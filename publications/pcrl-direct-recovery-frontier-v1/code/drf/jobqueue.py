"""One shared heavy-job queue for the PCRL direct-recovery frontier campaign (every role and every agent uses it; derived unchanged
from frc.jobqueue (itself from nrm.jobqueue) except for this campaign's paths, start time, CPU limit, environment names and code packages).

Every numerical unit (and every failed attempt) runs through `run`, which
- records, BEFORE launch, immutable SHA-256 hashes of the code it can import (every .py file of the campaign package drf, of the
  verifier's code folder and of the imported source packages: nrm, tsz, cts, ohn, ihs, acs_replication, scm_bench, ffv,
  task_conditioned_representation, task_headroom, mechanism_design), the git HEAD, whether those files differ from HEAD, the
  interpreter and its numerical library versions (units/<unit>/CODE_AT_START.json; the combined digest goes into the ledger's
  unit_start line),
- holds one of TWO slot locks (at most two heavy numerical jobs at once) with FIFO tickets,
- forces one numerical thread per job (OMP/BLAS/torch thread variables),
- keeps the sum of running RSS reservations <= 8 GiB,
- refuses to start when the campaign CPU budget (12 CPU-hours including failed attempts, retries, reader fits, reviewers' and
  agents' numerical work charged with `charge`) is used, when the 8-hour research window has ended for a research stage, or when
  the 30-minute closeout allowance has ended for any stage,
- records wall time, child CPU (RUSAGE_CHILDREN delta), sampled peak RSS, return code, output hashes and the unit's own counters
  (UNIT_STATS.json) in an append-only ledger.
Runtime state lives under .private/pcrl_direct_recovery_frontier_v1/compute (never committed).
"""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / '.private' / 'pcrl_direct_recovery_frontier_v1' / 'compute'
CPU_LIMIT_S = 12 * 3600
ELAPSED_LIMIT_H = 8.0                      # the brief's eight elapsed hours (research window 7.5 h + 30-minute closeout allowance)
RESEARCH_LIMIT_H = 7.5
CLOSEOUT_STAGES = ('custody', 'closeout', 'publication', 'deployment_check')
RSS_LIMIT_GIB = 8.0
SLOTS = 2
THREAD_ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                               'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'TORCH_NUM_THREADS')}
CAMPAIGN_START_UTC = '2026-10-25T02:17:51+00:00'   # first `date -u` of this campaign
CODE_PACKAGES = ('drf', 'sar', 'results/pcrl_direct_recovery_frontier_v1/roles/verifier/code', 'ddr', 'frc', 'fsa', 'nrm', 'tsz', 'cts', 'ohn', 'ihs',
                 'acs_replication', 'scm_bench', 'ffv', 'task_conditioned_representation', 'task_headroom', 'mechanism_design')
UNIT_ENV, STATS_ENV = 'DRF_UNIT_ID', 'DRF_UNIT_STATS'
VERSION_PROBE = ('import json, sys, importlib.metadata as m\nout = {"python": sys.version.split()[0], "executable": sys.executable}\n'
                 'for p in ("numpy", "scipy", "scikit-learn", "torch", "catboost", "psutil"):\n'
                 '    try:\n        out[p] = m.version(p)\n    except Exception:\n        out[p] = None\nprint(json.dumps(out))')


def now():
    return dt.datetime.now(dt.timezone.utc)


def _ledger():
    return ROOT / 'ledger.jsonl'


def _append(rec):
    ROOT.mkdir(parents=True, exist_ok=True)
    with open(_ledger(), 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec, sort_keys=True) + '\n')
        fcntl.flock(f, fcntl.LOCK_UN)


def entries():
    if not _ledger().exists():
        return []
    return [json.loads(l) for l in _ledger().read_text().splitlines() if l.strip()]


def start_utc():
    return dt.datetime.fromisoformat(CAMPAIGN_START_UTC)


def charged_cpu_s():
    tot = 0.0
    for e in entries():
        if e.get('event') in ('unit_end', 'charge'):
            v = float(e.get('charged_cpu_s', 0.0))
            tot += max(v, 0.0) if math.isfinite(v) else math.inf      # fail closed on a corrupt record
    return tot


def running_reservations():
    out = {}
    for e in entries():
        if e.get('event') == 'unit_start':
            out[e['unit_id']] = float(e['rss_reservation_gib'])
        elif e.get('event') == 'unit_end':
            out.pop(e['unit_id'], None)
    return out


def init():
    ROOT.mkdir(parents=True, exist_ok=True)
    os.chmod(ROOT.parent, 0o700)
    p = ROOT / 'START_UTC.json'
    if p.exists():
        return json.loads(p.read_text())
    rec = {'start_utc': CAMPAIGN_START_UTC, 'queue_opened_utc': now().isoformat(timespec='seconds'), 'cpu_limit_s': CPU_LIMIT_S,
           'elapsed_limit_h': ELAPSED_LIMIT_H, 'research_limit_h': RESEARCH_LIMIT_H, 'rss_limit_gib': RSS_LIMIT_GIB, 'slots': SLOTS}
    with open(p, 'x') as f:
        json.dump(rec, f, indent=2)
    _append({'event': 'init', **rec})
    return rec


def charge(label, cpu_s, note):
    """Charge non-queue work (agent analysis, light scripts, out-of-queue upper estimates) to the same budget."""
    cpu_s = float(cpu_s)
    if not math.isfinite(cpu_s) or cpu_s < 0:
        raise ValueError(f'a charge must be a finite, non-negative CPU-second upper estimate (got {cpu_s!r})')
    _append({'event': 'charge', 'label': label, 'charged_cpu_s': float(cpu_s), 'note': note, 'utc': now().isoformat(timespec='seconds')})


def close(note):
    """Close the queue at campaign end: an append-only event; every later `run` is refused (charges are still recorded)."""
    _append({'event': 'queue_closed', 'note': note, 'utc': now().isoformat(timespec='seconds')})


def _sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def _git(*a):
    r = subprocess.run(['git', '-C', str(REPO), *a], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def code_manifest():
    """SHA-256 of every importable source file of the campaign and source packages, at this moment."""
    files = {}
    for pkg in CODE_PACKAGES:
        d = REPO / pkg
        if not d.is_dir():
            continue
        for p in sorted(d.rglob('*.py')):
            if '__pycache__' in p.parts:
                continue
            files[str(p.relative_to(REPO))] = _sha(p)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    # -z: NUL-separated, unstripped entries (engineering review ENG1-01: the stripped porcelain output cut the first path's first
    # character in the records of every unit launched before commit 442b84694, committed 15:39:57Z on 2026-10-20; those rows are
    # kept and disclosed, never rewritten)
    r = subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain', '-z', '--', *CODE_PACKAGES], capture_output=True, text=True)
    entries = [e for e in r.stdout.split('\0') if e] if r.returncode == 0 else []
    return {'files': files, 'digest': digest, 'git_head': _git('rev-parse', 'HEAD'),
            'uncommitted_code_paths': [e[3:] for e in entries if len(e) > 3]}


def interpreter_versions(executable):
    ROOT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / 'interpreters.json'
    known = json.loads(cache.read_text()) if cache.is_file() else {}
    key = str(executable)
    stamp = f'{key}|{time.time_ns()}'           # probe every time: a pip install inside a venv leaves the executable's mtime unchanged
    if stamp not in known:
        r = subprocess.run([key, '-c', VERSION_PROBE], capture_output=True, text=True, env={**os.environ, **THREAD_ENV})
        known[stamp] = json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else {'error': r.stderr[-400:]}
        cache.write_text(json.dumps(known, indent=1, sort_keys=True))
    return known[stamp]


def _peak_rss_loop(proc, interval=0.5, deadline=None):
    try:
        import psutil
    except ImportError:
        return None
    peak = 0
    try:
        p = psutil.Process(proc.pid)
        while proc.poll() is None and (deadline is None or time.time() < deadline):
            try:
                tree = [p] + p.children(recursive=True)
                peak = max(peak, sum(c.memory_info().rss for c in tree if c.is_running()))
            except Exception:
                pass
            time.sleep(interval)
    except Exception:
        return None
    return peak


def run(unit_id, argv, stage, rss_gib, outputs=(), cwd=None, max_wall_s=6 * 3600):
    init()
    es = entries()
    if any(e.get('event') == 'queue_closed' for e in es):
        raise PermissionError('the campaign queue is closed (queue_closed event in the ledger)')
    if any(e.get('event') in ('unit_start', 'unit_end') and e['unit_id'] == unit_id for e in es):
        raise PermissionError(f'unit id {unit_id} already used (append-only; use a new versioned id)')
    elapsed_h = (now() - start_utc()).total_seconds() / 3600
    if elapsed_h > ELAPSED_LIMIT_H:
        raise PermissionError('elapsed budget (research window plus closeout allowance) exhausted')
    if elapsed_h > RESEARCH_LIMIT_H and stage not in CLOSEOUT_STAGES:
        raise PermissionError(f'research window ended; only closeout stages {CLOSEOUT_STAGES} may run')
    if charged_cpu_s() >= CPU_LIMIT_S:
        raise PermissionError('CPU budget exhausted')
    log_dir = ROOT / 'units' / unit_id
    log_dir.mkdir(parents=True, exist_ok=False)
    prov = code_manifest()
    prov['interpreter'] = interpreter_versions(argv[0]) if argv else None
    prov['argv'] = argv
    prov['recorded_utc'] = now().isoformat(timespec='seconds')
    (log_dir / 'CODE_AT_START.json').write_text(json.dumps(prov, indent=1, sort_keys=True))
    slot_files = [ROOT / f'slot{i}.lock' for i in range(SLOTS)]
    handle = None
    tdir = ROOT / 'tickets'; tdir.mkdir(parents=True, exist_ok=True)
    ticket = tdir / f'{time.time_ns():020d}_{os.getpid()}_{unit_id}'
    ticket.write_text(now().isoformat(timespec='seconds'))
    try:
        while handle is None:
            live = []
            for t in sorted(tdir.iterdir()):
                try:
                    pid = int(t.name.split('_')[1]); os.kill(pid, 0); live.append(t)
                except (ValueError, IndexError, ProcessLookupError):
                    t.unlink(missing_ok=True)
                except PermissionError:
                    live.append(t)
            if live and live[0] == ticket:
                for sf in slot_files:
                    f = open(sf, 'a+')
                    try:
                        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        handle, slot = f, sf.name
                        break
                    except BlockingIOError:
                        f.close()
            if handle is None:
                time.sleep(1)
    finally:
        ticket.unlink(missing_ok=True)
    try:
        # the code may have changed while waiting for a slot: re-hash at the actual launch and refuse silent drift
        launch = code_manifest()
        if launch['digest'] != prov['digest']:
            prov_launch = {'files_changed': sorted(k for k in set(prov['files']) | set(launch['files'])
                                                   if prov['files'].get(k) != launch['files'].get(k))}
            (log_dir / 'CODE_AT_LAUNCH.json').write_text(json.dumps({**launch, **prov_launch}, indent=1, sort_keys=True))
        res = running_reservations()
        if sum(res.values()) + rss_gib > RSS_LIMIT_GIB + 1e-9:
            raise PermissionError(f'RSS reservation would exceed {RSS_LIMIT_GIB} GiB: running {res}')
        env = {**os.environ, **THREAD_ENV, UNIT_ENV: unit_id, STATS_ENV: str(log_dir / 'UNIT_STATS.json')}
        _append({'event': 'unit_start', 'unit_id': unit_id, 'stage': stage, 'argv': argv, 'slot': slot, 'rss_reservation_gib': rss_gib,
                 'code_digest_at_launch': launch['digest'], 'code_digest_at_registration': prov['digest'],
                 'git_head': launch['git_head'], 'uncommitted_code_paths': launch['uncommitted_code_paths'],
                 'interpreter': prov['interpreter'], 'utc': now().isoformat(timespec='seconds')})
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        t0 = time.time()
        timed_out = False
        with open(log_dir / 'stdout.log', 'w') as so, open(log_dir / 'stderr.log', 'w') as se:
            proc = subprocess.Popen(argv, cwd=cwd or str(REPO), env=env, stdout=so, stderr=se)
            try:
                peak = _peak_rss_loop(proc, deadline=t0 + max_wall_s)
                try:
                    rc = proc.wait(timeout=max(0.0, t0 + max_wall_s - time.time()))
                except subprocess.TimeoutExpired:
                    timed_out = True
                    proc.kill(); proc.wait(); rc = -9          # reap it: RUSAGE_CHILDREN counts only waited-for children
            except BaseException:                              # runner interrupted: no orphan, CPU charged, reservation freed
                if proc.poll() is None:
                    proc.kill()
                proc.wait()
                after = resource.getrusage(resource.RUSAGE_CHILDREN)
                _append({'event': 'unit_end', 'unit_id': unit_id, 'stage': stage, 'returncode': proc.returncode,
                         'wall_s': round(time.time() - t0, 3),
                         'charged_cpu_s': round((after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime), 3),
                         'rss_reservation_gib': rss_gib, 'status': 'interrupted', 'output_sha256': {}, 'unit_stats': {},
                         'utc': now().isoformat(timespec='seconds')})
                raise
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
        end_code = code_manifest()                             # lazy imports can happen long after launch: record any drift
        changed = sorted(k for k in set(launch['files']) | set(end_code['files']) if launch['files'].get(k) != end_code['files'].get(k))
        hashes = {str(o): _sha(o) for o in outputs if Path(o).is_file()}
        stats_p = log_dir / 'UNIT_STATS.json'
        try:
            stats = json.loads(stats_p.read_text()) if stats_p.is_file() else {}
        except json.JSONDecodeError:
            stats = {'unreadable': True}
        rec = {'event': 'unit_end', 'unit_id': unit_id, 'stage': stage, 'returncode': rc, 'wall_s': round(time.time() - t0, 3),
               'charged_cpu_s': round(cpu, 3), 'child_maxrss_bytes_platform': after.ru_maxrss, 'sampled_peak_rss_bytes': peak,
               'rss_reservation_gib': rss_gib, 'status': 'done' if rc == 0 else 'failed', 'output_sha256': hashes,
               'unit_stats': stats, 'timed_out': timed_out, 'code_digest_at_end': end_code['digest'],
               'code_changed_during_run': changed, 'utc': now().isoformat(timespec='seconds')}
        _append(rec)
        return rec
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN); handle.close()


def summary():
    es = entries()
    ends = [e for e in es if e.get('event') == 'unit_end']
    return {'units_done': sum(e['status'] == 'done' for e in ends), 'units_failed': sum(e['status'] != 'done' for e in ends),
            'charged_cpu_s': round(charged_cpu_s(), 1), 'charged_cpu_h': round(charged_cpu_s() / 3600, 3),
            'agent_charges': [e for e in es if e.get('event') == 'charge'],
            'failed_units': [e['unit_id'] for e in ends if e['status'] != 'done'],
            'elapsed_h': round((now() - start_utc()).total_seconds() / 3600, 3), 'start_utc': CAMPAIGN_START_UTC}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('init')
    r = sub.add_parser('run'); r.add_argument('--unit', required=True); r.add_argument('--stage', required=True)
    r.add_argument('--rss-gib', type=float, required=True); r.add_argument('--output', action='append', default=[])
    r.add_argument('--max-wall-s', type=int, default=6 * 3600); r.add_argument('cmd_argv', nargs=argparse.REMAINDER)
    c = sub.add_parser('charge'); c.add_argument('--label', required=True); c.add_argument('--cpu-s', type=float, required=True); c.add_argument('--note', required=True)
    sub.add_parser('summary')
    cl = sub.add_parser('close'); cl.add_argument('--note', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'init':
        print(json.dumps(init()))
    elif a.cmd == 'run':
        cmd = a.cmd_argv[1:] if a.cmd_argv and a.cmd_argv[0] == '--' else a.cmd_argv
        rec = run(a.unit, cmd, a.stage, a.rss_gib, a.output, max_wall_s=a.max_wall_s)
        print(json.dumps(rec)); return 0 if rec['returncode'] == 0 else 1
    elif a.cmd == 'charge':
        charge(a.label, a.cpu_s, a.note)
    elif a.cmd == 'close':
        close(a.note)
    else:
        print(json.dumps(summary(), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
