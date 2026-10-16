"""One shared heavy-job queue for the PCRL mechanism-design campaign (data-free design, prototype and pilot preparation).

Every numerical unit (and every failed attempt) runs through `run`, which
- holds one of TWO slot locks (at most two heavy numerical jobs at once),
- forces one numerical thread per job (OMP/BLAS/torch thread variables),
- enforces a per-unit RSS reservation so the sum of running reservations stays <= 4 GiB,
- refuses to start when the campaign CPU budget (8 CPU-hours, including failed attempts, reviewer and verifier
  computation and the agent allowance logged with `charge`) or the 6-hour elapsed budget (from the campaign start)
  would be exceeded,
- records wall time, child CPU (RUSAGE_CHILDREN delta), peak RSS (sampled with psutil when
  available), return code and output hashes in an append-only ledger.
Runtime state lives under .private/mechanism_design/compute (never committed). No unit in this campaign may read
real-person arrays; see mechanism_design/no_real_data.py.
"""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / '.private' / 'mechanism_design' / 'compute'
CPU_LIMIT_S = 8 * 3600
ELAPSED_LIMIT_H = 6
RSS_LIMIT_GIB = 4.0
SLOTS = 2
THREAD_ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                               'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'TORCH_NUM_THREADS')}


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
    p = ROOT / 'START_UTC.json'
    if not p.exists():
        raise RuntimeError('queue not initialised')
    return dt.datetime.fromisoformat(json.loads(p.read_text())['start_utc'])


def charged_cpu_s():
    return sum(float(e.get('charged_cpu_s', 0.0)) for e in entries() if e.get('event') in ('unit_end', 'charge'))


def running_reservations():
    out = {}
    for e in entries():
        if e.get('event') == 'unit_start':
            out[e['unit_id']] = float(e['rss_reservation_gib'])
        elif e.get('event') == 'unit_end':
            out.pop(e['unit_id'], None)
    return out


CAMPAIGN_START_UTC = '2026-10-16T04:35:52+00:00'   # first campaign action (worktree/remote inspection); elapsed budget counts from here


def init():
    ROOT.mkdir(parents=True, exist_ok=True)
    os.chmod(ROOT.parent, 0o700)
    p = ROOT / 'START_UTC.json'
    if p.exists():
        return json.loads(p.read_text())
    rec = {'start_utc': CAMPAIGN_START_UTC, 'queue_opened_utc': now().isoformat(timespec='seconds'), 'cpu_limit_s': CPU_LIMIT_S, 'elapsed_limit_h': ELAPSED_LIMIT_H,
           'rss_limit_gib': RSS_LIMIT_GIB, 'slots': SLOTS}
    with open(p, 'x') as f:
        json.dump(rec, f, indent=2)
    _append({'event': 'init', **rec})
    return rec


def charge(label, cpu_s, note):
    """Charge non-queue work (agent analysis, light scripts) to the same budget."""
    _append({'event': 'charge', 'label': label, 'charged_cpu_s': float(cpu_s), 'note': note, 'utc': now().isoformat(timespec='seconds')})


def _sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def _peak_rss_loop(proc, interval=0.5):
    try:
        import psutil
    except ImportError:
        return None
    peak = 0
    try:
        p = psutil.Process(proc.pid)
        while proc.poll() is None:
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
    if any(e.get('event') == 'unit_end' and e['unit_id'] == unit_id for e in entries()):
        raise PermissionError(f'unit id {unit_id} already used (append-only; use a new versioned id)')
    elapsed_h = (now() - start_utc()).total_seconds() / 3600
    if elapsed_h > ELAPSED_LIMIT_H:
        raise PermissionError('elapsed budget exhausted')
    if charged_cpu_s() >= CPU_LIMIT_S:
        raise PermissionError('CPU budget exhausted')
    slot_files = [ROOT / f'slot{i}.lock' for i in range(SLOTS)]
    handle = None
    while handle is None:
        for sf in slot_files:
            f = open(sf, 'a+')
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                handle, slot = f, sf.name
                break
            except BlockingIOError:
                f.close()
        if handle is None:
            time.sleep(2)
    try:
        res = running_reservations()
        if sum(res.values()) + rss_gib > RSS_LIMIT_GIB + 1e-9:
            raise PermissionError(f'RSS reservation would exceed {RSS_LIMIT_GIB} GiB: running {res}')
        env = {**os.environ, **THREAD_ENV, 'MD_UNIT_ID': unit_id}
        _append({'event': 'unit_start', 'unit_id': unit_id, 'stage': stage, 'argv': argv, 'slot': slot, 'rss_reservation_gib': rss_gib,
                 'utc': now().isoformat(timespec='seconds')})
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        t0 = time.time()
        log_dir = ROOT / 'units' / unit_id
        log_dir.mkdir(parents=True, exist_ok=False)
        with open(log_dir / 'stdout.log', 'w') as so, open(log_dir / 'stderr.log', 'w') as se:
            proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=so, stderr=se)
            peak = _peak_rss_loop(proc)
            try:
                rc = proc.wait(timeout=max_wall_s)
            except subprocess.TimeoutExpired:
                proc.kill(); rc = -9
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
        hashes = {str(o): _sha(o) for o in outputs if Path(o).is_file()}
        rec = {'event': 'unit_end', 'unit_id': unit_id, 'stage': stage, 'returncode': rc, 'wall_s': round(time.time() - t0, 3),
               'charged_cpu_s': round(cpu, 3), 'child_maxrss_bytes_platform': after.ru_maxrss, 'sampled_peak_rss_bytes': peak,
               'rss_reservation_gib': rss_gib, 'status': 'done' if rc == 0 else 'failed', 'output_sha256': hashes,
               'utc': now().isoformat(timespec='seconds')}
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
            'start_utc': start_utc().isoformat() if (ROOT / 'START_UTC.json').exists() else None}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('init')
    r = sub.add_parser('run'); r.add_argument('--unit', required=True); r.add_argument('--stage', required=True)
    r.add_argument('--rss-gib', type=float, required=True); r.add_argument('--output', action='append', default=[])
    r.add_argument('--max-wall-s', type=int, default=6 * 3600); r.add_argument('cmd_argv', nargs=argparse.REMAINDER)
    c = sub.add_parser('charge'); c.add_argument('--label', required=True); c.add_argument('--cpu-s', type=float, required=True); c.add_argument('--note', required=True)
    sub.add_parser('summary')
    a = ap.parse_args(argv)
    if a.cmd == 'init':
        print(json.dumps(init()))
    elif a.cmd == 'run':
        cmd = a.cmd_argv[1:] if a.cmd_argv and a.cmd_argv[0] == '--' else a.cmd_argv
        rec = run(a.unit, cmd, a.stage, a.rss_gib, a.output, max_wall_s=a.max_wall_s)
        print(json.dumps(rec)); return 0 if rec['returncode'] == 0 else 1
    elif a.cmd == 'charge':
        charge(a.label, a.cpu_s, a.note)
    else:
        print(json.dumps(summary(), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
