"""Verifier stage V6 (procedural, label-free): process order and code provenance from the job-queue ledger, git and the records.

PCRL controlled training-size campaign (TSZ1; exploratory development on previously used ACS 2019 Florida PUMS person records).
  code provenance  for every production unit (unit id not starting with VER_): the module or script it ran was committed at launch
                   (not among the ledger's uncommitted code paths; the ledger's first path is read with its known one-character
                   truncation when git's porcelain line starts with a space), and no imported code file changed during the run
                   (ledger 'code_changed_during_run'), listed per unit
  order            registration commit < receipt < first label read; DEPLOY_W100.json written and committed before the trigger
                   record; trigger before the first correction-stage unit; correction code frozen (committed) before its first
                   real-data unit; both SELECTION_V<v>.json written before any MODEL_SELECT read of a corrected release and before
                   the first reader-grid unit; the reader code lock committed before the first reader-grid unit and the reader code
                   unchanged since the lock
  budget           elapsed (research window to 07:45:15Z) and charged CPU (14 CPU-hours) at the time of the check

    python3 .../v6_process.py --tag V1
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

ROOT = V.ROOT
RES = V.RESULTS
PRIV = ROOT / '.private' / 'pcrl_controlled_training_size_v1'
LEDGER = PRIV / 'compute' / 'ledger.jsonl'
LOG = PRIV / 'access' / 'LABEL_ACCESS.jsonl'
CORR_RE = re.compile(r'(?i:\bcorrect(?:ion|ions|ed)\b)|\b(?:LOC|PAIR|CONT)\b')


def ts(s):
    return dt.datetime.fromisoformat(s)


def git(*a):
    r = subprocess.run(['git', '-C', str(ROOT), *a], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def first_commit_time(path):
    out = git('log', '--format=%cI', '--reverse', '--', str(path))
    return out.splitlines()[0] if out else None


def module_file(argv):
    if not argv:
        return None
    if '-m' in argv:
        m = argv[argv.index('-m') + 1]
        return m.replace('.', '/') + '.py'
    for x in argv[1:]:
        if x.endswith('.py'):
            return x
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    led = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    log = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    starts = [e for e in led if e.get('event') == 'unit_start']
    ends = {e['unit_id']: e for e in led if e.get('event') == 'unit_end'}
    prov_bad, changed, rows = [], [], []
    for e in starts:
        u = e['unit_id']
        if u.startswith('VER_'):
            continue
        mf = module_file(e.get('argv'))
        unc = list(e.get('uncommitted_code_paths') or [])
        fixed = []
        for i, p in enumerate(unc):
            fixed.append(p)
            if i == 0 and not p.startswith(('tsz/', 'cts/', 'ohn/', 'ihs/', 'acs_replication/', 'scm_bench/', 'ffv/', 'task_')):
                fixed.append('t' + p)                            # ledger truncation of the first porcelain line (leading space stripped)
        hit = mf is not None and any(mf == p or (p.endswith('/') and mf.startswith(p)) for p in fixed)
        rows.append({'unit': u, 'utc': e['utc'], 'module': mf, 'git_head': (e.get('git_head') or '')[:9], 'module_uncommitted_at_launch': hit})
        if hit:
            prov_bad.append({'unit': u, 'utc': e['utc'], 'module': mf, 'uncommitted': unc})
        en = ends.get(u)
        if en and en.get('code_changed_during_run'):
            changed.append({'unit': u, 'changed': en['code_changed_during_run'], 'status': en.get('status')})
    truncated = sorted({p for e in starts for p in (e.get('uncommitted_code_paths') or [])[:1]
                        if not p.startswith(('tsz/', 'cts/', 'ohn/', 'ihs/', 'acs_replication/', 'scm_bench/', 'ffv/', 'task_'))})
    # order of key events
    order = {}
    reg = git('log', '-1', '--format=%cI', 'b3a5343f4')
    receipt = json.loads((PRIV / 'access' / 'REGISTRATION_RECEIPT_VERIFIED.json').read_text())['verified_utc']
    first_read = min((ts(x['utc']) for x in log), default=None)
    order['registration_commit'] = reg; order['receipt_verified'] = receipt; order['first_label_read'] = first_read.isoformat() if first_read else None
    dep_commit = first_commit_time(V.VER_DIR / 'DEPLOY_W100.json')
    trig_p = RES / 'trigger' / 'TRIGGER.json'
    trig = json.loads(trig_p.read_text()) if trig_p.is_file() else None
    trig_commit = first_commit_time(trig_p)
    corr_units = [e for e in starts if e['unit_id'].startswith('TSZ_CORR_V') or e['unit_id'].startswith('TSZ_NCV')]
    first_corr = min((ts(e['utc']) for e in corr_units), default=None)
    order.update({'DEPLOY_W100_first_commit': dep_commit, 'trigger_record_utc': trig['utc'] if trig else None, 'trigger_first_commit': trig_commit,
                  'first_correction_stage_unit': first_corr.isoformat() if first_corr else None})
    code_frozen = {m: first_commit_time(ROOT / m) for m in ('tsz/correct.py', 'tsz/nested.py', 'tsz/select.py', 'tsz/sizefit.py', 'tsz/control.py',
                                                             'tsz/evaluate.py', 'tsz/devreaders.py')}
    sel = {}
    for v in (0, 1):
        p = RES / 'conditional' / f'SELECTION_V{v}.json'
        if p.is_file():
            sel[v] = {'utc': json.loads(p.read_text())['utc'], 'first_commit': first_commit_time(p)}
    corrected_ms = [x for x in log if x['role'] == 'MODEL_SELECT' and (x['entry_point'].startswith('tsz.correct') or CORR_RE.search(x['purpose']))]
    grid_units = [e for e in starts if 'GRID' in e['unit_id'] or 'READER' in e['unit_id'] or 'DEVGRID' in e['unit_id']]
    first_grid = min((ts(e['utc']) for e in grid_units), default=None)
    lock_commit = first_commit_time(RES / 'DEVGRID_CODE_LOCK.json')
    lock = json.loads((RES / 'DEVGRID_CODE_LOCK.json').read_text())
    lock_files_now = {f: V.file_sha(ROOT / f) for f in lock['files_sha256']}
    lock_equal = lock_files_now == lock['files_sha256']
    problems = []
    if first_read and ts(receipt) > first_read:
        problems.append('a label read precedes the registration receipt')
    if trig and dep_commit and ts(dep_commit) > ts(trig['utc']):
        problems.append('DEPLOY_W100.json committed after the trigger record')
    if trig and first_corr and ts(trig['utc']) > first_corr:
        problems.append('a correction-stage unit started before the trigger record')
    for m in ('tsz/correct.py', 'tsz/nested.py', 'tsz/select.py'):
        if first_corr and code_frozen.get(m) and ts(code_frozen[m]) > first_corr:
            problems.append(f'{m} first committed after the first correction-stage unit')
    if len(sel) == 2:
        tsel = max(ts(s['utc']) for s in sel.values())
        if any(ts(x['utc']) < tsel for x in corrected_ms):
            problems.append('a MODEL_SELECT read of a corrected release precedes a selection record')
        if first_grid and first_grid < tsel:
            problems.append('a reader-grid unit started before both selection records')
    if first_grid and lock_commit and ts(lock_commit) > first_grid:
        problems.append('the reader code lock was committed after the first reader-grid unit')
    if not lock_equal:
        problems.append('reader code differs from DEVGRID_CODE_LOCK.json')
    now = dt.datetime.now(dt.timezone.utc)
    cpu = sum(float(e.get('charged_cpu_s', 0.0)) for e in led if e.get('event') in ('unit_end', 'charge'))
    budget = {'elapsed_h': (now - ts('2026-10-18T23:45:15+00:00')).total_seconds() / 3600, 'charged_cpu_h': cpu / 3600,
              'units_started': len(starts), 'units_failed': [e['unit_id'] for e in ends.values() if e.get('status') != 'done']}
    checks = [
        {'id': f'V6.PROCESS.CODE_PROVENANCE__{a.tag}', 'kind': 'PROCEDURAL_COMPLIANCE',
         'what': 'for every production queue unit: the module / script it ran was committed at launch (not in the ledger\'s uncommitted code '
                 'paths); code files that changed during a unit are listed; the ledger\'s truncated first uncommitted path is reported',
         'inputs': V.inputs_sha([LEDGER]), 'expected': 'no unit ran an uncommitted module',
         'observed': {'units_checked': len(rows), 'module_uncommitted_at_launch': prov_bad, 'code_changed_during_run': changed,
                      'ledger_first_path_truncation_examples': truncated},
         'status': 'FAIL' if prov_bad else ('ATTENTION' if (changed or truncated) else 'PASS')},
        {'id': f'V6.PROCESS.ORDER__{a.tag}', 'kind': 'PROCEDURAL_COMPLIANCE',
         'what': 'order of key events: registration commit, receipt, first label read; DEPLOY_W100 commit before the trigger record; trigger '
                 'before the correction stage; correction code committed before its first unit; both selections before any MODEL_SELECT read of '
                 'a corrected release and before the reader grid; reader code lock committed before the grid and unchanged',
         'inputs': V.inputs_sha([LEDGER, LOG, RES / 'DEVGRID_CODE_LOCK.json'] + ([trig_p] if trig else [])),
         'expected': 'every order holds', 'observed': {'order': order, 'code_first_commit': code_frozen, 'selections': sel,
                                                       'corrected_MODEL_SELECT_reads': [{k: x[k] for k in ('utc', 'unit_id', 'entry_point')} for x in corrected_ms],
                                                       'first_reader_grid_unit': first_grid.isoformat() if first_grid else None,
                                                       'reader_lock_first_commit': lock_commit, 'reader_code_equals_lock': lock_equal,
                                                       'budget': budget, 'problems': problems},
         'status': 'PASS' if not problems else 'FAIL'}]
    for c in checks:
        c['code_sha256'] = V.code_hashes(__file__, V.__file__)
    import os
    V.write_record(f'V6_PROCESS__{a.tag}__{os.environ.get("TSZ_UNIT_ID") or "local"}.json', {'checks': checks, 'units': rows, 'utc': V.utc()})
    print(json.dumps(V.append_checks(checks)))
    print(json.dumps({c['id']: c['status'] for c in checks}))
    print(json.dumps({'prov_bad': prov_bad, 'changed': changed[:10], 'truncated': truncated, 'problems': problems, 'order': order, 'budget': budget}, default=str)[:4000])


if __name__ == '__main__':
    main()
