"""Verifier stage V4 (procedural, label-free): exposure from the label-access log.

PCRL controlled training-size campaign (TSZ1; exploratory development on previously used ACS 2019 Florida PUMS person records).
Reads .private/<campaign>/access/LABEL_ACCESS.jsonl, the job-queue ledger, the registration receipt, ACCESS_POLICY.json and
ACCESS_AMENDMENTS.json (with the git commit time of each amendment), and the label-free MODEL_FIT / CALIBRATION / MODEL_SELECT row
ids and households. Every row-set hash of a MODEL_FIT read is recomputed from the verifier's own subset reconstruction.

Checks (each line, then each unit):
  closed roles      no line for ATTACK_FIT / ATTACK_VALIDATION / DEV_REPORT / SEALED_ASSESSMENT
  receipt           no line before the registration receipt's verified_utc
  policy            operation x role x fields x entry point allowed (registered policy + committed amendments, each committed before
                    its first use)
  rows              'fit' / 'select_ncv' lines: rows_sha256, people, households == own reconstruction of the named registered subset;
                    'select_ncv' only S100; role-wide reads (CALIBRATION, MODEL_SELECT) == own full-role row hash
  units             every unit with a 'fit' read reads one subset only, the one its unit id / purpose names; no MODEL_SELECT read by
                    a unit that has a 'fit' or 'select_ncv' read; A1_PO_16 size fits read occupation only
  ledger            every non-null unit id exists in the job-queue ledger and its reads fall inside [unit_start, unit_end]
  null unit ids     listed (deviation 4 discloses two selftest reads)
  corrected reads   MODEL_SELECT reads naming a corrected release (tsz.select's regex) before both SELECTION_V<v>.json exist

    python3 .../v4_access.py --tag V1
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

ROOT = V.ROOT
sys.path.insert(0, str(ROOT))
from tsz import common as C  # noqa: E402   (label-free features only)

RES = V.RESULTS
PRIV = ROOT / '.private' / 'pcrl_controlled_training_size_v1'
LOG = PRIV / 'access' / 'LABEL_ACCESS.jsonl'
LEDGER = PRIV / 'compute' / 'ledger.jsonl'
RECEIPT = PRIV / 'access' / 'REGISTRATION_RECEIPT_VERIFIED.json'
CLOSED = ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT', 'SEALED_ASSESSMENT')
CORR_RE = re.compile(r'(?i:\bcorrect(?:ion|ions|ed)\b)|\b(?:LOC|PAIR|CONT)\b')


def ts(s):
    return dt.datetime.fromisoformat(s)


def git_commit_time(path):
    r = subprocess.run(['git', '-C', str(ROOT), 'log', '--diff-filter=A', '--format=%cI', '--', str(path)], capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    return lines[-1] if lines else None


def amendment_commits():
    """{(operation, entry_point): first commit time (ISO) of the ACCESS_AMENDMENTS.json version that contains it}."""
    p = 'results/pcrl_controlled_training_size_v1/ACCESS_AMENDMENTS.json'
    r = subprocess.run(['git', '-C', str(ROOT), 'log', '--format=%H %cI', '--reverse', '--', p], capture_output=True, text=True)
    out = {}
    for line in r.stdout.splitlines():
        h, t = line.split()
        s = subprocess.run(['git', '-C', str(ROOT), 'show', f'{h}:{p}'], capture_output=True, text=True)
        if s.returncode != 0:
            continue
        for a in json.loads(s.stdout).get('amendments', []):
            out.setdefault((a['operation'], a['entry_point']), t)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    lines = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    ledger = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    receipt = json.loads(RECEIPT.read_text())
    policy = json.loads((RES / 'ACCESS_POLICY.json').read_text())
    amend_p = RES / 'ACCESS_AMENDMENTS.json'
    amend = json.loads(amend_p.read_text()) if amend_p.is_file() else {'amendments': []}
    amend_time = amendment_commits()
    F = C.features('MODEL_FIT')
    masks = V.subset_masks(F['unit'])
    sub_rec = {sid: {'rows_sha256': V.arr_sha(np.asarray(F['row_id'])[m]), 'people': int(m.sum()),
                     'households': int(len(np.unique(np.asarray(F['unit'])[m])))} for sid, m in masks.items()}
    full = {}
    for r in ('CALIBRATION', 'MODEL_SELECT'):
        Fr = C.features(r)
        full[r] = {'rows_sha256': V.arr_sha(np.asarray(Fr['row_id'])), 'people': int(len(Fr['row_id'])), 'households': int(len(np.unique(Fr['unit'])))}
    eps = {op: list(x) for op, x in policy['entry_points'].items()}
    for am in amend['amendments']:
        eps.setdefault(am['operation'], []).append(am['entry_point'])
    ops = policy['operations']
    rec_t = ts(receipt['verified_utc'])
    problems = {k: [] for k in ('closed_role', 'before_receipt', 'policy', 'rows', 'units', 'ledger', 'sizefit_fields', 'amendment_timing',
                                'corrected_reads_before_selection')}
    starts = {e['unit_id']: ts(e['utc']) for e in ledger if e.get('event') == 'unit_start'}
    ends = {e['unit_id']: ts(e['utc']) for e in ledger if e.get('event') == 'unit_end'}
    null_units = []
    by_unit = {}
    sel_files = [RES / 'conditional' / f'SELECTION_V{v}.json' for v in (0, 1)]
    sel_time = None
    if all(p.is_file() for p in sel_files):
        sel_time = max(ts(json.loads(p.read_text())['utc']) for p in sel_files)
    for i, e in enumerate(lines):
        t = ts(e['utc'])
        role, op, ep = e['role'], e['operation'], e['entry_point'].split(':')[0]
        if role in CLOSED:
            problems['closed_role'].append(i)
        if t < rec_t:
            problems['before_receipt'].append(i)
        pol = ops.get(op)
        allowed_ep = any(ep == x or ep.startswith(x if x.endswith('/') else x + '.') or (x.endswith('/') and ep.startswith(x)) for x in eps.get(op, []))
        if pol is None or role not in pol['roles'] or not set(e['fields']) <= set(pol['fields']) or not allowed_ep:
            problems['policy'].append({'line': i, 'operation': op, 'role': role, 'fields': e['fields'], 'entry_point': ep})
        for am in amend['amendments']:
            if am['entry_point'] == ep and am['operation'] == op:
                ct = amend_time.get((op, ep))
                if ct is None or ts(ct) > t:
                    problems['amendment_timing'].append({'line': i, 'entry_point': ep, 'amendment_commit': ct, 'read_utc': e['utc']})
        if role == 'MODEL_FIT':
            sid = e.get('subset')
            if sid is None:
                exp = {'rows_sha256': V.arr_sha(np.asarray(F['row_id'])), 'people': 20000, 'households': 13337}
            else:
                exp = sub_rec.get(sid)
            if exp is None or any(e.get(k) != exp[k] for k in ('rows_sha256', 'people', 'households')):
                problems['rows'].append({'line': i, 'subset': sid})
            if op == 'select_ncv' and sid != 'S100':
                problems['rows'].append({'line': i, 'subset': sid, 'why': 'select_ncv outside S100'})
        elif role in full:
            if e.get('subset') is not None or any(e.get(k) != full[role][k] for k in ('rows_sha256', 'people', 'households')):
                problems['rows'].append({'line': i, 'role': role})
        if op == 'fit' and ep == 'tsz.sizefit' and e['fields'] != ['occupation']:
            problems['sizefit_fields'].append(i)
        if op == 'calibrate' and (role != 'CALIBRATION' or e['fields'] != ['occupation']):
            problems['policy'].append({'line': i, 'why': 'calibrate fields'})
        u = e.get('unit_id')
        if u is None:
            null_units.append({'line': i, 'utc': e['utc'], 'entry_point': e['entry_point'], 'operation': op, 'role': role,
                               'fields': e['fields'], 'subset': e.get('subset'), 'purpose': e['purpose'][:90]})
        else:
            by_unit.setdefault(u, []).append(e)
            if u not in starts:
                problems['ledger'].append({'line': i, 'unit_id': u, 'why': 'unit not in the ledger'})
            else:
                end = ends.get(u)
                if t < starts[u].replace(microsecond=0) or (end is not None and t > end):
                    problems['ledger'].append({'line': i, 'unit_id': u, 'why': 'read outside the unit interval', 'utc': e['utc']})
        if role == 'MODEL_SELECT' and (ep.startswith('tsz.correct') or CORR_RE.search(e['purpose'])):
            if sel_time is None or t < sel_time:
                problems['corrected_reads_before_selection'].append({'line': i, 'utc': e['utc'], 'purpose': e['purpose'][:120]})
    unit_summary = {}
    for u, es in by_unit.items():
        ops_u = sorted({x['operation'] for x in es})
        subsets = sorted({x.get('subset') for x in es if x['role'] == 'MODEL_FIT' and x['operation'] == 'fit'})
        ms = any(x['role'] == 'MODEL_SELECT' for x in es)
        unit_summary[u] = {'operations': ops_u, 'fit_subsets': subsets, 'reads_MODEL_SELECT': ms, 'n_reads': len(es)}
        if ('fit' in ops_u or 'select_ncv' in ops_u) and ms:
            problems['units'].append({'unit_id': u, 'why': 'a fitting unit read MODEL_SELECT labels'})
        if len(subsets) > 1:
            problems['units'].append({'unit_id': u, 'why': f'several subsets {subsets}'})
        if subsets and subsets[0] is not None:
            tok = subsets[0]
            named = re.findall(r'S(?:50|70)_O[0-3]|S100', u + ' ' + ' '.join(x['purpose'] for x in es))
            if named and any(n != tok for n in set(named)):
                problems['units'].append({'unit_id': u, 'why': f'unit/purpose names {sorted(set(named))} but read {tok}'})
    status = 'PASS' if not any(problems.values()) else 'FAIL'
    disclosed_null = [x for x in null_units if x['entry_point'] == 'tsz.sizefit:selftest' and x['subset'] == 'S100' and x['fields'] == ['occupation']]
    chk = [{'id': f'V4.ACCESS.EXPOSURE__{a.tag}', 'kind': 'PROCEDURAL_COMPLIANCE',
            'what': 'label-access log audit: closed roles never read; no read before the registration receipt; every read allowed by '
                    'operation x role x fields x entry point (amendments committed before first use); every MODEL_FIT read touched exactly '
                    'its registered subset (own row-set reconstruction); select_ncv only S100; role-wide reads complete; every fitting unit '
                    'read one subset (the one it names) and no MODEL_SELECT label; A1_PO_16 fits read occupation only; reads inside their '
                    'unit interval in the queue ledger; no MODEL_SELECT read naming a corrected release before both selections exist',
            'inputs': V.inputs_sha([LOG, LEDGER, RECEIPT, RES / 'ACCESS_POLICY.json'] + ([amend_p] if amend_p.is_file() else [])),
            'expected': 'no problem in any category',
            'observed': {'n_lines': len(lines), 'n_units': len(by_unit), 'problems': problems,
                         'operations': sorted({(x['operation'], x['role'], x['entry_point'].split(':')[0]) for x in lines}),
                         'selection_time': None if sel_time is None else sel_time.isoformat(), 'amendment_commit_times': {f'{k[0]}|{k[1]}': t for k, t in amend_time.items()},
                         'unit_summary': unit_summary},
            'status': status},
           {'id': f'V4.ACCESS.NULL_UNIT_READS__{a.tag}', 'kind': 'PROCEDURAL_COMPLIANCE',
            'what': 'label reads without a queue unit id (outside the job queue): listed and compared with the disclosure (PROTOCOL_DEVIATIONS '
                    'item 4: two tsz.sizefit selftest reads of MODEL_FIT S100 occupation at 00:20:13Z and 00:20:27Z)',
            'inputs': V.inputs_sha([LOG, RES / 'PROTOCOL_DEVIATIONS.md']), 'expected': 'exactly the two disclosed selftest reads',
            'observed': {'null_unit_reads': null_units, 'n_disclosed_pattern': len(disclosed_null)},
            'status': 'ATTENTION' if (len(null_units) == 2 and len(disclosed_null) == 2) else ('PASS' if not null_units else 'FAIL'),
            'note': 'ATTENTION = disclosed deviation (reads happened; arrays discarded per the disclosure; policy-allowed operation and rows)'}]
    for c in chk:
        c['code_sha256'] = V.code_hashes(__file__, V.__file__)
    V.write_record(f'V4_ACCESS__{a.tag}.json', {'checks': chk, 'utc': V.utc()})
    print(json.dumps(V.append_checks(chk)))
    print(json.dumps({c['id']: c['status'] for c in chk}))
    print(json.dumps({k: v for k, v in problems.items() if v}, default=str)[:3000])


if __name__ == '__main__':
    main()
