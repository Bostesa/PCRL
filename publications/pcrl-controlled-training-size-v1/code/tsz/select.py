"""Strength selection of the conditional correction stage (PCRL controlled training-size campaign `pcrl_controlled_training_size_v1`;
EXPLORATORY DEVELOPMENT on previously used records, not confirmation). Runs ONLY if the registered trigger fired, after the nested
cost tables exist and BEFORE any MODEL_SELECT read of a corrected release. No label is read here.

Census context: public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida; occupation group (5 broad groups)
from six raw fields; SEX (recorded demographic column) entered the nested tables only as a MODEL_FIT training / evaluation label.

    python3 -m tsz.select run --version V     ->  results/<campaign>/conditional/SELECTION_V<v>.json (write-once)
    python3 -m tsz.select selftest            invented tables only

For each selection arm (LOC = primary, PAIR = secondary; RULES.md 8.1): the nested cross-fitted table NESTED_<arm>.json
(tsz.nested, schema TSZ1-nested-v1; its npz hash-checked) is converted by `rule_table` (the ONE adapter point) into the cost table
of the statistical reviewer's frozen rule `tsz.rules.select_strength(cost_table, version)` (U = estimate + 2 SE <= allowance per
statistic and version, prefix rule over the registered strength set; identity always admissible). CONT gets no selection of its
own: it is evaluated at the selected strength of the arm it matches (CONT@s*_LOC, CONT@s*_PAIR). The record keeps the whole audit
trail, the nested-table SHA-256s, the rules SHA-256s and a check that no MODEL_SELECT label read of a corrected release precedes it.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import common as C
from . import correct as CR
from . import nested as NS

SELECTION_ARMS = ('LOC', 'PAIR')
STATS = NS.STATS
SEL_DIR = C.RESULTS / 'conditional'
_CORR_WORDS = re.compile(r'(?i:\bcorrect(?:ion|ions|ed)\b)|\b(?:LOC|PAIR|CONT)\b')   # arm names case-sensitive (views are lower case)


def rule_table(nested_rec):
    """ADAPTER POINT: TSZ1-nested-v1 `per_strength` {skey: cost_summary} -> tsz.rules.select_strength's cost table
    {skey: {statistic: {'estimate': pooled mean increment vs s = 0, 'se': household-clustered SE}}} (all strengths, s = 0 included;
    the rule checks that the identity row is exactly zero)."""
    return {k: {j: {'estimate': row['stats'][j], 'se': row['se'][j]} for j in STATS} for k, row in nested_rec['per_strength'].items()}


def corrected_ms_reads(events):
    """MODEL_SELECT label reads of a corrected release (an entry point inside tsz.correct, or a purpose naming the correction)."""
    out = []
    for e in events:
        if e.get('role') != 'MODEL_SELECT':
            continue
        if str(e.get('entry_point', '')).startswith('tsz.correct') or _CORR_WORDS.search(str(e.get('purpose', ''))):
            out.append({k: e.get(k) for k in ('utc', 'unit_id', 'entry_point', 'operation', 'purpose', 'fields')})
    return out


def _log_digest():
    if not C.LABEL_LOG.is_file():
        return {'lines': 0, 'sha256': None}
    b = C.LABEL_LOG.read_bytes()
    return {'lines': b.count(b'\n'), 'sha256': hashlib.sha256(b).hexdigest()}


def run(v, nested_root=None, out_dir=None, RU=None, events=None):
    t0 = time.process_time()
    v = int(v)
    RU = RU or importlib.import_module('tsz.rules')
    if not hasattr(RU, 'select_strength'):
        raise PermissionError('tsz.rules.select_strength (the frozen rule) is not available')
    out = Path(out_dir or SEL_DIR) / f'SELECTION_V{v}.json'
    if out.exists():
        raise PermissionError(f'{out} exists (write-once)')
    events = C.access_events() if events is None else events
    bad = corrected_ms_reads(events)
    if bad:
        raise PermissionError(f'MODEL_SELECT label reads of a corrected release precede the selection: {bad}')
    root = Path(nested_root) if nested_root is not None else NS.ndir(v)
    S_rule = tuple(float(s) for s in RU.STRENGTHS)
    results, tables, info = {}, {}, {}
    for arm in SELECTION_ARMS:
        rec = NS.load_nested(v, arm, root)                       # NESTED_<arm>.npz hash-checked against its record
        if rec.get('schema') != 'TSZ1-nested-v1' or int(rec['version']) != v or rec['arm'] != arm:
            raise AssertionError(f'NESTED_{arm}.json is not the TSZ1-nested-v1 table of version {v}')
        if tuple(float(s) for s in rec['strengths']) != S_rule:
            raise AssertionError(f'NESTED_{arm} strengths {rec["strengths"]} differ from the registered set {S_rule}')
        tab = rule_table(rec)
        res = RU.select_strength(tab, v)
        if float(res['selected']) not in S_rule:
            raise AssertionError('the rule returned a strength outside the registered set')
        results[arm], tables[arm] = res, rec['per_strength']
        info[arm] = {'json_sha256': C.sha(root / f'NESTED_{arm}.json'), 'npz_sha256': rec['npz_sha256'], 'unit_id': rec.get('unit_id'),
                     'utc': rec.get('utc'), 'strength_source': rec.get('strength_source'), 'targets': rec.get('targets'),
                     'pooled': rec.get('pooled'), 'training_sizes': {k: f['training'] for k, f in rec.get('folds', {}).items()},
                     'disclosure': rec.get('disclosure')}
    sel = {arm: float(results[arm]['selected']) for arm in SELECTION_ARMS}
    rules_json = Path(getattr(RU, 'RULES_PATH', CR.RULES_JSON))
    record = {'schema': 'TSZ1-selection-v1', 'version': v, 'rule': 'tsz.rules.select_strength (statistical reviewer, frozen)',
              'selected': sel, 'primary_arm': CR.PRIMARY, 'primary_selected': sel[CR.PRIMARY],
              'cont_matched': {'CONT@LOC': sel['LOC'], 'CONT@PAIR': sel['PAIR'],
                               'note': 'CONT has no selection of its own; it is evaluated at the selected strength of the arm it matches'},
              'identity': {arm: sel[arm] == 0.0 for arm in SELECTION_ARMS},
              'audit': results, 'cost_tables': tables, 'nested_tables': info,
              'rules': {'rules_sha256_reported_by_rule': results[CR.PRIMARY].get('rules_sha256'),
                        'rules_json_sha256': C.sha(rules_json) if rules_json.is_file() else None,
                        'rules_json_committed': bool(rules_json.is_file() and C.committed(rules_json)),
                        'rules_py_sha256': C.sha(Path(RU.__file__)), 'strengths': list(S_rule)},
              'model_select_reads_of_corrected_releases_before_selection': [], 'label_access_log_at_selection': _log_digest(),
              'labels_read': False, 'model_select_used': False,
              'disclosure': NS.DISCLOSURE, 'code_sha256': {'tsz/select.py': C.sha(Path(__file__)), 'tsz/nested.py': C.sha(Path(NS.__file__))},
              'interpreter': sys.executable, 'cpu_s': time.process_time() - t0, 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV)}
    C.write_json_x(out, record)
    print(json.dumps({'version': v, 'selected': sel, 'flags': {a: results[a].get('flags') for a in SELECTION_ARMS}}), flush=True)
    return record


# ------------------------------------------------------------------ selftest (invented tables only; the real frozen rule)
def _invented_nested(root, v, arm, strengths, est_fn, se_fn, n=200):
    rng = np.random.default_rng(int(hashlib.sha256(f'{v}|{arm}'.encode()).hexdigest()[:8], 16))
    root.mkdir(parents=True, exist_ok=True)
    npz_sha = C.save_npz_x(root / f'NESTED_{arm}.npz', witness_oof=rng.dirichlet(np.ones(5), size=n), delta_oof=rng.normal(size=(n, 5)))
    per = {}
    for s in strengths:
        per[CR.skey(s)] = {'stats': {j: est_fn(s, j) for j in STATS}, 'se': {j: se_fn(s, j) for j in STATS}, 'n': n}
    rec = {'schema': 'TSZ1-nested-v1', 'version': v, 'arm': arm, 'strengths': list(strengths), 'strength_source': 'selftest',
           'per_strength': per, 'npz_sha256': npz_sha, 'folds': {}, 'disclosure': NS.DISCLOSURE}
    C.write_json_x(root / f'NESTED_{arm}.json', rec)


def _expected(est_fn, se_fn, strengths, allow, z=2.0):
    sel = 0.0
    for s in [x for x in strengths if x > 0]:
        if all(est_fn(s, j) + z * se_fn(s, j) <= allow[j] for j in STATS):
            sel = s
        else:
            break
    return sel


def selftest():
    t0 = time.process_time()
    RU = importlib.import_module('tsz.rules')
    S = tuple(float(s) for s in RU.STRENGTHS)
    checks = {}
    tmp = Path(tempfile.mkdtemp(prefix='tsz_select_selftest_'))
    for v in C.VERSIONS:
        allow = RU.allowances()[v]
        # LOC: costs grow linearly in s and cross the allowance between two registered strengths; PAIR: fails at once (identity)
        loc_est = lambda s, j, a=allow: 0.9 * a[j] * s / 0.25
        loc_se = lambda s, j, a=allow: 0.02 * a[j] * (s > 0)
        pair_est = lambda s, j, a=allow: (2.0 * a[j] if s > 0 else 0.0)
        pair_se = lambda s, j: 0.0
        root = tmp / f'V{v}'
        _invented_nested(root, v, 'LOC', S, loc_est, loc_se)
        _invented_nested(root, v, 'PAIR', S, pair_est, pair_se)
        rec = run(v, nested_root=root, out_dir=tmp / 'out', RU=RU, events=[])
        exp = _expected(loc_est, loc_se, S, allow)
        checks[f'V{v}_loc_selected_equals_independent_prefix_rule'] = {'selected': rec['selected']['LOC'], 'expected': exp,
                                                                        'pass': rec['selected']['LOC'] == exp and 0.0 < exp < 1.0}
        checks[f'V{v}_pair_identity_when_smallest_fails'] = rec['selected']['PAIR'] == 0.0 and rec['identity']['PAIR']
        checks[f'V{v}_cont_matched'] = rec['cont_matched']['CONT@LOC'] == exp and rec['cont_matched']['CONT@PAIR'] == 0.0
        checks[f'V{v}_nested_sha_recorded'] = rec['nested_tables']['LOC']['npz_sha256'] == json.loads((root / 'NESTED_LOC.json').read_text())['npz_sha256']
        try:
            run(v, nested_root=root, out_dir=tmp / 'out', RU=RU, events=[]); checks[f'V{v}_write_once'] = False
        except PermissionError:
            checks[f'V{v}_write_once'] = True
    # pass-after-fail: a later passing strength is never selected
    allow = RU.allowances()[0]
    nm = lambda s, j, a=allow: (2.0 * a[j] if abs(s - 0.1) < 1e-12 else 0.1 * a[j] * (s > 0))
    root = tmp / 'PAF'
    _invented_nested(root, 0, 'LOC', S, nm, lambda s, j: 0.0)
    _invented_nested(root, 0, 'PAIR', S, nm, lambda s, j: 0.0)
    rec = run(0, nested_root=root, out_dir=tmp / 'out_paf', RU=RU, events=[])
    checks['pass_after_fail_never_selected'] = rec['selected']['LOC'] == 0.03 and 'PASS_AFTER_FAIL' in rec['audit']['LOC']['flags']
    # refusals: a MODEL_SELECT read of a corrected release before selection; a table on another strength set; a tampered npz
    ev = [{'role': 'MODEL_SELECT', 'entry_point': 'tsz.evaluate:score', 'operation': 'benchmark', 'purpose': 'realized costs of the LOC release'}]
    try:
        run(1, nested_root=tmp / 'V1', out_dir=tmp / 'out_refuse', RU=RU, events=ev); checks['refuses_after_ms_read_of_correction'] = False
    except PermissionError:
        checks['refuses_after_ms_read_of_correction'] = True
    unrelated = ['task-stage benchmark of W100_V0', 'paired contrasts 50->100', 'uncorrected witness; correctness check',
                 'development reader grid TASK seed 0 (pair view; reader FIT / VALIDATION / SCORE)', 'locality control']
    checks['ignores_unrelated_ms_reads'] = corrected_ms_reads([{'role': 'MODEL_SELECT', 'entry_point': 'tsz.evaluate:ms', 'purpose': p}
                                                               for p in unrelated]) == []
    flagged = ['realized costs of the corrected releases', 'correction stage readers', 'reader grid CONT@0.1 V0', 'PAIR release',
               'Correction V1']
    checks['flags_corrected_release_reads'] = len(corrected_ms_reads([{'role': 'MODEL_SELECT', 'entry_point': 'tsz.devreaders:run',
                                                                       'purpose': p} for p in flagged])) == len(flagged)
    root = tmp / 'BADSET'
    _invented_nested(root, 0, 'LOC', (0.0, 0.1, 1.0), loc_est, loc_se)
    _invented_nested(root, 0, 'PAIR', S, pair_est, pair_se)
    try:
        run(0, nested_root=root, out_dir=tmp / 'out_bad', RU=RU, events=[]); checks['refuses_other_strength_set'] = False
    except AssertionError:
        checks['refuses_other_strength_set'] = True
    root = tmp / 'TAMPER'
    _invented_nested(root, 0, 'LOC', S, loc_est, loc_se)
    _invented_nested(root, 0, 'PAIR', S, pair_est, pair_se)
    with open(root / 'NESTED_LOC.npz', 'ab') as f:
        f.write(b'x')
    try:
        run(0, nested_root=root, out_dir=tmp / 'out_tamper', RU=RU, events=[]); checks['refuses_tampered_npz'] = False
    except PermissionError:
        checks['refuses_tampered_npz'] = True
    tab = rule_table({'per_strength': {'0.0': {'stats': {j: 0.0 for j in STATS}, 'se': {j: 0.0 for j in STATS}}}})
    checks['adapter_shape'] = set(tab['0.0']) == set(STATS) and set(tab['0.0']['brier']) == {'estimate', 'se'}
    ok = all((x['pass'] if isinstance(x, dict) else bool(x)) for x in checks.values())
    print(json.dumps({'selftest': 'PASS' if ok else 'FAIL', 'module': 'tsz.select', 'checks': checks, 'rule': 'tsz.rules.select_strength (real)',
                      'data': 'invented cost tables only; no label read', 'cpu_s': round(time.process_time() - t0, 2)}, default=str))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    x = sub.add_parser('run'); x.add_argument('--version', type=int, required=True, choices=C.VERSIONS)
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        return selftest()
    run(a.version)
    return 0


if __name__ == '__main__':
    sys.exit(main())
