#!/usr/bin/env python3
"""RFA1 verifier, wave 4 (c): the R1 second split claims/RFA_R1_D2R/D2R.json recomputed from the verifier's own entries.

RFA1_R1_ROB (S0, S1; R1-R4 x k0-k3) recomputed cell by cell with the verifier's code (v3_main.recompute: winners, SCORE AUCs,
bootstrap draws, paired entries vs GRID.json / BOOT.npz), merged with the verifier's RFA1_R1_MAIN entries (out/V3_MAIN.json, unit
RFA_VER_W3_MAIN_v1), then (i) rfa.rules.d2r unchanged on the merged entries with the record's comparators and refit-effectiveness
label, (ii) the verifier's own per-comparator reading (vagg2.d2r_comparator); both compared with the record.
SEX through rfa.common.load(..., 'verify') from this __main__, in a queue unit.

    python3 -m rfa.jobqueue run --unit RFA_VER_W4_R1ROB_v1 --stage verify --rss-gib 3 -- \\
        <repository>/.venv/bin/python results/pcrl_reader_fit_aware_release_v1/roles/verifier/code/v4_r1_rob.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT))

import vagg2 as B  # noqa: E402
import v3_main as M3  # noqa: E402

OUT = HERE.parent / 'out' / 'V4_R1_D2R.json'
RES = ROOT / 'results' / 'pcrl_reader_fit_aware_release_v1'
TOL = 1e-12


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if not os.environ.get('RFA_UNIT_ID'):
        sys.exit('refused: labels are read only inside an rfa.jobqueue verify unit')
    t0 = time.process_time()
    from rfa import common as C
    S = C.load('MODEL_SELECT', ('features', 'sex'), 'verify',
               'verify: wave 4 recomputation of the R1 second-split grid RFA1_R1_ROB and the R1 D2R reading (verifier wave 4)')
    S.pop('X', None)
    sex, unit, row_id = S['sex'], S['unit'], S['row_id']
    res = {'schema': 'RFA1-verifier-r1-d2r-v4', 'unit_id': os.environ.get('RFA_UNIT_ID'), 'checks': {}}
    chk = res['checks']
    pv = {v: M3.recompute('RFA1_R1_ROB', 'RF1_FULL', v, sex, unit, row_id) for v in (0, 1)}
    for v in (0, 1):
        chk[f'grid|RFA1_R1_ROB|S{v}|reproduced_exactly'] = (not pv[v]['mismatches'] and pv[v]['provenance_ok'] and
                                                         all(x <= TOL for x in pv[v]['worst'].values()))
    res['grid'] = {str(v): {k: x for k, x in pv[v].items() if k not in ('paired', 'nontable')} for v in (0, 1)}
    rob_new = M3.merge(pv)
    main_ent = json.loads((HERE.parent / 'out' / 'V3_MAIN.json').read_text())['inputs']['entries']['RFA1_R1_MAIN']
    # run v1 of this unit (RFA_VER_W4_R1ROB_v1) merged only the ROB members and crashed (KeyError B2TWIN); like rfa.rfclaims.d2r,
    # the merged grid holds every comparator: ROB members get registered + robustness cells, MAIN-only references registered cells only
    rob = {}
    for comp in sorted(set(main_ent) | set(rob_new)):
        rob[comp] = {}
        for view in ('pair', 'income', 'occupation'):
            reg = (main_ent.get(comp) or {}).get(view) or {'cells': {}, 'cell_se': {}}
            e = (rob_new.get(comp) or {}).get(view) or {'cells': {}, 'cell_se': {}}
            rob[comp][view] = {'cells': {**reg['cells'], **e['cells']}, 'cell_se': {**reg['cell_se'], **e['cell_se']}}
    D = json.loads((RES / 'claims' / 'RFA_R1_D2R' / 'D2R.json').read_text())
    rec_d = D['d2r']
    from rfa import rules as RU
    mine_frozen = RU.d2r({'grid_valid': True, 'release': 'RF1_FULL', 'refit': D['refit_effectiveness'], 'comparators': D['comparators'],
                          'rob': rob})
    chk['frozen_d2r_on_verifier_entries_label_equals_record'] = mine_frozen['label'] == rec_d['label']
    chk['frozen_d2r_per_comparator_labels_equal_record'] = all(
        mine_frozen['per_comparator'][c]['label'] == p['label'] for c, p in rec_d['per_comparator'].items())
    chk['frozen_d2r_sign_changes_and_recipient_failures_equal_record'] = (sorted(mine_frozen['sign_changes']) == sorted(rec_d['sign_changes']) and
                                                                         sorted(mine_frozen['recipient_failures']) == sorted(rec_d['recipient_failures']))
    refit_ok = (D['refit_effectiveness'] or {}).get('label') == 'REFIT_GENUINELY_EFFECTIVE'
    own = {c: B.d2r_comparator(rob[c], refit_ok) for c in rob}
    agree, means = {}, {}
    for c, p in rec_d['per_comparator'].items():
        if c in own:
            agree[c] = own[c]['label'] == p['label']
            if p.get('robustness_mean') is not None:
                means[c] = (p['robustness_mean'], own[c]['robustness_mean'])
        else:
            agree[c] = p['label'] == 'NOT_RUN'                  # not a member of the robustness bank: NOT_RUN in the record
    chk['own_per_comparator_labels_equal_record'] = all(agree.values())
    chk['own_robustness_means_equal_record (abs <= 1e-12)'] = all(abs(a - b) <= TOL for a, b in means.values())
    sc_me = sorted(c for c in own if own[c].get('sign_change') and c in rec_d['per_comparator'])
    rf_me = sorted(c for c in own if own[c].get('recipient_failed') and c in rec_d['per_comparator'])
    chk['own_sign_changes_and_recipient_failures_equal_record'] = sc_me == sorted(rec_d['sign_changes']) and rf_me == sorted(rec_d['recipient_failures'])
    dec = json.loads((RES / 'claims' / 'RFA_R1' / 'DECISIONS.json').read_text())['decisions']['D2']['comparisons']
    chk['decisive_comparators_are_the_registered_R1_D2_points'] = (D['comparators']['START'] == 'NEURAL_RAW' and D['comparators']['CONT'] == 'RF1_TASK' and
                                                                   all(D['comparators'][k] == dec[k]['point'] for k in ('STOPFIT', 'ALTCRIT', 'RAWJ')))
    res['record'] = {'label': rec_d['label'], 'sign_changes': rec_d['sign_changes'], 'recipient_failures': rec_d['recipient_failures'],
                     'comparators': D['comparators'], 'refit': (D['refit_effectiveness'] or {}).get('label')}
    res['frozen_on_verifier_entries'] = {'label': mine_frozen['label'], 'sign_changes': mine_frozen['sign_changes'],
                                         'recipient_failures': mine_frozen['recipient_failures']}
    res['own_per_comparator'] = own
    res['robustness_means_record_vs_verifier'] = means
    res['all_pass'] = all(bool(x) for x in chk.values())
    res['cpu_s'] = time.process_time() - t0
    res['code_sha256'] = {f: sha256(HERE / f) for f in ('v4_r1_rob.py', 'v3_main.py', 'vagg2.py')}
    if OUT.exists():
        raise PermissionError(f'{OUT.name} exists (write-once)')
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    print(json.dumps({'all_pass': res['all_pass'], 'failed': [k for k, x in chk.items() if not x], 'cpu_s': res['cpu_s'],
                      'label': mine_frozen['label']}, indent=1))
    return 0 if res['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
