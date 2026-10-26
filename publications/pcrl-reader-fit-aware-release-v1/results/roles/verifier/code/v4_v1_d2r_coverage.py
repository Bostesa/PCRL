#!/usr/bin/env python3
"""RFA1 verifier, wave 4: closes the coverage gap of the wave-3 v1 D2R check (label-free).

Wave 3 (unit RFA_VER_W3_MAIN_v1) compared the verifier's own per-comparator D2R reading with claims/RFA_v1_D2R/D2R.json only for the
8 members of the robustness bank RFA1_ROB; the record's 5 MAIN-only references (B2TWIN, TR100, TT100, TWJ, W; NOT_RUN by
construction) were not compared. Here the verifier's saved entries are merged as in wave 4 (c) (the union of comparators: RFA1_MAIN
registered cells from out/V2_GRID.json, RFA1_ROB cells from out/V3_MAIN.json), and (i) rfa.rules.d2r unchanged and (ii) the
verifier's own vagg2.d2r_comparator are compared with the record for all 13 comparators.

    <repository>/.venv/bin/python results/pcrl_reader_fit_aware_release_v1/roles/verifier/code/v4_v1_d2r_coverage.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT))
import vagg2 as B  # noqa: E402
from rfa import rules as RU  # noqa: E402

RES = ROOT / 'results' / 'pcrl_reader_fit_aware_release_v1'
OUT = HERE.parent / 'out' / 'V4_V1_D2R_COVERAGE.json'
TOL = 1e-12


def main():
    t0 = time.process_time()
    main_ent = json.loads((HERE.parent / 'out' / 'V2_GRID.json').read_text())['inputs']['entries']
    rob_ent = json.loads((HERE.parent / 'out' / 'V3_MAIN.json').read_text())['inputs']['entries']['RFA1_ROB']
    reg = {f'S{v}R{r}k{k}' for v in (0, 1) for r in (1, 2) for k in (0, 1)}
    merged = {}
    for comp in sorted(set(main_ent) | set(rob_ent)):
        merged[comp] = {}
        for view in ('pair', 'income', 'occupation'):
            a = (main_ent.get(comp) or {}).get(view) or {'cells': {}, 'cell_se': {}}
            b = (rob_ent.get(comp) or {}).get(view) or {'cells': {}, 'cell_se': {}}
            rob_cells = {c: x for c, x in b['cells'].items() if c not in reg}
            rob_se = {c: x for c, x in b['cell_se'].items() if c not in reg}
            same_reg = all(abs(b['cells'][c] - a['cells'][c]) <= TOL for c in b['cells'] if c in reg and c in a['cells'])
            merged[comp][view] = {'cells': {**a['cells'], **rob_cells}, 'cell_se': {**a['cell_se'], **rob_se}, '_reg_overlap_equal': same_reg}
    D = json.loads((RES / 'claims' / 'RFA_v1_D2R' / 'D2R.json').read_text())
    rec = D['d2r']
    clean = {c: {vw: {k: x for k, x in e.items() if not k.startswith('_')} for vw, e in d.items()} for c, d in merged.items()}
    fro = RU.d2r({'grid_valid': True, 'release': 'RF_FULL', 'refit': D['refit_effectiveness'], 'comparators': D['comparators'], 'rob': clean})
    refit_ok = (D['refit_effectiveness'] or {}).get('label') == 'REFIT_GENUINELY_EFFECTIVE'
    own = {c: B.d2r_comparator(clean[c], refit_ok) for c in clean}
    chk = {
        'record_has_13_comparators': len(rec['per_comparator']) == 13,
        'registered_cells_of_ROB_equal_MAIN (overlap)': all(e['_reg_overlap_equal'] for d in merged.values() for e in d.values()),
        'frozen_label_equals_record': fro['label'] == rec['label'],
        'frozen_per_comparator_labels_equal_record_all_13': all(fro['per_comparator'][c]['label'] == p['label'] for c, p in rec['per_comparator'].items()),
        'frozen_sign_changes_and_failures_equal_record': sorted(fro['sign_changes']) == sorted(rec['sign_changes'])
        and sorted(fro['recipient_failures']) == sorted(rec['recipient_failures']),
        'own_per_comparator_labels_equal_record_all_13': all(own[c]['label'] == p['label'] for c, p in rec['per_comparator'].items()),
        'own_NOT_RUN_exactly_the_5_MAIN_only_references': sorted(c for c in rec['per_comparator'] if own[c]['label'] == 'NOT_RUN')
        == ['B2TWIN', 'TR100', 'TT100', 'TWJ', 'W'],
        'own_robustness_means_equal_record (abs <= 1e-12)': all(abs(own[c]['robustness_mean'] - p['robustness_mean']) <= TOL
                                                                 for c, p in rec['per_comparator'].items() if p.get('robustness_mean') is not None),
    }
    res = {'schema': 'RFA1-verifier-v1-d2r-coverage-v4', 'checks': chk, 'all_pass': all(chk.values()),
           'own_labels': {c: own[c]['label'] for c in sorted(own) if c in rec['per_comparator']},
           'frozen': {'label': fro['label'], 'sign_changes': fro['sign_changes'], 'recipient_failures': fro['recipient_failures']},
           'code_sha256': {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in ('v4_v1_d2r_coverage.py', 'vagg2.py')},
           'cpu_s': time.process_time() - t0}
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'all_pass': res['all_pass'], 'failed': [k for k, x in chk.items() if not x], 'own_labels': res['own_labels'],
                      'cpu_s': res['cpu_s']}, indent=1))
    return 0 if res['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
