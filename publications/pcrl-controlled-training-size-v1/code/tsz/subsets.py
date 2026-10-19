"""Label-free manifest of the training subsets and the nested cross-fitting folds of the PCRL controlled training-size campaign.

    python3 -m tsz.subsets manifest      writes results/pcrl_controlled_training_size_v1/SUBSETS.json (no label is read)

Definitions (frozen in the registration; tsz.common): four household orderings O0..O3 of MODEL_FIT (households sorted by
SHA-256('TSZ1_order_<o>|<household id>')); nested subsets S50(o) = the first ceil(0.5 G) households, S70(o) = the first ceil(0.7 G),
S100 = all G = 13,337 households (identical for every ordering: one fit per version, aliased); whole households always. Nested
strength-selection folds of the conditional stage: ohn.common.household_folds(MODEL_FIT units, 10, 'TSZ1_ncv10'). Fold functions
inherited from the source recipes (A1_PO_16 inner CV 'OHN_A_cv_S<v>', cross-fitting 'OHN1_crossfit') are computed on all MODEL_FIT
units and restricted to a subset's rows (the source's sub-pipeline convention).
"""
from __future__ import annotations
import hashlib
import json
import sys

import numpy as np

from . import common as C


def manifest():
    from ohn import common as OC
    F = C.features('MODEL_FIT')
    unit = F['unit']
    problems = C.check_subsets(unit)
    summ = C.subset_summary(unit)
    nested = {}
    for o in C.ORDERINGS:
        a, b = C.subset_mask(unit, f'S50_{o}'), C.subset_mask(unit, f'S70_{o}')
        nested[o] = {'S50_in_S70': bool(not (a & ~b).any()), 'S70_in_S100': True,
                     'added_50_to_70_people': int((b & ~a).sum()), 'added_70_to_100_people': int((~b).sum())}
    f10 = C.ncv_folds(unit)
    ncv = {'tag': C.NCV_TAG, 'k': C.NCV_K, 'people_per_fold': np.bincount(f10).tolist(),
           'households_per_fold': [int(len(np.unique(unit[f10 == k]))) for k in range(C.NCV_K)],
           'training_people_per_outer_fold': [int((f10 != k).sum()) for k in range(C.NCV_K)]}
    inner = {}
    for v in C.VERSIONS:
        fi = OC.household_folds(unit, 5, f'OHN_A_cv_S{v}')
        inner[f'V{v}'] = {sid: np.bincount(fi[C.subset_mask(unit, sid)], minlength=5).tolist() for sid in C.subset_ids()}
    rec = {'schema': 'TSZ1-subsets-v1', 'utc': C.now(), 'labels_read': False, 'problems': problems,
           'definition': {'orderings': list(C.ORDERINGS), 'order_tag': f'{C.ORDER_TAG}_<o>', 'fractions_percent': list(C.FRACTIONS),
                          'S50_households': C.n_households(summ['model_fit_households'], 50),
                          'S70_households': C.n_households(summ['model_fit_households'], 70),
                          'S100_households': summ['model_fit_households'],
                          'unit': 'whole households; nested within each ordering; S100 identical for every ordering (one fit per version)'},
           **summ, 'nesting': nested, 'nested_selection_folds': ncv,
           'a1_inner_fold_people_per_subset': inner,
           'row_set_sha256': {sid: C.arr_sha(np.asarray(F['row_id'][C.subset_mask(unit, sid)])) for sid in C.subset_ids()},
           'model_fit_unit_sha256': hashlib.sha256(np.ascontiguousarray(unit).tobytes()).hexdigest()}
    C.write_json(C.RESULTS / 'SUBSETS.json', rec)
    print(json.dumps({'problems': problems, 'subsets': {k: (v['people'], v['households']) for k, v in summ['subsets'].items()}}))
    return 0 if not problems else 1


if __name__ == '__main__':
    if sys.argv[1:] != ['manifest']:
        raise SystemExit(__doc__)
    sys.exit(manifest())
