"""Challenger check (campaign pcrl_input_and_pair_release_v1): label-free support and release-identifiability counts.

Why: Designer B's exhaustion argument cites "17,744 distinct six-field records"; Designer A's PROJ control (E[SFG release | twin
release]) is only informative if the twin's released vectors do NOT already identify the person's record. Both are label-free
questions about the admitted MODEL_FIT features and the cached MODEL_FIT out-of-fold task predictions (TASK.pkl), which the
challenger brief allows.

Opens: ddr.common.features('MODEL_FIT') (features, unit; no label) and .private/pcrl_new_release_mechanisms_v1/fa_cache/V*/TASK.pkl
(label-free cached out-of-fold predictions). No label file, nothing of MODEL_SELECT, no reader cache.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'
import json
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, ROOT)
T0 = time.process_time()
from ddr import common as C                                     # noqa: E402
from ohn import fam_a as FA                                     # noqa: E402

F = C.features('MODEL_FIT')
NAMES = tuple(map(str, F['feature_names']))
Z = FA.compact_po(F['X'], NAMES)
out = {'n': int(len(Z)), 'households': int(len(np.unique(F['unit'])))}
out['distinct_six_field_records'] = int(len(np.unique(Z, axis=0)))

# how coarse are Designer A's cheap rungs: MAR3 (married | previously | never) and MAR1 (suppressed)
mar = Z[:, 4].astype(int)
out['MAR_levels'] = int(len(np.unique(mar)))
Z_mar3 = Z.copy()
Z_mar3[:, 4] = np.array([0, 1, 1, 1, 2, 0])[np.clip(mar, 0, 5)]
Z_mar1 = Z.copy()
Z_mar1[:, 4] = 0
out['distinct_after_MAR3'] = int(len(np.unique(Z_mar3, axis=0)))
out['distinct_after_MAR1'] = int(len(np.unique(Z_mar1, axis=0)))

for v in (0, 1):
    with open(os.path.join(ROOT, f'.private/pcrl_new_release_mechanisms_v1/fa_cache/V{v}/TASK.pkl'), 'rb') as f:
        T = pickle.load(f)
    pI = np.clip(np.asarray(T['oof']['inc'], np.float64), 1e-12, 1)
    pO = np.clip(np.asarray(T['oof']['occ'], np.float64), 1e-12, 1)
    W = np.column_stack([pI, pO]).round(12)
    r = {'distinct_twin_pair_wires': int(len(np.unique(W, axis=0))),
         'distinct_twin_income_wires': int(len(np.unique(pI.round(12), axis=0))),
         'distinct_twin_occupation_wires': int(len(np.unique(pO.round(12), axis=0))),
         'joint_decision_cells_used': int(len(np.unique(pI.argmax(1) * 5 + pO.argmax(1)))),
         'income_decision_share_class1': float(pI.argmax(1).mean())}
    # does the twin's released pair identify the six-field record? (out-of-fold wires; a fold model is shared by its fold's rows)
    _, inv = np.unique(W, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    _, zinv = np.unique(Z, axis=0, return_inverse=True)
    zinv = np.asarray(zinv).ravel()
    sizes = np.bincount(inv)
    r['mean_people_per_wire'] = float(sizes.mean())
    r['share_of_people_on_a_unique_wire'] = float(np.mean(sizes[inv] == 1))
    # of the people sharing a wire, how many share the same record?
    same = 0
    for g in np.flatnonzero(sizes > 1):
        m = inv == g
        same += int(len(np.unique(zinv[m])) == 1)
    r['wire_groups_gt1'] = int((sizes > 1).sum())
    r['wire_groups_gt1_that_are_one_record'] = same
    out[f'V{v}'] = r

out['cpu_s'] = round(time.process_time() - T0, 1)
print(json.dumps(out, indent=1))
with open(os.path.join(HERE, 'CHECK_SUPPORT.json'), 'w') as f:
    json.dump(out, f, indent=1)
