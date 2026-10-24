"""Investigator A, R2 design, label-free check (no label file opened; no SEX used; nothing read from MODEL_SELECT).

The lambda = 0 fits of FA_TFRO_R1 read neither task labels nor SEX (their task term is distillation toward the frozen HGB teacher's
out-of-fold probabilities). So they can be recomputed now from admitted MODEL_FIT / CALIBRATION features and frozen arrays:
  1. bitwise reproduction of R1's saved FA_TFRO_R1_CONT occupation vectors on MODEL_FIT and CALIBRATION (the reuse premise);
  2. the lambda = 0 cross-fit bag (mean of the 5 fold residuals) against the lambda = 0 full fit (the matched-control premise of R2):
     mean KL and max |difference| of the occupation vectors on CALIBRATION and MODEL_FIT, beside the size of the full movement
     (KL from SUCC to the lambda = 0 full fit at s = 1).
Run: OMP_NUM_THREADS=1 <repository>/.venv/bin/python results/pcrl_strong_anchor_release_v1/roles/investigator_a/r2_check_lambda0_labelfree.py
"""
import json
import os
import sys
import time
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from sar import common as C, famkit as FK, famrun as FR, release as RL  # noqa: E402
from sar.fa import tfr as V1, tfro as R1  # noqa: E402


def kl(P, Q):
    P, Q = np.clip(P, 1e-300, None), np.clip(Q, 1e-300, None)
    return float(np.mean(np.sum(P * (np.log(P) - np.log(Q)), 1)))


t0 = time.process_time()
out = {'utc': C.now(), 'labels_read': 'none (lambda = 0 fits read no label and no SEX)', 'versions': {}}
for v in (0, 1):
    M = C.features('MODEL_FIT')
    Cf = C.features('CALIBRATION')
    T, _ = V1.task_cache(v)
    sm = V1._succ(v, M['X'], M['feature_names'])
    sc = V1._succ(v, Cf['X'], Cf['feature_names'])
    A = V1.blocks(FK.compact(M['X'], M['feature_names']), (sm['income'], sm['occupation']), (T['oof']['inc'], T['oof']['occ']))
    from nrm.fa import prc
    Zc = FK.compact(Cf['X'], Cf['feature_names'])
    A_cal = V1.blocks(Zc, (sc['income'], sc['occupation']), (prc._proba(T['models']['inc'], Zc), prc._proba(T['models']['occ'], Zc)))
    unit = np.asarray(M['unit'])
    folds = FR.folds(unit, 'FA', v)
    full0 = R1.train(A, None, 0.0, R1._key(v, unit, 'rho0|full'))
    rel_full_mf = R1.release_state(full0, A, sm['income'])[1]
    rel_full_cal = R1.release_state(full0, A_cal, sc['income'])[1]
    saved, _ = RL.load('FA', 'FA_TFRO_R1_CONT', v, roles=('MODEL_FIT', 'CALIBRATION'))
    rec = {'reproduces_R1_CONT_bitwise': {'MODEL_FIT': bool(np.array_equal(saved['MODEL_FIT']['occupation'], rel_full_mf)),
                                          'CALIBRATION': bool(np.array_equal(saved['CALIBRATION']['occupation'], rel_full_cal))}}
    # the lambda = 0 cross-fit bag: mean of the 5 fold residuals (each fold fit with R1's fold keys)
    outs_cal, outs_mf = [], []
    for j in range(FR.N_XF):
        tr = folds != j
        st = R1.train(A[tr], None, 0.0, R1._key(v, unit[tr], f'rho0|f{j}'))
        W = st['checkpoints'][st['keep']]
        outs_cal.append(V1.net_numpy(W, V1.phi_apply(st['phi'], A_cal)))
        outs_mf.append(V1.net_numpy(W, V1.phi_apply(st['phi'], A)))
    bag_cal = R1.release_occ(A_cal, np.mean(outs_cal, 0), 1.0)
    bag_mf = R1.release_occ(A, np.mean(outs_mf, 0), 1.0)
    rec['bag_vs_full'] = {'CALIBRATION': {'mean_kl': kl(rel_full_cal, bag_cal), 'max_abs': float(np.max(np.abs(rel_full_cal - bag_cal)))},
                          'MODEL_FIT': {'mean_kl': kl(rel_full_mf, bag_mf), 'max_abs': float(np.max(np.abs(rel_full_mf - bag_mf)))}}
    rec['full_movement_kl_from_SUCC'] = {'CALIBRATION': kl(rel_full_cal, sc['occupation']), 'MODEL_FIT': kl(rel_full_mf, sm['occupation'])}
    rec['fold_spread_kl_mean_to_bag_CAL'] = float(np.mean([kl(R1.release_occ(A_cal, o, 1.0), bag_cal) for o in outs_cal]))
    out['versions'][f'V{v}'] = rec
out['cpu_s'] = round(time.process_time() - t0, 1)
p = Path(__file__).with_name('r2_check_lambda0_labelfree.json')
p.write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
print(json.dumps(out, indent=1, sort_keys=True))
