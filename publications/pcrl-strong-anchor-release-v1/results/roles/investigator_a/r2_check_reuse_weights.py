"""Investigator A, R2 design, label-free: R2's recomputed lambda = 0 full fit (R1's key) against the trained weights stored in R1's saved
lambda = 0 models (FA_TFRO_R1_CONT and FA_TFRO_R1_TASK_LO), bitwise. No label, no SEX (lambda = 0), nothing from MODEL_SELECT.
Run: OMP_NUM_THREADS=1 <repository>/.venv/bin/python results/pcrl_strong_anchor_release_v1/roles/investigator_a/r2_check_reuse_weights.py
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

from sar import common as C, famkit as FK  # noqa: E402
from sar.fa import tfr as V1, tfro as R1, tfrb as R2  # noqa: E402

t0 = time.process_time()
out = {'utc': C.now(), 'labels_read': 'none', 'versions': {}}
M = C.features('MODEL_FIT')
for v in (0, 1):
    T, _ = V1.task_cache(v)
    sm = V1._succ(v, M['X'], M['feature_names'])
    A = V1.blocks(FK.compact(M['X'], M['feature_names']), (sm['income'], sm['occupation']), (T['oof']['inc'], T['oof']['occ']))
    full0 = R1.train(A, None, 0.0, R1._key(v, np.asarray(M['unit']), 'rho0|full'))
    W0 = full0['checkpoints'][full0['keep']]
    out['versions'][f'V{v}'] = {cfg: R2.weights_equal(W0, FK.load_model('FA', cfg, v)['net']) for cfg in R2.r1_saved_points(v).values()}
out['cpu_s'] = round(time.process_time() - t0, 1)
Path(__file__).with_name('r2_check_reuse_weights.json').write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
print(json.dumps(out, indent=1, sort_keys=True))
