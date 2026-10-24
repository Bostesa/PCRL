"""Investigator A label-free CPU check of sar.fa.tfr at MODEL_FIT size (no label file opened; no configuration saved).

The packed array uses admitted MODEL_FIT features, SUCC's frozen outputs and the admitted HGB teachers' frozen out-of-fold outputs.
The task term of the fit is label-free (distillation toward the teachers' probabilities); the critic run uses INVENTED SEX (random),
for timing only. Two epochs per run, extrapolated to the recipe's 20.
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
from sar.fa import tfr  # noqa: E402

t0 = time.process_time()
M = C.features('MODEL_FIT')
s = tfr._succ(0, M['X'], M['feature_names'])
T, _ = tfr.task_cache(0)
A = tfr.blocks(FK.compact(M['X'], M['feature_names']), (s['income'], s['occupation']), (T['oof']['inc'], T['oof']['occ']))
t_prep = time.process_time() - t0
n_tr = int(0.8 * len(A))
fake_sex = np.random.default_rng(0).integers(0, 2, n_tr)          # invented, timing only
r0 = tfr.train(A[:n_tr], None, 0.0, 'cpu-check-task', epochs=2)
r1 = tfr.train(A[:n_tr], fake_sex, 1.0, 'cpu-check-crit', epochs=2)
p_t, o_t = tfr.release_state(r0, A[:n_tr])


def kl(P, Q):
    P, Q = np.clip(P, 1e-300, None), np.clip(Q, 1e-300, None)
    return float(np.mean(np.sum(P * (np.log(P) - np.log(Q)), 1)))


out = {'utc': C.now(), 'labels_read': 'none (SEX invented for the critic timing run)', 'rows_per_fold_fit': n_tr,
       'prep_cpu_s': round(t_prep, 1), 'task_only_cpu_s_per_epoch': round(r0['info']['cpu_s'] / 2, 2),
       'critic_cpu_s_per_epoch': round(r1['info']['cpu_s'] / 2, 2),
       'est_cpu_s_per_20_epoch_fit': {'task_only': round(10 * r0['info']['cpu_s'], 1), 'critics': round(10 * r1['info']['cpu_s'], 1)},
       'label_free_movement_after_2_task_epochs': {
           'kl_teacher_to_succ_occ': kl(T['oof']['occ'][:n_tr], s['occupation'][:n_tr]),
           'kl_teacher_to_release_occ': kl(T['oof']['occ'][:n_tr], o_t),
           'kl_teacher_to_succ_inc': kl(T['oof']['inc'][:n_tr], s['income'][:n_tr]),
           'kl_teacher_to_release_inc': kl(T['oof']['inc'][:n_tr], p_t)},
       'total_cpu_s': None}
out['total_cpu_s'] = round(time.process_time() - t0, 1)
Path(__file__).with_name('check_cpu_labelfree.json').write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
print(json.dumps(out, indent=1, sort_keys=True))
