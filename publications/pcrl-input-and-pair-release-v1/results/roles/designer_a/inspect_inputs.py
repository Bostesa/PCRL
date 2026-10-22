"""Designer A (campaign pcrl_input_and_pair_release_v1): label-free inspection of the admitted MODEL_FIT features and the two cached
out-of-fold prediction files named in the design brief. Opens no label file, nothing of MODEL_SELECT, no reader cache."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[k] = '1'
import sys, time, pickle, json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, ROOT)
t0 = time.process_time()
from ddr import common as C
from ohn import fam_a as FA

F = C.features('MODEL_FIT')
names = tuple(map(str, F['feature_names']))
Z = FA.compact_po(F['X'], names)
out = {'n': int(len(Z)), 'households': int(len(np.unique(F['unit'])))}
for j, nm in enumerate(('AGEP', 'WKHP', 'COW', 'SCHL', 'MAR', 'POBP')):
    v, c = np.unique(Z[:, j], return_counts=True)
    out[nm] = {'levels': int(len(v)), 'min': float(v.min()), 'max': float(v.max())}
    if len(v) <= 12:
        out[nm]['counts'] = {str(a): int(b) for a, b in zip(v, c)}
for b in ('COW', 'MAR'):
    out[b + '_names'] = [n for n in names if n.startswith(b + '=')]
po = Z[:, 5]
out['POBP_bands'] = {'us_lt60': int((po < 60).sum()), 'fl_12': int((po == 12).sum()), 'foreign_ge100': int((po >= 100).sum()),
                     'other_999': int((po == 999).sum())}
v, c = np.unique(po[po >= 60], return_counts=True)
order = np.argsort(-c)[:15]
out['POBP_top_foreign'] = {str(v[i]): int(c[i]) for i in order}
for v_ in (0, 1):
    with open(os.path.join(ROOT, f'.private/pcrl_new_release_mechanisms_v1/fa_cache/V{v_}/TASK.pkl'), 'rb') as f:
        T = pickle.load(f)
    with open(os.path.join(ROOT, f'.private/pcrl_finite_routing_comparison_v1/fa_cache/V{v_}/SEXSIDE.pkl'), 'rb') as f:
        S = pickle.load(f)
    out[f'V{v_}'] = {'task_keys': sorted(T.keys()), 'T': {k: int(x) for k, x in T['T'].items()},
                     'oof_shapes': {k: list(np.shape(x)) for k, x in T['oof'].items()},
                     'sex_keys': sorted(S.keys()), 'sex_oof': sorted(S['oof'].keys()),
                     'row_sha_match': bool(T['row_id_sha256'] == C.array_sha(np.asarray(F['row_id'])) if hasattr(C, 'array_sha') else None),
                     'pi_mean': float(S['oof']['sex'][:, 1].mean()), 'pi_q': [float(q) for q in np.percentile(S['oof']['sex'][:, 1], [5, 25, 50, 75, 95])],
                     'feature_names_len': int(len(T['feature_names']))}
out['cpu_s'] = round(time.process_time() - t0, 2)
print(json.dumps(out, indent=1))
