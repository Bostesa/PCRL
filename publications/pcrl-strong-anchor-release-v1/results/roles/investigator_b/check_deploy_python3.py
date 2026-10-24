"""Label-free check (investigator B): the FB deployment path imports and runs under python3 (numpy only, no torch), for a plain
network payload and for a CTRL (DQ) payload, on admitted MODEL_FIT features (features only) with invented weights in a temporary
model directory. Run from the worktree root:  python3 results/pcrl_strong_anchor_release_v1/roles/investigator_b/check_deploy_python3.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import numpy as np  # noqa: E402

from sar import common as C  # noqa: E402
from sar import ctlkit as CK  # noqa: E402
from sar import deploy  # noqa: E402
from sar import famkit as FK  # noqa: E402
from sar.fb import pfd as P  # noqa: E402

m = deploy.family_module('FB')
rng = np.random.default_rng(0)
F = C.features('MODEL_FIT')
X = F['X'][:300]
W = {}
for t, k in (('income', 2), ('occupation', 5)):
    shapes = [(64, X.shape[1]), (64, 64), (16, 64), (k, 16)]
    W[t] = [((rng.normal(size=s) * 0.3).astype(np.float32), (rng.normal(size=s[0]) * 0.1).astype(np.float32)) for s in shapes]
bI, bO = P.release_weights(W, np.asarray(X, dtype=np.float64))
basis = CK.dq_fit(bI, bO)
old = C.RUNS
tmp = Path(tempfile.mkdtemp(prefix='fb_deploy_py3_'))
try:
    C.RUNS = tmp
    FK.write_model('FB', 'FB_PFD', 0, {'feature_names': tuple(F['feature_names']), 'op': 'net', 'weights': W})
    FK.write_model('FB', 'FB_PFD_DQ_HI', 0, {'feature_names': tuple(F['feature_names']), 'op': 'dq', 'weights': W, 'dq_basis': basis,
                                             'delta': 0.5, 'dq_tag': 'TEST'})
    FK.write_model('FB', 'FB_PFO_DQ_HI', 0, {'feature_names': tuple(F['feature_names']), 'op': 'dq_occ', 'weights': W, 'dq_basis': basis,
                                             'delta': 0.5, 'dq_tag': 'TEST'})
    net = m.predict('FB_PFD', 0, X, F['feature_names'])
    dq = m.predict('FB_PFD_DQ_HI', 0, X, F['feature_names'])
    dq2 = m.predict('FB_PFD_DQ_HI', 0, X, F['feature_names'])
    dq_occ = m.predict('FB_PFO_DQ_HI', 0, X, F['feature_names'])           # revision R1 (sar.fb.pfo via the package dispatch)
    paths = [str(p.relative_to(tmp)) for p in m.model_paths('FB_PFD_DQ_HI', 0)]
finally:
    C.RUNS = old
want = CK.dq_release(basis, bI, bO, FK.compact(np.asarray(X, dtype=np.float64), F['feature_names']), 0.5, 'TEST')
r1_ok = bool(np.array_equal(dq_occ['income'], bI) and np.array_equal(dq_occ['occupation'], want[1]))
out = {'deploy_module': m.__name__, 'has_predict': hasattr(m, 'predict'), 'has_model_paths': hasattr(m, 'model_paths'),
       'net_equals_forward_bitwise': bool(np.array_equal(net['income'], bI) and np.array_equal(net['occupation'], bO)),
       'dq_equals_shared_law_bitwise': bool(np.array_equal(dq['income'], want[0]) and np.array_equal(dq['occupation'], want[1])),
       'dq_deterministic': bool(np.array_equal(dq['occupation'], dq2['occupation'])),
       'sums_to_one': bool(all(np.allclose(r[t].sum(1), 1, atol=1e-9, rtol=0) for r in (net, dq) for t in ('income', 'occupation'))),
       'r1_dq_occ_income_passed_through_and_occupation_shared_law_bitwise': r1_ok,
       'model_paths': paths, 'torch_imported': 'torch' in sys.modules, 'python': sys.version.split()[0]}
out['all_ok'] = all(out[k] for k in ('has_predict', 'has_model_paths', 'net_equals_forward_bitwise', 'dq_equals_shared_law_bitwise',
                                     'dq_deterministic', 'sums_to_one',
                                     'r1_dq_occ_income_passed_through_and_occupation_shared_law_bitwise')) and not out['torch_imported']
print(json.dumps(out))
