"""Label-free check (investigator B): the FS_CPI deployment path runs under python3 (no torch) through sar.fs, on admitted MODEL_FIT
features with the admitted SUCC files: a zero residual releases SUCC; the ALT law at SUCC's own beta and maps releases SUCC's income; the
CTRL law is the shared DQ law on SUCC's income; the occupation output is SUCC's bitwise in every case. Temporary model directory only.
Run from the worktree root:  python3 results/pcrl_strong_anchor_release_v1/roles/investigator_b/check_fs_deploy_python3.py"""
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
from sar.fs import cpi  # noqa: E402

m = deploy.family_module('FS')
Fe = C.features('MODEL_FIT')
X, nm = Fe['X'][:300], Fe['feature_names']
sI, sO = cpi.succ_arrays(0, np.asarray(X, np.float64), nm)
TF = cpi._tf()
T, _ = TF.task_cache(0)
Zc = FK.compact(np.asarray(X, np.float64), nm)
from nrm.fa import prc  # noqa: E402
A = TF.blocks(Zc, (sI, sO), (prc._proba(T['models']['inc'], Zc), prc._proba(T['models']['occ'], Zc)))
phi = TF.phi_fit(A)
d = TF.phi_apply(phi, A).shape[1]
W0 = [np.zeros((64, d)), np.zeros(64), np.zeros((64, 64)), np.zeros(64), np.zeros((1, 64)), np.zeros(1)]
st = cpi.h3_state(0)
basis = CK.dq_fit(sI, sO)
old = C.RUNS
tmp = Path(tempfile.mkdtemp(prefix='fs_deploy_py3_'))
try:
    C.RUNS = tmp
    base = {'feature_names': tuple(nm), 'frozen_sha256': TF.frozen_sha(0)}
    FK.write_model('FS', 'FS_CPI', 0, {**base, 'op': 'res', 'net': W0, 'phi': phi, 'hgb_inc': T['models']['inc'], 'hgb_occ': T['models']['occ']})
    FK.write_model('FS', 'FS_CPI_DQ_HI', 0, {**base, 'op': 'dq_inc', 'dq_basis': basis, 'delta': 0.5, 'dq_tag': 'TEST'})
    FK.write_model('FS', 'FS_CPI_BETA_LO', 0, {**base, 'op': 'alt', 'beta': float(st['beta']), 'stack_w': st['stack_w'],
                                                'platt_w': st['platt_w'], 'h3_config': st['config']})
    res, dq, alt = (m.predict(c, 0, X, nm) for c in ('FS_CPI', 'FS_CPI_DQ_HI', 'FS_CPI_BETA_LO'))
    paths = [str(p.relative_to(tmp)) for p in m.model_paths('FS_CPI', 0)]
finally:
    C.RUNS = old
want = CK.dq_release(basis, sI, sO, Zc, 0.5, 'TEST')[0]
out = {'deploy_module': m.__name__, 'occupation_is_succ_bitwise': all(np.array_equal(r['occupation'], sO) for r in (res, dq, alt)),
       'res_zero_residual_max_abs_diff': float(np.abs(res['income'] - sI).max()),
       'alt_at_succ_beta_max_abs_diff': float(np.abs(alt['income'] - sI).max()),
       'dq_inc_is_shared_law_bitwise': bool(np.array_equal(dq['income'], want)), 'model_paths': paths,
       'torch_imported': 'torch' in sys.modules, 'python': sys.version.split()[0]}
out['all_ok'] = (out['occupation_is_succ_bitwise'] and out['res_zero_residual_max_abs_diff'] < 1e-12
                 and out['alt_at_succ_beta_max_abs_diff'] < 1e-12 and out['dq_inc_is_shared_law_bitwise'] and not out['torch_imported'])
print(json.dumps(out))
