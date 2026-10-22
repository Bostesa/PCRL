"""Designer A, campaign pcrl_input_and_pair_release_v1: LABEL-FREE plug-in check of the release-only explanations of shared
field generalization (SFG, steps MAR1 + WK3), under the plug-in reader of feasibility.py (a cross-fitted HGB regression of the cached
propensity pi on the released logits: a denoised, data-rich reader).
  * PROJ: E[ SFG release logits | twin release logits ] (the best release-only imitation of SFG's vectors; never more spend, Jensen);
  * TILT: a release-only tilt of the twin's logits against psi = E[pi | twin release] (ZTILT-like), at a scale matching SFG's spend.
No label file, nothing of MODEL_SELECT, no reader cache."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[k] = '1'
import sys, time, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
T0 = time.process_time()
import feasibility as FE


def softmax_from_logits(L):
    li = L[:, 0]; ri = 1 / (1 + np.exp(-li))
    lo = np.column_stack([np.zeros(len(L)), L[:, 1:]]); po = np.exp(lo - lo.max(1, keepdims=True))
    return np.column_stack([1 - ri, ri]), po / po.sum(1, keepdims=True)


def summ(ei, eo, pi, ri, ro, b):
    L = FE.logits(ri, ro)
    r = {'kl_inc': FE.kl(ei, ri), 'kl_occ': FE.kl(eo, ro), 'pair_auc': FE.leak(L, pi), 'inc_auc': FE.leak(L[:, :1], pi),
         'occ_auc': FE.leak(L[:, 1:], pi)}
    if b is not None:
        r['d_spend_total'] = r['kl_inc'] + r['kl_occ'] - b['kl_inc'] - b['kl_occ']; r['d_pair'] = r['pair_auc'] - b['pair_auc']
    return {k: round(x, 5) for k, x in r.items()}


out = {'what': __doc__.strip().splitlines()[0]}
for v in (0, 1):
    ei, eo, pi = FE.load(v)
    Lt = FE.logits(ei, eo)
    res = {}
    ri0, ro0 = FE.release(FE.Z, ei, eo)                       # identity smoother (spend baseline, as in feasibility.py)
    res['IDENTITY'] = b = summ(ei, eo, pi, ri0, ro0, None)
    ri1, ro1 = FE.release(FE.coarsen(FE.Z, ['MAR1', 'WK3']), ei, eo)
    res['SFG_MAR1_WK3'] = summ(ei, eo, pi, ri1, ro1, b)
    L1 = FE.logits(ri1, ro1)
    Lp = np.column_stack([FE.reg(Lt, L1[:, j]) for j in range(5)])
    rip, rop = softmax_from_logits(Lp)
    res['PROJ_of_SFG'] = summ(ei, eo, pi, rip, rop, b)
    psi = FE.reg(Lt, pi); c = psi - psi.mean()
    beta = (Lt - Lt.mean(0)).T @ c / (c @ c)                  # linear covariance of each released logit with psi
    for a in (0.5, 1.0, 1.5, 2.0):
        rit, rot = softmax_from_logits(Lt - a * np.outer(c, beta))
        res[f'TILT_a{a}'] = summ(ei, eo, pi, rit, rot, None)
        res[f'TILT_a{a}']['d_spend_vs_twin'] = round(FE.kl(ei, rit) + FE.kl(eo, rot), 5)
    res['TWIN_raw'] = summ(ei, eo, pi, ei, eo, None)
    out[f'V{v}'] = res
out['cpu_s_total'] = round(time.process_time() - T0, 1)
with open(os.path.join(HERE, 'FEASIBILITY3.json'), 'w') as f:
    json.dump(out, f, indent=1)
for v in ('V0', 'V1'):
    for k, r in out[v].items():
        print(v, k, r)
print('cpu', out['cpu_s_total'])
