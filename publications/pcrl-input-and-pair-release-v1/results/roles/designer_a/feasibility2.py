"""Designer A, campaign pcrl_input_and_pair_release_v1: LABEL-FREE plug-in check of field-structured (interaction-constrained)
task models, the other input-side device, and of combining one with the best generalization steps of feasibility.py.
Same plug-in world and helpers as feasibility.py (cached eta, pi; no label file, nothing of MODEL_SELECT, no reader cache).
Releases are cross-fitted HGB regressions of the cached task LOGITS on Z under sklearn interaction constraints, then softmax."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[k] = '1'
import sys, time, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
T0 = time.process_time()
import feasibility as FE                                     # re-runs nothing heavy: feasibility.py guards its main block below
from sklearn.ensemble import HistGradientBoostingRegressor

A, W, CW, S, M, P = range(6)
CONSTRAINTS = {'FREE': None,
               'MAR_MAIN': [[A, W, CW, S, P], [M]],           # MAR enters only as a main effect
               'MAR_COW_MAIN': [[A, W, S, P], [M], [CW]],
               'PAIRWISE': [[i, j] for i in range(6) for j in range(i + 1, 6)],
               'ADDITIVE': [[i] for i in range(6)]}


def reg_ic(R, y, ic):
    out = np.zeros(len(y))
    for f in (0, 1):
        tr, te = FE.FOLD != f, FE.FOLD == f
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=40,
                                          categorical_features=FE.CAT, interaction_cst=ic, random_state=0).fit(R[tr], y[tr])
        out[te] = m.predict(R[te])
    return out


def release_ic(Zh, ei, eo, ic):
    li = reg_ic(Zh, np.log(ei[:, 1] / ei[:, 0]), ic)
    lo = np.column_stack([np.zeros(len(Zh))] + [reg_ic(Zh, np.log(eo[:, k] / eo[:, 0]), ic) for k in range(1, 5)])
    ri = 1 / (1 + np.exp(-li)); po = np.exp(lo - lo.max(1, keepdims=True))
    return np.column_stack([1 - ri, ri]), po / po.sum(1, keepdims=True)


out = {'what': __doc__.strip().splitlines()[0]}
for v in (0, 1):
    ei, eo, pi = FE.load(v)
    res = {}
    for steps in ([], ['MAR1', 'WK3']):
        Zh = FE.coarsen(FE.Z, steps)
        for nm, ic in CONSTRAINTS.items():
            if v == 1 and nm not in ('FREE', 'MAR_MAIN', 'ADDITIVE'):
                continue
            if steps and nm not in ('FREE', 'MAR_MAIN'):
                continue
            ri, ro = release_ic(Zh, ei, eo, ic)
            L = FE.logits(ri, ro)
            res['+'.join(steps + [nm])] = {'kl_inc': FE.kl(ei, ri), 'kl_occ': FE.kl(eo, ro), 'pair_auc': FE.leak(L, pi),
                                           'inc_auc': FE.leak(L[:, :1], pi), 'occ_auc': FE.leak(L[:, 1:], pi),
                                           'brier_occ_men': float(np.sum((1 - pi) * np.sum((ro - eo) ** 2, 1)) / np.sum(1 - pi)),
                                           'brier_occ_women': float(np.sum(pi * np.sum((ro - eo) ** 2, 1)) / np.sum(pi)),
                                           'acc_loss_inc': float(np.mean(ei.max(1) - ei[np.arange(FE.N), ri.argmax(1)])),
                                           'acc_loss_occ': float(np.mean(eo.max(1) - eo[np.arange(FE.N), ro.argmax(1)]))}
    b = res['FREE']
    for k, r in res.items():
        r['d_spend_total'] = r['kl_inc'] + r['kl_occ'] - b['kl_inc'] - b['kl_occ']
        r['d_pair'] = r['pair_auc'] - b['pair_auc']
        for kk in list(r):
            r[kk] = round(r[kk], 5)
    out[f'V{v}'] = res
out['cpu_s_total'] = round(time.process_time() - T0, 1)
with open(os.path.join(HERE, 'FEASIBILITY2.json'), 'w') as f:
    json.dump(out, f, indent=1)
for v in ('V0', 'V1'):
    for k, r in out[v].items():
        print(v, k, r)
print('cpu', out['cpu_s_total'])
