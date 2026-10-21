"""Designer A, second label-free pass (same inputs and plug-in world as feasibility.py; no label file, nothing of MODEL_SELECT).

Adds INFORMATION-REMOVING SEX-blind controls at matched plug-in T1 NLL spend (feasibility.py showed that flattening and temperature
are near-invertible maps of the twin and do not lower plug-in reader AUC at all, so they are not a fair control):
  COARSE@l   Euclidean k-means of log qO into K cells (sklearn KMeans, n_init 1), released = in-sample cell mean of qO; K from a
             ladder, the one whose T1 spend is closest to the STD point's (spend reported, not exactly matched)
  NOISE@l    softmax(log qO + sigma * eps), eps ~ N(0, I5), one draw per person (fixed seed); sigma by bisection on the T1 spend
Also: STD at lambda 1.25 / 1.5 (over-standardization), and the fixed weight w at 0.35 / 0.40 / 0.45 (lambda = 1): guard balance.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feasibility as F                                     # noqa: E402

K_LADDER = (24, 32, 48, 64, 96, 128, 192, 256, 384)


def coarse_path(qO):
    from sklearn.cluster import KMeans
    L = np.log(qO)
    out = {}
    for K in K_LADDER:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            lab = KMeans(n_clusters=K, n_init=1, random_state=0).fit_predict(L)
        C = np.zeros((K, qO.shape[1]))
        np.add.at(C, lab, qO)
        C /= np.bincount(lab, minlength=K)[:, None]
        out[K] = F.norm(C[lab])
    return out


def run(v, extras=True):
    d = F.load(v)
    n = len(d['pi'])
    pi, qO, qI, q0, q1 = d['pi'], d['qO'], d['qI'], d['q0'], d['q1']
    w = float(pi.mean())
    spend = lambda r: float(np.mean(F.ce(qO, r) - F.ce(qO, qO)))
    r1 = F.norm((1 - w) * q0 + w * q1)
    rel = {'TWIN': qO, 'STD_1.0': r1, 'STD_0.5': F.norm(0.5 * qO + 0.5 * r1)}
    info = {'w': w}
    cp = coarse_path(qO)
    info['coarse_spend_by_K'] = {K: spend(r) for K, r in cp.items()}
    rng = np.random.default_rng(4242 + v)
    eps = rng.normal(size=qO.shape)
    for l in (0.5, 1.0):
        tgt = spend(rel[f'STD_{l}'])
        K = min(cp, key=lambda k: abs(info['coarse_spend_by_K'][k] - tgt))
        rel[f'COARSE@{l}'] = cp[K]
        s = F.match(lambda s: spend(F.norm(np.exp(np.log(qO) + s * eps))), tgt, 0.0, 3.0)
        rel[f'NOISE@{l}'] = F.norm(np.exp(np.log(qO) + s * eps))
        info[f'COARSE@{l}'] = {'K': K, 'spend': info['coarse_spend_by_K'][K], 'target': tgt}
        info[f'NOISE@{l}'] = {'sigma': s, 'spend': spend(rel[f'NOISE@{l}'])}
    if extras:
        for l in (1.25, 1.5):
            rel[f'STD_{l}'] = F.norm((1 - l) * qO + l * r1)
        for wx in (0.35, 0.40, 0.45):
            rel[f'STDw{wx}'] = F.norm((1 - wx) * q0 + wx * q1)
    U = {k: {'T1': F.util(r, qO, qO), 'T2': F.util(r, qO, F.norm((1 - pi)[:, None] * q0 + pi[:, None] * q1)),
             'guards': F.guards(r, qO, d)} for k, r in rel.items()}
    rng = np.random.default_rng(20261021 + v)              # the same FIT blocks and pseudo-SEX draws as feasibility.py
    perm = rng.permutation(n)
    blocks = [perm[i * F.N_FIT:(i + 1) * F.N_FIT] for i in range(5)]
    ys = [(rng.random(F.N_FIT) < pi[idx]).astype(int) for idx in blocks]
    R = {}
    for k, r in rel.items():
        R[k] = {'occ': F.reader_auc(F.feats([r]), pi, blocks, ys, ('LOG_LR', 'HGB15')),
                'pair': F.reader_auc(F.feats([qI, r]), pi, blocks, ys, ('LOG_LR', 'HGB15'))}
    return {'version': v, 'info': info, 'utility': U, 'readers': R}


if __name__ == '__main__':
    t0 = time.process_time()
    res = {'runs': [run(0, extras=True), run(1, extras=False)]}
    res['cpu_s'] = round(time.process_time() - t0, 2)
    (F.OUT / 'FEASIBILITY2.json').write_text(json.dumps(res, indent=1))
    print('CPU_SECONDS', res['cpu_s'])
