"""Label-free design diagnostic (coordinator; no label file is read): how much of the cached MODEL_FIT out-of-fold SEX propensity
pi(x) is carried by the task posteriors. Expected AUCs are computed under the propensity model itself (S ~ Bernoulli(pi)), so they
are model-based proxies, not measured recoveries. Inputs: the cached out-of-fold predictions TASK.pkl (new release mechanisms) and
SEXSIDE.pkl (finite-routing comparison), versions 0 and 1.

    python3 results/pcrl_dual_design_release_v1/roles/coordinator/labelfree/propensity_structure.py
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[5]


def expected_auc(score, pi):
    """P(score_i > score_j) for i ~ women, j ~ men under S ~ Bernoulli(pi) independently per person (ties count one half)."""
    o = np.argsort(score, kind='mergesort')
    s, p = score[o], pi[o]
    _, idx = np.unique(s, return_index=True)
    w1, w0 = np.add.reduceat(p, idx), np.add.reduceat(1 - p, idx)
    below0 = np.cumsum(w0) - w0
    num = np.sum(w1 * below0) + 0.5 * (np.sum(w1 * w0) - np.sum(p * (1 - p)))
    return float(num / (p.sum() * (1 - p).sum() - np.sum(p * (1 - p))))


def cv_pred(F, y):
    out = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(F):
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, random_state=0)
        m.fit(F[tr], y[tr])
        out[te] = m.predict(F[te])
    return out


def main():
    t0 = time.process_time()
    res = {}
    for v in (0, 1):
        T = pickle.load(open(ROOT / f'.private/pcrl_new_release_mechanisms_v1/fa_cache/V{v}/TASK.pkl', 'rb'))
        S = pickle.load(open(ROOT / f'.private/pcrl_finite_routing_comparison_v1/fa_cache/V{v}/SEXSIDE.pkl', 'rb'))
        qI, qO, pi = T['oof']['inc'], T['oof']['occ'], np.clip(S['oof']['sex'][:, 1], 0.01, 0.99)
        lpi = np.log(pi / (1 - pi))
        lI, lO = np.log(qI[:, 1:] / qI[:, :1]), np.log(qO[:, 1:] / qO[:, :1])
        r = {'n': int(len(pi)), 'mean_pi': float(pi.mean()), 'eauc_pi': expected_auc(pi, pi)}
        for name, F in (('income', lI), ('occupation', lO), ('pair', np.column_stack([lI, lO]))):
            e = cv_pred(F, lpi)
            r[f'eauc_E[pi|{name}]'] = expected_auc(e, pi)
            r[f'r2_logitpi_on_{name}'] = float(1 - np.var(lpi - e) / np.var(lpi))
        qO0, qO1 = S['oof']['occ_s'][0], S['oof']['occ_s'][1]
        mix = (1 - pi)[:, None] * qO0 + pi[:, None] * qO1
        r['kl_qO_vs_pi_mixture'] = float(np.mean(np.sum(qO * np.log(qO / mix), 1)))
        pbar = float(pi.mean())
        mmp = (1 - pbar) * qO0 + pbar * qO1
        r['kl_qO_vs_mmp_fixed_pbar'] = float(np.mean(np.sum(qO * np.log(qO / mmp), 1)))
        res[f'V{v}'] = r
    res['cpu_s'] = time.process_time() - t0
    print(json.dumps(res, indent=1))
    (Path(__file__).parent / 'PROPENSITY_STRUCTURE.json').write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
