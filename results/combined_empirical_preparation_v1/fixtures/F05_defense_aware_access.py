"""F5 -- defense-aware attacker vs pre-noise (insider) access vs query averaging.

Gaussian toy, closed form + Monte Carlo (numpy/scipy/sklearn, independent).
  s ~ Bernoulli(1/2);  h = mu_s + tau * e,  e ~ N(0, I_d);  release r = h + sigma * z.
  mu_0 = 0, mu_1 = delta * u (u unit vector). Equal class covariances, so the Bayes
  score is linear and AUC = Phi( delta / sqrt(2 * v) ) with v the per-row noise
  variance along u.
  (a) defense-aware, one release, knows sigma:     v = tau^2 + sigma^2
  (b) insider with clean h at test time:           v = tau^2
  (c) N fresh-noise queries averaged:              v = tau^2 + sigma^2 / N
  (d) N queries with persistent (repeated) noise:  v = tau^2 + sigma^2   (no gain)
Monte-Carlo attackers:
  a_logit_on_r      logistic regression trained on released r (no sigma knowledge)
  a_lrt_clean_fit   class Gaussians fitted on CLEAN training h, covariance + sigma^2 I,
                    scores single released test rows (same construction as
                    durable-guarantees utils/battery.py:_lrt_scores)
  a_lrt_deconv      class Gaussians fitted on RELEASED training r only, covariance
                    minus sigma^2 I (knows sigma, never sees h), + sigma^2 I at scoring
  b_logit_on_h, c_avgN, d_persistN as above.
"""
import json
import math

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def lrt_scores(mu0, mu1, C0, C1, X):
    out = []
    for mu, C in ((mu0, C0), (mu1, C1)):
        L = np.linalg.cholesky(C)
        zz = np.linalg.solve(L, (X - mu).T)
        out.append(-0.5 * (zz ** 2).sum(0) - np.log(np.diag(L)).sum())
    return out[1] - out[0]


def auc_ci(y, sc, B=200, seed=0):
    rng = np.random.default_rng(seed)
    a = roc_auc_score(y, sc)
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        bs.append(roc_auc_score(y[i], sc[i]))
    return {"auc": float(a), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}


def main():
    d, delta, tau, sigma = 4, 1.0, 1.0, 2.0
    u = np.zeros(d); u[0] = 1.0
    analytic = {
        "a_defense_aware_one_release": norm.cdf(delta / math.sqrt(2 * (tau ** 2 + sigma ** 2))),
        "b_clean_h_insider": norm.cdf(delta / math.sqrt(2 * tau ** 2)),
        **{f"c_fresh_average_N{N}": norm.cdf(delta / math.sqrt(2 * (tau ** 2 + sigma ** 2 / N))) for N in (1, 4, 16)},
        **{f"d_persistent_N{N}": norm.cdf(delta / math.sqrt(2 * (tau ** 2 + sigma ** 2))) for N in (1, 4, 16)},
    }
    rng = np.random.default_rng(2026)
    ntr, nte = 20000, 20000
    def draw(n):
        s = rng.integers(0, 2, n)
        h = delta * s[:, None] * u + tau * rng.normal(size=(n, d))
        return s, h
    s_tr, h_tr = draw(ntr)
    s_te, h_te = draw(nte)
    r_tr = h_tr + sigma * rng.normal(size=h_tr.shape)
    r_te = h_te + sigma * rng.normal(size=h_te.shape)
    mc = {}
    lr = LogisticRegression(max_iter=1000).fit(r_tr, s_tr)
    mc["a_logit_on_r"] = auc_ci(s_te, lr.decision_function(r_te))
    I = np.eye(d)
    m0, m1 = h_tr[s_tr == 0].mean(0), h_tr[s_tr == 1].mean(0)
    C0, C1 = np.cov(h_tr[s_tr == 0].T), np.cov(h_tr[s_tr == 1].T)
    mc["a_lrt_clean_fit_sigma_known"] = auc_ci(s_te, lrt_scores(m0, m1, C0 + sigma ** 2 * I, C1 + sigma ** 2 * I, r_te))
    q0, q1 = r_tr[s_tr == 0].mean(0), r_tr[s_tr == 1].mean(0)
    D0 = np.cov(r_tr[s_tr == 0].T) - sigma ** 2 * I
    D1 = np.cov(r_tr[s_tr == 1].T) - sigma ** 2 * I
    fix = lambda M: (lambda w, V: V @ np.diag(np.maximum(w, 1e-6)) @ V.T)(*np.linalg.eigh((M + M.T) / 2))
    mc["a_lrt_deconv_from_released_only"] = auc_ci(s_te, lrt_scores(q0, q1, fix(D0) + sigma ** 2 * I, fix(D1) + sigma ** 2 * I, r_te))
    lrh = LogisticRegression(max_iter=1000).fit(h_tr, s_tr)
    mc["b_logit_on_clean_h"] = auc_ci(s_te, lrh.decision_function(h_te))
    for N in (1, 4, 16):
        avg_te = h_te + sigma * rng.normal(size=(N,) + h_te.shape).mean(0)
        avg_tr = h_tr + sigma * rng.normal(size=(N,) + h_tr.shape).mean(0)
        mc[f"c_fresh_average_N{N}"] = auc_ci(s_te, LogisticRegression(max_iter=1000).fit(avg_tr, s_tr).decision_function(avg_te))
        z_te = sigma * rng.normal(size=h_te.shape)
        pers_te = np.mean([h_te + z_te for _ in range(N)], axis=0)    # same noise every query
        mc[f"d_persistent_N{N}"] = auc_ci(s_te, lr.decision_function(pers_te))
    print(json.dumps({"id": "F5", "params": {"d": d, "delta": delta, "tau": tau, "sigma": sigma,
                      "n_train": ntr, "n_test": nte},
                      "analytic_auc": {k: float(v) for k, v in analytic.items()}, "monte_carlo": mc}, indent=1))


if __name__ == "__main__":
    main()
