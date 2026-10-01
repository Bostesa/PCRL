"""F6 -- accounting for an appended CONSTANT channel.

Population fact: appending a constant c to the wire, [H, J] -> [H, J, c], changes
no conditional distribution, so the population (Bayes) recovery increment is 0.
Finite-sample fact: a FRESH fit on the wider input is a different random
function (different initialisation shape / optimisation path), so measured
recovery differs, with either sign.

Fixture: S ~ Bern(1/2); H (4 dims, weak), J (1 dim, stronger). Learner: sklearn
MLPClassifier(32), fresh fit per condition, same seed. Shared evaluation people
(n_eval = 3000). For each of 6 seeds we report
  mean per-person log-loss difference D = loss_ext - loss_base, paired 95% t-CI,
  AUC difference with paired bootstrap CI,
  the same statistics for a pure REFIT control (identical input, seed + 1000),
  and naive unpaired CIs (two independent bootstraps) for contrast.
"""
import json
import warnings

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")


def fit_pred(X, y, Xe, seed):
    m = MLPClassifier((32,), max_iter=400, random_state=seed).fit(X, y)
    return np.clip(m.predict_proba(Xe)[:, 1], 1e-7, 1 - 1e-7)


def ll(y, p):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired(y, p_new, p_old, seed):
    D = ll(y, p_new) - ll(y, p_old)
    m, se = D.mean(), D.std(ddof=1) / np.sqrt(len(D))
    t = stats.t.ppf(0.975, len(D) - 1)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(300):
        i = rng.integers(0, len(y), len(y))
        bs.append(roc_auc_score(y[i], p_new[i]) - roc_auc_score(y[i], p_old[i]))
    return {"mean_logloss_diff": float(m), "paired_t_ci95": [float(m - t * se), float(m + t * se)],
            "covers_zero": bool(m - t * se <= 0 <= m + t * se),
            "auc_diff": float(roc_auc_score(y, p_new) - roc_auc_score(y, p_old)),
            "auc_diff_paired_boot_ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
            "frac_people_with_nonzero_loss_change": float(np.mean(np.abs(D) > 1e-9))}


def unpaired_auc_ci(y, p, seed):
    rng = np.random.default_rng(seed)
    bs = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(300))]
    return [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def main():
    rng = np.random.default_rng(606)
    nf, ne = 2000, 3000
    s = rng.integers(0, 2, nf + ne)
    H = rng.normal(size=(nf + ne, 4)) + 0.25 * s[:, None]
    J = (0.9 * s + rng.normal(size=nf + ne))[:, None]
    base = np.c_[H, J]
    ext = np.c_[H, J, np.full(nf + ne, 1.0)]
    f, e = slice(0, nf), slice(nf, nf + ne)
    rows = {}
    for seed in range(6):
        pb = fit_pred(base[f], s[f], base[e], seed)
        px = fit_pred(ext[f], s[f], ext[e], seed)
        pr = fit_pred(base[f], s[f], base[e], seed + 1000)
        rows[f"seed{seed}"] = {
            "auc_base": float(roc_auc_score(s[e], pb)), "auc_ext": float(roc_auc_score(s[e], px)),
            "ext_minus_base": paired(s[e], px, pb, seed),
            "refit_control_minus_base": paired(s[e], pr, pb, seed),
            "unpaired_auc_ci_base": unpaired_auc_ci(s[e], pb, seed),
            "unpaired_auc_ci_ext": unpaired_auc_ci(s[e], px, seed + 7),
        }
    diffs = [r["ext_minus_base"]["auc_diff"] for r in rows.values()]
    refit = [r["refit_control_minus_base"]["auc_diff"] for r in rows.values()]
    def seed_ci(v):
        v = np.asarray(v); m = v.mean(); h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        return {"mean": float(m), "t_ci95_over_fits": [float(m - h), float(m + h)], "covers_zero": bool(m - h <= 0 <= m + h)}
    lld = [r["ext_minus_base"]["mean_logloss_diff"] for r in rows.values()]
    llr = [r["refit_control_minus_base"]["mean_logloss_diff"] for r in rows.values()]
    across = {"auc_diff_ext_minus_base": seed_ci(diffs), "auc_diff_refit_control": seed_ci(refit),
              "logloss_diff_ext_minus_base": seed_ci(lld), "logloss_diff_refit_control": seed_ci(llr)}
    print(json.dumps({"id": "F6", "population_increment": 0.0, "per_seed": rows, "across_fits": across,
                      "auc_diffs_ext_minus_base": diffs, "auc_diffs_refit_control": refit,
                      "n_negative_ext_minus_base": int(sum(d < 0 for d in diffs)),
                      "n_paired_logloss_ci_cover_zero": int(sum(r["ext_minus_base"]["covers_zero"] for r in rows.values())),
                      "sd_ext_minus_base": float(np.std(diffs, ddof=1)), "sd_refit_control": float(np.std(refit, ddof=1))},
                     indent=1))


if __name__ == "__main__":
    main()
