"""F7 -- positive and null controls for an adaptive attacker slate, with replay.

Slate (fixed before seeing data): standardised logistic regression, histogram
gradient boosting, MLP(64), kNN(25). Selection: minimum validation log loss.
Metric: held-out test AUC (macro one-vs-rest for multiclass) with bootstrap CI.
Controls
  C1 direct inclusion : z contains s (+ small noise) and 5 noise dims    -> AUC ~ 1
  C2 XOR interaction  : s = 1{x1>0} xor 1{x2>0}, z = [x1, x2, 3 noise] -> linear ~0.5, nonlinear high
  C3 planted contrast : K = 10 balanced classes, z1 = +-1 by class half + N(0,1.5^2) noise
                        (first attempt used 1.35: seed-1 replay DA = 0.0509 sat on the 0.05
                        threshold; kept as outputs/F07_first_attempt_noise1.35.json);
                        max per-class (dominant-axis) ridge R2 < 0.05 'passes',
                        yet the slate recovers the half membership and the class
  C4 null             : z independent of s                               -> AUC CI contains 0.5
Replay: everything re-run with a second, disjoint seed (data + fits).
"""
import json
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def slate(seed):
    return {
        "logit": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "hgb": HistGradientBoostingClassifier(max_iter=150, random_state=seed),
        "mlp64": make_pipeline(StandardScaler(), MLPClassifier((64,), max_iter=400, random_state=seed)),
        "knn25": make_pipeline(StandardScaler(), KNeighborsClassifier(25)),
    }


def auc(y, P):
    return roc_auc_score(y, P[:, 1]) if P.shape[1] == 2 else roc_auc_score(y, P, multi_class="ovr", average="macro")


def run_slate(Z, s, seed, n_fit, n_val):
    f, v, t = slice(0, n_fit), slice(n_fit, n_fit + n_val), slice(n_fit + n_val, len(s))
    res, probs = {}, {}
    for name, m in slate(seed).items():
        m.fit(Z[f], s[f])
        res[name] = {"val_logloss": float(log_loss(s[v], m.predict_proba(Z[v]), labels=m.classes_)),
                     "test_auc": float(auc(s[t], m.predict_proba(Z[t])))}
        probs[name] = m.predict_proba(Z[t])
    best = min(sorted(res), key=lambda k: res[k]["val_logloss"])
    rng = np.random.default_rng(seed)
    yt, P = s[t], probs[best]
    bs = []
    for _ in range(300):
        i = rng.integers(0, len(yt), len(yt))
        if len(np.unique(yt[i])) == len(np.unique(yt)):
            bs.append(auc(yt[i], P[i]))
    return {"selected": best, "selected_test_auc": res[best]["test_auc"],
            "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "all": res}, probs[best], t


def max_per_class_r2(Z, s, K):
    Zc = Z - Z.mean(0)
    G = Zc.T @ Zc + 1e-6 * np.eye(Z.shape[1])
    r = []
    for k in range(K):
        y = (s == k).astype(float); yc = y - y.mean()
        w = np.linalg.solve(G, Zc.T @ yc)
        r.append(1 - ((yc - Zc @ w) ** 2).sum() / (yc ** 2).sum())
    return float(max(r))


def controls(seed):
    rng = np.random.default_rng(seed)
    out = {}
    n, nf, nv = 6000, 2000, 1000
    s = rng.integers(0, 2, n)
    Z = np.c_[s + 0.05 * rng.normal(size=n), rng.normal(size=(n, 5))]
    out["C1_direct"], _, _ = run_slate(Z, s, seed, nf, nv)
    x = rng.normal(size=(n, 2))
    s = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype(int)
    Z = np.c_[x, rng.normal(size=(n, 3))]
    out["C2_xor"], _, _ = run_slate(Z, s, seed, nf, nv)
    n3 = 12000
    s = rng.integers(0, 10, n3)
    Z = np.c_[np.where(s < 5, 1.0, -1.0) + 1.5 * rng.normal(size=n3), rng.normal(size=(n3, 4))]
    r, P, t = run_slate(Z, s, seed, 4000, 2000)
    half = (s[t] < 5).astype(int)
    r["dominant_axis_r2_on_test_rows"] = max_per_class_r2(Z[t], s[t], 10)
    r["dominant_axis_passes_0.05"] = r["dominant_axis_r2_on_test_rows"] <= 0.05
    r["half_membership_auc_from_selected_probs"] = float(roc_auc_score(half, P[:, :5].sum(1)))
    prior = np.bincount(s[:4000], minlength=10) / 4000
    r["test_logloss_selected"] = float(log_loss(s[t], P, labels=np.arange(10)))
    r["test_logloss_prior_only"] = float(log_loss(s[t], np.tile(prior, (len(s[t]), 1)), labels=np.arange(10)))
    out["C3_planted_contrast_K10"] = r
    s = rng.integers(0, 2, n)
    Z = rng.normal(size=(n, 6))
    r, _, _ = run_slate(Z, s, seed, nf, nv)
    r["ci_contains_0.5"] = bool(r["ci95"][0] <= 0.5 <= r["ci95"][1])
    out["C4_null"] = r
    return out


def verdicts(o):
    return {"C1_recovers": o["C1_direct"]["selected_test_auc"] > 0.99,
            "C2_linear_misses": o["C2_xor"]["all"]["logit"]["test_auc"] < 0.55,
            "C2_slate_recovers": o["C2_xor"]["selected_test_auc"] > 0.9,
            "C3_DA_passes_but_slate_recovers": bool(o["C3_planted_contrast_K10"]["dominant_axis_passes_0.05"]
                                                   and o["C3_planted_contrast_K10"]["half_membership_auc_from_selected_probs"] > 0.75),
            "C4_null_ci_contains_half": o["C4_null"]["ci_contains_0.5"]}


def main():
    a, b = controls(0), controls(1)
    print(json.dumps({"id": "F7", "seed0": a, "seed1_replay": b, "verdicts_seed0": verdicts(a),
                      "verdicts_seed1": verdicts(b), "replay_agrees": verdicts(a) == verdicts(b)}, indent=1))


if __name__ == "__main__":
    main()
