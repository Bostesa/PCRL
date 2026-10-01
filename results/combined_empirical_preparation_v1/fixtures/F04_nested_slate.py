"""F4 -- nested (singleton / ancestor) predictors in coalition and appended-channel audits.

Independent sklearn fixture. Protocol mirrors the ACS audits in spirit only:
fixed candidate slate per input; selection = minimum validation log loss;
reported metric = test AUC (and test log loss) of the selected candidate.

Part 1 (two recipients). S ~ Bernoulli(1/2). View A: 2 dims, mean shift with S.
View B: 150 dims of pure noise. Coalition AB = [A, B].
  (i)  coalition slate fitted on AB only            -> can lose to A alone
  (ii) coalition slate + nested singleton predictors (A-only / B-only fits that
       read their own columns of AB and ignore the rest) -> >= A alone up to
       validation-selection noise.
Population truth: AB carries exactly the information of A (B is independent
noise), so the population coalition increment is 0, never negative.

Part 2 (appended channel). Wire_J = [H, J] where J is an existing released
channel (informative); extension wire = [H, J, R] with R = 60 noise dims.
  ancestors 'H-only'     : candidates fitted on H, reading only H columns
  ancestors 'H-only + J' : additionally the candidates fitted on [H, J]
Increment of interest: recovery(extension) - recovery(J). Population value 0.
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
        "logit_C1": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
        "knn25": make_pipeline(StandardScaler(), KNeighborsClassifier(25)),
        "hgb": HistGradientBoostingClassifier(max_iter=100, random_state=seed),
        "mlp64": make_pipeline(StandardScaler(), MLPClassifier((64,), max_iter=300, random_state=seed)),
    }


class Projected:
    def __init__(self, model, cols):
        self.model, self.cols = model, cols

    def predict_proba(self, X):
        return self.model.predict_proba(X[:, self.cols])


def fit_slate(Xf, yf, seed, prefix, cols=None):
    out = {}
    for name, m in slate(seed).items():
        m.fit(Xf if cols is None else Xf[:, cols], yf)
        out[prefix + name] = m if cols is None else Projected(m, cols)
    return out


def select_and_score(cands, Xv, yv, Xt, yt):
    vl = {k: log_loss(yv, np.clip(c.predict_proba(Xv)[:, 1], 1e-6, 1 - 1e-6)) for k, c in cands.items()}
    best = min(sorted(vl), key=lambda k: vl[k])
    p = cands[best].predict_proba(Xt)[:, 1]
    return {"selected": best, "val_logloss": float(vl[best]), "test_auc": float(roc_auc_score(yt, p)),
            "test_logloss": float(log_loss(yt, np.clip(p, 1e-6, 1 - 1e-6)))}, p


def paired_boot(yt, p1, p2, B=400, seed=0):
    rng = np.random.default_rng(seed)
    n = len(yt)
    d = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if yt[i].min() == yt[i].max():
            continue
        d.append(roc_auc_score(yt[i], p1[i]) - roc_auc_score(yt[i], p2[i]))
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


def part1(seed):
    rng = np.random.default_rng(seed)
    nf, nv, nt = 300, 300, 4000
    n = nf + nv + nt
    s = rng.integers(0, 2, n)
    A = rng.normal(size=(n, 2)) + 1.0 * s[:, None] * np.array([1.0, 0.5])
    B = rng.normal(size=(n, 150))
    AB = np.c_[A, B]
    f, v, t = slice(0, nf), slice(nf, nf + nv), slice(nf + nv, n)
    ca = fit_slate(A[f], s[f], seed, "A/")
    cb = fit_slate(B[f], s[f], seed, "B/")
    cab = fit_slate(AB[f], s[f], seed, "AB/")
    rA, pA = select_and_score(ca, A[v], s[v], A[t], s[t])
    rB, pB = select_and_score(cb, B[v], s[v], B[t], s[t])
    rAB, pAB = select_and_score(cab, AB[v], s[v], AB[t], s[t])
    nested = dict(cab)
    nested.update(fit_slate(AB[f], s[f], seed, "inherited_A/", cols=list(range(2))))
    nested.update(fit_slate(AB[f], s[f], seed, "inherited_B/", cols=list(range(2, 152))))
    rN, pN = select_and_score(nested, AB[v], s[v], AB[t], s[t])
    return {"A_alone": rA, "B_alone": rB, "AB_without_nested": rAB, "AB_with_nested": rN,
            "combination_effect_without_nested": rAB["test_auc"] - max(rA["test_auc"], rB["test_auc"]),
            "combination_effect_with_nested": rN["test_auc"] - max(rA["test_auc"], rB["test_auc"]),
            "paired_CI_AB_without_minus_A": paired_boot(s[t], pAB, pA),
            "paired_CI_AB_with_minus_A": paired_boot(s[t], pN, pA)}


def part2(seed):
    rng = np.random.default_rng(100 + seed)
    nf, nv, nt = 400, 400, 4000
    n = nf + nv + nt
    s = rng.integers(0, 2, n)
    H = rng.normal(size=(n, 3)) + 0.4 * s[:, None]
    J = (1.2 * s + rng.normal(size=n))[:, None]
    R = rng.normal(size=(n, 60))
    WJ, WE = np.c_[H, J], np.c_[H, J, R]
    f, v, t = slice(0, nf), slice(nf, nf + nv), slice(nf + nv, n)
    h_anc_J = fit_slate(WJ[f], s[f], seed, "H_anc/", cols=[0, 1, 2])
    j_own = fit_slate(WJ[f], s[f], seed, "J_own/")
    rJ, pJ = select_and_score({**j_own, **h_anc_J}, WJ[v], s[v], WJ[t], s[t])
    e_own = fit_slate(WE[f], s[f], seed, "E_own/")
    h_anc_E = fit_slate(WE[f], s[f], seed, "H_anc/", cols=[0, 1, 2])
    j_anc_E = {"J_anc/" + k.split("/", 1)[1]: Projected(m, [0, 1, 2, 3]) for k, m in j_own.items()}
    rEh, pEh = select_and_score({**e_own, **h_anc_E}, WE[v], s[v], WE[t], s[t])
    rEhj, pEhj = select_and_score({**e_own, **h_anc_E, **j_anc_E}, WE[v], s[v], WE[t], s[t])
    return {"J_condition": rJ, "extension_H_only_ancestors": rEh, "extension_H_and_J_ancestors": rEhj,
            "increment_H_only": rEh["test_auc"] - rJ["test_auc"],
            "increment_H_and_J": rEhj["test_auc"] - rJ["test_auc"],
            "paired_CI_increment_H_only": paired_boot(s[t], pEh, pJ),
            "paired_CI_increment_H_and_J": paired_boot(s[t], pEhj, pJ)}


def main():
    out = {"id": "F4", "part1_two_recipients": {}, "part2_appended_channel": {}}
    for seed in (0, 1, 2):
        out["part1_two_recipients"][f"seed{seed}"] = part1(seed)
        out["part2_appended_channel"][f"seed{seed}"] = part2(seed)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
