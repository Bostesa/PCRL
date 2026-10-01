"""F2 -- per-class (dominant-axis) R2 versus the full categorical diagnostic.

Independent numpy implementation. What PCRL's dominant axis computes
(origin/main pcrl/evaluation/certificates.py:47-117): for each class k, in-sample
ridge (lam=1e-6) R2 of the indicator 1{y=k} on centred H; R2_DA = max_k R2_k.
It is a maximum over the K indicator directions only.

Full categorical diagnostic used here (max R2 over ALL linear contrasts a'Y):
  Y_d   = one-hot with one reference class dropped (columns for K-1 classes), centred
  S_yy  = Y_d'Y_d/n  (full rank iff every class has support in the sample)
  S_zz  = Z_c'Z_c/n + eps I,  eps = 1e-10 * trace(S_zz)/d   (numerical guard only)
  S_yz  = Y_d'Z_c/n
  rho2  = lambda_max( S_yy^{-1/2} S_yz S_zz^{-1} S_zy S_yy^{-1/2} )
        = squared first canonical correlation = max_a R2(a'Y on Z) over contrasts a.
Eigenvalues below 1e-12 are treated as 0. If a class has zero support, S_yy is
singular: that class is reported as unsupported and dropped (pseudo-inverse would
silently change the contrast space) -- reported, not hidden.
Invariance to the dropped reference class is checked numerically.

Bound proved in-line (Cauchy-Schwarz): for any priors, rho2 <= (K-1) * max_k R2_k.
So the gap between the canonical diagnostic and the dominant axis is at most K-1;
for K=3 at most 2 (it cannot be 'every indicator small, contrast large' with K=3).
The fixture shows K=3 (gap <= 2, attained 4/3 for the e1-e2 contrast) and K=10
(gap ~ 9: each indicator R2 < 0.05 while the half-vs-half contrast has rho2 ~ 0.4).
"""
import hashlib
import json

import numpy as np

LAM = 1e-6


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def per_class_r2(Z, y, K):
    Zc = Z - Z.mean(0)
    G = Zc.T @ Zc + LAM * np.eye(Z.shape[1])
    out = []
    for k in range(K):
        t = (y == k).astype(float)
        tc = t - t.mean()
        w = np.linalg.solve(G, Zc.T @ tc)
        out.append(1 - ((tc - Zc @ w) ** 2).sum() / (tc ** 2).sum())
    return np.array(out)


def onehot_r2(Z, y, K):
    Y = np.eye(K)[y]
    Zc = Z - Z.mean(0)
    Yc = Y - Y.mean(0)
    B = np.linalg.solve(Zc.T @ Zc + LAM * np.eye(Z.shape[1]), Zc.T @ Yc)
    return 1 - ((Yc - Zc @ B) ** 2).sum() / (Yc ** 2).sum()


def canonical_rho2(Z, y, K, ref=0):
    n, d = Z.shape
    support = np.bincount(y, minlength=K)
    present = [k for k in range(K) if support[k] > 0]
    keep = [k for k in present if k != ref] if ref in present else present[1:]
    Yd = (y[:, None] == np.array(keep)[None, :]).astype(float)
    Yc = Yd - Yd.mean(0)
    Zc = Z - Z.mean(0)
    Szz = Zc.T @ Zc / n
    eps = 1e-10 * np.trace(Szz) / d
    Szz = Szz + eps * np.eye(d)
    Syy = Yc.T @ Yc / n
    Syz = Yc.T @ Zc / n
    ev, U = np.linalg.eigh(Syy)
    Syy_mhalf = U @ np.diag(1 / np.sqrt(ev)) @ U.T
    M = Syy_mhalf @ Syz @ np.linalg.solve(Szz, Syz.T) @ Syy_mhalf
    lam, V = np.linalg.eigh((M + M.T) / 2)
    lam = np.where(lam < 1e-12, 0.0, lam)
    a_d = Syy_mhalf @ V[:, -1]                     # contrast coefficients on kept columns
    a = np.zeros(K)
    a[keep] = a_d
    return float(lam[-1]), a, {"unsupported_classes": [k for k in range(K) if support[k] == 0],
                               "Syy_min_eig": float(ev.min()), "eps": float(eps)}


def r2_of_target(Z, t):
    Zc = Z - Z.mean(0)
    tc = t - t.mean()
    w = np.linalg.solve(Zc.T @ Zc + LAM * np.eye(Z.shape[1]), Zc.T @ tc)
    return 1 - ((tc - Zc @ w) ** 2).sum() / (tc ** 2).sum()


def make_case(name):
    if name == "K3_balanced_e1_minus_e2":
        rng = np.random.default_rng(21)
        n, K = 4000, 3
        y = rng.integers(0, K, n)
        c = (y == 1).astype(float) - (y == 2).astype(float)
        Z = np.c_[c + 0.35 * rng.normal(size=n), rng.normal(size=(n, 2))]
    elif name == "K3_imbalanced_e1_minus_e2":
        rng = np.random.default_rng(22)
        n, K = 4000, 3
        y = rng.choice(K, n, p=[0.8, 0.1, 0.1])
        c = (y == 1).astype(float) - (y == 2).astype(float)
        Z = np.c_[c + 0.35 * rng.normal(size=n), rng.normal(size=(n, 2))]
    elif name == "K10_balanced_half_contrast":
        rng = np.random.default_rng(23)
        n, K = 6000, 10
        y = rng.integers(0, K, n)
        c = np.where(y < 5, 1.0, -1.0)
        Z = np.c_[c + 1.35 * rng.normal(size=n), rng.normal(size=(n, 4))]
    else:
        raise ValueError(name)
    return Z, y, K


CASES = ["K3_balanced_e1_minus_e2", "K3_imbalanced_e1_minus_e2", "K10_balanced_half_contrast"]


def main():
    out = {"id": "F2", "cases": {}}
    for name in CASES:
        Z, y, K = make_case(name)
        r2k = per_class_r2(Z, y, K)
        rho2s = [canonical_rho2(Z, y, K, ref=r)[0] for r in range(K)]
        rho2, a, diag = canonical_rho2(Z, y, K, ref=0)
        r2_star = r2_of_target(Z, a[y])                  # direct regression of the optimal contrast
        rng = np.random.default_rng(99)
        rand = max(r2_of_target(Z, rng.normal(size=K)[y]) for _ in range(300))
        pi = np.bincount(y, minlength=K) / len(y)
        out["cases"][name] = {
            "n": int(len(y)), "K": K, "sha_Z": sha(Z), "sha_y": sha(y),
            "class_counts": np.bincount(y, minlength=K).tolist(),
            "per_class_r2": r2k.round(6).tolist(),
            "dominant_axis_max": float(r2k.max()),
            "onehot_r2": float(onehot_r2(Z, y, K)),
            "canonical_rho2": rho2,
            "canonical_rho2_by_reference_class": rho2s,
            "reference_invariance_max_diff": float(max(rho2s) - min(rho2s)),
            "optimal_contrast_normalised": (a / np.abs(a).max()).round(4).tolist(),
            "direct_r2_of_optimal_contrast": float(r2_star),
            "best_of_300_random_contrasts": float(rand),
            "gap_ratio_rho2_over_DA": float(rho2 / r2k.max()),
            "bound_K_minus_1": K - 1,
            "bound_holds": bool(rho2 <= (K - 1) * r2k.max() + 1e-12),
            "passes_tau_0.05": {"onehot": bool(onehot_r2(Z, y, K) <= 0.05),
                                "dominant_axis": bool(r2k.max() <= 0.05),
                                "canonical": bool(rho2 <= 0.05)},
            "diag": diag,
        }
    # Monte-Carlo check of the bound rho2 <= (K-1) * max_k R2_k on random configurations
    rng = np.random.default_rng(7)
    worst = 0.0
    for t in range(300):
        K = int(rng.integers(2, 9))
        n = 600
        pri = rng.dirichlet(np.ones(K) * rng.uniform(0.3, 3))
        y = rng.choice(K, n, p=pri)
        if np.bincount(y, minlength=K).min() < 2:
            continue
        d = int(rng.integers(1, 6))
        Z = rng.normal(size=(K, d))[y] * rng.uniform(0, 2) + rng.normal(size=(n, d))
        r = canonical_rho2(Z, y, K)[0] / max(per_class_r2(Z, y, K).max(), 1e-15) / (K - 1)
        worst = max(worst, r)
    out["bound_check_random_300"] = {"max_of_rho2_over_(K-1)DA": float(worst), "bound_violated": bool(worst > 1 + 1e-9)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
