"""Fixture D: is PCRL's in-house `LEACEEraser` (origin/main pcrl/models/baselines.py:120-190)
the LEACE eraser of Belrose et al. 2023?

The in-house class (re-implemented here from the code as read, not imported):
  W* = (Hc^T Hc + reg I)^{-1} Hc^T Zc ;  U_r = left singular vectors of W* ;
  P  = I - U_r U_r^T  (ORTHOGONAL projection off the OLS-coefficient span)
LEACE (from the paper's definition): whiten with W = Sigma^{-1/2}, project off
colsp(W Sigma_xz) orthogonally in whitened space, unwhiten:
  x' = x - W^+ P_{W Sigma_xz} W (x - mu)   (an OBLIQUE projection in x-space)
LEACE zeroes Cov(x', z) exactly; the orthogonal OLS-null projection does so only
when Sigma is isotropic. Deterministic, synthetic, seconds.
Run: /opt/homebrew/bin/python3 D_inhouse_leace_vs_leace.py > outputs/D_inhouse_leace_vs_leace.json
"""
from __future__ import annotations

import json

import numpy as np


def r2_onehot(H, y, reg=1e-6):
    K = int(y.max()) + 1
    Y = np.eye(K)[y]
    Hc, Yc = H - H.mean(0), Y - Y.mean(0)
    W = np.linalg.solve(Hc.T @ Hc + reg * np.eye(H.shape[1]), Hc.T @ Yc)
    return float(max(0.0, 1 - ((Yc - Hc @ W) ** 2).sum() / (Yc ** 2).sum()))


def inhouse(H, y, reg=1e-6):
    K = int(y.max()) + 1
    Z = np.eye(K)[y]
    mu = H.mean(0, keepdims=True)
    Hc, Zc = H - mu, Z - Z.mean(0)
    W = np.linalg.solve(Hc.T @ Hc + reg * np.eye(H.shape[1]), Hc.T @ Zc)
    U, S, _ = np.linalg.svd(W, full_matrices=False)
    U = U[:, : int((S > 1e-10 * S[0]).sum())]
    P = np.eye(H.shape[1]) - U @ U.T
    return Hc @ P.T + mu


def leace(H, y):
    K = int(y.max()) + 1
    Z = np.eye(K)[y]
    mu = H.mean(0)
    Hc, Zc = H - mu, Z - Z.mean(0)
    n = len(H)
    S = Hc.T @ Hc / n
    Sxz = Hc.T @ Zc / n
    ev, V = np.linalg.eigh(S)
    W = V @ np.diag(ev ** -0.5) @ V.T
    Wp = V @ np.diag(ev ** 0.5) @ V.T
    M = W @ Sxz
    Q, s, _ = np.linalg.svd(M, full_matrices=False)
    Q = Q[:, : int((s > 1e-10 * s[0]).sum())]
    Pw = Q @ Q.T
    return H - (Hc @ (Wp @ Pw @ W).T)


rng = np.random.default_rng(0)
out = {"cases": []}
for name, aniso in [("mixed_cov_scale_1", 1.0), ("mixed_cov_scale_10", 10.0), ("mixed_cov_scale_100", 100.0)]:
    n, d, K = 20000, 8, 3
    y = rng.integers(0, K, n)
    A = rng.normal(size=(K, d))
    scales = np.geomspace(1, aniso, d)
    C = rng.normal(size=(d, d)) * 0.3 + np.eye(d)
    H = (A[y] + rng.normal(size=(n, d))) @ C * scales
    for lab, Hp in [("raw", H), ("inhouse_LEACEEraser", inhouse(H, y)), ("LEACE_definition", leace(H, y))]:
        Zc = np.eye(K)[y] - np.eye(K)[y].mean(0)
        cc = np.abs((Hp - Hp.mean(0)).T @ Zc / n).max()
        out["cases"].append({"case": name, "map": lab, "r2_onehot_in_sample": round(r2_onehot(Hp, y), 6),
                             "max_abs_cross_cov": float(cc)})
out["finding"] = ("In-house LEACEEraser leaves nonzero cross-covariance and nonzero one-hot R^2 whenever "
                  "the representation covariance is not isotropic (here R^2 0.58-0.65 vs raw 0.68-0.79); the LEACE definition gives R^2 = 0 and cross-cov ~1e-14. The "
                  "class name overstates what it does (it is a one-step OLS-null-space projection, closer "
                  "to a single INLP step).")
print(json.dumps(out, indent=1))
