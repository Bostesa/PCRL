"""Synthetic diagnostic: can the historical aggregate path produce identity residuals?

At 135e440e6b16ca753465d0492193e33e63280dff (the commit that wrote the committed
dominant_axis_audit.json files), LinearComplianceCertificate.check solved the
ridge system on the caller's dtype (float32 from torch representations), while
compute_dominant_axis_r2 cast to float64. This script reproduces both
conventions on synthetic, nearly collinear 64-d float32 representations and
reports |aggregate - sum_k w_k R2_k|. It does not open any checkpoint and does
not establish what happened in a particular historical cell.
"""
import json
import sys

import numpy as np


def agg_historical(H, y, K, lam=1e-6):  # dtype of H preserved (float32 path)
    Z = np.eye(K)[y].astype(H.dtype)
    Hc = H - H.mean(0, keepdims=True)
    Zc = Z - Z.mean(0, keepdims=True)
    G = Hc.T @ Hc + lam * np.eye(H.shape[1], dtype=H.dtype)
    W = np.linalg.solve(G, Hc.T @ Zc)
    return max(0.0, float(1 - ((Zc - Hc @ W) ** 2).sum() / (Zc ** 2).sum()))


def per_class_float64(H, y, K, lam=1e-6):
    H = H.astype(np.float64)
    Hc = H - H.mean(0, keepdims=True)
    G = Hc.T @ Hc + lam * np.eye(H.shape[1])
    out, pri = [], []
    for k in range(K):
        z = (y == k).astype(float)
        zc = z - z.mean()
        w = np.linalg.solve(G, Hc.T @ zc)
        out.append(1 - ((zc - Hc @ w) ** 2).sum() / (zc ** 2).sum())
        pri.append(z.mean())
    p = np.array(pri)
    wts = p * (1 - p) / (p * (1 - p)).sum()
    return float(wts @ np.array(out))


rows = []
for seed in range(20):
    rng = np.random.default_rng(seed)
    n, d, K, r = 6000, 64, 4, 4  # effective rank ~4 of 64 dims (collapsed representation)
    y = rng.choice(K, size=n, p=[0.18, 0.52, 0.265, 0.035])
    B = rng.normal(size=(r, d))
    H = (rng.normal(size=(n, r)) + 0.08 * np.eye(K)[y]) @ B + 1e-3 * rng.normal(size=(n, d))
    H = (H * 0.3).astype(np.float32)
    a32 = agg_historical(H, y, K)
    a64 = agg_historical(H.astype(np.float64), y, K)
    combo = per_class_float64(H, y, K)
    rows.append({"seed": seed, "agg_float32": a32, "agg_float64": a64, "combo_float64": combo,
                 "resid_float32": abs(a32 - combo), "resid_float64": abs(a64 - combo)})
summary = {"max_resid_float32": max(r["resid_float32"] for r in rows),
           "max_resid_float64": max(r["resid_float64"] for r in rows),
           "median_resid_float32": float(np.median([r["resid_float32"] for r in rows]))}
json.dump({"summary": summary, "rows": rows}, sys.stdout, indent=1)
