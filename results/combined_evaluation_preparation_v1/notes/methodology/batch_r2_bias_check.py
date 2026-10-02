"""Synthetic check: null bias of the PCRL TRAINING-TIME / Cotter-selection R^2.

PCRL origin/main pcrl/training/losses.py:VerificationRegularizer (ridge 1e-4, centred, in-sample) is
applied per minibatch (v2_trainer.py:880 training constraint; :1090 validation/Cotter selection),
batch_size 256 (v2_trainer.py:202), representation dim 64. With NO attribute signal, the in-sample
R^2 of a one-hot attribute on n rows and r non-degenerate directions is about r/(n-1).
This script measures that null level for several effective ranks. Synthetic data only; numpy only.
Run: python3 batch_r2_bias_check.py
"""
import json

import numpy as np


def r2_ridge(H, z, reg):
    k = int(z.max()) + 1
    Z = np.eye(k)[z]; Hc = H - H.mean(0); Zc = Z - Z.mean(0)
    W = np.linalg.solve(Hc.T @ Hc + reg * np.eye(H.shape[1]), Hc.T @ Zc)
    return max(0.0, 1 - ((Zc - Hc @ W) ** 2).sum() / (Zc ** 2).sum())


rng = np.random.default_rng(3)
out = {}
for n in (256, 4096):
    for r in (64, 32, 16, 8, 3):
        vals = []
        for _ in range(200):
            z = rng.integers(0, 2, n)                         # binary attribute, independent of H
            B = rng.standard_normal((r, 64))
            H = 0.3 * rng.standard_normal((n, r)) @ B / np.sqrt(r)   # rank-r representation, no signal
            vals.append(r2_ridge(H, z, 1e-4))
        out[f"n={n},rank={r}"] = {"mean_null_r2": float(np.mean(vals)), "approx_r_over_n_minus_1": r / (n - 1),
                                  "share_above_0.05": float(np.mean(np.array(vals) > 0.05))}
print(json.dumps(out, indent=1))
