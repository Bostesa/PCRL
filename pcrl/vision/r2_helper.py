"""Tikhonov-regularised one-hot linear R^2.

Mirrors ``pcrl.training.v2_trainer._linear_r2_train`` so vision compliance
metrics are directly comparable to the tabular pipeline. Re-implemented here
(not imported) because v2_trainer is locked.
"""
from __future__ import annotations

import numpy as np


def linear_r2(H, Z, reg: float = 1e-6) -> float:
    """Train-set linear R^2 of optimal Tikhonov-regularised one-hot predictor.

    Args:
        H: (N, d) features.
        Z: (N,) integer class labels.
        reg: ridge.

    Returns:
        scalar R^2 in [0, 1].
    """
    Z = Z.astype("int64") if hasattr(Z, "astype") else np.asarray(Z, dtype=np.int64)
    n_classes = int(Z.max()) + 1
    Z_oh = np.eye(n_classes)[Z]
    n, d = H.shape
    H_c = H - H.mean(axis=0, keepdims=True)
    Z_c = Z_oh - Z_oh.mean(axis=0, keepdims=True)
    gram = H_c.T @ H_c + reg * np.eye(d)
    W = np.linalg.solve(gram, H_c.T @ Z_c)
    Z_pred = H_c @ W
    ss_res = ((Z_c - Z_pred) ** 2).sum()
    ss_tot = (Z_c ** 2).sum()
    return float(max(0.0, 1.0 - ss_res / max(ss_tot, 1e-12)))
