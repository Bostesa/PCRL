"""Independent rank AUC and ATT tune split for later privacy replay.

This module does not import attack runner scoring or selection functions.
"""

from __future__ import annotations

import hashlib

import numpy as np


SPLIT_SALT = "pcrl-method-portfolio-v1|attack-tune|20261008|"


def rank_auc(label: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUC with midranks for exactly tied scores."""
    y = np.asarray(label)
    p = np.asarray(score, dtype=np.float64)
    if (y.ndim != 1 or p.shape != y.shape or len(y) == 0
            or not np.isfinite(p).all() or not np.isin(y, (0, 1)).all()):
        raise ValueError("binary labels and finite aligned scores required")
    positive = int(np.count_nonzero(y == 1))
    negative = len(y) - positive
    if not positive or not negative:
        raise ValueError("AUC requires both protected-label classes")
    order = np.argsort(p, kind="mergesort")
    sorted_score = p[order]
    sorted_label = y[order]
    rank_sum = 0.0
    start = 0
    while start < len(y):
        stop = start + 1
        while stop < len(y) and sorted_score[stop] == sorted_score[start]:
            stop += 1
        midrank = (start + 1 + stop) / 2.0
        rank_sum += midrank * int(np.count_nonzero(sorted_label[start:stop] == 1))
        start = stop
    return float((rank_sum - positive * (positive + 1) / 2) /
                 (positive * negative))


def alias_auc(label: np.ndarray, score: np.ndarray) -> dict:
    direct = rank_auc(label, score)
    return {"positive_alias_auc": direct, "reverse_alias_auc": 1 - direct,
            "fixed_alias_max_auc": max(direct, 1 - direct)}


def attack_tune_indices(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct source's fixed ATT group split without importing it."""
    unit = np.asarray(groups)
    if unit.dtype != np.int64 or unit.ndim != 1 or np.unique(unit).size != 4061:
        raise ValueError("ATT unit identity/count differs")
    ordered = sorted(np.unique(unit), key=lambda value: (
        hashlib.sha256((SPLIT_SALT + str(int(value))).encode()).hexdigest(),
        int(value)))
    tune_groups = np.array(ordered[:1015], dtype=np.int64)
    tune_flag = np.isin(unit, tune_groups)
    if not tune_flag.any() or tune_flag.all():
        raise ValueError("ATT tune split is empty")
    return np.flatnonzero(~tune_flag), np.flatnonzero(tune_flag)
