"""M5 endpoint arithmetic independent of M4 and inherited evaluator imports.

The caller supplies already admitted, identically ordered role arrays. This
module has no data loader, model fit, selection rule, or result writer.
"""

from __future__ import annotations

import numpy as np


NLL_CLIP = 1e-12


def binary_auc(labels: object, score: object) -> float:
    """Mann-Whitney AUC with half credit for exact score ties."""
    y = np.asarray(labels)
    s = np.asarray(score, dtype=np.float64)
    if (y.ndim != 1 or s.shape != y.shape or len(y) == 0
            or not np.all(np.isfinite(s)) or not np.all(np.isin(y, (0, 1)))):
        raise ValueError("binary AUC needs finite aligned labels and scores")
    positives = int(np.count_nonzero(y == 1))
    negatives = len(y) - positives
    if not positives or not negatives:
        raise ValueError("binary AUC needs both classes")
    order = np.argsort(s, kind="mergesort")
    sorted_y, sorted_s = y[order], s[order]
    concordant = 0.0
    negatives_before = 0
    i = 0
    while i < len(y):
        j = i + 1
        while j < len(y) and sorted_s[j] == sorted_s[i]:
            j += 1
        positive_tied = int(np.count_nonzero(sorted_y[i:j] == 1))
        negative_tied = j - i - positive_tied
        concordant += positive_tied * (negatives_before + 0.5 * negative_tied)
        negatives_before += negative_tied
        i = j
    return float(concordant / (positives * negatives))


def task_metrics(labels: object, probability: object, *, width: int) -> dict:
    """Accuracy, natural-log NLL, summed multiclass Brier and class recall."""
    y = np.asarray(labels)
    p = np.asarray(probability, dtype=np.float64)
    if (type(width) is not int or width < 2 or p.ndim != 2
            or p.shape != (len(y), width) or len(y) == 0
            or not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1)
            or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-8)
            or y.shape != (len(p),) or not np.issubdtype(y.dtype, np.integer)
            or np.any((y < 0) | (y >= width))):
        raise ValueError("task probabilities or ordered labels invalid")
    predicted = np.argmax(p, axis=1)
    rows = np.arange(len(y))
    nll = -np.log(np.maximum(p[rows, y], NLL_CLIP))
    onehot = np.eye(width, dtype=np.float64)[y]
    brier = np.sum((p - onehot) ** 2, axis=1)
    counts = [int(np.count_nonzero(y == cls)) for cls in range(width)]
    recall = [float(np.mean(predicted[y == cls] == cls)) if counts[cls] else None
              for cls in range(width)]
    return {
        "rows": len(y),
        "accuracy": float(np.mean(predicted == y)),
        "nll": float(np.mean(nll)),
        "brier": float(np.mean(brier)),
        "class_counts": counts,
        "per_class_recall": recall,
    }
