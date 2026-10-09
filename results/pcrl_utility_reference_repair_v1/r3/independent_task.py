"""R3 task endpoints and inclusive point gate, independent of runner code."""

from __future__ import annotations

from decimal import Decimal
from typing import Mapping

import numpy as np


CAPS = {"accuracy": 0.010, "nll": 0.010, "brier": 0.005}


def task_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels)
    if (p.ndim != 2 or p.shape[0] == 0 or p.shape[1] < 2
            or y.ndim != 1 or y.shape[0] != p.shape[0]
            or y.dtype.kind not in "iu" or not np.isfinite(p).all()
            or np.any(p < 0) or np.any(p > 1)
            or np.max(np.abs(p.sum(axis=1) - 1.0)) > 1e-8
            or np.any(y < 0) or np.any(y >= p.shape[1])):
        raise ValueError("invalid aligned probabilities or labels")
    y = y.astype(np.int64, copy=False)
    prediction = np.argmax(p, axis=1)
    class_counts = np.bincount(y, minlength=p.shape[1])
    correct_counts = np.bincount(y[prediction == y], minlength=p.shape[1])
    recall = [float(correct_counts[i] / class_counts[i]) if class_counts[i] else None
              for i in range(p.shape[1])]
    onehot = np.eye(p.shape[1], dtype=np.float64)[y]
    chosen = p[np.arange(len(y)), y]
    return {
        "accuracy": float(np.mean(prediction == y)),
        "nll": float(np.mean(-np.log(np.maximum(chosen, 1e-12)))),
        "brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "recall": recall,
        "class_counts": [int(n) for n in class_counts],
    }


def gate_against_references(candidate: Mapping[str, float],
                            references: Mapping[str, Mapping[str, float]]) -> dict:
    """Return costs and margins. Decimal(str(float)) avoids binary cap ambiguity."""
    if set(references) != {"U0", "Ucal"}:
        raise ValueError("both frozen references are required")
    costs = {}
    margins = {}
    by_reference = {}
    for name, ref in references.items():
        cost_dec = {
            "accuracy": Decimal(str(ref["accuracy"])) - Decimal(str(candidate["accuracy"])),
            "nll": Decimal(str(candidate["nll"])) - Decimal(str(ref["nll"])),
            "brier": Decimal(str(candidate["brier"])) - Decimal(str(ref["brier"])),
        }
        costs[name] = {key: float(value) for key, value in cost_dec.items()}
        margins[name] = {key: float(Decimal(str(CAPS[key])) - value)
                         for key, value in cost_dec.items()}
        by_reference[name] = all(value <= Decimal(str(CAPS[key]))
                                 for key, value in cost_dec.items())
    return {"caps": dict(CAPS), "costs": costs, "margins": margins,
            "by_reference": by_reference,
            "point_eligible": all(by_reference.values())}
