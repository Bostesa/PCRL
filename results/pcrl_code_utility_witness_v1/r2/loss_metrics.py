"""Small pure-array task and distillation metrics for the pre-refit audit."""

from __future__ import annotations

import numpy as np


FLOOR = 1e-12
TASKS = {"income": 2, "occupation": 6}


def validate(prob: np.ndarray, y: np.ndarray, task: str) -> tuple[np.ndarray, np.ndarray]:
    width = TASKS[task]
    p = np.asarray(prob)
    label = np.asarray(y)
    if (p.dtype != np.float64 or p.ndim != 2 or p.shape[1] != width
            or label.dtype != np.int64 or label.shape != (len(p),)
            or not np.isfinite(p).all() or np.any(p < 0)
            or not np.allclose(p.sum(axis=1), 1, atol=1e-8, rtol=0)
            or np.any((label < 0) | (label >= width))):
        raise ValueError("task probability, class order or label schema differs")
    return p, label


def task_metrics(prob: np.ndarray, y: np.ndarray, task: str) -> dict:
    p, label = validate(prob, y, task)
    width = TASKS[task]
    decision = np.argmax(p, axis=1)
    nll = -np.log(np.maximum(p[np.arange(len(label)), label], FLOOR))
    onehot = np.eye(width)[label]
    brier = np.sum((p - onehot) ** 2, axis=1)
    counts = np.bincount(label, minlength=width)
    majority = int(np.argmax(counts))
    return {
        "accuracy": float(np.mean(decision == label)),
        "nll": float(np.mean(nll)),
        "brier": float(np.mean(brier)),
        "constant_accuracy": float(np.mean(label == majority)),
        "accuracy_gain_over_constant": float(np.mean(decision == label) - np.mean(label == majority)),
        "class_counts": counts.astype(int).tolist(),
        "per_class_recall": [float(np.mean(decision[label == c] == c)) if counts[c]
                             else None for c in range(width)],
    }


def distillation_kl(teacher: np.ndarray, student: np.ndarray, task: str) -> float:
    """KL(teacher || uncalibrated head), matching PyTorch batchmean semantics."""
    width = TASKS[task]
    p = np.asarray(teacher)
    q = np.asarray(student)
    if (p.dtype != np.float64 or q.dtype != np.float64 or p.ndim != 2
            or p.shape != q.shape or p.shape[1] != width or not np.isfinite(p).all()
            or not np.isfinite(q).all() or np.any(p < 0) or np.any(q < 0)
            or not np.allclose(p.sum(axis=1), 1, atol=1e-8, rtol=0)
            or not np.allclose(q.sum(axis=1), 1, atol=1e-8, rtol=0)):
        raise ValueError("teacher/student probability schema differs")
    positive = p > 0
    term = np.zeros_like(p)
    term[positive] = p[positive] * (np.log(p[positive]) - np.log(np.maximum(q[positive], FLOOR)))
    return float(np.mean(np.sum(term, axis=1)))


def code_states(bits: np.ndarray) -> dict:
    code = np.asarray(bits)
    if (code.dtype != np.uint8 or code.ndim != 2 or code.shape[1] != 10
            or not np.isin(code, (0, 1)).all()):
        raise ValueError("old or compiled code requires ten hard uint8 bits")
    ids = (code.astype(np.int64) * (1 << np.arange(10, dtype=np.int64))).sum(axis=1)
    values, counts = np.unique(ids, return_counts=True)
    return {"global_capacity_states": 1024, "occupied_states": int(len(values)),
            "observed_entropy_bits": float(-np.sum((counts / len(code)) * np.log2(counts / len(code))))}
