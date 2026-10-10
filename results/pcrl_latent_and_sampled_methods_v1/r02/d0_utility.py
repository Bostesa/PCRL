"""Point utility screen for historical D0 score releases on INNER people.

This pure evaluator handles one release and one source seed. Source admission,
R03 custody and role access live in the separate queued worker. An arm is
eligible for A comparator escalation if *either* seed passes this full gate;
the worker never chooses a favorable seed or a release draw.
"""

from __future__ import annotations

import numpy as np


TASKS = (("income", "q1", "d1", 2), ("occupation", "q2", "d2", 6))
REFS = ("U0", "Ucal")
CLIP = 1e-12


def _valid_q(q: np.ndarray, n: int, width: int) -> np.ndarray:
    q = np.asarray(q)
    if (q.dtype != np.float64 or q.shape != (n, width)
            or not np.isfinite(q).all() or np.any(q < 0)
            or np.max(np.abs(q.sum(axis=1) - 1), initial=0) > 1e-8):
        raise ValueError("D0/reference probability wire differs")
    return q


def _metrics(q: np.ndarray, y: np.ndarray, width: int) -> dict:
    decision = np.argmax(q, axis=1)
    onehot = np.eye(width, dtype=np.float64)[y]
    return {"rows": int(len(y)),
            "accuracy": float(np.mean(decision == y)),
            "nll": float(np.mean(-np.log(np.maximum(q[np.arange(len(y)), y], CLIP)))),
            "brier": float(np.mean(np.sum((q - onehot) ** 2, axis=1))),
            "per_class_recall": [float(np.mean(decision[y == c] == c))
                                 if np.any(y == c) else None for c in range(width)],
            "class_counts": [int(np.sum(y == c)) for c in range(width)]}


def _comparisons(candidate: dict, references: dict) -> dict:
    out = {}
    for name in REFS:
        ref = references[name]
        delta = {metric: candidate[metric] - ref[metric]
                 for metric in ("accuracy", "nll", "brier")}
        out[name] = {"delta": delta,
                     "point_eligible": (delta["accuracy"] >= -0.010
                                        and delta["nll"] <= 0.010
                                        and delta["brier"] <= 0.005)}
    return out


def evaluate_inner_utility(release: dict, references: dict,
                           labels: dict) -> dict:
    """Compute all 12 fixed utility clauses on identical ordered INNER rows."""
    if set(references) != set(REFS) or set(labels) != {"income", "occupation"}:
        raise ValueError("D0 utility requires both references and both tasks")
    n = len(np.asarray(labels["income"]))
    out = {}
    for task, qkey, dkey, width in TASKS:
        y = np.asarray(labels[task])
        if y.dtype != np.int64 or y.shape != (n,) or np.any((y < 0) | (y >= width)):
            raise ValueError("INNER task label schema differs")
        q = _valid_q(release[qkey], n, width)
        d = np.asarray(release[dkey])
        if d.dtype != np.int64 or d.shape != (n,) or not np.array_equal(d, np.argmax(q, axis=1)):
            raise ValueError("D0 decision differs from strict argmax of emitted q")
        candidate = _metrics(q, y, width)
        ref_metrics = {name: _metrics(_valid_q(references[name][qkey], n, width),
                                      y, width) for name in REFS}
        comparison = _comparisons(candidate, ref_metrics)
        out[task] = {"candidate": candidate, "references": ref_metrics,
                     "comparisons": comparison,
                     "point_eligible_both_refs": all(comparison[name]["point_eligible"]
                                                      for name in REFS)}
    return {"rows": n, "tasks": out,
            "point_eligible_all_tasks_refs": all(out[t]["point_eligible_both_refs"]
                                                  for t, *_ in TASKS)}
