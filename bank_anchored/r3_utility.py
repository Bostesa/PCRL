"""Pure task utility for a bank-anchored output law.

No role file, protected label, candidate trainer or reference model is read here.
`mass` describes the law of the actual *single* released code pair. The
expected loss averages losses of fixed-table outputs, never probabilities.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


TASKS = (("income", 0, 2), ("occupation", 1, 6))
METRICS = ("accuracy", "nll", "brier")
FLOOR = 1e-12
CAPS = {
    "fit": {"accuracy": -.002, "nll": .001, "brier": .0005},
    "inner": {"accuracy": -.010, "nll": .010, "brier": .005},
}


def _validate(law: Mapping, tables: tuple[np.ndarray, np.ndarray],
              labels: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    required = {"code_r1", "code_r2", "mass", "valid", "b0_code_r1", "b0_code_r2"}
    if not isinstance(law, Mapping) or not required.issubset(law):
        raise ValueError("bank law fields missing")
    mass = np.asarray(law["mass"])
    valid = np.asarray(law["valid"])
    if (mass.dtype != np.float64 or mass.ndim != 2 or mass.shape[0] == 0
            or valid.dtype != np.bool_ or valid.shape != mass.shape
            or not np.isfinite(mass).all() or np.any(mass < 0)
            or np.any(mass[~valid] != 0)
            or not np.allclose(mass.sum(axis=1), 1, rtol=0, atol=1e-12)
            or np.any(~valid.any(axis=1))):
        raise ValueError("bank law mass or support invalid")
    n, slots = mass.shape
    if len(tables) != 2 or len(labels) != 2:
        raise ValueError("two fixed recipient tables and labels required")
    for _, purpose, width in TASKS:
        table = np.asarray(tables[purpose])
        code = np.asarray(law[f"code_r{purpose + 1}"])
        b0 = np.asarray(law[f"b0_code_r{purpose + 1}"])
        y = np.asarray(labels[purpose])
        if (table.dtype != np.float64 or table.ndim != 2
                or table.shape[1] != width or not len(table)
                or not np.isfinite(table).all() or np.any(table < 0)
                or not np.allclose(table.sum(axis=1), 1, rtol=0, atol=1e-10)
                or code.dtype != np.int64 or code.shape != (n, slots)
                or b0.dtype != np.int64 or b0.shape != (n,)
                or np.any(code < 0) or np.any(code >= len(table))
                or np.any(b0 < 0) or np.any(b0 >= len(table))
                or y.dtype != np.int64 or y.shape != (n,)
                or np.any(y < 0) or np.any(y >= width)):
            raise ValueError("fixed bank table, code or label schema invalid")
        if np.any(valid & (table[code].argmax(axis=2) != table[b0].argmax(axis=1)[:, None])):
            raise ValueError("action changes frozen bank decision class")
    same_b0 = ((np.asarray(law["code_r1"]) == np.asarray(law["b0_code_r1"])[:, None])
               & (np.asarray(law["code_r2"]) == np.asarray(law["b0_code_r2"])[:, None])
               & valid)
    if np.any(~same_b0.any(axis=1)):
        raise ValueError("exact bank identity action absent from support")
    return mass, valid


def _costs(prob: np.ndarray, label: np.ndarray) -> dict[str, np.ndarray]:
    width = prob.shape[-1]
    correct = (prob.argmax(axis=-1) == label)
    chosen = np.take_along_axis(prob, label[..., None], axis=-1)[..., 0]
    onehot = np.eye(width, dtype=np.float64)[label]
    return {"accuracy": correct.astype(np.float64),
            "nll": -np.log(np.maximum(chosen, FLOOR)),
            "brier": np.sum((prob - onehot) ** 2, axis=-1)}


def evaluate_law(law: Mapping, tables: tuple[np.ndarray, np.ndarray],
                 labels: tuple[np.ndarray, np.ndarray]) -> dict[str, dict]:
    """Return per-person expected and B0 task costs for both recipients.

    Duplicate public action pairs contribute only their summed probability
    mass. Nonzero invalid-slot mass and action class switching are rejected.
    """
    mass, _ = _validate(law, tables, labels)
    out = {}
    for task, purpose, _ in TASKS:
        table = np.asarray(tables[purpose])
        code = np.asarray(law[f"code_r{purpose + 1}"])
        b0_code = np.asarray(law[f"b0_code_r{purpose + 1}"])
        y = np.asarray(labels[purpose])
        action = _costs(table[code], np.broadcast_to(y[:, None], code.shape))
        baseline = _costs(table[b0_code], y)
        expected = {metric: np.sum(mass * action[metric], axis=1)
                    for metric in METRICS}
        out[task] = {"expected": expected, "b0": baseline}
    return out


def point_clause(metric: str, margin: float, regime: str) -> dict:
    """Apply inclusive signed-margin gate; accuracy is candidate minus ref."""
    if regime not in CAPS or metric not in METRICS or not np.isfinite(margin):
        raise ValueError("point clause regime, metric or value invalid")
    cap = CAPS[regime][metric]
    passes = margin >= cap if metric == "accuracy" else margin <= cap
    return {"metric": metric, "regime": regime, "margin": float(margin),
            "cap": cap, "status": "PASS" if passes else "FAIL"}
