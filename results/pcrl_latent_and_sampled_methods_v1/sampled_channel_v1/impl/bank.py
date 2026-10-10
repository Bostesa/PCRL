"""Fixed public score bank and input-local choices with global code IDs.

Only the decoded probability vector and strict decision determine a public ID.
The old token and the input-local choice position remain provider-private.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DISTANCE_FLOOR = 1e-12


def _probability_rows(value) -> np.ndarray:
    rows = np.asarray(value)
    if rows.dtype != np.dtype("float64") or rows.ndim != 2 or min(rows.shape) < 1:
        raise ValueError("public bank needs nonempty float64 probability rows")
    if (not np.isfinite(rows).all() or np.any(rows < 0) or
            not np.allclose(rows.sum(axis=1), 1, atol=1e-12, rtol=0)):
        raise ValueError("public bank rows must be probability vectors")
    out = np.ascontiguousarray(rows, dtype="<f8").copy()
    out[out == 0] = 0.0  # A signed floating zero is the same public value.
    decision = np.argmax(out, axis=1)
    if out.shape[1] > 1:
        others = np.where(np.eye(out.shape[1], dtype=bool)[decision], -np.inf, out)
        if np.any(out[np.arange(len(out)), decision] <= np.max(others, axis=1)):
            raise ValueError("every bank vector needs a strict predicted class")
    return out


def symmetric_kl(left, right) -> float:
    """Symmetric KL after a distance-only 1e-12 floor and renormalization."""
    p, q = (np.asarray(x, dtype=np.float64) for x in (left, right))
    if p.ndim != 1 or q.shape != p.shape or p.size < 1:
        raise ValueError("distance vectors have incompatible shapes")
    if (not np.isfinite(p).all() or not np.isfinite(q).all() or
            np.any(p < 0) or np.any(q < 0)):
        raise ValueError("distance vectors must be finite and nonnegative")
    p = np.maximum(p, DISTANCE_FLOOR)
    q = np.maximum(q, DISTANCE_FLOOR)
    p = p / p.sum()
    q = q / q.sum()
    return float(0.5 * (np.sum(p * np.log(p / q)) +
                        np.sum(q * np.log(q / p))))


@dataclass(frozen=True)
class FixedBank:
    probability: np.ndarray
    decision: np.ndarray
    old_to_global: tuple[int, ...]

    @classmethod
    def from_source(cls, source_probability, token_id, token_class) -> "FixedBank":
        source = _probability_rows(source_probability)
        source_class = np.argmax(source, axis=1)
        old_ids = np.asarray(token_id)
        old_classes = np.asarray(token_class)
        if (old_ids.dtype != np.dtype("int64") or
                old_ids.shape != (len(source),) or
                not np.array_equal(old_ids, np.arange(len(source), dtype=np.int64)) or
                old_classes.dtype != np.dtype("int64") or
                old_classes.shape != (len(source),) or
                not np.array_equal(old_classes, source_class)):
            raise ValueError("source token IDs/order/classes differ from decoded bank")
        keys = [(int(source_class[i]), source[i].tobytes()) for i in range(len(source))]
        unique = sorted(set(keys))
        index = {key: i for i, key in enumerate(unique)}
        probability = np.stack([
            np.frombuffer(key[1], dtype="<f8").copy() for key in unique])
        decision = np.array([key[0] for key in unique], dtype=np.int64)
        probability.setflags(write=False)
        decision.setflags(write=False)
        return cls(probability, decision, tuple(index[key] for key in keys))


def class_medoid(bank: FixedBank, predicted_class: int) -> int:
    """Unweighted symmetric-KL medoid, breaking ties by global code ID."""
    ids = np.flatnonzero(bank.decision == predicted_class)
    if len(ids) == 0:
        raise ValueError("class has no public bank entry")
    total = [sum(symmetric_kl(bank.probability[i], bank.probability[j])
                 for j in ids) for i in ids]
    return int(min(zip(total, ids), key=lambda pair: (pair[0], pair[1]))[1])


def build_actions(bank: FixedBank, *, input_class, fine_to_old_token) -> tuple[tuple[int, ...], ...]:
    """Identity, two nearest same-class public codes, class medoid; dedup."""
    classes = np.asarray(input_class)
    old_ids = np.asarray(fine_to_old_token)
    if (classes.dtype != np.dtype("int64") or old_ids.dtype != np.dtype("int64") or
            classes.ndim != 1 or old_ids.shape != classes.shape or len(classes) < 1):
        raise ValueError("fine-cell class and old-token maps need matching int64 vectors")
    if np.any((old_ids < 0) | (old_ids >= len(bank.old_to_global))):
        raise ValueError("fine-cell old token is outside the fixed source bank")
    medoid = {int(c): class_medoid(bank, int(c)) for c in np.unique(classes)}
    answer = []
    for c, old_id in zip(classes, old_ids):
        c = int(c)
        identity = bank.old_to_global[int(old_id)]
        if int(bank.decision[identity]) != c:
            raise ValueError("identity public code changes predicted class")
        neighbors = [int(i) for i in np.flatnonzero(bank.decision == c) if i != identity]
        neighbors.sort(key=lambda i: (symmetric_kl(bank.probability[identity],
                                                   bank.probability[i]), i))
        ordered = (identity, *neighbors[:2], medoid[c])
        answer.append(tuple(dict.fromkeys(ordered)))
    return tuple(answer)
