"""Pure independent reconstruction of complete saved release arrays.

No R1/R2/R3/R4 service, runner, scoring, or selection implementation is
imported. A separate queued caller supplies byte-pinned arrays and packages.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib

import numpy as np

from results.pcrl_code_utility_witness_v1.r5.independent_decoder import (
    continuous_head_probabilities, probabilities_from_hard_bits,
)
from results.pcrl_code_utility_witness_v1.r5.independent_checks import task_means, task_rows


HARD_MEMBERS = frozenset({"row_id", "unit", "code0", "prob0", "decision0",
                          "code1", "prob1", "decision1"})
FEATURE_MEMBERS = frozenset({"row_id", "unit", "features0", "prob0", "decision0",
                             "features1", "prob1", "decision1"})


def tensor_digest(state: Mapping[str, np.ndarray]) -> str:
    """Recompute the package's sorted tensor-content digest independently."""
    digest = hashlib.sha256()
    for name in sorted(state):
        array = np.ascontiguousarray(np.asarray(state[name]))
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError("package tensor dtype or values differ")
        digest.update(name.encode() + b"\0")
        digest.update(str(array.dtype).encode() + b"\0")
        digest.update(str(array.shape).encode() + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def _wire_common(wire: Mapping[str, np.ndarray], row_id: np.ndarray,
                 unit: np.ndarray, kind: str) -> None:
    members = FEATURE_MEMBERS if kind == "feature" else HARD_MEMBERS
    if set(wire) != members:
        raise ValueError("complete transmitted array inventory differs")
    row, group = np.asarray(wire["row_id"]), np.asarray(wire["unit"])
    if (row.dtype != np.int64 or group.dtype != np.int64
            or not np.array_equal(row, row_id) or not np.array_equal(group, unit)):
        raise ValueError("release and admitted ordered people differ")


def _metric_summary(wire: Mapping[str, np.ndarray], labels: tuple[np.ndarray, np.ndarray]) -> dict:
    result = {}
    for purpose, task, classes in ((0, "income", 2), (1, "occupation", 6)):
        rows = task_rows(labels[purpose], wire[f"prob{purpose}"], classes,
                         wire[f"decision{purpose}"])
        value = task_means(rows)
        counts = np.bincount(labels[purpose], minlength=classes)
        majority = int(np.argmax(counts))
        decision = wire[f"decision{purpose}"]
        value.update({
            "constant_accuracy": float(np.mean(labels[purpose] == majority)),
            "accuracy_gain_over_constant": float(value["accuracy"] - np.mean(labels[purpose] == majority)),
            "class_counts": counts.tolist(),
            "per_class_recall": [float(np.mean(decision[labels[purpose] == cls] == cls))
                                 if counts[cls] else None for cls in range(classes)],
        })
        result[task] = value
    return result


def replay_hard(wire: Mapping[str, np.ndarray], row_id: np.ndarray,
                unit: np.ndarray, labels: tuple[np.ndarray, np.ndarray],
                state: Mapping[str, np.ndarray], bits: int,
                alphas: tuple[float | None, float | None]) -> dict:
    """Validate the actual hard-bit wire and independently rebuild both heads."""
    _wire_common(wire, row_id, unit, "hard")
    if bits not in (10, 16):
        raise ValueError("unsealed bit count")
    occupied = {}
    largest = 0.0
    for purpose, task, width in ((0, "income", 2), (1, "occupation", 6)):
        code = np.asarray(wire[f"code{purpose}"])
        probability = np.asarray(wire[f"prob{purpose}"])
        decision = np.asarray(wire[f"decision{purpose}"])
        if (code.dtype != np.uint8 or code.shape != (len(row_id), bits)
                or probability.dtype != np.float64
                or probability.shape != (len(row_id), width)
                or decision.dtype != np.int64 or decision.shape != (len(row_id),)):
            raise ValueError("hard-code wire dtype or shape differs")
        rebuilt = probabilities_from_hard_bits(state, code, purpose, bits, alphas[purpose])
        gap = float(np.max(np.abs(probability - rebuilt)))
        if gap > 2e-12 or not np.array_equal(decision, rebuilt.argmax(axis=1)):
            raise ValueError("output bypasses or differs from public hard bits")
        largest = max(largest, gap)
        _, count = np.unique(code, axis=0, return_counts=True)
        frequency = count / count.sum()
        occupied[task] = {"global_capacity_states": 1 << bits,
                          "occupied_states": int(len(count)),
                          "entropy_bits": float(-(frequency * np.log2(frequency)).sum())}
    return {"metrics": _metric_summary(wire, labels), "states": occupied,
            "maximum_decoder_probability_gap": largest, "rows": len(row_id)}


def replay_bank(wire: Mapping[str, np.ndarray], row_id: np.ndarray,
                unit: np.ndarray, labels: tuple[np.ndarray, np.ndarray],
                tables: tuple[np.ndarray, np.ndarray]) -> dict:
    """Decode canonical ten-bit bank IDs using only fixed public tables."""
    _wire_common(wire, row_id, unit, "bank")
    states = {}
    largest = 0.0
    for purpose, task, classes in ((0, "income", 2), (1, "occupation", 6)):
        code = np.asarray(wire[f"code{purpose}"])
        p = np.asarray(wire[f"prob{purpose}"])
        d = np.asarray(wire[f"decision{purpose}"])
        table = np.asarray(tables[purpose], dtype=np.float64)
        if (code.dtype != np.uint8 or code.shape != (len(row_id), 10)
                or not np.isin(code, (0, 1)).all()
                or p.dtype != np.float64 or p.shape != (len(row_id), classes)
                or d.dtype != np.int64 or d.shape != (len(row_id),)
                or table.ndim != 2 or table.shape[1] != classes
                or not 1 <= len(table) <= 1024):
            raise ValueError("bank wire/table schema differs")
        ids = code.astype(np.int64) @ (1 << np.arange(10, dtype=np.int64))
        if np.any(ids >= len(table)):
            raise ValueError("bank code lies outside fixed public table")
        rebuilt = table[ids]
        gap = float(np.max(np.abs(p - rebuilt)))
        if gap > 2e-12 or not np.array_equal(d, rebuilt.argmax(axis=1)):
            raise ValueError("bank output bypasses public code")
        largest = max(largest, gap)
        states[task] = {"global_table_states": int(len(table)),
                        "occupied_states": int(np.unique(ids).size),
                        "minimum_capacity_bits": int((len(table)-1).bit_length())}
    return {"metrics": _metric_summary(wire, labels), "states": states,
            "maximum_decoder_probability_gap": largest, "rows": len(row_id)}


def replay_feature(wire: Mapping[str, np.ndarray], row_id: np.ndarray,
                   unit: np.ndarray, labels: tuple[np.ndarray, np.ndarray],
                   state: Mapping[str, np.ndarray],
                   alphas: tuple[float | None, float | None]) -> dict:
    """Rebuild F's output from the actual transmitted continuous features."""
    _wire_common(wire, row_id, unit, "feature")
    largest = 0.0
    for purpose, classes in ((0, 2), (1, 6)):
        p = np.asarray(wire[f"prob{purpose}"])
        d = np.asarray(wire[f"decision{purpose}"])
        if p.dtype != np.float64 or p.shape != (len(row_id), classes) or d.dtype != np.int64:
            raise ValueError("F output wire schema differs")
        rebuilt = continuous_head_probabilities(state, wire[f"features{purpose}"],
                                                purpose, alphas[purpose])
        gap = float(np.max(np.abs(p - rebuilt)))
        if gap > 2e-12 or not np.array_equal(d, rebuilt.argmax(axis=1)):
            raise ValueError("F output bypasses transmitted features")
        largest = max(largest, gap)
    return {"metrics": _metric_summary(wire, labels),
            "maximum_decoder_probability_gap": largest, "rows": len(row_id),
            "states": {"income": {"transmitted_continuous_dimensions": 16},
                       "occupation": {"transmitted_continuous_dimensions": 16}}}
