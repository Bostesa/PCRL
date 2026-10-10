"""Independent scalar checks for bank-release verification.

This module intentionally does not import candidate, runner, endpoint or
selection code. It reads only explicit public tables and saved public laws.
"""

from __future__ import annotations

import math

import numpy as np


def public_id_from_bits(bits: np.ndarray) -> np.ndarray:
    value = np.asarray(bits)
    if (value.ndim != 2 or value.shape[1] != 10
            or not np.issubdtype(value.dtype, np.integer)
            or not np.isin(value, (0, 1)).all()):
        raise ValueError("public code must be ten binary columns")
    return np.asarray([
        sum(int(bit) << index for index, bit in enumerate(row))
        for row in value
    ], dtype=np.int64)


def audit_table(table: np.ndarray, classes: np.ndarray) -> dict[str, int]:
    values = np.asarray(table)
    labels = np.asarray(classes)
    if (values.ndim != 2 or values.shape[1] not in (2, 6)
            or len(values) < 1 or len(values) > 1024
            or values.dtype != np.float64
            or labels.shape != (len(values),)
            or not np.issubdtype(labels.dtype, np.integer)
            or not np.isfinite(values).all() or np.any(values < 0)
            or not np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-10)):
        raise ValueError("bank table shape, dtype or simplex differs")
    keys: set[tuple[int, bytes]] = set()
    for row, decision in zip(values, labels):
        k = int(decision)
        if k < 0 or k >= len(row) or int(np.argmax(row)) != k:
            raise ValueError("bank class differs from probability decision")
        if sum(float(v) == float(row[k]) for v in row) != 1:
            raise ValueError("bank probability has tied decision")
        key = (k, row.astype("<f8", copy=False).tobytes())
        if key in keys:
            raise ValueError("duplicate global probability/decision state")
        keys.add(key)
    return {"states": len(keys), "bits_required": (len(keys) - 1).bit_length()}


def aggregate_pair_law(actions: np.ndarray, mass: np.ndarray,
                       state_counts: tuple[int, int]) -> list[dict[tuple[int, int], float]]:
    codes = np.asarray(actions)
    weights = np.asarray(mass)
    if (codes.ndim != 3 or codes.shape[2] != 2
            or not np.issubdtype(codes.dtype, np.integer)
            or weights.shape != codes.shape[:2]
            or not np.isfinite(weights).all() or np.any(weights < 0)
            or np.any(codes < 0)
            or np.any(codes[:, :, 0] >= state_counts[0])
            or np.any(codes[:, :, 1] >= state_counts[1])):
        raise ValueError("action pair law invalid")
    result = []
    for row, q in zip(codes, weights):
        if not math.isclose(float(sum(float(x) for x in q)), 1.0, rel_tol=0,
                            abs_tol=1e-10):
            raise ValueError("action probabilities do not normalize")
        merged: dict[tuple[int, int], float] = {}
        for pair, probability in zip(row, q):
            if probability:
                key = int(pair[0]), int(pair[1])
                merged[key] = merged.get(key, 0.0) + float(probability)
        result.append(merged)
    return result


def expected_costs(law: list[dict[tuple[int, int], float]],
                   tables: tuple[np.ndarray, np.ndarray],
                   labels: np.ndarray) -> dict[str, dict[str, float]]:
    """Mean fixed-output losses over exact per-person public pair laws."""
    truth = np.asarray(labels)
    if (truth.shape != (len(law), 2)
            or not np.issubdtype(truth.dtype, np.integer)
            or len(law) == 0):
        raise ValueError("task label dimensions differ")
    arrays = tuple(np.asarray(table) for table in tables)
    for table in arrays:
        if (table.ndim != 2 or table.shape[1] not in (2, 6)
                or not np.isfinite(table).all()):
            raise ValueError("fixed bank table differs")
    sums = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for i, row in enumerate(law):
        if not math.isclose(sum(row.values()), 1.0, rel_tol=0, abs_tol=1e-10):
            raise ValueError("public pair law does not normalize")
        for purpose, table in enumerate(arrays):
            y = int(truth[i, purpose])
            if y < 0 or y >= table.shape[1]:
                raise ValueError("task class outside bank table")
            for pair, probability in row.items():
                code = int(pair[purpose])
                if code < 0 or code >= len(table) or probability < 0:
                    raise ValueError("public code or mass outside bank law")
                vector = table[code]
                sums[purpose][0] += probability * (int(np.argmax(vector)) == y)
                sums[purpose][1] += probability * -math.log(max(float(vector[y]), 1e-12))
                sums[purpose][2] += probability * sum(
                    (float(value) - int(k == y)) ** 2
                    for k, value in enumerate(vector)
                )
    return {
        name: {"accuracy": sums[p][0] / len(law),
               "nll": sums[p][1] / len(law),
               "brier": sums[p][2] / len(law)}
        for p, name in enumerate(("income", "occupation"))
    }
