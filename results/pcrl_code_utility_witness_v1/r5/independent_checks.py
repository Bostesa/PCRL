"""Independent arithmetic for the code-utility-witness verification boundary.

This module imports no candidate, deployment, runner endpoint, or selection code.
It accepts already admitted, ordered arrays. Callers must check source hashes,
role identity, row/group alignment and stage seals before supplying those arrays.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import ceil, log2

import numpy as np


NLL_FLOOR = 1e-12
ACCURACY_CAP = -0.010  # candidate accuracy minus reference accuracy
NLL_CAP = 0.010       # candidate loss minus reference loss
BRIER_CAP = 0.005


def checked_probabilities(probabilities: np.ndarray, classes: int) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    if (p.ndim != 2 or p.shape[1] != classes or len(p) == 0
            or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
            or np.max(np.abs(p.sum(axis=1) - 1.0)) > 1e-8):
        raise ValueError("invalid public probability matrix")
    return p


def task_rows(
    labels: np.ndarray, probabilities: np.ndarray, classes: int,
    decisions: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return one value per person; class ties follow NumPy's first argmax."""
    p = checked_probabilities(probabilities, classes)
    y = np.asarray(labels)
    if (y.shape != (len(p),) or not np.issubdtype(y.dtype, np.integer)
            or np.any((y < 0) | (y >= classes))):
        raise ValueError("invalid task labels")
    prediction = np.argmax(p, axis=1)
    if decisions is not None and not np.array_equal(np.asarray(decisions), prediction):
        raise ValueError("transmitted decision disagrees with public probabilities")
    chosen = p[np.arange(len(p)), y]
    return {
        "accuracy": (prediction == y).astype(np.float64),
        "nll": -np.log(np.maximum(chosen, NLL_FLOOR)),
        "brier": np.sum(p * p, axis=1) + 1.0 - 2.0 * chosen,
    }


def task_means(rows: Mapping[str, np.ndarray], weight: np.ndarray | None = None) -> dict[str, float]:
    values = {name: np.asarray(rows[name], dtype=np.float64)
              for name in ("accuracy", "nll", "brier")}
    length = len(values["accuracy"])
    if any(value.shape != (length,) for value in values.values()):
        raise ValueError("task row lengths differ")
    w = np.ones(length, dtype=np.float64) if weight is None else np.asarray(weight, dtype=np.float64)
    if (w.shape != (length,) or not np.isfinite(w).all() or np.any(w < 0)
            or w.sum() <= 0):
        raise ValueError("invalid nonnegative row weights")
    return {name: float(np.dot(value, w) / w.sum()) for name, value in values.items()}


def utility_differences(candidate: Mapping[str, float], reference: Mapping[str, float]) -> dict[str, float]:
    return {key: float(candidate[key] - reference[key])
            for key in ("accuracy", "nll", "brier")}


def point_utility_pass(differences: Mapping[str, float]) -> bool:
    return (differences["accuracy"] >= ACCURACY_CAP
            and differences["nll"] <= NLL_CAP
            and differences["brier"] <= BRIER_CAP)


def select_capacity(rows: list[Mapping]) -> dict[str, object]:
    """Replay the sealed 10-then-16-bit point ladder without looking at AUC."""
    expected = {(seed, task, reference) for seed in (0, 1)
                for task in ("income", "occupation") for reference in ("U0", "Ucal")}
    groups: dict[int, dict[tuple[int, str, str], dict[str, float]]] = {10: {}, 16: {}}
    for row in rows:
        bits = row["bits"]
        key = (row["seed"], row["task"], row["reference"])
        if bits not in groups or key not in expected or key in groups[bits]:
            raise ValueError("unregistered or duplicate capacity/seed/task/reference row")
        values = {metric: float(row[metric]) for metric in ("accuracy", "nll", "brier")}
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("nonfinite utility difference")
        groups[bits][key] = values
    if set(groups[10]) != expected or (groups[16] and set(groups[16]) != expected):
        raise ValueError("incomplete capacity utility grid")
    ten_failed = sorted(key for key in expected if not point_utility_pass(groups[10][key]))
    if not ten_failed:
        if groups[16]:
            raise ValueError("sixteen-bit evaluation followed a passing ten-bit gate")
        return {"status": "TEN_BIT_POINT_FEASIBLE", "selected_bits": 10,
                "ten_failed_cells": [], "sixteen_failed_cells": None}
    if not groups[16]:
        return {"status": "SIXTEEN_BIT_STAGE_REQUIRED", "selected_bits": None,
                "ten_failed_cells": ten_failed, "sixteen_failed_cells": None}
    sixteen_failed = sorted(key for key in expected if not point_utility_pass(groups[16][key]))
    return {"status": ("LEARNER_RECIPE_NOT_UTILITY_FEASIBLE" if sixteen_failed
                       else "SIXTEEN_BIT_POINT_FEASIBLE"),
            "selected_bits": None if sixteen_failed else 16,
            "ten_failed_cells": ten_failed, "sixteen_failed_cells": sixteen_failed}


def minimal_public_bits(public_probabilities: np.ndarray, decisions: np.ndarray,
                        independent_tags: np.ndarray | None = None,
                        public_code_ids: np.ndarray | None = None) -> dict[str, int]:
    """Count globally distinct transmitted states, including any extra tag.

    The caller must supply the entire public decoder table. Include code IDs if
    they are transmitted: duplicate probabilities with different IDs then count
    as different wire states. A newly compiled canonical bank can instead pass
    its canonical IDs. Probabilities use exact float64 equality, never a
    post-hoc tolerance merge.
    """
    p = np.asarray(public_probabilities, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("public decoder table must be rank two")
    p = checked_probabilities(p, p.shape[1])
    d = np.asarray(decisions)
    if (d.shape != (len(p),) or not np.issubdtype(d.dtype, np.integer)
            or not np.array_equal(d, np.argmax(p, axis=1))):
        raise ValueError("public decisions invalid")
    columns = [p, d[:, None].astype(np.float64)]
    if independent_tags is not None:
        tags = np.asarray(independent_tags)
        if tags.shape != (len(p),) or not np.issubdtype(tags.dtype, np.integer):
            raise ValueError("public tag invalid")
        columns.append(tags[:, None].astype(np.float64))
    if public_code_ids is not None:
        codes = np.asarray(public_code_ids)
        if (codes.shape != (len(p),) or not np.issubdtype(codes.dtype, np.integer)
                or np.any(codes < 0) or np.any(codes >= 2**53)):
            raise ValueError("public code IDs invalid")
        columns.append(codes[:, None].astype(np.float64))
    states = int(np.unique(np.concatenate(columns, axis=1), axis=0).shape[0])
    return {"distinct_public_states": states,
            "minimum_bits": 0 if states == 1 else ceil(log2(states))}


def verify_hard_code_wire(bits: np.ndarray, probabilities: np.ndarray,
                          decisions: np.ndarray, public_table: np.ndarray,
                          bit_order: str = "little", tolerance: float = 1e-8) -> dict[str, float | int]:
    """Reconstruct outputs from the public bits and fixed table alone."""
    code = np.asarray(bits)
    if (code.ndim != 2 or not np.issubdtype(code.dtype, np.integer)
            or code.shape[1] not in (10, 16) or not np.isin(code, (0, 1)).all()
            or bit_order not in ("little", "big")):
        raise ValueError("invalid ten- or sixteen-bit hard public code")
    p = checked_probabilities(probabilities, np.asarray(public_table).shape[-1])
    table = checked_probabilities(public_table, p.shape[1])
    if table.shape[0] != 1 << code.shape[1] or len(p) != len(code):
        raise ValueError("public decoder table/code length mismatch")
    powers = (1 << np.arange(code.shape[1], dtype=np.int64))
    if bit_order == "big":
        powers = powers[::-1]
    ids = code.astype(np.int64) @ powers
    discrepancy = float(np.max(np.abs(p - table[ids])))
    if discrepancy > tolerance:
        raise ValueError("public task output is not code-derived")
    if not np.array_equal(np.asarray(decisions), np.argmax(p, axis=1)):
        raise ValueError("decision is not code-derived")
    return {"rows": len(code), "bits": code.shape[1],
            "occupied_states": int(np.unique(ids).size),
            "max_probability_discrepancy": discrepancy}


def oriented_auc(scores: np.ndarray, sex: np.ndarray,
                 weight: np.ndarray | None = None) -> float:
    """Weighted tie-aware AUC, oriented only after the raw AUC is computed."""
    score = np.asarray(scores, dtype=np.float64)
    y = np.asarray(sex)
    if (score.ndim != 1 or y.shape != score.shape or not np.isfinite(score).all()
            or not np.isin(y, (0, 1)).all()):
        raise ValueError("invalid attacker score or SEX label")
    w = np.ones(len(y), dtype=np.float64) if weight is None else np.asarray(weight, dtype=np.float64)
    if w.shape != y.shape or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("invalid attacker weights")
    order = np.argsort(score, kind="stable")
    sorted_score = score[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_score) != 0) + 1]
    positive = np.add.reduceat(w[order] * y[order], starts)
    negative = np.add.reduceat(w[order] * (1 - y[order]), starts)
    if positive.sum() <= 0 or negative.sum() <= 0:
        raise ValueError("attacker weights omit a SEX class")
    below = np.cumsum(negative) - negative
    raw = float(np.dot(positive, below + 0.5 * negative) /
                (positive.sum() * negative.sum()))
    return max(raw, 1.0 - raw)


def simultaneous_group_bounds(group_id: np.ndarray, statistic: Callable[[np.ndarray], np.ndarray],
                              clause_count: int, seed: int, replicates: int,
                              full_family_count: int | None = None) -> dict[str, np.ndarray | float]:
    """Paired-group bounds; optional full-K t critical retains missing slots.

    The caller fixes clause order, group IDs, RNG, replicate count and family
    before reading results. Model seeds are scored in a single statistic call,
    so each draw resamples the same people across all seeds and arms. When
    `full_family_count` is set, it counts all prospective clauses, including
    dependency-missing ones, and uses a two-sided Bonferroni Student-t critical
    with number-of-groups-minus-one degrees of freedom. The default max-|t|
    branch is only suitable for a complete, fixed observed family.
    """
    groups = np.asarray(group_id)
    if groups.ndim != 1 or len(groups) == 0 or replicates < 2 or clause_count < 1:
        raise ValueError("invalid interval family")
    _, inverse = np.unique(groups, return_inverse=True)
    group_count = int(inverse.max()) + 1
    point = np.asarray(statistic(np.ones(len(groups), dtype=np.float64)), dtype=np.float64)
    if point.shape != (clause_count,) or not np.isfinite(point).all():
        raise ValueError("invalid fixed-family point vector")
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    draws = np.empty((replicates, clause_count), dtype=np.float64)
    for index in range(replicates):
        count = np.bincount(rng.integers(group_count, size=group_count), minlength=group_count)
        draws[index] = statistic(count[inverse].astype(np.float64))
    if not np.isfinite(draws).all():
        raise ValueError("invalid bootstrap vector")
    standard_error = draws.std(axis=0, ddof=1)
    constant = np.max(np.abs(draws - point), axis=0) <= 1e-12
    standard_error[constant] = 0.0
    scale = np.maximum(standard_error, 1e-12)
    standardized = np.abs((draws - point) / scale)
    standardized[:, constant] = 0.0
    if full_family_count is None:
        critical = float(np.quantile(standardized.max(axis=1), .95, method="higher"))
    else:
        if full_family_count < clause_count:
            raise ValueError("full family cannot be smaller than estimable subset")
        from scipy.stats import t as student_t
        critical = float(student_t.ppf(1 - .05 / (2 * full_family_count), group_count - 1))
    half = critical * np.where(constant, 0.0, scale)
    return {"point": point, "standard_error": standard_error,
            "lower": point - half, "upper": point + half,
            "numerically_constant": constant, "critical_95": critical}


def sampled_expected_losses(action_probability: np.ndarray, output_bank: np.ndarray,
                            labels: np.ndarray) -> dict[str, np.ndarray]:
    """Exact fixed-bank expected NLL/Brier for the ACTUAL sampled service."""
    q = np.asarray(action_probability, dtype=np.float64)
    bank = np.asarray(output_bank, dtype=np.float64)
    if (q.ndim != 2 or bank.ndim != 2 or q.shape[1] != len(bank)
            or np.any(q < 0) or not np.isfinite(q).all()
            or np.max(np.abs(q.sum(axis=1) - 1)) > 1e-8):
        raise ValueError("invalid channel probability law")
    bank = checked_probabilities(bank, bank.shape[1])
    y = np.asarray(labels)
    if (y.shape != (len(q),) or not np.issubdtype(y.dtype, np.integer)
            or np.any((y < 0) | (y >= bank.shape[1]))):
        raise ValueError("invalid channel labels")
    chosen = bank[:, y].T
    nll = -np.log(np.maximum(chosen, NLL_FLOOR))
    brier = np.sum(bank * bank, axis=1)[None, :] + 1 - 2 * chosen
    return {"nll": np.sum(q * nll, axis=1),
            "brier": np.sum(q * brier, axis=1)}
