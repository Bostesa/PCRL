"""Independent finite-reader AUC and fixed-family inference arithmetic.

This module contains no imports from the defense, attack, utility, or
selection runners.  The release and reader custody checks belong to the
queued R5 workers that call it.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import t as student_t


ARMS = ("P0", "O_J", "V_J", "V_P", "V_D", "V_L", "V_S12", "V_S21")
VIEWS = ("r1", "r2", "pair")
SUMMARIES = ("s0_d0", "s0_d1", "s0_d2", "s1_d0", "s1_d1", "s1_d2",
             "s0_mean", "s1_mean", "mean")
STOCHASTIC = frozenset(("O_J", "V_J", "V_P", "V_L", "V_S12", "V_S21"))
K = 948
BOOTSTRAPS = 2048
GROUPS = 2234


def paired_plan(unit: np.ndarray, *, replicates: int = BOOTSTRAPS,
                expected_groups: int = GROUPS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One duplicate-group resample and one coupled draw plan for all arms."""
    unit = np.asarray(unit)
    if unit.dtype != np.int64 or unit.ndim != 1:
        raise ValueError("INNER duplicate-group IDs must be one int64 vector")
    unique, inverse = np.unique(unit, return_inverse=True)
    if len(unique) != expected_groups:
        raise ValueError("INNER duplicate-group count differs")
    generator = np.random.Generator(np.random.PCG64DXSM(20261010))
    group_samples = generator.integers(0, expected_groups,
                                       size=(replicates, expected_groups), dtype=np.int64)
    weights = np.empty((replicates + 1, expected_groups), dtype=np.int64)
    weights[0] = 1
    for index, sample in enumerate(group_samples, start=1):
        weights[index] = np.bincount(sample, minlength=expected_groups)
    # Generate these after the whole group-resampling batch, preserving the
    # prospectively fixed source stream and coupling each seed across arms.
    draws = generator.integers(0, 3, size=(replicates, 2, 3), dtype=np.int64)
    return inverse.astype(np.int64, copy=False), weights, draws


def auc_by_group_weights(y: np.ndarray, score: np.ndarray, group_index: np.ndarray,
                         group_weights: np.ndarray) -> np.ndarray:
    """Weighted Mann-Whitney AUC with half credit for exact score ties."""
    truth = np.asarray(y)
    score = np.asarray(score)
    group = np.asarray(group_index)
    weights = np.asarray(group_weights)
    if (truth.dtype != np.int64 or truth.ndim != 1 or
            score.dtype != np.float64 or score.shape != truth.shape or
            group.dtype != np.int64 or group.shape != truth.shape or
            weights.ndim != 2 or weights.shape[1] <= int(group.max(initial=-1)) or
            not np.isin(truth, (0, 1)).all() or not np.isfinite(score).all() or
            np.any(weights < 0) or np.any(group < 0)):
        raise ValueError("reader AUC input schema differs")
    order = np.argsort(score, kind="stable")
    ordered = score[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(ordered) != 0) + 1]
    person_weights = weights[:, group[order]].astype(np.float64, copy=False)
    pos = np.add.reduceat(person_weights * truth[order], boundaries, axis=1)
    neg = np.add.reduceat(person_weights * (1 - truth[order]), boundaries, axis=1)
    neg_below = np.cumsum(neg, axis=1) - neg
    numerator = np.sum(pos * (neg_below + neg / 2), axis=1)
    denominator = np.sum(pos, axis=1) * np.sum(neg, axis=1)
    if np.any(denominator == 0):
        raise ValueError("bootstrap has no positive or negative SEX row")
    return numerator / denominator


def reader_envelope(y: np.ndarray, predictions: dict[str, np.ndarray],
                    frozen_order: tuple[str, ...], group_index: np.ndarray,
                    group_weights: np.ndarray) -> np.ndarray:
    """Defense-aware envelope, including each reader's reversed orientation."""
    if len(frozen_order) == 0 or set(frozen_order) != set(predictions):
        raise ValueError("frozen reader inventory differs")
    envelope = np.full(len(group_weights), 0.5, dtype=np.float64)
    for name in frozen_order:
        auc = auc_by_group_weights(y, np.asarray(predictions[name]), group_index, group_weights)
        np.maximum(envelope, np.maximum(auc, 1 - auc), out=envelope)
    return envelope


def summarize_release_vectors(vectors: dict[tuple[str, int, int, str], np.ndarray],
                              eligible: set[str], draw_plan: np.ndarray,
                              *, replicates: int = BOOTSTRAPS) -> dict[tuple[str, str, str], np.ndarray]:
    """Summaries use cached whole-release draws, never per-row draw mixing."""
    if not eligible.issubset(ARMS) or draw_plan.shape != (replicates, 2, 3):
        raise ValueError("fixed arm or coupled draw family differs")
    out = {}
    for arm in eligible:
        for seed in (0, 1):
            for view in VIEWS:
                draws = []
                for draw in range(3):
                    key = (arm, seed, draw if arm in STOCHASTIC else 0, view)
                    vector = np.asarray(vectors[key])
                    if vector.dtype != np.float64 or vector.shape != (replicates + 1,):
                        raise ValueError("reader envelope vector length differs")
                    out[(f"s{seed}_d{draw}", arm, view)] = vector
                    draws.append(vector)
                cube = np.stack(draws, axis=1)
                sampled = np.take_along_axis(cube[1:], draw_plan[:, seed, :], axis=1)
                out[(f"s{seed}_mean", arm, view)] = np.r_[cube[0].mean(), sampled.mean(axis=1)]
        for view in VIEWS:
            out[("mean", arm, view)] = (out[("s0_mean", arm, view)] +
                                        out[("s1_mean", arm, view)]) / 2
    return out


def fixed_recovery_family(summaries: dict[tuple[str, str, str], np.ndarray],
                          eligible: set[str], *, replicates: int = BOOTSTRAPS,
                          groups: int = GROUPS) -> dict[str, dict]:
    """All 756 signed pair slots; absent arms keep explicit conditional nulls."""
    group_critical = float(student_t.ppf(1 - 0.05 / (2 * K), groups - 1))
    release_critical = float(student_t.ppf(1 - 0.05 / (2 * K), 2))
    slots: dict[str, dict] = {}
    for summary in SUMMARIES:
        fixed_draw = summary[2:4] == "_d"
        critical = group_critical if fixed_draw else release_critical
        for left, right in combinations(ARMS, 2):
            for view in VIEWS:
                key = f"{summary}:recovery:{left}:{right}:{view}:auc"
                if left not in eligible or right not in eligible:
                    slots[key] = {"point": None, "se": None, "lower": None,
                                  "upper": None, "status": "NOT_ESTIMABLE_DEPENDENCY"}
                    continue
                difference = summaries[(summary, left, view)] - summaries[(summary, right, view)]
                point = float(difference[0])
                draws = difference[1:]
                se = 0.0 if np.all(draws == point) else float(np.std(draws, ddof=1))
                slots[key] = {"point": point, "se": se,
                              "lower": point - critical * se,
                              "upper": point + critical * se,
                              "df": groups - 1 if fixed_draw else 2,
                              "K": K, "status": "BOUND_ONLY"}
    if len(slots) != 756:
        raise AssertionError("recovery family cardinality differs")
    return slots


def oriented_bounds(slots: dict[str, dict], candidate: str, control: str,
                    view: str, summary: str = "mean") -> tuple[float, float, float]:
    if candidate == control or candidate not in ARMS or control not in ARMS:
        raise ValueError("candidate/control identity differs")
    first, second = sorted((candidate, control), key=ARMS.index)
    record = slots[f"{summary}:recovery:{first}:{second}:{view}:auc"]
    if record["point"] is None:
        raise ValueError("required recovery comparison is not estimable")
    if control == first:
        return record["point"], record["lower"], record["upper"]
    return -record["point"], -record["upper"], -record["lower"]
