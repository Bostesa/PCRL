"""Independent fixed-family D interval arithmetic on admitted ordered arrays.

No candidate, R02, or inherited scorer is imported. The caller verifies every
input file, role identity and reader receipt before constructing records.
"""

from __future__ import annotations

import hashlib
from itertools import combinations
from typing import Mapping

import numpy as np

from .independent_replay import d_finite_family_ids


ARMS = ("D_T", "D_K", "D_L", "D_S12", "D_S21", "D_C", "D_F")
ANCHORS = ("U0", "Ucal")
VIEWS = ("r1", "r2", "pair")
TASKS = (("income", "prob_1", 2), ("occupation", "prob_2", 6))
METRICS = ("accuracy", "nll", "brier")
REPLICATES = 2048
GROUPS = 2234
FLOOR = 1e-12
SEED_BYTES = (20261010).to_bytes(16, "big") + hashlib.sha256(
    b"pcrl-latent-sampled-inference-v1").digest()[:16]


class _Reader:
    def __init__(self, scores: np.ndarray, sex: np.ndarray):
        score = np.asarray(scores, dtype=np.float64)
        if score.shape != sex.shape or not np.isfinite(score).all():
            raise ValueError("fixed reader score shape/value invalid")
        self.order = np.argsort(score, kind="stable")
        ordered = score[self.order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(ordered) != 0) + 1]
        self.sex = sex[self.order]

    def oriented_auc(self, weight: np.ndarray) -> float:
        ordered_weight = weight[self.order]
        positive = np.add.reduceat(ordered_weight * self.sex, self.starts)
        negative = np.add.reduceat(ordered_weight * (1 - self.sex), self.starts)
        if positive.sum() <= 0 or negative.sum() <= 0:
            raise ValueError("bootstrap omitted one SEX class")
        below = np.cumsum(negative) - negative
        auc = float(np.dot(positive, below + 0.5 * negative) /
                    (positive.sum() * negative.sum()))
        return max(auc, 1 - auc)


class _Release:
    def __init__(self, raw: Mapping, truth: Mapping, sex: np.ndarray):
        n = len(sex)
        if (not np.array_equal(raw["row_id"], truth["row_id"])
                or not np.array_equal(raw["unit"], truth["unit"])):
            raise ValueError("release ordered person/group identity differs")
        self.loss: dict[str, dict[str, np.ndarray]] = {}
        for task, field, width in TASKS:
            p = np.asarray(raw[field], dtype=np.float64)
            y = np.asarray(truth[task], dtype=np.int64)
            if (p.shape != (n, width) or y.shape != (n,)
                    or not np.isfinite(p).all() or np.any(p < 0)
                    or np.any(p > 1) or np.max(np.abs(p.sum(axis=1) - 1)) > 1e-8
                    or np.any(y < 0) or np.any(y >= width)):
                raise ValueError("task probability/label contract differs")
            chosen = p[np.arange(n), y]
            self.loss[task] = {
                "accuracy": (np.argmax(p, axis=1) == y).astype(np.float64),
                "nll": -np.log(np.maximum(chosen, 1e-12)),
                "brier": np.sum(p * p, axis=1) + 1 - 2 * chosen,
            }
        bank = raw["reader_predictions"]
        if set(bank) != set(VIEWS):
            raise ValueError("all complete-view reader banks required")
        self.readers = {}
        for view in VIEWS:
            if not bank[view] or len(bank[view]) != len(set(bank[view])):
                raise ValueError("fixed reader bank empty or repeated")
            self.readers[view] = [_Reader(value, sex) for value in bank[view].values()]

    def score(self, weight: np.ndarray, denominator: float) -> dict:
        task = {name: {metric: float(np.dot(weight, per_row) / denominator)
                       for metric, per_row in fields.items()}
                for name, fields in self.loss.items()}
        auc = {view: max(reader.oriented_auc(weight) for reader in self.readers[view])
               for view in VIEWS}
        return {"task": task, "auc": auc}


def _prepare(seed_records: Mapping[int, Mapping]) -> tuple[np.ndarray, dict]:
    if set(seed_records) != {0, 1}:
        raise ValueError("both registered model seeds required")
    first = seed_records[0]["truth"]
    row_id = np.asarray(first["row_id"], dtype=np.int64)
    unit = np.asarray(first["unit"], dtype=np.int64)
    if (row_id.ndim != 1 or unit.shape != row_id.shape
            or len(row_id) == 0 or len(np.unique(row_id)) != len(row_id)
            or len(np.unique(unit)) != GROUPS):
        raise ValueError("INNER ordered row/group count differs")
    prepared = {}
    for seed in (0, 1):
        record = seed_records[seed]
        truth = record["truth"]
        if (not np.array_equal(row_id, truth["row_id"])
                or not np.array_equal(unit, truth["unit"])
                or any(not np.array_equal(first[name], truth[name])
                       for name in ("income", "occupation", "sex"))):
            raise ValueError("paired seed identities or truth differ")
        sex = np.asarray(truth["sex"], dtype=np.int64)
        if sex.shape != row_id.shape or not np.isin(sex, (0, 1)).all():
            raise ValueError("ordered binary SEX invalid")
        releases = record["releases"]
        if set(releases) != set(ARMS).union(ANCHORS):
            raise ValueError("nine-release D registry incomplete")
        prepared[seed] = {name: _Release(releases[name], truth, sex)
                          for name in (*ARMS, *ANCHORS)}
    return unit, prepared


def _margins(prepared: Mapping[int, Mapping], weight: np.ndarray) -> np.ndarray:
    denominator = float(weight.sum())
    if denominator <= 0:
        raise ValueError("bootstrap weight sum zero")
    by_seed = []
    for seed in (0, 1):
        scored = {name: release.score(weight, denominator)
                  for name, release in prepared[seed].items()}
        values = []
        for arm in ARMS:
            for task, _, _ in TASKS:
                for ref in ANCHORS:
                    for metric in METRICS:
                        values.append(scored[arm]["task"][task][metric] -
                                      scored[ref]["task"][task][metric])
        for first, second in combinations(ARMS, 2):
            for view in VIEWS:
                values.append(scored[first]["auc"][view] -
                              scored[second]["auc"][view])
        for arm in ARMS:
            for ref in ANCHORS:
                for view in VIEWS:
                    values.append(scored[arm]["auc"][view] -
                                  scored[ref]["auc"][view])
        one = np.asarray(values, dtype=np.float64)
        if one.shape != (189,) or not np.isfinite(one).all():
            raise ValueError("one-seed D interval vector invalid")
        by_seed.append(one)
    return np.concatenate((by_seed[0], by_seed[1], (by_seed[0] + by_seed[1]) / 2))


def fixed_d_intervals(seed_records: Mapping[int, Mapping]) -> dict:
    """Run the prospectively fixed 567-clause paired max-|t| procedure."""
    names = d_finite_family_ids()
    if len(names) != 567:
        raise AssertionError("D family size differs")
    unit, prepared = _prepare(seed_records)
    ordered, inverse = np.unique(unit, return_inverse=True)
    if len(ordered) != GROUPS:
        raise AssertionError("INNER group count changed")
    point = _margins(prepared, np.ones(len(unit), dtype=np.float64))
    rng = np.random.Generator(np.random.PCG64DXSM(20261010))
    draws = np.empty((REPLICATES, len(names)), dtype=np.float64)
    for b in range(REPLICATES):
        sampled = rng.choice(ordered, size=len(ordered), replace=True)
        counts = np.bincount(np.searchsorted(ordered, sampled), minlength=GROUPS)
        draws[b] = _margins(prepared, counts[inverse].astype(np.float64))
    se = draws.std(axis=0, ddof=1)
    zero = np.max(np.abs(draws - point), axis=0) <= FLOOR
    se[zero] = 0.0
    scale = np.maximum(se, FLOOR)
    student = np.zeros_like(draws)
    student[:, ~zero] = np.abs((draws[:, ~zero] - point[~zero]) / scale[~zero])
    critical = float(np.quantile(student.max(axis=1), .95, method="higher"))
    half = critical * np.where(zero, 0.0, scale)
    clauses = [{"name": name, "point": float(point[j]), "se": float(se[j]),
                "lower": float(point[j] - half[j]), "upper": float(point[j] + half[j]),
                "numerically_constant": bool(zero[j])}
               for j, name in enumerate(names)]
    return {"schema": "pcrl-r04-independent-d-interval-v1",
            "status": "FIXED_567_INTERVALS_COMPUTED",
            "clause_count": len(clauses), "bootstrap_replicates": REPLICATES,
            "inner_groups": GROUPS, "critical_95": critical,
            "seed_bytes_sha256": hashlib.sha256(SEED_BYTES).hexdigest(),
            "numpy_version": np.__version__, "clauses": clauses}
