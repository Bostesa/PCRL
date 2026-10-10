"""Independent paired-group and three-draw A finite-family intervals.

The verifier receives hash-checked ordered INNER arrays and frozen reader
predictions. It imports no A runner, R02 scorer or inherited evaluator.
"""

from __future__ import annotations

from itertools import combinations
import hashlib
from typing import Mapping

import numpy as np
from scipy.stats import t as student_t

from .audit_a_family_s1 import ARMS, REFERENCES, SUMMARIES, VIEWS, independent_ids
from .independent_d_intervals import _Release


STOCHASTIC = frozenset(("A_JOINT_STOCH", "A_TASK_STOCH", "A_CENTRAL_PRODUCT",
                        "A_LOCAL", "A_SEQ12", "A_SEQ21"))
NAMES = (*ARMS, *REFERENCES)
TASKS = ("income", "occupation")
METRICS = ("accuracy", "nll", "brier")
GROUPS = 2234
REPLICATES = 2048
K = 6885
FLOOR = 1e-12
SEED_GROUP_BYTES = (20261010).to_bytes(16, "big") + hashlib.sha256(
    b"pcrl-latent-sampled-inference-v1").digest()[:16]
SEED_DRAW_BYTES = (20261011).to_bytes(16, "big") + hashlib.sha256(
    b"pcrl-a-release-inference-v1").digest()[:16]


def _prepare(records: Mapping[int, Mapping]) -> tuple[np.ndarray, dict]:
    if set(records) != {0, 1}:
        raise ValueError("A inference needs both paired seeds")
    first = records[0]["truth"]
    row = np.asarray(first["row_id"], dtype=np.int64)
    unit = np.asarray(first["unit"], dtype=np.int64)
    if (row.ndim != 1 or unit.shape != row.shape or len(np.unique(unit)) != GROUPS
            or not np.all(np.diff(row) > 0)):
        raise ValueError("A INNER row/group identity differs")
    prepared = {}
    for seed in (0, 1):
        truth = records[seed]["truth"]
        if any(not np.array_equal(first[name], truth[name]) for name in
               ("row_id", "unit", "income", "occupation", "sex")):
            raise ValueError("A paired seed people or truth differ")
        sex = np.asarray(truth["sex"], dtype=np.int64)
        if sex.shape != row.shape or not np.isin(sex, (0, 1)).all():
            raise ValueError("A binary INNER SEX truth invalid")
        releases = records[seed]["releases"]
        if not set(releases) <= set(NAMES):
            raise ValueError("unregistered A release in interval records")
        prepared[seed] = {}
        for name, draws in releases.items():
            wanted = {0, 1, 2} if name in STOCHASTIC else {0}
            if set(draws) != wanted:
                raise ValueError("A release draw set differs from arm contract")
            prepared[seed][name] = {draw: _Release(raw, truth, sex)
                                     for draw, raw in draws.items()}
    return unit, prepared


def _base(scored: Mapping[str, dict | None]) -> np.ndarray:
    values = []
    for arm in ARMS:
        for task in TASKS:
            for ref in REFERENCES:
                for metric in METRICS:
                    a, b = scored.get(arm), scored.get(ref)
                    values.append(np.nan if a is None or b is None else
                                  a["task"][task][metric] - b["task"][task][metric])
    for left, right in combinations(ARMS, 2):
        for view in VIEWS:
            a, b = scored.get(left), scored.get(right)
            values.append(np.nan if a is None or b is None else
                          a["auc"][view] - b["auc"][view])
    for arm in ARMS:
        for ref in REFERENCES:
            for view in VIEWS:
                a, b = scored.get(arm), scored.get(ref)
                values.append(np.nan if a is None or b is None else
                              a["auc"][view] - b["auc"][view])
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (765,) or np.any(np.isinf(result)):
        raise ValueError("A base contrast vector invalid")
    return result


def _compose(base: Mapping[tuple[int, int], np.ndarray],
             draws: Mapping[int, tuple[int, int, int]]) -> np.ndarray:
    if (set(base) != {(s, d) for s in (0, 1) for d in (0, 1, 2)}
            or set(draws) != {0, 1}
            or any(len(draws[s]) != 3 or any(d not in (0, 1, 2) for d in draws[s])
                   for s in (0, 1))):
        raise ValueError("A six fixed draws or coupled mean indices invalid")
    mean0 = np.mean([base[0, d] for d in draws[0]], axis=0)
    mean1 = np.mean([base[1, d] for d in draws[1]], axis=0)
    result = np.concatenate([base[s, d] for s in (0, 1) for d in (0, 1, 2)] +
                            [mean0, mean1, (mean0 + mean1) / 2])
    if result.shape != (K,):
        raise ValueError("A nine-summary family width invalid")
    return result


def _margins(prepared: Mapping[int, Mapping], weight: np.ndarray,
             draws: Mapping[int, tuple[int, int, int]]) -> np.ndarray:
    denominator = float(weight.sum())
    if denominator <= 0:
        raise ValueError("A bootstrap selected zero people")
    scored = {(seed, name, draw): release.score(weight, denominator)
              for seed in (0, 1)
              for name, draw_map in prepared[seed].items()
              for draw, release in draw_map.items()}
    base = {}
    for seed in (0, 1):
        for draw in (0, 1, 2):
            view = {name: scored.get((seed, name, draw if name in STOCHASTIC else 0))
                    for name in NAMES}
            base[seed, draw] = _base(view)
    return _compose(base, draws)


def _bounds(names: tuple[str, ...], point: np.ndarray,
            boot: np.ndarray) -> tuple[list[dict], str, float]:
    available = np.isfinite(point)
    if (len(names) != K or point.shape != (K,) or not np.any(available)
            or np.any(np.isinf(point))
            or boot.shape != (REPLICATES, int(available.sum()))
            or not np.isfinite(boot).all()):
        raise ValueError("A fixed-family point/bootstrap availability differs")
    selected = point[available]
    se = boot.std(axis=0, ddof=1)
    constant = np.max(np.abs(boot - selected), axis=0) <= FLOOR
    se[constant] = 0.0
    scale = np.maximum(se, FLOOR)
    if available.all():
        t = np.zeros_like(boot)
        t[:, ~constant] = np.abs((boot[:, ~constant] - selected[~constant]) /
                                  scale[~constant])
        critical = float(np.quantile(np.max(t, axis=1), .95, method="higher"))
        critical_by_clause = np.full(len(selected), critical)
        family = "two-sided-maximum-absolute-studentized-full-K"
    else:
        quantile = 1 - .05 / (2 * K)
        group = float(student_t.ppf(quantile, GROUPS - 1))
        release = float(student_t.ppf(quantile, 2))
        critical_by_clause = np.asarray([
            group if name.startswith(("s0_d", "s1_d")) else release
            for name, present in zip(names, available) if present])
        critical = float(critical_by_clause.max())
        family = "two-sided-fixed-K-Bonferroni-t-approximation"
    half = critical_by_clause * np.where(constant, 0.0, scale)
    finite = iter(zip(selected, se, selected - half, selected + half, constant))
    clauses = []
    for name, present in zip(names, available):
        if not present:
            clauses.append({"name": name, "point_margin": None, "se": None,
                            "lower": None, "upper": None,
                            "status": "INCOMPLETE_MISSING_ARM",
                            "numerical_se_status": None})
            continue
        p, s, lower, upper, z = next(finite)
        clauses.append({"name": name, "point_margin": float(p), "se": float(s),
                        "lower": float(lower), "upper": float(upper),
                        "status": "BOUND_ONLY",
                        "numerical_se_status": "NUMERICALLY_CONSTANT" if z else "VARIABLE"})
    return clauses, family, critical


def independent_a_intervals(seed_records: Mapping[int, Mapping]) -> dict:
    """Evaluate all 6885 predeclared slots without changing K for missing arms."""
    names = independent_ids()
    if len(names) != K:
        raise AssertionError("independent A family order differs")
    unit, prepared = _prepare(seed_records)
    ordered, inverse = np.unique(unit, return_inverse=True)
    if len(ordered) != GROUPS:
        raise AssertionError("A INNER group count changed")
    original_draws = {0: (0, 1, 2), 1: (0, 1, 2)}
    point = _margins(prepared, np.ones(len(unit), np.float64), original_draws)
    available = np.isfinite(point)
    group_rng = np.random.Generator(np.random.PCG64DXSM(20261010))
    draw_rng = np.random.Generator(np.random.PCG64DXSM(20261011))
    boot = np.empty((REPLICATES, int(available.sum())), dtype=np.float64)
    for b in range(REPLICATES):
        sampled = group_rng.choice(ordered, size=GROUPS, replace=True)
        group_counts = np.bincount(np.searchsorted(ordered, sampled), minlength=GROUPS)
        draws = {seed: tuple(int(x) for x in draw_rng.integers(0, 3, size=3))
                 for seed in (0, 1)}
        current = _margins(prepared, group_counts[inverse].astype(np.float64), draws)
        if not np.array_equal(np.isfinite(current), available):
            raise ValueError("A bootstrap finite-family availability changed")
        boot[b] = current[available]
    clauses, family, critical = _bounds(names, point, boot)
    missing = {f"s{seed}": [name for name in NAMES if name not in prepared[seed]]
               for seed in (0, 1)}
    return {"schema": "pcrl-r04-independent-a-crossed-interval-v1",
            "status": "FIXED_6885_INTERVALS_COMPUTED",
            "clause_count": K, "available_clause_count": int(available.sum()),
            "missing_clause_count": int((~available).sum()),
            "missing_arms_by_seed": missing, "family_complete": bool(available.all()),
            "bootstrap_replicates": REPLICATES, "inner_groups": GROUPS,
            "critical_family": family, "critical_95": critical,
            "fixed_family_denominator": K, "studentization_floor": FLOOR,
            "group_seed_bytes_sha256": hashlib.sha256(SEED_GROUP_BYTES).hexdigest(),
            "draw_seed_bytes_sha256": hashlib.sha256(SEED_DRAW_BYTES).hexdigest(),
            "numpy_version": np.__version__, "summaries": list(SUMMARIES),
            "clauses": clauses}
