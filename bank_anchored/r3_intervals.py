"""Fixed-family paired-group intervals for exact-law task utility.

Inputs are owner-only R3 per-person vectors, not candidate model code. This
module computes the 360 utility slots in the predeclared K=1,575 family and
leaves all missing arms null. It does not score recovery readers.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.stats import t as student_t

from .r3_family import ARMS, METRICS, REFERENCES, TASKS, ordered_clause_ids


BOOTSTRAP_SEED = 20261010
BOOTSTRAP_DRAWS = 2048
K_FULL = 1575
ALPHA = .05
REAL_GROUPS = 2234
FIELDS = {"row_id", "unit"} | {
    f"{name}_{task}_{metric}"
    for name in ("candidate", "B0", "U0", "Ucal")
    for task in TASKS for metric in METRICS
}


def _status(metric: str, lower: float, upper: float) -> str:
    if metric == "accuracy":
        return "PASS" if lower >= -.010 else "VIOLATION" if upper < -.010 else "UNRESOLVED"
    cap = .010 if metric == "nll" else .005
    return "PASS" if upper <= cap else "VIOLATION" if lower > cap else "UNRESOLVED"


def expected_utility_intervals(
    records: Mapping[tuple[str, int], Mapping[str, np.ndarray]],
    *, expected_groups: int = REAL_GROUPS,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, dict]:
    """Use one shared duplicate-group bootstrap stream across all records.

    The default is the sealed real-data procedure. Small group/draw arguments
    only support data-free tests; queue execution must use both defaults.
    """
    names = tuple(name for name in ordered_clause_ids() if ":utility:" in name)
    null = {"point": None, "se": None, "lower": None, "upper": None,
            "status": "NOT_ESTIMABLE_DEPENDENCY"}
    result = {name: null.copy() for name in names}
    if not records:
        return result
    if (expected_groups < 2 or bootstrap_draws < 2
            or any(arm not in ARMS or seed not in (0, 1) for arm, seed in records)):
        raise ValueError("utility family key, group count or draw count invalid")
    first = next(iter(records.values()))
    if set(first) != FIELDS:
        raise ValueError("private cost-vector inventory differs")
    row = np.asarray(first["row_id"])
    unit = np.asarray(first["unit"])
    if (row.dtype != np.int64 or unit.dtype != np.int64 or row.ndim != 1
            or unit.shape != row.shape or not len(row)
            or not np.all(np.diff(row) > 0)
            or np.unique(unit).size != expected_groups):
        raise ValueError("paired INNER person/group schema differs")
    for item in records.values():
        if (set(item) != FIELDS
                or not np.array_equal(item["row_id"], row)
                or not np.array_equal(item["unit"], unit)):
            raise ValueError("paired INNER rows/groups or vector inventory differ")
        for field in FIELDS - {"row_id", "unit"}:
            value = np.asarray(item[field])
            if value.dtype != np.float64 or value.shape != row.shape or not np.isfinite(value).all():
                raise ValueError("per-person expected cost vector invalid")

    groups, inverse = np.unique(unit, return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    rng = np.random.Generator(np.random.PCG64DXSM(BOOTSTRAP_SEED))
    sampled = rng.integers(0, len(groups), size=(bootstrap_draws, len(groups)), dtype=np.int64)
    weights = np.stack([np.bincount(line, minlength=len(groups)) for line in sampled])
    denominators = weights @ count
    critical = float(student_t.ppf(1 - ALPHA / (2 * K_FULL), expected_groups - 1))
    per_seed: dict[str, dict[int, tuple[float, np.ndarray]]] = {}
    for (arm, seed), item in records.items():
        for task in TASKS:
            for reference in REFERENCES:
                for metric in METRICS:
                    vector = (np.asarray(item[f"candidate_{task}_{metric}"])
                              - np.asarray(item[f"{reference}_{task}_{metric}"]))
                    group_sum = np.bincount(inverse, weights=vector, minlength=len(groups))
                    replicate = (weights @ group_sum) / denominators
                    point = float(np.mean(vector))
                    name = f"s{seed}:utility:{arm}:{task}:{reference}:{metric}"
                    se = (0.0 if np.max(np.abs(replicate - point)) <= 1e-12
                          else float(np.std(replicate, ddof=1)))
                    lower, upper = point - critical * se, point + critical * se
                    result[name] = {"point": point, "se": se, "lower": lower,
                                    "upper": upper, "status": _status(metric, lower, upper)}
                    base = f"utility:{arm}:{task}:{reference}:{metric}"
                    per_seed.setdefault(base, {})[seed] = (point, replicate)
    for base, by_seed in per_seed.items():
        if set(by_seed) != {0, 1}:
            continue
        point = (by_seed[0][0] + by_seed[1][0]) / 2
        replicate = (by_seed[0][1] + by_seed[1][1]) / 2
        se = (0.0 if np.max(np.abs(replicate - point)) <= 1e-12
              else float(np.std(replicate, ddof=1)))
        lower, upper = point - critical * se, point + critical * se
        metric = base.rsplit(":", 1)[-1]
        result[f"mean:{base}"] = {"point": point, "se": se,
                                  "lower": lower, "upper": upper,
                                  "status": _status(metric, lower, upper)}
    return result
