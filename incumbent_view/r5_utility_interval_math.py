"""Independent paired-group utility intervals for the frozen 192 K948 slots."""
from __future__ import annotations

from math import fsum

import numpy as np
from scipy.stats import t as student_t

from .r5_attack_math import ARMS, K


TASKS = ("income", "occupation")
REFERENCES = ("U0", "Ucal")
METRICS = ("accuracy", "nll", "brier")
GROUPS = 2234
BOOTSTRAPS = 2048
FIELDS = {"row_id", "unit"} | {
    f"{identity}_{task}_{metric}"
    for identity in ("candidate", "P0", "U0", "Ucal")
    for task in TASKS for metric in METRICS
}


def status(metric: str, lower: float, upper: float) -> str:
    if metric == "accuracy":
        return "PASS" if lower >= -0.010 else ("VIOLATION" if upper < -0.010 else "UNRESOLVED")
    cap = 0.010 if metric == "nll" else 0.005
    return "PASS" if upper <= cap else ("VIOLATION" if lower > cap else "UNRESOLVED")


def interval_family(records: dict[tuple[str, int], dict[str, np.ndarray]],
                    *, groups: int = GROUPS, replicates: int = BOOTSTRAPS) -> dict[str, dict]:
    """No runner scorer or interval implementation is imported."""
    out = {f"s{seed}:utility:{arm}:{task}:{reference}:{metric}":
           {"point": None, "se": None, "lower": None, "upper": None,
            "status": "NOT_ESTIMABLE_DEPENDENCY"}
           for seed in (0, 1) for arm in ARMS for task in TASKS
           for reference in REFERENCES for metric in METRICS}
    if len(out) != 192:
        raise AssertionError("frozen utility family cardinality differs")
    if not records:
        return out
    exemplar = next(iter(records.values()))
    row = np.asarray(exemplar["row_id"])
    unit = np.asarray(exemplar["unit"])
    if (row.dtype != np.int64 or unit.dtype != np.int64 or row.ndim != 1 or
            unit.shape != row.shape or len(np.unique(row)) != len(row)):
        raise ValueError("common INNER row/group identity differs")
    _, inverse = np.unique(unit, return_inverse=True)
    if int(inverse.max()) + 1 != groups:
        raise ValueError("fixed duplicate-person group count differs")
    for (arm, seed), item in records.items():
        if ((arm not in ARMS) or seed not in (0, 1) or set(item) != FIELDS or
                not np.array_equal(item["row_id"], row) or
                not np.array_equal(item["unit"], unit)):
            raise ValueError("paired utility source role or columns differ")
        for name in FIELDS - {"row_id", "unit"}:
            array = np.asarray(item[name])
            if array.dtype != np.float64 or array.shape != row.shape or not np.isfinite(array).all():
                raise ValueError("person task cost vector differs")
    rng = np.random.Generator(np.random.PCG64DXSM(20261010))
    sampled = rng.integers(0, groups, size=(replicates, groups), dtype=np.int64)
    weights = np.empty((replicates, groups), dtype=np.float64)
    for replicate, drawn in enumerate(sampled):
        weights[replicate] = np.bincount(drawn, minlength=groups)
    group_sizes = np.bincount(inverse).astype(np.float64)
    denominator = weights @ group_sizes
    if np.any(denominator == 0):
        raise ValueError("bootstrap lost every INNER person")
    critical = float(student_t.ppf(1 - 0.05 / (2 * K), groups - 1))
    for (arm, seed), item in records.items():
        for task in TASKS:
            for reference in REFERENCES:
                for metric in METRICS:
                    difference = item[f"candidate_{task}_{metric}"] - item[f"{reference}_{task}_{metric}"]
                    sums = np.bincount(inverse, weights=difference, minlength=groups)
                    replicates_value = (weights @ sums) / denominator
                    point = fsum(map(float, difference)) / len(difference)
                    se = 0.0 if np.all(difference == difference[0]) else float(np.std(replicates_value, ddof=1))
                    lower, upper = point - critical * se, point + critical * se
                    name = f"s{seed}:utility:{arm}:{task}:{reference}:{metric}"
                    out[name] = {"point": point, "se": se, "lower": lower,
                                 "upper": upper, "status": status(metric, lower, upper),
                                 "df": groups - 1, "K": K}
    return out
