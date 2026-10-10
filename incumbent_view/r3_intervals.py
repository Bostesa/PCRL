"""Fixed K948 seed-specific exact-law utility intervals, paired duplicate groups."""
from collections.abc import Mapping
import numpy as np
from scipy.stats import t as student_t
from .r3_family import ARMS, TASKS, REFERENCES, METRICS, K, ordered_clause_ids
from .r3_utility import stable_mean

BOOTSTRAP_SEED = 20261010
BOOTSTRAP_DRAWS = 2048
REAL_GROUPS = 2234
ALPHA = .05
FIELDS = {"row_id", "unit"} | {f"{a}_{t}_{m}" for a in ("candidate", "P0", "U0", "Ucal") for t in TASKS for m in METRICS}


def status(metric, lower, upper):
    if metric == "accuracy":
        return "PASS" if lower >= -.010 else "VIOLATION" if upper < -.010 else "UNRESOLVED"
    cap = .010 if metric == "nll" else .005
    return "PASS" if upper <= cap else "VIOLATION" if lower > cap else "UNRESOLVED"


def expected_utility_intervals(records: Mapping, *, expected_groups=REAL_GROUPS, bootstrap_draws=BOOTSTRAP_DRAWS):
    names = tuple(x for x in ordered_clause_ids() if ":utility:" in x)
    result = {x: {"point": None, "se": None, "lower": None, "upper": None, "status": "NOT_ESTIMABLE_DEPENDENCY"} for x in names}
    if not records:
        return result
    if expected_groups < 2 or bootstrap_draws < 2 or any(a not in ARMS or s not in (0, 1) for a, s in records):
        raise ValueError("invalid interval family key or fixture sizes")
    first = next(iter(records.values()))
    row, unit = np.asarray(first["row_id"]), np.asarray(first["unit"])
    if row.dtype != np.int64 or unit.dtype != np.int64 or row.ndim != 1 or unit.shape != row.shape or not len(row) or np.unique(row).size != len(row) or np.unique(unit).size != expected_groups:
        raise ValueError("paired person/group schema invalid")
    for item in records.values():
        if set(item) != FIELDS or not np.array_equal(item["row_id"], row) or not np.array_equal(item["unit"], unit):
            raise ValueError("all records must share exact row/group order and field inventory")
        for field in FIELDS - {"row_id", "unit"}:
            v = np.asarray(item[field])
            if v.dtype != np.float64 or v.shape != row.shape or not np.isfinite(v).all():
                raise ValueError("person cost vector invalid")
    groups, inverse = np.unique(unit, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    rng = np.random.Generator(np.random.PCG64DXSM(BOOTSTRAP_SEED))
    indices = rng.integers(0, len(groups), (bootstrap_draws, len(groups)), dtype=np.int64)
    weights = np.stack([np.bincount(x, minlength=len(groups)) for x in indices])
    denominators = weights @ counts
    critical = float(student_t.ppf(1 - ALPHA / (2 * K), expected_groups - 1))
    for (arm, seed), item in records.items():
        for task in TASKS:
            for reference in REFERENCES:
                for metric in METRICS:
                    vector = item[f"candidate_{task}_{metric}"] - item[f"{reference}_{task}_{metric}"]
                    sums = np.bincount(inverse, weights=vector, minlength=len(groups))
                    replicate = weights @ sums / denominators
                    point = stable_mean(vector)
                    # Only an exact constant vector has structurally zero SE;
                    # do not suppress small variance by a numerical threshold.
                    se = 0. if np.all(vector == vector[0]) else float(np.std(replicate, ddof=1))
                    lower, upper = point - critical * se, point + critical * se
                    result[f"s{seed}:utility:{arm}:{task}:{reference}:{metric}"] = {
                        "point": point, "se": se, "lower": lower, "upper": upper,
                        "status": status(metric, lower, upper), "df": expected_groups - 1, "K": K}
    return result
