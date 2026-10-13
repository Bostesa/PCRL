"""Locked paired duplicate-group inference for conditional development.

Reader/kernel fitting is conditional and fixed. One multinomial group sample
is shared across ALL arms and BOTH model seeds. The finite reader envelope is
recomputed inside each replicate using orientations frozen on ATT validation.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import t

from .metrics import matrix_auc, score_group_matrices

BOOTSTRAP_REPLICATES = 2048
BOOTSTRAP_SEED = 2026101105
ALPHA = 0.05


def paired_group_counts(groups, *, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    values = np.unique(np.asarray(groups))
    if len(values) < 2 or replicates < BOOTSTRAP_REPLICATES:
        raise ValueError("at least 2048 resamples of two duplicate groups required")
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(values), np.full(len(values), 1.0 / len(values)), size=replicates)
    return values, counts


def envelope_distribution(reader_scores, mass, sex, groups, *, counts, group_values, chunk=32):
    """One seed/view: point max and per-replicate max over frozen orientations."""
    if not reader_scores:
        raise ValueError("declared finite reader slate is empty")
    count = np.asarray(counts)
    if count.ndim != 2 or count.shape[1] != len(group_values):
        raise ValueError("paired bootstrap group count matrix differs")
    result = np.full(len(count), -np.inf)
    points = {}
    for name, scores in reader_scores.items():
        matrix = score_group_matrices(scores, mass, sex, groups)
        if not np.array_equal(matrix["groups"], group_values):
            raise ValueError("bootstrap people not paired across arms/seeds")
        points[name] = float(matrix_auc(matrix, np.ones(len(group_values))))
        for start in range(0, len(count), chunk):
            values = matrix_auc(matrix, count[start:start + chunk])
            result[start:start + chunk] = np.maximum(result[start:start + chunk], values)
    return {"point": max(points.values()), "bootstrap": result, "reader_points": points}


def mean_distribution(values, groups, *, counts, group_values):
    """Expected per-person utility, retaining repeated rows in group resampling."""
    v = np.asarray(values, dtype=np.float64)
    g = np.asarray(groups)
    if v.ndim != 1 or g.shape != v.shape or not np.isfinite(v).all():
        raise ValueError("utility/group aligned finite vectors required")
    actual, index = np.unique(g, return_inverse=True)
    if not np.array_equal(actual, group_values):
        raise ValueError("utility and recovery bootstrap group identities differ")
    totals = np.bincount(index, weights=v, minlength=len(actual))
    row_count = np.bincount(index, minlength=len(actual))
    return {"point": float(np.mean(v)), "bootstrap": (counts @ totals) / (counts @ row_count)}


def simultaneous_interval(point, bootstrap, *, K, reporting_groups):
    """Approximate Bonferroni-t construction, never an exact coverage claim."""
    vals = np.asarray(bootstrap, dtype=np.float64)
    if vals.ndim != 1 or len(vals) < BOOTSTRAP_REPLICATES or not np.isfinite(vals).all():
        raise ValueError("at least 2048 finite paired bootstrap endpoints required")
    if type(K) is not int or K < 1 or type(reporting_groups) is not int or reporting_groups < 2:
        raise ValueError("locked positive K and reporting group count required")
    se = float(vals.std(ddof=1))
    df = reporting_groups - 1
    critical = float(t.ppf(1 - ALPHA / (2 * K), df))
    half = se * critical
    return {"point": float(point), "bootstrap_sd": se, "lower": float(point - half),
            "upper": float(point + half), "K": K, "df": df, "critical_value": critical,
            "replicates": len(vals), "alpha": ALPHA,
            "scope": "approximate conditional development; reader/kernel fixed; not post-selection confirmation"}


def upper_cap_status(interval, cap):
    if interval["upper"] <= cap:
        return "PASS"
    if interval["lower"] > cap:
        return "VIOLATION"
    return "UNRESOLVED"


def lower_target_status(interval, target):
    if interval["lower"] >= target:
        return "PASS"
    if interval["upper"] < target:
        return "SHORTFALL"
    return "UNRESOLVED"


def lower_strict_status(interval, target):
    """A positive same-budget effect needs a bound strictly above zero.

    Exact-boundary equality is unresolved under the frozen strict convention.
    Structural identity, when independently proved, is reported separately.
    """
    if interval["lower"] > target:
        return "PASS"
    if interval["upper"] < target:
        return "SHORTFALL"
    return "UNRESOLVED"
