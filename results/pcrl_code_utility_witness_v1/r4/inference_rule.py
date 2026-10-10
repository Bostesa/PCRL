"""Full-family approximate paired-group interval and orientation conventions.

The critical value is fixed by all 2,376 prospective clauses, including any
dependency-missing capacity branch or inadmissible inherited bank. It is not
recomputed from the subset with observed results. This module reads no data.
"""

from __future__ import annotations

import math

from scipy.stats import t

from .finite_family import ARMS, REFERENCES, ordered_clause_ids


ALPHA = 0.05
K_FULL = len(ordered_clause_ids())
INNER_GROUPS = 2234
DF = INNER_GROUPS - 1
BOOTSTRAP_DRAWS = 2048
CRITICAL_95 = float(t.ppf(1 - ALPHA / (2 * K_FULL), DF))
if K_FULL != 2376 or abs(CRITICAL_95 - 4.262627512538624) > 1e-12:
    raise AssertionError("full prospective family or critical value changed")


def interval_bounds(point: float, bootstrap_se: float) -> tuple[float, float]:
    """Two-sided full-K t/Bonferroni interval from paired-group bootstrap SE."""
    if not math.isfinite(point) or not math.isfinite(bootstrap_se) or bootstrap_se < 0:
        raise ValueError("finite point and nonnegative paired-group SE required")
    half = CRITICAL_95 * bootstrap_se
    return point - half, point + half


def utility_status(metric: str, point: float | None,
                   lower: float | None, upper: float | None) -> str:
    """Apply the unchanged inclusive common utility caps to a signed cost."""
    if point is None and lower is None and upper is None:
        return "NOT_ESTIMABLE_DEPENDENCY"
    if (metric not in ("accuracy", "nll", "brier")
            or any(value is None or not math.isfinite(value)
                   for value in (point, lower, upper))
            or not lower <= point <= upper):
        raise ValueError("utility interval or metric invalid")
    if metric == "accuracy":
        return "PASS" if lower >= -.010 else "VIOLATION" if upper < -.010 else "UNRESOLVED"
    cap = .010 if metric == "nll" else .005
    return "PASS" if upper <= cap else "VIOLATION" if lower > cap else "UNRESOLVED"


def orient_auc_contrast(clause_name: str, *, point: float, lower: float,
                        upper: float, candidate: str, control: str) -> dict:
    """Map raw first-minus-second to pair benefit or local worsening.

    A registered `pair` contrast is interpreted as control minus candidate.
    A registered `r1`/`r2` contrast is candidate minus control. Negation
    exchanges and negates interval endpoints. Inclusion in the family does not
    make a cross-contract anchor an eligible matched control.
    """
    if (clause_name not in ordered_clause_ids() or candidate == control
            or not all(math.isfinite(x) for x in (point, lower, upper))
            or not lower <= point <= upper):
        raise ValueError("AUC clause or interval invalid")
    summary, kind, first, second, view, metric = clause_name.split(":")
    if (summary not in ("s0", "s1", "mean") or metric != "auc"
            or kind not in ("arm_pair", "ref_pair")
            or {candidate, control} != {first, second}
            or candidate not in ARMS or control not in (*ARMS, *REFERENCES)):
        raise ValueError("AUC clause does not bind this candidate/control")
    if view == "pair":
        direction, threshold = "control_minus_candidate", .020
        sign = 1 if first == control else -1
    elif view in ("r1", "r2"):
        direction, threshold = "candidate_minus_control", .010
        sign = 1 if first == candidate else -1
    else:
        raise ValueError("unknown complete view")
    oriented = (point, lower, upper) if sign == 1 else (-point, -upper, -lower)
    estimate, lo, hi = oriented
    if view == "pair":
        status = "PASS" if lo >= threshold else "VIOLATION" if hi < threshold else "UNRESOLVED"
    else:
        status = "PASS" if hi <= threshold else "VIOLATION" if lo > threshold else "UNRESOLVED"
    return {"point": estimate, "lower": lo, "upper": hi,
            "direction": direction, "threshold": threshold, "status": status,
            "raw_clause": clause_name, "raw_sign": sign}
