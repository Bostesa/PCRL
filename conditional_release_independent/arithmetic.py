"""Independent definitions for the locked finite-law study.

Inputs are public formats or admitted role-specific arrays. No optimizer,
deployment, runner metric, or selection implementation is imported here.
"""
from __future__ import annotations

import math
from collections import defaultdict
import numpy as np


def check_law(rows, tolerance=1e-10):
    """Rows contain lists of (emitted tuple, mass), including aliases."""
    merged = []
    for row in rows:
        accumulated = defaultdict(float)
        for wire, mass in row:
            if not math.isfinite(mass) or mass < -tolerance:
                raise ValueError("invalid law mass")
            accumulated[tuple(wire)] += float(mass)
        if abs(math.fsum(accumulated.values()) - 1) > tolerance:
            raise ValueError("law does not sum to one")
        merged.append(dict(accumulated))
    return merged


def change_rate(rows, original_codes):
    return math.fsum(math.fsum(p for z, p in r.items() if tuple(z) != tuple(o))
                     for r, o in zip(rows, original_codes, strict=True)) / len(rows)


def bayes_objective(rows, sex):
    """Empirical joint recovery + half each empirical marginal recovery."""
    if len(rows) != len(sex) or not rows:
        raise ValueError("unpaired or empty law")
    totals = []
    for view in (None, 0, 1):
        mass = defaultdict(lambda: [0.0, 0.0])
        for law, label in zip(rows, sex, strict=True):
            if label not in (0, 1):
                raise ValueError("SEX is not binary")
            for code, weight in law.items():
                key = code if view is None else code[view]
                mass[key][int(label)] += weight / len(rows)
        totals.append(math.fsum(max(v) for v in mass.values()))
    return totals[0] + 0.5 * (totals[1] + totals[2])


def utility_rows(rows, labels, bank, recipient, nll_floor=1e-12):
    """Expected accuracy/NLL/source summed multiclass Brier per person."""
    result = np.zeros((len(rows), 3), dtype=np.float64)
    for i, (law, target) in enumerate(zip(rows, labels, strict=True)):
        for codes, mass in law.items():
            probability = np.asarray(bank[codes[recipient]], dtype=np.float64)
            if target < 0 or target >= len(probability):
                raise ValueError("task label outside bank")
            decision = int(np.argmax(probability))
            onehot = np.zeros(len(probability)); onehot[target] = 1
            result[i] += mass * np.array([
                float(decision == target),
                -math.log(max(float(probability[target]), nll_floor)),
                math.fsum(float(v) ** 2 for v in probability - onehot),
            ])
    return result


def integrated_auc(score_laws, labels, row_weights=None):
    """Weighted exact single-query AUC, with exact score ties.

    score_laws contains dictionaries score -> mass for each person.
    Bernoulli release draws are independent across different people. The
    numerator is a positive-negative pair expectation, not an AUC of a
    mean output or a finite Monte Carlo draw.
    """
    weights = np.ones(len(labels)) if row_weights is None else row_weights
    totals = defaultdict(lambda: [0.0, 0.0])
    for law, label, weight in zip(score_laws, labels, weights, strict=True):
        if label not in (0, 1) or weight < 0:
            raise ValueError("invalid label or weight")
        if abs(math.fsum(law.values()) - 1) > 1e-10:
            raise ValueError("invalid score law")
        for score, mass in law.items():
            if not math.isfinite(score) or mass < 0:
                raise ValueError("invalid score mass")
            totals[float(score)][int(label)] += float(weight) * float(mass)
    negative = math.fsum(v[0] for v in totals.values())
    positive = math.fsum(v[1] for v in totals.values())
    if negative <= 0 or positive <= 0:
        raise ValueError("AUC requires both classes")
    negative_below = 0.0
    pieces = []
    for score in sorted(totals):
        n, p = totals[score]
        pieces.append(p * (negative_below + 0.5 * n))
        negative_below += n
    return math.fsum(pieces) / (negative * positive)


def exhaustive_auc(score_laws, labels, row_weights=None):
    """Slow independent cross-outcome oracle for small fixtures."""
    weights = np.ones(len(labels)) if row_weights is None else row_weights
    numerator = []; positive = negative = 0.0
    for i, yi in enumerate(labels):
        if yi == 1:
            positive += weights[i]
            for j, yj in enumerate(labels):
                if yj == 0:
                    for sp, mp in score_laws[i].items():
                        for sn, mn in score_laws[j].items():
                            credit = float(sp > sn) + 0.5 * float(sp == sn)
                            numerator.append(weights[i] * weights[j] * mp * mn * credit)
        elif yi == 0:
            negative += weights[i]
        else:
            raise ValueError("invalid label")
    return math.fsum(numerator) / (positive * negative)


def validate_group_pairing(groups, sex):
    """A person's duplicates must share SEX; positive/negative coins differ."""
    observed = {}
    for group, label in zip(groups, sex, strict=True):
        if group in observed and observed[group] != label:
            raise ValueError("SEX varies within duplicate group")
        observed[group] = label
    return observed


def simultaneous_interval(point, bootstrap_values, family_size, group_count):
    from scipy.stats import t
    values = np.asarray(bootstrap_values, dtype=float)
    if len(values) < 2048 or not np.all(np.isfinite(values)):
        raise ValueError("at least 2048 finite paired bootstrap values required")
    if family_size < 1 or group_count < 2:
        raise ValueError("invalid inference family")
    sd = float(np.std(values, ddof=1))
    critical = float(t.ppf(1 - 0.05 / (2 * family_size), group_count - 1))
    return {"point": float(point), "sd": sd, "critical": critical,
            "lower": float(point - critical * sd), "upper": float(point + critical * sd)}


def cap_status(lower, upper, cap, direction="upper"):
    if direction == "upper":
        return "PASS" if upper <= cap else "VIOLATION" if lower > cap else "UNRESOLVED"
    if direction == "lower":
        return "PASS" if lower >= cap else "VIOLATION" if upper < cap else "UNRESOLVED"
    raise ValueError("unknown boundary direction")


def lp_certificate(c, x, aub, bub, aeq, beq, lower, upper,
                   dual_ub, dual_eq, dual_lower, dual_upper):
    """Scipy/HiGHS marginal convention for a minimization LP.

    Inequality and upper-bound marginals are nonpositive; lower-bound
    marginals are nonnegative. Equality marginals are unrestricted.
    Strong duality concerns the supplied finite matrix, never population.
    """
    c, x = np.asarray(c), np.asarray(x)
    ubres = np.asarray(bub) - aub @ x
    eqres = aeq @ x - np.asarray(beq)
    lo = np.asarray(lower); hi = np.asarray(upper)
    dl = np.asarray(dual_lower); du = np.asarray(dual_upper)
    yu = np.asarray(dual_ub); ye = np.asarray(dual_eq)
    stationarity = c - aub.T @ yu - aeq.T @ ye - dl - du
    primal = float(c @ x)
    dual = float(np.asarray(bub) @ yu + np.asarray(beq) @ ye)
    finite_lo = np.isfinite(lo); finite_hi = np.isfinite(hi)
    dual += float(lo[finite_lo] @ dl[finite_lo] + hi[finite_hi] @ du[finite_hi])
    maxabs = lambda v: float(np.max(np.abs(v), initial=0))
    return {
        "objective": primal, "dual_objective": dual, "gap": primal - dual,
        "max_inequality_violation": float(np.max(-ubres, initial=0)),
        "max_equality_residual": maxabs(eqres),
        "max_bound_violation": float(max(0, np.max(lo-x, initial=0), np.max(x-hi, initial=0))),
        "stationarity_residual": maxabs(stationarity),
        "dual_sign_violation": float(max(0, np.max(yu, initial=0), np.max(-dl, initial=0), np.max(du, initial=0))),
        "unbounded_bound_dual_residual": max(maxabs(dl[~finite_lo]), maxabs(du[~finite_hi])),
        "complementarity_residual": max(maxabs(yu * ubres), maxabs(dl[finite_lo] * (x-lo)[finite_lo]), maxabs(du[finite_hi] * (hi-x)[finite_hi])),
    }
