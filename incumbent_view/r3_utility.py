"""Pure exact-law and actual-wire task utility; no data loading or SEX access."""
from collections.abc import Mapping
from decimal import Decimal, localcontext
from math import fsum
import numpy as np

TASKS = (("income", 0, 2), ("occupation", 1, 6))
METRICS = ("accuracy", "nll", "brier")
FLOOR = 1e-12
NUMERIC_ENVELOPE = 1e-10
CAPS = {"fit": {"accuracy": -.002, "nll": .001, "brier": .0005},
        "inner": {"accuracy": -.010, "nll": .010, "brier": .005}}


def validate(law, tables, labels):
    required = {"code_r1", "code_r2", "mass", "valid", "b0_code_r1", "b0_code_r2"}
    if not isinstance(law, Mapping) or not required.issubset(law):
        raise ValueError("P0 law keys missing")
    mass, valid = np.asarray(law["mass"]), np.asarray(law["valid"])
    if (mass.dtype != np.float64 or mass.ndim != 2 or not len(mass)
            or valid.dtype != np.bool_ or valid.shape != mass.shape
            or not np.isfinite(mass).all() or np.any(mass < 0)
            or np.any(mass[~valid] != 0) or np.any(~valid.any(axis=1))
            or not np.allclose(mass.sum(axis=1), 1, rtol=0, atol=1e-12)):
        raise ValueError("law probabilities invalid")
    n, width = mass.shape
    if len(tables) != 2 or len(labels) != 2:
        raise ValueError("two recipient tables/labels required")
    for _, j, classes in TASKS:
        table, y = np.asarray(tables[j]), np.asarray(labels[j])
        code, p0 = np.asarray(law[f"code_r{j+1}"]), np.asarray(law[f"b0_code_r{j+1}"])
        if (table.dtype != np.float64 or table.ndim != 2 or table.shape[1] != classes
                or not len(table) or not np.isfinite(table).all() or np.any(table < 0)
                or not np.allclose(table.sum(axis=1), 1, rtol=0, atol=1e-10)
                or code.dtype != np.int64 or code.shape != (n, width)
                or p0.dtype != np.int64 or p0.shape != (n,)
                or np.any(code < 0) or np.any(code >= len(table))
                or np.any(p0 < 0) or np.any(p0 >= len(table))
                or y.dtype != np.int64 or y.shape != (n,) or np.any(y < 0) or np.any(y >= classes)):
            raise ValueError("table, global code or label schema invalid")
        if np.any(valid & (table[code].argmax(-1) != table[p0].argmax(-1)[:, None])):
            raise ValueError("support changes P0 decision class")
    identity = valid & (law["code_r1"] == law["b0_code_r1"][:, None]) & (law["code_r2"] == law["b0_code_r2"][:, None])
    if np.any(~identity.any(1)):
        raise ValueError("identity missing from support")
    return mass


def costs(prob, y):
    prob, y = np.asarray(prob), np.asarray(y)
    chosen = np.take_along_axis(prob, y[..., None], -1)[..., 0]
    return {"accuracy": (prob.argmax(-1) == y).astype(np.float64),
            "nll": -np.log(np.maximum(chosen, FLOOR)),
            "brier": np.sum((prob - np.eye(prob.shape[-1])[y]) ** 2, axis=-1)}


def evaluate_law(law, tables, labels):
    mass = validate(law, tables, labels)
    out = {}
    for task, j, _ in TASKS:
        code, p0, y = law[f"code_r{j+1}"], law[f"b0_code_r{j+1}"], labels[j]
        action = costs(tables[j][code], np.broadcast_to(y[:, None], code.shape))
        expected = {m: np.sum(mass * action[m], axis=1, dtype=np.float64) for m in METRICS}
        baseline = costs(tables[j][p0], y)
        # Structural identity gets the exact original vector, avoiding a spurious
        # zero margin from a repeated mass sum. This requires an exact unit law.
        identity = (np.sum((mass > 0), axis=1) == 1) & np.all((mass == 0) | (mass == 1), axis=1)
        chosen_p0 = np.all((mass == 0) | ((code == p0[:, None]) & (law[f"code_r{2-j}"] == law[f"b0_code_r{2-j}"][:, None])), axis=1)
        for m in METRICS:
            expected[m][identity & chosen_p0] = baseline[m][identity & chosen_p0]
        out[task] = {"expected": expected, "p0": baseline}
    return out


def evaluate_realization(codes, tables, labels):
    if len(codes) != 2:
        raise ValueError("two actual codes required")
    out = {}
    for task, j, _ in TASKS:
        code = np.asarray(codes[j])
        if code.dtype != np.int64 or code.shape != np.asarray(labels[j]).shape or np.any(code < 0) or np.any(code >= len(tables[j])):
            raise ValueError("actual wire code invalid")
        out[task] = costs(tables[j][code], labels[j])
    return out


def stable_mean(values):
    values = np.asarray(values)
    if values.dtype != np.float64 or values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite float64 person vector required")
    return fsum(map(float, values)) / len(values)


def point_clause(metric, margin, regime="inner"):
    if regime not in CAPS or metric not in METRICS or not np.isfinite(margin):
        raise ValueError("clause arguments invalid")
    cap = CAPS[regime][metric]
    passed = margin >= cap if metric == "accuracy" else margin <= cap
    return {"metric": metric, "regime": regime, "margin": float(margin), "cap": cap,
            "status": "PASS" if passed else "FAIL"}


def decimal_fit_check(law, tables, labels, evaluated):
    """Full binary-input Decimal(50) cost replay, independent of numpy log/sums.

    Decimal.from_float preserves the supplied binary data; precision checks do
    not pretend rounded bank parameters are unknown real probabilities.
    """
    validate(law, tables, labels)
    out = {}
    with localcontext() as ctx:
        ctx.prec = 50
        d = Decimal.from_float
        floor = d(FLOOR)
        for task, j, classes in TASKS:
            table = tables[j]
            decimal_table = [[d(float(x)) for x in row] for row in table]
            cache = {}
            def scalar(code, y, metric):
                key = (int(code), int(y), metric)
                if key not in cache:
                    p = decimal_table[int(code)]
                    cache[key] = (Decimal(int(np.argmax(table[int(code)]) == y)) if metric == "accuracy" else
                                  -max(p[int(y)], floor).ln() if metric == "nll" else
                                  sum(((x - Decimal(int(k == y))) ** 2 for k, x in enumerate(p)), Decimal(0)))
                return cache[key]
            n = len(labels[j])
            for metric in METRICS:
                total = Decimal(0)
                max_row_error = 0.
                for i, y in enumerate(labels[j]):
                    row = sum((d(float(w)) * scalar(c, y, metric)
                               for c, w in zip(law[f"code_r{j+1}"][i], law["mass"][i]) if w != 0), Decimal(0))
                    base = scalar(law[f"b0_code_r{j+1}"][i], y, metric)
                    total += row - base
                    primary_row = evaluated[task]["expected"][metric][i] - evaluated[task]["p0"][metric][i]
                    max_row_error = max(max_row_error, float(abs((row - base) - d(float(primary_row)))))
                independent = float(total / Decimal(n))
                primary = stable_mean(evaluated[task]["expected"][metric] - evaluated[task]["p0"][metric])
                error = abs(primary - independent)
                slack = (primary - CAPS["fit"][metric] if metric == "accuracy" else CAPS["fit"][metric] - primary)
                robust = ("PASS" if slack >= NUMERIC_ENVELOPE else "FAIL" if slack < -NUMERIC_ENVELOPE else "NUMERIC_BOUNDARY_UNRESOLVED")
                out[f"{task}:{metric}"] = {**point_clause(metric, primary, "fit"),
                    "decimal_margin": independent, "observed_margin_error": error,
                    "max_person_arithmetic_error": max_row_error,
                    "mathematical_slack_estimate": slack, "numeric_error_envelope": NUMERIC_ENVELOPE,
                    "numerical_admission": "PASS" if max(error, max_row_error) <= NUMERIC_ENVELOPE else "FAIL",
                    "robust_status": robust}
    return out
