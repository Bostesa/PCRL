"""Small convex finite-law release programmes; no population inference."""

import itertools

import cvxpy as cp
import numpy as np

from .finite import conditional_mi, information_radius_bracket
from .synthetic import task_coefficients, task_cost


def _conditional_tables(law, role):
    base = law.sum(axis=-1)
    if role == "A":
        return base.sum(axis=2)
    if role == "AB":
        return base.reshape(base.shape[0], -1, base.shape[-1])
    raise ValueError(role)


def _cmi_expression(tables, q):
    terms = []
    for h in range(tables.shape[1]):
        sht = tables[:, h, :]
        sh = sht.sum(axis=1)
        ph = sh.sum()
        if ph == 0:
            continue
        a = sht @ q
        m = cp.sum(a, axis=0)
        b = cp.multiply(sh[:, None] / ph, cp.reshape(m, (1, q.shape[1]), order="C"))
        for s in range(len(sh)):
            if sh[s] > 0:
                terms.append(cp.sum(cp.rel_entr(a[s, :], b[s, :])))
    return cp.sum(cp.hstack(terms)) if terms else cp.Constant(0)


def _zero_equations(tables, q):
    equations = []
    for h in range(tables.shape[1]):
        sht = tables[:, h, :]
        sh = sht.sum(axis=1)
        ph = sh.sum()
        if ph == 0:
            continue
        a = sht @ q
        m = cp.sum(a, axis=0)
        for s in range(len(sh)):
            if sh[s] > 0:
                coef = sht[s, :] - (sh[s] / ph) * sht.sum(axis=0)
                equations.append((a[s, :] == (sh[s] / ph) * m, coef))
    return equations


def _cmi_gradient(tables, q):
    """Gradient at a strictly positive channel on supported cells."""
    grad = np.zeros_like(q)
    for h in range(tables.shape[1]):
        sht = tables[:, h, :]
        sh = sht.sum(axis=1)
        ph = sh.sum()
        if ph == 0:
            continue
        a = sht @ q
        m = a.sum(axis=0)
        for s in range(len(sh)):
            if sh[s] > 0:
                ratio = a[s, :] / ((sh[s] / ph) * m)
                grad += sht[s, :, None] * np.log(ratio)[None, :]
    return grad


def optimize_channel(nominal, laws, budget, method):
    """Minimize nominal cost at a CMI or all-input-radius budget.

    `robust` checks every finite listed law; `nominal` checks first law.
    The published Diaz-style finite-law Shannon-CMI adaptation is exactly
    `robust`, so no duplicate optimization is performed.
    """
    if method not in ("nominal", "robust", "capacity"):
        raise ValueError(method)
    n_t = nominal.shape[-2]
    q = cp.Variable((n_t, 2), nonneg=True)
    constraints = [cp.sum(q, axis=1) == 1]
    zero_meta = []
    privacy_meta = []
    r = None
    if method == "capacity":
        if budget == 0:
            for t in range(1, n_t):
                constraint = q[t, :] == q[0, :]
                constraints.append(constraint)
                coef = np.zeros(n_t)
                coef[t], coef[0] = 1, -1
                zero_meta.append((constraint, coef))
        else:
            r = cp.Variable(2, nonneg=True)
            constraints.append(cp.sum(r) == 1)
            for t in range(n_t):
                constraint = cp.sum(cp.rel_entr(q[t, :], r)) <= budget
                constraints.append(constraint)
                privacy_meta.append((constraint, t))
    else:
        selected = laws[:1] if method == "nominal" else laws
        for law in selected:
            for role in ("A", "AB"):
                tables = _conditional_tables(law, role)
                if budget == 0:
                    for constraint, coef in _zero_equations(tables, q):
                        constraints.append(constraint)
                        zero_meta.append((constraint, coef))
                else:
                    constraint = _cmi_expression(tables, q) <= budget
                    constraints.append(constraint)
                    privacy_meta.append((constraint, law, role))
    objective = cp.Minimize(cp.sum(cp.multiply(task_coefficients(nominal), q)))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10,
                  tol_feas=1e-10, max_iter=10000)
    if q.value is None or problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"solver failed: {problem.status}")
    channel = np.maximum(q.value, 0)
    channel /= channel.sum(axis=1, keepdims=True)
    # For any multipliers, the affine Lagrangian minimum over each row
    # simplex is a global lower bound. A small guard covers floating summation.
    coeff = task_coefficients(nominal)
    affine = coeff.copy()
    offset = 0.0
    if budget == 0:
        for constraint, coef in zero_meta:
            dual = np.asarray(constraint.dual_value, dtype=float).reshape(2)
            affine += coef[:, None] * dual[None, :]
    elif method == "capacity":
        alpha = 1e-9
        q0 = (1 - alpha) * channel + alpha / 2
        r0 = (1 - alpha) * np.maximum(np.asarray(r.value), 0) + alpha / 2
        r0 /= r0.sum()
        affine_r = np.zeros(2)
        for constraint, t in privacy_meta:
            lam = max(0.0, float(constraint.dual_value))
            g = float(np.sum(q0[t] * np.log(q0[t] / r0)))
            gq = np.log(q0[t] / r0) + 1
            gr = -q0[t] / r0
            affine[t] += lam * gq
            affine_r += lam * gr
            offset += lam * (g - gq @ q0[t] - gr @ r0 - budget)
        offset += float(np.min(affine_r))
    else:
        q0 = (1 - 1e-9) * channel + 5e-10
        for constraint, law, role in privacy_meta:
            lam = max(0.0, float(constraint.dual_value))
            tables = _conditional_tables(law, role)
            grad = _cmi_gradient(tables, q0)
            axes = (1,) if role == "A" else (1, 2)
            g = conditional_mi(law, q0, axes)
            affine += lam * grad
            offset += lam * (g - float(np.sum(grad * q0)) - budget)
    numeric_guard = 1e-7
    objective_lower_bound = float(np.sum(np.min(affine, axis=1)) + offset - numeric_guard)
    cmi = [{role: conditional_mi(law, channel, axes)
            for role, axes in (("A", (1,)), ("AB", (1, 2)))} for law in laws]
    max_cmi = max(v for row in cmi for v in row.values())
    if method == "capacity":
        radius = information_radius_bracket(channel, tol=1e-9)
        if radius["upper"] > budget + 1e-7:
            raise RuntimeError("radius constraint failed independent check")
    else:
        radius = None
        if method == "robust" and max_cmi > budget + 1e-7:
            raise RuntimeError("CMI constraint failed independent check")
        if method == "nominal" and max(cmi[0].values()) > budget + 1e-7:
            raise RuntimeError("nominal CMI constraint failed independent check")
    return {"method": method, "budget": budget, "status": problem.status,
            "cost": task_cost(nominal, channel), "channel": channel.tolist(),
            "cmi_by_law": cmi, "max_full_view_cmi": max_cmi,
            "radius": radius, "solver_value": float(problem.value),
            "objective_lower_bound": objective_lower_bound,
            "objective_upper_bound": task_cost(nominal, channel),
            "dual_numeric_guard": numeric_guard,
            "simplex_residual": float(np.max(np.abs(channel.sum(axis=1) - 1))),
            "solver_iterations": problem.solver_stats.num_iters}


def deterministic_controls(law, laws):
    out = []
    for assignment in itertools.product(range(2), repeat=law.shape[-2]):
        q = np.eye(2)[list(assignment)]
        cmi = [{role: conditional_mi(p, q, axes)
                for role, axes in (("A", (1,)), ("AB", (1, 2)))} for p in laws]
        out.append({"assignment": assignment, "cost": task_cost(law, q),
                    "cmi_by_law": cmi,
                    "max_full_view_cmi": max(v for row in cmi for v in row.values())})
    return sorted(out, key=lambda x: (x["cost"], x["assignment"]))
