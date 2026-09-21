"""Finite randomized channels with independently checked conditional information.

Tables use ``p[s, c, t]`` and channels use ``Q[t, z]``. Information and losses
are in nats. Cost entries already contain the input-state probability mass:
the objective is ``sum(cost * Q)``, never another multiplication by ``p(t)``.

At positive budgets the affine joint masses enter CVXPY's relative-entropy
cone. Zero budgets use the equivalent independence equations in a HiGHS LP.
All returned candidates are checked using NumPy/SciPy, outside CVXPY.
"""

from __future__ import annotations

import operator
import time
import warnings
from collections.abc import Mapping

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.special import rel_entr


ACCEPTANCE_TOLERANCES = {
    "simplex": 1e-7,
    "nonnegative": 1e-8,
    "support": 1e-7,
    "cmi": 1e-7,
    "zero_linear": 1e-8,
    "witness_objective": 1e-7,
    "repair": 1e-7,
    "privacy_row_roundoff_relative": 16 * np.finfo(float).eps,
}
SOLVER_SETTINGS = {
    "HIGHS": {"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9},
    "CLARABEL": {"max_iter": 500, "tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9, "tol_feas": 1e-9},
    "SCS": {"max_iters": 100000, "eps": 1e-7, "acceleration_lookback": 10, "normalize": True},
}


def _size(value, name):
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _indices(values, size, name):
    raw = np.asarray(values)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if not np.all(np.isfinite(raw)) or not np.all(raw == np.floor(raw)):
        raise ValueError(f"{name} must contain finite integers")
    if np.any(raw < 0) or np.any(raw >= size):
        raise ValueError(f"{name} contains an out-of-range index")
    return raw.astype(np.int64)


def _weights(weights, n):
    w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (n,) or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite, nonnegative, and match the observations")
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("weights must have positive finite total mass")
    return w / total


def joint_table(s, c, t, n_s, n_c, n_t, weights=None):
    """Normalized empirical joint table, retaining every declared empty cell."""
    n_s, n_c, n_t = _size(n_s, "n_s"), _size(n_c, "n_c"), _size(n_t, "n_t")
    s, c, t = _indices(s, n_s, "s"), _indices(c, n_c, "c"), _indices(t, n_t, "t")
    if not (len(s) == len(c) == len(t)):
        raise ValueError("s, c, t must have the same length")
    w = _weights(weights, len(t))
    out = np.zeros((n_s, n_c, n_t), dtype=float)
    np.add.at(out, (s, c, t), w)
    return out


def cost_table(t, y, action_probabilities, n_t, weights=None):
    """Joint-mass-weighted binary cross-entropy for each state/action.

    ``action_probabilities[i,z]`` is the frozen prediction for observation i
    and action z. A one-dimensional action array is broadcast to all rows.
    Probabilities must be strictly between zero and one; any chosen clipping
    belongs to the declared decoder, not an implicit modification here.
    """
    n_t = _size(n_t, "n_t")
    t = _indices(t, n_t, "t")
    y = np.asarray(y, dtype=float)
    if y.shape != t.shape or not np.all(np.isfinite(y)) or np.any((y < 0) | (y > 1)):
        raise ValueError("y must match t and lie in [0,1]")
    predictions = np.asarray(action_probabilities, dtype=float)
    if predictions.ndim == 1:
        predictions = np.broadcast_to(predictions, (len(t), len(predictions)))
    if predictions.ndim != 2 or predictions.shape[0] != len(t) or predictions.shape[1] < 1:
        raise ValueError("action_probabilities must have shape (n_observations,n_actions)")
    if not np.all(np.isfinite(predictions)) or np.any((predictions <= 0) | (predictions >= 1)):
        raise ValueError("action probabilities must lie strictly between zero and one")
    w = _weights(weights, len(t))
    losses = -y[:, None] * np.log(predictions) - (1-y[:, None]) * np.log1p(-predictions)
    out = np.zeros((n_t, predictions.shape[1]), dtype=float)
    np.add.at(out, t, w[:, None] * losses)
    return out


def _law(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 3 or min(p.shape) < 1:
        raise ValueError("joint law must have nonempty shape (n_s,n_c,n_t)")
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("joint law must be finite and nonnegative")
    if not np.isclose(p.sum(), 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError("joint law must be normalized")
    return p


def cmi(p, q):
    """Compute I(S;Z|C) in nats independently of the optimizer.

    Uses the closed convention 0*log(0/b)=0, including b=0. No smoothing,
    pseudo-counts, or positive floor is introduced into the joint law.
    """
    p = _law(p)
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[0] != p.shape[2] or q.shape[1] < 1:
        raise ValueError("channel shape does not match joint law")
    if not np.all(np.isfinite(q)) or np.any(q < 0):
        raise ValueError("channel must be finite and nonnegative")
    if not np.allclose(q.sum(axis=1), 1, atol=1e-8, rtol=0):
        raise ValueError("channel must have stochastic rows")
    joint = np.einsum("sct,tz->scz", p, q)
    p_sc = p.sum(axis=2)
    p_c = p_sc.sum(axis=0)
    p_s_given_c = np.divide(p_sc, p_c[None, :], out=np.zeros_like(p_sc), where=p_c[None, :] > 0)
    independent = p_s_given_c[:, :, None] * joint.sum(axis=0)[None, :, :]
    value = float(rel_entr(joint, independent).sum())
    return max(0.0, value)  # Only removes negative roundoff in a nonnegative quantity.


def privacy_matrix(p):
    """A with rows (s,c) such that A@Q=0 iff I(S;Z|C)=0.

    A[s,c,t] = p(s,c,t) - p(s|c)*p(c,t), flattened over (s,c).
    Empty sensitive/context cells contribute zero rows. A whole row whose
    difference is at floating-point cancellation scale is represented by zero
    before any equation scaling. Independent CMI always uses the original law.
    """
    p = _law(p)
    p_sc, p_ct = p.sum(axis=2), p.sum(axis=0)
    p_c = p_ct.sum(axis=1)
    conditional = np.divide(p_sc, p_c[None, :], out=np.zeros_like(p_sc), where=p_c[None, :] > 0)
    independent = conditional[:, :, None] * p_ct[None, :, :]
    difference = p - independent
    scale = np.max(np.abs(p), axis=2) + np.max(np.abs(independent), axis=2)
    cancellation = np.max(np.abs(difference), axis=2) <= ACCEPTANCE_TOLERANCES["privacy_row_roundoff_relative"] * scale
    difference[cancellation] = 0.0
    return difference.reshape(-1, p.shape[2])


def _support(n_t, roles, state_mass, parent, cost):
    if state_mass is None:
        mass = next(iter(roles.values())).sum(axis=(0, 1)) if roles else np.ones(n_t)
    else:
        mass = np.asarray(state_mass, dtype=float)
    if mass.shape != (n_t,) or not np.all(np.isfinite(mass)) or np.any(mass < 0):
        raise ValueError("state_mass must be a finite nonnegative vector")
    if parent is None:
        parents = np.arange(n_t)
    else:
        raw = np.asarray(parent)
        if raw.shape != (n_t,) or not np.issubdtype(raw.dtype, np.number) or not np.all(np.isfinite(raw)):
            raise ValueError("parent must be a finite integer vector with one entry per state")
        if np.any(raw < 0) or not np.all(raw == np.floor(raw)):
            raise ValueError("parent IDs must be nonnegative integers")
        parents = raw.astype(np.int64)
    missing = mass == 0
    if np.any(np.abs(cost[missing]) > 1e-14):
        raise ValueError("unsupported states have nonzero objective mass")
    if any(np.any(p.sum(axis=(0, 1))[missing] > 1e-14) for p in roles.values()):
        raise ValueError("state_mass marks a privacy-supported state as unsupported")
    ties, absent = {}, []
    for t in np.flatnonzero(missing):
        siblings = np.flatnonzero((parents == parents[t]) & ~missing)
        if len(siblings):
            weights = mass[siblings] / mass[siblings].sum()
            ties[int(t)] = (siblings, weights)
        else:
            absent.append(int(t))
    report = {
        "n_states": n_t,
        "supported_states": np.flatnonzero(~missing).tolist(),
        "unsupported_states": np.flatnonzero(missing).tolist(),
        "absent_parent_ids": sorted(set(parents[absent].tolist())),
        "state_mass": mass.tolist(),
        "parent": parents.tolist(),
        "ties": {str(t): {"siblings": ids.tolist(), "weights": w.tolist()} for t, (ids, w) in ties.items()},
        "default_mass_source": "first_role" if state_mass is None and roles else ("all_states" if state_mass is None else "supplied"),
    }
    return parents, ties, absent, report


def _support_equations(n_t, n_z, ties, absent, zero_action):
    rows, cols, data, rhs = [], [], [], []
    for t, (siblings, weights) in ties.items():
        for z in range(n_z):
            row = len(rhs)
            rows.append(row); cols.append(t*n_z+z); data.append(1.0)
            for sibling, weight in zip(siblings, weights):
                rows.append(row); cols.append(int(sibling)*n_z+z); data.append(-float(weight))
            rhs.append(0.0)
    for t in absent:
        for z in range(n_z):
            rows.append(len(rhs)); cols.append(t*n_z+z); data.append(1.0)
            rhs.append(float(z == zero_action))
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(rhs), n_t*n_z))
    return matrix, np.asarray(rhs)


def _clean_channel(raw, ties, absent, zero_action):
    """Remove only numerical simplex noise; all changes are audited afterward."""
    if raw is None:
        return None, float("inf")
    raw = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(raw)) or np.min(raw) < -ACCEPTANCE_TOLERANCES["nonnegative"]:
        return None, float("inf")
    q = np.maximum(raw, 0)
    mass = q.sum(axis=1)
    if np.any(mass <= 0) or np.max(np.abs(mass-1)) > ACCEPTANCE_TOLERANCES["simplex"]:
        return None, float("inf")
    q = q / mass[:, None]
    for t, (siblings, weights) in ties.items():
        q[t] = weights @ q[siblings]
    for t in absent:
        q[t] = 0
        q[t, zero_action] = 1
    return q, float(np.max(np.abs(q-raw)))


def _evaluate(q, roles, budget, matrices, normalized_matrix, support_eq, support_rhs, cost, witness_objective=None):
    if q is None or not np.all(np.isfinite(q)):
        return {"feasible": False, "reason": "no_finite_candidate"}
    residuals = {
        "simplex": float(np.max(np.abs(q.sum(axis=1)-1))),
        "nonnegative": float(max(0.0, -np.min(q))),
        "support": float(np.max(np.abs(support_eq @ q.ravel()-support_rhs), initial=0.0)),
        "zero_linear": float(np.max(np.abs(normalized_matrix @ q), initial=0.0)),
        "zero_linear_unscaled": {name: float(np.max(np.abs(a @ q), initial=0.0)) for name, a in matrices.items()},
    }
    try:
        leaks = {name: cmi(p, q) for name, p in roles.items()}
    except ValueError:
        return {"feasible": False, "reason": "invalid_channel", **residuals}
    residuals["cmi"] = leaks
    residuals["cmi_excess"] = {name: max(0.0, value-budget) for name, value in leaks.items()} if budget is not None else {}
    objective = float(np.sum(cost*q))
    residuals["witness_objective_excess"] = max(0.0, objective-witness_objective) if witness_objective is not None else 0.0
    feasible = (
        residuals["simplex"] <= ACCEPTANCE_TOLERANCES["simplex"]
        and residuals["nonnegative"] <= ACCEPTANCE_TOLERANCES["nonnegative"]
        and residuals["support"] <= ACCEPTANCE_TOLERANCES["support"]
        and all(value <= ACCEPTANCE_TOLERANCES["cmi"] for value in residuals["cmi_excess"].values())
        and (budget != 0 or residuals["zero_linear"] <= ACCEPTANCE_TOLERANCES["zero_linear"])
    )
    return {"feasible": bool(feasible), "objective": objective, **residuals}


def solve(cost, roles: Mapping[str, np.ndarray], budget: float | None,
          parent: np.ndarray | None = None, state_mass: np.ndarray | None = None,
          zero_action: int = 0, embedded_q=None):
    """Minimize a frozen joint-mass cost with every supplied privacy constraint.

    ``roles`` can contain both U/PWGTP laws and any local/coalition views; no
    role is dropped. ``None`` removes privacy constraints, while ``0`` uses
    exact conditional-independence equations. Supply unweighted estimation
    frequencies in ``state_mass`` for the prescribed unsupported-child rule.
    If omitted, the first role's marginal is used (all states when roles is
    empty). ``parent[t]`` identifies the coarse cell; absent coarse cells use
    ``zero_action``. A verified ``embedded_q`` can have fine rows or coarse
    rows indexed by parent; it supplies an objective upper-bound witness.

    ``status`` retains ``optimal_inaccurate``. A feasible witness returned
    after solver failure has status ``feasible_witness``, never ``optimal``.
    ``feasible`` is numerical empirical feasibility at the reported fixed
    tolerances, not a population privacy certificate.
    """
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or min(cost.shape) < 1 or not np.all(np.isfinite(cost)):
        raise ValueError("cost must be a finite nonempty state-by-action matrix")
    n_t, n_z = cost.shape
    try:
        zero_action = operator.index(zero_action)
    except TypeError as exc:
        raise ValueError("zero_action must be an action index") from exc
    if not 0 <= zero_action < n_z:
        raise ValueError("zero_action is outside the action alphabet")
    if budget is not None:
        budget = float(budget)
        if not np.isfinite(budget) or budget < 0:
            raise ValueError("budget must be nonnegative and finite, or None")
    if not isinstance(roles, Mapping):
        raise ValueError("roles must map names to joint laws")
    roles = {str(name): _law(p) for name, p in roles.items()}
    if any(p.shape[2] != n_t for p in roles.values()):
        raise ValueError("all roles must use the same input-state alphabet as cost")
    parents, ties, absent, support = _support(n_t, roles, state_mass, parent, cost)
    support_eq, support_rhs = _support_equations(n_t, n_z, ties, absent, zero_action)
    matrices = {name: privacy_matrix(p) for name, p in roles.items()}
    stacked = np.vstack(list(matrices.values())) if matrices else np.zeros((0, n_t))
    scales = np.max(np.abs(stacked), axis=1, initial=0)
    nonzero = scales > 0
    # Row scaling preserves every nonzero equality and protects rare contexts
    # against being ignored merely because they have a small absolute mass.
    normalized = stacked[nonzero] / scales[nonzero, None]
    rank = int(np.linalg.matrix_rank(stacked)) if stacked.size else 0
    ranks = {name: int(np.linalg.matrix_rank(a)) for name, a in matrices.items()}
    witness, witness_details, witness_objective = None, None, None
    if embedded_q is not None:
        witness = np.asarray(embedded_q, dtype=float)
        if witness.shape != (n_t, n_z):
            if witness.ndim == 2 and witness.shape[1] == n_z and witness.shape[0] > int(parents.max()):
                witness = witness[parents]
            else:
                raise ValueError("embedded witness has incompatible shape")
        witness_details = _evaluate(witness, roles, budget, matrices, normalized, support_eq, support_rhs, cost)
        if not witness_details["feasible"]:
            raise ValueError("embedded witness fails independent feasibility checks")
        witness_objective = witness_details["objective"]

    attempts, candidates = [], []

    def record(raw, solver, status, started, reported_objective=None, error=None, warning_messages=()):
        q, repair = _clean_channel(raw, ties, absent, zero_action)
        residuals = _evaluate(q, roles, budget, matrices, normalized, support_eq, support_rhs, cost, witness_objective)
        accepted = (
            residuals["feasible"] and repair <= ACCEPTANCE_TOLERANCES["repair"]
            and residuals["witness_objective_excess"] <= ACCEPTANCE_TOLERANCES["witness_objective"]
            and status in {"optimal", "optimal_inaccurate"}
        )
        attempt = {
            "solver": solver, "status": status, "accepted": bool(accepted),
            "seconds": time.perf_counter()-started,
            "reported_objective": float(reported_objective) if reported_objective is not None and np.isfinite(reported_objective) else None,
            "repair_max": repair if np.isfinite(repair) else None,
            "residuals": residuals, "warnings": list(warning_messages),
        }
        if raw is not None:
            raw_simplex = float(np.max(np.abs(np.asarray(raw).sum(axis=1)-1)))
            raw_minimum = float(np.min(raw))
            attempt["raw_simplex"] = raw_simplex if np.isfinite(raw_simplex) else None
            attempt["raw_minimum"] = raw_minimum if np.isfinite(raw_minimum) else None
        if error is not None:
            attempt["error"] = str(error)
        attempts.append(attempt)
        if accepted:
            candidates.append((q, status, solver, residuals))
        return accepted and status == "optimal"

    if budget is None or budget == 0 or not roles:
        stochastic = sparse.kron(sparse.eye(n_t), np.ones((1, n_z)), format="csr")
        eq_parts, rhs_parts = [stochastic, support_eq], [np.ones(n_t), support_rhs]
        if budget == 0 and len(normalized):
            eq_parts.append(sparse.kron(sparse.csr_matrix(normalized), sparse.eye(n_z), format="csr"))
            rhs_parts.append(np.zeros(len(normalized)*n_z))
        started = time.perf_counter()
        try:
            answer = linprog(
                cost.ravel(), A_eq=sparse.vstack(eq_parts, format="csr"), b_eq=np.concatenate(rhs_parts),
                A_ub=sparse.csr_matrix(cost.ravel()[None, :]) if witness is not None else None,
                b_ub=np.array([witness_objective]) if witness is not None else None,
                bounds=(0, None), method="highs", options=dict(SOLVER_SETTINGS["HIGHS"]),
            )
            raw = answer.x.reshape(n_t, n_z) if answer.x is not None else None
            status = "optimal" if answer.success else {1: "iteration_limit", 2: "infeasible", 3: "unbounded"}.get(answer.status, "solver_error")
            record(raw, "HIGHS", status, started, answer.fun, None if answer.success else answer.message)
        except Exception as exc:
            record(None, "HIGHS", "solver_error", started, error=exc)
    else:
        import cvxpy as cp

        q_variable = cp.Variable((n_t, n_z), nonneg=True)
        constraints = [cp.sum(q_variable, axis=1) == 1]
        if len(support_rhs):
            constraints.append(support_eq @ cp.reshape(q_variable, (n_t*n_z,), order="C") == support_rhs)
        for p in roles.values():
            p_sc, p_ct = p.sum(axis=2), p.sum(axis=0)
            p_c = p_ct.sum(axis=1)
            conditional = np.divide(p_sc, p_c[None, :], out=np.zeros_like(p_sc), where=p_c[None, :] > 0)
            a_coeff = p.reshape(-1, n_t)
            b_coeff = (conditional[:, :, None] * p_ct[None, :, :]).reshape(-1, n_t)
            supported = p_sc.ravel() > 0
            a = a_coeff[supported] @ q_variable
            b = b_coeff[supported] @ q_variable
            constraints.append(cp.sum(cp.rel_entr(a, b)) <= budget)
        objective = cp.sum(cp.multiply(cost, q_variable))
        if witness is not None:
            constraints.append(objective <= witness_objective)
        problem = cp.Problem(cp.Minimize(objective), constraints)
        if not problem.is_dcp():
            raise RuntimeError("finite channel problem unexpectedly violates convexity rules")
        for solver in ("CLARABEL", "SCS"):
            started = time.perf_counter()
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    value = problem.solve(solver=solver, warm_start=False, verbose=False, **SOLVER_SETTINGS[solver])
                complete = record(q_variable.value, solver, str(problem.status), started, value,
                                  warning_messages=[str(w.message) for w in caught])
                if complete:
                    break
            except Exception as exc:
                record(None, solver, "solver_error", started, error=exc)

    if candidates:
        q, status, solver, residuals = min(candidates, key=lambda result: (result[1] != "optimal", result[3]["objective"]))
    elif witness is not None:
        q, status, solver, residuals = witness.copy(), "feasible_witness", "embedded", witness_details
    else:
        q, status, solver, residuals = None, "failed", None, {"feasible": False, "reason": "no_accepted_solver_candidate"}
    return {
        "Q": q, "objective": residuals.get("objective"), "status": status, "solver": solver,
        "feasible": residuals["feasible"], "optimal": status == "optimal",
        "residuals": residuals, "rank": rank, "role_ranks": ranks, "support": support,
        "attempts": attempts, "witness": witness_details,
        "tolerances": dict(ACCEPTANCE_TOLERANCES), "budget": budget, "information_units": "nats",
    }
