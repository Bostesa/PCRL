"""Finite-channel LP, phase I and immutable fixed-attack cut exchange.

P1 cuts use expected attacker *loss* floors: <A,Q> >= rho - delta.
These are fitted-risk constraints, not population privacy guarantees.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .method import rowwise_unconstrained, validate_channel

SIMPLEX_TOL = 1e-7
NONNEGATIVE_TOL = 1e-8
PRIMAL_TOL = 1e-7
EXCHANGE_VIOLATION_TOL = 1e-5
HIGHS_OPTIONS = {"primal_feasibility_tolerance": 1e-9,
                 "dual_feasibility_tolerance": 1e-9}


def _cost(cost):
    c = np.asarray(cost, dtype=np.float64)
    if c.ndim != 2 or min(c.shape) < 1 or not np.isfinite(c).all():
        raise ValueError("cost must be finite with shape (states,tokens)")
    return c


def _cuts(cuts: Sequence[Mapping], shape):
    if isinstance(cuts, (str, bytes)):
        raise ValueError("cuts must be a sequence of mappings")
    parsed, ids = [], set()
    for item in cuts:
        if not isinstance(item, Mapping):
            raise ValueError("every cut must be a mapping")
        cid = item.get("id")
        if not isinstance(cid, str) or not cid or cid in ids:
            raise ValueError("cut IDs must be unique nonempty strings")
        ids.add(cid)
        a = np.asarray(item.get("coeff"), dtype=np.float64)
        floor = float(item.get("floor"))
        if a.shape != shape or not np.isfinite(a).all() or not np.isfinite(floor):
            raise ValueError("cut coefficient/floor shape or finiteness mismatch")
        if "rho" in item or "delta" in item:
            if "rho" not in item or "delta" not in item:
                raise ValueError("rho and delta must be supplied together")
            rho, delta = float(item["rho"]), float(item["delta"])
            if not np.isfinite(rho) or not np.isfinite(delta) or abs(rho-delta-floor) > 1e-10:
                raise ValueError("cut floor must equal rho - delta")
        metadata = {k: v for k, v in item.items() if k != "coeff"}
        json.dumps(metadata, sort_keys=True, allow_nan=False)
        parsed.append({**metadata, "coeff": a, "floor": floor})
    return parsed


def bank_sha256(cuts, shape):
    """Hash exact coefficients, floors and provenance; order-independent by ID."""
    parsed = _cuts(cuts, shape)
    digest = hashlib.sha256()
    digest.update(b"P1_LOSS_FLOOR_BANK_V1")
    digest.update(np.asarray(shape, dtype="<i8").tobytes())
    for cut in sorted(parsed, key=lambda item: item["id"]):
        meta = {k: v for k, v in cut.items() if k != "coeff"}
        digest.update(json.dumps(meta, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        digest.update(np.ascontiguousarray(cut["coeff"], dtype="<f8").tobytes())
    return digest.hexdigest()


def replay_p1(q, cost, cuts):
    """Independent primal replay from stored matrices and a stored channel."""
    c = _cost(cost)
    parsed = _cuts(cuts, c.shape)
    q = validate_channel(q, n_states=c.shape[0], n_tokens=c.shape[1], tolerance=SIMPLEX_TOL)
    losses = {cut["id"]: float(np.sum(cut["coeff"] * q)) for cut in parsed}
    slacks = {cut["id"]: losses[cut["id"]] - cut["floor"] for cut in parsed}
    return {"objective": float(np.sum(c*q)), "losses": losses, "slacks": slacks,
            "maximum_cut_violation": float(max((max(0., -v) for v in slacks.values()), default=0.)),
            "simplex_residual": float(np.max(np.abs(q.sum(axis=1)-1))),
            "minimum_entry": float(np.min(q)), "bank_sha256": bank_sha256(parsed, c.shape)}


def constant_maps(cost, cuts):
    """Check all constant token maps before attempting any privacy bank."""
    c = _cost(cost)
    parsed = _cuts(cuts, c.shape)
    answer = []
    for z in range(c.shape[1]):
        losses = {cut["id"]: float(cut["coeff"][:, z].sum()) for cut in parsed}
        slacks = {cut["id"]: losses[cut["id"]]-cut["floor"] for cut in parsed}
        answer.append({"token": z, "objective": float(c[:, z].sum()),
                       "minimum_slack": float(min(slacks.values())) if slacks else None,
                       "maximum_cut_violation": float(max((max(0., -s) for s in slacks.values()), default=0.)),
                       "feasible": all(s >= -PRIMAL_TOL for s in slacks.values())})
    return answer


def _simplex_matrix(n_states, n_tokens, extra=0):
    base = sparse.kron(sparse.eye(n_states, format="csr"),
                       np.ones((1, n_tokens)), format="csr")
    return sparse.hstack((base, sparse.csr_matrix((n_states, extra))), format="csr") if extra else base


def _clean(raw, shape):
    if raw is None:
        return None, None
    raw = np.asarray(raw, dtype=np.float64).reshape(shape)
    if not np.isfinite(raw).all() or raw.min() < -NONNEGATIVE_TOL:
        return None, None
    q = np.maximum(raw, 0.)
    mass = q.sum(axis=1)
    if np.any(mass <= 0) or np.max(np.abs(mass-1)) > SIMPLEX_TOL:
        return None, None
    q /= mass[:, None]
    return q, float(np.max(np.abs(q-raw)))


def phase_one_p1(cost, cuts, *, time_limit=None):
    """Minimize a common nonnegative relaxation of all attack-loss floors."""
    c = _cost(cost)
    parsed = _cuts(cuts, c.shape)
    n_states, n_tokens = c.shape
    n = c.size
    row = np.zeros(n+1)
    row[-1] = 1.
    a_ub = np.vstack([np.r_[-cut["coeff"].ravel(), -1.] for cut in parsed]) if parsed else None
    b_ub = -np.asarray([cut["floor"] for cut in parsed]) if parsed else None
    options = dict(HIGHS_OPTIONS)
    if time_limit is not None:
        options["time_limit"] = float(time_limit)
    result = linprog(row, A_ub=a_ub, b_ub=b_ub,
                     A_eq=_simplex_matrix(n_states, n_tokens, 1), b_eq=np.ones(n_states),
                     bounds=[(0, None)]*(n+1), method="highs", options=options)
    q, repair = _clean(result.x[:n], c.shape) if result.x is not None else (None, None)
    violation = None
    if q is not None:
        replay = replay_p1(q, c, parsed)
        violation = replay["maximum_cut_violation"]
    return {"Q": q, "status": "optimal" if result.success else
            {1: "time_or_iteration_limit", 2: "infeasible", 3: "unbounded"}.get(result.status, "solver_error"),
            "message": result.message, "minimum_common_violation": float(result.fun) if result.fun is not None else None,
            "replayed_maximum_violation": violation, "repair_max": repair,
            "solver": "HiGHS", "bank_sha256": bank_sha256(parsed, c.shape)}


def solve_p1(cost, cuts, *, time_limit=None):
    """Solve one fixed-bank affine P1 programme after constants and phase I.

    No infeasible bank is silently relaxed.  Call ``relax_cuts`` explicitly for
    the separately labeled descriptive phase-I fallback.
    """
    c = _cost(cost)
    parsed = _cuts(cuts, c.shape)
    constants = constant_maps(c, parsed)
    phase = phase_one_p1(c, parsed, time_limit=time_limit)
    phase_record = {key: value for key, value in phase.items() if key != "Q"}
    base = {"constant_maps": constants, "phase_one": phase_record,
            "bank_sha256": bank_sha256(parsed, c.shape)}
    if phase["status"] != "optimal" or phase["minimum_common_violation"] is None:
        return {**base, "Q": None, "status": "phase_one_unresolved", "feasible": False}
    if phase["minimum_common_violation"] > PRIMAL_TOL:
        return {**base, "Q": None, "status": "registered_bank_infeasible", "feasible": False}
    n_states, n_tokens = c.shape
    a_ub = np.vstack([-cut["coeff"].ravel() for cut in parsed]) if parsed else None
    b_ub = -np.asarray([cut["floor"] for cut in parsed]) if parsed else None
    a_eq = _simplex_matrix(n_states, n_tokens)
    b_eq = np.ones(n_states)
    options = dict(HIGHS_OPTIONS)
    if time_limit is not None:
        options["time_limit"] = float(time_limit)
    result = linprog(c.ravel(), A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                     bounds=(0, None), method="highs", options=options)
    q, repair = _clean(result.x, c.shape) if result.x is not None else (None, None)
    if q is None:
        return {**base, "Q": None, "status": "solver_failed", "feasible": False,
                "solver_status": result.status, "solver_message": result.message}
    replay = replay_p1(q, c, parsed)
    feasible = bool(replay["maximum_cut_violation"] <= PRIMAL_TOL and
                    replay["simplex_residual"] <= SIMPLEX_TOL and
                    replay["minimum_entry"] >= -NONNEGATIVE_TOL and repair <= SIMPLEX_TOL)
    lower = None
    dual = None
    if result.success:
        m_ub = np.minimum(np.asarray(result.ineqlin.marginals, dtype=float), 0.) if parsed else np.zeros(0)
        m_eq = np.asarray(result.eqlin.marginals, dtype=float)
        reduced = c.ravel() - a_eq.T @ m_eq
        if parsed:
            reduced -= a_ub.T @ m_ub
        minimum_reduced = float(np.min(reduced))
        lower = float(np.dot(m_eq, b_eq) +
                      (np.dot(m_ub, b_ub) if parsed else 0.) +
                      n_states*min(0., minimum_reduced))
        dual = {"lower_bound": lower, "minimum_reduced_cost": minimum_reduced,
                "maximum_positive_inequality_multiplier": float(max(0., np.max(m_ub, initial=0.))),
                "source": "reconstructed HiGHS dual marginals with reduced-cost guard",
                "not_interval_arithmetic": True}
    status = ("optimal" if result.success and feasible else
              ("infeasible_replay" if result.success else "solver_unresolved"))
    return {**base, "Q": q, "status": status, "feasible": feasible and result.success,
            "solver": "HiGHS", "solver_status": result.status,
            "solver_message": result.message, "solver_reported_objective": result.fun,
            "objective": replay["objective"], "replay": replay, "repair_max": repair,
            "dual": dual, "dual_lower_bound": lower,
            "fixed_bank_gap": replay["objective"]-lower if lower is not None else None}


def relax_cuts(cuts, amount, shape):
    """Explicit descriptive fallback, lowering every attack-loss floor equally."""
    amount = float(amount)
    if not np.isfinite(amount) or amount < 0:
        raise ValueError("relaxation must be finite and nonnegative")
    out = []
    for cut in _cuts(cuts, shape):
        c = dict(cut)
        c["floor"] -= amount
        c.pop("rho", None)
        c.pop("delta", None)
        c["descriptive_phase_one_relaxation"] = amount
        out.append(c)
    return out


def add_violated_cuts(existing, proposals, channels: Mapping[str, np.ndarray], shape,
                      *, tolerance=EXCHANGE_VIOLATION_TOL):
    """Retain old bank; add only new attacks violating an LP or MILP candidate."""
    current = _cuts(existing, shape)
    proposed = _cuts(proposals, shape)
    seen = {cut["id"]: bank_sha256([cut], shape) for cut in current}
    checked = {name: validate_channel(q, n_states=shape[0], n_tokens=shape[1])
               for name, q in channels.items()}
    added, log = [], []
    for cut in proposed:
        cid = cut["id"]
        fingerprint = bank_sha256([cut], shape)
        if cid in seen:
            if seen[cid] != fingerprint:
                raise ValueError("same cut ID has different coefficient/provenance")
            continue
        violations = {name: float(cut["floor"]-np.sum(cut["coeff"]*q))
                      for name, q in checked.items()}
        add = any(value > tolerance for value in violations.values())
        log.append({"id": cid, "violations": violations, "added": add})
        if add:
            current.append(cut)
            added.append(cid)
            seen[cid] = fingerprint
    return {"cuts": current, "added_ids": added, "oracle_log": log,
            "bank_sha256": bank_sha256(current, shape)}


def solve_p0(cost, roles, budget, *, state_mass=None, parent=None, zero_action=0):
    """Use the historical registered finite-CMI primitive for P0 arms."""
    from experiments.pcrl_task_directed_release_v1 import finite
    return finite.solve(cost, roles, budget, parent=parent,
                        state_mass=state_mass, zero_action=zero_action)
