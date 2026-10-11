"""Finite-map builders and bounded solvers for registered matched controls.

The product and local kernels are bilinear families; only a coordinate with
the other factor frozen is affine. Deterministic fitting uses binary weights,
never a rounded stochastic fit. This module loads no scientific role file.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import sparse


def _groups(group):
    group = np.asarray(group)
    if group.dtype.kind not in "iu" or group.size == 0 or np.any(group < 0):
        raise ValueError("groups must be nonempty nonnegative integer indices")
    unique = np.unique(group)
    if not np.array_equal(unique, np.arange(int(unique[-1]) + 1)):
        raise ValueError("group indices must be dense")
    return group.astype(np.int64, copy=False), len(unique)


def _factor(probability, width):
    probability = np.asarray(probability, dtype=np.float64)
    if (probability.ndim != 2 or probability.shape[1] != width
            or not np.isfinite(probability).all() or np.any(probability < 0)
            or not np.allclose(probability.sum(1), 1, rtol=0, atol=1e-12)):
        raise ValueError("frozen factor is not a finite normalized law")
    return probability


def product_map(group, frozen_other, fitted_recipient, width=4):
    """CSR mapping of variable marginal weights to an independent joint law."""
    group, count = _groups(group)
    frozen_other = _factor(frozen_other, width)
    if group.ndim != 1 or len(group) != len(frozen_other) or fitted_recipient not in (1, 2):
        raise ValueError("product coordinate contract differs")
    n = len(group)
    row, a, b = np.indices((n, width, width))
    coordinate = a if fitted_recipient == 1 else b
    fixed_coordinate = b if fitted_recipient == 1 else a
    values = frozen_other[row, fixed_coordinate]
    matrix = sparse.coo_matrix((values.ravel(),
        ((row * width * width + a * width + b).ravel(),
         (group[row] * width + coordinate).ravel())),
        shape=(n * width * width, count * width)).tocsr()
    matrix.eliminate_zeros()
    return matrix


def sequential_map(frozen_first, second_group, first_recipient, width=4):
    """Second-stage affine map, grouped by public emitted-first code.

    ``second_group[i,a]`` must be computed from input context and the actual
    emitted first-recipient global code, so aliased slots receive one group.
    No fine action position is a deployment input or transmitted field.
    """
    frozen_first = _factor(frozen_first, width)
    second_group, count = _groups(second_group)
    n = len(frozen_first)
    if second_group.shape != (n, width) or first_recipient not in (1, 2):
        raise ValueError("sequential coordinate contract differs")
    row, a, b = np.indices((n, width, width))
    first_action = a if first_recipient == 1 else b
    second_action = b if first_recipient == 1 else a
    values = frozen_first[row, first_action]
    return sparse.coo_matrix((values.ravel(),
        ((row * width * width + a * width + b).ravel(),
         (second_group[row, first_action] * width + second_action).ravel())),
        shape=(n * width * width, count * width)).tocsr()


def categorical_map(group, actions=16):
    """Context-shared joint categorical kernel; integrality is a solver option."""
    group, count = _groups(group)
    if group.ndim != 1 or actions < 1:
        raise ValueError("categorical map contract differs")
    row, action = np.indices((len(group), actions))
    return sparse.coo_matrix((np.ones(row.size),
        ((row * actions + action).ravel(),
         (group[row] * actions + action).ravel())),
        shape=(len(group) * actions, count * actions)).tocsr()


def simplex_rows(groups, width):
    """Two <= rows per simplex, expressing exact normalization."""
    if groups < 1 or width < 1:
        raise ValueError("empty simplex")
    index = np.arange(groups * width)
    positive = sparse.coo_matrix((np.ones(len(index)), (index // width, index)),
        shape=(groups, groups * width)).tocsr()
    return sparse.vstack((positive, -positive), format="csr"), np.r_[np.ones(groups), -np.ones(groups)]


def template_bounds(group, valid):
    """Only templates valid for every row in a shared group can receive mass."""
    group, count = _groups(group)
    valid = np.asarray(valid)
    if valid.dtype != bool or valid.ndim != 2 or group.shape != (len(valid),):
        raise ValueError("grouped support contract differs")
    allowed = np.ones((count, valid.shape[1]), bool)
    np.logical_and.at(allowed, group, valid)
    if not allowed.any(1).all():
        raise ValueError("shared group has no executable action")
    return np.c_[np.zeros(allowed.size), allowed.ravel().astype(np.float64)]


@dataclass(frozen=True)
class ControlCeilings:
    """Bounds must be frozen in the campaign lock, not changed after outcomes."""
    starts: int = 2
    rounds: int = 4
    seconds: float = 600.
    feasibility_tolerance: float = 1e-8
    relative_mip_gap: float = 1e-6


def append_simplexes(rows, count, width):
    from .lp import ConstraintRows
    matrix, rhs = simplex_rows(count, width)
    names = tuple(f"simplex/{direction}/{g}" for direction in ("upper", "lower") for g in range(count))
    return ConstraintRows(tuple(rows.names) + names,
        sparse.vstack((rows.matrix, matrix), format="csr"), np.r_[rows.rhs, rhs])


def map_law(emitted1, emitted2, identity1, identity2, mapping, bounds, weights=None, width=4):
    """Create the shared R2 affine law without fitting or reading any labels."""
    from .lp import AffineLaw
    if mapping.shape[1] % width:
        raise ValueError("mapping parameter width differs")
    eq, rhs = simplex_rows(mapping.shape[1] // width, width)
    count = len(rhs) // 2
    return AffineLaw(emitted1, emitted2, identity1, identity2,
        np.zeros(np.asarray(emitted1).shape, np.float64), mapping, bounds, weights,
        eq[:count], rhs[:count])


def initial_factor(bounds, mode, width=4):
    """Two predeclared starts: exact KEEP and uniform executable templates."""
    bounds = np.asarray(bounds, dtype=np.float64).reshape(-1, width, 2)
    allowed = bounds[:, :, 1] > 0
    if mode == "keep":
        if not allowed[:, 0].all():
            raise ValueError("KEEP is not executable in each group")
        answer = np.zeros(allowed.shape, np.float64)
        answer[:, 0] = 1.
    elif mode == "uniform":
        answer = allowed / allowed.sum(1, keepdims=True)
    else:
        raise ValueError("unregistered factor start")
    return answer


def fit_independent_product(emitted1, emitted2, identity1, identity2, groups1, groups2,
                            valid1, valid2, sensitive, constraints_builder,
                            ceilings=ControlCeilings(), weights=None):
    """Bounded alternating LPs, not a joint LP or global bilinear certificate.

    Local controls pass groups determined only by each own P0 input. Central
    product controls pass the same input-context groups for both recipients.
    Each failed frozen-factor subproblem is recorded; no global family
    infeasibility claim follows. No outcome-dependent restart is added.
    """
    import time
    from .lp import solve_protection, direct_bayes
    groups1, _ = _groups(groups1)
    groups2, _ = _groups(groups2)
    bounds = (template_bounds(groups1, valid1), template_bounds(groups2, valid2))
    if ceilings.starts != 2:
        raise ValueError("exactly the two registered starts are required")
    begun = time.monotonic()
    attempts, best = [], None
    for start in ("keep", "uniform"):
        factor = [initial_factor(bounds[0], start), initial_factor(bounds[1], start)]
        stopped = False
        for iteration in range(ceilings.rounds):
            for recipient in (1, 2):
                remaining = ceilings.seconds - (time.monotonic() - begun)
                if remaining <= 0:
                    stopped = True
                    break
                own = recipient - 1
                other = 1 - own
                group = (groups1, groups2)[own]
                other_group = (groups1, groups2)[other]
                mapping = product_map(group, factor[other][other_group], recipient)
                law = map_law(emitted1, emitted2, identity1, identity2, mapping, bounds[own], weights)
                rows = constraints_builder(law)
                result = solve_protection(law, sensitive, rows, time_limit=remaining,
                                          tolerance=ceilings.feasibility_tolerance)
                attempts.append({"start": start, "round": iteration, "recipient": recipient,
                                 "result": result})
                if not result["success"]:
                    stopped = True
                    break
                factor[own] = result["theta"].reshape(-1, 4)
                objective, views = direct_bayes(law, result["theta"], sensitive)
                if best is None or objective < best["objective"] - 1e-12:
                    best = {"objective": objective, "views": views, "factors": [v.copy() for v in factor],
                            "law": law, "theta": result["theta"].copy(), "result": result,
                            "start": start, "round": iteration, "recipient": recipient}
            if stopped:
                break
    return {"success": best is not None, "selected": best, "attempts": attempts,
            "solver_class": "bounded alternating coordinate LPs for a bilinear family",
            "global_optimality": "UNRESOLVED", "global_bilinear_gap": None,
            "elapsed_seconds": time.monotonic() - begun}


def solve_binary_control(law, sensitive, constraints, ceilings=ControlCeilings()):
    """Fit binary context/action weights with a genuine mixed-integer solver."""
    from scipy.optimize import milp, Bounds, LinearConstraint
    from .lp import protection_program, solve_program
    program = protection_program(law, sensitive, constraints)
    # The continuous relaxation is a lower bound for this exact MIP family.
    # It is separately counted by the caller, not reported as the MIP fit.
    relaxation = solve_program(program, time_limit=ceilings.seconds / 4,
                               tolerance=ceilings.feasibility_tolerance)
    linear = [LinearConstraint(program.inequality_matrix, -np.inf, program.inequality_rhs)]
    if len(program.equality_rhs):
        linear.append(LinearConstraint(program.equality_matrix, program.equality_rhs, program.equality_rhs))
    integrality = np.r_[np.ones(program.parameters, dtype=np.int32),
                       np.zeros(len(program.objective) - program.parameters, dtype=np.int32)]
    result = milp(program.objective, integrality=integrality,
                  bounds=Bounds(program.bounds[:, 0], program.bounds[:, 1]), constraints=linear,
                  options={"time_limit": ceilings.seconds * .75,
                           "mip_rel_gap": ceilings.relative_mip_gap})
    cert = {"scope": "registered binary context-action kernel and fixed empirical constraints",
            "solver_status": int(result.status), "solver_message": str(result.message),
            "optimality_certified": False, "feasible": False,
            "mip_gap": getattr(result, "mip_gap", None),
            "mip_dual_bound": getattr(result, "mip_dual_bound", None),
            "mip_nodes": getattr(result, "mip_node_count", None),
            "relaxation_objective": relaxation["certificate"].get("primal_objective"),
            "relaxation_certified": relaxation["success"]}
    theta = None
    if result.x is not None:
        x = np.asarray(result.x, np.float64)
        theta = x[:program.parameters]
        violation = max(0., float((program.inequality_matrix @ x - program.inequality_rhs).max(initial=0)),
                        float(np.abs(program.equality_matrix @ x - program.equality_rhs).max(initial=0)),
                        float((program.bounds[:, 0] - x).max(initial=0)),
                        float((x - program.bounds[:, 1]).max(initial=0)))
        integer_error = float(np.abs(theta - np.rint(theta)).max(initial=0))
        cert.update(maximum_primal_violation=violation, maximum_integer_error=integer_error,
                    primal_objective=float(program.objective @ x),
                    feasible=bool(max(violation, integer_error) <= ceilings.feasibility_tolerance))
        cert["optimality_certified"] = bool(cert["feasible"] and result.success
                and getattr(result, "mip_gap", np.inf) <= ceilings.relative_mip_gap)
    return {"success": cert["feasible"], "theta": theta,
            "primal": None if result.x is None else result.x.copy(), "certificate": cert,
            "program": program, "relaxation": relaxation, "solver_class": "binary MILP"}
