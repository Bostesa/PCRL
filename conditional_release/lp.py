"""Sparse affine finite-law optimization; no loaders or private file access.

All actions denote actual emitted global code tuples. Public metadata is
required to be a deterministic function of those tuples by the caller.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.optimize import linprog


@dataclass
class ConstraintRows:
    names: list[str]
    matrix: object
    rhs: np.ndarray

    def __post_init__(self):
        self.matrix = sparse.csr_matrix(self.matrix, dtype=np.float64)
        self.rhs = np.asarray(self.rhs, dtype=np.float64)
        if (self.matrix.shape[0] != len(self.names) or self.rhs.shape != (len(self.names),)
                or len(set(self.names)) != len(self.names)
                or not np.isfinite(self.matrix.data).all() or not np.isfinite(self.rhs).all()):
            raise ValueError("named constraint rows invalid")


@dataclass
class AffineLaw:
    emitted1: np.ndarray
    emitted2: np.ndarray
    identity1: np.ndarray
    identity2: np.ndarray
    base: np.ndarray
    delta: object
    bounds: object
    weights: object = None
    equality_matrix: object = None
    equality_rhs: object = None

    def __post_init__(self):
        self.emitted1 = np.asarray(self.emitted1)
        self.emitted2 = np.asarray(self.emitted2)
        self.identity1 = np.asarray(self.identity1)
        self.identity2 = np.asarray(self.identity2)
        self.base = np.asarray(self.base, dtype=np.float64)
        if (self.emitted1.dtype != np.int64 or self.emitted2.dtype != np.int64
                or self.emitted1.ndim != 2 or self.emitted2.shape != self.emitted1.shape
                or self.base.shape != self.emitted1.shape
                or self.identity1.dtype != np.int64 or self.identity2.dtype != np.int64
                or self.identity1.shape != (len(self.base),) or self.identity2.shape != (len(self.base),)
                or not np.isfinite(self.base).all() or len(self.base) == 0
                or np.any(self.emitted1 < 0) or np.any(self.emitted2 < 0)):
            raise ValueError("affine law arrays invalid")
        self.delta = sparse.csr_matrix(self.delta, dtype=np.float64)
        if self.delta.shape[0] != self.base.size or not np.isfinite(self.delta.data).all():
            raise ValueError("affine derivative shape differs")
        self.bounds = np.asarray(self.bounds, dtype=np.float64)
        if (self.bounds.shape != (self.delta.shape[1], 2) or np.isnan(self.bounds).any()
                or np.isposinf(self.bounds[:, 0]).any() or np.isneginf(self.bounds[:, 1]).any()
                or np.any(self.bounds[:, 0] > self.bounds[:, 1])):
            raise ValueError("parameter bounds invalid")
        self.weights = np.ones(len(self.base)) / len(self.base) if self.weights is None else np.asarray(self.weights, dtype=np.float64)
        if self.weights.shape != (len(self.base),) or np.any(self.weights < 0) or not np.isclose(self.weights.sum(), 1):
            raise ValueError("empirical probability weights invalid")
        p = self.delta.shape[1]
        self.equality_matrix = sparse.csr_matrix((0, p)) if self.equality_matrix is None else sparse.csr_matrix(self.equality_matrix, dtype=np.float64)
        self.equality_rhs = np.empty(0) if self.equality_rhs is None else np.asarray(self.equality_rhs, dtype=np.float64)
        if self.equality_matrix.shape != (len(self.equality_rhs), p):
            raise ValueError("affine equality contract differs")

    @property
    def parameters(self):
        return self.delta.shape[1]

    def probabilities(self, theta, tolerance=1e-8):
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.parameters,) or not np.isfinite(theta).all():
            raise ValueError("coefficient shape invalid")
        q = self.base + (self.delta @ theta).reshape(self.base.shape)
        if q.min() < -tolerance or not np.allclose(q.sum(1), 1, atol=tolerance, rtol=0):
            raise ValueError("kernel mass invalid")
        return q

    def expected_coefficients(self, loss, mask=None):
        loss = np.asarray(loss, dtype=np.float64)
        if loss.shape != self.base.shape or not np.isfinite(loss).all():
            raise ValueError("expected-loss array invalid")
        mask = np.ones(len(loss), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        if mask.shape != (len(loss),) or not mask.any():
            raise ValueError("empty or malformed expectation subset")
        w = self.weights * mask
        w /= w.sum()
        linear = (w[:, None] * loss).ravel()
        return float(linear @ self.base.ravel()), np.asarray(linear @ self.delta).ravel()

    def change_coefficients(self):
        different = (self.emitted1 != self.identity1[:, None]) | (self.emitted2 != self.identity2[:, None])
        return self.expected_coefficients(different.astype(np.float64))


def build_mixture_problem(emitted1, emitted2, identity1, identity2, donor, contexts,
                          coefficient_mode="conditional", weights=None):
    donor = np.asarray(donor, dtype=np.float64)
    c = np.asarray(contexts)
    e1, e2 = np.asarray(emitted1), np.asarray(emitted2)
    i1, i2 = np.asarray(identity1), np.asarray(identity2)
    if (donor.shape != e1.shape or c.dtype != np.int64 or c.shape != (len(donor),)
            or not np.isfinite(donor).all() or donor.min() < 0
            or not np.allclose(donor.sum(1), 1, atol=1e-12, rtol=0)):
        raise ValueError("frozen donor/context contract invalid")
    if coefficient_mode not in ("scalar", "conditional"):
        raise ValueError("unknown mixture coefficient mode")
    if c.min() < 0 or not np.array_equal(np.unique(c), np.arange(c.max() + 1)):
        raise ValueError("contexts must be contiguous nonnegative IDs")
    own = (e1 == i1[:, None]) & (e2 == i2[:, None])
    if not own.any(1).all():
        raise ValueError("identity action absent")
    base = np.zeros_like(donor)
    base[np.arange(len(donor)), own.argmax(1)] = 1
    p = 1 if coefficient_mode == "scalar" else int(c.max()) + 1
    param = np.zeros_like(c) if coefficient_mode == "scalar" else c
    row, action = np.nonzero(donor - base)
    delta = sparse.coo_matrix(((donor - base)[row, action],
                              (row * donor.shape[1] + action, param[row])), shape=(donor.size, p)).tocsr()
    return AffineLaw(e1, e2, i1, i2, base, delta, np.tile([0., 1.], (p, 1)), weights)


@dataclass
class LPProgram:
    objective: np.ndarray
    inequality_matrix: object
    inequality_rhs: np.ndarray
    equality_matrix: object
    equality_rhs: np.ndarray
    bounds: np.ndarray
    names: list[str]
    parameters: int
    objective_constant: float = 0.


def _validated_constraints(law, constraints):
    if constraints.matrix.shape[1] != law.parameters:
        raise ValueError("constraint parameter count differs")
    return constraints.matrix, constraints.rhs


def linear_program(law, constraints, objective_coeff, objective_constant=0., maximize=False):
    a, b = _validated_constraints(law, constraints)
    coeff = np.asarray(objective_coeff, dtype=np.float64)
    if coeff.shape != (law.parameters,) or not np.isfinite(coeff).all():
        raise ValueError("linear objective invalid")
    sign = -1. if maximize else 1.
    return LPProgram(sign * coeff, a, b, law.equality_matrix, law.equality_rhs,
                     law.bounds.copy(), list(constraints.names), law.parameters, sign * float(objective_constant))


def protection_program(law, sensitive, constraints):
    sex = np.asarray(sensitive)
    if sex.dtype != np.int64 or sex.shape != (len(law.base),) or not np.isin(sex, [0, 1]).all():
        raise ValueError("binary protected labels not aligned")
    a, b = _validated_constraints(law, constraints)
    projections, offsets, epigraph_weights, names = [], [], [], []
    offset = 0
    for view, values, weight in (("pair", np.c_[law.emitted1.ravel(), law.emitted2.ravel()], 1.),
                                  ("r1", law.emitted1.ravel()[:, None], .5),
                                  ("r2", law.emitted2.ravel()[:, None], .5)):
        unique, inv = np.unique(values, axis=0, return_inverse=True)
        z = len(unique)
        for s in (0, 1):
            weighted = np.repeat(law.weights * (sex == s), law.base.shape[1])
            incidence = sparse.coo_matrix((weighted, (inv, np.arange(law.base.size))), shape=(z, law.base.size)).tocsr()
            projections.append(incidence @ law.delta)
            offsets.append(np.asarray(incidence @ law.base.ravel()).ravel())
            names.extend(f"epigraph/{view}/sex{s}/output{k}" for k in range(z))
        epigraph_weights.extend([weight] * z)
        offset += z
    t = len(epigraph_weights)
    epigraph_blocks, cursor = [], 0
    for k in range(0, len(projections), 2):
        z = projections[k].shape[0]
        neg = sparse.coo_matrix((-np.ones(z), (np.arange(z), cursor + np.arange(z))), shape=(z, t)).tocsr()
        epigraph_blocks.extend((neg, neg))
        cursor += z
    m = sparse.vstack(projections, format="csr")
    epi = sparse.vstack(epigraph_blocks, format="csr")
    matrix = sparse.vstack((sparse.hstack((a, sparse.csr_matrix((a.shape[0], t)))),
                             sparse.hstack((m, epi))), format="csr")
    rhs = np.r_[b, -np.concatenate(offsets)]
    eq = sparse.hstack((law.equality_matrix, sparse.csr_matrix((len(law.equality_rhs), t))), format="csr")
    return LPProgram(np.r_[np.zeros(law.parameters), epigraph_weights], matrix, rhs, eq,
                     law.equality_rhs, np.vstack((law.bounds, np.tile([0., np.inf], (t, 1)))),
                     list(constraints.names) + names, law.parameters)


def primal_dual_certificate(program, result, tolerance=1e-7):
    if not result.success or result.x is None:
        return {"certified": False, "status": int(result.status), "message": str(result.message),
                "infeasibility_certificate_available": False}
    x = np.asarray(result.x)
    y = np.asarray(result.ineqlin.marginals)
    ye = np.asarray(result.eqlin.marginals)
    lo = np.asarray(result.lower.marginals)
    up = np.asarray(result.upper.marginals)
    margin = program.inequality_rhs - program.inequality_matrix @ x
    eqerr = program.equality_matrix @ x - program.equality_rhs
    stationarity = program.objective - program.inequality_matrix.T @ y - program.equality_matrix.T @ ye - lo - up
    lb, ub = program.bounds.T
    dual = float(program.inequality_rhs @ y + program.equality_rhs @ ye
                 + lb[np.isfinite(lb)] @ lo[np.isfinite(lb)]
                 + ub[np.isfinite(ub)] @ up[np.isfinite(ub)] + program.objective_constant)
    primal = float(program.objective @ x + program.objective_constant)
    precise_primal = float(np.sum(np.asarray(program.objective, dtype=np.longdouble)
                                 * np.asarray(x, dtype=np.longdouble), dtype=np.longdouble)
                           + np.longdouble(program.objective_constant))
    accumulation_error = abs(primal - precise_primal)
    violation = max(0., float(-margin.min(initial=0)),
                    float((lb - x).max(initial=0)), float((x - ub).max(initial=0)),
                    float(np.abs(eqerr).max(initial=0)))
    dual_violation = max(0., float(y.max(initial=0)), float((-lo).max(initial=0)), float(up.max(initial=0)),
                         float(np.abs(stationarity).max(initial=0)),
                         float(np.abs(lo[~np.isfinite(lb)]).max(initial=0)),
                         float(np.abs(up[~np.isfinite(ub)]).max(initial=0)))
    complementarity = max(float(np.abs(y * margin).max(initial=0)),
                          float(np.abs(lo[np.isfinite(lb)] * (x - lb)[np.isfinite(lb)]).max(initial=0)),
                          float(np.abs(up[np.isfinite(ub)] * (ub - x)[np.isfinite(ub)]).max(initial=0)))
    gap = primal - dual
    return {"certified": bool(max(violation, dual_violation, complementarity, abs(gap), accumulation_error) <= tolerance),
            "scope": "fixed supplied empirical affine law and finite constraint/support list",
            "tolerance": tolerance, "primal_objective": primal, "dual_objective": dual,
            "primal_dual_gap": gap, "maximum_primal_violation": violation,
            "maximum_dual_violation": dual_violation, "maximum_complementarity_error": complementarity,
            "objective_longdouble_accumulation_difference": accumulation_error,
            "minimum_inequality_margin": float(margin.min()) if len(margin) else None,
            "active_constraint_names": [name for name, slack in zip(program.names, margin) if abs(slack) <= tolerance],
            "status": int(result.status)}


def solve_program(program, time_limit=300., tolerance=1e-8):
    result = linprog(program.objective, A_ub=program.inequality_matrix,
                     b_ub=program.inequality_rhs, A_eq=program.equality_matrix,
                     b_eq=program.equality_rhs, bounds=program.bounds, method="highs",
                     options={"time_limit": float(time_limit), "threads": 1, "primal_feasibility_tolerance": tolerance,
                              "dual_feasibility_tolerance": tolerance})
    cert = primal_dual_certificate(program, result, max(1e-7, 10 * tolerance))
    return {"success": bool(result.success and cert["certified"]), "status": int(result.status),
            "message": str(result.message), "theta": None if result.x is None else result.x[:program.parameters].copy(),
            "primal": None if result.x is None else result.x.copy(), "certificate": cert,
            "inequality_dual": None if not result.success else result.ineqlin.marginals.copy(),
            "equality_dual": None if not result.success else result.eqlin.marginals.copy(),
            "lower_dual": None if not result.success else result.lower.marginals.copy(),
            "upper_dual": None if not result.success else result.upper.marginals.copy(),
            "program": program}


def program_size(program):
    """Actual sparse storage receipt; solver workspace needs extra reservation."""
    matrices = (program.inequality_matrix.tocsr(), program.equality_matrix.tocsr())
    storage = sum(a.data.nbytes + a.indices.nbytes + a.indptr.nbytes for a in matrices)
    storage += sum(a.nbytes for a in (program.objective, program.inequality_rhs,
                                    program.equality_rhs, program.bounds))
    return {"parameters": program.parameters, "variables": len(program.objective),
            "inequalities": len(program.inequality_rhs), "equalities": len(program.equality_rhs),
            "matrix_nonzeros": sum(a.nnz for a in matrices), "sparse_input_bytes": int(storage),
            "solver_workspace_not_included": True}


def solve_linear(law, constraints, objective_coeff, objective_constant=0., maximize=False, **kwargs):
    return solve_program(linear_program(law, constraints, objective_coeff, objective_constant, maximize), **kwargs)


def solve_protection(law, sensitive, constraints, **kwargs):
    return solve_program(protection_program(law, sensitive, constraints), **kwargs)


def certify_infeasibility(program, **kwargs):
    """One separately counted Phase-I solve; no hidden extra optimization.

    A positive certified minimum common constraint relaxation proves no
    feasible point for these bounded finite constraints. All equalities are
    replaced by two inequalities with the same relaxation. Parameter bounds
    are retained, and must themselves be nonempty.
    """
    n = len(program.objective)
    blocks = [program.inequality_matrix, program.equality_matrix, -program.equality_matrix]
    a = sparse.vstack(blocks, format="csr")
    rhs = np.r_[program.inequality_rhs, program.equality_rhs, -program.equality_rhs]
    relaxed = sparse.hstack((a, -np.ones((len(rhs), 1))), format="csr")
    names = list(program.names) + [f"phaseI/equality/{k}/positive" for k in range(len(program.equality_rhs))]
    names += [f"phaseI/equality/{k}/negative" for k in range(len(program.equality_rhs))]
    phase = LPProgram(np.r_[np.zeros(n), 1.], relaxed, rhs, sparse.csr_matrix((0, n + 1)),
                      np.empty(0), np.vstack((program.bounds, [0., np.inf])), names, n + 1)
    answer = solve_program(phase, **kwargs)
    answer["infeasibility_certified"] = bool(answer["success"] and answer["certificate"]["dual_objective"] > answer["certificate"]["tolerance"])
    answer["certificate"]["scope"] = "positive lower bound for common-relaxation Phase-I on original bounded finite constraints"
    answer["original_constraints"] = len(program.inequality_rhs)
    return answer


def direct_bayes(law, theta, sensitive):
    q = law.probabilities(theta)
    sex = np.asarray(sensitive)
    if sex.dtype != np.int64 or sex.shape != (len(q),) or not np.isin(sex, [0, 1]).all():
        raise ValueError("binary protected labels not aligned")
    answers = {}
    for view, values in (("pair", np.c_[law.emitted1.ravel(), law.emitted2.ravel()]),
                          ("r1", law.emitted1.ravel()[:, None]), ("r2", law.emitted2.ravel()[:, None])):
        unique, inv = np.unique(values, axis=0, return_inverse=True)
        mass = np.stack([np.bincount(inv, weights=(q * (law.weights * (sex == b))[:, None]).ravel(),
                                    minlength=len(unique)) for b in (0, 1)])
        answers[view] = float(mass.max(0).sum())
    return answers["pair"] + .5 * (answers["r1"] + answers["r2"]), answers


def omitted_column_report(program, inequality_dual, equality_dual, columns, equality_columns, costs,
                          scope="all supplied omitted columns", tolerance=1e-7):
    """Price every supplied omitted nonnegative unbounded variable.

    Caller must enumerate ALL allowed omitted columns to claim full-support
    optimality; a restricted-master certificate cannot establish that claim.
    """
    cols = sparse.csr_matrix(columns)
    eqcols = sparse.csr_matrix(equality_columns)
    costs = np.asarray(costs, dtype=np.float64)
    if cols.shape[0] != len(program.inequality_rhs) or eqcols.shape != (len(program.equality_rhs), cols.shape[1]) or costs.shape != (cols.shape[1],):
        raise ValueError("pricing column scope/shape differs")
    reduced = costs - cols.T @ inequality_dual - eqcols.T @ equality_dual
    return {"scope": scope, "columns_evaluated": len(costs), "minimum_reduced_cost": float(reduced.min()) if len(reduced) else None,
            "negative_reduced_cost_count": int((reduced < -tolerance).sum()),
            "all_supplied_columns_dual_feasible": bool(np.all(reduced >= -tolerance)), "reduced_cost": reduced}
