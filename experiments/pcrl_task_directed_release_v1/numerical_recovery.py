"""One deterministic Branch D retry of a positive-budget finite channel.

Only the entropy-cone coordinates change. For m = p(s,c) > 0,

    rel_entr(a, b) = m * rel_entr(a / m, b / m),

where a = sum_t p(s,c,t) Q[t,z] and
b = p(s|c) sum_t p(c,t) Q[t,z]. Thus a/m is P(z|s,c)
and b/m is P(z|c). Positive homogeneity holds also at the closed
relative-entropy boundary (0,0). Cells with m=0 have a=b=0 for every
nonnegative Q and contribute exactly zero; only those cells are omitted.
No smoothing, mass floor, row renormalization of laws, or changed budget
is introduced. Every supplied role gets its own original-mass-weighted sum.

The cost, row simplex, support equations, nonnegativity and optional
witness objective upper bound are retained. Consequently the feasible
set and objective are identical in exact arithmetic, even if the witness
has only a numerical feasibility check. Removing the witness bound would
need an additional exact feasibility argument; this module never removes it.

Exactly one CLARABEL call uses finite.py's registered settings. The same
NumPy/SciPy acceptance checks and tolerances run afterward. Rejected retries
return no Q, including when a valid witness exists. ``optimal`` means an
accepted solver-reported ``optimal`` status, not an independent dual proof;
``optimal_inaccurate`` never sets it. This function performs no artifact I/O.
Registration, once-per-unit enforcement and preservation of originals belong
to the caller; invoking this pure solver twice would perform two retries.
"""

from __future__ import annotations

import hashlib
import operator
import time
import warnings
from collections.abc import Mapping

import numpy as np

from . import finite


def _array_hash(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode('ascii'))
    h.update(repr(value.shape).encode('ascii'))
    h.update(value.tobytes())
    return h.hexdigest()


def normalized_entropy_coefficients(p):
    """Return exact-law masses and conditional coefficients, without flooring.

    Rows have the same (s,c) order as ``p.reshape(-1,n_t)`` after omission
    of zero-mass cells. Multiplication by Q gives the two cone arguments.
    Coefficients are divided analytically before CVXPY canonicalization;
    otherwise a small joint-mass coefficient would remain inside the cone.
    """
    p = finite._law(p)
    n_s, n_c, n_t = p.shape
    p_sc, p_ct = p.sum(axis=2), p.sum(axis=0)
    p_c = p_ct.sum(axis=1)
    supported = p_sc.ravel() > 0
    masses = p_sc.ravel()[supported].copy()
    contexts = np.tile(np.arange(n_c), n_s)[supported]
    a_coeff = p.reshape(-1, n_t)[supported] / masses[:, None]
    b_coeff = p_ct[contexts] / p_c[contexts, None]
    if not np.isfinite(a_coeff).all() or not np.isfinite(b_coeff).all():
        raise FloatingPointError('Nonfinite conditional entropy coefficients')
    details = {
        'joint_law_sha256': _array_hash(p),
        'supported_cells': int(len(masses)),
        'omitted_zero_mass_cells': int(supported.size-len(masses)),
        'positive_mass_minimum': float(masses.min()),
        'positive_mass_maximum': float(masses.max()),
        'a_coefficient_row_sum_error': float(np.max(np.abs(a_coeff.sum(axis=1)-1))),
        'b_coefficient_row_sum_error': float(np.max(np.abs(b_coeff.sum(axis=1)-1))),
    }
    return masses, a_coeff, b_coeff, details


def solve_normalized_retry(cost, roles, budget, parent=None, state_mass=None,
                           zero_action=0, embedded_q=None):
    """Solve the same positive-budget problem once using normalized cone inputs.

    Arguments match ``finite.solve`` for the positive-budget entropy branch.
    Both U/PWGTP and local/coalition laws must be supplied unchanged. Invalid
    inputs raise before solving. A solver failure or failed independent check
    returns ``accepted=False``, ``feasible=False``, ``optimal=False``, ``Q=None``;
    attempt diagnostics retain any candidate's empirical constraint residuals.
    No option changes solver settings, tolerances, law weighting or retry count.
    """
    import cvxpy as cp

    cost = np.array(cost, dtype=float, copy=True)
    if cost.ndim != 2 or min(cost.shape) < 1 or not np.all(np.isfinite(cost)):
        raise ValueError('cost must be a finite nonempty state-by-action matrix')
    n_t, n_z = cost.shape
    if budget is None or isinstance(budget, bool):
        raise ValueError('Normalized entropy retry requires a finite positive budget')
    budget = float(budget)
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError('Normalized entropy retry requires a finite positive budget')
    if not isinstance(roles, Mapping) or not roles:
        raise ValueError('Normalized entropy retry requires nonempty roles')
    original_role_count = len(roles)
    roles = {str(name): finite._law(p).copy() for name, p in roles.items()}
    if len(roles) != original_role_count:
        raise ValueError('Role names collide after string conversion')
    if any(p.shape[2] != n_t for p in roles.values()):
        raise ValueError('All roles must use the cost input-state alphabet')
    try:
        zero_action = operator.index(zero_action)
    except TypeError as exc:
        raise ValueError('zero_action must be an action index') from exc
    if not 0 <= zero_action < n_z:
        raise ValueError('zero_action is outside the action alphabet')

    parents, ties, absent, support = finite._support(n_t, roles, state_mass, parent, cost)
    support_eq, support_rhs = finite._support_equations(n_t, n_z, ties, absent, zero_action)
    matrices = {name: finite.privacy_matrix(p) for name, p in roles.items()}
    stacked = np.vstack(list(matrices.values()))
    scales = np.max(np.abs(stacked), axis=1, initial=0)
    nonzero = scales > 0
    normalized = stacked[nonzero] / scales[nonzero, None]
    witness_details, witness_objective = None, None
    if embedded_q is not None:
        witness = np.array(embedded_q, dtype=float, copy=True)
        if witness.shape != (n_t, n_z):
            if witness.ndim == 2 and witness.shape[1] == n_z and witness.shape[0] > int(parents.max()):
                witness = witness[parents]
            else:
                raise ValueError('Embedded witness has incompatible shape')
        witness_details = finite._evaluate(witness, roles, budget, matrices, normalized,
            support_eq, support_rhs, cost)
        if not witness_details['feasible']:
            raise ValueError('Embedded witness fails independent feasibility checks')
        witness_objective = witness_details['objective']

    q_variable = cp.Variable((n_t, n_z), nonneg=True)
    constraints = [cp.sum(q_variable, axis=1) == 1]
    if len(support_rhs):
        constraints.append(support_eq @ cp.reshape(q_variable, (n_t*n_z,), order='C') == support_rhs)
    role_details = {}
    for name, law in roles.items():
        masses, a_coeff, b_coeff, role_details[name] = normalized_entropy_coefficients(law)
        terms = cp.rel_entr(a_coeff @ q_variable, b_coeff @ q_variable)
        constraints.append(cp.sum(cp.multiply(masses[:, None], terms)) <= budget)
    objective = cp.sum(cp.multiply(cost, q_variable))
    if witness_objective is not None:
        constraints.append(objective <= witness_objective)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    if not problem.is_dcp():
        raise RuntimeError('Normalized entropy retry unexpectedly violates convexity rules')

    settings = dict(finite.SOLVER_SETTINGS['CLARABEL'])
    tolerances = dict(finite.ACCEPTANCE_TOLERANCES)
    started = time.perf_counter()
    raw, value, solver_status, error = None, None, 'solver_error', None
    warning_messages = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            value = problem.solve(solver='CLARABEL', warm_start=False, verbose=False, **settings)
        warning_messages = [str(w.message) for w in caught]
        solver_status = str(problem.status)
        raw = q_variable.value
    except Exception as exc:
        error = str(exc)
    q, repair = finite._clean_channel(raw, ties, absent, zero_action)
    residuals = finite._evaluate(q, roles, budget, matrices, normalized,
        support_eq, support_rhs, cost, witness_objective)
    accepted = bool(residuals['feasible'] and repair <= tolerances['repair']
        and residuals['witness_objective_excess'] <= tolerances['witness_objective']
        and solver_status in ('optimal', 'optimal_inaccurate'))
    rejection_reasons = []
    if not residuals['feasible']:
        rejection_reasons.append('independent_constraint_check_failed')
    if repair > tolerances['repair']:
        rejection_reasons.append('repair_tolerance_exceeded')
    if residuals.get('witness_objective_excess', 0) > tolerances['witness_objective']:
        rejection_reasons.append('witness_objective_tolerance_exceeded')
    if solver_status not in ('optimal', 'optimal_inaccurate'):
        rejection_reasons.append('solver_status_not_accepted')
    attempt = {
        'solver': 'CLARABEL', 'status': solver_status, 'accepted': accepted,
        'seconds': time.perf_counter()-started,
        'reported_objective': float(value) if value is not None and np.isfinite(value) else None,
        'repair_max': repair if np.isfinite(repair) else None,
        'residuals': residuals, 'warnings': warning_messages,
        'rejection_reasons': rejection_reasons,
    }
    if raw is not None:
        raw_simplex = float(np.max(np.abs(np.asarray(raw).sum(axis=1)-1)))
        raw_minimum = float(np.min(raw))
        attempt['raw_simplex'] = raw_simplex if np.isfinite(raw_simplex) else None
        attempt['raw_minimum'] = raw_minimum if np.isfinite(raw_minimum) else None
    if error is not None:
        attempt['error'] = error
    if problem.solver_stats is not None:
        attempt['iterations'] = problem.solver_stats.num_iters
    returned_residuals = residuals if accepted else {
        'feasible': False, 'reason': 'no_accepted_retry_candidate',
        'candidate_constraint_feasible': bool(residuals['feasible']),
    }
    return {
        'Q': q if accepted else None,
        'objective': residuals.get('objective') if accepted else None,
        'status': solver_status if accepted else 'failed',
        'solver': 'CLARABEL', 'solver_status': solver_status,
        'accepted': accepted, 'feasible': accepted, 'optimal': accepted and solver_status == 'optimal',
        'residuals': returned_residuals,
        'rank': int(np.linalg.matrix_rank(stacked)),
        'role_ranks': {name: int(np.linalg.matrix_rank(a)) for name, a in matrices.items()},
        'support': support, 'attempts': [attempt], 'witness': witness_details,
        'tolerances': tolerances, 'solver_settings': settings,
        'budget': budget, 'information_units': 'nats',
        'diagnostics': {
            'formulation': 'p_sc_times_rel_entr_conditional_probabilities_v1',
            'roles': role_details, 'cost_sha256': _array_hash(cost),
            'witness_objective_constraint_retained': witness_objective is not None,
            'witness_objective_postcheck_tolerance': tolerances['witness_objective'],
            'solver_call_limit': 1, 'fallback': 'none',
            'original_artifacts_written': False,
            'equivalence': 'positive homogeneity of closed relative entropy; all other constraints retained',
            'optimality_scope': 'accepted solver-reported optimal status only; no independent dual certificate',
            'feasibility_scope': 'empirical numerical constraints at unchanged finite.py tolerances',
        },
    }
