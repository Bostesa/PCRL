"""Finite-channel machinery for a role-constrained stochastic release (RCSR).

A finite code `T` is computed from permitted A-side inputs; the release is a token `Z` drawn from
`Q[t, :] = Pr(Z = . | T = t)`. For each role `r` with protected attribute `S_r` and existing view
`C_r`, the design constraint is `I(S_r; Z | C_r) <= delta_r`.

**Scope, once, and it governs everything below.** Every quantity here is computed on a *specified
finite model* `p(s, c, t)`. Convexity, feasibility and optimality are properties of that model.
They are not population statements and they do not survive coarsening of a continuous view:
`test_coarse_conditioning_hides_the_leak_entirely` exhibits `I(S;Z|bin(C)) = 0` alongside
`I(S;Z|C) = log 2`. Empirical protection is established by attacks on the full continuous view, not
here.

Model layout: a role model is an array `p[s, c, t]` summing to 1. A task model is the same shape
with `y` in place of `s`, so `cmi` computes utility information as well as leakage.

Assumes `Z — T — (S, C)` is Markov, which is the mechanism definition.

Structure of the two results used by the caller:

* `feasible_direction` implements the simultaneous-view analogue of the finite perfect-privacy
  nullspace test. For a single role the unconditional case is Rassouli & Gündüz, *On Perfect
  Privacy*, Proposition 1 (`dim(Null(P_{X|W}) \\ Null(P_{Y|W})) != 0`). The conditional,
  multi-role version -- an intersection over roles **and** over conditioning values -- is not in
  that paper and is the part that must be validated numerically here.
* `solve` is convex: relative entropy is jointly convex and both of its arguments are affine in
  `Q`, so each constraint sublevel set is convex; the objective is linear. **No rate constraint is
  ever added** (see REGISTRATION §0): the convex/LP characterization is known to fail when one is
  active.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-300


# ------------------------------------------------------------------ model algebra
def _parts(p):
    """`p(c,t)`, `p(s|c)` and `p(c)` from a role model `p[s, c, t]`."""
    p = np.asarray(p, dtype=np.float64)
    p_ct = p.sum(axis=0)                                  # [C, T]
    p_c = p_ct.sum(axis=1)                                # [C]
    with np.errstate(invalid='ignore', divide='ignore'):
        p_s_given_c = np.where(p_c[None, :] > 0, p.sum(axis=2) / np.where(p_c > 0, p_c, 1)[None, :], 0.0)
    return p, p_ct, p_c, p_s_given_c


def cmi(p, q) -> float:
    """`I(S; Z | C)` in nats for the finite model `p[s, c, t]` under the channel `q[t, z]`."""
    p, p_ct, _, p_s_given_c = _parts(p)
    q = np.asarray(q, dtype=np.float64)
    a = np.einsum('sct,tz->scz', p, q)                    # p(s, c, z)
    b = p_s_given_c[:, :, None] * np.einsum('ct,tz->cz', p_ct, q)[None, :, :]   # p(s|c) p(c, z)
    mask = a > 0
    return float(np.sum(a[mask] * np.log(a[mask] / np.maximum(b[mask], EPS))))


def constraint_rows(p) -> np.ndarray:
    """`A[(s,c), t] = p(t,s,c) - p(s|c) p(t,c)`.

    For a binary release with `q_t = Pr(Z=1|T=t)`, conditional independence holds exactly iff
    `A q = 0`; the `Z=0` case follows because `A 1 = 0`.
    """
    p, p_ct, _, p_s_given_c = _parts(p)
    rows = p - p_s_given_c[:, :, None] * p_ct[None, :, :]
    return rows.reshape(-1, p.shape[2])


def random_channel(n_t: int, n_z: int, rng) -> np.ndarray:
    q = rng.random((n_t, n_z)) + 1e-3
    return q / q.sum(1, keepdims=True)


# ------------------------------------------------------------------ feasibility
def feasible_direction(roles, task, *, tol: float = None, box: float = 0.499) -> dict:
    """Is there a binary release that is exactly private for every role and still useful?

    Stacks `A_r` for all roles, takes the numerical nullspace, and asks whether the task rows `B`
    vanish on it. `A_r 1 = B 1 = 0`, so any `f` in the intersection scales into `q = 1/2 + alpha f`.

    Returns the direction, the achievable scale, the singular spectrum and how the answer moves
    with the tolerance -- an estimated model has no exact nullspace, so the threshold is part of
    the claim, not an implementation detail.
    """
    a = np.vstack([constraint_rows(p) for p in roles])
    b = constraint_rows(task)
    n_t = a.shape[1]
    u, s, vt = np.linalg.svd(a)
    padded = np.concatenate([s, np.zeros(max(0, n_t - len(s)))])
    if tol is None:
        tol = max(a.shape) * np.finfo(float).eps * (padded[0] if padded.size else 1.0)
        tol = max(tol, 1e-12)
    null = vt[padded <= tol]
    by_tolerance = {f'{t:g}': int(np.sum(padded <= t)) for t in (1e-14, 1e-12, 1e-10, 1e-8, 1e-6)}

    direction, scale, task_norm = np.zeros(n_t), 0.0, 0.0
    if len(null):
        # Pick the direction in the nullspace that carries the most task signal.
        projected = b @ null.T
        if projected.size:
            _, _, wt = np.linalg.svd(projected, full_matrices=False)
            direction = wt[0] @ null
            task_norm = float(np.linalg.norm(b @ direction))
            peak = float(np.max(np.abs(direction)))
            if peak > 0:
                direction = direction / peak
                scale = box
    return {'feasible': bool(task_norm > max(tol, 1e-9) and scale > 0),
            'direction': direction, 'scale': scale, 'task_norm': task_norm,
            'null_dim': int(len(null)), 'singular_values': padded.tolist(),
            'tolerance': float(tol), 'null_dim_by_tolerance': by_tolerance,
            'scope': 'this finite model and this input code only; not an ACS statement'}


# ------------------------------------------------------------------ convex program
def _cvxpy():
    import cvxpy as cp
    return cp


def _cmi_expr(cp, p, q_var):
    """`I(S;Z|C)` as a DCP expression.

    `sum kl_div(a, b) = sum a log(a/b) - sum a + sum b`, and both sums are 1 here, so the
    identity is exact rather than an approximation.
    """
    p, p_ct, _, p_s_given_c = _parts(p)
    n_s, n_c, _ = p.shape
    terms = []
    for s in range(n_s):
        for c in range(n_c):
            a = p[s, c, :] @ q_var                                  # p(s, c, z)
            b = p_s_given_c[s, c] * (p_ct[c, :] @ q_var)            # p(s|c) p(c, z)
            terms.append(cp.sum(cp.kl_div(a, b)))
    return cp.sum(cp.hstack(terms)) if terms else cp.Constant(0)


def _report(q, roles, deltas, problem, extra):
    verified = {name: cmi(p, q) for name, p in roles.items()}
    out = {'Q': q, 'status': problem.status, 'objective': float(problem.value),
           'row_sum_residual': float(np.max(np.abs(q.sum(1) - 1.0))),
           'negativity_residual': float(max(0.0, -q.min())),
           'verified_cmi': verified,
           'constraint_violation': {n: max(0.0, verified[n] - deltas[n]) for n in roles},
           'scope': 'optimal for this fitted finite model; not a population guarantee'}
    out.update(extra)
    return out


def solve(*, roles: dict, cost, deltas: dict, n_z: int, solver: str = None,
          clean: bool = True) -> dict:
    """Minimise a fixed linear cost subject to `I(S_r; Z | C_r) <= delta_r` for every role.

    Convex: the objective is linear in `Q` and every constraint is a sublevel set of a convex
    function of `Q`. Reports feasibility residuals and, where the solver supplies one, a certified
    optimality gap -- a success string alone is not accepted (REGISTRATION G4).
    """
    cp = _cvxpy()
    cost = np.asarray(cost, dtype=np.float64)
    n_t = cost.shape[0]
    p_t = next(iter(roles.values())).sum(axis=(0, 1))
    q = cp.Variable((n_t, n_z), nonneg=True)
    objective = cp.Minimize(cp.sum(cp.multiply(p_t[:, None] * cost, q)))
    constraints = [cp.sum(q, axis=1) == 1]
    for name, p in roles.items():
        constraints.append(_cmi_expr(cp, p, q) <= deltas[name])
    problem = cp.Problem(objective, constraints)
    value, problem, stats, attempts = _solve_verified(cp, problem, q, roles, deltas, solver, clean)
    gap = None
    for attr in ('gap', 'duality_gap'):
        if stats is not None and getattr(stats, attr, None) is not None:
            gap = float(getattr(stats, attr))
    return _report(value, roles, deltas, problem,
                   {'optimality_gap': gap, 'solver': (stats.solver_name if stats else solver),
                    'n_iters': (stats.num_iters if stats else None), 'formulation': 'convex_cmi',
                    'solver_attempts': attempts})


def _solve_verified(cp, problem, q, roles, deltas, solver, clean, tol: float = 1e-7):
    """Solve, then check the answer independently; retry on another solver if it does not hold.

    REGISTRATION G4: a solver's success string is not accepted as evidence. The accepted answer is
    the one whose recomputed `cmi` actually satisfies the budget, preferring the cheaper objective
    among those that do. Exponential-cone problems at `delta = 0` sit on the boundary and routinely
    report `optimal_inaccurate` while being numerically fine, so the recomputation is the test.
    """
    order = [solver] if solver else [cp.CLARABEL, cp.SCS]
    attempts, best = [], None
    for candidate in order:
        try:
            problem.solve(solver=candidate)
        except cp.error.SolverError as exc:
            attempts.append({'solver': str(candidate), 'status': f'error: {exc}'})
            continue
        if q.value is None:
            attempts.append({'solver': str(candidate), 'status': problem.status})
            continue
        value = np.asarray(q.value, dtype=np.float64)
        if clean:                   # project tiny solver noise back into the simplex
            value = np.clip(value, 0.0, None)
            value = value / value.sum(1, keepdims=True)
        worst = max((cmi(p, value) - deltas[n] for n, p in roles.items()), default=0.0)
        attempts.append({'solver': str(candidate), 'status': problem.status,
                         'verified_worst_violation': float(max(0.0, worst)),
                         'objective': float(problem.value)})
        stats = getattr(problem, 'solver_stats', None)
        key = (max(0.0, worst) > tol, float(problem.value))
        if best is None or key < best[0]:
            best = (key, value, problem, stats)
        if max(0.0, worst) <= tol and problem.status == 'optimal':
            break
    if best is None:
        raise RuntimeError(f'no solver returned a channel; attempts: {attempts}')
    return best[1], best[2], best[3], attempts


def solve_lp(*, roles: dict, cost, epsilons: dict, n_z: int, solver: str = None,
             clean: bool = True) -> dict:
    """Linear-programming alternative: a pointwise likelihood-ratio bound per role.

    `exp(-eps) p(z|c) <= p(z|s,c) <= exp(eps) p(z|c)` for every supported `(s,c,z)`. These are
    linear in `Q`, and they control the entire modelled categorical output law rather than a set
    of moments -- so they can be materially stricter than an average information budget. The
    existing `.001`-nat empirical allowance is **not** automatically an appropriate `eps`.
    """
    cp = _cvxpy()
    cost = np.asarray(cost, dtype=np.float64)
    n_t = cost.shape[0]
    p_t = next(iter(roles.values())).sum(axis=(0, 1))
    q = cp.Variable((n_t, n_z), nonneg=True)
    objective = cp.Minimize(cp.sum(cp.multiply(p_t[:, None] * cost, q)))
    constraints = [cp.sum(q, axis=1) == 1]
    for name, p in roles.items():
        eps = epsilons[name]
        p_arr, p_ct, p_c, _ = _parts(p)
        p_sc = p_arr.sum(axis=2)
        for c in range(p_arr.shape[1]):
            if p_c[c] <= 0:
                continue
            marginal = (p_ct[c, :] / p_c[c]) @ q                    # p(z | c)
            for s in range(p_arr.shape[0]):
                if p_sc[s, c] <= 0:
                    continue
                joint = (p_arr[s, c, :] / p_sc[s, c]) @ q           # p(z | s, c)
                constraints += [joint <= np.exp(eps) * marginal,
                                joint >= np.exp(-eps) * marginal]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver or cp.CLARABEL)
    value = np.asarray(q.value, dtype=np.float64)
    if clean:
        value = np.clip(value, 0.0, None)
        value = value / value.sum(1, keepdims=True)
    stats = getattr(problem, 'solver_stats', None)
    out = _report(value, roles, {n: np.inf for n in roles}, problem,
                  {'optimality_gap': None, 'solver': (stats.solver_name if stats else solver),
                   'formulation': 'lp_likelihood_ratio'})
    out['max_log_ratio'] = max_log_ratio(roles, value)
    return out


def max_log_ratio(roles: dict, q) -> float:
    """Largest realised `|log p(z|s,c) - log p(z|c)|` over supported cells, across roles."""
    worst = 0.0
    for p in roles.values():
        p_arr, p_ct, p_c, _ = _parts(p)
        p_sc = p_arr.sum(axis=2)
        for c in range(p_arr.shape[1]):
            if p_c[c] <= 0:
                continue
            marginal = (p_ct[c, :] / p_c[c]) @ q
            for s in range(p_arr.shape[0]):
                if p_sc[s, c] <= 0:
                    continue
                joint = (p_arr[s, c, :] / p_sc[s, c]) @ q
                ok = (marginal > 1e-12) & (joint > 1e-12)
                if ok.any():
                    worst = max(worst, float(np.max(np.abs(np.log(joint[ok] / marginal[ok])))))
    return worst


# ------------------------------------------------------------------ the released object
def sample_release(q, t, rng) -> np.ndarray:
    """Draw one token per row and return it as a fixed public one-hot.

    The released object is the **sampled token**. Releasing `Q[t,:]`, its logits or an expected
    prototype is a different mechanism and may disclose `T` (REGISTRATION G3): the review's own
    fixture records the probability-vector release leaking 0.4748 and 0.2499 nats in the two toys
    where the sampled token leaks exactly 0.
    """
    q = np.asarray(q, dtype=np.float64)
    t = np.asarray(t)
    draws = np.array([rng.choice(q.shape[1], p=q[i] / q[i].sum()) for i in t])
    out = np.zeros((len(t), q.shape[1]), dtype=np.float64)
    out[np.arange(len(t)), draws] = 1.0
    return out


def is_sampled_token(values) -> bool:
    """Gate check: every row is a one-hot indicator, not a probability vector."""
    a = np.asarray(values, dtype=np.float64)
    return bool(a.ndim == 2 and np.all((a == 0.0) | (a == 1.0)) and np.all(a.sum(1) == 1.0))
