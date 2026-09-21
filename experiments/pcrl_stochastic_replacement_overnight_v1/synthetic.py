"""Registered synthetic comparison (§9): the constrained stochastic optimum against EVERY
deterministic map.

On a small finite model the deterministic family is enumerable, so this is a **certified**
comparison, not a heuristic control. It is a mathematical statement about the specified finite model.
It is not an ACS finding, and it does not establish global superiority of stochastic mechanisms over
all deterministic mechanisms in general.

Established mathematics is cited in `METHOD.md` §2; nothing here is claimed as new.
"""
from __future__ import annotations

import itertools

import numpy as np

from experiments.pcrl_stochastic_channel_v1 import channel as ch


def role_model(p_t, p_s_given_t, p_c_given_t=None) -> np.ndarray:
    """Build `p[s, c, t]` from a code marginal and conditionals."""
    p_t = np.asarray(p_t, np.float64)
    p_s1 = np.asarray(p_s_given_t, np.float64)
    n_t = len(p_t)
    if p_c_given_t is None:
        out = np.zeros((2, 1, n_t))
        out[1, 0, :] = p_t * p_s1
        out[0, 0, :] = p_t * (1 - p_s1)
        return out
    p_c1 = np.asarray(p_c_given_t, np.float64)
    out = np.zeros((2, 2, n_t))
    for s in (0, 1):
        ps = p_s1 if s else 1 - p_s1
        for c in (0, 1):
            pc = p_c1 if c else 1 - p_c1
            out[s, c, :] = p_t * ps * pc              # S and C conditionally independent given T
    return out


def deterministic_maps(n_t: int, k: int):
    """Every map `T -> Z`. There are `k ** n_t` of them; keep the model small."""
    for assignment in itertools.product(range(k), repeat=n_t):
        q = np.zeros((n_t, k))
        q[np.arange(n_t), assignment] = 1.0
        yield assignment, q


def enumerate_best_deterministic(roles: dict, deltas: dict, cost, p_t, k: int) -> dict:
    """Cheapest deterministic map that satisfies every role budget. Exhaustive, hence certified."""
    cost = np.asarray(cost, np.float64)
    p_t = np.asarray(p_t, np.float64)
    n_t = len(p_t)
    best, feasible, total = None, 0, 0
    for assignment, q in deterministic_maps(n_t, k):
        total += 1
        worst = max(ch.cmi(p, q) - deltas[name] for name, p in roles.items())
        if worst <= 1e-12:
            feasible += 1
            obj = float(np.sum(p_t[:, None] * cost * q))
            if best is None or obj < best['objective']:
                best = {'assignment': list(assignment), 'objective': obj,
                        'cmi': {name: ch.cmi(p, q) for name, p in roles.items()}}
    return {'best': best, 'n_feasible': feasible, 'n_enumerated': total,
            'certified': True,
            'note': 'exhaustive over all k**|T| deterministic maps, so this IS the deterministic optimum'}


def compare(roles: dict, deltas: dict, cost, p_t, k: int) -> dict:
    """Convex stochastic optimum versus the certified deterministic optimum, at matched budgets."""
    stochastic = ch.solve(roles=roles, cost=cost, deltas=deltas, n_z=k)
    det = enumerate_best_deterministic(roles, deltas, cost, p_t, k)
    out = {'k': k, 'deltas': dict(deltas), 'roles': sorted(roles),
           'stochastic': {'objective': stochastic['objective'],
                          'verified_cmi': stochastic['verified_cmi'],
                          'constraint_violation': stochastic['constraint_violation'],
                          'status': stochastic['status'], 'solver': stochastic['solver'],
                          'Q': stochastic['Q'].tolist()},
           'deterministic': det}
    if det['best'] is not None:
        gap = det['best']['objective'] - stochastic['objective']
        out['advantage_of_randomization'] = float(gap)
        out['ratio'] = (float(det['best']['objective'] / stochastic['objective'])
                        if stochastic['objective'] > 0 else None)
        out['strictly_better'] = bool(gap > 1e-6)
    else:
        out['advantage_of_randomization'] = None
        out['strictly_better'] = None
        out['deterministic_infeasible'] = True
    return out


def registered_model() -> dict:
    """The one small synthetic problem, fixed here before it was run.

    Five code states. `S` depends on `T` through `p_s1 = (.10, .35, .50, .65, .90)`, whose overall
    rate is exactly `.5`.

    **Note on structure, corrected after a fixture caught an overclaim in an earlier docstring.**
    Several proper subsets *do* average to `.5` — `{2}`, `{0,4}`, `{1,3}` — so a nonconstant
    deterministic map *can* be exactly private under the **trivial** context. That is deliberate and
    is what makes the comparison informative rather than rigged: under the local role alone,
    randomization has no advantage to find, and the enumeration confirms it. The interesting case is
    the **coalition** role, where `C` also depends on `T` and destroys that coincidence.

    `C` is a coarse public context with `p_c1 = (.20, .80, .50, .30, .70)`, conditionally independent
    of `S` given `T`. `Y = 1[T in {1,3}]` is the useful target; cost is `0` when the token matches `Y`
    and `1` otherwise, so the constant map costs `P(Y=1) = 0.4`.
    """
    p_t = np.full(5, 0.2)
    p_s1 = np.array([0.10, 0.35, 0.50, 0.65, 0.90])
    p_c1 = np.array([0.20, 0.80, 0.50, 0.30, 0.70])
    local = role_model(p_t, p_s1)
    coalition = role_model(p_t, p_s1, p_c1)
    y = np.array([0, 1, 0, 1, 0])
    return {'p_t': p_t, 'p_s1': p_s1, 'p_c1': p_c1, 'y': y,
            'roles': {'A/S': local, 'AB/S': coalition},
            'overall_p_s1': float(np.sum(p_t * p_s1))}


def cost_from_target(y, k: int) -> np.ndarray:
    """`0` if the token equals the target, else `1`. Tokens beyond 2 are unused actions."""
    n_t = len(y)
    cost = np.ones((n_t, k))
    for t in range(n_t):
        cost[t, int(y[t]) % k] = 0.0
    return cost
