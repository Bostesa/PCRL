"""Data-free active-constraint fixtures: own solver vs campaign guard.project_slack vs scipy SLSQP.

Pure numpy/scipy on synthetic vectors; no person-level array, checkpoint or reader is touched.
"""
from __future__ import annotations
import json
import sys


def slsqp(p, A, b, r):
    import numpy as np
    from scipy.optimize import minimize
    cons = [{'type': 'ineq', 'fun': (lambda d, a=a, bb=bb: bb - a @ d), 'jac': (lambda d, a=a: -a)} for a, bb in zip(A, b)]
    cons.append({'type': 'ineq', 'fun': lambda d: r * r - d @ d, 'jac': lambda d: -2 * d})
    x0 = np.zeros_like(p)
    out = minimize(lambda d: float((d - p) @ (d - p)), x0, jac=lambda d: 2 * (d - p), constraints=cons, method='SLSQP',
                   options={'ftol': 1e-16, 'maxiter': 2000})
    return out.x, out


def cases(rng):
    import numpy as np
    r = 0.01
    out = []
    for dim in (65, 325):
        for t in range(40):
            A = rng.normal(size=(3, dim)) * rng.choice([1e-3, 1.0, 1e2], size=(3, 1))
            p = rng.normal(size=dim) * rng.choice([1e-4, 3e-3, 2e-2])
            # slacks small so that halfspaces bind
            b = np.abs(rng.normal(size=3)) * rng.choice([0.0, 1e-6, 1e-4]) * np.linalg.norm(A, axis=1) * 0.01
            p = p + 0.02 * A[0] / np.linalg.norm(A[0])  # push into first halfspace
            out.append(('random', dim, p, A, b, r))
        a = rng.normal(size=dim); p = 0.008 * a / np.linalg.norm(a) + rng.normal(size=dim) * 1e-3
        out.append(('duplicate_rows', dim, p, np.vstack([a, a, rng.normal(size=dim)]), np.array([1e-4, 1e-4, 1.0]), r))
        out.append(('duplicate_rows_scaled', dim, p, np.vstack([a, 3 * a, rng.normal(size=dim)]), np.array([1e-4, 3e-4, 1.0]), r))
        out.append(('duplicate_rows_unequal_slack', dim, p, np.vstack([a, a, rng.normal(size=dim)]), np.array([2e-4, 1e-4, 1.0]), r))
        out.append(('antiparallel_slab', dim, p, np.vstack([a, -a, rng.normal(size=dim)]), np.array([1e-5, 1e-5, 1.0]), r))
        out.append(('antiparallel_zero_width', dim, p, np.vstack([a, -a, rng.normal(size=dim)]), np.array([0.0, 0.0, 1.0]), r))
        out.append(('ball_active_only', dim, rng.normal(size=dim), rng.normal(size=(3, dim)), np.full(3, 1e3), r))
        q = rng.normal(size=dim); out.append(('ball_and_halfspace', dim, 0.05 * q / np.linalg.norm(q), np.vstack([q + 0.3 * rng.normal(size=dim), rng.normal(size=(2, dim))]), np.array([1e-3, 10., 10.]), r))
        out.append(('zero_row_plus_active', dim, p, np.vstack([a, np.zeros(dim), rng.normal(size=dim)]), np.array([1e-5, 0.0, 1.0]), r))
        out.append(('all_zero_slack', dim, p, rng.normal(size=(3, dim)), np.zeros(3), r))
        out.append(('inactive', dim, rng.normal(size=dim) * 1e-4, rng.normal(size=(3, dim)), np.full(3, 1e2), r))
    return out


def main(seed=20261015):
    import numpy as np
    from two_recipient_adapter_independent.solver import solve
    from two_recipient_adapter.guard import project_slack
    rng = np.random.default_rng(seed)
    rows = []
    for name, dim, p, A, b, r in cases(rng):
        d_own, info = solve(p, A, b, r)
        d_c, qc = project_slack(p, A, b, r)
        d_s, so = slsqp(p, A, b, r)
        nrm = np.linalg.norm(A, axis=1); keep = nrm > 0
        def viol(d):
            return float(max(0.0, np.max((A[keep] @ d - b[keep]) / nrm[keep]))) if keep.any() else 0.0
        obj = lambda d: float(np.sum((d - p) ** 2))
        rows.append({'case': name, 'dim': dim, 'own_vs_campaign_max_abs': float(np.max(np.abs(d_own - d_c))),
                     'own_vs_slsqp_max_abs': float(np.max(np.abs(d_own - d_s))),
                     'objective_own': obj(d_own), 'objective_campaign': obj(d_c), 'objective_slsqp': obj(d_s),
                     'own_minus_slsqp_objective': obj(d_own) - obj(d_s),
                     'violation_own': viol(d_own), 'violation_campaign': viol(d_c), 'violation_slsqp': viol(d_s),
                     'norm_own': float(np.linalg.norm(d_own)), 'slsqp_success': bool(so.success),
                     'own_active': info['active_set'], 'own_ball': info.get('ball_active'),
                     'campaign_active': qc['active_set'], 'campaign_ball': qc['norm_bound_active'],
                     'campaign_fallback': qc['conservative_zero_fallback']})
    agree = [x for x in rows if x['own_vs_campaign_max_abs'] <= 1e-12 * max(1.0, 1.0)]
    feas = all(x['violation_own'] <= 1e-10 and x['violation_campaign'] <= 1e-10 and x['norm_own'] <= 0.01 * (1 + 1e-12) for x in rows)
    # SLSQP is a local first-order-accurate reference: own must be no worse (objective) than any feasible SLSQP point
    slsqp_ok = [x for x in rows if x['violation_slsqp'] <= 1e-9 and np.linalg.norm(1) and True]
    no_worse = all(x['own_minus_slsqp_objective'] <= 1e-12 + 1e-8 * x['objective_slsqp'] for x in slsqp_ok)
    summary = {'n_cases': len(rows), 'n_own_campaign_agree_1e-12': len(agree),
               'max_own_vs_campaign': max(x['own_vs_campaign_max_abs'] for x in rows),
               'max_own_vs_slsqp': max(x['own_vs_slsqp_max_abs'] for x in rows),
               'n_active_halfspace_cases': sum(bool(x['own_active']) for x in rows),
               'n_ball_active_cases': sum(bool(x['own_ball']) for x in rows),
               'all_feasible_own_and_campaign': bool(feas), 'own_no_worse_than_feasible_slsqp': bool(no_worse),
               'n_slsqp_feasible': len(slsqp_ok), 'campaign_fallbacks': sum(x['campaign_fallback'] for x in rows),
               'active_set_agree': sum(sorted(x['own_active']) == sorted(x['campaign_active']) for x in rows)}
    summary['pass'] = bool(summary['n_own_campaign_agree_1e-12'] == len(rows) and feas and no_worse and summary['campaign_fallbacks'] == 0)
    return {'schema': 'TRA1-independent-solver-fixtures-v1', 'seed': seed, 'summary': summary, 'cases': rows}


if __name__ == '__main__':
    out = main()
    with open(sys.argv[1], 'x') as f:
        json.dump(out, f, indent=2, sort_keys=True, default=float)
    print(json.dumps(out['summary'], indent=1))
