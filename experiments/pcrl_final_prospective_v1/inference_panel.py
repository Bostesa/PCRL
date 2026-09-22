"""Registered endpoint families, intersection-union decisions and secondary intervals.

Estimand for any paired endpoint: for each anchor, the weighted ratio
sum_i w_i (L_plus,i - L_minus,i) / sum_i w_i over the role's complete-case final
rows (w=1 or PWGTP); the endpoint is the equal mean of the three anchor ratios.
Standard errors come from one common household multinomial bootstrap over all
endpoints (predecessor ``paired_household_bounds``; 10,000 accepted draws; seed
fixed below; identical pairs keep exact zero variance).

Primary (per candidate M in {Q, D17}; alpha .025 each, family-wise .05 by
Bonferroni over the two candidates):
  task    dU(M) = CE_task(M) - CE_task(J);   clause passes iff UB_{.975}(dU) <= -.003
  privacy dG(M) = CE_J - CE_M per role;      clause passes iff UB_{.975}(dG) <= .001
  UB_{.975} = estimate + z_{.975} * SE (one-sided, pointwise). Decision = all ten.
Secondary (separate family): Q versus each comparator; task CE_Q - CE_comp,
sensitive CE_comp - CE_Q; two-sided simultaneous 95% Bonferroni over the generated
family size.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .common import ANCHORS, CANDIDATES, PANEL, PRIMARY_ROLES, SECONDARY_COMPARATORS, TASK_ROLE, WEIGHTINGS

ALPHA_FAMILY = .05
ALPHA_PER_CANDIDATE = .025
TASK_MARGIN = -.003
PRIVACY_CAP = .001
SECONDARY_ALPHA = .05
BOOT = {'n_boot': 10000, 'seed': 20260923}


def primary_endpoints():
    out = []
    for m in CANDIDATES:
        for w in WEIGHTINGS:
            out.append({'id': f'primary|{m}|task|{w}', 'family': 'primary', 'candidate': m, 'clause': 'task',
                        'role': TASK_ROLE, 'weighting': w, 'plus': m, 'minus': 'J', 'threshold': TASK_MARGIN,
                        'orientation': 'CE_task(M) - CE_task(J); negative favors M'})
        for role in PRIMARY_ROLES:
            for w in WEIGHTINGS:
                out.append({'id': f'primary|{m}|{role}|{w}', 'family': 'primary', 'candidate': m,
                            'clause': 'privacy', 'role': role, 'weighting': w, 'plus': 'J', 'minus': m,
                            'threshold': PRIVACY_CAP,
                            'orientation': '(CE_H-CE_M)-(CE_H-CE_J) = CE_J - CE_M; negative favors M'})
    return out


def secondary_endpoints():
    out = []
    for c in SECONDARY_COMPARATORS:
        for w in WEIGHTINGS:
            out.append({'id': f'secondary|Q-vs-{c}|task|{w}', 'family': 'secondary', 'comparator': c,
                        'role': TASK_ROLE, 'weighting': w, 'plus': 'Q', 'minus': c,
                        'orientation': 'CE_task(Q) - CE_task(comp); negative favors Q'})
        for role in PRIMARY_ROLES:
            for w in WEIGHTINGS:
                out.append({'id': f'secondary|Q-vs-{c}|{role}|{w}', 'family': 'secondary', 'comparator': c,
                            'role': role, 'weighting': w, 'plus': c, 'minus': 'Q',
                            'orientation': 'recovery(Q)-recovery(comp) = CE_comp - CE_Q; negative favors Q'})
    return out


def descriptive_endpoints():
    """Unadjusted, clearly labeled: recovery over H and task difference versus J for every panel member."""
    out = []
    for m in PANEL:
        if m == 'H':
            continue
        for role in PRIMARY_ROLES:
            for w in WEIGHTINGS:
                out.append({'id': f'descriptive|recovery_over_H|{m}|{role}|{w}', 'family': 'descriptive',
                            'role': role, 'weighting': w, 'plus': 'H', 'minus': m,
                            'orientation': 'CE_H - CE_M; positive = measured recovery beyond H'})
        for w in WEIGHTINGS:
            out.append({'id': f'descriptive|task_minus_H|{m}|{w}', 'family': 'descriptive', 'role': TASK_ROLE,
                        'weighting': w, 'plus': m, 'minus': 'H', 'orientation': 'CE_task(M) - CE_task(H)'})
            if m != 'J':
                out.append({'id': f'descriptive|task_minus_J|{m}|{w}', 'family': 'descriptive', 'role': TASK_ROLE,
                            'weighting': w, 'plus': m, 'minus': 'J', 'orientation': 'CE_task(M) - CE_task(J)'})
    for w in WEIGHTINGS:
        for role in PRIMARY_ROLES:
            out.append({'id': f'descriptive|D17-vs-Q|{role}|{w}', 'family': 'descriptive', 'role': role,
                        'weighting': w, 'plus': 'Q', 'minus': 'D17',
                        'orientation': 'CE_Q - CE_D17 = recovery(D17) - recovery(Q)'})
    return out


def family_counts():
    p, s, d = primary_endpoints(), secondary_endpoints(), descriptive_endpoints()
    return {'primary': len(p), 'primary_per_candidate': len(p)//len(CANDIDATES), 'secondary': len(s),
            'secondary_comparators': len(SECONDARY_COMPARATORS), 'descriptive': len(d),
            'secondary_arithmetic': f'{len(SECONDARY_COMPARATORS)} comparators x (2 task + 4 roles x 2 weightings) = {len(s)}'}


def z_primary():
    return float(norm.ppf(1-ALPHA_PER_CANDIDATE))


def z_secondary(m):
    return float(norm.isf(SECONDARY_ALPHA/(2*m)))


def upper_bound(estimate, se, z):
    """One-sided upper bound; exact zero-variance pairs return the estimate itself."""
    if se == 0:
        return float(estimate)
    return float(estimate+z*se)


def clause_passes(endpoint, ub):
    return bool(ub <= endpoint['threshold'])


def conjunction(clauses):
    """Intersection-union decision: the candidate passes only if every clause passes.

    Missing/invalid measurements are failures (unresolved), never passes.
    """
    if not clauses:
        return False
    return all(c.get('passed') is True for c in clauses)


def decide(endpoints, bounds):
    """bounds: {id: {'estimate', 'bootstrap_se', ...}} -> per-candidate decisions."""
    z = z_primary()
    result = {}
    for m in CANDIDATES:
        clauses = []
        for e in endpoints:
            if e.get('candidate') != m:
                continue
            b = bounds.get(e['id'])
            if b is None or not np.isfinite(b.get('estimate', np.nan)) or not np.isfinite(b.get('bootstrap_se', np.nan)):
                clauses.append({**e, 'passed': False, 'status': 'invalid_or_missing_measurement'})
                continue
            ub = upper_bound(b['estimate'], b['bootstrap_se'], z)
            lb = b['estimate'] if b['bootstrap_se'] == 0 else b['estimate']-z*b['bootstrap_se']
            passed = clause_passes(e, ub)
            if passed:
                status = 'passed'
            elif lb > 0:
                status = 'failed_demonstrated_adverse_vs_J'
            elif lb > e['threshold']:
                status = 'failed_demonstrated_beyond_margin'
            else:
                status = 'unresolved'
            clauses.append({**e, 'estimate': b['estimate'], 'bootstrap_se': b['bootstrap_se'],
                            'upper_bound_97_5': ub, 'lower_bound_97_5': lb,
                            'passed': passed, 'status': status,
                            'anchor_estimates': b.get('anchor_estimates')})
        result[m] = {'decision': 'pass' if conjunction(clauses) else 'fail', 'clauses': clauses,
                     'clauses_passed': sum(c['passed'] for c in clauses), 'clauses_total': len(clauses),
                     'alpha': ALPHA_PER_CANDIDATE, 'z_one_sided': z}
    return result


def secondary_intervals(endpoints, bounds):
    m = len(endpoints)
    z = z_secondary(m)
    rows = []
    for e in endpoints:
        b = bounds[e['id']]
        se = b['bootstrap_se']
        lo, hi = (b['estimate'], b['estimate']) if se == 0 else (b['estimate']-z*se, b['estimate']+z*se)
        rows.append({**e, 'estimate': b['estimate'], 'bootstrap_se': se, 'lower': lo, 'upper': hi,
                     'anchor_estimates': b.get('anchor_estimates'),
                     'excludes_zero': bool(lo > 0 or hi < 0)})
    return {'family_size': m, 'z': z, 'alpha': SECONDARY_ALPHA, 'rows': rows}


def descriptive_intervals(endpoints, bounds):
    z = float(norm.isf(.025))
    return [{**e, 'estimate': bounds[e['id']]['estimate'], 'bootstrap_se': bounds[e['id']]['bootstrap_se'],
             'lower_unadjusted_95': bounds[e['id']]['estimate']-z*bounds[e['id']]['bootstrap_se'],
             'upper_unadjusted_95': bounds[e['id']]['estimate']+z*bounds[e['id']]['bootstrap_se'],
             'anchor_estimates': bounds[e['id']].get('anchor_estimates')} for e in endpoints]


def contrasts_from_losses(endpoints, losses):
    """losses[(short, anchor, role)] = dict(ids, households, weights, loss). Masks must match exactly."""
    out = []
    for e in endpoints:
        anchors = []
        for a in ANCHORS:
            plus, minus = losses[(e['plus'], a, e['role'])], losses[(e['minus'], a, e['role'])]
            if not np.array_equal(plus['ids'], minus['ids']):
                raise ValueError(f'Complete-case masks differ between interfaces for {e["id"]}')
            w = np.ones(len(plus['ids'])) if e['weighting'] == 'unweighted' else plus['weights']
            anchors.append({'household': plus['households'].astype(str), 'difference': plus['loss']-minus['loss'],
                            'weights': w})
        out.append({'id': e['id'], 'anchors': anchors})
    return out


def interpret_secondary(sec):
    """Per comparator: task non-inferiority at .001, no sensitive cost beyond .001, strictly lower recovery."""
    out = {}
    for c in SECONDARY_COMPARATORS:
        rows = [r for r in sec['rows'] if r['comparator'] == c]
        task = [r for r in rows if r['role'] == TASK_ROLE]
        sens = [r for r in rows if r['role'] != TASK_ROLE]
        out[c] = {'task_noninferior_at_001_both_weightings': all(r['upper'] <= .001 for r in task),
                  'task_strictly_better_both_weightings': all(r['upper'] < 0 for r in task),
                  'task_strictly_worse_any_weighting': any(r['lower'] > 0 for r in task),
                  'no_sensitive_cost_beyond_001_all_eight': all(r['upper'] <= .001 for r in sens),
                  'strictly_lower_recovery_endpoints': [f"{r['role']}|{r['weighting']}" for r in sens if r['upper'] < 0],
                  'strictly_higher_recovery_endpoints': [f"{r['role']}|{r['weighting']}" for r in sens if r['lower'] > 0]}
    return out
