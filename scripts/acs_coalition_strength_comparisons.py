"""Pure, finite comparison rules for the frozen coalition-strength grid.

No fitting, model loading, coefficient selection or metric imputation occurs here.
"""
from __future__ import annotations

from collections import defaultdict
import math
import statistics


SOURCE_TASKS = ('income_binary', 'civilian_at_work', 'public_coverage')
UTILITY_TASKS = (*SOURCE_TASKS, 'same_residence', 'commute_over20')
AUDIT_ROLES = {'A': ('public_coverage', 'commute_over20', 'SEX', 'RAC1P'),
               'B': ('income_binary', 'civilian_at_work', 'same_residence', 'SEX', 'RAC1P'),
               'AB': ('SEX', 'RAC1P')}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def difference(left, right):
    return left-right if finite(left) and finite(right) else None


def complete_and(values):
    """Missing relevant evidence is unassessable, even beside a known failure."""
    values = list(values)
    return None if any(v is None for v in values) else all(values)


def source_check(point, parent, allowance=.01, roundoff=1e-12):
    rows = {}
    for task in SOURCE_TASKS:
        value, reference = point.get('utility', {}).get(task), parent.get(task)
        delta = difference(value, reference)
        rows[task] = {'loss': value if finite(value) else None,
            'parent_loss': reference if finite(reference) else None,
            'loss_minus_parent': delta,
            'excess_over_allowance': delta-allowance if delta is not None else None,
            'pass': value <= reference+allowance+roundoff if delta is not None else None}
    return {'tasks': rows, 'pass': complete_and(r['pass'] for r in rows.values())}


def evaluate_pair(j, local, parent, panel_tasks, delta, attribute='SEX', roundoff=1e-12):
    """Compare one fixed J/local pair, seed, weighting, scope and audit budget.

    A point contains utility[task] and gains['view/target']; gains already refer
    to the same fixed prior. Source checks never use native training heads.
    """
    if not finite(delta) or delta < 0 or not finite(roundoff) or roundoff < 0:
        raise ValueError('Nonnegative finite utility delta and roundoff required')
    if attribute not in ('SEX', 'RAC1P') or not set(panel_tasks) <= set(UTILITY_TASKS):
        raise ValueError('Undeclared attribute or utility component')
    sj, sl = source_check(j, parent, roundoff=roundoff), source_check(local, parent, roundoff=roundoff)
    utility_differences = {t: difference(j.get('utility', {}).get(t), local.get('utility', {}).get(t)) for t in UTILITY_TASKS}
    audit_differences = {f'{v}/{t}': difference(j.get('gains', {}).get(f'{v}/{t}'), local.get('gains', {}).get(f'{v}/{t}'))
                         for v, targets in AUDIT_ROLES.items() for t in targets}
    close = {t: abs(utility_differences[t]) <= delta+roundoff if utility_differences[t] is not None else None for t in panel_tasks}
    directional = {t: utility_differences[t] <= delta+roundoff if utility_differences[t] is not None else None for t in panel_tasks}
    required_tasks = set(SOURCE_TASKS) | set(panel_tasks)
    missing = [f'{name}/utility/{task}' for name, point in (('J', j), ('Iplus', local))
               for task in UTILITY_TASKS if task in required_tasks and not finite(point.get('utility', {}).get(task))]
    missing.extend(f'parent/{t}' for t in SOURCE_TASKS if not finite(parent.get(t)))
    attack_key = 'AB/'+attribute
    missing.extend(f'{name}/gain/{attack_key}' for name, point in (('J', j), ('Iplus', local))
                   if not finite(point.get('gains', {}).get(attack_key)))
    gain_difference = audit_differences[attack_key]
    strict = gain_difference < -roundoff if gain_difference is not None else None
    source_both = complete_and((sj['pass'], sl['pass']))
    statuses = {}
    exclusions = {}
    for name, utility_test in (('close', close), ('directional', directional)):
        reasons = [f'{label}/source/{t}' for label, check in (('J', sj), ('Iplus', sl))
                   for t, row in check['tasks'].items() if row['pass'] is False]
        reasons.extend('utility/'+t for t, passed in utility_test.items() if passed is False)
        if strict is False:
            reasons.append('no_strict_'+attribute+'_gain_improvement')
        reasons.extend('missing/'+item for item in missing)
        value = None if missing else complete_and((source_both, complete_and(utility_test.values()), strict))
        statuses['qualifies_'+name] = value
        statuses['assessment_'+name] = 'unassessable' if value is None else 'qualifies' if value else 'excluded'
        exclusions[name] = reasons
    return {'source_J': sj, 'source_Iplus': sl, 'both_source_feasible': source_both,
        'utility_differences': utility_differences, 'audit_gain_differences': audit_differences,
        'close_by_task': close, 'directional_by_task': directional,
        'utility_close': complete_and(close.values()), 'utility_directional': complete_and(directional.values()),
        'J_gain': j.get('gains', {}).get(attack_key) if finite(j.get('gains', {}).get(attack_key)) else None,
        'Iplus_gain': local.get('gains', {}).get(attack_key) if finite(local.get('gains', {}).get(attack_key)) else None,
        'gain_difference': gain_difference, 'strict_gain_improvement': strict,
        'missing_required': missing, 'exclusion_reasons': exclusions, **statuses}


def fixed_seed_summary(rows, identity_fields, expected_seeds=(0, 1, 2)):
    """Never stitch strengths across seeds or average only eligible observations."""
    groups = defaultdict(dict)
    for row in rows:
        identity = tuple(row[k] for k in identity_fields)
        if row['seed'] in groups[identity]:
            raise ValueError('Duplicate seed for one fixed comparison')
        if row['seed'] not in expected_seeds:
            raise ValueError('Unexpected experimental seed')
        groups[identity][row['seed']] = row
    result = []
    for identity, by_seed in groups.items():
        entry = dict(zip(identity_fields, identity))
        entry['expected_seeds'] = list(expected_seeds)
        entry['present_seeds'] = sorted(by_seed)
        entry['both_source_feasible_per_seed'] = {seed: by_seed.get(seed, {}).get('both_source_feasible') for seed in expected_seeds}
        entry['both_source_feasible_seed_count'] = sum(value is True for value in entry['both_source_feasible_per_seed'].values())
        for kind in ('close', 'directional'):
            eligible = {seed: complete_and((by_seed.get(seed, {}).get('both_source_feasible'),
                        by_seed.get(seed, {}).get('utility_'+kind))) for seed in expected_seeds}
            entry[kind+'_source_and_utility_eligible_per_seed'] = eligible
            entry[kind+'_source_and_utility_eligible_seed_count'] = sum(value is True for value in eligible.values())
            entry[kind+'_source_and_utility_assessable_seed_count'] = sum(value is not None for value in eligible.values())
            values = {seed: by_seed.get(seed, {}).get('qualifies_'+kind) for seed in expected_seeds}
            entry[kind+'_qualifying_seed_count'] = sum(v is True for v in values.values())
            entry[kind+'_assessable_seed_count'] = sum(v is not None for v in values.values())
            entry[kind+'_per_seed'] = values
            entry[kind+'_all_seeds_qualify'] = complete_and(values.values())
        differences = [by_seed.get(seed, {}).get('gain_difference') for seed in expected_seeds]
        entry['gain_difference_per_seed'] = dict(zip(expected_seeds, differences))
        entry['gain_difference_mean'] = statistics.mean(differences) if all(finite(v) for v in differences) else None
        entry['gain_difference_sd'] = statistics.stdev(differences) if all(finite(v) for v in differences) else None
        for metric, components in (('utility_differences', UTILITY_TASKS),
                                   ('audit_gain_differences', [f'{v}/{t}' for v, ts in AUDIT_ROLES.items() for t in ts])):
            summaries = {}
            for component in components:
                values = [by_seed.get(seed, {}).get(metric, {}).get(component) for seed in expected_seeds]
                complete = all(finite(v) for v in values)
                summaries[component] = {'per_seed': dict(zip(expected_seeds, values)),
                    'mean': statistics.mean(values) if complete else None,
                    'sample_sd': statistics.stdev(values) if complete else None}
            entry[metric] = summaries
        result.append(entry)
    return result


def dominates(left, right, components, roundoff=1e-12):
    if not components:
        raise ValueError('A Pareto vector must name at least one component')
    if not all(finite(point.get(c)) for point in (left, right) for c in components):
        return None
    return (all(left[c] <= right[c]+roundoff for c in components)
            and any(left[c] < right[c]-roundoff for c in components))


def pareto_membership(points, components, roundoff=1e-12):
    """Unknown alternatives prevent a complete nondominance assertion."""
    if len({p['condition'] for p in points}) != len(points):
        raise ValueError('Repeated physical model, including duplicate zero anchor')
    complete = {p['condition']: all(finite(p.get(c)) for c in components) for p in points}
    all_complete = all(complete.values())
    rows = []
    for point in points:
        missing = [c for c in components if not finite(point.get(c))]
        witnesses = [other['condition'] for other in points if other['condition'] != point['condition']
                     and dominates(other, point, components, roundoff) is True]
        status = ('unassessable_point' if missing else 'dominated' if witnesses else
                  'nondominated' if all_complete else 'not_dominated_among_assessable_points')
        rows.append({'condition': point['condition'], 'components': list(components),
            'values': {c: point.get(c) if finite(point.get(c)) else None for c in components},
            'missing_components': missing, 'dominating_conditions': witnesses,
            'all_alternatives_complete': all_complete, 'status': status})
    return rows
