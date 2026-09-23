"""Locked 2018 development endpoint family and paired household inference.

This is an empirical development comparison, not a fresh confirmation study.
Endpoint enumeration occurs before loading outer assessment losses. Each endpoint
has a fixed sign: task CE(candidate)-CE(comparator), sensitive recovery difference
CE(comparator)-CE(candidate). Negative sensitive differences favor candidate.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import numpy as np

from experiments.pcrl_task_directed_release_v1.uncertainty import paired_household_bounds
from .audit import ROLES

TASK_ROLE = 'utility:A/same_residence'
SENSITIVE_ROLES = tuple(role for role in ROLES if role.startswith('attack:'))
WEIGHTINGS = ('U', 'PWGTP')
PRIMARY_MARGIN = {'task': .001, 'target': -.002, 'guard': .001}


def enumerate_endpoints(slots: Sequence[Mapping], *, alias_of: Mapping[str, str] | None = None) -> list[dict]:
    """Create full primary comparison family from locked slots and comparators.

    slots: [{'id': candidate, 'comparators': [names...]}], at most two.
    alias_of maps a comparator name to a canonical release identity. Repeated
    aliases collapse to one endpoint but every original name is retained.
    """
    if not 1 <= len(slots) <= 2:
        raise ValueError('Primary family requires one or two locked candidate slots')
    aliases = {} if alias_of is None else dict(alias_of)
    endpoints: list[dict] = []
    seen_candidates: set[str] = set()
    for slot in slots:
        candidate = slot['id']
        if not isinstance(candidate, str) or not candidate or candidate in seen_candidates:
            raise ValueError('Selected candidate IDs must be unique nonempty strings')
        seen_candidates.add(candidate)
        named = list(slot['comparators'])
        if not named:
            raise ValueError('Each candidate needs registered matched comparators')
        groups: dict[str, list[str]] = {}
        for name in named:
            if not isinstance(name, str) or not name:
                raise ValueError('Comparator names must be nonempty strings')
            canonical = aliases.get(name, name)
            # Exact aliases remain a declared comparator: their zero difference
            # cannot satisfy the material AB/RAC1P improvement clause.
            groups.setdefault(canonical, []).append(name)
        for comparator, all_names in groups.items():
            for role in ROLES:
                kind = ('task' if role == TASK_ROLE else
                        'target' if role == 'attack:AB/RAC1P' else 'guard')
                for weighting in WEIGHTINGS:
                    endpoints.append({
                        'id': f'{candidate}|{comparator}|{role}|{weighting}',
                        'candidate': candidate, 'comparator': comparator,
                        'comparator_names': sorted(set(all_names)),
                        'role': role, 'weighting': weighting, 'clause': kind,
                        'threshold': PRIMARY_MARGIN[kind],
                        'plus': candidate if kind == 'task' else comparator,
                        'minus': comparator if kind == 'task' else candidate,
                        'orientation': ('CE_candidate - CE_comparator' if kind == 'task'
                                        else 'CE_comparator - CE_candidate = recovery_candidate - recovery_comparator'),
                    })
    ids = [e['id'] for e in endpoints]
    if len(set(ids)) != len(ids):
        raise ValueError('Endpoint IDs collided')
    return endpoints


def capability_endpoints(candidate_ids: Sequence[str]) -> list[dict]:
    """Point-estimate H capability screen, reported with separate uncertainty."""
    return [{
        'id': f'{candidate}|H|{TASK_ROLE}|{weighting}',
        'candidate': candidate, 'comparator': 'H', 'role': TASK_ROLE,
        'weighting': weighting, 'plus': candidate, 'minus': 'H',
        'clause': 'capability', 'threshold': -.01,
        'orientation': 'CE_candidate - CE_H',
    } for candidate in candidate_ids for weighting in WEIGHTINGS]


def _private_row(item: Mapping) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(item['ids'])
    household = np.asarray(item['households']).astype(str)
    weights = np.asarray(item['weights'], dtype=np.float64)
    loss = np.asarray(item['loss'], dtype=np.float64)
    if (ids.ndim != 1 or len(ids) == 0 or household.shape != ids.shape
            or weights.shape != ids.shape or loss.shape != ids.shape
            or len(np.unique(ids)) != len(ids) or not np.isfinite(weights).all()
            or (weights < 0).any() or weights.sum() <= 0
            or not np.isfinite(loss).all() or (loss < 0).any()):
        raise ValueError('Invalid original-person private score record')
    return ids, household, weights, loss


def contrasts_from_scores(endpoints: Sequence[Mapping], scores: Mapping, *, anchors=(0, 1, 2)) -> list[dict]:
    """Build private paired contrasts; never expose person rows in public result."""
    if len(anchors) != 3 or len(set(anchors)) != 3:
        raise ValueError('Three distinct historical anchors are required')
    contrasts = []
    for endpoint in endpoints:
        records = []
        for anchor in anchors:
            role = endpoint['role']
            left = _private_row(scores[(endpoint['plus'], anchor, role)])
            right = _private_row(scores[(endpoint['minus'], anchor, role)])
            ids, hh, w, loss = left
            ids_b, hh_b, w_b, loss_b = right
            if (not np.array_equal(ids, ids_b) or not np.array_equal(hh, hh_b)
                    or not np.array_equal(w, w_b)):
                raise ValueError(f'Paired masks/households/weights differ: {endpoint["id"]}, anchor {anchor}')
            used_w = np.ones(len(ids), dtype=np.float64) if endpoint['weighting'] == 'U' else w
            records.append({'household': hh, 'difference': loss - loss_b, 'weights': used_w})
        contrasts.append({'id': endpoint['id'], 'anchors': records})
    return contrasts


def evaluate_family(endpoints: Sequence[Mapping], private_scores: Mapping, *, n_boot=10000,
                    seed=20260923, alpha=.05) -> dict:
    """One common household bootstrap for the complete frozen endpoint family."""
    if not endpoints:
        raise ValueError('Empty endpoint family')
    contrasts = contrasts_from_scores(endpoints, private_scores)
    result = paired_household_bounds(contrasts, n_boot=n_boot, seed=seed, alpha=alpha)
    if result['family_size'] != len(endpoints):
        raise AssertionError('Multiplicity denominator changed')
    rows = []
    for endpoint in endpoints:
        bounds = result['bounds'][endpoint['id']]
        upper = bounds['upper']
        lower = bounds['lower']
        rows.append({**endpoint, 'estimate': bounds['estimate'],
                     'bootstrap_se': bounds['bootstrap_se'], 'lower': lower, 'upper': upper,
                     'passed_upper_bound': bool(upper <= endpoint['threshold']),
                     'point_screen_passed': bool(bounds['estimate'] <= endpoint['threshold']),
                     'demonstrated_adverse': bool(lower > endpoint['threshold']),
                     'anchor_estimates': bounds['anchor_estimates']})
    by_candidate = {}
    for candidate in sorted({e['candidate'] for e in endpoints}):
        clauses = [r for r in rows if r['candidate'] == candidate]
        by_candidate[candidate] = {
            'all_primary_clauses_passed': bool(clauses) and all(r['passed_upper_bound'] for r in clauses),
            'passed': sum(r['passed_upper_bound'] for r in clauses), 'total': len(clauses),
        }
    return {'family_size': len(rows), 'alpha': alpha,
            'critical_value_two_sided': result['critical_value_adjusted'],
            'bootstrap': result['bootstrap'], 'rows': rows, 'decisions': by_candidate,
            'scope': '2018 repeatedly used development; conditional on frozen fitted models and selection'}


def family_manifest(slots: Sequence[Mapping], *, alias_of: Mapping[str, str] | None = None) -> dict:
    endpoints = enumerate_endpoints(slots, alias_of=alias_of)
    return {'schema': 1, 'n_endpoints': len(endpoints), 'endpoints': endpoints,
            'multiplicity': 'two-sided Bonferroni over complete selected-panel endpoint family',
            'sampling_unit': 'household; common multinomial draw across roles, arms and anchors',
            'anchor_aggregation': 'equal mean of three anchor-specific weighted ratios; anchors share households'}


def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('fixture',))
    args = parser.parse_args()
    if args.command == 'fixture':
        manifest = family_manifest([{'id': 'candidate', 'comparators': ['D17', 'D_U1', 'deterministic', 'gradient']}])
        print(json.dumps({'status': 'pass', 'n_endpoints': manifest['n_endpoints'],
                          'first': manifest['endpoints'][0]['id']}, sort_keys=True))


if __name__ == '__main__':
    _main()
