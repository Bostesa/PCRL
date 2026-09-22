"""Validation-only route selection and prospective, explicit evaluation contrasts.

Inputs are aggregate ``fit_release_audits()['summary']`` dictionaries, never test
losses or fitted objects. A configuration has one anchor-averaged score per role
and weighting; a route selects one configuration for all anchors and weightings.
The selected deployment predictor remains the already chosen validation recipe.
Independent/fixed-decoder diagnostics cannot replace it after route selection.

An empty required label-matched family prevents competitive superiority. A route
without a passing configuration retains its nearest descriptive nominee. Neither
that nominee nor a later favorable evaluation result retroactively passes its
validation screen. Both routes remain in the same explicit multiplicity family.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from .config import PRIMARY, configuration, digest, mechanism_id

ANCHORS = (0, 1, 2)
WEIGHTS = ('unweighted', 'PWGTP')
TASK_ROLE = 'utility:A/same_residence'
ATTACK_ROLES = tuple('attack:' + role for role in PRIMARY)
REQUIRED_ROLES = (TASK_ROLE,) + ATTACK_ROLES
ROUTES = ('utility_first', 'protection_first')
HISTORICAL = ('leace_A0', 'splince_A0', 'optnet16_L1', 'optnet16_L2', 'optnet16_C1')


def default_families(*, registered_extensions=None, registered_controls=None):
    """Fixed label-matched control families, including unavailable attempted arms."""
    codes = ('T0', 'Ttask', 'Trisk')
    groups = {
        'continuous_task': ['continuous_task'],
        'deterministic_actions': [mechanism_id(c, 'U', None) for c in codes],
        'direct_code': [c + '_code' for c in codes],
        'withholding': [f'{c}_withhold_{p:g}' for c in codes for p in (.25, .5, .75)],
        'randomized_response': [f'{c}_rr_{p:g}' for c in codes for p in (.25, .5, .75)],
        'supervised_LEACE': ['leace_supervised'],
        'supervised_SPLINCE': ['splince_supervised'],
        'constant_null': ['constant_best', 'independent_token'],
    }
    for name, spec in sorted((registered_extensions or {}).items()):
        if spec['policy'] == 'U' and name not in groups['deterministic_actions']:
            groups['deterministic_actions'].append(name)
    control_groups = {'withhold': 'withholding', 'rr': 'randomized_response',
                      'constant_best': 'constant_null', 'independent_token': 'constant_null'}
    for name, spec in sorted((registered_controls or {}).items()):
        groups[control_groups[spec['baseline_family']]].append(name)
    return {name: {'configurations': values, 'label_matched': True, 'required': True}
            for name, values in groups.items()}


def _registered(spec):
    return isinstance(spec, dict) and any(spec.get(key) for key in
        ('registration_id', 'extra_registration_sha256', 'robustness_registration_sha256'))


def _conditioning(spec):
    return spec.get('conditioning_family', 'primary')


def _matched_name(code, policy, budget, actions, conditioning='primary'):
    base = mechanism_id(code, policy, budget, actions)
    return base + '_fineC' if conditioning == 'primary_intersect_fineC4' else base


def _names(values, label):
    if (not isinstance(values, (list, tuple)) or
            any(not isinstance(v, str) or not v.strip() for v in values) or
            len(set(values)) != len(values)):
        raise ValueError(f'{label} must be an explicit list of unique nonempty names')
    return list(values)


def _number(value):
    if isinstance(value, (bool, str)) or not isinstance(value, (int, float)):
        raise ValueError('Validation CE must be finite nonnegative numeric values')
    if not math.isfinite(value) or value < 0:
        raise ValueError('Validation CE must be finite nonnegative numeric values')
    return float(value)


def _prepare_validation(validation):
    if not isinstance(validation, dict):
        raise ValueError('Provide configuration -> anchor -> validation-summary mappings')
    aggregates, choices, unavailable = {}, {}, {}
    for name, raw in sorted(validation.items()):
        if not isinstance(name, str) or not name or not isinstance(raw, dict):
            raise ValueError('Invalid validation configuration record')
        anchors = {}
        for key, summary in raw.items():
            if key not in (0, 1, 2, '0', '1', '2') or isinstance(key, bool) or int(key) in anchors:
                raise ValueError('Validation anchors must be unique 0,1,2')
            anchors[int(key)] = summary
        missing = sorted(set(ANCHORS) - set(anchors))
        if missing:
            unavailable[name] = {'reason': 'incomplete_three_anchor_audit', 'missing_anchors': missing}
            continue
        schemas = [set(anchors[a]) for a in ANCHORS]
        if not all(set(REQUIRED_ROLES) <= schema for schema in schemas):
            raise ValueError(f'{name}: missing registered primary validation roles')
        if schemas[1:] != schemas[:1] * 2:
            raise ValueError(f'{name}: role schema differs across anchors')
        role_means, selected = {}, {}
        for a in ANCHORS:
            selected[str(a)] = {}
            for role, entry in anchors[a].items():
                if not isinstance(entry, dict) or not isinstance(entry.get('selection'), str):
                    raise ValueError('Every validation role requires its frozen selected predictor ID')
                if not entry['selection']:
                    raise ValueError('Empty selected predictor ID')
                if 'validation' not in entry:
                    raise ValueError('Validation summary required; evaluation metrics are not accepted')
                selected[str(a)][role] = {
                    'selection': entry['selection'],
                    'independent_selection': entry.get('independent_selection'),
                    'metric_used': 'validation of selected deployment predictor',
                }
        for role in sorted(schemas[0]):
            role_means[role] = {
                weighting: math.fsum(_number(anchors[a][role]['validation'][field])
                                     for a in ANCHORS) / 3
                for weighting, field in (('unweighted', 'unweighted'), ('PWGTP', 'weighted'))}
        utility = role_means[TASK_ROLE]
        aggregates[name] = {'utility': utility,
                            'balanced_utility': .5 * (utility['unweighted'] + utility['PWGTP']),
                            'sensitive': {role: role_means[role] for role in ATTACK_ROLES},
                            'roles': role_means}
        choices[name] = selected
    for required in ('H', 'J'):
        if required not in aggregates:
            raise ValueError(f'Complete three-anchor {required} validation baseline required')
    return aggregates, choices, unavailable


def _deltas(record, j):
    utility = {w: record['utility'][w] - j['utility'][w] for w in WEIGHTS}
    # Recovery(candidate)-Recovery(J) = loss(J)-loss(candidate), same H cancels.
    sensitive = {role: {w: j['sensitive'][role][w] - record['sensitive'][role][w]
                        for w in WEIGHTS} for role in ATTACK_ROLES}
    values = [sensitive[r][w] for r in ATTACK_ROLES for w in WEIGHTS]
    return {'utility_delta_J': utility, 'sensitive_increment_J': sensitive,
            'maximum_sensitive_increment_J': max(values),
            'minimum_sensitive_increment_J': min(values)}


def _rank(name, aggregates, route):
    row = aggregates[name]
    if route == 'utility_first':
        return (row['balanced_utility'], name)
    return (row['maximum_sensitive_increment_J'], row['balanced_utility'], name)


def _screen(row, route, cfg):
    sensitive = row['maximum_sensitive_increment_J']
    utility = max(row['utility_delta_J'].values())
    if route == 'utility_first':
        violations = [utility + cfg['utility_material_gain'], sensitive - cfg['sensitive_allowance']]
        passes = all(v <= 0 for v in violations)
    else:
        violations = [utility - cfg['utility_allowance'], sensitive - cfg['sensitive_allowance'],
                      row['minimum_sensitive_increment_J']]
        passes = all(v <= 0 for v in violations) and row['minimum_sensitive_increment_J'] < 0
    return bool(passes), max(0., max(violations))


def select_validation(validation, *, candidate_ids, families=None, registered_extensions=None,
                      registered_controls=None):
    """Select two common-configuration routes from validation summaries alone.

    ``validation[name][anchor]`` is exactly the role-summary dictionary returned
    by evaluation.fit_release_audits. Missing arms/anchors stay unavailable;
    complete H and J are mandatory. ``families`` explicitly names control sets
    with ``configurations``, ``label_matched``, ``required`` fields. The default
    covers the eight prospectively specified label-matched families.

    Positive L/C registered maps alone are primary candidates. Additional maps
    require explicit ``registered_extensions[name]`` specs with a prospective
    registration ID/hash, input, policy, budget and max_actions. Registered branch
    A U33 and ``registered_controls`` join their existing default control families.
    """
    cfg = configuration()
    candidates = _names(candidate_ids, 'candidate_ids')
    specs = {m['configuration']: {k: m[k] for k in ('input', 'policy', 'budget', 'max_actions')}
             for m in cfg['maps']}
    for name, spec in (registered_extensions or {}).items():
        if not _registered(spec):
            raise ValueError('Extensions require an explicit prospective registration_id')
        if (spec.get('input') not in ('T0', 'Ttask', 'Trisk') or spec.get('policy') not in ('L', 'C', 'U')
                or spec.get('max_actions') not in (17, 33)
                or (spec.get('budget') is not None and
                    (not isinstance(spec['budget'], (int, float)) or not math.isfinite(spec['budget'])
                     or spec['budget'] < 0))
                or ((spec['policy'] == 'U') != (spec.get('budget') is None))):
            raise ValueError('Invalid prospectively registered extension specification')
        conditioning = _conditioning(spec)
        if conditioning not in ('primary', 'primary_intersect_fineC4'):
            raise ValueError('Unknown registered conditioning family')
        if conditioning != 'primary' and (spec.get('branch') != 'C' or spec['input'] != 'Trisk'
                or spec['max_actions'] != 17 or spec['policy'] not in ('L', 'C')
                or spec.get('matched_frontier') != 'Trisk_fineC_a17'):
            raise ValueError('Invalid fine-conditioning extension specification')
        if name != _matched_name(spec['input'], spec['policy'], spec['budget'], spec['max_actions'], conditioning):
            raise ValueError('Extension ID does not match its registered mechanism specification')
        if name in specs and specs[name] != {k: spec[k] for k in specs[name]}:
            raise ValueError('An extension cannot change an existing registered map')
        specs[name] = copy.deepcopy(spec)
    standard_controls = {n for family in default_families().values() for n in family['configurations']}
    for name, spec in (registered_controls or {}).items():
        if not _registered(spec):
            raise ValueError('Controls require an explicit prospective registration_id')
        canonical = spec.get('canonical_release')
        family = spec.get('baseline_family')
        valid_family = ((family == 'withhold' and '_withhold_' in str(canonical))
                        or (family == 'rr' and '_rr_' in str(canonical))
                        or (family in ('constant_best', 'independent_token') and family == canonical))
        if (spec.get('configuration', name) != name or canonical not in standard_controls
                or name != canonical + '_a33' or spec.get('branch') != 'A'
                or spec.get('kind') != 'control' or spec.get('max_actions') != 33
                or spec.get('label_matched') is not True or not valid_family):
            raise ValueError('Invalid prospectively registered control specification')
    for name in candidates:
        spec = specs.get(name)
        if (spec is None or spec['input'] not in ('T0', 'Ttask', 'Trisk')
                or spec['policy'] not in ('L', 'C') or spec['budget'] is None or spec['budget'] <= 0):
            raise ValueError('Primary candidates must be registered positive-budget L/C configurations')
    families = copy.deepcopy(default_families(registered_extensions=registered_extensions,
        registered_controls=registered_controls) if families is None else families)
    if not isinstance(families, dict):
        raise ValueError('Explicit family mapping required')
    for name, family in families.items():
        if not isinstance(name, str) or not name or not isinstance(family, dict):
            raise ValueError('Invalid control family')
        family['configurations'] = _names(family['configurations'], 'family configurations')
        if not isinstance(family.get('label_matched'), bool) or not isinstance(family.get('required'), bool):
            raise ValueError('Declare label_matched and required booleans for every family')
    aggregates, choices, unavailable = _prepare_validation(validation)
    for name in set(candidates) | {c for f in families.values() for c in f['configurations']}:
        if name not in aggregates and name not in unavailable:
            unavailable[name] = {'reason': 'no_complete_validation_audit', 'missing_anchors': list(ANCHORS)}
    j = copy.deepcopy(aggregates['J'])
    for row in aggregates.values():
        row.update(_deltas(row, j))
    available_candidates = [name for name in candidates if name in aggregates]
    routes = {}
    for route in ROUTES:
        screens = {name: _screen(aggregates[name], route, cfg['inference']) for name in available_candidates}
        passing = [name for name in available_candidates if screens[name][0]]
        nominee = (min(passing, key=lambda n: _rank(n, aggregates, route)) if passing else
                   min(available_candidates, key=lambda n: (screens[n][1], *_rank(n, aggregates, route)))
                   if available_candidates else None)
        family_nominees = {}
        for family_name, family in sorted(families.items()):
            available = [name for name in family['configurations'] if name in aggregates]
            if route == 'utility_first':
                eligible = [name for name in available if aggregates[name]['maximum_sensitive_increment_J']
                            <= cfg['inference']['sensitive_allowance']]
            else:
                eligible = [name for name in available if max(aggregates[name]['utility_delta_J'].values())
                            <= cfg['inference']['utility_allowance']]
            selected = min(eligible, key=lambda n: _rank(n, aggregates, route)) if eligible else None
            family_nominees[family_name] = {
                'configuration': selected, 'eligible': sorted(eligible),
                'attempted': list(family['configurations']), 'available': sorted(available),
                'unavailable': sorted(set(family['configurations']) - set(available)),
                'label_matched': family['label_matched'], 'required': family['required'],
                'empty_eligibility': not eligible,
            }
        empty = [name for name, family in family_nominees.items()
                 if family['label_matched'] and family['required'] and family['configuration'] is None]
        screen_passed = nominee is not None and screens[nominee][0]
        routes[route] = {
            'nominee': nominee, 'screen_passed': bool(screen_passed),
            'descriptive_only': not screen_passed,
            'selection_reason': 'screen_pass' if screen_passed else 'nearest_descriptive' if nominee else 'no_complete_candidate',
            'passing_configurations': sorted(passing),
            'validation_frontier': {n: {'screen_passed': screens[n][0], 'maximum_violation': screens[n][1]}
                                    for n in sorted(available_candidates)},
            'family_nominees': family_nominees, 'empty_required_families': empty,
            'competitive_validation_eligible': bool(screen_passed and not empty
                and any(f['label_matched'] and f['required'] for f in family_nominees.values())),
            'competitive_claim_requires': 'all required family eligibility nonempty; adjusted candidate and comparator caps; '
                                          'all required adjusted pairwise conjunctions',
        }
    return {
        'schema': 1, 'validation_only': True, 'selection_frozen': False,
        'routes': routes, 'route_multiplicity': list(ROUTES), 'candidate_ids': candidates,
        'families': families, 'mechanism_specs': specs, 'aggregates': aggregates,
        'registered_controls': copy.deepcopy(registered_controls or {}),
        'predictor_choices': choices, 'unavailable_configurations': unavailable,
        'descriptive_configurations': sorted(aggregates),
        'historical_comparators': [n for n in HISTORICAL if n in aggregates],
        'historical_comparators_unavailable': [n for n in HISTORICAL if n not in aggregates],
        'configuration_digest': digest(cfg),
        'validation_summary_digest': digest(json.loads(json.dumps(validation, allow_nan=False))),
        'margins': copy.deepcopy(cfg['inference']),
        'selected_metric': 'selected deployment expected log loss; independent and fixed-decoder diagnostics stay separate',
        'anchor_rule': 'equal mean of exactly three anchor estimates, separately for each weighting',
        'sensitive_orientation': 'loss(comparator)-loss(candidate); common H recovery cancels algebraically',
        'frontier_inference': 'full available grid descriptive; simultaneous bounds apply only to the frozen endpoint list',
    }


def _check(name, threshold, *, strict=False, bound='upper'):
    return {'name': name, 'bound': bound, 'operator': ('<' if strict else '<=') if bound == 'upper'
            else ('>' if strict else '>='), 'threshold': float(threshold)}


def build_contrast_family(selection, *, matched_specs=None):
    """Enumerate every selected endpoint before evaluation; M is exactly list length.

    Terms specify per-person selected losses and signed coefficients. Inference
    recomputes each anchor's actual weighted ratio and then averages three anchors.
    Multi-term historical half-headroom/direct-role endpoints use the same paired
    people, never differences between independently aggregated recoveries.
    """
    if not selection.get('validation_only'):
        raise ValueError('A validation-only selection record is required')
    available = set(selection['aggregates'])
    specs = selection['mechanism_specs']
    for name, spec in (matched_specs or {}).items():
        if name not in specs or spec != specs[name]:
            raise ValueError('Matched specifications must already be registered unchanged at selection')
    endpoints, attribution = [], {}
    margins = selection['margins']

    def add(route, purpose, role, weighting, terms, checks=(), comparator=None):
        if any(term[0] not in available for term in terms):
            raise ValueError('Endpoint attempted to include an unavailable comparison')
        key = {'route': route, 'purpose': purpose, 'role': role, 'weighting': weighting,
               'terms': [{'configuration': n, 'role': r, 'coefficient': float(c),
                          'predictor': 'frozen_validation_selection'} for n, r, c in terms]}
        endpoint = {**key, 'id': 'contrast_' + digest(key)[:24], 'anchors': list(ANCHORS),
                    'checks': list(checks), 'comparator': comparator,
                    'metric': 'paired per-person expected log loss',
                    'aggregation': 'actual within-anchor weighted ratio, equal three-anchor mean'}
        endpoints.append(endpoint)

    def pair(route, purpose, candidate, comparator, *, utility_margin=None, strict_utility=False,
             sensitive_margin=None):
        for w in WEIGHTS:
            checks = [] if utility_margin is None else [_check('utility', utility_margin, strict=strict_utility)]
            add(route, purpose + '_utility', TASK_ROLE, w,
                [(candidate, TASK_ROLE, 1), (comparator, TASK_ROLE, -1)], checks, comparator)
            for role in ATTACK_ROLES:
                checks = [] if sensitive_margin is None else [
                    _check('sensitive_noninferiority', sensitive_margin),
                    _check('strict_sensitive_improvement', 0, strict=True)]
                add(route, purpose + '_sensitive', role, w,
                    [(comparator, role, 1), (candidate, role, -1)], checks, comparator)

    def attribution_caps(route, purpose, configuration_name):
        for w in WEIGHTS:
            if route == 'utility_first':
                for role in ATTACK_ROLES:
                    add(route, 'attribution_' + purpose + '_evaluated_J_cap', role, w,
                        [('J', role, 1), (configuration_name, role, -1)],
                        [_check('attribution_privacy_cap', margins['sensitive_allowance'])], configuration_name)
            else:
                add(route, 'attribution_' + purpose + '_evaluated_J_cap', TASK_ROLE, w,
                    [(configuration_name, TASK_ROLE, 1), ('J', TASK_ROLE, -1)],
                    [_check('attribution_utility_cap', margins['utility_allowance'])], configuration_name)

    # A fixed H/J descriptive core remains even when no complete candidate exists.
    pair('core', 'J_relative_H', 'J', 'H')
    for route in ROUTES:
        record = selection['routes'][route]
        candidate = record['nominee']
        attribution[route] = {}
        if candidate is None:
            continue
        gain = -margins['utility_material_gain'] if route == 'utility_first' else margins['utility_allowance']
        for w in WEIGHTS:
            add(route, 'candidate_J_utility_gate', TASK_ROLE, w,
                [(candidate, TASK_ROLE, 1), ('J', TASK_ROLE, -1)], [_check('utility_gate', gain)], 'J')
            add(route, 'historical_01_utility', TASK_ROLE, w,
                [(candidate, TASK_ROLE, 1), ('J', TASK_ROLE, -1)],
                [_check('separate_inherited_utility_gain', -margins['historical_utility_gain'])], 'J')
            for role in ATTACK_ROLES:
                add(route, 'candidate_J_privacy_cap', role, w,
                    [('J', role, 1), (candidate, role, -1)],
                    [_check('privacy_cap', margins['sensitive_allowance']),
                     _check('strict_sensitive_improvement', 0, strict=True)], 'J')
                add(route, 'historical_half_headroom', role, w,
                    [('H', role, .5), ('J', role, .5), (candidate, role, -1)],
                    [_check('separate_inherited_half_headroom', 0)], 'J')
        compared = {candidate}
        for family_name, family in sorted(record['family_nominees'].items()):
            comparator = family['configuration']
            if comparator is None:
                continue
            compared.add(comparator)
            prefix = 'family_' + family_name
            pair(route, prefix, candidate, comparator,
                 utility_margin=0 if route == 'utility_first' else margins['utility_allowance'],
                 strict_utility=route == 'utility_first', sensitive_margin=margins['sensitive_allowance'])
            for w in WEIGHTS:
                if route == 'utility_first':
                    for role in ATTACK_ROLES:
                        add(route, 'family_evaluated_J_privacy_cap', role, w,
                            [('J', role, 1), (comparator, role, -1)],
                            [_check('comparator_privacy_cap', margins['sensitive_allowance'])], comparator)
                else:
                    add(route, 'family_evaluated_J_utility_cap', TASK_ROLE, w,
                        [(comparator, TASK_ROLE, 1), ('J', TASK_ROLE, -1)],
                        [_check('comparator_utility_cap', margins['utility_allowance'])], comparator)
        for comparator in selection['historical_comparators']:
            pair(route, 'historical_' + comparator, candidate, comparator,
                 utility_margin=0 if route == 'utility_first' else margins['utility_allowance'],
                 strict_utility=route == 'utility_first', sensitive_margin=margins['sensitive_allowance'])
            compared.add(comparator)
        for comparator in sorted(compared):
            pair(route, 'H_relative_' + comparator, comparator, 'H')
        # Direct A-to-AB differences use the same target rows and weighting.
        for target in ('SEX', 'RAC1P'):
            for w in WEIGHTS:
                ra, rab = 'attack:A/' + target, 'attack:AB/' + target
                add(route, 'direct_coalition_gain', rab, w,
                    [(candidate, ra, 1), (candidate, rab, -1)])
        spec = specs[candidate]
        code, policy, budget, actions = (spec[k] for k in ('input', 'policy', 'budget', 'max_actions'))
        conditioning = _conditioning(spec)
        matched = {}
        if code == 'Trisk':
            matched['risk_refinement'] = [_matched_name(c, policy, budget, actions, conditioning)
                                          for c in ('T0', 'Ttask')]
        elif code == 'Ttask':
            matched['task_refinement'] = [_matched_name('T0', policy, budget, actions, conditioning)]
        frontier = None
        if policy == 'C':
            attempted = [_matched_name(code, 'L', b, actions, conditioning) for b in (.0005, .002, .01)]
            local_available = [n for n in attempted if n in available and n in specs
                               and _conditioning(specs[n]) == conditioning]
            if route == 'utility_first':
                eligible = [n for n in local_available if selection['aggregates'][n]['maximum_sensitive_increment_J']
                            <= margins['sensitive_allowance']]
            else:
                eligible = [n for n in local_available if max(selection['aggregates'][n]['utility_delta_J'].values())
                            <= margins['utility_allowance']]
            local_nominee = min(eligible, key=lambda n: _rank(n, selection['aggregates'], route)) if eligible else None
            frontier = {'attempted': attempted, 'available': local_available,
                        'missing': sorted(set(attempted) - set(local_available)),
                        'eligible': sorted(eligible), 'nominee': local_nominee,
                        'complete': len(local_available) == len(attempted),
                        'conditioning_family': conditioning, 'max_actions': actions, 'input': code,
                        'eligibility_rule': 'all eight J privacy caps' if route == 'utility_first'
                                            else 'both J utility allowances',
                        'ranking': 'balanced utility, lexical ID' if route == 'utility_first'
                                   else 'maximum J sensitive increment, balanced utility, lexical ID'}
            matched['coalition'] = [local_nominee] if local_nominee else []
            same_budget = _matched_name(code, 'L', budget, actions, conditioning)
            if same_budget in local_available:
                pair(route, 'attribution_coalition_same_budget', candidate, same_budget)
        matched['randomization'] = [mechanism_id(code, 'U', None, actions), code + '_code']
        simple_frontiers = {}
        suffix = '_a33' if actions == 33 else ''
        for label, kind in (('withholding', 'withhold'), ('randomized_response', 'rr')):
            attempted = [f'{code}_{kind}_{mix:g}{suffix}' for mix in (.25, .5, .75)]
            control_available = [n for n in attempted if n in available
                and (actions == 17 or n in selection.get('registered_controls', {}))]
            if route == 'utility_first':
                eligible = [n for n in control_available if selection['aggregates'][n]['maximum_sensitive_increment_J']
                            <= margins['sensitive_allowance']]
            else:
                eligible = [n for n in control_available if max(selection['aggregates'][n]['utility_delta_J'].values())
                            <= margins['utility_allowance']]
            nominee = min(eligible, key=lambda n: _rank(n, selection['aggregates'], route)) if eligible else None
            simple_frontiers[label] = {'attempted': attempted, 'available': control_available,
                'missing': sorted(set(attempted) - set(control_available)), 'eligible': sorted(eligible),
                'nominee': nominee, 'complete': len(control_available) == len(attempted),
                'input': code, 'max_actions': actions}
            if nominee is not None:
                matched['randomization'].append(nominee)
        for purpose, comparators in matched.items():
            missing = [name for name in comparators if name not in available]
            complete = not missing and bool(comparators)
            if purpose == 'coalition':
                complete = complete and frontier['complete']
            elif purpose == 'randomization':
                complete = complete and all(f['complete'] and f['nominee'] is not None for f in simple_frontiers.values())
                missing += [name for f in simple_frontiers.values() for name in f['missing']]
            attribution[route][purpose] = {'comparators': comparators, 'missing': missing,
                                           'complete': complete, 'input': code, 'policy': policy,
                                           'budget': budget, 'max_actions': actions,
                                           'conditioning_family': conditioning,
                                           'claim_eligible': bool(complete),
                                           'claim_eligible_meaning': 'availability only; adjusted conjunction must pass'}
            if purpose == 'coalition':
                attribution[route][purpose].update(local_frontier=frontier, same_budget_comparator=same_budget,
                    same_budget_available=same_budget in local_available)
            elif purpose == 'randomization':
                attribution[route][purpose]['simple_control_frontiers'] = simple_frontiers
            for comparator in comparators:
                if comparator in available:
                    pair(route, 'attribution_' + purpose + '_' + comparator, candidate, comparator,
                         utility_margin=0 if route == 'utility_first' else margins['utility_allowance'],
                         strict_utility=route == 'utility_first', sensitive_margin=margins['sensitive_allowance'])
            # Even descriptive/incomplete nominations retain prospective cap endpoints;
            # no subsequently favorable result can fill a missing validation frontier.
            for name in sorted({candidate, *(n for n in comparators if n in available)}):
                attribution_caps(route, purpose, name)
    ids = [e['id'] for e in endpoints]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate endpoint definition; resolve overlapping family nominees explicitly')
    def references(route, purpose, check, comparator=None, roles=None):
        records = [e for e in endpoints if e['route'] == route and e['purpose'] == purpose
                   and (comparator is None or e['comparator'] == comparator)
                   and (roles is None or e['role'] in roles)]
        if not records or any(check not in {c['name'] for c in e['checks']} for e in records):
            raise AssertionError('Missing prospectively required endpoint/check')
        return [{'endpoint': e['id'], 'check': check} for e in records]

    claims = {}
    for route in ROUTES:
        record = selection['routes'][route]
        if not record['screen_passed']:
            blocked = {'kind': 'blocked', 'reason': 'validation screen did not pass; descriptive nominee only'}
            claims[route] = {'historical_J': blocked, 'competitive': copy.deepcopy(blocked)}
            continue
        historical = references(route, 'candidate_J_utility_gate', 'utility_gate')
        historical += references(route, 'candidate_J_privacy_cap', 'privacy_cap')
        if route == 'protection_first':
            historical.append({'kind': 'any', 'clauses': references(
                route, 'candidate_J_privacy_cap', 'strict_sensitive_improvement')})
        historical = {'kind': 'all', 'clauses': historical}
        if not record['competitive_validation_eligible']:
            competitive = {'kind': 'blocked', 'reason': 'one or more required label-matched families lack eligibility, '
                           'or no required label-matched family was declared',
                           'empty_required_families': record['empty_required_families']}
        else:
            clauses = [historical]
            for family_name, family in sorted(record['family_nominees'].items()):
                comparator = family['configuration']
                if not family['label_matched'] or comparator is None:
                    continue
                prefix = 'family_' + family_name
                clauses += references(route, prefix + '_utility', 'utility', comparator)
                if route == 'utility_first':
                    clauses += references(route, 'family_evaluated_J_privacy_cap', 'comparator_privacy_cap', comparator)
                else:
                    clauses += references(route, prefix + '_sensitive', 'sensitive_noninferiority', comparator)
                    clauses.append({'kind': 'any', 'clauses': references(
                        route, prefix + '_sensitive', 'strict_sensitive_improvement', comparator)})
                    clauses += references(route, 'family_evaluated_J_utility_cap', 'comparator_utility_cap', comparator)
            competitive = {'kind': 'all', 'clauses': clauses}
        claims[route] = {'historical_J': historical, 'competitive': competitive}
    attribution_claims = {route: {} for route in ROUTES}
    for route in ROUTES:
        candidate = selection['routes'][route]['nominee']
        for purpose, record in attribution[route].items():
            if not record['claim_eligible']:
                reason = ('matched local frontier incomplete or has no eligible nominee' if purpose == 'coalition'
                          else 'matched simple-control frontier incomplete/empty or comparison unavailable'
                          if purpose == 'randomization' else 'one or more matched comparisons unavailable')
                formula = {'kind': 'blocked', 'reason': reason}
            else:
                clauses = []
                for comparator in record['comparators']:
                    prefix = 'attribution_' + purpose + '_' + comparator
                    clauses += references(route, prefix + '_utility', 'utility', comparator)
                    clauses += references(route, prefix + '_sensitive', 'sensitive_noninferiority', comparator)
                    if route == 'protection_first':
                        roles = ('attack:AB/SEX', 'attack:AB/RAC1P') if purpose == 'coalition' else None
                        clauses.append({'kind': 'any', 'clauses': references(route, prefix + '_sensitive',
                                        'strict_sensitive_improvement', comparator, roles=roles)})
                check = 'attribution_privacy_cap' if route == 'utility_first' else 'attribution_utility_cap'
                for name in sorted({candidate, *record['comparators']}):
                    clauses += references(route, 'attribution_' + purpose + '_evaluated_J_cap', check, name)
                formula = {'kind': 'all', 'clauses': clauses}
            attribution_claims[route][purpose] = formula
            record['adjusted_claim_formula'] = copy.deepcopy(formula)
    return {
        'schema': 1, 'prospective': True, 'endpoints': endpoints, 'family_size': len(endpoints),
        'route_multiplicity': list(ROUTES), 'attribution': attribution,
        'claim_formulas': claims, 'attribution_claim_formulas': attribution_claims,
        'evaluation_configurations': sorted(available),
        'selected_contrast_configurations': sorted({t['configuration'] for e in endpoints for t in e['terms']}),
        'claim_scope': 'both routes retained; either fully supported route may be reported; no post-hoc route winner',
        'descriptive_grid': selection['descriptive_configurations'],
        'multiplicity_rule': 'M = len(endpoints), alpha/(2*M); corresponding adjusted one-sided bounds',
        'empty_eligibility_rule': 'never evidence of competitive superiority',
        'historical_half_headroom_scope': 'separate inherited criterion; only interpretable with positive J headroom',
    }


def freeze_selection(selection, *, out_dir, matched_specs=None):
    """Write CONTRASTS first, then the selection permit; no overwrite or data access."""
    directory = Path(out_dir)
    selection_path, contrasts_path = directory / 'SELECTION.json', directory / 'CONTRASTS.json'
    if selection_path.exists() or contrasts_path.exists():
        raise FileExistsError('Refusing to overwrite a frozen selection or contrast family')
    contrasts = build_contrast_family(selection, matched_specs=matched_specs)
    encoded = json.dumps(contrasts, indent=2, sort_keys=True, allow_nan=False) + '\n'
    sha = hashlib.sha256(encoded.encode()).hexdigest()
    frozen = {**copy.deepcopy(selection), 'selection_frozen': True, 'contrasts_sha256': sha,
              'contrast_family_size': len(contrasts['endpoints']),
              'evaluation_configurations': contrasts['evaluation_configurations'],
              'selected_contrast_configurations': contrasts['selected_contrast_configurations'],
              'attribution': contrasts['attribution'], 'claim_formulas': contrasts['claim_formulas'],
              'attribution_claim_formulas': contrasts['attribution_claim_formulas']}
    directory.mkdir(parents=True, exist_ok=True)
    with contrasts_path.open('x') as handle:
        handle.write(encoded)
    with selection_path.open('x') as handle:
        json.dump(frozen, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    return frozen
