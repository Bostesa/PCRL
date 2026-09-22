"""Prospective selection/contrast contracts using synthetic validation summaries."""
import json

import pytest

from experiments.pcrl_task_directed_release_v1 import selection


A = 'Trisk_C_0.002_a17'
B = 'T0_L_0.0005_a17'
C = 'Ttask_C_0.002_a17'
PRIMARY = ('A/SEX', 'A/RAC1P', 'AB/SEX', 'AB/RAC1P')


def scores(u=.5, s=.5, weighted_u=None, weighted_s=None):
    rows = {}
    for role, value, weighted in [('utility:A/same_residence', u, weighted_u)] + [
            ('attack:' + role, s, weighted_s) for role in PRIMARY]:
        rows[role] = {'selection': 'chosen', 'independent_selection': 'independent',
                      'validation': {'unweighted': value, 'weighted': value if weighted is None else weighted,
                                     'balanced': .5 * (value + (value if weighted is None else weighted))}}
    return rows


def anchors(**kwargs):
    return {a: scores(**kwargs) for a in (0, 1, 2)}


def base():
    return {'H': anchors(u=.6, s=.6), 'J': anchors(),
            A: anchors(u=.49, s=.4998), B: anchors(u=.5005, s=.52),
            'continuous_task': anchors(u=.495, s=.4999)}


def families():
    return {'continuous_task': {'configurations': ['continuous_task'], 'label_matched': True, 'required': True}}


def test_two_routes_use_one_common_configuration_and_both_weightings():
    values = base()
    values[C] = anchors(u=.48, s=.5, weighted_s=.498)  # fails only PWGTP privacy cap
    result = selection.select_validation(values, candidate_ids=[A, B, C], families=families())
    assert result['routes']['utility_first']['nominee'] == A
    assert result['routes']['protection_first']['nominee'] == B
    assert all(r['screen_passed'] for r in result['routes'].values())
    assert result['routes']['utility_first']['family_nominees']['continuous_task']['configuration'] == 'continuous_task'
    assert result['predictor_choices'][A]['0']['utility:A/same_residence']['selection'] == 'chosen'
    assert result['aggregates'][A]['utility']['unweighted'] == pytest.approx(.49)


def test_equal_anchor_mean_not_worst_anchor_and_no_per_anchor_selection():
    values = base()
    values[A] = {0: scores(u=.47), 1: scores(u=.50), 2: scores(u=.50)}
    got = selection.select_validation(values, candidate_ids=[A], families=families())
    assert got['routes']['utility_first']['nominee'] == A
    assert got['routes']['utility_first']['screen_passed']
    assert got['aggregates'][A]['utility']['unweighted'] == pytest.approx(.49)


def test_empty_required_family_blocks_competitive_superiority():
    values = base()
    values['continuous_task'] = anchors(u=.4, s=.4)  # no UF family privacy eligibility
    got = selection.select_validation(values, candidate_ids=[A], families=families())
    route = got['routes']['utility_first']
    assert route['screen_passed']
    assert route['family_nominees']['continuous_task']['configuration'] is None
    assert not route['competitive_validation_eligible']
    assert route['empty_required_families'] == ['continuous_task']
    assert route['family_nominees']['continuous_task']['attempted'] == ['continuous_task']


def test_no_screen_pass_freezes_nearest_descriptive_and_lexical_tie():
    values = base()
    values[A] = anchors(u=.499, s=.5)
    values[C] = anchors(u=.499, s=.5)
    got = selection.select_validation(values, candidate_ids=[A, C], families=families())
    assert got['routes']['utility_first']['nominee'] == min(A, C)
    assert not got['routes']['utility_first']['screen_passed']
    assert got['routes']['utility_first']['descriptive_only']
    assert not got['routes']['utility_first']['competitive_validation_eligible']


def test_missing_anchor_not_silently_averaged_and_no_nominee_keeps_grid():
    values = base()
    values[A] = {0: scores(u=.1), 1: scores(u=.1)}
    got = selection.select_validation(values, candidate_ids=[A], families=families())
    assert got['routes']['utility_first']['nominee'] is None
    assert got['unavailable_configurations'][A]['missing_anchors'] == [2]
    assert set(got['descriptive_configurations']) == {'H', 'J', B, 'continuous_task'}
    family = selection.build_contrast_family(got)
    assert family['endpoints']  # H/J descriptive core is prospective even with no nominee


def test_contrasts_include_family_caps_and_exact_matched_attribution():
    values = base()
    matches = ['T0_C_0.002_a17', 'Ttask_C_0.002_a17', 'Trisk_L_0.002_a17',
               'Trisk_U_unconstrained_a17', 'Trisk_code']
    values.update({name: anchors(u=.495, s=.5) for name in matches})
    got = selection.select_validation(values, candidate_ids=[A, B], families=families())
    family = selection.build_contrast_family(got)
    endpoints = family['endpoints']
    assert family['family_size'] == len(endpoints) == len({e['id'] for e in endpoints})
    assert family['route_multiplicity'] == ['utility_first', 'protection_first']
    assert all(e['anchors'] == [0, 1, 2] for e in endpoints)
    caps = [e for e in endpoints if e['purpose'] == 'family_evaluated_J_privacy_cap'
            and e['route'] == 'utility_first']
    assert len(caps) == 8
    assert all(e['checks'][0]['threshold'] == .001 for e in caps)
    gate = [e for e in endpoints if e['purpose'] == 'candidate_J_utility_gate' and e['route'] == 'utility_first']
    assert len(gate) == 2 and all(e['checks'][0]['threshold'] == -.003 for e in gate)
    attributions = family['attribution']['utility_first']
    assert attributions['risk_refinement']['comparators'] == matches[:2]
    assert attributions['coalition']['comparators'] == [matches[2]]
    assert not attributions['risk_refinement']['missing']
    privacy = next(e for e in caps if e['weighting'] == 'PWGTP')
    assert privacy['terms'][0]['configuration'] == 'J'
    assert privacy['terms'][1]['configuration'] == 'continuous_task'
    assert [t['coefficient'] for t in privacy['terms']] == [1., -1.]
    pf = family['claim_formulas']['protection_first']['competitive']
    assert pf['kind'] == 'all'
    # Historical-J and each family require an ANY strict privacy improvement,
    # alongside ALL NI checks; requiring all eight strict improvements is wrong.
    disjunctions = [c for c in pf['clauses'] if c.get('kind') == 'any']
    assert len(disjunctions) == 1 and len(disjunctions[0]['clauses']) == 8
    assert pf['clauses'][0]['kind'] == 'all'
    assert pf['clauses'][0]['clauses'][-1]['kind'] == 'any'


def test_freeze_writes_complete_ids_and_hashed_family_without_overwrite(tmp_path):
    got = selection.select_validation(base(), candidate_ids=[A, B], families=families())
    result = selection.freeze_selection(got, out_dir=tmp_path)
    saved = json.loads((tmp_path / 'SELECTION.json').read_text())
    family = json.loads((tmp_path / 'CONTRASTS.json').read_text())
    assert saved['selection_frozen'] is True
    assert saved['contrasts_sha256'] == result['contrasts_sha256']
    assert saved['contrast_family_size'] == len(family['endpoints'])
    assert saved['predictor_choices'][A]['2']['attack:A/SEX']['selection'] == 'chosen'
    assert saved['attribution_claim_formulas'] == family['attribution_claim_formulas']
    with pytest.raises(FileExistsError):
        selection.freeze_selection(got, out_dir=tmp_path)


def test_nonfinite_validation_and_unregistered_candidates_rejected():
    values = base()
    values[A][0]['utility:A/same_residence']['validation']['weighted'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        selection.select_validation(values, candidate_ids=[A], families=families())
    with pytest.raises(ValueError, match='registered'):
        selection.select_validation(base(), candidate_ids=['freeform'], families=families())
    with pytest.raises(ValueError, match='unique'):
        selection.select_validation(base(), candidate_ids=[A, A], families=families())


def test_family_nominees_are_route_specific_strongest_eligible_controls():
    values = base()
    values['constant_best'] = anchors(u=.496, s=.4999)
    values['independent_token'] = anchors(u=.5005, s=.54)
    declared = {'null': {'configurations': ['constant_best', 'independent_token'],
                         'label_matched': True, 'required': True}}
    got = selection.select_validation(values, candidate_ids=[A, B], families=declared)
    assert got['routes']['utility_first']['family_nominees']['null']['configuration'] == 'constant_best'
    assert got['routes']['protection_first']['family_nominees']['null']['configuration'] == 'independent_token'


def test_expanded_dictionary_requires_registration_and_never_matches_a17_controls():
    name = 'Trisk_C_0.002_a33'
    values = {**base(), name: anchors(u=.48), 'T0_C_0.002_a17': anchors(), C: anchors()}
    with pytest.raises(ValueError, match='registered'):
        selection.select_validation(values, candidate_ids=[name], families=families())
    spec = {'input': 'Trisk', 'policy': 'C', 'budget': .002, 'max_actions': 33,
            'registration_id': 'synthetic_prospective_extension'}
    got = selection.select_validation(values, candidate_ids=[name], families=families(),
                                      registered_extensions={name: spec})
    family = selection.build_contrast_family(got)
    matched = family['attribution']['utility_first']['risk_refinement']
    assert matched['missing'] == ['T0_C_0.002_a33', 'Ttask_C_0.002_a33']
    assert not matched['claim_eligible']
    with pytest.raises(ValueError, match='unchanged'):
        selection.build_contrast_family(got, matched_specs={name: {**spec, 'max_actions': 17}})


def test_default_families_retain_all_registered_label_controls():
    defaults = selection.default_families()
    assert set(defaults) == {'continuous_task', 'deterministic_actions', 'direct_code', 'withholding',
                             'randomized_response', 'supervised_LEACE', 'supervised_SPLINCE', 'constant_null'}
    assert sum(len(v['configurations']) for v in defaults.values()) == 29
    assert all(v['label_matched'] and v['required'] for v in defaults.values())


def test_mixed_json_anchor_key_types_have_stable_digest():
    values = base()
    values[A]['1'] = values[A].pop(1)
    got = selection.select_validation(values, candidate_ids=[A], families=families())
    assert got['validation_summary_digest']


def attribution_fixture():
    values = base()
    values.update({
        'Trisk_L_0.0005_a17': anchors(u=.493, s=.501),
        'Trisk_L_0.002_a17': anchors(u=.495, s=.504),
        'Trisk_L_0.01_a17': anchors(u=.5005, s=.52),
        'T0_C_0.002_a17': anchors(u=.495, s=.5),
        'Ttask_C_0.002_a17': anchors(u=.495, s=.5),
    })
    return values


def leaves(formula):
    if 'endpoint' in formula:
        return [formula]
    return [leaf for clause in formula.get('clauses', []) for leaf in leaves(clause)]


def test_coalition_matches_best_route_specific_complete_local_frontier():
    got = selection.select_validation(attribution_fixture(), candidate_ids=[A], families=families())
    family = selection.build_contrast_family(got)
    uf = family['attribution']['utility_first']['coalition']
    pf = family['attribution']['protection_first']['coalition']
    assert uf['comparators'] == ['Trisk_L_0.0005_a17']
    assert pf['comparators'] == ['Trisk_L_0.01_a17']
    assert uf['same_budget_comparator'] == 'Trisk_L_0.002_a17'
    assert uf['local_frontier']['complete'] and len(uf['local_frontier']['attempted']) == 3
    assert any(e['purpose'] == 'attribution_coalition_same_budget_utility' for e in family['endpoints'])


def test_narrow_attribution_has_own_caps_and_is_not_gated_by_main_J_screen():
    values = attribution_fixture()
    values[A] = anchors(u=.499, s=.5)  # neither main screen passes
    got = selection.select_validation(values, candidate_ids=[A], families=families())
    assert not got['routes']['utility_first']['screen_passed']
    family = selection.build_contrast_family(got)
    formula = family['attribution_claim_formulas']['utility_first']['risk_refinement']
    assert formula['kind'] == 'all'
    refs = leaves(formula)
    eps = {e['id']: e for e in family['endpoints']}
    utility = [eps[r['endpoint']] for r in refs if r['check'] == 'utility']
    assert len(utility) == 4  # both weightings against BOTH T0 and Ttask
    assert all(e['checks'][0]['operator'] == '<' and e['checks'][0]['threshold'] == 0 for e in utility)
    assert len([r for r in refs if r['check'] == 'sensitive_noninferiority']) == 16
    assert len([r for r in refs if r['check'] == 'attribution_privacy_cap']) == 24
    assert family['claim_formulas']['utility_first']['historical_J']['kind'] == 'blocked'


def test_coalition_PF_any_strict_is_AB_only_with_all_eight_privacy_NI():
    got = selection.select_validation(attribution_fixture(), candidate_ids=[A], families=families())
    family = selection.build_contrast_family(got)
    formula = family['attribution_claim_formulas']['protection_first']['coalition']
    refs = leaves(formula)
    eps = {e['id']: e for e in family['endpoints']}
    strict = [eps[r['endpoint']] for r in refs if r['check'] == 'strict_sensitive_improvement']
    assert len(strict) == 4
    assert {e['role'] for e in strict} == {'attack:AB/SEX', 'attack:AB/RAC1P'}
    assert len([r for r in refs if r['check'] == 'sensitive_noninferiority']) == 8
    assert len([r for r in refs if r['check'] == 'attribution_utility_cap']) == 4
    assert any(c.get('kind') == 'any' for c in formula['clauses'])


@pytest.mark.parametrize('missing,empty', [(True, False), (False, True)])
def test_incomplete_or_empty_local_frontier_never_supports_coalition_claim(missing, empty):
    values = attribution_fixture()
    if missing:
        del values['Trisk_L_0.01_a17']
    if empty:
        for n in ['Trisk_L_0.0005_a17', 'Trisk_L_0.002_a17', 'Trisk_L_0.01_a17']:
            values[n] = anchors(u=.6, s=.4)
    got = selection.select_validation(values, candidate_ids=[A], families=families())
    family = selection.build_contrast_family(got)
    for route in selection.ROUTES:
        assert family['attribution_claim_formulas'][route]['coalition']['kind'] == 'blocked'
        assert not family['attribution'][route]['coalition']['claim_eligible']


def test_branch_A_controls_and_U33_join_existing_families_only_when_registered():
    u = 'Trisk_U_unconstrained_a33'
    map_spec = {'input': 'Trisk', 'policy': 'U', 'budget': None, 'max_actions': 33,
                'branch': 'A', 'extra_registration_sha256': 'registered'}
    control = 'T0_withhold_0.25_a33'
    control_spec = {'configuration': control, 'canonical_release': 'T0_withhold_0.25',
                    'kind': 'control', 'label_matched': True, 'branch': 'A', 'max_actions': 33,
                    'baseline_family': 'withhold', 'extra_registration_sha256': 'registered'}
    values = {**base(), u: anchors(u=.49), control: anchors(u=.49)}
    got = selection.select_validation(values, candidate_ids=[A], registered_extensions={u: map_spec},
                                      registered_controls={control: control_spec})
    assert u in got['families']['deterministic_actions']['configurations']
    assert control in got['families']['withholding']['configurations']
    assert len(got['families']) == 8


def test_fine_conditioning_frontier_never_matches_primary_L_or_primary_risk_controls():
    name = A + '_fineC'
    extensions = {}
    values = attribution_fixture()
    for policy in ('L', 'C'):
        for budget in (.0005, .002, .01):
            key = f'Trisk_{policy}_{budget:g}_a17_fineC'
            extensions[key] = {'input': 'Trisk', 'policy': policy, 'budget': budget, 'max_actions': 17,
                'registration_id': 'synthetic-fine', 'branch': 'C',
                'conditioning_family': 'primary_intersect_fineC4', 'matched_frontier': 'Trisk_fineC_a17'}
            values[key] = anchors(u=.49 if policy == 'C' else .495, s=.51)
    got = selection.select_validation(values, candidate_ids=[name], families=families(),
                                      registered_extensions=extensions)
    family = selection.build_contrast_family(got)
    coalition = family['attribution']['utility_first']['coalition']
    assert coalition['local_frontier']['complete']
    assert all(n.endswith('_fineC') for n in coalition['local_frontier']['attempted'])
    assert coalition['comparators'][0].endswith('_fineC')
    assert not family['attribution']['utility_first']['risk_refinement']['claim_eligible']


def formula_passes(formula, endpoints, upper):
    """Independent small oracle for the public all/any/blocked/check contract."""
    if 'endpoint' in formula:
        check = next(c for c in endpoints[formula['endpoint']]['checks'] if c['name'] == formula['check'])
        value = upper[formula['endpoint']]
        return value < check['threshold'] if check['operator'] == '<' else value <= check['threshold']
    if formula['kind'] == 'blocked':
        return False
    assert formula['clauses']  # empty conjunction cannot be evidence
    values = [formula_passes(c, endpoints, upper) for c in formula['clauses']]
    return all(values) if formula['kind'] == 'all' else any(values)


def test_adjusted_coalition_claim_fails_A_only_strict_gain_or_any_NI_or_common_cap_failure():
    got = selection.select_validation(attribution_fixture(), candidate_ids=[A], families=families())
    family = selection.build_contrast_family(got)
    endpoints = {e['id']: e for e in family['endpoints']}
    formula = family['attribution_claim_formulas']['protection_first']['coalition']
    upper = {key: 0. for key in endpoints}
    relevant = leaves(formula)
    a = next(r['endpoint'] for r in relevant if r['check'] == 'sensitive_noninferiority'
             and endpoints[r['endpoint']]['role'] == 'attack:A/SEX')
    upper[a] = -.01
    assert not formula_passes(formula, endpoints, upper)
    ab = next(r['endpoint'] for r in relevant if r['check'] == 'strict_sensitive_improvement')
    upper[ab] = -.0001
    assert formula_passes(formula, endpoints, upper)
    upper[a] = .00101
    assert not formula_passes(formula, endpoints, upper)
    upper[a] = .001
    assert formula_passes(formula, endpoints, upper)
    cap = next(r['endpoint'] for r in relevant if r['check'] == 'attribution_utility_cap')
    upper[cap] = .00101
    assert not formula_passes(formula, endpoints, upper)


def test_adjusted_risk_claim_requires_both_comparators_and_strict_both_weightings():
    got = selection.select_validation(attribution_fixture(), candidate_ids=[A], families=families())
    family = selection.build_contrast_family(got)
    endpoints = {e['id']: e for e in family['endpoints']}
    formula = family['attribution_claim_formulas']['utility_first']['risk_refinement']
    upper = {key: -.0001 for key in endpoints}
    assert formula_passes(formula, endpoints, upper)
    task_weighted = next(r['endpoint'] for r in leaves(formula) if r['check'] == 'utility'
                        and endpoints[r['endpoint']]['comparator'] == 'Ttask_C_0.002_a17'
                        and endpoints[r['endpoint']]['weighting'] == 'PWGTP')
    upper[task_weighted] = 0.
    assert not formula_passes(formula, endpoints, upper)


def test_local_frontier_eligibility_respects_PWGTP_and_lexical_ties():
    values = attribution_fixture()
    values['Trisk_L_0.0005_a17'] = anchors(u=.48, s=.5, weighted_s=.498)
    values['Trisk_L_0.002_a17'] = anchors(u=.495, s=.5)
    values['Trisk_L_0.01_a17'] = anchors(u=.495, s=.5)
    got = selection.select_validation(values, candidate_ids=[A], families=families())
    family = selection.build_contrast_family(got)
    uf = family['attribution']['utility_first']['coalition']['local_frontier']
    assert uf['nominee'] == 'Trisk_L_0.002_a17'
    assert 'Trisk_L_0.0005_a17' not in uf['eligible']


def test_branch_controls_without_provenance_or_wrong_family_rejected():
    name = 'constant_best_a33'
    spec = {'configuration': name, 'canonical_release': 'constant_best', 'kind': 'control',
            'label_matched': True, 'branch': 'A', 'max_actions': 33, 'baseline_family': 'constant_best'}
    with pytest.raises(ValueError, match='registration'):
        selection.select_validation(base(), candidate_ids=[A], registered_controls={name: spec})
    spec.update(registration_id='synthetic', baseline_family='rr')
    with pytest.raises(ValueError, match='control specification'):
        selection.select_validation(base(), candidate_ids=[A], registered_controls={name: spec})
