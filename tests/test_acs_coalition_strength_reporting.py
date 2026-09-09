"""Focused joins and finite-grid regression tests; no model or data fitting."""
import copy
import itertools
import json
from pathlib import Path

import pytest

from scripts import summarize_acs_coalition_strength as report


RULES = json.loads((Path(__file__).parents[1]/'results/redesign_20260909_acs_coalition_strength_v1/comparison_rules.json').read_text())


def manifest():
    rows = []
    for seed, interface in itertools.product((0, 1, 2), ('F', 'P')):
        rows.append(dict(seed=seed, interface=interface, family='I', beta=0, condition=interface+'_I', reused=True))
        for family, beta in itertools.product(('J', 'Iplus'), (.025, .05, .1, .2)):
            rows.append(dict(seed=seed, interface=interface, family=family, beta=beta,
                condition=interface+'_'+family+'_b'+str(beta).replace('.', 'p'), reused=beta == .1))
    return {'systems': rows}


def synthetic_points(entries):
    result = {}
    for e in entries:
        for weight, scope, budget in itertools.product(RULES['weights'], RULES['audit_scopes'], RULES['audit_budgets']):
            parent = {t: .3 for t in report.SOURCE_TASKS}
            utility = {t: .305 for t in report.UTILITY_TASKS}
            gains = {v+'/'+t: .1+e['beta']*(1 if e['family'] == 'J' else 2) for v, ts in report.AUDIT_ROLES.items() for t in ts}
            result[e['seed'], e['condition'], weight, scope, budget] = {**e, 'utility': utility, 'parent': parent,
                'gains': gains, 'candidate_ids': {k: 'mlp' for k in gains},
                'coverage': {k: {'complete': not k.endswith('RAC1P')} for k in gains}}
    return result


def test_registry_has54physical_and60_family_aliases():
    entries, aliases = report.registry(manifest())
    assert len(entries) == 54 and len(aliases) == 60
    for seed, interface in itertools.product((0, 1, 2), ('F', 'P')):
        assert aliases[seed, interface, 'J', '0'] is aliases[seed, interface, 'Iplus', '0']
    with pytest.raises(ValueError):
        report.registry({'systems': entries+[entries[0]]})


def test_full_cube_fixed_pair_identity_and_support_are_preserved():
    entries, aliases = report.registry(manifest())
    cube = report.pair_cube(synthetic_points(entries), aliases, RULES)
    assert len(cube) == 57600
    assert len({tuple(r[k] for k in ('seed', *report.PAIR_FIELDS)) for r in cube}) == 57600
    assert len([r for r in cube if r['attribute'] == 'SEX']) == 28800
    aggregates = report.fixed_seed_summary(cube, report.PAIR_FIELDS)
    assert len(aggregates) == 19200
    for row in aggregates:
        assert row['present_seeds'] == [0, 1, 2]
        assert row['close_assessable_seed_count'] == 3
    anchor = [r for r in cube if r['J_beta'] == '0' and r['Iplus_beta'] == '0']
    assert all(r['shared_physical_anchor'] and r['qualifies_close'] is False for r in anchor)
    anchor_summaries = [r for r in aggregates if r['J_beta'] == '0' and r['Iplus_beta'] == '0']
    assert all(r['close_source_and_utility_eligible_seed_count'] == 3 and r['close_qualifying_seed_count'] == 0 for r in anchor_summaries)
    race = [r for r in cube if r['attribute'] == 'RAC1P']
    assert all(r['J_attribute_support_complete'] is False for r in race)
    assert any(r['qualifies_close'] is True for r in race)  # Numerical result, not full-support certificate.


def test_missing_new_system_is_not_filled_by_historical_anchor():
    entries, aliases = report.registry(manifest())
    points = synthetic_points(entries)
    name = aliases[1, 'F', 'J', '0.025']['condition']
    for weight, scope, budget in itertools.product(RULES['weights'], RULES['audit_scopes'], RULES['audit_budgets']):
        p = points[1, name, weight, scope, budget]
        p['utility'] = {k: None for k in p['utility']}
        p['gains'] = {k: None for k in p['gains']}
    cube = report.pair_cube(points, aliases, RULES)
    subset = [r for r in cube if r['seed'] == 1 and r['J_condition'] == name]
    assert subset and all(r['qualifies_close'] is None and r['qualifies_directional'] is None for r in subset)


def test_native_join_preserves_weights_and_rejects_duplicate_split():
    score = {'log_loss': .3}
    raw = dict(seed=0, condition='F_J', view='A', target='income_binary', split='test', score=score,
        person_weighted={'log_loss': .4}, prediction_sha256='fixed')
    rows = report.native_rows({'raw_metrics': [raw]}, 'F_J', 'F_J_b0p1')
    assert rows[0]['condition'] == 'F_J_b0p1'
    assert rows[0]['scores']['test']['log_loss'] == .3
    assert rows[0]['scores']['test_person_weighted']['log_loss'] == .4
    with pytest.raises(ValueError):
        report.native_rows({'raw_metrics': [raw, copy.deepcopy(raw)]})


def test_budget_and_scope_are_part_of_fixed_pair_summary_identity():
    entries, aliases = report.registry(manifest())
    p = synthetic_points(entries)
    name = aliases[0, 'F', 'J', '0.025']['condition']
    p[0, name, 'person_weighted', 'expanded_catchup', 360]['gains']['AB/SEX'] = .001
    rows = report.pair_cube(p, aliases, RULES)
    wanted = [r for r in rows if r['seed'] == 0 and r['J_condition'] == name and r['Iplus_beta'] == '0.025'
        and r['attribute'] == 'SEX' and r['panel'] == 'source_only' and r['delta'] == 0]
    changed = [r for r in wanted if r['gain_difference'] < -.05]
    assert len(changed) == 1
    assert (changed[0]['weighting'], changed[0]['scope'], changed[0]['audit_budget']) == ('person_weighted', 'expanded_catchup', 360)


def test_deterministic_gzip_has_no_filename_or_clock(tmp_path):
    path = tmp_path/'evidence.csv';path.write_text('a,b\n1,2\n')
    first = report.gzip_export(path)
    second = report.gzip_export(path)
    assert first == second
    data = (tmp_path/'evidence.csv.gz').read_bytes()
    assert data[3] & 8 == 0 and data[4:8] == b'\0'*4


def test_complete_vector_does_not_turn_one_SEX_benefit_into_a_tradeoff_claim():
    row = dict(seed=0, interface='F', J_beta='0.025', Iplus_beta='0.05', J_condition='J', Iplus_condition='L',
        weighting='unweighted', scope='expanded_catchup', audit_budget=360, attribute='SEX', panel='full_authorized', delta=0,
        utility_differences={t: 0. for t in report.UTILITY_TASKS},
        audit_gain_differences={v+'/'+t: 0. for v, ts in report.AUDIT_ROLES.items() for t in ts})
    row['audit_gain_differences']['AB/SEX'] = -.01
    row['audit_gain_differences']['B/RAC1P'] = .001
    result = report.vector_tradeoffs([row])[0]
    assert result['J_dominates_Iplus_numeric'] is False and result['Iplus_dominates_J_numeric'] is False
    assert result['components_worse_for_J'] == ['gain/B/RAC1P']
    row['utility_differences']['commute_over20'] = None
    assert report.vector_tradeoffs([row])[0]['J_dominates_Iplus_numeric'] is None
