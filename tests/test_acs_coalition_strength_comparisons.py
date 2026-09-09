"""Numerical fixtures for fixed-grid comparisons; no scientific fitting."""
import math
import pytest

from scripts.acs_coalition_strength_comparisons import (
    SOURCE_TASKS, UTILITY_TASKS, dominates, evaluate_pair, fixed_seed_summary,
    pareto_membership, source_check,
)


def point(loss=.30, gain=.02):
    return {'utility': {t: loss for t in UTILITY_TASKS},
            'gains': {f'{v}/{t}': gain for v, ts in
                {'A': ('public_coverage', 'commute_over20', 'SEX', 'RAC1P'),
                 'B': ('income_binary', 'civilian_at_work', 'same_residence', 'SEX', 'RAC1P'),
                 'AB': ('SEX', 'RAC1P')}.items() for t in ts}}


PARENT = {t: .30 for t in SOURCE_TASKS}


def compare(j=None, local=None, panel=UTILITY_TASKS, delta=.001):
    return evaluate_pair(j or point(gain=.01), local or point(), PARENT, panel, delta)


def test_both_source_checks_reject_an_average_allowance():
    j = point(gain=.01)
    j['utility'].update(income_binary=.32, civilian_at_work=.20, public_coverage=.20)
    r = compare(j, delta=1)
    assert r['source_J']['pass'] is False
    assert r['both_source_feasible'] is False and r['qualifies_directional'] is False
    assert 'J/source/income_binary' in r['exclusion_reasons']['directional']


def test_local_source_failure_excludes_an_improved_j():
    local = point();local['utility']['income_binary'] = .32
    r = compare(local=local, delta=1)
    assert r['source_J']['pass'] and r['source_Iplus']['pass'] is False
    assert r['qualifies_close'] is False


def test_native_source_heads_cannot_substitute_for_downstream_losses():
    j = point(gain=.01);j['utility'].pop('income_binary');j['native'] = {'income_binary': .10}
    r = compare(j)
    assert r['assessment_close'] == 'unassessable'
    assert 'J/utility/income_binary' in r['missing_required']


def test_close_uses_every_component_without_cancellation():
    j = point(gain=.01);j['utility']['same_residence'] += .005;j['utility']['commute_over20'] -= .005
    r = compare(j)
    assert sum(r['utility_differences'].values()) == pytest.approx(0)
    assert r['utility_close'] is False and r['utility_directional'] is False


def test_directional_allows_large_improvements_while_close_does_not():
    j = point(loss=.20, gain=.01)
    r = compare(j)
    assert r['qualifies_directional'] is True and r['qualifies_close'] is False


def test_missing_unincluded_task_remains_visible_without_invalidating_source_panel():
    j = point(gain=.01);j['utility']['commute_over20'] = None
    r = compare(j, panel=SOURCE_TASKS)
    assert r['qualifies_close'] is True and r['utility_differences']['commute_over20'] is None
    assert compare(j)['qualifies_close'] is None


@pytest.mark.parametrize('value', [None, float('nan'), float('inf'), -float('inf'), True])
def test_nonfinite_or_boolean_required_metric_is_unassessable(value):
    j = point(gain=.01);j['gains']['AB/SEX'] = value
    r = compare(j)
    assert r['qualifies_close'] is None and r['gain_difference'] is None


def test_missing_relevant_value_dominates_a_known_failure_without_hiding_it():
    j = point(gain=.01);j['utility']['income_binary'] = .40;j['utility']['same_residence'] = None
    r = compare(j)
    assert r['assessment_close'] == 'unassessable'
    assert 'J/source/income_binary' in r['exclusion_reasons']['close']


def test_zero_anchor_is_a_match_but_never_a_strict_improvement():
    j = point();r = compare(j, j, delta=0)
    assert r['utility_close'] is True and r['gain_difference'] == 0
    assert r['strict_gain_improvement'] is False and r['qualifies_close'] is False


def test_roundoff_is_separate_from_practical_utility_delta():
    j = point(gain=.02-5e-13)
    assert compare(j)['strict_gain_improvement'] is False
    j['gains']['AB/SEX'] = .02-2e-12
    assert compare(j)['strict_gain_improvement'] is True
    j['utility']['same_residence'] += .00100000001
    assert compare(j)['utility_close'] is False


def test_weighted_parent_is_its_own_same_seed_reference():
    j = point(loss=.295)
    assert source_check(j, PARENT)['pass'] is True
    assert source_check(j, {t: .28 for t in SOURCE_TASKS})['pass'] is False


def test_fixed_pairs_do_not_stitch_a_different_strength_for_each_seed():
    rows = []
    for pair, seed in [('a', 0), ('b', 1), ('b', 2)]:
        rows.append({'pair': pair, 'seed': seed, **compare()})
    r = {x['pair']: x for x in fixed_seed_summary(rows, ['pair'])}
    assert r['a']['close_qualifying_seed_count'] == 1
    assert r['b']['close_qualifying_seed_count'] == 2
    assert r['a']['close_all_seeds_qualify'] is None and r['b']['close_all_seeds_qualify'] is None


def test_fixed_pair_means_include_excluded_seeds():
    rows = []
    for seed, delta in enumerate([-.1, -.2, .2]):
        result = compare(point(gain=.02+delta))
        rows.append({'pair': 'fixed', 'seed': seed, **result})
    r = fixed_seed_summary(rows, ['pair'])[0]
    assert r['close_qualifying_seed_count'] == 2
    assert r['gain_difference_mean'] == pytest.approx(-.1/3)
    assert r['gain_difference_mean'] != pytest.approx(-.15)


def test_duplicate_fixed_pair_seed_is_rejected():
    row = {'pair': 'fixed', 'seed': 0, **compare()}
    with pytest.raises(ValueError, match='Duplicate seed'):
        fixed_seed_summary([row, row], ['pair'])


def test_pareto_is_componentwise_not_a_two_dimensional_projection():
    x = {'residence': .3, 'gain': .01, 'income': .40}
    y = {'residence': .4, 'gain': .02, 'income': .30}
    assert dominates(x, y, ['residence', 'gain'])
    assert dominates(x, y, ['residence', 'gain', 'income']) is False


def test_pareto_roundoff_does_not_use_utility_matching_delta():
    x, y = {'u': .3005, 'g': .01}, {'u': .30, 'g': .02}
    assert dominates(x, y, ['u', 'g']) is False
    assert dominates(y, x, ['u', 'g']) is False


def test_missing_pareto_component_is_never_dropped():
    points = [{'condition': 'x', 'u': .3, 'g': .01}, {'condition': 'y', 'u': .2, 'g': None}]
    r = pareto_membership(points, ['u', 'g'])
    assert r[1]['status'] == 'unassessable_point'
    assert r[0]['status'] == 'not_dominated_among_assessable_points'
    assert dominates(points[0], points[1], ['u', 'g']) is None


def test_pareto_shared_anchor_cannot_be_counted_twice():
    p = {'condition': 'F_I', 'u': .30, 'g': .02}
    with pytest.raises(ValueError, match='Repeated physical'):
        pareto_membership([p, p], ['u', 'g'])


def test_negative_gains_are_retained_and_attributes_not_averaged():
    j = point(gain=-.01);j['gains']['AB/RAC1P'] = .20
    r = compare(j)
    assert r['J_gain'] == -.01 and r['strict_gain_improvement']
    race = evaluate_pair(j, point(), PARENT, UTILITY_TASKS, .001, attribute='RAC1P')
    assert race['strict_gain_improvement'] is False
