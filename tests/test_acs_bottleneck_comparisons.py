"""Bottleneck report boundaries: frozen choices, fixed PCA parent, null coverage."""
import pytest

from scripts.summarize_acs_bottleneck import (
    ATTRIBUTES, SOURCE_TASKS, UTILITY_TASKS, pca_policy, make_index, get_value, paired_rows,
    catchup_diagnostics, selectors,
)


def row(candidate, selected=False, independent=False, val=.5, test=.6):
    return dict(seed=0, role='audit', release='D_protected', target='SEX',
                candidate_id=candidate, family='mlp' if candidate.startswith('mlp') else candidate,
                selected=selected, independent_selected=independent,
                selected_within_family=independent, auc_selected=False,
                validation={'log_loss': val}, test={'log_loss': test})


def test_saved_adversary_is_diagnostic_even_with_the_best_scores():
    saved = row('saved_adversary', val=.001, test=.001)
    assert selectors(saved) == ['candidate:saved_adversary']
    # Runner preserves a true singleton-family flag; it still creates no
    # eligible family/primary selection in reporting.
    assert selectors({**saved, 'selected_within_family': True}) == ['candidate:saved_adversary']
    with pytest.raises(ValueError, match='diagnostic-only'):
        selectors({**saved, 'selected': True})
    with pytest.raises(ValueError, match='diagnostic-only'):
        selectors({**saved, 'independent_selected': True})


def test_catchup_primary_and_fresh_primary_do_not_reselect_on_development():
    fresh = row('mlp_0', independent=True, val=.5, test=.3)
    catchup = row('catchup', selected=True, val=.4, test=.6)
    index = make_index([{'seed': 0, 'raw_metrics': [fresh, catchup]}])
    args = (index, 0, 'audit', 'D_protected', 'SEX', 'test')
    assert get_value(*args) == .6
    assert get_value(*args, selector='independent_selected') == .3
    with pytest.raises(ValueError, match='not one of the five'):
        selectors({**catchup, 'independent_selected': True})


def test_reference_primary_is_its_five_candidate_independent_primary():
    reference = row('mlp_1', selected=True)
    reference['release'] = 'E_pca'
    del reference['independent_selected']
    assert 'independent_selected' in selectors(reference)


def test_both_learned_arms_use_original_pca_for_all_parent_margins():
    parent = {**dict.fromkeys(SOURCE_TASKS, .4), 'same_residence': .3, 'commute_over20': .6}
    method = {**dict.fromkeys(SOURCE_TASKS, .409), 'same_residence': .35, 'commute_over20': .7}
    coverage = {t: {'complete': True} for t in ATTRIBUTES}
    result = pca_policy(method, parent, {'B_rich_bank': .5, 'C_tree_bank': .55},
                        dict.fromkeys(ATTRIBUTES, .625), dict.fromkeys(ATTRIBUTES, .4),
                        dict.fromkeys(ATTRIBUTES, .8), coverage)
    assert result['fixed_parent'] == 'E_pca'
    assert result['source_preservation']['pass'] is True
    assert result['residential_retention']['parent_headroom'] == pytest.approx(.2)
    assert result['residential_retention']['retained_fraction'] == pytest.approx(.75)
    assert result['joint_pass'] is True
    # A task-only model with a lower source loss cannot silently replace PCA.
    assert result['source_preservation']['tasks']['income_binary']['parent_log_loss'] == .4


def test_nonpositive_parent_gain_or_limited_race_never_becomes_a_pass():
    utility = {**dict.fromkeys(SOURCE_TASKS, .4), 'same_residence': .3, 'commute_over20': .6}
    result = pca_policy(utility, utility, {'B_rich_bank': .5, 'C_tree_bank': .55},
                        {'SEX': .9, 'RAC1P': .7}, {'SEX': .85, 'RAC1P': .4},
                        {'SEX': .8, 'RAC1P': .8},
                        {'SEX': {'complete': True}, 'RAC1P': {'complete': False, 'limitations': {'fit': [3]}}})
    assert result['attribute_halving']['SEX']['parent_attribute_gain'] < 0
    assert result['attribute_halving']['SEX']['pass'] is None
    assert result['attribute_halving']['RAC1P']['numeric_halving_inequality'] is True
    assert result['attribute_halving']['RAC1P']['pass'] is None
    assert result['joint_pass'] is None


def test_catchup_diagnostic_keeps_negative_gains_and_split_differences():
    fresh = row('mlp_0', independent=True, val=.5, test=.3)
    catchup = row('catchup', selected=True, val=.4, test=.6)
    saved = row('saved_adversary', val=.7, test=.8)
    prior = {**row('prior', selected=True, independent=True, val=.55, test=.55), 'release': 'prior'}
    records = [{'seed': 0, 'raw_metrics': [fresh, catchup, saved, prior]}]
    index = make_index(records)
    vals = [r for r in catchup_diagnostics(records, index) if r['release'] == 'D_protected' and r['target'] == 'SEX']
    validation = next(r for r in vals if r['split'] == 'validation')
    development = next(r for r in vals if r['split'] == 'development_evaluation')
    assert validation['primary_minus_independent'] == pytest.approx(-.1)
    assert development['primary_minus_independent'] == pytest.approx(.3)
    assert development['primary_attribute_gain'] == pytest.approx(-.05)


def test_protected_minus_task_only_keeps_all_five_utility_tasks_paired():
    raw = []
    for i, target in enumerate(UTILITY_TASKS):
        for release, delta in [('C_bottleneck', 0.), ('D_protected', .01 * (i - 2))]:
            raw.append(dict(seed=0, role='transfer', release=release, target=target,
                            candidate_id='logistic', family='logistic', selected=True,
                            selected_within_family=True, auc_selected=False,
                            validation={'log_loss': .4 + delta}, test={'log_loss': .5 + delta}))
    records = [{'seed': 0, 'raw_metrics': raw}]
    pairs = paired_rows(records, make_index(records))
    selected = [r for r in pairs if r['comparison'] == 'protected_minus_task_only' and r['role'] == 'transfer'
                and r['selector'] == 'selected' and r['split'] == 'development_evaluation' and r['metric'] == 'log_loss']
    assert {r['target'] for r in selected} == set(UTILITY_TASKS)
    for row_ in selected:
        assert row_['left'] == 'D_protected'
        assert row_['right'] == 'C_bottleneck'
        assert row_['left_minus_right'] == pytest.approx(.01 * (UTILITY_TASKS.index(row_['target']) - 2))
