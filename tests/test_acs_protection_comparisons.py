"""Exact provisional-margin checks; no fitting, real data, or held-out access."""
import math

import pytest

from scripts.summarize_acs_protection import (
    ATTRIBUTES, SOURCE_TASKS, attribute_halving, coverage_status,
    feature_bank_comparison, joint_status, loss_headroom, parent_comparison,
    residential_retention, source_preservation, statistics, make_index, get_value,
)


def full_coverage(k=2):
    return coverage_status(k, [30] * k, [20] * k, [10] * k, [1.] * k)


@pytest.mark.parametrize('parent,erased,prior', [(.5, .4, .5), (.6, .5, .5), (None, .4, .5), (.4, math.nan, .5)])
def test_nonpositive_or_undefined_headroom_has_no_retention_ratio(parent, erased, prior):
    result = loss_headroom(parent, erased, prior)
    assert result['ratio_defined'] is False
    assert result['retained_fraction'] is None
    assert result['headroom_reduction_fraction'] is None


def test_source_margin_applies_to_each_task_without_averaging():
    parent = dict.fromkeys(SOURCE_TASKS, .5)
    erased = {**parent, 'income_binary': .51, 'civilian_at_work': .49}
    assert source_preservation(parent, erased)['pass'] is True
    erased['public_coverage'] = .510001
    result = source_preservation(parent, erased)
    assert result['pass'] is False
    assert result['tasks']['public_coverage']['pass'] is False
    del erased['public_coverage']
    assert source_preservation(parent, erased)['pass'] is None


def test_residential_margin_uses_stronger_bank_on_the_same_split():
    # Stronger bank .5, parent .25: retaining half requires erased <= .375.
    passed = residential_retention(.25, .375, .5, .75)
    assert passed['reference_log_loss'] == .5
    assert passed['retained_fraction'] == .5
    assert passed['pass'] is True
    assert residential_retention(.25, .375001, .5, .75)['pass'] is False
    assert residential_retention(.5, .4, .5, .75)['pass'] is None
    assert residential_retention(.25, .3, None, .75)['pass'] is None


def test_attribute_halving_keeps_signed_gain_and_requires_positive_headroom():
    result = attribute_halving(.5, .625, .75, full_coverage())
    assert result['parent_attribute_gain'] == .25
    assert result['erased_attribute_gain'] == .125
    assert result['pass'] is True
    assert attribute_halving(.5, .62499, .75, full_coverage())['pass'] is False
    beyond_prior = attribute_halving(.5, .875, .75, full_coverage())
    assert beyond_prior['erased_attribute_gain'] == -.125
    assert beyond_prior['pass'] is True
    assert attribute_halving(.75, .8, .75, full_coverage())['pass'] is None
    assert attribute_halving(.8, .9, .75, full_coverage())['pass'] is None


def test_rare_category_coverage_and_exposed_control_failures_are_not_passes():
    support = [20] * 9
    missing = support.copy()
    missing[3] = 0
    limited = coverage_status(9, missing, support, support, [1.] * 9)
    assert limited['complete'] is False
    assert limited['limitations'] == {'fit': [3]}
    result = attribute_halving(.5, .75, 1., limited)
    assert result['pass'] is None
    assert result['ratio_defined'] is False
    assert result['retained_fraction'] is None
    assert result['numeric_halving_inequality'] is True
    recalls = [1.] * 9
    recalls[4] = 0.
    weak_control = coverage_status(9, support, support, support, recalls)
    assert weak_control['complete'] is False
    assert weak_control['limitations'] == {'exposed_recall': [4]}
    assert attribute_halving(.5, .75, 1., weak_control)['pass'] is None


def test_coverage_requires_fixed_schema_at_each_stage():
    assert full_coverage(9)['complete'] is True
    result = coverage_status(9, [1] * 9, [1] * 8, [1] * 9, [1.] * 9)
    assert result['complete'] is False
    assert result['limitations']['validation'] == list(range(9))
    assert coverage_status(2, [1, 1], [1, 1], [1, 0], [1., None])['complete'] is False


def test_feature_bank_comparison_needs_utility_and_both_attribute_margins():
    coverage = {target: full_coverage(2 if target == 'SEX' else 9) for target in ATTRIBUTES}
    feature = {'SEX': .5, 'RAC1P': 1.}
    bank = {'SEX': .505, 'RAC1P': 1.}
    prior = {'SEX': .75, 'RAC1P': 1.5}
    assert feature_bank_comparison(.49, .5, feature, bank, prior, coverage)['joint_pass'] is True
    assert feature_bank_comparison(.490001, .5, feature, bank, prior, coverage)['joint_pass'] is False
    feature['SEX'] = .499999
    # Better race leakage cannot cancel a sex-margin failure.
    feature['RAC1P'] = 1.2
    failed = feature_bank_comparison(.48, .5, feature, bank, prior, coverage)
    assert failed['attributes']['SEX']['pass'] is False
    assert failed['joint_pass'] is False
    feature['SEX'] = .5
    coverage['RAC1P'] = {'complete': False, 'limitations': {'fit': [3]}}
    limited = feature_bank_comparison(.48, .5, feature, bank, prior, coverage)
    assert limited['joint_pass'] is None
    assert limited['attributes']['RAC1P']['numeric_inequality'] is True
    assert limited['attributes']['RAC1P']['pass'] is None


def test_validation_and_development_are_assessed_independently():
    parent = {**dict.fromkeys(SOURCE_TASKS, .5), 'same_residence': .25, 'commute_over20': .6}
    erased = {**dict.fromkeys(SOURCE_TASKS, .5), 'same_residence': .375, 'commute_over20': .9}
    coverage = {target: full_coverage(2 if target == 'SEX' else 9) for target in ATTRIBUTES}
    args = ({'B_rich_bank': .5, 'C_tree_bank': .75}, dict.fromkeys(ATTRIBUTES, .5),
            dict.fromkeys(ATTRIBUTES, .625), dict.fromkeys(ATTRIBUTES, .75), coverage)
    validation = parent_comparison(parent, erased, *args)
    assert validation['joint_pass'] is True
    assert validation['commute_margin'] is None
    assert validation['utility_differences']['commute_over20'] == pytest.approx(.3)
    development = parent_comparison(parent, {**erased, 'same_residence': .4}, *args)
    assert development['joint_pass'] is False
    assert validation['joint_pass'] is True  # No cross-split mutable state.


def test_joint_logic_never_turns_incomplete_evidence_into_a_pass():
    assert joint_status([True, True]) is True
    assert joint_status([True, None]) is None
    assert joint_status([False, None]) is False
    assert joint_status([]) is None


def test_seed_aggregation_preserves_undefined_categories():
    result = statistics([.3, None, .5])
    assert result['mean'] is None
    assert result['sample_sd'] is None
    assert result['n_defined'] == 2
    assert statistics([.3])['sample_sd'] is None
    assert statistics([.3, .5])['sample_sd'] == pytest.approx(math.sqrt(.02))


def test_report_reuses_saved_primary_family_and_auroc_selections():
    common = dict(seed=0, role='audit', release='D_compressed', target='SEX', family='mlp')
    # Development winner differs from validation-selected primary; do not reselect.
    rows = [{**common, 'candidate_id': 'mlp_0', 'selected': True, 'selected_within_family': True,
             'auc_selected': False, 'validation': {'log_loss': .5}, 'test': {'log_loss': .6}},
            {**common, 'candidate_id': 'mlp_1', 'selected': False, 'selected_within_family': False,
             'auc_selected': True, 'validation': {'log_loss': .55}, 'test': {'log_loss': .4}}]
    index = make_index([{'seed': 0, 'raw_metrics': rows}])
    args = (index, 0, 'audit', 'D_compressed', 'SEX', 'test')
    assert get_value(*args) == .6
    assert get_value(*args, selector='family:mlp') == .6
    assert get_value(*args, selector='auc_selected') == .4
    assert get_value(*args, selector='candidate:mlp_1') == .4
