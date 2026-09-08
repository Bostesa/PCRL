"""Small regressions for the PCA16 diagnostic's changed primary comparison."""
import copy

import pytest

from scripts.summarize_acs_pca16 import (
    make_index, primary_selector, get_value, utility_criteria, SOURCE_TASKS,
)


def audit_row(release, candidate, selected, independent, test):
    return dict(seed=0, role='audit', release=release, target='RAC1P', candidate_id=candidate,
                family='catchup' if candidate == 'catchup' else 'mlp', selected=selected,
                independent_selected=independent, selected_within_family=independent,
                auc_selected=False, validation={'log_loss': .5 if selected else .6}, test={'log_loss': test})


def test_primary_uses_matched_independent_and_preserves_historical_flags():
    raw = [audit_row('D_protected', 'mlp_0', False, True, .8),
           audit_row('D_protected', 'catchup', True, False, .7),
           audit_row('PCA16', 'mlp_1', True, True, .75)]
    raw[1]['selected_within_family'] = True  # Singleton catch-up family.
    original = copy.deepcopy(raw)
    index = make_index([{'seed': 0, 'raw_metrics': raw}])
    args = (index, 0, 'audit', 'D_protected', 'RAC1P', 'test')
    assert get_value(*args, selector='primary') == .8
    assert get_value(*args, selector='historical_catchup_inclusive') == .7
    assert get_value(*args, selector='selected') == .7
    assert get_value(*args, selector='family:mlp') == .8
    assert get_value(*args, selector='family:catchup') == .7
    assert raw == original
    assert primary_selector('transfer', 'D_protected') == 'selected'
    assert primary_selector('audit', 'PCA16') == 'selected'


def test_source_and_residence_parent_stays_original_pca_not_learned_arm():
    parent = {**dict.fromkeys(SOURCE_TASKS, .4), 'same_residence': .25, 'commute_over20': .6}
    method = {**dict.fromkeys(SOURCE_TASKS, .409), 'same_residence': .375, 'commute_over20': .7}
    result = utility_criteria(method, parent, {'B_rich_bank': .5, 'C_tree_bank': .6})
    assert result['parent'] == 'E_pca'
    assert result['source_preservation']['pass'] is True
    assert result['residential_retention']['retained_fraction'] == .5
    assert result['residential_retention']['pass'] is True
    assert result['all_task_pca16_minus_parent']['commute_over20'] == pytest.approx(.1)
    assert 'joint_pass' not in result  # This diagnostic has no all-attribute gate.


def test_source_failure_and_undefined_headroom_remain_distinct():
    parent = {**dict.fromkeys(SOURCE_TASKS, .4), 'same_residence': .5, 'commute_over20': .6}
    method = {**parent, 'income_binary': .41001}
    result = utility_criteria(method, parent, {'B_rich_bank': .5, 'C_tree_bank': .6})
    assert result['source_preservation']['pass'] is False
    assert result['residential_retention']['pass'] is None
    assert result['residential_retention']['retained_fraction'] is None
