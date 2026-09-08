"""Stage reporting must preserve the fork contrasts and audit boundary."""
import copy

import pytest

import scripts.summarize_acs_pca16_init as reporting
from scripts.summarize_acs_pca16_init import (
    make_index, paired_rows, get_value, precise, UTILITY_TASKS, STAGE_PAIRS,
)


def row(release, role='transfer', target='same_residence', candidate='logistic', selected=True, independent=True, loss=.5):
    return dict(seed=0, role=role, release=release, target=target, candidate_id=candidate,
                family='catchup' if candidate == 'catchup' else candidate.split('_')[0],
                selected=selected, independent_selected=independent,
                selected_within_family=True, auc_selected=False, validation={'log_loss': loss}, test={'log_loss': loss})


def test_final_audits_use_fresh_primary_and_preserve_inclusive_choices():
    raw = [row('D_init', 'audit', 'SEX', 'mlp_0', False, True, .6),
           row('D_init', 'audit', 'SEX', 'catchup', True, False, .5)]
    before = copy.deepcopy(raw)
    index = make_index([{'seed': 0, 'raw_metrics': raw}])
    args = (index, 0, 'audit', 'D_init', 'SEX', 'test')
    assert get_value(*args, selector='primary') == .6
    assert get_value(*args, selector='catchup_inclusive') == .5
    assert get_value(*args, selector='family:mlp') == .6
    assert raw == before


def test_unaudited_snapshots_cannot_acquire_fabricated_audit_scores():
    with pytest.raises(ValueError, match='no attribute audits'):
        make_index([{'seed': 0, 'raw_metrics': [row('I', 'audit', 'SEX')]}])


def test_all_stage_contrasts_preserve_direction_and_each_task():
    levels = {'PCA16': .4, 'I': .4, 'W': .42, 'C_init': .43, 'D_init': .435,
              'C_bottleneck': .45, 'D_protected': .46}
    raw = [row(release, target=target, loss=level) for release, level in levels.items() for target in UTILITY_TASKS]
    pairs = paired_rows([0], make_index([{'seed': 0, 'raw_metrics': raw}]))
    selected = [p for p in pairs if p['selector'] == 'primary' and p['split'] == 'development_evaluation' and p['metric'] == 'log_loss' and p['role'] == 'transfer']
    assert len(selected) == len(STAGE_PAIRS) * len(UTILITY_TASKS)
    for result in selected:
        assert result['left_minus_right'] == pytest.approx(levels[result['left']] - levels[result['right']])
    assert not any(p['role'] == 'audit' and (p['left'] in ('I', 'W') or p['right'] in ('I', 'W')) for p in pairs)


def test_tiny_identity_difference_does_not_print_as_exact_zero():
    assert precise(1e-10) == '1.000e-10'
    assert precise(-1e-10) == '-1.000e-10'
    assert precise(0.) == '0.000000'


def test_full_final_criteria_keep_race_coverage_limit_under_both_audit_sets(monkeypatch):
    monkeypatch.setattr(reporting, 'audit_coverage', lambda index, selections, seed, release, target, split, selector:
                        {'complete': target == 'SEX', 'limitations': {} if target == 'SEX' else {'fit': [3]}})
    releases = ('E_pca', *reporting.FINAL, *reporting.BANKS)
    raw = []
    for release in releases:
        for target in UTILITY_TASKS:
            loss = (.25 if release == 'E_pca' else .375 if release in reporting.FINAL else .5) if target == 'same_residence' else .5
            raw.append(row(release, target=target, loss=loss))
        for target in reporting.ATTRIBUTES:
            raw.append(row(release, 'audit', target, 'mlp_0', loss=.625 if release in reporting.FINAL else .5))
    raw += [row('prior', 'audit', target, 'prior', loss=.75) for target in reporting.ATTRIBUTES]
    policies, banks = reporting.final_criteria([0], make_index([{'seed': 0, 'raw_metrics': raw}]), {0: {}})
    assert len(policies) == 8  # Two final arms × two splits × two audit sets.
    for policy in policies:
        assert policy['fixed_parent'] == 'E_pca'
        assert policy['source_preservation']['pass'] is True
        assert policy['residential_retention']['pass'] is True
        assert policy['attribute_halving']['SEX']['pass'] is True
        assert policy['attribute_halving']['RAC1P']['numeric_halving_inequality'] is True
        assert policy['attribute_halving']['RAC1P']['pass'] is None
        assert policy['joint_pass'] is None
    assert len(banks) == 32
    assert all(r['numeric_joint_inequalities'] is True and r['joint_pass'] is None for r in banks)
