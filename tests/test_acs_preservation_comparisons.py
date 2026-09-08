"""Reporting boundaries and predeclared paired directions on artificial scores."""
import copy

import pytest

import scripts.summarize_acs_preservation as report


def row(release, role='transfer', target='same_residence', candidate='logistic',
        selected=True, independent=True, loss=.5, weighted=None):
    score = {'log_loss': loss}
    return {'seed': 0, 'role': role, 'release': release, 'target': target,
            'candidate_id': candidate, 'family': 'catchup' if candidate == 'catchup' else candidate.split('_')[0],
            'selected': selected, 'independent_selected': independent, 'selected_within_family': True,
            'auc_selected': False, 'validation': score.copy(), 'test': score.copy(),
            'validation_person_weighted': {'log_loss': loss if weighted is None else weighted},
            'test_person_weighted': {'log_loss': loss if weighted is None else weighted}}


def test_normalization_preserves_inputs_and_historical_provenance():
    raw = {'seed': 0, 'raw_metrics': [row('W'), row('C_init')], 'source_sha256': {'old.py': 'original'}}
    selection = {'fitting_records': {'transfer/W/same_residence': {'source_sha256': 'old'},
                                     'transfer/C_init/same_residence': {'source_sha256': 'older'}}}
    before = copy.deepcopy((raw, selection))
    record, selected = report.normalize_unit(raw, selection, 'beta_0p1')
    assert (raw, selection) == before
    assert [r['release'] for r in record['raw_metrics']] == ['beta_0p1_W', 'C_init']
    assert selected['fitting_records']['transfer/beta_0p1_W/same_residence']['source_sha256'] == 'old'
    assert record['source_sha256'] == raw['source_sha256']


def test_repeated_references_require_exact_agreement():
    unit = ({'seed': 0, 'raw_metrics': [row('I')]}, {'fitting_records': {}})
    records, _ = report.merge_units([unit, copy.deepcopy(unit)])
    assert len(records[0]['raw_metrics']) == 1
    changed = copy.deepcopy(unit)
    changed[0]['raw_metrics'][0]['test']['log_loss'] = .1
    with pytest.raises(ValueError, match='Inconsistent reused evidence'):
        report.merge_units([unit, changed])


def test_fresh_and_inclusive_selections_and_weighting_are_preserved():
    release = report.FINAL[0]
    rows = [row(release, 'audit', 'SEX', 'mlp_0', False, True, .6, .9),
            row(release, 'audit', 'SEX', 'catchup', True, False, .7, .1)]
    index = report.make_index([{'seed': 0, 'raw_metrics': rows}])
    assert report.get_value(index, 0, 'audit', release, 'SEX', 'test', selector='primary') == .6
    assert report.get_value(index, 0, 'audit', release, 'SEX', 'test', selector='catchup_inclusive') == .7
    assert report.get_value(index, 0, 'audit', release, 'SEX', 'test_person_weighted', selector='primary') == .9
    assert report.get_value(index, 0, 'audit', release, 'SEX', 'test_person_weighted', selector='catchup_inclusive') == .1


@pytest.mark.parametrize('release', report.STAGES)
def test_no_teacher_audit_can_be_relabelled_as_an_unaudited_stage(release):
    with pytest.raises(ValueError, match='no independently executed'):
        report.make_index([{'seed': 0, 'raw_metrics': [row(release, 'audit', 'SEX')]}])


def test_all_requested_directions_and_tasks_remain_in_pairs():
    stages = set(v for _, left, right in report.comparison_matrix() for v in (left, right))
    levels = {r: .1 + .001*j for j, r in enumerate(sorted(stages))}
    raw = [row(r, target=t, loss=levels[r]) for r in stages for t in report.UTILITY_TASKS]
    pairs = report.paired_rows([0], report.make_index([{'seed': 0, 'raw_metrics': raw}]))
    selected = [r for r in pairs if r['role'] == 'transfer' and r['selector'] == 'primary' and r['split'] == 'development_evaluation']
    assert len(selected) == len(report.comparison_matrix()) * len(report.UTILITY_TASKS)
    assert {r['comparison'] for r in selected} == {'historical_warmup', 'warmup', 'warmup_minus_historical',
        'continuation', 'preservation_minus_beta0', 'persistent_minus_warmup_only', 'protection_D_minus_C', 'beta1_minus_beta0p1'}
    for r in selected:
        assert r['left_minus_right'] == pytest.approx(levels[r['left']] - levels[r['right']])
    assert not any(r['role'] == 'audit' and (r['left'] in report.STAGES or r['right'] in report.STAGES) for r in pairs)


def test_missing_seed_is_undefined_instead_of_favorable_mean():
    rows = [{'comparison': 'warmup', 'left': 'a', 'right': 'b', 'role': 'transfer',
             'target': 'same_residence', 'selector': 'primary', 'split': 'development_evaluation',
             'metric': 'log_loss', 'seed': seed, 'left_minus_right': value}
            for seed, value in ((0, -.1), (1, None), (2, -.2))]
    summarized = report.summarize_pairs(rows)[0]
    assert summarized['mean'] is None
    assert summarized['n_defined'] == 2
    assert summarized['per_seed'] == {'0': -.1, '1': None, '2': -.2}


def test_final_criteria_keep_original_parent_and_missing_race_under_both_weights(monkeypatch):
    monkeypatch.setattr(report, 'audit_coverage', lambda index, selection, seed, release, target, split, selector:
        {'complete': target == 'SEX', 'limitations': {} if target == 'SEX' else {'fit': [3]}})
    raw = []
    for release in ('E_pca', *report.FINAL, *report.BANKS, 'C_init', 'D_init'):
        for target in report.UTILITY_TASKS:
            if target == 'same_residence':
                value = .25 if release == 'E_pca' else .375 if release in report.FINAL else .5
            else:
                value = .5 if release == 'E_pca' else .55 if release in report.FINAL else .8
            raw.append(row(release, target=target, loss=value))
        for target in report.ATTRIBUTES:
            raw.append(row(release, 'audit', target, 'mlp_0', loss=.625 if release in report.FINAL else .5))
    raw.extend(row('prior', 'audit', t, 'prior', loss=.75) for t in report.ATTRIBUTES)
    _, policies, banks, _ = report.criteria_rows([0], report.make_index([{'seed': 0, 'raw_metrics': raw}]), {0: {}})
    assert len(policies) == len(report.FINAL) * 4 * 2
    for policy in policies:
        assert policy['fixed_parent'] == 'E_pca'
        assert policy['source_preservation']['pass'] is False
        assert policy['residential_retention']['pass'] is True
        assert policy['attribute_halving']['SEX']['pass'] is True
        assert policy['attribute_halving']['RAC1P']['pass'] is None
        assert policy['joint_pass'] is False
    assert len(banks) == len(policies) * len(report.BANKS)
