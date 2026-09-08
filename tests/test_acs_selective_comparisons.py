"""Selective reporting boundaries on artificial saved scores, without fitting."""
import copy

import pytest

from scripts import summarize_acs_selective as report


def row(release, role='transfer', target='same_residence', candidate='logistic',
        selected=True, independent=True, loss=.5, weighted=None, budget=None):
    result = {'seed': 0, 'role': role, 'release': release, 'target': target,
        'candidate_id': candidate, 'family': 'catchup' if candidate == 'catchup' else candidate.split('_')[0],
        'selected': selected, 'independent_selected': independent,
        'selected_within_family': True, 'auc_selected': False}
    result.update({key: {'log_loss': loss if 'weighted' not in key or weighted is None else weighted}
                   for key in report.SPLITS})
    if budget is not None:
        result['audit_budget'] = budget
    return result


def test_historical_alias_preserves_score_and_actual_budget():
    evidence = report.Evidence()
    raw = row('C_persistent', 'audit', 'SEX')
    original = copy.deepcopy(raw)
    metadata = {'source_sha256': {'original.py': 'old-hash'}}
    evidence.add(raw, metadata, 360, report.ROOT/'results/old/metrics.json', report.historical_alias)
    saved = next(iter(evidence.rows[360].values()))
    assert raw == original
    assert saved['release'] == 'R_rho0p1_C'
    assert saved['origin_release'] == 'C_persistent'
    assert saved['evidence_audit_budget'] == 120
    assert saved['report_audit_budget'] == 360
    assert next(iter(evidence.meta[360].values())) == metadata


def test_reuse_conflict_rejected_but_nested_budgets_are_separate():
    evidence = report.Evidence()
    path = report.ROOT/'results/old/metrics.json'
    raw = row('E_direct', 'audit', 'SEX', budget=120)
    evidence.add(raw, {}, 120, path)
    evidence.add(copy.deepcopy(raw), {}, 120, path)
    changed = row('E_direct', 'audit', 'SEX', loss=.2, budget=360)
    with pytest.raises(ValueError, match='Conflicting reused'):
        evidence.add(changed, {}, 120, path)
    evidence.add(changed, {}, 360, path)
    assert len(evidence.rows[120]) == len(evidence.rows[360]) == 1


def test_unweighted_selection_carries_weighted_prediction_without_reselection():
    release = 'E_rho0_D'
    rows = [row(release, 'audit', 'SEX', 'mlp_0', False, True, .6, .9),
            row(release, 'audit', 'SEX', 'catchup', True, False, .7, .1)]
    index = report.index_records([{'seed': 0, 'raw_metrics': rows}])
    assert report.get_value(index, 0, 'audit', release, 'SEX', 'test', selector='primary') == .6
    assert report.get_value(index, 0, 'audit', release, 'SEX', 'test_person_weighted', selector='primary') == .9
    assert report.get_value(index, 0, 'audit', release, 'SEX', 'test_person_weighted', selector='pooled') == .1


@pytest.mark.parametrize('release', report.STAGES)
def test_unaudited_stage_cannot_inherit_teacher_audit(release):
    with pytest.raises(ValueError, match='not independently audited'):
        report.index_records([{'seed': 0, 'raw_metrics': [row(release, 'audit', 'SEX')]}])


def test_full_prespecified_pairs_have_fixed_signs_and_all_five_tasks():
    comparisons = report.comparison_matrix()
    names = {v for _, left, right in comparisons for v in (left, right)}
    values = {name: .1 + i/1000 for i, name in enumerate(sorted(names))}
    raw = [row(name, target=target, loss=values[name]) for name in names for target in report.UTILITY_TASKS]
    index = report.index_records([{'seed': 0, 'raw_metrics': raw}])
    pairs = report.paired_rows([0], {120: index, 360: index})
    utility = [p for p in pairs if p['role'] == 'transfer' and p['split'] == 'development_evaluation']
    assert len(utility) == len(comparisons)*5
    assert {p['comparison'] for p in utility} == {'E_minus_R', 'E_minus_S', 'rho0_minus_rho0p1',
        'D_minus_C', 'W_minus_I', 'final_minus_W', 'new_final_minus_beta0', 'learned_minus_own_teacher'}
    for pair in utility:
        assert pair['left_minus_right'] == pytest.approx(values[pair['left']]-values[pair['right']])
    assert not any(p['role'] == 'audit' and (p['left'] in report.STAGES or p['right'] in report.STAGES) for p in pairs)
    assert sum(c == 'new_final_minus_beta0' for c, _, _ in comparisons) == 10


def test_missing_seed_does_not_create_favorable_subset_mean():
    rows = [{'seed': seed, 'group': 'one', 'left_minus_right': value}
            for seed, value in ((0, -.2), (1, None), (2, -.1))]
    result = report.aggregate_rows(rows, ('group',))[0]
    assert result['mean'] is None and result['n_defined'] == 2
    assert result['per_seed'] == {'0': -.2, '1': None, '2': -.1}


def test_parent_and_support_criteria_survive_both_budgets_and_weights(monkeypatch):
    monkeypatch.setattr(report, 'audit_coverage', lambda index, selection, seed, release, target, split, selector:
        {'complete': target == 'SEX', 'limitations': {} if target == 'SEX' else {'fit': [3]}})
    raw = []
    for release in report.RELEASES:
        for target in report.UTILITY_TASKS:
            value = (.25 if release == 'E_pca' else .375 if release in report.FINAL else .5) if target == 'same_residence' else (
                .5 if release == 'E_pca' else .55 if release in report.FINAL else .8)
            raw.append(row(release, target=target, loss=value))
        if release not in report.STAGES:
            raw.extend(row(release, 'audit', target, loss=.75 if release == 'prior' else .625 if release in report.FINAL else .5)
                       for target in report.ATTRIBUTES)
    index = report.index_records([{'seed': 0, 'raw_metrics': raw}])
    _, policies, banks, _ = report.criteria([0], {120: index, 360: index}, {120: {0: {}}, 360: {0: {}}})
    finals = [p for p in policies if p['release'] in report.FINAL]
    assert len(finals) == len(report.FINAL)*4*2*2
    for policy in finals:
        assert policy['fixed_parent'] == 'E_pca' and policy['parent_actual_audit_budget'] == 120
        assert policy['source_preservation']['pass'] is False
        assert policy['residential_retention']['pass'] is True
        assert policy['attribute_halving']['SEX']['pass'] is True
        assert policy['attribute_halving']['RAC1P']['pass'] is None
        assert policy['joint_pass'] is False
    assert len(banks) == len(finals)*len(report.BANKS)
    parent_changes, bank_changes, ranks = report.audit_change_rows([0], {120: index, 360: index}, policies, banks)
    assert all(r['numeric_changed'] is False for r in parent_changes + bank_changes)
    assert all(r['rank_changed'] is False for r in ranks)


def test_attack_gain_contrast_is_negative_loss_contrast():
    raw = [row('E_rho0_C', 'audit', 'SEX', loss=.6), row('R_rho0_C', 'audit', 'SEX', loss=.4)]
    index = report.index_records([{'seed': 0, 'raw_metrics': raw}])
    pairs = report.paired_rows([0], {120: index, 360: index})
    pair = next(p for p in pairs if p['comparison'] == 'E_minus_R' and p['left'] == 'E_rho0_C'
                and p['target'] == 'SEX' and p['split'] == 'development_evaluation' and p['selector'] == 'primary')
    assert pair['signed_gain_left_minus_right'] == pytest.approx(-pair['left_minus_right'])


def test_reconstruction_interaction_uses_four_matched_scores_and_fixed_sign():
    values = {'E_rho0_C': .6, 'E_rho0p1_C': .5, 'R_rho0_C': .43, 'R_rho0p1_C': .4,
              'S_rho0_C': .52, 'S_rho0p1_C': .5}
    raw = [row(release, 'audit', 'SEX', loss=loss) for release, loss in values.items()]
    index = report.index_records([{'seed': 0, 'raw_metrics': raw}])
    interactions = report.interaction_rows(report.paired_rows([0], {120: index, 360: index}))
    selected = {r['reference_teacher']: r for r in interactions if r['arm'] == 'C' and r['target'] == 'SEX'
                and r['audit_budget'] == 360 and r['selector'] == 'primary' and r['split'] == 'development_evaluation'}
    assert selected['R']['left_minus_right'] == pytest.approx(.07)
    assert selected['S']['left_minus_right'] == pytest.approx(.08)
    assert selected['R']['signed_gain_interaction'] == pytest.approx(-.07)
    missing = [r for r in interactions if r['arm'] == 'D']
    assert all(r['left_minus_right'] is None for r in missing)
