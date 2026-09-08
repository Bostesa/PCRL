"""Focused read-only reporting checks; no scientific model fits."""
import pytest

from scripts.summarize_acs_coalition import (
    AUDIT_ROLES, COMPACT_EXPORTS, SCOPES, Evidence, aggregate_metrics, audit_coverage, build_index,
    coalition_singleton_rows, compact_exports, control_row, export_candidates, mean_sd, paired_rows, per_seed_rows,
)


def scored(condition='F_I', view='A', target='SEX', val=.6, test=.6, seed=0,
           scope=SCOPES, cid='wire__logistic', budget=360):
    k = 9 if target == 'RAC1P' else 2
    def score(value):
        return {'log_loss': value, 'n_classes': k, 'support': [4]*k,
                'per_class': [{'class_index': i, 'recall': 1.} for i in range(k)]}
    return dict(seed=seed, condition=condition, role='audit', view=view, target=target,
                audit_budget=budget, candidate_id=cid, selected_scopes=list(scope),
                scores={'validation': score(val), 'test': score(test),
                        'validation_person_weighted': score(val+.01), 'test_person_weighted': score(test+.02)})


def test_opposing_task_audit_is_binary_and_has_no_invented_threshold():
    row = scored(target='public_coverage')
    result = audit_coverage(row, {'fit_support': [10, 20]}, 'test', row)
    assert result['n_classes'] == 2 and result['complete']
    assert result['opposing_task_threshold'] is None
    assert 'limitation_census_codes' not in result


def test_missing_race_category_remains_unassessable():
    row = scored(target='RAC1P')
    support = [10]*9; support[3] = 0
    result = audit_coverage(row, {'fit_support': support}, 'test', row)
    assert not result['complete'] and result['n_classes'] == 9
    assert 4 in result['limitation_census_codes']['fit']


def test_missing_seed_does_not_shrink_race_schema():
    result = audit_coverage(None, {}, 'test', None, target='RAC1P')
    assert result['n_classes'] == 9 and result['complete'] is False
    assert result['limitation_census_codes']['fit'] == list(range(1, 10))


def test_duplicate_selection_rejected_without_development_selection():
    a = scored(cid='one'); b = scored(cid='two', test=.1)
    with pytest.raises(ValueError, match='Multiple selected candidates'):
        build_index([a, b])


def test_duplicate_candidate_identity_rejected():
    row = scored()
    with pytest.raises(ValueError, match='Duplicate candidate'):
        build_index([row, row])


def test_paired_gain_sign_and_weighting_use_same_candidate():
    rows = [scored(condition='F_I', test=.55), scored(condition='F_J', test=.60)]
    index = build_index(rows)
    result = [r for r in paired_rows(index, [0]) if r['comparison'] == 'J_minus_I'
              and r['left'] == 'F_J' and r['view'] == 'A' and r['target'] == 'SEX'
              and r['audit_budget'] == 360 and r['scope'] == SCOPES[0]]
    for row in result:
        if row['split'].startswith('development'):
            assert row['left_minus_right'] == pytest.approx(.05)
            assert row['signed_gain_left_minus_right'] == pytest.approx(-.05)


def test_missing_seed_prevents_favorable_subset_average():
    index = build_index([scored(seed=0)])
    records = per_seed_rows(index, [0, 1, 2])
    aggregate = [r for r in aggregate_metrics(records) if r['condition'] == 'F_I'
                 and r['view'] == 'A' and r['target'] == 'SEX' and r['audit_budget'] == 360
                 and r['scope'] == SCOPES[0] and r['split'] == 'development_evaluation'
                 and r['metric'] == 'log_loss'][0]
    assert aggregate['n_defined'] == 1 and aggregate['n_seeds'] == 3
    assert aggregate['mean'] is None and aggregate['complete'] is False
    assert aggregate['observed_mean'] == pytest.approx(.6)
    assert '(1/3; partial)' in mean_sd(aggregate)


def test_prior_budget_none_reused_without_inventing_fitting_budget():
    prior = scored(condition='prior', view='control', budget=None, scope=('prior',))
    index = build_index([prior])
    assert control_row(index, 0, 'prior', 'SEX', 120) is prior
    assert control_row(index, 0, 'prior', 'SEX', 360) is prior


def test_coalition_validation_inclusion_does_not_force_development_monotonicity():
    index = build_index([scored(view='A', val=.6, test=.4),
                         scored(view='B', val=.65, test=.5),
                         scored(view='AB', val=.55, test=.7)])
    rows = coalition_singleton_rows(index, [0])
    row = next(r for r in rows if r['condition'] == 'F_I' and r['singleton'] == 'A'
               and r['audit_budget'] == 360 and r['target'] == 'SEX' and r['scope'] == SCOPES[0]
               and r['split'] == 'development_evaluation')
    assert row['coalition_minus_singleton_loss'] == pytest.approx(.3)
    assert row['coalition_minus_singleton_gain'] == pytest.approx(-.3)
    assert row['development_monotonicity_required'] is False


def test_illegal_coalition_validation_selection_rejected():
    index = build_index([scored(view='A', val=.5), scored(view='AB', val=.6)])
    with pytest.raises(ValueError, match='included singleton'):
        coalition_singleton_rows(index, [0])


def test_all_eleven_audit_roles_are_declared_and_reserved_roles_distinct():
    assert sum(map(len, AUDIT_ROLES.values())) == 11
    assert 'commute_over20' in AUDIT_ROLES['A']
    assert 'same_residence' in AUDIT_ROLES['B']
    assert AUDIT_ROLES['AB'] == ('SEX', 'RAC1P')


def test_nullable_control_fit_support_stays_undefined_in_exports(tmp_path):
    import csv
    import json
    evidence = Evidence()
    evidence.rows.append({**scored(condition='prior', view='control', budget=None, scope=('prior',)),
        'metadata': {'fit_support': None, 'seed': 1260000}, 'origin_metrics': 'toy/metrics.json'})
    result = export_candidates(tmp_path, evidence)
    assert result['candidate_records'] == 1
    with (tmp_path/'PER_CLASS.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8 and all(r['fit_class_support'] == '' for r in rows)
    lineage = json.loads((tmp_path/'CANDIDATE_LINEAGE.json').read_text())[0]
    assert lineage['seed'] == 0 and lineage['candidate_seed'] == 1260000


def test_compact_publication_roundtrips_deterministically_without_host_filename(tmp_path):
    import gzip
    for name in COMPACT_EXPORTS:
        (tmp_path/name).write_text('fixed evidence\n')
    first = compact_exports(tmp_path)
    second = compact_exports(tmp_path)
    assert first == second
    for name in COMPACT_EXPORTS:
        data = (tmp_path/(name+'.gz')).read_bytes()
        assert gzip.decompress(data) == (tmp_path/name).read_bytes()
        assert data[3] & 8 == 0 and data[4:8] == b'\0'*4
