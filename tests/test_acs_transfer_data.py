"""Small schema/isolation regressions; no real ACS data or final test is read."""
from copy import deepcopy
import itertools
import json

import numpy as np
import pandas as pd
import pytest

from experiments import acs_transfer_data as data


def frame(n=120):
    i = np.arange(n)
    return pd.DataFrame({
        'SERIALNO': [f'001{j // 2:05d}' for j in i],
        'SPORDER': i % 2 + 1, 'PWGTP': np.ones(n),
        'AGEP': 19 + i % 16, 'WKHP': 10 + i % 40,
        'SCHL': 1 + i % 24, 'MAR': 1 + i % 5,
        'RELP': i % 18, 'CIT': 1 + i % 5,
        'DIS': 1 + i % 2, 'DEAR': 1 + i % 2,
        'DEYE': 1 + i % 2, 'DREM': 1 + i % 2,
        'PINCP': i * 1000., 'ESR': 1 + i % 6,
        'PUBCOV': 1 + i % 2, 'MIG': 1 + i % 3,
        'JWMNP': 1 + i % 70, 'SEX': 1 + i % 2,
        'RAC1P': 1 + i % 9,
    })


def test_allowlist_excludes_tasks_aliases_identifiers_and_masks(tmp_path):
    assert data.FEATURES == (
        'AGEP', 'WKHP', 'SCHL', 'MAR', 'RELP', 'CIT',
        'DIS', 'DEAR', 'DEYE', 'DREM',
    )
    forbidden = set(data.SOURCE_COLUMNS + data.HELDOUT_COLUMNS
                    + data.AUDIT_COLUMNS + data.KEY_COLUMNS)
    forbidden.update({'COW', 'NATIVITY', 'FER', 'MIGPUMA', 'MIGSP', 'RAC2P',
                      'RACWHT', 'WAGP', 'HINS4', 'HICOV', 'WRK', 'NWAB',
                      'JWAP', 'JWDP', 'commute_valid', 'income_binary'})
    assert not forbidden.intersection(data.FEATURES)
    original = frame()
    for name in forbidden - set(original):
        original[name] = 123
    path = tmp_path / 'raw.csv'
    original.to_csv(path, index=False)
    loaded, _ = data.load_cohort(path, cap=1000)
    assert not (forbidden - set(data.SOURCE_COLUMNS + data.HELDOUT_COLUMNS
                                + data.AUDIT_COLUMNS + data.KEY_COLUMNS)).intersection(loaded)
    # A features-only frame succeeds: preprocessing cannot depend on labels,
    # their masks, identifiers, or survey weights being present.
    features = loaded.loc[:, data.FEATURES]
    preprocessor = data.CovariatePreprocessor().fit(features)
    result = preprocessor.transform(features)
    assert np.isfinite(result).all()
    assert preprocessor.metadata()['fitted_columns'] == list(data.FEATURES)


def test_household_sampling_is_complete_and_label_independent(tmp_path):
    first = frame(240)
    path = tmp_path / 'first.csv'
    first.to_csv(path, index=False)
    selected, metadata = data.load_cohort(path, cap=71, sample_seed=14)
    assert len(selected) == 70
    assert metadata['sample_households'] == 35
    assert selected.groupby('SERIALNO').size().eq(2).all()
    assert selected.SERIALNO.str.startswith('001').all()
    altered = first.copy()
    for name in data.SOURCE_COLUMNS + data.HELDOUT_COLUMNS + data.AUDIT_COLUMNS:
        altered[name] = np.arange(len(altered))[::-1]
    second_path = tmp_path / 'second.csv'
    altered.to_csv(second_path, index=False)
    second, second_metadata = data.load_cohort(second_path, cap=71, sample_seed=14)
    assert selected[['SERIALNO', 'SPORDER']].equals(second[['SERIALNO', 'SPORDER']])
    assert metadata['raw_row_hash'] == second_metadata['raw_row_hash']


def test_cohort_boundaries_and_duplicate_counts(tmp_path):
    source = frame(8)
    source['AGEP'] = [18, 19, 34, 35, 25, 25, 25, 25]
    source['PWGTP'] = [1, 1, 1, 1, 0, -1, np.nan, 1]
    # Three identical copies of the final person remove two rows, not one.
    source = pd.concat([source, source.iloc[[-1]], source.iloc[[-1]]], ignore_index=True)
    path = tmp_path / 'duplicates.csv'
    source.to_csv(path, index=False)
    result, metadata = data.load_cohort(path, cap=1000)
    assert result.AGEP.tolist() == [19, 34, 25]
    assert metadata['cohort_rows_before_dedup'] == 5
    assert metadata['duplicate_rows_removed'] == 2
    conflicting = source.copy()
    conflicting.loc[len(conflicting) - 1, 'WKHP'] = 88
    conflicting.to_csv(path, index=False)
    with pytest.raises(ValueError, match='Conflicting duplicate'):
        data.load_cohort(path)


def test_seven_pools_keep_households_and_person_rows_disjoint():
    source = frame(240)
    pools = data.split_households(source, seed=7)
    assert tuple(pools) == data.POOLS
    assert all(len(indices) > 0 for indices in pools.values())
    assert np.array_equal(np.sort(np.concatenate(list(pools.values()))), np.arange(len(source)))
    for left, right in itertools.combinations(pools, 2):
        assert np.intersect1d(pools[left], pools[right]).size == 0
        assert set(source.iloc[pools[left]].SERIALNO).isdisjoint(source.iloc[pools[right]].SERIALNO)
    again = data.split_households(source, seed=7)
    for key in pools:
        assert np.array_equal(pools[key], again[key])
    assert not np.array_equal(pools['test'], data.split_households(source, seed=8)['test'])


def test_source_labels_only_need_source_columns_and_preserve_missingness():
    source = pd.DataFrame({
        'PINCP': [-19998, 0, 50000, 50001, 4209995, np.nan, -19999, 4209996],
        'ESR': [1, 2, 3, 4, 5, 6, 0, np.nan],
        'PUBCOV': [1, 2, 1, 2, 1, 2, 0, np.nan],
    })
    labels = data.source_labels(source, income_edges=np.array([0., 50000.]))
    assert labels['income_binary'].tolist() == [0, 0, 0, 1, 1, -1, -1, -1]
    assert labels['esr'].tolist() == [0, 1, 2, 3, 4, 5, -1, -1]
    assert labels['pubcov'].tolist() == [0, 1, 0, 1, 0, 1, -1, -1]
    assert labels['joint'].tolist() == [3, 0, 1, 4, 5, -1, -1, -1]
    assert np.array_equal(data.fit_income_edges(pd.DataFrame({'PINCP': [100., 100., np.nan]})), [100.])
    with pytest.raises(ValueError, match='No fitting income'):
        data.fit_income_edges(pd.DataFrame({'PINCP': [np.nan, np.inf, 4209996]}))


def test_heldout_and_audit_validity_masks_are_explicit_and_fixed():
    # No ESR dependency: all Census-valid commuters are eligible, including
    # military workers, and invalid/home-worker observations are never negatives.
    heldout = pd.DataFrame({'MIG': [1, 2, 3, np.nan, 0, 4, 1, 3],
                           'JWMNP': [0, np.nan, 201, 20, 21, 1, 200, 20.5]})
    labels = data.heldout_labels(heldout)
    assert labels['same_residence'].tolist() == [1, 0, 0, -1, -1, -1, 1, 0]
    assert labels['commute_over20'].tolist() == [-1, -1, -1, 0, 1, 0, 1, -1]
    audit = data.audit_labels(pd.DataFrame({'SEX': [1, 2, np.nan, 3], 'RAC1P': [1, 9, np.nan, 10]}))
    assert audit['SEX'].tolist() == [0, 1, -1, -1]
    assert audit['RAC1P'].tolist() == [0, 8, -1, -1]
    coverage = data.support(audit, {'SEX': 2, 'RAC1P': 9})
    assert coverage['RAC1P']['class_counts'] == [1, 0, 0, 0, 0, 0, 0, 0, 1]
    assert coverage['RAC1P']['missing_or_inapplicable'] == 2


def test_preprocessing_is_fit_only_and_distinguishes_unseen_from_invalid():
    fit = frame(4).loc[:, data.FEATURES]
    fit['AGEP'] = [19, 20, 21, 22]
    fit['WKHP'] = [10., np.nan, 30., 0.]
    fit['MAR'] = [1, 1, 3, 3]
    preprocessor = data.CovariatePreprocessor().fit(fit)
    before = deepcopy(preprocessor.metadata())
    assert preprocessor.numeric['WKHP'] == pytest.approx([20., 20., np.sqrt(50.)])
    validation = fit.copy()
    validation['AGEP'] = [34, 34, 34, 34]
    validation['WKHP'] = [99., 99., 99., 99.]
    validation['MAR'] = [2., 99., np.nan, 1.]
    validation['MIG'] = [1, 2, 3, 1]  # Extra target columns remain inaccessible inputs.
    validation['commute_valid'] = True
    transformed = preprocessor.transform(validation)
    assert before == preprocessor.metadata()
    names = preprocessor.feature_names
    assert transformed[:, names.index('MAR=unseen')].tolist() == [1, 0, 0, 0]
    assert transformed[:, names.index('MAR=missing')].tolist() == [0, 1, 1, 0]
    assert transformed[:, names.index('MAR=1')].tolist() == [0, 0, 0, 1]
    assert np.isfinite(transformed).all()


def test_final_test_barrier_requires_all_declared_selections(tmp_path):
    expected = ['source', 'transfer', 'audit']
    with pytest.raises(RuntimeError, match='sealed'):
        data.require_test_selection(tmp_path, expected)
    path = tmp_path / 'selection_before_test.json'
    path.write_text(json.dumps({'head_selections': {'source': {}, 'transfer': {}}}))
    with pytest.raises(RuntimeError, match='sealed'):
        data.require_test_selection(tmp_path, expected)
    path.write_text(json.dumps({'head_selections': {k: {'checkpoint': 'selected'} for k in expected}}))
    digest = data.require_test_selection(tmp_path, expected)
    assert digest == data.sha_file(path)
    assert len(digest) == 64
    # An unexpected replacement key cannot satisfy the barrier by count alone.
    path.write_text(json.dumps({'head_selections': {'source': {}, 'transfer': {}, 'wrong': {}}}))
    with pytest.raises(RuntimeError, match='sealed'):
        data.require_test_selection(tmp_path, expected)
