"""Deployment replay contracts; all fixture rows and model parameters are synthetic."""
import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from experiments.acs_transfer_data import CovariatePreprocessor, FEATURES, NUMERIC, CATEGORICAL
from experiments.pcrl_task_directed_release_v1 import inference


def fixture():
    numeric = {'AGEP': [40., 30., 10.], 'WKHP': [40., 35., 5.]}
    categories = {key: ([1, 2, 3, 4, 5] if key == 'SCHL' else [0] if key == 'RELP' else [1])
                  for key in CATEGORICAL}
    names = [v for key in NUMERIC for v in (key, key + '_missing')]
    for key in CATEGORICAL:
        names += [f'{key}={v}' for v in categories[key]] + [key + '=missing', key + '=unseen']
    metadata = {'numeric': numeric, 'categories': categories, 'feature_names': names,
                'numeric_dtype': 'float32', 'fitted_columns': list(FEATURES)}
    assert len(names) == 32
    pca = PCA(n_components=32, svd_solver='full')
    pca.components_ = np.eye(32, dtype=np.float32)
    pca.mean_ = np.arange(32, dtype=np.float32) / 100
    pca.n_features_in_ = pca.n_components_ = 32
    pca.explained_variance_ = np.ones(32)
    mapper = inference.FrozenInputMap(metadata, pca, np.arange(32) / 10,
                                       np.ones(32) * 2, {'fixture': 'synthetic'})
    frame = pd.DataFrame({key: [40., 50., np.nan] if key in NUMERIC else
                         [1., 2., 99.] if key != 'RELP' else [0., 1., 99.] for key in FEATURES})
    h = np.array([[.25, .75, .4, .6], [.6, .4, .9, .1], [.1, .9, .3, .7]], np.float32)
    return mapper, frame, h


def test_exact_historical_preprocessing_and_standardization_without_fit(monkeypatch):
    mapper, frame, h = fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError('Deployment must never fit')
    monkeypatch.setattr(CovariatePreprocessor, 'fit', forbidden)
    monkeypatch.setattr(PCA, 'fit', forbidden)
    historical = CovariatePreprocessor()
    historical.numeric = copy.deepcopy(mapper.preprocessing['numeric'])
    historical.categories = copy.deepcopy(mapper.preprocessing['categories'])
    historical.feature_names = list(mapper.preprocessing['feature_names'])
    expected_pca = mapper.pca.transform(historical.transform(frame))
    expected = ((expected_pca.astype(np.float64) - mapper.input_mean) / mapper.input_scale).astype(np.float32)
    before = frame.copy(deep=True)
    got = mapper.transform(frame, h)
    np.testing.assert_array_equal(got.x_a, expected)
    assert got.x_a.dtype == np.float32
    assert got.h_a.dtype == h.dtype and got.h_a.tobytes() == h.tobytes()
    pd.testing.assert_frame_equal(frame, before)
    assert got.h_a is not h
    # Reordering permitted raw columns cannot change a name-based transform.
    np.testing.assert_array_equal(mapper.transform(frame[list(reversed(FEATURES))], h).x_a, expected)


@pytest.mark.parametrize('extra', ['SEX', 'RAC1P', 'MIG', 'PINCP', 'ESR', 'PUBCOV',
                                  'JWMNP', 'SERIALNO', 'SPORDER', 'PWGTP', 'H_B'])
def test_forbidden_raw_columns_are_rejected(extra):
    mapper, frame, h = fixture()
    with pytest.raises(ValueError, match='exactly|permitted'):
        mapper.transform(frame.assign(**{extra: 1}), h)


def test_missing_duplicate_columns_h_b_and_nonfinite_h_rejected():
    mapper, frame, h = fixture()
    for bad in [frame.drop(columns='AGEP'), pd.concat([frame, frame[['AGEP']]], axis=1)]:
        with pytest.raises(ValueError):
            mapper.transform(bad, h)
    with pytest.raises(ValueError, match='H_A'):
        mapper.transform(frame, np.column_stack((h, np.ones((3, 2)))))
    bad = h.copy(); bad[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        mapper.transform(frame, bad)
    with pytest.raises(TypeError):
        mapper.transform(frame, h, h_b=np.zeros((3, 2)))


def test_export_reload_is_exact_and_hash_guard_precedes_unpickle(tmp_path):
    mapper, frame, h = fixture()
    manifest = mapper.export(tmp_path / 'deploy')
    restored = inference.FrozenInputMap.load_export(tmp_path / 'deploy')
    np.testing.assert_array_equal(restored.transform(frame, h).x_a, mapper.transform(frame, h).x_a)
    assert set(manifest['files']) == {'preprocessing.json', 'pca.joblib', 'standardizer.npz'}
    with pytest.raises(FileExistsError):
        mapper.export(tmp_path / 'deploy')
    path = tmp_path / 'deploy' / 'pca.joblib'
    path.write_bytes(path.read_bytes() + b'corruption')
    with pytest.raises(ValueError, match='hash'):
        inference.FrozenInputMap.load_export(tmp_path / 'deploy')


def test_emitted_wire_contains_only_token_and_byte_identical_h():
    _, _, h = fixture()
    probabilities = np.array([[1., 0., 0.], [0., 0., 1.], [.25, .5, .25]])
    before = probabilities.copy()
    wire = inference.emit_token(h, probabilities, rng=np.random.default_rng(19))
    assert set(wire) == {'h_a', 'token'}
    assert wire['token'].shape == (3,) and wire['token'].dtype.kind in 'iu'
    assert wire['token'][:2].tolist() == [0, 2]
    assert wire['h_a'].dtype == h.dtype and wire['h_a'].tobytes() == h.tobytes()
    assert not wire['h_a'].flags.writeable and not wire['token'].flags.writeable
    np.testing.assert_array_equal(probabilities, before)
    repeated = inference.emit_token(h, probabilities, rng=np.random.default_rng(19))
    np.testing.assert_array_equal(wire['token'], repeated['token'])
    with pytest.raises(ValueError):
        inference.emit_token(h, probabilities * .5, rng=np.random.default_rng(19))
