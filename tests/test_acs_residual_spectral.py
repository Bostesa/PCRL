"""Small synthetic mathematics checks; never loads ACS or reserved labels."""
import importlib.util
import io

import joblib
import numpy as np
import pytest


def core():
    assert importlib.util.find_spec('experiments.acs_residual_spectral') is not None, 'spectral core must exist'
    from experiments import acs_residual_spectral
    return acs_residual_spectral


def fixture(n=180):
    rng = np.random.default_rng(48)
    T = rng.normal(size=(n, 32))
    a = rng.uniform(.05, .95, (n, 2))
    hA = np.column_stack((1-a[:, 0], a[:, 0], 1-a[:, 1], a[:, 1]))
    b = rng.uniform(.05, .95, n)
    hB = np.column_stack((1-b, b))
    labels = {'SEX': np.arange(n) % 2, 'RAC1P': np.arange(n) % 9, 'public_coverage': (np.arange(n)//2) % 2}
    return T, hA, hB, labels, np.arange(n)//2


def test_fit_is_whitened_optimal_and_serializable_without_B_at_inference():
    m = core()
    T, hA, hB, labels, households = fixture()
    model, d = m.fit_spectral(T, hA, hB, labels, households, seed=0)
    assert model.rank == 128
    assert model.output_rank == 16
    V = model.features(T, hA)
    np.testing.assert_allclose(V.mean(0), 0, atol=1e-10)
    np.testing.assert_allclose(V.T @ V / len(V), np.eye(model.rank), atol=1e-8)
    Q = model.qA.transform(hA[:, [1, 3]])
    np.testing.assert_allclose(Q.T @ V / len(V), 0, atol=1e-9)
    assert set(model.maps) == {'spectral_S0', 'spectral_M025', 'spectral_M1', 'spectral_L025', 'spectral_L1', 'spectral_C025', 'spectral_C1', 'spectral_L2'}
    for arm, W in model.maps.items():
        np.testing.assert_allclose(W.T @ W, np.eye(16), atol=1e-10)
        assert np.all(W[np.argmax(abs(W), axis=0), np.arange(16)] >= 0)
        assert d['arms'][arm]['objective_gap'] < 1e-9
        assert d['arms'][arm]['reconstruction_identity_abs_error'] < 1e-9
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    restored = joblib.load(buf)
    np.testing.assert_array_equal(restored.transform(T[:13], hA[:13], 'spectral_C1'), model.transform(T[:13], hA[:13], 'spectral_C1'))


def test_basis_deduplication_handles_binary_square_and_constant():
    m = core()
    x = np.array([[0., 4.], [1., 4.], [0., 4.], [1., 4.]])
    basis = m.PolynomialBasis.fit(x)
    np.testing.assert_array_equal(basis.transform(x), [[1., -1.], [1., 1.], [1., -1.], [1., 1.]])


def test_moment_normalization_full_onehot_and_missing_mask():
    m = core()
    V = np.array([[-1.], [1.], [-1.], [1.], [999.]])
    y = np.array([0, 1, 0, 1, -1])
    pred = np.tile([.5, .5], (5, 1))
    P, d = m.moment_penalty(V, np.ones((5, 1)), y, pred, 2)
    np.testing.assert_allclose(P, [[1.]])
    assert d['raw_trace'] == .5
    assert d['valid_rows'] == 4
    np.testing.assert_allclose([v['squared_norm'] for v in d['classes']], [.25, .25])


def test_group_folds_and_absent_classes_are_explicit():
    m = core()
    T, hA, hB, labels, households = fixture(72)
    labels['RAC1P'][:] = 0
    labels['SEX'][:6] = -1
    model, d = m.fit_spectral(T, hA, hB, labels, households, seed=4)
    folds = np.asarray(d['fold_assignments'])
    for household in np.unique(households):
        assert len(np.unique(folds[households == household])) == 1
    race = d['nuisances']['local']['RAC1P']
    assert race['unsupported_classes'] == list(range(1, 9))
    assert all(f['predictor'] == 'constant_one_class' for f in race['folds'])
    assert d['penalties']['local']['attributes']['RAC1P']['zero_trace']
    assert d['penalties']['local']['denominator'] == 3
    assert d['penalties']['coalition']['denominator'] == 2
    assert d['penalties']['local']['attributes']['SEX']['valid_rows'] == 66
    heldout = model.moment_diagnostics(T[:12], hA[:12], hB[:12], {k: v[:12] for k, v in labels.items()})
    assert heldout['prediction_source'] == 'equal_ensemble_of_three_training_fold_models'


def test_finite_source_lp_and_counterexamples():
    d = core().numerical_fixtures()
    assert d['independent_useful_sensitive_lp']['error'] == pytest.approx(0)
    assert d['useful_equals_sensitive_lp']['error'] == pytest.approx(.5)
    for name in ('independent_useful_sensitive_lp', 'useful_equals_sensitive_lp'):
        assert d[name]['success']
        assert d[name]['primal_residual_max'] < 1e-9
    assert d['xor']['marginal_moment'] == 0
    assert d['xor']['interaction_moment'] == 1
    assert d['nonlinear_first_moment_failure']['first_moment'] == 0
    assert d['nonlinear_first_moment_failure']['recovery_accuracy'] == 1


def test_rank_deficiency_and_schema_fail_closed():
    m = core()
    T, hA, hB, labels, households = fixture(12)
    model, d = m.fit_spectral(T, hA, hB, labels, households, seed=0)
    assert 0 < model.output_rank < 16
    assert model.transform(T, hA, 'spectral_S0').shape == (12, model.output_rank)
    assert d['rank_mismatch_with_16d_baselines']
    labels['reserved'] = np.zeros(12)
    with pytest.raises(ValueError, match='exactly'):
        m.fit_spectral(T, hA, hB, labels, households, seed=0)


def test_degenerate_bandwidth_and_zero_utility_abort_fitting():
    m = core()
    T, hA, hB, labels, households = fixture(36)
    T[:] = 2.
    with pytest.raises(ValueError, match='positive pairwise'):
        m.fit_spectral(T, hA, hB, labels, households, seed=0)


def test_zero_utility_aborts_fitting():
    m = core()
    T, hA, hB, labels, households = fixture(36)
    T[:] = 2.
    T[:, 0] = hA[:, 1]
    with pytest.raises(ValueError, match='utility trace'):
        m.fit_spectral(T, hA, hB, labels, households, seed=0)


def test_negligible_scale_uses_unit_scale_without_amplification():
    m = core()
    T, hA, hB, labels, households = fixture(36)
    T[:, 0] *= 1e-13
    model, _ = m.fit_spectral(T, hA, hB, labels, households, seed=0)
    assert model.t_scale[0] == 1.


def test_nonconverged_real_nuisance_fit_aborts(monkeypatch):
    m = core()
    original = m.LogisticRegression
    def constrained_solver(**kwargs):
        kwargs['max_iter'] = 1
        return original(**kwargs)
    monkeypatch.setattr(m, 'LogisticRegression', constrained_solver)
    T, hA, hB, labels, households = fixture(36)
    with pytest.raises(ArithmeticError, match='nuisance.*converg'):
        m.fit_spectral(T, hA, hB, labels, households, seed=0)
