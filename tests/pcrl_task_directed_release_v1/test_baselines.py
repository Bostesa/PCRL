"""Synthetic contracts for the supervised full auxiliary-channel erasers."""
from itertools import product
import json

import numpy as np
import pytest


from experiments.pcrl_task_directed_release_v1 import baselines


def fixture(seed=47, n=720):
    rng = np.random.default_rng(seed)
    protected = {'SEX': rng.integers(0, 2, n), 'RAC1P': rng.choice([0, 2, 8], n)}
    task = {name: rng.integers(0, 2, n) for name in
            ('same_residence', 'income_binary', 'civilian_at_work')}
    x = rng.normal(size=(n, 33))
    x[:, 0] += 2 * protected['SEX']
    x[:, 1] += 2 * (protected['RAC1P'] == 2)
    x[:, 2] += 2 * (protected['RAC1P'] == 8)
    for j, y in enumerate(task.values()):
        x[:, j + 3] += 2 * y
    x[:, -1] += 3 * task['same_residence']
    return x, protected, task


def cross(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return (x - x.mean(0)).T @ (y - y.mean(0)) / (len(x) - 1)


def concepts(protected):
    return np.column_stack([y == value for y in protected.values() for value in np.unique(y)])


def test_joint_multiclass_erasure_and_three_task_preservation(tmp_path):
    """Fails if erasing a scalar or the wrong projection orientation, or omitting a task."""
    x, protected, task = fixture()
    fitted = baselines.fit_supervised_erasers(x, protected, task, out_dir=tmp_path)
    z, y = concepts(protected), np.column_stack(list(task.values()))
    assert np.max(np.abs(cross(x, z))) > .1
    for eraser in fitted.values():
        assert not isinstance(eraser, dict)
        value = eraser.transform(x)
        assert value.shape == x.shape and np.isfinite(value).all()
        np.testing.assert_allclose(cross(value, z), 0, atol=1e-9)
        np.testing.assert_allclose(value.mean(0), x.mean(0), atol=1e-10)
        np.testing.assert_allclose(eraser.transform(value), value, atol=1e-8)
    np.testing.assert_allclose(cross(fitted['splince_supervised'].transform(x), y),
                               cross(x, y), atol=1e-9)
    assert fitted['leace_supervised'].metadata['protected_observed_classes']['RAC1P'] == [0, 2, 8]


def test_leace_least_change_retains_label_independent_noise(tmp_path):
    """Direct hand-derived oracle: subtract conditional mean shift, not an orthogonal feature cut."""
    s, u, v = np.array(list(product([-1., 1.], repeat=3))).T
    x = np.zeros((len(s), 33))
    x[:, 0], x[:, 1] = s + 2 * u, u
    protected = {'SEX': ((s + 1) / 2).astype(int), 'RAC1P': np.zeros(len(s), int)}
    task = {'same_residence': ((v + 1) / 2).astype(int),
            'income_binary': ((u + 1) / 2).astype(int),
            'civilian_at_work': ((v + 1) / 2).astype(int)}
    eraser = baselines.fit_supervised_erasers(x, protected, task, out_dir=tmp_path)['leace_supervised']
    expected = x.copy()
    expected[:, 0] = 2 * u
    np.testing.assert_allclose(eraser.transform(x), expected, atol=1e-10)
    # Constant/off-support coordinates are preserved, rather than zeroed at deployment.
    deployed = x.copy()
    deployed[:, 10] = 17
    np.testing.assert_allclose(eraser.transform(deployed)[:, 10], 17, atol=1e-12)


@pytest.mark.parametrize('complement', [False, True])
def test_exact_intersection_is_reported_without_fallback(tmp_path, complement):
    """A task identical to a protected label cannot be both erased and covariance-preserved."""
    x, protected, task = fixture()
    task['same_residence'] = (1 - protected['SEX'] if complement else protected['SEX'].copy())
    fitted = baselines.fit_supervised_erasers(x, protected, task, out_dir=tmp_path)
    diagnostic = fitted['splince_supervised']
    assert isinstance(diagnostic, dict)
    assert diagnostic['status'] == 'infeasible'
    assert diagnostic['intersection_dimension'] >= 1
    certificate = diagnostic['infeasibility_certificate']
    assert certificate['kind'] == 'exact_binary_label_relation_and_nonzero_covariance'
    assert certificate['task'] == 'same_residence'
    assert certificate['protected'] == 'SEX'
    assert int(certificate['exact_covariance_numerator']) != 0
    assert diagnostic['fallback_used'] is False
    assert not (tmp_path / 'splince_supervised' / 'map.npz').exists()
    saved = json.loads((tmp_path / 'splince_supervised' / 'diagnostics.json').read_text())
    assert saved['status'] == 'infeasible'


def test_maps_round_trip_without_pickle_and_include_covariances(tmp_path):
    """Saved artifacts must contain the fitted operation, not only scores or a refit recipe."""
    x, protected, task = fixture()
    fitted = baselines.fit_supervised_erasers(x, protected, task, out_dir=tmp_path)
    for name, original in fitted.items():
        restored = baselines.AffineEraser.load(tmp_path / name)
        np.testing.assert_array_equal(restored.transform(x), original.transform(x))
        with np.load(tmp_path / name / 'map.npz', allow_pickle=False) as z:
            assert {'projection', 'mean', 'covariance_xx', 'covariance_xz', 'covariance_xy',
                    'whitener', 'unwhitener', 'fit_row_indices'} <= set(z.files)
            assert z['projection'].shape == (33, 33)
        assert json.loads((tmp_path / name / 'diagnostics.json').read_text())['status'] == 'fitted'
    with pytest.raises(FileExistsError):
        baselines.fit_supervised_erasers(x, protected, task, out_dir=tmp_path)


def test_missing_labels_are_excluded_never_fabricated(tmp_path):
    """Complete-case masks must record the actual distinct sensitive/task missingness."""
    x, protected, task = fixture()
    protected['SEX'][0] = -1
    task['same_residence'][1] = -1
    fitted = baselines.fit_supervised_erasers(x, protected, task, out_dir=tmp_path)
    assert fitted['leace_supervised'].metadata['fit_rows'] == len(x) - 1
    assert fitted['splince_supervised'].metadata['fit_rows'] == len(x) - 2
    for name, want in [('leace_supervised', 1), ('splince_supervised', 2)]:
        with np.load(tmp_path / name / 'map.npz', allow_pickle=False) as z:
            np.testing.assert_array_equal(z['fit_row_indices'], np.arange(want, len(x)))


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_and_outputs_are_rejected(tmp_path, bad):
    """No nonfinite input can reach fitting or deployment silently."""
    x, protected, task = fixture()
    invalid = x.copy()
    invalid[0, 0] = bad
    with pytest.raises(ValueError, match='finite'):
        baselines.fit_supervised_erasers(invalid, protected, task, out_dir=tmp_path / 'bad')
    eraser = baselines.fit_supervised_erasers(x, protected, task, out_dir=tmp_path / 'good')['leace_supervised']
    with pytest.raises(ValueError, match='finite'):
        eraser.transform(invalid)


def test_h_is_not_accepted_and_auxiliary_fit_is_pure(tmp_path):
    """Reject accidentally appended H_A; never mutate caller input or accept an H parameter."""
    x, protected, task = fixture()
    before = x.copy()
    h = np.ones((len(x), 4))
    with pytest.raises(ValueError, match='33'):
        baselines.fit_supervised_erasers(np.column_stack([h, x]), protected, task, out_dir=tmp_path / 'wrong')
    with pytest.raises(TypeError):
        baselines.fit_supervised_erasers(x, protected, task, h_a=h, out_dir=tmp_path / 'wrong_kw')
    eraser = baselines.fit_supervised_erasers(x, protected, task, out_dir=tmp_path / 'good')['leace_supervised']
    eraser.transform(x)
    np.testing.assert_array_equal(x, before)
    np.testing.assert_array_equal(h, np.ones((len(x), 4)))
    with pytest.raises(ValueError, match='33'):
        eraser.transform(np.column_stack([h, x]))


def test_label_schema_and_insufficient_complete_cases_rejected(tmp_path):
    x, protected, task = fixture()
    for bad in [1.5, 9, -2, np.nan]:
        invalid = {k: v.astype(float) for k, v in protected.items()}
        invalid['RAC1P'][0] = bad
        with pytest.raises(ValueError):
            baselines.fit_supervised_erasers(x, invalid, task, out_dir=tmp_path / str(bad))
    invalid = {**protected, 'RAC1P': np.full(len(x), -1)}
    with pytest.raises(ValueError, match='complete'):
        baselines.fit_supervised_erasers(x, invalid, task, out_dir=tmp_path / 'empty')


def test_near_intersection_is_numerical_refusal_not_false_infeasibility(tmp_path):
    """A one-dimensional U'V can have condition number1 yet amplify by millions."""
    a, b, noise = np.array(list(product([0., 1.], repeat=3))).T
    target = (a * b).astype(int)
    x = np.zeros((len(a), 33))
    x[:, 0] = a
    x[:, 1] = noise + 1e-7 * target
    protected = {'SEX': a.astype(int), 'RAC1P': np.zeros(len(a), int)}
    tasks = {key: target for key in ('same_residence', 'income_binary', 'civilian_at_work')}
    diagnostic = baselines.fit_supervised_erasers(x, protected, tasks, out_dir=tmp_path)['splince_supervised']
    assert isinstance(diagnostic, dict)
    assert diagnostic['status'] == 'numerically_unresolved'
    assert diagnostic['intersection_dimension'] == 0
    assert diagnostic['inverse_norm_UtV'] > 1e6
    assert not (tmp_path / 'splince_supervised' / 'map.npz').exists()


def test_numerical_rank_collapse_does_not_prove_exact_infeasibility(tmp_path):
    """Independent directions below SVD resolution remain algebraically distinct."""
    a, b, noise = np.array(list(product([0., 1.], repeat=3))).T
    target = (a * b).astype(int)
    x = np.zeros((len(a), 33))
    x[:, 0], x[:, 1] = a, noise + 1e-11 * target
    protected = {'SEX': a.astype(int), 'RAC1P': np.zeros(len(a), int)}
    tasks = {key: target for key in ('same_residence', 'income_binary', 'civilian_at_work')}
    directions = np.column_stack([cross(x, a)[:2], cross(x, target)[:2]])
    assert np.linalg.det(directions) != 0
    diagnostic = baselines.fit_supervised_erasers(x, protected, tasks, out_dir=tmp_path)['splince_supervised']
    assert diagnostic['intersection_dimension'] >= 1  # numerical rank is still refused
    assert diagnostic['status'] == 'numerically_unresolved'
    assert diagnostic['reason'] == 'numerical_rank_intersection_without_exact_certificate'
    assert diagnostic['infeasibility_certificate'] is None
    assert not (tmp_path / 'splince_supervised' / 'map.npz').exists()


def test_finite_but_overflowing_covariance_is_refused(tmp_path):
    """Finite values do not justify producing infinities during second-moment fitting."""
    x, protected, tasks = fixture()
    x *= 1e200
    with pytest.raises(ValueError, match='[Nn]onfinite'):
        baselines.fit_supervised_erasers(x, protected, tasks, out_dir=tmp_path)


def test_splince_matches_independent_constrained_least_squares_oracle(tmp_path):
    """Constraint checks alone miss a feasible but unnecessarily destructive map."""
    x, protected, tasks = fixture()
    fitted = baselines.fit_supervised_erasers(x, protected, tasks, out_dir=tmp_path)['splince_supervised']
    cxx = cross(x, x)
    cxz = cross(x, concepts(protected))
    cxy = cross(x, np.column_stack(list(tasks.values())))
    # Independent constrained Frobenius problem in Cholesky coordinates:
    # minimize ||B-L||_F subject to B A=[0,Cxy], with B=P L.
    l = np.linalg.cholesky(cxx)
    w = np.linalg.inv(l)
    a = w @ np.column_stack([cxz, cxy])
    target = np.column_stack([np.zeros_like(cxz), cxy])
    b = l + (target - l @ a) @ np.linalg.pinv(a)
    oracle = b @ w
    np.testing.assert_allclose(fitted.projection, oracle, atol=1e-9)
