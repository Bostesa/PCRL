"""Small regressions for the development-only application screen."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import expit

from experiments import screen_pcrl_applications as screen


@pytest.fixture
def har_fixture():
    """All six recorded activities, repeated for 21 synthetic participants."""
    rng = np.random.default_rng(517)
    subject = np.repeat(np.arange(101, 122), 12)
    activity = np.tile(np.repeat(np.arange(6), 2), 21)
    active = (activity < 3).astype(int)
    x = np.column_stack((2*active - 1 + rng.normal(0, .4, len(active)),
                         .5*(subject-100) + rng.normal(0, .2, len(active)),
                         activity + rng.normal(0, .4, len(active)),
                         rng.normal(size=len(active))))
    return x, activity, subject


@pytest.fixture
def fitted_reference(har_fixture, tmp_path):
    x, activity, subject = har_fixture
    result = screen.learned_har_reference(x, activity, subject, tmp_path)
    with np.load(tmp_path/'learned_task_reference.npz') as saved:
        checkpoint = {key: saved[key].copy() for key in saved.files}
    return x, activity, subject, result, checkpoint


def test_learned_reference_keeps_participants_and_rows_disjoint(fitted_reference):
    x, activity, subject, result, checkpoint = fitted_reference
    names = ('task_fit', 'attacker_fit', 'development')
    groups = [set(result['split'][name]['subject_ids']) for name in names]
    assert [len(group) for group in groups] == [13, 4, 4]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert set.union(*groups) == set(subject)
    indices = [checkpoint[f'indices_{name}'] for name in names]
    np.testing.assert_array_equal(np.sort(np.concatenate(indices)), np.arange(len(x)))
    for name, rows, people in zip(names, indices, groups):
        assert set(subject[rows]) == people
        assert result['split'][name]['rows'] == len(rows)
        assert result['task_utility'][name]['activity_support'] == np.bincount(
            activity[rows], minlength=6).tolist()
    assert result['privacy_pass'] is None


def test_scaler_and_saved_predictions_use_only_task_fitting_data(fitted_reference):
    x, _, _, _, checkpoint = fitted_reference
    fit = checkpoint['indices_task_fit']
    np.testing.assert_allclose(checkpoint['scaler_mean'], x[fit].mean(0), atol=1e-12)
    np.testing.assert_allclose(checkpoint['scaler_scale'], x[fit].std(0), atol=1e-12)
    # Participant offsets make fitting on the whole pool detectably different.
    assert np.max(np.abs(checkpoint['scaler_mean'] - x.mean(0))) > .01
    np.testing.assert_array_equal(checkpoint['classes'], [0, 1])
    for name in ('task_fit', 'attacker_fit', 'development'):
        rows = checkpoint[f'indices_{name}']
        standardized = (x[rows] - checkpoint['scaler_mean'])/checkpoint['scaler_scale']
        expected = expit(standardized @ checkpoint['coef'].T + checkpoint['intercept'])[:, 0]
        np.testing.assert_allclose(checkpoint[f'probability_{name}'], expected, atol=1e-12)


def test_binary_task_outputs_are_aligned_with_the_genuine_six_class_attack(fitted_reference):
    _, activity, _, result, checkpoint = fitted_reference
    # Main converts source activity IDs 1..6 to 0..5. Thus source IDs <=3
    # correspond exactly to this bit, with six-way activity remaining the attack.
    active = ((activity+1) <= 3).astype(int)
    fit, dev = checkpoint['indices_attacker_fit'], checkpoint['indices_development']
    pfit, pdev = checkpoint['probability_attacker_fit'], checkpoint['probability_development']
    bins = screen.CONFIG['learned_reference']['probability_bins']
    codes = {
        'oracle_active_bit': (active[fit], active[dev], 2),
        'learned_hard_active_output': ((pfit >= .5).astype(int), (pdev >= .5).astype(int), 2),
        'learned_active_probability_20_bins': (np.minimum((pfit*bins).astype(int), bins-1),
                                             np.minimum((pdev*bins).astype(int), bins-1), bins),
    }
    for name, (cfit, cdev, size) in codes.items():
        expected = screen.conditional_attack(cfit, cdev, activity[fit], activity[dev],
                                            n_codes=size, n_classes=6)
        actual = result['activity_leakage'][name]
        assert actual == expected
        assert len(actual['attribute_fit_support']) == 6
        assert len(actual['attribute_development_support']) == 6
        assert actual['privacy_pass'] is None
    assert result['activity_leakage']['oracle_active_bit']['conditional_accuracy'] > 1/6


def test_conditional_attacker_fits_probabilities_and_prior_only_on_attacker_rows():
    zfit = np.repeat(np.arange(6), np.arange(1, 7))
    cfit = zfit % 2
    zeval = np.tile(np.arange(6), 3)
    ceval = zeval % 2
    score = screen.conditional_attack(cfit, ceval, zfit, zeval, n_codes=2, n_classes=6)
    counts = np.ones((2, 6))
    for code, target in zip(cfit, zfit):
        counts[code, target] += 1
    probabilities = counts/counts.sum(1, keepdims=True)
    prior = (np.bincount(zfit, minlength=6)+1)/(len(zfit)+6)
    expected_loss = -np.log(probabilities[ceval, zeval]).mean()
    expected_prior = -np.log(prior[zeval]).mean()
    assert score['conditional_log_loss_nats'] == pytest.approx(expected_loss)
    assert score['prior_log_loss_nats'] == pytest.approx(expected_prior)
    assert score['log_loss_gain_nats'] == pytest.approx(expected_prior-expected_loss)


@pytest.mark.parametrize('missing_split', ['fit', 'development'])
def test_missing_activity_class_fails_closed(missing_split):
    fit = np.arange(6 if missing_split != 'fit' else 5)
    dev = np.arange(6 if missing_split != 'development' else 5)
    with pytest.raises(ValueError, match='Undefined attribute coverage'):
        screen.conditional_attack(fit % 2, dev % 2, fit, dev, n_codes=2, n_classes=6)


def test_main_reads_only_designated_training_contents(har_fixture, tmp_path, monkeypatch):
    """Exercise the loader boundary with tiny files, never the actual datasets."""
    x, activity, subject = har_fixture
    monkeypatch.chdir(tmp_path)
    har = Path('data/UCI HAR Dataset/train')
    har.mkdir(parents=True)
    np.savetxt(har/'y_train.txt', activity+1, fmt='%d')
    np.savetxt(har/'subject_train.txt', subject, fmt='%d')
    np.savetxt(har/'X_train.txt', x)
    diabetes = Path('data/diabetes_processed')
    diabetes.mkdir()
    keys = ('primary_diagnosis_category', 'readmission_outcome', 'medication_change_outcome',
            'race', 'gender', 'age_bucket')
    np.savez(diabetes/'train.npz', **{key: np.arange(12) % 2 for key in keys})
    forbidden = [Path('data/UCI HAR Dataset/test/X_test.txt'), diabetes/'test.npz']
    for path in forbidden:
        path.parent.mkdir(exist_ok=True)
        path.write_text('OFFICIAL TEST CONTENT MUST REMAIN UNREAD')
    forbidden_absolute = {path.resolve() for path in forbidden}
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert path.resolve() not in forbidden_absolute
        return original_open(path, *args, **kwargs)

    allowed = {str(har/name) for name in ('y_train.txt', 'subject_train.txt', 'X_train.txt')}
    allowed.add(str(diabetes/'train.npz'))
    loaded = []
    original_loadtxt, original_load = np.loadtxt, np.load

    def guarded_loadtxt(path, *args, **kwargs):
        assert str(path) in allowed
        loaded.append(str(path))
        return original_loadtxt(path, *args, **kwargs)

    def guarded_load(path, *args, **kwargs):
        assert str(path) in allowed
        loaded.append(str(path))
        return original_load(path, *args, **kwargs)

    original_hash = screen.sha256

    def guarded_hash(path):
        assert path not in forbidden
        return original_hash(path)

    # Historical oracle-value assertions do not apply to this fabricated fixture;
    # retain the real data-loading and learned-reference execution around them.
    def fixture_oracle(name, arrays, tasks, attrs, rng, **kwargs):
        if name == 'har_official_train':
            np.testing.assert_array_equal(arrays['activity'], activity)
            np.testing.assert_array_equal(arrays['is_active'], activity < 3)
            return {'metrics': [{'task': 'is_active', 'attribute': 'activity',
                                 'conditional_accuracy': .357207615593835,
                                 'log_loss_gain_nats': .687252540516186}]}
        assert name == 'diabetes_processed_train'
        return {'metrics': [{'task': 'primary_diagnosis_category', 'attribute': 'gender',
                             'conditional_accuracy': .5436201385189131}]}

    monkeypatch.setattr(screen.np, 'loadtxt', guarded_loadtxt)
    monkeypatch.setattr(screen.np, 'load', guarded_load)
    monkeypatch.setattr(Path, 'open', guarded_open)
    monkeypatch.setattr(screen, 'sha256', guarded_hash)
    monkeypatch.setattr(screen, 'oracle_screen', fixture_oracle)
    monkeypatch.setattr(screen.subprocess, 'check_output', lambda *a, **k: 'fixture-commit\n')
    monkeypatch.setattr(screen.platform, 'platform', lambda: 'test-fixture')
    monkeypatch.setattr(sys, 'argv', ['screen_pcrl_applications', '--out-dir', 'screen'])
    screen.main()
    assert set(loaded) == allowed
    result = json.loads(Path('screen/metrics.json').read_text())
    assert result['final_test_access'] is False
    assert set(result['used_training_files_sha256']) == allowed
    assert result['learned_output_reference']['privacy_pass'] is None
    # Inventorying names/sizes of existing test files is explicitly allowed.
    inventory = json.loads(Path('screen/local_data_inventory.json').read_text())
    assert len(inventory['datasets']) == 2
