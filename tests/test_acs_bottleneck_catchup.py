"""Direct-coordinate fidelity and fresh-optimizer catch-up regressions."""
import copy
import inspect
import json

import numpy as np
import pytest
import torch

from experiments.acs_bottleneck_catchup import CATCHUP_CONFIG, fit_catchup
from experiments.acs_transfer_heads import _network, _state_hash, load_candidate


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def example(n=67, nv=41, classes=2):
    rng = np.random.default_rng(988)
    x = (rng.normal(size=(n, 16))*3+7).astype(np.float32)
    xv = (rng.normal(size=(nv, 16))*2-5).astype(np.float32)
    y = np.arange(n, dtype=np.int64) % classes
    yv = np.arange(nv, dtype=np.int64) % classes
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(671)
        model = _network(16, [64, 32], classes).eval().requires_grad_(False)
    return model, x, y, xv, yv


def test_saved_audit_and_epoch_zero_use_exact_direct_coordinates():
    model, x, y, xv, yv = example()
    source_hash = _state_hash(model)
    rng_before = torch.get_rng_state().clone()
    result = fit_catchup(model, x, y, xv, yv, 2, 77, epochs=0)
    with torch.no_grad():
        direct = torch.softmax(model(torch.from_numpy(xv)), dim=1).double().numpy()
    for kind in ('saved', 'catchup'):
        candidate = result[kind]
        np.testing.assert_array_equal(candidate.predict_proba(xv), direct)
        np.testing.assert_array_equal(candidate.preprocessing.mean, np.zeros(16))
        np.testing.assert_array_equal(candidate.preprocessing.scale, np.ones(16))
        assert candidate.preprocessing.fitted is False
        assert not candidate.model.training
        assert not any(value.requires_grad for value in candidate.model.parameters())
        assert candidate.metadata['audit_kind'] == kind
        assert candidate.metadata['preprocessing_fit_rows'] == 0
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert _state_hash(model) == source_hash
    assert result['saved'].metadata['fit_rows'] == 0
    assert result['catchup'].metadata['optimizer_steps'] == 0
    assert result['metadata']['initial_fidelity_exact'] is True
    assert not any('test' in name for name in inspect.signature(fit_catchup).parameters)


def test_optimizer_is_reset_despite_prior_adversary_training():
    model, x, y, xv, yv = example()
    model.requires_grad_(True)
    prior_optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    for _ in range(3):
        prior_optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(torch.from_numpy(x)), torch.from_numpy(y)).backward()
        prior_optimizer.step()
    model.eval().requires_grad_(False)
    original_hash = _state_hash(model)
    original_grads = [value.grad.clone() for value in model.parameters()]
    inherited = {'optimizer_steps': 3, 'row_exposures': 3*len(y), 'source_split': 'representation_fit'}
    expected = copy.deepcopy(model).train().requires_grad_(True)
    fresh = torch.optim.Adam(expected.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    order = np.random.default_rng(89+700000).permutation(len(y))
    fresh.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(expected(torch.from_numpy(x[order])), torch.from_numpy(y[order]))
    loss.backward()
    fresh.step()
    result = fit_catchup(model, x, y, xv, yv, 2, 89, epochs=1, inherited_exposure=inherited)
    meta = result['catchup'].metadata
    assert meta['optimizer']['restored_state'] is False
    assert meta['optimizer']['initial_state_entries'] == 0
    assert meta['optimizer']['final_state_steps'] == [1]*6
    assert meta['final_state_hash'] == _state_hash(expected)
    assert meta['training_row_exposures'] == len(y)
    assert result['saved'].metadata['training_row_exposures'] == 0
    assert meta['inherited_exposure'] == inherited
    meta['inherited_exposure']['row_exposures'] = 999
    assert inherited['row_exposures'] == 3*len(y)
    assert _state_hash(model) == original_hash
    for value, original in zip(model.parameters(), original_grads):
        assert torch.equal(value.grad, original)
    assert all(int(state['step']) == 3 for state in prior_optimizer.state.values())


def test_actual_budget_and_validation_checkpoint_minimum():
    model, x, y, xv, yv = example(n=513, nv=97)
    result = fit_catchup(model, x, y, xv, yv, 2, 523, epochs=7)
    meta = result['catchup'].metadata
    assert meta['optimizer_steps'] == 21
    assert meta['training_row_exposures'] == 7*513
    assert meta['row_exposure_min'] == meta['row_exposure_max'] == 7
    assert [row['epoch'] for row in meta['validation_curve']] == [0, 5, 7]
    assert [row['optimizer_steps'] for row in meta['validation_curve']] == [0, 15, 21]
    best = min(meta['validation_curve'], key=lambda row: (row['validation_log_loss'], row['epoch']))
    assert meta['selected_epoch'] == best['epoch']
    assert meta['selected_optimizer_steps'] == best['optimizer_steps']
    assert meta['validation_scores']['log_loss'] == best['validation_log_loss']
    assert _state_hash(result['catchup'].model) == meta['selected_state_hash']
    assert result['saved'].metadata['validation_scores']['log_loss'] == meta['validation_curve'][0]['validation_log_loss']
    json.dumps(result['metadata'], allow_nan=False)


def test_validation_labels_change_only_selection_not_training():
    model, x, y, xv, yv = example(n=273)
    a = fit_catchup(model, x, y, xv, yv, 2, 818, epochs=2)
    b = fit_catchup(model, x, y, xv, 1-yv, 2, 818, epochs=2)
    for key in ('initial_state_hash', 'final_state_hash', 'schedule_hash', 'optimizer_steps', 'training_row_exposures'):
        assert a['catchup'].metadata[key] == b['catchup'].metadata[key]
    assert a['catchup'].metadata['validation_hashes']['y'] != b['catchup'].metadata['validation_hashes']['y']
    np.testing.assert_array_equal(a['saved'].predict_proba(xv), b['saved'].predict_proba(xv))


@pytest.mark.parametrize('kind', ['saved', 'catchup'])
def test_historical_save_load_preserves_identity_coordinates_and_selected_weights(kind, tmp_path):
    model, x, y, xv, yv = example(classes=9)
    result = fit_catchup(model, x, y, xv, yv, 9, 5, epochs=1)
    candidate = result[kind]
    candidate.save(tmp_path/kind)
    restored = load_candidate(tmp_path/kind)
    assert restored.metadata == candidate.metadata
    assert restored.metadata['audit_kind'] == kind
    np.testing.assert_array_equal(restored.predict_proba(xv), candidate.predict_proba(xv))
    np.testing.assert_array_equal(restored.preprocessing.mean, np.zeros(16))
    np.testing.assert_array_equal(restored.preprocessing.scale, np.ones(16))
    assert not any(value.requires_grad for value in restored.model.parameters())


def test_missing_race_categories_remain_in_the_schema():
    model, x, y, xv, yv = example(classes=9)
    y = np.where(y == 3, 0, y)
    yv = np.where(yv == 3, 0, yv)
    result = fit_catchup(model, x, y, xv, yv, 9, 56, epochs=1)
    assert result['catchup'].metadata['fit_support'][3] == 0
    assert result['catchup'].metadata['fit_coverage_complete'] is False
    for kind in ('saved', 'catchup'):
        assert result[kind].predict_proba(xv).shape == (len(xv), 9)
        score = result[kind].metadata['validation_scores']
        assert score['coverage_complete'] is False
        assert score['macro_auroc'] is None
        assert score['per_class'][3]['recall'] is None
    assert CATCHUP_CONFIG['epochs'] == 120


def test_invalid_coordinates_or_network_are_rejected_before_fitting():
    model, x, y, xv, yv = example()
    with pytest.raises(ValueError, match='direct-coordinate'):
        fit_catchup(model, x[:, :15], y, xv, yv, 2, 1)
    invalid = x.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match='direct-coordinate'):
        fit_catchup(model, invalid, y, xv, yv, 2, 1)
    with pytest.raises(ValueError, match='architecture'):
        fit_catchup(_network(16, [8, 4], 2), x, y, xv, yv, 2, 1)
