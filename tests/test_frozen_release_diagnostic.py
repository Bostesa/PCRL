"""Alignment and immutability regressions for frozen-release attack diagnostics."""

import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from experiments.frozen_release_adversaries import (
    SavedAdversaryPredictor,
    TARGET_COLUMNS,
    TARGET_NAMES,
    fit_continuations,
)
from experiments.nonlinear_conflict_training import clone_state, state_digest
from experiments.nonlinear_release_training import ReleaseAdversaries
from experiments import run_frozen_release_diagnostic as runner


def _saved_attack_fixture():
    """Distinct feature/target scales make ordering mistakes visible."""
    adversaries = ReleaseAdversaries(repr_dim=3, hidden=4)
    with torch.no_grad():
        for parameter in adversaries.parameters():
            parameter.zero_()
        for p, network in enumerate(adversaries.networks):
            network[0].weight[0, 0 if p < 2 else 3] = 1.
            network[2].weight[0, 0] = 1.
        adversaries.networks[0][4].weight[:, 0] = torch.tensor([1., 2.])
        adversaries.networks[0][4].bias.copy_(torch.tensor([0., 1.]))
        adversaries.networks[1][4].weight[:, 0] = torch.tensor([3., 4.])
        adversaries.networks[1][4].bias.copy_(torch.tensor([2., 3.]))
        adversaries.networks[2][4].weight[0, 0] = 5.
        adversaries.networks[2][4].bias[0] = 4.
    release = {
        'projections': torch.eye(3).repeat(2, 1, 1), 'biases': torch.zeros(2, 3),
        'feature_mean': torch.tensor([[10., 0., 0.], [20., 0., 0.]]),
        'feature_std': torch.tensor([[2., 1., 1.], [3., 1., 1.]]),
    }
    target_mean = np.array([100., 200., 300.], dtype=np.float32)
    target_std = np.array([2., 3., 5.], dtype=np.float32)
    n1, n2 = np.array([1., 2., 3., 4.]), np.array([2., 4., 6., 8.])
    p1 = np.column_stack((10 + 2*n1, np.zeros((len(n1), 2))))
    p2 = np.column_stack((20 + 3*n2, np.zeros((len(n2), 2))))
    views = [p1, p2, np.concatenate((p1, p2), axis=1)]
    expected = np.column_stack((200 + 3*n1, 300 + 5*(2*n1 + 1),
                                100 + 2*(3*n2 + 2), 300 + 5*(4*n2 + 3),
                                300 + 5*(5*n2 + 4)))
    return adversaries, release, target_mean, target_std, views, expected


def test_saved_adversaries_preserve_preprocessing_and_all_five_target_columns():
    adversaries, release, mean, std, views, expected = _saved_attack_fixture()
    predictor = SavedAdversaryPredictor(adversaries, release, mean, std)
    assert tuple(TARGET_COLUMNS) == (1, 2, 0, 2, 2)
    assert tuple(TARGET_NAMES) == ('p1_V', 'p1_S', 'p2_U', 'p2_S', 'combined_S')
    np.testing.assert_allclose(predictor.predict(views), expected, atol=1e-5)
    labels = np.arange(12).reshape(4, 3) * 2. + 50
    scores = predictor.score(views, labels)
    assert set(scores) == set(TARGET_NAMES)
    for j, name in enumerate(TARGET_NAMES):
        target = labels[:, TARGET_COLUMNS[j]]
        assert scores[name]['mse'] == pytest.approx(np.mean((expected[:, j] - target)**2))
        assert scores[name]['r2'] == pytest.approx(1 - np.mean((expected[:, j] - target)**2) / target.var())


def test_saved_predictor_clones_checkpoint_parameters_and_normalizer_buffers():
    adversaries, release, mean, std, views, expected = _saved_attack_fixture()
    predictor = SavedAdversaryPredictor(adversaries, release, mean, std)
    # Mutating caller-owned tensors after construction must not alter replay.
    with torch.no_grad():
        for parameter in adversaries.parameters():
            parameter.fill_(20.)
        release['feature_mean'].fill_(500.)
        release['feature_std'].fill_(100.)
    mean[:] = -700.
    std[:] = 40.
    np.testing.assert_allclose(predictor.predict(views), expected, atol=1e-5)


def test_zero_epoch_continuation_preserves_saved_weights_and_coordinates_despite_shifted_fitting_rows(tmp_path):
    adversaries, release, mean, std, views, expected = _saved_attack_fixture()
    shifted = [views[0] + 1000, views[1] - 1000]
    shifted += [np.concatenate(shifted, axis=1)]
    y_fit = np.arange(12).reshape(4, 3) + 1000.
    y_val = np.arange(12).reshape(4, 3) - 1000.
    source_hash = state_digest(adversaries.state_dict())
    release_hash = state_digest(release)
    result = fit_continuations(adversaries, release, mean, std, shifted, y_fit, views, y_val,
                               seed=97, out_dir=tmp_path/'zero_epoch', epochs=0, batch_size=2, kinds=('B',))
    np.testing.assert_allclose(result['B'].predict(views), expected, atol=1e-5)
    np.testing.assert_array_equal(result['B'].preprocessing.feature_mean, release['feature_mean'])
    np.testing.assert_array_equal(result['B'].preprocessing.feature_std, release['feature_std'])
    np.testing.assert_array_equal(result['B'].target_mean, mean)
    np.testing.assert_array_equal(result['B'].target_std, std)
    assert result['B'].metadata['initial_state_hashes'] == [source_hash, source_hash]
    assert result['B'].metadata['optimizer_steps_total'] == 0
    assert state_digest(adversaries.state_dict()) == source_hash
    assert state_digest(release) == release_hash


def test_continuation_and_fresh_attackers_match_schedules_and_select_only_validation(tmp_path):
    adversaries, release, mean, std, views, _ = _saved_attack_fixture()
    rng = np.random.default_rng(809)
    y_fit = mean + rng.normal(size=(4, 3))*std
    y_val = mean + rng.normal(size=(4, 3))*std
    result = fit_continuations(adversaries, release, mean, std, views, y_fit, views, y_val,
                               seed=98, out_dir=tmp_path/'fits', epochs=2, batch_size=2, validation_interval=1)
    b, c = result['B'].metadata, result['C'].metadata
    assert b['schedule_hashes'] == c['schedule_hashes']
    assert b['optimizer_steps_total'] == c['optimizer_steps_total'] == 8
    assert b['preprocessing_hash'] == c['preprocessing_hash']
    assert b['architecture'] == c['architecture']
    initial_b = torch.load(tmp_path/'fits/B/initialization_restart_0.pt', weights_only=True)
    initial_c = torch.load(tmp_path/'fits/C/initialization_restart_0.pt', weights_only=True)
    assert {key: value.shape for key, value in initial_b['adversaries'].items()} == {
        key: value.shape for key, value in initial_c['adversaries'].items()}
    assert initial_b['initial_state_hash'] != initial_c['initial_state_hash']
    for kind in ('B', 'C'):
        metadata = result[kind].metadata
        assert len(metadata['candidates']) == 2
        for target in TARGET_NAMES:
            best = min((row['validation_mse'][target], candidate['restart_index'], row['epoch'])
                       for candidate in metadata['candidates'] for row in candidate['validation_curve'])
            selected = metadata['selection'][target]
            assert (selected['validation_mse'], selected['restart_index'], selected['epoch']) == best
        for candidate in metadata['candidates']:
            assert candidate['optimizer_steps'] == 4
            assert candidate['validation_curve'][0]['optimizer_steps'] == 0
        # Evaluation on a new split cannot alter the selected weights/scalers.
        before = [state_digest(state) for state in result[kind].selected_states]
        result[kind].score(views, y_val + 300)
        assert [state_digest(state) for state in result[kind].selected_states] == before
    for row in result['metadata']['schedules']:
        with np.load(tmp_path/'fits'/row['artifact']) as schedule:
            assert schedule['indices'].min() >= 0
            assert schedule['indices'].max() < len(y_fit)
            assert len(schedule['indices']) == 2*len(y_fit)


def test_frozen_parameters_buffers_maps_and_representative_outputs_survive_attacker_fit(tmp_path):
    class PurposeEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)
            self.bn = torch.nn.BatchNorm1d(3)
            self.dropout = torch.nn.Dropout(.8)

        def forward(self, x, purpose):
            return self.dropout(self.bn(self.linear(x)))

    adversaries, release, mean, std, _, _ = _saved_attack_fixture()
    model = {'encoder': PurposeEncoder(), 'heads': torch.nn.ModuleList([torch.nn.Linear(3, 1) for _ in range(2)]),
             'adversaries': adversaries,
             'checkpoint': {'training_release': copy.deepcopy(release),
                            'target_mean': torch.from_numpy(mean.copy()), 'target_std': torch.from_numpy(std.copy())},
             'erasers': {**{f'p{p}_matrix': np.eye(3) for p in (1, 2)},
                         **{f'p{p}_center': np.zeros(3) for p in (1, 2)}},
             'training_maps': {'375': copy.deepcopy(release), '400': copy.deepcopy(release)}}
    for key in ('encoder', 'heads', 'adversaries'):
        runner.freeze(model[key])
        assert not model[key].training
        assert all(not parameter.requires_grad for parameter in model[key].parameters())
    rng = np.random.default_rng(333)
    data = {split: {'x': rng.normal(size=(12, 3)), 'y': rng.normal(size=(12, 3)),
                    'ids': np.arange(12)+100*i} for i, split in enumerate(('attacker_fit', 'validation'))}
    before = runner.integrity(model)
    views, _, _ = runner.cache_releases(model, data, tmp_path/'cache_before')
    fit_continuations(model['adversaries'], model['checkpoint']['training_release'], mean, std,
                      views['attacker_fit'], data['attacker_fit']['y'], views['validation'], data['validation']['y'],
                      seed=99, out_dir=tmp_path/'fits', epochs=1, batch_size=6)
    assert runner.integrity(model) == before
    repeated, _, _ = runner.cache_releases(model, data, tmp_path/'cache_after')
    assert runner.integrity(model) == before
    for split in data:
        for first, second in zip(views[split], repeated[split]):
            np.testing.assert_array_equal(first, second)
    # The integrity record must include buffers, not merely parameters.
    model['encoder'].bn.running_mean.add_(1)
    assert runner.integrity(model) != before


def test_diagnostic_test_is_blocked_until_selection_and_uses_a_distinct_stream(monkeypatch, tmp_path):
    config = copy.deepcopy(runner.CONFIG)
    config['test_n'] = 8
    monkeypatch.setattr(runner, 'CONFIG', config)
    manifest = {'preprocessing_mean': [0.]*8, 'preprocessing_std': [1.]*8}
    selection = tmp_path/'selection_before_test.json'
    with pytest.raises(RuntimeError, match='selection'):
        runner.diagnostic_test(97, manifest, selection)
    selection.write_text(json.dumps({'test_generated': False}))
    test = runner.diagnostic_test(97, manifest, selection)
    assert test['x'].shape == (8, 8)
    assert test['y'].shape == (8, 3)
    expected_seed = 1000004 + 100*97
    np.testing.assert_array_equal(test['ids'], np.arange(8)+expected_seed*100000)
    assert expected_seed not in (500004+100*97, 900004+100*97)
