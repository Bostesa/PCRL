"""Protocol regressions for the fixed nonlinear purpose-conflict experiment."""

import copy
import json

import numpy as np
import pytest
import torch

from experiments.nonlinear_conflict_probes import (
    ProbeConfig,
    fit_affine_probe,
    fit_mlp_probe,
)
from experiments.nonlinear_conflict_training import adapt, clone_state, pretrain
from experiments.run_nonlinear_conflict import (
    THRESHOLDS,
    calibrate_and_apply,
    frozen_views,
    make_data,
    select_configuration,
    selection_row,
)


def _probe_arrays():
    rng = np.random.default_rng(992)
    x_fit = rng.normal(size=(48, 4)) + np.array([2.0, -3.0, 1.0, 0.0])
    x_fit[:, 3] = 5.0
    x_val = rng.normal(size=(24, 4)) + 40.0
    x_test = rng.normal(size=(16, 4)) - 30.0
    coefficients = np.array([[2., -.5], [-1., 1.], [.5, 2.], [0., 0.]])
    intercept = np.array([7., -2.])
    return x_fit, x_fit @ coefficients + intercept, x_val, x_val @ coefficients + intercept, x_test, x_test @ coefficients + intercept


def test_nonlinear_probe_preprocessing_uses_fitting_rows_only():
    x_fit, y_fit, x_val, y_val, x_test, y_test = _probe_arrays()
    config = ProbeConfig(hidden=4, epochs=2, batch_size=16, initialization_offsets=(0,))
    affine = fit_affine_probe(x_fit, y_fit, x_val, y_val, target_names=['U', 'S'], config=config)
    mlp = fit_mlp_probe(x_fit, y_fit, x_val, y_val, target_names=['U', 'S'], seed=99, config=config)
    for probe in (affine, mlp):
        preprocessing = probe.preprocessing
        np.testing.assert_allclose(preprocessing.x_mean, x_fit.mean(axis=0))
        np.testing.assert_allclose(preprocessing.y_mean, y_fit.mean(axis=0))
        np.testing.assert_allclose(preprocessing.x_scale[:3], x_fit.std(axis=0)[:3])
        np.testing.assert_allclose(preprocessing.y_scale, y_fit.std(axis=0))
        assert preprocessing.x_active.tolist() == [True, True, True, False]
        before = copy.deepcopy(preprocessing.__dict__)
        prediction = probe.predict(x_test)
        assert prediction.shape == y_test.shape
        for name, value in before.items():
            np.testing.assert_array_equal(getattr(preprocessing, name), value)
    np.testing.assert_allclose(affine.predict(x_test), y_test, atol=1e-10)


def test_nonlinear_probe_selection_uses_validation_mse_per_target():
    x_fit, y_fit, x_val, y_val, _, _ = _probe_arrays()
    config = ProbeConfig(hidden=4, epochs=2, batch_size=16, initialization_offsets=(0, 1))
    probe = fit_mlp_probe(x_fit, y_fit, x_val, y_val, target_names=['U', 'S'], seed=102, config=config)
    history = probe.metadata['history']
    for j, target in enumerate(('U', 'S')):
        best = min(history, key=lambda row: row['validation_mse'][j])
        selected = probe.metadata['selection'][target]
        assert selected['validation_mse'] == pytest.approx(best['validation_mse'][j])
        assert selected['initialization_seed'] == best['initialization_seed']
        assert selected['epoch'] == best['epoch']
    # Every checkpoint records genuine optimizer progress; initialization has 0.
    initial = [row for row in history if row['epoch'] == 0]
    assert all(row['optimizer_steps'] == 0 for row in initial)


@pytest.fixture(scope='module')
def matched_adaptation(tmp_path_factory):
    """Small real optimization run, with two arms differing only in strength."""
    torch.set_num_threads(1)
    rng = np.random.default_rng(81)
    x = rng.normal(size=(80, 8)).astype(np.float32)
    y = np.column_stack((x[:, 0] + .1*x[:, 1]**2,
                         x[:, 2] - .2*x[:, 3], x[:, 4] + .1*x[:, 5]**3))
    cfg = {
        'pretrain': {'epochs': 3, 'eval_every': 1, 'batch_size': 16, 'hidden_dim': 8},
        'adaptation': {'steps': 3, 'batch_size': 16, 'rank': 2, 'alpha': 2.},
    }
    directory = tmp_path_factory.mktemp('nonlinear_training')
    pretrained = pretrain(x[:64], y[:64, :2], x[64:], y[64:, :2],
                          seed=7, out_dir=directory/'pretrain', config=cfg)
    backbone = clone_state(pretrained['encoder'])
    arms = [adapt(pretrained, x[:64], y[:64], seed=7, strength=strength,
                  out_dir=directory/f'arm_{strength}', config=cfg) for strength in (0., 1.)]
    return pretrained, backbone, arms, directory


def test_shared_pretraining_has_true_initialization_and_validation_selection(matched_adaptation):
    pretrained, _, _, directory = matched_adaptation
    metadata = pretrained['metadata']
    history = metadata['validation_history']
    best = min(history, key=lambda row: row['mean_mse'])
    assert metadata['selected_epoch'] == best['epoch']
    assert metadata['selected_optimizer_steps'] == best['optimizer_steps']
    assert metadata['selected_val_mse'] == best['mean_mse']
    initial = torch.load(directory/'pretrain/initialization.pt', weights_only=True)
    final = torch.load(directory/'pretrain/final.pt', weights_only=True)
    assert initial['epoch'] == initial['optimizer_steps'] == 0
    assert initial['optimizer']['state'] == {}
    assert final['optimizer_steps'] == metadata['optimizer_steps'] == 12
    assert initial['state_hash'] != final['state_hash']


def test_adaptation_arms_match_initial_checkpoint_backbone_and_update_budget(matched_adaptation):
    _, backbone, arms, directory = matched_adaptation
    first, second = [arm['metadata'] for arm in arms]
    assert first['initial_state_hash'] == second['initial_state_hash']
    assert first['pretrained_state_hash'] == second['pretrained_state_hash']
    assert first['schedule_hash'] == second['schedule_hash']
    assert first['optimizer_steps'] == second['optimizer_steps'] == 3
    assert first['dual_update_count'] == 0
    assert second['dual_update_count'] == 3
    initial = [torch.load(directory/f'arm_{strength}/initialization.pt', weights_only=True)
               for strength in (0., 1.)]
    for section in ('encoder', 'heads'):
        assert initial[0][section].keys() == initial[1][section].keys()
        for name in initial[0][section]:
            assert torch.equal(initial[0][section][name], initial[1][section][name])
    for arm in arms:
        for name, value in arm['encoder'].backbone.state_dict().items():
            assert torch.equal(value, backbone[name])
        assert all(not p.requires_grad for p in arm['encoder'].backbone.parameters())


def test_adaptation_receives_task_and_protection_gradients_upstream_of_nonlinearity(matched_adaptation):
    _, _, arms, _ = matched_adaptation
    for arm in arms:
        metadata = arm['metadata']
        diagnostics = metadata['gradient_diagnostics']
        assert diagnostics['task_first_layer_B_l2'] > 1e-8
        assert diagnostics['protection_first_layer_B_l2'] > 1e-8
        assert metadata['first_layer_parameter_delta_l2'] > 1e-8
        assert not metadata['erasure_during_training']
        assert metadata['constraint_thresholds']['combined_S'] == .10
        assert metadata['constraint_thresholds']['P1_V'] == .05
        assert 'combined_S' in diagnostics['constraint_values']


def test_nonlinear_splits_are_disjoint_and_final_test_cannot_affect_preprocessing():
    fit_data, fit_manifest = make_data(2)
    all_data, all_manifest = make_data(2, include_test=True)
    assert 'test' not in fit_data
    assert 'test' not in fit_manifest['split_seeds']
    assert len(set(all_manifest['split_seeds'].values())) == 5
    assert fit_manifest['preprocessing_mean'] == all_manifest['preprocessing_mean']
    assert fit_manifest['preprocessing_std'] == all_manifest['preprocessing_std']
    for split in fit_data:
        np.testing.assert_array_equal(fit_data[split]['x'], all_data[split]['x'])
        np.testing.assert_array_equal(fit_data[split]['y'], all_data[split]['y'])
    ids = [data['ids'] for data in all_data.values()]
    assert len(np.unique(np.concatenate(ids))) == sum(len(v) for v in ids)
    np.testing.assert_allclose(fit_data['representation_train']['x'].mean(0), 0, atol=1e-14)
    np.testing.assert_allclose(fit_data['representation_train']['x'].std(0), 1, atol=1e-14)
    assert not np.allclose(all_data['test']['x'].mean(0), 0, atol=1e-6)


def _validation_row(utility, *, combined=.01, individual=.01, strength=0.):
    return {'utility': utility, 'strength': strength,
            'leakage': {key: combined if key == 'combined_S' else individual for key in THRESHOLDS}}


def test_nonlinear_selection_uses_constraint_specific_thresholds_and_validation_utility():
    rows = {
        'best_feasible': _validation_row(.9, combined=.08, strength=1.),
        'lower_utility': _validation_row(.8, strength=0.),
        'infeasible': _validation_row(.99, individual=.051, strength=.1),
    }
    selected = select_configuration(rows)
    assert selected['key'] == 'best_feasible'
    assert selected['feasible']
    assert selected['selection_split'] == 'validation'
    assert all(value == 0 for value in selected['violations'].values())
    # Normalized excess matters when no method is feasible: 1.5x the joint
    # threshold is less violation than 2x one individual threshold.
    joint_excess = _validation_row(.4, combined=.15)
    individual_excess = _validation_row(.9)
    individual_excess['leakage']['p1_V'] = .10
    selected = select_configuration({'joint': joint_excess, 'individual': individual_excess})
    assert selected['key'] == 'joint'
    assert not selected['feasible']
    assert selected['total_normalized_violation'] == pytest.approx(.5)


def test_undefined_attacker_cannot_silently_make_a_nonlinear_candidate_feasible():
    evaluation = {
        'attack': {kind: {key: {'r2': .01} for key in THRESHOLDS} for kind in ('linear', 'mlp')},
        'task': {'mlp': {'p1_U': {'r2': .9}, 'p2_V': {'r2': .9}}},
    }
    # In this order, Python max(finite, NaN) would silently return finite.
    evaluation['attack']['mlp']['p1_S']['r2'] = float('nan')
    unknown = selection_row(evaluation, strength=1.)
    selected = select_configuration({'unknown': unknown, 'valid': _validation_row(.5)})
    assert selected['key'] == 'valid'
    assert not selected['all_candidates']['unknown']['feasible']


def test_encoder_is_frozen_before_views_are_extracted():
    class FreezeCheckingEncoder(torch.nn.Linear):
        def forward(self, x):
            assert not self.training
            assert not torch.is_grad_enabled()
            assert all(not parameter.requires_grad for parameter in self.parameters())
            return super().forward(x)

    encoder = FreezeCheckingEncoder(8, 8)
    original = clone_state(encoder)
    data = {'calibration': {'x': np.ones((8, 8))},
            'representation_train': {'x': np.zeros((8, 8))}}
    views = frozen_views(encoder, data, adapted=False)
    assert set(views) == set(data)
    for name, value in encoder.state_dict().items():
        assert torch.equal(value, original[name])


def test_final_eraser_calibration_does_not_use_other_splits():
    rng = np.random.default_rng(909)
    labels = rng.normal(size=(96, 3))
    calibration = np.column_stack((labels, rng.normal(size=(96, 5))))
    raw = {'calibration': [calibration, calibration.copy()],
           'representation_train': [rng.normal(size=(64, 8)), rng.normal(size=(64, 8))],
           'validation': [rng.normal(size=(32, 8)), rng.normal(size=(32, 8))]}
    erasers, views = calibrate_and_apply(raw, labels, erased=True)
    changed = copy.deepcopy(raw)
    for split in ('representation_train', 'validation'):
        changed[split] = [x * 100 + 10 for x in changed[split]]
    other_erasers, _ = calibrate_and_apply(changed, labels, erased=True)
    for purpose, prohibited in enumerate(((1, 2), (0, 2))):
        np.testing.assert_array_equal(erasers[purpose].P, other_erasers[purpose].P)
        np.testing.assert_array_equal(erasers[purpose].bias, other_erasers[purpose].bias)
        h = views['calibration'][purpose]
        z = labels[:, prohibited]
        cov = (h-h.mean(0)).T @ (z-z.mean(0)) / len(z)
        np.testing.assert_allclose(cov, 0, atol=1e-10)


def test_tiny_orchestration_serializes_selection_before_generating_test(monkeypatch, tmp_path):
    """Exercise the real five-arm pipeline only on a disposable tiny fixture."""
    import experiments.run_nonlinear_conflict as runner
    from experiments.nonlinear_conflict_training import TRAIN_CONFIG
    from experiments.nonlinear_conflict_probes import PROBE_CONFIG

    config = copy.deepcopy(runner.CONFIG)
    config['split_sizes'] = {'representation_train': 64, 'calibration': 64,
                             'attacker_fit': 48, 'validation': 32, 'test': 32}
    monkeypatch.setattr(runner, 'CONFIG', config)
    config['training'] = copy.deepcopy(TRAIN_CONFIG)
    config['training']['pretrain'].update(epochs=2, eval_every=1, batch_size=16, hidden_dim=8)
    config['training']['adaptation'].update(steps=2, batch_size=16, rank=2, alpha=2.)
    config['probes'] = copy.deepcopy(PROBE_CONFIG)
    config['probes'].update(epochs=1, validation_interval=1, hidden=4, batch_size=16,
                            initialization_offsets=[0])
    config = json.loads(json.dumps(config))
    directory = tmp_path/'seed_0'
    test_was_generated = []
    original_generate = runner.generate_split
    original_fit = runner.fit_method_probes

    def guarded_generate(seed, split):
        if split == 'test':
            selection_path = directory/'selection_before_test.json'
            assert selection_path.exists()
            selection = json.loads(selection_path.read_text())
            assert selection['test_generated'] is False
            assert selection['grid']['selection_split'] == 'validation'
            test_was_generated.append(True)
        return original_generate(seed, split)

    def guarded_fit(*args, **kwargs):
        assert not test_was_generated, 'No probe may be fitted after final test is generated'
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(runner, 'generate_split', guarded_generate)
    monkeypatch.setattr(runner, 'fit_method_probes', guarded_fit)
    result = runner.run_seed(0, directory, config)
    assert test_was_generated == [True]
    saved = json.loads((directory/'metrics.json').read_text())
    assert saved['selection']['key'] == result['selection']['key']
    assert len(saved['methods']) == 5
    assert set(saved['pretraining_test']) == {'before', 'after'}
    for method in saved['methods'].values():
        assert 'test' in method and 'validation' in method
        assert 'calibration_covariance' in method
        if method['adaptation'] is not None:
            assert method['adaptation']['optimizer_steps'] == 2
    runner.summarize(tmp_path)
    table = (tmp_path/'TABLE.md').read_text()
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert 'Every forbidden target' in table
    assert summary['seeds'] == [0]
    assert len(summary['rows']) == 5
