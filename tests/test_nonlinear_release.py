"""Regression checks for utility-constrained nonlinear release selection."""

import copy
import hashlib
import json

import numpy as np
import pytest
import torch
from concept_erasure import LeaceEraser

from experiments import run_nonlinear_release as runner
from experiments.nonlinear_release_training import (
    TRAIN_CONFIG,
    fit_training_erasers,
    protection_penalty,
    train_adversarial,
)


def _evaluation(u=.995, v=.995, individual=.01, combined=.01):
    task = {name: {'r2': value} for name, value in (('p1_U', u), ('p2_V', v))}
    attacks = {key: {'r2': combined if key == 'combined_S' else individual}
               for key in runner.THRESHOLDS}
    return {'task': {'linear': copy.deepcopy(task), 'mlp': task},
            'attack': {kind: copy.deepcopy(attacks) for kind in ('linear', 'mlp')}}


def test_release_selection_requires_each_purpose_utility_floor():
    imbalanced = _evaluation(u=1., v=.98, individual=0., combined=0.)
    eligible = _evaluation(u=.991, v=.991, individual=.04, combined=.08)
    result = runner.select_configuration({'mean_only': imbalanced, 'both': eligible})
    assert result['key'] == 'both'
    assert result['utility_feasible']
    assert not result['all_candidates']['mean_only']['utility_pass']
    assert result['all_candidates']['both']['utility_pass']
    assert result['selection_split'] == 'validation'


def test_release_selection_minimizes_maximum_ratio_not_sum_of_violations():
    # Balanced risk .8 beats concentrated risk .9 even though the latter has
    # a smaller sum. Combined-S uses its own .10 threshold.
    balanced = _evaluation(individual=.04, combined=.08)
    concentrated = _evaluation(individual=0., combined=0.)
    concentrated['attack']['mlp']['p1_V']['r2'] = .045
    selected = runner.select_configuration({'balanced': balanced, 'concentrated': concentrated})
    assert selected['key'] == 'balanced'
    assert selected['all_candidates']['balanced']['worst_ratio'] == pytest.approx(.8)
    assert selected['all_candidates']['concentrated']['worst_ratio'] == pytest.approx(.9)
    assert selected['protection_pass']


def test_release_risk_ties_favor_minimum_purpose_utility_and_missing_floor_is_diagnostic():
    high_mean = _evaluation(u=.999, v=.993)
    high_minimum = _evaluation(u=.994, v=.994)
    assert runner.select_configuration({'higher_mean': high_mean, 'higher_min': high_minimum})['key'] == 'higher_min'
    result = runner.select_configuration({'a': _evaluation(u=.96, v=.99),
                                           'b': _evaluation(u=.98, v=.98)})
    assert result['key'] == 'b'
    assert result['diagnostic_only']
    assert not result['utility_feasible']


@pytest.mark.parametrize('undefined', [None, float('nan')])
def test_undefined_release_attacker_never_passes_even_with_high_utility(undefined):
    unknown = _evaluation()
    # The linear-first/MLP-second ordering must not hide NaN with max().
    unknown['attack']['mlp']['p2_S']['r2'] = undefined
    assessment = runner.assess(unknown)
    assert assessment['utility_pass']
    assert not assessment['defined']
    assert not assessment['protection_pass']
    assert not assessment['feasible']
    selected = runner.select_configuration({'unknown': unknown, 'known': _evaluation()})
    assert selected['key'] == 'known'


def test_release_test_is_fresh_and_fit_streams_remain_identical():
    # A nonpilot seed ensures this test never materializes seeds 0/1/2's
    # sealed final data before their actual experiment selection.
    seed = 97
    fitting, before = runner.make_data(seed)
    complete, after = runner.make_data(seed, include_test=True)
    old_complete, old_manifest = runner.old.make_data(seed, include_test=True)
    assert 'test' not in fitting
    assert after['split_seeds']['test'] == 900004 + 100*seed
    assert old_manifest['split_seeds']['test'] == 500004 + 100*seed
    for split in fitting:
        np.testing.assert_array_equal(fitting[split]['x'], complete[split]['x'])
        np.testing.assert_array_equal(fitting[split]['y'], complete[split]['y'])
        np.testing.assert_array_equal(fitting[split]['ids'], old_complete[split]['ids'])
    assert before['preprocessing_mean'] == after['preprocessing_mean']
    assert before['preprocessing_std'] == after['preprocessing_std']
    assert not np.intersect1d(complete['test']['ids'], old_complete['test']['ids']).size
    assert after['splits']['test']['target_sha256'] != old_manifest['splits']['test']['target_sha256']


def test_prediction_only_release_uses_actual_task_heads_without_target_access():
    class PurposeEncoder(torch.nn.Module):
        def forward(self, features, purpose):
            return features + 10*purpose

    heads = torch.nn.ModuleList([torch.nn.Linear(3, 1), torch.nn.Linear(3, 1)])
    with torch.no_grad():
        heads[0].weight.copy_(torch.tensor([[2., 0., -1.]]))
        heads[0].bias.fill_(3.)
        heads[1].weight.copy_(torch.tensor([[0., -1., .5]]))
        heads[1].bias.fill_(-2.)
    x = np.arange(18, dtype=np.float64).reshape(6, 3)
    model = {'encoder': PurposeEncoder(), 'heads': heads}
    # Deliberately omit targets: prediction-only release must not access them.
    views = runner.prediction_views(model, {'validation': {'x': x}})['validation']
    expected_u = 2*x[:, 0] - x[:, 2] + 3
    expected_v = -(x[:, 1]+10) + .5*(x[:, 2]+10) - 2
    np.testing.assert_allclose(views[0][:, 0], expected_u)
    np.testing.assert_allclose(views[1][:, 0], expected_v)
    np.testing.assert_allclose(views[2], np.column_stack((expected_u, expected_v)))
    assert [view.shape[1] for view in views] == [1, 1, 2]


def test_training_eraser_fit_stops_gradients_but_release_application_remains_differentiable():
    class IdentityPurposeEncoder(torch.nn.Module):
        def forward(self, features, purpose):
            return features

    rng = np.random.default_rng(414)
    holdout = torch.tensor(rng.normal(size=(64, 8)), dtype=torch.float32, requires_grad=True)
    labels = np.column_stack((holdout.detach().numpy()[:, :2],
                              holdout.detach().numpy()[:, 2] + .2*holdout.detach().numpy()[:, 3]**2))
    release = fit_training_erasers(IdentityPurposeEncoder(), holdout, labels)
    assert holdout.grad is None
    assert release.metadata['fit_gradients'] is False
    assert release.metadata['application_differentiable'] is True
    assert not list(release.parameters())
    assert all(not tensor.requires_grad for tensor in release.buffers())
    for purpose, prohibited in enumerate(((1, 2), (0, 2))):
        actual = LeaceEraser.fit(holdout.detach().double(), torch.tensor(labels[:, prohibited]),
                                **TRAIN_CONFIG['leace'])
        torch.testing.assert_close(release(holdout.detach(), purpose),
                                   actual(holdout.detach().double()).float(), atol=1e-6, rtol=1e-6)
    current = torch.randn(16, 8, requires_grad=True)
    loss = release(current, 0).square().mean()
    loss.backward()
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()
    assert current.grad.norm() > 1e-6
    assert holdout.grad is None


def test_protection_penalty_has_the_opposing_encoder_and_adversary_loss_signs():
    mse = torch.tensor([.2, .3, .4, .5, .6], requires_grad=True)
    weights = torch.tensor([.01, .02, .03, .04, .05])
    thresholds = torch.tensor([.05, .05, .05, .05, .10])
    encoder_penalty = protection_penalty(mse, weights, thresholds)
    encoder_gradient = torch.autograd.grad(encoder_penalty, mse, retain_graph=True)[0]
    attacker_gradient = torch.autograd.grad(mse.mean(), mse)[0]
    torch.testing.assert_close(encoder_gradient, -weights / len(weights))
    assert torch.all(attacker_gradient > 0)
    # Raising attack MSE should lower the encoder objective; satisfied privacy
    # constraints can legitimately contribute a negative Lagrangian term.
    assert protection_penalty(torch.ones(5)*1.2, weights, thresholds) < 0


@pytest.fixture(scope='module')
def release_training_fixture(tmp_path_factory):
    from experiments.nonlinear_conflict_training import adapt, pretrain

    torch.set_num_threads(1)
    rng = np.random.default_rng(31)
    x = rng.normal(size=(128, 8)).astype(np.float32)
    y = np.column_stack((x[:, 0] + .1*x[:, 1]**2,
                         x[:, 2] - .2*x[:, 3], x[:, 4] + .1*x[:, 5]**3)).astype(np.float32)
    root = tmp_path_factory.mktemp('release_training')
    prior = root/'prior'
    prior_seed = prior/'seed_97'
    config = {'pretrain': {'epochs': 1, 'eval_every': 1, 'batch_size': 16, 'hidden_dim': 8},
              'adaptation': {'steps': 1, 'batch_size': 16, 'rank': 2, 'alpha': 2.}}
    pre = pretrain(x[:96], y[:96, :2], x[96:], y[96:, :2], seed=97,
                   out_dir=prior_seed/'pretraining', config=config)
    # Small authentic prior artifacts exercise the new read-only loader paths.
    for name, weight in [('C_task_only', 0.), ('D_protection_0.1', .1), ('D_protection_1', 1.)]:
        adapt(pre, x[:96], y[:96], seed=97, strength=weight,
              out_dir=prior_seed/name/'adaptation', config=config)
    tiny = copy.deepcopy(TRAIN_CONFIG)
    tiny.update(eraser_holdout_size=32, steps=2, batch_size=16, adversary_warmup_steps=2,
                adversary_steps_per_encoder=2, eraser_refresh_every=1,
                adversary_hidden=4, rank=2, alpha=2.)
    arms = [train_adversarial(pre, x[:96], y[:96], seed=97, method=method,
                             weight=0. if method == 'task' else .01,
                             out_dir=root/method, config=tiny) for method in ('task', 'fixed', 'dual')]
    return {'pre': pre, 'x': x[:96], 'y': y[:96], 'root': root, 'prior': prior,
            'arms': arms, 'config': tiny}


def test_release_training_matches_actor_and_adversary_initialization_and_update_budgets(release_training_fixture):
    fixture = release_training_fixture
    metadata = [arm['metadata'] for arm in fixture['arms']]
    for field in ('initial_state_hash', 'pretrained_state_hash', 'adversary_initial_state_hash',
                  'schedule_hash', 'adversary_schedule_hash', 'trainable_parameter_count',
                  'adversary_parameter_count'):
        assert len({m[field] for m in metadata}) == 1, field
    initial = [torch.load(fixture['root']/method/'initialization.pt', weights_only=True)
               for method in ('task', 'fixed', 'dual')]
    for checkpoint in initial:
        assert checkpoint['optimizer_steps'] == {'encoder': 0, 'adversary': 0, 'dual': 0}
        assert checkpoint['encoder_optimizer']['state'] == {}
        assert checkpoint['adversary_optimizer']['state'] == {}
    for second in initial[1:]:
        for section in ('encoder', 'heads', 'adversaries'):
            for name, tensor in initial[0][section].items():
                assert torch.equal(tensor, second[section][name])
    for m in metadata:
        assert m['optimizer_steps'] == 2
        assert m['adversary_optimizer_steps'] == 6
        assert m['adversary_network_update_count'] == 18
        assert m['initial_backbone_hash'] == m['final_backbone_hash']
        assert m['gradient_diagnostics']['task_first_layer_B_l2'] > 0
        assert m['gradient_diagnostics']['protection_first_layer_B_l2'] > 0
        assert m['first_layer_parameter_delta_l2'] > 0
    assert [m['dual_update_count'] for m in metadata] == [0, 0, 2]


def test_training_only_eraser_pool_and_adversary_normalization_are_disjoint(release_training_fixture):
    fixture = release_training_fixture
    expected_input_hash = hashlib.sha256(fixture['x'][:32].tobytes()).hexdigest()
    expected_label_hash = hashlib.sha256(fixture['y'][:32].astype(np.float64).tobytes()).hexdigest()
    for arm in fixture['arms']:
        metadata = arm['metadata']
        assert metadata['eraser_fit_indices'] == list(range(32))
        assert metadata['update_indices_first'] == 32
        assert metadata['update_indices_last'] == 95
        np.testing.assert_allclose(metadata['target_mean'], fixture['y'][32:].astype(np.float64).mean(0), atol=1e-7)
        np.testing.assert_allclose(metadata['target_std'], fixture['y'][32:].astype(np.float64).std(0), atol=1e-7)
        refreshes = metadata['eraser_refresh_history']
        assert [row['encoder_optimizer_steps'] for row in refreshes] == [0, 1, 2]
        for row in refreshes:
            assert row['fit_n'] == 32
            assert row['fit_input_sha256'] == expected_input_hash
            assert row['fit_labels_sha256'] == expected_label_hash
            assert row['fit_gradients'] is False


def test_saved_release_loss_and_dual_updates_have_correct_signs(release_training_fixture):
    for arm in release_training_fixture['arms']:
        metadata = arm['metadata']
        for step in metadata['training_history']:
            weights = np.array(step['weights_before'])
            violation = 1 - np.array(step['normalized_adversary_mse']) - np.array(step['thresholds'])
            penalty = np.mean(weights * violation)
            assert step['protection_penalty'] == pytest.approx(penalty, abs=1e-7)
            assert step['encoder_objective'] == pytest.approx(step['task_loss'] + penalty, abs=1e-7)
            if metadata['method'] == 'dual':
                expected = np.clip(weights + metadata['config']['dual_lr_multiplier'] *
                                   metadata['initial_weight'] * violation, 0, metadata['config']['dual_max'])
            else:
                expected = weights
            np.testing.assert_allclose(step['weights_after'], expected, atol=1e-8)


def test_tiny_release_pipeline_seals_selection_before_fresh_test_and_serializes_all_arms(
    monkeypatch, tmp_path, release_training_fixture,
):
    from experiments import nonlinear_release_training as training

    fixture = release_training_fixture
    old_config = copy.deepcopy(runner.old.CONFIG)
    sizes = {'representation_train': 96, 'calibration': 64, 'attacker_fit': 48,
             'validation': 32, 'test': 32}
    old_config['split_sizes'] = sizes
    monkeypatch.setattr(runner.old, 'CONFIG', old_config)
    config = copy.deepcopy(runner.CONFIG)
    config['split_sizes'] = sizes
    config['training'] = copy.deepcopy(fixture['config'])
    config['probes'].update(epochs=0, hidden=4, initialization_offsets=[0])
    config = json.loads(json.dumps(config))
    monkeypatch.setattr(runner, 'CONFIG', config)
    directory = tmp_path/'seed_97'
    original_data = runner.make_data
    original_probe_fit = runner.old.fit_method_probes
    original_train = training.train_adversarial
    generated_test = []

    def guarded_data(seed, include_test=False):
        assert seed == 97  # A fixture seed outside the real pilot.
        if include_test:
            selection_path = directory/'selection_before_test.json'
            assert selection_path.exists()
            selection = json.loads(selection_path.read_text())
            assert selection['test_generated'] is False
            assert selection['fresh_test_rng_seed'] == 900004 + 100*97
            assert set(selection['selection']) == {'D', 'E'}
            assert all(row['selection_split'] == 'validation' for row in selection['selection'].values())
            generated_test.append(True)
        return original_data(seed, include_test=include_test)

    def guarded_probe_fit(*args, **kwargs):
        assert not generated_test
        return original_probe_fit(*args, **kwargs)

    def guarded_train(*args, **kwargs):
        assert not generated_test
        return original_train(*args, **kwargs)

    monkeypatch.setattr(runner, 'make_data', guarded_data)
    monkeypatch.setattr(runner.old, 'fit_method_probes', guarded_probe_fit)
    monkeypatch.setattr(training, 'train_adversarial', guarded_train)
    result = runner.run_seed(97, directory, config, prior=fixture['prior'])
    assert generated_test == [True]
    saved = json.loads((directory/'metrics.json').read_text())
    assert saved['selection'] == result['selection']
    assert len(saved['methods']) == 10
    assert set(saved['diagnostics']) == {'exposed_target'}
    assert all(saved['matching'].values())
    assert saved['methods']['B_prediction_only']['release_dimensions'] == [1, 1, 2]
    assert 'test' in saved['methods']['B_prediction_only']['direct_prediction']
    for record in [*saved['methods'].values(), *saved['diagnostics'].values()]:
        assert 'validation' in record
        assert 'test' in record
        assert 'assessment_test' in record
