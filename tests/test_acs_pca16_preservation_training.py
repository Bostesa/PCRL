"""Focused artificial-data preservation regressions; never access ACS."""
import copy
import inspect

import numpy as np
import pytest
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from experiments import acs_bottleneck_training as t
from experiments.acs_transfer_models import state_digest


def data(n=37, seed=19):
    rng = np.random.default_rng(seed)
    x = (rng.normal(size=(n, 32)) * np.linspace(.3, 3., 32) + np.linspace(-4., 5., 32)).astype(np.float32)
    x[:, 12] = 7.
    source = {k: (np.arange(n) % 2).astype(np.int64) for k in t.SOURCE_SCHEMA}
    source['income_binary'][::9] = -1
    attrs = {'SEX': np.arange(n) % 2, 'RAC1P': np.arange(n) % 9}
    return x, source, attrs


def statistics(x):
    raw = x.astype(np.float64)
    std = raw.std(0)
    return {'mean': raw.mean(0).tolist(), 'scale': np.where(std > 1e-12, std, 1.).tolist()}


@pytest.fixture(scope='module')
def units(tmp_path_factory):
    x, source, attrs = data()
    xv, yv, _ = data(23, 20)
    out = tmp_path_factory.mktemp('preservation')
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            historical = t.train_pair(x, source, attrs, xv, yv, 2, out / 'historical',
                                      miniature=True, initialization='pca16')
            result = {beta: t.train_preservation_unit(x, source, attrs, xv, yv, 2, out / str(beta),
                       beta=beta, miniature=True, fitting_statistics=statistics(x)) for beta in (0., .1, 1.)}
    finally:
        torch.set_num_threads(old)
    return result, historical, out, (x, source, attrs, xv, yv)


def test_loss_is_raw_coordinate_mse_detaches_teacher_and_fixed_scales():
    released = torch.arange(48, dtype=torch.float32).reshape(3, 16).requires_grad_()
    teacher = (released.detach() * .7 - 5.).requires_grad_()
    scale = torch.linspace(.2, 3., 16, requires_grad=True)
    loss = t.preservation_loss(released, teacher, scale)
    expected = ((released.detach() - teacher.detach()) / scale.detach()).square().mean()
    assert loss.item() == expected.item()
    loss.backward()
    torch.testing.assert_close(released.grad, 2 * (released.detach() - teacher.detach()) / scale.detach().square() / 48)
    assert teacher.grad is None and scale.grad is None
    with pytest.raises(ValueError, match='scales'):
        t.preservation_loss(released, teacher, torch.zeros(16))


def test_zero_beta_exactly_reproduces_historical_entire_update_path(units):
    results, historical, out, _ = units
    zero = results[0.]
    for stage in ('initialization.pt', 'warm_base.pt', 'warm_adversary.pt'):
        assert t.tree_digest(torch.load(out / 'historical' / stage, weights_only=True)) == t.tree_digest(
            torch.load(out / '0.0' / stage, weights_only=True))
    for name, arm in zero['metadata']['arms'].items():
        old_name = 'C_bottleneck' if name.startswith('C_') else 'D_protected'
        old = historical['metadata']['arms'][old_name]
        for field in ('final_model_hash', 'final_adversary_hash', 'final_mapper_optimizer_hash', 'final_adversary_optimizer_hash'):
            assert arm[field] == old[field]
        assert t.tree_digest(torch.load(out / '0.0' / name / 'final.pt', weights_only=True)) == t.tree_digest(
            torch.load(out / 'historical' / old_name / 'final.pt', weights_only=True))
    assert zero['metadata']['schedules'] == historical['metadata']['schedules']


def test_common_initialization_exact_four_way_forks_and_schedule_boundaries(units):
    results, historical, out, _ = units
    expected_names = ['C_warmup_only', 'D_warmup_only', 'C_persistent', 'D_persistent']
    for beta in (.1, 1.):
        result, meta = results[beta], results[beta]['metadata']
        assert list(result['arms']) == expected_names
        assert meta['initialization_hashes'] == historical['metadata']['initialization_hashes']
        assert meta['schedules'] == historical['metadata']['schedules']
        assert meta['adversary_initialization_hash'] == historical['metadata']['adversary_initialization_hash']
        forks = [torch.load(out / str(beta) / name / 'fork.pt', weights_only=True) for name in expected_names]
        assert len({t.tree_digest(fork) for fork in forks}) == 1
        assert forks[0]['mapper_optimizer_state']['state'] and forks[0]['adversary_optimizer_state']['state']
        wb = torch.load(out / str(beta) / 'warm_base.pt', weights_only=True)
        wa = torch.load(out / str(beta) / 'warm_adversary.pt', weights_only=True)
        assert state_digest(wb['model_state']) == state_digest(wa['model_state'])
        assert t.tree_digest(wb['mapper_optimizer_state']) == t.tree_digest(wa['mapper_optimizer_state'])
        for curve in meta['shared_curves']['warm_base']:
            assert curve['preservation_coefficient'] == beta
            assert curve['reconstruction_coefficient'] == .1
        for curve in meta['shared_curves']['warm_adversary']:
            assert curve['preservation_coefficient'] == 0.
        for name, arm in meta['arms'].items():
            active = beta if name.endswith('_persistent') else 0.
            assert arm['continuation_preservation_coefficient'] == active
            assert arm['optimizer_counts_including_common'] == {'mapper_optimizer_steps': 9, 'adversary_optimizer_steps': 21}
            assert arm['continuation_mapper_optimizer_steps'] == 6
            assert arm['continuation_adversary_optimizer_steps'] == 18
            assert arm['schedule_hash'] == meta['schedules']['continuation']['sha256']
            for curve in arm['curve']:
                assert curve['preservation_coefficient'] == active
                assert curve['protection_coefficient'] == (-.1 if name.startswith('D_') else 0.)
                assert curve['fit_objective'] == pytest.approx(curve['fit_base_loss'] + curve['fit_applied_preservation'] + curve['fit_applied_protection'])
        assert len({arm['final_model_hash'] for arm in meta['arms'].values()}) == 4


def test_gradient_isolation_perturbation_and_snapshot_immutability(units):
    results, _, out, (x, _, _, xv, _) = units
    for beta in (.1, 1.):
        result, meta = results[beta], results[beta]['metadata']
        assert meta['teacher']['immutable_verified'] and meta['teacher']['detached']
        np.testing.assert_array_equal(meta['teacher']['coordinate_scale'], statistics(x)['scale'][:16])
        assert meta['teacher']['coordinate_scale'][12] == 1.
        check = meta['initialization']['perturbed_preservation_gradient_check']
        assert check['analytic_gradient_passed'] and check['readout_bias_gradient_l2'] > 0
        assert check['loss'] == pytest.approx(.125 ** 2, abs=1e-7)
        diagnostics = list(meta['stage_gradient_diagnostics'].values())
        for arm in meta['arms'].values():
            diagnostics += [arm['fixed_batch_gradient_diagnostics_at_shared_fork'], arm['fixed_batch_gradient_diagnostics_at_final']]
        for diag in diagnostics:
            assert diag['training_state_unchanged'] and diag['torch_rng_unchanged']
            assert diag['optimizer_steps'] == 0 and diag['preservation_nonmapper_gradients_all_absent']
            assert diag['applied_preservation_mapper_l2'] == pytest.approx(diag['preservation_coefficient'] * diag['preservation_mapper_l2'], rel=2e-6, abs=1e-12)
        assert meta['stage_gradient_diagnostics']['initialization']['preservation_loss'] < 1e-12
        assert meta['stage_gradient_diagnostics']['after_base_warmup']['preservation_mapper_l2'] > 0
        for name, model in result['snapshots'].items():
            assert not model.training and not any(p.requires_grad for p in model.parameters())
            before = state_digest(model.state_dict())
            for raw in (x, xv):
                model.release(raw)
            assert state_digest(model.state_dict()) == before == meta['snapshots'][name]['model_hash']
            checkpoint = torch.load(out / str(beta) / meta['snapshots'][name]['checkpoint'], weights_only=True)
            assert state_digest(checkpoint['model_state']) == before


def test_preservation_does_not_directly_change_head_or_decoder_gradients(units):
    results, _, _, (x, source, attrs, _, _) = units
    model = copy.deepcopy(results[.1]['snapshots']['W']).requires_grad_(True)
    left, right = copy.deepcopy(model), copy.deepcopy(model)
    observers = nn.ModuleDict()
    source = {k: torch.tensor(v) for k, v in source.items()}
    attrs = {k: torch.tensor(v) for k, v in attrs.items()}
    priors = results[.1]['metadata']['prior_entropies']
    t.mapper_update(left, observers, t._adam(left.parameters()), left.standardize(x), source, attrs, priors, False)
    t.mapper_update(right, observers, t._adam(right.parameters()), right.standardize(x), source, attrs, priors, False,
                    preservation_beta=1., teacher=torch.from_numpy(x[:, :16]))
    assert state_digest(left.mapper.state_dict()) != state_digest(right.mapper.state_dict())
    assert state_digest(left.heads.state_dict()) == state_digest(right.heads.state_dict())
    assert state_digest(left.decoder.state_dict()) == state_digest(right.decoder.state_dict())


def test_validation_never_changes_updates_and_invalid_access_rejected(units, tmp_path):
    results, _, _, (x, source, attrs, xv, yv) = units
    changed = {k: np.where(v < 0, -1, 1 - v) for k, v in yv.items()}
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            repeated = t.train_preservation_unit(x, source, attrs, xv * 3 + 100, changed, 2, tmp_path / 'changed',
                          beta=.1, miniature=True, fitting_statistics=statistics(x))
    finally:
        torch.set_num_threads(old)
    for name, arm in results[.1]['metadata']['arms'].items():
        for field in ('final_model_hash', 'final_adversary_hash', 'final_mapper_optimizer_hash', 'final_adversary_optimizer_hash'):
            assert arm[field] == repeated['metadata']['arms'][name][field]
    path = tmp_path / 'invalid'
    for kwargs in ({'beta': .2, 'miniature': True}, {'beta': 0.}, {'beta': .1},
                   {'beta': .1, 'miniature': True, 'fit_pool': 'attacker_fit'},
                   {'beta': .1, 'miniature': True, 'fitting_statistics': statistics(x + 1)}):
        with pytest.raises(ValueError):
            t.train_preservation_unit(x, source, attrs, xv, yv, 2, path, **kwargs)
        assert not path.exists()
    with pytest.raises(ValueError, match='whitelist'):
        t.train_preservation_unit(x, {**source, 'same_residence': source['income_binary']}, attrs, xv, yv, 2,
                                  path, beta=.1, miniature=True)
    assert not any('test' in key or 'reserved' in key for key in inspect.signature(t.train_preservation_unit).parameters)
