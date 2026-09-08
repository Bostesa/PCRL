"""Analytic PCA16 initialization regressions on artificial data only."""
import copy

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits

from experiments import acs_bottleneck_training as training
from experiments.acs_transfer_models import state_digest


def data(n, seed):
    rng = np.random.default_rng(seed)
    x = (rng.normal(size=(n, 32)) * np.linspace(.1, 3, 32) + np.linspace(-4, 5, 32)).astype(np.float32)
    # Also exercise the fitting-only floor for an exactly constant coordinate.
    x[:, 12] = 7.
    source = {k: (np.arange(n) % 2).astype(np.int64) for k in training.SOURCE_SCHEMA}
    source['income_binary'][::9] = -1
    attrs = {'SEX': np.arange(n) % 2, 'RAC1P': np.arange(n) % 9}
    return x, source, attrs


def test_analytic_overwrite_preserves_nonmapper_state_and_torch_rng():
    x, _, _ = data(37, 16)
    raw = x.astype(np.float64)
    std = raw.std(0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1270200)
        model = training.BottleneckModel(raw.mean(0), np.where(std > 1e-12, std, 1.))
        original = copy.deepcopy(model.state_dict())
        rng = torch.get_rng_state().clone()
        meta = training.initialize_pca16_mapper(model)
        assert torch.equal(rng, torch.get_rng_state())
    for key, value in original.items():
        if not key.startswith('mapper.'):
            assert torch.equal(value, model.state_dict()[key])
    assert meta['nonmapper_unchanged_by_overwrite'] and meta['torch_rng_unchanged_by_overwrite']
    assert all(p.requires_grad for p in model.parameters())
    first, last = model.mapper[0], model.mapper[2]
    assert torch.equal(first.weight[:32], torch.eye(32))
    assert torch.equal(first.weight[32:], -torch.eye(32))
    assert torch.count_nonzero(first.bias) == 0
    assert torch.count_nonzero(last.weight[:, meta['unused_readout_columns']]) == 0
    assert torch.equal(last.bias, model.input_mean[:16].float())
    np.testing.assert_allclose(model.freeze().release(x), x[:, :16], atol=1e-5, rtol=1e-5)
    # The identity is global, including a constant fitting coordinate that
    # varies in validation. It is not implemented by replaying fitting rows.
    xv, _, _ = data(23, 17)
    xv[:, 12] = np.linspace(-2, 9, len(xv))
    np.testing.assert_allclose(model.release(xv), xv[:, :16], atol=1e-5, rtol=1e-5)


@pytest.fixture(scope='module')
def pairs(tmp_path_factory):
    x, source, attrs = data(37, 18)
    xv, yv, _ = data(23, 19)
    out = tmp_path_factory.mktemp('pca16_init')
    result = {}
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            for mode in ('random', 'pca16'):
                result[mode] = training.train_pair(x, source, attrs, xv, yv, 2, out / mode,
                                                  miniature=True, initialization=mode)
    finally:
        torch.set_num_threads(old)
    return result, out, x, xv


def test_initial_parity_disposable_gradient_and_nonmapper_adversary_identity(pairs):
    pair, out, x, xv = pairs
    random, pca = (pair[k]['metadata'] for k in ('random', 'pca16'))
    init = pca['initialization']
    for pool in ('representation_fit', 'source_validation'):
        assert init['parity'][pool]['passed']
        assert init['parity'][pool]['atol'] == init['parity'][pool]['rtol'] == 1e-5
        assert init['parity'][pool]['rms_error'] <= init['parity'][pool]['max_absolute_error'] < 1e-5
    diagnostic = init['unused_readout_gradient_check']
    assert diagnostic['base_unused_readout_gradient_l2'] > 0
    assert diagnostic['source_unused_readout_gradient_l2'] > 0
    assert diagnostic['actual_model_unchanged'] and diagnostic['disposable_model']
    assert diagnostic['optimizer_steps'] == 0
    originals = {mode: torch.load(out / mode / 'initialization.pt', weights_only=True) for mode in pair}
    for name, value in originals['random']['model_state'].items():
        if not name.startswith('mapper.'):
            assert torch.equal(value, originals['pca16']['model_state'][name])
    assert random['adversary_initialization_hash'] == pca['adversary_initialization_hash']
    assert random['schedules'] == pca['schedules']
    assert random['config'] == pca['config']
    for mode in pair:
        assert originals[mode]['counters'] == {'mapper_optimizer_steps': 0, 'adversary_optimizer_steps': 0, 'epoch': 0}
        assert originals[mode]['mapper_optimizer_state']['state'] == {}
    for raw in (x, xv):
        np.testing.assert_allclose(pair['pca16']['snapshots']['I'].release(raw), raw[:, :16], atol=1e-5, rtol=1e-5)


def test_frozen_i_w_snapshots_and_matched_final_steps(pairs):
    pair, out, x, _ = pairs
    for mode, result in pair.items():
        meta = result['metadata']
        assert set(result['snapshots']) == ({'I', 'W'} if mode == 'pca16' else set())
        for name, model in result['snapshots'].items():
            assert not model.training and not any(p.requires_grad for p in model.parameters())
            before = state_digest(model.state_dict())
            checkpoint = torch.load(out / mode / meta['snapshots'][name]['checkpoint'], weights_only=True)
            assert before == state_digest(checkpoint['model_state']) == meta['snapshots'][name]['model_hash']
            model.release(x)
            assert before == state_digest(model.state_dict())
        warm = torch.load(out / mode / 'warm_adversary.pt', weights_only=True)
        if mode == 'pca16':
            assert state_digest(warm['model_state']) == meta['snapshots']['W']['model_hash']
            assert meta['snapshots']['W']['unchanged_across_adversary_warmup']
        forks = [torch.load(out / mode / arm / 'fork.pt', weights_only=True) for arm in result['arms']]
        assert training.tree_digest(forks[0]) == training.tree_digest(forks[1])
        for arm in result['arms']:
            arm_meta = meta['arms'][arm]
            assert arm_meta['optimizer_counts_including_common'] == {'mapper_optimizer_steps': 9, 'adversary_optimizer_steps': 21}
            assert arm_meta['continuation_mapper_optimizer_steps'] == 6
            assert arm_meta['continuation_adversary_optimizer_steps'] == 18
            assert arm_meta['selected_epoch'] == 2
            final = torch.load(out / mode / arm / 'final.pt', weights_only=True)
            assert all(int(v['step']) == 9 for v in final['mapper_optimizer_state']['state'].values())
            assert all(int(v['step']) == 21 for v in final['adversary_optimizer_state']['state'].values())


def test_unknown_initialization_rejected_before_output_creation(tmp_path):
    x, source, attrs = data(23, 20)
    directory = tmp_path / 'invalid'
    with pytest.raises(ValueError, match='Initialization'):
        training.train_pair(x, source, attrs, x, source, 0, directory, initialization='fit_new_pca')
    assert not directory.exists()
