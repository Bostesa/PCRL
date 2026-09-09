"""Artificial exact-fork and protection-strength regression checks."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits

from experiments import acs_bottleneck_training as old
from experiments import acs_coalition_training as historical
from experiments import acs_coalition_strength_training as strength


def artificial():
    rng = np.random.default_rng(1347)
    raw = (rng.normal(size=(37, 32))*np.linspace(.2, 2., 32)+np.linspace(-2., 1., 32)).astype(np.float32)
    raw[:, 12] = 3.
    val = rng.normal(size=(19, 32)).astype(np.float32)
    source = {name: np.arange(len(raw), dtype=np.int64)%2 for name in historical.SOURCE_SCHEMA}
    source['income_binary'][::7] = -1
    attrs = {'SEX': np.arange(len(raw), dtype=np.int64)%2, 'RAC1P': np.arange(len(raw), dtype=np.int64)%9}
    attrs['RAC1P'][::11] = -1
    sv = {name: np.arange(len(val), dtype=np.int64)%2 for name in historical.SOURCE_SCHEMA}
    x64 = raw.astype(np.float64)
    pre = {'mean': x64.mean(0), 'scale': np.where(x64.std(0)>1e-12, x64.std(0), 1.)}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1270000)
        initial = old.BottleneckModel(pre['mean'], pre['scale'])
    old.initialize_pca16_mapper(initial)
    return raw, source, attrs, pre, initial.state_dict(), val, sv


@pytest.fixture(scope='module')
def experiment(tmp_path_factory):
    raw, source, attrs, pre, initial, val, sv = artificial()
    root = tmp_path_factory.mktemp('coalition_strength')
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    historical_globals = copy.deepcopy(historical.COEFFICIENTS)
    try:
        with threadpool_limits(limits=1):
            base = historical.train_seed(raw, source, attrs, pre, initial, {}, 0, root/'historical', miniature=True, pca_val=val, source_val=sv)
            trained = {}
            for interface in ('F', 'P'):
                for regime in ('Iplus', 'J'):
                    for beta in strength.BETAS:
                        name = f'{interface}_{regime}_{beta}'
                        trained[name] = strength.train_continuation(raw, source, attrs, pre,
                            root/'historical'/f'{interface}_{regime}'/'fork.pt', base['metadata'], 0, root/name,
                            interface=interface, regime=regime, beta=beta, pca_val=val, source_val=sv, miniature=True)
    finally:
        torch.set_num_threads(previous)
    assert historical.COEFFICIENTS == historical_globals
    return root, base, trained, (raw, source, attrs, pre, initial, val, sv)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def test_beta_point_one_full_continuation_exactly_matches_historical(experiment):
    root, _, trained, _ = experiment
    for interface in ('F', 'P'):
        for regime in ('Iplus', 'J'):
            expected = load(root/'historical'/f'{interface}_{regime}'/'final.pt')
            actual = load(root/f'{interface}_{regime}_0.1'/'final.pt')
            assert old.tree_digest(actual) == old.tree_digest(expected)
            metadata = trained[f'{interface}_{regime}_0.1']['metadata']
            assert metadata['coefficients'] == historical.COEFFICIENTS[regime]


def test_beta_zero_both_regimes_are_historical_I_bitwise(experiment):
    root, _, _, _ = experiment
    for interface in ('F', 'P'):
        expected = load(root/'historical'/f'{interface}_I'/'final.pt')
        a, b = (load(root/f'{interface}_{regime}_0.0'/'final.pt') for regime in ('Iplus', 'J'))
        assert old.tree_digest(a) == old.tree_digest(b) == old.tree_digest(expected)


def test_full_adam_rng_schedule_forks_and_unchanged_historical_evidence(experiment):
    root, base, trained, _ = experiment
    for name, result in trained.items():
        metadata = result['metadata']; expected = load(Path(metadata['historical_fork_path']))
        actual, final = load(root/name/'fork.pt'), load(root/name/'final.pt')
        assert old.tree_digest(actual) == old.tree_digest(expected)
        assert old.tree_digest(expected) == metadata['historical_fork_tree_sha256']
        assert metadata['historical_fork_unchanged'] and metadata['caller_rng_unchanged']
        assert metadata['warmup_refitted'] is False
        assert metadata['counts'] == {'mapper_optimizer_steps': 9, 'adversary_optimizer_steps': 21}
        assert {int(s['step']) for s in final['mapper_optimizer']['state'].values()} == {9}
        assert {int(s['step']) for s in final['adversary_optimizer']['state'].values()} == {21}
        assert len(final['mapper_optimizer']['state']) == 14 and len(final['adversary_optimizer']['state']) == 54
        assert final['schedule_state']['schedules'] == base['metadata']['schedules']
        assert final['schedule_state']['completed_epochs'] == 2 and final['schedule_state']['next_minibatch_index'] == 0
        assert torch.equal(final['torch_rng_state'], expected['torch_rng_state'])
        assert metadata['mapper_exposure_per_row'] == 3 and metadata['observer_exposure_per_row'] == 7
        assert len(metadata['observer_valid_label_exposures']) == 9
        assert metadata['source_label_hashes'] == base['metadata']['source_label_hashes']
        assert metadata['attribute_label_hashes'] == base['metadata']['attribute_label_hashes']
        assert metadata['prior_entropies'] == base['metadata']['prior_entropies']
        assert metadata['reserved_labels_received'] is False and metadata['final_evaluation_received'] is False
        assert not any('decoder' in name for name in final['model_state'])


def test_objective_gradient_is_affine_and_added_difference_scales(experiment):
    root, base, _, data = experiment
    raw, source, attrs, pre, _, _, _ = data
    source = old._labels(source, len(raw), historical.SOURCE_SCHEMA, 'source')
    attrs = old._labels(attrs, len(raw), historical.ATTRIBUTE_SCHEMA, 'attrs')
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        with threadpool_limits(limits=1):
            for interface in ('F', 'P'):
                checkpoint = load(root/'historical'/f'{interface}_I'/'fork.pt')
                model, observers, _, _ = strength.restore_fork(checkpoint, pre, interface, 0)
                losses, _ = historical.components(model, observers, model.standardize(raw), source, attrs, base['metadata']['prior_entropies'], interface)
                parameters = list(model.parameters())
                def gradient(value):
                    grads = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
                    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(parameters,grads)])
                l0 = losses['source']-.1*losses['individual']; g0 = gradient(l0)
                assert torch.linalg.vector_norm(g0)>0
                zero_gradients = []
                for regime in ('Iplus', 'J'):
                    extra = losses['extra_local' if regime == 'Iplus' else 'coalition']
                    signed_unit = gradient(-extra)
                    assert torch.linalg.vector_norm(signed_unit)>0
                    for beta in strength.BETAS:
                        actual = l0-beta*extra if beta else l0
                        torch.testing.assert_close(actual-l0, -beta*extra, atol=8e-8, rtol=3e-6)
                        # Subtracting two float32 full gradients introduces
                        # cancellation; endpoint update/state parity is tested
                        # bitwise separately, without a tolerance.
                        torch.testing.assert_close(gradient(actual)-g0, beta*signed_unit, atol=2e-7, rtol=3e-5)
                        if beta:
                            assert not torch.allclose(gradient(actual), beta*signed_unit)
                        else: zero_gradients.append(gradient(actual))
                assert torch.equal(*zero_gradients)
    finally:
        torch.set_num_threads(previous)


def test_gradient_routing_and_recorded_coefficients(experiment):
    root, base, trained, data = experiment
    for name, result in trained.items():
        m = result['metadata']
        for point in ('fork_diagnostic', 'final_diagnostic'):
            diag = m[point]
            assert diag['state_unchanged'] and diag['rng_unchanged']
            assert diag['coefficients'] == {'source': 1., **strength.coefficients(m['regime'],m['beta'])}
            assert diag['coefficients']['individual'] == -.1
            for owner, other in (('A','B'),('B','A')):
                assert diag['routing'][f'local_{owner}_to_{other}_mapper_l2'] == 0.
                assert diag['routing'][f'extra_{owner}_to_{other}_mapper_l2'] == 0.
                assert diag['groups'][owner+'_mapper']['coalition_raw_l2']>0.
                if m['interface']=='F':
                    assert diag['groups'][owner+'_heads']['combined_protection_l2']==0.
                else:
                    assert diag['groups'][owner+'_heads']['individual_raw_l2']>0.
            if m['beta']==0:
                for group in diag['groups'].values():
                    assert group['extra_local_applied_l2']==group['coalition_applied_l2']==0.
            if m['beta']==.1:
                recorded = base['metadata']['arms'][m['interface']+'_'+m['regime']][point]
                assert diag['groups']==recorded['groups'] and diag['routing']==recorded['routing']
        assert not result['model'].training and not any(p.requires_grad for p in result['model'].parameters())
        p = result['model'].wires(data[0], 'P')
        assert [p[k].shape[1] for k in ('A','B','AB')]==[2,1,3]
        json.dumps(m, allow_nan=False)


def test_new_grid_and_data_identity_guards(experiment,tmp_path):
    root, base, _, data = experiment
    raw, source, attrs, pre, _, val, sv = data
    args=(raw,source,attrs,pre,root/'historical/F_Iplus/fork.pt',base['metadata'],0)
    for beta in (0.,.1):
        with pytest.raises(ValueError,match='endpoints'):
            strength.train_continuation(*args,tmp_path/str(beta),interface='F',regime='Iplus',beta=beta)
    for regime,beta in (('I',.025),('J',.3)):
        with pytest.raises(ValueError,match='grid'):
            strength.coefficients(regime,beta)
    altered={**source,'same_residence':np.zeros(len(raw),np.int64)}
    with pytest.raises(ValueError,match='whitelist'):
        strength.train_continuation(raw,altered,attrs,*args[3:],tmp_path/'reserved',interface='F',regime='Iplus',beta=.025,miniature=True)
    changed=raw.copy();changed[0,0]+=.1
    with pytest.raises(ValueError,match='identity'):
        strength.train_continuation(changed,*args[1:],tmp_path/'changed',interface='F',regime='Iplus',beta=.025,miniature=True)
    bad=copy.deepcopy(base['metadata']);bad['schedules']['continuation']['seed']+=1
    with pytest.raises(ValueError,match='schedule'):
        strength.train_continuation(raw,source,attrs,pre,args[4],bad,0,tmp_path/'schedule',interface='F',regime='Iplus',beta=.025,miniature=True)
