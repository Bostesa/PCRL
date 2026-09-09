"""Miniature numerical, literal-Adam and shared-source-trajectory checks."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from experiments import acs_bottleneck_training as old
from experiments import acs_coalition_training as historical
from experiments import acs_coalition_strength_training as strength
from experiments import acs_source_guard_training as guard


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        with threadpool_limits(limits=1): yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize('r,g,expected', [
    ([1., 2.], [[1., 0.], [0., 1.], [0., 0.]], [0., 0.]),
    ([-1., 2.], [[1., 0.], [0., 1.], [0., 0.]], [-1., 0.]),
    ([2., 3.], [[1., 0.], [-1., 0.], [0., 0.]], [0., 3.]),
    ([2., 3.], [[1., 0.], [2., 0.], [0., 0.]], [0., 3.]),
    ([2., 3.], [[0., 0.], [0., 0.], [0., 0.]], [2., 3.]),
    ([0., 0.], [[1., 0.], [-1., 0.], [0., 1.]], [0., 0.]),
])
def test_analytical_halfspaces_zero_dependent_opposing(r, g, expected):
    v, d = guard.project_three_halfspaces(r, g)
    np.testing.assert_allclose(v, expected, atol=2e-14, rtol=0)
    assert d['svd_rcond'] == 1e-12
    assert np.all(np.asarray(d['ideal_dots']) <= d['primal_tolerances'])
    assert d['stationarity_l2'] <= d['stationarity_tolerance']
    assert np.all(np.abs(d['complementarity_abs']) <= d['complementarity_tolerances'])
    assert d['active_subset_order'] == [list(a) for a in sorted(guard.ACTIVE_SUBSETS)]


def test_projection_against_independent_scipy_qp_and_gradient_svd_rank():
    rng = np.random.default_rng(891)
    for _ in range(12):
        g, r = rng.normal(size=(3, 7)), rng.normal(size=7)
        v, d = guard.project_three_halfspaces(r, g)
        result = minimize(lambda z: .5*np.sum((z-r)**2), np.zeros_like(r), jac=lambda z: z-r,
            constraints=[{'type': 'ineq', 'fun': lambda z: -(g@z), 'jac': lambda z: -g}],
            method='SLSQP', options={'ftol': 1e-13, 'maxiter': 100})
        assert result.success
        np.testing.assert_allclose(v, result.x, atol=2e-7, rtol=2e-7)
        assert abs(.5*d['distance_squared']-result.fun) < 1e-10
    # The active gradient matrix retains a direction that a Gram rcond=1e-12
    # would drop.  Its inequalities and KKT equations still have to pass.
    g = np.array([[1., 0., 0.], [0., 1e-8, 0.], [0., 0., 0.]])
    v, d = guard.project_three_halfspaces(np.array([1., 1., 3.]), g)
    np.testing.assert_allclose(v, [0., 0., 3.], atol=1e-12)
    assert d['active_rank'] == 2
    with pytest.raises(FloatingPointError): guard.project_three_halfspaces([np.nan, 1.], np.zeros((3, 2)))


def test_constraint_rescaling_near_dependency_and_deterministic_ties():
    g = np.array([[1., 0., 1.], [0., 1., 1.], [1., 1., 2.+1e-13]])
    r = np.array([2., -1., 1.])
    first, meta = guard.project_three_halfspaces(r, g)
    again, duplicate = guard.project_three_halfspaces(r, g)
    np.testing.assert_array_equal(first, again)
    assert meta == duplicate
    scaled, _ = guard.project_three_halfspaces(r, g*np.array([.1, 10., 2.])[:, None])
    np.testing.assert_allclose(first, scaled, atol=1e-12, rtol=0)


def test_literal_adam_proposals_preserve_none_zero_and_inherited_moments():
    parameters = [torch.nn.Parameter(torch.tensor([1., 2.])), torch.nn.Parameter(torch.tensor([3.]))]
    optimizer = old._adam(parameters)
    for p in parameters: p.grad = torch.ones_like(p)
    optimizer.step()
    initial = copy.deepcopy(optimizer.state_dict()); initial_parameters = [p.detach().clone() for p in parameters]
    gradients = [torch.zeros_like(parameters[0]), None]
    proposed, state = guard.disposable_adam(parameters, optimizer, gradients)
    assert old.tree_digest(initial) == old.tree_digest(optimizer.state_dict())
    assert all(torch.equal(p, value) for p, value in zip(parameters, initial_parameters))
    for p, grad in zip(parameters, gradients): p.grad = grad
    optimizer.step()
    assert old.tree_digest(state) == old.tree_digest(optimizer.state_dict())
    assert all(torch.equal(p, value) for p, value in zip(parameters, proposed))
    assert not torch.equal(proposed[0], initial_parameters[0])  # Inherited momentum still acts.
    assert torch.equal(proposed[1], initial_parameters[1])
    assert [int(s['step']) for s in state['state'].values()] == [2, 1]


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
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    root = tmp_path_factory.mktemp('source_guard')
    raw, source, attrs, pre, initial, val, sv = artificial()
    caller = torch.get_rng_state().clone()
    old_globals = copy.deepcopy(historical.COEFFICIENTS)
    try:
        with threadpool_limits(limits=1):
            base = historical.train_seed(raw, source, attrs, pre, initial, {}, 0, root/'historical', miniature=True, pca_val=val, source_val=sv)
            guarded = {}
            for interface in ('F', 'P'):
                for regime, beta in guard.GUARDED:
                    name = f'{interface}_{regime}_{beta}'
                    guarded[name] = guard.train_guarded_continuation(raw, source, attrs, pre,
                        root/'historical'/f'{interface}_I'/'fork.pt', base['metadata'], 0, root/name,
                        interface=interface, regime=regime, beta=beta, pca_val=val, source_val=sv, miniature=True)
            shared = guard.train_source_only_shared(raw, source, attrs, pre,
                {interface: root/'historical'/f'{interface}_I'/'fork.pt' for interface in ('F', 'P')},
                base['metadata'], 0, root/'shared', pca_val=val, source_val=sv, miniature=True)
    finally:
        torch.set_num_threads(previous)
    assert torch.equal(caller, torch.get_rng_state()) and historical.COEFFICIENTS == old_globals
    return root, base, guarded, shared, (raw, source, attrs, pre, initial, val, sv)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def test_guarded_saved_proposals_support_cast_and_retained_state(experiment):
    root, base, guarded, _, data = experiment
    raw, source, attrs, pre, _, _, _ = data
    for name, result in guarded.items():
        meta = result['metadata']; directory = root/name
        assert meta['reserved_labels_received'] is False and meta['final_evaluation_received'] is False
        assert meta['diagnostic_epochs'] == [1, 2] and len(meta['epoch_update_summaries']) == 2
        assert meta['counts'] == {'mapper_optimizer_steps': 9, 'adversary_optimizer_steps': 21}
        assert old.tree_digest(load(directory/'fork.pt')) == old.tree_digest(load(meta['historical_fork_path']))
        for point in meta['detailed_steps']:
            record = load(point['path']); scalar = record['scalar']
            assert old.tree_digest(record['post_optimizer']) == old.tree_digest(record['full_proposal_optimizer'])
            mask = record['support_mask'].numpy()
            proposed_source = guard._flat(record['source_proposal'])
            proposed_full = guard._flat(record['full_proposal'])
            assert np.array_equal(proposed_source[~mask], proposed_full[~mask])
            accepted = guard._flat(record['accepted_parameters'])
            v, r, g = (record[k].numpy() for k in ('ideal_increment', 'raw_increment', 'task_gradient_matrix'))
            realized = (accepted-proposed_source)[mask]
            np.testing.assert_array_equal(realized, record['realized_increment'].numpy())
            expected64 = proposed_source.copy(); expected64[mask] += v
            expected32 = guard._flat(guard._unflat(expected64, record['source_proposal']))
            np.testing.assert_array_equal(accepted, expected32)  # No cast correction.
            np.testing.assert_allclose(g@(realized-v), g@record['cast_error'].numpy(), atol=1e-18)
            assert np.all(np.asarray(scalar['actual_minus_ideal_dots']) <= scalar['cast_plus_roundoff_bounds'])
            assert np.all(np.asarray(scalar['actual_dots']) <= scalar['actual_dot_excess_bounds'])
            assert len(record['native_losses']) == 4 and len(record['observers_after_scheduled_updates']) > 0
            assert record['counts_after_update']['mapper_optimizer_steps'] in (4, 7)
            assert meta['source_label_hashes'] == base['metadata']['source_label_hashes']
            if meta['interface'] == 'F': assert not any('.heads.' in n for n in scalar['support_parameter_names'])
            else: assert any('.heads.' in n for n in scalar['support_parameter_names'])


def test_shared_T_exactly_matches_both_separate_interface_trajectories(experiment):
    root, base, _, shared, data = experiment
    raw, source, attrs, pre, _, _, _ = data
    source = old._labels(source, len(raw), historical.SOURCE_SCHEMA, 'source')
    attrs = old._labels(attrs, len(raw), historical.ATTRIBUTE_SCHEMA, 'attrs')
    orders, _ = old._orders(len(raw), 2, base['metadata']['schedules']['continuation']['seed'])
    final_states = []
    for interface in ('F', 'P'):
        checkpoint = load(root/'historical'/f'{interface}_I'/'fork.pt')
        model, observers, optimizer, observer_optimizer = strength.restore_fork(checkpoint, pre, interface, 0)
        x = model.standardize(raw)
        for order in orders:
            for start in range(0, len(raw), 16):
                ix = order[start:start+16]; ys, ya = old._batch(source, ix), old._batch(attrs, ix)
                for _ in range(3):
                    historical.observer_update(model, observers, observer_optimizer, x[ix], {**ys, **ya}, base['metadata']['prior_entropies'], interface)
                historical.mapper_update(model, observers, optimizer, x[ix], ys, ya, base['metadata']['prior_entropies'], interface, None)
        actual = load(root/'shared'/f'{interface}_T'/'final.pt')
        for key, value in (('model_state', model.state_dict()), ('mapper_optimizer', optimizer.state_dict()),
                           ('adversary_state', observers.state_dict()), ('adversary_optimizer', observer_optimizer.state_dict())):
            assert old.tree_digest(actual[key]) == old.tree_digest(value)
        final_states.append(old.tree_digest((actual['model_state'], actual['mapper_optimizer'])))
        assert all(d['scalar']['float64_parameter_round_trip'] is False for d in shared['arms'][interface]['metadata']['detailed_steps'])
    assert final_states[0] == final_states[1] == shared['metadata']['final_forward_state_sha256']
    assert shared['arms']['F']['model'] is shared['arms']['P']['model']
    assert shared['metadata']['unique_new_forward_continuations'] == 1


def test_zero_eligible_source_constraint_and_guard_immutability_before_commit(experiment):
    root, base, _, _, data = experiment
    raw, source, attrs, pre, _, _, _ = data
    checkpoint = load(root/'historical/F_I/fork.pt')
    model, observers, optimizer, _ = strength.restore_fork(checkpoint, pre, 'F', 0)
    labels = old._labels(source, len(raw), historical.SOURCE_SCHEMA, 'source')
    labels['income_binary'][:] = -1
    attrs = old._labels(attrs, len(raw), historical.ATTRIBUTE_SCHEMA, 'attrs')
    observer_before = old.tree_digest(observers.state_dict()); rng = torch.get_rng_state().clone()
    record, evidence = guard.guard_step(model, observers, optimizer, model.standardize(raw), labels, attrs,
        base['metadata']['prior_entropies'], 'F', 'J', .1, capture=True)
    assert record['zero_constraint_tasks'] == ['income_binary']
    assert np.array_equal(evidence['task_gradient_matrix'][0].numpy(), np.zeros(record['support_dimension']))
    assert old.tree_digest(observers.state_dict()) == observer_before and torch.equal(rng, torch.get_rng_state())


def test_all_protection_zero_dispatch_is_literal_source_adam_without_cast(experiment):
    root, base, _, _, data = experiment
    raw, source, _, pre, _, _, _ = data
    model, _, optimizer, _ = strength.restore_fork(load(root/'historical/F_I/fork.pt'), pre, 'F', 0)
    labels = old._labels(source, len(raw), historical.SOURCE_SCHEMA, 'source')
    x = model.standardize(raw); parameters = list(model.parameters())
    loss = historical.source_loss(model.forward_parts(x)[1], labels)[0]
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    source_proposal, source_state = guard.disposable_adam(parameters, optimizer, gradients)
    full_proposal, full_state = guard.disposable_adam(parameters, optimizer, gradients)
    assert old.tree_digest((source_proposal, source_state)) == old.tree_digest((full_proposal, full_state))
    assert np.array_equal(guard._flat(full_proposal)-guard._flat(source_proposal), np.zeros(6355))
    record, _ = guard.source_step(model, optimizer, x, labels)
    assert old.tree_digest((list(model.parameters()), optimizer.state_dict())) == old.tree_digest((source_proposal, full_state))
    assert record['float64_parameter_round_trip'] is False
    assert record['coefficients'] == {'source': 1., 'individual': 0., 'extra_local': 0., 'coalition': 0.}


def test_training_whitelist_and_fixed_guard_grid(experiment,tmp_path):
    root, base, _, _, data = experiment
    raw, source, attrs, pre, _, val, sv = data
    args=(raw,source,attrs,pre,root/'historical/F_I/fork.pt',base['metadata'],0)
    with pytest.raises(ValueError,match='predeclared'):
        guard.train_guarded_continuation(*args,tmp_path/'badbeta',interface='F',regime='J',beta=.2,miniature=True)
    forbidden={**source,'same_residence':np.zeros(len(raw),np.int64)}
    with pytest.raises(ValueError,match='whitelist'):
        guard.train_guarded_continuation(raw,forbidden,*args[2:],tmp_path/'reserved',interface='F',regime='J',beta=.1,miniature=True)
