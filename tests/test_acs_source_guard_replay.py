"""Small independent guard algebra and evidence-boundary checks; no ACS fit."""
import copy

import numpy as np
import pytest
import torch

from scripts.replay_acs_source_guard import condition_identity, independent_projection, literal_adam
from experiments.acs_bottleneck_training import tree_digest
from tests.test_acs_source_guard_training import experiment


@pytest.fixture(autouse=True)
def one_numerical_thread():
    from threadpoolctl import threadpool_limits
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        with threadpool_limits(limits=1):
            yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize('name,expected', [
    ('F_G_J', ('F', 'J', .1)), ('P_G_L025', ('P', 'Iplus', .025)),
    ('F_G_L20', ('F', 'Iplus', .2)), ('P_T', ('P', 'T', 0.)),
])
def test_only_authorized_condition_identity(name, expected):
    assert condition_identity(name) == expected
    with pytest.raises(ValueError):
        condition_identity('F_I')


@pytest.mark.parametrize('raw,gradients,expected', [
    ([1., -2., 3.], np.eye(3), [0., -2., 0.]),
    ([-1., -2., -3.], np.eye(3), [-1., -2., -3.]),
    ([1., 2., 3.], np.zeros((3, 3)), [1., 2., 3.]),
    ([1., 2., 3.], [[1., 0., 0.], [1., 0., 0.], [-1., 0., 0.]], [0., 2., 3.]),
    ([1., 2., 3.], [[1., 0., 0.], [0., 1., 0.], [1., 1., 0.]], [0., 0., 3.]),
])
def test_independent_projection_handles_all_inactive_empty_and_dependent_constraints(raw, gradients, expected):
    from experiments.acs_source_guard_training import project_three_halfspaces
    replay = independent_projection(raw, gradients)
    production, metadata = project_three_halfspaces(raw, gradients)
    np.testing.assert_allclose(replay['value'], expected, atol=2e-12, rtol=0.)
    np.testing.assert_allclose(production, replay['value'], atol=2e-12, rtol=0.)
    assert metadata['active_subset'] == replay['active']


def test_rank_is_resolved_on_gradient_matrix_and_nonfinite_is_rejected():
    # Singular value 1e-8 is retained at rcond1e-12 on G, but would be lost on GG'.
    gradients = np.diag([1., 1e-8, 0.])
    replay = independent_projection(np.ones(3), gradients)
    np.testing.assert_allclose(replay['value'], [0., 0., 1.], atol=2e-12, rtol=0.)
    assert replay['rank'] == 2
    with pytest.raises(FloatingPointError):
        independent_projection([float('nan')], np.ones((3, 1)))


def test_literal_adam_preserves_none_zero_clock_and_momentum_exactly():
    parameters = [torch.nn.Parameter(torch.tensor([1., -2.])),
                  torch.nn.Parameter(torch.tensor([3., 4.])),
                  torch.nn.Parameter(torch.tensor([5., -6.]))]
    optimizer = torch.optim.Adam(parameters, lr=.001)
    for i, parameter in enumerate(parameters):
        parameter.grad = torch.tensor([.2, -.3])*(i+1)
    optimizer.step()
    current = [p.detach().clone() for p in parameters]
    state = copy.deepcopy(optimizer.state_dict()); before = tree_digest(state)
    gradients = [None, torch.zeros(2), torch.tensor([-.4, .8])]
    proposed, replayed = literal_adam(current, state, gradients)
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient
    optimizer.step()
    assert tree_digest(replayed) == tree_digest(optimizer.state_dict())
    assert tree_digest(state) == before
    for actual, expected in zip(parameters, proposed):
        torch.testing.assert_close(actual, expected, atol=0., rtol=0.)
    torch.testing.assert_close(proposed[0], current[0], atol=0., rtol=0.)
    assert not torch.equal(proposed[1], current[1]), 'Zero gradient must retain inherited Adam momentum'
    assert [int(replayed['state'][k]['step']) for k in range(3)] == [1, 2, 2]


def test_literal_adam_initial_none_does_not_create_optimizer_state():
    parameters = [torch.nn.Parameter(torch.tensor([1.])), torch.nn.Parameter(torch.tensor([2.]))]
    optimizer = torch.optim.Adam(parameters, lr=.001)
    gradients = [None, torch.zeros(1)]
    proposed, replayed = literal_adam(parameters, optimizer.state_dict(), gradients)
    assert set(replayed['state']) == {1}
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient
    optimizer.step()
    assert tree_digest(replayed) == tree_digest(optimizer.state_dict())
    for expected, actual in zip(proposed, parameters):
        torch.testing.assert_close(expected, actual, atol=0., rtol=0.)


@pytest.mark.parametrize('interface,family,beta', [('F', 'Iplus', .025), ('P', 'J', .1)])
def test_saved_guard_step_replays_gradients_literal_adam_and_float32_bound(interface, family, beta):
    from tests.test_acs_source_guard_training import artificial
    from experiments import acs_coalition_training as historical
    from experiments import acs_source_guard_training as guard
    from scripts.replay_acs_source_guard import verify_guard_evidence
    from scripts import verify_acs_bottleneck_scores as check
    raw, source, attributes, pre, initial, _, _ = artificial()
    model = historical.CoalitionModel(pre, initial)
    observers, _ = historical.make_observers(interface, 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    source = {k: torch.from_numpy(v) for k, v in source.items()}
    attributes = {k: torch.from_numpy(v) for k, v in attributes.items()}
    priors = historical.label_priors(source, attributes)
    x = model.standardize(raw)
    guard.source_step(model, optimizer, x, source)
    _, evidence = guard.guard_step(model, observers, optimizer, x, source, attributes, priors,
                                   interface, family, beta, capture=True)
    evidence['observers_after_scheduled_updates'] = copy.deepcopy(observers.state_dict())
    labels = {k: v.numpy() for k, v in {**source, **attributes}.items()}
    errors_before = len(check.errors)
    replay = verify_guard_evidence(evidence, x.numpy(), labels, priors, interface, family, beta)
    assert replay['literal_adam_exact'] and replay['float32_actual_bounds_passed']
    assert len(check.errors) == errors_before


@pytest.mark.parametrize('system', ['P_guarded', 'F_shared_T'])
def test_complete_artificial_guard_or_shared_T_to_frozen_heads_audits_and_scores(experiment, tmp_path, system):
    import json
    from experiments import acs_coalition_audits as audits
    from experiments import run_acs_coalition as pipeline
    from scripts import verify_acs_bottleneck_scores as check
    from scripts.verify_acs_coalition import literal_wires
    _, _, guarded, shared, fixture = experiment
    raw, _, _, _, _, validation, _ = fixture
    interface = system[0]
    trained = guarded['P_J_0.1'] if interface == 'P' else shared['arms']['F']
    model, observers = trained['model'], trained['observers']
    initial = tree_digest((model.state_dict(), observers.state_dict()))
    # A finite artificial integration fixture: no ACS records, no new policy,
    # production configuration untouched; the audit helper's explicit miniature
    # contract uses one two-epoch trajectory with its first-epoch nested prefix.
    pools = {'downstream_fit': raw.copy(), 'downstream_validation': validation.copy(),
             'attacker_fit': raw[::-1].copy(), 'attacker_validation': validation[::-1].copy()}
    wire, derived = pipeline.arrange(model, pools, interface)
    labels = {pool: {target: np.arange(len(x), dtype=np.int64)%classes
                    for target, classes in audits.CLASSES.items()} for pool, x in pools.items()}
    weights = {pool: np.arange(len(x), dtype=np.float64)+1 for pool, x in pools.items()}
    utility_indices = {target: np.arange(len(raw)) for target in pipeline.TASKS}
    audit_indices = {target: np.arange(len(raw)) for target in audits.TARGETS}
    release_hash = tree_digest((wire, derived))
    utility = pipeline.fit_utilities(wire, labels, utility_indices, 0, tmp_path)
    result = audits.fit_condition_audits(
        {v: {p: x for p, x in a.items() if p.startswith('attacker_')} for v, a in wire.items()},
        {v: {p: x for p, x in a.items() if p.startswith('attacker_')} for v, a in derived.items()},
        {p: y for p, y in labels.items() if p.startswith('attacker_')}, audit_indices,
        observers, 0, tmp_path/'audits', interface=interface,
        inherited_exposure={key: {'role': key, 'artificial_fixture_only': True} for key in observers},
        miniature=True, epochs=2, nested_epochs=1,
        budget={'histgb': {'max_iter': 2, 'max_leaf_nodes': 3}})
    frozen_selection = {'utility': utility['selection'], 'audits': result['selection']}
    selection_path = tmp_path/'selection_before_test.json'
    selection_path.write_text(json.dumps(frozen_selection, sort_keys=True))
    frozen_selection_hash = check.sha(selection_path)
    # Reserved artificial evaluation is constructed only after the release and
    # all validation choices above are frozen.
    test = (validation*.75+.125).astype(np.float32)
    test_labels = {target: np.arange(len(test), dtype=np.int64)%classes
                   for target, classes in audits.CLASSES.items()}
    test_weights = np.arange(len(test), dtype=np.float64)+3
    test_wire, test_derived = model.wires(test, interface), model.wires(test, 'P')
    literal, literal_derived, _ = literal_wires(model.state_dict(), test, interface)
    for view in ('A', 'B', 'AB'):
        np.testing.assert_array_equal(literal[view], test_wire[view])
        np.testing.assert_array_equal(literal_derived[view], test_derived[view])
    pipeline.score_condition(0, 'P_G_J' if interface == 'P' else 'F_T', wire, derived,
                             test_wire, test_derived, utility, result, labels, test_labels,
                             weights, test_weights, tmp_path)
    assert initial == tree_digest((model.state_dict(), observers.state_dict()))
    assert release_hash == tree_digest((wire, derived))
    assert frozen_selection_hash == check.sha(selection_path)
    records = json.loads((tmp_path/'metrics.json').read_text())['raw_metrics']
    assert len(records) == (212 if interface == 'P' else 362)
    errors_before = len(check.errors)
    with np.load(tmp_path/'predictions.npz') as predictions:
        assert len(predictions.files) == 2*len(records)
        for row in records:
            y = test_labels[row['target']]
            probability = predictions[f"{row['role']}/{row['view']}/{row['target']}/{row['audit_budget']}/{row['candidate_id']}/test"]
            classes = 9 if row['target'] == 'RAC1P' else 2
            for key, weight in [('test', None), ('test_person_weighted', test_weights)]:
                check.compare(check.independent_scores(y, probability, classes, weight), row['scores'][key], 'artificial_complete_pipeline')
    assert len(check.errors) == errors_before
    assert result['metadata']['counts']['own_catchup_trajectories'] == 9
    assert all('saved_adversary' not in choice for roles in result['selection'].values()
               for scopes in roles.values() for choice in scopes.values())


def test_independent_global_freeze_accepts_only_complete_or_ordered_final_prefix():
    from scripts.replay_acs_source_guard import CONDITIONS, expected_system_identities, verify_global_gate
    expected = expected_system_identities()
    entries = [{'seed': seed, 'condition': name, 'interface': identity[0], 'family': identity[1],
                'beta': identity[2], 'reused': identity[3], 'original_condition': identity[4]}
               for (seed, name), identity in expected.items()]
    record = {'training_permanently_closed': True, 'new_reserved_fitting_started': False,
              'historical_reserved_outcomes_previously_known': True, 'systems': entries,
              'missing': [], 'all_intended_releases_frozen': True, 'explicit_partial_prefix': False,
              'created_utc': '2026-09-09T03:00:00+00:00'}
    completion = {'training_complete': True, 'reserved_labels_accessed': False,
                  'completed_utc': '2026-09-09T02:59:00+00:00'}
    start = {'all_intended_frozen': True, 'first_reserved_labels_after_global_freeze': True,
             'started_utc': '2026-09-09T03:01:00+00:00'}
    assert verify_global_gate(record, [completion]*24, [start])['frozen_systems'] == 48
    bad = copy.deepcopy(completion); bad['completed_utc'] = '2026-09-09T03:01:00+00:00'
    with pytest.raises(AssertionError):
        verify_global_gate(record, [bad]+[completion]*23, [start])
    prefix = {(0, name) for name in CONDITIONS[:3]}
    partial = copy.deepcopy(record)
    partial['systems'] = [e for e in entries if e['reused'] or (e['seed'],e['condition']) in prefix]
    partial['missing'] = [e for e in entries if not e['reused'] and (e['seed'],e['condition']) not in prefix]
    partial.update(all_intended_releases_frozen=False, explicit_partial_prefix=True)
    partial_start = {**start, 'all_intended_frozen': False}
    assert verify_global_gate(partial, [completion]*3, [partial_start])['frozen_systems'] == 27
    skipped = next(e for e in partial['systems'] if e['condition']=='P_T')
    replacement = next(e for e in partial['missing'] if e['seed']==0 and e['condition']=='F_G_L025')
    partial['systems'].remove(skipped); partial['systems'].append(replacement)
    partial['missing'].remove(replacement); partial['missing'].append(skipped)
    with pytest.raises(AssertionError, match='completed prefix'):
        verify_global_gate(partial, [completion]*3, [partial_start])


def test_source_reconstruction_matches_T_update_without_any_observer():
    from tests.test_acs_source_guard_training import artificial
    from experiments import acs_coalition_training as historical
    from experiments import acs_source_guard_training as guard
    from scripts.reconstruct_acs_source_guard_T import native_components
    from scripts.replay_acs_source_guard import assert_gradient_lists
    raw, source, _, pre, initial, _, _ = artificial()
    model = historical.CoalitionModel(pre, initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    x = model.standardize(raw)
    for _ in range(3):
        before = copy.deepcopy(model.state_dict()); before_optimizer = copy.deepcopy(optimizer.state_dict())
        source_tensors = {k:torch.from_numpy(y) for k,y in source.items()}
        _, evidence = guard.source_step(model, optimizer, x, source_tensors, capture=True)
        state, loss, _ = native_components(before, x.numpy(), source)
        names = [name for name in state if name.startswith('branches.')]
        gradients = torch.autograd.grad(loss, [state[name] for name in names], allow_unused=True)
        values, replayed_optimizer = literal_adam([state[name] for name in names], before_optimizer, gradients)
        assert_gradient_lists(gradients, evidence['source_gradients'])
        assert_gradient_lists(values, [model.state_dict()[name] for name in names])
        assert tree_digest(replayed_optimizer) == tree_digest(optimizer.state_dict())
