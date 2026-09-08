import copy
import json

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits

from experiments import acs_coalition_audits as audit
from experiments.acs_transfer_heads import FittedCandidate, InputStandardizer, _state_hash, metrics


def observer(width, classes):
    return torch.nn.Sequential(torch.nn.Linear(width, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, classes)).eval().requires_grad_(False)


def fixture(interface):
    rng = np.random.default_rng(71)
    dims = (16, 16) if interface in ('F', 'E') else (2, 1)
    wire = {view: {} for view in audit.VIEWS}
    labels = {}
    for pool, n in (('attacker_fit', 54), ('attacker_validation', 27)):
        a, b = (rng.normal(size=(n, width)).astype(np.float32) for width in dims)
        if interface == 'E': b = a.copy()
        wire['A'][pool], wire['B'][pool], wire['AB'][pool] = a, b, np.column_stack((a, b))
        labels[pool] = {target: np.arange(n, dtype=np.int64)%classes for target, classes in audit.CLASSES.items()}
    indices = {target: np.arange(54) for target in audit.TARGETS}
    observers = {key: observer(sum(dims) if key.startswith('AB__') else dims[0 if key.startswith('A__') else 1],
                               audit.CLASSES[key.split('__')[1]]) for key in audit.OBSERVER_ROLES}
    return wire, labels, indices, observers


def test_fixed_roles_and_view_seeds():
    assert sum(map(len, audit.AUDIT_ROLES.values())) == 11
    assert len(audit.OBSERVER_ROLES) == 9
    assert not any('same_residence' in key or 'commute_over20' in key for key in audit.OBSERVER_ROLES)
    assert audit.role_seed(2, 'AB', 'RAC1P') == 1260221
    assert audit.role_seed(1, 'B', 'income_binary', catchup=True) == 1300112
    seeds = []
    for seed in (0, 1, 2):
        for view, targets in audit.AUDIT_ROLES.items():
            for target in targets:
                base = audit.role_seed(seed, view, target)
                seeds.extend([base, base+10000])
                if view+'__'+target in audit.OBSERVER_ROLES:
                    seeds.append(audit.role_seed(seed, view, target, catchup=True))
    assert len(seeds) == len(set(seeds))


@pytest.mark.parametrize('width', [1, 2, 3, 16, 32])
def test_saved_start_dimensions_fidelity_and_nested_actual_states(width):
    torch.set_num_threads(1)
    rng = np.random.default_rng(5)
    x = rng.normal(size=(16, width)).astype(np.float32)
    y = np.arange(16, dtype=np.int64)%2
    model = observer(width, 2)
    initial, mode, flags = _state_hash(model), model.training, [p.requires_grad for p in model.parameters()]
    with threadpool_limits(limits=1):
        result = audit.fit_role_catchup(model, x, y, x[:8], y[:8], 2, 1300000,
            inherited_exposure={'role': 'A__SEX', 'original_passes': 260}, epochs=2, nested_epochs=1)
    assert _state_hash(model) == initial and model.training == mode
    assert [p.requires_grad for p in model.parameters()] == flags
    assert np.array_equal(result['saved'].predict_proba(x), result['saved'].predict_proba(x.copy()))
    assert result['nested120'].metadata['validation_curve'] == result['nested360'].metadata['validation_curve'][:2]
    assert result['nested360'].metadata['validation_scores']['log_loss'] <= result['nested120'].metadata['validation_scores']['log_loss']
    for epoch, checkpoint in result['training_checkpoints']['catchup'].items():
        assert checkpoint['epoch'] == epoch and checkpoint['optimizer_steps'] == epoch
        assert all(int(value['step']) == epoch for value in checkpoint['optimizer_state']['state'].values())
    assert result['metadata']['initial_fidelity_exact'] and result['metadata']['source_unchanged']


def test_P_full_toy_pools_deduplicate_and_inherit_every_candidate(tmp_path):
    torch.set_num_threads(1)
    wire, labels, indices, observers = fixture('P')
    original = {key: _state_hash(value) for key, value in observers.items()}
    with threadpool_limits(limits=1):
        result = audit.fit_condition_audits(wire, copy.deepcopy(wire), labels, indices, observers, 0, tmp_path/'audit',
            interface='P', inherited_exposure={key: {'role': key, 'row_passes': 260} for key in observers},
            miniature=True, epochs=2, nested_epochs=1, budget={'histgb': {'max_iter': 2, 'max_leaf_nodes': 3}})
    assert original == {key: _state_hash(value) for key, value in observers.items()}
    assert result['metadata']['counts'] == {'new_five_candidate_roles': 11, 'reused_five_candidate_roles': 0,
        'own_catchup_trajectories': 9, 'new_fresh_mlp_trajectories': 22, 'new_static_candidates': 33, 'P_derived_roles_deduplicated': 11}
    for budget, roles in result['candidates'].items():
        assert len(roles) == 11
        assert set(roles['A/commute_over20']) == {'wire__'+cid for cid in audit.FRESH}
        assert set(roles['B/same_residence']) == {'wire__'+cid for cid in audit.FRESH}
        for target in ('SEX', 'RAC1P'):
            coalition = roles['AB/'+target]
            assert len(coalition) == 21
            assert len(result['selection_pools'][budget]['AB/'+target]['standard_independent']) == 15
            assert len(result['selection_pools'][budget]['AB/'+target]['expanded_catchup']) == 18
            for view in ('A', 'B'):
                for cid, candidate in roles[view+'/'+target].items():
                    inherited = coalition['inherited_'+view+'__'+cid]
                    np.testing.assert_array_equal(inherited.predict_proba({'wire': wire['AB']['attacker_validation']}),
                                                  candidate.predict_proba({'wire': wire[view]['attacker_validation']}))
            for scope, cid in result['selection'][budget]['AB/'+target].items():
                assert 'saved_adversary' not in cid
                assert coalition[cid].metadata['validation_scores']['log_loss'] == min(coalition[k].metadata['validation_scores']['log_loss']
                    for k in result['selection_pools'][budget]['AB/'+target][scope])
    saved = json.loads((tmp_path/'audit/audit_selection.json').read_text())
    assert not saved['development_received'] and saved['P_native_equals_wire_deduplicated']


def test_no_development_or_hidden_P_input_and_exact_coalition_alignment():
    wire, labels, indices, _ = fixture('P')
    bad = copy.deepcopy(wire); bad['A']['test'] = bad['A']['attacker_validation']
    with pytest.raises(ValueError, match='Development'): audit._validate_inputs(bad, None, labels, indices, 'P')
    bad = copy.deepcopy(wire); bad['AB']['attacker_fit'][0, 0] += 1
    with pytest.raises(ValueError, match='exact same-person'): audit._validate_inputs(bad, None, labels, indices, 'P')
    bad = copy.deepcopy(wire); bad['A']['attacker_fit'] = np.zeros((54, 16), np.float32)
    with pytest.raises(ValueError, match='shape'): audit._validate_inputs(bad, None, labels, indices, 'P')
    derived = copy.deepcopy(wire); derived['A']['attacker_fit'][0, 0] += 1
    with pytest.raises(ValueError, match='must equal'): audit._validate_inputs(wire, derived, labels, indices, 'P')


def test_F_native_projection_is_the_public_derived_bundle(tmp_path):
    torch.set_num_threads(1)
    model = observer(2, 2)
    base = FittedCandidate('mlp', model, InputStandardizer(np.zeros(2), np.ones(2)), 2,
        {'input_dim': 2, 'family': 'mlp', 'validation_scores': {'log_loss': .4}})
    path = tmp_path/'base'; path.mkdir(); (path/'metadata.json').write_text(json.dumps(base.metadata))
    own = audit._wrapped(base, 'derived__mlp_0', 'derived', 'A', 'SEX', 360, path)
    roles = {'A/SEX': {'derived__mlp_0': own}, 'B/SEX': {}, 'AB/SEX': {},
             'A/RAC1P': {}, 'B/RAC1P': {}, 'AB/RAC1P': {}}
    audit.inherit_singletons(roles, {'wire': (16, 16), 'derived': (2, 1)})
    native = np.array([[.2, .8, .3], [.4, .6, .1]], np.float32)
    inherited = roles['AB/SEX']['inherited_A__derived__mlp_0']
    np.testing.assert_array_equal(inherited.predict_proba({'wire': np.zeros((2, 32)), 'derived': native}),
                                  own.predict_proba({'wire': np.ones((2, 16)), 'derived': native[:, :2]}))
    with pytest.raises(ValueError, match='available public-derived'): inherited.predict_proba({'wire': np.zeros((2, 32))})
