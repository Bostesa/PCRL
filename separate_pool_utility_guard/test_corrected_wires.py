"""Finite fixtures only; execute through the ROOT engineering queue."""
import copy
import pytest

@pytest.fixture(autouse=True)
def queued():
    from .scheduler import assert_active_from_env
    assert_active_from_env(stage='admission')


def fixture_model():
    import torch
    from task_conditioned_representation.useful_service import UsefulModel
    torch.manual_seed(913)
    model = UsefulModel([f'x{i}' for i in range(83)])
    model.eval()
    return model


def test_hybrid_factorization_and_literal_pair_excludes_raw_income():
    import torch
    from .wires import bind_geometry, corrected_trial_wires, prepare_geometry
    model = fixture_model()
    binding = bind_geometry(model, source_seed=0, checkpoint_sha256='a'*64)
    geometry = prepare_geometry(binding)
    x = torch.randn(19, 83)
    wires = corrected_trial_wires(model, x, geometry, variant='RAW_HYBRID')
    z = model.encode(0, x).double()
    b = geometry.bases[0]
    assert torch.equal(wires[0]['representation'], (z @ b.T) @ b)
    assert not torch.equal(wires[0]['representation'], z)
    assert torch.equal(wires[1]['representation'], model.encode(1, x))
    assert [w['full'].shape[1] for w in wires] == [21, 29]
    assert torch.equal(torch.cat([w['full'] for w in wires], 1)[:, :16], wires[0]['representation'])
    assert torch.allclose(wires[0]['probabilities'], corrected_trial_wires(model,x,geometry,variant='RAW_FULL')[0]['probabilities'],atol=1e-10,rtol=0)
    for arm in ('SEPARATE_GUARD','GLOBAL_UPDATE_GUARD','OLD_BATCH_GUARD','ORDINARY','TASK_ONLY'):
        assert torch.equal(corrected_trial_wires(model,x,geometry,variant=arm)[0]['full'], wires[0]['full'])


def test_binding_rejects_old_hash_and_changed_income_head():
    import torch
    from .wires import bind_geometry, corrected_trial_wires, prepare_geometry
    model = fixture_model(); binding = bind_geometry(model, source_seed=1, checkpoint_sha256='b'*64)
    stale = copy.deepcopy(binding); stale['heads'][0]['B'][0][0] += 0.1
    with pytest.raises(PermissionError): prepare_geometry(stale)
    geometry = prepare_geometry(binding)
    with torch.no_grad(): model.head[0].weight[0,0] += 0.1
    with pytest.raises(PermissionError): corrected_trial_wires(model, torch.zeros(2,83), geometry)


def test_frozen_guard_detects_basis_and_signed_zero_tensor_changes():
    import torch
    from .wires import bind_geometry, prepare_geometry, FrozenIncomeGuard
    model = fixture_model()
    with torch.no_grad(): model.head[0].bias[0] = 0.
    geometry = prepare_geometry(bind_geometry(model,source_seed=0,checkpoint_sha256='c'*64))
    guard = FrozenIncomeGuard(model, geometry)
    guard.verify()
    with torch.no_grad(): geometry.bases[0][0,0] += 1e-8
    with pytest.raises(AssertionError): guard.verify()
    with torch.no_grad(): geometry.bases[0][0,0] -= 1e-8
    fresh_guard = FrozenIncomeGuard(model, geometry)
    with torch.no_grad(): model.head[0].bias[0] = -0.
    with pytest.raises(AssertionError): fresh_guard.verify()


def test_role_batching_identical_under_equal_frozen_income_models():
    import numpy as np
    import torch
    from .wires import bind_geometry, prepare_geometry, forward_role, ROLE_FORWARD_BATCH_SIZE
    model = fixture_model(); second = copy.deepcopy(model)
    geometry = prepare_geometry(bind_geometry(model,source_seed=0,checkpoint_sha256='d'*64))
    raw = np.zeros((ROLE_FORWARD_BATCH_SIZE+3,83),dtype=np.float32)
    raw[:,0] = np.arange(len(raw),dtype=np.float32)/100
    with torch.no_grad(): second.head[1].bias.add_(.1)
    first = forward_role(model,raw,model.names,geometry,variant='RAW_HYBRID')
    other = forward_role(second,raw,second.names,geometry,variant='ORDINARY')
    np.testing.assert_array_equal(first[0]['wire'], other[0]['wire'])


def test_old_corrected_trial_lock_cannot_authorize_new_trial():
    import json
    from .recipe import validate_science_lock
    from pathlib import Path
    old = json.loads((Path(__file__).resolve().parents[2] / 'pcrl-compressed-income-safeguard-v1/results/pcrl_compressed_income_safeguard_v1/DIRECT_TRIAL_LOCK.json').read_text())
    with pytest.raises(PermissionError):
        validate_science_lock(old, seed=0, checkpoint_sha256=old['starting_checkpoints']['0']['sha256'],
                              geometry_sha256=old['geometry_bindings']['0'], role_split_sha256='a'*64)


def test_residual_coordinate_changes_cannot_enter_income_release():
    import torch
    from .wires import bind_geometry, prepare_geometry, corrected_trial_wires
    model = fixture_model()
    with torch.no_grad():
        model.head[0].weight.zero_();model.head[0].bias.zero_()
        model.head[0].weight[0,0]=-1.;model.head[0].weight[1,0]=1.
    model.encode=lambda i,x:x[:,:16]
    geometry=prepare_geometry(bind_geometry(model,source_seed=0,checkpoint_sha256='e'*64))
    x=torch.zeros(2,83);x[1,1:16]=1e12
    wire=corrected_trial_wires(model,x,geometry)[0]
    assert torch.equal(wire['full'][0],wire['full'][1])
    assert wire['decision'].tolist()==[0,0]
    assert wire['probabilities'].tolist()==[[.5,.5],[.5,.5]]


def test_science_lock_binds_start_geometry_recipe_and_split():
    from .recipe import validate_science_lock, RECIPE, canonical_sha
    lock={'schema':'SPG1-science-lock-v1','campaign':'pcrl_separate_pool_utility_guard_v1','owner':'ROOT',
          'status':'DIRECT_TRIAL_AUTHORIZED','INNER_truth_accessed':False,'VALIDATION_outcomes_opened':False,
          'recipe_sha256':canonical_sha(RECIPE),'role_split_sha256':'f'*64,
          'starting_checkpoints':{'0':'a'*64,'1':'b'*64},'geometry_bindings':{'0':'c'*64,'1':'d'*64}}
    assert validate_science_lock(lock,seed=0,checkpoint_sha256='a'*64,geometry_sha256='c'*64,role_split_sha256='f'*64)
    for key,value in (('recipe_sha256','0'*64),('role_split_sha256','0'*64),('VALIDATION_outcomes_opened',True)):
        changed=copy.deepcopy(lock);changed[key]=value
        with pytest.raises(PermissionError):validate_science_lock(changed,seed=0,checkpoint_sha256='a'*64,geometry_sha256='c'*64,role_split_sha256='f'*64)
    with pytest.raises(PermissionError):validate_science_lock(lock,seed=0,checkpoint_sha256='a'*64,geometry_sha256='e'*64,role_split_sha256='f'*64)
