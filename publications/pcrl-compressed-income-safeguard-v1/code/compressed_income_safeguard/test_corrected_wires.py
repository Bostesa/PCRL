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
    for arm in ('CONSTRAINED','ORDINARY','LOCAL_TWIN','TASK_ONLY'):
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


def test_direct_authorization_rejects_old_campaign():
    from .pilot import validate_direct_lock, RECIPE, canonical_sha
    old = {'owner':'ROOT','status':'PILOT_TRIGGERED','recipe_sha256':canonical_sha(RECIPE),'INNER_used_for_trigger':False}
    with pytest.raises(PermissionError): validate_direct_lock(old, seed=0, checkpoint={'sha256':'a'*64}, geometry_sha256='b'*64)


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


def test_direct_lock_binds_both_starts_and_current_geometry_recipe():
    from .pilot import validate_direct_lock, RECIPE, canonical_sha
    starts={'0':{'path':'checkpoint0.pt','sha256':'a'*64},'1':{'path':'checkpoint1.pt','sha256':'b'*64}}
    lock={'schema':'CIS1-direct-trial-lock-v1','campaign':'pcrl_compressed_income_safeguard_v1',
          'owner':'ROOT','status':'DIRECT_TRIAL_AUTHORIZED','authorization':'SOURCE_HANDOFF_EXPLICIT_DIRECT_TRIAL',
          'INNER_truth_accessed':False,'recipe_sha256':canonical_sha(RECIPE),'starting_checkpoints':starts,
          'geometry_bindings':{'0':'c'*64,'1':'d'*64}}
    assert validate_direct_lock(lock,seed=0,checkpoint=starts['0'],geometry_sha256='c'*64)
    changed=copy.deepcopy(lock);changed['recipe_sha256']='7c4ca6110be3c4774c9cdfdcff46babd44c117a39cf81e47d7f727de7bbf1528'
    with pytest.raises(PermissionError):validate_direct_lock(changed,seed=0,checkpoint=starts['0'],geometry_sha256='c'*64)
    with pytest.raises(PermissionError):validate_direct_lock(lock,seed=0,checkpoint=starts['0'],geometry_sha256='e'*64)
