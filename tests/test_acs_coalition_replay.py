import numpy as np
import pytest
import torch
from scripts import verify_acs_coalition as replay


def literal_state():
    rng = np.random.default_rng(79)
    state = {'input_mean':torch.tensor(rng.normal(size=32)), 'input_scale':torch.tensor(rng.uniform(.5,2,size=32))}
    for view, targets in replay.ASSIGNED.items():
        for layer, shape in [('0',(64,32)),('2',(16,64))]:
            prefix = f'branches.{view}.mapper.{layer}.'
            state[prefix+'weight'] = torch.from_numpy(rng.normal(0,.1,shape).astype(np.float32))
            state[prefix+'bias'] = torch.from_numpy(rng.normal(0,.1,shape[0]).astype(np.float32))
        for target in targets:
            prefix = f'branches.{view}.heads.{target}.'
            state[prefix+'weight'] = torch.from_numpy(rng.normal(0,.1,(1,16)).astype(np.float32))
            state[prefix+'bias'] = torch.from_numpy(rng.normal(0,.1,1).astype(np.float32))
    return state


def test_literal_public_interface_uses_only_assigned_heads_and_same_person_concat():
    torch.set_num_threads(1)
    state = literal_state(); raw = np.random.default_rng(10).normal(size=(7,32)).astype(np.float32)
    wire, derived, native = replay.literal_wires(state,raw,'F')
    p, again, second = replay.literal_wires(state,raw,'P')
    assert {v:a.shape[1] for v,a in wire.items()} == {'A':16,'B':16,'AB':32}
    assert {v:a.shape[1] for v,a in p.items()} == {'A':2,'B':1,'AB':3}
    for view in p:
        np.testing.assert_array_equal(p[view],derived[view]); np.testing.assert_array_equal(p[view],again[view])
    np.testing.assert_array_equal(wire['AB'],np.concatenate((wire['A'],wire['B']),1))
    recomposed, _ = replay.public_probabilities(state,wire)
    for view in p: np.testing.assert_array_equal(recomposed[view],p[view])
    for i,target in enumerate(replay.ASSIGNED['A']): np.testing.assert_array_equal(native[target][:,1],p['A'][:,i])
    np.testing.assert_array_equal(native['public_coverage'][:,1],p['B'][:,0])
    assert sum(value.numel() for key,value in state.items() if key.startswith('branches.')) == 6355
    with pytest.raises(ValueError): replay.literal_wires(state,raw[:,:16],'F')


def test_projection_selects_original_public_view_and_canonical_memory():
    a = np.arange(20,dtype=np.float32).reshape(10,2); b = -np.ones((10,1),np.float32)
    release = {'derived/AB/test':np.concatenate((a,b),1)}
    meta = {'view':'AB','space':'derived','projection_columns':[0,1]}
    mask = np.arange(10)%2 == 0
    x = replay.candidate_input(release,meta,'test',mask)
    np.testing.assert_array_equal(x,a[mask]); assert x.flags.c_contiguous
    meta['projection_columns'] = [2]
    np.testing.assert_array_equal(replay.candidate_input(release,meta,'test',mask),b[mask])


def test_three_scope_selection_excludes_saved_but_keeps_every_legal_inheritance():
    candidates = {cid:{'space':space} for cid,space in [('wire__logistic','wire'),('derived__mlp_0','derived'),
        ('catchup','wire'),('saved_adversary','wire'),('inherited_A__wire__mlp_0','wire'),
        ('inherited_A__derived__mlp_1','derived'),('inherited_B__catchup','wire'),('inherited_B__saved_adversary','wire')]}
    assert replay.expected_pool(candidates,'standard_independent') == ['inherited_A__wire__mlp_0','wire__logistic']
    assert replay.expected_pool(candidates,'expanded_independent') == ['derived__mlp_0','inherited_A__derived__mlp_1','inherited_A__wire__mlp_0','wire__logistic']
    assert replay.expected_pool(candidates,'expanded_catchup') == ['catchup','derived__mlp_0','inherited_A__derived__mlp_1','inherited_A__wire__mlp_0','inherited_B__catchup','wire__logistic']
