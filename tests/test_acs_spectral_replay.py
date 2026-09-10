"""Pure independent verification fixtures; no scientific fitting."""
import copy
import numpy as np
import pytest
from scripts.replay_acs_residual_spectral import (SCOPES, eligible, verify_selection,
    expected_h_columns, literal_labels, literal_kernel)

def candidate(space='wire',loss=1.,diagnostic=False):
    return {'space':space,'validation_scores':{'log_loss':loss},'diagnostic_only':diagnostic}

def fixture():
    cs={'wire__logistic':candidate(loss=.9),'wire__mlp_0':candidate(loss=.8),
        'derived__mlp_0':candidate('derived',.7),'catchup':candidate(loss=.6),
        'saved_adversary':candidate(loss=.01,diagnostic=True),
        'wire__kernel__0.0001':candidate(loss=.5),'wire__kernel__0.01':candidate(loss=.5),
        'derived__kernel__1.0':candidate('derived',.4),
        'anchor__catchup':candidate(loss=.55),
        'inherited_A__wire__kernel__1.0':candidate(loss=.45),
        'inherited_B__saved_adversary':candidate(loss=0.,diagnostic=True)}
    pools={scope:eligible(cs,scope) for scope in SCOPES}
    choices={scope:min(ids,key=lambda cid:(cs[cid]['validation_scores']['log_loss'],cid)) for scope,ids in pools.items()}
    return cs,choices,pools

def test_all_six_scope_memberships_and_deterministic_selection():
    cs,choices,pools=fixture();verify_selection(cs,choices,pools)
    assert choices['standard_independent']=='wire__mlp_0'
    assert choices['expanded_independent']=='derived__mlp_0'
    assert choices['expanded_catchup']=='anchor__catchup'
    assert choices['kernel_standard_independent']=='inherited_A__wire__kernel__1.0'
    assert choices['kernel_expanded_independent']=='derived__kernel__1.0'
    for scope,ids in pools.items():
        assert all('saved_adversary' not in cid for cid in ids)
        if not scope.startswith('kernel_'):assert all('kernel__' not in cid for cid in ids)
        if not scope.endswith('catchup'):assert all('catchup' not in cid for cid in ids)
    # Exact tie uses full stored ID.
    del cs['inherited_A__wire__kernel__1.0']
    ids=eligible(cs,'kernel_standard_independent')
    assert min(ids,key=lambda cid:(cs[cid]['validation_scores']['log_loss'],cid))=='wire__kernel__0.0001'

def test_selection_rejects_omitted_opportunity_and_development_winner():
    cs,choices,pools=fixture();bad=copy.deepcopy(pools)
    bad['kernel_standard_independent'].remove('wire__kernel__0.01')
    with pytest.raises(AssertionError):verify_selection(cs,choices,bad)
    wrong=dict(choices);wrong['standard_independent']='wire__logistic'
    with pytest.raises(AssertionError):verify_selection(cs,wrong,pools)

def test_dynamic_raw_routes_and_nested_singleton_projection():
    assert expected_h_columns('AB',13,None)==[0,1,2,3,13,14]
    assert expected_h_columns('AB',13,[4,5])==[13,14]
    assert expected_h_columns('A',13,[1,3])==[1,3]

def test_literal_labels_masks_full_race_schema_and_eligibility():
    import pandas as pd
    f=pd.DataFrame({'SEX':[1,2,9],'RAC1P':[1,9,0],'PINCP':[50000,50001,np.nan],
        'ESR':[1,6,0],'PUBCOV':[1,2,0],'MIG':[1,3,0],'JWMNP':[20,21,20.5]})
    labels=literal_labels(f)
    assert labels['SEX'].tolist()==[0,1,-1]
    assert labels['RAC1P'].tolist()==[0,8,-1]
    assert labels['income_binary'].tolist()==[0,1,-1]
    assert labels['civilian_at_work'].tolist()==[1,0,-1]
    assert labels['public_coverage'].tolist()==[1,0,-1]
    assert labels['same_residence'].tolist()==[1,0,-1]
    assert labels['commute_over20'].tolist()==[0,1,-1]

def test_literal_kernel_clip_full_classes_without_refit():
    from types import SimpleNamespace
    model=SimpleNamespace(preprocessing=SimpleNamespace(mean=np.array([1.]),scale=np.array([2.])),
        phase=np.zeros(2),omega=np.array([[1.,2.]]),center=np.zeros(2),
        prior=np.array([.5,.5,0.]),coef=np.array([[10.,-10.,0.],[-2.,2.,0.]]))
    result=literal_kernel(model,np.array([[1.],[2.]]))
    assert result.shape==(2,3) and np.isfinite(result).all() and (result>0).all()
    np.testing.assert_allclose(result.sum(1),1)

def test_replay_normalizes_raw_projected_memory_order_before_inference(tmp_path):
    from scripts.replay_acs_residual_spectral import Replay
    replay=Replay(tmp_path,tmp_path,True)
    path=tmp_path/'candidate'
    def check_layout(x):
        assert x.flags.c_contiguous
        return np.column_stack((x[:,0],x[:,1]))
    replay.models[str(path)]=check_layout
    wire=np.arange(60,dtype=float).reshape(10,6)
    projected=wire[:,[0,2]]
    assert projected.flags.f_contiguous and not projected.flags.c_contiguous
    result=replay.predict(path,{},projected,'validation')
    assert np.array_equal(result,projected)
