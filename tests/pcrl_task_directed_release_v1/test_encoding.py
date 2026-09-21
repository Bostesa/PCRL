import numpy as np
from scipy.special import expit,logit


def test_risk_refinement_can_split_a_task_degenerate_parent_without_labels():
    from experiments.pcrl_task_directed_release_v1.encoding import Codebook
    x=np.zeros((80,32)); r=np.zeros(80)
    risk=np.column_stack((np.r_[np.zeros(40),np.ones(40)],np.r_[np.ones(40),np.zeros(40)]))
    code=Codebook.fit(r,x,risk,seed=11)
    got=code.assign(r,x,risk)
    assert np.unique(got['T0']).tolist()==[0]
    assert len(np.unique(got['Ttask']))==1
    assert len(np.unique(got['Trisk']))==2
    for name in ('Ttask','Trisk'):
        assert np.array_equal(got[name]//2,got['T0'])


def test_task_ties_use_features_and_never_identifiers():
    from experiments.pcrl_task_directed_release_v1.encoding import Codebook
    x=np.zeros((60,32));x[:,0]=np.arange(60)
    code=Codebook.fit(np.zeros(60),x,np.ones((60,2))*.5,seed=1)
    got=code.assign(np.zeros(60),x,np.ones((60,2))*.5)
    assert len(np.unique(got['Ttask']))==2
    assert code.diagnostics['task_tie_projection_cells']==[0]


def test_dictionary_offsets_satisfy_independent_soft_label_stationary_equation():
    from experiments.pcrl_task_directed_release_v1.encoding import fit_dictionary
    b=np.array([.15,.25,.45,.6,.7,.8]);p=expit(logit(b)+.7)
    w=np.array([1.,2.,3.,4.,5.,6.]);d=fit_dictionary(p,b,w,max_actions=2)
    assert 0. in d['offsets'] and len(d['offsets'])==2
    assert abs(d['offsets'][1]-.7)<1e-10
    assert abs(np.dot(.5+.5*w/w.mean(),expit(logit(b)+d['offsets'][1])-p))<1e-10


def test_local_partition_is_independent_of_b_service_and_label_outcomes():
    from experiments.pcrl_task_directed_release_v1.encoding import ServicePartitions
    rng=np.random.default_rng(19);ha=rng.random((100,4));hb=rng.random((100,2))
    cells=ServicePartitions.fit(ha,hb,seed=0)
    a,ab=cells.assign(ha,hb)
    a2,ab2=cells.assign(ha,1-hb)
    assert np.array_equal(a,a2)
    assert np.any(ab!=ab2)
    assert np.array_equal(ab//2,a)


def test_reserved_dictionary_strengthens_common_calibration_without_widening_primary_actions():
    from types import SimpleNamespace
    from experiments.pcrl_task_directed_release_v1.encoding import Encoder
    from experiments.pcrl_task_directed_release_v1.data import RuntimeInputs
    predictor=SimpleNamespace(predict_proba=lambda x:np.tile([.4,.6],(len(x),1)))
    risk=SimpleNamespace(predict=lambda x:np.ones((len(x.x_a),11))/11)
    code=SimpleNamespace(assign=lambda r,x,s:{k:np.zeros(len(r),int) for k in ('T0','Ttask','Trisk')})
    encoder=Encoder(predictor,predictor,risk,code,
                    {17:{'offsets':np.array([0.,1.])},33:{'offsets':np.array([0.,-.5,.5])}},None,{})
    encoded=encoder.encode(RuntimeInputs(np.zeros((4,32)),np.ones((4,4))*.5))
    assert encoded['actions'][17].shape==(4,2)
    assert encoded['actions'][33].shape==(4,3)
    assert encoded['global_offsets'].shape==(4,4)
    assert np.array_equal(encoded['global_offsets'][:,:2],encoded['actions'][17])
    assert np.array_equal(encoded['global_offsets'][:,2:],encoded['actions'][33][:,1:])
