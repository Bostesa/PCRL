import numpy as np
from experiments.acs_spectral_audits import fit_kernel, route_ancestor, expected_losses

def test_kernel_replays_and_has_finite_full_probabilities(tmp_path):
    rng=np.random.default_rng(17); x=rng.normal(size=(70,4)); y=(x[:,0]*x[:,1]>0).astype(int)
    result=fit_kernel(x[:50],y[:50],x[50:],y[50:],3,27,tmp_path/'kernel',features=24)
    assert len(result)==3
    for candidate in result.values():
        p=candidate.predict_proba(x)
        assert p.shape==(70,3) and np.all(p>0)
        np.testing.assert_allclose(p.sum(1),1)
        assert candidate.metadata['fit_support'][2]==0
        assert candidate.metadata['normal_equation_max_error']<1e-9
    import joblib
    restored=joblib.load(tmp_path/'kernel'/'0.01'/'model.joblib')
    assert np.array_equal(restored.predict_proba(x),result['0.01'].predict_proba(x))

def test_raw_ancestor_projection_variable_rank():
    assert route_ancestor('AB',11,None)==(0,1,2,3,11,12)
    assert route_ancestor('AB',11,(4,5))==(11,12)
    assert route_ancestor('A',11,None)==(0,1,2,3)

def test_withholding_averages_losses_not_probabilities():
    a=np.array([0.,1.,4.]); b=np.array([3.,2.,0.]); w=np.array([2.,1.,3.])
    for p in [0,.25,.5,.75,1]:
        loss=expected_losses(a,b,p)
        assert np.allclose(loss,(1-p)*a+p*b)
        assert np.isclose(loss@w/w.sum(),(1-p)*(a@w/w.sum())+p*(b@w/w.sum()))

def test_kernel_resume_rejects_changed_validation_and_seed(tmp_path):
    import pytest
    rng=np.random.default_rng(21);x=rng.normal(size=(30,3));y=np.arange(30)%2
    path=tmp_path/'k';fit_kernel(x[:20],y[:20],x[20:],y[20:],2,12,path,features=12)
    with pytest.raises(AssertionError):
        fit_kernel(x[:20],y[:20],x[20:]+.01,y[20:],2,12,path,features=12)
    with pytest.raises(AssertionError):
        fit_kernel(x[:20],y[:20],x[20:],y[20:],2,13,path,features=12)
