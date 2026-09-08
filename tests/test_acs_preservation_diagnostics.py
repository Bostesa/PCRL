import numpy as np
import pytest
from experiments.acs_preservation_diagnostics import fit_affine,evaluate_affine


def test_affine_recovers_moved_coordinates_out_of_fit_and_keeps_fit_intercept():
    rng=np.random.default_rng(8);teacher=rng.normal(size=(100,16));test=rng.normal(size=(31,16))
    matrix=rng.normal(size=(16,16));offset=rng.normal(size=16)
    mean=np.linspace(-2,2,32);scale=np.linspace(.5,2,32)
    fit=fit_affine(teacher@matrix+offset,teacher,mean,scale)
    score=evaluate_affine(fit,test@matrix+offset,test,mean,scale)
    assert fit['rank']==17 and score['affine_mean_mse']<1e-20
    assert score['direct_mean_mse']>1
    prior=((test-mean[:16])/scale[:16] - ((teacher-mean[:16])/scale[:16]).mean(0))**2
    np.testing.assert_allclose(score['prior_per_coordinate_mse'],prior.mean(0))
    with pytest.raises(ValueError):fit_affine(teacher,teacher,mean,scale,fit_pool='test')


def test_fixed_rank_tolerance_and_constant_release_have_finite_prior_behavior():
    rng=np.random.default_rng(2);y=rng.normal(size=(40,16));x=np.zeros((40,16))
    fitted=fit_affine(x,y,np.zeros(32),np.ones(32));assert fitted['rank']==1
    score=evaluate_affine(fitted,x,y,np.zeros(32),np.ones(32))
    np.testing.assert_allclose(score['affine_mean_mse'],score['prior_mean_mse'])
    with pytest.raises(ValueError):fit_affine(x,y,np.zeros(32),np.ones(32),rcond=1e-6)
