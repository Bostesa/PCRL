"""Data-free risk fixtures; execute only under the coordinator's queue."""
import numpy as np
import pytest
from compressed_income_safeguard.geometry import build_geometry, projected_release, audit_geometry
from compressed_income_safeguard.gradient_worker import project_displacement, finite_step, parameter_vector


def test_binary_rank_and_nullspace_no_extra_channel():
    w=np.array([[1.,0.,0.],[-1.,0.,0.]])
    g=build_geometry(w,np.zeros(2))
    z=np.array([[2.,1e12,-1e12],[2.,-7.,19.]])
    release=projected_release(z,g)
    assert g['rank']==1
    assert np.array_equal(release['representation'][0],release['representation'][1])
    assert release['full'].shape==(2,8)
    assert set(release)=={'representation','centered_logits','probabilities','decision','full'}
    assert audit_geometry(z,g)['decision_preserving']


def test_centering_six_classes_rank_and_actual_projected_head():
    w=np.vstack((np.eye(5),np.zeros((1,5))))
    g=build_geometry(w,np.arange(6,dtype=float))
    assert g['rank']==5
    z=np.arange(15,dtype=float).reshape(3,5)
    r=projected_release(z,g)
    actual=r['representation']@w.T+np.arange(6)
    actual-=actual.mean(1,keepdims=True)
    np.testing.assert_array_equal(r['centered_logits'],actual)
    assert audit_geometry(z,g)['passed']


def test_zero_rank_exact_tie_chooses_first_class():
    g=build_geometry(np.ones((2,3)),np.zeros(2))
    r=projected_release(np.array([[1.,2.,3.]]),g)
    assert g['rank']==0 and r['decision'].tolist()==[0]
    assert audit_geometry(np.array([[1.,2.,3.]]),g)['near_tie_rows']==1


def test_rank_boundary_is_explicitly_unavailable():
    # Relative cutoff is 1e-12; within [0.5 tau,2 tau] is ambiguous.
    w=np.diag([1.,1e-12,0.]);w=np.vstack((w,-w))
    with pytest.raises(ValueError,match='rank boundary'):
        build_geometry(w,np.zeros(6))


def test_qp_two_opposed_constraints_preserves_zero_witness():
    d,receipt=project_displacement(np.array([3.,2.]),np.array([[1.,0.],[-1.,0.]]),1.)
    np.testing.assert_allclose(d,[0.,1.],atol=1e-12)
    assert receipt['zero_feasible'] and receipt['certified']


def test_task_gradient_scale_does_not_remove_tiny_guard():
    d,receipt=project_displacement(np.array([1.,2.]),np.array([[1e-30,0.]]),10.)
    assert d[0]<=1e-10 and receipt['zero_gradients']==0


def test_finite_step_rejects_curvature_and_restores_parameters():
    import torch
    p=torch.nn.Parameter(torch.tensor([0.],dtype=torch.float64))
    def task():return {'nll':float(p.square().sum()),'brier':float(p.square().sum())}
    before=parameter_vector([p])
    log=finite_step([p],np.array([1.]),np.zeros((2,1)),task,1.)
    assert log['status']=='REJECTED_ZERO_STEP'
    np.testing.assert_array_equal(parameter_vector([p]),before)


def test_finite_step_tests_actual_float32_parameter_displacement():
    import torch
    p=torch.nn.Parameter(torch.tensor([1e8],dtype=torch.float32))
    log=finite_step([p],np.array([1e-4]),np.zeros((2,1)),lambda:{'nll':0.,'brier':0.},1.)
    assert log['status']=='ROUNDING_ZERO_STEP'
    assert log['actual_displacement_norm']==0.
