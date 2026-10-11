"""Queued numerical fixtures derived from locked R2 four-row finite law."""
import numpy as np


def fixture():
    from conditional_release.lp import ConstraintRows
    identity=np.arange(4,dtype=np.int64)
    actions=np.stack((identity,np.array([4,4,5,5],np.int64),identity,identity),axis=1)
    e1=np.repeat(actions,4,axis=1);e2=np.tile(actions,(1,4))
    valid=np.tile([True,True,False,False],(4,1))
    cost=np.array([1.,1.,9.,9.])[:,None]*((e1!=identity[:,None])|(e2!=identity[:,None]))
    def constraints(law):
        constant,coefficient=law.expected_coefficients(cost)
        return ConstraintRows(["heterogeneous/maximum-cost"],coefficient[None,:],np.array([1.-constant]))
    return identity,e1,e2,valid,np.array([0,0,1,1],np.int64),np.array([0,1,0,1],np.int64),constraints


def test_binary_fit_produces_exact_executable_one_hot_not_rounded_lp():
    from conditional_release.controls import categorical_map,template_bounds,map_law,solve_binary_control,ControlCeilings
    identity,e1,e2,valid,groups,sex,constraints=fixture()
    pairvalid=(valid[:,:,None]&valid[:,None,:]).reshape(4,16)
    law=map_law(e1,e2,identity,identity,categorical_map(groups,16),template_bounds(groups,pairvalid),width=16)
    result=solve_binary_control(law,sex,constraints(law),ControlCeilings(seconds=10.))
    assert result["success"]
    assert result["solver_class"]=="binary MILP"
    assert np.isin(result["theta"],[0.,1.]).all()
    assert np.isin(law.probabilities(result["theta"]),[0.,1.]).all()
    assert result["certificate"]["relaxation_certified"]
    assert result["certificate"]["maximum_primal_violation"]<=1e-8


def test_alternating_selected_law_is_actual_product_not_joint_relaxation():
    from conditional_release.controls import fit_independent_product,ControlCeilings
    identity,e1,e2,valid,groups,sex,constraints=fixture()
    result=fit_independent_product(e1,e2,identity,identity,groups,groups,valid,valid,sex,constraints,
                                   ControlCeilings(rounds=1,seconds=10.))
    assert result["success"]
    selected=result["selected"]
    q=selected["law"].probabilities(selected["theta"]).reshape(4,4,4)
    one,two=selected["factors"]
    np.testing.assert_allclose(q,one[groups,:,None]*two[groups,None,:],atol=1e-12)
    assert result["global_optimality"]=="UNRESOLVED"


def test_sequential_second_stage_preserves_exact_first_marginal_both_orders():
    from conditional_release.controls import fit_sequential,ControlCeilings
    identity,e1,e2,valid,groups,sex,constraints=fixture()
    for first in (1,2):
        result=fit_sequential(e1,e2,identity,identity,groups,valid,valid,sex,constraints,first,
                              ControlCeilings(seconds=10.))
        assert result["success"]
        q=result["law"].probabilities(result["theta"]).reshape(4,4,4)
        marginal=q.sum(2 if first==1 else 1)
        np.testing.assert_allclose(marginal,result["first_theta"][groups],atol=1e-12)
