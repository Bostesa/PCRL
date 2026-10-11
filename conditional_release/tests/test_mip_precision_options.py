"""Queued engineering conformance to unchanged executable 1e-8 guard."""
import numpy as np


def test_actual_binary_solver_receives_stricter_precision_without_new_budget(monkeypatch):
    import scipy.optimize
    from conditional_release.controls import categorical_map,template_bounds,map_law,solve_binary_control,ControlCeilings
    from conditional_release.tests.test_control_solvers import fixture
    real=scipy.optimize.milp;seen=[]
    def audited(*args,**kwargs):
        seen.append(dict(kwargs['options']))
        return real(*args,**kwargs)
    monkeypatch.setattr(scipy.optimize,'milp',audited)
    identity,e1,e2,valid,groups,sex,constraints=fixture()
    pairvalid=(valid[:,:,None]&valid[:,None,:]).reshape(4,16)
    law=map_law(e1,e2,identity,identity,categorical_map(groups,16),template_bounds(groups,pairvalid),width=16)
    result=solve_binary_control(law,sex,constraints(law),ControlCeilings(seconds=10.))
    assert len(seen)==1
    options=seen[0]
    for key in ('mip_feasibility_tolerance','primal_feasibility_tolerance','dual_feasibility_tolerance'):
        assert options[key]==1e-9
    assert options['threads']==1 and options['time_limit']<=7.5
    assert options['mip_rel_gap']==1e-6
    assert result['success'] and np.isin(result['theta'],[0.,1.]).all()
    assert result['certificate']['maximum_primal_violation']<=1e-8
    assert result['certificate']['maximum_integer_error']<=1e-8
