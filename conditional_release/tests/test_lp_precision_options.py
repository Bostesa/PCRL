"""Queued actual fixed LP option conformance and unchanged certificate cap."""
import numpy as np


def test_corrected_lp_fixed_accuracy_preserves_program_and_checker(monkeypatch):
    from conditional_release import lp,accurate_memory_lp
    from conditional_release.tests.test_wide_control_maps import fixture
    from conditional_release.wide_control_fit import coordinate_law,widths
    template,metadata,contexts=fixture();groups=np.maximum(contexts,0)
    other=np.zeros((len(contexts),widths(metadata)[1]));other[:,0]=1.
    law=coordinate_law(template,metadata,contexts,groups,other,1)
    constraints=lp.ConstraintRows([],np.empty((0,law.parameters)),np.empty(0));sex=np.array([0,1,0],np.int64)
    original=lp.linprog;seen=[]
    def audited(*args,**kwargs):
        seen.append(dict(kwargs['options']));return original(*args,**kwargs)
    monkeypatch.setattr(lp,'linprog',audited)
    result=accurate_memory_lp.solve_protection(law,sex,constraints,time_limit=10.,tolerance=1e-8)
    assert len(seen)==1 and result['success']
    assert seen[0]['primal_feasibility_tolerance']==seen[0]['dual_feasibility_tolerance']==1e-10
    assert seen[0]['threads']==1 and seen[0]['time_limit']==10.
    assert result['certificate']['tolerance']==1e-7
    assert result['certificate']['maximum_primal_violation']<=1e-8
    old=lp.protection_program(law,sex,constraints);new=result['program']
    assert (old.inequality_matrix!=new.inequality_matrix).nnz==0
    assert (old.equality_matrix!=new.equality_matrix).nnz==0
    np.testing.assert_array_equal(old.objective,new.objective)
