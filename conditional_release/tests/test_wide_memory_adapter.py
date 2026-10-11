"""Queue-only numerical parity for unchanged WIDE low-memory programs."""
import numpy as np
from conditional_release import lp,memory_lp
from conditional_release.controls import ControlCeilings,solve_binary_control
from conditional_release.wide_control_fit import fit_product,fit_sequential,coordinate_law,widths
from conditional_release.tests.test_wide_control_maps import fixture


def no_cost(law):
    return lp.ConstraintRows([],np.empty((0,law.parameters)),np.empty(0))


def test_wide_coordinate_encoded_program_and_bayes_exact_parity():
    law,metadata,contexts=fixture();groups=np.maximum(contexts,0)
    other=np.zeros((len(contexts),widths(metadata)[1]));other[:,0]=1.
    law=coordinate_law(law,metadata,contexts,groups,other,1)
    sex=np.array([0,1,0],np.int64);rows=no_cost(law)
    one,two=lp.protection_program(law,sex,rows),memory_lp.protection_program(law,sex,rows)
    for key in ('objective','inequality_rhs','equality_rhs','bounds'):
        np.testing.assert_array_equal(getattr(one,key),getattr(two,key))
    for key in ('inequality_matrix','equality_matrix'):
        assert (getattr(one,key)!=getattr(two,key)).nnz==0
    assert one.names==two.names
    theta=np.zeros(law.parameters);theta.reshape(2,-1)[:,0]=1
    assert lp.direct_bayes(law,theta,sex)==memory_lp.direct_bayes(law,theta,sex)


def test_wide_product_and_both_sequential_backends_same_optimum():
    template,metadata,contexts=fixture();group=np.maximum(contexts,0);sex=np.array([0,1,0],np.int64)
    ceiling=ControlCeilings(rounds=1,seconds=10.)
    archived=[]
    def consume(index,attempt):
        assert attempt['result']['program'].parameters>0
        archived.append((index,attempt['result']['certificate']))
    original=fit_product(template,metadata,contexts,group,group,sex,no_cost,ceiling)
    small=fit_product(template,metadata,contexts,group,group,sex,no_cost,ceiling,
                      solver_module=memory_lp,attempt_callback=consume)
    assert original['success'] and small['success']
    assert len(archived)==len(small['attempts'])
    assert all('program' not in attempt['result'] for attempt in small['attempts'])
    np.testing.assert_allclose(original['selected']['objective'],small['selected']['objective'],atol=1e-12)
    for first in (1,2):
        one=fit_sequential(template,metadata,contexts,sex,no_cost,first,ceiling)
        two=fit_sequential(template,metadata,contexts,sex,no_cost,first,ceiling,solver_module=memory_lp)
        assert one['success'] and two['success']
        np.testing.assert_allclose(one['objective'],two['objective'],atol=1e-12)


def test_wide_binary_low_memory_builder_preserves_true_one_hot_optimum():
    template,metadata,contexts=fixture();sex=np.array([0,1,0],np.int64)
    one=solve_binary_control(template,sex,no_cost(template),ControlCeilings(seconds=10.))
    two=solve_binary_control(template,sex,no_cost(template),ControlCeilings(seconds=10.),
                             protection_program_builder=memory_lp.protection_program)
    assert one['success'] and two['success']
    assert np.isin(two['theta'],[0.,1.]).all()
    np.testing.assert_allclose(one['certificate']['primal_objective'],two['certificate']['primal_objective'],atol=1e-12)
