from fractions import Fraction
import numpy as np
import pytest
from experiments.pcrl_task_directed_release_v1.zero_certificate import certify


def test_exact_supported_nonconstant_witness_with_absent_child_and_parent():
    law=np.array([[[9,1,1,0,0]],[[1,9,9,0,0]]])
    report,q=certify({'A/SEX/U':law},[10,10,10,0,0],[0,1,2,0,3])
    assert report['exact_rank']==1 and report['exact_nullity']==2
    assert report['nonconstant_supported_kernel_exists']
    assert q[3]==q[0] and q[4]==['1','0']
    assert all(sum(Fraction(v) for v in row)==1 for row in q)
    assert report['maximum_independent_float_cmi']<1e-14


def test_absent_columns_do_not_make_constant_system_nonconstant():
    law=np.zeros((3,1,5),int);law[:,0,:3]=np.eye(3,dtype=int)
    report,q=certify({'A/S/U':law},[1,1,1,0,0],[0,1,2,0,3])
    assert report['exact_rank']==2 and report['exact_nullity']==1
    assert report['constant_only_on_supported_states'] and q is None


def test_noninteger_mass_cannot_be_called_exact():
    with pytest.raises(ValueError,match='integer cell masses'):
        certify({'A/S/W':np.array([[[.5,1]],[[1,.5]]])},[1,1],[0,1])


@pytest.mark.parametrize('law,parents,actions',[
    (np.zeros((2,1,3),int),[0,1,2],2),
    (np.zeros((0,1,3),int),[0,1,2],2),
    (np.ones((2,1,3),int),[0,np.inf,np.inf],2),
    (np.ones((2,1,3),int),[0,1,2],2.5),
])
def test_invalid_certificate_schema_rejected_at_entry(law,parents,actions):
    with pytest.raises(ValueError):certify({'role':law},[1,1,1],parents,n_actions=actions)


def test_empty_law_cannot_receive_even_constant_certificate():
    with pytest.raises(ValueError,match='positive observed mass'):
        certify({'role':np.zeros((2,1,1),int)},[1],[0])


@pytest.mark.parametrize('actions,zero',[(1,0),(3,2)])
def test_certificate_reports_actual_action_schema_including_single_action_exception(actions,zero):
    report,q=certify({'role':np.ones((2,1,3),int)},[2,2,2],[0,1,2],n_actions=actions,zero_action=zero)
    assert report['action_count']==actions and report['zero_action']==zero
    assert report['exact_nullity']==3
    assert report['constant_only_on_supported_states']==(actions==1)
    assert (q is None)==(actions==1)
