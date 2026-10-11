"""Data-free ragged-axis tests; optimizer laws remain unchanged."""
import numpy as np


def fixture():
    from conditional_release.test_controls import _space
    from conditional_release.templates import build_template_law
    ids=np.array([0,1,2],np.int64)
    contexts=np.array([0,1,-1],np.int64)
    template,metadata=build_template_law(_space(),ids,ids,contexts,"WIDE",10_000_000)
    return template,metadata,contexts


def test_full_bank_product_map_preserves_padding_and_identity_fallback():
    from conditional_release.wide_control_fit import coordinate_law
    template,metadata,contexts=fixture()
    group=np.array([0,1,0],np.int64)
    other=np.array([[.3,.7,0,0,0],[.5,.5,0,0,0],[1.,0,0,0,0]])
    own=np.array([[.2,.8,0,0,0],[.4,.6,0,0,0]])
    law=coordinate_law(template,metadata,contexts,group,other,1)
    q=law.probabilities(own.ravel()).reshape(3,5,5)
    np.testing.assert_allclose(q[:2],own[:, :,None]*other[:2,None,:])
    assert q[2,0,0]==1 and np.count_nonzero(q[2])==1
    assert law.bounds[1,1]==1 # Unsupported row must not force context0 KEEP.


def test_full_bank_second_stage_preserves_first_marginal():
    from conditional_release.wide_control_fit import sequential_second_law
    template,metadata,contexts=fixture()
    first=np.array([[.3,.7,0,0,0],[.5,.5,0,0,0],[1.,0,0,0,0]])
    groups=np.repeat(np.array([0,1,0],np.int64)[:,None],5,axis=1)
    later=np.array([[.2,.8,0,0,0],[.4,.6,0,0,0]])
    for recipient in (1,2):
        law=sequential_second_law(template,metadata,contexts,groups,first,recipient)
        q=law.probabilities(later.ravel()).reshape(3,5,5)
        np.testing.assert_allclose(q[:2].sum(2 if recipient==1 else 1),first[:2])
        assert q[2,0,0]==1 and np.count_nonzero(q[2])==1
