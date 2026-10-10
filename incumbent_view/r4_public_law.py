"""Public-law feature under a fixed uniform *original-code-pair* prior.

This is a reader feature, not an asserted provider-input distribution. The
prior is deliberately data-free and enumerates each possible pair of original
P0 public global IDs once. It uses no labels or private per-person inputs.
"""
import numpy as np


def uniform_output_masses(bank1,bank2,policy=None):
    left,right=np.asarray(bank1),np.asarray(bank2)
    if (left.dtype!=np.float64 or right.dtype!=np.float64 or
            left.ndim!=2 or left.shape[1]!=2 or right.ndim!=2 or right.shape[1]!=6 or
            not np.isfinite(left).all() or not np.isfinite(right).all()):
        raise ValueError("exact two P0 public probability banks required")
    n1,n2=len(left),len(right)
    source1=np.repeat(np.arange(n1,dtype=np.int64),n2)
    source2=np.tile(np.arange(n2,dtype=np.int64),n1)
    if policy is None:
        action1=source1[:,None]
        action2=source2[:,None]
        mass=np.ones((len(source1),1),dtype=np.float64)
    else:
        for r,bank in ((1,left),(2,right)):
            if not np.array_equal(getattr(policy.space,f"probability{r}"),bank):
                raise ValueError("public policy decoder table/order differs from original bank")
        prepared=policy.space.prepare(source1,source2)
        mass=policy.law_prepared(prepared)
        action1,action2=prepared["code_r1"],prepared["code_r2"]
        if (mass.dtype!=np.float64 or action1.dtype!=np.int64 or action2.dtype!=np.int64 or
                mass.shape!=action1.shape or mass.shape!=action2.shape or
                not np.isfinite(mass).all() or np.any(mass<0) or
                np.any(action1<0) or np.any(action1>=n1) or
                np.any(action2<0) or np.any(action2>=n2) or
                not np.allclose(mass.sum(1),1.,atol=1e-12,rtol=0)):
            raise ValueError("public policy law over original ID space invalid")
    joint=np.zeros((n1,n2),np.float64)
    np.add.at(joint,(action1.ravel(),action2.ravel()),mass.ravel()/len(source1))
    if not np.isfinite(joint).all() or np.any(joint<0) or not np.isclose(joint.sum(),1,atol=1e-10,rtol=0):
        raise ValueError("uniform public prior output law invalid")
    return {"r1":joint.sum(1),"r2":joint.sum(0),"pair":joint,
            "source_prior":"uniform_over_all_original_P0_global_code_pairs",
            "actual_public_law":policy is not None}


def row_log_mass(masses,view,code,other_code=None):
    a=np.asarray(code)
    if a.dtype!=np.int64 or a.ndim!=1:
        raise ValueError("observed public global code ID differs")
    if view=="r1":
        if other_code is not None or np.any(a<0) or np.any(a>=len(masses["r1"])):
            raise ValueError("recipient1 feature needs only its own code")
        v=masses["r1"][a]
    elif view=="r2":
        if other_code is not None or np.any(a<0) or np.any(a>=len(masses["r2"])):
            raise ValueError("recipient2 feature needs only its own code")
        v=masses["r2"][a]
    elif view=="pair":
        b=np.asarray(other_code)
        if (b.dtype!=np.int64 or b.shape!=a.shape or np.any(a<0) or
                np.any(a>=len(masses["r1"])) or np.any(b<0) or np.any(b>=len(masses["r2"]))):
            raise ValueError("aligned coalition feature requires both public codes")
        v=masses["pair"][a,b]
    else:raise ValueError("unknown public view")
    return np.log(np.maximum(v,1e-12))[:,None]
