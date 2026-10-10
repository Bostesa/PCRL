"""Independent finite policy-law reconstruction from frozen public bank and weights.

This module imports no R2 policy, space, optimizer, endpoint or selection code.
It is used by R5 only after a selected package has been frozen and hash-pinned.
"""
from __future__ import annotations

import numpy as np

ARMS=("P0","O_J","V_J","V_P","V_D","V_L","V_S12","V_S21")


def _summary(prob):
    entropy=-(prob*np.log(np.maximum(prob,1e-12))).sum(axis=-1)
    ranked=np.sort(prob,axis=-1)
    return np.stack((entropy,ranked[...,-1]-ranked[...,-2]),axis=-1)


def _features(left,right):
    return np.concatenate((left,right,_summary(left),_summary(right)),axis=-1)


def _gate(features,weights):
    w0,b0,w1,b1=weights
    return np.tanh(features@w0+b0)@w1+b1


def _simplex(logits,valid):
    masked=np.where(valid,logits,-np.inf)
    exponent=np.exp(masked-masked.max(axis=-1,keepdims=True))
    return exponent/exponent.sum(axis=-1,keepdims=True)


def _mix(prob,eta):
    out=eta*prob
    out[...,0]+=1-eta
    return out


def frozen_space(bank,package):
    """Check package tables/support against original global IDs and return arrays."""
    tables=[];actions=[];valid=[]
    for recipient in (1,2):
        public=bank["recipients"][str(recipient)]
        table=np.asarray(public["probability"],np.float64)
        if public["global_ids"]!=list(range(len(table))):
            raise PermissionError("original global IDs were renumbered")
        action=np.repeat(np.arange(len(table),dtype=np.int64)[:,None],4,axis=1)
        mask=np.zeros(action.shape,dtype=bool)
        for source,row in enumerate(public["support"]):
            if not row or len(row)>4 or row[0]!=source:
                raise PermissionError("frozen P0 support differs")
            action[source,:len(row)]=row;mask[source,:len(row)]=True
        for name,value in ((f"probability{recipient}",table),
                           (f"actions{recipient}",action),(f"valid{recipient}",mask)):
            if not np.array_equal(package[name],value):
                raise PermissionError("selected policy package changed original public bank/support")
        tables.append(table);actions.append(action);valid.append(mask)
    return tables,actions,valid


def reconstructed_law(bank,package,receipt,source_code1,source_code2):
    """Rebuild the 16-action law using only exact frozen package arrays.

    Returns code arrays, valid positions, and masses. This is an independent
    deployment calculation, not a claim about which checkpoint should win.
    """
    arm=receipt["policy_arm"]
    if arm not in ARMS:
        raise ValueError("selected family differs")
    tables,actions,valid=frozen_space(bank,package)
    c1=np.asarray(source_code1);c2=np.asarray(source_code2)
    if (c1.dtype!=np.int64 or c2.dtype!=np.int64 or c1.ndim!=1 or c2.shape!=c1.shape
            or np.any(c1<0) or np.any(c1>=len(tables[0]))
            or np.any(c2<0) or np.any(c2>=len(tables[1]))):
        raise ValueError("private original P0 code input differs")
    a1=actions[0][c1];a2=actions[1][c2]
    code1=np.repeat(a1,4,axis=1);code2=np.tile(a2,(1,4))
    allowed=np.repeat(valid[0][c1],4,axis=1)&np.tile(valid[1][c2],(1,4))
    weights=tuple(tuple(package[f"gate{i}_{j}"] for j in range(4))
                  for i in range(receipt["gate_count"]))
    eta1=float(receipt["eta1"]);eta2=float(receipt["eta2"])
    if not (0<=eta1<=1 and 0<=eta2<=1):
        raise ValueError("selected identity backoff differs")
    p1=tables[0][c1];p2=tables[1][c2]
    if arm=="P0":
        if weights or eta1!=0 or eta2!=0:
            raise PermissionError("exact P0 fallback package contains a fitted gate")
        mass=np.zeros(allowed.shape,np.float64);mass[:,0]=1.
    elif arm in ("O_J","V_J","V_P","V_D"):
        soft=_simplex(_gate(_features(p1,p2),weights[0]),allowed)
        if arm=="V_D":
            mass=np.zeros_like(soft);mass[np.arange(len(soft)),soft.argmax(1)]=1.
        elif arm=="V_P":
            pair=soft.reshape(-1,4,4)
            left=_mix(pair.sum(2),eta1);right=_mix(pair.sum(1),eta2)
            mass=(left[:,:,None]*right[:,None,:]).reshape(-1,16)
        else:
            mass=eta1*soft;mass[:,0]+=1-eta1
    elif arm=="V_L":
        left=_simplex(_gate(np.c_[p1,_summary(p1)],weights[0]),valid[0][c1])
        right=_simplex(_gate(np.c_[p2,_summary(p2)],weights[1]),valid[1][c2])
        mass=(_mix(left,eta1)[:,:,None]*_mix(right,eta2)[:,None,:]).reshape(-1,16)
    else:
        first=0 if arm=="V_S12" else 1
        source=c1 if first==0 else c2
        first_mass=_mix(_simplex(_gate(_features(p1,p2),weights[0]),valid[first][source]),eta1)
        tensor=np.zeros((len(c1),4,4),np.float64)
        later=1-first
        for slot in range(4):
            emitted=tables[first][actions[first][source,slot]]
            input1,input2=(emitted,p2) if first==0 else (p1,emitted)
            conditional=_mix(_simplex(_gate(_features(input1,input2),weights[1]),
                                        valid[later][c2 if later==1 else c1]),eta2)
            if first==0:tensor[:,slot,:]=first_mass[:,slot,None]*conditional
            else:tensor[:,:,slot]=first_mass[:,slot,None]*conditional
        mass=tensor.reshape(-1,16)
    if (mass.shape!=allowed.shape or not np.isfinite(mass).all() or np.any(mass<0)
            or np.any(mass[~allowed]!=0) or not np.allclose(mass.sum(1),1,atol=1e-12,rtol=0)):
        raise ArithmeticError("independent reconstructed public law invalid")
    return {"code_r1":code1,"code_r2":code2,"valid":allowed,"mass":mass,
            "b0_code_r1":c1,"b0_code_r2":c2}
