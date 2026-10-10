"""Executable incumbent-code-only family laws with original public IDs."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .r2_space import Space,central_features,summaries,identity_q,check_law

ARMS=("P0","O_J","V_J","V_P","V_D","V_L","V_S12","V_S21")
_FORWARDS={"numpy_policy_forward_calls":0,"numpy_policy_forward_rows":0,
           "numpy_policy_forward_action_positions":0}


def reset_forward_counts():
    for key in _FORWARDS:_FORWARDS[key]=0


def forward_counts():return dict(_FORWARDS)


def logits(x,w):
    return np.tanh(x@w[0]+w[1])@w[2]+w[3]


def softmax(x,valid):
    x=np.where(valid,x,-np.inf);x=np.exp(x-np.max(x,axis=-1,keepdims=True))
    return x/x.sum(axis=-1,keepdims=True)


def factor_mix(q,eta):
    out=q*eta;out[...,0]+=1-eta;return out


@dataclass
class Policy:
    space:Space
    arm:str
    weights:tuple
    eta1:float=0.
    eta2:float=0.

    def __post_init__(self):
        if self.arm not in ARMS or not 0<=self.eta1<=1 or not 0<=self.eta2<=1:raise ValueError("policy family/boundary invalid")
        self.weights=tuple(tuple(np.asarray(a,dtype=np.float64).copy() for a in w) for w in self.weights)
        if self.arm!="P0" and not self.weights:raise ValueError("fitted policy missing weights")

    @property
    def parameter_count(self):return sum(a.size for w in self.weights for a in w)

    @property
    def law_kind(self):
        return {"P0":"incumbent_identity_v1","O_J":"incumbent_joint_v1","V_J":"incumbent_joint_v1",
                "V_P":"incumbent_product_v1","V_L":"incumbent_local_v1","V_D":"incumbent_deterministic_v1",
                "V_S12":"incumbent_seq12_v1","V_S21":"incumbent_seq21_v1"}[self.arm]

    def sample_public_pair(self,prob1,prob2,code1,code2,uniform=None):
        if (not np.array_equal(np.asarray(prob1),self.space.probability1[int(code1)]) or
            not np.array_equal(np.asarray(prob2),self.space.probability2[int(code2)])):
            raise ValueError("private P0 decoded values differ from fixed source table")
        u=None if uniform is None else np.asarray([uniform],dtype=np.float64)
        a,b=self.sample_codes(np.asarray([code1],np.int64),np.asarray([code2],np.int64),u)
        return int(a[0]),int(b[0])

    def sparse_law(self,prob1,prob2,code1,code2):
        if (not np.array_equal(np.asarray(prob1),self.space.probability1[int(code1)]) or
            not np.array_equal(np.asarray(prob2),self.space.probability2[int(code2)])):
            raise ValueError("private P0 probability input differs")
        p=self.space.prepare(np.asarray([code1],np.int64),np.asarray([code2],np.int64))
        q=self.law_prepared(p)[0]
        return tuple((int(p["code_r1"][0,s]),int(p["code_r2"][0,s]),float(q[s])) for s in np.flatnonzero(q))

    def law(self,c1,c2):
        p=self.space.prepare(c1,c2);return self.law_prepared(p)

    def law_prepared(self,p):
        _FORWARDS["numpy_policy_forward_calls"]+=1
        _FORWARDS["numpy_policy_forward_rows"]+=len(p["code_r1"])
        _FORWARDS["numpy_policy_forward_action_positions"]+=p["code_r1"].size
        if self.arm=="P0":return identity_q(p)
        if self.arm in ("O_J","V_J","V_P","V_D"):
            source=softmax(logits(p["features"],self.weights[0]),p["valid"])
            if self.arm=="V_D":
                q=np.zeros_like(source);q[np.arange(len(q)),source.argmax(1)]=1.
            elif self.arm=="V_P":
                pair=source.reshape(-1,4,4)
                a=factor_mix(pair.sum(2),self.eta1);b=factor_mix(pair.sum(1),self.eta2)
                q=(a[:,:,None]*b[:,None,:]).reshape(-1,16)
            else:
                q=self.eta1*source;q[:,0]+=1-self.eta1
            return check_law(p,q)
        if self.arm=="V_L":
            p1=self.space.probability1[p["b0_code_r1"]];p2=self.space.probability2[p["b0_code_r2"]]
            a=softmax(logits(np.c_[p1,summaries(p1)],self.weights[0]),self.space.valid1[p["b0_code_r1"]])
            b=softmax(logits(np.c_[p2,summaries(p2)],self.weights[1]),self.space.valid2[p["b0_code_r2"]])
            return check_law(p,(factor_mix(a,self.eta1)[:,:,None]*factor_mix(b,self.eta2)[:,None,:]).reshape(-1,16))
        first=1 if self.arm=="V_S12" else 2
        c=p[f"b0_code_r{first}"];v=getattr(self.space,f"valid{first}")[c]
        a=factor_mix(softmax(logits(p["features"],self.weights[0]),v),self.eta1)
        q=np.zeros((len(c),4,4),dtype=np.float64)
        src1=self.space.probability1[p["b0_code_r1"]];src2=self.space.probability2[p["b0_code_r2"]]
        for i in range(4):
            emitted=getattr(self.space,f"probability{first}")[getattr(self.space,f"actions{first}")[c,i]]
            # The first transmitted code changes probabilities, NEVER support.
            x=central_features(emitted,src2) if first==1 else central_features(src1,emitted)
            later=3-first;vl=getattr(self.space,f"valid{later}")[p[f"b0_code_r{later}"]]
            b=factor_mix(softmax(logits(x,self.weights[1]),vl),self.eta2)
            if first==1:q[:,i,:]=a[:,i,None]*b
            else:q[:,:,i]=a[:,i,None]*b
        return check_law(p,q.reshape(-1,16))

    def sample_codes(self,c1,c2,uniform=None):
        p=self.space.prepare(c1,c2);q=self.law_prepared(p);n=len(q)
        if self.arm in ("P0","V_D"):
            slot=q.argmax(1)
        elif self.arm in ("V_P","V_L","V_S12","V_S21"):
            u=np.asarray(uniform,dtype=np.float64)
            if u.shape!=(n,2) or np.any(u<0) or np.any(u>=1):raise ValueError("factor/seq requires two frozen private coins")
            pair=q.reshape(-1,4,4)
            if self.arm in ("V_P","V_L","V_S12"):
                ma=pair.sum(2);a=(np.cumsum(ma,axis=1)<=u[:,0,None]).sum(1).clip(0,3)
                # For P/L this conditional equals independent second marginal.
                mb=(pair.sum(1) if self.arm in ("V_P","V_L") else
                    pair[np.arange(n),a,:]/ma[np.arange(n),a,None])
                b=(np.cumsum(mb,axis=1)<=u[:,1,None]).sum(1).clip(0,3)
            else:
                mb=pair.sum(1);b=(np.cumsum(mb,axis=1)<=u[:,1,None]).sum(1).clip(0,3)
                ma=pair[np.arange(n),:,b]/mb[np.arange(n),b,None]
                a=(np.cumsum(ma,axis=1)<=u[:,0,None]).sum(1).clip(0,3)
            slot=4*a+b
        else:
            u=np.asarray(uniform,dtype=np.float64)
            if u.shape!=(n,) or np.any(u<0) or np.any(u>=1):raise ValueError("joint requires one frozen private coin")
            slot=(np.cumsum(q,axis=1)<=u[:,None]).sum(1).clip(0,15)
        # Protect inverse-CDF floating endpoint from an invalid padded slot.
        bad=q[np.arange(n),slot]<=0
        if bad.any():
            slot[bad]=np.asarray([np.flatnonzero(row>0)[-1] for row in q[bad]])
        return p["code_r1"][np.arange(n),slot],p["code_r2"][np.arange(n),slot]
