"""Frozen original P0 code space and pure law/float64 cost arithmetic.

This module has no scientific loader and does not open role files.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
import numpy as np

CLIP = 1e-12
SAFETY = 1e-10
CAPS = {"accuracy": .002, "nll": .001, "brier": .0005}
INITIAL_ETA = tuple(.05 * .5**k for k in range(41)) + (0.,)
ETA = tuple(.5**k for k in range(41)) + (0.,)
FOLD_SALT = "IVR1|FIT-crossfold|20261010|"
_PREPROCESS={"group_fold_calls":0,"group_fold_rows":0,"prepared_calls":0,
             "prepared_rows":0,"prepared_action_positions":0,
             "cost_matrix_calls":0,"cost_matrix_rows":0,"cost_matrix_action_positions":0}


def reset_preprocess_counts():
    for key in _PREPROCESS:_PREPROCESS[key]=0


def preprocess_counts():return dict(_PREPROCESS)


def group_folds(unit):
    values = np.asarray(unit)
    if values.ndim != 1 or values.dtype.kind not in "iuU":
        raise ValueError("group IDs must be exact one-dimensional integer/string IDs")
    out=np.asarray([int(hashlib.sha256((FOLD_SALT+str(v)).encode()).hexdigest(),16)%3
                    for v in values], dtype=np.int64)
    _PREPROCESS["group_fold_calls"]+=1;_PREPROCESS["group_fold_rows"]+=len(values)
    return out


def summaries(p):
    p=np.asarray(p,dtype=np.float64)
    ent=-(p*np.log(np.clip(p,CLIP,1))).sum(axis=-1)
    s=np.sort(p,axis=-1)
    return np.stack((ent,s[...,-1]-s[...,-2]),axis=-1)


def central_features(p1,p2):
    return np.concatenate((p1,p2,summaries(p1),summaries(p2)),axis=-1)


@dataclass(frozen=True)
class Space:
    probability1: np.ndarray
    probability2: np.ndarray
    actions1: np.ndarray
    actions2: np.ndarray
    valid1: np.ndarray
    valid2: np.ndarray

    def __post_init__(self):
        for r in (1,2):
            p=np.asarray(getattr(self,f"probability{r}"))
            a=np.asarray(getattr(self,f"actions{r}"))
            v=np.asarray(getattr(self,f"valid{r}"))
            if (p.dtype!=np.float64 or p.ndim!=2 or p.shape[1] not in (2,6)
                or not np.isfinite(p).all() or np.any(p<0)
                or not np.allclose(p.sum(1),1,atol=1e-12,rtol=0)
                or a.dtype!=np.int64 or a.shape!=(len(p),4)
                or v.dtype!=bool or v.shape!=a.shape or not np.all(v[:,0])
                or not np.array_equal(a[:,0],np.arange(len(p)))
                or np.any(a<0) or np.any(a>=len(p))):
                raise ValueError("original-ID P0 table/support contract differs")
            for row in range(len(p)):
                if len(np.unique(a[row,v[row]]))!=v[row].sum():
                    raise ValueError("support duplicates public action")
                if np.any(p[a[row,v[row]]].argmax(1)!=p[row].argmax()):
                    raise ValueError("support changed P0 decision")
            for name,x in ((f"probability{r}",p),(f"actions{r}",a),(f"valid{r}",v)):
                x=x.copy(); x.setflags(write=False); object.__setattr__(self,name,x)

    def prepare(self,code1,code2):
        c1,c2=np.asarray(code1),np.asarray(code2)
        if c1.dtype!=np.int64 or c2.dtype!=np.int64 or c1.ndim!=1 or c2.shape!=c1.shape:
            raise ValueError("private P0 codes not aligned int64")
        if np.any(c1<0) or np.any(c1>=len(self.probability1)) or np.any(c2<0) or np.any(c2>=len(self.probability2)):
            raise ValueError("private P0 ID outside bank")
        a1=self.actions1[c1]; a2=self.actions2[c2]
        code_r1=np.repeat(a1,4,axis=1); code_r2=np.tile(a2,(1,4))
        valid=np.repeat(self.valid1[c1],4,axis=1)&np.tile(self.valid2[c2],(1,4))
        _PREPROCESS["prepared_calls"]+=1;_PREPROCESS["prepared_rows"]+=len(c1)
        _PREPROCESS["prepared_action_positions"]+=valid.size
        return {"code_r1":code_r1,"code_r2":code_r2,"valid":valid,
                "b0_code_r1":c1.copy(),"b0_code_r2":c2.copy(),
                "features":central_features(self.probability1[c1],self.probability2[c2])}


def identity_q(prepared):
    q=np.zeros(prepared["valid"].shape,dtype=np.float64);q[:,0]=1
    return q


def check_law(prepared,q):
    q=np.asarray(q)
    if q.dtype!=np.float64 or q.shape!=prepared["valid"].shape or not np.isfinite(q).all() or np.any(q<0) or np.any(q[~prepared["valid"]]!=0) or not np.allclose(q.sum(1),1,atol=1e-12,rtol=0):
        raise ValueError("actual finite action mass invalid")
    return q


def costs(space,prepared,y1,y2):
    out={}
    for r,y in ((1,y1),(2,y2)):
        p=getattr(space,f"probability{r}")[prepared[f"code_r{r}"]]
        y=np.asarray(y)
        if y.dtype!=np.int64 or y.shape!=(len(p),) or np.any(y<0) or np.any(y>=p.shape[-1]):
            raise ValueError("task labels not aligned admitted class order")
        target=np.eye(p.shape[-1])[y]
        out[f"r{r}_nll"]=-np.log(np.clip(np.take_along_axis(p,y[:,None,None],axis=2).squeeze(2),CLIP,1))
        out[f"r{r}_brier"]=np.sum((p-target[:,None,:])**2,axis=2)
        out[f"r{r}_accuracy"]=(p.argmax(2)==y[:,None]).astype(np.float64)
    _PREPROCESS["cost_matrix_calls"]+=1;_PREPROCESS["cost_matrix_rows"]+=len(prepared["code_r1"])
    _PREPROCESS["cost_matrix_action_positions"]+=prepared["code_r1"].size
    return out


def accurate_margin(q,cost,key,mask=None):
    mask=np.ones(len(q),bool) if mask is None else np.asarray(mask,dtype=bool)
    # Subtract identity row costs BEFORE summation; avoids cancellation of means.
    diff=cost[mask]-cost[mask,0,None]
    sign=-1 if key.endswith("accuracy") else 1
    row=np.sum(q[mask]*diff,axis=1,dtype=np.float64)*sign
    value=math.fsum(float(v) for v in row)/len(row)
    high=np.sum(np.asarray(q[mask],dtype=np.longdouble)*np.asarray(diff,dtype=np.longdouble),axis=1)*sign
    hp=float(np.sum(high,dtype=np.longdouble)/len(high))
    return value,abs(value-hp)


def utility_receipt(q,costs_by_key,fold):
    result={}; feasible=True
    for group,mask in (("all",np.ones(len(q),bool)),*((str(k),fold==k) for k in range(3))):
        if not mask.any():raise ValueError("empty internal group fold")
        for key,c in costs_by_key.items():
            margin,error=accurate_margin(q,c,key,mask)
            cap=CAPS[key.split("_")[-1]]
            identity=np.all(q[:,0]==1.)
            passed=margin<=cap-(0. if identity else SAFETY) and error<=SAFETY
            result[f"{group}/{key}"]={"margin":margin,"cap":cap,"numeric_error":error,"slack":cap-margin,"feasible":bool(passed)}
            feasible &= passed
    return {"feasible":bool(feasible),"clauses":result,"numeric_safety":SAFETY}


def bayes_objective(prepared,q,sex):
    sex=np.asarray(sex,dtype=np.int64);n=len(q)
    vals=[]
    z2=int(prepared["code_r2"].max())+1
    for key in (prepared["code_r1"],prepared["code_r2"],prepared["code_r1"]*z2+prepared["code_r2"]):
        unique,inverse=np.unique(key,return_inverse=True)
        count=np.stack([np.bincount(inverse.ravel(),weights=(q*(sex==s)[:,None]/n).ravel(),minlength=len(unique)) for s in (0,1)])
        vals.append(float(count.max(0).sum()))
    return vals[2]+.5*(vals[0]+vals[1]),{"r1":vals[0],"r2":vals[1],"pair":vals[2]}


def decimal_cost_check(space,prepared,q,y1,y2):
    """Independent 50-digit cost construction from exact float inputs.

    Decimal logs/Brier avoid relying on longdouble's platform mantissa. This
    is a numerical engineering replay, separate from statistical uncertainty.
    """
    from decimal import Decimal,localcontext
    out={}
    with localcontext() as context:
        context.prec=50
        for r,y in ((1,y1),(2,y2)):
            p=getattr(space,f"probability{r}")
            table={}
            for code,vector in enumerate(p):
                v=[Decimal.from_float(float(x)) for x in vector]
                for label in range(p.shape[1]):
                    nll=-max(v[label],Decimal("1e-12")).ln()
                    brier=sum((x-Decimal(int(j==label)))**2 for j,x in enumerate(v))
                    table[code,label]=(nll,brier,Decimal(int(vector.argmax()==label)))
            for name,index in (("nll",0),("brier",1),("accuracy",2)):
                total=Decimal(0)
                for row in range(len(q)):
                    identity=table[int(prepared[f"b0_code_r{r}"][row]),int(y[row])][index]
                    for slot in np.flatnonzero(q[row]):
                        value=table[int(prepared[f"code_r{r}"][row,slot]),int(y[row])][index]-identity
                        total+=Decimal.from_float(float(q[row,slot]))*value
                out[f"r{r}_{name}"]=float(total/Decimal(len(q)))*(-1 if name=="accuracy" else 1)
    return out
