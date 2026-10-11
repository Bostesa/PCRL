"""Group-cross-fitted weighted actual-code categorical and 64/32 readers.

All arrays supplied by an admitted worker. No role loader or final attacker is
imported. Fitted losses are on actual emitted states, not mean probabilities.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .r2_space import CLIP,SAFETY,check_law,identity_q

READER_UPDATES=50
READER_LR=.001
READER_ALPHA=1e-4
VIEWS=("r1","r2","pair")


def state_ids(space,prepared,view):
    if view=="r1":return prepared["code_r1"],len(space.probability1)
    if view=="r2":return prepared["code_r2"],len(space.probability2)
    if view=="pair":return prepared["code_r1"]*len(space.probability2)+prepared["code_r2"],len(space.probability1)*len(space.probability2)
    raise ValueError("unknown actual-wire view")


def wire_features(space,view,state):
    """Categorical identity + EVERY varying field of source P0 public wire.

    Public recipient/release/model/source/bank identifiers and lengths are
    constants bound by the package receipt. Decision/prob/code are deterministic
    functions of original global code and included explicitly here.
    """
    state=np.asarray(state,dtype=np.int64)
    parts=[]
    for r in ((1,2) if view=="pair" else (int(view[-1]),)):
        p=getattr(space,f"probability{r}"); z=len(p)
        code=(state//len(space.probability2) if r==1 else state%len(space.probability2)) if view=="pair" else state
        bits=(z-1).bit_length()
        parts.extend((np.eye(z,dtype=np.float64)[code],p[code],np.eye(p.shape[1])[p[code].argmax(1)],
                      ((code[:,None]>>np.arange(bits))&1).astype(np.float64)))
    return np.ascontiguousarray(np.concatenate(parts,axis=1))


def weighted_counts(ids,width,q,sex,mask):
    return np.stack([np.bincount(ids[mask].ravel(),weights=(q[mask]*(sex[mask]==s)[:,None]).ravel(),minlength=width)
                     for s in (0,1)],axis=1)


def categorical_probability(count,prior):
    total=count.sum(1)
    return np.where(total>0,(count[:,1]+1)/(total+2),prior)


def _fit_nonlinear(feature,count,seed):
    import torch
    torch.set_num_threads(1)
    active=count.sum(1)>0
    if not active.any():raise ValueError("reader has no weighted training actions")
    x=feature[active];c=count[active]
    weight=c.sum(1);target=c[:,1]/weight
    # Actual weighted StandardScaler counterpart, frozen after fitting.
    mean=np.average(x,axis=0,weights=weight)
    scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=weight));scale[scale<1e-12]=1
    x=(x-mean)/scale
    rng=np.random.default_rng(seed)
    shapes=((feature.shape[1],64),(64,),(64,32),(32,),(32,1),(1,))
    values=[rng.normal(0, .02, size=s) if len(s)==2 else np.zeros(s) for s in shapes]
    w=[torch.nn.Parameter(torch.as_tensor(a,dtype=torch.float64)) for a in values]
    opt=torch.optim.Adam(w,lr=READER_LR)
    tx=torch.as_tensor(x);tw=torch.as_tensor(weight);ty=torch.as_tensor(target)
    for _ in range(READER_UPDATES):
        opt.zero_grad()
        h=torch.relu(tx@w[0]+w[1]);h=torch.relu(h@w[2]+w[3]);logit=(h@w[4]+w[5]).squeeze(1)
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(logit,ty,reduction="none")*tw).sum()/tw.sum()
        loss=loss+READER_ALPHA*.5*sum(a.square().sum() for a in (w[0],w[2],w[4]))/tw.sum()
        loss.backward();opt.step()
    arrays=[a.detach().numpy().copy() for a in w]
    xx=(feature-mean)/scale
    h=np.maximum(xx@arrays[0]+arrays[1],0);h=np.maximum(h@arrays[2]+arrays[3],0)
    logit=(h@arrays[4]+arrays[5]).ravel()
    # Stable sigmoid without evaluating overflowing unused branch.
    prediction=np.empty(len(logit));pos=logit>=0
    prediction[pos]=1/(1+np.exp(-logit[pos]));e=np.exp(logit[~pos]);prediction[~pos]=e/(1+e)
    return prediction,{"mean":mean,"scale":scale,**{f"w{k}":a for k,a in enumerate(arrays)}}


@dataclass
class Reader:
    name:str
    view:str
    loss:np.ndarray
    baseline:float
    models:dict
    round_index:int
    start_index:int


def fit_round(space,prepared,q,sex,fold,*,round_index,start_index,seed,views=VIEWS):
    q=check_law(prepared,q);sex=np.asarray(sex,dtype=np.int64)
    result=[]
    for v_index,view in enumerate(views):
        ids,width=state_ids(space,prepared,view)
        feature=wire_features(space,view,np.arange(width,dtype=np.int64))
        losses={family:np.zeros(q.shape,dtype=np.float64) for family in ("prior","categorical","nonlinear")}
        models={family:{} for family in losses}
        for k in range(3):
            train=fold!=k;test=fold==k
            if not train.any() or not test.any():raise ValueError("reader cross-fit empty fold")
            count=weighted_counts(ids,width,q,sex,train)
            prior=float((sex[train].sum()+1)/(train.sum()+2))
            cat=categorical_probability(count,prior)
            nonlinear,model=_fit_nonlinear(feature,count,seed+10000*start_index+1000*round_index+100*v_index+k)
            probabilities={"prior":np.full(width,prior),"categorical":cat,"nonlinear":nonlinear}
            for family,p in probabilities.items():
                p=np.clip(p[ids[test]],CLIP,1-CLIP)
                losses[family][test]=np.where(sex[test,None]==1,-np.log(p),-np.log1p(-p))
            models["prior"][f"f{k}"]=np.asarray([prior])
            models["categorical"][f"f{k}"]=count
            models["nonlinear"][f"f{k}"]=model
        for family,loss in losses.items():
            result.append(Reader(f"s{start_index}r{round_index}/{view}/{family}",view,loss,
                                 float(loss[:,0].mean()),models[family],round_index,start_index))
    return result


def risks(q,readers,view):
    return {r.name:float(np.mean(np.sum(q*r.loss,axis=1))) for r in readers if r.view==view}


def reader_guard_receipt(q,readers,fold,views=("r1","r2")):
    out={};feasible=True
    for reader in readers:
        if reader.view not in views:continue
        for k,mask in (("all",np.ones(len(q),bool)),*((str(j),fold==j) for j in range(3))):
            diff=np.sum(q[mask]*(reader.loss[mask]-reader.loss[mask,0,None]),axis=1)
            # Unlike utility, guards permit ONLY the predeclared numeric tolerance.
            import math
            excess=math.fsum(float(x) for x in diff)/len(diff)
            pass_=excess>=-SAFETY
            out[f"{reader.name}/{k}"]={"ce_difference_vs_p0":excess,"feasible":bool(pass_)}
            feasible&=pass_
    return {"feasible":bool(feasible),"clauses":out,"numeric_tolerance":SAFETY}


def nomination(q,readers):
    pair=risks(q,readers,"pair")
    constant=[v for key,v in pair.items() if key.endswith("/prior")]
    if not pair or not constant:raise ValueError("nomination missing pair/prior readers")
    strongest=min(pair.values());prior=min(constant)
    return {"pair_min_ce":strongest,"constant_prior_ce":prior,
            "reader_recovery_gap":prior-strongest,"pair_reader_losses":pair}
