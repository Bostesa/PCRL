"""Bounded three-round actual-wire policy optimization, supplied arrays only.

No final attack, role loader, calibration or checkpoint search on INNER.
Approximate Adam/straight-through optimization; no global certificate.
"""
from __future__ import annotations
import time
import math
import numpy as np
from .r2_space import (ETA,INITIAL_ETA,SAFETY,CAPS,group_folds,costs,identity_q,
                      utility_receipt,bayes_objective,central_features,summaries)
from .r2_policy import Policy
from .r2_readers import fit_round,reader_guard_receipt,risks,nomination


def initial_weights(width,hidden,output,seed,start):
    rng=np.random.default_rng(seed+start)
    weights=(rng.normal(0,.01,(width,hidden)),rng.normal(0,.01,hidden),
             np.zeros((hidden,output)),np.zeros(output))
    if start==0:weights[-1][0]=4.
    return weights


def _snapshot(weights):
    return tuple(tuple(a.detach().numpy().copy() for a in w) for w in weights)


def _shape_weights(arm,seed,start):
    if arm=="V_L":return (initial_weights(4,16,4,seed,start),initial_weights(8,16,4,seed+20,start))
    if arm in ("V_S12","V_S21"):
        return (initial_weights(12,12,4,seed,start),initial_weights(12,12,4,seed+20,start))
    return (initial_weights(12,16,16,seed,start),)


def _torch_law(space,p,w,arm,eta1,eta2,*,torch,stage=None,frozen_first=None,temperature=1.):
    tx=torch.as_tensor(p["features"])
    valid=torch.as_tensor(p["valid"])
    def lg(x,ww):return torch.tanh(x@ww[0]+ww[1])@ww[2]+ww[3]
    def sm(x,v):return torch.softmax(x.masked_fill(~v,-torch.inf),dim=-1)
    def mix(q,eta):
        identity=torch.zeros_like(q);identity[...,0]=1
        return eta*q+(1-eta)*identity
    if arm in ("O_J","V_J","V_P","V_D"):
        raw=sm(lg(tx,w[0])/temperature,valid)
        if arm=="V_D":
            one=torch.nn.functional.one_hot(raw.argmax(1),16).to(torch.float64)
            return one+raw-raw.detach() # HARD forward; straight-through training
        if arm=="V_P":
            pair=raw.reshape(-1,4,4)
            return (mix(pair.sum(2),eta1)[:,:,None]*mix(pair.sum(1),eta2)[:,None,:]).reshape(-1,16)
        return mix(raw,eta1)
    if arm=="V_L":
        marg=[]
        for r in (1,2):
            pp=getattr(space,f"probability{r}")[p[f"b0_code_r{r}"]]
            x=torch.as_tensor(np.c_[pp,summaries(pp)])
            v=torch.as_tensor(getattr(space,f"valid{r}")[p[f"b0_code_r{r}"]])
            marg.append(mix(sm(lg(x,w[r-1]),v),eta1 if r==1 else eta2))
        return (marg[0][:,:,None]*marg[1][:,None,:]).reshape(-1,16)
    first=1 if arm=="V_S12" else 2;later=3-first
    vf=torch.as_tensor(getattr(space,f"valid{first}")[p[f"b0_code_r{first}"]])
    a=(mix(sm(lg(tx,w[0]),vf),eta1) if frozen_first is None else
       torch.as_tensor(frozen_first))
    src1=space.probability1[p["b0_code_r1"]];src2=space.probability2[p["b0_code_r2"]]
    vl=torch.as_tensor(getattr(space,f"valid{later}")[p[f"b0_code_r{later}"]])
    conditional=[]
    for i in range(4):
        emitted=getattr(space,f"probability{first}")[getattr(space,f"actions{first}")[p[f"b0_code_r{first}"],i]]
        x=central_features(emitted,src2) if first==1 else central_features(src1,emitted)
        b=(torch.zeros((len(a),4),dtype=torch.float64) if stage==1 else
           mix(sm(lg(torch.as_tensor(x),w[1]),vl),eta2))
        if stage==1:b[:,0]=1
        conditional.append(a[:,i,None]*b)
    pair=torch.stack(conditional,dim=1)
    if first==2:pair=pair.transpose(1,2)
    return pair.reshape(-1,16)


def _bayes_torch(p,q,sex,torch):
    n=len(q);result=[]
    z2=int(p["code_r2"].max())+1
    for ids in (p["code_r1"],p["code_r2"],p["code_r1"]*z2+p["code_r2"]):
        _,inv=np.unique(ids,return_inverse=True)
        ix=torch.as_tensor(inv.ravel());width=int(inv.max())+1
        count=[torch.zeros(width,dtype=torch.float64).scatter_add(0,ix,(q*torch.as_tensor((sex==s)[:,None])/n).ravel()) for s in (0,1)]
        result.append(torch.maximum(*count).sum())
    return result[2]+.5*(result[0]+result[1])


def _objective(q,p,sex,readers,arm,torch,*,stage=None):
    if arm=="O_J":return _bayes_torch(p,q,sex,torch)
    views=("r1","r2") if arm=="V_L" else (("r1" if arm=="V_S12" else "r2",) if stage==1 else ("pair",))
    objective=torch.zeros((),dtype=torch.float64)
    for v in views:
        values=[(q*torch.as_tensor(r.loss)).sum(1).mean() for r in readers if r.view==v]
        if not values:raise ValueError("optimizer missing retained actual-wire reader")
        objective-=torch.stack(values).min()
    return objective


def _constraints(q,cost,fold,readers,torch,*,task_views=(1,2),guard_views=("r1","r2")):
    result=[]
    for mask in (np.ones(len(q),bool),*(fold==k for k in range(3))):
        tq=q[torch.as_tensor(mask)]
        for key,c in cost.items():
            if int(key[1]) not in task_views:continue
            diff=torch.as_tensor((c-c[:,0,None])[mask])
            sign=-1 if key.endswith("accuracy") else 1
            cap=CAPS[key.split("_")[-1]]
            result.append(sign*(tq*diff).sum(1).mean()-cap+SAFETY)
        for reader in readers:
            if reader.view not in guard_views:continue
            diff=torch.as_tensor((reader.loss-reader.loss[:,0,None])[mask])
            result.append(-(tq*diff).sum(1).mean()-SAFETY)
    return result


def feasible(policy,p,cost,fold,readers,*,guard_views=("r1","r2"),task_views=(1,2)):
    q=policy.law_prepared(p)
    utility=utility_receipt(q,{k:v for k,v in cost.items() if int(k[1]) in task_views},fold)
    guards=reader_guard_receipt(q,readers,fold,guard_views)
    return utility["feasible"] and guards["feasible"],q,utility,guards


def backoff(space,p,arm,weights,cost,fold,readers,*,initial=False,first_eta=None,stage=None):
    from .r2_projection import law_coefficients,constraint_coefficients,polynomial_feasible
    schedule=INITIAL_ETA if initial else ETA
    if arm=="V_D":schedule=(1.,)
    first=1 if arm=="V_S12" else 2
    task_views=(first,) if stage==1 else (1,2)
    guard_views=(f"r{first}",) if stage==1 else ("r1","r2")
    def make(eta):
        if stage==1:policy=Policy(space,arm,weights,eta,0.)
        elif first_eta is not None:policy=Policy(space,arm,weights,first_eta,eta)
        else:policy=Policy(space,arm,weights,eta,eta)
        return policy
    coefficients=None
    if arm!="V_D":
        law=law_coefficients(*(make(eta).law_prepared(p) for eta in (0.,.5,1.)))
        coefficients=constraint_coefficients(law,cost,fold,readers,task_views,guard_views)
    for eta in schedule:
        # Never let a conservative polynomial prefilter discard exact identity.
        if eta!=0 and coefficients is not None and not polynomial_feasible(eta,coefficients):continue
        policy=make(eta)
        passed,q,u,g=feasible(policy,p,cost,fold,readers,task_views=task_views,guard_views=guard_views)
        if passed:return policy,q,{"utility":u,"guards":g,"eta1":policy.eta1,"eta2":policy.eta2}
    # Sequential second backoff must NOT change the frozen first recipient.
    # A whole-P0 fallback is separate and explicitly returned by caller.
    return None,None,{"dependency_failure":True,"stage":stage}


def _rank(policy,p,sex,readers,arm,stage=None):
    q=policy.law_prepared(p)
    if arm=="O_J":return bayes_objective(p,q,sex)[0]
    if arm=="V_L":return -sum(min(risks(q,readers,v).values()) for v in ("r1","r2"))
    view=("r1" if arm=="V_S12" else "r2") if stage==1 else "pair"
    return -min(risks(q,readers,view).values())


def fit_arm(space,arm,code1,code2,y1,y2,sex,unit,*,seed,cpu_seconds=1800.):
    """Complete mandated fixed fit; no early success stop or final-role lookup.

    Returns selected deployable policy, scalar receipts and PRIVATE checkpoints
    and reader arrays. Caller separates public/private serialization.
    """
    import torch
    torch.set_num_threads(1)
    if arm not in ("O_J","V_J","V_P","V_D","V_L","V_S12","V_S21"):raise ValueError("unknown fitted arm")
    p=space.prepare(code1,code2);sex=np.asarray(sex,dtype=np.int64);fold=group_folds(unit)
    if sex.shape!=(len(code1),) or not np.isin(sex,(0,1)).all():raise ValueError("protected fitting labels differ")
    c=costs(space,p,y1,y2);clock=time.process_time()
    checkpoints=[];slate=[];round_receipts=[];incomplete=False;frozen=None;frozen_slate=[]
    stages=(1,2) if arm in ("V_S12","V_S21") else (None,)
    first=1 if arm=="V_S12" else 2
    for stage in stages:
        stage_checkpoints=[];stage_readers=[]
        for start in range(2):
            init=_shape_weights(arm,seed+100*int(stage or 0),start)
            if stage==2:init=(frozen.weights[0],init[1])
            w=[tuple(torch.nn.Parameter(torch.as_tensor(a.copy())) for a in ww) for ww in init]
            if stage==2:
                for a in w[0]:a.requires_grad_(False)
            if stage==1:
                for a in w[1]:a.requires_grad_(False)
            trainable=[a for ww in w for a in ww if a.requires_grad]
            opt=torch.optim.Adam(trainable,lr=.01)
            current=Policy(space,arm,init,frozen.eta1 if stage==2 else .05,0. if stage==1 else .05)
            dual=None;retained=list(frozen_slate)
            for round_index in range(3):
                if time.process_time()-clock>=cpu_seconds:incomplete=True;break
                q=current.law_prepared(p)
                views=(f"r{first}",) if stage==1 else ((f"r{3-first}","pair") if stage==2 else ("r1","r2","pair"))
                new=fit_round(space,p,q,sex,fold,round_index=round_index,start_index=start,
                              seed=seed+100000*int(stage or 0),views=views)
                # stage label makes independently auditable unique reader IDs.
                for reader in new:reader.name=f"stage{stage}/{reader.name}"
                retained.extend(new);stage_readers.extend(new);slate.extend(new)
                safe,_,initial_receipt=backoff(space,p,arm,_snapshot(w),c,fold,retained,
                                              initial=True,first_eta=frozen.eta1 if stage==2 else None,stage=stage)
                if safe is not None:current=safe
                # Identity-backoff may select zero for deployment. Exploration
                # still uses nonzero .05: infeasible exploratory states are
                # allowed optimizer iterates, never returned checkpoints.
                eta1=frozen.eta1 if stage==2 else .05
                eta2=0. if stage==1 else .05
                fixedfirst=None
                if stage==2:
                    fp=frozen.law_prepared(p).reshape(-1,4,4)
                    fixedfirst=fp.sum(2 if first==1 else 1)
                per_round=50 if arm=="V_L" or stage is not None else 100
                for update in range(per_round+1):
                    total_update=round_index*per_round+update
                    if update==0 or total_update%25==0 or update==per_round:
                        weights=_snapshot(w)
                        candidate,_,receipt=backoff(space,p,arm,weights,c,fold,retained,
                                                    first_eta=frozen.eta1 if stage==2 else None,stage=stage)
                        entry={"stage":stage,"start":start,"round":round_index,"update":total_update,
                               "weights":weights,"policy":candidate,"receipt":receipt}
                        stage_checkpoints.append(entry);checkpoints.append(entry)
                        if candidate is not None:current=candidate
                    if update==per_round:break
                    if time.process_time()-clock>=cpu_seconds:incomplete=True;break
                    opt.zero_grad()
                    tq=_torch_law(space,p,w,arm,eta1,eta2,torch=torch,stage=stage,frozen_first=fixedfirst,
                                  temperature=.5**(total_update/299) if arm=="V_D" else 1.)
                    obj=_objective(tq,p,sex,retained,arm,torch,stage=stage)
                    residual=_constraints(tq,c,fold,retained,torch,
                              task_views=(first,) if stage==1 else (1,2),
                              guard_views=(f"r{first}",) if stage==1 else ("r1","r2"))
                    if dual is None or len(dual)!=len(residual):dual=np.zeros(len(residual))
                    loss=obj+sum(float(d)*r+50*torch.clamp(r,min=0).square() for d,r in zip(dual,residual))
                    loss.backward();torch.nn.utils.clip_grad_norm_(trainable,5);opt.step()
                    if (total_update+1)%25==0:
                        dual=np.maximum(0,dual+100*np.asarray([float(r.detach()) for r in residual]))
                round_receipts.append({"stage":stage,"start":start,"round":round_index,"completed":not incomplete,
                                       "policy_updates":per_round,"reader_count":len(new),"initial_backoff":initial_receipt})
                if incomplete:break
            if arm=="V_D" and not incomplete:
                candidates_for_start=[e for e in stage_checkpoints if e["start"]==start and e["policy"] is not None]
                best=min(candidates_for_start,key=lambda e:(_rank(e["policy"],p,sex,retained,arm),e["update"])) if candidates_for_start else None
                polished=(best["weights"] if best else _snapshot(w))
                for slot in range(16):
                    for delta in (-.5,-.1,.1,.5):
                        if time.process_time()-clock>=cpu_seconds:incomplete=True;break
                        trial=tuple(tuple(a.copy() for a in ww) for ww in polished)
                        trial[0][3][slot]+=delta
                        candidate,_,receipt=backoff(space,p,arm,trial,c,fold,retained)
                        current_policy=Policy(space,arm,polished,1.,1.)
                        current_ok=feasible(current_policy,p,c,fold,retained)[0]
                        accepted=candidate is not None and (not current_ok or _rank(candidate,p,sex,retained,arm)<_rank(current_policy,p,sex,retained,arm)-1e-10)
                        entry={"stage":stage,"start":start,"round":3,"update":300,
                               "weights":trial,"policy":candidate,"receipt":receipt,
                               "phase":"polish","slot":slot,"delta":delta,"accepted":accepted}
                        # All 64 attempts retained; stronger custody than source
                        # accepted-only records, with unchanged fixed trial list.
                        checkpoints.append(entry)
                        if accepted:polished=trial;stage_checkpoints.append(entry)
                    if incomplete:break
            # Always archive terminal/stopped arrays, even outside cadence.
            checkpoints.append({"stage":stage,"start":start,"round":-1,"update":-1,
                                "weights":_snapshot(w),"policy":None,"receipt":{"terminal_stopped":incomplete}})
            if incomplete:break
        if incomplete:break
        selection_slate=list(frozen_slate)+stage_readers
        candidates=[]
        # Recheck EVERY recorded checkpoint on combined retained slate. For
        # stage2 only its second eta changes; first stays bitwise frozen.
        for entry in stage_checkpoints:
            candidate,_,receipt=backoff(space,p,arm,entry["weights"],c,fold,selection_slate,
                                first_eta=frozen.eta1 if stage==2 else None,stage=stage)
            entry["final_replay"]=receipt;entry["final_policy"]=candidate
            if candidate is not None:
                candidates.append((_rank(candidate,p,sex,selection_slate,arm,stage),entry["start"],entry["update"],candidate,entry))
        identity=Policy(space,"P0",())
        if stage==1 or stage is None:
            candidates.append((_rank(identity,p,sex,selection_slate,arm,stage),-1,-1,identity,{"exact_identity":True}))
        if not candidates:
            # Whole-P0 fallback distinct from modification of frozen first law.
            return identity,{"status":"COMPLETE_WHOLE_P0_DEPENDENCY_FALLBACK","stage":stage,
                    "checkpoints":checkpoints,"readers":slate,"rounds":round_receipts,"elapsed_cpu_seconds":time.process_time()-clock}
        chosen=min(candidates,key=lambda row:row[:3])
        frozen=chosen[3]
        if stage==1:
            # Identity must be represented within sequential family to continue
            # genuine stage-two optimization with frozen exact first action.
            if frozen.arm=="P0":frozen=Policy(space,arm,stage_checkpoints[0]["weights"],0.,0.)
            frozen_slate=selection_slate
            for reader in frozen_slate:reader.models["stage1_frozen_marker"]=np.asarray([1])
    if incomplete:
        return None,{"status":"INCOMPLETE","checkpoints":checkpoints,"readers":slate,
                     "rounds":round_receipts,"elapsed_cpu_seconds":time.process_time()-clock}
    q=frozen.law_prepared(p)
    allreaders=frozen_slate+stage_readers if stages==(1,2) else stage_readers
    final_receipt={"status":"COMPLETE_IDENTITY_ALIAS" if np.array_equal(q,identity_q(p)) else "COMPLETE_SELECTED",
                   "selection":{"start":chosen[1],"update":chosen[2],"stage":stage},
                   "utility":utility_receipt(q,c,fold),"guards":reader_guard_receipt(q,allreaders,fold),
                   "nomination":nomination(q,allreaders),"eta1":frozen.eta1,"eta2":frozen.eta2,
                   "parameter_count":frozen.parameter_count,"checkpoints":checkpoints,"readers":slate,
                   "rounds":round_receipts,"elapsed_cpu_seconds":time.process_time()-clock}
    return frozen,final_receipt
