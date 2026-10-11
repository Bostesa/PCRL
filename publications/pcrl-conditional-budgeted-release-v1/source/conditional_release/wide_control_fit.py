"""Registered full-bank controls using variable class-compatible action axes.

All numerical entry points are called by the queued control worker. These
builders consume already admitted arrays, never load protected labels.
"""
from __future__ import annotations
import time
import numpy as np
from scipy import sparse
from .lp import AffineLaw,solve_protection,direct_bayes
from .controls import ControlCeilings,simplex_rows,initial_factor,register_second_groups,template_bounds,apply_second_groups


def widths(metadata):
    return tuple(max(len(d[f"recipient{r}_templates"]) for d in metadata["contexts"]) for r in (1,2))


def pad_template(law,metadata,width):
    """Zero-pad a valid template law; keep original parameters and output IDs."""
    old=law.base.shape[1]
    if width<=old:return law,metadata
    n=len(law.base)
    one=np.repeat(law.identity1[:,None],width,axis=1);two=np.repeat(law.identity2[:,None],width,axis=1)
    base=np.zeros((n,width),np.float64)
    one[:,:old]=law.emitted1;two[:,:old]=law.emitted2;base[:,:old]=law.base
    coo=law.delta.tocoo()
    row=(coo.row//old)*width+coo.row%old
    delta=sparse.coo_matrix((coo.data,(row,coo.col)),shape=(n*width,law.parameters)).tocsr()
    answer=AffineLaw(one,two,law.identity1,law.identity2,base,delta,law.bounds,law.weights,
                     law.equality_matrix,law.equality_rhs)
    metadata=dict(metadata,maximum_actions_per_row=width)
    return answer,metadata


def axis_valid(contexts,metadata,recipient):
    maximum=widths(metadata)[recipient-1]
    valid=np.zeros((len(contexts),maximum),bool)
    valid[:,0]=True
    for description in metadata["contexts"]:
        valid[np.asarray(contexts)==description["context"],:len(description[f"recipient{recipient}_templates"])]=True
    return valid


def coordinate_law(template,metadata,contexts,groups,fixed_other,recipient):
    """One independent-product marginal coordinate as an exact affine law."""
    contexts=np.asarray(contexts,np.int64);groups=np.asarray(groups,np.int64)
    axis=widths(metadata)[recipient-1]
    count=int(groups.max())+1
    fixed_other=np.asarray(fixed_other,np.float64)
    if fixed_other.shape!=(len(contexts),widths(metadata)[2-recipient]):
        raise ValueError("fixed class-bank factor width differs")
    rows=[];columns=[];values=[]
    for d in metadata["contexts"]:
        selected=np.flatnonzero(contexts==d["context"])
        w1,w2=(len(d[f"recipient{r}_templates"]) for r in (1,2))
        a,b=np.indices((w1,w2));action=(a*w2+b).ravel()
        own=(a if recipient==1 else b).ravel()
        other=(b if recipient==1 else a).ravel()
        rows.append((selected[:,None]*template.base.shape[1]+action).ravel())
        columns.append((groups[selected,None]*axis+own).ravel())
        values.append(fixed_other[selected[:,None],other].ravel())
    delta=sparse.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(columns))),
                            shape=(template.base.size,count*axis)).tocsr()
    delta.eliminate_zeros()
    active=contexts>=0
    bounds=template_bounds(groups[active],axis_valid(contexts,metadata,recipient)[active])
    equality,rhs=simplex_rows(count,axis)
    return AffineLaw(template.emitted1,template.emitted2,template.identity1,template.identity2,template.base,
                     delta,bounds,template.weights,equality[:count],rhs[:count])


def sequential_second_law(template,metadata,contexts,groups,frozen_first,first_recipient):
    """Frozen-first law with second weights keyed by actual emitted first code."""
    contexts=np.asarray(contexts,np.int64);groups=np.asarray(groups,np.int64)
    second=3-first_recipient
    maximum=widths(metadata)[second-1]
    count=int(groups.max())+1
    rows=[];columns=[];values=[]
    for d in metadata["contexts"]:
        selected=np.flatnonzero(contexts==d["context"])
        w1,w2=(len(d[f"recipient{r}_templates"]) for r in (1,2))
        a,b=np.indices((w1,w2));action=(a*w2+b).ravel()
        first_action=(a if first_recipient==1 else b).ravel()
        second_action=(b if first_recipient==1 else a).ravel()
        rows.append((selected[:,None]*template.base.shape[1]+action).ravel())
        columns.append((groups[selected[:,None],first_action]*maximum+second_action).ravel())
        values.append(frozen_first[selected[:,None],first_action].ravel())
    delta=sparse.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(columns))),
                            shape=(template.base.size,count*maximum)).tocsr()
    delta.eliminate_zeros()
    first_width=widths(metadata)[first_recipient-1]
    later_valid=np.repeat(axis_valid(contexts,metadata,second)[:,None,:],first_width,axis=1).reshape(-1,maximum)
    active=np.repeat(contexts>=0,first_width)
    bounds=template_bounds(groups.ravel()[active],later_valid[active])
    equality,rhs=simplex_rows(count,maximum)
    return AffineLaw(template.emitted1,template.emitted2,template.identity1,template.identity2,template.base,
                     delta,bounds,template.weights,equality[:count],rhs[:count])


def fit_product(template,metadata,contexts,groups1,groups2,sensitive,constraints_builder,ceilings=ControlCeilings(), *, solver_module=None, attempt_callback=None):
    backend = solver_module
    solve = solve_protection if backend is None else backend.solve_protection
    bayes = direct_bayes if backend is None else backend.direct_bayes
    begun=time.monotonic();width=widths(metadata)
    groups=(np.asarray(groups1,np.int64),np.asarray(groups2,np.int64))
    active=np.asarray(contexts)>=0
    bounds=[template_bounds(g[active],axis_valid(contexts,metadata,r)[active]) for r,g in enumerate(groups,1)]
    attempts=[];best=None
    for start in ("keep","uniform"):
        factors=[initial_factor(b,start,width=w) for b,w in zip(bounds,width)]
        stop=False
        for iteration in range(ceilings.rounds):
            for recipient in (1,2):
                own=recipient-1;other=1-own
                if time.monotonic()-begun>=ceilings.seconds:stop=True;break
                law=coordinate_law(template,metadata,contexts,groups[own],factors[other][groups[other]],recipient)
                rows=constraints_builder(law)
                remaining=ceilings.seconds-(time.monotonic()-begun)
                if remaining<=0:stop=True;break
                result=solve(law,sensitive,rows,time_limit=remaining,tolerance=ceilings.feasibility_tolerance)
                attempt={"start":start,"round":iteration,"recipient":recipient,"result":result}
                if attempt_callback is not None:
                    attempt_callback(len(attempts),attempt)
                    # Full primal/dual/matrices are already immutable on disk.
                    # Keep only what selection needs; no array/result history.
                    result={k:result[k] for k in ("success","theta","certificate","solver_class") if k in result}
                    attempt["result"]=result
                attempts.append(attempt)
                if not result["success"]:stop=True;break
                factors[own]=result["theta"].reshape(-1,width[own])
                objective,views=bayes(law,result["theta"],sensitive)
                if best is None or objective<best["objective"]-1e-12:
                    best={"law":law,"theta":result["theta"],"objective":objective,"views":views,
                          "factors":[v.copy() for v in factors],"result":result,
                          "start":start,"round":iteration,"recipient":recipient}
            if stop:break
    return {"success":best is not None,"selected":best,"attempts":attempts,
            "solver_class":"bounded alternating coordinate LPs, variable full-class-bank axes",
            "global_optimality":"UNRESOLVED","elapsed_seconds":time.monotonic()-begun}


def emitted_first_matrix(template,metadata,contexts,recipient):
    maximum=widths(metadata)[recipient-1]
    own=(template.identity1 if recipient==1 else template.identity2)
    output=np.repeat(own[:,None],maximum,axis=1)
    for d in metadata["contexts"]:
        selected=np.flatnonzero(np.asarray(contexts)==d["context"])
        axis=np.asarray(d[f"recipient{recipient}_templates"],np.int64)
        mapped=np.broadcast_to(axis,(len(selected),len(axis))).copy()
        mapped[:,axis==-1]=own[selected,None]
        output[selected,:len(axis)]=mapped
    return output


def fit_sequential(template,metadata,contexts,sensitive,constraints_builder,first_recipient,ceilings=ControlCeilings(), *, solver_module=None):
    backend = solver_module
    solve = solve_protection if backend is None else backend.solve_protection
    bayes = direct_bayes if backend is None else backend.direct_bayes
    begun=time.monotonic();width=widths(metadata);first=first_recipient-1;later=1-first
    contexts=np.asarray(contexts,np.int64)
    # Unsupported rows have constant identity mass in the template and do not
    # enter the active context parameter families.
    positive=contexts>=0
    group=contexts.copy();group[~positive]=0
    emitted=emitted_first_matrix(template,metadata,contexts,first_recipient)
    _,keys,counts=register_second_groups(group[positive],emitted[positive],axis_valid(contexts,metadata,first_recipient)[positive])
    keys_groups=np.zeros(emitted.shape,np.int64)
    keys_groups[positive]=apply_second_groups(group[positive],emitted[positive],keys)
    keep=np.zeros((len(group),width[later]));keep[:,0]=1.
    law1=coordinate_law(template,metadata,contexts,group,keep,first_recipient)
    result1=solve(law1,sensitive,constraints_builder(law1),
                              time_limit=max(.001,ceilings.seconds/2-(time.monotonic()-begun)),
                              tolerance=ceilings.feasibility_tolerance)
    record={"success":False,"first_result":result1,"second_result":None,
            "second_group_keys":keys,"second_group_uniform_counts":counts,
            "solver_class":"two freeze-first LP stages, variable full-class-bank axes"}
    if not result1["success"]:return record
    theta1=result1["theta"].reshape(-1,width[first])
    frozen_first=theta1[group]
    law2=sequential_second_law(template,metadata,contexts,keys_groups,frozen_first,first_recipient)
    result2=solve(law2,sensitive,constraints_builder(law2),
                              time_limit=max(.001,min(ceilings.seconds/2,ceilings.seconds-(time.monotonic()-begun))),
                              tolerance=ceilings.feasibility_tolerance)
    record["second_result"]=result2
    if result2["success"]:
        objective,views=bayes(law2,result2["theta"],sensitive)
        record.update(success=True,first_theta=theta1,second_theta=result2["theta"].reshape(-1,width[later]),
                      law=law2,theta=result2["theta"],objective=objective,views=views)
    return record
