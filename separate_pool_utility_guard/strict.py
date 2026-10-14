"""Inherited strict batch safeguard, copied byte-for-byte from the corrected
compressed-income trial (gradient_worker.py) and renamed strict_*; used only
by OLD_BATCH_GUARD and, unguarded, by ORDINARY/TASK_ONLY. No numerical import.
"""
from __future__ import annotations
from itertools import combinations

QP_TOL=1e-10
FINITE_TOL=1e-10
BACKTRACK=(1.,.5,.25,.125)
NORM_REPAIR_SAFETY=1e-6
NORM_REPAIR_ATTEMPTS=16


def strict_project_displacement(proposal,task_gradients,max_norm):
    """min 1/2||d-proposal||² subject to a_j.d<=0 and ||d||<=max_norm.

    Two differentiable task guards only. Normalize every strictly nonzero
    row, including tiny gradients. Homogeneous cone projection followed by
    radial clipping is the Euclidean projection onto cone intersect ball.
    """
    import numpy as np
    p=np.asarray(proposal,dtype=np.float64);a=np.asarray(task_gradients,dtype=np.float64)
    if p.ndim!=1 or not p.size or a.ndim!=2 or a.shape[1]!=p.size or len(a)>2:raise ValueError('at most two aligned task guards')
    if not np.isfinite(p).all() or not np.isfinite(a).all() or not np.isfinite(max_norm) or max_norm<=0:raise ValueError('finite positive bounded proposal required')
    # Scale before norm to avoid underflow removing a real tiny guard.
    row_scale=np.max(np.abs(a),axis=1) if len(a) else np.zeros(0)
    keep=row_scale>0
    aa=a[keep]/row_scale[keep,None];aa/=np.linalg.norm(aa,axis=1)[:,None]
    candidates=[];tol=QP_TOL*max(1.,float(np.linalg.norm(p)))
    for size in range(len(aa)+1):
        for active in combinations(range(len(aa)),size):
            at=aa[list(active)]
            if size:
                multipliers=np.linalg.lstsq(at@at.T,at@p,rcond=None)[0]
                d=p-at.T@multipliers
                if np.max(np.abs(at@d))>tol or np.any(multipliers < -tol):continue
            else:d=p.copy();multipliers=np.zeros(0)
            if len(aa) and np.max(aa@d)>tol:continue
            candidates.append((float(np.sum((d-p)**2)),d,active,multipliers))
    fallback=not candidates
    if fallback:d=np.zeros_like(p);active=();multipliers=np.zeros(0)
    else:_,d,active,multipliers=min(candidates,key=lambda item:item[0])
    norm=float(np.linalg.norm(d));ball_active=norm>max_norm
    if ball_active:d=d*(max_norm/norm)
    residual=aa@d
    violation=float(max(0.,float(residual.max()))) if len(aa) else 0.
    return d,{'zero_feasible':True,'zero_gradients':int((~keep).sum()),
              'guard_count':len(a),'active_set':list(active),'multipliers':multipliers.tolist(),
              'norm_bound_active':ball_active,'proposal_norm':float(np.linalg.norm(p)),
              'displacement_norm':float(np.linalg.norm(d)),'normalized_constraint_products':residual.tolist(),
              'max_normalized_violation':violation,'conservative_zero_fallback':fallback,
              'certified':bool(violation<=tol and np.linalg.norm(d)<=max_norm+tol)}


def parameter_vector(parameters):
    import numpy as np
    return np.concatenate([p.detach().cpu().double().reshape(-1).numpy() for p in parameters])


def _assign(parameters,vector):
    import torch
    offset=0
    with torch.no_grad():
        for p in parameters:
            end=offset+p.numel();p.copy_(torch.from_numpy(vector[offset:end].copy()).reshape(p.shape).to(p.dtype));offset=end
    if offset!=len(vector):raise ValueError('parameter displacement dimension differs')


def strict_finite_step(parameters,proposal,task_gradients,task_evaluator,max_norm,*,guarded=True):
    """Apply a bounded SGD displacement, certify actual dtype-rounded movement.

    task_evaluator reads only the authorized FIT-update batch and returns NLL
    and summed Brier. Any failed step restores exactly the saved parameters.
    No momentum/Adam state can bypass a projected displacement.
    """
    import numpy as np
    before=parameter_vector(parameters);baseline=task_evaluator()
    if set(baseline)!={'nll','brier'} or not all(np.isfinite(list(baseline.values()))):raise ValueError('finite two-metric FIT safeguard required')
    a=np.asarray(task_gradients,dtype=np.float64)
    d,qp=strict_project_displacement(proposal,a if guarded else np.zeros((0,len(before))),max_norm)
    attempts=[];accepted=None;actual=np.zeros_like(before)
    row_scale=np.max(np.abs(a),axis=1) if len(a) else np.zeros(0)
    keep=row_scale>0;normal=a[keep]/row_scale[keep,None]
    if len(normal):normal/=np.linalg.norm(normal,axis=1)[:,None]
    for fraction in BACKTRACK if guarded else (1.,):
        # A float64 vector clipped exactly to the ball can round outside it
        # when copied to float32 parameters. Repair only the actual norm,
        # identically in every arm, before evaluating task feasibility. The
        # scalar contractions preserve the proposed homogeneous cone; actual
        # rounded halfspace products are still checked separately below.
        norm_trials=[];norm_scale=1.;norm_certified=False
        for repair_attempt in range(NORM_REPAIR_ATTEMPTS):
            _assign(parameters,before+fraction*norm_scale*d)
            actual=parameter_vector(parameters)-before
            actual_norm=float(np.linalg.norm(actual))
            norm_certified=bool(np.isfinite(actual_norm) and actual_norm<=max_norm)
            norm_trials.append({'attempt':repair_attempt,'scale':norm_scale,
                                'actual_norm':actual_norm,'bounded':norm_certified})
            if norm_certified:break
            if not np.isfinite(actual_norm):break
            if repair_attempt+1==NORM_REPAIR_ATTEMPTS:break
            norm_scale*=min(1.,max_norm/actual_norm)*(1.-NORM_REPAIR_SAFETY)
        products=normal@actual if len(normal) else np.zeros(0)
        actual_norm=float(np.linalg.norm(actual));metric=task_evaluator()
        finite=all(np.isfinite(list(metric.values())))
        local=not len(products) or bool(np.max(products)<=QP_TOL*max(1.,actual_norm))
        bounded=norm_certified
        safe=finite and all(metric[k]<=baseline[k]+FINITE_TOL for k in baseline)
        passed=finite and bounded and (not guarded or (local and safe))
        attempts.append({'fraction':fraction,'metrics':metric,'actual_norm':actual_norm,
                         'norm_repair_scale':norm_scale,'actual_effective_fraction':fraction*norm_scale,
                         'norm_repair_attempts':norm_trials,
                         'actual_normalized_constraint_products':products.tolist(),
                         'first_order_check':local,'finite_step_check':safe,'accepted':passed})
        if passed:accepted=fraction;break
        _assign(parameters,before)
    if accepted is None:
        _assign(parameters,before);actual=np.zeros_like(before);status='REJECTED_ZERO_STEP'
    elif not np.any(actual):status='ROUNDING_ZERO_STEP'
    else:status='ACCEPTED'
    return {'status':status,'accepted_fraction':accepted,'actual_displacement_norm':float(np.linalg.norm(actual)),
            'baseline':baseline,'attempts':attempts,'projection':qp,'guarded':guarded,
            'actual_norm_repair':{'all_arms':True,'safety_fraction':NORM_REPAIR_SAFETY,
                                  'maximum_attempts':NORM_REPAIR_ATTEMPTS,'task_metrics_used':False},
            'finite_step_tolerance':FINITE_TOL,'backtrack':list(BACKTRACK)}


def flat_gradient(loss,parameters):
    import numpy as np
    import torch
    if not loss.requires_grad:return np.zeros(sum(p.numel() for p in parameters))
    values=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    return np.concatenate([np.zeros(p.numel()) if v is None else v.detach().cpu().double().reshape(-1).numpy() for p,v in zip(parameters,values)])


def gradient_summary(vectors):
    import numpy as np
    norms={name:float(np.linalg.norm(v)) for name,v in vectors.items()};pairs={}
    for left,right in combinations(vectors,2):
        dot=float(vectors[left]@vectors[right]);den=norms[left]*norms[right]
        cosine=dot/den if den>0 else None
        pairs[left+'__'+right]={'dot':dot,'cosine':cosine,
                               'same_minimization_direction_conflict':bool(dot<0) if den>0 else None}
    return {'norms':norms,'pairs':pairs,'zero_norm_cosine':None,'normalization':'mean row loss; Euclidean parameter-gradient norm; float64 reductions'}


