"""Separately written projection, mixed-dtype movement and log certification."""
from __future__ import annotations


def normalized_guards(guards):
    import numpy as np
    guards=np.asarray(guards,dtype=np.float64)
    maxima=np.max(abs(guards),axis=1) if len(guards) else np.empty(0)
    rows=guards[maxima>0]/maxima[maxima>0,None]
    return rows/np.linalg.norm(rows,axis=1)[:,None] if len(rows) else rows


def project(proposal, guards, radius=.01):
    """Moreau polar projection using independent nonnegative least squares."""
    import numpy as np
    from scipy.optimize import nnls
    p=np.asarray(proposal,dtype=np.float64);g=np.asarray(guards,dtype=np.float64)
    if p.ndim!=1 or not p.size or g.ndim!=2 or g.shape[1]!=len(p) or len(g)>2:
        raise ValueError('two aligned safeguard guards required')
    if not np.isfinite(p).all() or not np.isfinite(g).all() or not np.isfinite(radius) or radius<=0:
        raise ValueError('finite bounded displacement required')
    a=normalized_guards(g)
    if len(a):
        dual,_=nnls(a.T,p,maxiter=100);d=p-a.T@dual
    else:d=p.copy()
    length=np.linalg.norm(d)
    if length>radius:d*=radius/length
    if len(a) and np.max(a@d)>1e-10*max(1.,np.linalg.norm(p)):
        raise ArithmeticError('independent cone solution uncertified')
    return d


def finite_step(arrays, proposal, guards, evaluator, *, guarded=True, radius=.01):
    """Arrays retain separate dtypes; reject restores saved bytes exactly."""
    import numpy as np
    original=[a.copy() for a in arrays]
    before=np.concatenate([a.ravel().astype(np.float64) for a in arrays])
    baseline=evaluator();a=normalized_guards(guards)
    d=project(proposal,guards if guarded else np.empty((0,len(before))),radius)
    def assign(vector):
        at=0
        for array in arrays:
            size=array.size;array[...] = vector[at:at+size].reshape(array.shape).astype(array.dtype);at+=size
    def restore():
        for array,saved in zip(arrays,original):array[...] = saved
    attempts=[]
    for fraction in ((1.,.5,.25,.125) if guarded else (1.,)):
        scale=1.;trials=[]
        for repair in range(16):
            assign(before+fraction*scale*d)
            actual=np.concatenate([v.ravel().astype(np.float64) for v in arrays])-before
            norm=float(np.linalg.norm(actual));bounded=np.isfinite(norm) and norm<=radius
            trials.append({'actual_norm':norm,'scale':scale,'bounded':bool(bounded)})
            if bounded or not np.isfinite(norm):break
            scale*=min(1.,radius/norm)*(1.-1e-6)
        metrics=evaluator();products=a@actual
        finite=all(np.isfinite(v) for v in metrics.values())
        first=not len(products) or np.max(products)<=1e-10*max(1.,norm)
        safe=finite and all(metrics[k]<=baseline[k]+1e-10 for k in ('nll','brier'))
        accepted=bool(finite and bounded and (not guarded or first and safe))
        attempts.append({'fraction':fraction,'metrics':metrics,'actual_norm':norm,
                         'accepted':accepted,'norm_repair_attempts':trials})
        if accepted:
            return {'status':'ACCEPTED' if np.any(actual) else 'ROUNDING_ZERO_STEP',
                    'actual_displacement':actual,'accepted_fraction':fraction,'attempts':attempts}
        restore()
    restore()
    return {'status':'REJECTED_ZERO_STEP','actual_displacement':np.zeros_like(before),
            'accepted_fraction':None,'attempts':attempts}


def audit_steps(records, expected_proposals=800):
    """Check every logged proposal, including failures and accepted zero moves."""
    import math
    counts={'accepted_nonzero':0,'feasibility_accepted':0,'rejected':0,'zero':0,'shortened':0,'norm_repaired':0}
    by_epoch={}
    for record in records:
        step=record.get('step',record)
        status=step['status'];fraction=step['accepted_fraction'];norm=step['actual_displacement_norm']
        if status not in ('ACCEPTED','ROUNDING_ZERO_STEP','REJECTED_ZERO_STEP'):
            raise ValueError('unregistered finite step status')
        if not math.isfinite(norm) or not 0<=norm<=.01: raise ValueError('actual movement outside bound')
        accepted=[a for a in step['attempts'] if a['accepted']]
        if status=='REJECTED_ZERO_STEP':
            if fraction is not None or norm!=0 or accepted:raise ValueError('rejection must restore exact zero movement')
            counts['rejected']+=1
        else:
            if len(accepted)!=1 or fraction!=accepted[0]['fraction']:raise ValueError('unique accepted attempt required')
            if (status=='ROUNDING_ZERO_STEP')!=(norm==0):raise ValueError('zero-step status differs')
            counts['feasibility_accepted']+=1
            counts['accepted_nonzero']+=status=='ACCEPTED'
            counts['shortened']+=status=='ACCEPTED' and fraction<1
        counts['zero']+=norm==0
        if step.get('guarded'):
            for attempt in accepted:
                if not all(attempt['metrics'][k]<=step['baseline'][k]+1e-10 for k in ('nll','brier')):
                    raise ValueError('finite safeguard sign or tolerance failure')
                if any(v>1e-10*max(1.,norm) for v in attempt['actual_normalized_constraint_products']):
                    raise ValueError('actual movement violates task halfspace')
        for attempt in step['attempts']:
            if len(attempt['norm_repair_attempts'])>16:raise ValueError('excess rounding repairs')
        counts['norm_repaired']+=any(len(a['norm_repair_attempts'])>1 for a in step['attempts'])
        if record.get('frozen_income_bitwise_unchanged') is False or step.get('frozen_income_bitwise_unchanged') is False:
            raise ValueError('frozen income changed')
        epoch=record.get('epoch',0);by_epoch.setdefault(epoch,[]).append((status,norm))
    if len(records)!=expected_proposals:raise ValueError('complete proposal log required')
    return {'proposal_count':len(records),'counts':counts,
            'epochs':{str(e):{'proposals':len(v),'accepted_fraction':sum(s=='ACCEPTED' for s,n in v)/len(v),
                             'feasibility_accepted_fraction':sum(s!='REJECTED_ZERO_STEP' for s,n in v)/len(v),
                             'zero_stall_fraction':sum(n==0 for s,n in v)/len(v)} for e,v in by_epoch.items()}}
