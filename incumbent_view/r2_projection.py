"""Exact fixed-schedule affine/quadratic identity-backoff prefilter.

No new search points are introduced. Selected points still receive the full
direct float64 endpoint/guard replay by r2_train.feasible.
"""
import math
import numpy as np
from .r2_space import CAPS,SAFETY


def law_coefficients(q0,qhalf,q1):
    b=2*(q1-2*qhalf+q0)
    a=q1-q0-b
    return q0,a,b


def constraint_coefficients(coeff,cost,fold,readers,task_views=(1,2),guard_views=("r1","r2")):
    out=[]
    for mask in (np.ones(len(fold),bool),*(fold==k for k in range(3))):
        for key,c in cost.items():
            if int(key[1]) not in task_views:continue
            sign=-1 if key.endswith("accuracy") else 1
            diff=c-c[:,0,None]
            vals=[sign*math.fsum(float(v) for v in np.sum(x[mask]*diff[mask],axis=1))/mask.sum() for x in coeff]
            out.append((tuple(vals),CAPS[key.split("_")[-1]]-SAFETY,"upper",key))
        for r in readers:
            if r.view not in guard_views:continue
            diff=r.loss-r.loss[:,0,None]
            vals=[math.fsum(float(v) for v in np.sum(x[mask]*diff[mask],axis=1))/mask.sum() for x in coeff]
            out.append((tuple(vals),-SAFETY,"lower",r.name))
    return out


def polynomial_feasible(eta,coefficients):
    for (c,a,b),limit,direction,_ in coefficients:
        value=c+eta*a+eta*eta*b
        if direction=="upper" and value>limit:return False
        if direction=="lower" and value<limit:return False
    return True
