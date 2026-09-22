"""Exact support-aware finite perfect-privacy certificates; no optimization.

ACS survey weights and cell counts are integers. Multiplying each conditional
independence equation by its positive denominators gives an integer matrix.
Exact elimination separates real supported directions from absent-state freedom.
This is an empirical finite-model statement, not a population privacy result.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
from functools import reduce
import json
import math
import operator
from pathlib import Path
import numpy as np
from .math_replay import conditional_information


def integer_counts(value):
    a=np.asarray(value)
    if a.ndim!=3 or min(a.shape)<1 or not np.isfinite(a).all() or np.any(a<0) or np.any(a!=np.floor(a)) or np.any(a>2**53):
        raise ValueError('Exact certificate requires nonnegative exactly represented integer cell masses')
    if not np.any(a>0):raise ValueError('Each constrained law needs positive observed mass')
    return np.asarray(a,dtype=object)


def integer_equations(counts,supported):
    a=integer_counts(counts);ns,nc,nt=a.shape
    a=np.asarray([int(x) for x in a.ravel()],dtype=object).reshape(a.shape)
    rows=[]
    for c in range(nc):
        total=int(a[:,c,:].sum())
        for s in range(ns):
            sc=int(a[s,c,:].sum())
            row=[int(a[s,c,t])*total-sc*int(a[:,c,t].sum()) for t in supported]
            if any(row):rows.append(row)
    return rows


def _primitive(row):
    divisor=reduce(math.gcd,(abs(v) for v in row),0)
    if divisor:
        row=[v//divisor for v in row]
        if next(v for v in row if v)<0:row=[-v for v in row]
    return row


def exact_echelon(rows,width):
    """Fraction-free integer elimination with exact primitive-row reduction."""
    a=[_primitive(list(row)) for row in rows if any(row)];pivots=[];rank=0
    for col in range(width):
        candidates=[j for j in range(rank,len(a)) if a[j][col]]
        if not candidates:continue
        index=min(candidates,key=lambda j:(abs(a[j][col]).bit_length(),j))
        a[rank],a[index]=a[index],a[rank]
        pivot=a[rank][col]
        for j in range(rank+1,len(a)):
            factor=a[j][col]
            if not factor:continue
            g=math.gcd(abs(pivot),abs(factor))
            a[j]=_primitive([(pivot//g)*x-(factor//g)*y for x,y in zip(a[j],a[rank])])
        pivots.append(col);rank+=1
        if rank==len(a):break
    return a[:rank],pivots


def certify(counts_by_role,state_mass,parents,*,n_actions=2,zero_action=0):
    mass=np.asarray(state_mass)
    parents=np.asarray(parents)
    if mass.ndim!=1 or not np.isfinite(mass).all() or np.any(mass<0) or np.any(mass!=np.floor(mass)):
        raise ValueError('State support must be integer nonnegative counts')
    if parents.shape!=mass.shape or not np.isfinite(parents).all() or np.any(parents!=np.floor(parents)) or np.any(parents<0):
        raise ValueError('Invalid fixed parent map')
    try:n_actions,zero_action=operator.index(n_actions),operator.index(zero_action)
    except TypeError as error:raise ValueError('Action counts and indices must be integers') from error
    if n_actions<1 or not 0<=zero_action<n_actions:raise ValueError('Invalid action schema')
    supported=np.flatnonzero(mass>0).tolist();m=len(supported)
    if not m or not counts_by_role:raise ValueError('Observed support and constrained roles required')
    rows=[]
    for counts in counts_by_role.values():
        if np.asarray(counts).shape[-1]!=len(mass):raise ValueError('State schema mismatch')
        if np.any(np.asarray(counts)[...,mass==0]!=0):raise ValueError('Protected mass in unsupported state')
        rows.extend(integer_equations(counts,supported))
    if any(sum(row)!=0 for row in rows):raise AssertionError('Constant supported kernel must be feasible')
    echelon,pivots=exact_echelon(rows,m);rank=len(pivots);nullity=m-rank
    record={'supported_states':m,'unsupported_states':int(np.sum(mass==0)),
            'exact_rank':rank,'exact_nullity':nullity,'nonzero_integer_equations':len(rows),
            'constant_only_on_supported_states':nullity==1 or n_actions==1,
            'nonconstant_supported_kernel_exists':nullity>1 and n_actions>1,
            'arithmetic':'integer elimination and rational witness verification',
            'scope':'fixed empirical laws on observed support; no held-out utility or population privacy claim'}
    if not record['nonconstant_supported_kernel_exists']:return record,None
    free=[i for i in range(m) if i not in pivots];direction=None
    for col in free:
        v=[Fraction(0) for _ in range(m)];v[col]=Fraction(1)
        for row,pivot in reversed(list(zip(echelon,pivots))):
            v[pivot]=-sum(Fraction(row[j])*v[j] for j in range(pivot+1,m))/row[pivot]
        mean=sum(v)/m;v=[x-mean for x in v]
        if any(v):direction=v;break
    if direction is None:raise AssertionError('Nonconstant nullspace direction missing')
    if any(sum(Fraction(a)*b for a,b in zip(row,direction)) for row in rows):
        raise AssertionError('Exact direction violates an empirical independence equation')
    scale=max(abs(v) for v in direction);other=next(i for i in range(n_actions) if i!=zero_action)
    q=[[Fraction(0) for _ in range(n_actions)] for _ in mass]
    for t,v in zip(supported,direction):
        q[t][zero_action]=Fraction(1,2)+v/(4*scale)
        q[t][other]=Fraction(1,2)-v/(4*scale)
    for t in np.flatnonzero(mass==0):
        siblings=[s for s in supported if parents[s]==parents[t]]
        if siblings:
            denominator=sum(int(mass[s]) for s in siblings)
            q[t]=[sum(int(mass[s])*q[s][z] for s in siblings)/denominator for z in range(n_actions)]
        else:q[t][zero_action]=Fraction(1)
    if any(sum(row)!=1 or any(v<0 or v>1 for v in row) for row in q):raise AssertionError('Invalid rational kernel')
    for row in rows:
        for z in (zero_action,other):
            if sum(Fraction(a)*q[t][z] for a,t in zip(row,supported)):
                raise AssertionError('Rational witness violates exact zero privacy')
    floats=np.asarray(q,float)
    cmis={role:conditional_information(np.asarray(counts,float)/np.asarray(counts,float).sum(),floats)
          for role,counts in counts_by_role.items()}
    record.update(maximum_independent_float_cmi=max(cmis.values()),independent_float_cmi=cmis,
                  supported_row_range=float(np.ptp(floats[supported,zero_action])),
                  rational_equations_verified=True,rational_simplex_verified=True)
    if record['maximum_independent_float_cmi']>1e-10:raise AssertionError('Rational witness float CMI disagreement')
    return record,[[str(v) for v in row] for row in q]


def run_certificates():
    from . import run
    from .config import configuration,mechanism_id
    schedule=json.loads((run.OUT/'RESOURCE_SCHEDULE.json').read_text())
    if not schedule.get('frozen_before_comparative_outcomes'):raise RuntimeError('Resource schedule must be frozen')
    reports=[];unavailable=[]
    for spec in configuration()['maps']:
        if spec['budget']!=0:continue
        anchor,name=spec['anchor'],spec['configuration'];base=run.anchor_dir(anchor)/'maps'/name
        if not (base/'ACCEPTED.json').exists():unavailable.append(spec['id']);continue
        receipt=run._map_receipt(anchor,name)
        meta=json.loads((base/'metadata.json').read_text())
        with np.load(base/'tables.npz',allow_pickle=False) as saved:
            mass=saved['state_mass'].copy();laws={}
            for role in meta['constrained_roles']:
                stem,weight=role.rsplit('/',1)
                laws[role]=saved['support/'+stem+('/count' if weight=='U' else '/sumw')].copy()
        family=spec['input'];parents=np.arange(len(mass)) if family=='T0' else np.arange(len(mass))//2
        with np.load(base/'Q.npz',allow_pickle=False) as saved:n_actions=saved['Q'].shape[1]
        # Primary dictionary construction registers zero as action0.
        report,witness=certify(laws,mass,parents,n_actions=n_actions,zero_action=0)
        report.update(anchor=anchor,configuration=name,map_receipt_sha256=run.sha(base/'ACCEPTED.json'))
        if witness is not None:
            target=run.OUT/'private/zero_certificates'/f'{name}_anchor_{anchor}.json'
            run.atomic(target,{'Q_rational':witness,'report':report})
            report['private_witness_sha256']=run.sha(target)
        reports.append(report)
    output={'written_utc':run.now(),'certificates':reports,'unavailable':unavailable,
            'P7_supported':any(r['nonconstant_supported_kernel_exists'] for r in reports)}
    run.atomic(run.OUT/'ZERO_PRIVACY_CERTIFICATES.json',output)
    return output


if __name__=='__main__':
    result=run_certificates();print(json.dumps({'certificates':len(result['certificates']),'P7_supported':result['P7_supported']}))
