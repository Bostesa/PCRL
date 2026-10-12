"""Independent frozen-reader envelopes and paired-group endpoint inference."""
from __future__ import annotations
import numpy as np
from scipy.stats import t

class FixedAUC:
    def __init__(self,sex,score):
        sex,score=np.asarray(sex,dtype=int),np.asarray(score,dtype=float)
        if set(np.unique(sex))!={0,1} or not np.isfinite(score).all():raise ValueError('invalid AUC inputs')
        self.order=np.argsort(score,kind='stable')
        sorted_score=score[self.order]
        self.positive=sex[self.order]
        self.starts=np.r_[0,np.flatnonzero(sorted_score[1:]!=sorted_score[:-1])+1]
    def evaluate(self,weights):
        w=np.asarray(weights,dtype=float)[self.order]
        pos=np.add.reduceat(w*self.positive,self.starts)
        neg=np.add.reduceat(w*(1-self.positive),self.starts)
        denominator=pos.sum()*neg.sum()
        if denominator<=0:raise ValueError('resampled AUC lacks a class')
        return float(np.sum(pos*(np.cumsum(neg)-neg+.5*neg))/denominator)

def interval(point,reps,k,groups):
    se=float(np.std(reps,ddof=1));critical=float(t.ppf(1-.05/(2*k),df=groups-1))
    return {'point':float(point),'se':se,'lower':float(point-critical*se),'upper':float(point+critical*se),'critical':critical,
            'replicates':len(reps),'groups':groups,'K':k,'approximate_conditional_development':True}

def clause(point,reps,k,groups,target,side):
    result=interval(point,reps,k,groups)
    result.update(target=target,side=side)
    if side=='upper':
        result['status']='PASS' if result['upper']<=target else 'MEASURED_VIOLATION' if result['lower']>target else 'UNRESOLVED'
    elif side=='lower':
        result['status']='PASS' if result['lower']>=target else 'MEASURED_VIOLATION' if result['upper']<target else 'UNRESOLVED'
    else:raise ValueError('invalid clause sign')
    return result

def bootstrap_counts(groups):
    unique,inverse=np.unique(groups,return_inverse=True)
    # Exact registered shared draw law, not separately resampled source seeds.
    draws=np.random.default_rng(2026101208).integers(0,len(unique),size=(2048,len(unique)))
    for row in draws:
        counts=np.bincount(row,minlength=len(unique))
        yield counts[inverse]

def scores_envelope(scores,sex,groups):
    filters={'full':lambda n:True,'score':lambda n:n.split('__')[-1].startswith('OUTPUT_'),
             'representation':lambda n:n.split('__')[-1].startswith('REP_')}
    prepared={}
    for view in ('r1','r2','pair'):
        bank={name[len(view)+2:]:FixedAUC(sex,p) for name,p in scores.items()
              if name.startswith(view+'__') and not name[len(view)+2:].startswith(('PLANTED','SHUFFLED'))}
        if not bank:raise ValueError('missing actual view reader bank')
        for contract,keep in filters.items():
            selected={name:a for name,a in bank.items() if keep(name)}
            if not selected:raise ValueError('missing contract reader bank')
            prepared[(contract,view)]=selected
    ones=np.ones(len(sex));point={key:max(a.evaluate(ones) for a in bank.values()) for key,bank in prepared.items()}
    replicates={key:np.empty(2048,dtype=float) for key in prepared}
    # Evaluate each frozen reader once per replicate, reuse it across contracts.
    byview={view:prepared[('full',view)] for view in ('r1','r2','pair')}
    for b,w in enumerate(bootstrap_counts(groups)):
        for view,bank in byview.items():
            values={name:a.evaluate(w) for name,a in bank.items()}
            for contract,keep in filters.items():replicates[(contract,view)][b]=max(v for n,v in values.items() if keep(n))
    return point,replicates
