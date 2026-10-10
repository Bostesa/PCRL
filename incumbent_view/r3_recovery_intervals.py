"""Frozen-slate complete-view recovery intervals for the K948 family.

No attack model fitting, reader selection, or candidate code import occurs here.
Each bootstrap replicate recomputes the maximum orientation of every already
frozen public-view reader after resampling aligned duplicate-person groups.
"""
from itertools import combinations
import numpy as np
from scipy.stats import t as student_t
from .r3_family import ARMS, VIEWS, RECOVERY_SUMMARIES, K

BOOTSTRAPS=2048
GROUPS=2234
ALPHA=.05
STOCHASTIC=frozenset(('O_J','V_J','V_P','V_L','V_S12','V_S21'))


def weighted_auc_batch(y,score,group_index,weights,*,groups=GROUPS):
    truth=np.asarray(y);values=np.asarray(score);group=np.asarray(group_index);w=np.asarray(weights)
    if (truth.dtype!=np.int64 or truth.ndim!=1 or values.dtype!=np.float64 or values.shape!=truth.shape
        or group.dtype!=np.int64 or group.shape!=truth.shape or w.ndim!=2 or w.shape[1]!=groups
        or not np.isin(truth,(0,1)).all() or not np.isfinite(values).all() or np.any(w<0)
        or np.any(group<0) or np.any(group>=groups)):
        raise ValueError('weighted AUC person/group/reader schema differs')
    order=np.argsort(values,kind='mergesort');sorted_score=values[order]
    starts=np.r_[0,np.flatnonzero(sorted_score[1:]!=sorted_score[:-1])+1]
    person_weights=w[:,group[order]].astype(np.float64,copy=False)
    positive=np.add.reduceat(person_weights*truth[order][None,:],starts,axis=1)
    negative=np.add.reduceat(person_weights*(1-truth[order])[None,:],starts,axis=1)
    lower=np.cumsum(negative,axis=1)-negative
    numerator=np.sum(positive*(lower+.5*negative),axis=1)
    denominator=positive.sum(1)*negative.sum(1)
    if np.any(denominator<=0):raise ValueError('paired group bootstrap lost a SEX class')
    return numerator/denominator


def bootstrap_plan(unit,*,bootstrap_draws=BOOTSTRAPS,groups=GROUPS):
    unit=np.asarray(unit)
    if unit.dtype!=np.int64 or unit.ndim!=1 or np.unique(unit).size!=groups:raise ValueError('fixed INNER duplicate-group inventory differs')
    _,inverse=np.unique(unit,return_inverse=True)
    rng=np.random.Generator(np.random.PCG64DXSM(20261010))
    samples=rng.integers(0,groups,size=(bootstrap_draws,groups),dtype=np.int64)
    weights=np.concatenate((np.ones((1,groups),dtype=np.int64),np.stack([np.bincount(x,minlength=groups) for x in samples])),axis=0)
    # Draw after ALL group samples; each seed gets one coupled three-index
    # whole-release tuple per replicate, shared across all arms and views.
    draw_indices=rng.integers(0,3,size=(bootstrap_draws,2,3),dtype=np.int64)
    return inverse.astype(np.int64),weights,draw_indices


def frozen_envelope(y,predictions,reader_order,group_index,weights,*,groups=GROUPS):
    if not reader_order or set(predictions)!=set(reader_order):raise ValueError('frozen reader prediction inventory differs')
    envelope=np.full(len(weights),.5,dtype=np.float64)
    for reader in reader_order:
        score=np.asarray(predictions[reader])
        auc=weighted_auc_batch(y,score,group_index,weights,groups=groups)
        envelope=np.maximum(envelope,np.maximum(auc,1-auc))
    return envelope


def recovery_intervals(values,eligible_arms,draw_indices,*,bootstrap_draws=BOOTSTRAPS,groups=GROUPS):
    """Return all 756 possible pair slots, null where an arm was not attacked."""
    eligible=frozenset(eligible_arms)
    if not eligible.issubset(ARMS) or draw_indices.shape!=(bootstrap_draws,2,3) or np.any((draw_indices<0)|(draw_indices>2)):
        raise ValueError('eligible family or coupled mechanism draw plan differs')
    summarized={}
    for arm in eligible:
        for seed in (0,1):
            for view in VIEWS:
                draws=[np.asarray(values[(arm,seed,d if arm in STOCHASTIC else 0,view)]) for d in range(3)]
                if any(x.dtype!=np.float64 or x.shape!=(bootstrap_draws+1,) or not np.isfinite(x).all() for x in draws):raise ValueError('fixed reader envelope vector differs')
                for d,source in enumerate(draws):summarized[(f's{seed}_d{d}',arm,view)]=source
                matrix=np.stack(draws,axis=1)
                sampled=matrix[1:][np.arange(bootstrap_draws)[:,None],draw_indices[:,seed,:]].mean(1)
                summarized[(f's{seed}_mean',arm,view)]=np.r_[matrix[0].mean(),sampled]
        for view in VIEWS:summarized[('mean',arm,view)]=(summarized[('s0_mean',arm,view)]+summarized[('s1_mean',arm,view)])/2
    group_critical=float(student_t.ppf(1-ALPHA/(2*K),groups-1))
    draw_critical=float(student_t.ppf(1-ALPHA/(2*K),2))
    out={}
    for summary in RECOVERY_SUMMARIES:
        crit=group_critical if '_d' in summary else draw_critical
        for first,second in combinations(ARMS,2):
            for view in VIEWS:
                name=f'{summary}:recovery:{first}:{second}:{view}:auc'
                if first not in eligible or second not in eligible:
                    out[name]={'point':None,'se':None,'lower':None,'upper':None,'status':'NOT_ESTIMABLE_DEPENDENCY'}
                    continue
                diff=summarized[(summary,first,view)]-summarized[(summary,second,view)]
                point=float(diff[0]);replicates=diff[1:]
                se=0. if np.all(replicates==point) else float(np.std(replicates,ddof=1))
                out[name]={'point':point,'se':se,'lower':point-crit*se,'upper':point+crit*se,
                           'status':'BOUND_ONLY','df':groups-1 if '_d' in summary else 2,'K':K}
    if len(out)!=756:raise AssertionError('fixed recovery family length changed')
    return out,summarized
