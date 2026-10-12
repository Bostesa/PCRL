"""Deterministic metrics and paired duplicate-group conditional inference."""
from __future__ import annotations
import numpy as np
from scipy.stats import t


def auc(y,score,weights=None):
    return float(bootstrap_auc(y,score,np.ones((1,len(y))) if weights is None else np.asarray(weights)[None,:])[0])


def bootstrap_auc(y,score,weights):
    y=np.asarray(y);s=np.asarray(score,dtype=np.float64);w=np.asarray(weights,dtype=np.float64)
    if y.ndim!=1 or s.shape!=y.shape or not np.isin(y,[0,1]).all() or np.unique(y).size!=2:raise ValueError('AUC requires aligned scores and both binary classes')
    if w.ndim!=2 or w.shape[1]!=len(y) or not np.isfinite(s).all() or not np.isfinite(w).all() or np.any(w<0):raise ValueError('invalid score or bootstrap weight')
    order=np.argsort(s,kind='stable');sorted_s=s[order];starts=np.r_[0,np.flatnonzero(sorted_s[1:]!=sorted_s[:-1])+1]
    pos=np.add.reduceat(w[:,order]*(y[order]==1),starts,axis=1);neg=np.add.reduceat(w[:,order]*(y[order]==0),starts,axis=1)
    denom=pos.sum(1)*neg.sum(1)
    if np.any(denom<=0):raise ValueError('bootstrap draw lacks both SEX classes')
    return (pos*(np.cumsum(neg,axis=1)-.5*neg)).sum(1)/denom


def group_bootstrap(unit,replicates,seed):
    g=np.asarray(unit);unique,inverse=np.unique(g,return_inverse=True);rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(unique),size=(replicates,len(unique)))
    count=np.zeros((replicates,len(unique)),dtype=np.int32)
    np.add.at(count,(np.arange(replicates)[:,None],draws),1)
    return count[:,inverse]


def task_vectors(y,probability,class_order=None):
    y=np.asarray(y);p=np.asarray(probability,dtype=np.float64)
    if p.ndim!=2 or y.shape!=(len(p),) or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1,atol=1e-10,rtol=0):raise ValueError('task probability convention differs')
    order=np.arange(p.shape[1]) if class_order is None else np.asarray(class_order)
    if order.shape!=(p.shape[1],) or len(np.unique(order))!=len(order) or not np.isin(y,order).all():raise ValueError('task class order differs')
    target=(y[:,None]==order).astype(float);correct=(order[p.argmax(1)]==y).astype(float)
    return {'accuracy':correct,'nll':-(target*np.log(np.maximum(p,1e-12))).sum(1),'brier':np.square(p-target).sum(1)}


def task_metrics(y,probability,class_order=None):
    y=np.asarray(y);p=np.asarray(probability,dtype=float);order=np.arange(p.shape[1]) if class_order is None else np.asarray(class_order)
    v=task_vectors(y,p,order);decision=order[p.argmax(1)]
    recall=[float(np.mean(decision[y==k]==k)) if np.any(y==k) else 0. for k in order]
    prior=np.array([(y==k).mean() for k in order]);constant=np.tile(prior,(len(y),1));cv=task_vectors(y,constant,order)
    return {**{k:float(a.mean()) for k,a in v.items()},'balanced_accuracy':float(np.mean(recall)),'per_class_recall':recall,
            'class_order':order.tolist(),'class_counts':[int((y==k).sum()) for k in order],
            'confidence':float(p.max(1).mean()),'constant_prediction':{k:float(a.mean()) for k,a in cv.items()},
            'gain_over_constant':{'accuracy':float(v['accuracy'].mean()-cv['accuracy'].mean()),'nll':float(cv['nll'].mean()-v['nll'].mean()),'brier':float(cv['brier'].mean()-v['brier'].mean())}}


def family_status(point_eligible,complete):
    return 'incomplete' if not complete else 'eligible' if point_eligible else 'diagnostic_ineligible'


def interval(point,replicates,K,groups,alpha=.05):
    a=np.asarray(replicates,dtype=np.float64)
    if a.ndim!=1 or len(a)<2048 or not np.isfinite(a).all() or K<1 or groups<2:raise ValueError('interval registry/bootstrap invalid')
    se=float(a.std(ddof=1));critical=float(t.ppf(1-alpha/(2*K),df=groups-1));half=critical*se
    return {'point':float(point),'bootstrap_se':se,'lower':float(point-half),'upper':float(point+half),'K':int(K),'df':int(groups-1),'replicates':len(a),'critical':critical,'construction':'paired-group bootstrap SE times Bonferroni t; approximate conditional-development coverage'}


def point_utility(candidate,reference):
    return {'accuracy_loss':float(reference['accuracy']-candidate['accuracy']),'nll_excess':float(candidate['nll']-reference['nll']),'brier_excess':float(candidate['brier']-reference['brier'])}
