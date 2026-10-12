"""Independent format-based mathematics for the deterministic TCR1 service.

Implemented from pinned source laws and protocol definitions, never by calling
candidate kernels, models, metrics, selection or deployment implementations.
"""
from __future__ import annotations
import hashlib
import math
from statistics import NormalDist
import numpy as np

def softmax(a):
    a = np.asarray(a, dtype=np.float64)
    p = np.exp(a - a.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)

def fold_pipeline(pipe):
    scaler, clf = pipe.steps[0][1], pipe.steps[-1][1]
    w = np.asarray(clf.coef_, dtype=np.float64) / scaler.scale_
    b = np.asarray(clf.intercept_, dtype=np.float64) - w @ scaler.mean_
    if w.shape[0] == 1:
        w, b = np.concatenate((-w/2, w/2)), np.concatenate((-b/2, b/2))
    else:
        w, b = w-w.mean(axis=0), b-b.mean()
    return w, b, np.asarray(clf.classes_)

def encoder(state, r, x):
    import torch
    import torch.nn.functional as F
    a = torch.as_tensor(x, dtype=torch.float32)
    with torch.no_grad():
        for layer in (0, 2, 4):
            a = F.linear(a, state[f'enc.{r}.{layer}.weight'], state[f'enc.{r}.{layer}.bias'])
            if layer != 4:
                a = F.relu(a)
    return a.double().numpy()

def own_release(state, x):
    out = {}
    for r in (0, 1):
        z = encoder(state, r, x)
        h = state[f'head.{r}.weight'].double().numpy()
        b = state[f'head.{r}.bias'].double().numpy()
        c = z @ h.T + b
        c -= c.mean(axis=1, keepdims=True)
        p = softmax(c)
        d = p.argmax(axis=1)
        out.update({f'r{r+1}':z, f'c{r+1}':c, f'p{r+1}':p, f'hard{r+1}':d})
    return out

def wire(release, r):
    return np.column_stack([release[f'r{r}'], release[f'c{r}'], release[f'p{r}'], release[f'hard{r}']])

def metrics(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=np.float64)
    if y.shape != (len(p),) or p.ndim != 2 or (y<0).any() or (y>=p.shape[1]).any():
        raise ValueError('invalid task labels/probabilities')
    if not np.isfinite(p).all() or (p<0).any() or np.max(abs(p.sum(1)-1))>1e-10:
        raise ValueError('invalid probability simplex')
    d = p.argmax(axis=1)
    recalls = [float((d[y==c]==c).mean()) if (y==c).any() else 0.0 for c in range(p.shape[1])]
    truth = np.eye(p.shape[1])[y]
    return {'accuracy':float((d==y).mean()), 'nll':float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean()),
            'brier':float(((p-truth)**2).sum(1).mean()), 'balanced_accuracy':float(np.mean(recalls)),
            'per_class_recall':recalls, 'mean_confidence':float(p.max(axis=1).mean()),
            'class_counts':[int((y==c).sum()) for c in range(p.shape[1])]}

def task_rows(y, p):
    y = np.asarray(y, dtype=int)
    return np.column_stack([(p.argmax(1)==y).astype(float), -np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)), ((p-np.eye(p.shape[1])[y])**2).sum(1)])

def auc(y, score, weights=None):
    y, score = np.asarray(y,dtype=int), np.asarray(score,dtype=float)
    if set(np.unique(y)) != {0,1} or not np.isfinite(score).all():
        raise ValueError('AUC requires two classes and finite scores')
    w = np.ones(len(y)) if weights is None else np.asarray(weights,dtype=float)
    order = np.argsort(score,kind='stable')
    s, label, wt = score[order], y[order], w[order]
    starts = np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
    pos = np.add.reduceat(wt*label,starts)
    neg = np.add.reduceat(wt*(1-label),starts)
    denom = pos.sum()*neg.sum()
    if denom <= 0:
        raise ValueError('empty resampled AUC class')
    return float((pos*(np.cumsum(neg)-neg+neg/2)).sum()/denom)

def fit_fold(groups):
    return np.array([int(hashlib.sha256(('IVR1|FIT-crossfold|20261010|'+str(g)).encode()).hexdigest(),16)%3 for g in groups])

def normalized_task_violation(current, baseline):
    return max(v for a,b in zip(current,baseline) for v in ((b['accuracy']-a['accuracy'])/.010,(a['nll']-b['nll'])/.010,(a['brier']-b['brier'])/.005))

def common_select(rows, strengths, epochs, utility=False):
    configs = []
    for strength in strengths:
        for epoch in epochs:
            pair = [r for r in rows if r['strength']==strength and r['epoch']==epoch]
            if len(pair)!=2 or set(r['seed'] for r in pair)!={0,1}:
                continue
            configs.append({'strength':strength,'epoch':epoch,'violation':max(r['violation'] for r in pair),
                'nll':sum(r['nll'] for r in pair),'sensitive':sum(r['sensitive'] for r in pair)/2})
    if not configs:
        raise ValueError('no common two-seed configuration')
    tie = lambda r:(r['violation'],r['nll'],strengths.index(r['strength']),epochs.index(r['epoch']))
    fallback = min(configs,key=tie)
    feasible = [r for r in configs if r['violation']<=1]
    if not feasible:
        return fallback,fallback
    selected = min(feasible,key=(lambda r:(r['nll'],epochs.index(r['epoch']),strengths.index(r['strength']))) if utility else (lambda r:(r['sensitive'],*tie(r))))
    return selected,fallback

def squared_distance(x,y):
    return np.maximum((x*x).sum(1)[:,None]+(y*y).sum(1)[None,:]-2*x@y.T,0)

def normalization(ref,initial_std=None):
    sd = np.std(ref,axis=0)
    initial_std = sd if initial_std is None else initial_std
    floor = np.maximum(1e-12,1e-8*initial_std)
    scale = np.maximum(sd,floor)
    mean = np.mean(ref,axis=0)
    v = (ref-mean)/scale
    d = squared_distance(v,v)
    ordered = np.sort(d[np.triu_indices(len(v),1)])
    # Registered torch.median convention: lower central order statistic.
    median = max(float(ordered[(len(ordered)-1)//2]),1e-12)
    return mean,scale,median,floor

def gram_bank(x,y,ref,initial_std=None):
    mean,scale,median,_ = normalization(ref,initial_std)
    x,y = (x-mean)/scale,(y-mean)/scale
    denominators = median*np.array([.25,.5,1,2,4])**2
    return tuple(np.exp(-squared_distance(a,b)[None,:,:]/denominators[:,None,None]) for a,b in ((x,x),(y,y),(x,y)))

def core_power(kx,ky,kxy,nblocks=4):
    n = kx.shape[-1]
    offmean = lambda k:(k.sum(axis=(-1,-2))-np.trace(k,axis1=-2,axis2=-1))/(n*(n-1))
    mmd = offmean(kx)+offmean(ky)-2*offmean(kxy)
    h = kx+ky-kxy-kxy.swapaxes(-1,-2)
    variance = 4*((h.mean(-1)**2).mean(-1)-h.mean(axis=(-1,-2))**2)
    z = np.sqrt(nblocks)*mmd/np.sqrt(np.maximum(variance,1e-6))-NormalDist().inv_cdf(.95)
    power = np.array([NormalDist().cdf(float(v)) for v in z])
    return power,mmd,variance

def aligned_product(first,second,first_ids,second_ids):
    if not np.array_equal(first_ids,second_ids):
        raise ValueError('recipient alignment mismatch')
    return tuple(a*b for a,b in zip(first,second))

def fixtures():
    # Label-mediated sensitive information persists in the population while
    # each within-label empirical law is identical across sensitive arms.
    noextra = []
    for label in (0,1):
        signal = np.full((32,2),float(label))
        noextra.append(float(core_power(*gram_bank(signal,signal,signal))[0].max()))
    base = np.r_[np.zeros(16),np.ones(16)][:,None]
    reverse = 1-base
    first = gram_bank(base,base,np.vstack((base,base)))
    second = gram_bank(base,reverse,np.vstack((base,reverse)))
    ids = np.arange(64)
    pair = core_power(*aligned_product(first,second,ids,ids))[0]
    local = [float(core_power(*k)[0].max()) for k in (first,second)]
    rejected = False
    try: aligned_product(first,second,ids,ids[::-1])
    except ValueError: rejected=True
    planted_triplet = gram_bank(np.zeros((32,2)),np.ones((32,2)),np.vstack((np.zeros((32,2)),np.ones((32,2)))))
    planted = float(core_power(*planted_triplet)[0].max())
    metric = metrics(np.array([0,1]),np.array([[.8,.2],[.25,.75]]))
    metric_ok = abs(metric['brier']-(.08+.125)/2)<1e-15 and metric['accuracy']==1
    tied_auc = auc(np.array([0,1,0,1]),np.array([.2,.2,.8,.8]))
    if max(noextra)>.05+1e-12 or max(local)>.05+1e-12 or pair.max()<=.99 or planted<=.99 or not rejected or not metric_ok or tied_auc!=.5:
        raise ValueError('independent correctness fixture failure')
    return {'within_label_identical_power':noextra,'XOR_local_power':local,'XOR_pair_power':float(pair.max()),
            'alignment_rejected':rejected,'within_label_planted_power':planted,'summed_brier_fixture':True,'tied_auc':tied_auc}

def coverage(y1,y2,sex,groups,mask=None):
    mask = np.ones(len(groups),dtype=bool) if mask is None else mask
    tables = {}
    for name,cols in [('income',[y1]),('occupation',[y2]),('pair',[y1,y2])]:
        labels = np.column_stack(cols)
        rows = []
        for vals in sorted(set(map(tuple,labels[mask]))):
            choose = mask & (labels==vals).all(1)
            counts = [len(set(groups[choose & (sex==s)])) for s in (0,1)]
            rows.append({'labels':list(map(int,vals)),'rows':int(choose.sum()),'mass':float(choose.sum()/mask.sum()),
                         'distinct_groups_by_sex':counts,'supported':min(counts)>=32})
        covered = sum(r['mass'] for r in rows if r['supported'])
        tables[name] = {'strata':rows,'covered_mass':covered,'uncovered_mass':1-covered}
    return tables

def group_bootstrap_weights(groups,replicates=2048,seed=20261012):
    unique,inverse = np.unique(groups,return_inverse=True)
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        counts = np.bincount(rng.integers(0,len(unique),size=len(unique)),minlength=len(unique))
        yield counts[inverse]

def simultaneous_interval(point,replicates,k,groups):
    from scipy.stats import t
    se = float(np.std(replicates,ddof=1))
    critical = float(t.ppf(1-.05/(2*k),df=groups-1))
    return {'point':float(point),'se':se,'lower':float(point-critical*se),'upper':float(point+critical*se),'critical':critical}
