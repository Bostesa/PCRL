"""Frozen task teachers, nested input codes and a task-directed action library."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import joblib
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit,logit
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from .data import RuntimeInputs,household_roles,array_hash
from .audits import fit_predictor_slate,balanced_person_weights,loss_scores

CLIP=1e-5


def clipped(p):
    p=np.asarray(p,float)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Invalid task probabilities before declared clipping')
    return np.clip(p,CLIP,1-CLIP)


def quantile_cuts(x,n):
    x=np.asarray(x,float)
    cuts=np.unique(np.quantile(x,np.arange(1,n)/n))
    return cuts[(cuts>x.min())&(cuts<x.max())]


def projection(x):
    a=np.cos(np.arange(1,np.asarray(x).shape[1]+1));a/=np.linalg.norm(a)
    return np.asarray(x)@a


def two_centers(x,seed):
    x=np.asarray(x,float)
    if not len(x):raise ValueError('Cannot cluster an empty training set')
    axis=projection(x);first=x[np.argmin(axis)];second=x[np.argmax(axis)]
    if np.array_equal(first,second):second=x[np.argmax(np.sum((x-first)**2,axis=1))]
    if np.array_equal(first,second):return np.stack((first,second))
    km=KMeans(n_clusters=2,init=np.stack((first,second)),n_init=1,random_state=seed).fit(x)
    centers=km.cluster_centers_
    order=sorted(range(2),key=lambda i:tuple(centers[i]))
    return centers[order]


def nearest(x,centers):
    return np.argmin(np.sum((np.asarray(x)[:,None,:]-np.asarray(centers)[None,:,:])**2,axis=2),axis=1)


@dataclass
class Codebook:
    cuts:np.ndarray
    task_medians:np.ndarray
    task_tie_medians:np.ndarray
    risk_centers:np.ndarray
    diagnostics:dict

    @classmethod
    def fit(cls,r,x,risk,seed):
        r=np.asarray(r,float);x=np.asarray(x,float);risk=np.asarray(risk,float)
        if r.ndim!=1 or x.shape!=(len(r),32) or risk.ndim!=2 or len(risk)!=len(r):
            raise ValueError('Aligned residual, PCA32 and predicted-risk vectors required')
        if not all(np.isfinite(a).all() for a in (r,x,risk)):raise ValueError('Nonfinite codes')
        cuts=quantile_cuts(r,32);t=np.searchsorted(cuts,r,side='right');n=len(cuts)+1
        med=np.empty(n);tie=np.empty(n);centers=[];fallback=[];generic=[];counts=[]
        global_centers=two_centers(risk,202609210+seed)
        px=projection(x)
        for parent in range(n):
            rows=np.flatnonzero(t==parent);counts.append(len(rows))
            med[parent]=float(np.median(r[rows])) if len(rows) else 0.
            ties=rows[r[rows]==med[parent]]
            tie[parent]=float(np.median(px[ties])) if len(ties) else 0.
            if len(np.unique(px[ties]))>1:generic.append(parent)
            if len(rows)<40:centers.append(global_centers);fallback.append(parent)
            else:centers.append(two_centers(risk[rows],202609210+100*seed+parent+1))
        obj=cls(cuts,med,tie,np.asarray(centers),{
            'nominal_T0':n,'nominal_refined':2*n,'training_parent_counts':counts,
            'task_tie_projection_cells':generic,'risk_global_fallback_cells':fallback,
            'risk_initialization':'fixed projection extrema; farthest-vector repair only if extrema identical',
            'risk_cluster_scaling':'unscaled predicted SEX2 + RAC1P9 probabilities',
            'duplicate_coarse_cuts_merged':31-len(cuts)})
        assigned=obj.assign(r,x,risk)
        obj.diagnostics['realized_training_states']={k:len(np.unique(v)) for k,v in assigned.items()}
        obj.diagnostics['risk_constant_predictor']=bool(np.max(np.ptp(risk,axis=0))<1e-12)
        return obj

    def assign(self,r,x,risk):
        r=np.asarray(r,float);x=np.asarray(x,float);risk=np.asarray(risk,float)
        if not all(np.isfinite(a).all() for a in (r,x,risk)):raise ValueError('Nonfinite code inputs')
        t=np.searchsorted(self.cuts,r,side='right');p=projection(x)
        task=(r>self.task_medians[t])|((r==self.task_medians[t])&(p>self.task_tie_medians[t]))
        child=np.empty(len(t),dtype=np.int64)
        for parent in range(self.n_states('T0')):
            rows=t==parent
            child[rows]=nearest(risk[rows],self.risk_centers[parent])
        result={'T0':t.astype(np.int64),'Ttask':2*t+task.astype(np.int64),'Trisk':2*t+child}
        for name in ('Ttask','Trisk'):
            if not np.array_equal(result[name]//2,result['T0']):raise AssertionError('Refinement lost parent')
        return result

    def n_states(self,family):return (len(self.cuts)+1)*(1 if family=='T0' else 2)

    def parents(self,family):
        return np.arange(self.n_states('T0')) if family=='T0' else np.repeat(np.arange(self.n_states('T0')),2)


@dataclass
class RiskPredictor:
    mean:np.ndarray
    scale:np.ndarray
    models:dict
    classes:dict

    @classmethod
    def fit(cls,inputs,protected,weights):
        x=inputs.features();mean=x.mean(0);scale=x.std(0);scale[scale<1e-12]=1
        xn=(x-mean)/scale;models={};classes={}
        for name,k in [('SEX',2),('RAC1P',9)]:
            y=np.asarray(protected[name]);ok=y>=0;observed=np.unique(y[ok])
            if not len(observed):raise ValueError(f'No {name} risk training labels')
            classes[name]=observed.tolist()
            if len(observed)==1:models[name]=int(observed[0])
            else:
                model=LogisticRegression(C=1.,max_iter=2000)
                model.fit(xn[ok],y[ok],sample_weight=balanced_person_weights(np.asarray(weights)[ok]))
                models[name]=model
        return cls(mean,scale,models,classes)

    def probabilities(self,inputs):
        x=(inputs.features()-self.mean)/self.scale;out={}
        for name,k in [('SEX',2),('RAC1P',9)]:
            m=self.models[name];p=np.zeros((len(x),k))
            if isinstance(m,int):p[:,m]=1.
            else:p[:,m.classes_]=m.predict_proba(x)
            out[name]=(1-k*1e-8)*p+1e-8
        return out

    def predict(self,inputs):
        p=self.probabilities(inputs);return np.column_stack((p['SEX'],p['RAC1P']))


@dataclass
class ServicePartitions:
    centers:np.ndarray
    b_median:float

    @classmethod
    def fit(cls,ha,hb,seed,n_local=2):
        ha,hb=np.asarray(ha),np.asarray(hb)
        if ha.ndim!=2 or ha.shape[1]!=4 or hb.shape!=(len(ha),2):raise ValueError('Service ownership mismatch')
        centers=KMeans(n_clusters=n_local,n_init=10,random_state=202609210+seed).fit(ha[:,[1,3]]).cluster_centers_
        centers=centers[sorted(range(n_local),key=lambda i:tuple(centers[i]))]
        return cls(centers,float(np.median(hb[:,1])))

    def assign(self,ha,hb):
        ha,hb=np.asarray(ha),np.asarray(hb)
        if ha.shape[1]!=4 or hb.shape!=(len(ha),2):raise ValueError('Service ownership mismatch')
        a=nearest(ha[:,[1,3]],self.centers)
        return a,2*a+(hb[:,1]>self.b_median).astype(np.int64)


def fit_dictionary(p,b,weights,max_actions=17):
    p,b=clipped(p),clipped(b);r=logit(p)-logit(b);w=balanced_person_weights(weights)
    cuts=quantile_cuts(r,max_actions-1) if max_actions>2 else np.array([])
    groups=np.searchsorted(cuts,r,side='right');offsets=[0.];records=[]
    for group in range(len(cuts)+1):
        ix=groups==group
        if not ix.any():continue
        base=logit(b[ix]);target=p[ix];wg=w[ix]
        def stationary(a):return float(np.dot(wg,expit(base+a)-target))
        if stationary(-12)>=0:a=-12.;status='lower_bound'
        elif stationary(12)<=0:a=12.;status='upper_bound'
        else:a=float(brentq(stationary,-12.,12.,xtol=1e-13));status='interior'
        records.append({'group':group,'rows':int(ix.sum()),'offset':a,'root_residual':stationary(a),'status':status})
        if all(abs(a-v)>1e-10 for v in offsets):offsets.append(a)
    # Zero first is a stable, explicit absent-parent action. Other values retain quantile order.
    return {'offsets':np.asarray(offsets),'groups':records,'residual_cuts':cuts,
            'max_actions':max_actions,'duplicates_removed':1+len(records)-len(offsets),
            'fit_loss':'unclipped sigmoid soft-label CE; direct mechanism losses use declared probability clipping',
            'zero_action':0}


def action_probabilities(b,offsets):return clipped(expit(logit(clipped(b))[:,None]+np.asarray(offsets)[None,:]))


@dataclass
class Encoder:
    task:object
    baseline:object
    risk:RiskPredictor
    code:Codebook
    dictionaries:dict
    partitions:ServicePartitions
    metadata:dict

    def encode(self,inputs):
        if not isinstance(inputs,RuntimeInputs):raise TypeError('Encoder accepts RuntimeInputs only')
        p=clipped(self.task.predict_proba(inputs.features())[:,1])
        b=clipped(self.baseline.predict_proba(inputs.h_a)[:,1])
        r=logit(p)-logit(b);risk=self.risk.predict(inputs)
        codes=self.code.assign(r,inputs.x_a,risk)
        actions={k:action_probabilities(b,d['offsets']) for k,d in self.dictionaries.items()}
        global_offsets=np.column_stack([actions[k] if k==17 else actions[k][:,1:]
                                       for k in sorted(actions)])
        return {'p':p,'b':b,'r':r,'risk':risk,'codes':codes,
                'actions':actions,'global_offsets':global_offsets}


def fit_encoder(ctx,out_dir):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=False)
    rf=ctx['pools']['representation_fit'];anchor=ctx['anchor'];roles=household_roles(rf['households'])
    f=roles['teacher_fit'];v=roles['teacher_internal_validation'];cb=np.sort(np.r_[f,v])
    valid=rf['labels']['same_residence']>=0;f=f[valid[f]];v=v[valid[v]]
    inputs=RuntimeInputs(rf['x'],rf['ha']);features=inputs.features();teachers={};meta={}
    for name,x in [('task',features),('baseline',rf['ha'])]:
        fitted=fit_predictor_slate(x[f],rf['labels']['same_residence'][f],rf['weights'][f],
                                  x[v],rf['labels']['same_residence'][v],rf['weights'][v],
                                  2,20261000+100*anchor+(name=='baseline'),out/name)
        teachers[name]=fitted['candidates'][fitted['selection']]
        meta[name]={'selection':fitted['selection'],'slate':fitted['metadata']}
    # Protected models use all designated teacher-fit rows (not target-label complete cases).
    risk_rows=roles['teacher_fit']
    risk=RiskPredictor.fit(RuntimeInputs(rf['x'][risk_rows],rf['ha'][risk_rows]),
                          {k:rf['labels'][k][risk_rows] for k in ('SEX','RAC1P')},rf['weights'][risk_rows])
    cb_inputs=RuntimeInputs(rf['x'][cb],rf['ha'][cb])
    p=clipped(teachers['task'].predict_proba(cb_inputs.features())[:,1]);b=clipped(teachers['baseline'].predict_proba(cb_inputs.h_a)[:,1])
    r=logit(p)-logit(b);risk_values=risk.predict(cb_inputs)
    code=Codebook.fit(r,cb_inputs.x_a,risk_values,anchor)
    # Reserve the one registered recovery dictionary before outcomes. Its
    # global H-only offsets are available symmetrically from the first audit.
    # No 33-action Q is fitted unless branch A triggers.
    dictionaries={k:fit_dictionary(p,b,rf['weights'][cb],k) for k in (17,33)}
    partitions=ServicePartitions.fit(rf['ha'][cb],rf['hb'][cb],anchor)
    meta.update({'teacher_train_rows':len(f),'teacher_internal_validation_rows':len(v),
                 'codebook_rows':len(cb),'mechanism_rows':len(roles['mechanism']),
                 'teacher_frozen_without_refit':True,'risk_observed_classes':risk.classes,
                 'codebook':code.diagnostics,
                 'split_hashes':{k:array_hash(rf['raw_rows'][ix]) for k,ix in roles.items()}})
    encoder=Encoder(teachers['task'],teachers['baseline'],risk,code,dictionaries,partitions,meta)
    # Out-of-fit risk competence is evidence, kept private until the resource schedule is fixed.
    m=roles['mechanism'];pred=risk.probabilities(RuntimeInputs(rf['x'][m],rf['ha'][m]));meta['risk_out_of_fit']={}
    for name,k in [('SEX',2),('RAC1P',9)]:
        y=rf['labels'][name][m];ok=y>=0
        loss=-np.log(pred[name][np.arange(len(y))[ok],y[ok]])
        meta['risk_out_of_fit'][name]={'scores':loss_scores(loss,rf['weights'][m][ok]),
                                       'class_counts':np.bincount(y[ok],minlength=k).tolist()}
    joblib.dump(encoder,out/'encoder.joblib')
    def clean(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,Path):return str(v)
        raise TypeError(type(v).__name__)
    (out/'metadata.json').write_text(json.dumps(meta,default=clean,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(out/'split_rows.npz',**roles)
    return encoder,roles
