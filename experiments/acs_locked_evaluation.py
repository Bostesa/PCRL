"""Pure inference/scoring primitives for the locked ACS household evaluation.

No fitting, optimization, candidate selection, calibration or model refresh.
Identifiers and per-person products must remain local.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.acs_transfer_data import CovariatePreprocessor
from experiments.acs_transfer_heads import load_candidate, metrics

SALT = 'PCRL_LOCKED_EVAL_20260910_V1'
NAMESPACE = 'ACS/2018/1-Year/CA/person'
ARMS = ('H','E','A0','L025','L20','J')
SCOPES = ('standard_independent','expanded_independent','expanded_catchup')
TASKS = ('income_binary','civilian_at_work','public_coverage','same_residence','commute_over20')
ROLES = {'A':('public_coverage','commute_over20','SEX','RAC1P'),
         'B':('income_binary','civilian_at_work','same_residence','SEX','RAC1P'),
         'AB':('SEX','RAC1P')}


def encoding(*parts):
    return json.dumps(list(parts),ensure_ascii=True,separators=(',',':'))


def canonical_people(frame):
    f=frame.copy()
    if '_raw_row' not in f:f['_raw_row']=np.arange(len(f),dtype=np.int64)
    age=pd.to_numeric(f.AGEP,errors='raise');weight=pd.to_numeric(f.PWGTP,errors='raise')
    f=f.loc[age.between(19,34)&age.eq(np.floor(age))&weight.gt(0)].copy()
    for key in ('SERIALNO','SPORDER'):
        if f[key].isna().any() or not f[key].map(lambda v:isinstance(v,str) and bool(v)).all():
            raise ValueError('Missing or non-string person/group key: '+key)
    dup=f.duplicated(['SERIALNO','SPORDER'],keep=False)
    for _,g in f.loc[dup].groupby(['SERIALNO','SPORDER'],sort=False):
        if len(g.drop(columns='_raw_row').drop_duplicates())!=1:
            raise ValueError('Conflicting duplicate person keys')
    f=f.drop_duplicates(['SERIALNO','SPORDER'],keep='first')
    f['household_id']=[encoding(NAMESPACE,v) for v in f.SERIALNO]
    f['person_id']=[encoding(NAMESPACE,h,p) for h,p in zip(f.SERIALNO,f.SPORDER)]
    return f.reset_index(drop=True)


def household_order(ids):
    return sorted(set(ids),key=lambda h:(hashlib.sha256(encoding(SALT,h).encode('utf-8')).digest(),h))


def select_households(frame,excluded,cap=30000):
    remainder=frame.loc[~frame.household_id.isin(excluded)]
    sizes=remainder.groupby('household_id').size();chosen=[];total=0
    for h in household_order(sizes.index):
        if total+int(sizes[h])>cap:break
        chosen.append(h);total+=int(sizes[h])
    rank={h:i for i,h in enumerate(chosen)}
    result=remainder.loc[remainder.household_id.isin(chosen)].copy()
    result['_household_order']=result.household_id.map(rank)
    return result.sort_values(['_household_order','person_id']).reset_index(drop=True)


def frozen_preprocessor(metadata):
    p=CovariatePreprocessor()
    p.numeric=metadata['numeric'];p.categories=metadata['categories'];p.feature_names=metadata['feature_names']
    return p


def project_predict(model,raw,columns=None):
    x=np.asarray(raw,dtype=np.float64)
    if columns is not None:x=x[:,columns]
    # Projection is deliberately before the candidate's own saved transform.
    return model.predict_proba(np.ascontiguousarray(x))


def losses(y,probabilities):
    y=np.asarray(y);p=np.asarray(probabilities,dtype=np.float64)
    if p.ndim!=2 or len(p)!=len(y):raise ValueError('Probability/label shape mismatch')
    if not np.isfinite(p).all() or (p<0).any() or (p>1+1e-8).any():raise ValueError('Nonfinite/invalid model output')
    if not np.allclose(p.sum(1),1,atol=1e-6,rtol=0):raise ValueError('Invalid class probability sums')
    valid=y>=0
    if (y[valid]>=p.shape[1]).any():raise ValueError('Class outside frozen schema')
    clipped=np.clip(p,1e-12,1);clipped/=clipped.sum(1,keepdims=True)
    out=np.full(len(y),np.nan)
    out[valid]=-np.log(clipped[np.flatnonzero(valid),y[valid]])
    return out


def score_predictions(y,p,weights):
    y=np.asarray(y);mask=y>=0;per_person=losses(y,p)
    if not mask.any():return {'unweighted':None,'PWGTP':None,'status':'no eligible targets'},per_person
    records={}
    for name,w in [('unweighted',None),('PWGTP',np.asarray(weights)[mask])]:
        r=metrics(y[mask],p[mask],p.shape[1],w)
        ww=np.ones(mask.sum()) if w is None else w
        conf=np.zeros((p.shape[1],p.shape[1]))
        np.add.at(conf,(y[mask],p[mask].argmax(1)),ww)
        r['confusion']=conf.tolist();r['effective_sample_size']=float(ww.sum()**2/(ww@ww))
        records[name]=r
    return records,per_person


def match(j,c,fresh_reference,panel,delta):
    both=all(z[t]<=r+.01+1e-12 for z in (j,c) for t,r in fresh_reference.items())
    differences={t:j[t]-c[t] for t in j}
    return {'both_source_feasible':both,
            'utility_close':all(abs(differences[t])<=delta+1e-12 for t in panel),
            'utility_directional':all(differences[t]<=delta+1e-12 for t in panel),
            'differences':differences}


def household_sums(differences,groups,weights):
    """differences: seed x person x contrast; weights: person x contrast.

    Missing masks must be encoded as zero weights. Missing person loss at zero
    weight is ignored; missing loss at positive weight remains unassessable.
    """
    d=np.asarray(differences,dtype=float);w=np.asarray(weights,dtype=float)
    _,g=np.unique(groups,return_inverse=True);n=int(g.max()+1) if len(g) else 0
    if d.ndim!=3 or d.shape[1:]!=w.shape or len(g)!=len(w):raise ValueError('Unaligned bootstrap inputs')
    if not np.isfinite(w).all() or (w<0).any():raise ValueError('Invalid weights')
    sums=np.zeros((d.shape[0],n,d.shape[2]));den=np.zeros((n,d.shape[2]))
    np.add.at(den,g,w)
    for s in range(len(d)):np.add.at(sums[s],g,np.where(w>0,d[s],0)*w)
    return sums,den


def bootstrap_values(sums,den,draws):
    result=np.empty((len(draws),den.shape[1]))
    for b,draw in enumerate(draws):
        multiplicity=np.bincount(draw,minlength=len(den))
        numerator=np.einsum('g,sgj->sj',multiplicity,sums)
        denominator=multiplicity@den
        with np.errstate(divide='ignore',invalid='ignore'):result[b]=(numerator/denominator).mean(0)
    return result


def intervals(original,replicates):
    point=np.asarray(original,float);bs=np.asarray(replicates,float);sd=bs.std(0,ddof=1)
    assessable=np.isfinite(point)&np.isfinite(bs).all(0)&np.isfinite(sd)&(sd>0)
    # Simultaneous family is exactly six; do not silently shrink it.
    q=None
    if len(point)==6 and assessable.all():
        q=float(np.quantile(np.max(np.abs((bs-point)/sd),axis=1),.95,method='linear'))
    rows=[]
    for j in range(len(point)):
        finite=np.isfinite(point[j]) and np.isfinite(bs[:,j]).all()
        status='assessable' if assessable[j] else 'degenerate' if finite and sd[j]==0 else 'unassessable'
        rows.append({'point':float(point[j]) if np.isfinite(point[j]) else None,'bootstrap_sd':float(sd[j]) if np.isfinite(sd[j]) else None,
                     'percentile':np.quantile(bs[:,j],[.025,.975],method='linear').tolist() if finite else None,
                     'simultaneous':[float(point[j]-q*sd[j]),float(point[j]+q*sd[j])] if q is not None else None,
                     'status':status,'bad_replicates':int((~np.isfinite(bs[:,j])).sum())})
    return {'replicates':len(bs),'quantile_method':'numpy linear (Hyndman–Fan type 7)','q':q,'contrasts':rows,
            'family_status':'assessable' if q is not None else 'six-contrast simultaneous family unavailable'}


def inference_graph(pca,anchors,teacher,checkpoint,condition):
    """All full anchor columns retained in a lossless float64 wire."""
    import torch
    aa=np.asarray(anchors['A'],dtype=np.float64);bb=np.asarray(anchors['B'],dtype=np.float64)
    ps=None;aux=None
    if condition=='E':aux=teacher.apply(pca[:,:16]).astype(np.float64)
    elif condition!='H':
        state=checkpoint['model_state']
        with torch.no_grad():
            x=torch.tensor((pca.astype(float)-state['input_mean'].numpy())/state['input_scale'].numpy(),dtype=torch.float32)
            h=torch.relu(torch.nn.functional.linear(x,state['branch.mapper.0.weight'],state['branch.mapper.0.bias']))
            h=torch.nn.functional.linear(h,state['branch.mapper.2.weight'],state['branch.mapper.2.bias']);aux=h.double().numpy();ps=[]
            for t in TASKS[:2]:
                p=torch.sigmoid(torch.nn.functional.linear(h,state[f'branch.heads.{t}.weight'],state[f'branch.heads.{t}.bias']))
                ps.append(torch.cat([1-p,p],1).double().numpy())
    a=aa if aux is None else np.column_stack([aa,aux]);wire={'A':a,'B':bb,'AB':np.column_stack([a,bb])}
    derived=None if ps is None else {'A':np.column_stack([aa,*ps]),'B':bb,'AB':np.column_stack([aa,*ps,bb])}
    assert np.array_equal(wire['A'][:,:4],aa) and np.array_equal(wire['AB'][:,-2:],bb)
    return {'wire':wire,'derived':derived}


def require_independence(lock):
    if not lock.get('independence_verified') or lock.get('sample_rows',0)==0:
        raise RuntimeError('Scientific evaluation blocked: no defensibly unused locked households')


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,allow_nan=False)+'\n'
    if path.exists():
        if path.read_text()!=text:raise RuntimeError('Refusing incompatible completed evidence: '+str(path))
        return
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(text);temporary.replace(path)
