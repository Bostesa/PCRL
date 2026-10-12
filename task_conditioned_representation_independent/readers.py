"""Independent reader feature algebra, restoration and refitting.

Candidate scaler pickle classes are mapped to a local state-only container;
loading saved readers never imports their scientific implementation.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np

READERS = ('LR_C1','MLP_64x32','HGB_15x200','DA_MLP_64x32','REP_LR_C1','REP_MLP_64x32','REP_HGB_15x200','OUTPUT_LR_C1','OUTPUT_HGB_15x200')

class ScaleState:
    def transform(self,x):
        return (np.asarray(x,dtype=np.float64)-self.mean_)/self.scale_

def load_saved_reader(path,validate=True):
    from joblib.numpy_pickle import NumpyUnpickler
    from joblib.numpy_pickle_utils import _validate_fileobject_and_memmap
    class IndependentUnpickler(NumpyUnpickler):
        def find_class(self,module,name):
            if module=='task_conditioned_representation' or module.startswith('task_conditioned_representation.'):
                if module=='task_conditioned_representation.attacks' and name=='FiniteScaler':
                    return ScaleState
                raise ImportError('candidate scientific pickle class refused: '+module+'.'+name)
            return super().find_class(module,name)
    filename = str(Path(path))
    with open(filename,'rb') as raw:
        with _validate_fileobject_and_memmap(raw,filename,None) as (stream,mmap):
            record = IndependentUnpickler(filename,stream,True,mmap_mode=None).load()
    if validate and (set(record)!={'model','scaler'} or not np.array_equal(record['model'].classes_,[0,1])):
        raise ValueError('invalid reader file/class order')
    return record

def features(r1,r2,view,name):
    a,b = np.asarray(r1,dtype=np.float64),np.asarray(r2,dtype=np.float64)
    if a.shape!=(len(a),21) or b.shape!=(len(a),29):
        raise ValueError('full wire shape mismatch')
    local = [a,b]
    transform = 'defense' if name.startswith('DA_') else 'representation' if name.startswith('REP_') else 'output' if name.startswith('OUTPUT_') else 'full'
    wires = []
    for r,x in enumerate(local):
        k=(2,6)[r]
        if transform=='full': z=x
        elif transform=='representation': z=x[:,:16]
        elif transform=='output': z=x[:,16+k:]
        else: z=np.column_stack((x,np.log(np.maximum(x[:,16+k:16+2*k],np.finfo(float).eps))))
        wires.append(z)
    if view in ('r1','r2'):
        return wires[int(view[-1])-1]
    if view!='pair':raise ValueError('invalid view')
    combined = np.column_stack(wires)
    if transform=='defense':
        zprod = np.einsum('ni,nj->nij',a[:,:16],b[:,:16]).reshape(len(a),256)
        lprod = np.einsum('ni,nj->nij',a[:,16:18],b[:,16:22]).reshape(len(a),12)
        combined = np.column_stack((combined,zprod,lprod))
    return combined

def split(groups):
    groups = np.asarray(groups)
    if len(groups)!=4064 or len(np.unique(groups))!=4061:raise ValueError('wrong ATT groups')
    rank = sorted((hashlib.sha256(('pcrl-method-portfolio-v1|attack-tune|20261008|'+str(int(g))).encode()).hexdigest(),int(g)) for g in np.unique(groups))
    val = np.isin(groups,[g for h,g in rank[:1015]])
    return np.flatnonzero(~val),np.flatnonzero(val)

def refit(x,y,name,seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    scaler=ScaleState()
    scaler.mean_=x.mean(axis=0)
    scaler.raw_scale_=x.std(axis=0)
    scaler.scale_=np.where(scaler.raw_scale_>0,scaler.raw_scale_,1.0)
    if 'MLP' in name:
        model=MLPClassifier(hidden_layer_sizes=(64,32),activation='relu',solver='adam',alpha=1e-4,max_iter=300,early_stopping=False,random_state=2026101205+seed)
    elif 'HGB' in name:
        model=HistGradientBoostingClassifier(max_leaf_nodes=15,max_iter=200,learning_rate=.05,early_stopping=False,random_state=2026101205+seed)
    else:model=LogisticRegression(C=1,max_iter=500,solver='lbfgs')
    model.fit(scaler.transform(x),y)
    return {'model':model,'scaler':scaler}

def predict(record,x):
    if not np.array_equal(record['model'].classes_,[0,1]):raise ValueError('wrong reader classes')
    return record['model'].predict_proba(record['scaler'].transform(x))[:,1]

def shuffled_groups(y,groups,seed):
    out=np.empty_like(y)
    buckets={}
    for g in np.unique(groups):
        rows=np.flatnonzero(groups==g)
        if len(np.unique(y[rows]))!=1:raise ValueError('duplicate labels contradict')
        buckets.setdefault(len(rows),[]).append(rows)
    rng=np.random.default_rng(seed)
    for size in sorted(buckets):
        rows=buckets[size]
        order=rng.permutation(len(rows))
        for dst,src in zip(rows,order):out[dst]=y[rows[src]]
    return out
