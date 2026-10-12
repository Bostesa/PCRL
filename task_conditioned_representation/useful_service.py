"""Useful source encoders with algebraically folded deployed logistic heads.

Encoder arithmetic remains float32, as deployed U0; head arithmetic is float64.
Protected arms update these very heads; every output reads its released feature.
"""
from __future__ import annotations
import json
import numpy as np
import torch
from .source import pinned


def fold_estimator(pipeline):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    steps=list(pipeline.named_steps.values())
    if len(steps)!=2 or not isinstance(steps[0],StandardScaler) or not isinstance(steps[1],LogisticRegression):raise ValueError('expected fitted scaler/logistic pipeline')
    scaler,lr=steps
    if not np.array_equal(lr.classes_,np.arange(len(lr.classes_))):raise ValueError('source class order differs')
    scale=np.asarray(scaler.scale_,np.float64) if scaler.with_std else np.ones(lr.coef_.shape[1])
    mean=np.asarray(scaler.mean_,np.float64) if scaler.with_mean else np.zeros(lr.coef_.shape[1])
    if scale.shape!=(16,) or not np.isfinite(scale).all() or np.any(scale<=0):raise ValueError('invalid scaler')
    w=np.asarray(lr.coef_,np.float64)/scale[None,:]
    b=np.asarray(lr.intercept_,np.float64)-w@mean
    k=len(lr.classes_)
    if k==2:
        if w.shape!=(1,16) or b.shape!=(1,):raise ValueError('binary logistic convention differs')
        # sklearn binary predict_proba uses sigmoid(d), equivalent to softmax(-d/2,+d/2).
        w=np.concatenate((-w/2,w/2));b=np.concatenate((-b/2,b/2))
        convention='binary sigmoid(d) = softmax(-d/2,+d/2); ties choose class0'
    else:
        if w.shape!=(k,16) or b.shape!=(k,) or lr.solver=='liblinear' or getattr(lr,'multi_class','auto')=='ovr':raise ValueError('nonmultinomial logistic source cannot fold to softmax')
        w=w-w.mean(0,keepdims=True);b=b-b.mean()
        convention='multinomial softmax with centered affine logits; ties choose first source class'
    return w,b,{'classes':lr.classes_.tolist(),'coef_shape':list(lr.coef_.shape),'solver':lr.solver,'multi_class':getattr(lr,'multi_class','auto'),'C':float(lr.C),'convention':convention,'scaler_n_samples_seen':np.asarray(scaler.n_samples_seen_).tolist()}


class UsefulModel(torch.nn.Module):
    def __init__(self,names):
        super().__init__()
        self.enc=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(83,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,16)) for _ in range(2)])
        self.head=torch.nn.ModuleList([torch.nn.Linear(16,k,dtype=torch.float64) for k in (2,6)])
        self.names=tuple(map(str,names));self.class_orders=[[0,1],list(range(6))];self.Ks=[2,6]

    def encode(self,i,X):return self.enc[i](X.to(dtype=torch.float32))

    def logits(self,i,z):
        l=self.head[i](z.to(dtype=torch.float64));return l-l.mean(1,keepdim=True)

    def forward(self,X):
        z=[self.encode(i,X) for i in range(2)];l=[self.logits(i,z[i]) for i in range(2)]
        return z,l,[torch.softmax(x,1) for x in l]

    def n_params(self):return {'encoder_each':sum(p.numel() for p in self.enc[0].parameters()),'heads':[sum(p.numel() for p in h.parameters()) for h in self.head],'total':sum(p.numel() for p in self.parameters())}

    def route_raw(self,raw,names):
        if tuple(map(str,names))!=self.names:raise ValueError('wrong or reordered feature schema')
        x=np.asarray(raw)
        if x.ndim!=2 or x.shape[1]!=83 or not np.issubdtype(x.dtype,np.floating) or not np.isfinite(x).all():raise ValueError('finite raw83 required')
        with torch.no_grad():
            z,l,p=self(torch.from_numpy(np.ascontiguousarray(x,dtype=np.float32)))
            out=[]
            for i in range(2):
                zz,ll,pp=(v[i].numpy() for v in (z,l,p));d=pp.argmax(1).astype(np.int64)
                out.append({'representation':zz,'centered_logits':ll,'probability':pp,'decision':d,'wire':np.concatenate((zz,ll,pp,d[:,None]),axis=1)})
        return out


def load_useful_model(manifest,seed):
    import joblib
    if seed not in (0,1):raise ValueError('seed not admitted')
    r=manifest['seeds'][str(seed)];pins=r['pins']
    for item in pins.values():pinned(item)
    complete=json.loads(pinned(pins['complete']).read_text())
    for filename,key in (('model.pt','model'),('head_0.joblib','head0'),('head_1.joblib','head1'),('record.json','record')):
        if complete['files'].get(filename)!=pins[key]['sha256']:raise ValueError('source COMPLETE binding differs')
    record=json.loads(pinned(pins['record']).read_text())
    if int(record['seed'])!=seed:raise ValueError('source seed mismatch')
    bank=json.loads(pinned(r['public_bank']).read_text());model=UsefulModel(bank['feature_names'])
    state=torch.load(pinned(pins['model']),map_location='cpu',weights_only=True)
    # Historical affine heads are discarded: the deployed useful heads are fitted logistic pipelines.
    own=model.state_dict()
    for k in own:
        if k.startswith('enc.'):own[k]=state[k]
    model.load_state_dict(own,strict=True)
    if any(not torch.equal(v,state[k]) for k,v in model.state_dict().items() if k.startswith('enc.')):raise ValueError('source encoder tensor parity failed')
    info=[]
    for i,k in enumerate(('head0','head1')):
        w,b,meta=fold_estimator(joblib.load(pinned(pins[k])))
        with torch.no_grad():model.head[i].weight.copy_(torch.from_numpy(w));model.head[i].bias.copy_(torch.from_numpy(b))
        info.append(meta)
    model.source_bindings={'seed':seed,'feature_schema_sha256':bank['feature_schema_sha256'],'files':{k:v['sha256'] for k,v in pins.items()},'heads':info}
    model.eval();return model
