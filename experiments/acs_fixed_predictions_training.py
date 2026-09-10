"""Explicit fixed-anchor/A-only study. Historical trainers remain unchanged."""
from __future__ import annotations
import copy, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from experiments import acs_bottleneck_training as old
from experiments import acs_coalition_training as hist
from experiments.acs_transfer_heads import _network, metrics, _state_hash
from experiments.acs_transfer_data import write_json, array_hash

ARMS=('A0','L025','L20','J')
COEF={'A0':(0.,0.,0.),'L025':(-.1,-.025,0.),'L20':(-.1,-.2,0.),'J':(-.1,0.,-.1)}
class Auxiliary(nn.Module):
    def __init__(self, pre, original):
        super().__init__()
        self.branch=hist.PurposeBranch(original,hist.PURPOSE_TASKS['A'])
        for k in ('mean','scale'):
            self.register_buffer('input_'+k,torch.tensor(pre[k],dtype=torch.float64))
            assert torch.equal(self.get_buffer('input_'+k),original['input_'+k])
    def standardize(self,p):
        return torch.tensor((np.asarray(p,np.float64)-self.input_mean.numpy())/self.input_scale.numpy(),dtype=torch.float32)
    def forward(self,x):
        h=self.branch.mapper(x)
        return h,{t:head(h) for t,head in self.branch.heads.items()}
    def freeze(self):return self.eval().requires_grad_(False)


def make_observers(seed,augmented=True):
    dims={'A':20 if augmented else 4,'B':2,'AB':22 if augmented else 6}
    obs=nn.ModuleDict();meta={}
    for role,k in hist.ROLE_SCHEMA.items():
        v,t=role.split('__');s=20260911+100*seed+10*hist.VIEW_INDEX[v]+hist.TARGET_ORDER.index(t)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(s);obs[role]=_network(dims[v],[64,32],k)
        meta[role]={'seed':s,'input_dim':dims[v],'classes':k,'parameters':sum(p.numel() for p in obs[role].parameters()),'initial_sha256':_state_hash(obs[role])}
    return obs,meta


def tensor_wires(h,anchors):
    a=anchors['A'] if h is None else torch.cat((anchors['A'],h),1)
    b=anchors['B'];return {'A':a,'B':b,'AB':torch.cat((a,b),1)}


def array_wires(aux,anchors,model=None):
    """Authoritative wire is float64; both historical probability columns survive."""
    a=anchors['A'].copy() if aux is None else np.column_stack((anchors['A'],np.asarray(aux,np.float64)))
    b=anchors['B'].copy();wire={'A':a,'B':b,'AB':np.column_stack((a,b))};derived=None
    if model is not None:
        with torch.no_grad():
            h=torch.tensor(aux,dtype=torch.float32);probs=[]
            for head in model.branch.heads.values():
                p=torch.sigmoid(head(h));probs.append(torch.cat((1-p,p),1).numpy().astype(np.float64))
        d=np.column_stack((anchors['A'],*probs));derived={'A':d,'B':b.copy(),'AB':np.column_stack((d,b))}
    assert all(x.dtype==np.float64 for x in wire.values())
    return wire,derived


def source_components(logits,source):
    _,tasks,support=old.masked_source_bce(logits,{t:source[t] for t in hist.PURPOSE_TASKS['A']})
    return sum(.25*tasks[t] for t in hist.PURPOSE_TASKS['A']),tasks,support


def components(model,observers,x,anchors,source,attrs,priors):
    h,logits=model(x);base,tasks,support=source_components(logits,source)
    protection,ce,known=hist.observer_losses(observers,tensor_wires(h,anchors),{**source,**attrs},priors)
    return {'source':base,**protection},{'tasks':tasks,'support':support,'ce':ce,'observer_support':known}


def observer_step(observers,opt,wires,labels,priors,skip_b=False):
    opt.zero_grad(set_to_none=True)
    losses=[]
    for role,net in observers.items():
        v,t=role.split('__')
        if skip_b and v=='B':continue
        scores=net(wires[v].detach());y=labels[t];valid=y>=0
        losses.append(torch.nn.functional.cross_entropy(scores[valid],y[valid]) if valid.any() else scores.sum()*0.)
    # Always original nine-role denominator, including when exact B trajectory is aliased.
    loss=torch.stack(losses).sum()/9
    assert torch.isfinite(loss);loss.backward();opt.step()


def forward_step(model,observers,opt,x,anchors,source,attrs,priors,arm):
    opt.zero_grad(set_to_none=True);observers.zero_grad(set_to_none=True);observers.requires_grad_(False)
    try:
        if arm=='A0':loss=source_components(model(x)[1],source)[0]
        else:
            c,_=components(model,observers,x,anchors,source,attrs,priors)
            loss=c['source']
            for k,w in zip(('individual','extra_local','coalition'),COEF[arm]):
                if w:loss=loss+w*c[k]
        assert torch.isfinite(loss);loss.backward();opt.step()
    finally:observers.requires_grad_(True)


def diagnostic(model,obs,x,a,s,y,priors,arm):
    before=old.tree_digest((model.state_dict(),obs.state_dict()));rng=torch.get_rng_state().clone()
    c,d=components(model,obs,x,a,s,y,priors);params=list(model.parameters());groups={'mapper':list(model.branch.mapper.parameters()),'heads':list(model.branch.heads.parameters())}
    out={'losses':{k:float(v.detach()) for k,v in c.items()},'task_losses':{k:float(v.detach()) for k,v in d['tasks'].items()},'observer_ce':{k:float(v.detach()) for k,v in d['ce'].items()},'coefficients':dict(zip(c,(1.,*COEF[arm]))),'gradients':{}}
    for name,p in groups.items():
        out['gradients'][name]={}
        for k,v in c.items():
            g=torch.autograd.grad(v,p,retain_graph=True,allow_unused=True)
            norm=math.sqrt(sum(float((t.double()**2).sum()) for t in g if t is not None))
            out['gradients'][name][k]={'raw_norm':norm,'applied_norm':abs(out['coefficients'][k])*norm}
    # B-only objective cannot reach auxiliary parameters.
    b_loss=sum(d['ce'][k] for k in d['ce'] if k.startswith('B__'))
    bg=torch.autograd.grad(b_loss,params,allow_unused=True)
    assert all(g is None or not torch.count_nonzero(g) for g in bg)
    assert before==old.tree_digest((model.state_dict(),obs.state_dict())) and torch.equal(rng,torch.get_rng_state())
    out.update(B_gradient_zero=True,immutable=True);return out


def copy_b(target,to,source,so):
    """Alias identical stationary B tensors AND each Adam slot; no extra fitting."""
    for role in target:
        if not role.startswith('B__'):continue
        target[role].load_state_dict(source[role].state_dict())
        for p,q in zip(target[role].parameters(),source[role].parameters()):
            if q in so.state:to.state[p]=copy.deepcopy(so.state[q])
    assert all(_state_hash(target[r])==_state_hash(source[r]) for r in target if r.startswith('B__'))


def train_seed(pca,source_y,attr_y,pre,original,anchors,teacher,seed,out,miniature=False):
    """No dataframe, reserved label, downstream score or evaluation array accepted."""
    start=time.perf_counter();out=Path(out);out.mkdir(parents=True,exist_ok=False)
    source=old._labels(source_y,len(pca),hist.SOURCE_SCHEMA,'source');attrs=old._labels(attr_y,len(pca),hist.ATTRIBUTE_SCHEMA,'attribute')
    model=Auxiliary(pre,original);assert sum(p.numel() for p in model.parameters())==3186
    x=model.standardize(pca);a={k:torch.tensor(v,dtype=torch.float32) for k,v in anchors.items()}
    priors=hist.label_priors(source,attrs);opt=old._adam(model.parameters())
    epochs=(1,1,2) if miniature else (60,20,80);bs=16 if miniature else 256
    schedules={n:old._orders(len(x),e,1280000+100*seed+offset) for n,e,offset in zip(('source','observer','continuation'),epochs,(0,100,200))}
    ix0=schedules['source'][0][0][:bs];empty=nn.ModuleDict()
    with torch.no_grad():err=model(x)[0].numpy().astype(float)-np.asarray(pca)[:,:16]
    assert np.allclose(err,0,atol=1e-5,rtol=1e-5) or np.allclose(model(x)[0].detach().numpy(),np.asarray(pca)[:,:16],atol=1e-5,rtol=1e-5)
    def save(path,m,o,mo,oo,phase,steps,os):
        torch.save({'model_state':m.state_dict() if m is not None else None,'adversary_state':o.state_dict(),'mapper_optimizer':mo.state_dict() if mo else None,'adversary_optimizer':oo.state_dict() if oo else None,'torch_rng_state':torch.get_rng_state(),'phase':phase,'counts':{'mapper_steps':steps,'observer_steps':os},'schedules':{k:{'sha256':v[1],'epochs':len(v[0])} for k,v in schedules.items()}},path)
    save(out/'initial.pt',model,empty,opt,None,'initial',0,0)
    steps=0;curve=[]
    for epoch,order in enumerate(schedules['source'][0],1):
        for start0 in range(0,len(order),bs):
            ix=order[start0:start0+bs];forward_step(model,empty,opt,x[ix],{k:v[ix] for k,v in a.items()},old._batch(source,ix),old._batch(attrs,ix),priors,'A0');steps+=1
        if epoch%5==0 or epoch==epochs[0]:
            with torch.no_grad():base,tasks,_=source_components(model(x)[1],source)
            curve.append({'epoch':epoch,'tasks':{k:float(v) for k,v in tasks.items()},'trainable_source':float(base)})
    obs,ometa=make_observers(seed);oo=old._adam(obs.parameters())
    save(out/'observer_initial.pt',model,obs,opt,oo,'observer_initial',steps,0)
    forward_hash=old.tree_digest((model.state_dict(),opt.state_dict()));os=0
    for order in schedules['observer'][0]:
        for start0 in range(0,len(order),bs):
            ix=order[start0:start0+bs]
            with torch.no_grad():w=tensor_wires(model(x[ix])[0],{k:v[ix] for k,v in a.items()})
            observer_step(obs,oo,w,{**old._batch(source,ix),**old._batch(attrs,ix)},priors);os+=1
    assert forward_hash==old.tree_digest((model.state_dict(),opt.state_dict()))
    save(out/'fork.pt',model,obs,opt,oo,'fork',steps,os)
    fork=old.tree_digest((model.state_dict(),obs.state_dict(),opt.state_dict(),oo.state_dict()));arms={};metadata={}
    rng=torch.get_rng_state().clone();b_path=[]
    for arm in ARMS:
        m,o=copy.deepcopy(model),copy.deepcopy(obs);mo=old._adam(m.parameters());ao=old._adam(o.parameters());mo.load_state_dict(copy.deepcopy(opt.state_dict()));ao.load_state_dict(copy.deepcopy(oo.state_dict()));torch.set_rng_state(rng)
        assert fork==old.tree_digest((m.state_dict(),o.state_dict(),mo.state_dict(),ao.state_dict()))
        dest=out/arm;dest.mkdir();save(dest/'fork.pt',m,o,mo,ao,'fork',steps,os)
        diag=diagnostic(m,o,x[ix0],{k:v[ix0] for k,v in a.items()},old._batch(source,ix0),old._batch(attrs,ix0),priors,arm);cur=[];ns,no=steps,os
        for epoch,order in enumerate(schedules['continuation'][0],1):
            for start0 in range(0,len(order),bs):
                ix=order[start0:start0+bs];aa={k:v[ix] for k,v in a.items()};s=old._batch(source,ix);y=old._batch(attrs,ix)
                with torch.no_grad():w=tensor_wires(m(x[ix])[0],aa)
                for _ in range(3):observer_step(o,ao,w,{**s,**y},priors,skip_b=arm!='A0');no+=1
                if arm=='A0':
                    b_path.append({r:{k:v.detach().clone() for k,v in o[r].state_dict().items()} for r in o if r.startswith('B__')})
                else:
                    for r,state in b_path[ns-steps].items():o[r].load_state_dict(state)
                forward_step(m,o,mo,x[ix],aa,s,y,priors,arm);ns+=1
            if epoch%5==0 or epoch==epochs[2]:
                with torch.no_grad():c,d=components(m,o,x,a,source,attrs,priors)
                cur.append({'epoch':epoch,'components':{k:float(v) for k,v in c.items()},'source_tasks':{k:float(v) for k,v in d['tasks'].items()}})
        if arm!='A0':copy_b(o,ao,arms['A0'][1],arms['A0'][2])
        save(dest/'final.pt',m,o,mo,ao,'final',ns,no)
        metadata[arm]={'fork_sha256':fork,'counts':{'mapper_steps':ns,'observer_steps':no},'B_continuation_alias':None if arm=='A0' else 'A0','curve':cur,'fork_diagnostic':diag,'final_diagnostic':diagnostic(m,o,x[ix0],{k:v[ix0] for k,v in a.items()},old._batch(source,ix0),old._batch(attrs,ix0),priors,arm)}
        arms[arm]=(m.freeze(),o.eval(),ao)
    torch.save(b_path,out/'B_trajectory.local.pt')
    # Static observer controls start BEFORE observer warmup, never a warmed B state.
    for name,aux in [('H',None),('E',torch.tensor(teacher,dtype=torch.float32))]:
        o,meta=make_observers(seed,name=='E');ao=old._adam(o.parameters());dest=out/name;dest.mkdir();save(dest/'initial.pt',None,o,None,ao,'static_initial',0,0)
        w=tensor_wires(aux,a);no=0
        for phase,repeats in [('observer',1),('continuation',3)]:
            for order in schedules[phase][0]:
                for start0 in range(0,len(order),bs):
                    ix=order[start0:start0+bs]
                    for _ in range(repeats):observer_step(o,ao,{k:v[ix] for k,v in w.items()},{**old._batch(source,ix),**old._batch(attrs,ix)},priors,skip_b=True);no+=1
        copy_b(o,ao,arms['A0'][1],arms['A0'][2]);save(dest/'final.pt',None,o,None,ao,'static_final',0,no)
        metadata[name]={'observer_metadata':meta,'counts':{'mapper_steps':0,'observer_steps':no},'B_full_trajectory_alias':'A0','stationary_input_history':True};arms[name]=(None,o.eval(),ao)
    b={name:{r:_state_hash(o[r]) for r in o if r.startswith('B__')} for name,(_,o,_) in arms.items()};assert all(v==b['A0'] for v in b.values())
    cov=source_y['public_coverage'];valid=cov>=0;constant=.5*metrics(cov[valid],anchors['B'][valid],2)['log_loss'] if valid.any() else 0.
    record={'seed':seed,'miniature':miniature,'initial_parity':{'max_abs':float(abs(err).max()),'rms':float(np.sqrt((err**2).mean()))},'source_curve':curve,'fixed_coverage_constant':constant,'constant_loss_rule':'historical full-vector scorer, clip each class to[1e-12,1], normalize clipped rows as original scorer; omitted from backward arithmetic','priors':priors,'observer_metadata':ometa,'arms':metadata,'B_final_identity':b,'schedules':{k:{'sha256':v[1],'epochs':len(v[0])} for k,v in schedules.items()},'fit_rows':len(x),'batch_size':bs,'reserved_labels_received':False,'runtime_seconds':time.perf_counter()-start}
    write_json(out/'training.json',record);return arms,record
