"""Matched ordinary online adversarial control with actual own-head wires.

Adapted from validated jcv/rgj online CE-risk machinery. No external outputs
or source logits enter the critic. Critic exposure is explicitly accounted.
"""
from dataclasses import dataclass
import random
import torch
import torch.nn.functional as F
from .models import actual_wires
from .kernels import DiagonalReference
from .trainer import seed_for

@dataclass(frozen=True)
class RawBatch:
    rows:tuple
    sex:tuple

class RawRegistry:
    def __init__(self,sex,batch_size=256):
        self.sex=tuple(map(int,sex));self.batch_size=min(batch_size,len(self.sex))
        if set(self.sex)!={0,1}:raise ValueError('both admitted sensitive groups required')
    def batch(self,rng):
        rows=tuple(rng.sample(range(len(self.sex)),self.batch_size))
        return RawBatch(rows,tuple(self.sex[i] for i in rows))
    def draw(self,rng,family=None):
        return {'critic_batches':[self.batch(rng) for _ in range(5)],'encoder_batch':self.batch(rng)}

class RawObjective:
    def __init__(self,seed,sex):
        self.references=None;self.update_enabled=True;self.critic_steps=0
        s=torch.as_tensor(sex,dtype=torch.long);prior=torch.bincount(s,minlength=2).double()/len(s)
        self.prior=prior;self.entropy=float(-(prior*prior.log()).sum())
        if self.entropy<=1e-12:raise ValueError('sensitive entropy unsupported')
        self.critics=[];self.optimizers=[]
        for view,d in enumerate((21,29,50)):
            bank=[];opts=[]
            for kind in ('linear','mlp32','mlp64'):
                torch.manual_seed(seed_for('TCR1rawcritic',seed,view,kind))
                c=(torch.nn.Linear(d,2,dtype=torch.float64) if kind=='linear' else torch.nn.Sequential(torch.nn.Linear(d,32,dtype=torch.float64),torch.nn.ReLU(),torch.nn.Linear(32,2,dtype=torch.float64)) if kind=='mlp32' else torch.nn.Sequential(torch.nn.Linear(d,64,dtype=torch.float64),torch.nn.ReLU(),torch.nn.Linear(64,64,dtype=torch.float64),torch.nn.ReLU(),torch.nn.Linear(64,2,dtype=torch.float64)))
                bank.append(c);opts.append(torch.optim.Adam(c.parameters(),lr=3e-3))
            self.critics.append(bank);self.optimizers.append(opts)

    def refresh_model_reference(self,model,x_fixed):
        with torch.no_grad():w=[v['full'] for v in actual_wires(model,x_fixed)]
        prior=self.references or [None]*3
        self.references=[DiagonalReference.fit(v,p) for v,p in zip([w[0],w[1],torch.cat(w,1)],prior)]
        return [r.snapshot() for r in self.references]

    def _views(self,model,x):
        wires=[v['full'] for v in actual_wires(model,x)]
        return [r.normalize(v) for r,v in zip(self.references,[wires[0],wires[1],torch.cat(wires,1)])]

    def _critic_update(self,model,x,batch):
        with torch.no_grad():views=self._views(model,x[list(batch.rows)])
        target=torch.tensor(batch.sex,dtype=torch.long)
        for bank,opts,view in zip(self.critics,self.optimizers,views):
            for critic,optimizer in zip(bank,opts):
                for p in critic.parameters():p.requires_grad_(True)
                loss=F.cross_entropy(critic(view),target)
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();self.critic_steps+=1

    def warm(self,model,x,registry,seed):
        rng=random.Random(seed_for('TCR1rawwarm',seed))
        for _ in range(80):self._critic_update(model,x,registry.batch(rng))
        # Eighty updates per critic; nine critics = 720 optimizer updates.
        return {'warm_updates_per_critic':80,'warm_critic_optimizer_steps':self.critic_steps}

    def loss(self,model,x,draws,family,active=(0,1)):
        if self.references is None:raise RuntimeError('raw fixed FIT reference missing')
        if self.update_enabled and torch.is_grad_enabled():
            for batch in draws['critic_batches']:self._critic_update(model,x,batch)
        batch=draws['encoder_batch'];views=self._views(model,x[list(batch.rows)]);target=torch.tensor(batch.sex,dtype=torch.long)
        terms=[];critic_ces=[]
        for bank,view in zip(self.critics,views):
            for c in bank:
                for p in c.parameters():p.requires_grad_(False)
            losses=torch.stack([F.cross_entropy(c(view),target) for c in bank]+[view.sum()*0+self.entropy])
            terms.append(1-losses.min()/self.entropy);critic_ces.append(losses.detach().tolist())
        zero=views[0].sum()*0
        sensitive=sum(terms[i] for i in active)+terms[2]
        return {'local0':terms[0],'local1':terms[1],'pair':terms[2],'sensitive':sensitive,'target_loss':zero,'target0':zero,'target1':zero,'diagnostics':{'critic_ces':critic_ces,'training_sensitive_entropy':self.entropy,'critic_optimizer_steps':self.critic_steps,'reference':[r.snapshot() for r in self.references]}}

    def state_dict(self):
        return {'critics':[[c.state_dict() for c in b] for b in self.critics],'optimizers':[[o.state_dict() for o in b] for b in self.optimizers],'prior':self.prior,'entropy':self.entropy,'critic_optimizer_steps':self.critic_steps,'references':self.references}

    def load_state_dict(self,state):
        if not torch.equal(self.prior,state['prior']):raise ValueError('raw prior differs')
        for bank,states in zip(self.critics,state['critics']):
            for c,s in zip(bank,states):c.load_state_dict(s)
        for opts,states in zip(self.optimizers,state['optimizers']):
            for o,s in zip(opts,states):o.load_state_dict(s)
        self.critic_steps=state['critic_optimizer_steps'];self.references=state['references']
