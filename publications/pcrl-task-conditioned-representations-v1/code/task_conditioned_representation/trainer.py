"""Bounded continuation with separate uniform task and conditional privacy draws.

Only worker entrypoints launch numerical computation. All selection uses FIT;
INNER labels are absent from this module's inputs.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json,math,os,random,statistics,time
import torch
import torch.nn.functional as F
from .models import actual_wires

@dataclass(frozen=True)
class Recipe:
    epochs:int=20
    checkpoints:tuple=(0,5,10,20)
    batch_size:int=256
    learning_rate:float=1e-4
    weight_decay:float=0.
    clip:float=5.
    reference_size:int=256
    gradient_log_every:int=20
    def __post_init__(self):
        if not 0<=self.epochs<=40 or 0 not in self.checkpoints or any(e<0 or e>self.epochs for e in self.checkpoints):raise ValueError('unregistered checkpoint schedule')
        if tuple(sorted(set(self.checkpoints)))!=self.checkpoints or self.batch_size<1 or self.learning_rate<=0 or self.weight_decay!=0 or self.clip<=0:raise ValueError('recipe differs')


def seed_for(*items):return int(hashlib.sha256('|'.join(map(str,items)).encode()).hexdigest()[:16],16)%2**63

def sha_file(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def to_json(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {str(k):to_json(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [to_json(v) for v in value]
    return value

def atomic_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w') as f:json.dump(to_json(value),f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def gradient_norm(loss,parameters,retain_graph=True):
    if not loss.requires_grad:return 0.
    gradients=torch.autograd.grad(loss,parameters,retain_graph=retain_graph,allow_unused=True)
    square=sum(float(g.detach().to(torch.float64).square().sum()) for g in gradients if g is not None)
    return math.sqrt(square)

def calibrate_ratio(task_norms,protection_norms):
    if len(task_norms)!=16 or len(protection_norms)!=16:raise ValueError('exactly sixteen fixed draws required')
    finite=all(math.isfinite(float(x)) for x in list(task_norms)+list(protection_norms))
    t=statistics.median(task_norms) if finite else None;p=statistics.median(protection_norms) if finite else None
    out={'task_norms':list(task_norms),'protection_norms':list(protection_norms),'median_task':t,'median_protection':p,'norm_floor':1e-12,'coefficient_ceiling':1e4,'supported':False}
    if not finite or t<=1e-12 or p<=1e-12:out['reason']='nonfinite_or_below_floor';return out
    ratio=t/p
    if 3*ratio>1e4:out['reason']='largest_registered_coefficient_above_ceiling';return out
    out.update(supported=True,coefficient_per_ratio=ratio,coefficients={str(r):r*ratio for r in (.3,1.,3.)},reason=None);return out


def task_loss(model,x,labels,active=(0,1)):
    # Uniform row sampling estimates each task population risk, with equal task weights.
    return sum(F.cross_entropy(model.head[i](model.encode(i,x).to(torch.float64)),labels[i]) for i in active)

@torch.no_grad()
def task_metrics(model,x,labels):
    out=[]
    for i,w in enumerate(actual_wires(model,x)):
        p=w['probabilities'];y=labels[i];decision=w['decision'];k=p.shape[1]
        recall=[float((decision[y==c]==c).to(torch.float64).mean()) if bool((y==c).any()) else 0. for c in range(k)]
        prior=torch.bincount(y,minlength=k).to(torch.float64)/len(y)
        const=prior[None].expand(len(y),-1);truth=F.one_hot(y,k).to(torch.float64)
        out.append({'accuracy':float((decision==y).to(torch.float64).mean()),'nll':float(-p[torch.arange(len(y)),y].clamp_min(1e-12).log().mean()),'brier':float((p-truth).square().sum(1).mean()),'balanced_accuracy':sum(recall)/k,'per_class_recall':recall,'class_counts':torch.bincount(y,minlength=k).tolist(),'mean_confidence':float(p.max(1).values.mean()),'constant_accuracy':float(prior.max()),'constant_nll':float(-prior[y].clamp_min(1e-12).log().mean()),'constant_brier':float((const-truth).square().sum(1).mean()),'accuracy_gain_over_constant':float((decision==y).to(torch.float64).mean()-prior.max())})
    return out

def normalized_violation(metrics,baseline):
    values=[]
    for a,b in zip(metrics,baseline):values.extend(((b['accuracy']-a['accuracy'])/.01,(a['nll']-b['nll'])/.01,(a['brier']-b['brier'])/.005))
    return max(values)


def select_common(rows,family,strengths=(.3,1.,3.),forced=None):
    grouped={}
    for r in rows:
        if r['family']==family:grouped.setdefault((r['strength'],r['epoch']),{})[r['seed']]=r
    configs=[]
    for (strength,epoch),seeds in grouped.items():
        if set(seeds)!={0,1}:continue
        if strength not in strengths:continue
        records=[seeds[0],seeds[1]]
        configs.append({'strength':strength,'epoch':epoch,'violation':max(r['violation'] for r in records),'nll':sum(r['nll'] for r in records),'sensitive':sum(r.get('sensitive',0.) for r in records),'nominees':records})
    if not configs:raise ValueError('no common two-seed configuration')
    tie=lambda c:(c['violation'],c['nll'],strengths.index(c['strength']),c['epoch'])
    fallback=min(configs,key=tie)
    feasible=[c for c in configs if c['violation']<=1]
    if forced is not None:
        chosen=next((c for c in configs if (c['strength'],c['epoch'])==tuple(forced)),None)
        if chosen is None:raise ValueError('exact twin configuration absent')
    elif family in ('U_NEW','U_TASKPOWER'):
        chosen=min(feasible,key=lambda c:(c['nll'],c['epoch'])) if feasible else fallback
    else:
        chosen=min(feasible,key=lambda c:(c['sensitive'],)+tie(c)) if feasible else fallback
    return dict(chosen,fallback=fallback,selection='FIT_ONLY_COMMON_CONFIGURATION',surrogate_feasible=chosen['violation']<=1)


def _save(model,optimizer,path,recipe,family,seed,strength,coefficient,epoch,updates,task_generator,privacy_rng,reference,rows,objective=None,selection_active=None,selection_family=None):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    payload={'schema':'TCR1-trainable-own-head-checkpoint-v1','model':model.state_dict(),'optimizer':optimizer.state_dict(),'feature_names':list(model.names),'class_orders':model.class_orders,'source_bindings':getattr(model,'source_bindings',{}),'recipe':asdict(recipe),'family':family,'selection_active':list(selection_active) if selection_active is not None else None,'selection_family':selection_family,'seed':seed,'strength':strength,'coefficient':coefficient,'epoch':epoch,'updates_per_recipient':updates[:],'task_rng_state':task_generator.get_state(),'privacy_rng_state':privacy_rng.getstate(),'reference':reference,'selection_history':rows,'objective_state':objective.state_dict() if objective is not None and hasattr(objective,'state_dict') else None}
    torch.save(payload,tmp);os.chmod(tmp,0o600);os.replace(tmp,p);return sha_file(p)


def fit(model,x,labels,train_indices,selection_indices,recipe,family,seed,coefficient,output,*,strength=0.,target_enabled=None,objective=None,registry=None,selection_registry=None,active=(0,1),initial_baseline=None,stage_tag='simultaneous',initial_updates=(0,0),selection_active=None,selection_family=None):
    if target_enabled is None:target_enabled=family!='U_NEW'
    needs_kernel=target_enabled or coefficient!=0
    objective_family='U_TASKPOWER' if coefficient==0 and target_enabled else family
    selection_active=tuple(active if selection_active is None else selection_active)
    selection_family=objective_family if selection_family is None else selection_family
    if not selection_active or any(i not in (0,1) for i in selection_active) or len(set(selection_active))!=len(selection_active):raise ValueError('invalid selection recipient set')
    if not math.isfinite(coefficient) or coefficient<0 or coefficient>1e4:raise ValueError('derived coefficient outside registered range')
    if needs_kernel and (objective is None or registry is None or selection_registry is None):raise ValueError('registered objective and strata required')
    for i in range(2):
        for p in list(model.enc[i].parameters())+list(model.head[i].parameters()):p.requires_grad_(i in active)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(parameters,lr=recipe.learning_rate,weight_decay=recipe.weight_decay,betas=(.9,.999),eps=1e-8)
    task_generator=torch.Generator().manual_seed(seed_for('TCR1task',seed,stage_tag));privacy_rng=random.Random(seed_for('TCR1privacy',seed,stage_tag))
    train_indices=torch.as_tensor(train_indices,dtype=torch.long);selection_indices=torch.as_tensor(selection_indices,dtype=torch.long)
    tx=x[train_indices];ty=tuple(y[train_indices] for y in labels);vx=x[selection_indices];vy=tuple(y[selection_indices] for y in labels)
    baseline=initial_baseline or task_metrics(model,vx,vy)
    reference_indices=torch.randperm(len(tx),generator=torch.Generator().manual_seed(seed_for('TCR1reference',seed)))[:recipe.reference_size]
    output=Path(output);output.mkdir(parents=True,exist_ok=True);rows=[];logs=[];updates=list(initial_updates);step=0;started=time.monotonic()
    for epoch in range(recipe.epochs+1):
        reference=objective.refresh_model_reference(model,tx[reference_indices]) if needs_kernel else None
        if epoch in recipe.checkpoints:
            model.eval();metrics=task_metrics(model,vx,vy)
            sensitive=0.;diagnostics=None
            if needs_kernel:
                val_rng=random.Random(seed_for('TCR1selectiondraw',seed))
                with torch.no_grad():
                    losses=objective.loss(model,vx,selection_registry.draw(val_rng,selection_family),selection_family,selection_active)
                sensitive=float(losses['sensitive']);diagnostics=losses['diagnostics']
            row={'family':family,'seed':seed,'strength':strength,'epoch':epoch,'metrics':metrics,'violation':normalized_violation(metrics,baseline),'nll':sum(t['nll'] for t in metrics),'sensitive':sensitive,'selection_active':list(selection_active),'selection_family':selection_family,'updates_per_recipient':updates[:],'checkpoint':str(output/f'epoch_{epoch}.pt')}
            row['checkpoint_sha256']=_save(model,optimizer,row['checkpoint'],recipe,family,seed,strength,coefficient,epoch,updates,task_generator,privacy_rng,reference,rows,objective,selection_active,selection_family)
            rows.append(row);atomic_json(output/f'epoch_{epoch}_selection.json',dict(row,diagnostics=diagnostics))
            atomic_json(output/'FIT_CHECKPOINTS.json',rows)
        if epoch==recipe.epochs:break
        model.train();order=torch.randperm(len(tx),generator=task_generator)
        for indices in order.split(recipe.batch_size):
            ce=task_loss(model,tx[indices],tuple(y[indices] for y in ty),active)
            terms=objective.loss(model,tx,registry.draw(privacy_rng,objective_family),objective_family,active) if needs_kernel else None
            target=terms['target_loss'] if target_enabled else ce*0
            sensitive=terms['sensitive'] if coefficient!=0 else ce*0
            loss=ce+target+coefficient*sensitive
            if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite objective')
            if step%recipe.gradient_log_every==0:
                tasknorm=gradient_norm(ce,parameters);protectionnorm=gradient_norm(sensitive,parameters);targetnorm=gradient_norm(target,parameters)
                logs.append({'epoch':epoch+1,'step':step,'ce':float(ce.detach()),'target':float(target.detach()),'sensitive':float(sensitive.detach()),'task_gradient_norm':tasknorm,'sensitive_gradient_norm':protectionnorm,'target_gradient_norm':targetnorm,'effective_ratio':abs(coefficient)*protectionnorm/tasknorm if tasknorm>1e-12 else None,'head_norms':[float(h.weight.detach().double().norm()) for h in model.head], 'latent_variances':[z.detach().double().var(0,correction=0).tolist() for z in [model.encode(i,tx[indices]) for i in range(2)]]})
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):raise FloatingPointError('nonfinite parameter gradient')
            torch.nn.utils.clip_grad_norm_(parameters,recipe.clip);optimizer.step()
            for i in active:updates[i]+=1
            step+=1
        atomic_json(output/'GRADIENT_SCALE_LOG.json',logs)
        atomic_json(output/'RUNNING.json',{'epoch':epoch+1,'updates_per_recipient':updates,'elapsed_s':time.monotonic()-started})
    result={'family':family,'seed':seed,'strength':strength,'coefficient':coefficient,'recipe':asdict(recipe),'checkpoints':rows,'updates_per_recipient':updates,'steps':step,'elapsed_s':time.monotonic()-started,'selection_baseline':'exact_source_U0_epoch0_on_FIT_fold2','train_rows':len(tx),'selection_rows':len(vx),'active':list(active),'selection_active':list(selection_active),'selection_family':selection_family,'target_enabled':target_enabled,'parameter_counts':{'encoders':[sum(p.numel() for p in e.parameters()) for e in model.enc],'heads':[sum(p.numel() for p in h.parameters()) for h in model.head],'all':sum(p.numel() for p in model.parameters()),'trainable_this_stage':sum(p.numel() for p in parameters)},'transformations':['encoder0','actual_affine_head0','encoder1','actual_affine_head1'],'task_population_weights':[1.,1.]}
    atomic_json(output/'FIT_RESULT.json',result);return result
