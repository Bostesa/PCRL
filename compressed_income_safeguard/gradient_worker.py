"""Queue-owned FIT gradient regeneration and constrained-update prototype.

No module import loads arrays, models or critics. Numerical entrypoint checks
the new campaign queue first. Saved gradient norms are never called tensors.
"""
from __future__ import annotations
from itertools import combinations
from pathlib import Path
import hashlib,json,os

QP_TOL=1e-10
FINITE_TOL=1e-10
BACKTRACK=(1.,.5,.25,.125)
NORM_REPAIR_SAFETY=1e-6
NORM_REPAIR_ATTEMPTS=16


def project_displacement(proposal,task_gradients,max_norm):
    """min 1/2||d-proposal||² subject to a_j.d<=0 and ||d||<=max_norm.

    Two differentiable task guards only. Normalize every strictly nonzero
    row, including tiny gradients. Homogeneous cone projection followed by
    radial clipping is the Euclidean projection onto cone intersect ball.
    """
    import numpy as np
    p=np.asarray(proposal,dtype=np.float64);a=np.asarray(task_gradients,dtype=np.float64)
    if p.ndim!=1 or not p.size or a.ndim!=2 or a.shape[1]!=p.size or len(a)>2:raise ValueError('at most two aligned task guards')
    if not np.isfinite(p).all() or not np.isfinite(a).all() or not np.isfinite(max_norm) or max_norm<=0:raise ValueError('finite positive bounded proposal required')
    # Scale before norm to avoid underflow removing a real tiny guard.
    row_scale=np.max(np.abs(a),axis=1) if len(a) else np.zeros(0)
    keep=row_scale>0
    aa=a[keep]/row_scale[keep,None];aa/=np.linalg.norm(aa,axis=1)[:,None]
    candidates=[];tol=QP_TOL*max(1.,float(np.linalg.norm(p)))
    for size in range(len(aa)+1):
        for active in combinations(range(len(aa)),size):
            at=aa[list(active)]
            if size:
                multipliers=np.linalg.lstsq(at@at.T,at@p,rcond=None)[0]
                d=p-at.T@multipliers
                if np.max(np.abs(at@d))>tol or np.any(multipliers < -tol):continue
            else:d=p.copy();multipliers=np.zeros(0)
            if len(aa) and np.max(aa@d)>tol:continue
            candidates.append((float(np.sum((d-p)**2)),d,active,multipliers))
    fallback=not candidates
    if fallback:d=np.zeros_like(p);active=();multipliers=np.zeros(0)
    else:_,d,active,multipliers=min(candidates,key=lambda item:item[0])
    norm=float(np.linalg.norm(d));ball_active=norm>max_norm
    if ball_active:d=d*(max_norm/norm)
    residual=aa@d
    violation=float(max(0.,float(residual.max()))) if len(aa) else 0.
    return d,{'zero_feasible':True,'zero_gradients':int((~keep).sum()),
              'guard_count':len(a),'active_set':list(active),'multipliers':multipliers.tolist(),
              'norm_bound_active':ball_active,'proposal_norm':float(np.linalg.norm(p)),
              'displacement_norm':float(np.linalg.norm(d)),'normalized_constraint_products':residual.tolist(),
              'max_normalized_violation':violation,'conservative_zero_fallback':fallback,
              'certified':bool(violation<=tol and np.linalg.norm(d)<=max_norm+tol)}


def parameter_vector(parameters):
    import numpy as np
    return np.concatenate([p.detach().cpu().double().reshape(-1).numpy() for p in parameters])


def _assign(parameters,vector):
    import torch
    offset=0
    with torch.no_grad():
        for p in parameters:
            end=offset+p.numel();p.copy_(torch.from_numpy(vector[offset:end].copy()).reshape(p.shape).to(p.dtype));offset=end
    if offset!=len(vector):raise ValueError('parameter displacement dimension differs')


def finite_step(parameters,proposal,task_gradients,task_evaluator,max_norm,*,guarded=True):
    """Apply a bounded SGD displacement, certify actual dtype-rounded movement.

    task_evaluator reads only the authorized FIT-update batch and returns NLL
    and summed Brier. Any failed step restores exactly the saved parameters.
    No momentum/Adam state can bypass a projected displacement.
    """
    import numpy as np
    before=parameter_vector(parameters);baseline=task_evaluator()
    if set(baseline)!={'nll','brier'} or not all(np.isfinite(list(baseline.values()))):raise ValueError('finite two-metric FIT safeguard required')
    a=np.asarray(task_gradients,dtype=np.float64)
    d,qp=project_displacement(proposal,a if guarded else np.zeros((0,len(before))),max_norm)
    attempts=[];accepted=None;actual=np.zeros_like(before)
    row_scale=np.max(np.abs(a),axis=1) if len(a) else np.zeros(0)
    keep=row_scale>0;normal=a[keep]/row_scale[keep,None]
    if len(normal):normal/=np.linalg.norm(normal,axis=1)[:,None]
    for fraction in BACKTRACK if guarded else (1.,):
        # A float64 vector clipped exactly to the ball can round outside it
        # when copied to float32 parameters. Repair only the actual norm,
        # identically in every arm, before evaluating task feasibility. The
        # scalar contractions preserve the proposed homogeneous cone; actual
        # rounded halfspace products are still checked separately below.
        norm_trials=[];norm_scale=1.;norm_certified=False
        for repair_attempt in range(NORM_REPAIR_ATTEMPTS):
            _assign(parameters,before+fraction*norm_scale*d)
            actual=parameter_vector(parameters)-before
            actual_norm=float(np.linalg.norm(actual))
            norm_certified=bool(np.isfinite(actual_norm) and actual_norm<=max_norm)
            norm_trials.append({'attempt':repair_attempt,'scale':norm_scale,
                                'actual_norm':actual_norm,'bounded':norm_certified})
            if norm_certified:break
            if not np.isfinite(actual_norm):break
            if repair_attempt+1==NORM_REPAIR_ATTEMPTS:break
            norm_scale*=min(1.,max_norm/actual_norm)*(1.-NORM_REPAIR_SAFETY)
        products=normal@actual if len(normal) else np.zeros(0)
        actual_norm=float(np.linalg.norm(actual));metric=task_evaluator()
        finite=all(np.isfinite(list(metric.values())))
        local=not len(products) or bool(np.max(products)<=QP_TOL*max(1.,actual_norm))
        bounded=norm_certified
        safe=finite and all(metric[k]<=baseline[k]+FINITE_TOL for k in baseline)
        passed=finite and bounded and (not guarded or (local and safe))
        attempts.append({'fraction':fraction,'metrics':metric,'actual_norm':actual_norm,
                         'norm_repair_scale':norm_scale,'actual_effective_fraction':fraction*norm_scale,
                         'norm_repair_attempts':norm_trials,
                         'actual_normalized_constraint_products':products.tolist(),
                         'first_order_check':local,'finite_step_check':safe,'accepted':passed})
        if passed:accepted=fraction;break
        _assign(parameters,before)
    if accepted is None:
        _assign(parameters,before);actual=np.zeros_like(before);status='REJECTED_ZERO_STEP'
    elif not np.any(actual):status='ROUNDING_ZERO_STEP'
    else:status='ACCEPTED'
    return {'status':status,'accepted_fraction':accepted,'actual_displacement_norm':float(np.linalg.norm(actual)),
            'baseline':baseline,'attempts':attempts,'projection':qp,'guarded':guarded,
            'actual_norm_repair':{'all_arms':True,'safety_fraction':NORM_REPAIR_SAFETY,
                                  'maximum_attempts':NORM_REPAIR_ATTEMPTS,'task_metrics_used':False},
            'finite_step_tolerance':FINITE_TOL,'backtrack':list(BACKTRACK)}


def flat_gradient(loss,parameters):
    import numpy as np
    import torch
    if not loss.requires_grad:return np.zeros(sum(p.numel() for p in parameters))
    values=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    return np.concatenate([np.zeros(p.numel()) if v is None else v.detach().cpu().double().reshape(-1).numpy() for p,v in zip(parameters,values)])


def gradient_summary(vectors):
    import numpy as np
    norms={name:float(np.linalg.norm(v)) for name,v in vectors.items()};pairs={}
    for left,right in combinations(vectors,2):
        dot=float(vectors[left]@vectors[right]);den=norms[left]*norms[right]
        cosine=dot/den if den>0 else None
        pairs[left+'__'+right]={'dot':dot,'cosine':cosine,
                               'same_minimization_direction_conflict':bool(dot<0) if den>0 else None}
    return {'norms':norms,'pairs':pairs,'zero_norm_cosine':None,'normalization':'mean row loss; Euclidean parameter-gradient norm; float64 reductions'}


def occupation_refinement_step(model,x,occupation,raw_terms,*,arm,coefficient,geometry,frozen_guard=None,learning_rate=.01,max_norm=.01):
    """Runnable single pilot update; coordinator owns exposure and critic jobs.

    raw_terms comes from the corrected projected-income/full-occupation wires,
    after the same five critic updates in every arm. Even the local twin and
    task-only arm train the pair bank for equal compute; only the update term
    is removed. Caller must freeze one recipe before calling this function.
    """
    import numpy as np
    import torch
    import torch.nn.functional as f
    from .wires import corrected_trial_wires, FrozenIncomeGuard
    if arm not in ('CONSTRAINED','ORDINARY','LOCAL_TWIN','TASK_ONLY'):raise ValueError('unregistered pilot arm')
    if not np.isfinite(coefficient) or coefficient<0 or learning_rate<=0:raise ValueError('invalid frozen coefficient or learning rate')
    frozen_guard = frozen_guard or FrozenIncomeGuard(model,geometry)
    frozen_guard.verify()
    for name,p in model.named_parameters():p.requires_grad_(name.startswith(('enc.1.','head.1.')))
    parameters=list(model.enc[1].parameters())+list(model.head[1].parameters())
    w=corrected_trial_wires(model,x,geometry)[1];p=w['probabilities'];rows=torch.arange(len(x))
    ce=f.cross_entropy(w['centered_logits'],occupation)
    nll=-p[rows,occupation].clamp_min(1e-12).log().mean()
    brier=(p-f.one_hot(occupation,6).to(torch.float64)).square().sum(1).mean()
    zero=ce*0
    sensitive=zero if arm=='TASK_ONLY' else raw_terms['local1']+(zero if arm=='LOCAL_TWIN' else raw_terms['pair'])
    task_gradient=flat_gradient(ce,parameters);privacy_gradient=flat_gradient(sensitive,parameters)
    proposal=-learning_rate*(task_gradient+coefficient*privacy_gradient)
    guards=np.vstack((flat_gradient(nll,parameters),flat_gradient(brier,parameters)))
    def evaluate():
        with torch.no_grad():
            pp=corrected_trial_wires(model,x,geometry)[1]['probabilities']
            return {'nll':float(-pp[rows,occupation].clamp_min(1e-12).log().mean()),
                    'brier':float((pp-f.one_hot(occupation,6).to(torch.float64)).square().sum(1).mean())}
    result=finite_step(parameters,proposal,guards,evaluate,max_norm,guarded=arm in ('CONSTRAINED','LOCAL_TWIN'))
    frozen_guard.verify()
    return dict(result,arm=arm,coefficient=coefficient,learning_rate=learning_rate,
                gradient_summary=gradient_summary({'task_ce':task_gradient,'sensitive_penalty':privacy_gradient}),
                frozen_income_bitwise_unchanged=True,optimizer='bounded SGD displacement; no momentum or preconditioning')


def batch_gradients(model,x,income,occupation,*,geometry,sensitive_loss=None,kernel_loss=None):
    """Return actual tensors (private) and summaries in two registered blocks.

    All sensitive losses here are minimized penalty losses: adversarial risk
    1-min(critic CE,entropy)/entropy, not the critic's own minimization CE.
    """
    import numpy as np
    import torch
    import torch.nn.functional as f
    from .wires import corrected_trial_wires, FrozenIncomeGuard
    model.eval()
    for p in model.parameters():p.requires_grad_(True)
    wires=corrected_trial_wires(model,x,geometry);p=wires[1]['probabilities'];n=len(x)
    if not n:raise ValueError('nonempty fixed FIT batch required')
    losses={'task_ce':sum(f.cross_entropy(w['centered_logits'],y) for w,y in zip(wires,(income,occupation))),
            'occupation_ce':f.cross_entropy(wires[1]['centered_logits'],occupation),
            'occupation_nll':-p[torch.arange(n),occupation].clamp_min(1e-12).log().mean(),
            'occupation_brier':(p-f.one_hot(occupation,6).to(torch.float64)).square().sum(1).mean()}
    if sensitive_loss is not None:losses['sensitive_penalty']=sensitive_loss(model,x)
    if kernel_loss is not None:losses['conditional_kernel']=kernel_loss(model)
    blocks={'all_actual_parameters':list(model.named_parameters()),
            'occupation_encoder_and_own_head':[(name,p) for name,p in model.named_parameters() if name.startswith(('enc.1.','head.1.'))]}
    summaries={};tensors={}
    for block,items in blocks.items():
        vectors={name:flat_gradient(loss,[p for _,p in items]) for name,loss in losses.items()}
        summaries[block]=gradient_summary(vectors)
        summaries[block]['parameter_order']=[name for name,_ in items]
        summaries[block]['parameter_sizes']=[p.numel() for _,p in items]
        tensors[block]=vectors
    return {'loss_values':{k:float(v.detach()) for k,v in losses.items()},'blocks':summaries},tensors


def _sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def run(spec_path):
    """Frozen spec: manifest/checkpoint pins, explicit FIT indices, output path.

    Predecessor full-income critics are incompatible with the hybrid release.
    Their sensitive gradient is MISSING, never relabeled or treated as zero.
    """
    from .evaluation_worker import queue_config
    active,spec,verify,queue_guard=queue_config(spec_path,'admission')
    from compressed_income_safeguard.scheduler import assert_active_from_env
    import numpy as np
    if spec.get('operation')!='regenerate_fixed_fit_gradients':raise ValueError('unregistered gradient operation')
    import torch
    torch.set_num_threads(1)
    from compressed_income_safeguard.source import load_manifest,load_role
    from task_conditioned_representation.models import load_checkpoint
    from .wires import prepare_geometry
    batches=spec['batch_indices']
    if not 1<=len(batches)<=4 or any(not 1<=len(b)<=256 or len(set(b))!=len(b) for b in batches):raise ValueError('small fixed batch roster differs')
    m=load_manifest(spec['manifest'],spec['manifest_sha256'])
    data=load_role(m,int(spec['seed']),'OSF_DEFENSE_FIT',{'features','income','occupation','sex'})
    for b in batches:
        if any(not isinstance(i,int) or i<0 or i>=len(data['X']) for i in b):raise ValueError('batch index outside FIT')
        if not np.isin(data['fold'][b],[0,1]).all():raise PermissionError('gradient batch outside FIT folds 0/1')
    model,payload=load_checkpoint(spec['checkpoint'],spec['checkpoint_sha256'])
    if int(payload['seed'])!=int(spec['seed']):raise ValueError('checkpoint source seed differs')
    geometry=prepare_geometry(json.loads(verify(spec['geometry']['path'],spec['geometry']['sha256']).read_text()), expected_sha256=spec['geometry_binding_sha256'],source_seed=int(spec['seed']),checkpoint_sha256=spec['geometry_checkpoint_sha256'])
    # Predecessor critics and their full-income references are incompatible
    # with this corrected interface; never relabel their gradient as hybrid.
    objective=None
    records=[];tensor_batches={}
    for j,indices in enumerate(batches):
        assert_active_from_env(stage='admission')
        x=torch.from_numpy(data['X'][indices]);yi=torch.as_tensor(data['income'][indices]);yo=torch.as_tensor(data['occupation'][indices]);sex=torch.as_tensor(data['sex'][indices])
        summary,tensors=batch_gradients(model,x,yi,yo,geometry=geometry)
        summary.update(batch_number=j,rows=len(indices),sensitive_gradient_status='MISSING_CORRECTED_WIRE_CRITIC_STATE',
                       fixed_batch_index_sha256=hashlib.sha256(json.dumps(indices,separators=(',',':')).encode()).hexdigest())
        records.append(summary);tensor_batches[str(j)]={block:{name:torch.from_numpy(vector) for name,vector in v.items()} for block,v in tensors.items()}
    out=Path(spec['private_output'])
    if '.private' not in out.resolve().parts or out.exists():raise PermissionError('fresh private gradient directory required')
    out.mkdir(parents=True,mode=0o700)
    torch.save({'schema':'CIS1-regenerated-fit-gradient-tensors-v1','batches':tensor_batches},out/'gradients.pt');(out/'gradients.pt').chmod(0o600)
    result={'schema':'CIS1-regenerated-fit-gradients-v1','seed':int(spec['seed']),'checkpoint_sha256':spec['checkpoint_sha256'],
            'tensor_sha256':_sha(out/'gradients.pt'),'records':records,'source_gradient_logs_not_regenerated_tensors':True,
            'new_critic_fit_count':0,'model_updates':0,'INNER_labels_opened':False,'conditional_kernel_gradient_status':'NOT_REGENERATED_BY_THIS_FIXED_BATCH_OPERATION'}
    for path in (out/'SUMMARY.json',Path(spec['output'])):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True)
    run(parser.parse_args().spec)
