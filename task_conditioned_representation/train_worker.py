"""Authenticated queue worker for FIT-only baselines, kernel arms and calibration."""
from __future__ import annotations
import argparse,json,os,random
from pathlib import Path


def run(spec_path):
    spec=json.loads(Path(spec_path).read_text())
    from .scheduler import assert_active_from_env
    assert_active_from_env(stage='admission' if spec['operation'] in ('calibrate','timing') else 'protection')
    import torch
    torch.set_num_threads(1)
    from .source import load_manifest,load_role
    from .useful_service import load_useful_model
    from .objectives import KernelObjective,StrataRegistry
    from .trainer import Recipe,fit,seed_for,task_loss,gradient_norm,calibrate_ratio,atomic_json,task_metrics
    m=load_manifest(spec['manifest'],spec['manifest_sha256']);seed=int(spec['seed'])
    data=load_role(m,seed,'OSF_DEFENSE_FIT',{'features','income','occupation','sex'})
    model=load_useful_model(m,seed)
    x=torch.from_numpy(data['X']);labels=tuple(torch.as_tensor(data[k],dtype=torch.long) for k in ('income','occupation'))
    ti=torch.as_tensor((data['fold']!=2).nonzero()[0],dtype=torch.long);vi=torch.as_tensor((data['fold']==2).nonzero()[0],dtype=torch.long)
    def registry(ix):
        ids=ix.numpy();return StrataRegistry(data['income'][ids],data['occupation'][ids],data['sex'][ids],data['unit'][ids])
    train_registry=registry(ti);val_registry=registry(vi);objective=KernelObjective()
    # Freeze selection diagnostic weights/support from optimization population.
    # Held-out support still must independently supply 32 distinct groups/arm.
    training_weights={(kind,key):weight for kind,key,pools,weight in train_registry.specs}
    val_registry.specs=[(kind,key,pools,training_weights[(kind,key)]) for kind,key,pools,weight in val_registry.specs if (kind,key) in training_weights]
    for row in val_registry.coverage:
        identity=(row['kind'],tuple(row['key']))
        row['training_supported']=identity in training_weights
        row['fixed_training_weight']=training_weights.get(identity,0.)
        row['selection_supported']=row['supported'] and row['training_supported']
    family=spec['family'];recipe=Recipe(**{k:tuple(v) if k=='checkpoints' else v for k,v in spec.get('recipe',{}).items()})
    if family=='RAW_JOINT':
        from .controls import RawObjective,RawRegistry
        train_registry=RawRegistry(data['sex'][ti.numpy()],recipe.batch_size)
        val_registry=RawRegistry(data['sex'][vi.numpy()],recipe.batch_size)
        objective=RawObjective(seed,data['sex'][ti.numpy()])
        reference=x[ti][torch.randperm(len(ti),generator=torch.Generator().manual_seed(seed_for('TCR1reference',seed)))[:recipe.reference_size]]
        objective.refresh_model_reference(model,reference)
        if spec['operation']=='calibrate':
            objective.warm(model,x[ti],train_registry,seed)
            warm_path=Path(spec['private_output'])/'raw_warm.pt';warm_path.parent.mkdir(parents=True,exist_ok=True)
            torch.save(objective.state_dict(),warm_path);os.chmod(warm_path,0o600)
        elif spec['operation']=='fit':
            from .trainer import sha_file
            if sha_file(spec['raw_warm_checkpoint'])!=spec['raw_warm_sha256']:raise ValueError('raw warm checkpoint hash differs')
            objective.load_state_dict(torch.load(spec['raw_warm_checkpoint'],map_location='cpu',weights_only=False))
        if spec['operation']=='calibrate':objective.update_enabled=False
    if spec['operation']=='calibrate':
        tx=x[ti];ty=tuple(y[ti] for y in labels)
        ref=torch.randperm(len(tx),generator=torch.Generator().manual_seed(seed_for('TCR1reference',seed)))[:recipe.reference_size]
        objective.refresh_model_reference(model,tx[ref]);privacy_rng=random.Random(seed_for('TCR1calibrationprivacy',seed));task_rng=torch.Generator().manual_seed(seed_for('TCR1calibrationtask',seed))
        parameters=list(model.parameters());tasks=[];protections=[]
        for draw in range(16):
            indices=torch.randperm(len(tx),generator=task_rng)[:recipe.batch_size]
            ce=task_loss(model,tx[indices],tuple(y[indices] for y in ty));loss=objective.loss(model,tx,train_registry.draw(privacy_rng,family),family)['sensitive']
            tasks.append(gradient_norm(ce,parameters));protections.append(gradient_norm(loss,parameters))
        out=dict(calibrate_ratio(tasks,protections),family=family,seed=seed,parameter_blocks='both_encoders_and_actual_heads_all_parameters',initialization='exact_folded_U0_epoch0',draws=16,rows='FIT_internal_folds_0_1',reference=objective.refresh_model_reference(model,tx[ref]))
        if family=='RAW_JOINT':
            from .trainer import sha_file
            out['warm_checkpoint_sha256']=sha_file(warm_path)
            out['critic_warm_updates_per_critic']=80;out['critic_warm_optimizer_steps']=objective.critic_steps
        atomic_json(spec['output'],out);return out
    if spec['operation']=='timing':
        import time
        if spec.get('steps',2) not in (1,2):raise ValueError('engineering timing bound is at most two updates')
        tx=x[ti];ty=tuple(y[ti] for y in labels);objective.refresh_model_reference(model,tx[:recipe.reference_size]);parameters=list(model.parameters());privacy_rng=random.Random(seed_for('TCR1timing',seed));durations=[]
        optimizer=torch.optim.Adam(parameters,lr=recipe.learning_rate)
        for repeat in range(spec.get('steps',2)):
            started=time.monotonic();indices=torch.arange(recipe.batch_size)
            loss=task_loss(model,tx[indices],tuple(y[indices] for y in ty))+objective.loss(model,tx,train_registry.draw(privacy_rng,family),family)['sensitive']
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(parameters,recipe.clip);optimizer.step();durations.append(time.monotonic()-started)
        out={'seed':seed,'family':family,'steps':len(durations),'step_elapsed_s':durations,'train_rows':len(ti),'selection_rows':len(vi),'science_fit':False,'discarded_model':True}
        atomic_json(spec['output'],out);return out
    initial_baseline=task_metrics(model,x[vi],tuple(y[vi] for y in labels))
    coefficient=float(spec.get('coefficient',0.));strength=float(spec.get('strength',0.))
    if family in ('SEQ12','SEQ21') and coefficient!=0:
        import copy
        first,second=(0,1) if family=='SEQ12' else (1,0)
        source_model=copy.deepcopy(model)
        first_result=fit(model,x,labels,ti,vi,recipe,'LOCAL_TWIN',seed,coefficient,Path(spec['private_output'])/'stage1',strength=strength,objective=objective,registry=train_registry,selection_registry=val_registry,active=(first,),initial_baseline=initial_baseline)
        first_rows={r['epoch']:r for r in first_result['checkpoints']}
        source_row=dict(first_rows[0],family=family)
        combined=[source_row];branches=[]
        for e in recipe.checkpoints:
            if e==0:continue
            model=copy.deepcopy(source_model)
            checkpoint=torch.load(first_rows[e]['checkpoint'],map_location='cpu',weights_only=False)
            model.load_state_dict(checkpoint['model'])
            frozen={k:v.detach().clone() for k,v in model.state_dict().items() if k.startswith((f'enc.{first}.',f'head.{first}.'))}
            stage_recipe=Recipe(**dict(spec.get('recipe',{}),epochs=e,checkpoints=(0,e)))
            second_result=fit(model,x,labels,ti,vi,stage_recipe,family,seed,coefficient,Path(spec['private_output'])/f'stage2_from_{e}',strength=strength,objective=objective,registry=train_registry,selection_registry=val_registry,active=(second,),initial_baseline=initial_baseline,initial_updates=tuple(first_rows[e]['updates_per_recipient']))
            if any(not torch.equal(v,model.state_dict()[k]) for k,v in frozen.items()):raise AssertionError('frozen first encoder/head changed')
            last=second_result['checkpoints'][-1]
            if last['updates_per_recipient'][0]!=last['updates_per_recipient'][1]:raise AssertionError('sequential matched e/e updates differ')
            combined.append(last);branches.append(second_result)
        result={'family':family,'seed':seed,'strength':strength,'checkpoints':combined,'stages':[first_result]+branches,'updates_per_recipient':combined[-1]['updates_per_recipient'],'first_encoder_and_head_frozen':True,'aggregate_encoder_epochs':recipe.epochs+sum(e for e in recipe.checkpoints if e),'control_extra_search_exposure_percent':37.5}
    else:
        result=fit(model,x,labels,ti,vi,recipe,family,seed,coefficient,spec['private_output'],strength=strength,objective=objective,registry=train_registry,selection_registry=val_registry,initial_baseline=initial_baseline,target_enabled=spec.get('target_enabled',False if family=='RAW_JOINT' else None))
    if family=='RAW_JOINT':result['critic_optimizer_steps']=objective.critic_steps
    # Public receipt includes no raw rows, learned tensors, or private file paths.
    public=dict(result)
    public.pop('stages',None)
    public['checkpoints']=[{k:v for k,v in row.items() if k!='checkpoint'} for row in result['checkpoints']]
    atomic_json(spec['output'],public)
    atomic_json(Path(spec['private_output'])/'PRIVATE_FIT_RESULT.json',result)
    return public


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();run(a.spec)
if __name__=='__main__':main()
