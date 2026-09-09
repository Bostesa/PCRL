"""Bounded exact reconstruction of unsaved source-only displacement diagnostics.

Original T fits are immutable. This disposable source-forward replay exports
diagnostics only and must reproduce every saved step and the full final Adam
state exactly. It fits no observer or auditor and publishes no replacement map.
"""
from __future__ import annotations
import argparse
import copy
import datetime
import gzip
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.replay_acs_source_guard import TASKS, literal_adam, flat, assert_gradient_lists
from scripts.verify_acs_coalition_training import literal_parts
from scripts.verify_acs_coalition_strength_training import seed_context
from scripts import verify_acs_bottleneck_scores as check
from experiments.acs_bottleneck_training import tree_digest


def native_components(original, standardized, labels):
    state = {k:v.detach().clone().requires_grad_(k.startswith('branches.')) for k,v in original.items()}
    _, logits = literal_parts(state, torch.from_numpy(standardized))
    tasks = {}
    for target in TASKS:
        view = 'B' if target=='public_coverage' else 'A'
        scores=logits[view][target].reshape(-1); y=torch.from_numpy(labels[target]); valid=y>=0
        tasks[target] = F.binary_cross_entropy_with_logits(scores[valid],y[valid].float()) if valid.any() else scores.sum()*0.
    # Preserve the original nested equal-purpose reduction order exactly.
    source=torch.stack((torch.stack((tasks[TASKS[0]],tasks[TASKS[1]])).mean(),tasks[TASKS[2]])).mean()
    return state,source,tasks


def run(out):
    tick=time.perf_counter(); cfg=check.read(out/'config.json'); reuse=check.read(out/'REUSE_MANIFEST.json')
    releases=check.read(out/'RELEASE_MANIFEST.json')
    assert releases['training_permanently_closed'],'No reconstruction before final representation freeze'
    root=out/'t_source_reconstruction';root.mkdir(exist_ok=False)
    sources=[Path(__file__).resolve(),ROOT/'scripts/replay_acs_source_guard.py',
             ROOT/'scripts/verify_acs_coalition_training.py',ROOT/'scripts/verify_acs_coalition_strength_training.py']
    inputs={}
    for seed in (0,1,2):
        directory=out/f'seed_{seed}'/'training'
        for interface in ('F','P'):
            path=directory/(interface+'_T');assert (path/'complete.json').exists()
            for name in ('fork.pt','final.pt','training.json','complete.json'):
                p=path/name;inputs[str(p.relative_to(ROOT))]=check.sha(p)
        p=directory/'T_shared_step_diagnostics.jsonl.gz';inputs[str(p.relative_to(ROOT))]=check.sha(p)
    source_snapshot=root/'source';source_snapshot.mkdir()
    for path in sources:
        (source_snapshot/path.name).write_bytes(path.read_bytes())
    freeze={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'Three existing shared T forward trajectories; diagnostics only, no model replacement or selection',
        'why':'Original source-only all-step stream omitted displacement norms/dots; guarded streams are complete',
        'input_sha256':inputs,'source_sha256':{str(p.relative_to(ROOT)):check.sha(p) for p in sources},
        'source_snapshot_sha256':{str(p.relative_to(ROOT)):check.sha(p) for p in source_snapshot.iterdir()},
        'protocol_freeze_sha256':check.sha(out/'protocol_freeze.json'),'global_release_manifest_sha256':check.sha(out/'RELEASE_MANIFEST.json'),
        'seeds':[0,1,2],'epochs':80,'batch_size':256,'schedule_seed_base':1280200,
        'method':'Literal native BCE gradients and literal current Adam algebra from historical fork; no observer work',
        'acceptance':'Exact every original source-loss record, all fixed diagnostic pre/post states and final model/Adam/RNG/schedule',
        'new_models_selected_or_published':0,'new_observers_or_auditors':0,'maximum_seconds':180,
        'accounting':'Additional reconstruction wall time is conservatively charged to scientific compute'}
    (root/'protocol_freeze.json').write_text(json.dumps(freeze,indent=2,allow_nan=False)+'\n')
    parent=ROOT/cfg['parent_results'];raw_path=ROOT/check.read(parent/'config.json')['raw_path']
    assert check.sha(raw_path)==check.read(parent/'schema_support.json')['raw_sha256']
    raw=pd.read_csv(raw_path,usecols=['PINCP','ESR','PUBCOV','SEX','RAC1P'],low_memory=False)
    report={'created_utc':freeze['created_utc'],'protocol_freeze_sha256':check.sha(root/'protocol_freeze.json'),
        'raw_sha256':check.sha(raw_path),'source_forward_trajectories':3,'new_scientific_result_models':0,
        'new_observer_or_auditor_fits':0,'reserved_labels_read':False,'seeds':{},'total_steps':0}
    for seed in (0,1,2):
        started=time.perf_counter();context=seed_context(out,cfg,seed,raw,reuse)
        directory=out/f'seed_{seed}'/'training'; original=context['forks']['F']
        state=copy.deepcopy(original['model_state']);optimizer=copy.deepcopy(original['mapper_optimizer'])
        caller_rng=torch.get_rng_state().clone();names=[name for name in state if name.startswith('branches.')]
        n=len(context['pca']);batches=(n+255)//256
        mapper_mask=np.concatenate([np.full(state[name].numel(),'.mapper.' in name) for name in names])
        rng=np.random.default_rng(1280200+100*seed);details=[];step=0
        destination=root/f'seed_{seed}_steps.jsonl.gz'
        with gzip.open(directory/'T_shared_step_diagnostics.jsonl.gz','rt') as prior_stream,gzip.open(destination,'xt') as output:
            for epoch in range(1,81):
                order=rng.permutation(n)
                for start in range(0,n,256):
                    if time.perf_counter()-tick>freeze['maximum_seconds']:
                        raise TimeoutError('Bounded T reconstruction ceiling reached; original fits remain intact')
                    ix=order[start:start+256]; labels={k:y[ix] for k,y in context['labels'].items()}
                    before=state; before_optimizer=optimizer
                    state,source,tasks=native_components(before,context['standardized'][ix],labels)
                    parameters=[state[name] for name in names]
                    task_grad=[torch.autograd.grad(tasks[t],parameters,retain_graph=True,allow_unused=True) for t in TASKS]
                    gradients=torch.autograd.grad(source,parameters,allow_unused=True)
                    values,optimizer=literal_adam(parameters,before_optimizer,gradients)
                    ds=flat(values)-flat(parameters);g=np.stack([flat(v,parameters) for v in task_grad])
                    previous=json.loads(prior_stream.readline());step+=1
                    assert previous['epoch']==epoch and previous['minibatch_start']==start
                    assert previous['batch_indices_sha256']==check.array_hash(ix)
                    assert previous['source_objective']==float(source.detach())
                    assert previous['source_task_losses']=={t:float(tasks[t].detach()) for t in TASKS}
                    state={k:v.detach() for k,v in state.items()};state.update(zip(names,values))
                    row={'seed':seed,'epoch':epoch,'minibatch_start':start,'batch_indices_sha256':check.array_hash(ix),
                        'original_forward_optimizer_steps':60*batches+step,'source_support':previous['source_support'],
                        'source_objective':previous['source_objective'],'source_task_losses':previous['source_task_losses'],
                        'source_displacement_l2':float(np.linalg.norm(ds)),'full_source_displacement_dots':(g@ds).tolist(),
                        'mapper_only_source_displacement_dots':(g[:,mapper_mask]@ds[mapper_mask]).tolist(),
                        'raw_increment_l2':0.,'ideal_increment_l2':0.,'actual_increment_l2':0.,'projection_distance':0.,
                        'raw_dots':[0.,0.,0.],'ideal_dots':[0.,0.,0.],'actual_increment_dots':[0.,0.,0.],
                        'projection_applied':False,'active_subset':[],
                        'source_full_accepted_are_exact_aliases':True,'original_source_loss_record_exact':True}
                    output.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
                    if start==0 and epoch in (1,20,40,60,80):
                        for interface in ('F','P'):
                            evidence=torch.load(directory/(interface+'_T')/f'diagnostic_epoch_{epoch:03d}.pt',map_location='cpu',weights_only=True)
                            assert tree_digest(before)==tree_digest(evidence['pre_model_state'])
                            assert tree_digest(before_optimizer)==tree_digest(evidence['pre_optimizer'])
                            assert_gradient_lists(gradients,evidence['source_gradients'])
                            assert tree_digest(state)==tree_digest(evidence['post_model_state'])
                            assert tree_digest(optimizer)==tree_digest(evidence['post_optimizer'])
                        _,after_source,after_tasks=native_components(state,context['standardized'][ix],labels)
                        native_before={'source':float(source.detach()),'tasks':previous['source_task_losses']}
                        native_after={'source':float(after_source.detach()),'tasks':{t:float(after_tasks[t].detach()) for t in TASKS}}
                        details.append({'epoch':epoch,'state_and_Adam_exact_for_both_interfaces':True,
                            'native_losses':{'pre':native_before,'source':native_after,'full':native_after,'accepted':native_after},
                            **{k:row[k] for k in ('source_displacement_l2','full_source_displacement_dots','mapper_only_source_displacement_dots')}})
            assert not prior_stream.readline()
        for interface in ('F','P'):
            final=torch.load(directory/(interface+'_T')/'final.pt',map_location='cpu',weights_only=True)
            assert tree_digest(state)==tree_digest(final['model_state'])
            assert tree_digest(optimizer)==tree_digest(final['mapper_optimizer'])
            assert final['counts']['mapper_optimizer_steps']==140*batches and final['schedule_state']['completed_epochs']==80
            assert torch.equal(final['torch_rng_state'],original['torch_rng_state'])
        assert torch.equal(caller_rng,torch.get_rng_state())
        report['seeds'][str(seed)]={'steps':step,'all_original_source_loss_records_exact':True,
            'both_final_model_and_Adam_states_exact':True,'five_fixed_pre_post_states_exact_for_both_interfaces':True,
            'diagnostic_steps':details,'step_diagnostics_path':str(destination.relative_to(ROOT)),
            'step_diagnostics_sha256':check.sha(destination),'runtime_seconds':time.perf_counter()-started}
        report['total_steps']+=step
    for path,h in inputs.items():assert check.sha(ROOT/path)==h,'Original fitted evidence changed'
    report.update(passed=True,original_evidence_unchanged=True,runtime_seconds=time.perf_counter()-tick)
    (root/'REPLAY.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/redesign_20260909_acs_source_guard_v1')
    args=parser.parse_args();torch.set_num_threads(1)
    with threadpool_limits(limits=1):result=run(args.out.resolve())
    print(json.dumps({'passed':result['passed'],'steps':result['total_steps'],'runtime_seconds':result['runtime_seconds']}))
