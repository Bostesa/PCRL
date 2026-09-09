"""Fixed finite coefficient comparison; no historical mutation or prefix refitting."""
from __future__ import annotations
import argparse
import ast
import datetime
import json
import shutil
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from experiments import run_acs_coalition as hist
from experiments.acs_transfer_data import array_hash, audit_labels, sha_file, support, write_json
from experiments.acs_transfer_heads import _state_hash, metrics
from experiments.run_acs_selective import load_context
from experiments.run_acs_bottleneck import source_binaries
from experiments.run_acs_protection import ROOT, TASKS, now, read
from experiments.run_acs_transfer import subset_indices

DEFAULT_OUT=ROOT/'results/redesign_20260909_acs_coalition_strength_v1'
BETAS=(0.,.025,.05,.1,.2)
NEW_BETAS=(.025,.05,.2)


def condition_name(interface,family,beta):
    if interface not in ('F','P') or family not in ('I','Iplus','J') or beta not in BETAS:raise ValueError('Fixed condition identity required')
    if beta==0:return interface+'_I'
    if family=='I':raise ValueError('I is the shared zero anchor')
    return interface+'_'+family+'_b'+str(beta).replace('.','p')


def matrix(cfg,out):
    entries=[];parent=ROOT/cfg['coalition_reference_results']
    for seed in (0,1,2):
        for interface in ('F','P'):
            for beta in BETAS:
                for family in (('I',) if beta==0 else ('Iplus','J')):
                    name=condition_name(interface,family,beta);reused=beta in (0.,.1)
                    original=interface+'_'+family if reused else name
                    directory=(parent if reused else out)/f'seed_{seed}'
                    entries.append({'seed':seed,'condition':name,'interface':interface,'family':family,'beta':beta,'reused':reused,'original_condition':original,
                        'evidence_directory':str((directory/original).relative_to(ROOT)),
                        'training_checkpoint':str((directory/'training'/original/'final.pt').relative_to(ROOT)),
                        'historical_fork':str((parent/f'seed_{seed}'/'training'/(interface+'_I')/'fork.pt').relative_to(ROOT))})
    assert len(entries)==54 and len({(e['seed'],e['condition']) for e in entries})==54
    return entries


def check_config(cfg):
    old=read(ROOT/cfg['coalition_reference_results']/'config.json')
    for key in ('head_budget','attacker_budget','head_mlp_epochs','audit_mlp_epochs','catchup_epochs','audit_restart_offset','audit_min_samples_leaf','margins','utility_tasks','seeds','threads','parity_tolerance','extended_audit_epochs','target_order','view_ids','observer_seed_base','fresh_seed_base','catchup_seed_base','attacker_subset_seed_base','source_task_assignment','reserved_task_assignment','audit_roles','observer_roles','training','forward_parameter_count','extra_local_reduction'):
        assert cfg[key]==old[key],key
    assert cfg['study']=='acs_coalition_strength_v1' and cfg['beta_grid']==list(BETAS) and cfg['new_betas']==list(NEW_BETAS)
    assert cfg['ordinary_individual_coefficient']==.1 and cfg['families']==['Iplus','J']
    assert cfg['maximum_scientific_seconds']==10800 and cfg['maximum_total_work_seconds']==21600 and cfg['reserved_reporting_seconds']==2700
    assert cfg['global_release_freeze_required'] and cfg['total_paired_systems']==54 and cfg['new_paired_systems']==36 and cfg['reused_paired_systems']==18
    expected=[condition_name(v,f,b) for v in ('F','P') for b in NEW_BETAS for f in ('Iplus','J')]
    assert cfg['new_condition_order']==expected
    assert cfg['execution_order']=={'training':{'seeds':[0,1,2],'conditions':expected},'global_release_freeze':'all54 before new reserved heads/audits','evaluation':{'seeds':[0,1,2],'conditions':expected}}
    for e in matrix(cfg,DEFAULT_OUT):
        b=e['beta'];f=e['family']
        assert cfg['objectives'][e['condition']]=={'individual':-.1,'extra_local':-b if f=='Iplus' else 0.,'coalition':-b if f=='J' else 0.}
    assert cfg['audit_mlp_epochs']==cfg['catchup_epochs']==120 and cfg['extended_audit_epochs']==360


def source_paths():
    pending=[ROOT/'experiments/run_acs_coalition_strength.py',ROOT/'experiments/acs_coalition_strength_training.py',ROOT/'scripts/run_acs_coalition_strength_bounded.py'];found=set()
    while pending:
        path=pending.pop()
        if path in found:continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names=[]
            if isinstance(node,ast.Import):names=[n.name for n in node.names]
            if isinstance(node,ast.ImportFrom) and node.module:
                names=[node.module]+[node.module+'.'+n.name for n in node.names]
            for name in names:
                p=ROOT.joinpath(*name.split('.')).with_suffix('.py')
                if p.exists() and p not in found:pending.append(p)
    return sorted(found)


def frozen_hashes(out):
    return {str(p.relative_to(ROOT)):sha_file(p) for p in [out/'PROTOCOL.md',out/'PREFIT_REVIEW.md',out/'config.json',out/'comparison_rules.json',*source_paths()]}


def verify_freeze(out):
    cfg=read(out/'config.json');check_config(cfg);freeze=read(out/'protocol_freeze.json')
    assert frozen_hashes(out)==freeze['scientific_and_protocol_sha256'],'Unrecorded scientific/configuration amendment'
    assert sha_file(out/'REUSE_MANIFEST.json')==freeze['reuse_manifest_sha256']
    assert sha_file(out/'PREFIT_IDENTITY.json')==freeze['prefit_identity_sha256']
    return cfg


def verify_complete(path):
    record=read(path)
    for name,h in record['files_sha256'].items():assert sha_file(ROOT/name)==h,name
    return record


def complete_record(directory,filename,extra):
    p=directory/filename
    if p.exists():raise FileExistsError(p)
    hashes={str(q.relative_to(ROOT)):sha_file(q) for q in sorted(directory.rglob('*')) if q.is_file()}
    record={'completed_utc':now(),'files_sha256':hashes,**extra};write_json(p,record);return record


def prepare(out):
    cfg=read(out/'config.json');check_config(cfg)
    assert not (out/'protocol_freeze.json').exists(),'Never replace a frozen protocol'
    parent=ROOT/cfg['coalition_reference_results'];hist.verify_freeze(parent)
    historical_files={}
    # Bind all retained objects and compact scores to the reviewed, completed study.
    for seed in (0,1,2):
        directory=parent/f'seed_{seed}';c=read(directory/'complete.json')
        for name,h in c['compact_sha256'].items():
            p=directory/name;assert sha_file(p)==h;historical_files[str(p.relative_to(ROOT))]=h
        for e in read(directory/'local_artifacts.json'):
            assert sha_file(ROOT/e['path'])==e['sha256'],e['path'];historical_files[e['path']]=e['sha256']
        for name in ('complete.json','local_artifacts.json','release_freeze.json'):
            p=directory/name;historical_files[str(p.relative_to(ROOT))]=sha_file(p)
    for name in ('config.json','PROTOCOL.md','protocol_freeze.json','EXECUTION_AMENDMENTS.json','SCORE_REPLAY.json','TRAINING_REPLAY.json','FINAL_REPORT_REPLAY.json','FITTING_COUNTS.json'):
        p=parent/name;historical_files[str(p.relative_to(ROOT))]=sha_file(p)
    identities={}
    for seed in (0,1,2):
        frame,cohort,pools,refs,pre=load_context(cfg,seed)
        olddir=parent/f'seed_{seed}';training=read(olddir/'training/training.json')
        with np.load(olddir/'split_rows.npz') as rows:
            assert set(rows.files)==set(pools)
            for pool,ix in pools.items():assert np.array_equal(frame.iloc[ix]._raw_row.to_numpy(),rows[pool])
        assert pre==read(olddir/'parent_provenance.json')['preprocessing']
        source=source_binaries(frame.iloc[pools['representation_fit']],refs.source.selection['income_edges']);attrs=audit_labels(frame.iloc[pools['representation_fit']])
        from experiments.acs_coalition_strength_training import restore_fork
        forks={}
        for interface in ('F','P'):
            fork=olddir/'training'/(interface+'_I')/'fork.pt'
            cp=torch.load(fork,weights_only=True)
            model,observers,optimizer,observer_optimizer=restore_fork(cp,pre,interface,seed)
            # Helper checks complete model, observers, both Adam states and schedule.
            cp=torch.load(fork,weights_only=True)
            from experiments.acs_coalition_training import _hashes
            from experiments import acs_bottleneck_training as base
            tensor_hashes={'model':_state_hash(model),'adversaries':_state_hash(observers),
                'mapper_optimizer':base.tree_digest(optimizer.state_dict()),'adversary_optimizer':base.tree_digest(observer_optimizer.state_dict())}
            for family in ('I','Iplus','J'):
                oldarm=training['arms'][interface+'_'+family]
                assert tensor_hashes==oldarm['fork_hashes']
                final=torch.load(olddir/'training'/(interface+'_'+family)/'final.pt',weights_only=True)
                assert final['counts']==oldarm['counts']
                assert final['schedule_state']['completed_epochs']==80
                assert _state_hash_from_dict(final['model_state'])==oldarm['final_hashes']['model']
                assert _state_hash_from_dict(final['adversary_state'])==oldarm['final_hashes']['adversaries']
            forks[interface]={'path':str(fork.relative_to(ROOT)),'sha256':sha_file(fork),'tensor_optimizer_hashes':tensor_hashes,'counts':cp['counts'],'schedule_state':cp['schedule_state'],'torch_rng_sha256':array_hash(cp['torch_rng_state'].numpy())}
        identities[str(seed)]={'preprocessing':pre,'pool_raw_row_hashes':{p:array_hash(frame.iloc[ix]._raw_row.to_numpy()) for p,ix in pools.items()},'source_label_hashes':{k:array_hash(v) for k,v in source.items()},'attribute_label_hashes':{k:array_hash(v) for k,v in attrs.items()},'forks':forks,'reserved_labels_accessed':False}
    systems=matrix(cfg,out)
    for e in systems:
        if e['reused']:
            dest=ROOT/e['evidence_directory'];e['original_files_sha256']={str(p.relative_to(ROOT)):sha_file(p) for p in (dest/'metrics.json',dest/'selection_before_test.json',dest/'releases.npz',dest/'predictions.npz',ROOT/e['training_checkpoint'])}
            e['original_fitting_exposure']={'source_epochs':140,'observer_passes':260,'fresh_attacker_epochs':[120,360],'catchup_epochs':[120,360],'saved_start_epoch0_eligible':True,'utility_labels':2048,'utility_mlp_epochs':40}
    reuse={'created_utc':now(),'original_commit':cfg['starting_commit'],'systems':systems,'historical_files_sha256':historical_files,'controls':{str(s):{'directory':str((parent/f'seed_{s}'/'controls').relative_to(ROOT)),'direct_E_directory':str((parent/f'seed_{s}'/'E').relative_to(ROOT)),'native_source':str((parent/f'seed_{s}'/'native_source.json').relative_to(ROOT)),'split_rows':str((parent/f'seed_{s}'/'split_rows.npz').relative_to(ROOT))} for s in (0,1,2)},'regeneration':[],'historical_training_or_audit_refits':0,'original_scientific_hashes_preserved':True,'reuse_validation':'Historical complete/hash records and independent SCORE_REPLAY/TRAINING_REPLAY certificates; exact current files bound before new fitting'}
    write_json(out/'REUSE_MANIFEST.json',reuse);write_json(out/'PREFIT_IDENTITY.json',{'created_utc':now(),'seeds':identities})
    write_json(out/'protocol_freeze.json',{'created_utc':now(),'starting_commit':cfg['starting_commit'],'scientific_and_protocol_sha256':frozen_hashes(out),'reuse_manifest_sha256':sha_file(out/'REUSE_MANIFEST.json'),'prefit_identity_sha256':sha_file(out/'PREFIT_IDENTITY.json'),'historical_sources':'original records remain unchanged','regeneration':[]})
    update_matrix(out)


def _state_hash_from_dict(state):
    from experiments.acs_transfer_models import state_digest
    return state_digest(state)


def ordered_new(out):
    cfg=read(out/'config.json');entries=matrix(cfg,out)
    return [next(e for e in entries if e['seed']==s and e['condition']==n) for s in (0,1,2) for n in cfg['new_condition_order']]


def update_matrix(out):
    cfg=read(out/'config.json');entries=matrix(cfg,out)
    for e in entries:
        directory=out/f"seed_{e['seed']}";name=e['condition']
        e['training_complete']=e['reused'] or (directory/'training'/name/'complete.json').exists()
        e['evaluation_complete']=e['reused'] or (directory/name/'complete.json').exists()
    train=sum(e['training_complete'] for e in entries);evaluate=sum(e['evaluation_complete'] for e in entries)
    record={'updated_utc':now(),'status':'complete' if train==evaluate==54 else 'partial','systems':entries,'planned_paired_systems':54,'reused_paired_systems':18,'new_paired_systems_trained':train-18,'new_paired_systems_evaluated':evaluate-18,'completed_paired_systems':evaluate,'new_branch_mappers':2*(train-18),'historical_prefix_refits':0,'global_release_manifest_frozen':(out/'RELEASE_MANIFEST.json').exists(),'missing_training':[e['condition']+'/seed'+str(e['seed']) for e in entries if not e['training_complete']],'missing_evaluation':[e['condition']+'/seed'+str(e['seed']) for e in entries if not e['evaluation_complete']]}
    write_json(out/'EXECUTED_MATRIX.json',record);write_json(out/'progress.json',record);return record


def train_unit(out,cfg,e):
    from experiments.acs_coalition_strength_training import train_continuation
    seed=e['seed'];name=e['condition'];directory=out/f'seed_{seed}';dest=directory/'training'/name
    if (dest/'complete.json').exists():verify_complete(dest/'complete.json');return
    if dest.exists():raise RuntimeError('Preserve partial training and record recovery before continuing: '+str(dest))
    print(f'TRAIN seed{seed} {name}: exact historical fork,80 continuation epochs',flush=True)
    tick=time.perf_counter();frame,cohort,pools,refs,pre=load_context(cfg,seed);pca=refs.releases['E_pca'];edges=refs.source.selection['income_edges']
    directory.mkdir(exist_ok=True);(directory/'training').mkdir(exist_ok=True)
    rows={p:frame.iloc[ix]._raw_row.to_numpy() for p,ix in pools.items()}
    if (directory/'split_rows.npz').exists():
        with np.load(directory/'split_rows.npz') as z:assert all(np.array_equal(z[p],v) for p,v in rows.items())
    else:np.savez_compressed(directory/'split_rows.npz',**rows)
    source=source_binaries(frame.iloc[pools['representation_fit']],edges);attrs=audit_labels(frame.iloc[pools['representation_fit']]);sv=source_binaries(frame.iloc[pools['source_validation']],edges)
    parent=ROOT/cfg['coalition_reference_results']/f'seed_{seed}'
    trained=train_continuation(pca['representation_fit'],source,attrs,pre,ROOT/e['historical_fork'],read(parent/'training/training.json'),seed,dest,interface=e['interface'],regime=e['family'],beta=e['beta'],pca_val=pca['source_validation'],source_val=sv)
    wire,derived=hist.arrange(trained['model'],pca,e['interface'])
    values={f'{space}/{view}/{pool}':x for space,views in [('wire',wire),('derived',derived)] for view,arr in views.items() for pool,x in arr.items()}
    np.savez_compressed(dest/'frozen_releases.npz',**values)
    write_json(dest/'release_freeze.json',{'created_utc':now(),'seed':seed,'condition':name,'beta':e['beta'],'family':e['family'],'model_sha256':_state_hash(trained['model']),'observer_sha256':_state_hash(trained['observers']),'outputs':hist.view_hashes(wire,derived),'checkpoint_sha256':sha_file(dest/'final.pt'),'reserved_labels_accessed':False,'protocol_sha256':sha_file(out/'protocol_freeze.json')})
    complete_record(dest,'complete.json',{'seed':seed,'condition':name,'training_complete':True,'reserved_labels_accessed':False,'unit_wall_seconds':time.perf_counter()-tick});update_matrix(out)


def freeze_all_releases(out):
    cfg=verify_freeze(out);dest=out/'RELEASE_MANIFEST.json'
    if dest.exists():
        record=read(dest)
        for e in record['systems']:
            assert sha_file(ROOT/e['checkpoint_path'])==e['checkpoint_sha256']
            assert sha_file(ROOT/e['release_record_path'])==e['release_record_sha256']
        return record
    entries=[];reuse=read(out/'REUSE_MANIFEST.json')
    for e in matrix(cfg,out):
        if e['reused']:
            p=ROOT/e['evidence_directory'];r=p.parent/'release_freeze.json';checkpoint=ROOT/e['training_checkpoint'];source=read(r)
            outputs=source['outputs'][e['original_condition']]
        else:
            p=out/f"seed_{e['seed']}"/'training'/e['condition'];verify_complete(p/'complete.json');r=p/'release_freeze.json';source=read(r);checkpoint=p/'final.pt';outputs=source['outputs']
        entries.append({**e,'checkpoint_path':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':sha_file(checkpoint),'release_record_path':str(r.relative_to(ROOT)),'release_record_sha256':sha_file(r),'outputs':outputs})
    assert len(entries)==54
    record={'created_utc':now(),'all_54_systems_frozen':True,'new_reserved_fitting_started':False,'historical_reserved_outcomes_previously_known':True,'systems':entries,'protocol_sha256':sha_file(out/'protocol_freeze.json')}
    write_json(dest,record);update_matrix(out);return record


def load_final(cfg,e):
    from experiments.acs_coalition_training import CoalitionModel,make_observers
    pre=read(ROOT/cfg['coalition_reference_results']/f"seed_{e['seed']}"/'parent_provenance.json')['preprocessing']
    initial,_=hist.historical_initial(cfg,e['seed']);cp=torch.load(ROOT/e['training_checkpoint'],weights_only=True)
    model=CoalitionModel(pre,initial);model.load_state_dict(cp['model_state']);model.freeze()
    observers,_=make_observers(e['interface'],e['seed']);observers.load_state_dict(cp['adversary_state']);observers.eval().requires_grad_(False)
    return model,observers


def evaluate_unit(out,cfg,e):
    from experiments.acs_coalition_audits import fit_condition_audits
    global_freeze=freeze_all_releases(out);assert global_freeze['all_54_systems_frozen']
    seed=e['seed'];name=e['condition'];directory=out/f'seed_{seed}';dest=directory/name
    if (dest/'complete.json').exists():verify_complete(dest/'complete.json');return
    if dest.exists():raise RuntimeError('Preserve partially fitted evidence and record recovery before continuing: '+str(dest))
    tick=time.perf_counter();dest.mkdir();print(f'EVALUATE seed{seed} {name}: utilities,11 wire roles,public compositions,9 catch-ups',flush=True)
    frame,cohort,pools,refs,pre=load_context(cfg,seed);pca=refs.releases['E_pca'];edges=refs.source.selection['income_edges'];model,observers=load_final(cfg,e)
    wire,derived=hist.arrange(model,pca,e['interface']);before=hist.view_hashes(wire,derived);training_freeze=read(directory/'training'/name/'release_freeze.json');assert before==training_freeze['outputs']
    state=_state_hash(model);ostate=_state_hash(observers)
    for views in (wire,derived):
        for a in views.values():
            for value in a.values():value.setflags(write=False)
    # The sole new reserved-label access boundary is after the complete54 global freeze.
    labels={p:hist.all_labels(frame.iloc[pools[p]],edges) for p in ('downstream_fit','downstream_validation','attacker_fit','attacker_validation')}
    weights={p:frame.iloc[pools[p]].PWGTP.to_numpy(float) for p in labels}
    ti={t:subset_indices(labels['downstream_fit'][t],cfg['head_budget'],1230000+100*seed+j) for j,t in enumerate(TASKS)}
    ai={t:subset_indices(labels['attacker_fit'][t],cfg['attacker_budget'],1240000+100*seed+j) for j,t in enumerate(hist.TARGETS)}
    indices={'task_fit_indices':{t:{'n':len(i),'pool_indices_sha256':array_hash(i),'raw_rows_sha256':array_hash(frame.iloc[pools['downstream_fit'][i]]._raw_row.to_numpy())} for t,i in ti.items()},'audit_fit_indices':{t:{'n':len(i),'pool_indices_sha256':array_hash(i),'raw_rows_sha256':array_hash(frame.iloc[pools['attacker_fit'][i]]._raw_row.to_numpy())} for t,i in ai.items()}}
    parent=ROOT/cfg['coalition_reference_results']/f'seed_{seed}';assert indices==read(parent/'indices.json')
    hist.write_once_same(directory/'indices.json',indices);hist.write_once_same(directory/'support.json',{p:support(ys,hist.TARGETS) for p,ys in labels.items()})
    write_json(dest/'fit_start.json',{'started_utc':now(),'global_release_manifest_sha256':sha_file(out/'RELEASE_MANIFEST.json'),'all_54_frozen':True,'first_reserved_labels_after_global_freeze':True})
    times={};t=time.perf_counter();utility=hist.fit_utilities(wire,labels,ti,seed,dest);times['utility_seconds']=time.perf_counter()-t
    exposure=read(parent/(e['interface']+'_I')/'selection_before_test.json')['inherited_exposure']
    t=time.perf_counter();audits=fit_condition_audits({v:{p:x for p,x in a.items() if p.startswith('attacker_')} for v,a in wire.items()}, {v:{p:x for p,x in a.items() if p.startswith('attacker_')} for v,a in derived.items()}, {p:y for p,y in labels.items() if p.startswith('attacker_')},ai,observers,seed,dest/'audits',interface=e['interface'],inherited_exposure=exposure)
    times['audit_seconds']=time.perf_counter()-t
    write_json(dest/'selection_before_test.json',{'created_utc':now(),'utility':utility['selection'],'utility_metadata':utility['metadata'],'audits':audits['selection'],'release_freeze_sha256':sha_file(out/'RELEASE_MANIFEST.json'),'inherited_exposure':exposure,'coefficient_identity':{'interface':e['interface'],'family':e['family'],'beta':e['beta']}})
    # Development scoring follows this condition's frozen unweighted-validation selections.
    tf=frame.iloc[pools['test']];testpca=refs.evaluation_releases(tf)['E_pca'];ty=hist.all_labels(tf,edges);tw=tf.PWGTP.to_numpy(float)
    testwire=model.wires(testpca,e['interface']);testderived=model.wires(testpca,'P')
    t=time.perf_counter();hist.score_condition(seed,name,wire,derived,testwire,testderived,utility,audits,labels,ty,weights,tw,dest)
    save={f'{space}/{view}/{p}':x for space,views,test in [('wire',wire,testwire),('derived',derived,testderived)] for view,a in views.items() for p,x in {**a,'test':test[view]}.items()}
    np.savez_compressed(dest/'releases.npz',**save)
    composition=[]
    for pool in (*pca,'test'):
        w=wire if pool!='test' else None
        if e['interface']=='F':
            for view in ('A','B'):
                features=wire[view][pool] if pool!='test' else testwire[view]
                with torch.no_grad():value=torch.cat([torch.sigmoid(head(torch.from_numpy(features.copy()))) for head in model.branches[view].heads.values()],1).numpy()
                expected=derived[view][pool] if pool!='test' else testderived[view];assert np.array_equal(value,expected)
                composition.append({'condition':name,'view':view,'pool':pool,'exact':True,'public_heads_only':True,'raw_rows_requested':False,'derived_sha256':array_hash(value)})
        else:
            for view in ('A','B','AB'):
                assert np.array_equal(wire[view][pool] if pool!='test' else testwire[view],derived[view][pool] if pool!='test' else testderived[view])
    write_json(dest/'public_composition_parity.json',{'records':composition,'P_deduplicated':e['interface']=='P','all_exact':True})
    native=[];pred={};sv=source_binaries(frame.iloc[pools['source_validation']],edges)
    for split,x,y,w in [('source_validation',pca['source_validation'],sv,frame.iloc[pools['source_validation']].PWGTP.to_numpy(float)),('test',testpca,source_binaries(tf,edges),tw)]:
        for target,prob in model.native_probabilities(x).items():
            valid=y[target]>=0;pred[f'{name}/{target}/{split}']=prob
            native.append({'seed':seed,'condition':name,'view':'B' if target=='public_coverage' else 'A','target':target,'split':split,'score':metrics(y[target][valid],prob[valid],2),'person_weighted':metrics(y[target][valid],prob[valid],2,w[valid]),'prediction_sha256':array_hash(prob),'selection':'fixed native head'})
    write_json(dest/'native_source.json',{'raw_metrics':native});np.savez_compressed(dest/'native_predictions.npz',**pred)
    assert state==_state_hash(model) and ostate==_state_hash(observers) and before==hist.view_hashes(wire,derived)
    times['read_only_scoring_seconds']=time.perf_counter()-t;times['unit_wall_seconds']=time.perf_counter()-tick
    write_json(dest/'runtime.json',times)
    local=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha_file(p)} for p in dest.rglob('*') if p.is_file() and (p.suffix in ('.pt','.npz','.joblib') or 'fitted' in p.parts)]
    write_json(dest/'local_artifacts.json',local)
    complete_record(dest,'complete.json',{'seed':seed,'condition':name,'evaluation_complete':True,'global_release_manifest_sha256':sha_file(out/'RELEASE_MANIFEST.json'),'unit_wall_seconds':time.perf_counter()-tick});update_matrix(out)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=DEFAULT_OUT);ap.add_argument('--phase',choices=['prepare','train','freeze','evaluate'],required=True);ap.add_argument('--max-units',type=int,default=36);args=ap.parse_args();out=args.out.resolve()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if args.phase=='prepare':prepare(out);return
        cfg=verify_freeze(out)
        if args.phase=='freeze':freeze_all_releases(out);return
        if args.phase=='evaluate':freeze_all_releases(out)
        completed=0
        for e in ordered_new(out):
            path=out/f"seed_{e['seed']}"/('training' if args.phase=='train' else '')/e['condition']/'complete.json'
            if path.exists():verify_complete(path);continue
            elapsed=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(cfg['started_utc'])).total_seconds()
            if elapsed>=cfg['maximum_total_work_seconds']-cfg['reserved_reporting_seconds']:break
            try:(train_unit if args.phase=='train' else evaluate_unit)(out,cfg,e)
            except Exception as exc:
                write_json(out/f"failure_{args.phase}_{e['seed']}_{e['condition']}_{int(time.time())}.json",{'created_utc':now(),'phase':args.phase,'unit':e,'error':repr(exc),'traceback':traceback.format_exc(),'partial_artifacts_preserved':True});raise
            completed+=1
            if completed>=args.max_units:break
        if args.phase=='train' and all((out/f"seed_{e['seed']}"/'training'/e['condition']/'complete.json').exists() for e in ordered_new(out)):freeze_all_releases(out)
        update_matrix(out)
if __name__=='__main__':main()
