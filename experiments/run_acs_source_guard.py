"""Explicit source-guard orchestration over unchanged coalition evaluation helpers."""
from __future__ import annotations
import argparse,ast,copy,datetime,hashlib,json,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from experiments import run_acs_coalition as hist
from experiments import run_acs_coalition_strength as strength
from experiments.acs_transfer_data import array_hash,audit_labels,sha_file,support,write_json
from experiments.acs_transfer_heads import _state_hash,metrics
from experiments.run_acs_selective import load_context
from experiments.run_acs_bottleneck import source_binaries
from experiments.run_acs_protection import ROOT,TASKS,now,read
from experiments.run_acs_transfer import subset_indices
DEFAULT_OUT=ROOT/'results/redesign_20260909_acs_source_guard_v1'
NEW_ARMS={'G_J':('J',.1),'G_L025':('Iplus',.025),'G_L20':('Iplus',.2),'T':('T',0.)}
verify_complete=strength.verify_complete
complete_record=strength.complete_record


def matrix(cfg,out):
    old=strength.matrix(read(ROOT/cfg['strength_reference_results']/'config.json'),ROOT/cfg['strength_reference_results']);entries=[]
    for seed in (0,1,2):
        for name in cfg['condition_order']:
            interface,arm=name.split('_',1);family,beta=NEW_ARMS[arm]
            entries.append({'seed':seed,'condition':name,'arm':arm,'interface':interface,'family':family,'beta':beta,'guarded':arm!='T','reused':False,'original_condition':name,'forward_alias':f'seed_{seed}/T_shared' if arm=='T' else f'seed_{seed}/{name}','evidence_directory':str((out/f'seed_{seed}'/name).relative_to(ROOT)),'training_checkpoint':str((out/f'seed_{seed}'/'training'/name/'final.pt').relative_to(ROOT)),'historical_fork':str((ROOT/cfg['coalition_reference_results']/f'seed_{seed}'/'training'/(interface+'_I')/'fork.pt').relative_to(ROOT))})
        for interface in ('F','P'):
            for arm,family,beta in [('U_J','J',.1),('U_L025','Iplus',.025),('U_L20','Iplus',.2),('I','I',0.)]:
                e=copy.deepcopy(next(x for x in old if x['seed']==seed and x['interface']==interface and x['family']==family and x['beta']==beta))
                e.update(arm=arm,guarded=False,reused=True,forward_alias=f"historical/{seed}/{e['condition']}");entries.append(e)
    assert len(entries)==48 and len({(e['seed'],e['condition']) for e in entries})==48
    return entries


def check_config(cfg):
    old=read(ROOT/cfg['strength_reference_results']/'config.json')
    for key in ('head_budget','attacker_budget','head_mlp_epochs','audit_mlp_epochs','catchup_epochs','audit_restart_offset','audit_min_samples_leaf','margins','utility_tasks','seeds','threads','parity_tolerance','extended_audit_epochs','target_order','view_ids','observer_seed_base','fresh_seed_base','catchup_seed_base','attacker_subset_seed_base','source_task_assignment','reserved_task_assignment','audit_roles','observer_roles','training','forward_parameter_count','extra_local_reduction'):
        assert cfg[key]==old[key],key
    assert cfg['study']=='acs_source_guard_v1'
    assert (cfg['maximum_scientific_seconds'],cfg['maximum_total_work_seconds'],cfg['reserved_reporting_seconds'])==(5400,14400,1800)
    assert (cfg['new_paired_systems'],cfg['reused_paired_systems'],cfg['total_paired_systems'],cfg['new_unique_forward_continuations'],cfg['new_branch_mappers'])==(24,24,48,21,42)
    order=['F_T','P_T','F_G_J','F_G_L025','F_G_L20','P_G_J','P_G_L025','P_G_L20'];assert cfg['condition_order']==order
    assert cfg['execution_order']['evaluation']=={'seeds':[0,1,2],'conditions':order}
    assert cfg['execution_order']['training']=={'seeds':[0,1,2],'conditions':['shared_T',*order[2:]]}
    assert cfg['audit_mlp_epochs']==cfg['catchup_epochs']==120 and cfg['extended_audit_epochs']==360
    assert cfg['guard']['svd_relative_rank_tolerance']==1e-12 and cfg['guard']['diagnostic_epochs']==[1,20,40,60,80]
    for v in ('F','P'):
        for arm,(f,b) in NEW_ARMS.items():
            assert cfg['objectives'][v+'_'+arm]=={'family':f,'beta':b,'guarded':arm!='T','individual':-.1 if arm!='T' else 0.,'extra_local':-b if f=='Iplus' else 0.,'coalition':-b if f=='J' else 0.}


def source_paths():
    pending=[ROOT/p for p in ('experiments/run_acs_source_guard.py','experiments/acs_source_guard_training.py','scripts/run_acs_source_guard_bounded.py')];found=set()
    while pending:
        path=pending.pop()
        if path in found:continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names=[]
            if isinstance(node,ast.Import):names=[n.name for n in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:names=[node.module]+[node.module+'.'+n.name for n in node.names]
            for name in names:
                p=ROOT.joinpath(*name.split('.')).with_suffix('.py')
                if p.exists() and p not in found:pending.append(p)
    return sorted(found)


def comparison_implementation_hashes():
    path=ROOT/'scripts/report_acs_source_guard.py';tree=ast.parse(path.read_text())
    names=('registry','score_key','parent_scores','points_from_index','pair_cube','aggregate_pairs','comparison_endpoints','fixed_contrasts','feasibility_rows','vector_tradeoffs','pareto_rows','criteria_rows')
    nodes={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    hashes={name:hashlib.sha256(ast.dump(nodes[name],include_attributes=False).encode()).hexdigest() for name in names}
    constants={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('ARMS','NEW_ARMS','GUARDED_ARMS','PAIR_FIELDS'):
                    constants[target.id]=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
    assert len(constants)==4
    return {'functions_ast_sha256':hashes,'constants_ast_sha256':constants,'historical_helpers_sha256':{str(p.relative_to(ROOT)):sha_file(p) for p in (ROOT/'scripts/acs_coalition_strength_comparisons.py',ROOT/'scripts/summarize_acs_coalition.py',ROOT/'scripts/summarize_acs_coalition_strength.py')}}


def frozen_hashes(out):
    return {str(p.relative_to(ROOT)):sha_file(p) for p in [out/'PROTOCOL.md',out/'PREFIT_REVIEW.md',out/'config.json',out/'comparison_rules.json',out/'NUMERICAL_POLICY.json',*source_paths()]}


def verify_freeze(out):
    cfg=read(out/'config.json');check_config(cfg);f=read(out/'protocol_freeze.json')
    assert f['scientific_and_protocol_sha256']==frozen_hashes(out),'Frozen scientific recipe changed'
    assert f['comparison_implementation']==comparison_implementation_hashes(),'Frozen comparison definitions changed'
    for name,key in [('REUSE_MANIFEST.json','reuse_manifest_sha256'),('PREFIT_IDENTITY.json','prefit_identity_sha256')]:assert sha_file(out/name)==f[key]
    return cfg


def prepare(out):
    from experiments.acs_coalition_strength_training import restore_fork
    from experiments import acs_bottleneck_training as base
    cfg=read(out/'config.json');check_config(cfg);assert not (out/'protocol_freeze.json').exists()
    prior=ROOT/cfg['strength_reference_results'];strength.verify_freeze(prior)
    systems=matrix(cfg,out);hashes={};regenerated=[]
    previous=read(prior/'REUSE_MANIFEST.json')
    # Existing certificates establish the history; bind actual preserved bytes.
    for name,h in previous['historical_files_sha256'].items():
        assert sha_file(ROOT/name)==h,name;hashes[name]=h
    for e in systems:
        if not e['reused']:continue
        d=ROOT/e['evidence_directory'];cp=ROOT/e['training_checkpoint']
        if (d/'complete.json').exists():
            complete=verify_complete(d/'complete.json');hashes.update(complete['files_sha256'])
            hashes[str((d/'complete.json').relative_to(ROOT))]=sha_file(d/'complete.json')
            training=cp.parent;complete=verify_complete(training/'complete.json');hashes.update(complete['files_sha256'])
            hashes[str((training/'complete.json').relative_to(ROOT))]=sha_file(training/'complete.json')
        for p in (d/'metrics.json',d/'selection_before_test.json',d/'releases.npz',d/'predictions.npz',cp):
            hashes[str(p.relative_to(ROOT))]=sha_file(p)
        e['original_files_sha256']={str(p.relative_to(ROOT)):sha_file(p) for p in (d/'metrics.json',d/'selection_before_test.json',d/'releases.npz',d/'predictions.npz',cp)}
        e['original_fitting_exposure']={'source_epochs':140,'observer_passes':260,'fresh_attacker_epochs':[120,360],'catchup_epochs':[120,360],'saved_start_epoch0_eligible':True,'utility_labels':2048,'utility_mlp_epochs':40}
    for name in ('PROTOCOL.md','config.json','comparison_rules.json','REUSE_MANIFEST.json','protocol_freeze.json','PREFIT_IDENTITY.json','RELEASE_MANIFEST.json','SCORE_REPLAY.json','TRAINING_REPLAY.json','FINAL_COMPARISON_REPLAY_PUBLICATION.json','COMPACT_UNIT_EVIDENCE.json'):
        p=prior/name;hashes[str(p.relative_to(ROOT))]=sha_file(p)
    identities={};parent=ROOT/cfg['coalition_reference_results']
    for seed in (0,1,2):
        frame,cohort,pools,refs,pre=load_context(cfg,seed);d=parent/f'seed_{seed}';oldid=read(prior/'PREFIT_IDENTITY.json')['seeds'][str(seed)]
        assert pre==oldid['preprocessing']
        poolhash={p:array_hash(frame.iloc[ix]._raw_row.to_numpy()) for p,ix in pools.items()};assert poolhash==oldid['pool_raw_row_hashes']
        source=source_binaries(frame.iloc[pools['representation_fit']],refs.source.selection['income_edges']);attrs=audit_labels(frame.iloc[pools['representation_fit']])
        assert {k:array_hash(y) for k,y in source.items()}==oldid['source_label_hashes'];assert {k:array_hash(y) for k,y in attrs.items()}==oldid['attribute_label_hashes']
        forks={};states=[]
        for v in ('F','P'):
            p=d/'training'/(v+'_I')/'fork.pt';cp=torch.load(p,weights_only=True);m,a,o,ao=restore_fork(cp,pre,v,seed)
            assert sha_file(p)==oldid['forks'][v]['sha256']
            assert sum(x.numel() for x in m.parameters())==6355
            forks[v]={'path':str(p.relative_to(ROOT)),'sha256':sha_file(p),'model':_state_hash(m),'mapper_optimizer':base.tree_digest(o.state_dict()),'observer':_state_hash(a),'observer_optimizer':base.tree_digest(ao.state_dict()),'schedule_state':cp['schedule_state'],'counts':cp['counts'],'rng_sha256':array_hash(cp['torch_rng_state'].numpy())}
            states.append(cp)
        assert base.tree_digest(states[0]['model_state'])==base.tree_digest(states[1]['model_state'])
        assert base.tree_digest(states[0]['mapper_optimizer'])==base.tree_digest(states[1]['mapper_optimizer'])
        assert states[0]['schedule_state']==states[1]['schedule_state'] and states[0]['counts']==states[1]['counts']
        assert torch.equal(states[0]['torch_rng_state'],states[1]['torch_rng_state'])
        identities[str(seed)]={**oldid,'source_only_F_P_forward_Adam_schedule_RNG_exact':True,'source_only_forks':forks,'reserved_labels_accessed':False}
    reuse={'created_utc':now(),'original_commit':cfg['starting_commit'],'systems':systems,'historical_files_sha256':hashes,'controls':previous['controls'],'regeneration':regenerated,'historical_training_or_audit_refits':0,'original_scientific_hashes_preserved':True,'contextual_grid':str(prior/'EXECUTED_MATRIX.json'),'validation':'Published unchanged historical replay certificates and exact retained object hashes; no historical fitting'}
    write_json(out/'REUSE_MANIFEST.json',reuse);write_json(out/'PREFIT_IDENTITY.json',{'created_utc':now(),'seeds':identities})
    write_json(out/'FORWARD_ALIASES.json',{'source_only':[{ 'seed':s,'alias':f'seed_{s}/T_shared','systems':['F_T','P_T'],'forward_fits':1,'observer_sets':2} for s in (0,1,2)],'planned_new_unique_forward_continuations':21,'planned_new_interface_systems':24})
    write_json(out/'protocol_freeze.json',{'created_utc':now(),'starting_commit':cfg['starting_commit'],'scientific_and_protocol_sha256':frozen_hashes(out),'comparison_implementation':comparison_implementation_hashes(),'reuse_manifest_sha256':sha_file(out/'REUSE_MANIFEST.json'),'prefit_identity_sha256':sha_file(out/'PREFIT_IDENTITY.json'),'historical_sources':'original hashes preserved unchanged','regeneration':regenerated})
    update_matrix(out)


def ordered_new(out):
    cfg=read(out/'config.json');entries=matrix(cfg,out)
    return [next(e for e in entries if e['seed']==s and e['condition']==n) for s in (0,1,2) for n in cfg['condition_order']]


def ordered_training(out):
    entries=ordered_new(out)
    return [e for e in entries if e['condition']!='P_T']


def update_matrix(out):
    cfg=read(out/'config.json');entries=matrix(cfg,out)
    for e in entries:
        d=out/f"seed_{e['seed']}";e['training_complete']=e['reused'] or (d/'training'/e['condition']/'complete.json').exists();e['evaluation_complete']=e['reused'] or (d/e['condition']/'complete.json').exists()
    trained=[e for e in entries if e['training_complete'] and not e['reused']];evaluated=[e for e in entries if e['evaluation_complete'] and not e['reused']];unique=len({e['forward_alias'] for e in trained})
    record={'updated_utc':now(),'status':'complete' if len(evaluated)==24 else 'partial','systems':entries,'planned_paired_systems':48,'reused_paired_systems':24,'planned_new_interface_systems':24,'new_paired_systems_trained':len(trained),'new_paired_systems_evaluated':len(evaluated),'completed_paired_systems':len(evaluated)+24,'new_unique_forward_continuations':unique,'new_branch_mappers':2*unique,'historical_prefix_refits':0,'global_release_manifest_frozen':(out/'RELEASE_MANIFEST.json').exists(),'missing_training':[f"seed_{e['seed']}/{e['condition']}" for e in entries if not e['training_complete']],'missing_evaluation':[f"seed_{e['seed']}/{e['condition']}" for e in entries if not e['evaluation_complete']]}
    write_json(out/'EXECUTED_MATRIX.json',record);write_json(out/'progress.json',record);return record


def train_unit(out,cfg,e):
    from experiments.acs_source_guard_training import train_guarded_continuation,train_source_only_shared
    assert not (out/'RELEASE_MANIFEST.json').exists(),'Representation fitting permanently closed after global freeze'
    seed=e['seed'];name=e['condition'];directory=out/f'seed_{seed}';dest=directory/'training'/name
    if (dest/'complete.json').exists():verify_complete(dest/'complete.json');return
    if dest.exists():raise RuntimeError('Partial evidence must be preserved and diagnosed: '+str(dest))
    tick=time.perf_counter();print(f'TRAIN seed{seed} {name}; shared T first; exact fork',flush=True)
    frame,cohort,pools,refs,pre=load_context(cfg,seed);pca=refs.releases['E_pca'];edges=refs.source.selection['income_edges'];directory.mkdir(exist_ok=True);(directory/'training').mkdir(exist_ok=True)
    rows={p:frame.iloc[ix]._raw_row.to_numpy() for p,ix in pools.items()}
    if (directory/'split_rows.npz').exists():
        with np.load(directory/'split_rows.npz') as z:assert all(np.array_equal(z[p],v) for p,v in rows.items())
    else:np.savez_compressed(directory/'split_rows.npz',**rows)
    source=source_binaries(frame.iloc[pools['representation_fit']],edges);attrs=audit_labels(frame.iloc[pools['representation_fit']]);sv=source_binaries(frame.iloc[pools['source_validation']],edges)
    parent=ROOT/cfg['coalition_reference_results']/f'seed_{seed}';metadata=read(parent/'training/training.json')
    if e['arm']=='T':
        paths={v:parent/'training'/(v+'_I')/'fork.pt' for v in ('F','P')}
        trained=train_source_only_shared(pca['representation_fit'],source,attrs,pre,paths,metadata,seed,directory/'training',pca_val=pca['source_validation'],source_val=sv)
        systems=[(next(x for x in matrix(cfg,out) if x['seed']==seed and x['condition']==v+'_T'),trained['arms'][v]) for v in ('F','P')]
    else:
        trained=train_guarded_continuation(pca['representation_fit'],source,attrs,pre,ROOT/e['historical_fork'],metadata,seed,dest,interface=e['interface'],regime=e['family'],beta=e['beta'],pca_val=pca['source_validation'],source_val=sv)
        systems=[(e,trained)]
    for entry,result in systems:
        dest=directory/'training'/entry['condition'];wire,derived=hist.arrange(result['model'],pca,entry['interface'])
        np.savez_compressed(dest/'frozen_releases.npz',**{f'{space}/{view}/{pool}':x for space,views in [('wire',wire),('derived',derived)] for view,arr in views.items() for pool,x in arr.items()})
        write_json(dest/'release_freeze.json',{'created_utc':now(),'seed':seed,'condition':entry['condition'],'beta':entry['beta'],'family':entry['family'],'forward_alias':entry['forward_alias'],'model_sha256':_state_hash(result['model']),'observer_sha256':_state_hash(result['observers']),'outputs':hist.view_hashes(wire,derived),'checkpoint_sha256':sha_file(dest/'final.pt'),'reserved_labels_accessed':False,'protocol_sha256':sha_file(out/'protocol_freeze.json')})
        complete_record(dest,'complete.json',{'seed':seed,'condition':entry['condition'],'training_complete':True,'forward_alias':entry['forward_alias'],'shared_forward_fit':entry['arm']=='T','reserved_labels_accessed':False,'unit_wall_seconds':time.perf_counter()-tick})
    if e['arm']=='T':
        shared=trained['metadata'];write_json(directory/'training'/'T_shared_manifest.json',{'created_utc':now(),**shared,'interface_checkpoints':{v:sha_file(directory/'training'/(v+'_T')/'final.pt') for v in ('F','P')}})
    update_matrix(out)


def freeze_all_releases(out,allow_partial=False):
    cfg=verify_freeze(out);dest=out/'RELEASE_MANIFEST.json'
    if dest.exists():
        record=read(dest)
        for e in record['systems']:
            assert sha_file(ROOT/e['checkpoint_path'])==e['checkpoint_sha256'];assert sha_file(ROOT/e['release_record_path'])==e['release_record_sha256']
        return record
    entries=[];missing=[]
    for e in matrix(cfg,out):
        if e['reused']:
            p=ROOT/e['evidence_directory'];checkpoint=ROOT/e['training_checkpoint'];r=checkpoint.parent/'release_freeze.json'
            if not r.exists():r=p.parent/'release_freeze.json';outputs=read(r)['outputs'][e['original_condition']]
            else:outputs=read(r)['outputs']
        else:
            p=out/f"seed_{e['seed']}"/'training'/e['condition']
            if not (p/'complete.json').exists():missing.append(e);continue
            verify_complete(p/'complete.json');r=p/'release_freeze.json';checkpoint=p/'final.pt';outputs=read(r)['outputs']
        entries.append({**e,'checkpoint_path':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':sha_file(checkpoint),'release_record_path':str(r.relative_to(ROOT)),'release_record_sha256':sha_file(r),'outputs':outputs})
    if missing and not allow_partial:raise RuntimeError('All24 intended new releases must freeze before reserved evaluation')
    record={'created_utc':now(),'all_intended_releases_frozen':not missing,'training_permanently_closed':True,'explicit_partial_prefix':bool(missing),'new_reserved_fitting_started':False,'historical_reserved_outcomes_previously_known':True,'systems':entries,'missing':missing,'protocol_sha256':sha_file(out/'protocol_freeze.json')}
    write_json(dest,record);update_matrix(out);return record


load_final=strength.load_final


def evaluate_unit(out,cfg,e):
    from experiments.acs_coalition_audits import fit_condition_audits
    global_freeze=freeze_all_releases(out);assert global_freeze['training_permanently_closed']
    assert any(x['seed']==e['seed'] and x['condition']==e['condition'] for x in global_freeze['systems'])
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
    # The sole new reserved-label access boundary is after the complete new-system global freeze.
    labels={p:hist.all_labels(frame.iloc[pools[p]],edges) for p in ('downstream_fit','downstream_validation','attacker_fit','attacker_validation')}
    weights={p:frame.iloc[pools[p]].PWGTP.to_numpy(float) for p in labels}
    ti={t:subset_indices(labels['downstream_fit'][t],cfg['head_budget'],1230000+100*seed+j) for j,t in enumerate(TASKS)}
    ai={t:subset_indices(labels['attacker_fit'][t],cfg['attacker_budget'],1240000+100*seed+j) for j,t in enumerate(hist.TARGETS)}
    indices={'task_fit_indices':{t:{'n':len(i),'pool_indices_sha256':array_hash(i),'raw_rows_sha256':array_hash(frame.iloc[pools['downstream_fit'][i]]._raw_row.to_numpy())} for t,i in ti.items()},'audit_fit_indices':{t:{'n':len(i),'pool_indices_sha256':array_hash(i),'raw_rows_sha256':array_hash(frame.iloc[pools['attacker_fit'][i]]._raw_row.to_numpy())} for t,i in ai.items()}}
    parent=ROOT/cfg['coalition_reference_results']/f'seed_{seed}';assert indices==read(parent/'indices.json')
    hist.write_once_same(directory/'indices.json',indices);hist.write_once_same(directory/'support.json',{p:support(ys,hist.TARGETS) for p,ys in labels.items()})
    write_json(dest/'fit_start.json',{'started_utc':now(),'global_release_manifest_sha256':sha_file(out/'RELEASE_MANIFEST.json'),'all_intended_frozen':global_freeze['all_intended_releases_frozen'],'first_reserved_labels_after_global_freeze':True})
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
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=DEFAULT_OUT);ap.add_argument('--phase',choices=['prepare','train','freeze','evaluate'],required=True);ap.add_argument('--max-units',type=int,default=24);ap.add_argument('--allow-partial-freeze',action='store_true');a=ap.parse_args();out=a.out.resolve()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if a.phase=='prepare':prepare(out);return
        cfg=verify_freeze(out)
        if a.phase=='freeze':freeze_all_releases(out,a.allow_partial_freeze);return
        if a.phase=='evaluate':freeze_all_releases(out)
        completed=0
        for e in ordered_training(out) if a.phase=='train' else ordered_new(out):
            p=out/f"seed_{e['seed']}"/('training' if a.phase=='train' else '')/e['condition']/'complete.json'
            if p.exists():verify_complete(p);continue
            if a.phase=='evaluate' and not (out/f"seed_{e['seed']}"/'training'/e['condition']/'complete.json').exists():continue
            elapsed=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(cfg['started_utc'])).total_seconds()
            if elapsed>=cfg['maximum_total_work_seconds']-cfg['reserved_reporting_seconds']:break
            try:(train_unit if a.phase=='train' else evaluate_unit)(out,cfg,e)
            except Exception as exc:
                write_json(out/f"failure_{a.phase}_{e['seed']}_{e['condition']}_{int(time.time())}.json",{'created_utc':now(),'phase':a.phase,'unit':e,'error':repr(exc),'traceback':traceback.format_exc(),'partial_artifacts_preserved':True});raise
            completed+=1
            if completed>=a.max_units:break
        m=update_matrix(out)
        if a.phase=='train' and not m['missing_training']:freeze_all_releases(out)
if __name__=='__main__':main()
