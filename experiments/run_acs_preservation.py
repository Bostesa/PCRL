"""Fixed ACS nonlinear utility/protection comparison, with immutable references.

Reserved outcomes enter only post-freeze heads. Original evaluation people are
DEVELOPMENT EVALUATION. No new LEACE fit, source bank fit or encoder selection.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import platform
import time
import importlib.metadata

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from experiments.acs_transfer_data import (array_hash, audit_labels, load_cohort,
    require_test_selection, sha_file, source_labels, split_households, write_json)
from experiments.acs_transfer_heads import fit_candidates, load_candidate, _hash_array, _state_hash
from experiments.acs_protection_audits import fit_primary_auditors, select_candidates
from experiments.acs_protection_maps import FrozenAffineMap
from experiments.run_acs_protection import (ROOT, TASKS, AUDITS, ParentBundle, binary_tasks,
    feature_rank, frozen_digest, now, read, score_and_save, execution_paths as old_paths)
from experiments.run_acs_transfer import subset_indices, save_candidates
from experiments.acs_bottleneck_training import train_preservation_unit
from experiments.acs_bottleneck_catchup import fit_catchup

from experiments.run_acs_bottleneck import SavedReferences, REFERENCES, source_binaries, execution_paths as historical_paths, check_config as historical_check
from experiments.acs_preservation_diagnostics import fit_snapshots, evaluate_snapshots
DEFAULT_OUT = ROOT/'results/redesign_20260908_acs_preservation_v1'
ARMS = ('C_warmup_only','D_warmup_only','C_persistent','D_persistent')


def beta_name(beta):
    return 'beta_0p1' if beta == .1 else 'beta_1'


def check_config(cfg):
    original=read(ROOT/cfg['init_reference_results']/'config.json')
    historical_check(original)
    for key in ('training','references','utility_tasks','seeds','threads','head_budget',
                'attacker_budget','head_mlp_epochs','audit_mlp_epochs','catchup_epochs',
                'audit_restart_offset','audit_min_samples_leaf','margins','mapper_initialization','parity_tolerance'):
        assert cfg[key]==original[key], 'Fixed inherited recipe changed: '+key
    assert cfg['study']=='acs_pca16_output_preservation_v1'
    assert cfg['preservation_betas']==[.1,1.] and cfg['preservation_schedules']==['warmup_only','persistent']
    assert tuple(cfg['continuation_arms'])==ARMS and tuple(cfg['learned_arms'])==ARMS
    assert cfg['snapshot_releases']==['I','W',*ARMS]
    assert cfg['extended_audit_epochs']==360 and cfg['affine_rank_rcond']==1e-12
    assert cfg['maximum_scientific_seconds']==3600 and cfg['maximum_total_work_seconds']==14400


def execution_paths():
    return [*historical_paths(),*(ROOT/'experiments'/name for name in
        ('run_acs_preservation.py','acs_preservation_diagnostics.py','acs_preservation_audits.py','run_acs_preservation_extended.py'))]


def frozen_hashes(out):
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha_file(p)
            for p in [out/'config.json',out/'PROTOCOL.md',*execution_paths()]}


def prepare(out):
    cfg=read(out/'config.json');check_config(cfg)
    files=[]
    for key in ('parent_results','reference_results','bottleneck_reference_results','pca16_reference_results','init_reference_results'):
        parent=ROOT/cfg[key]
        files += [parent/k for k in ('config.json','PROTOCOL.md','protocol_freeze.json')]
        for seed in cfg['seeds']:
            files += [parent/f'seed_{seed}'/k for k in ('metrics.json','selection_before_test.json','local_artifacts.json')]
        if (parent/'SCORE_REPLAY.json').exists(): files.append(parent/'SCORE_REPLAY.json')
    if (out/'protocol_freeze.json').exists(): raise FileExistsError('Preserve protocol freeze')
    write_json(out/'protocol_freeze.json',{'created_utc':now(),'starting_commit':cfg['starting_commit'],
        'sha256':frozen_hashes(out),'reference_record_hashes':{str(p.relative_to(ROOT)):sha_file(p) for p in files},
        'evaluation_status':cfg['evaluation_status']})
    previous=read(ROOT/cfg['reference_results']/'environment.json')
    write_json(out/'environment.json',{'created_utc':now(),'platform':platform.platform(),
        'hardware':previous['hardware'],'threads':1,'python':platform.python_version(),
        'packages':{k:importlib.metadata.version(k) for k in previous['packages']}})


def execution_amendments(out):
    return {p.name:sha_file(p) for p in sorted(out.glob('EXECUTION_AMENDMENT_*.json'))}


def effective_execution_hashes(out):
    expected=read(out/'protocol_freeze.json')['sha256'].copy()
    for path in sorted(out.glob('EXECUTION_AMENDMENT_*.json')):
        amendment=read(path)
        assert amendment['original_protocol_freeze_sha256']==sha_file(out/'protocol_freeze.json')
        for name,change in amendment['source_changes'].items():
            assert name.startswith('experiments/') and expected[name]==change['before_sha256']
            expected[name]=change['after_sha256']
    return expected


def verify_freeze(out):
    check_config(read(out/'config.json'));freeze=read(out/'protocol_freeze.json')
    assert frozen_hashes(out)==effective_execution_hashes(out), 'Execution/protocol changed outside recorded freeze/amendments'
    for p,h in freeze['reference_record_hashes'].items(): assert sha_file(ROOT/p)==h, p


def preservation_history(cfg,seed,pair,releases):
    directory=ROOT/cfg['init_reference_results']/f'seed_{seed}'
    manifest={v['path']:v['sha256'] for v in read(directory/'local_artifacts.json')}
    used={}
    def checked(path):
        key=str(path.relative_to(ROOT));h=sha_file(path);assert manifest[key]==h,key
        used[key]=h;return path
    old=read(directory/'training/training.json');new=pair['metadata']
    initial=torch.load(checked(directory/'training/initialization.pt'),weights_only=True)['model_state']
    current=pair['snapshots']['I'].state_dict()
    assert set(initial)==set(current) and all(torch.equal(initial[k],current[k]) for k in initial)
    for key in ('config','schedules','adversary_initialization_hash','preprocessing','fit_input_sha256',
                'source_validation_input_sha256','source_label_hashes','attribute_label_hashes',
                'source_validation_label_hashes','common_optimizer_counts'):
        assert old[key]==new[key], 'Historical identity failed: '+key
    for arm in ARMS:
        oldarm=old['arms']['C_bottleneck' if arm.startswith('C_') else 'D_protected']
        for key in ('schedule_hash','continuation_mapper_optimizer_steps','continuation_adversary_optimizer_steps',
                    'optimizer_counts_including_common','mapper_row_exposures','adversary_row_exposures'):
            assert oldarm[key]==new['arms'][arm][key], key
    with np.load(checked(directory/'release_I.npz')) as z:
        for pool,values in releases['I'].items(): assert np.array_equal(values,z[pool]),pool
    checked(directory/'predictions.npz')
    selected=read(directory/'selection_before_test.json');previous=read(directory/'metrics.json')
    names=('I','W','C_init','D_init','C_bottleneck','D_protected','PCA16')
    rename=lambda n:'W_historical' if n=='W' else n
    raw=[{**copy.deepcopy(r),'release':rename(r['release']),'reused_reference':True}
         for r in previous['raw_metrics'] if r['release'] in names]
    records={}
    for key,rec in selected['fitting_records'].items():
        role,name,target=key.split('/')
        if name not in names: continue
        records[f'{role}/{rename(name)}/{target}']={field:selected[field].get(key) for field in
            ('head_selections','family_selections','auroc_selections','independent_selections')}
        records[f'{role}/{rename(name)}/{target}']['record']=rec
    return {'raw':raw,'records':records,'used_files':used,'identities':{
        'initial_full_state_bitwise_equal':True,'I_release_bitwise_equal':True,
        'historical_training_recipe_and_schedules_exact':True,
        'indices':{k:selected[k] for k in ('task_fit_indices','audit_fit_indices')},
        'historical_metrics_sha256':sha_file(directory/'metrics.json'),
        'historical_selection_sha256':sha_file(directory/'selection_before_test.json')}}


def run_unit(out,cfg,beta,seed,*,miniature=False):
    if not miniature: verify_freeze(out)
    tick=time.perf_counter();directory=out/beta_name(beta)/f'seed_{seed}';directory.mkdir(parents=True,exist_ok=False)
    pcfg=read(ROOT/cfg['parent_results']/'config.json')
    support_record=read(ROOT/cfg['parent_results']/'schema_support.json')
    assert sha_file(ROOT/pcfg['raw_path'])==support_record['raw_sha256']
    frame,cohort=load_cohort(ROOT/pcfg['raw_path'],pcfg['sample_cap'],pcfg['sample_seed'])
    pools=split_households(frame,seed);refs=SavedReferences(cfg,seed,frame,pools)
    edges=refs.source.selection['income_edges'];pca=refs.releases['E_pca']
    original_support=refs.previous['support_by_pool']
    for p,i in pools.items(): assert original_support[p]['raw_row_sha256']==array_hash(frame.iloc[i]._raw_row.to_numpy())
    write_json(directory/'support.json',original_support)
    np.savez_compressed(directory/'split_rows.npz',**{p:frame.iloc[i]._raw_row.to_numpy() for p,i in pools.items()})
    reuse_seconds=time.perf_counter()-tick;t=time.perf_counter()
    historical_training=read(ROOT/cfg['init_reference_results']/f'seed_{seed}'/'training/training.json')
    pair=train_preservation_unit(pca['representation_fit'],source_binaries(frame.iloc[pools['representation_fit']],edges),
        audit_labels(frame.iloc[pools['representation_fit']]),pca['source_validation'],
        source_binaries(frame.iloc[pools['source_validation']],edges),seed,directory/'training',miniature=miniature,
        beta=beta,fitting_statistics=historical_training['preprocessing'])
    training_seconds=time.perf_counter()-t
    final_arms=pair['arms']
    models={**pair.get('snapshots',{}),**{n:arm['model'] for n,arm in final_arms.items()}}
    releases={n:{p:model.release(x) for p,x in pca.items()} for n,model in models.items()}
    history=preservation_history(cfg,seed,pair,releases)
    training_completed_utc=now()
    for arrays in releases.values():
        for a in arrays.values(): a.setflags(write=False)
    def learned_state():
        return {n:{'model':_state_hash(model),
                   'adversaries':_state_hash(final_arms[n]['adversaries']) if n in final_arms else None}
                for n,model in models.items()}
    before=learned_state();outputs=frozen_digest(releases)
    release_meta={n:copy.deepcopy(refs.previous['release_metadata'][n]) for n in REFERENCES}
    for n,a in releases.items():
        release_meta[n]={'parent':'E_pca','erased':False,'dimension':16,'dtype':'float32',
            'bytes_per_record':64,'training_arm':n,**feature_rank(a['representation_fit'])}
    write_json(directory/'release_freeze.json',{'created_utc':now(),'release_metadata':release_meta,
        'learned_state':before,'output_hashes':outputs,'reference_state':refs.initial,
        'fit_pool':'representation_fit','fit_raw_row_sha256':original_support['representation_fit']['raw_row_sha256'],
        'reserved_labels_entered_release_fitting':False,'final_epoch_selection':'fixed80; I/W diagnostics never selectable',
        'scientific_training_completed_utc':training_completed_utc,
        'initialization_history':history['identities'] if history else None})
    affine,diagnostics=fit_snapshots(releases,pca,pair['metadata']['preprocessing'],directory)
    write_json(directory/'affine_freeze.json',{'created_utc':now(),'fits':diagnostics,'sha256':{p.name:sha_file(p) for p in directory.glob('affine_*.npz')}})
    # First downstream construction/access to reserved labels after immutable maps.
    fit_y=binary_tasks(frame.iloc[pools['downstream_fit']],edges)
    val_y=binary_tasks(frame.iloc[pools['downstream_validation']],edges)
    audit_y=audit_labels(frame.iloc[pools['attacker_fit']]);audit_v=audit_labels(frame.iloc[pools['attacker_validation']])
    ti={t:subset_indices(fit_y[t],cfg['head_budget'],1230000+100*seed+j) for j,t in enumerate(TASKS)}
    ai={t:subset_indices(audit_y[t],cfg['attacker_budget'],1240000+100*seed+j) for j,t in enumerate(AUDITS)}
    def index_record(indices,pool):
        return {t:{'n':len(i),'pool_indices_sha256':array_hash(i),
            'raw_rows_sha256':array_hash(frame.iloc[pools[pool][i]]._raw_row.to_numpy())} for t,i in indices.items()}
    index_records={'task_fit_indices':index_record(ti,'downstream_fit'),'audit_fit_indices':index_record(ai,'attacker_fit')}
    t=time.perf_counter();refs.verify_reuse(fit_y,val_y,audit_y,audit_v,ti,ai,index_records)
    if history:
        for identity in history['identities'].values():
            if isinstance(identity,dict) and 'task_fit_indices' in identity:
                assert identity['task_fit_indices']==index_records['task_fit_indices']
                assert identity['audit_fit_indices']==index_records['audit_fit_indices']
    reference_prediction_verification_seconds=time.perf_counter()-t
    fitted={};selections={};records={};family_selections={};auroc_selections={};independent_selections={}
    def accept(key,result):
        candidates=result['candidates']
        scores={k:c.metadata['validation_scores'] for k,c in candidates.items() if k!='saved_adversary'}
        chosen=select_candidates(scores);selections[key]=chosen['selected_family'];auroc_selections[key]=chosen['selected_auroc']
        independent={k:v for k,v in scores.items() if k!='catchup'}
        independent_choice=select_candidates(independent)
        selection_split='downstream_validation' if key.startswith('transfer/') else 'attacker_validation'
        for choice,eligible in ((chosen,scores),(independent_choice,independent)):
            choice['selection_split']=selection_split
            choice['auroc_scope']='diagnostic selection among '+', '.join(eligible)+'; MLP checkpoints selected by validation log loss; saved adversary excluded'
        independent_selections[key]=independent_choice['selected_family']
        fam=lambda k: k if k in ('catchup','saved_adversary') else candidates[k].metadata['family']
        family_selections[key]={f:min((k for k in candidates if fam(k)==f),
            key=lambda k:(candidates[k].metadata['validation_scores']['log_loss'],k)) for f in {fam(k) for k in candidates}}
        result.update(chosen)
        result['metadata'].update(chosen)
        result['metadata'].update({'validation_scores':{k:c.metadata['validation_scores'] for k,c in candidates.items()},
            'candidate_ids':list(candidates),'independent_candidate_ids':list(independent),
            'primary_selection_candidates':list(scores),'independent_selection':independent_choice,
            'saved_adversary_diagnostic_only':True,'family_selections':family_selections[key]})
        if 'catchup' in candidates:
            for metric in ('optimizer_steps','training_row_exposures'):
                old_key='total_mlp_'+metric
                result['metadata']['independent_'+old_key]=result['metadata'][old_key]
                result['metadata']['total_new_mlp_'+metric]=result['metadata'][old_key]+candidates['catchup'].metadata[metric]
            result['metadata']['total_mlp_budget_scope']='independent fresh MLP restarts only; total_new_mlp fields additionally include catchup, excluding inherited representation-training exposure'
        save_candidates(result,directory/'fitted'/key)
        fitted[key]=candidates;records[key]=result['metadata']
    task_seconds=0.;audit_seconds=0.;catchup_seconds=0.
    audit_budget={'mlp':{'epochs':1},'histgb':{'max_iter':1}} if miniature else None
    for release,arrays in releases.items():
        if release=='I': continue  # Exact historical I state and predictions verified above.
        print(f'seed {seed}: {release}: final utility and independent audits',flush=True)
        t=time.perf_counter()
        for j,target in enumerate(TASKS):
            i=ti[target];vi=np.flatnonzero(val_y[target]>=0)
            accept(f'transfer/{release}/{target}',fit_candidates(arrays['downstream_fit'][i],fit_y[target][i],
                arrays['downstream_validation'][vi],val_y[target][vi],2,1250000+100*seed+j,
                budget={'mlp':{'epochs':1 if miniature else cfg['head_mlp_epochs']}}))
        task_seconds+=time.perf_counter()-t
        if release not in final_arms:
            continue  # I/W utility-only snapshots have no new attribute audit.
        training_arm=release
        for j,(target,k) in enumerate(AUDITS.items()):
            i=ai[target];vi=np.flatnonzero(audit_v[target]>=0);t=time.perf_counter()
            result=fit_primary_auditors(arrays['attacker_fit'][i],audit_y[target][i],
                arrays['attacker_validation'][vi],audit_v[target][vi],k,1260000+100*seed+j,budget=audit_budget)
            audit_seconds+=time.perf_counter()-t;t=time.perf_counter()
            train_meta=pair['metadata'];arm_meta=train_meta['arms'][training_arm]
            exposure_passes=(train_meta['config']['warm_adversary_epochs']+
                train_meta['config']['continuation_epochs']*train_meta['config']['adversary_updates_per_mapper_step'])
            inherited={'target':target,'fit_pool':train_meta['fit_pool'],
                'fit_rows':train_meta['preprocessing']['fit_rows'],
                'fit_input_sha256':train_meta['fit_input_sha256'],
                'fit_raw_row_sha256':original_support['representation_fit']['raw_row_sha256'],
                'training_label_sha256':train_meta['attribute_label_hashes'][target],
                'fit_coverage':train_meta['attribute_fit_coverage'][target],
                'common_warm_adversary_optimizer_steps':train_meta['common_optimizer_counts']['adversary_optimizer_steps'],
                'common_warm_adversary_row_exposures':train_meta['common_adversary_row_exposures'],
                'continuation_adversary_optimizer_steps':arm_meta['continuation_adversary_optimizer_steps'],
                'continuation_adversary_row_exposures':arm_meta['adversary_row_exposures'],
                'total_optimizer_steps':arm_meta['optimizer_counts_including_common']['adversary_optimizer_steps'],
                'total_row_exposures':train_meta['common_adversary_row_exposures']+arm_meta['adversary_row_exposures'],
                'total_row_passes_equivalent':exposure_passes,
                'total_known_label_exposures':train_meta['attribute_fit_coverage'][target]['known']*exposure_passes,
                'final_adversary_state_hash':_state_hash(final_arms[release]['adversaries'][target]),
                'continuation_schedule_hash':arm_meta['schedule_hash']}
            caught=fit_catchup(final_arms[release]['adversaries'][target],arrays['attacker_fit'][i],audit_y[target][i],
                arrays['attacker_validation'][vi],audit_v[target][vi],k,1300000+100*seed+j,
                epochs=1 if miniature else cfg['catchup_epochs'],
                inherited_exposure=inherited)
            result['candidates'].update({'catchup':caught['catchup'],'saved_adversary':caught['saved']})
            result['metadata']['candidates'].update({cid:c.metadata for cid,c in result['candidates'].items()})
            result['metadata']['catchup_diagnostic']=caught['metadata']
            accept(f'audit/{release}/{target}',result)
            catchup_seconds+=time.perf_counter()-t
    # Import verified historical reference records without refits or new choices.
    for key,record in refs.selection['fitting_records'].items():
        if key.split('/')[1] not in (*REFERENCES,'prior','exposed'): continue
        records[key]=record;selections[key]=refs.selection['head_selections'][key]
        independent_selections[key]=selections[key];family_selections[key]=refs.selection['family_selections'][key]
        auroc_selections[key]=refs.selection['auroc_selections'][key]
    if history:
        for key,entry in history['records'].items():
            records[key]=entry['record'];selections[key]=entry['head_selections']
            independent_selections[key]=entry['independent_selections']
            family_selections[key]=entry['family_selections'];auroc_selections[key]=entry['auroc_selections']
    provenance={'created_utc':now(),'used_reference_files_sha256':refs.used_files,
        'used_original_files_sha256':refs.source.used_files,'original_source_selection':refs.source.selection,
        'reference_metrics_path':str((refs.directory/'metrics.json').relative_to(ROOT)),
        'reference_metrics_sha256':sha_file(refs.directory/'metrics.json'),
        'reference_prediction_reuse':'Verified fitting rows/labels, recipe, exact standardizers, fitted-state and validation probability replay; original predictions/metrics retained.',
        'verified_reference_candidates':refs.checked_candidates,
        'additional_history':history['identities'] if history else None,
        'additional_history_files_sha256':history['used_files'] if history else {},'regeneration':'None; frozen inference only; no PCA, bank, encoder or eraser refit'}
    write_json(directory/'parent_provenance.json',provenance)
    record={'created_utc':now(),'evaluation_status':cfg['evaluation_status'],'head_selections':selections,
        'independent_selections':independent_selections,'family_selections':family_selections,
        'auroc_selections':auroc_selections,'fitting_records':records,**index_records,
        'release_freeze_sha256':sha_file(directory/'release_freeze.json'),
        'protocol_freeze_sha256':sha_file(out/'protocol_freeze.json') if not miniature else 'miniature_only'}
    write_json(directory/'selection_before_test.json',record)
    selection_hash=require_test_selection(directory,list(selections))
    print(f'seed {seed}: selections saved; opening DEVELOPMENT EVALUATION',flush=True)
    eval_started=now();t=time.perf_counter();test_frame=frame.iloc[pools['test']]
    ref_test=refs.evaluation_releases(test_frame)
    test={n:model.release(ref_test['E_pca']) for n,model in models.items()}
    if history:
        with np.load(ROOT/cfg['pca16_reference_results']/f'seed_{seed}'/'release_PCA16.npz') as z:
            expected=z['test'];assert np.allclose(test['I'],expected,atol=1e-5,rtol=1e-5)
            e=test['I'].astype(np.float64)-expected.astype(np.float64)
            write_json(directory/'evaluation_parity.json',{'atol':1e-5,'rtol':1e-5,
                'max_abs':float(np.abs(e).max()),'rms':float(np.sqrt(np.mean(e**2))),
                'initial_sha256':array_hash(test['I']),'historical_PCA16_sha256':array_hash(expected),
                'selected_before_access':selection_hash})
    evaluate_snapshots(affine,diagnostics,test,ref_test['E_pca'],pair['metadata']['preprocessing'])
    write_json(directory/'preservation_diagnostics.json',diagnostics)
    predictions={}
    val=score_and_save(fitted,selections,
        {r:{n:a[p] for n,a in releases.items()} for r,p in [('transfer','downstream_validation'),('audit','attacker_validation')]},
        {'transfer':val_y,'audit':audit_v},
        {r:frame.iloc[pools[p]].PWGTP.to_numpy(float) for r,p in [('transfer','downstream_validation'),('audit','attacker_validation')]},
        'validation',predictions,cfg)
    evaluated=score_and_save(fitted,selections,{'transfer':test,'audit':test},
        {'transfer':binary_tasks(test_frame,edges),'audit':audit_labels(test_frame)},
        {r:test_frame.PWGTP.to_numpy(float) for r in ('transfer','audit')},'test',predictions,cfg)
    raw=[]
    for key,candidates in fitted.items():
        role,release,target=key.split('/')
        for cid,c in candidates.items():
            family=cid if cid in ('catchup','saved_adversary') else c.metadata['family']
            raw.append({'seed':seed,'role':role,'release':release,'parent':'E_pca','erased':False,'target':target,
                'candidate_id':cid,'family':family,'selected':selections[key]==cid,
                'independent_selected':independent_selections[key]==cid,
                'selected_within_family':family_selections[key][family]==cid,'auc_selected':auroc_selections[key]==cid,
                'validation':val[key,cid]['score'],'validation_person_weighted':val[key,cid]['weighted'],
                'test':evaluated[key,cid]['score'],'test_person_weighted':evaluated[key,cid]['weighted'],
                'reused_reference':False})
    for old in refs.previous['raw_metrics']:
        if old['release'] in (*REFERENCES,'prior','exposed'):
            raw.append({**copy.deepcopy(old),'independent_selected':old['selected'],'reused_reference':True})
    if history: raw.extend(history['raw'])
    np.savez_compressed(directory/'predictions.npz',**predictions)
    for n,a in releases.items(): np.savez_compressed(directory/f'release_{n}.npz',**a,test=test[n])
    integrity={'learned_states_unchanged':before==learned_state(),
        'new_releases_unchanged':outputs==frozen_digest(releases),
        'reference_states_unchanged':refs.initial==refs.fingerprint(),
        'reference_files_unchanged':all(sha_file(ROOT/p)==h for p,h in {**refs.used_files,**refs.source.used_files}.items()),
        'history_files_unchanged':not history or all(sha_file(ROOT/p)==h for p,h in history['used_files'].items()),
        'selection_unchanged':sha_file(directory/'selection_before_test.json')==selection_hash,
        'selection_created_utc':record['created_utc'],'evaluation_started_utc':eval_started,
        'selection_sha256':selection_hash,'evaluation_output_hashes':{n:array_hash(x) for n,x in test.items()}}
    assert all(v for k,v in integrity.items() if k.endswith('_unchanged')), 'Final frozen-state integrity failed'
    local=[{'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'bytes':p.stat().st_size,
            'sha256':sha_file(p),'availability':'local only; regenerate in new directory with fixed runner'}
           for p in directory.rglob('*') if p.is_file() and p.suffix in ('.pt','.npz','.joblib')]
    write_json(directory/'local_artifacts.json',local)
    runtime={'total_seconds':time.perf_counter()-tick,'reference_loading_seconds':reuse_seconds,
        'representation_training_seconds':training_seconds,'reference_prediction_verification_seconds':reference_prediction_verification_seconds,
        'utility_fitting_seconds':task_seconds,'independent_audit_seconds':audit_seconds,'catchup_seconds':catchup_seconds,
        'evaluation_serialization_integrity_seconds':time.perf_counter()-t}
    result={'seed':seed,'beta':beta,'evaluation_status':cfg['evaluation_status'],'cohort':cohort,'raw_metrics':raw,
        'release_metadata':release_meta,'support_by_pool':original_support,'integrity':integrity,'runtime':runtime}
    write_json(directory/'metrics.json',result)
    print(json.dumps({'seed':seed,'runtime':runtime,'integrity':integrity}),flush=True)
    return result



def main():
    import datetime
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    p.add_argument('--prepare',action='store_true');p.add_argument('--core',action='store_true')
    args=p.parse_args();out=args.out.resolve();cfg=read(out/'config.json');torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if args.prepare: prepare(out)
        if args.core:
            progress_path=out/'progress.json'
            progress=read(progress_path) if progress_path.exists() else {'completed_units':[],'scientific_seconds':0.,'status':'running'}
            order=[(b,s) for b in cfg['preservation_betas'] for s in cfg['seeds']]
            for index,(beta,seed) in enumerate(order):
                path=out/beta_name(beta)/f'seed_{seed}'/'metrics.json'
                if path.exists():
                    if [beta,seed] not in progress['completed_units']: raise RuntimeError('Completed evidence missing progress accounting')
                    continue
                remaining=cfg['maximum_scientific_seconds']-progress['scientific_seconds']
                elapsed=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(cfg['started_utc'])).total_seconds()
                estimate=max(progress.get('unit_seconds',[60.]))
                if remaining < estimate*1.25 or elapsed+estimate*1.25>cfg['maximum_total_work_seconds']-900:
                    progress.update(status='partial: insufficient time for another complete unit');break
                tick=time.perf_counter()
                try:
                    result=run_unit(out,cfg,beta,seed)
                except Exception as error:
                    progress['scientific_seconds']+=time.perf_counter()-tick
                    progress.update(status='incomplete unit retained after failure',failed_unit=[beta,seed],
                        failure=type(error).__name__+': '+str(error),updated_utc=now())
                    write_json(progress_path,progress)
                    raise
                seconds=time.perf_counter()-tick
                progress['scientific_seconds']+=seconds;progress['completed_units'].append([beta,seed])
                progress.setdefault('unit_seconds',[]).append(seconds)
                progress.update(updated_utc=now(),projected_core_total_seconds=progress['scientific_seconds']+(5-index)*max(progress['unit_seconds']))
                write_json(progress_path,progress)
                print(json.dumps(progress),flush=True)
            else: progress['status']='core complete'
            write_json(progress_path,progress)


if __name__=='__main__': main()
