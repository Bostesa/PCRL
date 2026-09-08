"""Fixed selective-teacher study, sequential immutable units and nested audits."""
from __future__ import annotations
import argparse
import copy
import datetime
import json
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from experiments.acs_transfer_data import array_hash, audit_labels, load_cohort, sha_file, split_households, write_json
from experiments.acs_transfer_heads import fit_candidates, _state_hash
from experiments.acs_protection_audits import AUDIT_BUDGET
from experiments.acs_preservation_audits import fit_extended_auditors, fit_extended_catchup, save_extended_audits
from experiments.run_acs_preservation_extended import _selection
from experiments.run_acs_bottleneck import SavedReferences, source_binaries
from experiments.run_acs_preservation import execution_paths as inherited_paths
from experiments.run_acs_protection import ROOT, TASKS, AUDITS, binary_tasks, frozen_digest, feature_rank, now, read, score_and_save
from experiments.run_acs_transfer import subset_indices, save_candidates

DEFAULT_OUT = ROOT/'results/redesign_20260908_acs_selective_preservation_v1'


def unit_name(teacher, rho):
    assert teacher in ('R', 'E', 'S') and rho in (0., .1)
    return teacher+'_rho'+('0p1' if rho == .1 else '0')


def unit_path(out, unit):
    return out/('static' if unit[0] == 'static' else unit_name(unit[0], unit[1]))/f'seed_{unit[-1]}'


def check_config(cfg):
    old=read(ROOT/cfg['preservation_reference_results']/'config.json')
    for key in ('training','references','utility_tasks','seeds','threads','head_budget','attacker_budget',
                'head_mlp_epochs','audit_mlp_epochs','catchup_epochs','audit_restart_offset',
                'audit_min_samples_leaf','margins','mapper_initialization','parity_tolerance','extended_audit_epochs'):
        assert cfg[key] == old[key], 'Inherited fixed configuration changed: '+key
    assert cfg['study']=='acs_selective_preservation_v1'
    assert cfg['teachers']==['R','E','S'] and cfg['reconstruction_coefficients']==[.1,0.]
    assert cfg['preservation_betas']==[1.] and cfg['preservation_schedules']==['persistent']
    assert cfg['continuation_arms']==cfg['learned_arms']==['C','D']
    assert cfg['snapshot_releases']==['I','W','C','D']
    assert cfg['permutation_seed_base']==20260908 and cfg['affine_rank_rcond']==cfg['ratio_variance_floor']==1e-12
    expected=[['static',s] for s in range(3)]+[[t,r,s] for s in range(3) for t,r in [('E',.1),('R',0.)]]+[['E',0.,s] for s in range(3)]+[[t,r,s] for s in range(3) for t,r in [('S',.1),('S',0.)]]
    assert cfg['execution_order']==expected
    assert cfg['maximum_scientific_seconds']==3600 and cfg['maximum_total_work_seconds']==14400 and cfg['reserved_reporting_seconds']==1800


def execution_paths():
    return list(dict.fromkeys([*inherited_paths(),*(ROOT/'experiments'/p for p in
        ('run_acs_selective.py','acs_selective_teachers.py','acs_selective_diagnostics.py'))]))


def frozen_hashes(out):
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha_file(p) for p in [out/'config.json',out/'PROTOCOL.md',*execution_paths()]}


def prepare(out):
    cfg=read(out/'config.json');check_config(cfg)
    assert not (out/'protocol_freeze.json').exists(), 'Never overwrite a frozen study'
    files=[]
    for key in ('parent_results','reference_results','bottleneck_reference_results','pca16_reference_results','init_reference_results','preservation_reference_results'):
        parent=ROOT/cfg[key]
        files.extend(parent/p for p in ('config.json','PROTOCOL.md','protocol_freeze.json'))
        files.extend(p for p in parent.rglob('*.json') if p.name in ('metrics.json','selection_before_test.json','local_artifacts.json','release_freeze.json') and 'fitted' not in p.parts and 'operational_attempts' not in p.parts)
    write_json(out/'protocol_freeze.json',{'created_utc':now(),'starting_commit':cfg['starting_commit'],
        'sha256':frozen_hashes(out),'reference_record_hashes':{str(p.relative_to(ROOT)):sha_file(p) for p in files},
        'historical_source_hashes':'retained in original freezes, not replaced with new source identities'})
    environment=read(ROOT/cfg['preservation_reference_results']/'environment.json')
    write_json(out/'environment.json',{**environment,'created_utc':now(),'reused_existing_environment':True})


def verify_freeze(out):
    cfg=read(out/'config.json');check_config(cfg);freeze=read(out/'protocol_freeze.json')
    assert freeze['sha256']==frozen_hashes(out), 'Frozen execution changed'
    for p,h in freeze['reference_record_hashes'].items(): assert sha_file(ROOT/p)==h,p


def load_context(cfg, seed):
    pcfg=read(ROOT/cfg['parent_results']/'config.json')
    support=read(ROOT/cfg['parent_results']/'schema_support.json')
    assert sha_file(ROOT/pcfg['raw_path'])==support['raw_sha256']
    frame,cohort=load_cohort(ROOT/pcfg['raw_path'],pcfg['sample_cap'],pcfg['sample_seed'])
    pools=split_households(frame,seed);refs=SavedReferences(cfg,seed,frame,pools)
    pre=read(ROOT/cfg['init_reference_results']/f'seed_{seed}'/'training/training.json')['preprocessing']
    return frame,cohort,pools,refs,pre


def historical_identity(cfg,seed,pair,releases):
    """Exact reused I tensors/coordinates; preserve all historical source identities."""
    parent=ROOT/cfg['init_reference_results']/f'seed_{seed}'
    manifest={v['path']:v['sha256'] for v in read(parent/'local_artifacts.json')}
    used={}
    def checked(path):
        name=str(path.relative_to(ROOT));assert sha_file(path)==manifest[name],name
        used[name]=manifest[name];return path
    initial=torch.load(checked(parent/'training/initialization.pt'),weights_only=True)
    old=initial['model_state'];current=pair['snapshots']['I'].state_dict()
    assert set(old)==set(current) and all(torch.equal(old[k],current[k]) for k in old)
    with np.load(checked(parent/'release_I.npz')) as z:
        for pool,x in releases['I'].items(): assert np.array_equal(x,z[pool]),pool
    old_meta=read(parent/'training/training.json');new=pair['metadata']
    for key in ('preprocessing','schedules','adversary_initialization_hash','source_label_hashes','attribute_label_hashes','source_validation_label_hashes'):
        assert old_meta[key]==new[key],key
    checked(parent/'predictions.npz')
    return {'initial_full_tensors_equal':True,'I_release_bitwise_equal':True,'original_schedules_labels_preprocessing_equal':True,'files_sha256':used}


def verify_historical_artifacts(cfg,seed,frame,pools):
    """Read-only verification once per seed; no historical fitting or reselection."""
    roots=[ROOT/cfg[k]/f'seed_{seed}' for k in ('init_reference_results','pca16_reference_results','reference_results')]
    old=ROOT/cfg['preservation_reference_results']
    roots.extend([old/'beta_1'/f'seed_{seed}',old/'extended'/f'seed_{seed}'])
    used={}
    for parent in roots:
        with np.load(parent/'split_rows.npz') as rows:
            for pool,ix in pools.items(): assert np.array_equal(rows[pool],frame.iloc[ix]._raw_row.to_numpy())
        for entry in read(parent/'local_artifacts.json'):
            path=ROOT/entry['path'];assert sha_file(path)==entry['sha256'],str(path)
            used[entry['path']]=entry['sha256']
    training=read(old/'beta_1'/f'seed_{seed}'/'training/training.json')
    assert training['preservation_config']['beta']==1.
    for arm in ('C_persistent','D_persistent'):
        assert training['arms'][arm]['selected_epoch']==80
    return {'files_sha256':used,'historical_raw_rho0p1_beta1_persistent_reused':True,
        'historical_release_predictions_checkpoints_hashed':True,'household_rows_exact':True,
        'regenerated_artifacts':[],'recipe':'original frozen protocols and nested360 evidence retained'}


def fit_static_auditors(xf,yf,xv,yv,k,seed):
    candidates={}
    for cid,family,leaf in (('logistic','logistic',None),('hist_gb_20','histgb',20),('hist_gb_5','histgb',5)):
        budget=copy.deepcopy(AUDIT_BUDGET)
        if leaf is not None: budget['histgb']['min_samples_leaf']=leaf
        candidate=fit_candidates(xf,yf,xv,yv,k,seed,families=(family,),budget=budget)['candidates'][family]
        candidate.metadata.update(candidate_id=cid,base_seed=seed,restart_index=None,restart_seed_offset=None,checkpoint_criterion='fixed_configuration')
        candidates[cid]=candidate
    return candidates


def empty_record():
    return {k:{} for k in ('head_selections','independent_selections','family_selections','auroc_selections','fitting_records')}


def accept(candidates,key,record):
    primary,independent,families,meta=_selection(candidates)
    if key.startswith('transfer/'):
        meta['selection_split']='downstream_validation'
    record['head_selections'][key]=primary['selected_family']
    record['independent_selections'][key]=independent['selected_family']
    record['family_selections'][key]=families
    record['auroc_selections'][key]=primary['selected_auroc']
    record['fitting_records'][key]=meta


def rows_from_scores(fitted,record,val,evaluated,seed,budget=None):
    raw=[]
    for key,candidates in fitted.items():
        role,release,target=key.split('/')
        for cid,candidate in candidates.items():
            family=cid if cid in ('catchup','saved_adversary') else candidate.metadata['family']
            row={'seed':seed,'role':role,'release':release,'parent':'E_pca','erased':False,'target':target,
                'candidate_id':cid,'family':family,'selected':record['head_selections'][key]==cid,
                'independent_selected':record['independent_selections'][key]==cid,
                'selected_within_family':record['family_selections'][key][family]==cid,
                'auc_selected':record['auroc_selections'][key]==cid,
                'validation':val[key,cid]['score'],'validation_person_weighted':val[key,cid]['weighted'],
                'test':evaluated[key,cid]['score'],'test_person_weighted':evaluated[key,cid]['weighted'],
                'reused_reference':False}
            if budget is not None: row['audit_budget']=budget
            raw.append(row)
    return raw


def inherited_exposure(pair,release,target,support):
    meta=pair['metadata'];arm=meta['arms'][release]
    passes=meta['config']['warm_adversary_epochs']+meta['config']['continuation_epochs']*meta['config']['adversary_updates_per_mapper_step']
    return {'target':target,'fit_pool':meta['fit_pool'],'fit_rows':meta['preprocessing']['fit_rows'],
        'fit_input_sha256':meta['fit_input_sha256'],'fit_raw_row_sha256':support['representation_fit']['raw_row_sha256'],
        'training_label_sha256':meta['attribute_label_hashes'][target],'fit_coverage':meta['attribute_fit_coverage'][target],
        'common_warm_adversary_optimizer_steps':meta['common_optimizer_counts']['adversary_optimizer_steps'],
        'continuation_adversary_optimizer_steps':arm['continuation_adversary_optimizer_steps'],
        'total_optimizer_steps':arm['optimizer_counts_including_common']['adversary_optimizer_steps'],
        'total_row_exposures':meta['common_adversary_row_exposures']+arm['adversary_row_exposures'],
        'total_row_passes_equivalent':passes,'total_known_label_exposures':meta['attribute_fit_coverage'][target]['known']*passes,
        'final_adversary_state_hash':_state_hash(pair['arms'][release]['adversaries'][target]),
        'continuation_schedule_hash':arm['schedule_hash']}


def run_unit(out,cfg,unit):
    from experiments.acs_bottleneck_training import train_selective_unit
    from experiments.acs_selective_teachers import build_teachers, load_teachers
    from experiments.acs_selective_diagnostics import fit_snapshots, evaluate_snapshots
    verify_freeze(out);started=time.perf_counter();seed=unit[-1];is_static=unit[0]=='static'
    directory=unit_path(out,unit);directory.mkdir(parents=True,exist_ok=False)
    frame,cohort,pools,refs,pre=load_context(cfg,seed)
    pca=refs.releases['E_pca'];edges=refs.source.selection['income_edges'];support=refs.previous['support_by_pool']
    write_json(directory/'support.json',support)
    np.savez_compressed(directory/'split_rows.npz',**{p:frame.iloc[ix]._raw_row.to_numpy() for p,ix in pools.items()})
    phase={};tick=time.perf_counter();pair=None;models={}
    if is_static:
        teachers=build_teachers(pca['representation_fit'][:,:16],audit_labels(frame.iloc[pools['representation_fit']]),seed,directory/'teachers',
            original_mean=pre['mean'][:16],original_scale=pre['scale'][:16],fit_raw_rows=frame.iloc[pools['representation_fit']]._raw_row.to_numpy())
        phase['teacher_fitting_seconds']=time.perf_counter()-tick
        write_json(directory/'historical_reuse.json',verify_historical_artifacts(cfg,seed,frame,pools))
        releases={name+'_direct':{p:teachers['maps'][name].apply(x[:,:16]) for p,x in pca.items()} for name in ('E','S')}
        teacher_by_release={'E_direct':'E','S_direct':'S'}
        initial_identity=None;final_arms={}
    else:
        teacher,rho,_=unit
        teachers=load_teachers(out/'static'/f'seed_{seed}'/'teachers')
        target=pca['representation_fit'][:,:16] if teacher=='R' else teachers['maps'][teacher].apply(pca['representation_fit'][:,:16])
        pair=train_selective_unit(pca['representation_fit'],source_binaries(frame.iloc[pools['representation_fit']],edges),
            audit_labels(frame.iloc[pools['representation_fit']]),pca['source_validation'],
            source_binaries(frame.iloc[pools['source_validation']],edges),seed,directory/'training',
            teacher_targets=target,teacher_name=teacher,reconstruction_weight=rho,fitting_statistics=pre)
        phase['representation_training_seconds']=time.perf_counter()-tick
        models={**pair['snapshots'],**{n:v['model'] for n,v in pair['arms'].items()}}
        releases={n:{p:model.release(x) for p,x in pca.items()} for n,model in models.items()}
        initial_identity=historical_identity(cfg,seed,pair,releases);final_arms=pair['arms']
        teacher_by_release={n:teacher for n in releases}
    for arrays in releases.values():
        for x in arrays.values(): x.setflags(write=False)
    def state():
        return {'models':{n:_state_hash(m) for n,m in models.items()},'adversaries':{n:_state_hash(a['adversaries']) for n,a in final_arms.items()},
            'teachers':{n:m.fingerprint() for n,m in teachers['maps'].items()}}
    before=state();outputs=frozen_digest(releases)
    write_json(directory/'release_freeze.json',{'created_utc':now(),'output_hashes':outputs,'state':before,'initialization_identity':initial_identity,
        'representation_fitting_complete':True,'reserved_labels_received':False,'fit_raw_row_sha256':support['representation_fit']['raw_row_sha256'],
        'protocol_freeze_sha256':sha_file(out/'protocol_freeze.json'),
        'release_metadata':{n:{'dimension':16,'teacher':teacher_by_release[n],'static':is_static,**feature_rank(a['representation_fit'])} for n,a in releases.items()}})
    affine,geometry=fit_snapshots(releases,pca,pre,teachers['maps'],directory,teacher_by_release)
    write_json(directory/'affine_freeze.json',{'created_utc':now(),'fits':geometry,
        'files_sha256':{p.name:sha_file(p) for p in directory.glob('affine_*.npz')}})
    # First new reserved outcome access follows the complete C/D and decoder freeze.
    fy=binary_tasks(frame.iloc[pools['downstream_fit']],edges);vy=binary_tasks(frame.iloc[pools['downstream_validation']],edges)
    ay=audit_labels(frame.iloc[pools['attacker_fit']]);av=audit_labels(frame.iloc[pools['attacker_validation']])
    ti={t:subset_indices(fy[t],cfg['head_budget'],1230000+100*seed+j) for j,t in enumerate(TASKS)}
    ai={t:subset_indices(ay[t],cfg['attacker_budget'],1240000+100*seed+j) for j,t in enumerate(AUDITS)}
    def indices(ix,pool):
        return {t:{'n':len(v),'pool_indices_sha256':array_hash(v),'raw_rows_sha256':array_hash(frame.iloc[pools[pool][v]]._raw_row.to_numpy())} for t,v in ix.items()}
    indices_record={'task_fit_indices':indices(ti,'downstream_fit'),'audit_fit_indices':indices(ai,'attacker_fit')}
    historical_indices=read(ROOT/cfg['init_reference_results']/f'seed_{seed}'/'selection_before_test.json')
    assert all(indices_record[k]==historical_indices[k] for k in indices_record)
    utility={};urecord=empty_record();audits={120:{},360:{}};arecord={120:empty_record(),360:empty_record()}
    phase.update(utility_fitting_seconds=0.,independent_audit_seconds=0.,catchup_seconds=0.)
    for release,arrays in releases.items():
        if release=='I': continue
        print(f'{unit}: {release}: utility',flush=True);tick=time.perf_counter()
        for j,target in enumerate(TASKS):
            ix,valid=ti[target],np.flatnonzero(vy[target]>=0);key=f'transfer/{release}/{target}'
            result=fit_candidates(arrays['downstream_fit'][ix],fy[target][ix],arrays['downstream_validation'][valid],vy[target][valid],2,
                1250000+100*seed+j,budget={'mlp':{'epochs':cfg['head_mlp_epochs']}})
            utility[key]=result['candidates'];accept(utility[key],key,urecord);save_candidates(result,directory/'fitted'/key)
        phase['utility_fitting_seconds']+=time.perf_counter()-tick
        if not is_static and release not in final_arms: continue
        print(f'{unit}: {release}: nested120/360 audits',flush=True)
        for j,(target,k) in enumerate(AUDITS.items()):
            ix,valid=ai[target],np.flatnonzero(av[target]>=0);key=f'audit/{release}/{target}'
            xf,yf,xv,yv=arrays['attacker_fit'][ix],ay[target][ix],arrays['attacker_validation'][valid],av[target][valid]
            tick=time.perf_counter();statics=fit_static_auditors(xf,yf,xv,yv,k,1260000+100*seed+j)
            fresh=fit_extended_auditors(xf,yf,xv,yv,k,1260000+100*seed+j,static_candidates=statics)
            save_extended_audits(fresh,directory/'fitted'/key/'fresh');phase['independent_audit_seconds']+=time.perf_counter()-tick
            caught=None
            if not is_static:
                tick=time.perf_counter();caught=fit_extended_catchup(final_arms[release]['adversaries'][target],xf,yf,xv,yv,k,1300000+100*seed+j,
                    inherited_exposure=inherited_exposure(pair,release,target,support))
                save_extended_audits(caught,directory/'fitted'/key/'saved_start');phase['catchup_seconds']+=time.perf_counter()-tick
            for budget in audits:
                nested='nested'+str(budget);candidates=dict(fresh[nested]['candidates'])
                if caught: candidates.update(catchup=caught[nested],saved_adversary=caught['saved'])
                audits[budget][key]=candidates;accept(candidates,key,arecord[budget])
                arecord[budget]['fitting_records'][key].update(audit_budget=budget,catchup_available=caught is not None,
                    catchup_trajectory=caught[nested].metadata if caught else None,fresh_trajectory=fresh['metadata'][nested])
    provenance={'used_reference_files_sha256':refs.used_files,'used_original_files_sha256':refs.source.used_files,'initialization_identity':initial_identity,
        'teacher_manifest_sha256':sha_file((directory if is_static else out/'static'/f'seed_{seed}')/'teachers'/'teachers.json')}
    write_json(directory/'parent_provenance.json',provenance)
    selection={'created_utc':now(),'evaluation_status':cfg['evaluation_status'],**urecord,'budgets':{str(k):v for k,v in arecord.items()},**indices_record,
        'release_freeze_sha256':sha_file(directory/'release_freeze.json'),'affine_freeze_sha256':sha_file(directory/'affine_freeze.json'),
        'protocol_freeze_sha256':sha_file(out/'protocol_freeze.json')}
    write_json(directory/'selection_before_test.json',selection);selection_hash=sha_file(directory/'selection_before_test.json')
    print(f'{unit}: selections frozen; DEVELOPMENT EVALUATION',flush=True);evaluation_started=now();tick=time.perf_counter()
    tf=frame.iloc[pools['test']];test_pca=refs.evaluation_releases(tf)['E_pca']
    test={n:teachers['maps'][n[0]].apply(test_pca[:,:16]) for n in releases} if is_static else {n:m.release(test_pca) for n,m in models.items()}
    evaluate_snapshots(affine,geometry,test,test_pca,pre,teachers['maps'])
    write_json(directory/'geometry.json',geometry)
    raw=[];predictions={}
    for budget,fitted,record in [(None,utility,urecord),*((b,audits[b],arecord[b]) for b in audits)]:
        current={};role='transfer' if budget is None else 'audit';pool='downstream_validation' if role=='transfer' else 'attacker_validation'
        val=score_and_save(fitted,record['head_selections'],{role:{n:a[pool] for n,a in releases.items()}},{role:vy if role=='transfer' else av},
            {role:frame.iloc[pools[pool]].PWGTP.to_numpy(float)},'validation',current,cfg)
        evaluated=score_and_save(fitted,record['head_selections'],{role:test},{role:binary_tasks(tf,edges) if role=='transfer' else audit_labels(tf)},
            {role:tf.PWGTP.to_numpy(float)},'test',current,cfg)
        prefix='' if budget is None else f'budget{budget}/'
        predictions.update({prefix+k:v for k,v in current.items()})
        raw.extend(rows_from_scores(fitted,record,val,evaluated,seed,budget))
    np.savez_compressed(directory/'predictions.npz',**predictions)
    for name,a in releases.items(): np.savez_compressed(directory/f'release_{name}.npz',**a,test=test[name])
    integrity={'states_unchanged':before==state(),'releases_unchanged':outputs==frozen_digest(releases),
        'references_unchanged':refs.initial==refs.fingerprint(),'selection_unchanged':selection_hash==sha_file(directory/'selection_before_test.json'),
        'source_files_unchanged':all(sha_file(ROOT/p)==h for p,h in {**refs.used_files,**refs.source.used_files}.items()),
        'selection_created_utc':selection['created_utc'],'evaluation_started_utc':evaluation_started,
        'evaluation_output_hashes':{n:array_hash(a) for n,a in test.items()}}
    assert all(v for k,v in integrity.items() if k.endswith('_unchanged'))
    local=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha_file(p),'availability':'local only; reproduction instructions'}
        for p in directory.rglob('*') if p.is_file() and p.suffix in ('.pt','.npz','.joblib')]
    write_json(directory/'local_artifacts.json',local)
    phase['evaluation_serialization_seconds']=time.perf_counter()-tick;phase['total_seconds']=time.perf_counter()-started
    result={'seed':seed,'unit':unit,'teacher':'static' if is_static else unit[0],'rho':None if is_static else unit[1],
        'raw_metrics':raw,'cohort':cohort,'support_by_pool':support,'integrity':integrity,'runtime':phase,'evaluation_status':cfg['evaluation_status'],
        'new_final_models':0 if is_static else 2,'new_static_teachers':2 if is_static else 0,'completed_budgets':[120,360]}
    write_json(directory/'metrics.json',result)
    print(json.dumps({'unit_complete':unit,'runtime':phase}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true');parser.add_argument('--max-units',type=int)
    args=parser.parse_args();out=args.out.resolve();cfg=read(out/'config.json');torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if args.prepare: prepare(out)
        if args.run:
            progress_path=out/'progress.json'
            progress=read(progress_path) if progress_path.exists() else {'completed_units':[],'scientific_seconds':0.,'unit_seconds':[],'status':'running'}
            ran=0
            for index,unit in enumerate(cfg['execution_order']):
                path=unit_path(out,unit)
                if (path/'metrics.json').exists():
                    assert unit in progress['completed_units']
                    manifest=read(path/'completion.json')
                    assert all(sha_file(path/p)==h for p,h in manifest['sha256'].items())
                    continue
                elapsed=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(cfg['started_utc'])).total_seconds()
                estimate=max(progress['unit_seconds'] or [100.])
                if progress['scientific_seconds']+estimate*1.3>3600 or elapsed+estimate*1.3>14400-1800:
                    progress['status']='partial: insufficient budget for next complete unit';break
                tick=time.perf_counter()
                try: result=run_unit(out,cfg,unit)
                except Exception as error:
                    progress['scientific_seconds']+=time.perf_counter()-tick
                    progress.update(status='failed unit preserved',failed_unit=unit,failure=repr(error),updated_utc=now())
                    write_json(progress_path,progress);raise
                seconds=time.perf_counter()-tick
                write_json(path/'completion.json',{'created_utc':now(),'unit':unit,'sha256':{str(p.relative_to(path)):sha_file(p) for p in path.rglob('*') if p.is_file() and 'fitted' not in p.parts}})
                progress['completed_units'].append(unit);progress['unit_seconds'].append(seconds);progress['scientific_seconds']+=seconds
                progress.update(status='running',updated_utc=now(),projected_total_scientific_seconds=progress['scientific_seconds']+(len(cfg['execution_order'])-index-1)*max(progress['unit_seconds']))
                write_json(progress_path,progress);print(json.dumps(progress),flush=True);ran+=1
                if args.max_units and ran>=args.max_units: break
            if len(progress['completed_units'])==len(cfg['execution_order']): progress['status']='complete'
            write_json(progress_path,progress)


if __name__=='__main__': main()
