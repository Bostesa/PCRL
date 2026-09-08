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
from experiments.acs_transfer_heads import fit_candidates, _state_hash, metrics
from experiments.acs_protection_audits import AUDIT_BUDGET
from experiments.acs_preservation_audits import fit_extended_auditors, fit_extended_catchup, save_extended_audits
from experiments.run_acs_preservation_extended import _selection
from experiments.run_acs_bottleneck import SavedReferences, source_binaries
from experiments.run_acs_preservation import execution_paths as inherited_paths
from experiments.run_acs_protection import ROOT, TASKS, AUDITS, binary_tasks, frozen_digest, feature_rank, now, read, score_and_save
from experiments.run_acs_transfer import subset_indices, save_candidates

from experiments.run_acs_selective import load_context, fit_static_auditors, empty_record, accept, rows_from_scores, inherited_exposure
from experiments.run_acs_selective import execution_paths as selective_paths

DEFAULT_OUT=ROOT/'results/redesign_20260908_acs_restricted_inputs_v1'

def unit_path(out,unit):
    return out/f'{unit[0]}_{unit[1]}'/f'seed_{unit[-1]}'

def check_config(cfg):
    old=read(ROOT/cfg['selective_reference_results']/'config.json')
    for key in ('references','utility_tasks','seeds','threads','head_budget','attacker_budget','head_mlp_epochs',
                'audit_mlp_epochs','catchup_epochs','audit_restart_offset','audit_min_samples_leaf','margins','parity_tolerance','extended_audit_epochs'):
        assert cfg[key]==old[key],key
    training=dict(old['training']);training.update(input_dim=48,reconstruction_weight=0.)
    assert cfg['training']==training
    assert cfg['study']=='acs_restricted_inputs_v1' and cfg['teachers']==['E','S']
    assert cfg['input_conditions']==['F','K'] and cfg['reconstruction_coefficients']==[0.]
    assert cfg['preservation_betas']==[1.] and cfg['preservation_schedules']==['persistent']
    assert cfg['source_bank_preservation_beta']==0. and cfg['orthogonal_seed_base']==20260909
    assert cfg['mapper_initialization']=='shifted_teacher_identity_48'
    assert cfg['continuation_arms']==cfg['learned_arms']==['C','D'] and cfg['snapshot_releases']==['I','W','C','D']
    assert cfg['affine_rank_rcond']==cfg['ratio_variance_floor']==1e-12
    assert cfg['bank_output_order']==['income_binary','civilian_at_work','public_coverage']
    assert cfg['execution_order']==[[t,a,s] for t in ['E','S'] for s in range(3) for a in ['F','K','bank']]
    assert cfg['maximum_scientific_seconds']==3600 and cfg['maximum_total_work_seconds']==14400 and cfg['reserved_reporting_seconds']==1800

def execution_paths():
    return list(dict.fromkeys([*selective_paths(),ROOT/'experiments/run_acs_restricted.py',ROOT/'experiments/acs_restricted_training.py']))

def frozen_hashes(out):
    paths=[out/'config.json',out/'PROTOCOL.md',*execution_paths(),*(out/f'initial_Q_seed_{s}.npz' for s in range(3))]
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha_file(p) for p in paths}

def prepare(out):
    from experiments.acs_restricted_training import generate_q
    cfg=read(out/'config.json');check_config(cfg)
    assert not (out/'protocol_freeze.json').exists(),'Never overwrite a frozen study'
    for seed in cfg['seeds']:
        with (out/f'initial_Q_seed_{seed}.npz').open('xb') as handle: np.savez_compressed(handle,Q=generate_q(seed))
    parent=ROOT/cfg['selective_reference_results'];files=[parent/'protocol_freeze.json',parent/'config.json']
    # Preserve original hashes and verify exactly the frozen teachers/checkpoints, no regeneration.
    used={}
    for seed in cfg['seeds']:
        directory=parent/'static'/f'seed_{seed}'
        for entry in read(directory/'local_artifacts.json'):
            if '/teachers/' in entry['path']:
                assert sha_file(ROOT/entry['path'])==entry['sha256'];used[entry['path']]=entry['sha256']
        files.extend(directory/p for p in ('metrics.json','selection_before_test.json','geometry.json','teachers/teachers.json','teachers/teacher_prefit_freeze.json'))
        initial=ROOT/cfg['init_reference_results']/f'seed_{seed}'/'training/initialization.pt'
        entries={e['path']:e['sha256'] for e in read(initial.parent.parent/'local_artifacts.json')}
        key=str(initial.relative_to(ROOT));assert sha_file(initial)==entries[key];used[key]=entries[key]
    for path in parent.rglob('metrics.json'):
        if 'fitted' not in path.parts: files.extend([path,path.parent/'selection_before_test.json'])
    write_json(out/'protocol_freeze.json',{'created_utc':now(),'starting_commit':cfg['starting_commit'],
        'sha256':frozen_hashes(out),'reference_record_hashes':{str(p.relative_to(ROOT)):sha_file(p) for p in files},
        'required_local_reference_hashes':used,'regenerated_artifacts':[],
        'historical_source_hashes':'retained in original freezes, not replaced with new source identities',
        'initial_Q':{str(s):{'file_sha256':sha_file(out/f'initial_Q_seed_{s}.npz'),'seed':20260909+s} for s in cfg['seeds']}})
    write_json(out/'environment.json',{**read(parent/'environment.json'),'created_utc':now(),'reused_existing_environment':True})

def verify_freeze(out):
    check_config(read(out/'config.json'));freeze=read(out/'protocol_freeze.json')
    assert freeze['sha256']==frozen_hashes(out),'Frozen execution changed'
    for p,h in {**freeze['reference_record_hashes'],**freeze['required_local_reference_hashes']}.items(): assert sha_file(ROOT/p)==h,p

def run_unit(out,cfg,unit):
    from experiments.acs_restricted_training import train_restricted_unit, train_source_bank
    from experiments.acs_selective_teachers import load_teachers
    from experiments.acs_selective_diagnostics import fit_snapshots, evaluate_snapshots
    verify_freeze(out);started=time.perf_counter();seed=unit[-1];is_bank=unit[1]=='bank';teacher,access,_=unit
    directory=unit_path(out,unit);directory.mkdir(parents=True,exist_ok=False)
    frame,cohort,pools,refs,pre=load_context(cfg,seed)
    pca=refs.releases['E_pca'];edges=refs.source.selection['income_edges'];support=refs.previous['support_by_pool']
    write_json(directory/'support.json',support)
    np.savez_compressed(directory/'split_rows.npz',**{p:frame.iloc[ix]._raw_row.to_numpy() for p,ix in pools.items()})
    phase={};tick=time.perf_counter();pair=None;models={}
    teachers=load_teachers(ROOT/cfg['selective_reference_results']/'static'/f'seed_{seed}'/'teachers')
    teacher_arrays={p:teachers['maps'][teacher].apply(x[:,:16]) for p,x in pca.items()}
    assert np.array_equal(teacher_arrays['representation_fit'],teachers['fit_targets'][teacher])
    assert np.array_equal(pca['representation_fit'][:,:16],teachers['fit_targets']['R'])
    assert np.array_equal(pre['mean'][:16],teachers['original_mean']) and np.array_equal(pre['scale'][:16],teachers['original_scale'])
    initial_path=ROOT/cfg['init_reference_results']/f'seed_{seed}'/'training/initialization.pt'
    initial=torch.load(initial_path,weights_only=True)['model_state']
    with np.load(out/f'initial_Q_seed_{seed}.npz') as saved: q=saved['Q'].copy()
    common=dict(teacher_name=teacher,fitting_statistics=pre,initial_state=initial,orthogonal_q=q)
    sy=source_binaries(frame.iloc[pools['representation_fit']],edges)
    sv=source_binaries(frame.iloc[pools['source_validation']],edges)
    if is_bank:
        pair=train_source_bank(teacher_arrays['representation_fit'],sy,teacher_arrays['source_validation'],sv,seed,directory/'training',**common)
    else:
        raw=dict(raw_fit=pca['representation_fit'],raw_val=pca['source_validation']) if access=='F' else {}
        pair=train_restricted_unit(teacher_arrays['representation_fit'],sy,audit_labels(frame.iloc[pools['representation_fit']]),
            teacher_arrays['source_validation'],sv,seed,directory/'training',access=access,**common,**raw)
    phase['representation_training_seconds']=time.perf_counter()-tick
    all_models={**pair['snapshots'],**{n:v['model'] for n,v in pair['arms'].items()}}
    models={'B':all_models['B']} if is_bank else all_models
    def infer(model,t,raw=None,native=False):
        kw={'raw':raw} if access=='F' else {}
        return model.source_probabilities(t,**kw) if native else model.release(t,**kw)
    def released(model,t,raw=None):
        if is_bank: return np.column_stack([infer(model,t,raw,native=True)[k][:,1] for k in cfg['bank_output_order']]).astype(np.float32)
        return infer(model,t,raw)
    releases={n:{p:released(m,teacher_arrays[p],x) for p,x in pca.items()} for n,m in models.items()}
    parity={}
    for p,t in teacher_arrays.items():
        value=infer(all_models['I'],t,pca[p]);error=value.astype(float)-t.astype(float)
        assert np.allclose(value,t,**cfg['parity_tolerance'])
        parity[p]={'max_abs':float(np.max(np.abs(error))),'rms':float(np.sqrt(np.mean(error**2))),
                   'teacher_sha256':array_hash(t),'initial_sha256':array_hash(value)}
    initial_identity={'teacher_output_parity':parity,'tolerance':cfg['parity_tolerance'],
        'Q_sha256':array_hash(q),'original_checkpoint_sha256':sha_file(initial_path),
        'direct_teacher_evidence_scope':'reused starting-function reference; numerical parity, not newly fitted I heads/audits'}
    final_arms={} if is_bank else pair['arms']
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
        'release_metadata':{n:{'dimension':3 if is_bank else 16,'teacher':teacher_by_release[n],'static':False,'bank':is_bank,**feature_rank(a['representation_fit'])} for n,a in releases.items()}})
    affine,geometry=({}, {'snapshots':{},'scope':'bank releases only three source probabilities; no16-coordinate bank interface'}) if is_bank else fit_snapshots(releases,pca,pre,teachers['maps'],directory,teacher_by_release)
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
        if not is_bank and release not in final_arms: continue
        print(f'{unit}: {release}: nested120/360 audits',flush=True)
        for j,(target,k) in enumerate(AUDITS.items()):
            ix,valid=ai[target],np.flatnonzero(av[target]>=0);key=f'audit/{release}/{target}'
            xf,yf,xv,yv=arrays['attacker_fit'][ix],ay[target][ix],arrays['attacker_validation'][valid],av[target][valid]
            tick=time.perf_counter();statics=fit_static_auditors(xf,yf,xv,yv,k,1260000+100*seed+j)
            fresh=fit_extended_auditors(xf,yf,xv,yv,k,1260000+100*seed+j,static_candidates=statics)
            save_extended_audits(fresh,directory/'fitted'/key/'fresh');phase['independent_audit_seconds']+=time.perf_counter()-tick
            caught=None
            if not is_bank:
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
        'teacher_manifest_sha256':sha_file(ROOT/cfg['selective_reference_results']/'static'/f'seed_{seed}'/'teachers'/'teachers.json')}
    write_json(directory/'parent_provenance.json',provenance)
    selection={'created_utc':now(),'evaluation_status':cfg['evaluation_status'],**urecord,'budgets':{str(k):v for k,v in arecord.items()},**indices_record,
        'release_freeze_sha256':sha_file(directory/'release_freeze.json'),'affine_freeze_sha256':sha_file(directory/'affine_freeze.json'),
        'protocol_freeze_sha256':sha_file(out/'protocol_freeze.json')}
    write_json(directory/'selection_before_test.json',selection);selection_hash=sha_file(directory/'selection_before_test.json')
    print(f'{unit}: selections frozen; DEVELOPMENT EVALUATION',flush=True);evaluation_started=now();tick=time.perf_counter()
    tf=frame.iloc[pools['test']];test_pca=refs.evaluation_releases(tf)['E_pca']
    test_teacher=teachers['maps'][teacher].apply(test_pca[:,:16])
    test={n:released(m,test_teacher,test_pca) for n,m in models.items()}
    value=infer(all_models['I'],test_teacher,test_pca);assert np.allclose(value,test_teacher,**cfg['parity_tolerance'])
    error=value.astype(float)-test_teacher.astype(float)
    write_json(directory/'initial_evaluation_parity.json',{'max_abs':float(np.max(np.abs(error))),'rms':float(np.sqrt(np.mean(error**2))),'tolerance':cfg['parity_tolerance']})
    if not is_bank: evaluate_snapshots(affine,geometry,test,test_pca,pre,teachers['maps'])
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
    # Native source predictions are fixed heads, never downstream-selected probes.
    native_rows=[];native_arrays={}
    for name,model in models.items():
        for split,t,raw_input,y,w in (
            ('source_validation',teacher_arrays['source_validation'],pca['source_validation'],sv,frame.iloc[pools['source_validation']].PWGTP.to_numpy(float)),
            ('test',test_teacher,test_pca,source_binaries(tf,edges),tf.PWGTP.to_numpy(float))):
            probs=infer(model,t,raw_input,native=True)
            for target in cfg['bank_output_order']:
                valid=y[target]>=0;p=probs[target]
                native_arrays[f'{split}/{name}/{target}']=p
                native_rows.append({'seed':seed,'release':name,'target':target,'split':split,
                    'score':metrics(y[target][valid],p[valid],2),'person_weighted':metrics(y[target][valid],p[valid],2,w[valid]),
                    'probability_sha256':array_hash(p),'selection':'fixed native source head; no downstream probe'})
    write_json(directory/'native_source.json',{'output_order':cfg['bank_output_order'],'raw_metrics':native_rows})
    np.savez_compressed(directory/'native_predictions.npz',**native_arrays)
    composition=[]
    if access!='F':
        # All paths below invoke student or bank with teacher values alone.
        for budget,fitted in audits.items():
            for key,candidates in fitted.items():
                _,name,target=key.split('/')
                for cid,candidate in candidates.items():
                    record={'seed':seed,'teacher':teacher,'source_release':name,'bank':is_bank,'target':target,
                        'candidate_id':cid,'audit_budget':budget,'model_state_sha256':_state_hash(models[name]),
                        'raw_input_argument_supplied':False,'teacher_only_composition':True,'splits':{}}
                    for split,t,x in [('validation',teacher_arrays['attacker_validation'],releases[name]['attacker_validation']),('test',test_teacher,test[name])]:
                        valid=(av[target] if split=='validation' else audit_labels(tf)[target])>=0
                        composed=candidate.predict_proba(released(models[name],t)[valid])
                        direct=candidate.predict_proba(x[valid])
                        stored=predictions[f'budget{budget}/{split}/{key}/{cid}']
                        assert np.array_equal(direct,stored)
                        assert np.allclose(composed,direct,**cfg['parity_tolerance'])
                        record['splits'][split]={'bitwise_equal':bool(np.array_equal(composed,direct)),
                            'max_abs':float(np.max(np.abs(composed-direct))),'rms':float(np.sqrt(np.mean((composed-direct)**2))),
                            'composed_probability_sha256':array_hash(composed),'direct_probability_sha256':array_hash(direct)}
                    composition.append(record)
    write_json(directory/'composition_parity.json',{'eligible':access!='F','all_existing_audit_candidates':composition,
        'scope':'expanded witness family; additional source/observer fitting exposure and model depth, not matched direct-teacher budget'})
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
    result={'seed':seed,'unit':unit,'teacher':teacher,'access':access,'rho':0.,
        'raw_metrics':raw,'cohort':cohort,'support_by_pool':support,'integrity':integrity,'runtime':phase,'evaluation_status':cfg['evaluation_status'],
        'new_final_models':1 if is_bank else 2,'new_static_teachers':0,'new_source_banks':int(is_bank),'completed_budgets':[120,360]}
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
                estimate=max(progress['unit_seconds'] or [150.])
                remaining_block=3 if unit[1]=='F' else (2 if unit[1]=='K' else 1)
                if progress['scientific_seconds']+estimate*remaining_block*1.3>3600 or elapsed+estimate*remaining_block*1.3>14400-1800:
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
                if len(progress['completed_units'])==3:
                    write_json(out/'FIRST_BLOCK_PROJECTION.json',{'created_utc':now(),'completed_units':progress['completed_units'],'first_block_seconds':progress['scientific_seconds'],'projected_core_seconds':progress['scientific_seconds']*6,'policy':'complete full unchanged blocks within ceilings'})
                if args.max_units and ran>=args.max_units: break
            if len(progress['completed_units'])==len(cfg['execution_order']): progress['status']='complete'
            write_json(progress_path,progress)


if __name__=='__main__': main()
