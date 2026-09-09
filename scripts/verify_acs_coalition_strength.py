"""Independent replay of new coalition-strength releases and frozen auditors.

Historical verification remains immutable and is reused through recorded hashes.
No experimental fitting, historical rerun, optimizer step or development selection.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import verify_acs_bottleneck_scores as check
from scripts.verify_acs_coalition import (
    TARGETS, TASKS, ASSIGNED, AUTHORIZED, ROLES, FRESH, SCOPES,
    literal_wires, public_probabilities, candidate_input, expected_pool, check_curves, verify_native,
)

NEW_BETAS = {'0p025':.025,'0p05':.05,'0p2':.2}
HISTORICAL_COMMIT = '9461fe29c4f7db337ea5dc20e295dc8d02c3d9e2'


def condition_identity(name):
    """Parse only the predeclared new endpoints; aliases never become new fits."""
    interface, regime, encoded = name.split('_')
    if interface not in ('F','P') or regime not in ('Iplus','J') or not encoded.startswith('b') or encoded[1:] not in NEW_BETAS:
        raise ValueError('Only the36 new fixed-grid continuations are replayed here')
    return interface,regime,NEW_BETAS[encoded[1:]]


def published_hash(path,commit=HISTORICAL_COMMIT):
    """Bind preserved local bytes to the reviewed Git object, not a new hash alone."""
    relative=str(Path(path).resolve().relative_to(ROOT))
    content=subprocess.run(['git','show',commit+':'+relative],cwd=ROOT,check=True,capture_output=True).stdout
    digest=hashlib.sha256(content).hexdigest()
    assert check.sha(path)==digest,('published bytes changed',relative)
    return digest


def candidate_belongs_to_condition(path,dest):
    """An inherited singleton is local to this deployment, never another beta."""
    candidate=Path(path).resolve()
    allowed=(Path(dest)/'audits/fitted').resolve()
    if not candidate.is_relative_to(allowed):
        raise ValueError('Candidate crossed a condition/interface/seed deployment boundary')
    return candidate


def verify_release_arrays(releases,checkpoint,pca,condition,outputs,report):
    interface,_,_=condition_identity(condition)
    for pool,x in pca.items():
        wires,derived,_=literal_wires(checkpoint['model_state'],x,interface)
        for space,values in [('wire',wires),('derived',derived)]:
            for view,value in values.items():
                np.testing.assert_array_equal(value,releases[space+'/'+view+'/'+pool])
                report['literal_release_arrays']+=1
                if pool!='test': assert check.array_hash(value)==outputs[space][view][pool]
        if interface=='F':
            composed,_=public_probabilities(checkpoint['model_state'],{v:releases['wire/'+v+'/'+pool] for v in ('A','B')})
            for view in ('A','B','AB'):
                np.testing.assert_array_equal(composed[view],releases['derived/'+view+'/'+pool])
                report['public_head_composition_arrays']+=1
        else:
            for view in ('A','B','AB'):
                np.testing.assert_array_equal(releases['wire/'+view+'/'+pool],releases['derived/'+view+'/'+pool])


def verify_condition(dest,seed,condition,checkpoint,releases,release_freeze,release_freeze_hash,
                     completion,fit_indices,frames,label,rows,report,cache,score_cache,seen,terminals):
    condition_identity(condition)
    selected = check.read(dest/'selection_before_test.json'); audits = check.read(dest/'audits/audit_selection.json')
    interface,family,beta=condition_identity(condition)
    assert selected['coefficient_identity']=={'interface':interface,'family':family,'beta':beta}
    assert selected['release_freeze_sha256'] == release_freeze_hash
    assert release_freeze['created_utc'] <= selected['created_utc'] <= completion['completed_utc']
    assert not audits['development_received'] and audits['interface'] == condition[0]
    report['new_trajectory_counts'][str(seed)+'/'+condition] = audits['counts']
    expected_roles_count = 22 if condition.startswith('F') else 11
    assert audits['counts']['new_five_candidate_roles'] == expected_roles_count
    assert audits['counts']['new_fresh_mlp_trajectories'] == expected_roles_count*2
    assert audits['counts']['own_catchup_trajectories'] == 9
    assert audits['counts']['reused_five_candidate_roles'] == 0
    assert selected['audits'] == audits['selections']
    for role, record in selected['utility_metadata'].items():
        choices = record['candidates']
        assert selected['utility'][role] == min(choices,key=lambda c:(choices[c]['validation_scores']['log_loss'],c))
    expected_roles = {v+'/'+t for v,ts in ROLES.items() for t in ts}
    for budget in ('120','360'):
        assert set(audits['candidates'][budget]) == expected_roles
        for role, candidates in audits['candidates'][budget].items():
            view,target = role.split('/'); own = {'wire__'+c for c in FRESH}
            if condition.startswith('F'): own |= {'derived__'+c for c in FRESH}
            if target not in ('same_residence','commute_over20'): own |= {'saved_adversary','catchup'}
            inherited = {f'inherited_{v}__{cid}' for v in ('A','B') for cid in audits['candidates'][budget][v+'/'+target]} if view == 'AB' else set()
            assert set(candidates) == own|inherited
            for scope in SCOPES:
                eligible = expected_pool(candidates,scope)
                assert eligible == audits['selection_pools'][budget][role][scope]
                winner = min(eligible,key=lambda c:(candidates[c]['validation_scores']['log_loss'],c))
                assert winner == audits['selections'][budget][role][scope]; report['selection_pools'] += 1
                if view == 'AB': assert all(candidates[winner]['validation_scores']['log_loss'] <= candidates[c]['validation_scores']['log_loss'] for c in eligible)
    measurements = check.read(dest/'metrics.json')['raw_metrics']
    expected_rows = {('audit',*role.split('/'),int(budget),cid) for budget,roles in audits['candidates'].items() for role,candidates in roles.items() for cid in candidates}
    expected_rows |= {('utility',*role.split('/'),None,cid) for role,record in selected['utility_metadata'].items() for cid in record['candidates']}
    assert len(measurements) == len(expected_rows)
    assert {(r['role'],r['view'],r['target'],r['audit_budget'],r['candidate_id']) for r in measurements} == expected_rows
    with np.load(dest/'predictions.npz') as predictions:
        assert len(predictions.files) == len(measurements)*2
        for row in measurements:
            role,view,target,cid,budget = (row[k] for k in ('role','view','target','candidate_id','audit_budget'))
            assert row['seed'] == seed and row['condition'] == condition
            if role == 'audit':
                meta = audits['candidates'][str(budget)][view+'/'+target][cid]
                path = candidate_belongs_to_condition(meta['base_candidate_directory'],dest); base_meta = check.read(path/'metadata.json')
                assert check.sha(path/'metadata.json') == meta['base_metadata_sha256']
                assert all(meta[k] == v for k,v in base_meta.items() if k not in ('candidate_id',))
                scopes = [s for s,c in audits['selections'][str(budget)][view+'/'+target].items() if c == cid]
                if meta['inherited_singleton']:
                    origin = meta['source_view']; original = audits['candidates'][str(budget)][origin+'/'+target][meta['source_candidate_id']]
                    assert original['base_candidate_directory'] == str(path) and meta['space'] == original['space']
                    widths = (2,1) if meta['space'] == 'derived' or condition.startswith('P') else (16,16)
                    expected_columns = list(range(widths[0])) if origin == 'A' else list(range(widths[0],sum(widths)))
                    assert meta['projection_columns'] == expected_columns
                assert meta['actual_auditor_input_dimension'] == base_meta['input_dim']
            else:
                assert target in AUTHORIZED[view] and budget is None
                meta = selected['utility_metadata'][view+'/'+target]['candidates'][cid]; base_meta = meta
                path = dest/'fitted/utility'/view/target/cid
                assert check.read(path/'metadata.json') == meta
                scopes = ['utility'] if selected['utility'][view+'/'+target] == cid else []
            assert row['selected_scopes'] == scopes
            classes = 9 if target == 'RAC1P' else 2
            fitpool,valpool = ('downstream_fit','downstream_validation') if role == 'utility' else ('attacker_fit','attacker_validation')
            ix = fit_indices[role,target]; yf,mf = label[fitpool,target]; yv,mv = label[valpool,target]
            inputs = (lambda p,m: candidate_input(releases,meta,p,m)) if role == 'audit' else (lambda p,m: np.ascontiguousarray(releases['wire/'+view+'/'+p][m]))
            xf,xv = inputs(fitpool,ix).astype(np.float64),inputs(valpool,mv).astype(np.float64)
            unique = str(path)
            if unique not in cache: cache[unique] = check.load_inference(path,base_meta)
            predict,mean,scale,modelstate = cache[unique]
            if unique not in seen:
                seen.add(unique); report['unique_saved_candidate_paths'] += 1
                assert base_meta['n_classes'] == classes
                assert base_meta['validation_hashes'] == {'x':check.array_hash(xv),'y':check.array_hash(yv[mv].astype(np.int64))}
                saved = base_meta.get('audit_kind') == 'saved'; caught = base_meta.get('audit_kind') == 'catchup'
                if not saved:
                    assert base_meta['fit_hashes'] == {'x':check.array_hash(xf),'y':check.array_hash(yf[ix].astype(np.int64))}
                    assert base_meta['fit_support'] == np.bincount(yf[ix],minlength=classes).tolist()
                if saved or caught:
                    np.testing.assert_array_equal(mean,np.zeros(xf.shape[1])); np.testing.assert_array_equal(scale,np.ones(xf.shape[1]))
                    source_view = meta['source_view']; prefix = source_view+'__'+target+'.'
                    original = {k.removeprefix(prefix):v for k,v in checkpoint['adversary_state'].items() if k.startswith(prefix)}
                    assert check.state_hash(original) == base_meta['source_state_hash']
                    exposure = base_meta['inherited_exposure']; ty,tm = label['representation_fit',target]; ty = np.where(tm,ty,-1).astype(np.int64)
                    assert exposure['source_fit_raw_row_hash'] == check.array_hash(rows['representation_fit'])
                    assert exposure['fit_label_hash'] == check.array_hash(ty)
                    assert exposure['representation_fitting_passes'] == 260 and exposure['known_fitting_labels'] == int(tm.sum())
                    assert exposure['label_presentations'] == int(tm.sum())*260; report['inherited_exposure_checks'] += 1
                    if saved:
                        assert check.state_hash(modelstate) == check.state_hash(original)
                        np.testing.assert_array_equal(predict(xv),check.network_probability(original,xv)); report['saved_observer_fidelity'] += 1
                    else:
                        assert base_meta['initial_state_hash'] == check.state_hash(original)
                        assert base_meta['optimizer']['initial_state_entries'] == 0 and not base_meta['optimizer']['restored_state']
                else:
                    np.testing.assert_array_equal(mean,xf.mean(0)); std = xf.std(0)
                    np.testing.assert_array_equal(scale,np.where(std>1e-12,std,1.))
                original_cid = path.name
                expected_seed = ((1250000+100*seed+TASKS.index(target)) if role == 'utility' else (1300000 if caught or saved else 1260000)+100*seed+10*('A','B','AB').index(meta['source_view'])+TARGETS.index(target))
                if original_cid == 'mlp_1': expected_seed += 10000
                assert base_meta['seed'] == expected_seed
                check_curves(path,base_meta,len(ix),expected_seed,budget,report,terminals)
            for split,pool in [('validation',valpool),('test','test')]:
                y,mask = label[pool,target]; p = predictions[f'{role}/{view}/{target}/{budget}/{cid}/{split}']
                np.testing.assert_array_equal(predict(inputs(pool,mask)),p); report['prediction_sets'] += 1
                for suffix,weighted in [('',False),('_person_weighted',True)]:
                    skey = (seed,pool,target,check.array_hash(p),weighted)
                    if skey not in score_cache: score_cache[skey] = check.independent_scores(y[mask],p,classes,frames[pool].PWGTP.to_numpy(float)[mask] if weighted else None)
                    check.compare(score_cache[skey],row['scores'][split+suffix],f'{seed}/{condition}/{role}/{view}/{target}/{budget}/{cid}/{split}{suffix}'); report['score_dictionaries'] += 1
                if role == 'audit' and meta['inherited_singleton']:
                    direct = predictions[f'audit/{meta["source_view"]}/{target}/{budget}/{meta["source_candidate_id"]}/{split}']
                    np.testing.assert_array_equal(p,direct); report['inherited_prediction_sets'] += 1
            check.compare(base_meta['validation_scores'],row['scores']['validation'],f'{seed}/{condition}/{role}/{view}/{target}/{budget}/{cid}/validation_metadata')
            report['candidate_records'] += 1
        history = check.read(dest/'history_attack_replay.json')
        expected_history = {(r['view']+'/'+r['target'],r['candidate_id'],k) for r in measurements if r['role']=='audit' and r['audit_budget']==360 and r['selected_scopes'] for k in (1,2,4)}
        assert {(r['role'],r['candidate_id'],r['calls']) for r in history['records']} == expected_history
        for r in history['records']:
            view,target = r['role'].split('/'); meta = audits['candidates']['360'][r['role']][r['candidate_id']]
            _,mask = label['test',target]; x = candidate_input(releases,meta,'test',mask)
            repeated = np.stack([x]*r['calls'],1)
            assert all(np.array_equal(repeated[:,i],x) for i in range(r['calls']))
            predict = cache[meta['base_candidate_directory']][0]
            p = predict(np.ascontiguousarray(repeated[:,0]))
            assert check.array_hash(p) == r['prediction_sha256']; report['history_prediction_sets'] += 1


def check_completion(path,report,seen_hashes):
    record=check.read(path)
    for name,h in record['files_sha256'].items():
        if name not in seen_hashes:
            assert check.sha(ROOT/name)==h,('completed artifact changed',name)
            seen_hashes[name]=h;report['local_artifact_hashes']+=1
        else: assert seen_hashes[name]==h
    return record


def expected_system_identities():
    identities={}
    for seed in (0,1,2):
        for interface in ('F','P'):
            identities[seed,interface+'_I']=(interface,'I',0.,True,interface+'_I')
            for encoded,beta in (*NEW_BETAS.items(),('0p1',.1)):
                for family in ('Iplus','J'):
                    name=f'{interface}_{family}_b{encoded}'
                    identities[seed,name]=(interface,family,beta,beta==.1,interface+'_'+family if beta==.1 else name)
    return identities


def verify_global_gate(global_freeze,training_completions,fit_starts):
    """Validate one global54 freeze, including units outside a partial replay."""
    assert global_freeze['all_54_systems_frozen'] and not global_freeze['new_reserved_fitting_started']
    assert global_freeze['historical_reserved_outcomes_previously_known']
    expected=expected_system_identities()
    systems=global_freeze['systems']
    assert len(systems)==len(expected)==54
    assert len({(e['seed'],e['condition']) for e in systems})==54
    for e in systems:
        assert (e['interface'],e['family'],e['beta'],e['reused'],e['original_condition'])==expected[e['seed'],e['condition']]
    assert len(training_completions)==36
    assert all(c['training_complete'] and not c['reserved_labels_accessed'] and c['completed_utc']<=global_freeze['created_utc'] for c in training_completions)
    for start in fit_starts:
        assert start['all_54_frozen'] and start['first_reserved_labels_after_global_freeze']
        assert global_freeze['created_utc']<=start['started_utc']


def verify_reuse(out,cfg,report):
    manifest=check.read(out/'REUSE_MANIFEST.json');historical=ROOT/cfg['coalition_reference_results']
    assert manifest['original_commit']==HISTORICAL_COMMIT
    assert manifest['historical_training_or_audit_refits']==0 and manifest['regeneration']==[]
    assert manifest['original_scientific_hashes_preserved']
    assert len(manifest['systems'])==54 and sum(e['reused'] for e in manifest['systems'])==18
    certificates={}
    for name in ('SCORE_REPLAY.json','TRAINING_REPLAY.json','FINAL_REPORT_REPLAY.json','protocol_freeze.json','EXECUTION_AMENDMENTS.json'):
        certificates[name]=published_hash(historical/name)
    score=check.read(historical/'SCORE_REPLAY.json')
    assert score['passed'] and score['complete_matrix'] and score['seeds']==[0,1,2]
    assert score['script_sha256']==check.sha(ROOT/'scripts/verify_acs_coalition.py')
    assert score['metric_helper_sha256']==check.sha(check.__file__)
    assert score['checkpoint_helper_sha256']==check.sha(ROOT/'scripts/verify_acs_preservation_extended.py')
    assert check.read(historical/'TRAINING_REPLAY.json')['passed']
    for seed in (0,1,2):
        directory=historical/f'seed_{seed}'
        for name in ('complete.json','local_artifacts.json','release_freeze.json','training/training.json'):
            published_hash(directory/name)
        completed=check.read(directory/'complete.json')
        assert check.sha(directory/'local_artifacts.json')==completed['local_artifact_manifest_sha256']
        for name,h in completed['compact_sha256'].items():
            assert manifest['historical_files_sha256'][str((directory/name).relative_to(ROOT))]==h
        for entry in check.read(directory/'local_artifacts.json'):
            assert manifest['historical_files_sha256'][entry['path']]==entry['sha256']
    for name,h in manifest['historical_files_sha256'].items():
        assert check.sha(ROOT/name)==h,('historical evidence changed',name)
    for e in manifest['systems']:
        if e['reused']:
            for name,h in e['original_files_sha256'].items():
                assert manifest['historical_files_sha256'][name]==h
    report['historical_reuse']={'paired_systems':18,'new_control_or_static_fits':0,
        'historical_files_hashed':len(manifest['historical_files_sha256']),
        'mature_verification_certificates':certificates,'repeated_historical_model_or_score_inference':False,
        'scope':'Current retained artifacts match published completed-study hashes; original mature full prediction/score replay remains the reference.'}


def verify(out,seeds=None,max_units=None):
    tick=time.perf_counter();check.errors.clear()
    check.max_error,check.max_error_path,check.numeric_comparisons=0.,None,0
    cfg=check.read(out/'config.json');freeze=check.read(out/'protocol_freeze.json')
    for name,h in freeze['scientific_and_protocol_sha256'].items(): assert check.sha(ROOT/name)==h,('frozen execution source',name)
    assert freeze['reuse_manifest_sha256']==check.sha(out/'REUSE_MANIFEST.json')
    assert freeze['prefit_identity_sha256']==check.sha(out/'PREFIT_IDENTITY.json')
    global_freeze=check.read(out/'RELEASE_MANIFEST.json');global_hash=check.sha(out/'RELEASE_MANIFEST.json')
    assert global_freeze['protocol_sha256']==check.sha(out/'protocol_freeze.json')
    report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'New fixed-grid paired models, every audit candidate and legal singleton projection, public-head composition, validation pools, nested terminal states, native heads and raw-label weighted/unweighted scores; historical evidence hash reuse only',
        'script_sha256':check.sha(__file__),'metric_helper_sha256':check.sha(check.__file__),
        'literal_replay_helper_sha256':check.sha(ROOT/'scripts/verify_acs_coalition.py'),
        'checkpoint_helper_sha256':check.sha(ROOT/'scripts/verify_acs_preservation_extended.py'),
        'protocol_freeze_sha256':check.sha(out/'protocol_freeze.json'),'global_release_manifest_sha256':global_hash,
        'evaluation_status':cfg['evaluation_status'],'absolute_metric_tolerance':check.TOL,
        'seed_evidence':{},'new_trajectory_counts':{},'limitations':['No optimizer steps or historical fitting replay.',
        'Alternative coefficients/interfaces/seeds are separate deployments; no cross-condition attacker transfer.',
        'Inherited public-head and observer exposure remains separate from independent attacker fitting.',
        'Absent independent race-category support remains unassessable; finite predictive audit only.']}
    for key in ('candidate_records','score_dictionaries','prediction_sets','unique_saved_candidate_paths','selection_pools',
                'inherited_prediction_sets','literal_release_arrays','public_head_composition_arrays','native_score_dictionaries',
                'native_predictions','mlp_schedules','actual_terminal_checkpoints','nested_trajectory_prefixes','saved_observer_fidelity',
                'inherited_exposure_checks','history_prediction_sets','local_artifact_hashes'): report[key]=0
    verify_reuse(out,cfg,report)
    hashed={};training_completions=[];fit_starts=[]
    for e in global_freeze['systems']:
        assert check.sha(ROOT/e['checkpoint_path'])==e['checkpoint_sha256']
        assert check.sha(ROOT/e['release_record_path'])==e['release_record_sha256']
        source=check.read(ROOT/e['release_record_path'])
        outputs=source['outputs'][e['original_condition']] if e['reused'] else source['outputs']
        assert outputs==e['outputs']
        if not e['reused']:
            directory=out/f"seed_{e['seed']}"
            training_completions.append(check_completion(directory/'training'/e['condition']/'complete.json',report,hashed))
            assert source['checkpoint_sha256']==e['checkpoint_sha256']
            assert source['protocol_sha256']==check.sha(out/'protocol_freeze.json')
            if (directory/e['condition']/'fit_start.json').exists():
                start=check.read(directory/e['condition']/'fit_start.json')
                assert start['global_release_manifest_sha256']==global_hash
                fit_starts.append(start)
    verify_global_gate(global_freeze,training_completions,fit_starts)
    report['global_release_gate']={'frozen_systems':54,'new_training_completions':36,'observed_fit_starts':len(fit_starts),'passed':True}
    selected_seeds=cfg['seeds'] if seeds is None else seeds
    assert selected_seeds and len(set(selected_seeds))==len(selected_seeds) and set(selected_seeds)<=set(cfg['seeds'])
    units=[e for seed in selected_seeds for name in cfg['new_condition_order'] for e in global_freeze['systems']
           if e['seed']==seed and e['condition']==name and not e['reused'] and (out/f'seed_{seed}'/name/'complete.json').exists()]
    if max_units is not None:
        assert max_units>0;units=units[:max_units]
    assert units,'No immutable completed new evaluation unit to replay'
    parent=ROOT/cfg['parent_results'];raw_path=ROOT/check.read(parent/'config.json')['raw_path']
    assert check.sha(raw_path)==check.read(parent/'schema_support.json')['raw_sha256']
    report['raw_sha256']=check.sha(raw_path)
    raw=pd.read_csv(raw_path,usecols=['MIG','JWMNP','PINCP','ESR','PUBCOV','SEX','RAC1P','PWGTP'],low_memory=False)
    cache,score_cache,seen={},{},set()
    for seed in selected_seeds:
        local=[e for e in units if e['seed']==seed]
        if not local: continue
        directory=out/f'seed_{seed}'
        with np.load(directory/'split_rows.npz') as z: rows={p:z[p].copy() for p in z.files}
        with np.load(parent/f'seed_{seed}/split_rows.npz') as z:
            assert set(rows)==set(z.files) and all(np.array_equal(v,z[p]) for p,v in rows.items())
        frames={p:raw.iloc[ix] for p,ix in rows.items()}
        labels={(p,t):check.labels(f,t) for p,f in frames.items() for t in TARGETS}
        with np.load(ROOT/cfg['reference_results']/f'seed_{seed}/release_E_pca.npz') as z:pca={p:z[p].copy() for p in z.files}
        indices=check.read(directory/'indices.json');fit_indices={}
        assert indices==check.read(ROOT/cfg['coalition_reference_results']/f'seed_{seed}/indices.json')
        for kind,pool,order,cap,base in [('utility','downstream_fit',TASKS,2048,1230000),('audit','attacker_fit',TARGETS,4096,1240000)]:
            for j,target in enumerate(order):
                _,valid=labels[pool,target];ix=np.random.default_rng(base+100*seed+j).permutation(np.flatnonzero(valid))[:cap]
                fit_indices[kind,target]=ix
                assert indices['task_fit_indices' if kind=='utility' else 'audit_fit_indices'][target]=={
                    'n':len(ix),'pool_indices_sha256':check.array_hash(ix),'raw_rows_sha256':check.array_hash(rows[pool][ix])}
        terminals={};unit_evidence={}
        for e in local:
            condition=e['condition'];dest=directory/condition
            complete=check_completion(dest/'complete.json',report,hashed)
            assert complete['seed']==seed and complete['condition']==condition and complete['evaluation_complete']
            assert complete['global_release_manifest_sha256']==global_hash
            checkpoint=torch.load(ROOT/e['checkpoint_path'],map_location='cpu',weights_only=True)
            training=check.read(directory/'training'/condition/'training.json')
            assert check.state_hash(checkpoint['model_state'])==training['final_hashes']['model']
            assert check.state_hash(checkpoint['adversary_state'])==training['final_hashes']['adversaries']
            assert (training['interface'],training['regime'],training['beta'])==condition_identity(condition)
            with np.load(dest/'releases.npz') as z:releases={p:z[p].copy() for p in z.files}
            verify_release_arrays(releases,checkpoint,pca,condition,e['outputs'],report)
            verify_condition(dest,seed,condition,checkpoint,releases,global_freeze,global_hash,complete,
                             fit_indices,frames,labels,rows,report,cache,score_cache,seen,terminals)
            verify_native(dest,{condition:checkpoint},pca,frames,report)
            unit_evidence[condition]={'complete_sha256':check.sha(dest/'complete.json'),'checkpoint_sha256':e['checkpoint_sha256']}
            # Every inherited base lives in this unit. Bound replay memory to one
            # deployment instead of retaining thousands of fitted trees/MLPs.
            cache.clear();score_cache.clear()
        report['seed_evidence'][str(seed)]={'units':unit_evidence,'actual_training_checkpoints':terminals}
    n_f=sum(e['interface']=='F' for e in units);n_p=len(units)-n_f
    expected_counts={
        'candidate_records':362*n_f+212*n_p,'prediction_sets':2*(362*n_f+212*n_p),
        'score_dictionaries':4*(362*n_f+212*n_p),'unique_saved_candidate_paths':257*n_f+147*n_p,
        'selection_pools':66*len(units),'inherited_prediction_sets':192*n_f+112*n_p,
        'literal_release_arrays':42*len(units),'public_head_composition_arrays':21*n_f,
        'native_predictions':6*len(units),'native_score_dictionaries':12*len(units),
        'mlp_schedules':111*n_f+67*n_p,'actual_terminal_checkpoints':106*n_f+62*n_p,
        'nested_trajectory_prefixes':53*n_f+31*n_p,'saved_observer_fidelity':9*len(units),
        'inherited_exposure_checks':27*len(units),
    }
    assert all(report[k]==value for k,value in expected_counts.items()),{
        k:(report[k],value) for k,value in expected_counts.items() if report[k]!=value}
    report['expected_scope_counts']=expected_counts
    report.update(passed=not check.errors,errors=check.errors,seeds=sorted({e['seed'] for e in units}),
        new_units_verified=len(units),complete_matrix=len(units)==36,
        numeric_values_compared=check.numeric_comparisons,max_absolute_metric_error=check.max_error,
        max_error_path=check.max_error_path,runtime_seconds=time.perf_counter()-tick)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/redesign_20260909_acs_coalition_strength_v1')
    parser.add_argument('--report',type=Path);parser.add_argument('--seeds',type=int,nargs='+');parser.add_argument('--max-units',type=int)
    args=parser.parse_args()
    if args.report and args.report.exists():raise FileExistsError('Preserve completed replay evidence')
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):result=verify(args.out.resolve(),args.seeds,args.max_units)
    rendered=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.report:
        with args.report.open('x') as handle:handle.write(rendered)
    else:print(rendered,end='')
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
