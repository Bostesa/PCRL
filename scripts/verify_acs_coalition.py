"""Independent coalition model, public composition, candidate and metric replay.

Read-only: no experiment fitter, scorer, optimizer step, new data or selection.
Raw-label metrics and literal tensor inference reuse the established replay code.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import verify_acs_bottleneck_scores as check
from scripts.verify_acs_preservation_extended import checkpoint_evidence
from scripts.verify_acs_selective import load_map, map_probability_free

TARGETS = ('SEX','RAC1P','income_binary','civilian_at_work','public_coverage','same_residence','commute_over20')
TASKS = ('same_residence','commute_over20','income_binary','civilian_at_work','public_coverage')
ASSIGNED = {'A': ('income_binary','civilian_at_work'), 'B': ('public_coverage',)}
AUTHORIZED = {'A': ('income_binary','civilian_at_work','same_residence'), 'B': ('public_coverage','commute_over20')}
ROLES = {'A': ('public_coverage','commute_over20','SEX','RAC1P'), 'B': ('income_binary','civilian_at_work','same_residence','SEX','RAC1P'), 'AB': ('SEX','RAC1P')}
FRESH = ('logistic','mlp_0','mlp_1','hist_gb_20','hist_gb_5')
SCOPES = ('standard_independent','expanded_independent','expanded_catchup')
CONDITIONS = ('F_I','F_Iplus','F_J','P_I','P_Iplus','P_J','E')


def public_probabilities(state, features):
    result, native = {}, {}
    with torch.no_grad():
        for view, targets in ASSIGNED.items():
            h = torch.from_numpy(np.ascontiguousarray(features[view], dtype=np.float32))
            values = []
            for target in targets:
                prefix = f'branches.{view}.heads.{target}.'
                p = torch.sigmoid(F.linear(h, state[prefix+'weight'], state[prefix+'bias']))
                values.append(p)
                native[target] = torch.cat((1.-p, p), 1).numpy()
            result[view] = torch.cat(values, 1).numpy()
    result['AB'] = np.concatenate((result['A'], result['B']), 1)
    return result, native


def literal_wires(state, raw, interface):
    if interface not in ('F','P'): raise ValueError('Declared public interface required')
    raw = np.asarray(raw, np.float32)
    if raw.ndim != 2 or raw.shape[1] != 32: raise ValueError('Original PCA32 required')
    x = ((raw.astype(np.float64)-state['input_mean'].numpy())/state['input_scale'].numpy()).astype(np.float32)
    features = {}
    with torch.no_grad():
        for view in ('A','B'):
            prefix = f'branches.{view}.mapper.'
            h = F.linear(torch.from_numpy(x), state[prefix+'0.weight'], state[prefix+'0.bias'])
            features[view] = F.linear(torch.relu(h), state[prefix+'2.weight'], state[prefix+'2.bias']).numpy()
    features['AB'] = np.concatenate((features['A'], features['B']), 1)
    probabilities, native = public_probabilities(state, features)
    return features if interface == 'F' else probabilities, probabilities, native


def candidate_input(releases, metadata, pool, mask):
    """Literal projection before fitted preprocessing; canonical memory layout."""
    x = releases[metadata['space']+'/'+metadata['view']+'/'+pool][mask]
    columns = metadata.get('projection_columns')
    if columns is not None: x = x[:, columns]
    return np.ascontiguousarray(x)


def expected_pool(candidates, scope):
    result = []
    for cid, metadata in candidates.items():
        if 'saved_adversary' in cid: continue
        if scope != 'expanded_catchup' and 'catchup' in cid: continue
        if scope == 'standard_independent' and metadata['space'] != 'wire': continue
        result.append(cid)
    return sorted(result)


def check_curves(path, metadata, n, expected_seed, budget, report, terminals):
    if 'validation_curve' not in metadata: return
    curve = metadata['validation_curve']; epochs = 40 if budget is None else budget
    assert metadata['parameters']['epochs'] == epochs
    assert [x['epoch'] for x in curve] == list(range(0, epochs+1, 5))
    steps = int(np.ceil(n/256))
    assert all(x['optimizer_steps'] == steps*x['epoch'] for x in curve)
    best = min(curve, key=lambda x: (x['validation_log_loss'], x['epoch']))
    assert metadata['selected_epoch'] == best['epoch']
    assert metadata['validation_scores']['log_loss'] == best['validation_log_loss']
    assert metadata['optimizer_steps'] == steps*epochs
    assert metadata['training_row_exposures'] == n*epochs
    assert metadata['schedule_seed'] == expected_seed+700000
    check.replay_schedule(metadata, n); report['mlp_schedules'] += 1
    if metadata.get('audit_kind') != 'catchup':
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(expected_seed)
            model = torch.nn.Sequential(torch.nn.Linear(metadata['input_dim'],64),torch.nn.ReLU(),
                torch.nn.Linear(64,32),torch.nn.ReLU(),torch.nn.Linear(32,metadata['n_classes']))
        assert check.state_hash(model.state_dict()) == metadata['initial_state_hash']
    if budget is not None:
        terminal = path.parent.parent/f'last_training_{path.name}_epoch{budget}.pt'
        terminals[str(terminal.relative_to(ROOT))] = checkpoint_evidence(terminal, metadata, n)
        report['actual_terminal_checkpoints'] += 1
        if budget == 360:
            lower = check.read(path.parent.parent/'nested120'/path.name/'metadata.json')
            assert curve[:len(lower['validation_curve'])] == lower['validation_curve']
            assert metadata['validation_scores']['log_loss'] <= lower['validation_scores']['log_loss']
            report['nested_trajectory_prefixes'] += 1


def verify(out, seeds=None):
    tick = time.perf_counter()
    check.errors.clear(); check.max_error, check.max_error_path, check.numeric_comparisons = 0., None, 0
    cfg = check.read(out/'config.json'); freeze = check.read(out/'protocol_freeze.json')
    expected_source = freeze['sha256'].copy(); amendments = {}
    amendment_path = out/'EXECUTION_AMENDMENTS.json'
    if amendment_path.exists():
        ledger = check.read(amendment_path)
        for name,value in ledger.get('historical_metadata_anchor_hashes',{}).items():
            assert check.sha(ROOT/name) == value
            original = subprocess.run(['git','show',ledger['historical_metadata_anchor_commit']+':'+name],cwd=ROOT,check=True,capture_output=True).stdout
            assert hashlib.sha256(original).hexdigest() == value
        for amendment in ledger['amendments']:
            for name,change in amendment['source_changes'].items():
                assert expected_source[name] == change['before_sha256']
                assert check.sha(ROOT/change['preserved_source_path']) == change['before_sha256']
                expected_source[name] = change['after_sha256']
        amendments[amendment_path.name] = check.sha(amendment_path)
    for name,value in expected_source.items(): assert check.sha(ROOT/name) == value, ('execution source',name)
    for field in ('reference_record_hashes','required_local_reference_hashes'):
        for name, value in freeze[field].items(): assert check.sha(ROOT/name) == value, (field,name)
    assert freeze['prefit_identity_sha256'] == check.sha(out/'PREFIT_IDENTITY.json')
    parent = ROOT/cfg['parent_results']; pcfg = check.read(parent/'config.json'); raw_path = ROOT/pcfg['raw_path']
    assert check.sha(raw_path) == check.read(parent/'schema_support.json')['raw_sha256']
    raw = pd.read_csv(raw_path,usecols=['MIG','JWMNP','PINCP','ESR','PUBCOV','SEX','RAC1P','PWGTP'],low_memory=False)
    available = [s for s in cfg['seeds'] if (out/f'seed_{s}/complete.json').exists()]
    seeds = available if seeds is None else seeds
    assert seeds and len(set(seeds)) == len(seeds) and not set(seeds)-set(available)
    report = {'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'Independent frozen paired models, all legal audit paths, public composition, exact projections, validation-only pools, nested checkpoints and raw-label metrics; no fitting',
        'script_sha256':check.sha(__file__),'metric_helper_sha256':check.sha(check.__file__),
        'checkpoint_helper_sha256':check.sha(ROOT/'scripts/verify_acs_preservation_extended.py'),
        'protocol_freeze_sha256':check.sha(out/'protocol_freeze.json'),'execution_amendments_sha256':amendments,'raw_sha256':check.sha(raw_path),
        'evaluation_status':cfg['evaluation_status'],'complete_matrix':seeds == [0,1,2],'seeds':seeds,
        'absolute_metric_tolerance':check.TOL,'candidate_records':0,'score_dictionaries':0,'prediction_sets':0,
        'unique_saved_candidate_paths':0,'selection_pools':0,'inherited_prediction_sets':0,'literal_release_arrays':0,
        'public_head_composition_arrays':0,'native_score_dictionaries':0,'native_predictions':0,
        'mlp_schedules':0,'actual_terminal_checkpoints':0,'nested_trajectory_prefixes':0,
        'saved_observer_fidelity':0,'inherited_exposure_checks':0,'history_prediction_sets':0,
        'local_artifact_hashes':0,'seed_evidence':{},'new_trajectory_counts':{},'limitations':['Optimizer state/counters/RNG inspected without refitting.',
            'Deterministic duplicates add no values; no changed model versions, access noise, or privacy theorem assessed.',
            'Absent race fitting categories remain unassessable.']}
    cache, score_cache, seen = {}, {}, set()
    for seed in seeds:
        directory = out/f'seed_{seed}'; completion = check.read(directory/'complete.json')
        for name, h in completion['compact_sha256'].items(): assert check.sha(directory/name) == h
        assert completion['local_artifact_manifest_sha256'] == check.sha(directory/'local_artifacts.json')
        for entry in check.read(directory/'local_artifacts.json'):
            assert check.sha(ROOT/entry['path']) == entry['sha256']; report['local_artifact_hashes'] += 1
        if (directory/'recovery.json').exists():
            recovery = check.read(directory/'recovery.json')
            assert recovery['models_retrained'] == recovery['completed_candidate_fits_retrained'] == 0
            assert recovery['outputs_match_original_freeze']
            assert recovery['amendments_sha256'] == check.sha(out/'EXECUTION_AMENDMENTS.json')
            old_files = check.read(out/'amendments/operational_001/PRESERVED_BEFORE_RECOVERY.json')['files_sha256']
            assert len(old_files) == recovery['prior_artifacts_preserved']
            assert all(check.sha(ROOT/name) == value for name,value in old_files.items())
        provenance = check.read(directory/'parent_provenance.json')
        for name,h in provenance['used_reference_files_sha256'].items(): assert check.sha(ROOT/name) == h
        with np.load(directory/'split_rows.npz') as z: rows = {p:z[p].copy() for p in z.files}
        with np.load(parent/f'seed_{seed}/split_rows.npz') as z:
            assert set(rows) == set(z.files) and all(np.array_equal(v,z[p]) for p,v in rows.items())
        frames = {p:raw.iloc[ix] for p,ix in rows.items()}
        label = {(p,t):check.labels(f,t) for p,f in frames.items() for t in TARGETS}
        with np.load(ROOT/cfg['reference_results']/f'seed_{seed}/release_E_pca.npz') as z: pca = {p:z[p].copy() for p in z.files}
        emap = load_map(ROOT/cfg['selective_reference_results']/f'static/seed_{seed}/teachers/map_E.npz')
        release_freeze = check.read(directory/'release_freeze.json')
        assert release_freeze['all_six_pairs_complete'] and not release_freeze['reserved_labels_received']
        assert release_freeze['teacher_E_hash'] == emap['metadata']['map_sha256']
        assert release_freeze['protocol_freeze_sha256'] == check.sha(out/'protocol_freeze.json')
        selection_freeze = check.read(directory/'selection_before_test.json')
        for name,h in selection_freeze['condition_selections_sha256'].items(): assert check.sha(directory/name/'selection_before_test.json') == h
        assert selection_freeze['indices_sha256'] == check.sha(directory/'indices.json')
        indices = check.read(directory/'indices.json'); fit_indices = {}
        for kind, pool, order, cap, base in [('utility','downstream_fit',TASKS,2048,1230000),('audit','attacker_fit',TARGETS,4096,1240000)]:
            for j,target in enumerate(order):
                y,valid = label[pool,target]; ix = np.random.default_rng(base+100*seed+j).permutation(np.flatnonzero(valid))[:cap]
                fit_indices[kind,target] = ix
                record = indices['task_fit_indices' if kind == 'utility' else 'audit_fit_indices'][target]
                assert record == {'n':len(ix),'pool_indices_sha256':check.array_hash(ix),'raw_rows_sha256':check.array_hash(rows[pool][ix])}
        terminals, states = {}, {}
        for name in ('I','W',*CONDITIONS[:-1]):
            path = directory/'training'/({'I':'initialization.pt','W':'warm_base.pt'}.get(name,name+'/final.pt'))
            states[name] = torch.load(path,map_location='cpu',weights_only=True)
        for condition in CONDITIONS:
            dest = directory/condition
            with np.load(dest/'releases.npz') as z: releases = {p:z[p].copy() for p in z.files}
            for pool, x in pca.items():
                if condition == 'E':
                    e = map_probability_free(emap,x[:,:16]); wires = {'A':e,'B':e,'AB':np.concatenate((e,e),1)}; derived = None
                else:
                    state = states[condition]['model_state']
                    assert check.state_hash(state) == release_freeze['model_hashes'][condition]
                    assert check.state_hash(states[condition]['adversary_state']) == release_freeze['observer_hashes'][condition]
                    wires, derived, _ = literal_wires(state,x,condition[0])
                for space, values in [('wire',wires),('derived',derived)]:
                    if values is None: continue
                    for view, value in values.items():
                        np.testing.assert_array_equal(value,releases[space+'/'+view+'/'+pool]); report['literal_release_arrays'] += 1
                        if pool != 'test': assert check.array_hash(value) == release_freeze['outputs'][condition][space][view][pool]
                if condition.startswith('F'):
                    composed,_ = public_probabilities(state,{v:releases['wire/'+v+'/'+pool] for v in ('A','B')})
                    for view in ('A','B','AB'):
                        np.testing.assert_array_equal(composed[view],releases['derived/'+view+'/'+pool]); report['public_head_composition_arrays'] += 1
            selected = check.read(dest/'selection_before_test.json'); audits = check.read(dest/'audits/audit_selection.json')
            assert selected['release_freeze_sha256'] == check.sha(directory/'release_freeze.json')
            assert release_freeze['created_utc'] <= selected['created_utc'] <= selection_freeze['created_utc'] <= completion['completed_utc']
            assert not audits['development_received'] and audits['interface'] == condition[0]
            report['new_trajectory_counts'][str(seed)+'/'+condition] = audits['counts']
            expected_roles_count = 22 if condition.startswith('F') else (9 if condition == 'E' else 11)
            assert audits['counts']['new_five_candidate_roles'] == expected_roles_count
            assert audits['counts']['new_fresh_mlp_trajectories'] == expected_roles_count*2
            assert audits['counts']['own_catchup_trajectories'] == (0 if condition == 'E' else 9)
            assert audits['counts']['reused_five_candidate_roles'] == (2 if condition == 'E' else 0)
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
                    if condition != 'E' and target not in ('same_residence','commute_over20'): own |= {'saved_adversary','catchup'}
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
                        path = Path(meta['base_candidate_directory']); base_meta = check.read(path/'metadata.json')
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
                        path = (ROOT/cfg['selective_reference_results']/f'static/seed_{seed}/fitted/transfer/E_direct'/target/cid if condition == 'E' else dest/'fitted/utility'/view/target/cid)
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
                            original = {k.removeprefix(prefix):v for k,v in states[condition]['adversary_state'].items() if k.startswith(prefix)}
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
        verify_native(directory,states,pca,frames,report)
        verify_controls(directory,seed,frames,fit_indices,report,cache,score_cache,terminals)
        report['seed_evidence'][str(seed)] = {'complete_sha256':check.sha(directory/'complete.json'),'actual_training_checkpoints':terminals}
    report.update(passed=not check.errors,errors=check.errors,numeric_values_compared=check.numeric_comparisons,
        max_absolute_metric_error=check.max_error,max_error_path=check.max_error_path,runtime_seconds=time.perf_counter()-tick)
    return report


def verify_native(directory,states,pca,frames,report):
    recorded = check.read(directory/'native_source.json')['raw_metrics']
    with np.load(directory/'native_predictions.npz') as predictions:
        assert len(predictions.files) == len(recorded) == len(states)*6
        for row in recorded:
            name,target,pool = row['condition'],row['target'],row['split']
            assert row['view'] == ('B' if target == 'public_coverage' else 'A') and row['selection'] == 'fixed native head'
            _,_,native = literal_wires(states[name]['model_state'],pca[pool],'F'); p = native[target]
            np.testing.assert_array_equal(p,predictions[name+'/'+target+'/'+pool]); assert check.array_hash(p) == row['prediction_sha256']
            y,mask = check.labels(frames[pool],target)
            for field,weight in [('score',None),('person_weighted',frames[pool].PWGTP.to_numpy(float)[mask])]:
                check.compare(check.independent_scores(y[mask],p[mask],2,weight),row[field],f'{directory.name}/native/{name}/{target}/{pool}/{field}'); report['native_score_dictionaries'] += 1
            report['native_predictions'] += 1


def verify_controls(directory,seed,frames,fit_indices,report,cache,score_cache,terminals):
    dest = directory/'controls'; selected = check.read(dest/'selection_before_test.json'); rows = check.read(dest/'metrics.json')['raw_metrics']
    assert len(rows) == 77
    assert {(r['target'],r['audit_budget'],r['candidate_id']) for r in rows} == {(t,b,c) for t in TARGETS for b in (120,360) for c in FRESH}|{(t,None,'prior') for t in TARGETS}
    with np.load(dest/'predictions.npz') as predictions:
        for row in rows:
            target,cid,budget = row['target'],row['candidate_id'],row['audit_budget']; k = 9 if target == 'RAC1P' else 2
            yf,mf = check.labels(frames['attacker_fit'],target); ix = fit_indices['audit',target]
            if cid == 'prior':
                path = dest/'fitted/prior'/target; meta = check.read(path/'metadata.json')
                assert meta == selected['prior_metadata'][target] and meta['pseudocount_per_class'] == 1.
                counts = np.bincount(yf[ix],minlength=k); probabilities = (counts+1.)/(len(ix)+k)
                assert meta['fit_support'] == counts.tolist() and meta['fit_label_hash'] == check.array_hash(yf[ix].astype(np.int64))
                assert row['selected_scopes'] == ['prior'] and meta['probabilities'] == probabilities.tolist()
                with np.load(path/'prior.npz') as z: np.testing.assert_array_equal(z['probabilities'],probabilities)
                predict = lambda x: np.broadcast_to(probabilities,(len(x),k)).copy()
            else:
                path = Path(selected['paths'][target][str(budget)][cid])
                meta = check.read(path/'metadata.json'); predict,mean,scale,_ = check.load_inference(path,meta)
                xf = np.eye(k)[yf[ix]]
                assert meta['fit_hashes'] == {'x':check.array_hash(xf),'y':check.array_hash(yf[ix].astype(np.int64))}
                yv,mv = check.labels(frames['attacker_validation'],target)
                assert meta['validation_hashes'] == {'x':check.array_hash(np.eye(k)[yv[mv]]),'y':check.array_hash(yv[mv].astype(np.int64))}
                check.compare(meta['validation_scores'],row['scores']['validation'],f'{directory.name}/control/{target}/{budget}/{cid}/validation_metadata')
                np.testing.assert_array_equal(mean,xf.mean(0)); std = xf.std(0); np.testing.assert_array_equal(scale,np.where(std>1e-12,std,1.))
                expected_seed = 1260000+100*seed+TARGETS.index(target)+(10000 if cid == 'mlp_1' else 0)
                assert meta['seed'] == expected_seed
                check_curves(path,meta,len(ix),expected_seed,budget,report,terminals)
                report['unique_saved_candidate_paths'] += 1
                candidates = {c:check.read(Path(p)/'metadata.json') for c,p in selected['paths'][target][str(budget)].items()}
                choice = min(candidates,key=lambda c:(candidates[c]['validation_scores']['log_loss'],c))
                assert choice == selected['selection'][str(budget)][target]
                assert row['selected_scopes'] == (['exposed'] if choice == cid else [])
            for split,pool in [('validation','attacker_validation'),('test','test')]:
                y,mask = check.labels(frames[pool],target); x = np.eye(k)[y[mask]]
                p = predictions[f'audit/control/{target}/{budget}/{cid}/{split}']
                np.testing.assert_array_equal(predict(x),p); report['prediction_sets'] += 1
                for suffix,weight in [('',None),('_person_weighted',frames[pool].PWGTP.to_numpy(float)[mask])]:
                    check.compare(check.independent_scores(y[mask],p,k,weight),row['scores'][split+suffix],f'{directory.name}/control/{target}/{budget}/{cid}/{split}{suffix}'); report['score_dictionaries'] += 1
            report['candidate_records'] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/redesign_20260908_acs_coalition_v1')
    parser.add_argument('--report',type=Path)
    parser.add_argument('--seeds',type=int,nargs='+')
    args = parser.parse_args()
    if args.report and args.report.exists(): raise FileExistsError('Preserve completed verification evidence')
    torch.set_num_threads(1)
    with threadpool_limits(limits=1): result = verify(args.out.resolve(),args.seeds)
    rendered = json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.report:
        with args.report.open('x') as handle: handle.write(rendered)
    else: print(rendered,end='')
    if not result['passed']: raise SystemExit(1)


if __name__ == '__main__': main()
