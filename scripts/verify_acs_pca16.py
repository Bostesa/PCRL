"""Replay only new PCA16 predictions/models; no fitting or historical audit rerun.

Uses published independent metric/inference helpers from the bottleneck replay.
Requires local original PCA map, cached arrays, raw records, and new fitted files.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import verify_acs_bottleneck_scores as check


def verify(out):
    started = time.perf_counter()
    cfg,freeze = check.read(out/'config.json'),check.read(out/'protocol_freeze.json')
    for group in ('sha256','reference_records'):
        for path,digest in freeze[group].items():
            assert check.sha(ROOT/path) == digest, path
    parent,cache,reference = (ROOT/cfg[k] for k in ('parent_results','pca_cache_results','reference_results'))
    raw_path = ROOT/check.read(parent/'config.json')['raw_path']
    assert check.sha(raw_path) == check.read(parent/'schema_support.json')['raw_sha256']
    raw = pd.read_csv(raw_path,usecols=['MIG','JWMNP','PINCP','ESR','PUBCOV','SEX','RAC1P','PWGTP'])
    report = {'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'Only new PCA16 scores/models and referenced file identities; no fitting or historical score rerun.',
        'evaluation_status':cfg['evaluation_status'],'script_sha256':check.sha(__file__),
        'helper_sha256':check.sha(ROOT/'scripts/verify_acs_bottleneck_scores.py'),
        'candidate_records':0,'score_dictionaries':0,'prediction_sets_replayed':0,
        'pca_component_orders_checked':0,'exact_slices_checked':0,'standardizers_checked':0,
        'primary_selections_checked':0,'family_selections_checked':0,'mlp_curves_schedules_checked':0,
        'reference_file_hashes_checked':len(freeze['reference_records']),'seeds':{}}
    for seed in cfg['seeds']:
        directory = out/f'seed_{seed}'
        measured = check.read(directory/'metrics.json')
        selection = check.read(directory/'selection_before_test.json')
        released = check.read(directory/'release_freeze.json')
        prior_selection = check.read(reference/f'seed_{seed}/selection_before_test.json')
        provenance = check.read(directory/'provenance.json')
        assert check.sha(ROOT/provenance['reference_metrics_path']) == provenance['reference_metrics_sha256']
        for path,digest in provenance['used_local_artifacts_sha256'].items():
            assert check.sha(ROOT/path) == digest
            report['reference_file_hashes_checked'] += 1
        for item in check.read(directory/'local_artifacts.json'):
            assert check.sha(ROOT/item['path']) == item['sha256']
        assert check.sha(directory/'release_freeze.json') == selection['release_freeze_sha256']
        assert check.sha(out/'protocol_freeze.json') == selection['protocol_freeze_sha256']
        assert selection['created_utc'] <= measured['integrity']['evaluation_started_utc']
        assert all(v for k,v in measured['integrity'].items() if k.endswith('_unchanged'))
        pca = joblib.load(parent/f'seed_{seed}/release_maps.joblib')['pca']
        assert joblib.hash(pca) == released['pca_object_hash'] and not pca.whiten
        assert pca.n_components_ == 32 and pca.svd_solver == 'full'
        assert check.array_hash(pca.components_) == released['components_sha256']
        assert [check.array_hash(row) for row in pca.components_] == released['component_row_sha256']
        assert check.array_hash(pca.mean_) == released['original_mean_sha256']
        assert (np.diff(pca.explained_variance_)<=0).all() and released['component_indices'] == list(range(16))
        report['pca_component_orders_checked'] += 1
        rows,arrays,pca32 = (dict(np.load(p)) for p in (directory/'split_rows.npz',directory/'release_PCA16.npz',cache/f'seed_{seed}/release_E_pca.npz'))
        old_rows = dict(np.load(reference/f'seed_{seed}/split_rows.npz'))
        assert set(rows) == set(old_rows) and all(np.array_equal(v,old_rows[p]) for p,v in rows.items())
        for pool,x in arrays.items():
            assert x.dtype == np.float32 and np.array_equal(x,pca32[pool][:,:16])
            assert check.array_hash(x) == (measured['integrity']['test_output_sha256'] if pool=='test' else released['output_hashes'][pool])
            report['exact_slices_checked'] += 1
        frames = {pool:raw.iloc[idx] for pool,idx in rows.items()}
        probabilities = dict(np.load(directory/'predictions.npz'))
        assert len(measured['raw_metrics']) == 20 and len(probabilities) == 40
        for row in measured['raw_metrics']:
            role,target,cid = (row[k] for k in ('role','target','candidate_id'))
            key = f'{role}/PCA16/{target}';k = 9 if target=='RAC1P' else 2
            for split in ('validation','test'):
                pool = 'test' if split=='test' else 'downstream_validation' if role=='transfer' else 'attacker_validation'
                y,valid = check.labels(frames[pool],target)
                p = probabilities[f'{split}/{key}/{cid}']
                for suffix,weight in (('',None),('_person_weighted',frames[pool].PWGTP.to_numpy(float)[valid])):
                    check.compare(check.independent_scores(y[valid],p,k,weight),row[split+suffix],f'{seed}/{key}/{cid}/{split}{suffix}')
                    report['score_dictionaries'] += 1
            assert row['selected'] == row['independent_selected'] == (selection['head_selections'][key]==cid)
            assert row['selected_within_family'] == (selection['family_selections'][key][row['family']]==cid)
            assert row['auc_selected'] == (selection['auroc_selections'][key]==cid)
            check.compare(selection['fitting_records'][key]['candidates'][cid]['validation_scores'],row['validation'],f'{seed}/{key}/{cid}/saved_validation_score')
            report['candidate_records'] += 1
        for key,record in selection['fitting_records'].items():
            role,_,target = key.split('/');candidates = record['candidates']
            assert record == check.read(directory/'fitted'/key/'selection.json')
            assert selection['head_selections'][key] == check.primary(candidates,list(candidates))
            assert record['selected_family'] == selection['head_selections'][key]
            expected_auc = check.primary(candidates,list(candidates),auc=True) if role=='audit' else None
            assert selection['auroc_selections'][key] == expected_auc
            report['primary_selections_checked'] += 1
            for family,cid in selection['family_selections'][key].items():
                assert cid == check.primary(candidates,[c for c in candidates if candidates[c]['family']==family])
                report['family_selections_checked'] += 1
            pool,vpool = ('downstream_fit','downstream_validation') if role=='transfer' else ('attacker_fit','attacker_validation')
            y,valid = check.labels(frames[pool],target);yv,vv = check.labels(frames[vpool],target)
            targets = ['same_residence','commute_over20','income_binary','civilian_at_work','public_coverage'] if role=='transfer' else ['SEX','RAC1P']
            fit_seed = (1230000 if role=='transfer' else 1240000)+100*seed+targets.index(target)
            indices = np.random.default_rng(fit_seed).permutation(np.flatnonzero(valid))[:2048 if role=='transfer' else 4096]
            field = 'task_fit_indices' if role=='transfer' else 'audit_fit_indices'
            assert selection[field][target] == prior_selection[field][target]
            assert selection[field][target]['pool_indices_sha256'] == check.array_hash(indices)
            xfit,xval = arrays[pool][indices].astype(np.float64),arrays[vpool][vv].astype(np.float64)
            for cid,candidate in candidates.items():
                path = directory/'fitted'/key/cid
                assert check.read(path/'metadata.json') == candidate
                predict,mean,scale,_ = check.load_inference(path,candidate)
                assert candidate['fit_hashes'] == {'x':check.array_hash(xfit),'y':check.array_hash(y[indices].astype(np.int64))}
                assert candidate['validation_hashes'] == {'x':check.array_hash(xval),'y':check.array_hash(yv[vv].astype(np.int64))}
                np.testing.assert_array_equal(mean,xfit.mean(0));std=xfit.std(0)
                np.testing.assert_array_equal(scale,np.where(std>1e-12,std,1.))
                assert candidate['fit_support'] == np.bincount(y[indices],minlength=candidate['n_classes']).tolist()
                report['standardizers_checked'] += 1
                for split,p in (('validation',vpool),('test','test')):
                    _,mask = check.labels(frames[p],target)
                    assert np.array_equal(predict(arrays[p][mask]),probabilities[f'{split}/{key}/{cid}'])
                    report['prediction_sets_replayed'] += 1
                if 'validation_curve' in candidate:
                    epochs = 40 if role=='transfer' else 120;curve=candidate['validation_curve']
                    assert [c['epoch'] for c in curve] == list(range(0,epochs+1,5))
                    best = min(curve,key=lambda c:(c['validation_log_loss'],c['epoch']))
                    assert candidate['selected_epoch']==best['epoch'] and candidate['selected_optimizer_steps']==best['optimizer_steps']
                    assert candidate['validation_scores']['log_loss']==best['validation_log_loss']
                    assert candidate['optimizer_steps'] == int(np.ceil(len(indices)/256))*epochs
                    assert candidate['training_row_exposures'] == len(indices)*epochs
                    check.replay_schedule(candidate,len(indices))
                    prior = prior_selection['fitting_records'][f'{role}/C_bottleneck/{target}']['candidates'][cid]
                    assert all(candidate[f]==prior[f] for f in ('initial_state_hash','schedule_hash','optimizer_steps','training_row_exposures'))
                    report['mlp_curves_schedules_checked'] += 1
        race = [row for row in measured['raw_metrics'] if row['target']=='RAC1P'][0]
        report['seeds'][str(seed)] = {'metrics_sha256':check.sha(directory/'metrics.json'),
            'race_support':{s:race[s]['support'] for s in ('validation','test')},
            'race_coverage':{s:race[s]['coverage_complete'] for s in ('validation','test')}}
    report.update(passed=not check.errors,errors=check.errors,max_absolute_metric_error=check.max_error,
        numeric_values_compared=check.numeric_comparisons,runtime_seconds=time.perf_counter()-started)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/redesign_20260908_acs_pca16_v1')
    parser.add_argument('--report',type=Path,help='Fresh report; default stdout; no overwrites')
    args=parser.parse_args()
    if args.report and args.report.exists():raise FileExistsError('Preserve existing evidence; use a fresh report')
    torch.set_num_threads(1);result=verify(args.out.resolve());text=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.report:
        with args.report.open('x') as handle:handle.write(text)
    else:print(text,end='')
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
