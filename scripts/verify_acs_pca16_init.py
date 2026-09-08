"""Verify new PCA16-initialized checkpoints, fixed snapshots, and new scores.

No fitting or historical scientific inference is repeated. Historical manifests,
metadata and the original initialization tensors establish the reference. The
published independent metric/inference helpers replay only this stage's models.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import verify_acs_bottleneck_scores as check
from scripts.verify_acs_bottleneck_artifacts import adam_steps
from experiments.acs_bottleneck_training import tree_digest

ARMS={'C_init':'C_bottleneck','D_init':'D_protected'}
SNAPSHOTS={'I':'initialization.pt','W':'warm_base.pt'}


def load(path):
    return torch.load(path,map_location='cpu',weights_only=True)


def map_output(state,x):
    with torch.no_grad():
        value=((x.astype(np.float64)-state['input_mean'].numpy())/state['input_scale'].numpy()).astype(np.float32)
        value=torch.relu(torch.nn.functional.linear(torch.from_numpy(value),state['mapper.0.weight'],state['mapper.0.bias']))
        return torch.nn.functional.linear(value,state['mapper.2.weight'],state['mapper.2.bias']).numpy()


def state_checks(directory,historical,pca,arrays,meta,freeze):
    train=directory/'training';old_meta=check.read(historical/'training/training.json')
    initial,base,warm=(load(train/name) for name in ('initialization.pt','warm_base.pt','warm_adversary.pt'))
    old_initial=load(historical/'training/initialization.pt')['model_state']
    state=initial['model_state'];n=len(pca['representation_fit']);per_epoch=int(np.ceil(n/256))
    assert initial['counters']=={'epoch':0,'mapper_optimizer_steps':0,'adversary_optimizer_steps':0}
    assert not initial['mapper_optimizer_state']['state'] and initial['adversary_state']=={}
    assert initial['adversary_optimizer_state'] is None
    assert set(state)==set(old_initial)
    nonmapper=[k for k in state if not k.startswith('mapper.')]
    assert all(torch.equal(state[k],old_initial[k]) for k in nonmapper)
    identity=meta['initialization'];assert identity['mode']=='pca16'
    assert identity['torch_rng_unchanged_by_overwrite'] and identity['nonmapper_unchanged_by_overwrite']
    assert identity['nonmapper_initial_state_sha256']==check.state_hash({k:state[k] for k in nonmapper})
    assert identity['all_mapper_parameters_trainable']
    np.testing.assert_array_equal(state['mapper.0.weight'].numpy(),np.concatenate([np.eye(32),-np.eye(32)]).astype(np.float32))
    np.testing.assert_array_equal(state['mapper.0.bias'].numpy(),np.zeros(64))
    readout=np.diag(state['input_scale'].numpy().astype(np.float32))[:16]
    np.testing.assert_array_equal(state['mapper.2.weight'].numpy(),np.concatenate([readout,-readout],axis=1))
    np.testing.assert_array_equal(state['mapper.2.bias'].numpy(),state['input_mean'].numpy()[:16].astype(np.float32))
    fit=pca['representation_fit'].astype(np.float64);std=fit.std(0)
    np.testing.assert_array_equal(state['input_mean'].numpy(),fit.mean(0))
    np.testing.assert_array_equal(state['input_scale'].numpy(),np.where(std>1e-12,std,1.))
    for field in ('config','preprocessing','schedules','adversary_initialization_hash','source_label_hashes',
                  'attribute_label_hashes','source_validation_label_hashes','fit_input_sha256',
                  'source_validation_input_sha256','common_optimizer_counts'):
        assert meta[field]==old_meta[field],('historical recipe',field)
    for phase,epochs,offset in (('warm_base',60,0),('warm_adversary',20,100),('continuation',80,200)):
        schedule=meta['schedules'][phase];assert schedule['seed']==1280000+100*meta['seed']+offset
        rng=np.random.default_rng(schedule['seed']);digest=hashlib.sha256()
        for _ in range(epochs):digest.update(rng.permutation(n).tobytes())
        assert digest.hexdigest()==schedule['sha256']
    assert check.state_hash(base['model_state'])==check.state_hash(warm['model_state'])==meta['snapshots']['W']['model_hash']
    assert tree_digest(base['mapper_optimizer_state'])==tree_digest(warm['mapper_optimizer_state'])
    assert meta['snapshots']['W']['unchanged_across_adversary_warmup']
    assert check.state_hash(state)==meta['snapshots']['I']['model_hash']
    adam_steps(base['mapper_optimizer_state'],60*per_epoch)
    adam_steps(warm['adversary_optimizer_state'],20*per_epoch)
    assert meta['common_base_row_exposures']==n*60 and meta['common_adversary_row_exposures']==n*20
    unused=identity['unused_readout_columns'];gradient=identity['unused_readout_gradient_check']
    assert gradient['disposable_model'] and gradient['actual_model_unchanged'] and gradient['optimizer_steps']==0
    assert gradient['base_unused_readout_gradient_l2']>0 and gradient['base_unused_readout_nonzero_entries']>0
    assert torch.count_nonzero(base['model_state']['mapper.2.weight'][:,unused])>0
    forks={a:load(train/old/'fork.pt') for a,old in ARMS.items()}
    assert tree_digest(forks['C_init'])==tree_digest(forks['D_init'])
    checkpoints={s:load(train/path) for s,path in SNAPSHOTS.items()}
    for arm,old in ARMS.items():
        checkpoint=load(train/old/'final.pt');checkpoints[arm]=checkpoint
        arm_meta=meta['arms'][old]
        for field in ('schedule_hash','continuation_mapper_optimizer_steps','continuation_adversary_optimizer_steps',
                      'optimizer_counts_including_common','mapper_row_exposures','adversary_row_exposures'):
            assert arm_meta[field]==old_meta['arms'][old][field]
        assert arm_meta['fork_hashes']==meta['shared_fork_hashes']
        for field,part in (('model_state','model'),('adversary_state','adversaries'),('mapper_optimizer_state','mapper_optimizer'),('adversary_optimizer_state','adversary_optimizer')):
            digest=tree_digest(forks[arm][field]) if 'optimizer' in field else check.state_hash(forks[arm][field])
            assert digest==meta['shared_fork_hashes'][part]
        expected={'mapper_optimizer_steps':140*per_epoch,'adversary_optimizer_steps':260*per_epoch,'continuation_epoch':80}
        assert checkpoint['counters']==expected and arm_meta['selected_epoch']==80
        adam_steps(checkpoint['mapper_optimizer_state'],140*per_epoch)
        adam_steps(checkpoint['adversary_optimizer_state'],260*per_epoch)
        for row in arm_meta['curve']:
            assert row['mapper_optimizer_steps']==(60+row['epoch'])*per_epoch
            assert row['adversary_optimizer_steps']==(20+3*row['epoch'])*per_epoch
        assert check.state_hash(checkpoint['model_state'])==arm_meta['final_model_hash']
        assert check.state_hash(checkpoint['adversary_state'])==arm_meta['final_adversary_hash']
        assert tree_digest(checkpoint['mapper_optimizer_state'])==arm_meta['final_mapper_optimizer_hash']
        assert tree_digest(checkpoint['adversary_optimizer_state'])==arm_meta['final_adversary_optimizer_hash']
        assert arm_meta['mapper_loss_uses_protection_gradient']==(arm=='D_init')
        assert (arm_meta['applied_protection_mapper_l2_at_shared_fork']>0)==(arm=='D_init')
    parity={}
    for arm,checkpoint in checkpoints.items():
        assert check.state_hash(checkpoint['model_state'])==freeze['learned_state'][arm]['model']
        if arm in ARMS:
            assert check.state_hash(checkpoint['adversary_state'])==freeze['learned_state'][arm]['adversaries']
        for pool,x in arrays[arm].items():
            np.testing.assert_array_equal(map_output(checkpoint['model_state'],pca[pool]),x)
            if arm=='I':
                target=pca[pool][:,:16];np.testing.assert_allclose(x,target,atol=1e-5,rtol=1e-5)
                parity[pool]={'max_absolute_error':float(np.abs(x.astype(np.float64)-target).max()),'atol':1e-5,'rtol':1e-5}
    return checkpoints,{'nonmapper_tensors_exact':len(nonmapper),'I_parity':parity,
        'W_unchanged_through_adversary_warmup':True,'exact_C_D_model_and_Adam_fork':True,
        'all_saved_Adam_steps_verified':True,'historical_training_recipe_and_schedules_match':True,
        'frozen_release_arrays_replayed':28,'unused_readout_updated_during_warmup':True}


def verify(out):
    started=time.perf_counter();cfg=check.read(out/'config.json');frozen=check.read(out/'protocol_freeze.json')
    for group in ('sha256','reference_record_hashes'):
        for path,digest in frozen[group].items():assert check.sha(ROOT/path)==digest,path
    parent,reference,historical,pca16=(ROOT/cfg[k] for k in ('parent_results','reference_results','bottleneck_reference_results','pca16_reference_results'))
    raw_path=ROOT/check.read(parent/'config.json')['raw_path']
    assert check.sha(raw_path)==check.read(parent/'schema_support.json')['raw_sha256']
    raw=pd.read_csv(raw_path,usecols=['MIG','JWMNP','PINCP','ESR','PUBCOV','SEX','RAC1P','PWGTP'])
    report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'New I/W/C_init/D_init checkpoints, snapshots, predictions and scores only; old initialization/metadata/manifests as references; no fitting or old scientific inference rerun.',
        'evaluation_status':cfg['evaluation_status'],'script_sha256':check.sha(__file__),
        'helper_sha256':{p:check.sha(ROOT/p) for p in ('scripts/verify_acs_bottleneck_scores.py','scripts/verify_acs_bottleneck_artifacts.py')},
        'new_candidate_records':0,'new_score_dictionaries':0,'new_prediction_sets_replayed':0,
        'unchanged_historical_records_checked':0,'primary_choices_checked':0,'independent_choices_checked':0,
        'family_choices_checked':0,'MLP_curves_schedules_checked':0,'fitting_standardizers_checked':0,
        'catchup_identity_coordinate_checks':0,'saved_adversary_fidelity_checks':0,'seeds':{}}
    for seed in cfg['seeds']:
        directory=out/f'seed_{seed}';hd=historical/f'seed_{seed}';p16dir=pca16/f'seed_{seed}'
        measured,selected,release_freeze,meta=(check.read(directory/path) for path in
            ('metrics.json','selection_before_test.json','release_freeze.json','training/training.json'))
        assert release_freeze['scientific_training_completed_utc']<=release_freeze['created_utc']<selected['created_utc']<=measured['integrity']['evaluation_started_utc']
        assert check.sha(directory/'selection_before_test.json')==measured['integrity']['selection_sha256']
        assert check.sha(directory/'release_freeze.json')==selected['release_freeze_sha256']
        assert check.sha(out/'protocol_freeze.json')==selected['protocol_freeze_sha256']
        assert all(v for k,v in measured['integrity'].items() if k.endswith('_unchanged'))
        for item in check.read(directory/'local_artifacts.json'):assert check.sha(ROOT/item['path'])==item['sha256']
        provenance=check.read(directory/'parent_provenance.json')
        for group in ('used_reference_files_sha256','used_original_files_sha256','additional_history_files_sha256'):
            for path,digest in provenance[group].items():assert check.sha(ROOT/path)==digest
        rows=dict(np.load(directory/'split_rows.npz'));old_rows=dict(np.load(hd/'split_rows.npz'))
        assert set(rows)==set(old_rows) and all(np.array_equal(v,old_rows[p]) for p,v in rows.items())
        frames={p:raw.iloc[i] for p,i in rows.items()};pca=dict(np.load(reference/f'seed_{seed}/release_E_pca.npz'))
        arrays={a:dict(np.load(directory/f'release_{a}.npz')) for a in (*SNAPSHOTS,*ARMS)}
        checkpoints,state_report=state_checks(directory,hd,pca,arrays,meta,release_freeze)
        with np.load(p16dir/'release_PCA16.npz') as cached:
            for pool in rows:np.testing.assert_allclose(arrays['I'][pool],cached[pool],atol=1e-5,rtol=1e-5)
        assert check.read(directory/'evaluation_parity.json')['selected_before_access']==measured['integrity']['selection_sha256']
        old_selection=check.read(hd/'selection_before_test.json')
        prior_records={}
        for path in (hd,p16dir,reference/f'seed_{seed}'):
            for r in check.read(path/'metrics.json')['raw_metrics']:
                prior_records[r['role'],r['release'],r['target'],r['candidate_id']]=r
        probabilities=dict(np.load(directory/'predictions.npz'));assert len(probabilities)==136
        new_rows=[r for r in measured['raw_metrics'] if not r['reused_reference']];assert len(new_rows)==68
        for row in measured['raw_metrics']:
            role,arm,target,cid=(row[k] for k in ('role','release','target','candidate_id'))
            if row['reused_reference']:
                old=prior_records[role,arm,target,cid]
                for field in old:
                    if field not in ('reused_reference','independent_selected'):assert row[field]==old[field]
                assert row['independent_selected']==old.get('independent_selected',old['selected'])
                report['unchanged_historical_records_checked']+=1;continue
            assert arm in arrays and (arm not in SNAPSHOTS or role=='transfer')
            key=f'{role}/{arm}/{target}';k=9 if target=='RAC1P' else 2
            for split in ('validation','test'):
                pool='test' if split=='test' else 'downstream_validation' if role=='transfer' else 'attacker_validation'
                y,mask=check.labels(frames[pool],target);p=probabilities[f'{split}/{key}/{cid}']
                for suffix,weight in (('',None),('_person_weighted',frames[pool].PWGTP.to_numpy(float)[mask])):
                    check.compare(check.independent_scores(y[mask],p,k,weight),row[split+suffix],f'{seed}/{key}/{cid}/{split}{suffix}')
                    report['new_score_dictionaries']+=1
            check.compare(selected['fitting_records'][key]['candidates'][cid]['validation_scores'],row['validation'],f'{seed}/{key}/{cid}/recorded_validation')
            assert row['selected']==(selected['head_selections'][key]==cid)
            assert row['independent_selected']==(selected['independent_selections'][key]==cid)
            assert row['auc_selected']==(selected['auroc_selections'][key]==cid)
            assert row['selected_within_family']==(selected['family_selections'][key][row['family']]==cid)
            report['new_candidate_records']+=1
        for key,record in selected['fitting_records'].items():
            role,arm,target=key.split('/')
            if arm not in arrays:continue
            assert record==check.read(directory/'fitted'/key/'selection.json')
            candidates=record['candidates'];eligible=[c for c in candidates if c!='saved_adversary'];independent=[c for c in eligible if c!='catchup']
            assert selected['head_selections'][key]==record['selected_family']==check.primary(candidates,eligible)
            assert selected['independent_selections'][key]==check.primary(candidates,independent)
            assert selected['auroc_selections'][key]==check.primary(candidates,eligible,auc=True)
            report['primary_choices_checked']+=1;report['independent_choices_checked']+=1
            family=lambda c:c if c in ('catchup','saved_adversary') else candidates[c]['family']
            for f,cid in selected['family_selections'][key].items():
                assert cid==check.primary(candidates,[c for c in candidates if family(c)==f]);report['family_choices_checked']+=1
            pool,vpool=('downstream_fit','downstream_validation') if role=='transfer' else ('attacker_fit','attacker_validation')
            yf,valid=check.labels(frames[pool],target);yv,vv=check.labels(frames[vpool],target)
            targets=cfg['utility_tasks'] if role=='transfer' else ['SEX','RAC1P']
            base=1230000 if role=='transfer' else 1240000;cap=2048 if role=='transfer' else 4096
            idx=np.random.default_rng(base+100*seed+targets.index(target)).permutation(np.flatnonzero(valid))[:cap]
            field='task_fit_indices' if role=='transfer' else 'audit_fit_indices'
            assert selected[field][target]==old_selection[field][target]
            assert selected[field][target]['pool_indices_sha256']==check.array_hash(idx)
            xf,xv=arrays[arm][pool][idx].astype(np.float64),arrays[arm][vpool][vv].astype(np.float64)
            for cid,candidate in candidates.items():
                path=directory/'fitted'/key/cid;assert check.read(path/'metadata.json')==candidate
                predict,mean,scale,network_state=check.load_inference(path,candidate)
                assert candidate['validation_hashes']=={'x':check.array_hash(xv),'y':check.array_hash(yv[vv].astype(np.int64))}
                if cid!='saved_adversary':
                    assert candidate['fit_hashes']=={'x':check.array_hash(xf),'y':check.array_hash(yf[idx].astype(np.int64))}
                    assert candidate['fit_support']==np.bincount(yf[idx],minlength=candidate['n_classes']).tolist()
                if cid in ('catchup','saved_adversary'):
                    np.testing.assert_array_equal(mean,np.zeros(16));np.testing.assert_array_equal(scale,np.ones(16))
                    saved_state={k.removeprefix(target+'.'):v for k,v in checkpoints[arm]['adversary_state'].items() if k.startswith(target+'.')}
                    assert candidate['source_state_hash']==check.state_hash(saved_state)
                    inherited=candidate['inherited_exposure'];ty,valid=check.labels(frames['representation_fit'],target)
                    assert inherited['training_label_sha256']==check.array_hash(np.where(valid,ty,-1).astype(np.int64))
                    assert inherited['total_row_passes_equivalent']==260 and inherited['total_row_exposures']==len(ty)*260
                    assert inherited['total_optimizer_steps']==int(np.ceil(len(ty)/256))*260
                    assert inherited['final_adversary_state_hash']==candidate['source_state_hash']
                    report['catchup_identity_coordinate_checks']+=1
                    if cid=='saved_adversary':
                        assert check.state_hash(network_state)==check.state_hash(saved_state)
                        assert candidate['optimizer_steps']==0 and candidate['fit_support'] is None
                        assert np.array_equal(predict(xv),check.network_probability(saved_state,xv))
                        report['saved_adversary_fidelity_checks']+=1
                    else:
                        assert candidate['initial_state_hash']==check.state_hash(saved_state)
                        assert candidate['optimizer']['initial_state_entries']==0 and not candidate['optimizer']['restored_state']
                        assert candidate['validation_curve'][0]['validation_log_loss']==candidates['saved_adversary']['validation_scores']['log_loss']
                else:
                    np.testing.assert_array_equal(mean,xf.mean(0));std=xf.std(0)
                    np.testing.assert_array_equal(scale,np.where(std>1e-12,std,1.));report['fitting_standardizers_checked']+=1
                for split,p in (('validation',vpool),('test','test')):
                    _,mask=check.labels(frames[p],target)
                    assert np.array_equal(predict(arrays[arm][p][mask]),probabilities[f'{split}/{key}/{cid}'])
                    report['new_prediction_sets_replayed']+=1
                if 'validation_curve' in candidate:
                    epochs=40 if role=='transfer' else 120;curve=candidate['validation_curve'];steps=int(np.ceil(len(idx)/256))
                    assert candidate['parameters']['epochs']==epochs and [r['epoch'] for r in curve]==list(range(0,epochs+1,5))
                    assert all(r['optimizer_steps']==steps*r['epoch'] for r in curve)
                    best=min(curve,key=lambda r:(r['validation_log_loss'],r['epoch']))
                    assert candidate['selected_epoch']==best['epoch'] and candidate['selected_optimizer_steps']==best['optimizer_steps']
                    assert candidate['validation_scores']['log_loss']==best['validation_log_loss']
                    assert candidate['optimizer_steps']==steps*epochs and candidate['training_row_exposures']==len(idx)*epochs
                    check.replay_schedule(candidate,len(idx))
                    old=old_selection['fitting_records'][f'{role}/C_bottleneck/{target}']['candidates'][cid]
                    assert all(candidate[f]==old[f] for f in ('schedule_hash','optimizer_steps','training_row_exposures'))
                    if cid!='catchup':assert candidate['initial_state_hash']==old['initial_state_hash']
                    report['MLP_curves_schedules_checked']+=1
        report['seeds'][str(seed)]={'states':state_report,'metrics_sha256':check.sha(directory/'metrics.json'),
            'selection_sha256':check.sha(directory/'selection_before_test.json'),'new_candidate_rows':68}
    report.update(passed=not check.errors,errors=check.errors,max_absolute_metric_error=check.max_error,
        numeric_values_compared=check.numeric_comparisons,runtime_seconds=time.perf_counter()-started)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/redesign_20260908_acs_pca16_init_v1')
    parser.add_argument('--report',type=Path,help='Fresh JSON file; default stdout; never overwritten')
    args=parser.parse_args()
    if args.report and args.report.exists():raise FileExistsError('Preserve historical reports; choose a fresh path')
    torch.set_num_threads(1);report=verify(args.out.resolve());rendered=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.report:
        with args.report.open('x') as handle:handle.write(rendered)
    else:print(rendered,end='')
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
