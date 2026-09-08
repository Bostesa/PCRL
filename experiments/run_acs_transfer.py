"""Bounded ACS source-task-to-withheld-task transfer; no protection training.

Run as a module. Each seed saves every downstream/attack selection before any
final-test release is extracted or scored. Large fitted objects stay local.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from experiments.acs_transfer_data import (
    AUDIT_NAMES, FEATURES, POOLS, SOURCE_KEYS, CovariatePreprocessor,
    array_hash, audit_labels, fit_income_edges, heldout_labels, load_cohort,
    require_test_selection, sha_file, source_labels, split_households, support, write_json,
)
from experiments.acs_transfer_heads import fit_candidates, fit_prior, metrics
from experiments.acs_transfer_models import fit_source_encoder, fit_source_tree_bank, source_scores, state_digest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'results/redesign_20260907_acs_transfer_v1'
TRANSFER = ('same_residence', 'commute_over20')
AUDITS = {'SEX': 2, 'RAC1P': 9}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def source_paths():
    return [ROOT / 'experiments' / name for name in (
        'run_acs_transfer.py', 'acs_transfer_data.py', 'acs_transfer_models.py', 'acs_transfer_heads.py')]


def frozen_hashes(out):
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha_file(p)
            for p in [out/'PROTOCOL.md', out/'config.json', *source_paths()]}


def verify_freeze(out):
    saved = json.loads((out/'protocol_freeze.json').read_text())
    current = frozen_hashes(out)
    if saved['sha256'] != current:
        raise RuntimeError('Frozen protocol/config/source changed; retain evidence and record explicit correction before running')
    if sha_file(out/'schema_support.json') != saved['schema_support_sha256']:
        raise RuntimeError('Frozen schema/support record changed')
    support_record=json.loads((out/'schema_support.json').read_text())
    if sha_file(ROOT/support_record['raw_path']) != support_record['raw_sha256']:
        raise RuntimeError('Raw ACS file differs from the frozen data identity')


def prepare(out, cfg):
    start = time.perf_counter()
    frame, cohort = load_cohort(ROOT/cfg['raw_path'], cfg['sample_cap'], cfg['sample_seed'])
    metadata = {'created_utc': now(), 'cohort': cohort,
                'raw_path': cfg['raw_path'], 'raw_sha256': sha_file(ROOT/cfg['raw_path']),
                'raw_bytes': (ROOT/cfg['raw_path']).stat().st_size,
                'feature_allowlist': list(FEATURES), 'audit_category_names': AUDIT_NAMES, 'seeds': {}}
    for seed in cfg['seeds']:
        pools = split_households(frame,seed)
        edge = fit_income_edges(frame.iloc[pools['representation_fit']])
        classes = {'income_binary':2,'income_bins':len(edge)+1,'esr':6,'pubcov':2,'joint':8}
        records = {}
        for pool, indices in pools.items():
            part = frame.iloc[indices]
            labels = {**source_labels(part,edge), **heldout_labels(part), **audit_labels(part)}
            records[pool] = {'persons':len(part),'households':int(part.SERIALNO.nunique()),
                             'raw_row_hash':array_hash(part._raw_row.to_numpy()),
                             'household_hash':array_hash(np.asarray(sorted(part.SERIALNO.unique()),dtype='U')),
                             'support':support(labels,{**classes,**{t:2 for t in TRANSFER},**AUDITS})}
        metadata['seeds'][str(seed)] = {'split_rng_seed':1210000+seed,'income_edges':edge.tolist(),
                                       'class_counts':classes,'pools':records}
    metadata['runtime_seconds'] = time.perf_counter()-start
    write_json(out/'schema_support.json',metadata)
    freeze = {'created_utc':now(),'sha256':frozen_hashes(out),
              'schema_support_sha256':sha_file(out/'schema_support.json'),
              'starting_commit':cfg['starting_commit'],
              'note':'Frozen before any ACS model comparison. Support checks examine validity/counts only.'}
    write_json(out/'protocol_freeze.json',freeze)
    return metadata


def subset_indices(labels, limit, seed):
    valid = np.flatnonzero(labels >= 0)
    return np.random.default_rng(seed).permutation(valid)[:limit]


def save_candidates(result, directory):
    for family,candidate in result['candidates'].items():
        candidate.save(directory/family)
    write_json(directory/'selection.json',result['metadata'])


def run_seed(out, cfg, seed, verify=True):
    if verify:
        if cfg != json.loads((out/'config.json').read_text()):
            raise RuntimeError('Supplied configuration differs from frozen config.json')
        verify_freeze(out)
    started = time.perf_counter()
    seed_dir = out/f'seed_{seed}'
    seed_dir.mkdir(exist_ok=False)
    torch.set_num_threads(cfg['threads'])
    frame, cohort = load_cohort(ROOT/cfg['raw_path'],cfg['sample_cap'],cfg['sample_seed'])
    pools = split_households(frame,seed)
    np.savez_compressed(seed_dir/'split_rows.npz',**{k:frame.iloc[v]._raw_row.to_numpy() for k,v in pools.items()})
    repr_frame,source_val_frame = (frame.iloc[pools[k]] for k in ('representation_fit','source_validation'))
    pre = CovariatePreprocessor().fit(repr_frame)
    write_json(seed_dir/'preprocessing.json',pre.metadata())
    edges = fit_income_edges(repr_frame)
    classes = {'income_binary':2,'income_bins':len(edges)+1,'esr':6,'pubcov':2,'joint':8}
    # Test rows are not transformed here. Only the fitting/validation pools enter development.
    x = {name:pre.transform(frame.iloc[indices]) for name,indices in pools.items() if name != 'test'}
    yfit,yval = source_labels(repr_frame,edges),source_labels(source_val_frame,edges)
    init_seed = 1220000+seed
    source_start = time.perf_counter()
    encoder_candidates=[]
    for lr in cfg['source_lr_grid']:
        print(f'seed {seed}: source encoder lr={lr}',flush=True)
        model,meta = fit_source_encoder(x['representation_fit'],yfit,x['source_validation'],yval,classes,
                                       seed=init_seed,lr=lr,epochs=cfg['source_epochs'],
                                       batch_size=cfg['source_batch_size'],out_dir=seed_dir/'source'/f'encoder_{lr}')
        encoder_candidates.append((model,meta,lr))
    model,model_meta,selected_lr = min(encoder_candidates,key=lambda z:(z[1]['validation_loss'],z[2]))
    encoder_time=time.perf_counter()-source_start
    tree_start=time.perf_counter()
    tree_candidates=[]
    for leaves in cfg['source_tree_leaf_grid']:
        print(f'seed {seed}: source tree leaves={leaves}',flush=True)
        tree,meta=fit_source_tree_bank(x['representation_fit'],yfit,x['source_validation'],yval,classes,
                                      seed=init_seed,max_leaf_nodes=leaves,max_iter=cfg['source_tree_iterations'],
                                      out_dir=seed_dir/'source'/f'tree_{leaves}')
        tree_candidates.append((tree,meta,leaves))
    tree,tree_meta,selected_leaves=min(tree_candidates,key=lambda z:(z[1]['validation_loss'],z[2]))
    tree_time=time.perf_counter()-tree_start
    joint_supported=all(np.bincount(y['joint'][y['joint']>=0],minlength=8).min()>0 for y in (yfit,yval))
    bank_keys=[k for k in SOURCE_KEYS if k!='joint' or joint_supported]
    bank_dim=sum(classes[k] for k in bank_keys)
    pca_start=time.perf_counter()
    pca=PCA(n_components=min(32,x['representation_fit'].shape[1],len(repr_frame)),svd_solver='full').fit(x['representation_fit'])
    pca_time=time.perf_counter()-pca_start
    compression=None
    compression_time=0.
    if model.repr_dim>bank_dim:
        tick=time.perf_counter()
        compression=PCA(n_components=bank_dim,svd_solver='full').fit(model.release(x['representation_fit']))
        compression_time=time.perf_counter()-tick
    joblib.dump({'pca':pca,'compression':compression},seed_dir/'release_maps.joblib',compress=3)
    # Frozen source identity is independent of held-out labels and all downstream selection.
    state_before=state_digest(model.state_dict())
    maps_before=joblib.hash({'tree':tree,'pca':pca,'compression':compression,'preprocessing':pre})
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise AssertionError('Source encoder is not frozen in evaluation mode')
    source_selection={'selected_lr':selected_lr,'selected_tree_leaves':selected_leaves,
                      'selected_encoder_state_sha256':state_before,'income_edges':edges.tolist(),
                      'class_counts':classes,'bank_keys':bank_keys,'joint_supported':bool(joint_supported),
                      'encoder_candidates':{str(lr):meta for _,meta,lr in encoder_candidates},
                      'tree_candidates':{str(leaves):meta for _,meta,leaves in tree_candidates},
                      'selected_utc':now(),'selection_uses':'source-validation mean five-head cross entropy only'}
    write_json(seed_dir/'source_selection.json',source_selection)
    def neural_bank(z,binary=False):
        p=model.probabilities(z)
        return np.column_stack([p['income_binary'][:,1],p['esr'][:,0],p['pubcov'][:,0]]) if binary else np.concatenate([p[k] for k in bank_keys],axis=1)
    def tree_bank(z):
        p=tree.probabilities(z)
        return np.concatenate([p[k] for k in bank_keys],axis=1)
    release_specs={'A_binary_bank':lambda z:neural_bank(z,True),'B_rich_bank':neural_bank,
                   'C_tree_bank':tree_bank,'D_features':model.release,
                   'E_pca':pca.transform,'F_covariates':lambda z:z.copy()}
    if compression is not None:
        release_specs['D_compressed']=lambda z:compression.transform(model.release(z))
    releases={}
    release_meta={}
    for name,extract in release_specs.items():
        tick=time.perf_counter()
        releases[name]={pool:np.asarray(extract(z),dtype=np.float32) for pool,z in x.items()}
        extraction=time.perf_counter()-tick
        arrays=releases[name]
        for a in arrays.values():
            a.setflags(write=False)
        dimension=arrays['representation_fit'].shape[1]
        release_meta[name]={'dimension':dimension,'dtype':'float32','bytes_per_record':dimension*4,
                            'development_extraction_seconds':extraction,'development_rows':sum(len(v) for v in arrays.values()),
                            'development_hashes':{p:array_hash(v) for p,v in arrays.items()}}
        np.savez_compressed(seed_dir/f'release_{name}.npz',**arrays)
    # Withheld values first enter fitting APIs only after all releases are frozen.
    fit_labels=heldout_labels(frame.iloc[pools['downstream_fit']])
    val_labels=heldout_labels(frame.iloc[pools['downstream_validation']])
    attack_fit=audit_labels(frame.iloc[pools['attacker_fit']])
    attack_val=audit_labels(frame.iloc[pools['attacker_validation']])
    task_indices={t:subset_indices(fit_labels[t],cfg['head_budget'],1230000+100*seed+j) for j,t in enumerate(TRANSFER)}
    audit_indices={t:subset_indices(attack_fit[t],cfg['attacker_budget'],1240000+100*seed+j) for j,t in enumerate(AUDITS)}
    head_budget={'mlp':{'epochs':cfg['head_mlp_epochs'],'batch_size':cfg['head_mlp_batch_size'],'lr':cfg['head_mlp_lr']},
                 'histgb':{'max_iter':cfg['attacker_tree_iterations'],'max_leaf_nodes':cfg['attacker_tree_leaves']}}
    fitted={}
    selections={}
    fit_records={}
    head_start=time.perf_counter()
    for release,arrays in releases.items():
        print(f'seed {seed}: downstream + audit {release}',flush=True)
        for j,target in enumerate(TRANSFER):
            idx=task_indices[target];vi=np.flatnonzero(val_labels[target]>=0)
            key=f'transfer/{release}/{target}'
            result=fit_candidates(arrays['downstream_fit'][idx],fit_labels[target][idx],
                                  arrays['downstream_validation'][vi],val_labels[target][vi],2,
                                  1250000+100*seed+j,budget=head_budget)
            save_candidates(result,seed_dir/'fitted'/key)
            fitted[key]=result['candidates'];selections[key]=result['selected_family'];fit_records[key]=result['metadata']
        for j,(target,nclasses) in enumerate(AUDITS.items()):
            idx=audit_indices[target];vi=np.flatnonzero(attack_val[target]>=0)
            key=f'audit/{release}/{target}'
            result=fit_candidates(arrays['attacker_fit'][idx],attack_fit[target][idx],
                                  arrays['attacker_validation'][vi],attack_val[target][vi],nclasses,
                                  1260000+100*seed+j,families=('logistic','mlp','histgb'),budget=head_budget)
            save_candidates(result,seed_dir/'fitted'/key)
            fitted[key]=result['candidates'];selections[key]=result['selected_family'];fit_records[key]=result['metadata']
    fitting_time=time.perf_counter()-head_start
    control_start=time.perf_counter()
    for j,(target,nclasses) in enumerate(AUDITS.items()):
        idx=audit_indices[target];vi=np.flatnonzero(attack_val[target]>=0)
        key=f'audit/exposed/{target}'
        result=fit_candidates(np.eye(nclasses)[attack_fit[target][idx]],attack_fit[target][idx],
                              np.eye(nclasses)[attack_val[target][vi]],attack_val[target][vi],nclasses,
                              1260000+100*seed+j,families=('logistic','mlp','histgb'),budget=head_budget)
        save_candidates(result,seed_dir/'fitted'/key)
        fitted[key]=result['candidates'];selections[key]=result['selected_family'];fit_records[key]=result['metadata']
    for role,labels,indices,classes_prior in (
        ('transfer',fit_labels,task_indices,{t:2 for t in TRANSFER}),
        ('audit',attack_fit,audit_indices,AUDITS)):
        validation=val_labels if role=='transfer' else attack_val
        for target,nclasses in classes_prior.items():
            key=f'{role}/prior/{target}'
            prior=fit_prior(labels[target][indices[target]],nclasses)
            vy=validation[target];vy=vy[vy>=0]
            prior.metadata['validation_scores']=metrics(vy,prior.predict_proba(np.zeros((len(vy),1))),nclasses)
            prior.save(seed_dir/'fitted'/key/'prior')
            fitted[key]={'prior':prior};selections[key]='prior';fit_records[key]={'candidates':{'prior':prior.metadata}}
    control_time=time.perf_counter()-control_start
    # Every candidate model is saved before the final test guard opens.
    record={'created_utc':now(),'head_selections':selections,'source_selection_sha256':sha_file(seed_dir/'source_selection.json'),
            'protocol_freeze_sha256':sha_file(out/'protocol_freeze.json') if verify else None,
            'release_metadata':release_meta,'fitting_records':fit_records,
            'task_fit_indices':{t:{'n':len(v),'pool_indices_sha256':array_hash(v),'raw_rows_sha256':array_hash(frame.iloc[pools['downstream_fit'][v]]._raw_row.to_numpy())} for t,v in task_indices.items()},
            'audit_fit_indices':{t:{'n':len(v),'pool_indices_sha256':array_hash(v),'raw_rows_sha256':array_hash(frame.iloc[pools['attacker_fit'][v]]._raw_row.to_numpy())} for t,v in audit_indices.items()}}
    write_json(seed_dir/'selection_before_test.json',record)
    selection_hash=require_test_selection(seed_dir,list(fitted))
    print(f'seed {seed}: all {len(selections)} selections saved; opening final test',flush=True)
    evaluation_started=now()
    evaluation_tick=time.perf_counter()
    test_frame=frame.iloc[pools['test']]
    xt=pre.transform(test_frame)
    test_releases={}
    for name,extract in release_specs.items():
        tick=time.perf_counter();value=np.asarray(extract(xt),dtype=np.float32)
        release_meta[name]['test_inference_seconds']=time.perf_counter()-tick
        release_meta[name]['test_rows']=len(xt)
        release_meta[name]['test_output_sha256']=array_hash(value)
        value.setflags(write=False);test_releases[name]=value
    yt=heldout_labels(test_frame);ya=audit_labels(test_frame)
    weights=test_frame.PWGTP.to_numpy(float)
    raw_metrics=[]
    predictions={}
    for key,candidates in fitted.items():
        role,release,target=key.split('/')
        y=(yt if role=='transfer' else ya)[target]
        nclasses=2 if role=='transfer' else AUDITS[target]
        valid=y>=0;truth=y[valid];w=weights[valid]
        if release=='prior':
            features=np.zeros((len(truth),1),dtype=np.float32)
        elif release=='exposed':
            features=np.eye(nclasses,dtype=np.float32)[truth]
        else:
            features=test_releases[release][valid]
        for family,candidate in candidates.items():
            prediction=candidate.predict_proba(features)
            predictions[f'{key}/{family}']=prediction
            raw_metrics.append({'seed':seed,'role':role,'release':release,'target':target,'family':family,
                                'selected':selections[key]==family,
                                'validation':candidate.metadata['validation_scores'],
                                'test':metrics(truth,prediction,nclasses),
                                'test_person_weighted':metrics(truth,prediction,nclasses,w)})
    np.savez_compressed(seed_dir/'test_predictions.npz',**predictions)
    source_test={'encoder':source_scores(model.probabilities(xt),source_labels(test_frame,edges),classes),
                 'tree':source_scores(tree.probabilities(xt),source_labels(test_frame,edges),classes)}
    state_after=state_digest(model.state_dict())
    maps_after=joblib.hash({'tree':tree,'pca':pca,'compression':compression,'preprocessing':pre})
    integrity={'source_state_before':state_before,'source_state_after':state_after,
               'fitted_maps_before':maps_before,'fitted_maps_after':maps_after,'fitted_maps_unchanged':maps_before==maps_after,
               'source_unchanged':state_before==state_after,
               'development_releases_unchanged':all(array_hash(a)==release_meta[name]['development_hashes'][pool]
                                                    for name,arrays in releases.items() for pool,a in arrays.items()),
               'selection_sha256':selection_hash,'selection_still_unchanged':sha_file(seed_dir/'selection_before_test.json')==selection_hash,
               'test_evaluation_started_utc':evaluation_started,'selection_created_utc':record['created_utc']}
    if not all(integrity[k] for k in ('source_unchanged','fitted_maps_unchanged','development_releases_unchanged','selection_still_unchanged')):
        raise AssertionError('Frozen state or saved selection changed')
    model_files=[p for p in seed_dir.rglob('*') if p.is_file() and p.suffix in ('.pt','.joblib','.npz')]
    local_manifest=[{'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
                     'bytes':p.stat().st_size,'sha256':sha_file(p),'availability':'local only; regenerate with runner'} for p in model_files]
    write_json(seed_dir/'local_artifacts.json',local_manifest)
    runtime={'total_seconds':time.perf_counter()-started,'source_encoder_search_seconds':encoder_time,
             'source_tree_search_seconds':tree_time,'pca_seconds':pca_time,'feature_compression_seconds':compression_time,
             'downstream_and_audit_seconds':fitting_time,'controls_seconds':control_time,
             'test_and_integrity_seconds':time.perf_counter()-evaluation_tick}
    result={'seed':seed,'cohort':cohort,'raw_metrics':raw_metrics,'release_metadata':release_meta,
            'source_selection':{'lr':selected_lr,'tree_leaves':selected_leaves,'state_sha256':state_before},
            'source_test':source_test,'integrity':integrity,'runtime':runtime}
    write_json(seed_dir/'metrics.json',result)
    print(json.dumps({'seed':seed,'runtime':runtime,'integrity':integrity}),flush=True)
    return result


def miniature(out,cfg):
    """Artificial plumbing fixture; does not evaluate or redesign the ACS tasks."""
    from tempfile import TemporaryDirectory
    rng=np.random.default_rng(89991)
    with TemporaryDirectory(prefix='pcrl-acs-mini-') as temp:
        temp=Path(temp);n=1400
        f=pd.DataFrame({'SERIALNO':[f'fixture_{i//2}' for i in range(n)],'SPORDER':1+np.arange(n)%2,
                        'PWGTP':rng.integers(1,100,n),'AGEP':rng.integers(19,35,n),'WKHP':rng.integers(1,70,n),
                        'SCHL':rng.integers(1,25,n),'MAR':rng.integers(1,6,n),'RELP':rng.integers(0,18,n),
                        'CIT':rng.integers(1,6,n),'DIS':rng.integers(1,3,n),'DEAR':rng.integers(1,3,n),
                        'DEYE':rng.integers(1,3,n),'DREM':rng.integers(1,3,n),'PINCP':rng.integers(-100,100000,n),
                        'ESR':rng.integers(1,7,n),'PUBCOV':rng.integers(1,3,n),'MIG':rng.integers(1,4,n),
                        'JWMNP':rng.integers(1,201,n),'SEX':rng.integers(1,3,n),'RAC1P':rng.integers(1,10,n)})
        csv=temp/'fixture.csv';f.to_csv(csv,index=False)
        small={**cfg,'raw_path':str(csv),'sample_cap':n,'source_epochs':2,'source_tree_iterations':2,
               'head_mlp_epochs':2,'attacker_tree_iterations':2,'head_budget':64,'attacker_budget':128}
        result=run_seed(temp,small,97,verify=False)
        record={'fixture':'1400 artificial rows, no ACS model outcomes; full fit/select/test/save plumbing',
                'integrity':result['integrity'],'raw_metric_rows':len(result['raw_metrics']),
                'release_dimensions':{k:v['dimension'] for k,v in result['release_metadata'].items()},
                'runtime':result['runtime'],'success':True,'created_utc':now()}
        write_json(out/'PIPELINE_CHECK.json',record)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--mini-check',action='store_true')
    parser.add_argument('--seeds',type=int,nargs='+')
    args=parser.parse_args();out=args.out.resolve();cfg=json.loads((out/'config.json').read_text())
    torch.set_num_threads(cfg['threads'])
    with threadpool_limits(limits=cfg['threads']):
        if args.prepare:
            if (out/'protocol_freeze.json').exists():
                raise RuntimeError('Existing freeze preserved; choose a fresh directory or document correction')
            prepare(out,cfg)
        if args.mini_check:
            miniature(out,cfg)
        if args.seeds:
            for seed in args.seeds:
                if seed not in cfg['seeds']:
                    raise ValueError('Seed not predeclared')
                run_seed(out,cfg,seed)


if __name__=='__main__':
    main()
