"""Frozen ACS parent releases plus one joint LEACE map; development evaluation.

Reuses the published ACS data/source/head infrastructure. No source model is
trained here. Historical test households are explicitly development evidence.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import subprocess
import time

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import concept_erasure

from experiments.acs_transfer_data import (
    AUDIT_NAMES, CovariatePreprocessor, array_hash, audit_labels, heldout_labels,
    load_cohort, require_test_selection, sha_file, source_labels, split_households,
    support, write_json,
)
from experiments.acs_transfer_heads import fit_candidates, fit_prior, metrics
from experiments.acs_transfer_models import SourceEncoder, state_digest
from experiments.run_acs_transfer import subset_indices, save_candidates
from experiments.acs_protection_audits import fit_primary_auditors
from experiments.acs_protection_maps import fit_joint_leace, NUMERICAL_POLICY, LEACE_OPTIONS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT/'results/redesign_20260908_acs_protection_v1'
PARENTS = ('A_binary_bank','B_rich_bank','C_tree_bank','D_compressed','E_pca')
TASKS = ('same_residence','commute_over20','income_binary','civilian_at_work','public_coverage')
AUDITS = {'SEX':2,'RAC1P':9}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def execution_paths():
    names = ('run_acs_protection.py','acs_protection_maps.py','acs_protection_audits.py',
             'acs_transfer_data.py','acs_transfer_models.py','acs_transfer_heads.py','run_acs_transfer.py')
    paths = [ROOT/'experiments'/n for n in names]
    paths += sorted(Path(inspect.getfile(concept_erasure)).parent.rglob('*.py'))
    return paths


def frozen_hashes(out):
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha_file(p)
            for p in [out/'PROTOCOL.md',out/'config.json',*execution_paths()]}


def verify_freeze(out):
    freeze = read(out/'protocol_freeze.json')
    if frozen_hashes(out) != freeze['sha256']:
        raise RuntimeError('Frozen protocol/config/execution source changed')
    for path, expected in freeze['parent_record_hashes'].items():
        if sha_file(ROOT/path) != expected:
            raise RuntimeError('Historical parent identity changed: '+path)


def prepare(out):
    cfg=read(out/'config.json'); parent=ROOT/cfg['parent_results']
    if tuple(cfg['parents'])!=PARENTS or tuple(cfg['utility_tasks'])!=TASKS:
        raise ValueError('Fixed parent/task schema changed')
    if (cfg['audit_mlp_epochs'] != 120 or cfg['audit_restart_offset'] != 10000
            or cfg['audit_min_samples_leaf'] != [20,5] or cfg['head_mlp_epochs'] != 40
            or cfg['head_budget'] != 2048 or cfg['attacker_budget'] != 4096):
        raise ValueError('Scientific budgets must match the fixed protocol')
    expected_leace={'covariance_relative_tolerance':NUMERICAL_POLICY['input_covariance_rtol'],
        'map_relative_tolerance':NUMERICAL_POLICY['projection_svd_rtol'],
        'svd_tolerance':LEACE_OPTIONS['svd_tol'],'shrinkage':False,'constrain_cov_trace':False,
        'ridge':0.,'affine':True,'covariance_dtype':'float64','release_dtype':'float32'}
    if cfg['leace']!=expected_leace:
        raise ValueError('Recorded erasure recipe must match fixed implementation')
    record_paths=[parent/'config.json',parent/'schema_support.json',parent/'protocol_freeze.json']
    for s in cfg['seeds']:
        record_paths += [parent/f'seed_{s}'/n for n in ('metrics.json','source_selection.json',
                         'preprocessing.json','selection_before_test.json','local_artifacts.json')]
    freeze={'created_utc':now(),'starting_commit':cfg['starting_commit'],
            'sha256':frozen_hashes(out),
            'parent_record_hashes':{str(p.relative_to(ROOT)):sha_file(p) for p in record_paths},
            'evaluation_status':cfg['evaluation_status']}
    if (out/'protocol_freeze.json').exists():
        raise RuntimeError('Preserve existing freeze; use a new directory')
    write_json(out/'protocol_freeze.json',freeze)
    write_json(out/'environment.json',{'created_utc':now(),'platform':platform.platform(),
        'machine':platform.machine(),'processor':platform.processor(),
        'hardware':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
        'threads':cfg['threads'],'python':platform.python_version(),
        'packages':{k:importlib.metadata.version(k) for k in
                    ('torch','numpy','pandas','scikit-learn','scipy','concept-erasure','joblib','threadpoolctl')}})


def binary_tasks(frame, edges):
    src=source_labels(frame,edges)
    return {**heldout_labels(frame),'income_binary':src['income_binary'],
            'civilian_at_work':np.where(src['esr']<0,-1,(src['esr']==0).astype(int)),
            'public_coverage':np.where(src['pubcov']<0,-1,(src['pubcov']==0).astype(int))}


class ParentBundle:
    """Load exact saved source tensors/maps and development arrays, without fitting."""
    def __init__(self, parent, seed, frame, pools):
        self.directory=parent/f'seed_{seed}'
        self.selection=read(self.directory/'source_selection.json')
        self.previous=read(self.directory/'metrics.json')
        self.used_files={}
        manifest={str(ROOT/v['path']):v for v in read(self.directory/'local_artifacts.json')}
        def checked(path):
            actual=sha_file(path)
            if str(path) not in manifest or actual!=manifest[str(path)]['sha256']:
                raise RuntimeError('Missing/changed saved parent artifact: '+str(path))
            self.used_files[str(path.relative_to(ROOT))]=actual
            return path
        rows=np.load(checked(self.directory/'split_rows.npz'))
        for pool, idx in pools.items():
            if not np.array_equal(rows[pool],frame.iloc[idx]._raw_row.to_numpy()):
                raise AssertionError('Original split row alignment changed')
        prep=read(self.directory/'preprocessing.json')
        self.pre=CovariatePreprocessor()
        self.pre.numeric,self.pre.categories,self.pre.feature_names=(prep[k] for k in ('numeric','categories','feature_names'))
        self.model=SourceEncoder.load(checked(self.directory/'source'/f"encoder_{self.selection['selected_lr']}"/'selected.pt'))
        self.tree=joblib.load(checked(self.directory/'source'/f"tree_{self.selection['selected_tree_leaves']}"/'source_tree_bank.joblib'))
        maps=joblib.load(checked(self.directory/'release_maps.joblib'))
        self.pca,self.compression=maps['pca'],maps['compression']
        self.releases={}
        for name in (*PARENTS,'F_covariates'):
            with np.load(checked(self.directory/f'release_{name}.npz')) as z:
                self.releases[name]={p:z[p].copy() for p in z.files}
            for pool,a in self.releases[name].items():
                if array_hash(a)!=self.previous['release_metadata'][name]['development_hashes'][pool]:
                    raise AssertionError('Parent release output differs')
                a.setflags(write=False)
        self.initial=self.fingerprint()
        if self.initial['encoder']!=self.selection['selected_encoder_state_sha256']:
            raise AssertionError('Wrong starting encoder tensors')

    def fingerprint(self):
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise AssertionError('Source model must remain frozen and in evaluation mode')
        return {'encoder':state_digest(self.model.state_dict()),
                'other_maps':joblib.hash((self.pre,self.tree,self.pca,self.compression))}

    def evaluation_releases(self, frame):
        x=self.pre.transform(frame); p=self.model.probabilities(x)
        keys=self.selection['bank_keys']; tree=self.tree.probabilities(x)
        releases={'A_binary_bank':np.column_stack([p['income_binary'][:,1],p['esr'][:,0],p['pubcov'][:,0]]),
                  'B_rich_bank':np.concatenate([p[k] for k in keys],axis=1),
                  'C_tree_bank':np.concatenate([tree[k] for k in keys],axis=1),
                  'D_compressed':self.compression.transform(self.model.release(x)),
                  'E_pca':self.pca.transform(x),'F_covariates':x}
        result={k:np.asarray(v,dtype=np.float32) for k,v in releases.items()}
        for name,a in result.items():
            if array_hash(a)!=self.previous['release_metadata'][name]['test_output_sha256']:
                raise AssertionError('Regenerated evaluation release differs from historical output')
            a.setflags(write=False)
        return result


def feature_rank(x):
    z=np.asarray(x,dtype=np.float64); z=z-z[0]; z=z-z.mean(axis=0)
    ev=np.linalg.eigvalsh(z.T@z/max(1,len(z)-1)).clip(0)
    retained=ev>1e-10*ev.max() if ev.max()>0 else np.zeros(len(ev),bool)
    p=ev[retained]/ev[retained].sum() if retained.any() else np.array([])
    return {'centered_covariance_rank':int(retained.sum()),
            'entropy_effective_rank':float(np.exp(-(p*np.log(p)).sum())) if len(p) else 0.,
            'covariance_eigenvalues':ev.tolist()}


def frozen_digest(releases):
    return {n:{p:array_hash(a) for p,a in arrays.items()} for n,arrays in releases.items()}


def score_and_save(fitted, selections, features, truths, weights, split, predictions, cfg):
    rows=[]
    for key,candidates in fitted.items():
        role,release,target=key.split('/'); y=truths[role][target]
        valid=y>=0; truth=y[valid]; nclasses=2 if role=='transfer' else AUDITS[target]
        if release=='prior': x=np.zeros((len(truth),1),dtype=np.float32)
        elif release=='exposed': x=np.eye(nclasses,dtype=np.float32)[truth]
        else: x=features[role][release][valid]
        for candidate_id,candidate in candidates.items():
            prob=candidate.predict_proba(x)
            predictions[f'{split}/{key}/{candidate_id}']=prob
            rows.append({'key':key,'candidate_id':candidate_id,'score':metrics(truth,prob,nclasses),
                         'weighted':metrics(truth,prob,nclasses,weights[role][valid])})
    return {(v['key'],v['candidate_id']):v for v in rows}


def run_seed(out,cfg,seed):
    verify_freeze(out); tick=time.perf_counter(); directory=out/f'seed_{seed}'
    directory.mkdir(exist_ok=False)
    parent=ROOT/cfg['parent_results']; pcfg=read(parent/'config.json'); oldsupport=read(parent/'schema_support.json')
    if sha_file(ROOT/pcfg['raw_path'])!=oldsupport['raw_sha256']:
        raise RuntimeError('Raw CSV identity changed')
    frame,cohort=load_cohort(ROOT/pcfg['raw_path'],pcfg['sample_cap'],pcfg['sample_seed'])
    pools=split_households(frame,seed)
    bundle=ParentBundle(parent,seed,frame,pools)
    reuse_seconds=time.perf_counter()-tick
    write_json(directory/'parent_provenance.json',{'original_source_selection':bundle.selection,
        'used_local_files_sha256':bundle.used_files,'initial_identity':bundle.initial,
        'regeneration':'No source models or parent maps refitted; evaluation arrays regenerated by frozen inference only.',
        'historical_support_source':str((parent/'schema_support.json').relative_to(ROOT))})
    np.savez_compressed(directory/'split_rows.npz',**{p:frame.iloc[i]._raw_row.to_numpy() for p,i in pools.items()})
    cal=frame.iloc[pools['representation_fit']]; labels=audit_labels(cal)
    erasers={}; releases={}; release_meta={}; et=time.perf_counter()
    for name in PARENTS:
        arrays=bundle.releases[name]
        eraser=fit_joint_leace(arrays['representation_fit'],labels,fit_split='representation_fit')
        erasers[name]=eraser; eraser.save(directory/f'map_{name}.npz')
        releases[name]=arrays
        releases[name+'_leace']={p:eraser.apply(a) for p,a in arrays.items()}
    releases['F_covariates']=bundle.releases['F_covariates']
    map_before={n:e.fingerprint() for n,e in erasers.items()}
    for name,arrays in releases.items():
        for a in arrays.values(): a.setflags(write=False)
        parent_name=name.removesuffix('_leace'); dim=arrays['representation_fit'].shape[1]
        release_meta[name]={'parent':parent_name,'erased':name.endswith('_leace'),'dimension':dim,
            'dtype':'float32','bytes_per_record':4*dim,**feature_rank(arrays['representation_fit']),
            'eraser':erasers[parent_name].metadata if name.endswith('_leace') else None}
    release_before=frozen_digest(releases)
    write_json(directory/'release_freeze.json',{'created_utc':now(),'release_metadata':release_meta,
        'output_hashes':release_before,'map_hashes':map_before,
        'calibration_pool':'representation_fit','calibration_raw_row_sha256':array_hash(cal._raw_row.to_numpy()),
        'calibration_attribute_support':support(labels,AUDITS),
        'reserved_task_labels_entered_release_fitting':False})
    erasure_seconds=time.perf_counter()-et
    # Reserved labels first enter fitting APIs after all release maps are frozen.
    edges=bundle.selection['income_edges']
    fit_y=binary_tasks(frame.iloc[pools['downstream_fit']],edges)
    val_y=binary_tasks(frame.iloc[pools['downstream_validation']],edges)
    audit_y=audit_labels(frame.iloc[pools['attacker_fit']]); audit_v=audit_labels(frame.iloc[pools['attacker_validation']])
    ti={t:subset_indices(fit_y[t],cfg['head_budget'],1230000+100*seed+j) for j,t in enumerate(TASKS)}
    ai={t:subset_indices(audit_y[t],cfg['attacker_budget'],1240000+100*seed+j) for j,t in enumerate(AUDITS)}
    fitted={}; selections={}; records={}; auroc_selections={}; family_selections={}
    def accept(key,result):
        save_candidates(result,directory/'fitted'/key)
        fitted[key]=result['candidates']; selections[key]=result['selected_family']; records[key]=result['metadata']
        auroc_selections[key]=result.get('selected_auroc')
        fams={c.metadata['family'] for c in result['candidates'].values()}
        family_selections[key]={f:min((k for k,c in result['candidates'].items() if c.metadata['family']==f),
            key=lambda k:(result['candidates'][k].metadata['validation_scores']['log_loss'],k)) for f in fams}
    task_seconds=0.;audit_seconds=0.
    for release,arrays in releases.items():
        print(f'seed {seed}: {release}: five task heads + five candidates per attribute',flush=True)
        t=time.perf_counter()
        for j,target in enumerate(TASKS):
            i=ti[target];vi=np.flatnonzero(val_y[target]>=0)
            accept(f'transfer/{release}/{target}',fit_candidates(arrays['downstream_fit'][i],fit_y[target][i],
                arrays['downstream_validation'][vi],val_y[target][vi],2,1250000+100*seed+j,
                budget={'mlp':{'epochs':cfg['head_mlp_epochs']}}))
        task_seconds+=time.perf_counter()-t; t=time.perf_counter()
        for j,(target,k) in enumerate(AUDITS.items()):
            i=ai[target];vi=np.flatnonzero(audit_v[target]>=0)
            accept(f'audit/{release}/{target}',fit_primary_auditors(arrays['attacker_fit'][i],audit_y[target][i],
                arrays['attacker_validation'][vi],audit_v[target][vi],k,1260000+100*seed+j))
        audit_seconds+=time.perf_counter()-t
    t=time.perf_counter()
    for j,(target,k) in enumerate(AUDITS.items()):
        i=ai[target];vi=np.flatnonzero(audit_v[target]>=0)
        accept(f'audit/exposed/{target}',fit_primary_auditors(np.eye(k)[audit_y[target][i]],audit_y[target][i],
            np.eye(k)[audit_v[target][vi]],audit_v[target][vi],k,1260000+100*seed+j))
    for role,y,v,indices,classes in [('transfer',fit_y,val_y,ti,{t:2 for t in TASKS}),('audit',audit_y,audit_v,ai,AUDITS)]:
        for target,k in classes.items():
            prior=fit_prior(y[target][indices[target]],k); vy=v[target][v[target]>=0]
            prior.metadata['validation_scores']=metrics(vy,prior.predict_proba(np.zeros((len(vy),1))),k)
            accept(f'{role}/prior/{target}',{'candidates':{'prior':prior},'selected_family':'prior',
                'metadata':{'candidates':{'prior':prior.metadata}}})
    control_seconds=time.perf_counter()-t
    # Support inspection never selects a release or changes a split.
    support_by_pool={p:{'persons':len(i),'households':int(frame.iloc[i].SERIALNO.nunique()),
        'raw_row_sha256':array_hash(frame.iloc[i]._raw_row.to_numpy()),
        'tasks':support(binary_tasks(frame.iloc[i],edges),{t:2 for t in TASKS}),
        'attributes':support(audit_labels(frame.iloc[i]),AUDITS)} for p,i in pools.items()}
    write_json(directory/'support.json',support_by_pool)
    def index_record(indices,pool):
        return {t:{'n':len(i),'pool_indices_sha256':array_hash(i),
            'raw_rows_sha256':array_hash(frame.iloc[pools[pool][i]]._raw_row.to_numpy())} for t,i in indices.items()}
    record={'created_utc':now(),'evaluation_status':cfg['evaluation_status'],'head_selections':selections,
        'family_selections':family_selections,'auroc_selections':auroc_selections,'fitting_records':records,
        'task_fit_indices':index_record(ti,'downstream_fit'),'audit_fit_indices':index_record(ai,'attacker_fit'),
        'release_freeze_sha256':sha_file(directory/'release_freeze.json'),
        'protocol_freeze_sha256':sha_file(out/'protocol_freeze.json')}
    write_json(directory/'selection_before_test.json',record)
    selection_hash=require_test_selection(directory,list(fitted))
    print(f'seed {seed}: {len(fitted)} selections saved; opening DEVELOPMENT EVALUATION',flush=True)
    eval_start=now();t=time.perf_counter()
    test_frame=frame.iloc[pools['test']]
    test=bundle.evaluation_releases(test_frame)
    for name,eraser in erasers.items(): test[name+'_leace']=eraser.apply(test[name])
    for a in test.values(): a.setflags(write=False)
    predictions={}
    val=score_and_save(fitted,selections,
        {r:{n:a[p] for n,a in releases.items()} for r,p in [('transfer','downstream_validation'),('audit','attacker_validation')]},
        {'transfer':val_y,'audit':audit_v},
        {r:frame.iloc[pools[p]].PWGTP.to_numpy(float) for r,p in [('transfer','downstream_validation'),('audit','attacker_validation')]},
        'validation',predictions,cfg)
    evaluation=score_and_save(fitted,selections,{'transfer':test,'audit':test},
        {'transfer':binary_tasks(test_frame,edges),'audit':audit_labels(test_frame)},
        {r:test_frame.PWGTP.to_numpy(float) for r in ('transfer','audit')},'test',predictions,cfg)
    raw=[]
    for key,candidates in fitted.items():
        role,release,target=key.split('/')
        for cid,c in candidates.items():
            raw.append({'seed':seed,'role':role,'release':release,'parent':release.removesuffix('_leace'),
                'erased':release.endswith('_leace'),'target':target,'candidate_id':cid,'family':c.metadata['family'],
                'selected':selections[key]==cid,'selected_within_family':family_selections[key][c.metadata['family']]==cid,
                'auc_selected':auroc_selections[key]==cid,'validation':val[key,cid]['score'],
                'validation_person_weighted':val[key,cid]['weighted'],'test':evaluation[key,cid]['score'],
                'test_person_weighted':evaluation[key,cid]['weighted']})
    np.savez_compressed(directory/'predictions.npz',**predictions)
    for n,a in releases.items(): np.savez_compressed(directory/f'release_{n}.npz',**a,test=test[n])
    integrity={'source_unchanged':bundle.fingerprint()==bundle.initial,
        'parent_files_unchanged':all(sha_file(ROOT/p)==h for p,h in bundle.used_files.items()),
        'erasers_unchanged':map_before=={n:e.fingerprint() for n,e in erasers.items()},
        'development_releases_unchanged':release_before==frozen_digest(releases),
        'selection_unchanged':sha_file(directory/'selection_before_test.json')==selection_hash,
        'selection_created_utc':record['created_utc'],'evaluation_started_utc':eval_start,
        'evaluation_parent_outputs_match_historical':True,'selection_sha256':selection_hash,
        'evaluation_output_hashes':{n:array_hash(a) for n,a in test.items()}}
    if not all(integrity[k] for k in ('source_unchanged','parent_files_unchanged','erasers_unchanged',
                                    'development_releases_unchanged','selection_unchanged')):
        raise AssertionError('Frozen release integrity failed')
    local=[{'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
            'bytes':p.stat().st_size,'sha256':sha_file(p),'availability':'local only; regenerate with runner'}
           for p in directory.rglob('*') if p.is_file() and p.suffix in ('.npz','.joblib','.pt')]
    write_json(directory/'local_artifacts.json',local)
    runtime={'total_seconds':time.perf_counter()-tick,'parent_loading_verification_seconds':reuse_seconds,
        'erasure_and_freeze_seconds':erasure_seconds,'utility_fitting_seconds':task_seconds,
        'audit_fitting_seconds':audit_seconds,'controls_seconds':control_seconds,
        'evaluation_serialization_integrity_seconds':time.perf_counter()-t}
    result={'seed':seed,'evaluation_status':cfg['evaluation_status'],'cohort':cohort,
        'raw_metrics':raw,'release_metadata':release_meta,'support_by_pool':support_by_pool,
        'integrity':integrity,'runtime':runtime}
    write_json(directory/'metrics.json',result)
    print(json.dumps({'seed':seed,'runtime':runtime,'integrity':integrity}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    p.add_argument('--prepare',action='store_true');p.add_argument('--seeds',type=int,nargs='+')
    args=p.parse_args();out=args.out.resolve();cfg=read(out/'config.json')
    torch.set_num_threads(cfg['threads'])
    with threadpool_limits(limits=cfg['threads']):
        if args.prepare: prepare(out)
        for seed in args.seeds or []:
            if seed not in cfg['seeds']: raise ValueError('Seed not predeclared')
            run_seed(out,cfg,seed)


if __name__=='__main__':
    main()
