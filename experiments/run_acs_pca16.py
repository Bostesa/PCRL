"""One fixed first16-coordinate control on existing ACS PCA32 releases."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from experiments.acs_transfer_data import (array_hash, audit_labels, load_cohort,
    require_test_selection, sha_file, split_households, write_json)
from experiments.acs_transfer_heads import fit_candidates
from experiments.acs_protection_audits import fit_primary_auditors
from experiments.run_acs_protection import (ROOT,TASKS,AUDITS,binary_tasks,now,read,
    score_and_save,execution_paths as parent_execution_paths)
from experiments.run_acs_transfer import subset_indices,save_candidates

DEFAULT_OUT=ROOT/'results/redesign_20260908_acs_pca16_v1'
FIT_POOLS=('downstream_fit','downstream_validation','attacker_fit','attacker_validation')


def slice_pca16(x32):
    x=np.asarray(x32)
    if x.ndim!=2 or x.shape[1]!=32 or x.dtype!=np.float32 or not np.isfinite(x).all():
        raise ValueError('Expected finite float32 PCA32 matrix in original order')
    x=np.ascontiguousarray(x[:,:16])
    return np.frombuffer(x.tobytes(),dtype=np.float32).reshape(x.shape)


def execution_paths():
    return [*parent_execution_paths(),Path(__file__).resolve()]


def freeze_hashes(out):
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha_file(p)
            for p in [out/'PROTOCOL.md',out/'config.json',*execution_paths()]}


def prepare(out):
    cfg=read(out/'config.json');assert cfg['seeds']==[0,1,2] and cfg['components']==list(range(16))
    assert cfg['head_budget']==2048 and cfg['attacker_budget']==4096 and cfg['threads']==1
    assert cfg['head_mlp_epochs']==40 and cfg['audit_mlp_epochs']==120
    assert cfg['audit_restart_offset']==10000 and cfg['audit_min_samples_leaf']==[20,5]
    reference=ROOT/cfg['reference_results']
    paths=[reference/'config.json',reference/'protocol_freeze.json',reference/'SCORE_REPLAY.json']
    parent=ROOT/cfg['parent_results'];cache=ROOT/cfg['pca_cache_results']
    paths += [parent/'config.json',parent/'schema_support.json',cache/'config.json']
    for seed in cfg['seeds']:
        paths += [reference/f'seed_{seed}'/n for n in ('metrics.json','selection_before_test.json',
                  'support.json','local_artifacts.json','release_freeze.json','parent_provenance.json')]
        paths += [parent/f'seed_{seed}'/n for n in ('source_selection.json','local_artifacts.json')]
        paths += [cache/f'seed_{seed}'/n for n in ('metrics.json','release_freeze.json','local_artifacts.json')]
    freeze={'created_utc':now(),'starting_commit':cfg['starting_commit'],'sha256':freeze_hashes(out),
            'reference_records':{str(p.relative_to(ROOT)):sha_file(p) for p in paths}}
    if (out/'protocol_freeze.json').exists():raise FileExistsError('Preserve existing freeze')
    write_json(out/'protocol_freeze.json',freeze)
    write_json(out/'environment.json',read(reference/'environment.json'))


def verify_freeze(out):
    f=read(out/'protocol_freeze.json');assert freeze_hashes(out)==f['sha256']
    for p,h in f['reference_records'].items():assert sha_file(ROOT/p)==h,p


def fit_release(arrays,fit_y,val_y,audit_y,audit_v,seed,directory,*,miniature=False):
    """Fit only new PCA16 candidates; never accepts final-evaluation arrays."""
    if set(arrays)!=set(FIT_POOLS):raise ValueError('Only the four fitting/validation release pools are allowed')
    assert set(fit_y)==set(val_y)==set(TASKS) and set(audit_y)==set(audit_v)==set(AUDITS)
    ti={t:subset_indices(fit_y[t],16 if miniature else 2048,1230000+100*seed+j) for j,t in enumerate(TASKS)}
    ai={t:subset_indices(audit_y[t],32 if miniature else 4096,1240000+100*seed+j) for j,t in enumerate(AUDITS)}
    fitted={};record={k:{} for k in ('head_selections','family_selections','auroc_selections','fitting_records')}
    def accept(key,result):
        save_candidates(result,directory/'fitted'/key);candidates=result['candidates'];fitted[key]=candidates
        record['head_selections'][key]=result['selected_family']
        record['auroc_selections'][key]=result.get('selected_auroc')
        record['fitting_records'][key]=result['metadata']
        record['family_selections'][key]={f:min((c for c,v in candidates.items() if v.metadata['family']==f),
            key=lambda c:(candidates[c].metadata['validation_scores']['log_loss'],c))
            for f in {v.metadata['family'] for v in candidates.values()}}
    for j,target in enumerate(TASKS):
        i=ti[target];v=val_y[target]>=0
        accept(f'transfer/PCA16/{target}',fit_candidates(arrays['downstream_fit'][i],fit_y[target][i],
            arrays['downstream_validation'][v],val_y[target][v],2,1250000+100*seed+j,
            budget={'mlp':{'epochs':1 if miniature else 40}}))
    for j,(target,k) in enumerate(AUDITS.items()):
        i=ai[target];v=audit_v[target]>=0
        accept(f'audit/PCA16/{target}',fit_primary_auditors(arrays['attacker_fit'][i],audit_y[target][i],
            arrays['attacker_validation'][v],audit_v[target][v],k,1260000+100*seed+j,
            budget={'mlp':{'epochs':1},'histgb':{'max_iter':1}} if miniature else None))
    record.update(task_fit_indices=ti,audit_fit_indices=ai)
    return fitted,record


def run_seed(out,cfg,seed):
    verify_freeze(out);tick=time.perf_counter();directory=out/f'seed_{seed}';directory.mkdir(exist_ok=False)
    parent=ROOT/cfg['parent_results'];calibration=ROOT/cfg['pca_cache_results'];reference=ROOT/cfg['reference_results']
    pcfg=read(parent/'config.json');schema=read(parent/'schema_support.json')
    assert sha_file(ROOT/pcfg['raw_path'])==schema['raw_sha256']
    frame,cohort=load_cohort(ROOT/pcfg['raw_path'],pcfg['sample_cap'],pcfg['sample_seed']);pools=split_households(frame,seed)
    olddir=calibration/f'seed_{seed}';refdir=reference/f'seed_{seed}';pdir=parent/f'seed_{seed}'
    manifests=[read(d/'local_artifacts.json') for d in (olddir,refdir,pdir)]
    hashes={str(ROOT/r['path']):r['sha256'] for m in manifests for r in m};used={}
    def checked(p):
        h=sha_file(p);assert hashes.get(str(p))==h, 'Required artifact changed/missing: '+str(p)
        used[str(p.relative_to(ROOT))]=h;return p
    pca=joblib.load(checked(pdir/'release_maps.joblib'))['pca'];pca_before=joblib.hash(pca)
    assert not pca.whiten and pca.n_components_==32 and pca.svd_solver=='full'
    ev=pca.explained_variance_;assert np.isfinite(ev).all() and (np.diff(ev)<=0).all()
    rowfile=checked(refdir/'split_rows.npz')
    with np.load(rowfile) as rows:
        for pool,idx in pools.items():assert np.array_equal(rows[pool],frame.iloc[idx]._raw_row.to_numpy())
    # Cached original PCA32 and covariates are verified by their original binary
    # hashes. Test contents are not opened until new candidate choices are saved.
    xp=checked(olddir/'release_E_pca.npz');fp=checked(olddir/'release_F_covariates.npz')
    previous=read(olddir/'metrics.json');support=read(refdir/'support.json')
    releases={};original_hashes={}
    with np.load(xp) as xs,np.load(fp) as fs:
        for pool in pools:
            if pool=='test':continue
            x=xs[pool]
            assert np.array_equal(pca.transform(fs[pool]).astype(np.float32),x),'Original PCA order/output differs'
            assert array_hash(x)==read(olddir/'release_freeze.json')['output_hashes']['E_pca'][pool]
            original_hashes[pool]=array_hash(x);releases[pool]=slice_pca16(x)
    # Existing reference predictions are reused, not refitted/reselected.
    for d in (refdir,olddir):checked(d/'predictions.npz')
    ref_selection=read(refdir/'selection_before_test.json')
    for key in ('head_budget','attacker_budget'):
        assert cfg[key]==read(reference/'config.json')[key]
    freeze={'created_utc':now(),'release':'PCA16','component_indices':list(range(16)),
        'dimension':16,'dtype':'float32','bytes_per_record':64,'whiten':False,'representation_fitted':False,
        'pca_object_hash':pca_before,'components_sha256':array_hash(pca.components_),
        'component_row_sha256':[array_hash(v) for v in pca.components_],
        'original_mean_sha256':array_hash(pca.mean_),'explained_variance':ev.tolist(),
        'explained_variance_ratio':pca.explained_variance_ratio_.tolist(),
        'first16_variance_fraction':float(pca.explained_variance_ratio_[:16].sum()),
        'output_hashes':{p:array_hash(x) for p,x in releases.items()},'pca32_output_hashes':original_hashes,
        'slice_before_any_head_standardization':True}
    write_json(directory/'release_freeze.json',freeze)
    setup_seconds=time.perf_counter()-tick
    edges=read(pdir/'source_selection.json')['income_edges']
    fit_y=binary_tasks(frame.iloc[pools['downstream_fit']],edges);val_y=binary_tasks(frame.iloc[pools['downstream_validation']],edges)
    audit_y=audit_labels(frame.iloc[pools['attacker_fit']]);audit_v=audit_labels(frame.iloc[pools['attacker_validation']])
    t=time.perf_counter();fitted,selection=fit_release({p:releases[p] for p in FIT_POOLS},fit_y,val_y,audit_y,audit_v,seed,directory)
    fitting_seconds=time.perf_counter()-t
    for field,pool in [('task_fit_indices','downstream_fit'),('audit_fit_indices','attacker_fit')]:
        selection[field]={target:{'n':len(i),'pool_indices_sha256':array_hash(i),
            'raw_rows_sha256':array_hash(frame.iloc[pools[pool][i]]._raw_row.to_numpy())} for target,i in selection[field].items()}
        assert selection[field]==ref_selection[field],'Changed fitting examples'
    selection.update(created_utc=now(),release_freeze_sha256=sha_file(directory/'release_freeze.json'),
        protocol_freeze_sha256=sha_file(out/'protocol_freeze.json'),primary_audit='five independent candidates; no saved PCA16 adversary')
    write_json(directory/'selection_before_test.json',selection);selection_hash=require_test_selection(directory,list(fitted))
    print(f'seed {seed}: seven suite selections saved; opening DEVELOPMENT EVALUATION',flush=True)
    evaluated_utc=now();t=time.perf_counter()
    with np.load(xp) as xs,np.load(fp) as fs:
        x=xs['test'];assert np.array_equal(pca.transform(fs['test']).astype(np.float32),x)
        assert array_hash(x)==previous['integrity']['evaluation_output_hashes']['E_pca']
        test=slice_pca16(x)
    predictions={}
    val=score_and_save(fitted,selection['head_selections'],
        {r:{'PCA16':releases[p]} for r,p in [('transfer','downstream_validation'),('audit','attacker_validation')]},
        {'transfer':val_y,'audit':audit_v},
        {r:frame.iloc[pools[p]].PWGTP.to_numpy(float) for r,p in [('transfer','downstream_validation'),('audit','attacker_validation')]},
        'validation',predictions,cfg)
    test_frame=frame.iloc[pools['test']]
    evaluation=score_and_save(fitted,selection['head_selections'],{r:{'PCA16':test} for r in ('transfer','audit')},
        {'transfer':binary_tasks(test_frame,edges),'audit':audit_labels(test_frame)},
        {r:test_frame.PWGTP.to_numpy(float) for r in ('transfer','audit')},'test',predictions,cfg)
    raw=[]
    for key,cs in fitted.items():
        role,release,target=key.split('/')
        for cid,c in cs.items():
            raw.append({'seed':seed,'role':role,'release':release,'parent':'E_pca','erased':False,'target':target,
                'candidate_id':cid,'family':c.metadata['family'],'selected':selection['head_selections'][key]==cid,
                'independent_selected':selection['head_selections'][key]==cid,
                'selected_within_family':selection['family_selections'][key][c.metadata['family']]==cid,
                'auc_selected':selection['auroc_selections'][key]==cid,
                'validation':val[key,cid]['score'],'validation_person_weighted':val[key,cid]['weighted'],
                'test':evaluation[key,cid]['score'],'test_person_weighted':evaluation[key,cid]['weighted']})
    np.savez_compressed(directory/'release_PCA16.npz',**releases,test=test)
    np.savez_compressed(directory/'predictions.npz',**predictions)
    np.savez_compressed(directory/'split_rows.npz',**{p:frame.iloc[i]._raw_row.to_numpy() for p,i in pools.items()})
    integrity={'pca_unchanged':joblib.hash(pca)==pca_before,
        'releases_unchanged':{p:array_hash(x) for p,x in releases.items()}==freeze['output_hashes'],
        'source_files_unchanged':all(sha_file(ROOT/p)==h for p,h in used.items()),
        'selection_unchanged':sha_file(directory/'selection_before_test.json')==selection_hash,
        'selection_created_utc':selection['created_utc'],'evaluation_started_utc':evaluated_utc,
        'test_output_sha256':array_hash(test)}
    assert all(v for k,v in integrity.items() if k.endswith('_unchanged'))
    write_json(directory/'provenance.json',{'used_local_artifacts_sha256':used,
        'reference_metrics_path':str((refdir/'metrics.json').relative_to(ROOT)),
        'reference_metrics_sha256':sha_file(refdir/'metrics.json'),
        'reference_score_verification_path':str((reference/'SCORE_REPLAY.json').relative_to(ROOT)),
        'reference_source_and_maps':'No refits or regeneration. Original source and prior PCA/bottleneck provenance preserved.'})
    write_json(directory/'local_artifacts.json',[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha_file(p)}
        for p in directory.rglob('*') if p.is_file() and p.suffix in ('.npz','.pt','.joblib')])
    runtime={'total_seconds':time.perf_counter()-tick,'setup_and_verification_seconds':setup_seconds,
        'new_utility_and_audit_fitting_seconds':fitting_seconds,'evaluation_serialization_seconds':time.perf_counter()-t}
    result={'seed':seed,'evaluation_status':cfg['evaluation_status'],'raw_metrics':raw,'support_by_pool':support,
        'cohort':cohort,'release_metadata':{'PCA16':freeze},'integrity':integrity,'runtime':runtime}
    write_json(directory/'metrics.json',result);print({'seed':seed,'runtime':runtime,'integrity':integrity},flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    p.add_argument('--prepare',action='store_true');p.add_argument('--seeds',nargs='+',type=int)
    args=p.parse_args();out=args.out.resolve();cfg=read(out/'config.json');torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if args.prepare:prepare(out)
        for seed in args.seeds or []:
            assert seed in cfg['seeds'];run_seed(out,cfg,seed)
if __name__=='__main__':main()
