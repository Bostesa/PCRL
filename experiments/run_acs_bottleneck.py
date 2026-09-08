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
from experiments.acs_bottleneck_training import train_pair
from experiments.acs_bottleneck_catchup import fit_catchup

DEFAULT_OUT = ROOT/'results/redesign_20260908_acs_bottleneck_v1'
REFERENCES = ('E_pca','E_pca_leace','B_rich_bank','B_rich_bank_leace','C_tree_bank','C_tree_bank_leace')
LEARNED = ('C_bottleneck','D_protected')
SOURCE_TASKS = ('income_binary','civilian_at_work','public_coverage')


def source_binaries(frame, edges):
    # Deliberately restrict actual columns as well as the return-key schema.
    y = source_labels(frame.loc[:, ['PINCP','ESR','PUBCOV']], edges)
    return {'income_binary':y['income_binary'],
            'civilian_at_work':np.where(y['esr']<0,-1,(y['esr']==0).astype(np.int64)),
            'public_coverage':np.where(y['pubcov']<0,-1,(y['pubcov']==0).astype(np.int64))}


def execution_paths():
    return list(dict.fromkeys([*old_paths(),*(ROOT/'experiments'/name for name in
        ('run_acs_bottleneck.py','acs_bottleneck_training.py','acs_bottleneck_catchup.py'))]))


def frozen_hashes(out):
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha_file(p)
            for p in [out/'config.json',out/'PROTOCOL.md',*execution_paths()]}


def check_config(cfg):
    expected={'input_dim':32,'mapper_hidden':64,'release_dim':16,'decoder_hidden':64,
        'adversary_hidden':[64,32],'source_weight':1.,'reconstruction_weight':.1,
        'protection_weight':.1,'learning_rate':.001,'batch_size':256,'warmup_epochs':60,
        'adversary_warmup_epochs':20,'continuation_epochs':80,'adversary_updates_per_mapper':3,
        'checkpoint':'fixed_final_epoch','prior_entropy':'empirical fitting labels, no smoothing',
        'optimizer':'Adam defaults, no weight decay or clipping'}
    assert cfg['training']==expected, 'Fixed scientific training configuration changed'
    assert tuple(cfg['references'])==REFERENCES and tuple(cfg['learned_arms'])==LEARNED
    assert tuple(cfg['utility_tasks'])==TASKS and cfg['seeds']==[0,1,2] and cfg['threads']==1
    for k,v in {'head_budget':2048,'attacker_budget':4096,'head_mlp_epochs':40,
                'audit_mlp_epochs':120,'catchup_epochs':120,'audit_restart_offset':10000,
                'audit_min_samples_leaf':[20,5]}.items():
        assert cfg[k]==v, 'Fixed scientific evaluation configuration changed: '+k
    assert cfg['margins']==read(ROOT/cfg['reference_results']/'config.json')['margins']


def prepare(out):
    cfg=read(out/'config.json');check_config(cfg)
    files=[]
    for name in ('parent_results','reference_results'):
        parent=ROOT/cfg[name]
        files += [parent/k for k in ('config.json','PROTOCOL.md','protocol_freeze.json')]
        for seed in cfg['seeds']:
            files += [parent/f'seed_{seed}'/k for k in
                ('metrics.json','selection_before_test.json','local_artifacts.json')]
    freeze={'created_utc':now(),'starting_commit':cfg['starting_commit'],
        'sha256':frozen_hashes(out),'reference_record_hashes':{str(p.relative_to(ROOT)):sha_file(p) for p in files},
        'evaluation_status':cfg['evaluation_status']}
    if (out/'protocol_freeze.json').exists(): raise FileExistsError('Preserve existing protocol freeze')
    write_json(out/'protocol_freeze.json',freeze)
    previous=read(ROOT/cfg['reference_results']/'environment.json')
    write_json(out/'environment.json',{'created_utc':now(),'platform':platform.platform(),
        'hardware':previous['hardware'],'threads':1,'python':platform.python_version(),
        'packages':{k:importlib.metadata.version(k) for k in previous['packages']}})


def verify_freeze(out):
    cfg=read(out/'config.json');check_config(cfg);freeze=read(out/'protocol_freeze.json')
    assert frozen_hashes(out)==freeze['sha256'], 'Execution/protocol changed after freeze'
    for p,h in freeze['reference_record_hashes'].items():
        assert sha_file(ROOT/p)==h, 'Reference record changed: '+p


class SavedReferences:
    """Reuse original immutable maps, arrays and recorded candidate predictions."""
    def __init__(self, cfg, seed, frame, pools):
        self.source=ParentBundle(ROOT/cfg['parent_results'],seed,frame,pools)
        self.directory=ROOT/cfg['reference_results']/f'seed_{seed}'
        self.previous=read(self.directory/'metrics.json')
        self.selection=read(self.directory/'selection_before_test.json')
        self.freeze=read(self.directory/'release_freeze.json')
        self.manifest={str(ROOT/v['path']):v['sha256'] for v in read(self.directory/'local_artifacts.json')}
        self.used_files={};self.releases={};self.maps={};self.checked_candidates=0
        rows=np.load(self.checked(self.directory/'split_rows.npz'))
        for p,idx in pools.items():
            assert np.array_equal(rows[p],frame.iloc[idx]._raw_row.to_numpy()), 'Original household split changed'
        oldcfg=read(self.directory.parent/'config.json')
        for key in ('head_budget','attacker_budget','head_mlp_epochs','audit_mlp_epochs',
                    'audit_restart_offset','audit_min_samples_leaf','utility_tasks'):
            assert cfg[key]==oldcfg[key], 'Reference recipe mismatch: '+key
        oldfreeze=read(self.directory.parent/'protocol_freeze.json')
        for p in old_paths():
            name=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
            assert sha_file(p)==oldfreeze['sha256'][name], 'Original scientific dependency changed'
        for name in REFERENCES:
            with np.load(self.checked(self.directory/f'release_{name}.npz')) as z:
                self.releases[name]={p:z[p].copy() for p in z.files if p!='test'}
            for pool,x in self.releases[name].items():
                assert array_hash(x)==self.freeze['output_hashes'][name][pool]
                x.setflags(write=False)
            if name.endswith('_leace'):
                parent=name.removesuffix('_leace')
                eraser=FrozenAffineMap.load(self.checked(self.directory/f'map_{parent}.npz'))
                assert eraser.fingerprint()==self.freeze['map_hashes'][parent]
                self.maps[parent]=eraser
            else:
                for pool,x in self.releases[name].items():
                    assert np.array_equal(x,self.source.releases[name][pool])
        for parent,e in self.maps.items():
            for pool,x in self.releases[parent].items():
                assert np.array_equal(e.apply(x),self.releases[parent+'_leace'][pool])
        self.checked(self.directory/'predictions.npz')
        self.initial=self.fingerprint()

    def checked(self,path):
        h=sha_file(path)
        assert self.manifest.get(str(path))==h, 'Missing or changed reference object: '+str(path)
        self.used_files[str(path.relative_to(ROOT))]=h
        return path

    def fingerprint(self):
        return {'source':self.source.fingerprint(),'erasers':{n:e.fingerprint() for n,e in self.maps.items()},
                'outputs':frozen_digest(self.releases)}

    def evaluation_releases(self,test_frame):
        # Only invoked after new head/audit selections are saved.
        parent=self.source.evaluation_releases(test_frame)
        test={n:parent[n] for n in REFERENCES if not n.endswith('_leace')}
        test.update({n+'_leace':e.apply(test[n]) for n,e in self.maps.items()})
        for n,x in test.items():
            assert array_hash(x)==self.previous['integrity']['evaluation_output_hashes'][n]
            with np.load(self.directory/f'release_{n}.npz') as z:
                assert np.array_equal(x,z['test'])
            x.setflags(write=False)
        return test

    def verify_reuse(self, fit_y, val_y, audit_y, audit_v, ti, ai, fit_records):
        """Check fitting rows, standardizers, selected states and validation outputs.

        No reference model is refit. Test probabilities remain unopened here.
        """
        for key,record in self.selection['fitting_records'].items():
            role,name,target=key.split('/')
            if name not in (*REFERENCES,'prior','exposed'): continue
            y,v,idx=(fit_y,val_y,ti) if role=='transfer' else (audit_y,audit_v,ai)
            yf=y[target][idx[target]];valid=v[target]>=0;yv=v[target][valid]
            k=2 if role=='transfer' else AUDITS[target]
            fp,vp=('downstream_fit','downstream_validation') if role=='transfer' else ('attacker_fit','attacker_validation')
            if name=='prior': xf=np.zeros((len(yf),1));xv=np.zeros((len(yv),1))
            elif name=='exposed': xf=np.eye(k)[yf];xv=np.eye(k)[yv]
            else: xf=self.releases[name][fp][idx[target]];xv=self.releases[name][vp][valid]
            for cid,metadata in record['candidates'].items():
                path=self.directory/'fitted'/key/cid
                for artifact in path.iterdir():
                    if artifact.suffix in ('.npz','.joblib','.pt'): self.checked(artifact)
                candidate=load_candidate(path)
                assert candidate.metadata==metadata, 'Reference candidate metadata changed'
                assert metadata['fit_rows']==len(yf)
                if name=='prior': assert metadata['fit_label_hash']==_hash_array(yf)
                else:
                    assert metadata['fit_hashes']=={'x':_hash_array(np.asarray(xf,dtype=np.float64)),'y':_hash_array(yf)}
                    assert metadata['validation_hashes']=={'x':_hash_array(np.asarray(xv,dtype=np.float64)),'y':_hash_array(yv)}
                    if candidate.preprocessing is not None:
                        mean=np.asarray(xf,dtype=np.float64).mean(0);std=np.asarray(xf,dtype=np.float64).std(0)
                        assert np.array_equal(mean,candidate.preprocessing.mean)
                        assert np.array_equal(np.where(std>1e-12,std,1.),candidate.preprocessing.scale)
                    if candidate.family=='mlp': assert _state_hash(candidate.model)==metadata['selected_state_hash']
                with np.load(self.directory/'predictions.npz') as probs:
                    assert np.array_equal(candidate.predict_proba(xv),probs[f'validation/{key}/{cid}'])
                self.checked_candidates+=1
        assert fit_records['task_fit_indices']==self.selection['task_fit_indices']
        assert fit_records['audit_fit_indices']==self.selection['audit_fit_indices']


def run_seed(out,cfg,seed,*,miniature=False):
    if not miniature: verify_freeze(out)
    tick=time.perf_counter();directory=out/f'seed_{seed}';directory.mkdir(exist_ok=False)
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
    pair=train_pair(pca['representation_fit'],source_binaries(frame.iloc[pools['representation_fit']],edges),
        audit_labels(frame.iloc[pools['representation_fit']]),pca['source_validation'],
        source_binaries(frame.iloc[pools['source_validation']],edges),seed,directory/'training',miniature=miniature)
    training_seconds=time.perf_counter()-t
    releases={n:{p:arm['model'].release(x) for p,x in pca.items()} for n,arm in pair['arms'].items()}
    for arrays in releases.values():
        for a in arrays.values(): a.setflags(write=False)
    def learned_state():
        return {n:{'model':_state_hash(arm['model']),
                   'adversaries':_state_hash(arm['adversaries'])} for n,arm in pair['arms'].items()}
    before=learned_state();outputs=frozen_digest(releases)
    release_meta={n:copy.deepcopy(refs.previous['release_metadata'][n]) for n in REFERENCES}
    for n,a in releases.items():
        release_meta[n]={'parent':'E_pca','erased':False,'dimension':16,'dtype':'float32',
            'bytes_per_record':64,'training_arm':n,**feature_rank(a['representation_fit'])}
    write_json(directory/'release_freeze.json',{'created_utc':now(),'release_metadata':release_meta,
        'learned_state':before,'output_hashes':outputs,'reference_state':refs.initial,
        'fit_pool':'representation_fit','fit_raw_row_sha256':original_support['representation_fit']['raw_row_sha256'],
        'reserved_labels_entered_release_fitting':False,'final_epoch_selection':'fixed80; no validation selection'})
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
        print(f'seed {seed}: {release}: final utility and independent audits',flush=True)
        t=time.perf_counter()
        for j,target in enumerate(TASKS):
            i=ti[target];vi=np.flatnonzero(val_y[target]>=0)
            accept(f'transfer/{release}/{target}',fit_candidates(arrays['downstream_fit'][i],fit_y[target][i],
                arrays['downstream_validation'][vi],val_y[target][vi],2,1250000+100*seed+j,
                budget={'mlp':{'epochs':1 if miniature else cfg['head_mlp_epochs']}}))
        task_seconds+=time.perf_counter()-t
        for j,(target,k) in enumerate(AUDITS.items()):
            i=ai[target];vi=np.flatnonzero(audit_v[target]>=0);t=time.perf_counter()
            result=fit_primary_auditors(arrays['attacker_fit'][i],audit_y[target][i],
                arrays['attacker_validation'][vi],audit_v[target][vi],k,1260000+100*seed+j,budget=audit_budget)
            audit_seconds+=time.perf_counter()-t;t=time.perf_counter()
            train_meta=pair['metadata'];arm_meta=train_meta['arms'][release]
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
                'final_adversary_state_hash':_state_hash(pair['arms'][release]['adversaries'][target]),
                'continuation_schedule_hash':arm_meta['schedule_hash']}
            caught=fit_catchup(pair['arms'][release]['adversaries'][target],arrays['attacker_fit'][i],audit_y[target][i],
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
    provenance={'created_utc':now(),'used_reference_files_sha256':refs.used_files,
        'used_original_files_sha256':refs.source.used_files,'original_source_selection':refs.source.selection,
        'reference_metrics_path':str((refs.directory/'metrics.json').relative_to(ROOT)),
        'reference_metrics_sha256':sha_file(refs.directory/'metrics.json'),
        'reference_prediction_reuse':'Verified fitting rows/labels, recipe, exact standardizers, fitted-state and validation probability replay; original predictions/metrics retained.',
        'verified_reference_candidates':refs.checked_candidates,'regeneration':'None; frozen inference only; no PCA, bank, encoder or eraser refit'}
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
    test={n:arm['model'].release(ref_test['E_pca']) for n,arm in pair['arms'].items()}
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
    np.savez_compressed(directory/'predictions.npz',**predictions)
    for n,a in releases.items(): np.savez_compressed(directory/f'release_{n}.npz',**a,test=test[n])
    integrity={'learned_states_unchanged':before==learned_state(),
        'new_releases_unchanged':outputs==frozen_digest(releases),
        'reference_states_unchanged':refs.initial==refs.fingerprint(),
        'reference_files_unchanged':all(sha_file(ROOT/p)==h for p,h in {**refs.used_files,**refs.source.used_files}.items()),
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
    result={'seed':seed,'evaluation_status':cfg['evaluation_status'],'cohort':cohort,'raw_metrics':raw,
        'release_metadata':release_meta,'support_by_pool':original_support,'integrity':integrity,'runtime':runtime}
    write_json(directory/'metrics.json',result)
    print(json.dumps({'seed':seed,'runtime':runtime,'integrity':integrity}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    p.add_argument('--prepare',action='store_true');p.add_argument('--seeds',type=int,nargs='+')
    args=p.parse_args();out=args.out.resolve();cfg=read(out/'config.json');torch.set_num_threads(cfg['threads'])
    with threadpool_limits(limits=cfg['threads']):
        if args.prepare: prepare(out)
        for seed in args.seeds or []:
            assert seed in cfg['seeds'];run_seed(out,cfg,seed)


if __name__=='__main__': main()
