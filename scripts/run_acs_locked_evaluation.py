"""Separated, resumable locked-evaluation phases. Never invokes training runners.

The published V1 has an empty defensible remainder; infer/score/uncertainty
therefore refuse to consume any outcome. Reproduction is freeze/verify only.
"""
from __future__ import annotations
import argparse, datetime, json, platform, subprocess, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits, threadpool_info
from experiments import acs_locked_evaluation as e
from experiments.acs_transfer_data import FEATURES,sha_file,array_hash
from experiments.acs_protection_maps import FrozenAffineMap
from experiments.acs_transfer_heads import load_candidate
from scripts.inventory_acs_locked_objects import ROOT,PARENT,OUT,read


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def source_hashes():
    paths=[ROOT/'experiments/acs_locked_evaluation.py',ROOT/'scripts/run_acs_locked_evaluation.py',
           ROOT/'scripts/inventory_acs_locked_objects.py',ROOT/'scripts/verify_acs_locked_independence.py',
           ROOT/'scripts/verify_acs_locked_objects.py',ROOT/'tests/test_acs_locked_evaluation.py',ROOT/'tests/test_acs_locked_independence.py',
           ROOT/'experiments/acs_transfer_data.py',ROOT/'experiments/acs_transfer_heads.py',
           ROOT/'experiments/acs_protection_maps.py',ROOT/'experiments/acs_transfer_models.py',
           ROOT/'scripts/verify_acs_bottleneck_scores.py',ROOT/'scripts/verify_acs_fixed_predictions_comparisons.py',
           ROOT/'scripts/summarize_acs_protection.py']
    return {str(p.relative_to(ROOT)):sha_file(p) for p in paths}


def verify_objects(objects):
    for path,r in objects['files'].items():
        if sha_file(ROOT/path)!=r['sha256']:raise ValueError('Frozen object changed: '+path)


def freeze(out):
    if (out/'LOCK.json').exists():return verify(out)
    data=read(out/'DATA_USE_INVENTORY.json');objects=read(out/'FROZEN_OBJECTS.json')
    assert objects['counts']['systems']==18 and objects['counts']['audit_selections']==1188
    assert objects['required_missing']==0
    verify_objects(objects)
    for r in read(out/'OBJECT_IDENTITY_VALIDATION.json')['checks']:
        assert sha_file(ROOT/r['selection_source'])==r['selection_sha256']
    lock={'created_utc':now(),'started_utc':read(out/'START.json')['started_utc'],
          'starting_head':read(out/'START.json')['starting_head'],'sample_rows':data['sample_rows'],
          'sample_households':data['sample_households'],'independence_verified':data['independence_verified'],
          'status':'locked_for_inference' if data['sample_rows'] and data['independence_verified'] else 'blocked_no_unused_households',
          'scientific_fits':0,'optimizer_updates':0,'candidate_reselections':0,'fresh_outcomes_scored':0,
          'analysis_source_hashes':source_hashes(),
          'records':{n:sha_file(out/n) for n in ('PROTOCOL.md','START.json','FROZEN_OBJECTS.json','DATA_USE_INVENTORY.json','DATA_INDEPENDENCE.md','OBJECT_IDENTITY_VALIDATION.json')},
          'local_files':data['local_files'],'bootstrap':{'replicates':2000,'rng_seed':20260910,'draw':'default_rng integers, household groups with replacement; same draws for all seeds/conditions/targets',
             'aggregation':'arithmetic mean of three per-seed metrics, no probability averaging','quantile':'linear',
             'contrasts':['residence H-J U','residence H-J PWGTP','AB SEX L025-J U','AB SEX L025-J PWGTP','AB SEX L20-J U','AB SEX L20-J PWGTP'],
             'audit_scope':'expanded_catchup','audit_budget':360},'environment':{'platform':platform.platform(),
             'python':platform.python_version(),'processor':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
             'threadpools':threadpool_info(),'torch_threads':torch.get_num_threads(),'inference_workers':1}}
    e.atomic_json(out/'LOCK.json',lock);print(lock['status']);return lock


def verify(out):
    lock=read(out/'LOCK.json')
    for n,h in lock['records'].items():assert sha_file(out/n)==h,n
    for n,h in lock['analysis_source_hashes'].items():assert sha_file(ROOT/n)==h,n
    for n,h in lock['local_files'].items():
        p=Path(n);p=p if p.is_absolute() else ROOT/p
        assert sha_file(p)==h,n
    for n,h in read(out/'DATA_USE_INVENTORY.json')['source_sha256'].items():
        assert sha_file(ROOT/n)==h,n
    for r in read(out/'OBJECT_IDENTITY_VALIDATION.json')['checks']:
        assert sha_file(ROOT/r['selection_source'])==r['selection_sha256']
    verify_objects(read(out/'FROZEN_OBJECTS.json'))
    return lock


def atomic_arrays(path,values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with np.load(path) as old:
            if set(old.files)!=set(values) or any(not np.array_equal(old[k],v,equal_nan=True) for k,v in values.items()):
                raise ValueError('Incompatible existing arrays: '+str(path))
        return
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**values)
    tmp.replace(path)


def sample_frame(out,with_labels=False):
    lock=verify(out);e.require_independence(lock)
    records=[json.loads(x) for x in (out/'local/sample_people.jsonl').read_text().splitlines()]
    cols=list(dict.fromkeys((*FEATURES,'SERIALNO','SPORDER','PWGTP',*(('PINCP','ESR','PUBCOV','MIG','JWMNP','SEX','RAC1P') if with_labels else ()))))
    f=pd.read_csv(ROOT/'data/folktables/2018/1-Year/psam_p06.csv',usecols=cols,dtype={'SERIALNO':str,'SPORDER':str})
    raw_rows=np.array([r['_raw_row'] for r in records]);f=f.iloc[raw_rows].copy();f['_raw_row']=raw_rows
    f=e.canonical_people(f)
    assert f.person_id.tolist()==[r['person_id'] for r in records]
    assert f.household_id.tolist()==[r['household_id'] for r in records]
    assert len(f)==lock['sample_rows'];return f


def infer(out):
    lock=verify(out);e.require_independence(lock);frame=sample_frame(out);objects=read(out/'FROZEN_OBJECTS.json')
    candidate_cache={};seed_cache={};units=[]
    for system in objects['systems']:
        s,c=system['seed'],system['condition'];dest=out/'local'/f'seed_{s}'/c
        if (dest/'inference_complete.json').exists():
            done=read(dest/'inference_complete.json');assert done['lock_sha256']==sha_file(out/'LOCK.json')
            for p,h in done['files'].items():assert sha_file(ROOT/p)==h
            units.append(done);continue
        tick=time.perf_counter()
        if s not in seed_cache:
            pre=e.frozen_preprocessor(read(ROOT/system['upstream']['preprocessing.json']))
            maps=joblib.load(ROOT/system['upstream']['release_maps.joblib']);state_before=joblib.hash((pre,maps))
            x=pre.transform(frame);pca=maps['pca'].transform(x).astype(np.float32)
            anchors={}
            for t,p in system['anchors'].items():
                model=load_candidate(ROOT/p);before=joblib.hash(model)
                anchors[t]=model.predict_proba(pca);assert joblib.hash(model)==before
            seed_cache[s]=(pca,{'A':np.column_stack([anchors[t] for t in e.TASKS[:2]]),'B':anchors['public_coverage']})
            assert joblib.hash((pre,maps))==state_before
        pca,anchors=seed_cache[s]
        cp=torch.load(ROOT/system['checkpoint'],map_location='cpu',weights_only=True)
        teacher=FrozenAffineMap.load(ROOT/system['teacher']) if system['teacher'] else None
        from experiments.acs_bottleneck_training import tree_digest
        checkpoint_before=tree_digest(cp);teacher_before=teacher.fingerprint() if teacher else None
        graph=e.inference_graph(pca,anchors,teacher,cp,c)
        assert tree_digest(cp)==checkpoint_before
        assert (teacher.fingerprint() if teacher else None)==teacher_before
        arrays={f'{space}/{v}':x for space,z in graph.items() if z is not None for v,x in z.items()}
        atomic_arrays(dest/'releases.npz',arrays);predictions={};records=[];aliases={}
        for role,rs in [('utility',system['utilities']),('audit',system['audits']),('H_witness',system['H_witnesses'])]:
            for r in rs:
                v,t=r['role'].split('/');x=graph[r['space']][v]
                if r.get('projection_columns') is not None:x=x[:,r['projection_columns']]
                x=np.ascontiguousarray(x);path=r['path'];key=path+'|'+array_hash(x)
                pid=str(len(aliases)) if key not in aliases else aliases[key]
                if key not in aliases:
                    if path not in candidate_cache:candidate_cache[path]=load_candidate(ROOT/path)
                    model=candidate_cache[path];state_before=joblib.hash(model)
                    predictions[pid]=e.project_predict(model,x)
                    assert joblib.hash(model)==state_before
                    aliases[key]=pid
                records.append({'kind':role,**r,'prediction_key':pid,'input_hash':array_hash(x),'prediction_hash':array_hash(predictions[pid])})
        atomic_arrays(dest/'predictions.npz',predictions)
        e.atomic_json(dest/'prediction_records.json',records)
        done={'seed':s,'condition':c,'completed_utc':now(),'runtime_seconds':time.perf_counter()-tick,
              'lock_sha256':sha_file(out/'LOCK.json'),'records':len(records),'unique_outputs':len(predictions),
              'row_order_hash':array_hash(frame._raw_row.to_numpy()),'scientific_fits':0,'optimizer_updates':0,
              'files':{str((dest/n).relative_to(ROOT)):sha_file(dest/n) for n in ('releases.npz','predictions.npz','prediction_records.json')}}
        e.atomic_json(dest/'inference_complete.json',done);units.append(done)
    verify_objects(objects);e.atomic_json(out/'INFERENCE_COMPLETION.json',{'systems':units,'complete':len(units)==18})



def verify_inference_completion(out):
    completion=read(out/'INFERENCE_COMPLETION.json');assert completion['complete']
    units=completion['systems'];expected={(s,c) for s in range(3) for c in e.ARMS}
    assert len(units)==18 and {(u['seed'],u['condition']) for u in units}==expected
    for u in units:
        dest=out/'local'/f"seed_{u['seed']}"/u['condition']
        assert read(dest/'inference_complete.json')==u
        assert u['lock_sha256']==sha_file(out/'LOCK.json')
        for path,h in u['files'].items():assert sha_file(ROOT/path)==h,path
    return len(units)


def bootstrap_draws(path,n_households):
    expected=np.random.default_rng(20260910).integers(0,n_households,size=(2000,n_households),dtype=np.int32)
    if path.exists():
        with np.load(path) as a:actual=a['draws']
        assert actual.dtype==np.int32 and np.array_equal(actual,expected),'Changed deterministic bootstrap draws'
        return actual
    atomic_arrays(path,{'draws':expected});return expected


def score(out):
    from scripts.verify_acs_bottleneck_scores import labels
    lock=verify(out);e.require_independence(lock);verify_inference_completion(out)
    frame=sample_frame(out,True);weights=frame.PWGTP.to_numpy(float);truth={}
    for t in ('SEX','RAC1P',*e.TASKS):
        y,m=labels(frame,t);truth[t]=np.where(m,y,-1)
    atomic_arrays(out/'local/labels_weights.npz',{**truth,'PWGTP':weights,'household_group':np.unique(frame.household_id,return_inverse=True)[1]})
    objects=read(out/'FROZEN_OBJECTS.json');results=[];prior_scores={}
    for r in objects['controls']:
        if r['role']!='prior':continue
        p=load_candidate(ROOT/r['path']).predict_proba(np.zeros((len(frame),1)))
        z,ll=e.score_predictions(truth[r['target']],p,weights);prior_scores[r['seed'],r['target']]=z
    for system in objects['systems']:
        s,c=system['seed'],system['condition'];dest=out/'local'/f'seed_{s}'/c;records=read(dest/'prediction_records.json');ls={}
        with np.load(dest/'predictions.npz') as p:
            for i,r in enumerate(records):
                t=r['role'].split('/')[1];z,ll=e.score_predictions(truth[t],p[r['prediction_key']],weights);ls[str(i)]=ll
                row={'seed':s,'condition':c,**r,'scores':z}
                if r['kind'] in ('audit','H_witness'):
                    row['recovery_gain']={w:prior_scores[s,t][w]['log_loss']-z[w]['log_loss'] if z[w] else None for w in ('unweighted','PWGTP')}
                results.append(row)
        atomic_arrays(dest/'losses.npz',ls)
    e.atomic_json(out/'SCORES.json',results);verify_objects(objects)


def uncertainty(out):
    lock=verify(out);e.require_independence(lock);verify_inference_completion(out);records=read(out/'SCORES.json')
    with np.load(out/'local/labels_weights.npz') as z:
        weights=z['PWGTP'];groups=z['household_group'];res=z['same_residence']>=0;sex=z['SEX']>=0
    index={(r['seed'],r['condition'],r['kind'],r['role'],r.get('budget'),r.get('scope')):r for r in records}
    def loss(s,c,role,kind):
        key=(s,c,kind,role,360 if kind=='audit' else None,'expanded_catchup' if kind=='audit' else None);r=index[key]
        dest=out/'local'/f'seed_{s}'/c
        with np.load(dest/'predictions.npz') as p,np.load(out/'local/labels_weights.npz') as z:return e.losses(z[role.split('/')[1]],p[r['prediction_key']])
    differences=[]
    for s in range(3):
        r=loss(s,'H','A/same_residence','utility')-loss(s,'J','A/same_residence','utility')
        # Gain(local)-gain(J) = loss(J)-loss(local); same frozen prior cancels.
        a=loss(s,'J','AB/SEX','audit')-loss(s,'L025','AB/SEX','audit')
        b=loss(s,'J','AB/SEX','audit')-loss(s,'L20','AB/SEX','audit')
        differences.append(np.column_stack([r,r,a,a,b,b]))
    sums,den=e.household_sums(differences,groups,np.column_stack([res,res*weights,sex,sex*weights,sex,sex*weights]))
    drawpath=out/'local/bootstrap_draws.npz'
    draws=bootstrap_draws(drawpath,len(den))
    bs=e.bootstrap_values(sums,den,draws)
    with np.errstate(divide='ignore',invalid='ignore'):point=(sums.sum(1)/den.sum(0)).mean(0)
    atomic_arrays(out/'local/bootstrap_values.npz',{'replicates':bs,'original':point,'household_sums':sums,'household_denominators':den})
    r=e.intervals(point,bs);r['draws_sha256']=sha_file(drawpath);e.atomic_json(out/'UNCERTAINTY.json',r)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);p.add_argument('--phase',required=True,choices=['freeze','infer','score','uncertainty','verify']);a=p.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if a.phase=='freeze':freeze(a.out)
        elif a.phase=='verify':verify(a.out);print('Locked records and historical object hashes verified')
        else:globals()[a.phase](a.out)

if __name__=='__main__':main()
