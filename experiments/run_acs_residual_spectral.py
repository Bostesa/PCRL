"""Resumable residual spectral study with an explicit global reserved-label gate."""
from __future__ import annotations
import argparse,datetime,json,time,platform,subprocess,copy
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from experiments import acs_spectral_audits as audit
from experiments import acs_fixed_predictions_audits as oldaudit
from experiments import run_acs_fixed_predictions as fixed
from experiments import run_acs_coalition as hist
from experiments.acs_transfer_data import sha_file,array_hash,write_json,load_cohort,split_households,audit_labels,support
from experiments.run_acs_transfer import subset_indices
from experiments.run_acs_protection import TASKS
ROOT=Path(__file__).resolve().parents[1]
HIST_ROOT=Path('/Users/nathansamson/PCRL')
NAME='redesign_20260910_acs_residual_spectral_v1'
OUT=ROOT/'results'/NAME
ARMS=tuple('spectral_'+n for n in ('S0','M025','M1','L025','L1','C025','C1','L2'))
HIST=('H','E','A0','L025','L20','J')

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def arrays(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def parent(root):return root/'results/redesign_20260909_acs_fixed_predictions_v1'
def wires(path):
    z=arrays(path);w={v:{} for v in oldaudit.VIEWS};d={v:{} for v in oldaudit.VIEWS}
    for key,x in z.items():
        space,v,p=key.split('/');(w if space=='wire' else d)[v][p]=x
    return w,d if d['A'] else None

def fit_maps(out,root):
    from experiments.acs_residual_spectral import fit_spectral
    if (out/'RELEASE_MANIFEST.json').exists():return
    if not (out/'PREFIT_FREEZE.json').exists():
        write_json(out/'PREFIT_FREEZE.json',{'created_utc':now(),'protocol_sha256':sha_file(out/'PROTOCOL.md'),'source_hashes':{str(p.relative_to(ROOT)):sha_file(p) for p in [ROOT/'experiments/acs_residual_spectral.py',ROOT/'experiments/acs_spectral_audits.py',ROOT/'experiments/run_acs_residual_spectral.py']}})
    else: assert read(out/'PREFIT_FREEZE.json')['protocol_sha256']==sha_file(out/'PROTOCOL.md')
    historical=parent(root);rawpath=root/'data/folktables/2018/1-Year/psam_p06.csv'
    # No residence/commute columns are loaded by this representation stage.
    raw=pd.read_csv(rawpath,usecols=['SEX','RAC1P','PUBCOV','SERIALNO'],dtype={'SERIALNO':str})
    for seed in (0,1,2):
        dest=out/f'seed_{seed}'; dest.mkdir(parents=True,exist_ok=True)
        if (dest/'maps_complete.json').exists():
            prior=read(dest/'maps_complete.json')
            assert prior['prefit_freeze_sha256']==sha_file(out/'PREFIT_FREEZE.json')
            for name,digest in prior['files'].items():assert sha_file(dest/name)==digest
            continue
        tick=time.perf_counter(); p=historical/f'seed_{seed}'; rows=arrays(p/'split_rows.npz');t=arrays(p/'pca.npz');a=arrays(p/'anchors.npz')
        frame=raw.iloc[rows['representation_fit']]
        labels=audit_labels(frame);pub=frame.PUBCOV.to_numpy(float)
        labels['public_coverage']=np.where(np.isin(pub,[1,2]),(pub==1).astype(int),-1)
        model,diagnostics=fit_spectral(t['representation_fit'],a['representation_fit/A'],a['representation_fit/B'],labels,frame.SERIALNO.to_numpy(),seed)
        assert set(model.maps)==set(ARMS)
        joblib.dump(model,dest/'maps.joblib')
        parity=[]
        for arm in ARMS:
            cache={}
            for pool,x in t.items():
                ha=a[pool+'/A'];hb=a[pool+'/B'];z=model.transform(x,ha,arm)
                wa=np.column_stack((ha,z));wab=np.column_stack((wa,hb))
                assert wa.dtype==np.float64 and np.array_equal(wa[:,:4],ha) and np.array_equal(wab[:,-2:],hb)
                for v,value in [('A',wa),('B',hb),('AB',wab)]:cache['wire/'+v+'/'+pool]=value
                parity.append({'arm':arm,'pool':pool,'anchor_A_exact':True,'anchor_B_exact':True,'dtype':str(wa.dtype),'rank':z.shape[1]})
            path=dest/'releases'/arm;path.mkdir(parents=True,exist_ok=True);np.savez_compressed(path/'releases.npz',**cache)
        write_json(dest/'matrix_diagnostics.json',diagnostics)
        write_json(dest/'anchor_parity.json',parity)
        write_json(dest/'maps_complete.json',{'created_utc':now(),'runtime_seconds':time.perf_counter()-tick,'maps':8,'prefit_freeze_sha256':sha_file(out/'PREFIT_FREEZE.json'),'nuisance_fits':15,'reserved_columns_loaded':False,
            'files':{str(p.relative_to(dest)):sha_file(p) for p in [dest/'maps.joblib',dest/'matrix_diagnostics.json',*dest.glob('releases/*/*.npz')]},
            'historical_inputs':{str(p.relative_to(root)):sha_file(p) for p in [p/'split_rows.npz',p/'pca.npz',p/'anchors.npz']}})
        print('MAPS_FROZEN',seed,round(time.perf_counter()-tick,2),flush=True)
    write_json(out/'RELEASE_MANIFEST.json',{'created_utc':now(),'maps':24,'all24_frozen':True,'training_closed':True,'reserved_new_evaluation_started':False,
        'protocol_sha256':sha_file(out/'PROTOCOL.md'),'seeds':{str(s):read(out/f'seed_{s}'/'maps_complete.json') for s in (0,1,2)},
        'source_hashes':{str(p.relative_to(ROOT)):sha_file(p) for p in [ROOT/'experiments/acs_residual_spectral.py',ROOT/'experiments/run_acs_residual_spectral.py',ROOT/'experiments/acs_spectral_audits.py']}})

def load_labels(root,seed):
    cfg=read(root/'results/redesign_20260907_acs_transfer_v1/config.json')
    frame,cohort=load_cohort(root/cfg['raw_path'],cfg['sample_cap'],cfg['sample_seed']);pools=split_households(frame,seed)
    with np.load(parent(root)/f'seed_{seed}'/'split_rows.npz') as z:
        for p,ix in pools.items(): assert np.array_equal(frame.iloc[ix]._raw_row.to_numpy(),z[p])
    # Fixed binary task definitions do not depend on income-bin cutpoints.
    labels={p:hist.all_labels(frame.iloc[ix],[]) for p,ix in pools.items() if p!='representation_fit'}
    weights={p:frame.iloc[pools[p]].PWGTP.to_numpy(float) for p in labels}
    return frame,pools,labels,weights

def score(out,seed,condition,w,d,u,attacks,labels,weights,oldpath=None):
    """Reuse exact historical candidate scores, recompute new predictions only."""
    historical_rows={};historical_preds=None
    if oldpath:
        historical_rows={(r['role'],r['view'],r['target'],r['audit_budget'],r['candidate_id']):r for r in read(oldpath/'metrics.json')['raw_metrics']}
        historical_preds=np.load(oldpath/'predictions.npz')
    records=[];preds={};cache={}
    def add(role,view,target,b,cid,c,chosen):
        key=(role,view,target,b,cid); saved=historical_rows.get(key); vals={}
        if saved:
            for split in ('validation','test'):
                pk=f'{role}/{view}/{target}/{b}/{cid}/{split}';preds[pk]=historical_preds[pk]
            records.append({**saved,'selected_scopes':chosen});return
        for split in ('validation','test'):
            pool=('downstream_validation' if role=='utility' else 'attacker_validation') if split=='validation' else 'test'
            y=labels[pool][target];valid=y>=0; bundle={'wire':w[view][pool][valid]}
            if d is not None:bundle['derived']=d[view][pool][valid]
            # Cache identical raw inputs / base candidate across inherited routes.
            if role=='audit':
                actual=bundle[c.space];actual=actual[:,c.columns] if c.columns is not None else actual
                actual=np.ascontiguousarray(actual)
                ident=(c.metadata['base_candidate_directory'],pool,array_hash(actual))
                if ident not in cache:cache[ident]=c.base.predict_proba(actual)
                prediction=cache[ident]
            else: prediction=c.predict_proba(bundle['wire'])
            pk=f'{role}/{view}/{target}/{b}/{cid}/{split}';preds[pk]=prediction
            from experiments.acs_transfer_heads import metrics
            vals[split]=metrics(y[valid],prediction,oldaudit.CLASSES[target]);vals[split+'_person_weighted']=metrics(y[valid],prediction,oldaudit.CLASSES[target],weights[pool][valid])
        records.append({'seed':seed,'condition':condition,'role':role,'view':view,'target':target,'audit_budget':b,'candidate_id':cid,'selected_scopes':chosen,'scores':vals})
    for role,cs in u['candidates'].items():
        v,t=role.split('/')
        for cid,c in cs.items():add('utility',v,t,None,cid,c,['utility'] if u['selection'][role]==cid else [])
    for b,roles in attacks['candidates'].items():
        for role,cs in roles.items():
            v,t=role.split('/')
            for cid,c in cs.items():add('audit',v,t,b,cid,c,[scope for scope,selected in attacks['selection'][b][role].items() if selected==cid])
    if historical_preds is not None:historical_preds.close()
    write_json(out/'metrics.json',{'raw_metrics':records,'evaluation_status':'DEVELOPMENT EVALUATION'})
    np.savez_compressed(out/'predictions.npz',**preds)


def reusable_utilities(w,labels,ti,seed,dest,h):
    from experiments.acs_transfer_heads import fit_candidates, _hash_array
    result={'candidates':{},'selection':{},'metadata':{}}
    hrecord=read(h/'selection_before_test.json')
    for view,targets in hist.AUTHORIZED.items():
        for target in targets:
            role=view+'/'+target; ix=ti[target]; valid=labels['downstream_validation'][target]>=0
            xf=w[view]['downstream_fit'][ix]; yf=labels['downstream_fit'][target][ix]
            xv=w[view]['downstream_validation'][valid]; yv=labels['downstream_validation'][target][valid]
            cs={}
            for cid in ('logistic','mlp'):
                path=(h if view=='B' else dest)/'fitted/utility'/view/target/cid
                if (path/'metadata.json').exists(): candidate=audit.load_base(path)
                else:
                    if path.exists():path.rename(path.with_name('incomplete_'+cid+'_'+str(time.time_ns())))
                    fitted=fit_candidates(xf,yf,xv,yv,2,1250000+100*seed+TASKS.index(target),families=(cid,),budget={'mlp':{'epochs':40}})
                    candidate=fitted['candidates'][cid];candidate.save(path)
                assert candidate.metadata['fit_hashes']=={'x':_hash_array(xf),'y':_hash_array(yf)}
                assert candidate.metadata['validation_hashes']=={'x':_hash_array(xv),'y':_hash_array(yv)}
                cs[cid]=candidate
            result['candidates'][role]=cs
            result['selection'][role]=min(cs,key=lambda k:(cs[k].metadata['validation_scores']['log_loss'],k))
            result['metadata'][role]={'candidates':{cid:c.metadata for cid,c in cs.items()},'B_alias':str(h) if view=='B' else None}
    return result

def verify_gate(out):
    gate=read(out/'RELEASE_MANIFEST.json')
    assert gate['all24_frozen'] and sha_file(out/'PROTOCOL.md')==gate['protocol_sha256']
    core='experiments/acs_residual_spectral.py'
    assert sha_file(ROOT/core)==gate['source_hashes'][core]
    for s,rec in gate['seeds'].items():
        for name,digest in rec['files'].items():assert sha_file(out/f'seed_{s}'/name)==digest
    return gate

def evaluate_seed(out,root,seed,max_units=None):
    gate=verify_gate(out)
    for s,rec in gate['seeds'].items():
        for name,digest in rec['files'].items(): assert sha_file(out/f'seed_{s}'/name)==digest
    historical=parent(root);frame,pools,labels,weights=load_labels(root,seed)
    ti={t:subset_indices(labels['downstream_fit'][t],2048,1230000+100*seed+j) for j,t in enumerate(TASKS)}
    ai={t:subset_indices(labels['attacker_fit'][t],4096,1240000+100*seed+j) for j,t in enumerate(hist.TARGETS)}
    oldix=read(historical/f'seed_{seed}'/'indices.json')
    assert oldix=={'utility':{t:array_hash(ix) for t,ix in ti.items()},'attacker':{t:array_hash(ix) for t,ix in ai.items()}}
    write_json(out/f'seed_{seed}'/'support.json',{p:support(y,hist.TARGETS) for p,y in labels.items()})
    write_json(out/f'seed_{seed}'/'indices.json',oldix)
    moment_path=out/f'seed_{seed}'/'heldout_moments.json'
    if not moment_path.exists():
        model=joblib.load(out/f'seed_{seed}'/'maps.joblib');t=arrays(historical/f'seed_{seed}'/'pca.npz');a=arrays(historical/f'seed_{seed}'/'anchors.npz')
        write_json(moment_path,{p:model.moment_diagnostics(t[p],a[p+'/A'],a[p+'/B'],{k:labels[p][k] for k in ('SEX','RAC1P','public_coverage')}) for p in ('source_validation','test')})
    np.savez_compressed(out/f'seed_{seed}'/'evaluation_labels.npz',**{p+'/'+t:y for p,ys in labels.items() for t,y in ys.items()},**{'weights/'+p:w for p,w in weights.items()})
    done=0
    for condition in HIST+ARMS:
        dest=out/f'seed_{seed}'/condition
        if (dest/'complete.json').exists():
            complete=read(dest/'complete.json')
            assert sha_file(dest/'predictions.npz')==complete['prediction_sha256']
            assert sha_file(dest/'metrics.json')==complete['metric_sha256']
            continue
        tick=time.perf_counter();dest.mkdir(parents=True,exist_ok=True); print('EVALUATE',seed,condition,flush=True)
        hp=historical/f'seed_{seed}'/condition if condition in HIST else None
        release=historical/f'seed_{seed}'/'training'/condition/'releases.npz' if hp else out/f'seed_{seed}'/'releases'/condition/'releases.npz'
        w,d=wires(release)
        with np.load(historical/f'seed_{seed}'/'anchors.npz') as a:
            for p,x in w['A'].items():
                assert np.array_equal(x[:,:4],a[p+'/A']) and np.array_equal(w['B'][p],a[p+'/B'])
        if hp:
            us=read(hp/'selection_before_test.json');u={'selection':us['utility'],'metadata':us['utility_metadata'],'candidates':{}}
            for role in u['selection']:
                utility_parent=historical/f'seed_{seed}'/'H' if role.startswith('B/') else hp
                u['candidates'][role]={cid:audit.load_base(utility_parent/'fitted/utility'/role/cid) for cid in ('logistic','mlp')}
            ha=audit.load_audits(hp/'audits')
        else:
            up=dest/'utility_state.joblib'
            if up.exists():u=joblib.load(up)
            else:
                u=reusable_utilities(w,labels,ti,seed,dest,historical/f'seed_{seed}'/'H');joblib.dump(u,up)
            ha=None
        ancestor=None if condition=='H' else audit.load_audits(out/f'seed_{seed}'/'H'/'audits')
        fit=lambda v:{view:{p:x for p,x in values.items() if p in oldaudit.FIT_POOLS} for view,values in v.items()}
        attacks=audit.build_audits(fit(w),fit(d) if d else None,{p:labels[p] for p in oldaudit.FIT_POOLS},ai,seed,dest/'audits',historical=ha,ancestor=ancestor)
        write_json(dest/'selection_before_test.json',{'created_utc':now(),'utility':u['selection'],'utility_metadata':u['metadata'],'audits':attacks['selection'],'release_manifest_sha256':sha_file(out/'RELEASE_MANIFEST.json')})
        score(dest,seed,condition,w,d,u,attacks,labels,weights,hp)
        write_json(dest/'complete.json',{'created_utc':now(),'runtime_seconds':time.perf_counter()-tick,'historical_fits':0,'new_utility_fits':0 if hp else 6,
            'audit_counts':attacks['metadata']['counts'],'prediction_sha256':sha_file(dest/'predictions.npz'),'metric_sha256':sha_file(dest/'metrics.json')})
        print('COMPLETE',seed,condition,round(time.perf_counter()-tick,2),flush=True)
        done+=1
        if max_units and done>=max_units:return

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);p.add_argument('--historical-root',type=Path,default=HIST_ROOT)
    p.add_argument('--phase',choices=['fit','evaluate'],required=True);p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);p.add_argument('--max-units',type=int);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True); start=time.perf_counter();torch.set_num_threads(1)
    error=None
    try:
        with threadpool_limits(limits=1):
            if a.phase=='fit':fit_maps(a.out,a.historical_root)
            else:
                for seed in a.seeds:evaluate_seed(a.out,a.historical_root,seed,a.max_units)
    except BaseException as exc:
        error=repr(exc)
        raise
    finally:
        events=read(a.out/'RUNTIME_EVENTS.json') if (a.out/'RUNTIME_EVENTS.json').exists() else []
        events.append({'phase':a.phase,'ended_utc':now(),'elapsed_seconds':time.perf_counter()-start,'seeds':a.seeds,'error':error})
        write_json(a.out/'RUNTIME_EVENTS.json',events)
if __name__=='__main__':main()
