"""Bounded fixed-source/A-auxiliary orchestration; old runners are immutable."""
from __future__ import annotations
import argparse,ast,copy,datetime,json,time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from experiments import run_acs_coalition as hist
from experiments import acs_fixed_predictions_training as train
from experiments import acs_fixed_predictions_audits as audit
from experiments.run_acs_selective import load_context
from experiments.acs_selective_teachers import load_teachers
from experiments.acs_transfer_data import array_hash,sha_file,write_json,audit_labels,support
from experiments.acs_transfer_heads import load_candidate,metrics,_hash_array,fit_candidates
from experiments.run_acs_transfer import subset_indices,save_candidates
from experiments.run_acs_bottleneck import source_binaries
from experiments.run_acs_protection import ROOT,TASKS,read,now
OUT=ROOT/'results/redesign_20260909_acs_fixed_predictions_v1'
CONDITIONS=('H','E','A0','L025','L20','J')

def closure():
    pending=[ROOT/p for p in ['experiments/run_acs_fixed_predictions.py','experiments/acs_fixed_predictions_training.py','experiments/acs_fixed_predictions_audits.py','scripts/run_acs_fixed_predictions_bounded.py','scripts/report_acs_fixed_predictions.py']];found=set()
    while pending:
        p=pending.pop()
        if p in found:continue
        found.add(p)
        for n in ast.walk(ast.parse(p.read_text())):
            names=[v.name for v in n.names] if isinstance(n,ast.Import) else [n.module,*[n.module+'.'+v.name for v in n.names]] if isinstance(n,ast.ImportFrom) and n.module else []
            for name in names:
                q=ROOT.joinpath(*name.split('.')).with_suffix('.py')
                if q.exists() and q not in found:pending.append(q)
    return sorted(found)

def frozen(out):return {str(p.relative_to(ROOT)):sha_file(p) for p in [out/'PROTOCOL.md',out/'PREFIT_REVIEW.md',out/'config.json',out/'comparison_rules.json',out/'INPUT_SCHEMA.json',*closure()]}
def verify(out):
    f=read(out/'protocol_freeze.json');assert frozen(out)==f['source_protocol_hashes']
    for p,h in f['reference_hashes'].items():assert sha_file(ROOT/p)==h,p
    return read(out/'config.json')

def prepare(out):
    cfg=read(out/'config.json');assert cfg['condition_order']==list(CONDITIONS);assert not (out/'protocol_freeze.json').exists()
    identities={};used={};anchor_report=[]
    for seed in (0,1,2):
        frame,cohort,pools,refs,pre=load_context(cfg,seed);dest=out/f'seed_{seed}';dest.mkdir()
        pca=dict(refs.releases['E_pca'])
        # Existing frozen input arrays only: no new reserved labels or selections.
        path=refs.directory/'release_E_pca.npz'
        with np.load(refs.checked(path)) as z:pca['test']=z['test']
        models={};anchors={pool:{} for pool in pca};records={};parent=refs.directory
        with np.load(refs.checked(parent/'predictions.npz')) as oldpred:
            for task in train.hist.PURPOSE_TASKS['A']+train.hist.PURPOSE_TASKS['B']:
                key='transfer/E_pca/'+task;cid=refs.selection['head_selections'][key];path=parent/'fitted'/key/cid
                model=hist.checked_candidate(parent,path,used);models[task]=model
                assert model.metadata==refs.selection['fitting_records'][key]['candidates'][cid]
                fy=source_binaries(frame.iloc[pools['downstream_fit']],refs.source.selection['income_edges'])[task]
                fi=subset_indices(fy,2048,1230000+100*seed+TASKS.index(task))
                assert model.metadata['fit_hashes']=={'x':_hash_array(pca['downstream_fit'][fi].astype(np.float64)),'y':_hash_array(fy[fi])}
                assert model.metadata['fit_rows']==2048
                for pool,x in pca.items():anchors[pool][task]=model.predict_proba(x)
                checks={}
                for split,pool in [('validation','downstream_validation'),('test','test')]:
                    y=source_binaries(frame.iloc[pools[pool]],refs.source.selection['income_edges'])[task];valid=y>=0
                    expected=oldpred[f'{split}/{key}/{cid}'];actual=anchors[pool][task][valid]
                    assert np.array_equal(expected,actual),(seed,task,pool,'anchor vector mismatch')
                    w=frame.iloc[pools[pool]].PWGTP.to_numpy(float)[valid]
                    checks[split]={'exact_vector':True,'prediction_sha256':array_hash(actual),'unweighted':metrics(y[valid],actual,2),'PWGTP':metrics(y[valid],actual,2,w)}
                for split,z in checks.items():
                    parent_scores=read(out/'comparison_rules.json')['original_parent_metric_identity'][str(seed)]['tasks'][task]['scores']
                    assert z['unweighted']['log_loss']==parent_scores[split]
                    assert z['PWGTP']['log_loss']==parent_scores[split+'_person_weighted']
                records[task]={'candidate_id':cid,'path':str(path.relative_to(ROOT)),'metadata':model.metadata,'parameters':sum(p.numel() for p in model.model.parameters()) if cid=='mlp' else int(model.model.coef_.size+model.model.intercept_.size),'checks':checks,'pool_prediction_hashes':{p:array_hash(a[task]) for p,a in anchors.items()}}
                anchor_report.append({'seed':seed,'task':task,**records[task]})
        arrays={p:{'A':np.column_stack([v[t] for t in train.hist.PURPOSE_TASKS['A']]),'B':v['public_coverage'].copy()} for p,v in anchors.items()}
        np.savez_compressed(dest/'anchors.npz',**{p+'/'+v:a for p,z in arrays.items() for v,a in z.items()})
        state,initial=hist.historical_initial(cfg,seed);m=train.Auxiliary(pre,state).freeze();assert sum(p.numel() for p in m.parameters())==3186
        teacher_dir=ROOT/cfg['selective_reference_results']/'static'/f'seed_{seed}'/'teachers';teachers=load_teachers(teacher_dir);teacher=teachers['maps']['E'];assert teacher.output_dim==16 if hasattr(teacher,'output_dim') else teachers['fit_targets']['E'].shape[1]==16
        e={p:teacher.apply(x[:,:16]) for p,x in pca.items()};assert np.array_equal(e['representation_fit'],teachers['fit_targets']['E'])
        np.savez_compressed(dest/'teacher.npz',**e);np.savez_compressed(dest/'pca.npz',**pca)
        np.savez_compressed(dest/'split_rows.npz',**{p:frame.iloc[ix]._raw_row.to_numpy() for p,ix in pools.items()})
        for p,h in {**refs.used_files,**refs.source.used_files}.items():used[p]=h
        used[str(initial.relative_to(ROOT))]=sha_file(initial)
        for record in read(teacher_dir.parent/'local_artifacts.json'):
            if '/teachers/' in record['path']:
                assert sha_file(ROOT/record['path'])==record['sha256'];used[record['path']]=record['sha256']
        identities[str(seed)]={'anchors':records,'preprocessing':pre,'initialization_path':str(initial.relative_to(ROOT)),'initialization_sha256':sha_file(initial),'E_fingerprint':teacher.fingerprint(),'E_dimension':16,'pool_hashes':{p:array_hash(frame.iloc[ix]._raw_row.to_numpy()) for p,ix in pools.items()},'cached_inputs':{str(p.relative_to(ROOT)):sha_file(p) for p in dest.glob('*.npz')},'source_fitting_hashes':{t:array_hash(y) for t,y in source_binaries(frame.iloc[pools['representation_fit']],refs.source.selection['income_edges']).items()}}
    for seed in range(3):
        for name in ('metrics.json','selection_before_test.json','predictions.npz'):
            p=ROOT/cfg['coalition_reference_results']/f'seed_{seed}'/'controls'/name;used[str(p.relative_to(ROOT))]=sha_file(p)
    write_json(out/'ANCHOR_PARITY.json',anchor_report);write_json(out/'PREFIT_IDENTITY.json',identities)
    write_json(out/'REUSE_MANIFEST.json',{'historical_hashes':used,'anchor_models':9,'teacher_maps':3,'historical_refits':0,'historical_source_identities':'original manifests retained; new code not attributed to old fits','static_controls':str(ROOT/cfg['coalition_reference_results']),'anchor_fitting_exposure':'original2048 source labels; readout fitting pools overlap upstream exposure'})
    used.update({str((out/p).relative_to(ROOT)):sha_file(out/p) for p in ('PREFIT_IDENTITY.json','ANCHOR_PARITY.json','REUSE_MANIFEST.json')})
    write_json(out/'protocol_freeze.json',{'created_utc':now(),'source_protocol_hashes':frozen(out),'reference_hashes':used});progress(out)

def load_inputs(out,seed):
    d=out/f'seed_{seed}'
    def load(n):
        with np.load(d/n) as z:return {p:z[p] for p in z.files}
    flat=load('anchors.npz');a={p:{v:flat[p+'/'+v] for v in ('A','B')} for p in {k.split('/')[0] for k in flat}}
    return load('pca.npz'),a,load('teacher.npz')

def load_final(out,seed,condition):
    ident=read(out/'PREFIT_IDENTITY.json')[str(seed)];cp=torch.load(out/f'seed_{seed}'/'training'/condition/'final.pt',weights_only=True)
    m=None
    if condition in train.ARMS:
        state=torch.load(ROOT/ident['initialization_path'],weights_only=True)['model_state'];m=train.Auxiliary(ident['preprocessing'],state);m.load_state_dict(cp['model_state']);m.freeze()
    o,_=train.make_observers(seed,condition!='H');o.load_state_dict(cp['adversary_state']);o.eval().requires_grad_(False)
    return m,o

def releases(m,a,pca,e,condition):
    wire={v:{} for v in train.hist.PURPOSE_TASKS}|{'AB':{}};derived=copy.deepcopy(wire) if m is not None else None
    for p,x in pca.items():
        with torch.no_grad():aux=m(m.standardize(x))[0].numpy() if m is not None else e[p] if condition=='E' else None
        w,d=train.array_wires(aux,a[p],m)
        for v in wire:
            wire[v][p]=w[v]
            if derived is not None:derived[v][p]=d[v]
    return wire,derived

def progress(out):
    systems=[{'seed':s,'condition':c,'learned':c in train.ARMS,'trained':(out/f'seed_{s}'/'training'/c/'final.pt').exists(),'evaluated':(out/f'seed_{s}'/c/'complete.json').exists()} for s in range(3) for c in CONDITIONS]
    r={'updated_utc':now(),'systems':systems,'planned_systems':18,'learned_continuations':sum(e['learned'] and e['trained'] for e in systems),'evaluated_systems':sum(e['evaluated'] for e in systems),'complete':all(e['evaluated'] for e in systems),'globally_frozen':(out/'RELEASE_MANIFEST.json').exists()};write_json(out/'EXECUTED_MATRIX.json',r);return r

def train_block(out,cfg,seed):
    assert not (out/'RELEASE_MANIFEST.json').exists();dest=out/f'seed_{seed}'
    if (dest/'training_complete.json').exists():return
    frame,cohort,pools,refs,pre=load_context(cfg,seed);pca,a,e=load_inputs(out,seed);state,path=hist.historical_initial(cfg,seed)
    arms,record=train.train_seed(pca['representation_fit'],source_binaries(frame.iloc[pools['representation_fit']],refs.source.selection['income_edges']),audit_labels(frame.iloc[pools['representation_fit']]),pre,state,a['representation_fit'],e['representation_fit'],seed,dest/'training')
    hashes={}
    for c,(m,o,_) in arms.items():
        w,d=releases(m,a,pca,e,c);p=dest/'training'/c/'releases.npz'
        np.savez_compressed(p,**{f'{space}/{v}/{pool}':x for space,views in [('wire',w),('derived',d)] if views is not None for v,arr in views.items() for pool,x in arr.items()})
        hashes[c]={'checkpoint_sha256':sha_file(p.parent/'final.pt'),'release_sha256':sha_file(p),'outputs':{space:{v:{pool:array_hash(x) for pool,x in arr.items()} for v,arr in views.items()} for space,views in [('wire',w),('derived',d)] if views is not None}}
    write_json(dest/'training_complete.json',{'created_utc':now(),'conditions':hashes,'reserved_labels_accessed':False,'runtime_seconds':record['runtime_seconds']});progress(out)

def freeze(out):
    if (out/'RELEASE_MANIFEST.json').exists():return read(out/'RELEASE_MANIFEST.json')
    r=progress(out);assert all(x['trained'] for x in r['systems'])
    write_json(out/'RELEASE_MANIFEST.json',{'created_utc':now(),'training_closed':True,'all18_frozen':True,'seeds':{s:read(out/f'seed_{s}'/'training_complete.json') for s in range(3)},'protocol_sha256':sha_file(out/'protocol_freeze.json')})

def load_audits(path):
    record=read(path/'audits/audit_selection.json');out={'metadata':record,'selection':{int(b):v for b,v in record['selections'].items()},'candidates':{}}
    cache={}
    for b,roles in record['candidates'].items():
        out['candidates'][int(b)]={}
        for role,cs in roles.items():
            out['candidates'][int(b)][role]={}
            for cid,meta in cs.items():
                p=Path(meta['base_candidate_directory']);assert sha_file(p/'metadata.json')==meta['base_metadata_sha256']
                if str(p) not in cache:cache[str(p)]=load_candidate(p)
                out['candidates'][int(b)][role][cid]=audit.AuditCandidate(cache[str(p)],meta['space'],tuple(meta['projection_columns']) if meta['projection_columns'] is not None else None,meta)
    return out

def utilities(wire,labels,ti,seed,dest,h=None):
    result={'candidates':{},'selection':{},'metadata':{}}
    for view,targets in hist.AUTHORIZED.items():
        for t in targets:
            role=view+'/'+t;ix=ti[t];valid=labels['downstream_validation'][t]>=0
            if view=='B' and h is not None:
                meta=read(h/'selection_before_test.json')['utility_metadata'][role];cs={cid:load_candidate(h/'fitted/utility'/view/t/cid) for cid in ('logistic','mlp')};meta={**meta,'alias':str(h/'fitted/utility'/view/t)}
                for c in cs.values():assert c.metadata['fit_hashes']['x']==_hash_array(wire[view]['downstream_fit'][ix])
            else:
                r=fit_candidates(wire[view]['downstream_fit'][ix],labels['downstream_fit'][t][ix],wire[view]['downstream_validation'][valid],labels['downstream_validation'][t][valid],2,1250000+100*seed+TASKS.index(t),budget={'mlp':{'epochs':40}});save_candidates(r,dest/'fitted/utility'/view/t);cs=r['candidates'];meta=r['metadata']
            result['candidates'][role]=cs;result['selection'][role]=min(cs,key=lambda k:(cs[k].metadata['validation_scores']['log_loss'],k));result['metadata'][role]=meta
    return result

def evaluate(out,cfg,seed,condition):
    assert (out/'RELEASE_MANIFEST.json').exists();dest=out/f'seed_{seed}'/condition
    if (dest/'complete.json').exists():return
    dest.mkdir();tick=time.perf_counter();print('EVALUATE',seed,condition,flush=True)
    frame,cohort,pools,refs,pre=load_context(cfg,seed);pca,a,e=load_inputs(out,seed);m,o=load_final(out,seed,condition);w,d=releases(m,a,pca,e,condition)
    with np.load(out/f'seed_{seed}'/'training'/condition/'releases.npz') as z:
        for space,views in [('wire',w),('derived',d)]:
            if views is not None:
                for v,arr in views.items():
                    for p,x in arr.items():assert np.array_equal(x,z[f'{space}/{v}/{p}'])
    labels={p:hist.all_labels(frame.iloc[pools[p]],refs.source.selection['income_edges']) for p in ('downstream_fit','downstream_validation','attacker_fit','attacker_validation','test')};weights={p:frame.iloc[pools[p]].PWGTP.to_numpy(float) for p in labels}
    ti={t:subset_indices(labels['downstream_fit'][t],cfg['head_budget'],1230000+100*seed+j) for j,t in enumerate(TASKS)};ai={t:subset_indices(labels['attacker_fit'][t],cfg['attacker_budget'],1240000+100*seed+j) for j,t in enumerate(hist.TARGETS)}
    hist.write_once_same(out/f'seed_{seed}'/'support.json',{p:support(y,hist.TARGETS) for p,y in labels.items()})
    hist.write_once_same(out/f'seed_{seed}'/'indices.json',{'utility':{t:array_hash(ix) for t,ix in ti.items()},'attacker':{t:array_hash(ix) for t,ix in ai.items()}})
    h=None if condition=='H' else out/f'seed_{seed}'/'H';u=utilities(w,labels,ti,seed,dest,h)
    ancestor=load_audits(h) if h is not None else None
    exposure={r:{'representation_fit_passes':260,'rows':len(pools['representation_fit']),'stationary':condition in ('H','E') or r.startswith('B__'),'B_alias':'shared A0 observer trajectory','source_labels':'only original three','attributes':'real SEX2/RAC1P9'} for r in audit.OBSERVER_ROLES}
    fit=lambda views:{v:{p:x for p,x in arr.items() if p in audit.FIT_POOLS} for v,arr in views.items()}
    audits=audit.fit_condition_audits(fit(w),fit(d) if d else None,{p:labels[p] for p in audit.FIT_POOLS},ai,o,seed,dest/'audits',interface=condition if condition in ('H','E') else 'learned',inherited_exposure=exposure,ancestor=ancestor)
    write_json(dest/'selection_before_test.json',{'created_utc':now(),'utility':u['selection'],'utility_metadata':u['metadata'],'audits':audits['selection'],'global_freeze_sha256':sha_file(out/'RELEASE_MANIFEST.json')})
    tw={v:arr['test'] for v,arr in w.items()};td={v:arr['test'] for v,arr in d.items()} if d else None
    hist.score_condition(seed,condition,w,d,tw,td,u,audits,labels,labels['test'],weights,weights['test'],dest)
    # Every anchor witness is replayed on RAW projected coordinates and compared with H on both pools.
    parity=[]
    if h is not None:
        with np.load(h/'predictions.npz') as hp,np.load(dest/'predictions.npz') as cp:
            for b,roles in audits['candidates'].items():
                for role,cs in roles.items():
                    for cid,c in cs.items():
                        if not cid.startswith('anchor__'):continue
                        for split in ('validation','test'):
                            key=f'audit/{role}/{b}/{cid}/{split}';old=f'audit/{role}/{b}/{cid.removeprefix("anchor__")}/{split}';assert np.array_equal(cp[key],hp[old]);parity.append({'role':role,'budget':b,'cid':cid,'split':split,'exact':True})
    write_json(dest/'anchor_projection_parity.json',parity)
    local={str(p.relative_to(ROOT)):sha_file(p) for p in dest.rglob('*') if p.is_file()};write_json(dest/'complete.json',{'created_utc':now(),'runtime_seconds':time.perf_counter()-tick,'files_sha256':local});progress(out)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);p.add_argument('--phase',choices=['prepare','train','freeze','evaluate'],required=True);p.add_argument('--max-units',type=int,default=18);args=p.parse_args();out=args.out.resolve();torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if args.phase=='prepare':prepare(out);return
        cfg=verify(out)
        if args.phase=='freeze':freeze(out);return
        n=0
        if args.phase=='train':
            for s in range(3):
                if (out/f'seed_{s}'/'training_complete.json').exists():continue
                print('TRAIN',s,flush=True);train_block(out,cfg,s);n+=1
                if n>=args.max_units:break
            if all((out/f'seed_{s}'/'training_complete.json').exists() for s in range(3)):freeze(out)
        else:
            for s in range(3):
                for c in CONDITIONS:
                    if (out/f'seed_{s}'/c/'complete.json').exists():continue
                    evaluate(out,cfg,s,c);n+=1
                    if n>=args.max_units:return
if __name__=='__main__':main()
