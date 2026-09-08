"""Finite two-purpose ACS study; historical contracts and artifacts stay read-only."""
from __future__ import annotations
import argparse
import ast
import datetime
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from experiments.acs_transfer_data import array_hash, audit_labels, sha_file, support, write_json
from experiments.acs_transfer_heads import fit_candidates, fit_prior, load_candidate, metrics, _hash_array, _state_hash
from experiments.acs_selective_teachers import load_teachers
from experiments.acs_preservation_audits import fit_extended_auditors, save_extended_audits
from experiments.run_acs_selective import load_context, fit_static_auditors
from experiments.run_acs_bottleneck import source_binaries
from experiments.run_acs_protection import ROOT, TASKS, binary_tasks, now, read
from experiments.run_acs_transfer import subset_indices, save_candidates

DEFAULT_OUT=ROOT/'results/redesign_20260908_acs_coalition_v1'
TARGETS={'SEX':2,'RAC1P':9,'income_binary':2,'civilian_at_work':2,'public_coverage':2,'same_residence':2,'commute_over20':2}
AUTHORIZED={'A':('income_binary','civilian_at_work','same_residence'),'B':('public_coverage','commute_over20')}
CONDITIONS=('F_I','F_Iplus','F_J','P_I','P_Iplus','P_J')
IDS=('logistic','mlp_0','mlp_1','hist_gb_20','hist_gb_5')


def check_config(cfg):
    old=read(ROOT/cfg['restricted_reference_results']/'config.json')
    for k in ('head_budget','attacker_budget','head_mlp_epochs','audit_mlp_epochs','catchup_epochs','audit_restart_offset','audit_min_samples_leaf','margins','utility_tasks','seeds','threads','parity_tolerance','extended_audit_epochs'):
        assert cfg[k]==old[k],k
    assert cfg['study']=='acs_coalition_v1' and cfg['condition_order']==list(CONDITIONS)
    assert cfg['target_order']==list(TARGETS) and cfg['execution_order']==[0,1,2]
    assert cfg['objectives']=={'I':{'individual':-.1,'extra_local':0.,'coalition':0.},'Iplus':{'individual':-.1,'extra_local':-.1,'coalition':0.},'J':{'individual':-.1,'extra_local':0.,'coalition':-.1}}
    assert cfg['extra_local_reduction']=='M_A+M_B SUM'
    assert cfg['maximum_scientific_seconds']==5400 and cfg['maximum_total_work_seconds']==14400 and cfg['reserved_reporting_seconds']==1800
    assert cfg['forward_parameter_count']==6355 and cfg['repetition_counts']==[1,2,4]


def execution_paths():
    """Static closure of local scientific imports, including explicitly dynamic ones."""
    pending=[ROOT/'experiments'/n for n in ('run_acs_coalition.py','acs_coalition_training.py','acs_coalition_audits.py')]+[ROOT/'scripts/run_acs_coalition_bounded.py']
    found=set()
    while pending:
        path=pending.pop()
        if path in found:continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names=[]
            if isinstance(node,ast.Import):names=[n.name for n in node.names]
            if isinstance(node,ast.ImportFrom) and node.module:names=[node.module]
            for name in names:
                q=ROOT.joinpath(*name.split('.')).with_suffix('.py')
                if q.exists() and q not in found:pending.append(q)
    return sorted(found)


def hashes(out):
    return {str(p.relative_to(ROOT)):sha_file(p) for p in [out/'config.json',out/'PROTOCOL.md',out/'PURPOSE_POLICY.md',out/'ACCESS_SCOPE.md',*execution_paths()]}


def historical_initial(cfg,seed):
    parent=ROOT/cfg['init_reference_results']/f'seed_{seed}'
    path=parent/'training/initialization.pt'
    manifest={v['path']:v['sha256'] for v in read(parent/'local_artifacts.json')}
    assert sha_file(path)==manifest[str(path.relative_to(ROOT))]
    return torch.load(path,weights_only=True)['model_state'],path


def prepare(out):
    from experiments.acs_coalition_training import CoalitionModel
    cfg=read(out/'config.json');check_config(cfg)
    assert not (out/'protocol_freeze.json').exists(),'Never replace a frozen study'
    identities={};used={}
    for seed in cfg['seeds']:
        frame,cohort,pools,refs,pre=load_context(cfg,seed)
        state,path=historical_initial(cfg,seed);model=CoalitionModel(pre,state).freeze()
        initial_hash=_state_hash(model);assert sum(p.numel() for p in model.parameters())==6355
        parity={}
        for pool,x in refs.releases['E_pca'].items():
            wires=model.wires(x,'F')
            for view in ('A','B'):
                e=wires[view].astype(float)-x[:,:16].astype(float)
                assert np.allclose(wires[view],x[:,:16],**cfg['parity_tolerance'])
                parity[pool+'/'+view]={'max_abs':float(np.abs(e).max()),'rms':float(np.sqrt(np.mean(e*e)))}
        teachers=load_teachers(ROOT/cfg['selective_reference_results']/'static'/f'seed_{seed}'/'teachers')
        assert np.array_equal(teachers['maps']['E'].apply(refs.releases['E_pca']['representation_fit'][:,:16]),teachers['fit_targets']['E'])
        identities[str(seed)]={'initial_model_sha256':initial_hash,'parameter_count':6355,'parity':parity,'preprocessing':pre,'original_initialization_path':str(path.relative_to(ROOT)),'original_initialization_sha256':sha_file(path),'teacher_E_map_sha256':teachers['maps']['E'].fingerprint(),'household_pool_raw_row_hashes':{p:array_hash(frame.iloc[ix]._raw_row.to_numpy()) for p,ix in pools.items()}}
        used.update(refs.used_files);used.update(refs.source.used_files);used[str(path.relative_to(ROOT))]=sha_file(path)
        directory=ROOT/cfg['selective_reference_results']/'static'/f'seed_{seed}'
        for e in read(directory/'local_artifacts.json'):
            if '/teachers/' in e['path']:assert sha_file(ROOT/e['path'])==e['sha256'];used[e['path']]=e['sha256']
    write_json(out/'PREFIT_IDENTITY.json',{'created_utc':now(),'seeds':identities,'reserved_labels_used':False,'historical_regeneration':False})
    refs={}
    for key in ('parent_results','reference_results','init_reference_results','selective_reference_results','restricted_reference_results','preservation_reference_results'):
        parent=ROOT/cfg[key]
        for name in ('config.json','PROTOCOL.md','protocol_freeze.json'):
            p=parent/name;refs[str(p.relative_to(ROOT))]=sha_file(p)
    for seed in cfg['seeds']:
        for parent in (ROOT/cfg['selective_reference_results']/'static'/f'seed_{seed}',ROOT/cfg['preservation_reference_results']/'extended'/f'seed_{seed}'):
            path=parent/'selection_before_test.json';refs[str(path.relative_to(ROOT))]=sha_file(path)
    write_json(out/'protocol_freeze.json',{'created_utc':now(),'starting_commit':cfg['starting_commit'],'sha256':hashes(out),'reference_record_hashes':refs,'required_local_reference_hashes':used,'prefit_identity_sha256':sha_file(out/'PREFIT_IDENTITY.json'),'historical_source_identities':'original historical freezes remain unchanged','regenerated_artifacts':[]})
    write_json(out/'environment.json',{**read(ROOT/cfg['restricted_reference_results']/'environment.json'),'created_utc':now()})


def verify_freeze(out):
    cfg=read(out/'config.json');check_config(cfg);f=read(out/'protocol_freeze.json')
    expected=dict(f['sha256'])
    amendments=out/'EXECUTION_AMENDMENTS.json'
    if amendments.exists():
        amended=read(amendments)
        for p,h in amended.get('historical_metadata_anchor_hashes',{}).items():assert sha_file(ROOT/p)==h,p
        for amendment in amended['amendments']:
            for path,change in amendment['source_changes'].items():
                assert expected[path]==change['before_sha256']
                assert sha_file(ROOT/change['preserved_source_path'])==change['before_sha256']
                expected[path]=change['after_sha256']
    assert expected==hashes(out),'Scientific source/protocol changed outside recorded amendment'
    assert f['prefit_identity_sha256']==sha_file(out/'PREFIT_IDENTITY.json')
    for p,h in {**f['reference_record_hashes'],**f['required_local_reference_hashes']}.items():assert sha_file(ROOT/p)==h,p


def checked_candidate(parent,path,used):
    manifest={e['path']:e['sha256'] for e in read(parent/'local_artifacts.json')}
    for p in path.iterdir():
        if p.is_file():
            key=str(p.relative_to(ROOT));digest=sha_file(p)
            if key in manifest:
                assert digest==manifest[key],key
            else:
                # Historical binary manifests omit metadata. Bind it to the
                # original published selection record rather than invent a hash.
                assert p.name=='metadata.json',key
                parts=path.relative_to(parent).parts;selection_path=parent/'selection_before_test.json'
                if (parent/'completion.json').exists():assert sha_file(selection_path)==read(parent/'completion.json')['sha256']['selection_before_test.json']
                record=read(selection_path)
                role_key='/'.join(parts[1:4])
                selected=record['budgets'][parts[-2].removeprefix('nested')] if parts[1]=='audit' else record
                assert read(p)==selected['fitting_records'][role_key]['candidates'][parts[-1]],key
                used[str((parent/'selection_before_test.json').relative_to(ROOT))]=sha_file(parent/'selection_before_test.json')
            used[key]=digest
    return load_candidate(path)


def reused_nested(parent,relative,used):
    root=parent/relative;result={'paths':{},'source':str(root)}
    for b in (120,360):
        paths={cid:root/f'nested{b}'/cid for cid in IDS}
        result[f'nested{b}']={'candidates':{cid:checked_candidate(parent,p,used) for cid,p in paths.items()}}
        result['paths'][b]={cid:str(p) for cid,p in paths.items()}
    return result


def write_once_same(path,value):
    if path.exists():assert read(path)==value,str(path)
    else:write_json(path,value)


def load_trained_seed(directory,pre,initial,seed):
    from experiments.acs_coalition_training import CoalitionModel,make_observers
    training=directory/'training';metadata=read(training/'training.json')
    assert set(metadata['arms'])==set(CONDITIONS)
    def model(path):
        checkpoint=torch.load(path,weights_only=True)
        result=CoalitionModel(pre,initial);result.load_state_dict(checkpoint['model_state']);return result.freeze(),checkpoint
    snapshots={name:model(training/filename)[0] for name,filename in [('I','initialization.pt'),('W','warm_base.pt')]}
    arms={}
    for condition in CONDITIONS:
        forward,checkpoint=model(training/condition/'final.pt')
        observers,_=make_observers(condition[0],seed);observers.load_state_dict(checkpoint['adversary_state'])
        arms[condition]={'model':forward,'observers':observers.eval().requires_grad_(False)}
    return {'snapshots':snapshots,'arms':arms,'metadata':metadata}


def load_completed_condition(directory,wire,labels,ti,seed):
    from experiments.acs_coalition_audits import AuditCandidate
    saved=read(directory/'selection_before_test.json')
    utility={'candidates':{},'selection':saved['utility'],'metadata':saved['utility_metadata']}
    for role,meta in saved['utility_metadata'].items():
        view,target=role.split('/');utility['candidates'][role]={}
        for cid,original in meta['candidates'].items():
            path=directory/'fitted/utility'/view/target/cid
            assert read(path/'metadata.json')==original
            candidate=load_candidate(path);valid=labels['downstream_validation'][target]>=0
            assert candidate.metadata['fit_hashes']=={'x':_hash_array(np.asarray(wire[view]['downstream_fit'][ti[target]],np.float64)),'y':_hash_array(labels['downstream_fit'][target][ti[target]])}
            assert metrics(labels['downstream_validation'][target][valid],candidate.predict_proba(wire[view]['downstream_validation'][valid]),2)==original['validation_scores']
            utility['candidates'][role][cid]=candidate
    record=read(directory/'audits/audit_selection.json');audits={'candidates':{},'selection':{int(b):v for b,v in record['selections'].items()},'metadata':record}
    assert record['selections']==saved['audits']
    for budget,roles in record['candidates'].items():
        audits['candidates'][int(budget)]={}
        for role,candidates in roles.items():
            audits['candidates'][int(budget)][role]={}
            for cid,meta in candidates.items():
                path=Path(meta['base_candidate_directory'])
                assert sha_file(path/'metadata.json')==meta['base_metadata_sha256']
                candidate=load_candidate(path)
                columns=tuple(meta['projection_columns']) if meta['projection_columns'] is not None else None
                audits['candidates'][int(budget)][role][cid]=AuditCandidate(candidate,meta['space'],columns,meta)
    return utility,audits


def all_labels(frame,edges):return {**binary_tasks(frame,edges),**audit_labels(frame)}


def arrange(model,pca,interface):
    wire={v:{} for v in ('A','B','AB')};derived={v:{} for v in wire}
    for pool,x in pca.items():
        w=model.wires(x,interface);d=model.wires(x,'P')
        for v in wire:
            wire[v][pool]=w[v];derived[v][pool]=d[v]
    return wire,derived


def view_hashes(wire,derived):
    return {space:{v:{p:array_hash(x) for p,x in a.items()} for v,a in views.items()} for space,views in [('wire',wire),('derived',derived)]}


def fit_utilities(wire,labels,ti,seed,directory,*,reuse_parent=None,used=None):
    fitted={};selection={};records={}
    for view,targets in AUTHORIZED.items():
        for target in targets:
            key=view+'/'+target;i=ti[target];v=np.flatnonzero(labels['downstream_validation'][target]>=0);j=TASKS.index(target)
            xf=wire[view]['downstream_fit'][i];yf=labels['downstream_fit'][target][i]
            xv=wire[view]['downstream_validation'][v];yv=labels['downstream_validation'][target][v]
            if reuse_parent:
                candidates={cid:checked_candidate(reuse_parent,reuse_parent/'fitted/transfer/E_direct'/target/cid,used) for cid in ('logistic','mlp')}
                for c in candidates.values():
                    assert c.metadata['fit_hashes']=={'x':_hash_array(np.asarray(xf,np.float64)),'y':_hash_array(yf)}
                    assert metrics(yv,c.predict_proba(xv),2)['log_loss']==c.metadata['validation_scores']['log_loss']
                meta={'reused':True,'source_directory':str(reuse_parent),'candidates':{cid:c.metadata for cid,c in candidates.items()}}
            else:
                result=fit_candidates(xf,yf,xv,yv,2,1250000+100*seed+j,budget={'mlp':{'epochs':40}})
                save_candidates(result,directory/'fitted/utility'/view/target);candidates=result['candidates'];meta=result['metadata']
            fitted[key]=candidates;selection[key]=min(candidates,key=lambda c:(candidates[c].metadata['validation_scores']['log_loss'],c));records[key]=meta
    return {'candidates':fitted,'selection':selection,'metadata':records}


def fit_controls(labels,ai,seed,directory,cfg,used):
    priors={};exposed={120:{},360:{}};selection={120:{},360:{}};metadata={};paths={}
    parent=ROOT/cfg['preservation_reference_results']/'extended'/f'seed_{seed}'
    for j,(target,k) in enumerate(TARGETS.items()):
        ix=ai[target];vi=np.flatnonzero(labels['attacker_validation'][target]>=0)
        yf=labels['attacker_fit'][target][ix];yv=labels['attacker_validation'][target][vi]
        prior=fit_prior(yf,k);prior.save(directory/'fitted/prior'/target);priors[target]=prior
        xf=np.eye(k)[yf];xv=np.eye(k)[yv]
        if target in ('SEX','RAC1P'):
            result=reused_nested(parent,Path('fitted/audit/exposed')/target/'fresh',used)
            for b in (120,360):
                for c in result[f'nested{b}']['candidates'].values():assert c.metadata['fit_hashes']=={'x':_hash_array(xf),'y':_hash_array(yf)}
            metadata[target]={'reused':True,'source':result['source']};paths[target]=result['paths']
        else:
            statics=fit_static_auditors(xf,yf,xv,yv,k,1260000+100*seed+j)
            result=fit_extended_auditors(xf,yf,xv,yv,k,1260000+100*seed+j,static_candidates=statics)
            dest=directory/'fitted/exposed'/target;save_extended_audits(result,dest)
            metadata[target]=result['metadata'];paths[target]={b:{cid:str(dest/f'nested{b}'/cid) for cid in IDS} for b in (120,360)}
        for b in (120,360):
            exposed[b][target]=result[f'nested{b}']['candidates']
            selection[b][target]=min(exposed[b][target],key=lambda c:(exposed[b][target][c].metadata['validation_scores']['log_loss'],c))
    record={'selection':selection,'metadata':metadata,'paths':paths,'prior_metadata':{t:c.metadata for t,c in priors.items()},'exposed_new_targets':list(TARGETS)[2:],'exposed_reused_targets':['SEX','RAC1P']}
    write_json(directory/'selection_before_test.json',record)
    return priors,exposed,record


def add_scores(rows,predictions,seed,condition,role,view,target,budget,cid,selected,predict,inputs,labels,weights):
    k=TARGETS[target];scores={}
    for split in ('validation','test'):
        y=labels[split][target];valid=y>=0;p=predict(inputs[split],valid)
        key=f'{role}/{view}/{target}/{budget}/{cid}/{split}';predictions[key]=p
        scores[split]=metrics(y[valid],p,k);scores[split+'_person_weighted']=metrics(y[valid],p,k,weights[split][valid])
    rows.append({'seed':seed,'condition':condition,'role':role,'view':view,'target':target,'audit_budget':budget,'candidate_id':cid,'selected_scopes':selected,'scores':scores})


def score_condition(seed,condition,wire,derived,testwire,testderived,utility,audits,labels,testlabels,weights,testweights,directory):
    rows=[];predictions={};parity=[];history_replay=[]
    for key,candidates in utility['candidates'].items():
        view,target=key.split('/')
        for cid,c in candidates.items():
            add_scores(rows,predictions,seed,condition,'utility',view,target,None,cid,['utility'] if utility['selection'][key]==cid else [],lambda x,v,c=c:c.predict_proba(x[v]),{'validation':wire[view]['downstream_validation'],'test':testwire[view]}, {'validation':labels['downstream_validation'],'test':testlabels},{'validation':weights['downstream_validation'],'test':testweights})
    for budget,roles in audits['candidates'].items():
        for key,candidates in roles.items():
            view,target=key.split('/')
            inputs={s:{'wire':w,'derived':d} for s,w,d in [('validation',wire[view]['attacker_validation'],derived[view]['attacker_validation'] if derived else None),('test',testwire[view],testderived[view] if testderived else None)]}
            for cid,c in candidates.items():
                selected=[scope for scope,choice in audits['selection'][budget][key].items() if choice==cid]
                add_scores(rows,predictions,seed,condition,'audit',view,target,budget,cid,selected,lambda x,v,c=c:c.predict_proba({n:a[v] for n,a in x.items() if a is not None}),inputs,{'validation':labels['attacker_validation'],'test':testlabels},{'validation':weights['attacker_validation'],'test':testweights})
                if budget==360 and selected:
                    valid=testlabels[target]>=0
                    base=predictions[f'audit/{view}/{target}/{budget}/{cid}/test']
                    for count in (1,2,4):
                        history={space:np.stack([value[valid]]*count,axis=1)[:,0] for space,value in inputs['test'].items() if value is not None}
                        replay=c.predict_proba(history)
                        assert np.array_equal(base,replay),(key,cid,count)
                        history_replay.append({'role':key,'candidate_id':cid,'calls':count,'exact':True,'prediction_sha256':array_hash(replay)})
                if cid.startswith('inherited_'):

                    # Same candidate on its original singleton rows must exactly replay projection.
                    origin,cid0=cid.removeprefix('inherited_').split('__',1)
                    original=candidates # actual role lookup is fixed by target and origin
                    for split in ('validation','test'):
                        base=predictions.get(f'audit/{origin}/{target}/{budget}/{cid0}/{split}')
                        value=predictions[f'audit/{view}/{target}/{budget}/{cid}/{split}']
                        assert base is not None and np.array_equal(base,value),(key,cid,split)
                        parity.append({'budget':budget,'target':target,'candidate':cid,'split':split,'exact':True,'prediction_sha256':array_hash(value)})
    write_json(directory/'metrics.json',{'raw_metrics':rows,'evaluation_status':'DEVELOPMENT EVALUATION'})
    np.savez_compressed(directory/'predictions.npz',**predictions)
    write_json(directory/'projection_parity.json',{'records':parity,'all_exact':True})
    write_json(directory/'history_attack_replay.json',{'records':history_replay,'scope':'DUPLICATION/INVARIANCE CHECK; each selected360 first-pair auditor replayed','all_exact':True})
    return rows


def run_seed(out,cfg,seed):
    from experiments.acs_coalition_training import train_seed
    from experiments.acs_coalition_audits import fit_condition_audits
    verify_freeze(out);started=time.perf_counter();directory=out/f'seed_{seed}';resuming=directory.exists()
    if resuming:
        assert (out/'EXECUTION_AMENDMENTS.json').exists(),'Explicit preserved recovery amendment required'
        assert (directory/'training/training.json').exists() and (directory/'release_freeze.json').exists(),'Only a fully frozen six-pair training block can be recovered without fitting'
    else:directory.mkdir()
    frame,cohort,pools,refs,pre=load_context(cfg,seed);pca=refs.releases['E_pca'];edges=refs.source.selection['income_edges']
    if resuming:
        with np.load(directory/'split_rows.npz') as rows:
            for p,ix in pools.items():assert np.array_equal(rows[p],frame.iloc[ix]._raw_row.to_numpy())
    else:np.savez_compressed(directory/'split_rows.npz',**{p:frame.iloc[ix]._raw_row.to_numpy() for p,ix in pools.items()})
    source=source_binaries(frame.iloc[pools['representation_fit']],edges);attrs=audit_labels(frame.iloc[pools['representation_fit']]);sv=source_binaries(frame.iloc[pools['source_validation']],edges)
    initial,initialpath=historical_initial(cfg,seed);phase={};tick=time.perf_counter()
    trained=load_trained_seed(directory,pre,initial,seed) if resuming else train_seed(pca['representation_fit'],source,attrs,pre,initial,cfg,seed,directory/'training',pca_val=pca['source_validation'],source_val=sv)
    phase['representation_training_seconds']=trained['metadata']['runtime_seconds']
    phase['training_loaded_without_steps']=resuming
    phase['training_load_or_fit_seconds_this_process']=time.perf_counter()-tick
    assert list(trained['arms'])==list(CONDITIONS)
    assert _state_hash(trained['snapshots']['I'])==read(out/'PREFIT_IDENTITY.json')['seeds'][str(seed)]['initial_model_sha256']
    releases={};models={n:a['model'] for n,a in trained['arms'].items()}
    for name,model in models.items():releases[name]=arrange(model,pca,name[0])
    teachers=load_teachers(ROOT/cfg['selective_reference_results']/'static'/f'seed_{seed}'/'teachers');emap=teachers['maps']['E']
    e={p:emap.apply(x[:,:16]) for p,x in pca.items()};releases['E']=({'A':e,'B':e,'AB':{p:np.concatenate([a,a],axis=1) for p,a in e.items()}},None)
    compositions=[]
    for name,(wire,derived) in releases.items():
        for views in (wire,derived):
            if views:
                for arr in views.values():
                    for x in arr.values():x.setflags(write=False)
        if name.startswith('F'):
            for view in ('A','B'):
                for pool,x in wire[view].items():
                    with torch.no_grad():
                        features=torch.from_numpy(x.copy())
                        value=torch.cat([torch.sigmoid(head(features)) for head in models[name].branches[view].heads.values()],1).numpy()
                    assert np.array_equal(value,derived[view][pool])
                    compositions.append({'condition':name,'view':view,'pool':pool,'public_heads_only':True,'raw_rows_requested':False,'exact':True,'derived_sha256':array_hash(value)})
    write_once_same(directory/'public_composition_prefit.json',{'records':compositions,'computed_from_released_F_and_public_heads_only':True})
    before={n:view_hashes(*r) if r[1] else {'wire':{v:{p:array_hash(x) for p,x in a.items()} for v,a in r[0].items()}} for n,r in releases.items()}
    state={n:_state_hash(m) for n,m in models.items()};observerstate={n:_state_hash(a['observers']) for n,a in trained['arms'].items()}
    freeze_record={'created_utc':now(),'all_six_pairs_complete':True,'reserved_labels_received':False,'outputs':before,'model_hashes':state,'observer_hashes':observerstate,'teacher_E_hash':emap.fingerprint(),'protocol_freeze_sha256':sha_file(out/'protocol_freeze.json')}
    if resuming:
        previous=read(directory/'release_freeze.json')
        assert {k:v for k,v in previous.items() if k!='created_utc'}=={k:v for k,v in freeze_record.items() if k!='created_utc'}
    else:write_json(directory/'release_freeze.json',freeze_record)
    # First permitted access to residence/commute labels for this seed follows all six final pairs.
    labels={p:all_labels(frame.iloc[pools[p]],edges) for p in ('downstream_fit','downstream_validation','attacker_fit','attacker_validation')}
    weights={p:frame.iloc[pools[p]].PWGTP.to_numpy(float) for p in labels}
    ti={t:subset_indices(labels['downstream_fit'][t],cfg['head_budget'],1230000+100*seed+j) for j,t in enumerate(TASKS)}
    ai={t:subset_indices(labels['attacker_fit'][t],cfg['attacker_budget'],1240000+100*seed+j) for j,t in enumerate(TARGETS)}
    ixrecord={'task_fit_indices':{t:{'n':len(i),'pool_indices_sha256':array_hash(i),'raw_rows_sha256':array_hash(frame.iloc[pools['downstream_fit'][i]]._raw_row.to_numpy())} for t,i in ti.items()},'audit_fit_indices':{t:{'n':len(i),'pool_indices_sha256':array_hash(i),'raw_rows_sha256':array_hash(frame.iloc[pools['attacker_fit'][i]]._raw_row.to_numpy())} for t,i in ai.items()}}
    historical=read(ROOT/cfg['init_reference_results']/f'seed_{seed}'/'selection_before_test.json')
    assert ixrecord['task_fit_indices']==historical['task_fit_indices']
    assert all(ixrecord['audit_fit_indices'][t]==historical['audit_fit_indices'][t] for t in ('SEX','RAC1P'))
    write_once_same(directory/'indices.json',ixrecord);write_once_same(directory/'support.json',{p:support(ys,TARGETS) for p,ys in labels.items()})
    fitted={};used={};phase.update(utility_seconds=0.,audit_seconds=0.,controls_seconds=0.)
    for name in (*CONDITIONS,'E'):
        dest=directory/name;wire,derived=releases[name]
        if (dest/'selection_before_test.json').exists():
            print(f'seed{seed} {name}: reloading completed utility and audit fits without any optimization',flush=True)
            fitted[name]=load_completed_condition(dest,wire,labels,ti,seed)
            phase['audit_seconds']+=sum(fitted[name][1]['metadata']['runtime'].values())
            phase['utility_seconds']+=sum(m.get('fit_runtime_seconds',0.) for m in fitted[name][0]['metadata'].values())
            continue
        if dest.exists():assert not any(dest.iterdir()),'Preserve and diagnose any partially fitted condition before retry'
        else:dest.mkdir()
        print(f'seed{seed} {name}: authorized utility then all fixed audit roles',flush=True)
        tick=time.perf_counter();parent=ROOT/cfg['selective_reference_results']/'static'/f'seed_{seed}' if name=='E' else None
        u=fit_utilities(wire,labels,ti,seed,dest,reuse_parent=parent,used=used);phase['utility_seconds']+=time.perf_counter()-tick
        reused={('A',t):reused_nested(parent,Path('fitted/audit/E_direct')/t/'fresh',used) for t in ('SEX','RAC1P')} if name=='E' else {}
        exposure={}
        if name!='E':
            for view,targets in cfg['observer_roles'].items():
                for t in targets:
                    y=attrs[t] if t in attrs else source[t]
                    exposure[view+'__'+t]={'representation_fit_rows':len(y),'known_fitting_labels':int((y>=0).sum()),'representation_fitting_passes':260,'label_presentations':int((y>=0).sum())*260,'warmup_epochs':20,'continuation_epochs':80,'detached_updates_per_mapper':3,'source_fit_raw_row_hash':array_hash(frame.iloc[pools['representation_fit']]._raw_row.to_numpy()),'fit_label_hash':array_hash(y),'additional_attacker_pool_exposure':'360 epochs; Adam reset; disjoint attacker pool'}
        tick=time.perf_counter();a=fit_condition_audits({v:{p:x for p,x in arr.items() if p.startswith('attacker_')} for v,arr in wire.items()}, {v:{p:x for p,x in arr.items() if p.startswith('attacker_')} for v,arr in derived.items()} if derived else None, {p:y for p,y in labels.items() if p.startswith('attacker_')},ai,trained['arms'][name]['observers'] if name!='E' else None,seed,dest/'audits',interface=name[0],inherited_exposure=exposure,reused=reused)
        phase['audit_seconds']+=time.perf_counter()-tick;fitted[name]=(u,a)
        write_json(dest/'selection_before_test.json',{'created_utc':now(),'utility':u['selection'],'utility_metadata':u['metadata'],'audits':a['selection'],'release_freeze_sha256':sha_file(directory/'release_freeze.json'),'inherited_exposure':exposure})
    print(f'seed{seed}: prior/exposed controls and final selection freeze',flush=True)
    tick=time.perf_counter();control=directory/'controls';control.mkdir();priors,exposed,controlrecord=fit_controls(labels,ai,seed,control,cfg,used);phase['controls_seconds']=time.perf_counter()-tick
    write_json(directory/'selection_before_test.json',{'created_utc':now(),'condition_selections_sha256':{n:sha_file(directory/n/'selection_before_test.json') for n in (*CONDITIONS,'E','controls')},'indices_sha256':sha_file(directory/'indices.json'),'all_design_choices_predeclared':True})
    print(f'seed{seed}: all selections frozen; DEVELOPMENT EVALUATION',flush=True);tick=time.perf_counter()
    tf=frame.iloc[pools['test']];testpca=refs.evaluation_releases(tf)['E_pca'];ty=all_labels(tf,edges);tw=tf.PWGTP.to_numpy(float);tests={}
    for name,model in models.items():
        tests[name]=(model.wires(testpca,name[0]),model.wires(testpca,'P'))
        assert all(np.array_equal(value,model.wires(testpca,name[0])[view]) for view,value in tests[name][0].items())
        if name.startswith('F'):
            for view in ('A','B'):
                with torch.no_grad():
                    x=torch.from_numpy(tests[name][0][view].copy())
                    value=torch.cat([torch.sigmoid(head(x)) for head in model.branches[view].heads.values()],1).numpy()
                assert np.array_equal(value,tests[name][1][view])
                compositions.append({'condition':name,'view':view,'pool':'test','public_heads_only':True,'raw_rows_requested':False,'exact':True,'derived_sha256':array_hash(value)})
    write_json(directory/'public_composition_parity.json',{'records':compositions,'all_exact':True})
    et=emap.apply(testpca[:,:16]);tests['E']=({'A':et,'B':et,'AB':np.concatenate([et,et],axis=1)},None)
    duplicate=[]
    for name,(wire,derived) in releases.items():
        testwire,testderived=tests[name];u,a=fitted[name]
        score_condition(seed,name,wire,derived,testwire,testderived,u,a,labels,ty,weights,tw,directory/name)
        save={f'wire/{view}/{p}':x for view,arr in wire.items() for p,x in {**arr,'test':testwire[view]}.items()}
        if derived:save.update({f'derived/{view}/{p}':x for view,arr in derived.items() for p,x in {**arr,'test':testderived[view]}.items()})
        np.savez_compressed(directory/name/'releases.npz',**save)
        for view,x in testwire.items():
            for k in cfg['repetition_counts']:
                hist=np.stack([x]*k,axis=1);assert all(np.array_equal(hist[:,j],x) for j in range(k))
                duplicate.append({'condition':name,'view':view,'calls':k,'exact_duplicate':True,'first_pair_predictions_identical':True,'value_sha256':array_hash(x),'history_sha256':array_hash(hist),'scope':'DUPLICATION/INVARIANCE CHECK; no fresh versions'})
        assert _state_hash(models[name])==state[name] if name in models else emap.fingerprint()==teachers['maps']['E'].fingerprint()
        if name in models:assert _state_hash(trained['arms'][name]['observers'])==observerstate[name]
    native=[];nativepred={}
    for name,model in {**trained['snapshots'],**models}.items():
        for split,x,y,w in [('source_validation',pca['source_validation'],sv,frame.iloc[pools['source_validation']].PWGTP.to_numpy(float)),('test',testpca,source_binaries(tf,edges),tw)]:
            for target,prob in model.native_probabilities(x).items():
                valid=y[target]>=0;nativepred[f'{name}/{target}/{split}']=prob
                native.append({'seed':seed,'condition':name,'view':'B' if target=='public_coverage' else 'A','target':target,'split':split,'score':metrics(y[target][valid],prob[valid],2),'person_weighted':metrics(y[target][valid],prob[valid],2,w[valid]),'prediction_sha256':array_hash(prob),'selection':'fixed native head'})
    write_json(directory/'native_source.json',{'raw_metrics':native});np.savez_compressed(directory/'native_predictions.npz',**nativepred)
    cr=[];cp={}
    for target,k in TARGETS.items():
        for b in (120,360):
            for cid,c in exposed[b][target].items():
                add_scores(cr,cp,seed,'exposed','audit','control',target,b,cid,['exposed'] if controlrecord['selection'][b][target]==cid else [],lambda x,v,c=c,k=k:c.predict_proba(np.eye(k)[x[v]]),{'validation':labels['attacker_validation'][target],'test':ty[target]},{'validation':labels['attacker_validation'],'test':ty},{'validation':weights['attacker_validation'],'test':tw})
        c=priors[target]
        add_scores(cr,cp,seed,'prior','audit','control',target,None,'prior',['prior'],lambda x,v,c=c:c.predict_proba(np.zeros((int(v.sum()),1))),{'validation':None,'test':None},{'validation':labels['attacker_validation'],'test':ty},{'validation':weights['attacker_validation'],'test':tw})
    write_json(control/'metrics.json',{'raw_metrics':cr});np.savez_compressed(control/'predictions.npz',**cp)
    write_json(directory/'duplicate_access.json',{'records':duplicate,'no_new_values':True,'new_versions_or_noise_tested':False})
    phase['read_only_scoring_seconds']=time.perf_counter()-tick
    for name,r in releases.items():
        current=view_hashes(*r) if r[1] else {'wire':{v:{p:array_hash(x) for p,x in a.items()} for v,a in r[0].items()}}
        assert current==before[name]
    write_json(directory/'parent_provenance.json',{'used_reference_files_sha256':{**used,**refs.used_files,**refs.source.used_files},'teacher_maps_refitted':False,'historical_models_retrained':False,'preprocessing':pre})
    phase['seed_process_seconds']=time.perf_counter()-started
    phase['recovery_preserved_completed_pairs']=6 if resuming else 0
    phase['recovery_preserved_completed_audit_conditions']=6 if resuming else 0
    if resuming:
        prior=read(out/'amendments/operational_001/PRESERVED_BEFORE_RECOVERY.json')['files_sha256']
        assert all(sha_file(ROOT/p)==h for p,h in prior.items()),'A pre-recovery artifact was overwritten'
        write_json(directory/'recovery.json',{'utc':now(),'prior_artifacts_preserved':len(prior),'models_retrained':0,'completed_candidate_fits_retrained':0,'outputs_match_original_freeze':True,'amendments_sha256':sha_file(out/'EXECUTION_AMENDMENTS.json')})
    write_json(directory/'runtime.json',phase)
    local=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha_file(p)} for p in directory.rglob('*') if p.is_file() and (p.suffix in ('.pt','.npz','.joblib') or 'fitted' in p.parts)]
    write_json(directory/'local_artifacts.json',local)
    compact={str(p.relative_to(directory)):sha_file(p) for p in directory.rglob('*.json') if 'fitted' not in p.parts}
    write_json(directory/'complete.json',{'completed_utc':now(),'seed':seed,'six_pairs_complete':True,'all_utility_audits_controls_complete':True,'compact_sha256':compact,'local_artifact_manifest_sha256':sha_file(directory/'local_artifacts.json'),'runtime_seconds':time.perf_counter()-started})
    return phase


def complete_seeds(out):
    result=[]
    for seed in (0,1,2):
        p=out/f'seed_{seed}'/'complete.json'
        if p.exists():
            d=read(p)
            for name,h in d['compact_sha256'].items():assert sha_file(p.parent/name)==h,name
            result.append(seed)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT_OUT);parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true');parser.add_argument('--max-seeds',type=int,default=3);args=parser.parse_args();out=args.out.resolve()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if args.prepare:prepare(out)
        if args.run:
            cfg=read(out/'config.json');verify_freeze(out);completed=complete_seeds(out);done=0
            for seed in cfg['execution_order']:
                if seed in completed:continue
                elapsed=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(cfg['started_utc'])).total_seconds()
                science=sum(read(out/f'seed_{s}'/'complete.json')['runtime_seconds'] for s in completed)
                estimated=max([read(out/f'seed_{s}'/'complete.json')['runtime_seconds'] for s in completed],default=0.)
                if science+estimated>cfg['maximum_scientific_seconds'] or elapsed+estimated>cfg['maximum_total_work_seconds']-cfg['reserved_reporting_seconds']:break
                try:run_seed(out,cfg,seed)
                except Exception as error:
                    write_json(out/f'failure_seed_{seed}_{int(time.time())}.json',{'utc':now(),'error':repr(error),'traceback':traceback.format_exc(),'seed':seed,'preserved_partial':True});raise
                completed=complete_seeds(out);done+=1
                write_json(out/'progress.json',{'completed_seeds':completed,'updated_utc':now(),'expected_seeds':[0,1,2]})
                if done>=args.max_seeds:break
            write_json(out/'EXECUTED_MATRIX.json',{'planned_conditions':list(CONDITIONS),'planned_seeds':[0,1,2],'completed_seeds':completed,'status':'complete' if completed==[0,1,2] else 'partial','final_paired_systems':6*len(completed),'final_branch_mappers':12*len(completed),'common_source_warmups':len(completed),'interface_observer_warmups':2*len(completed),'fresh_mlp_epochs':360,'catchup_epochs':360,'all_nine_observers_each_condition':True})
if __name__=='__main__':main()
