"""Independent saved-tensor inference and raw-label score/access replay. No fitting."""
from __future__ import annotations
import argparse,json,time,math
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from scripts import verify_acs_bottleneck_scores as check
from scripts.verify_acs_coalition import check_curves
from experiments.acs_transfer_data import load_cohort,split_households
from experiments.run_acs_transfer import subset_indices
from experiments.acs_bottleneck_training import tree_digest
ROOT=Path(__file__).resolve().parents[1]
TASKS=('same_residence','commute_over20','income_binary','civilian_at_work','public_coverage')
TARGETS=('SEX','RAC1P','income_binary','civilian_at_work','public_coverage','same_residence','commute_over20')
ARMS=('H','E','A0','L025','L20','J')

def read(p):return json.loads(Path(p).read_text())
def arrays(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def literal_aux(cp,pca):
    state=cp['model_state'];x=torch.tensor((pca.astype(float)-state['input_mean'].numpy())/state['input_scale'].numpy(),dtype=torch.float32)
    with torch.no_grad():
        h=torch.relu(torch.nn.functional.linear(x,state['branch.mapper.0.weight'],state['branch.mapper.0.bias']))
        h=torch.nn.functional.linear(h,state['branch.mapper.2.weight'],state['branch.mapper.2.bias']);ps=[]
        for task in ('income_binary','civilian_at_work'):
            p=torch.sigmoid(torch.nn.functional.linear(h,state[f'branch.heads.{task}.weight'],state[f'branch.heads.{task}.bias']))
            ps.append(torch.cat((1-p,p),1).double().numpy())
    return h.double().numpy(),ps

def verify_training(out,seed,rep):
    d=out/f'seed_{seed}'/'training';meta=read(d/'training.json');n=meta['fit_rows'];batch=math.ceil(n/256);common=torch.load(d/'fork.pt',weights_only=True);initial=torch.load(d/'initial.pt',weights_only=True)
    assert common['counts']=={'mapper_steps':60*batch,'observer_steps':20*batch}
    assert all(int(v['step'])==60*batch for v in common['mapper_optimizer']['state'].values())
    assert all(int(v['step'])==20*batch for v in common['adversary_optimizer']['state'].values())
    initial_identity=read(out/'PREFIT_IDENTITY.json')[str(seed)];original=torch.load(ROOT/initial_identity['initialization_path'],weights_only=True)['model_state']
    for k,v in initial['model_state'].items():
        name=k.removeprefix('branch.') if k.startswith('branch.') else k
        assert torch.equal(v,original[name])
    bfinal=None
    for c in ARMS:
        cp=torch.load(d/c/'final.pt',weights_only=True);assert cp['counts']['observer_steps']==260*batch
        assert all(int(v['step'])==260*batch for v in cp['adversary_optimizer']['state'].values())
        b={k:v for k,v in cp['adversary_state'].items() if k.startswith('B__')}
        if bfinal is None:bfinal=b
        else:assert tree_digest(b)==tree_digest(bfinal)
        if c not in ('H','E'):
            fork=torch.load(d/c/'fork.pt',weights_only=True);assert tree_digest(fork)==tree_digest(common)
            assert cp['counts']['mapper_steps']==140*batch
            assert all(int(v['step'])==140*batch for v in cp['mapper_optimizer']['state'].values())
        else:
            before=torch.load(d/c/'initial.pt',weights_only=True);assert not before['adversary_optimizer']['state'];assert before['counts']['observer_steps']==0
            learned_initial=torch.load(d/'observer_initial.pt',weights_only=True)
            assert tree_digest({k:v for k,v in before['adversary_state'].items() if k.startswith('B__')})==tree_digest({k:v for k,v in learned_initial['adversary_state'].items() if k.startswith('B__')})
        rep['training_systems']+=1


def literal_diagnostic(cp,pca,anchors,labels,priors,expected,report):
    state={k:v.detach().clone().requires_grad_(k.startswith('branch.')) for k,v in cp['model_state'].items()}
    x=torch.tensor((pca.astype(float)-state['input_mean'].numpy())/state['input_scale'].numpy(),dtype=torch.float32)
    h=torch.relu(torch.nn.functional.linear(x,state['branch.mapper.0.weight'],state['branch.mapper.0.bias']))
    h=torch.nn.functional.linear(h,state['branch.mapper.2.weight'],state['branch.mapper.2.bias'])
    tasks={}
    for target in ('income_binary','civilian_at_work'):
        logits=torch.nn.functional.linear(h,state[f'branch.heads.{target}.weight'],state[f'branch.heads.{target}.bias']).reshape(-1);y=torch.tensor(labels[target]);known=y>=0
        tasks[target]=torch.nn.functional.binary_cross_entropy_with_logits(logits[known],y[known].float()) if known.any() else logits.sum()*0
    aa=torch.cat((torch.tensor(anchors['A'],dtype=torch.float32),h),1);bb=torch.tensor(anchors['B'],dtype=torch.float32);views={'A':aa,'B':bb,'AB':torch.cat((aa,bb),1)};ce={}
    roles={'A':('public_coverage','SEX','RAC1P'),'B':('income_binary','civilian_at_work','SEX','RAC1P'),'AB':('SEX','RAC1P')}
    for v,targets in roles.items():
        for t in targets:
            prefix=v+'__'+t+'.';z=views[v]
            for layer in ('0','2','4'):
                z=torch.nn.functional.linear(z,cp['adversary_state'][prefix+layer+'.weight'],cp['adversary_state'][prefix+layer+'.bias'])
                if layer!='4':z=torch.relu(z)
            y=torch.tensor(labels[t]);known=y>=0;ce[v,t]=torch.nn.functional.cross_entropy(z[known],y[known]) if known.any() else z.sum()*0
    normalized={v:torch.stack([ce[v,t]/priors[t]['entropy'] for t in ts]).mean() for v,ts in roles.items()}
    sen={v:torch.stack([ce[v,t]/priors[t]['entropy'] for t in ('SEX','RAC1P')]).mean() for v in ('A','B')}
    losses={'source':.25*tasks['income_binary']+.25*tasks['civilian_at_work'],'individual':(normalized['A']+normalized['B'])/2,'extra_local':sen['A']+sen['B'],'coalition':normalized['AB']}
    for name,value in losses.items():assert float(value.detach())==expected['losses'][name]
    for group,prefix in [('mapper','branch.mapper.'),('heads','branch.heads.')]:
        parameters=[v for k,v in state.items() if k.startswith(prefix)]
        for name,loss in losses.items():
            grad=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
            norm=math.sqrt(sum(float((g.double()**2).sum()) for g in grad if g is not None))
            err=abs(norm-expected['gradients'][group][name]['raw_norm']);assert err<=1e-12
            assert abs(abs(expected['coefficients'][name])*norm-expected['gradients'][group][name]['applied_norm'])<=1e-12
            report['gradient_norm_checks']+=1
    assert not views['B'].requires_grad
    report['literal_loss_diagnostics']+=1


def run(out):
    started=time.perf_counter();cfg=read(out/'config.json');gate=read(out/'RELEASE_MANIFEST.json');assert gate['all18_frozen'] and gate['training_closed'];identity=read(out/'PREFIT_IDENTITY.json')
    rep={'started_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'training_systems':0,'anchor_prediction_arrays':0,'release_arrays':0,'candidate_records':0,'prediction_arrays':0,'score_dictionaries':0,'selection_pools':0,'ancestor_parities':0,'B_parities':0,'terminal_checkpoints':0,'saved_start_fidelity':0,'fit_metadata_checks':0,'historical_refits':0,'mlp_schedules':0,'actual_terminal_checkpoints':0,'nested_trajectory_prefixes':0,'gradient_norm_checks':0,'literal_loss_diagnostics':0};cache={};seen=set();score_cache={};terminal_seen=set();terminals={}
    parent=read(ROOT/cfg['parent_results']/'config.json');assert check.sha(ROOT/parent['raw_path'])==read(ROOT/cfg['parent_results']/'schema_support.json')['raw_sha256'];frame,_=load_cohort(ROOT/parent['raw_path'],parent['sample_cap'],parent['sample_seed'])
    for s in range(3):
        print('REPLAY',s,flush=True);verify_training(out,s,rep);d=out/f'seed_{s}';pools=split_households(frame,s);
        for path,h in identity[str(s)]['cached_inputs'].items():assert check.sha(ROOT/path)==h
        frames={p:frame.iloc[ix] for p,ix in pools.items()};label={(p,t):check.labels(f,t) for p,f in frames.items() for t in TARGETS};pca=arrays(d/'pca.npz');anchor=arrays(d/'anchors.npz');teacher=arrays(d/'teacher.npz');rows=arrays(d/'split_rows.npz')
        for p,ix in pools.items():assert np.array_equal(rows[p],frame.iloc[ix]._raw_row.to_numpy())
        original_pred=arrays(ROOT/cfg['reference_results']/f'seed_{s}'/'predictions.npz')
        for t,record in identity[str(s)]['anchors'].items():
            path=ROOT/record['path'];meta=read(path/'metadata.json');predict,mean,scale,state=check.load_inference(path,meta)
            offset={'income_binary':('A',0),'civilian_at_work':('A',2),'public_coverage':('B',0)}[t];v,j=offset
            for pool,x in pca.items():
                actual=predict(x);expected=anchor[pool+'/'+v][:,j:j+2];assert np.array_equal(actual,expected),(s,t,pool);rep['anchor_prediction_arrays']+=1
            for split,pool in [('validation','downstream_validation'),('test','test')]:
                y,mask=label[pool,t];p=anchor[pool+'/'+v][mask,j:j+2];assert np.array_equal(p,original_pred[f'{split}/transfer/E_pca/{t}/{record["candidate_id"]}'])
                for w,key in [(None,'unweighted'),(frames[pool].PWGTP.to_numpy(float)[mask],'PWGTP')]:check.compare(check.independent_scores(y[mask],p,2,w),record['checks'][split][key],f'anchor/{s}/{t}/{split}/{key}')
        hpred=None;b_scores={}
        tm=read(d/'training/training.json');ix0=np.random.default_rng(1280000+100*s).permutation(len(pca['representation_fit']))[:256]
        diagnostic_labels={t:np.where(label['representation_fit',t][1],label['representation_fit',t][0],-1)[ix0] for t in TARGETS[:5]}
        for condition in ('A0','L025','L20','J'):
            for stage,file in [('fork_diagnostic','fork.pt'),('final_diagnostic','final.pt')]:
                cp0=torch.load(d/'training'/condition/file,weights_only=True)
                assert tm['arms'][condition][stage]['coefficients']==dict(zip(('source','individual','extra_local','coalition'),{'A0':(1.,0.,0.,0.),'L025':(1.,-.1,-.025,0.),'L20':(1.,-.1,-.2,0.),'J':(1.,-.1,0.,-.1)}[condition]))
                literal_diagnostic(cp0,pca['representation_fit'][ix0],{v:anchor['representation_fit/'+v][ix0] for v in ('A','B')},diagnostic_labels,tm['priors'],tm['arms'][condition][stage],rep)
        for c in ARMS:
            dest=d/c;done=read(dest/'complete.json');selected=read(dest/'selection_before_test.json');audits=read(dest/'audits/audit_selection.json');assert gate['created_utc']<=selected['created_utc']<=done['created_utc'];cp=torch.load(d/'training'/c/'final.pt',weights_only=True);released=arrays(d/'training'/c/'releases.npz')
            frozen=gate['seeds'][str(s)]['conditions'][c]
            assert check.sha(d/'training'/c/'final.pt')==frozen['checkpoint_sha256'] and check.sha(d/'training'/c/'releases.npz')==frozen['release_sha256']
            for pool,x in pca.items():
                aa=anchor[pool+'/A'];bb=anchor[pool+'/B'];aux,ps=(literal_aux(cp,x) if c not in ('H','E') else (teacher[pool].astype(float),None) if c=='E' else (None,None))
                aw=aa if aux is None else np.column_stack((aa,aux));wire={'A':aw,'B':bb,'AB':np.column_stack((aw,bb))};derived=None if ps is None else {'A':np.column_stack((aa,*ps)),'B':bb,'AB':np.column_stack((aa,*ps,bb))}
                for space,views in [('wire',wire),('derived',derived)]:
                    if views is None:continue
                    for v,x0 in views.items():assert np.array_equal(x0,released[f'{space}/{v}/{pool}']);assert x0.dtype==np.float64;rep['release_arrays']+=1
            for b,roles in audits['candidates'].items():
                for role,cs in roles.items():
                    for scope,ids in ((k,audits['selection_pools'][b][role][k]) for k in ('standard_independent','expanded_independent','expanded_catchup')):
                        expected=[cid for cid,meta in cs.items() if not meta['diagnostic_only'] and (scope=='expanded_catchup' or 'catchup' not in cid) and (scope!='standard_independent' or meta['space']=='wire')]
                        assert sorted(expected)==ids
                        assert min(ids,key=lambda k:(cs[k]['validation_scores']['log_loss'],k))==audits['selections'][b][role][scope];rep['selection_pools']+=1
            predictions=arrays(dest/'predictions.npz');metrics=read(dest/'metrics.json')['raw_metrics']
            for row in metrics:
                role,v,t,cid,b=(row[k] for k in ('role','view','target','candidate_id','audit_budget'));k=9 if t=='RAC1P' else 2
                if role=='audit':
                    meta=audits['candidates'][str(b)][v+'/'+t][cid];path=Path(meta['base_candidate_directory']);base=read(path/'metadata.json');space=meta['space'];cols=meta['projection_columns']
                    # Only own condition or legal H dependencies are admissible.
                    assert path.is_relative_to(dest/'audits') or (c!='H' and path.is_relative_to(d/'H'/'audits'))
                    assert check.sha(path/'metadata.json')==meta['base_metadata_sha256']
                else:
                    record=selected['utility_metadata'][v+'/'+t];path=Path(record['alias'])/cid if 'alias' in record else dest/'fitted/utility'/v/t/cid;base=read(path/'metadata.json');assert base==record['candidates'][cid];meta=base;space='wire';cols=None
                    assert selected['utility'][v+'/'+t]==min(record['candidates'],key=lambda name:(record['candidates'][name]['validation_scores']['log_loss'],name))
                def inputs(pool,mask):
                    x=released[f'{space}/{v}/{pool}'][mask]
                    if cols is not None:x=x[:,cols]
                    return np.ascontiguousarray(x)
                if str(path) not in cache:cache[str(path)]=check.load_inference(path,base)
                predict,mean,scale,state=cache[str(path)]
                fp,vp=('downstream_fit','downstream_validation') if role=='utility' else ('attacker_fit','attacker_validation');yf,mf=label[fp,t];yv,mv=label[vp,t]
                fi=subset_indices(np.where(mf,yf,-1),2048 if role=='utility' else 4096,(1230000 if role=='utility' else 1240000)+100*s+(TASKS if role=='utility' else TARGETS).index(t));xf=inputs(fp,fi);xv=inputs(vp,mv)
                if str(path) not in seen:
                    seen.add(str(path));assert base['validation_hashes']=={'x':check.array_hash(xv),'y':check.array_hash(yv[mv].astype(np.int64))}
                    if base.get('audit_kind')!='saved':assert base['fit_hashes']=={'x':check.array_hash(xf),'y':check.array_hash(yf[fi].astype(np.int64))}
                    if base.get('audit_kind') in ('saved','catchup'):
                        assert np.array_equal(mean,np.zeros(len(mean))) and np.array_equal(scale,np.ones(len(scale)))
                        origin='H' if path.is_relative_to(d/'H'/'audits') else c;ocp=torch.load(d/'training'/origin/'final.pt',weights_only=True);ov=meta['source_view'];prefix=ov+'__'+t+'.';os={n.removeprefix(prefix):x for n,x in ocp['adversary_state'].items() if n.startswith(prefix)}
                        assert check.state_hash(os)==base['source_state_hash']
                        if base['audit_kind']=='saved':assert np.array_equal(predict(xv),check.network_probability(os,xv));rep['saved_start_fidelity']+=1
                    else:
                        assert np.array_equal(mean,xf.mean(0));sd=xf.std(0);assert np.array_equal(scale,np.where(sd>1e-12,sd,1.))
                    expected_seed=(1250000+100*s+TASKS.index(t)) if role=='utility' else (1300000 if base.get('audit_kind') in ('saved','catchup') else 1260000)+100*s+10*('A','B','AB').index(meta['source_view'])+TARGETS.index(t)
                    if path.name=='mlp_1':expected_seed+=10000
                    assert base['seed']==expected_seed
                    check_curves(path,base,len(fi),expected_seed,b,rep,terminals)
                    rep['fit_metadata_checks']+=1
                for split,pool in [('validation',vp),('test','test')]:
                    y,valid=label[pool,t];key=f'{role}/{v}/{t}/{b}/{cid}/{split}';p=predictions[key];actual=predict(inputs(pool,valid));assert np.array_equal(actual,p),(s,c,key);rep['prediction_arrays']+=1
                    for suffix,weighted in [('',False),('_person_weighted',True)]:
                        sk=(s,pool,t,check.array_hash(p),weighted)
                        if sk not in score_cache:score_cache[sk]=check.independent_scores(y[valid],p,k,frames[pool].PWGTP.to_numpy(float)[valid] if weighted else None)
                        check.compare(score_cache[sk],row['scores'][split+suffix],f'{s}/{c}/{key}{suffix}');rep['score_dictionaries']+=1
                    if c!='H' and cid.startswith('anchor__'):
                        assert np.array_equal(p,hpred[f'{role}/{v}/{t}/{b}/{cid.removeprefix("anchor__")}/{split}']);rep['ancestor_parities']+=1
                    if role=='audit' and cid.startswith('inherited_'):
                        sv,scid=cid.removeprefix('inherited_').split('__',1);assert np.array_equal(p,predictions[f'audit/{sv}/{t}/{b}/{scid}/{split}']);rep['ancestor_parities']+=1
                rep['candidate_records']+=1
            if c=='H':hpred=predictions
            # Actual B selected probabilities and score dictionaries identical across all six names.
            for row in metrics:
                if row['view']!='B' or not row['selected_scopes']:continue
                for scope in row['selected_scopes']:
                    key=(row['role'],row['target'],row['audit_budget'],scope)
                    if c=='H':b_scores[key]=row['scores']
                    else:assert b_scores[key]==row['scores'];rep['B_parities']+=1
            for p in (dest/'audits').rglob('last_training_*.pt'):
                if str(p) in terminal_seen:continue
                terminal_seen.add(str(p));z=torch.load(p,weights_only=False);assert z;rep['terminal_checkpoints']+=1
            for p,h in done['files_sha256'].items():assert check.sha(ROOT/p)==h
    assert not check.errors,check.errors[:5]
    rep['terminal_manifest']=terminals
    rep.update(completed_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),runtime_seconds=time.perf_counter()-started,max_score_error=check.max_error,numeric_comparisons=check.numeric_comparisons,passed=True)
    return rep
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'results/redesign_20260909_acs_fixed_predictions_v1');a=p.parse_args();torch.set_num_threads(1)
    with threadpool_limits(limits=1):r=run(a.out)
    (a.out/'INDEPENDENT_REPLAY.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
