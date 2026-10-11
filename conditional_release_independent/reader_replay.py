"""Independent frozen-reader prediction, ATT orientation and training-count replay."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
from .reconstruction import digest,admitted_arrays
from .arithmetic import integrated_auc,validate_group_pairing


def tables(bank,prior):
    p=[np.asarray(bank['recipients'][str(r)]['probability'],float) for r in (1,2)]
    if prior.shape!=(len(p[0]),len(p[1])) or np.any(prior<0) or abs(prior.sum()-1)>1e-10:raise ValueError('invalid public prior')
    out={};bits=[]
    for i,prob in enumerate(p):
        n,k=prob.shape;width=(n-1).bit_length();b=((np.arange(n)[:,None]>>np.arange(max(1,width)))&1).astype(float);bits.append(b)
        x=np.concatenate((np.eye(n),b,prob,np.eye(k)[prob.argmax(1)],np.full((n,1),i+1),np.full((n,1),width)),axis=1)
        marginal=prior.sum(axis=1-i)
        out['r'+str(i+1)]={'x':x,'bits':b,'probability':prob,'defense':np.c_[x,np.log(np.maximum(prob,1e-12)),np.log(np.maximum(marginal,1e-12))]}
    l=np.repeat(np.arange(len(p[0])),len(p[1]));r=np.tile(np.arange(len(p[1])),len(p[0]))
    out['pair']={'x':np.c_[out['r1']['x'][l],out['r2']['x'][r]],'bits':np.c_[bits[0][l],bits[1][r]],'probability':np.c_[p[0][l],p[1][r]],'defense':np.c_[out['r1']['defense'][l],out['r2']['defense'][r],(bits[0][l,:,None]*bits[1][r,None,:]).reshape(len(l),-1),np.log(np.maximum(prior.ravel(),1e-12))]}
    return out,l,r


def predictions(model,x):
    if model.get('scaler') is not None:x=(x-model['scaler'].mean_)/model['scaler'].scale_
    estimator=model['model']
    # Frozen third-party predictions, not runner scoring or refitting.
    return np.asarray(estimator.predict_proba(x)[:,list(estimator.classes_).index(1)],float)


def score_dicts(scores,mass):
    result=[]
    for ss,mm in zip(scores,mass):
        row={}
        for s,m in zip(ss,mm):
            if m>0:row[float(s)]=row.get(float(s),0.)+float(m)
        result.append(row)
    return result


def run(config):
    from conditional_release.scheduler import assert_active_from_env
    active=assert_active_from_env(stage='verification');state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text())
    pins={x['path']:x['sha256'] for x in state['units'][active['unit_id']]['spec']['sources']}
    def check(entry):
        if pins.get(entry['path'])!=entry['sha256'] or digest(entry['path'])!=entry['sha256']:raise PermissionError('unqueued reader input')
    if pins.get(str(Path(config).absolute()))!=digest(config):raise PermissionError('unqueued reader config')
    cfg=json.loads(Path(config).read_text());results=[]
    import joblib
    for item in cfg['readers']:
        check(item['bank']);bank=json.loads(Path(item['bank']['path']).read_text());root=Path(item['private_root'])
        def entry(name,sha):
            a={'path':str(root/name),'sha256':sha};check(a);return a
        check(item['receipt']);receipt=json.loads(Path(item['receipt']['path']).read_text())
        train=admitted_arrays(entry('TRAINING_INPUTS.npz',receipt['input_sha256']),{'row_id','unit','code_r1','code_r2','mass','sex','fit','tune'})
        prior=admitted_arrays(entry('PUBLIC_UNIFORM_PRIOR.npz',receipt['public_uniform_prior_sha256']),{'pair'})['pair']
        computed,left,right=tables(bank,prior)
        frozen_entry=entry('FROZEN_SCORE_TABLES.npz',receipt['score_table_sha256'])
        names={v+'__'+r for v in ('r1','r2','pair') for r in receipt['reader_order']+['SHUFFLED','PLANTED']}
        names.update('pair__IGNORE_'+ignored+'__'+r for ignored in ('R1','R2') for r in receipt['reader_order'])
        frozen=admitted_arrays(frozen_entry,names)
        validate_group_pairing(train['unit'],train['sex'])
        ranked=sorted((hashlib.sha256(('pcrl-method-portfolio-v1|attack-tune|20261008|'+str(int(g))).encode()).hexdigest(),int(g)) for g in np.unique(train['unit']))
        tune=np.flatnonzero(np.isin(train['unit'],[g for _,g in ranked[:1015]]));fit=np.flatnonzero(~np.isin(train['unit'],[g for _,g in ranked[:1015]]))
        if not np.array_equal(tune,train['tune']) or not np.array_equal(fit,train['fit']):raise ValueError('ATT split mismatch')
        indexes={'r1':train['code_r1'],'r2':train['code_r2'],'pair':train['code_r1']*len(bank['recipients']['2']['probability'])+train['code_r2']}
        maxgap=0.;checked=0
        for view in ('r1','r2','pair'):
            table_path=root/('TABLE_INPUTS_'+view+'.npz');check({'path':str(table_path),'sha256':item['table_sha256'][view]})
            archived=admitted_arrays({'path':str(table_path),'sha256':item['table_sha256'][view]},set(computed[view])|{'weighted_counts'})
            for key in computed[view]:
                gap=float(np.max(np.abs(archived[key]-computed[view][key]),initial=0));maxgap=max(maxgap,gap)
                if gap>1e-12:raise ValueError('independent wire feature table mismatch')
            counts=np.zeros((len(computed[view]['x']),2))
            for label in (0,1):
                use=fit[train['sex'][fit]==label]
                counts[:,label]=np.bincount(indexes[view][use].ravel(),weights=train['mass'][use].ravel(),minlength=len(counts))
            if np.max(np.abs(counts-archived['weighted_counts']),initial=0)>1e-10:raise ValueError('actual law training counts mismatch')
            for reader in receipt['reader_order']+['SHUFFLED','PLANTED']:
                modelname='MODEL_'+view+'__'+reader+'.joblib';model_entry=entry(modelname,receipt['model_files'][modelname]);model=joblib.load(model_entry['path'])
                kind='defense' if reader.startswith('DA_') else 'bits' if reader=='CODE_BIT_LR_C1' else 'probability' if reader.startswith('PROB_') else 'x'
                x=computed[view][kind]
                if reader=='CELL_A1':
                    prior_cell=(counts[:,1].sum()+1)/(counts.sum()+2);pred=np.full(len(counts),prior_cell);seen=counts.sum(1)>0;pred[seen]=(counts[seen,1]+1)/(counts[seen].sum(1)+2)
                    if not np.allclose(model['counts'],counts,rtol=0,atol=1e-10) or abs(model['prior']-prior_cell)>1e-12:raise ValueError('cell model mismatch')
                elif reader=='PLANTED':pred=predictions(model,np.c_[np.repeat(x,2,axis=0),np.tile([0.,1.],len(x))]).reshape(-1,2)
                else:pred=predictions(model,x)
                if reader not in ('SHUFFLED','PLANTED'):
                    auc=integrated_auc(score_dicts(pred[indexes[view][tune]],train['mass'][tune]),train['sex'][tune])
                    orient=1 if auc>=.5 else -1
                    reported=receipt['validation'][view][reader]
                    if abs(auc-reported['positive_auc'])>1e-12 or reported['selected_orientation']!=('positive' if orient==1 else 'reverse'):raise ValueError('ATT-only orientation mismatch')
                    if orient==-1:pred=1-pred
                gap=float(np.max(np.abs(pred-frozen[view+'__'+reader]),initial=0));maxgap=max(maxgap,gap)
                if gap>1e-12:raise ValueError('frozen model prediction mismatch')
                checked+=1
        for local,ix,ignored in (('r1',left,'R2'),('r2',right,'R1')):
            for reader in receipt['reader_order']:
                if not np.array_equal(frozen['pair__IGNORE_'+ignored+'__'+reader],frozen[local+'__'+reader][ix]):raise ValueError('ignore-other alias mismatch')
        results.append({'arm_id':item['arm_id'],'seed':item['seed'],'models_replayed':checked,'all_public_wire_states':True,'maximum_numeric_gap':maxgap,'ATT_group_split':'EXACT','law_weighted_counts':'PASS','ATT_orientation':'PASS','ignore_other_aliases':'EXACT','reader_receipt_sha256':item['receipt']['sha256']})
    report={'schema':'CBR1-independent-frozen-reader-replay-v1','status':'PASS','unit_id':active['unit_id'],'results':results,'reader_refits':0,'INNER_SEX_access':False,'scope':'Independent feature construction, weighted training counts, frozen third-party model predictions on every public wire state, ATT-only orientations and aliases. Public prior values are pinned inputs; their policy-law derivation is a separate deployment replay.'}
    with Path(cfg['output']).open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'PASS','banks':len(results),'models':sum(x['models_replayed'] for x in results)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);run(p.parse_args().config)
