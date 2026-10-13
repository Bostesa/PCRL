"""Queued exact useful-head admission and FIT stratum accounting; zero fitting."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path


def arr_sha(a):
    import numpy as np
    z=np.ascontiguousarray(a)
    return hashlib.sha256(str(z.dtype).encode()+str(z.shape).encode()+z.tobytes()).hexdigest()


def metrics(prob,y):
    import numpy as np
    p=np.asarray(prob,np.float64);n=len(y);k=p.shape[1];d=p.argmax(1)
    target=np.eye(k)[y];rec=[float((d[y==c]==c).mean()) if (y==c).any() else None for c in range(k)]
    return {'accuracy':float((d==y).mean()),'nll':float(-np.log(np.maximum(p[np.arange(n),y],1e-12)).mean()),'brier':float(((p-target)**2).sum(1).mean()),'balanced_accuracy':float(np.mean([r for r in rec if r is not None])),'per_class_recall':rec,'class_counts':np.bincount(y,minlength=k).tolist(),'mean_confidence':float(p.max(1).mean())}


def strata(fit,mask):
    import numpy as np
    u=fit['unit'][mask];s=fit['sex'][mask];y1=fit['income'][mask];y2=fit['occupation'][mask]
    # Labels of an exact-record duplicate group must agree; support counts unique people.
    for g in np.unique(u):
        ix=u==g
        if len(np.unique(np.c_[s[ix],y1[ix],y2[ix]],axis=0))!=1:raise AssertionError('duplicate-group labels differ')
    def one(a,keys):
        records=[]
        for key in keys:
            hit=np.all(a==np.asarray(key)[None,:],axis=1);count=[len(np.unique(u[hit&(s==v)])) for v in (0,1)]
            records.append({'labels':list(key),'rows':int(hit.sum()),'mass':float(hit.mean()),'distinct_groups_by_sex':count,'supported':min(count)>=32})
        return {'strata':records,'covered_mass':sum(r['mass'] for r in records if r['supported']),'uncovered_mass':sum(r['mass'] for r in records if not r['supported'])}
    return {'income':one(y1[:,None],[(i,) for i in range(2)]),'occupation':one(y2[:,None],[(i,) for i in range(6)]),'pair':one(np.c_[y1,y2],[(i,j) for i in range(2) for j in range(6)])}


def run(manifest_path,manifest_sha256,output,coverage_output,private_output):
    # Import numerical packages only after scheduler has independently authenticated this child.
    from .scheduler import assert_active_from_env
    assert_active_from_env(stage='admission')
    import joblib,numpy as np,torch
    from .source import load_manifest,load_role,pinned,ROLES,write_json,sha_file
    from .useful_service import load_useful_model
    m=load_manifest(manifest_path,manifest_sha256);reports=[];pairings={};models={}
    # Guard behavior is exercised before any request can reach filesystem hashing.
    for role,field in [('INNER_SELECTION','sex'),('INNER_SELECTION','income'),('INNER_SELECTION','occupation'),('ATTACK_FIT_NEW','income')]:
        try:load_role(m,0,role,(field,))
        except PermissionError:pass
        else:raise AssertionError('label guard admitted barred prefit field')
    for seed in (0,1):
        model=load_useful_model(m,seed);models[str(seed)]=model.source_bindings
        calpins=m['seeds'][str(seed)]['ucal'];cal=json.loads(pinned(calpins['record']).read_text());complete=json.loads(pinned(calpins['complete']).read_text())
        if complete['files']['record.json']!=calpins['record']['sha256'] or cal['schema']!='hcal-calU-v1' or cal['seed']!=seed:raise AssertionError('Ucal provenance differs')
        heads=[joblib.load(pinned(m['seeds'][str(seed)]['pins'][k])) for k in ('head0','head1')]
        for role in ROLES:
            fields={'features','references'}|({'income','occupation','sex'} if role=='OSF_DEFENSE_FIT' else set())
            data=load_role(m,seed,role,fields)
            if role in pairings:
                if any(not np.array_equal(data[k],pairings[role][k]) for k in ('row_id','unit')):raise AssertionError('source seeds not paired')
            else:pairings[role]={k:data[k] for k in ('row_id','unit')}
            if tuple(map(str,data['feature_names']))!=model.names:raise AssertionError('source feature schema differs')
            wires=model.route_raw(data['X'],data['feature_names']);tasks=[]
            for i in range(2):
                z=wires[i]['representation'];p=wires[i]['probability'];reference=heads[i].predict_proba(z.astype(np.float64));stored=data['references'][f'U0_{i+1}']
                gap=float(np.max(np.abs(p-reference)));storedgap=float(np.max(np.abs(reference-stored)))
                if gap>2e-12 or storedgap>2e-12 or not np.array_equal(p.argmax(1),reference.argmax(1)):raise AssertionError(f'folded useful head probability parity failed {seed} {role} {i}: {gap}/{storedgap}')
                if not np.array_equal(p.argmax(1),data['references'][f'd_{i+1}']):raise AssertionError('stored decision mismatch')
                alpha=float(cal['tables'][f'H-GLOBAL-TEMP|r{i+1}']['alpha'])
                logp=np.log(np.maximum(reference,1e-12));logp-=np.log(np.maximum(reference,1e-12).sum(1))[:,None];v=alpha*logp;v=np.exp(v-v.max(1,keepdims=True));q=v/v.sum(1,keepdims=True)
                ucgap=float(np.max(np.abs(q-data['references'][f'Ucal_{i+1}'])))
                if ucgap>2e-12:raise AssertionError('Ucal stored probability differs')
                item={'task':('income','occupation')[i],'folded_vs_deployed_max_abs':gap,'deployed_vs_stored_max_abs':storedgap,'Ucal_vs_stored_max_abs':ucgap,'Ucal_alpha':alpha,'decision_parity':True,'representation_sha256':arr_sha(z),'wire_dimension':wires[i]['wire'].shape[1]}
                if role=='OSF_DEFENSE_FIT':
                    y=data[('income','occupation')[i]];item['FIT_metrics']={'deployed_U0':metrics(reference,y),'folded_U0':metrics(p,y),'stored_Ucal':metrics(data['references'][f'Ucal_{i+1}'],y)}
                tasks.append(item)
            reports.append({'seed':seed,'role':role,'rows':len(data['X']),'groups':len(np.unique(data['unit'])),'row_id_sha256':arr_sha(data['row_id']),'group_sha256':arr_sha(data['unit']),'tasks':tasks})
        if seed==0:fit=load_role(m,0,'OSF_DEFENSE_FIT',{'features','income','occupation','sex'})
    overlap=[]
    for a,b in [('OSF_DEFENSE_FIT','ATTACK_FIT_NEW'),('OSF_DEFENSE_FIT','INNER_SELECTION'),('ATTACK_FIT_NEW','INNER_SELECTION')]:
        rr=len(np.intersect1d(pairings[a]['row_id'],pairings[b]['row_id']));gg=len(np.intersect1d(pairings[a]['unit'],pairings[b]['unit']))
        if rr or gg:raise AssertionError('roles overlap')
        overlap.append({'a':a,'b':b,'rows':rr,'groups':gg})
    folds={str(i):{'rows':int((fit['fold']==i).sum()),'groups':len(np.unique(fit['unit'][fit['fold']==i]))} for i in range(3)}
    coverage={'schema':'TCR1-stratum-coverage-v1','minimum_distinct_groups_per_sex':32,'whole_FIT':strata(fit,np.ones(len(fit['X']),bool)),'FIT_train_folds01':strata(fit,fit['fold']!=2),'FIT_selection_fold2':strata(fit,fit['fold']==2),'fold_counts':folds,'FIT_fold_sha256':arr_sha(fit['fold']),'no_INNER_labels_opened':True}
    private={'schema':'TCR1-useful-source-bindings-v1','models':models,'feature_names':list(model.names),'source_manifest_sha256':manifest_sha256,'parameters':model.n_params()}
    report={'schema':'TCR1-source-admission-v1','status':'PASS','source_manifest_sha256':manifest_sha256,'source_roles':reports,'role_overlap':overlap,'fold_counts':folds,'parameters':model.n_params(),'class_orders':model.class_orders,'head_metadata':{s:models[s]['heads'] for s in models},'float_convention':'source encoders float32; folded heads float64; whole-role forwarding','probability_tolerance':2e-12,'INNER_task_labels_opened':False,'INNER_SEX_opened_or_hashed':False,'new_fits':0,'guard_fixtures':'PASS'}
    write_json(output,report);write_json(coverage_output,coverage);write_json(private_output,private,True)
    return {'status':'PASS','output_sha256':sha_file(output),'coverage_sha256':sha_file(coverage_output)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--output',required=True);p.add_argument('--coverage-output',required=True);p.add_argument('--private-output',required=True)
    a=p.parse_args();print(json.dumps(run(a.manifest,a.manifest_sha256,a.output,a.coverage_output,a.private_output)))
