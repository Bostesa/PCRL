"""Independent control factor provenance, frozen-stage and actual FIT caps.

No optimizer, runner law, deployment, metric or selection implementation is
imported. Serialized LP arithmetic is checked by control_replay separately.
"""
import argparse, json, os
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from .reconstruction import digest, admitted_arrays, folds, merge_rows
from .primary_replay import load_archive
from .policy_replay import source_numbers, cost_tables
from .policy_laws import checked_law
from .contexts import context_cells
from .arithmetic import bayes_objective


def allowed_bounds(groups, valid):
    count=int(groups.max())+1
    support=np.ones((count,valid.shape[1]),bool)
    for group in range(count):
        rows=valid[groups==group]
        support[group]=rows.all(0) if len(rows) else False
        if not len(rows):support[group,0]=True
    return support


def second_keys(contexts, emitted, valid):
    counts={}
    for group,row,mask in zip(contexts,emitted,valid,strict=True):
        weight=1/int(mask.sum())
        for code in row[mask]:
            key=(int(group),int(code));counts[key]=counts.get(key,0.)+weight
    parents=[(int(g),-1) for g in np.unique(contexts)]
    children=sorted((k for k,v in counts.items() if v>=64),key=lambda k:(-counts[k],k))[:512-len(parents)]
    return np.asarray(parents+children,np.int64)


def run(config):
    from conditional_release.scheduler import assert_active_from_env
    active=assert_active_from_env(stage='verification')
    state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text())
    pins={x['path']:x['sha256'] for x in state['units'][active['unit_id']]['spec']['sources']}
    def checked(e):
        if pins.get(e['path'])!=e['sha256'] or digest(e['path'])!=e['sha256']:raise PermissionError('unregistered control replay input')
    if pins.get(str(Path(config).absolute()))!=digest(config):raise PermissionError('unregistered control replay config')
    cfg=json.loads(Path(config).read_text());checked(cfg['manifest']);source=json.loads(Path(cfg['manifest']['path']).read_text());results=[]
    for unit in cfg['controls']:
        checked(unit['receipt']);record=json.loads(Path(unit['receipt']['path']).read_text());seed=record['seed'];kind=record['kind']
        if record['status']!='COMPLETE':
            results.append({'unit_id':record['unit_id'],'status':record['status'],'selected_policy_available':False});continue
        checked(unit['policy_receipt']);receipt=json.loads(Path(unit['policy_receipt']['path']).read_text());checked(unit['package'])
        with ZipFile(unit['package']['path']) as z:names={x[:-4] for x in z.namelist()}
        package=admitted_arrays(unit['package'],names);numbers,_=source_numbers(source,seed)
        old=admitted_arrays(source['seeds'][str(seed)]['roles']['OSF_DEFENSE_FIT']['release'],{'row_id','unit','code_r1','code_r2','prob_r1','prob_r2','decision_r1','decision_r2'})
        sex=admitted_arrays(source['roles']['OSF_DEFENSE_FIT']['sex'],{'row_id','unit','sex'})
        if not all(np.array_equal(old[k],sex[k]) for k in ('row_id','unit')):raise ValueError('control FIT people mismatch')
        c1,c2=old['code_r1'],old['code_r2'];metadata=record['support_metadata']
        widths=tuple(max(len(d[f'recipient{r}_templates']) for d in metadata['contexts']) for r in (1,2))
        if kind=='LOCAL':
            _,contexts=np.unique(6*numbers['probability1'][c1].argmax(1)+numbers['probability2'][c2].argmax(1),return_inverse=True)
            own=[np.unique(c1),np.unique(c2)];groups=[np.searchsorted(own[0],c1),np.searchsorted(own[1],c2)]
        else:
            contexts=package['context_mapping'][context_cells(old['prob_r1'],old['prob_r2'])];groups=[np.maximum(contexts,0)]*2
        validity=[np.zeros((len(c1),w),bool) for w in widths]
        for v in validity:v[:,0]=True
        for d in metadata['contexts']:
            for r,v in enumerate(validity,1):v[contexts==d['context'],:len(d[f'recipient{r}_templates'])]=True
        archives=[]
        for item in record['solves']:
            e={'path':str(Path(unit['private_root'])/item['solve']['archive_name']),'sha256':item['solve']['archive_sha256']};checked(e);archives.append(load_archive(e))
        provenance=[]
        if kind in ('LOCAL','PRODUCT'):
            supports=[allowed_bounds(g[contexts>=0],v[contexts>=0]) for g,v in zip(groups,validity)]
            best=None;factors=None;laststart=None
            for index,(item,archive) in enumerate(zip(record['solves'],archives,strict=True)):
                start=item['start'];r=item['recipient']-1
                if start!=laststart:
                    factors=[]
                    for support in supports:
                        f=np.zeros(support.shape);f[:,0]=1 if start=='keep' else 0
                        if start=='uniform':f=support/support.sum(1,keepdims=True)
                        factors.append(f)
                    laststart=start
                if not item['solve']['success']:continue
                factors[r]=archive['theta'].reshape(-1,widths[r]).copy()
                trial=dict(package)
                if kind=='LOCAL':
                    for j,key in enumerate(('first_weights','second_weights')):
                        table=np.zeros(package[key].shape);table[:,0]=1;table[own[j],:widths[j]]=factors[j];trial[key]=table
                else:trial.update(first_weights=factors[0],second_weights=factors[1])
                a,b,q=checked_law(numbers,c1,c2,'CONTROL',receipt,trial);value=bayes_objective(merge_rows(a,b,q),sex['sex'])
                gap=abs(value-float(archive['objective']@archive['primal']))
                if gap>1e-7:raise ValueError('coordinate objective differs from complete deployed tuple')
                provenance.append({'start':start,'round':item['round'],'recipient':r+1,'actual_tuple_objective':value,'epigraph_gap':gap})
                if best is None or value<best['value']-1e-12:best={'value':value,'index':index,'provenance':provenance[-1],'first':trial['first_weights'].copy(),'second':trial['second_weights'].copy()}
            if best is None or not all(np.array_equal(package[key],best[name]) for key,name in (('first_weights','first'),('second_weights','second'))):raise ValueError('selected factors do not equal earliest best registered coordinate')
            selected={'archive_index':best['index'],**best['provenance']}
        elif kind.startswith('SEQ'):
            first=1 if kind=='SEQ12' else 2
            if receipt['first_recipient']!=first:raise ValueError('sequential order differs')
            if not np.array_equal(package['first_weights'].ravel(),archives[0]['theta']) or not np.array_equal(package['second_weights'].ravel(),archives[1]['theta']):raise ValueError('sequential first was refitted or selected weights differ from stages')
            active_rows=contexts>=0;owncodes=c1 if first==1 else c2
            emitted=np.repeat(owncodes[:,None],widths[first-1],axis=1)
            for d in metadata['contexts']:
                row=np.flatnonzero(contexts==d['context']);axis=np.asarray(d[f'recipient{first}_templates'])
                out=np.broadcast_to(axis,(len(row),len(axis))).copy();out[:,axis==-1]=owncodes[row,None];emitted[row,:len(axis)]=out
            keys=second_keys(contexts[active_rows],emitted[active_rows],validity[first-1][active_rows])
            if not np.array_equal(keys,package['second_keys']):raise ValueError('sequential groups differ from frozen potential actual-code hierarchy')
            selected={'first_stage_frozen_exactly':True,'second_keys_count':len(keys),'actual_emitted_first_code_groups':True}
        else:
            if not np.array_equal(package['first_weights'],archives[-1]['theta']):raise ValueError('deterministic weights differ from actual binary incumbent')
            checked(unit['independently_verified_program']);reference=load_archive(unit['independently_verified_program'])
            for archive in archives:
                for key in ('objective','inequality_data','inequality_indices','inequality_indptr','inequality_shape','inequality_rhs','equality_data','equality_indices','equality_indptr','equality_shape','equality_rhs','bounds'):
                    if archive[key].shape!=reference[key].shape or not np.allclose(archive[key],reference[key],rtol=0,atol=1e-12):raise ValueError('MIP full program differs from independently reconstructed WIDE program')
            selected={'actual_binary_incumbent_exact':True,'full_program_matches_independently_reconstructed_WIDE':True}
        a,b,q=checked_law(numbers,c1,c2,'CONTROL',receipt,package);fold=folds(old['unit']);caps=[]
        for r,code,labelname in ((1,a,'income'),(2,b,'occupation')):
            labels=admitted_arrays(source['roles']['OSF_DEFENSE_FIT'][labelname],{'row_id','unit',labelname})
            if not all(np.array_equal(labels[k],old[k]) for k in ('row_id','unit')):raise ValueError('control FIT task pairing')
            for metric,lookup in cost_tables(numbers['probability'+str(r)]).items():
                target=labels[labelname];base=lookup[target,(c1 if r==1 else c2)];excess=(q*lookup[target[:,None],code]).sum(1)-base
                cap={'accuracy':.002,'nll':.001,'brier':.0005}[metric]
                for view,mask in [('pooled',np.ones(len(q),bool))]+[(f'fold{j}',fold==j) for j in range(3)]:
                    value=float(excess[mask].mean());caps.append({'task':r,'metric':metric,'view':view,'excess':value,'cap':cap})
                    if value>cap+1e-7:raise ValueError('selected control violates matched STRICT fitting cap')
        objective=bayes_objective(merge_rows(a,b,q),sex['sex'])
        if abs(objective-record['fitted_objective'])>1e-7:raise ValueError('selected control objective differs')
        results.append({'unit_id':record['unit_id'],'seed':seed,'kind':kind,'status':'PASS','selected_provenance':selected,'registered_coordinate_attempts':provenance,'strict_fit_caps':caps,'actual_tuple_objective':objective,'INNER_SEX_access':False})
    report={'schema':'CBR1-independent-control-design-replay-v1','status':'PASS','unit_id':active['unit_id'],'results':results,'new_optimizer_fits':0,'scope':'Independent exact selected factor/stage bindings, label-free second-stage actual-code group hierarchy, all registered successful coordinate actual-tuple objectives and earliest-best selection, complete selected STRICT FIT task caps. Every fixed-reader guard and INNER utility row is checked by independent policy replay; LP arithmetic separately. No global product/sequential optimality claim.'}
    with Path(cfg['output']).open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'PASS','controls':len(results)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);run(p.parse_args().config)
