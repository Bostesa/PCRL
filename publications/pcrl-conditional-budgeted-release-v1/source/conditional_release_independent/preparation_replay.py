"""Independent donor/context/budget and guard coefficients from admitted FIT."""
from __future__ import annotations
import csv,json,os,hashlib,argparse
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from .reconstruction import digest,admitted_arrays,folds,source_joint_law
from .contexts import context_cells,hierarchy_from_counts


def costs(probabilities, labels):
    p=np.asarray(probabilities); y=np.asarray(labels)
    one=np.zeros((len(y),p.shape[-1]));one[np.arange(len(y)),y]=1
    return {'accuracy':-(np.argmax(p,axis=-1)==y[:,None]).astype(float),
            'nll':-np.log(np.maximum(p[np.arange(len(y))[:,None],np.arange(p.shape[1])[None,:],y[:,None]],1e-12)),
            'brier':np.sum((p-one[:,None,:])**2,axis=-1)}


def constraint_reconstruction(source,seed,prep):
    record=source['seeds'][str(seed)]
    wire=admitted_arrays(record['roles']['OSF_DEFENSE_FIT']['release'],{'row_id','unit','code_r1','code_r2','prob_r1','prob_r2','decision_r1','decision_r2'})
    receipt=json.loads(Path(record['policies']['V_J']['receipt']['path']).read_text())
    arrays=admitted_arrays(record['policies']['V_J']['package'],set(receipt['array_sha256']))
    a,b,donor=source_joint_law(arrays,wire['code_r1'],wire['code_r2'],1.)
    cells=context_cells(wire['prob_r1'],wire['prob_r2']);counts=np.bincount(cells,minlength=96)
    hierarchy=hierarchy_from_counts(counts);contexts=np.asarray(hierarchy['raw_to_context'])[cells]
    donor[contexts<0]=0;donor[contexts<0,0]=1
    for key,expected in (('code_r1',a),('code_r2',b),('b0_code_r1',wire['code_r1']),('b0_code_r2',wire['code_r2']),('donor',donor),('contexts',contexts),('fold',folds(wire['unit'])),('raw_fit_counts',counts),('context_mapping',hierarchy['raw_to_context']),('merged_fit_counts',hierarchy['merged_counts'])):
        if not np.allclose(prep[key],expected,atol=1e-14,rtol=0):raise ValueError('independent preparation mismatch: '+key)
    labels=[]
    for name in ('income','occupation'):
        item=admitted_arrays(source['roles']['OSF_DEFENSE_FIT'][name],{'row_id','unit',name})
        if not all(np.array_equal(item[k],wire[k]) for k in ('row_id','unit')):raise ValueError('task pairing mismatch')
        labels.append(item[name])
    refs=admitted_arrays(source['roles']['OSF_DEFENSE_FIT'][f'references_s{seed}'],{'row_id','unit','U0_1','U0_2','Ucal_1','Ucal_2','d_1','d_2'})
    if not all(np.array_equal(refs[k],wire[k]) for k in ('row_id','unit')):raise ValueError('reference pairing mismatch')
    action_costs={};reference_costs={ref:{} for ref in ('U0','Ucal')}
    for recipient,code,label in ((1,a,labels[0]),(2,b,labels[1])):
        action_costs.update({f'r{recipient}_{m}':v for m,v in costs(arrays[f'probability{recipient}'][code],label).items()})
        for ref in reference_costs:
            reference_costs[ref].update({f'r{recipient}_{m}':v[:,0] for m,v in costs(refs[f'{ref}_{recipient}'][:,None,:],label).items()})
    index_entry=record['fixed_readers']['index'];archive_entry=record['fixed_readers']['arrays']
    if digest(index_entry['path'])!=index_entry['sha256'] or digest(archive_entry['path'])!=archive_entry['sha256']:raise PermissionError('reader archive binding')
    index=json.loads(Path(index_entry['path']).read_text())
    with ZipFile(archive_entry['path']) as z:
        if any(not name.endswith('.npy') or not name.startswith('r') for name in z.namelist()):raise PermissionError('unadmitted reader member')
    readers={}
    with np.load(archive_entry['path'],allow_pickle=False) as archive:
        for i,reader in enumerate(index['readers']):
            if reader['view'] in ('r1','r2'):
                loss=archive[f'r{i}_loss'].copy();v=np.ascontiguousarray(loss)
                got=hashlib.sha256(str(v.dtype).encode()+str(v.shape).encode()+v.tobytes()).hexdigest()
                if got!=reader['loss_sha256']:raise ValueError('independent reader loss hash')
                readers[reader['name']]=loss
    delta=donor.copy();delta[:,0]-=1
    views=[('pooled',np.ones(len(donor),bool))]+[(f'fold{k}',prep['fold']==k) for k in range(3)]
    results={};translations=[]
    for mode in ('scalar','conditional'):
        param=np.zeros(len(contexts),int) if mode=='scalar' else np.maximum(contexts,0)
        size=1 if mode=='scalar' else max(1,len(hierarchy['labels']))
        for budget in ('STRICT','ORIGINAL'):
            coefficients=[];rhs=[];names=[]
            for view,mask in views:
                count=int(mask.sum())
                def slope(loss):
                    return np.bincount(param[mask],weights=np.sum(delta[mask]*loss[mask],axis=1)/count,minlength=size)
                for key,loss in action_costs.items():
                    metric=key.rsplit('_',1)[1];baseline=loss[:,0]
                    coeff=slope(loss-baseline[:,None])
                    for ref in (('P0',) if budget=='STRICT' else ('U0','Ucal')):
                        caps={'accuracy':.002,'nll':.001,'brier':.0005} if budget=='STRICT' else {'accuracy':.01,'nll':.01,'brier':.005}
                        excess=0. if ref=='P0' else float(np.mean(baseline[mask]-reference_costs[ref][key][mask]))
                        remaining=caps[metric]-excess
                        names.append(f'{budget}/{view}/{key}/{ref}');coefficients.append(coeff);rhs.append(remaining)
                        if mode=='scalar':translations.append((budget,view,key,ref,excess,remaining))
                for name,loss in readers.items():
                    names.append(f'CE_GUARD/{view}/{name}');coefficients.append(-slope(loss-loss[:,0,None]));rhs.append(1e-10)
            results[f'{mode}/{budget}']={'names':names,'matrix':np.asarray(coefficients),'rhs':np.asarray(rhs)}
    return {'constraints':results,'translations':translations,'hierarchy':hierarchy,'donor':donor,'wire':wire,'action_costs':action_costs,'reference_costs':reference_costs,'reader_losses':readers}


def run(config):
    from conditional_release.scheduler import assert_active_from_env
    active=assert_active_from_env(stage='verification')
    state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text())
    registered={i['path']:i['sha256'] for i in state['units'][active['unit_id']]['spec']['sources']}
    if registered.get(str(Path(config).absolute()))!=digest(config):raise PermissionError('replay config unqueued')
    cfg=json.loads(Path(config).read_text())
    for item in (cfg['manifest'],cfg['prep'],cfg['report'],cfg['translation']):
        if registered.get(item['path'])!=item['sha256'] or digest(item['path'])!=item['sha256']:raise PermissionError('independent replay input unqueued')
    source=json.loads(Path(cfg['manifest']['path']).read_text())
    prep=admitted_arrays(cfg['prep'],{'context_mapping','raw_fit_counts','merged_fit_counts','contexts','donor','fold','code_r1','code_r2','b0_code_r1','b0_code_r2','valid'})
    expected=constraint_reconstruction(source,cfg['seed'],prep)
    report=json.loads(Path(cfg['report']['path']).read_text())
    for setup,record in report['setup_dimensions_and_identity'].items():
        calc=expected['constraints'][setup]
        if record['constraint_names']!=calc['names'] or record['identity_feasible']!=bool(np.all(calc['rhs']>=0)):raise ValueError('constraint completeness or identity mismatch')
    with Path(cfg['translation']['path']).open() as f:rows=list(csv.DictReader(f))
    errors=[]
    for row,independent in zip(rows,expected['translations'],strict=True):
        budget,view,key,ref,excess,rhs=independent
        if (row['budget'],row['view'],row['task_metric'],row['reference'])!=(budget,view,key,ref):raise ValueError('budget translation identity')
        errors.extend([abs(float(row['p0_excess'])-excess),abs(float(row['remaining_rhs'])-rhs)])
    if max(errors)>1e-12:raise ValueError('budget translation values')
    private=Path(cfg['private_output']);private.mkdir(parents=True,exist_ok=False)
    for name,value in expected['constraints'].items():
        with (private/(name.replace('/','_')+'.npz')).open('xb') as f:np.savez_compressed(f,matrix=value['matrix'],rhs=value['rhs'])
    result={'schema':'CBR1-independent-preparation-replay-v1','status':'PASS','seed':cfg['seed'],'unit_id':active['unit_id'],
            'source_manifest_sha256':cfg['manifest']['sha256'],'preparation_sha256':cfg['prep']['sha256'],
            'independent_context_count':len(expected['hierarchy']['labels']),'fallback_rows':int(np.sum(prep['contexts']<0)),
            'translation_rows':len(rows),'maximum_translation_gap':max(errors),'complete_constraint_counts':{k:len(v['names']) for k,v in expected['constraints'].items()},
            'guard_reader_count':len(expected['reader_losses']),'new_reader_fits':0,'SEX_array_read':False,
            'scope':'Independent exact source donor/context/fold preparation, original/strict budget RHS and all restored local-reader guard coefficients. Cached independently derived constraints private for later solve replay.'}
    with Path(cfg['output']).open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'status':'PASS','seed':cfg['seed'],'constraints':sum(result['complete_constraint_counts'].values())}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);run(parser.parse_args().config)
