"""Independent live or backup-only raw83 per-recipient runtime replay."""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path


def _canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def wires(inputs,item):
    import numpy as np
    import torch
    from .replay import forward
    manifest=json.loads(inputs.path(item['deployment']).read_text())
    fields={'schema','campaign','service_id','seed','per_recipient_variants','feature_names','feature_schema_sha256',
            'class_orders','model_sha256','source_checkpoint_sha256','provenance','contract','parameter_count','geometry'}
    variants={name:['head_row_space','full'] for name in ('CONSTRAINED','ORDINARY','LOCAL_TWIN','TASK_ONLY','RAW_HYBRID')}
    variants.update({name:['full','full'] for name in ('RAW_FULL','U0_FULL')})
    variants.update({name:['head_row_space','head_row_space'] for name in ('RAW_PROJ','U0_PROJ')})
    service=manifest.get('service_id')
    if (set(manifest)!=fields or manifest['schema']!='CIS1-raw83-own-head-deployment-v1'
        or manifest['campaign']!='pcrl_compressed_income_safeguard_v1' or service not in variants
        or manifest['per_recipient_variants']!=variants[service] or service!=item['arm_id']
        or manifest['seed']!=item['seed'] or manifest['seed'] not in (0,1)
        or manifest['class_orders']!=[[0,1],list(range(6))] or manifest['parameter_count']!=21288
        or manifest['model_sha256']!=item['model_state']['sha256']):
        raise ValueError('independent deployed service identity differs')
    contract={'recipient_full_dimensions':[21,29],'pair_full_dimensions':50,'recipient_score_dimensions':[3,7],
              'pair_score_dimensions':10,'deterministic':True,'head_reads_released_representation':True,
              'role_forward_batch_size':1024,'projection_arithmetic':'torch_float64_q=z@B.T;released=q@B',
              'decision_rule':'argmax probabilities; first class in fixed source order','raw_z_or_q_co_release':False}
    if manifest['contract']!=contract:raise ValueError('literal deployed wire contract differs')
    names=manifest['feature_names'];forbidden={'sex','race','income','occupation','fnlwgt','row_id','unit','id','person_id','sampling_id','group_id','hidden_flag','y','label','target','teacher','teacher_score'}
    if (len(names)!=83 or len(set(names))!=83 or any(not isinstance(n,str) or not n or n.lower().split('=')[0] in forbidden for n in names)
        or hashlib.sha256(json.dumps(names,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()!=manifest['feature_schema_sha256']):
        raise ValueError('exact deployed feature order differs')
    state=torch.load(inputs.path(item['model_state']),map_location='cpu',weights_only=True);shapes={}
    for i,k in enumerate((2,6)):
        for layer,inn,out in ((0,83,64),(2,64,64),(4,64,16)):
            shapes[f'enc.{i}.{layer}.weight']=(out,inn);shapes[f'enc.{i}.{layer}.bias']=(out,)
        shapes[f'head.{i}.weight']=(k,16);shapes[f'head.{i}.bias']=(k,)
    if set(state)!=set(shapes) or any(not isinstance(v,torch.Tensor) or tuple(v.shape)!=shapes[k] or v.dtype!=(torch.float32 if k.startswith('enc.') else torch.float64) or not torch.isfinite(v).all() for k,v in state.items()):
        raise ValueError('independent deployed model tensor schema')
    meta=manifest['geometry'];binding=json.loads(inputs.path(item['geometry']).read_text());body=dict(binding);digest=body.pop('binding_sha256',None)
    if (set(meta)!={'file','sha256','binding_sha256','checkpoint_sha256','head_tensor_sha256'} or meta['file']!='geometry.json'
        or meta['sha256']!=item['geometry']['sha256'] or digest!=_canonical(body) or digest!=meta['binding_sha256']
        or body.get('schema')!='CIS1-per-seed-geometry-v1' or body.get('campaign')!=manifest['campaign']
        or body.get('source_seed')!=manifest['seed'] or body.get('checkpoint_sha256')!=meta['checkpoint_sha256']
        or body.get('head_tensor_sha256')!=meta['head_tensor_sha256'] or len(body.get('heads',[]))!=2):
        raise ValueError('independent deployed fixed geometry identity')
    for i,variant in enumerate(variants[service]):
        if variant=='full':continue
        g=body['heads'][i];b=np.asarray(g['B'],dtype=np.float64).reshape(g['rank'],16);hashes={}
        for name in ('weight','bias'):
            v=state[f'head.{i}.{name}'].contiguous().numpy();h=hashlib.sha256();h.update(str(v.dtype).encode());h.update(str(v.shape).encode());h.update(v.tobytes());hashes[name]=h.hexdigest()
        a=state[f'head.{i}.weight'].numpy();a=a-a.mean(0)
        if (g['source_dimension']!=16 or g['class_order']!=manifest['class_orders'][i]
            or not 0<=g['rank']<=len(g['class_order'])-1 or _canonical(hashes)!=meta['head_tensor_sha256'][i]
            or not np.isfinite(b).all() or not np.allclose(b@b.T,np.eye(len(b)),atol=1e-10,rtol=1e-10)
            or not np.allclose((a@b.T)@b,a,atol=1e-10,rtol=1e-10)):
            raise ValueError('independent projected head/basis parity')
    raw=inputs.arrays(item['input'],{'X','feature_names'})
    if raw['feature_names'].tolist()!=names or raw['X'].dtype.kind!='f' or not np.isfinite(raw['X']).all():raise ValueError('deployment input schema differs')
    with np.errstate(over='ignore'):x=np.ascontiguousarray(raw['X'],dtype=np.float32)
    if not np.isfinite(x).all():raise ValueError('deployment input overflows source dtype')
    actual=forward({'schema':'TCR1-trainable-own-head-checkpoint-v1','class_orders':manifest['class_orders'],'model':state},x,
                   wire_contract=dict(zip(('income','occupation'),variants[service])),geometry_binding=binding)
    expected=inputs.arrays(item['expected_wire'],{'r1','r2'})
    if any(not np.array_equal(actual[i],expected[f'r{i+1}']) for i in (0,1)):
        raise AssertionError('independent deployed literal wire differs')
    return actual


def run(inputs):
    import numpy as np,joblib
    from .replay import reader_table
    from .registry import SERVICES
    cfg=inputs.cfg;packages=cfg['packages'];restored=cfg['operation']=='restore'
    if restored:
        if not {'CONSTRAINED','ORDINARY','RAW_HYBRID','RAW_PROJ'}<={p['arm_id'] for p in packages}:
            raise ValueError('candidate, ordinary, hybrid and compressed backup representatives required')
        if not any(p.get('readers') for p in packages):raise ValueError('selected frozen reader backup representative required')
    elif len(packages)!=18 or {(p['arm_id'],p['seed']) for p in packages}!={(s,n) for s in SERVICES for n in (0,1)}:
        raise ValueError('all18 independent deployed source-seed packages required')
    records=[]
    for item in packages:
        actual=wires(inputs,item);checks=[]
        for reader in item.get('readers',[]):
            saved=joblib.load(inputs.path(reader['model']))
            if set(saved)!={'model','mean','scale'} or not np.array_equal(saved['model'].classes_,[0,1]):raise ValueError('independent frozen reader artifact schema')
            table=reader_table(*actual,reader['reader'],reader['view']);mean=np.asarray(saved['mean']);scale=np.asarray(saved['scale'])
            if mean.shape!=(table.shape[1],) or scale.shape!=mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any() or reader['orientation'] not in (-1,1):raise ValueError('independent deployed reader preprocessing')
            score=saved['model'].predict_proba((table-mean)/scale)[:,1]
            if reader['orientation']==-1:score=1-score
            expected=inputs.arrays(reader['expected'],{'row_id','unit','score'})
            if not np.array_equal(score,expected['score']):raise AssertionError('independent restored frozen reader scores differ')
            checks.append({'reader':reader['reader'],'view':reader['view'],'orientation':reader['orientation'],'status':'PASS'})
        records.append({'arm_id':item['arm_id'],'seed':item['seed'],'rows':len(actual[0]),'max_absolute_wire_difference':[0.,0.],
                        'deployment_sha256':item['deployment']['sha256'],'reader_checks':checks,'status':'PASS'})
    return {'schema':'CIS1-independent-deployment-replay-v1','status':'PASS','records':records,
            'backup_only':restored,'all_numerical_sources_from_restored_tree':restored,
            'scientific_runner_imports':False,'labels_read_or_hashed':False,'new_fits':0}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);args=parser.parse_args()
    from compressed_income_safeguard.scheduler import assert_active_from_env
    assert_active_from_env()
    metadata=json.loads(Path(args.spec).read_text())
    if metadata['operation']=='restore':
        from .restore_worker import RestoredInputs
        inputs=RestoredInputs(args.spec)
        inputs.path(Path(__file__))
    elif metadata['operation']=='live':
        from .replay import Inputs
        inputs=Inputs(args.spec)
    else:raise ValueError('independent deployment operation')
    result=run(inputs);output=Path(inputs.cfg['public_output']);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':'PASS','packages':len(result['records'])}))

if __name__=='__main__':main()
