"""Independent 42-bank fit census and actual role-wire alias graph."""
from __future__ import annotations
import argparse,json
from pathlib import Path


def run(inputs):
    import numpy as np
    from .registry import ARMS,READERS,score_keys
    from .replay import torch_load
    cfg=inputs.cfg;items=cfg['banks'];required={(a,s,e) for a in ARMS for s in (0,1) for e in (0,5,10,20)}
    required|={(a,s,None) for a in ('RAW_FULL','RAW_HYBRID','RAW_PROJ','U0_FULL','U0_PROJ') for s in (0,1)}
    if len(items)!=42 or {(i['service'],i['seed'],i['epoch']) for i in items}!=required:raise ValueError('exact 42 banks required')
    freezes={};by_hash={};tables={};score_tables={};records=[];main={'LR':0,'MLP':0,'HGB':0};controls=0
    for item in items:
        identity=(item['service'],item['seed'],item['epoch']);f=inputs.json(item['freeze']);freezes[identity]=f;by_hash[item['freeze']['sha256']]=identity
        if f['status']!='FROZEN_COMPLETE' or f['source_seed']!=item['seed'] or set(f['readers'])!=set(READERS):raise ValueError('complete 18-reader bank identity')
        if f['numerical_fit_count']!=len(f['accounting']):raise AssertionError('actual fitted reader accounting differs')
        for record in f['accounting']:
            if record['status']!='NEW':raise ValueError('fit census expects explicit new-fit records only')
            name=record['reader'];filename=f'MODEL_{record["view"]}__{name}.joblib'
            if filename not in f['model_files']:raise ValueError('new numerical fit lacks immutable model bytes')
            inputs.path(item['models'][filename])
            if item['models'][filename]['sha256']!=f['model_files'][filename]:raise ValueError('fitted model differs from freeze')
            if name in READERS:main['MLP' if 'MLP' in name else 'HGB' if 'HGB' in name else 'LR']+=1
            elif name in ('SHUFFLED','PLANTED'):
                if item['service'] not in ('RAW_FULL','U0_FULL'):raise ValueError('real controls on unregistered service')
                controls+=1
            else:raise ValueError('unregistered reader fit')
        for role,key,fields in (('FIT2','fit_release',{'row_id','unit','r1','r2'}),
                              ('ATT','att_release',{'row_id','unit','r1','r2','sex','train','validation'}),
                              ('INNER','inner_release',{'row_id','unit','r1','r2'})):
            a=inputs.arrays(item[key],fields);tables[identity,role]=a
            counts={'FIT2':(5211,5209),'ATT':(4064,4061),'INNER':(2235,2234)}
            if (len(a['unit']),len(np.unique(a['unit'])))!=counts[role]:raise ValueError('actual wire role census')
        for role,key in (('FIT2','fit_scores'),('ATT','att_scores'),('INNER','inner_scores')):
            scores=inputs.arrays(item[key]);score_tables[identity,role]=scores
            expected=set(score_keys('r1')+score_keys('r2')+score_keys('pair'))
            if not expected<=set(scores):raise ValueError('complete 18/18/54 score bank required')
            if any(np.asarray(scores[key]).shape!=(len(tables[identity,role]['unit']),) for key in expected):raise ValueError('score/release row counts differ')
            for reader in READERS:
                for view,ignored in (('r1','r2'),('r2','r1')):
                    left=scores[f'{view}__{reader}'];right=scores[f'pair__IGNORE_{ignored}__{reader}']
                    if left.dtype!=right.dtype or left.shape!=right.shape or left.tobytes()!=right.tobytes():raise AssertionError('ignore-other score alias bytes differ')
        records.append({'service':item['service'],'seed':item['seed'],'epoch':item['epoch'],
                        'new_reader_fits':f['numerical_fit_count'],'freeze_sha256':item['freeze']['sha256']})
    aliases=[]
    for identity,f in freezes.items():
        for view,names in f['reused_readers'].items():
            for reader,reuse in names.items():
                source=by_hash.get(reuse['source_freeze_sha256'])
                if source is None:raise ValueError('alias source is outside complete new-trial graph')
                sf=freezes[source];filename=f'MODEL_{view}__{reader}.joblib'
                if sf['model_files'].get(filename)!=reuse['model_sha256'] or sf['validation'][view][reader]['orientation']!=reuse['orientation']:
                    raise ValueError('aliased reader/model/orientation provenance differs')
                if f['model_seed']!=sf['model_seed'] or f['bindings']['preprocessing']!=sf['bindings']['preprocessing']:
                    raise ValueError('alias preprocessing/model seed differs')
                for role in ('FIT2','ATT','INNER'):
                    left=tables[identity,role];right=tables[source,role]
                    fields=('row_id','unit','r1','r2') if view=='pair' else ('row_id','unit',view)
                    fields+=('sex','train','validation') if role=='ATT' else ()
                    for field in fields:
                        a=left[field];b=right[field]
                        if a.dtype!=b.dtype or a.shape!=b.shape or a.tobytes()!=b.tobytes():raise AssertionError('literal aliased role bytes differ')
                    left=score_tables[identity,role][f'{view}__{reader}'];right=score_tables[source,role][f'{view}__{reader}']
                    if left.dtype!=right.dtype or left.shape!=right.shape or left.tobytes()!=right.tobytes():raise AssertionError('reused reader score curve bytes differ')
                aliases.append({'service':identity[0],'seed':identity[1],'epoch':identity[2],'view':view,'reader':reader,
                                'source_service':source[0],'source_epoch':source[2],'actual_role_wire_equality':'PASS'})
    tensor_records=[];sources={}
    for item in cfg['frozen_sources']:
        source=torch_load(inputs.path(item['checkpoint']));binding=inputs.json(item['geometry'])
        if binding['source_seed']!=item['seed'] or binding['checkpoint_sha256']!=item['checkpoint']['sha256']:raise ValueError('admitted source geometry identity')
        sources[item['seed']]=(source,binding)
    checkpoints=cfg['continuation_checkpoints']
    if len(checkpoints)!=32 or {(c['service'],c['seed'],c['epoch']) for c in checkpoints}!={(a,s,e) for a in ARMS for s in (0,1) for e in (0,5,10,20)}:raise ValueError('all32 continuation checkpoints required')
    for item in checkpoints:
        payload=torch_load(inputs.path(item['checkpoint']));source,binding=sources[item['seed']]
        names=[k for k in source['model'] if k.startswith(('enc.0.','head.0.'))]
        if len(names)!=8:raise ValueError('income parameter namespace')
        for name in names:
            left=payload['model'][name].detach().contiguous().numpy();right=source['model'][name].detach().contiguous().numpy()
            if left.dtype!=right.dtype or left.shape!=right.shape or left.tobytes()!=right.tobytes():raise AssertionError('frozen income parameter dtype/shape/bytes differ')
        saved=payload['wire_binding']
        if saved!=binding:raise AssertionError('checkpoint fixed wire geometry differs')
        for i in (0,1):
            left=np.asarray(saved['heads'][i]['B'],dtype=np.float64);right=np.asarray(binding['heads'][i]['B'],dtype=np.float64)
            if left.shape!=right.shape or left.tobytes()!=right.tobytes():raise AssertionError('frozen basis signed-zero bytes differ')
        tensor_records.append({'service':item['service'],'seed':item['seed'],'epoch':item['epoch'],'income_parameter_count':len(names),'income_tensor_bytes':'EXACT','fixed_basis_bytes':'EXACT'})
    if main!={'LR':444,'MLP':444,'HGB':444} or controls!=24:raise AssertionError('independent actual 1332+24 fit census differs')
    if len(aliases)!=42*54-1332:raise AssertionError('complete direct-view alias membership differs')
    return {'schema':'CIS1-independent-unit-fit-census-v1','status':'PASS','bank_count':42,'main_architecture_counts':main,
            'main_reader_fits':sum(main.values()),'logistic_real_control_fits':controls,'main_plus_control_fits':sum(main.values())+controls,
            'continuation_frozen_income_checks':tensor_records,'continuation_checkpoint_count':32,'score_banks_complete_18_18_54':42,'ignore_other_curve_alias_checks':42*3*36,
            'direct_view_alias_reader_memberships':len(aliases),'records':records,'alias_graph':aliases,
            'all_alias_roles_checked':['FIT2','ATT','INNER_label_free'],'scientific_runner_imports':False,
            'INNER_truth_opened_or_hashed':False,'aliases_are_not_extra_independent_evidence':True}


def main():
    from .replay import Inputs
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);args=parser.parse_args();inputs=Inputs(args.spec)
    result=run(inputs);out=Path(inputs.cfg['public_output']);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':'PASS','main_plus_controls':result['main_plus_control_fits']}))

if __name__=='__main__':main()
