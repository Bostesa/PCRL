"""Nomination-gated original INNER point utility for every frozen family."""
from __future__ import annotations
import argparse,json
from pathlib import Path


def admitted_checkpoint(lock,arm_id,seed,checkpoint_sha256):
    """One exact canonical nomination association; unrelated hash strings fail."""
    instances=lock.get('instances')
    if not isinstance(instances,list):return False
    matching=[item for item in instances if isinstance(item,dict) and item.get('arm_id')==arm_id and type(item.get('seed')) is int and item['seed']==seed]
    return len(matching)==1 and matching[0].get('checkpoint_sha256')==checkpoint_sha256


def utility_clause_pass(value,cap):
    """The frozen point requirement is literal, without tolerance slack."""
    return bool(value<=cap)


def run(spec_path):
    from .attack_worker import queue_config
    active,cfg,verify=queue_config(spec_path,'attacks')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from .source import load_manifest,load_role
    from .models import load_checkpoint
    from .inference import task_metrics,point_utility
    from .attacks import save_npz,write_json
    seal=verify(cfg['nomination_lock'],cfg['nomination_lock_sha256']);lock=json.loads(seal.read_text())
    if lock.get('status')!='FROZEN' or not lock.get('nominees'):raise PermissionError('all-family nomination seal absent')
    verify(cfg['manifest'],cfg['manifest_sha256']);manifest=load_manifest(cfg['manifest'],cfg['manifest_sha256'])
    root=Path(cfg['private_output'])
    if '.private' not in root.resolve().parts or root.exists():raise PermissionError('fresh private utility directory required')
    root.mkdir(parents=True,mode=0o700);records=[];arrays={};roles={}
    # Verify all frozen requested checkpoint bytes before opening any task label.
    for item in cfg['instances']:
        if not admitted_checkpoint(lock,item['arm_id'],item['seed'],item['checkpoint_sha256']):raise PermissionError('exact requested arm/seed/checkpoint absent or duplicated in canonical nomination instances')
        verify(item['checkpoint'],item['checkpoint_sha256'])
    for seed in (0,1):
        data=load_role(manifest,seed,'INNER_SELECTION',{'features','income','occupation','references'},nomination_seal=lock)
        roles[seed]=data
        if seed==0:
            arrays.update(row_id=data['row_id'],unit=data['unit'],income=data['income'],occupation=data['occupation'])
        elif not all(np.array_equal(data[k],arrays[k]) for k in ('row_id','unit','income','occupation')):raise PermissionError('utility source seeds are not same people/labels')
        for i in (1,2):
            for ref in ('U0','Ucal'):arrays[f's{seed}__{ref}__p{i}']=data['references'][f'{ref}_{i}']
    for item in cfg['instances']:
        seed=item['seed'];data=roles[seed];model,payload=load_checkpoint(item['checkpoint'],item['checkpoint_sha256'])
        if int(payload['seed'])!=seed:raise PermissionError('utility checkpoint source seed differs')
        wire=model.route_raw(data['X'],data['feature_names']);tasks={};clauses=[]
        for i,key in enumerate(('income','occupation')):
            probability=wire[i]['probability'];metrics=task_metrics(data[key],probability,model.class_orders[i]);references={}
            arrays[f"s{seed}__{item['arm_id']}__p{i+1}"]=probability
            for reference in ('U0','Ucal'):
                baseline=task_metrics(data[key],data['references'][f'{reference}_{i+1}'],model.class_orders[i]);delta=point_utility(metrics,baseline)
                references[reference]={'metrics':baseline,'difference':delta}
                for metric,cap in (('accuracy_loss',.010),('nll_excess',.010),('brier_excess',.005)):
                    clauses.append({'task':key,'reference':reference,'metric':metric,'value':delta[metric],'cap':cap,'pass':utility_clause_pass(delta[metric],cap)})
            tasks[key]={'metrics':metrics,'references':references}
        records.append({'arm_id':item['arm_id'],'seed':seed,'checkpoint_sha256':item['checkpoint_sha256'],'tasks':tasks,'clauses':clauses,'clauses_passed':sum(c['pass'] for c in clauses),'point_eligible_seed':all(c['pass'] for c in clauses)})
    family={}
    for arm in sorted({i['arm_id'] for i in records}):
        matched=[i for i in records if i['arm_id']==arm]
        if sorted(i['seed'] for i in matched)!=[0,1]:raise PermissionError('both paired source seeds required')
        family[arm]={'point_eligible':all(i['point_eligible_seed'] for i in matched),'passed':sum(i['clauses_passed'] for i in matched),'total':24}
    input_sha=save_npz(root/'UTILITY_INPUTS.npz',**arrays)
    out={'schema':'TCR1-original-INNER-point-utility-v1','nomination_lock_sha256':cfg['nomination_lock_sha256'],'source_manifest_sha256':cfg['manifest_sha256'],'unit_id':active['unit_id'],'INNER_SEX_opened':False,'records':records,'family_eligibility':family,'utility_input_sha256':input_sha}
    write_json(root/'UTILITY_COMPLETE.json',out);write_json(cfg['public_output'],out,False)
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();out=run(a.spec);print(json.dumps({'families':out['family_eligibility']}))
if __name__=='__main__':main()
