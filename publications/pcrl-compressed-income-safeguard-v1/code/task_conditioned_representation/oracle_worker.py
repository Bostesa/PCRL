"""Frozen FIT-trained true-task-label-only interpretation reference, never primary."""
from __future__ import annotations
import argparse,json
from pathlib import Path


def run(spec_path):
    from .attack_worker import queue_config
    active,cfg,verify=queue_config(spec_path,'attacks')
    import numpy as np
    from .source import load_manifest,load_role
    from .attacks import save_npz,write_json
    seal=json.loads(verify(cfg['nomination_lock'],cfg['nomination_lock_sha256']).read_text())
    if seal.get('status')!='FROZEN' or not seal.get('nominees'):raise PermissionError('nomination freeze required before INNER task labels')
    verify(cfg['manifest'],cfg['manifest_sha256']);m=load_manifest(cfg['manifest'],cfg['manifest_sha256'])
    fit=load_role(m,0,'OSF_DEFENSE_FIT',{'features','income','occupation','sex'})
    inner=load_role(m,0,'INNER_SELECTION',{'features','income','occupation'},nomination_seal=seal)
    table={};scores={};counts={}
    for view,k,fit_label,inner_label in (('r1',2,fit['income'],inner['income']),('r2',6,fit['occupation'],inner['occupation']),('pair',12,fit['income']*6+fit['occupation'],inner['income']*6+inner['occupation'])):
        count=np.zeros((k,2),dtype=np.int64);np.add.at(count,(fit_label,fit['sex']),1)
        probability=(count[:,1]+1)/(count.sum(1)+2);table[view]=probability;counts[view]=count;scores[view]=probability[inner_label]
    root=Path(cfg['private_output'])
    if '.private' not in root.resolve().parts or root.exists():raise PermissionError('fresh private oracle reference required')
    root.mkdir(parents=True,mode=0o700);table_sha=save_npz(root/'FIT_LABEL_ORACLE.npz',**table,**{k+'__counts':v for k,v in counts.items()})
    score_sha=save_npz(root/'INNER_LABEL_ORACLE.npz',row_id=inner['row_id'],unit=inner['unit'],income=inner['income'],occupation=inner['occupation'],**scores)
    out={'schema':'TCR1-FIT-trained-label-only-oracle-freeze-v1','status':'FROZEN_COMPLETE','unit_id':active['unit_id'],'table_sha256':table_sha,'score_sha256':score_sha,'nomination_lock_sha256':cfg['nomination_lock_sha256'],'training_role':'FIT all15434people including inherited internal validation; not ATT','smoothing':'Laplace1 per binary SEX class per true task class/labelpair','inference_inputs':'oracle true income/occupation labels, forbidden for primary deployment/readers','INNER_SEX_opened':False,'interpretation_only':True}
    write_json(root/'FROZEN_COMPLETE.json',out);write_json(cfg['public_output'],out,False);return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();run(a.spec);print(json.dumps({'status':'FROZEN_COMPLETE','contract':'oracle_true_task_label_only_interpretation'}))
if __name__=='__main__':main()
