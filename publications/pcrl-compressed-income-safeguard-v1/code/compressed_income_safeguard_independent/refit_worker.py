"""Freeze independent ATT-selected targets and run their admitted refits."""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path


def freeze(inputs):
    import numpy as np
    from .replay import envelopes,close
    from .registry import refit_targets,score_keys
    records=[]
    for item in inputs.cfg['banks']:
        att=inputs.arrays(item['att_release']);scores=inputs.arrays(item['att_scores']);receipt=inputs.json(item['freeze'])
        if receipt['source_seed']!=item['seed'] or receipt['status']!='FROZEN_COMPLETE':raise ValueError('independent ATT bank identity')
        actual=envelopes(scores,att['sex'],att['validation']);values={};orientations={}
        for view in ('r1','r2','pair'):
            for key in score_keys(view):
                name=key.split('__')[-1];actual_view=view
                if '__IGNORE_r2__' in key:actual_view='r1'
                elif '__IGNORE_r1__' in key:actual_view='r2'
                val=receipt['validation'][actual_view][name];orientation=val['orientation']
                if orientation not in (-1,1):raise ValueError('frozen reader orientation convention')
                auc=actual[view]['reader_auc'][key.split('__',1)[1]]
                close(auc,val['fixed_orientation_auc'],'independent ATT fixed orientation')
                if auc<.5-1e-12:raise ValueError('ATT orientation differs')
                values[key]=auc;orientations[key]=orientation
        records.append({**item,'ATT_validation_reader_auc':values,'orientations':orientations,'model_seed':receipt['model_seed']})
    targets=refit_targets(records)
    private=Path(inputs.cfg['private_output']);private.mkdir(parents=True,exist_ok=False,mode=0o700)
    target_file=private/'TARGETS.json'
    with target_file.open('x') as stream:json.dump(targets,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    target_file.chmod(0o600)
    digest=hashlib.sha256(target_file.read_bytes()).hexdigest()
    return {'schema':'CIS1-independent-refit-target-freeze-v1','status':'FROZEN',
            'target_registry_sha256':digest,'registered_attempts':54,'architecture_counts':targets['architecture_counts'],
            'targets':[{k:v for k,v in t.items() if k not in ('att_release','inner_release','freeze')} for t in targets['targets']],
            'selected_on':'independently recomputed ATT validation only','INNER_truth_opened_or_hashed':False}


def main():
    from .replay import Inputs,refit
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);args=parser.parse_args();inputs=Inputs(args.spec)
    operation=inputs.cfg['operation']
    if operation=='freeze_targets':result=freeze(inputs)
    elif operation=='refit':result=refit(inputs)
    else:raise ValueError('registered independent reader operation required')
    output=Path(inputs.cfg['public_output']);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':result.get('status','COMPLETE'),'operation':operation}))

if __name__=='__main__':main()
