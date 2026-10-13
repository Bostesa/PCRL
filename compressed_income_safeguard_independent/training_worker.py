"""Independent complete proposal accounting from immutable training logs."""
from __future__ import annotations
import argparse,json
from pathlib import Path


def run(inputs):
    import numpy as np
    from .replay import torch_load,close
    from .safeguard import audit_steps
    from .registry import ARMS
    cfg=inputs.cfg;packages=cfg['packages'];records=[];coefficients={};draw_hashes={}
    if len(cfg['normalizations'])!=2 or {n['seed'] for n in cfg['normalizations']}!={0,1}:raise ValueError('both source normalization receipts required')
    for item in cfg['normalizations']:
        receipt=inputs.json(item['receipt']);draws=torch_load(inputs.path(item['tensors']))
        if len(draws)!=16 or len(receipt['draws'])!=16:raise ValueError('all16 frozen normalization draws required')
        norms={key:[] for key in ('task_ce','sensitive_full_occupation_plus_pair')}
        for number,draw in enumerate(draws):
            for key in norms:
                norm=float(np.linalg.norm(draw[key].double().numpy()));norms[key].append(norm)
                close(norm,receipt['draws'][number]['summary']['norms'][key],'independent normalization norm',1e-10)
        task=float(np.median(norms['task_ce']));sensitive=float(np.median(norms['sensitive_full_occupation_plus_pair']))
        if min(task,sensitive)<=1e-12 or not np.isfinite([task,sensitive]).all():raise ValueError('invalid frozen normalization')
        coefficient=3*task/sensitive
        if not np.isfinite(coefficient) or coefficient>1e4:raise ValueError('invalid fixed coefficient')
        close(coefficient,receipt['coefficient'],'independent fixed coefficient',1e-10);coefficients[item['seed']]=coefficient
    if len(packages)!=8 or {(p['arm_id'],p['seed']) for p in packages}!={(a,s) for a in ARMS for s in (0,1)}:
        raise ValueError('eight mandatory continuation executions required')
    for package in packages:
        log=[json.loads(line) for line in inputs.path(package['update_log']).read_text().splitlines()]
        summary=inputs.json(package['summary']);actual=audit_steps(log,800)
        if len(log)!=800 or {r['epoch'] for r in log}!=set(range(1,21)) or [r['update_number'] for r in log]!=list(range(1,801)):
            raise ValueError('all800 proposals in all20 epochs required')
        for epoch in range(1,21):
            chunk=[r for r in log if r['epoch']==epoch]
            if len(chunk)!=40 or {r['step_in_epoch'] for r in chunk}!=set(range(40)):raise ValueError('40 distinct proposals per epoch')
        for number,step in enumerate(log,1):
            close(step['coefficient'],coefficients[package['seed']],'shared per-seed coefficient',1e-10)
            key=(package['seed'],number)
            prior=draw_hashes.setdefault(key,step['update_indices_sha256'])
            if prior!=step['update_indices_sha256']:raise ValueError('matched model proposal draw streams differ')
            if step['critic_optimizer_steps']!=480+30*number:raise ValueError('all six matched critics must train before each proposal')
            if step.get('frozen_income_bitwise_unchanged') is not True:raise ValueError('frozen income receipt missing')
            if (step['arm']!=package['arm_id'] or step['guarded']!=(package['arm_id'] in ('CONSTRAINED','LOCAL_TWIN'))):raise ValueError('matched arm safeguard assignment differs')
            if (step['all_gradient_summary']['privacy_gradient_disabled']!=(package['arm_id']=='TASK_ONLY')):raise ValueError('task-only applied sensitive gradient convention')
            if package['arm_id']=='TASK_ONLY' and step['all_gradient_summary']['norms']['sensitive_model_term']!=0:
                raise ValueError('TASK_ONLY applied a nonzero sensitive model gradient')
        if summary['critic_optimizer_steps']!=24480 or summary['model_updates']!=800:raise ValueError('per-arm planned execution counts differ')
        expected={e['epoch']:e for e in summary['epoch_accounting']}
        for epoch,values in actual['epochs'].items():
            saved=expected[int(epoch)]
            for key in ('proposals','accepted_fraction','zero_stall_fraction'):
                if values[key]!=saved[key]:raise AssertionError('independent epoch accounting differs '+key)
        records.append({'arm_id':package['arm_id'],'seed':package['seed'],**actual,'status':'PASS'})
    return {'schema':'CIS1-independent-training-accounting-v1','status':'PASS','records':records,
            'proposal_count':sum(r['proposal_count'] for r in records),'critic_optimizer_updates':8*24480,
            'normalization_draw_count':32,'critic_batch_draw_count':32640,
            'normalization_coefficients':coefficients,
            'arithmetic_scope':'complete stored-log certification; representative mixed-dtype live engineering is separate',
            'scientific_runner_imports':False,'INNER_truth_opened_or_hashed':False}


def main():
    from .replay import Inputs
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);args=p.parse_args();inputs=Inputs(args.spec)
    result=run(inputs);output=Path(inputs.cfg['public_output']);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':'PASS','proposal_count':result['proposal_count']}))

if __name__=='__main__':main()
