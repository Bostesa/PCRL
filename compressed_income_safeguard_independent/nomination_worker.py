"""Independently recompute FIT2 proxies and ATT envelopes before nomination."""
from __future__ import annotations
import argparse,json
from pathlib import Path


def run(inputs):
    from .replay import metrics,envelopes,close
    from .registry import ARMS,common_epoch
    cfg=inputs.cfg;starts={};records=[]
    for item in cfg['starts']:
        a=inputs.arrays(item['fit_inputs']);seed=item['seed'];keys=item['probability_keys']
        starts[seed]={task:metrics(a[task],a[keys[task]]) for task in ('income','occupation')}
    for item in cfg['records']:
        a=inputs.arrays(item['fit_inputs']);att=inputs.arrays(item['att_release']);scores=inputs.arrays(item['att_scores'])
        fitted={task:metrics(a[task],a[item['probability_keys'][task]]) for task in ('income','occupation')}
        bank=envelopes(scores,att['sex'],att['validation'])
        values={view+'__'+name:auc for view,r in bank.items() for name,auc in r['reader_auc'].items()}
        records.append({'arm_id':item['arm_id'],'seed':item['seed'],'epoch':item['epoch'],
                        'FIT2':fitted,'ATT_validation_reader_auc':values})
    choices=[common_epoch(records,starts,arm) for arm in ARMS]
    if cfg.get('runner_nomination'):
        saved=inputs.json(cfg['runner_nomination'])
        expected=saved.get('nominees',saved.get('arms'))
        for choice in choices:
            prior=expected[choice['arm_id']] if isinstance(expected,dict) else next(r for r in expected if r['arm_id']==choice['arm_id'])
            if choice['epoch']!=prior['epoch'] or choice['proxy_eligible']!=prior['proxy_eligible']:raise AssertionError('independent common-epoch nomination differs')
    return {'schema':'CIS1-independent-nomination-v1','status':'PASS' if all(c['epoch'] is not None for c in choices) else 'MISSING',
            'nominees':choices,'INNER_truth_opened_or_hashed':False,'scientific_runner_imports':False}


def main():
    from .replay import Inputs
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);args=p.parse_args();inputs=Inputs(args.spec)
    result=run(inputs);output=Path(inputs.cfg['public_output']);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':result['status'],'arms':len(result['nominees'])}))

if __name__=='__main__':main()
