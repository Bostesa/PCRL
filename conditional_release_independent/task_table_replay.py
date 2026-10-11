"""Independent absolute task means from emitted laws and allowed task labels."""
from __future__ import annotations
import argparse,csv,json,os
from pathlib import Path
import numpy as np
from .reconstruction import admitted_arrays,digest
from .policy_replay import cost_tables


def run(config):
    from conditional_release.scheduler import assert_active_from_env
    active=assert_active_from_env(stage='verification')
    state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text())
    sources={e['path']:e['sha256'] for e in state['units'][active['unit_id']]['spec']['sources']}
    def check(e):
        if sources.get(e['path'])!=e['sha256'] or digest(e['path'])!=e['sha256']:raise PermissionError('unqueued task-table replay input')
    check({'path':str(Path(config).absolute()),'sha256':digest(config)})
    cfg=json.loads(Path(config).read_text());check(cfg['manifest']);check(cfg['table'])
    source=json.loads(Path(cfg['manifest']['path']).read_text())
    with Path(cfg['table']['path']).open() as f:rows=list(csv.DictReader(f))
    if len(rows)!=408:raise ValueError('all 408 registered utility slots required')
    reported={(r['arm_id'],int(r['seed']),r['recipient'],r['reference'],r['metric']):r for r in rows}
    if len(reported)!=408:raise ValueError('duplicate task table identity')
    labels={};references={};banks={};people={}
    for seed in (0,1):
        label_values=[]
        for label in ('income','occupation'):
            e=source['roles']['INNER_SELECTION'][label];check(e)
            v=admitted_arrays(e,{'row_id','unit',label})
            if seed in people and not all(np.array_equal(v[k],people[seed][k]) for k in ('row_id','unit')):raise ValueError('task label pairing')
            people[seed]=v;label_values.append(v[label])
        labels[seed]=label_values
        e=source['roles']['INNER_SELECTION'][f'references_s{seed}'];check(e)
        references[seed]=admitted_arrays(e,{'row_id','unit','U0_1','U0_2','Ucal_1','Ucal_2','d_1','d_2'})
        if not all(np.array_equal(references[seed][k],people[seed][k]) for k in ('row_id','unit')):raise ValueError('reference people pairing')
        e=source['seeds'][str(seed)]['public_bank'];check(e);banks[seed]=json.loads(Path(e['path']).read_text())
    maximum=0.;computed=0;missing=0;covered=set()
    for arm in cfg['arms']:
        seed=arm['seed'];arm_id=arm['arm_id'];values={};refvalues={}
        if arm.get('law'):
            check(arm['law']);law=admitted_arrays(arm['law'],{'row_id','unit','code_r1','code_r2','mass'})
            if not all(np.array_equal(law[k],people[seed][k]) for k in ('row_id','unit')):raise ValueError('law/label pairing')
            for recipient in (1,2):
                y=labels[seed][recipient-1];p=np.asarray(banks[seed]['recipients'][str(recipient)]['probability'],float)
                for metric,table in cost_tables(p).items():
                    values[(recipient,metric)]=float(np.mean(np.sum(law['mass']*table[y[:,None],law['code_r'+str(recipient)]],axis=1)))
                    for ref in ('U0','Ucal'):
                        q=references[seed][f'{ref}_{recipient}']
                        if metric=='accuracy':v=-(q.argmax(1)==y).astype(float)
                        elif metric=='nll':v=-np.log(np.maximum(q[np.arange(len(y)),y],1e-12))
                        else:v=(q*q).sum(1)-2*q[np.arange(len(y)),y]+1
                        refvalues[(recipient,metric,ref)]=float(v.mean())
        passes=[]
        for recipient in (1,2):
            for ref in ('U0','Ucal'):
                for metric in ('accuracy','nll','brier'):
                    key=(arm_id,seed,'r'+str(recipient),ref,metric);covered.add(key);row=reported[key]
                    if not values:
                        if row['status']!='UNTESTED_DEPENDENCY' or row.get('absolute_metric'):raise ValueError('missing task arm estimated')
                        missing+=1;continue
                    own=values[(recipient,metric)];base=refvalues[(recipient,metric,ref)];excess=own-base
                    cap=.005 if metric=='brier' else .01;passes.append(excess<=cap)
                    expected={'absolute_metric':-own if metric=='accuracy' else own,'reference_absolute_metric':-base if metric=='accuracy' else base,'mean_loss_excess':excess}
                    if metric=='accuracy':expected['accuracy_change']=-excess
                    for name,value in expected.items():
                        gap=abs(float(row[name])-value);maximum=max(maximum,gap)
                        if gap>1e-12:raise ValueError('absolute emitted-law task mean differs')
                    if (row['original_point_cap_pass']=='True')!=(excess<=cap):raise ValueError('task cap boundary differs')
                    computed+=1
        if values:
            for key in covered:
                if key[:2]==(arm_id,seed) and (reported[key]['all_twelve_original_point_caps_pass']=='True')!=all(passes):raise ValueError('dual-reference conjunction differs')
    if covered!=set(reported):raise ValueError('task table slots omitted')
    report={'schema':'CBR1-independent-absolute-task-table-v1','status':'PASS','unit_id':active['unit_id'],'slots_checked':408,'computed_slots':computed,'missing_slots':missing,'maximum_arithmetic_gap':maximum,'INNER_SEX_access':False,'refits':0,'scope':'Absolute single-query expected task accuracy/NLL/summed Brier reconstructed from emitted code laws, public probability banks and allowed INNER task labels; both frozen references and all twelve-cap point conjunctions replayed. Missing dependency slots remain missing; no uncertainty or population claim.'}
    with Path(cfg['output']).open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'status':'PASS','slots':408,'computed':computed,'maximum_gap':maximum}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);run(p.parse_args().config)
