"""Post-freeze independent complete-service utility and recovery audit."""
from __future__ import annotations
import argparse,json
from pathlib import Path


def run(inputs):
    import numpy as np
    from .replay import metrics,vectors,envelopes,control_replay,compare_controls,close
    from .registry import SERVICES,CAPS,TASKS,task_deltas,strongest_control
    cfg=inputs.cfg;lock=inputs.seal();manifest=inputs.json(cfg['manifest']);truth={}
    label_hashes=lock.get('role_label_sha256',{}).get('INNER_SELECTION',{})
    if any(label_hashes.get(t)!=cfg['truth'][t]['sha256'] for t in ('income','occupation','sex')):
        raise PermissionError('independent truth pins differ from committed label custody')
    frozen_hashes=set(lock.get('reader_freeze_sha256',[]))
    if any(bank['freeze']['sha256'] not in frozen_hashes for bank in cfg['banks']):
        raise PermissionError('independent bank absent from committed reader freeze')
    for task in ('income','occupation','sex'):
        truth[task]=inputs.arrays(cfg['truth'][task],{'row_id','unit',task})
    people=truth['sex']
    if any(not np.array_equal(truth[t][k],people[k]) for t in TASKS for k in ('row_id','unit')):raise AssertionError('assessment truth people differ')
    banks=cfg['banks']
    if len(banks)!=18 or {(b['arm_id'],b['seed']) for b in banks}!={(a,s) for a in SERVICES for s in (0,1)}:
        raise ValueError('all nine final services and both seeds required')
    refs={s:inputs.arrays(manifest['roles']['INNER_SELECTION'][f'references_s{s}'],{'row_id','unit','U0_1','U0_2','Ucal_1','Ucal_2','d_1','d_2'}) for s in (0,1)}
    for r in refs.values():
        if any(not np.array_equal(r[k],people[k]) for k in ('row_id','unit')):raise ValueError('reference people differ')
    records=[];clauses=[];pair={};control_gates={k:people[k] for k in ('row_id','unit')}
    runner_gates=inputs.arrays(cfg['runner_control_gates'])
    if any(not np.array_equal(runner_gates[k],people[k]) for k in ('row_id','unit')):raise ValueError('required real-control gate people differ')
    private=Path(cfg['private_output']) if cfg.get('private_output') else None
    if private:private.mkdir(parents=True,exist_ok=False,mode=0o700)
    for bank in banks:
        release=inputs.arrays(bank['release'],{'row_id','unit','r1','r2'});scores=inputs.arrays(bank['scores']);freeze=inputs.json(bank['freeze'])
        if freeze['status']!='FROZEN_COMPLETE' or freeze['source_seed']!=bank['seed']:raise ValueError('final bank freeze identity')
        if freeze['inner_release_sha256']!=bank['release']['sha256'] or freeze['score_sha256']!=bank['scores']['sha256']:raise ValueError('final bank wire/score differs from frozen bytes')
        if any(not np.array_equal(release[k],people[k]) for k in ('row_id','unit')):raise ValueError('final wire and truth person pairing differs')
        service=bank['arm_id'];seed=bank['seed'];task_records={};metric_table={k:people[k] for k in ('row_id','unit')}
        for i,(task,classes) in enumerate(zip(TASKS,(2,6)),1):
            wire=release[f'r{i}'];p=wire[:,16+classes:16+2*classes]
            if wire.shape!=(len(people['unit']),17+2*classes) or not np.isfinite(wire).all() or not np.array_equal(wire[:,-1],p.argmax(1)):
                raise ValueError('actual own output wire convention')
            y=truth[task][task];v=vectors(y,p);actual=metrics(y,p);reference_records={}
            for metric,value in v.items():metric_table[f'own__{task}__{metric}']=value
            for reference in ('U0','Ucal'):
                reference_p=refs[seed][f'{reference}_{i}'];baseline=metrics(y,reference_p);delta=task_deltas(actual,baseline)
                reference_records[reference]={'metrics':baseline,'difference':delta}
                for metric,value in vectors(y,reference_p).items():metric_table[f'REF_{reference}__{task}__{metric}']=value
                for metric,value in delta.items():
                    clauses.append({'service':service,'seed':seed,'task':task,'reference':reference,'metric':metric,
                                    'point':value,'threshold':CAPS[metric],'point_pass':value<=CAPS[metric]})
            task_records[task]={'metrics':actual,'references':reference_records}
        recovery=envelopes(scores,people['sex']);controls=control_replay(scores,people['sex'],people['unit'],2026101207)
        pair.setdefault(service,{})[seed]=recovery['pair']['contracts']['complete']['auc']
        record={'arm_id':service,'seed':seed,'tasks':task_records,'recovery':recovery,'controls':controls}
        if service in ('U0_FULL','RAW_FULL'):
            att=inputs.arrays(bank['att_release']);att_scores=inputs.arrays(bank['att_scores'])
            if freeze['att_release_sha256']!=bank['att_release']['sha256'] or freeze['att_scores_sha256']!=bank['att_scores']['sha256']:
                raise ValueError('ATT real-control input differs from frozen bank')
            att_controls=control_replay(att_scores,att['sex'],att['unit'],2026101206,att['validation'])
            compare_controls(att_controls,freeze.get('ATT_validation_controls'),'independent ATT real controls')
            valid=(controls is not None and att_controls is not None
                   and all(v['passed'] for v in controls.values()) and all(v['passed'] for v in att_controls.values()))
            key=f'{service}__S{seed}';expected=np.full(len(people['unit']),valid,dtype=bool)
            actual=runner_gates.get(key)
            if actual is None or actual.shape!=expected.shape or actual.dtype.kind!='b' or not np.array_equal(actual,expected):
                raise AssertionError('independent required real-control gate vector differs '+key)
            control_gates[key]=expected;record['ATT_controls']=att_controls;record['control_gate_valid']=bool(valid)
        if bank.get('runner_metrics'):
            expected=inputs.json(bank['runner_metrics'])
            for task in TASKS:
                saved=expected['tasks'][task]['metrics']
                for metric in ('accuracy','nll','brier'):close(task_records[task]['metrics'][metric],saved[metric],'independent final '+service+' '+task+' '+metric)
        if bank.get('metric_vectors'):
            saved=inputs.arrays(bank['metric_vectors'])
            for key,value in metric_table.items():
                actual_key=bank.get('metric_key_map',{}).get(key,key)
                if actual_key not in saved:raise ValueError('independent metric-vector source key missing')
                close(value,saved[actual_key],'independent final row arithmetic '+actual_key)
        if private:np.savez_compressed(private/f'METRICS_{service}_S{seed}.npz',**metric_table)
        records.append(record)
    selected=strongest_control(clauses,pair)
    if private:np.savez_compressed(private/'CONTROL_GATES.npz',**control_gates)
    return {'schema':'CIS1-independent-complete-services-v1','status':'PASS','services':9,'fitted_seed_packages':18,
            'utility_clause_count':len(clauses),'records':records,'utility_clauses':clauses,'strongest_control':selected,
            'control_gate_validity':{k:bool(v[0]) for k,v in control_gates.items() if k not in ('row_id','unit')},
            'scientific_runner_imports':False,'stored_prediction_replay':True,'independent_model_forward':False,
            'INNER_truth_access':'after committed canonical REPORTING_LOCK','reporting_lock_sha256':cfg['reporting_lock']['sha256']}


def main():
    from .replay import Inputs
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);args=p.parse_args();inputs=Inputs(args.spec)
    result=run(inputs);output=Path(inputs.cfg['public_output']);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':'PASS','services':9,'utility_clause_count':result['utility_clause_count']}))

if __name__=='__main__':main()
