"""Public aggregate closeout arithmetic; no person arrays, fits or selection."""
from __future__ import annotations
import argparse, csv, hashlib, json
from collections import Counter
from pathlib import Path

def run(root: Path):
    task_path=root/'r5/ABSOLUTE_TASK_METRICS.csv'
    tasks=list(csv.DictReader(task_path.open()))
    endpoint_path=root/'PRIMARY_ENDPOINTS.csv'
    endpoints=list(csv.DictReader(endpoint_path.open()))
    if len(tasks)!=408 or len(endpoints)!=552 or len({x['endpoint_id'] for x in endpoints})!=552:
        raise ValueError('all prospective slots must remain')
    lock=json.loads((root/'ATTACKS_LOCK.json').read_text())
    selection=json.loads((root/'FIT_SELECTION_LOCK.json').read_text())
    if selection['nominee']!='W_STRICT' or lock['nominee']!='W_STRICT':
        raise ValueError('nominee identity changed')
    reports={}
    for path in sorted((root/'r5').glob('R5_REPORT_*_V2.json')):
        d=json.loads(path.read_text());key=(d['arm_id'],d['source_seed'])
        if key in reports or d['status']!='COMPLETE' or not d['all_controls_pass']:
            raise ValueError('report duplicate, missing control or incomplete')
        reports[key]=d
    if len(reports)!=20:raise ValueError('twenty complete fresh-reader reports required')
    arms={}
    for arm in dict.fromkeys(t['arm_id'] for t in tasks):
        own=[t for t in tasks if t['arm_id']==arm]
        point=all(t['original_point_cap_pass']=='True' for t in own)
        absolute={};excess={};worst={}
        for task in ('income','occupation'):
            for metric in ('accuracy','nll','brier'):
                for ref in ('U0','Ucal'):
                    rows=[t for t in own if t['task']==task and t['metric']==metric and t['reference']==ref]
                    if len(rows)!=2:raise ValueError('paired task slots omitted')
                    if all(t['absolute_metric'] for t in rows):
                        absolute[f'{task}_{metric}_{ref}']=sum(float(t['absolute_metric']) for t in rows)/2
                        excess[f'{task}_{metric}_{ref}']=sum(float(t['mean_loss_excess']) for t in rows)/2
                all_refs=[t for t in own if t['task']==task and t['metric']==metric and t['mean_loss_excess']]
                if all_refs:worst[f'{task}_{metric}']=max(float(t['mean_loss_excess']) for t in all_refs)
        auc={}
        if all((arm,s) in reports for s in (0,1)):
            auc={view:sum(reports[(arm,s)]['views'][view]['envelope_auc'] for s in (0,1))/2 for view in ('r1','r2','pair')}
        slots=[e for e in endpoints if e['kind']=='utility' and e['arm_id']==arm]
        arms[arm]={'point_eligible_both_seeds':point,'absolute_task_means':absolute,
                   'mean_excess_by_reference':excess,'worst_seed_reference_excess':worst,
                   'mean_declared_slate_auc':auc,'utility_status_counts':dict(Counter(e['status'] for e in slots)),
                   'original_reference_bound_counts':dict(Counter(e['status'] for e in slots if '/U0/' in e['endpoint_id'])),
                   'calibration_matched_bound_counts':dict(Counter(e['status'] for e in slots if '/Ucal/' in e['endpoint_id']))}
    rule=lock['descriptive_strongest_control'];pool=rule['pool']
    eligible=[a for a in pool if arms[a]['point_eligible_both_seeds'] and arms[a]['mean_declared_slate_auc']]
    minimum=min(arms[a]['mean_declared_slate_auc']['pair'] for a in eligible)
    strongest=next(a for a in eligible if arms[a]['mean_declared_slate_auc']['pair']<=minimum+rule['tie_tolerance'])
    nominee=selection['nominee'];w=arms[nominee]['mean_declared_slate_auc']['pair']
    result={'schema':'CBR1-public-closeout-summary-v1','nominee':nominee,'strongest_descriptive_control':strongest,
            'nominee_reselected':False,'registered_K':552,'status_counts':dict(Counter(e['status'] for e in endpoints)),
            'nominee_pair_benefit_against_P0':arms['P0']['mean_declared_slate_auc']['pair']-w,
            'nominee_pair_benefit_against_strongest_control':arms[strongest]['mean_declared_slate_auc']['pair']-w,
            'strongest_control_pair_benefit_against_P0_descriptive_only':arms['P0']['mean_declared_slate_auc']['pair']-minimum,
            'all_report_control_gates_pass':True,'arms':arms,
            'input_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [task_path,endpoint_path,root/'ATTACKS_LOCK.json',root/'FIT_SELECTION_LOCK.json']},
            'scope':'Fixed FIT nominee and predeclared descriptive control ranking; reused Adult exploratory development. No new primary contrast or confirmation.'}
    (root/'RESULT_SUMMARY.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    rows=[]
    for (arm,seed),d in reports.items():
        for view in ('r1','r2','pair'):
            rows.append({'arm_id':arm,'seed':seed,'view':view,'auc':d['views'][view]['envelope_auc'],'all_controls_pass':d['all_controls_pass'],'summary':'individual paired source seed'})
    for arm,d in arms.items():
        for view,auc in d['mean_declared_slate_auc'].items():rows.append({'arm_id':arm,'seed':'mean','view':view,'auc':auc,'all_controls_pass':True,'summary':'equal mean of two paired source seeds'})
    with (root/'COMPLETE_VIEW_RECOVERY.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:result[k] for k in ['nominee','strongest_descriptive_control','nominee_pair_benefit_against_P0','nominee_pair_benefit_against_strongest_control','status_counts']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
