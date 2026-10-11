"""Independent aggregate-only closeout arithmetic and frozen descriptive ranking."""
from __future__ import annotations
import argparse,csv,json,os,math
from collections import Counter
from pathlib import Path
from .reconstruction import digest


def run(config):
    from conditional_release.scheduler import assert_active_from_env
    active=assert_active_from_env(stage='verification')
    state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text())
    registered={i['path']:i['sha256'] for i in state['units'][active['unit_id']]['spec']['sources']}
    if registered.get(str(Path(config).absolute()))!=digest(config):raise PermissionError('unqueued config')
    cfg=json.loads(Path(config).read_text())
    def pin(x):
        if registered.get(x['path'])!=x['sha256'] or digest(x['path'])!=x['sha256']:raise PermissionError('unqueued claims input')
        return Path(x['path'])
    summary=json.loads(pin(cfg['summary']).read_text());lock=json.loads(pin(cfg['attacks_lock']).read_text());selection=json.loads(pin(cfg['selection']).read_text())
    with pin(cfg['endpoints']).open() as f: endpoints=list(csv.DictReader(f))
    with pin(cfg['task_table']).open() as f: task=list(csv.DictReader(f))
    with pin(cfg['recovery_table']).open() as f: recovery=list(csv.DictReader(f))
    gaps=[]
    def same(a,b):
        gap=abs(float(a)-float(b));gaps.append(gap)
        if gap>1e-12:raise ValueError('aggregate numeric mismatch')
    reports={}
    for item in cfg['reports']:
        j=json.loads(pin(item).read_text())
        if j['status']!='PASS':raise ValueError('independent report not PASS')
        row=j['results'][0];reports[(row['arm_id'],row['seed'])]=row
    if len(reports)!=20:raise ValueError('incomplete independently replayed reports')
    for row in recovery:
        if row['seed']=='mean': expected=math.fsum(reports[(row['arm_id'],s)]['views'][row['view']]['envelope_auc'] for s in (0,1))/2
        else:expected=reports[(row['arm_id'],int(row['seed']))]['views'][row['view']]['envelope_auc']
        same(expected,row['auc'])
        if row['all_controls_pass']!='True':raise ValueError('recovery control gate')
    if summary['registered_K']!=552 or len(endpoints)!=552 or summary['status_counts']!=dict(Counter(x['status'] for x in endpoints)):raise ValueError('registered status accounting')
    if summary['nominee']!='W_STRICT' or summary['nominee_reselected'] is not False:raise ValueError('nominee changed')
    # Selection's exact key is checked by its earlier independent nomination replay.
    if 'W_STRICT' not in json.dumps(selection):raise ValueError('nominee lock absent')
    aucs={};eligibility={}
    for arm,desc in summary['arms'].items():
        slots=[x for x in endpoints if x['kind']=='utility' and x['arm_id']==arm]
        if len(slots)!=24 or dict(Counter(x['status'] for x in slots))!=desc['utility_status_counts']:raise ValueError('utility status accounting')
        for ref,key in [('U0','original_reference_bound_counts'),('Ucal','calibration_matched_bound_counts')]:
            if dict(Counter(x['status'] for x in slots if x['array_key'].startswith(ref+'_')))!=desc[key]:raise ValueError('reference accounting')
        rows=[x for x in task if x['arm_id']==arm and x['status']=='COMPUTED_FROM_PUBLIC_RECEIPTS']
        eligible=bool(rows) and all(any(x['seed']==str(s) for x in rows) for s in (0,1)) and all(x['all_twelve_original_point_caps_pass']=='True' for x in rows)
        eligibility[arm]=eligible
        if eligible!=desc['point_eligible_both_seeds']:raise ValueError('dual-reference point eligibility')
        expectedmean={};expectedabsolute={};worst={}
        for taskname in ('income','occupation'):
            for metric in ('accuracy','nll','brier'):
                both=[x for x in rows if x['task']==taskname and x['metric']==metric]
                if both:worst[taskname+'_'+metric]=max(float(x['mean_loss_excess']) for x in both)
                for ref in ('U0','Ucal'):
                    subset=[x for x in both if x['reference']==ref]
                    if subset:
                        if len(subset)!=2:raise ValueError('unpaired means')
                        key=f'{taskname}_{metric}_{ref}'
                        expectedmean[key]=math.fsum(float(x['mean_loss_excess']) for x in subset)/2
                        expectedabsolute[key]=math.fsum(float(x['absolute_metric']) for x in subset)/2
        for field,expected in [('mean_excess_by_reference',expectedmean),('absolute_task_means',expectedabsolute),('worst_seed_reference_excess',worst)]:
            if set(desc[field])!=set(expected):raise ValueError('aggregate key mismatch')
            for key,value in expected.items():same(value,desc[field][key])
        if eligible:
            means={view:math.fsum(reports[(arm,s)]['views'][view]['envelope_auc'] for s in (0,1))/2 for view in ('r1','r2','pair')}
            for view,value in means.items():same(value,desc['mean_declared_slate_auc'][view])
            aucs[arm]=means
        elif desc['mean_declared_slate_auc']:raise ValueError('ineligible fresh recovery entered ranking')
    ranking=lock['descriptive_strongest_control'];pool=[x for x in ranking['pool'] if eligibility[x]];minimum=min(aucs[x]['pair'] for x in pool)
    strongest=next(x for x in pool if aucs[x]['pair']<=minimum+ranking['tie_tolerance'])
    if strongest!=summary['strongest_descriptive_control']:raise ValueError('frozen tie ranking')
    same(aucs['P0']['pair']-aucs['W_STRICT']['pair'],summary['nominee_pair_benefit_against_P0'])
    same(aucs[strongest]['pair']-aucs['W_STRICT']['pair'],summary['nominee_pair_benefit_against_strongest_control'])
    same(aucs['P0']['pair']-aucs[strongest]['pair'],summary['strongest_control_pair_benefit_against_P0_descriptive_only'])
    if not summary['all_report_control_gates_pass'] or not all(x['engineering_controls_pass'] for x in reports.values()):raise ValueError('probe gates')
    output={'schema':'CBR1-independent-closeout-claims-v1','status':'PASS','unit_id':active['unit_id'],'registered_slots':552,'independent_report_instances':20,'strongest_descriptive_control':strongest,'nominee':'W_STRICT','nominee_reselected':False,'maximum_arithmetic_gap':max(gaps,default=0),'nominee_pair_benefit_against_P0':aucs['P0']['pair']-aucs['W_STRICT']['pair'],'nominee_pair_benefit_against_strongest_control':aucs[strongest]['pair']-aucs['W_STRICT']['pair'],'W_utility_status_counts':summary['arms']['W_STRICT']['utility_status_counts'],'status_counts':summary['status_counts'],'scope':'All public arm aggregates, dual-reference point eligibility, fixed pool/tolerance/order strongest-control selection, complete-view recovery table and unchanged nominee. Recovery inputs are own independently replayed law results; interval verification is separate. No new primary contrast, nominee reselection or confirmation.'}
    with Path(cfg['output']).open('x') as f:json.dump(output,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'status':'PASS','strongest_control':strongest,'maximum_gap':output['maximum_arithmetic_gap']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);run(p.parse_args().config)
