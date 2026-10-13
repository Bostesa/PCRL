"""Independent, metadata-only direct-trial membership and nomination rules."""
from __future__ import annotations
import math

SERVICES = ('CONSTRAINED', 'ORDINARY', 'LOCAL_TWIN', 'TASK_ONLY', 'RAW_FULL',
            'RAW_HYBRID', 'RAW_PROJ', 'U0_FULL', 'U0_PROJ')
ARMS = SERVICES[:4]
CONTROLS = ('RAW_PROJ', 'RAW_HYBRID', 'RAW_FULL', 'ORDINARY', 'LOCAL_TWIN',
            'TASK_ONLY', 'U0_PROJ', 'U0_FULL')
TASKS = ('income', 'occupation')
CAPS = {'accuracy_loss': .010, 'nll_excess': .010, 'brier_excess': .005}
BASE = ('LR_C1', 'MLP_64x32', 'HGB_15x200', 'DA_MLP_64x32', 'REP_LR_C1',
        'REP_MLP_64x32', 'REP_HGB_15x200', 'OUTPUT_LR_C1', 'OUTPUT_HGB_15x200')
EXTENDED = tuple(c+'_'+m for c in ('LOGITS_OUTPUT','DECISION','CONFIDENCE')
                 for m in ('LR_C1','MLP_64x32','HGB_15x200'))
READERS = BASE + EXTENDED


def score_keys(view):
    if view not in ('r1','r2','pair'): raise ValueError('unknown recipient view')
    keys = [view+'__'+name for name in READERS]
    if view == 'pair':
        keys += ['pair__IGNORE_'+other+'__'+name
                 for other in ('r2','r1') for name in READERS]
    return keys


def membership():
    """Return all 240 semantic endpoints without runner table conventions."""
    endpoints = []
    for service in SERVICES:
        for seed in (0,1):
            for task in TASKS:
                for reference in ('U0','Ucal'):
                    for metric, cap in CAPS.items():
                        endpoints.append({'endpoint_id': f'utility|{service}|s{seed}|{task}|{reference}|{metric}',
                                          'kind':'mean_difference','service':service,'seed':seed,
                                          'task':task,'reference':reference,'metric':metric,
                                          'direction':'at_most','threshold':cap})
    for control in CONTROLS:
        for metric, view, direction, threshold in (
                ('pair_benefit','pair','at_least',.020),
                ('income_worsening','r1','at_most',.010),
                ('occupation_worsening','r2','at_most',.010)):
            endpoints.append({'endpoint_id':f'recovery|CONSTRAINED|{control}|{metric}',
                              'kind':'auc_envelope_difference','candidate':'CONSTRAINED',
                              'control':control,'metric':metric,'view':view,
                              'direction':direction,'threshold':threshold})
    assert len(endpoints)==240 and len({e['endpoint_id'] for e in endpoints})==240
    return endpoints


def validate_registry(registry):
    """Validate semantic membership, independently of array-key spelling."""
    expected = membership()
    if registry.get('K')!=len(expected) or registry.get('replicates')!=2048:
        raise ValueError('direct trial requires K240 and 2048 shared draws')
    if (registry.get('family')!='compressed_income_safeguard' or registry.get('bootstrap_seed')!=2026101401
        or registry.get('alpha')!=.05):raise ValueError('new frozen combined interval procedure required')
    actual=registry['endpoints']
    def utility_key(e):
        return (e.get('service',e.get('arm_id')),e.get('seed'),e.get('task'),e.get('reference'),e.get('metric'))
    def recovery_key(e): return (e.get('candidate','CONSTRAINED'),e.get('control'),e.get('metric'))
    utility=[e for e in actual if e['kind']=='mean_difference']
    recovery=[e for e in actual if e['kind']=='auc_envelope_difference']
    if len(actual)!=240 or len({e['endpoint_id'] for e in actual})!=240:
        raise ValueError('unique actual K240 membership required')
    if {utility_key(e) for e in utility}!={utility_key(e) for e in expected[:216]} or len(utility)!=216:
        raise ValueError('nine complete 24-clause services required')
    if {recovery_key(e) for e in recovery}!={recovery_key(e) for e in expected[216:]} or len(recovery)!=24:
        raise ValueError('eight complete candidate-control contrasts required')
    for e in actual:
        cap=CAPS[e['metric']] if e['kind']=='mean_difference' else (.020 if e['metric']=='pair_benefit' else .010)
        direction='at_least' if e['metric']=='pair_benefit' else 'at_most'
        if (e['threshold'],e['direction'])!=(cap,direction): raise ValueError('original criterion differs')
        if len(e['paired_seeds'])!=(1 if e['kind']=='mean_difference' else 2):
            raise ValueError('seed-specific usefulness and paired-mean recovery required')
        if e['kind']=='mean_difference':
            part=e['paired_seeds'][0];base=e['metric'].split('_')[0]
            own=f'own__{e["task"]}__{base}';reference=f'REF_{e["reference"]}__{e["task"]}__{base}'
            left,right=(reference,own) if base=='accuracy' else (own,reference)
            table=f'metrics__{e.get("service",e.get("arm_id"))}__S{e["seed"]}'
            if (part.get('source_seed'),part['left_table'],part['right_table'],part['left_key'],part['right_key'],part.get('sign',1))!=(e['seed'],table,table,left,right,1):
                raise ValueError('utility direction, source seed or service vector differs')
        if e['kind']=='auc_envelope_difference':
            if {p.get('source_seed') for p in e['paired_seeds']}!={0,1}:raise ValueError('two source seeds must remain paired')
            for part in e['paired_seeds']:
                expected_keys=set(score_keys(e['view']))
                if set(part['left_keys'])!=expected_keys or set(part['right_keys'])!=expected_keys:
                    raise ValueError('all applicable 18/54 score curves required')
                seed=part['source_seed'];candidate=f'scores__CONSTRAINED__S{seed}';control=f'scores__{e["control"]}__S{seed}'
                left,right=(control,candidate) if e['metric']=='pair_benefit' else (candidate,control)
                if (part['left_table'],part['right_table'],part['truth_table'],part['truth_key'],part.get('sign',1))!=(left,right,'truth','sex',1):
                    raise ValueError('recovery benefit/worsening sign or paired source differs')
            required={('control_gates',f'{anchor}__S{seed}') for anchor in ('U0_FULL','RAW_FULL') for seed in (0,1)}
            if {(d['table'],d['key']) for d in e.get('failure_dependencies',[])}!=required:
                raise ValueError('all four mandatory real control gates required')
    return True


def _finite(value):
    value=float(value)
    if not math.isfinite(value): raise ValueError('finite selection metrics required')
    return value


def task_deltas(task, baseline):
    return {'accuracy_loss':_finite(baseline['accuracy'])-_finite(task['accuracy']),
            'nll_excess':_finite(task['nll'])-_finite(baseline['nll']),
            'brier_excess':_finite(task['brier'])-_finite(baseline['brier'])}


def common_epoch(records, starts, arm):
    """Select only from complete both-seed FIT2/ATT records; no INNER inputs.

    records: arm_id, seed, epoch, FIT2={task:{accuracy,nll,brier}},
    ATT_validation_reader_auc={full score key: oriented validation AUC}.
    starts: seed->{task:{accuracy,nll,brier}} from exact RAW_HYBRID.
    """
    if arm not in ARMS or set(starts)!={0,1}: raise ValueError('mandatory arm/source starts')
    lookup={}
    for r in records:
        if r['arm_id']!=arm: continue
        key=(r['epoch'],r['seed'])
        if key in lookup or key[0] not in (0,5,10,20) or key[1] not in (0,1):
            raise ValueError('unique predeclared checkpoint per seed')
        lookup[key]=r
    if set(lookup)!={(epoch,seed) for epoch in (0,5,10,20) for seed in (0,1)}:
        return {'arm_id':arm,'status':'MISSING_REQUIRED_SEED_OR_CHECKPOINT','epoch':None}
    candidates=[]
    for epoch in (0,5,10,20):
        excess=[];nll=[];auc=[];deltas=[]
        for seed in (0,1):
            r=lookup[epoch,seed];seed_deltas={}
            for task in TASKS:
                d=task_deltas(r['FIT2'][task],starts[seed][task]);seed_deltas[task]=d
                excess += [v/CAPS[k]-1 for k,v in d.items()]
            values=r['ATT_validation_reader_auc']
            if not set(score_keys('pair'))<=values.keys(): raise ValueError('complete 54 ATT pair scores required')
            auc.append(max(_finite(values[key]) for key in score_keys('pair')))
            nll.append(_finite(r['FIT2']['occupation']['nll']));deltas.append(seed_deltas)
        candidates.append({'epoch':epoch,'proxy_eligible':max(excess)<=0,
                           'maximum_normalized_task_excess':max(excess),
                           'paired_occupation_nll':sum(nll)/2,'paired_ATT_pair_auc':sum(auc)/2,
                           'per_seed_deltas':deltas})
    eligible=[r for r in candidates if r['proxy_eligible']]
    if eligible: chosen=min(eligible,key=lambda r:(r['paired_ATT_pair_auc'],r['paired_occupation_nll'],r['epoch']))
    else: chosen=min(candidates,key=lambda r:(r['maximum_normalized_task_excess'],r['paired_occupation_nll'],r['paired_ATT_pair_auc'],r['epoch']))
    return {'arm_id':arm,'status':'NOMINATED' if eligible else 'NOMINATED_PROXY_INELIGIBLE',
            'epoch':chosen['epoch'],'proxy_eligible':chosen['proxy_eligible'],'candidates':candidates,
            'INNER_used':False,'source_training_saw_all_FIT':True}


def strongest_control(clauses, recovery):
    """Point eligibility uses the full 24-clause original conjunction."""
    eligible=[]
    for control in CONTROLS:
        records=[e for e in clauses if e.get('service',e.get('arm_id'))==control]
        keys={(e['seed'],e['task'],e['reference'],e['metric']) for e in records}
        required={(s,t,r,m) for s in (0,1) for t in TASKS for r in ('U0','Ucal') for m in CAPS}
        if len(records)!=24 or keys!=required: continue
        if not all(e.get('point') is not None and _finite(e['point'])<=CAPS[e['metric']] for e in records): continue
        if control not in recovery or set(recovery[control])!={0,1}: continue
        eligible.append((sum(_finite(recovery[control][s]) for s in (0,1))/2,CONTROLS.index(control),control))
    return {'status':'SELECTED' if eligible else 'MISSING_POINT_ELIGIBLE_CONTROL',
            'control':min(eligible)[2] if eligible else None,
            'eligible_controls':[p[2] for p in eligible],'tie_order':list(CONTROLS)}


def refit_targets(records):
    """Freeze 54 attempts from ATT validation, including repeated aliases."""
    targets=[]
    for service in SERVICES:
        for seed in (0,1):
            matches=[r for r in records if r['arm_id']==service and r['seed']==seed]
            if len(matches)!=1: raise ValueError('mandatory unique final service/seed ATT bank')
            r=matches[0]
            for view in ('r1','r2','pair'):
                values=r['ATT_validation_reader_auc'];keys=score_keys(view)
                if not set(keys)<=values.keys(): raise ValueError('mandatory complete ATT reader bank')
                key=min(keys,key=lambda k:(-_finite(values[k]),keys.index(k)))
                reader=key.split('__')[-1];actual_view=view
                if '__IGNORE_r2__' in key:actual_view='r1'
                elif '__IGNORE_r1__' in key:actual_view='r2'
                target={'target_id':f'{service}_S{seed}_{view}','arm_id':service,'seed':seed,
                        'view':actual_view,'requested_view':view,'reader':reader,'score_key':key,
                        'orientation':r['orientations'][key],
                        'model_seed':r['model_seed'],'att_release':r['att_release'],
                        'inner_release':r['inner_release'],'freeze':r['freeze'],
                        'selected_on':'ATT validation only','INNER_labels_opened':False}
                targets.append(target)
    return {'schema':'CIS1-independent-refit-targets-v1','status':'FROZEN',
            'registered_attempts':len(targets),'targets':targets,
            'architecture_counts':{kind:sum(kind in t['reader'] for t in targets)
                                   for kind in ('LR','MLP','HGB')}}
