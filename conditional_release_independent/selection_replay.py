"""Independent FIT-objective nomination from verified point-utility receipts."""
import argparse,json,os
from pathlib import Path
from .reconstruction import digest


def run(config):
    from conditional_release.scheduler import assert_active_from_env
    active=assert_active_from_env(stage='verification');state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text());pins={x['path']:x['sha256'] for x in state['units'][active['unit_id']]['spec']['sources']}
    if pins.get(str(Path(config).absolute()))!=digest(config):raise PermissionError('unqueued nomination config')
    cfg=json.loads(Path(config).read_text())
    def read(e):
        if pins.get(e['path'])!=e['sha256'] or digest(e['path'])!=e['sha256']:raise PermissionError('unqueued nomination input')
        return json.loads(Path(e['path']).read_text())
    lock=read(cfg['science']);priority=lock['candidate_priority'];candidates={};unitstatuses=[]
    for entry in cfg['independent_policy_receipts']:
        r=read(entry)
        if r['status']!='PASS':raise ValueError('unverified nomination inputs')
        for row in r['results']:
            arm=row['arm_id'];seed=row['seed']
            if arm not in priority:continue
            if (arm,seed) in candidates:raise ValueError('duplicate candidate record')
            eligible=len(row['inner_point_utility'])==12 and all(s['pass'] for s in row['inner_point_utility'])
            if eligible!=row['inner_point_eligible']:raise ValueError('independent nomination gate incomplete')
            candidates[(arm,seed)]=row
    for entry in cfg['formulation_receipts']:
        r=read(entry)
        if r['status']!='PASS':raise ValueError('formulation verification missing')
        for setup in r['setups']:
            arm=setup.get('arm_id')
            if arm is None:arm=('G' if setup['mode']=='scalar' else 'C')+'_'+setup['budget']
            unitstatuses.append({'arm_id':arm,'seed':r['seed'],'status':setup['status']})
            if setup['status']=='COMPLETE' and (arm,r['seed']) not in candidates:raise ValueError('complete candidate lacks independently verified utility')
    if {(s['arm_id'],s['seed']) for s in unitstatuses}!={(a,s) for a in priority for s in (0,1)}:raise ValueError('all sixteen candidate cells must be accounted for')
    eligible=[]
    for arm in priority:
        rows=[candidates.get((arm,s)) for s in (0,1)]
        if all(r is not None and r['inner_point_eligible'] for r in rows):eligible.append({'arm_id':arm,'mean_FIT_objective':sum(r['fitted_objective'] for r in rows)/2})
    def choose(pool):
        if not pool:return None
        best=min(x['mean_FIT_objective'] for x in pool)
        return min((x for x in pool if x['mean_FIT_objective']<=best+1e-10),key=lambda x:priority.index(x['arm_id']))
    nominee=choose(eligible);scalar=choose([x for x in eligible if x['arm_id'].startswith('G_')])
    if cfg.get('reported_nomination'):
        reported=read(cfg['reported_nomination'])
        if reported[cfg['reported_arm_key']]!=(None if nominee is None else nominee['arm_id']):raise ValueError('reported nomination differs from frozen ranking')
    result={'schema':'CBR1-independent-nomination-replay-v1','status':'PASS','unit_id':active['unit_id'],'candidate_cells':unitstatuses,'both_seed_point_eligible':eligible,'nominee':nominee,'best_scalar':scalar,'selection_outcome_inputs':'FIT objective and both-seed INNER task utility only; no INNER SEX or reporting AUC','tie_tolerance':1e-10,'priority':priority,'scope':'Point nomination and structural attribution are separate from simultaneous utility/recovery passage. All eight designs/two seeds accounted for; reporting remains exploratory.'}
    with Path(cfg['output']).open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'PASS','nominee':nominee,'best_scalar':scalar}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);run(p.parse_args().config)
