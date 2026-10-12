"""Queue-only comparison of independent results with the reported endpoint tables."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args()
    from task_conditioned_representation.scheduler import assert_active_from_env
    guard=assert_active_from_env(stage='verification')
    cfg=json.loads(Path(a.spec).read_text())
    def read(pin):
        assert_active_from_env(stage='verification')
        path=Path(pin['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=pin['sha256']:raise ValueError('comparison input hash changed')
        return json.loads(path.read_text())
    own=read(cfg['independent_replay']);reported=read(cfg['reported_inference'])
    if own['status']!='PASS_NEW_CONTINUOUS_SERVICE_REPLAY' or reported['K']!={'primary':51,'utility':240,'secondary':33}:raise ValueError('completed replay or multiplicity missing')
    count=0;missing=0;max_error=0.;checks=[]
    def compare(left,right,identity):
        nonlocal count,max_error
        for key in ('point','se','lower','upper','critical'):
            error=abs(float(left[key])-float(right[key]));max_error=max(max_error,error)
            if error>1e-10:raise ValueError(f'independent endpoint mismatch {identity}: {key} error={error}')
        count+=1
    for family,n in [('primary',51),('utility',240),('secondary',33)]:
        rows=reported['families'][family]
        if len(rows)!=n:raise ValueError('reported endpoint family incomplete')
        for row in rows:
            if row['interval'] is None:
                if family!='primary' or row['status']!='MISSING_DEPENDENCY':raise ValueError('unexpected missing endpoint')
                if row['kind']=='privacy':
                    gates=own['point_eligibility']
                    if all(gates[f'{arm}__s{seed}'] for arm in (row['arm'],row['comparator']) for seed in (0,1)):raise ValueError('available qualified comparison missing')
                missing+=1;continue
            if row['kind']=='utility':
                task={'income':0,'occupation':1}[row['task']];metric=row['metric'].split('_')[0]
                key=f"{row['arm']}__s{row['seed']}__task{task}__{row['reference']}__{metric}"
                table=own['primary_endpoints'] if family=='primary' else own['secondary_utility_endpoints']
            else:
                view={'pair_benefit':'pair','r1_worsening':'r1','r2_worsening':'r2'}[row['metric']]
                key=f"{row['comparator']}__{view}";table=own['primary_endpoints'] if family=='primary' else own['score_only_endpoints']
            compare(table[key],row['interval'],row['id'])
    recovery={(r['family'],r['seed']):r for r in own['complete_view_recovery']}
    legacy={(r['family'],r['seed']):r for r in own['legacy_replay']}
    for row in reported['absolute_recovery']:
        arm,seed,view=row['arm_id'],row['seed'],row['view'];scope='score' if row['contract']=='output' else row['contract']
        if seed is None:
            if arm=='H_RAW_CSTAR':target=own['historical_reference_descriptive'][f'{scope}__{view}']
            elif arm in ('P0','S12'):target=own['legacy_absolute_recovery'][f'{arm}__mean__{view}']
            else:target=own['absolute_recovery'][f'{arm}__{scope}__{view}']
            if target is not None:compare(target,row['interval'],f'absolute {arm}/{scope}/{view}')
        else:
            target=own['legacy_absolute_recovery'][f'{arm}__s{seed}__{view}'] if arm in ('P0','S12') else own['per_seed_absolute_recovery'][f'{arm}__s{seed}__{scope}__{view}']
            compare(target,row['interval'],f'absolute {arm}/s{seed}/{scope}/{view}')
    out={'schema':'TCR1-independent-reported-endpoint-comparison-v1','status':'PASS','guard':guard,'intervals_compared':count,'primary_missing_dependencies_preserved':missing,'maximum_numeric_error':max_error,'multiplicity':reported['K'],'scope':'independent model/reader/utility/AUC and paired2048 bootstrap replay compared with reported intervals; unavailable primary slots remain unavailable'}
    path=Path(cfg['output']);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':main()
