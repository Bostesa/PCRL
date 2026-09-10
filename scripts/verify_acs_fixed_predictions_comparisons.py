"""Independent CSV arithmetic replay. Does not import the reporting functions."""
import argparse,csv,json,itertools,math,statistics,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TASKS=('income_binary','civilian_at_work','public_coverage','same_residence','commute_over20')
SCOPES=('standard_independent','expanded_independent','expanded_catchup')
ARMS=('H','E','A0','L025','L20','J')
def read(p):return json.loads(p.read_text())
def rows(p):return list(csv.DictReader(p.open()))
def truth(x):return None if x in ('','None') else x=='True'
def main(out):
    tick=time.perf_counter();rules=read(out/'comparison_rules.json');cfg=read(out/'config.json');points={};result={'points':0,'selected_endpoints':0,'source_task_checks':0,'increment_checks':0,'pair_checks':0,'matching_checks':0,'aggregate_checks':0,'max_numeric_error':0.}
    def close(a,b):
        d=abs(float(a)-float(b));result['max_numeric_error']=max(result['max_numeric_error'],d);assert d<=1e-12,(a,b)
    for seed in range(3):
        controls=read(ROOT/cfg['coalition_reference_results']/f'seed_{seed}'/'controls/metrics.json')['raw_metrics'];priors={r['target']:r for r in controls if r['condition']=='prior'}
        for c in ARMS:
            data=read(out/f'seed_{seed}'/c/'metrics.json')['raw_metrics']
            for split,w,b,scope in itertools.product(('validation','test'),('unweighted','person_weighted'),(120,360),SCOPES):
                score=split+('_person_weighted' if w=='person_weighted' else '');p={'utility':{},'gains':{}}
                for r in data:
                    if r['role']=='utility' and r['selected_scopes']:p['utility'][r['target']]=r['scores'][score]['log_loss']
                    elif r['role']=='audit' and r['audit_budget']==b and scope in r['selected_scopes']:p['gains'][r['view']+'/'+r['target']]=priors[r['target']]['scores'][score]['log_loss']-r['scores'][score]['log_loss']
                assert len(p['utility'])==5 and len(p['gains'])==11;points[seed,c,split,w,b,scope]=p;result['points']+=1
    def key(r,cfield='condition'):return int(r['seed']),r[cfield],r['split'],r['weight'],int(r.get('budget') or 360),r.get('scope') or 'expanded_catchup'
    for r in rows(out/'PER_SEED.csv'):close(r['value'],points[key(r)][r['kind']][r['endpoint']]);result['selected_endpoints']+=1
    for r in rows(out/'SOURCE_FEASIBILITY.csv'):
        p=points[key(r)]['utility'];parent=rules['original_parent_metric_identity'][r['seed']]['tasks'];score=r['split']+('_person_weighted' if r['weight']=='person_weighted' else '');loss=p[r['task']];baseline=parent[r['task']]['scores'][score];close(r['loss'],loss);close(r['parent_loss'],baseline);close(r['excess_over_allowance'],loss-baseline-.01)
        assert truth(r['pass'])==(loss<=baseline+.01+1e-12);assert truth(r['all_source_pass'])==all(p[t]<=parent[t]['scores'][score]+.01+1e-12 for t in TASKS[:3]);result['source_task_checks']+=1
    for r in rows(out/'INCREMENTAL.csv'):
        k=key(r);x=points[k]['gains'][r['endpoint']];h=points[(k[0],'H',*k[2:])]['gains'][r['endpoint']];close(r['absolute_gain'],x);close(r['H_gain'],h);close(r['increment'],x-h);result['increment_checks']+=1
    for r in rows(out/'PAIRED.csv'):
        left=points[key(r,'left')][r['kind']][r['endpoint']];right=points[key(r,'right')][r['kind']][r['endpoint']];close(r['difference'],left-right);result['pair_checks']+=1
    match_groups={}
    for r in rows(out/'UTILITY_MATCHES.csv'):
        assert r['J']=='J' and r['comparator'] in ('L025','L20','A0','E','H');seed=int(r['seed']);k=key(r,'J');p=points[k];q=points[(seed,r['comparator'],*k[2:])];score=r['split']+('_person_weighted' if r['weight']=='person_weighted' else '');par={t:x['scores'][score] for t,x in rules['original_parent_metric_identity'][str(seed)]['tasks'].items()};delta=float(r['delta']);included=rules['panels'][r['panel']]
        ds={t:p['utility'][t]-q['utility'][t] for t in TASKS};dg={t:p['gains'][t]-q['gains'][t] for t in p['gains']}
        both=all(z['utility'][t]<=par[t]+.01+1e-12 for z in (p,q) for t in TASKS[:3]);cl=all(abs(ds[t])<=delta+1e-12 for t in included);dire=all(ds[t]<=delta+1e-12 for t in included);lower=dg['AB/'+r['attribute']] < -1e-12
        assert truth(r['both_source_feasible'])==both and truth(r['utility_close'])==cl and truth(r['utility_directional'])==dire
        assert truth(r['qualifies_close'])==(both and cl and lower) and truth(r['qualifies_directional'])==(both and dire and lower)
        for t,d in json.loads(r['utility_differences']).items():close(d,ds[t])
        for t,d in json.loads(r['audit_gain_differences']).items():close(d,dg[t])
        k=tuple(r[x] for x in ('J','comparator','split','weight','budget','scope','panel','delta','attribute'));match_groups.setdefault(k,[]).append((seed,both,cl,dire,lower));result['matching_checks']+=1
    for r in rows(out/'UTILITY_MATCHES_AGGREGATE.csv'):
        k=tuple(r[x] for x in ('J','comparator','split','weight','budget','scope','panel','delta','attribute'));v=match_groups[k];assert sorted(x[0] for x in v)==[0,1,2];assert int(r['n_seeds'])==3
        for name,fn in {'both_source_feasible':lambda x:x[1],'utility_close':lambda x:x[2],'utility_directional':lambda x:x[3],'eligible_close':lambda x:x[1] and x[2],'eligible_directional':lambda x:x[1] and x[3],'qualifies_close':lambda x:x[1] and x[2] and x[4],'qualifies_directional':lambda x:x[1] and x[3] and x[4]}.items():assert int(r[name])==sum(fn(x) for x in v)
        result['aggregate_checks']+=1
    for r in rows(out/'PAIRED_AGGREGATE.csv'):
        v=[]
        for seed in range(3):
            k=(seed,r['left'],r['split'],r['weight'],int(r['budget']),r['scope']);j=(seed,r['right'],*k[2:]);v.append(points[k][r['kind']][r['endpoint']]-points[j][r['kind']][r['endpoint']])
        close(r['mean'],sum(v)/3);close(r['sample_sd'],math.sqrt(sum((x-sum(v)/3)**2 for x in v)/2));assert int(r['n_seeds'])==3;result['aggregate_checks']+=1
    result.update(passed=True,runtime_seconds=time.perf_counter()-tick,independent_of_reporter=True,fixed_comparator_seed_identities=True)
    (out/'COMPARISON_REPLAY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'results/redesign_20260909_acs_fixed_predictions_v1');a=p.parse_args();main(a.out)
