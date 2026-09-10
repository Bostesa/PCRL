"""Frozen arithmetic for fixed-source predictions; no fitting or release selection."""
from __future__ import annotations
import argparse,csv,json,statistics,itertools
from pathlib import Path
from scripts.acs_coalition_strength_comparisons import evaluate_pair,source_check
ROOT=Path(__file__).resolve().parents[1]
CONDITIONS=('H','E','A0','L025','L20','J')
SCOPES=('standard_independent','expanded_independent','expanded_catchup')
def read(p):return json.loads(Path(p).read_text())
def csvout(p,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
def selected(out):
    cfg=read(out/'config.json');points={};candidate=[];flat=[]
    for s in range(3):
        control=read(ROOT/cfg['coalition_reference_results']/f'seed_{s}'/'controls/metrics.json')['raw_metrics']
        prior={r['target']:r['scores'] for r in control if r['condition']=='prior'}
        for c in CONDITIONS:
            p=out/f'seed_{s}'/c/'metrics.json'
            if not p.exists():continue
            rows=read(p)['raw_metrics'];candidate.extend(rows)
            for split,weight,budget,scope in itertools.product(('validation','test'),('unweighted','person_weighted'),(120,360),SCOPES):
                key=split+('_person_weighted' if weight=='person_weighted' else '')
                u={r['target']:r['scores'][key]['log_loss'] for r in rows if r['role']=='utility' and r['selected_scopes']}
                g={r['view']+'/'+r['target']:prior[r['target']][key]['log_loss']-r['scores'][key]['log_loss'] for r in rows if r['role']=='audit' and r['audit_budget']==budget and scope in r['selected_scopes']}
                point={'utility':u,'gains':g};points[s,c,split,weight,budget,scope]=point
                for kind,values in point.items():
                    for endpoint,value in values.items():flat.append({'seed':s,'condition':c,'split':split,'weight':weight,'budget':budget,'scope':scope,'kind':kind,'endpoint':endpoint,'value':value})
    return points,candidate,flat

def report(out):
    rules=read(out/'comparison_rules.json');points,candidates,flat=selected(out);csvout(out/'PER_SEED.csv',flat)
    incremental=[];paired=[];matches=[];feasibility=[]
    for (s,c,split,w,b,scope),point in points.items():
        parent={t:v['scores'][split+('_person_weighted' if w=='person_weighted' else '')] for t,v in rules['original_parent_metric_identity'][str(s)]['tasks'].items()}
        if b==360 and scope=='expanded_catchup':
            check=source_check(point,parent)
            for t,v in check['tasks'].items():feasibility.append({'seed':s,'condition':c,'split':split,'weight':w,'task':t,**v,'all_source_pass':check['pass']})
        h=points[s,'H',split,w,b,scope]
        for endpoint,value in point['gains'].items():incremental.append({'seed':s,'condition':c,'split':split,'weight':w,'budget':b,'scope':scope,'endpoint':endpoint,'absolute_gain':value,'H_gain':h['gains'][endpoint],'increment':value-h['gains'][endpoint],'sensitive_excess_reference':.005 if endpoint.split('/')[1] in ('SEX','RAC1P') else None})
        if c=='J':
            for local in ('L025','L20','A0','E','H'):
                if (s,local,split,w,b,scope) not in points:continue
                p=points[s,local,split,w,b,scope]
                for panel,tasks in rules['panels'].items():
                    for delta,attribute in itertools.product(rules['delta_values'],('SEX','RAC1P')):
                        value=evaluate_pair(point,p,parent,tasks,delta,attribute)
                        matches.append({'seed':s,'J':'J','comparator':local,'split':split,'weight':w,'budget':b,'scope':scope,'panel':panel,'delta':delta,'attribute':attribute,**value})
        for right in CONDITIONS:
            if right==c or (s,right,split,w,b,scope) not in points:continue
            p=points[s,right,split,w,b,scope]
            for kind,values in point.items():
                for ep,v in values.items():paired.append({'seed':s,'left':c,'right':right,'split':split,'weight':w,'budget':b,'scope':scope,'kind':kind,'endpoint':ep,'difference':v-p[kind][ep]})
    for name,rows in [('INCREMENTAL',incremental),('PAIRED',paired),('UTILITY_MATCHES',matches),('SOURCE_FEASIBILITY',feasibility)]:csvout(out/(name+'.csv'),rows)
    groups={}
    for r in paired:
        k=tuple((k,v) for k,v in r.items() if k not in ('seed','difference'));groups.setdefault(k,[]).append(r['difference'])
    agg=[{**dict(k),'n_seeds':len(v),'mean':statistics.mean(v),'sample_sd':statistics.stdev(v) if len(v)>1 else None} for k,v in groups.items()];csvout(out/'PAIRED_AGGREGATE.csv',agg)
    groups={}
    for r in matches:
        k=tuple((k,r[k]) for k in ('J','comparator','split','weight','budget','scope','panel','delta','attribute'));groups.setdefault(k,[]).append(r)
    aggm=[{**dict(k),'n_seeds':len(rs),**{name:sum(r[name] is True for r in rs) for name in ('both_source_feasible','utility_close','utility_directional','qualifies_close','qualifies_directional')},'eligible_close':sum(r['both_source_feasible'] is True and r['utility_close'] is True for r in rs),'eligible_directional':sum(r['both_source_feasible'] is True and r['utility_directional'] is True for r in rs)} for k,rs in groups.items()];csvout(out/'UTILITY_MATCHES_AGGREGATE.csv',aggm)
    # Every selected endpoint is public in flat table; full per-candidate scores remain hash-bound unit evidence.
    means=[]
    for c,split,w,b,scope in itertools.product(CONDITIONS,('validation','test'),('unweighted','person_weighted'),(120,360),SCOPES):
        ps=[points[s,c,split,w,b,scope] for s in range(3) if (s,c,split,w,b,scope) in points]
        if not ps:continue
        for kind in ('utility','gains'):
            for ep in ps[0][kind]:
                v=[p[kind][ep] for p in ps];means.append({'condition':c,'split':split,'weight':w,'budget':b,'scope':scope,'kind':kind,'endpoint':ep,'n_seeds':len(v),'mean':statistics.mean(v),'sample_sd':statistics.stdev(v) if len(v)>1 else None})
    csvout(out/'MEANS.csv',means)
    (out/'REPORT_COUNTS.json').write_text(json.dumps({'points':len(points),'candidate_records':len(candidates),'selected_endpoint_rows':len(flat),'matching_rows':len(matches),'paired_rows':len(paired),'all_means_include_all_available_seeds':True},indent=2)+'\n')
    return points,means,feasibility,incremental,aggm
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'results/redesign_20260909_acs_fixed_predictions_v1');a=p.parse_args();report(a.out)
