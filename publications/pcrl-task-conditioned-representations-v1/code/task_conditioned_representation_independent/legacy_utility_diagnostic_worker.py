"""Aggregate-only diagnostic for a preserved independent legacy utility failure."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args()
    from task_conditioned_representation.scheduler import assert_active_from_env
    guard=assert_active_from_env(stage='verification');cfg=json.loads(Path(a.spec).read_text())
    def checked(pin):
        assert_active_from_env(stage='verification');path=Path(pin['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=pin['sha256']:raise ValueError('legacy utility replay source hash differs')
        return path
    nomination=json.loads(checked(cfg['nomination_lock']).read_text())
    if nomination.get('status')!='FROZEN' or len(nomination.get('nominees',[]))!=22:raise PermissionError('all nominees must freeze before task labels')
    import numpy as np
    from .science import task_rows,metrics
    from .legacy_replay import law_check
    def npz(pin):
        with np.load(checked(pin),allow_pickle=False) as z:return {k:z[k] for k in z.files}
    # Data-free check distinguishes exact expected utility from mean-probability scoring.
    fixture=task_rows(np.array([0,0]),np.array([[1.,0.],[0.,1.]])).mean(0)
    if not np.allclose(fixture,[.5,.5*np.log(1e12),1.],atol=1e-12,rtol=0):raise ValueError('expected per-action metric fixture failed')
    source=json.loads(checked(cfg['source_manifest']).read_text());role=source['roles']['INNER_SELECTION'];x=npz(role['features'])
    ys=[]
    for name in ('income','occupation'):
        record=npz(role[name])
        if not all(np.array_equal(record[k],x[k]) for k in ('row_id','unit')):raise ValueError('legacy utility task people differ')
        ys.append(record[name])
    refs={s:npz(role[f'references_s{s}']) for s in (0,1)}
    reported=json.loads(checked(cfg['reported_utility']).read_text());results=[];eligibility={};max_error=0.;trace=[]
    source_utility=npz(cfg['reported_utility_inputs'])
    if len(cfg['instances'])!=4:raise ValueError('complete four legacy arms required')
    for item in cfg['instances']:
        bank=json.loads(checked(item['bank']).read_text());law=npz(item['inner_law']);indexes,mass=law_check(law,bank,x);n,actions=mass.shape;clauses=[];tasks=[]
        actual=next(r for r in reported['records'] if (r['arm_id'],r['seed'])==(item['family'],item['seed']))
        for r,y in enumerate(ys,1):
            probabilities=np.asarray(bank['recipients'][str(r)]['probability'],float)[indexes[f'r{r}']]
            vectors=task_rows(np.repeat(y,actions),probabilities.reshape(n*actions,probabilities.shape[-1])).reshape(n,actions,3)
            expected=np.sum(vectors*mass[:,:,None],axis=1);point=expected.mean(0);name=('income','occupation')[r-1]
            observed=actual['tasks'][name]['metrics']
            for j,metric in enumerate(('accuracy','nll','brier')):
                error=abs(point[j]-observed[metric]);max_error=max(max_error,float(error))
                if error>1e-12:raise ValueError('legacy expected task metric differs')
            tasks.append(dict(zip(('accuracy','nll','brier'),map(float,point))))
            for refname in ('U0','Ucal'):
                if not all(np.array_equal(refs[item['seed']][k],x[k]) for k in ('row_id','unit')):raise ValueError('legacy useful reference people differ')
                reference=metrics(y,refs[item['seed']][f'{refname}_{r}'])
                for j,(metric,cap) in enumerate([('accuracy',.010),('nll',.010),('brier',.005)]):
                    delta=reference[metric]-point[j] if metric=='accuracy' else point[j]-reference[metric]
                    reported_name={'accuracy':'accuracy_loss','nll':'nll_excess','brier':'brier_excess'}[metric]
                    clause=next(c for c in actual['clauses'] if (c['task'],c['reference'],c['metric'])==(name,refname,reported_name))
                    error=abs(delta-clause['value']);max_error=max(max_error,float(error))
                    trace.append({'family':item['family'],'seed':item['seed'],'task':name,'reference':refname,'metric':metric,'own_delta':float(delta),'reported_delta':clause['value'],'own_cap_pass':bool(delta<=cap),'reported_cap_pass':clause['pass'],'delta_error':float(error),'own_reference_metric':reference[metric],'reported_reference_metric':actual['tasks'][name]['references'][refname]['metrics'][metric],'utility_input_reference_metric':metrics(y,source_utility[f"s{item['seed']}__{refname}__p{r}"])[metric],'source_vs_utility_input_probability_error':float(np.max(abs(refs[item['seed']][f'{refname}_{r}']-source_utility[f"s{item['seed']}__{refname}__p{r}"]))),'candidate_metric':float(point[j]),'reported_candidate_metric':observed[metric]})
                    clauses.append({'task':name,'reference':refname,'metric':metric,'point':float(delta),'cap':cap,'pass':bool(delta<=cap)})
        results.append({'family':item['family'],'seed':item['seed'],'task_metrics':tasks,'point_eligible_seed':all(c['pass'] for c in clauses),'clauses':clauses})
    for family in ('P0','S12'):
        rows=[r for r in results if r['family']==family]
        if sorted(r['seed'] for r in rows)!=[0,1]:raise ValueError('legacy two-seed admission incomplete')
        eligibility[family]=all(r['point_eligible_seed'] for r in rows)
        pass
    out={'schema':'TCR1-independent-legacy-utility-aggregate-diagnostic-v1','status':'DIAGNOSTIC_COMPLETE','trace':trace,'guard':guard,'maximum_numeric_error':max_error,'family_point_eligibility':eligibility,'records':results,'data_free_expected_utility_fixture_passed':True,'INNER_SEX_opened':False,'primary_pool_changed':False,'fits':0}
    path=Path(cfg['output']);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':main()
