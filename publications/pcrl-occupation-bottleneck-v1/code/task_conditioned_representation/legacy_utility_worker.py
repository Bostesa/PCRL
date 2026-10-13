"""Queue-only original utility gates for fixed historical stochastic score laws."""
from __future__ import annotations
import argparse,json
from pathlib import Path


def run(spec_path):
    from .attack_worker import queue_config
    active,cfg,verify=queue_config(spec_path,'attacks')
    import numpy as np
    from .source import read_exact
    from .attacks import write_json
    from .inference import task_vectors,task_metrics,point_utility
    from .utility_worker import utility_clause_pass,reference_probability_key
    from conditional_release.attacks import law_arrays
    lock=json.loads(verify(cfg['nomination_lock']['path'],cfg['nomination_lock']['sha256']).read_text())
    if lock.get('status')!='FROZEN' or not lock.get('nominees'):raise PermissionError('global family nomination freeze absent')
    verify(cfg['utility_inputs']['path'],cfg['utility_inputs']['sha256'])
    with np.load(cfg['utility_inputs']['path'],allow_pickle=False) as z:u={k:z[k].copy() for k in z.files}
    if any(reference_probability_key(s,r,i) not in u for s in (0,1) for r in ('U0','Ucal') for i in (1,2)):raise PermissionError('distinct original reference probability keys absent')
    records=[]
    for item in cfg['instances']:
        bank=json.loads(verify(item['bank']['path'],item['bank']['sha256']).read_text());verify(item['inner_law']['path'],item['inner_law']['sha256'])
        law=read_exact(item['inner_law'],{'row_id','unit','code_r1','code_r2','mass'});law_arrays(law,bank)
        if not all(np.array_equal(law[k],u[k]) for k in ('row_id','unit')):raise PermissionError('historical task-law people do not match admitted INNER utility people')
        mass=law['mass'];n,a=mass.shape;tasks={};clauses=[]
        for i,key in enumerate(('income','occupation'),1):
            y=u[key];table=np.asarray(bank['recipients'][str(i)]['probability'],dtype=np.float64);p=table[law[f'code_r{i}']]
            vectors=task_vectors(np.repeat(y,a),p.reshape(n*a,p.shape[-1]))
            expected={k:(v.reshape(n,a)*mass).sum(1) for k,v in vectors.items()}
            metrics={k:float(v.mean()) for k,v in expected.items()}
            recall=[float(expected['accuracy'][y==c].mean()) if np.any(y==c) else 0. for c in range(p.shape[-1])]
            metrics.update(confidence=float((p.max(2)*mass).sum(1).mean()),balanced_accuracy=float(np.mean(recall)),per_class_recall=recall,class_order=list(range(p.shape[-1])),class_counts=[int((y==c).sum()) for c in range(p.shape[-1])])
            references={}
            for ref in ('U0','Ucal'):
                reference=task_metrics(y,u[reference_probability_key(item['seed'],ref,i)]);delta=point_utility(metrics,reference)
                references[ref]={'metrics':reference,'difference':delta}
                for name,cap in (('accuracy_loss',.010),('nll_excess',.010),('brier_excess',.005)):
                    clauses.append({'task':key,'reference':ref,'metric':name,'value':delta[name],'cap':cap,'pass':utility_clause_pass(delta[name],cap)})
            tasks[key]={'metrics':metrics,'references':references}
        records.append({'arm_id':item['arm_id'],'seed':item['seed'],'bank_sha256':item['bank']['sha256'],'inner_law_sha256':item['inner_law']['sha256'],'tasks':tasks,'clauses':clauses,'clauses_passed':sum(c['pass'] for c in clauses),'point_eligible_seed':all(c['pass'] for c in clauses)})
    eligibility={}
    for arm in ('P0','S12'):
        rows=[r for r in records if r['arm_id']==arm]
        if sorted(r['seed'] for r in rows)!=[0,1]:raise PermissionError('both fixed historical seeds required')
        eligibility[arm]={'point_eligible':all(r['point_eligible_seed'] for r in rows),'passed':sum(r['clauses_passed'] for r in rows),'total':24}
    out={'schema':'TCR1-fresh-historical-score-law-utility-v2','unit_id':active['unit_id'],'reference_namespace':'REF_U0/REF_Ucal original admitted service probabilities','metric_contract':'per-person exact finite-law expected decision accuracy, expected per-action NLL with floor1e-12 and expected summed multiclass Brier; no averaged-probability scoring','utility_input_sha256':cfg['utility_inputs']['sha256'],'records':records,'family_eligibility':eligibility,'INNER_SEX_opened':False,'new_fits_or_calibration':False,'primary_pool_changed':False,'secondary_contract':'literal historical code/probability/decision/metadata wire'}
    write_json(cfg['public_output'],out,False);write_json(cfg['private_output'],out);return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();out=run(a.spec);print(json.dumps({'status':'REPORTED','eligibility':out['family_eligibility']}))
if __name__=='__main__':main()
