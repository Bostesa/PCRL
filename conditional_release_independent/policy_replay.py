"""Independent deployment-law, utility/caps and FIT guard replay for all arms."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from .reconstruction import digest,admitted_arrays,folds,merge_rows
from .policy_laws import checked_law
from .guard_replay import source_guard_tables
from .arithmetic import bayes_objective,change_rate,validate_group_pairing


def cost_tables(p):
    result={}
    for metric in ('accuracy','nll','brier'):
        rows=[]
        for y in range(p.shape[1]):
            if metric=='accuracy':v=-(p.argmax(1)==y).astype(float)
            elif metric=='nll':v=-np.log(np.maximum(p[:,y],1e-12))
            else:v=(p*p).sum(1)-2*p[:,y]+1
            rows.append(v)
        result[metric]=np.asarray(rows)
    return result


def source_numbers(source,seed,policy='V_J'):
    rec=source['seeds'][str(seed)]['policies'][policy]
    if digest(rec['receipt']['path'])!=rec['receipt']['sha256']:raise PermissionError('source policy receipt')
    receipt=json.loads(Path(rec['receipt']['path']).read_text())
    return admitted_arrays(rec['package'],set(receipt['array_sha256'])),receipt


def run(config):
    from conditional_release.scheduler import assert_active_from_env
    active=assert_active_from_env(stage='verification');state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text());pins={x['path']:x['sha256'] for x in state['units'][active['unit_id']]['spec']['sources']}
    def check(e):
        if pins.get(e['path'])!=e['sha256'] or digest(e['path'])!=e['sha256']:raise PermissionError('unqueued policy replay input')
    if pins.get(str(Path(config).absolute()))!=digest(config):raise PermissionError('unqueued policy config')
    cfg=json.loads(Path(config).read_text());check(cfg['manifest']);source=json.loads(Path(cfg['manifest']['path']).read_text());results=[]
    for item in cfg['policies']:
        seed=item['seed'];numbers,historical=source_numbers(source,seed,'V_S21' if item['kind']=='V_S21' else 'V_J');receipt=historical;package=None
        if item.get('receipt'):
            check(item['receipt']);receipt=json.loads(Path(item['receipt']['path']).read_text());check(item['package'])
            with ZipFile(item['package']['path']) as z:names={n[:-4] for n in z.namelist()}
            package=admitted_arrays(item['package'],names)
            if receipt['package_sha256']!=item['package']['sha256']:raise ValueError('policy package binding')
            if receipt['public_bank_sha256']!=source['seeds'][str(seed)]['public_bank']['sha256']:raise ValueError('policy bank binding')
        guardtables=None;guards=[];maxgap=0.;utilitygates=[];fittedobjective=None;changerate=None
        for role,entry in item['role_laws'].items():
            check(entry);saved=admitted_arrays(entry,{'row_id','unit','code_r1','code_r2','mass'})
            old=admitted_arrays(source['seeds'][str(seed)]['roles'][role]['release'],{'row_id','unit','code_r1','code_r2','prob_r1','prob_r2','decision_r1','decision_r2'})
            if not all(np.array_equal(saved[k],old[k]) for k in ('row_id','unit')):raise ValueError('law people mismatch')
            a,b,q=checked_law(numbers,old['code_r1'],old['code_r2'],item['kind'],receipt,package)
            if not np.array_equal(a,saved['code_r1']) or not np.array_equal(b,saved['code_r2']):raise ValueError('actual emitted global code order mismatch')
            gap=float(np.max(np.abs(q-saved['mass']),initial=0));maxgap=max(maxgap,gap)
            if gap>1e-12:raise ValueError('independent serialized policy law mismatch')
            if role=='OSF_DEFENSE_FIT':
                sex=admitted_arrays(source['roles'][role]['sex'],{'row_id','unit','sex'});validate_group_pairing(sex['unit'],sex['sex'])
                if not all(np.array_equal(sex[k],old[k]) for k in ('row_id','unit')):raise ValueError('FIT SEX alignment')
                rows=merge_rows(a,b,q);fittedobjective=bayes_objective(rows,sex['sex']);changerate=change_rate(rows,np.c_[old['code_r1'],old['code_r2']]);fold=folds(old['unit']);guardtables=source_guard_tables(source,seed)
                for name,g in guardtables.items():
                    codes=a if g['view']=='r1' else b;original=old['code_r1'] if g['view']=='r1' else old['code_r2'];loss=g['table'][fold[:,None],sex['sex'][:,None],codes];base=g['table'][fold,sex['sex'],original]
                    excess=(q*loss).sum(1)-base
                    for view,mask in [('pooled',np.ones(len(q),bool))]+[(f'fold{f}',fold==f) for f in range(3)]:
                        margin=float(excess[mask].mean());guards.append({'name':name,'view':view,'CE_excess':margin})
                        if margin < -1.001e-7:raise ValueError('executable policy violates frozen CE guard')
                    del loss
                del rows
            if role=='INNER_SELECTION':
                labels=[]
                for label in ('income','occupation'):
                    target=admitted_arrays(source['roles'][role][label],{'row_id','unit',label})
                    if not all(np.array_equal(target[k],old[k]) for k in ('row_id','unit')):raise ValueError('INNER task pairing')
                    labels.append(target[label])
                refs=admitted_arrays(source['roles'][role][f'references_s{seed}'],{'row_id','unit','U0_1','U0_2','Ucal_1','Ucal_2','d_1','d_2'})
                check(item['utility']);utilitykeys={f'{ref}_r{r}_{m}' for ref in ('U0','Ucal') for r in (1,2) for m in ('accuracy','nll','brier')};expected=admitted_arrays(item['utility'],utilitykeys|{'row_id','unit'})
                if not all(np.array_equal(expected[k],old[k]) and np.array_equal(refs[k],old[k]) for k in ('row_id','unit')):raise ValueError('INNER utility/reference pairing')
                for r,codes,label in ((1,a,labels[0]),(2,b,labels[1])):
                    lookup=cost_tables(numbers['probability'+str(r)])
                    for metric,cost in lookup.items():
                        actual=(q*cost[label[:,None],codes]).sum(1)
                        for ref in ('U0','Ucal'):
                            p=refs[f'{ref}_{r}']
                            if metric=='accuracy':base=-(p.argmax(1)==label).astype(float)
                            elif metric=='nll':base=-np.log(np.maximum(p[np.arange(len(p)),label],1e-12))
                            else:base=(p*p).sum(1)-2*p[np.arange(len(p)),label]+1
                            excess=actual-base;key=f'{ref}_r{r}_{metric}';gap=float(np.max(np.abs(excess-expected[key]),initial=0));maxgap=max(maxgap,gap)
                            if gap>1e-12:raise ValueError('independent actual sampled-law expected utility mismatch')
                            cap=.005 if metric=='brier' else .01;mean=float(excess.mean());utilitygates.append({'reference':ref,'task_metric':f'r{r}_{metric}','margin':mean,'cap':cap,'pass':mean<=cap})
                if len(utilitygates)!=12:raise ValueError('incomplete INNER utility gates')
        if item.get('fitted_objective') is not None and abs(fittedobjective-item['fitted_objective'])>1e-7:raise ValueError('reported FIT objective differs')
        if item.get('actual_change_rate') is not None and abs(changerate-item['actual_change_rate'])>1e-10:raise ValueError('reported actual change differs')
        if item.get('inner_point_eligible') is not None and item['inner_point_eligible']!=all(x['pass'] for x in utilitygates):raise ValueError('original-reference point eligibility differs')
        results.append({'seed':seed,'arm_id':item['arm_id'],'role_laws':list(item['role_laws']),'maximum_numeric_gap':maxgap,'fitted_objective':fittedobjective,'actual_change_rate':changerate,'complete_CE_guards':guards,'inner_point_utility':utilitygates,'inner_point_eligible':all(x['pass'] for x in utilitygates),'INNER_SEX_access':False})
    report={'schema':'CBR1-independent-policy-utility-replay-v1','status':'PASS','unit_id':active['unit_id'],'results':results,'new_fits':0,'scope':'Independent deployment laws on every serialized allowed person, all INNER reference utility rows/point gates, FIT actual-tuple Bayes/change and 144 fixed-reader guards per policy. LP solver certificates are separately replayed; numerical fitting feasibility tolerance does not change scientific caps.'}
    with Path(cfg['output']).open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'PASS','policies':len(results)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);run(p.parse_args().config)
