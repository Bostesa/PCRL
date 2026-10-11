"""Independent all primary G/C witness, protection and Phase-I replay."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from scipy import sparse
from .reconstruction import digest,admitted_arrays,merge_rows
from .preparation_replay import constraint_reconstruction
from .arithmetic import lp_certificate,bayes_objective,change_rate,validate_group_pairing


def load_archive(entry):
    if digest(entry['path'])!=entry['sha256']:raise PermissionError('solve archive hash')
    with ZipFile(entry['path']) as z:
        names={s[:-4] for s in z.namelist() if s.endswith('.npy')}
        if len(names)!=len(z.namelist()):raise PermissionError('unexpected solve archive member')
    return admitted_arrays(entry,names)


def matrix(arrays,prefix):
    return sparse.csr_matrix((arrays[prefix+'_data'],arrays[prefix+'_indices'],arrays[prefix+'_indptr']),shape=tuple(arrays[prefix+'_shape']))


def certify(arrays,reported,tolerance=1e-7):
    result=lp_certificate(arrays['objective'],arrays['primal'],matrix(arrays,'inequality'),arrays['inequality_rhs'],matrix(arrays,'equality'),arrays['equality_rhs'],arrays['bounds'][:,0],arrays['bounds'][:,1],arrays['inequality_dual'],arrays['equality_dual'],arrays['lower_dual'],arrays['upper_dual'])
    residuals=[result[k] for k in ('max_inequality_violation','max_equality_residual','max_bound_violation','stationarity_residual','dual_sign_violation','unbounded_bound_dual_residual','complementarity_residual')]+[abs(result['gap'])]
    if max(residuals)>tolerance:raise ValueError('independent LP residual exceeds numerical tolerance')
    if abs(result['gap']-reported['primal_dual_gap'])>tolerance:raise ValueError('LP reported gap differs')
    return result


def run(config):
    from conditional_release.scheduler import assert_active_from_env
    active=assert_active_from_env(stage='verification')
    state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text())
    registered={i['path']:i['sha256'] for i in state['units'][active['unit_id']]['spec']['sources']}
    if registered.get(str(Path(config).absolute()))!=digest(config):raise PermissionError('unqueued independent config')
    cfg=json.loads(Path(config).read_text())
    for entry in (cfg['manifest'],cfg['prep'],cfg['receipt']):
        if registered.get(entry['path'])!=entry['sha256'] or digest(entry['path'])!=entry['sha256']:raise PermissionError('unqueued replay input')
    if not any('science' in Path(a['path']).name.lower() for a in state['amendments']):raise PermissionError('science lock missing before FIT SEX audit')
    source=json.loads(Path(cfg['manifest']['path']).read_text())
    prep=admitted_arrays(cfg['prep'],{'context_mapping','raw_fit_counts','merged_fit_counts','contexts','donor','fold','code_r1','code_r2','b0_code_r1','b0_code_r2','valid'})
    independent=constraint_reconstruction(source,cfg['seed'],prep)
    rec=json.loads(Path(cfg['receipt']['path']).read_text());reports=[];objective_gaps=[]
    sex=admitted_arrays(source['roles']['OSF_DEFENSE_FIT']['sex'],{'row_id','unit','sex'})
    if not all(np.array_equal(sex[k],independent['wire'][k]) for k in ('row_id','unit')):raise ValueError('FIT SEX pairing')
    validate_group_pairing(sex['unit'],sex['sex'])
    identity=np.column_stack((prep['b0_code_r1'],prep['b0_code_r2']))
    for setup in rec['setups']:
        mode,budget=setup['mode'],setup['budget'];constraints=independent['constraints'][f'{mode}/{budget}'];nparam=1 if mode=='scalar' else len(independent['hierarchy']['labels'])
        local={'mode':mode,'budget':budget,'status':setup['status'],'solves':[]}
        records=[('minimum_nll',setup['witnesses']['minimum_nll']),('maximum_actual_change',setup['witnesses']['maximum_actual_change'])]
        if setup.get('infeasibility'):records.append(('phaseI',setup['infeasibility']))
        if setup.get('protection'):records.append(('protection',setup['protection']))
        for name,item in records:
            path=Path(cfg['primary_private'])/item['archive_name'];entry={'path':str(path),'sha256':item['archive_sha256']}
            if registered.get(str(path))!=entry['sha256']:raise PermissionError('solve archive not queued')
            data=load_archive(entry);m=matrix(data,'inequality')
            size=len(constraints['names'])
            expected_matrix=constraints['matrix'];expected_rhs=constraints['rhs']
            if name=='phaseI':
                target=np.column_stack((expected_matrix,-np.ones(size)))
                if m.shape!=target.shape or np.max(np.abs(m.toarray()-target),initial=0)>1e-12:raise ValueError('PhaseI matrix formation')
            elif np.max(np.abs(m[:size,:nparam].toarray()-expected_matrix),initial=0)>1e-12:raise ValueError('independent complete utility/guard matrix mismatch')
            if np.max(np.abs(data['inequality_rhs'][:size]-expected_rhs),initial=0)>1e-12:raise ValueError('independent complete budget RHS mismatch')
            slot={'name':name,'success':item['success'],'archive_sha256':entry['sha256'],'matrix_and_RHS':'PASS'}
            if item['success']:
                cert=certify(data,item['certificate']);slot['certificate']=cert
                if name=='phaseI':
                    expected_infeasible=cert['dual_objective']>1e-7
                    if item['infeasibility_certified']!=expected_infeasible:raise ValueError('PhaseI infeasibility verdict')
                    slot['finite_family_infeasibility_certified']=expected_infeasible
                else:
                    theta=data['theta'];param=np.zeros(len(prep['contexts']),int) if mode=='scalar' else np.maximum(prep['contexts'],0)
                    q=prep['donor']*theta[param,None];q[:,0]+=1-theta[param]
                    rows=merge_rows(prep['code_r1'],prep['code_r2'],q)
                    actualchange=change_rate(rows,identity)
                    if name=='maximum_actual_change':
                        if abs(-cert['objective']-actualchange)>1e-10:raise ValueError('maxchange excludes aliases incorrectly')
                        slot['actual_change']=actualchange
                    if name=='minimum_nll':
                        cost=independent['action_costs']['r1_nll']+independent['action_costs']['r2_nll']
                        expected_nll=float(np.mean(np.sum(q*cost,axis=1)))
                        if abs(cert['objective']+item['objective_constant']-expected_nll)>1e-10:raise ValueError('minimum actual action NLL objective')
                        slot['expected_summed_nll']=expected_nll
                    if name=='protection':
                        objective=bayes_objective(rows,sex['sex']);gap=abs(objective-cert['objective']);objective_gaps.append(gap)
                        if gap>1e-7 or abs(objective-setup['fitted_objective'])>1e-7:raise ValueError('actual public tuple recovery objective')
                        if abs(actualchange-setup['actual_change_rate'])>1e-10:raise ValueError('fitted change rate')
                        slot.update(actual_change=actualchange,fitted_objective=objective,direct_epigraph_gap=gap)
            local['solves'].append(slot)
        reports.append(local)
    paired={s['mode']+'/'+s['budget']:s for s in reports}
    nesting=[]
    for budget in ('STRICT','ORIGINAL'):
        scalar=next((x['fitted_objective'] for x in paired['scalar/'+budget]['solves'] if x['name']=='protection' and x['success']),None)
        conditional=next((x['fitted_objective'] for x in paired['conditional/'+budget]['solves'] if x['name']=='protection' and x['success']),None)
        if scalar is not None and conditional is not None:
            if conditional>scalar+2e-7:raise ValueError('nested conditional optimum worse beyond numerical gaps')
            nesting.append({'budget':budget,'scalar_objective':scalar,'conditional_objective':conditional,'same_budget_reduction':scalar-conditional})
    report={'schema':'CBR1-independent-primary-LP-replay-v1','status':'PASS','unit_id':active['unit_id'],'seed':cfg['seed'],'setups':reports,'nested_allocation_comparisons':nesting,'maximum_direct_objective_gap':max(objective_gaps,default=None),'INNER_SEX_access':False,'new_optimizer_fits':0,'scope':'All archived primary witness/protection/PhaseI matrix completeness, finite-law margins, actual tuple objectives/change and LP primal-dual replay. No held-out privacy conclusion or population guarantee.'}
    with Path(cfg['output']).open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'PASS','setups':len(reports),'solves':sum(len(s['solves']) for s in reports)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);run(parser.parse_args().config)
