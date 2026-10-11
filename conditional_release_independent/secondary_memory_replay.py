"""Independent low-memory template replay; same law and every full column."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import numpy as np
from scipy import sparse
from .reconstruction import admitted_arrays,digest,folds,merge_rows
from .policy_laws import law_from_numbers
from .policy_replay import source_numbers,cost_tables
from .guard_replay import source_guard_tables
from .primary_replay import matrix,load_archive,certify
from .arithmetic import bayes_objective,change_rate
from .contexts import context_cells


def reconstruct(source,seed,metadata,context_mapping):
    numbers,_=source_numbers(source,seed);entry=source['seeds'][str(seed)]['roles']['OSF_DEFENSE_FIT']['release']
    old=admitted_arrays(entry,{'row_id','unit','code_r1','code_r2','prob_r1','prob_r2','decision_r1','decision_r2'})
    sex=admitted_arrays(source['roles']['OSF_DEFENSE_FIT']['sex'],{'row_id','unit','sex'});n=len(old['unit']);fold=folds(old['unit']);ctx=context_mapping[context_cells(old['prob_r1'],old['prob_r2'])]
    if not all(np.array_equal(old[k],sex[k]) for k in ('row_id','unit')):raise ValueError('secondary SEX pairing')
    p=metadata['parameters'];rec={'metadata':metadata};pkg={'theta':np.zeros(p),'context_mapping':context_mapping}
    a,b,base=law_from_numbers(numbers,old['code_r1'],old['code_r2'],'TEMPLATE',rec,pkg);params=np.full(a.shape,-1,int);bounds=np.zeros((p,2));eq=np.zeros((len(metadata['contexts']),p))
    for d in metadata['contexts']:
        selected=np.flatnonzero(ctx==d['context']);count=d['action_count'];offset=d['parameter_offset'];l=np.asarray(d['recipient1_templates']);r=np.asarray(d['recipient2_templates'])
        if count!=len(l)*len(r):raise ValueError('template Cartesian count')
        if metadata['support']=='WIDE':
            for recipient,axis in ((1,l),(2,r)):
                expected=np.r_[-1,np.flatnonzero(numbers['probability'+str(recipient)].argmax(1)==d['class_pair'][recipient-1])]
                if not np.array_equal(axis,expected):raise ValueError('WIDE omitted permitted code')
            allowed=np.ones(count)
        else:
            if not np.array_equal(l,np.arange(4)) or not np.array_equal(r,np.arange(4)):raise ValueError('NARROW axis')
            allowed=(numbers['valid1'][old['code_r1'][selected]][:,np.repeat(l,len(r))]&numbers['valid2'][old['code_r2'][selected]][:,np.tile(r,len(l))]).all(0)
        params[selected,:count]=np.arange(offset,offset+count);bounds[offset:offset+count,1]=allowed;eq[d['context'],offset:offset+count]=1
    labels=[]
    for name in ('income','occupation'):
        z=admitted_arrays(source['roles']['OSF_DEFENSE_FIT'][name],{'row_id','unit',name})
        if not all(np.array_equal(z[k],old[k]) for k in ('row_id','unit')):raise ValueError('secondary task pairing')
        labels.append(z[name])
    refs=admitted_arrays(source['roles']['OSF_DEFENSE_FIT'][f'references_s{seed}'],{'row_id','unit','U0_1','U0_2','Ucal_1','Ucal_2','d_1','d_2'})
    guards=source_guard_tables(source,seed);views=[('pooled',np.ones(n,bool))]+[(f'fold{f}',fold==f) for f in range(3)]
    def coeff(loss,mask):
        w=loss[mask]/int(mask.sum());pp=params[mask];valid=pp>=0
        return np.bincount(pp[valid],weights=w[valid],minlength=p)
    lookup={};references={}
    for recipient,codes,y in ((1,a,labels[0]),(2,b,labels[1])):
        for metric,t in cost_tables(numbers['probability'+str(recipient)]).items():lookup[f'r{recipient}_{metric}']=(t,y,codes)
        for ref in ('U0','Ucal'):
            prob=refs[f'{ref}_{recipient}'];references[f'{ref}_r{recipient}_accuracy']=-(prob.argmax(1)==y).astype(float);references[f'{ref}_r{recipient}_nll']=-np.log(np.maximum(prob[np.arange(n),y],1e-12));references[f'{ref}_r{recipient}_brier']=(prob*prob).sum(1)-2*prob[np.arange(n),y]+1
    def cost_items():
        for key,(t,y,codes) in lookup.items():yield key,t[y[:,None],codes]
    programs={}
    for budget in ('STRICT','ORIGINAL'):
        rows=[];rhs=[];names=[]
        for view,mask in views:
            for key,loss in cost_items():
                metric=key.rsplit('_',1)[1];slope=coeff(loss-loss[:,0,None],mask)
                for ref in (('P0',) if budget=='STRICT' else ('U0','Ucal')):
                    cap=({'accuracy':.002,'nll':.001,'brier':.0005} if budget=='STRICT' else {'accuracy':.01,'nll':.01,'brier':.005})[metric]
                    excess=0 if ref=='P0' else np.mean(loss[mask,0]-references[ref+'_'+key][mask]);rows.append(slope);rhs.append(cap-excess);names.append(f'{budget}/{view}/{key}/{ref}')
            for name,g in guards.items():
                codes=a if g['view']=='r1' else b;loss=g['table'][fold[:,None],sex['sex'][:,None],codes];rows.append(-coeff(loss-loss[:,0,None],mask));rhs.append(1e-10);names.append(f'CE_GUARD/{view}/{name}')
        programs[budget]={'matrix':sparse.csr_matrix(rows),'rhs':np.asarray(rhs),'names':names}
    mincost=np.zeros(a.shape)
    for key,loss in cost_items():
        if key.endswith('_nll'):mincost+=loss
    minimum=coeff(mincost,np.ones(n,bool));minimum_constant=float(np.mean((base*mincost).sum(1)));del mincost,loss
    changed=(a!=old['code_r1'][:,None])|(b!=old['code_r2'][:,None]);maximum=-coeff(changed.astype(float),np.ones(n,bool))
    # Sorted actual tuple epigraph incidence against every shared template.
    projections=[];offsets=[];weights=[];zs=[];valid=params.ravel()>=0
    factor=int(b.max())+1
    if int(a.max())*factor+int(b.max())>np.iinfo(np.int64).max:raise OverflowError('independent tuple encoding')
    for view,weight in (('pair',1.),('r1',.5),('r2',.5)):
        values=(a*factor+b).ravel() if view=='pair' else (a if view=='r1' else b).ravel()
        unique,inv=np.unique(values,return_inverse=True);z=len(unique);zs.append(z);weights.extend([weight]*z);del unique,values
        for label in (0,1):
            rowmass=np.repeat((sex['sex']==label)/n,a.shape[1]);use=valid&(rowmass>0)
            projection=sparse.coo_matrix((rowmass[use],(inv[use],params.ravel()[use])),shape=(z,p)).tocsr();projections.append(projection)
            offsets.append(np.bincount(inv,weights=base.ravel()*rowmass,minlength=z))
        del inv,rowmass,use,projection
    matrices=[];cursor=0;nt=len(weights)
    for idx,z in enumerate(zs):
        negative=sparse.coo_matrix((-np.ones(z),(np.arange(z),cursor+np.arange(z))),shape=(z,nt)).tocsr()
        for projection in projections[2*idx:2*idx+2]:matrices.append(sparse.hstack((projection,negative)))
        cursor+=z
    return {'old':old,'sex':sex['sex'],'a':a,'b':b,'base':base,'params':params,'bounds':bounds,'eq':sparse.csr_matrix(eq),'programs':programs,'minimum':minimum,'minimum_constant':minimum_constant,'maximum':maximum,'epi':sparse.vstack(matrices).tocsr(),'epirhs':-np.concatenate(offsets),'epicost':np.r_[np.zeros(p),weights]}


def run(config):
    from conditional_release.scheduler import assert_active_from_env
    active=assert_active_from_env(stage='verification');state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text());pins={e['path']:e['sha256'] for e in state['units'][active['unit_id']]['spec']['sources']}
    def check(e):
        if pins.get(e['path'])!=e['sha256'] or digest(e['path'])!=e['sha256']:raise PermissionError('unqueued secondary replay')
    if pins.get(str(Path(config).absolute()))!=digest(config):raise PermissionError('unqueued secondary config')
    cfg=json.loads(Path(config).read_text());check(cfg['manifest']);check(cfg['receipt']);source=json.loads(Path(cfg['manifest']['path']).read_text());report=json.loads(Path(cfg['receipt']['path']).read_text());prep=admitted_arrays(cfg['prep'],{'context_mapping','raw_fit_counts','merged_fit_counts','contexts','donor','fold','code_r1','code_r2','b0_code_r1','b0_code_r2','valid'})
    calc=reconstruct(source,cfg['seed'],report['support_metadata'],prep['context_mapping']);p=len(calc['bounds']);outputs=[]
    for setup in report['setups']:
        program=calc['programs'][setup['budget']];m=program['matrix'];rhs=program['rhs'];slot={'arm_id':setup['arm_id'],'status':setup['status'],'solves':[]}
        items=[('minimum_nll',setup['witnesses']['minimum_nll']),('maximum_actual_change',setup['witnesses']['maximum_actual_change'])]
        if setup.get('infeasibility'):items.append(('phaseI',setup['infeasibility']))
        if setup.get('protection'):items.append(('protection',setup['protection']))
        for name,item in items:
            entry={'path':str(Path(cfg['private_root'])/item['archive_name']),'sha256':item['archive_sha256']};check(entry);arr=load_archive(entry);actual=matrix(arr,'inequality')
            if name=='phaseI':expected=sparse.hstack((sparse.vstack((m,calc['eq'],-calc['eq'])),-np.ones((len(rhs)+2*calc['eq'].shape[0],1))));expectedrhs=np.r_[rhs,np.ones(calc['eq'].shape[0]),-np.ones(calc['eq'].shape[0])]
            elif name=='protection':expected=sparse.vstack((sparse.hstack((m,sparse.csr_matrix((m.shape[0],len(calc['epicost'])-p)))),calc['epi']));expectedrhs=np.r_[rhs,calc['epirhs']]
            else:expected=m;expectedrhs=rhs
            gap=actual-expected
            if actual.shape!=expected.shape or np.max(np.abs(gap.data),initial=0)>1e-12 or np.max(np.abs(arr['inequality_rhs']-expectedrhs),initial=0)>1e-12:raise ValueError('complete secondary matrix/RHS differs')
            expectedcost={'minimum_nll':calc['minimum'],'maximum_actual_change':calc['maximum'],'phaseI':np.r_[np.zeros(p),1.],'protection':calc['epicost']}[name]
            if np.max(np.abs(arr['objective']-expectedcost),initial=0)>1e-12:raise ValueError('secondary objective formation')
            if name!='phaseI':
                eqtarget=calc['eq'] if name!='protection' else sparse.hstack((calc['eq'],sparse.csr_matrix((calc['eq'].shape[0],len(calc['epicost'])-p))))
                egap=matrix(arr,'equality')-eqtarget
                if np.max(np.abs(egap.data),initial=0)>1e-12 or not np.array_equal(arr['equality_rhs'],np.ones(calc['eq'].shape[0])) or not np.array_equal(arr['bounds'][:p],calc['bounds']):raise ValueError('secondary normalization/support bounds')
            record={'name':name,'success':item['success'],'matrix_RHS_objective_support':'PASS','archive_sha256':entry['sha256']}
            if item['success']:
                cert=certify(arr,item['certificate']);record['certificate']=cert
                if name=='phaseI':
                    passed=cert['dual_objective']>1e-7
                    if passed!=item['infeasibility_certified']:raise ValueError('secondary finite infeasibility')
                    record['finite_family_infeasibility_certified']=passed
                else:
                    theta=arr['theta'];q=calc['base'].copy();valid=calc['params']>=0;q[valid]+=theta[calc['params'][valid]];rows=merge_rows(calc['a'],calc['b'],q)
                    if name=='protection':
                        objective=bayes_objective(rows,calc['sex'])
                        if abs(objective-cert['objective'])>1e-7 or abs(objective-setup['fitted_objective'])>1e-7:raise ValueError('secondary actual tuple objective')
                    if name=='maximum_actual_change':
                        rate=change_rate(rows,np.c_[calc['old']['code_r1'],calc['old']['code_r2']])
                        if abs(rate+cert['objective'])>1e-10:raise ValueError('secondary change alias accounting')
                    del rows,q
            slot['solves'].append(record)
        outputs.append(slot)
    result={'schema':'CBR1-independent-secondary-LP-replay-v1','status':'PASS','unit_id':active['unit_id'],'seed':cfg['seed'],'support':report['support'],'setups':outputs,'parameter_count':p,'all_declared_support_columns':True,'new_optimizer_fits':0,'INNER_SEX_access':False,'scope':'Independent complete template constraints/epigraph/objectives, support/equality bounds, every successful witness/protection certificate and Phase-I finite infeasibility. No population or held-out privacy claim.'}
    with Path(cfg['output']).open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'PASS','seed':cfg['seed'],'support':report['support']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);run(p.parse_args().config)
