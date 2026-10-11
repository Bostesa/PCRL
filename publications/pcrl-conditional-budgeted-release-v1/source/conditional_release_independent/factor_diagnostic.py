"""Independent archived simplex arithmetic, no scientific labels or fitting."""
import argparse,json,os
from pathlib import Path
import numpy as np
from .reconstruction import digest
from .primary_replay import load_archive,matrix
from .arithmetic import lp_certificate


def run(config):
    from conditional_release.scheduler import assert_active_from_env
    active=assert_active_from_env(stage='verification');state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text());pins={x['path']:x['sha256'] for x in state['units'][active['unit_id']]['spec']['sources']}
    if pins.get(str(Path(config).absolute()))!=digest(config):raise PermissionError('unregistered factor diagnostic')
    cfg=json.loads(Path(config).read_text());results=[]
    for e in cfg['archives']:
        if pins.get(e['path'])!=e['sha256']:raise PermissionError('unregistered factor archive')
        a=load_archive(e)
        if 'theta' not in a:
            results.append({'archive':Path(e['path']).name,'primal_available':False});continue
        theta=a['theta'];count=len(a['equality_rhs'])
        if count<1 or len(theta)%count:raise ValueError('factor simplex count/width differs')
        f=theta.reshape(count,-1);cert=None
        if all(k in a for k in ('inequality_dual','equality_dual','lower_dual','upper_dual')):
            cert=lp_certificate(a['objective'],a['primal'],matrix(a,'inequality'),a['inequality_rhs'],matrix(a,'equality'),a['equality_rhs'],a['bounds'][:,0],a['bounds'][:,1],a['inequality_dual'],a['equality_dual'],a['lower_dual'],a['upper_dual'])
        results.append({'archive':Path(e['path']).name,'primal_available':True,'groups':count,'width':f.shape[1],'minimum_weight':float(f.min()),'maximum_weight':float(f.max()),'negative_weight_count':int((f<0).sum()),'weights_below_minus_1e12':int((f < -1e-12).sum()),'maximum_row_sum_error':float(np.max(np.abs(f.sum(1)-1))),'maximum_bound_violation':float(max(0,np.max(a['bounds'][:len(theta),0]-theta),np.max(theta-a['bounds'][:len(theta),1]))),'objective':float(a['objective']@a['primal']),'independent_LP_certificate':cert})
    report={'schema':'CBR1-independent-simplex-numerical-diagnostic-v1','status':'PASS','unit_id':active['unit_id'],'results':results,'scientific_labels_read':False,'new_optimizer_fits':0,'scope':'Exact saved simplex weights/normalization and archived LP primal-dual arithmetic. PASS means diagnostic executed correctly; it does not admit an invalid probability law or certify a family infeasible.'}
    with Path(cfg['output']).open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'PASS','archives':len(results)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);run(p.parse_args().config)
