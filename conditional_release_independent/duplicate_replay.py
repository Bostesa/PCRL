"""Independent canonical-person raw input equality for cached deployment."""
import argparse,json,os
from pathlib import Path
import numpy as np
from .reconstruction import digest,admitted_arrays


def run(config):
    from conditional_release.scheduler import assert_active_from_env
    active=assert_active_from_env(stage='verification');state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text());pins={e['path']:e['sha256'] for e in state['units'][active['unit_id']]['spec']['sources']}
    def check(e):
        if pins.get(e['path'])!=e['sha256'] or digest(e['path'])!=e['sha256']:raise PermissionError('unregistered duplicate input')
    if pins.get(str(Path(config).absolute()))!=digest(config):raise PermissionError('unregistered duplicate config')
    cfg=json.loads(Path(config).read_text());check(cfg['manifest']);m=json.loads(Path(cfg['manifest']['path']).read_text());results=[]
    for role in ('OSF_DEFENSE_FIT','ATTACK_FIT_NEW','INNER_SELECTION'):
        e=m['roles'][role]['features'];check(e);raw=admitted_arrays(e,{'row_id','unit','X','feature_names'});groups,counts=np.unique(raw['unit'],return_counts=True);duplicate=groups[counts>1]
        for g in duplicate:
            ix=np.flatnonzero(raw['unit']==g);x=np.asarray(raw['X'][ix],np.float32)
            if not np.array_equal(x,np.repeat(x[:1],len(x),axis=0)):raise ValueError('one canonical person has different cached raw83 input')
        for seed in (0,1):
            e=m['seeds'][str(seed)]['roles'][role]['release'];check(e);wire=admitted_arrays(e,{'row_id','unit','code_r1','code_r2','prob_r1','prob_r2','decision_r1','decision_r2'})
            if not all(np.array_equal(raw[k],wire[k]) for k in ('row_id','unit')):raise ValueError('duplicate wire pairing differs')
            for g in duplicate:
                ix=np.flatnonzero(raw['unit']==g)
                for key in ('code_r1','code_r2','prob_r1','prob_r2','decision_r1','decision_r2'):
                    if not np.array_equal(wire[key][ix],np.repeat(wire[key][ix[:1]],len(ix),axis=0)):raise ValueError('canonical duplicate P0 wire differs')
        results.append({'role':role,'duplicate_groups':len(duplicate),'duplicate_rows':int(counts[counts>1].sum()),'raw83_float32_equality':'EXACT','both_seed_P0_wire_equality':'EXACT'})
    report={'schema':'CBR1-independent-duplicate-cache-admission-v1','status':'PASS','unit_id':active['unit_id'],'results':results,'scientific_labels_read':False,'scope':'Canonical duplicate raw83 inputs and both source seed P0 wire equal exactly, supporting cached one-provider-person deployment. SEX group consistency is checked separately on allowed FIT/ATT roles and after the global reporting freeze on INNER.'}
    with Path(cfg['output']).open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'status':'PASS','roles':len(results)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);run(p.parse_args().config)
