"""Queue-owned independent raw83 hybrid forwards and literal alias audit."""
from __future__ import annotations
import argparse,json
from pathlib import Path


def audit(inputs):
    import numpy as np
    from .replay import torch_load,forward,geometry,folds,close
    cfg=inputs.cfg;m=inputs.json(cfg['manifest']);features={}
    for role in ('OSF_DEFENSE_FIT','ATTACK_FIT_NEW','INNER_SELECTION'):
        features[role]=inputs.arrays(m['roles'][role]['features'],{'row_id','unit','X','feature_names'})
    records=[];cache={};frozen=[]
    for package in cfg['packages']:
        payload=torch_load(inputs.path(package['checkpoint']));names=payload['feature_names']
        contract=package['wire_contract'];binding=inputs.json(package['geometry_binding']) if package.get('geometry_binding') else payload.get('wire_binding')
        if payload.get('wire_binding') and payload['wire_binding']!=binding:
            raise AssertionError('checkpoint fixed basis differs from admitted geometry')
        if payload['seed']!=package['seed']:raise ValueError('source seed differs')
        if package.get('frozen_source_checkpoint'):
            source=torch_load(inputs.path(package['frozen_source_checkpoint']))
            for key,value in payload['model'].items():
                if key.startswith(('enc.0.','head.0.')):
                    original=source['model'][key]
                    if value.dtype!=original.dtype or value.shape!=original.shape or not np.array_equal(value.numpy().view(np.uint8),original.numpy().view(np.uint8)):
                        raise AssertionError('income tensor bytes changed')
            if source.get('wire_binding') and payload.get('wire_binding')!=source['wire_binding']:
                raise AssertionError('fixed source geometry binding changed')
            frozen.append({'arm_id':package['arm_id'],'seed':package['seed'],'income_tensor_bytes':'PASS','fixed_basis':'PASS'})
        for role,key in [('OSF_DEFENSE_FIT','fit_release'),('ATTACK_FIT_NEW','att_release'),('INNER_SELECTION','inner_release')]:
            f=features[role];ix=np.flatnonzero(folds(f['unit'])==2) if role=='OSF_DEFENSE_FIT' else np.arange(len(f['unit']))
            if tuple(map(str,f['feature_names']))!=tuple(names):raise AssertionError('checkpoint feature order differs')
            arrays=inputs.arrays(package[key]);expected={name:arrays[name] for name in ('row_id','unit','r1','r2')}
            if any(not np.array_equal(expected[k],f[k][ix]) for k in ('row_id','unit')):raise AssertionError('actual checkpoint and frozen release people differ')
            # Canonical batching is on the whole role, then FIT2 is sliced.
            # Slice-first float32 GEMM can change low bits and break aliases.
            actual=[a[ix] for a in forward(payload,f['X'],wire_contract=contract,geometry_binding=binding)]
            raw=[a[ix] for a in forward(payload,f['X'])];errors=[];checks=[]
            for i,task in enumerate(('income','occupation')):
                if contract[task]=='head_row_space':
                    w=payload['model'][f'head.{i}.weight'].numpy();b=payload['model'][f'head.{i}.bias'].numpy()
                    g=geometry(w,b,raw[i][:,:16],cfg['rule']);close(g['B'],binding['heads'][i]['B'],'independent head basis',1e-10)
                    if g['decision_mismatch_rows'] or g['AP_max_absolute_residual']>1e-10 or g['max_absolute_probability_difference']>1e-10:
                        raise AssertionError('projected own-head parity')
                    close(actual[i][:,:16],g['projected_z'],'factorized projection',1e-12)
                    checks.append({k:v for k,v in g.items() if k not in ('B','P','projected_z')})
                reference=expected[f'r{i+1}'];close(actual[i][:,:-1],reference[:,:-1],'independent actual checkpoint wire',2e-12)
                if not np.array_equal(actual[i][:,-1],reference[:,-1]):raise AssertionError('actual decision mismatch')
                errors.append(float(abs(actual[i]-reference).max()))
                cache[package['arm_id'],package['seed'],role,f'r{i+1}']=actual[i]
            records.append({'arm_id':package['arm_id'],'seed':package['seed'],'wire_contract':contract,'role':role,'rows':len(ix),
                            'checkpoint_sha256':package['checkpoint']['sha256'],'max_absolute_wire_difference':errors,
                            'projection_checks':checks,'decisions_exact':True,'status':'PASS'})
    aliases=[]
    for alias in cfg.get('aliases',[]):
        seed=alias['seed'];view=alias['view']
        for role in ('OSF_DEFENSE_FIT','ATTACK_FIT_NEW','INNER_SELECTION'):
            left=cache[alias['left'],seed,role,view];right=cache[alias['right'],seed,role,view]
            if not np.array_equal(left,right):raise AssertionError('declared literal wire alias differs')
            aliases.append({**alias,'role':role,'literal_wire_equality':'PASS'})
    return {'schema':'CIS1-independent-role-forwards-v1','status':'PASS','scientific_runner_imports':False,
            'forward_batch_size':1024,'package_count':len(cfg['packages']),'role_forward_count':len(records),
            'records':records,'frozen_checks':frozen,'aliases':aliases,
            'true_label_arrays_materialized':False,'INNER_label_bytes_accessed':False}


def main():
    from .replay import Inputs
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);args=p.parse_args();inputs=Inputs(args.spec)
    result=audit(inputs);out=Path(inputs.cfg['public_output']);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':'PASS','role_forward_count':result['role_forward_count']}))

if __name__=='__main__':main()
