"""Queue adapter for R7's independent functional checkpoint/view arithmetic.

This adapter supplies formats only; scientific forward/SVD functions are the
separate verifier implementation. No candidate or runner science is imported.
"""
import argparse,json
from pathlib import Path

def main():
 from .replay import Inputs,torch_load,forward,geometry,folds,close
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);args=p.parse_args();inputs=Inputs(args.spec)
 import numpy as np
 cfg=inputs.cfg;m=inputs.json(cfg['manifest']);features={}
 for role in ('OSF_DEFENSE_FIT','ATTACK_FIT_NEW','INNER_SELECTION'):
  features[role]=inputs.arrays(m['roles'][role]['features'],{'row_id','unit','X','feature_names'})
 records=[]
 for package in cfg['packages']:
  payload=torch_load(inputs.path(package['checkpoint']));names=payload['feature_names']
  for role,key in [('OSF_DEFENSE_FIT','fit_release'),('ATTACK_FIT_NEW','att_release'),('INNER_SELECTION','inner_release')]:
   f=features[role];ix=np.flatnonzero(folds(f['unit'])==2) if role=='OSF_DEFENSE_FIT' else np.arange(len(f['unit']))
   if tuple(map(str,f['feature_names']))!=tuple(names):raise AssertionError('checkpoint feature order differs')
   with np.load(inputs.path(package[key]),allow_pickle=False) as archive:
    if not {'row_id','unit','r1','r2'}<=set(archive.files):raise AssertionError('frozen release fields missing')
    expected={name:archive[name].copy() for name in ('row_id','unit','r1','r2')}
   if any(not np.array_equal(expected[k],f[k][ix]) for k in ('row_id','unit')):raise AssertionError('actual checkpoint and frozen release people differ')
   actual=forward(payload,f['X'][ix]);errors=[]
   if package['release_variant']=='head_row_space':
    for i,k in enumerate((2,6)):
     w=payload['model'][f'head.{i}.weight'].numpy();b=payload['model'][f'head.{i}.bias'].numpy();g=geometry(w,b,actual[i][:,:16],cfg['rule']);z=g['projected_z']
     logits=z@w.T+b;logits-=logits.mean(1,keepdims=True);e=np.exp(logits-logits.max(1,keepdims=True));prob=e/e.sum(1,keepdims=True);actual[i]=np.column_stack((z,logits,prob,prob.argmax(1)))
   for i,k in enumerate((2,6)):
    reference=expected[f'r{i+1}'];close(actual[i][:,:-1],reference[:,:-1],'independent actual checkpoint complete wire',2e-12)
    if not np.array_equal(actual[i][:,-1],reference[:,-1]):raise AssertionError('actual checkpoint decision mismatch')
    errors.append(float(abs(actual[i]-reference).max()))
   records.append({'arm_id':package['arm_id'],'seed':package['seed'],'variant':package['release_variant'],'role':role,'rows':len(ix),'checkpoint_sha256':package['checkpoint']['sha256'],'max_absolute_wire_difference':errors,'decisions_exact':True,'status':'PASS'})
 if len(records)!=216:raise AssertionError('all72packages three roles required')
 result={'status':'PASS','scientific_runner_imports':False,'independent_math_implementation':'R7 forward/geometry','adapter_owner':'ROOT formats only','package_count':72,'role_forward_count':216,'encoder_forward_count':432,'true_label_arrays_materialized':False,'authorized_ATT_SEX_archive_bytes_hashed':True,'INNER_label_bytes_accessed':False,'records':records}
 with Path(cfg['public_output']).open('x') as out:json.dump(result,out,sort_keys=True,indent=2);out.write('\n')
 print(json.dumps({'status':'PASS','actual_checkpoint_release_replays':216}))
if __name__=='__main__':main()
