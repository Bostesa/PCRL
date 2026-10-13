"""Finite engineering timing of the exact18reader pair slate; no Adult inputs."""
import argparse,json,time,warnings
from pathlib import Path

def run(spec):
 from .evaluation_worker import queue_config
 active,cfg,verify,guard=queue_config(spec,'admission')
 import numpy as np
 from .readers import tables,kind_for
 from .reader_specs import BASE_READERS,EXTENDED_READERS
 from task_conditioned_representation.attacks import fit_reader,attack_split,MODEL_SEED_BASE
 from .source import write_json
 rng=np.random.default_rng(2026101309);n=4064;groups=np.r_[np.arange(4061),np.arange(3)];train,val=attack_split(groups)
 z=[rng.normal(size=(n,16)) for _ in range(2)];y=(z[0][:,0]+z[1][:,0]>.0).astype(int);w=[]
 for i,k in enumerate((2,6)):
  logits=z[i]@rng.normal(size=(16,k));logits-=logits.mean(1,keepdims=True);p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True);w.append(np.column_stack((z[i],logits,p,p.argmax(1))))
 tab=tables(w[0],w[1]);records=[]
 for name in BASE_READERS+EXTENDED_READERS:
  guard();x=tab['pair'][kind_for(name)][train];c=time.process_time();t=time.perf_counter()
  with warnings.catch_warnings(record=True) as ws:
   warnings.simplefilter('always');model=fit_reader(x,y[train],name,MODEL_SEED_BASE)
  records.append({'reader':name,'architecture':'MLP' if 'MLP' in name else 'HGB' if 'HGB' in name else 'LR','rows':len(x),'columns':x.shape[1],'CPU_s':time.process_time()-c,'wall_s':time.perf_counter()-t,'warning_count':len(ws)})
  del model
 out={'status':'PASS','unit_id':active['unit_id'],'finite_engineering_fit_count':18,'Adult_arrays_opened':False,'actual_reader_slate':records,'forecast_main_reader_CPU_s':sum(max(r['CPU_s'] for r in records if r['architecture']==a)*444 for a in ['LR','MLP','HGB']),'forecast_is_not_measured_Adult_runtime':True};write_json(cfg['public_output'],out);return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();print(json.dumps(run(a.spec)))
