"""Queued descriptive summaries of the already frozen training logs.

This reports diagnostics only: no additional fit, selection or inference.
"""
from __future__ import annotations
import argparse,csv,json,math,statistics
from pathlib import Path

def run(spec):
 from .evaluation_worker import queue_config
 active,cfg,verify,guard=queue_config(spec,'verification');rows=[];accounting=[]
 for item in cfg['logs']:
  events=[json.loads(x) for x in verify(item['log']['path'],item['log']['sha256']).read_text().splitlines()]
  if len(events)!=800:raise ValueError('complete800proposal log required')
  for epoch in range(1,21):
   steps=[e for e in events if e['epoch']==epoch]
   if len(steps)!=40:raise ValueError('40 proposals per epoch required')
   values={}
   def add(k,v):
    if v is not None:
     if not math.isfinite(v):raise ValueError('nonfinite frozen diagnostic')
     values.setdefault(k,[]).append(float(v))
   for e in steps:
    add('actual_displacement_norm',e['actual_displacement_norm'])
    for k,v in e['all_gradient_summary']['norms'].items():add('gradient_norm/'+k,v)
    for k,v in e['all_gradient_summary']['pairs'].items():
     add('gradient_dot/'+k,v['dot']);add('gradient_cosine/'+k,v['cosine'])
    for k,v in e['baseline'].items():
     add('batch_before/'+k,v)
     final=e['attempts'][-1]['metrics'][k] if e['status']=='ACCEPTED' else v
     add('batch_after/'+k,final);add('batch_delta/'+k,final-v)
    for bank, losses in zip(('occupation','pair'),e['critic_ces']):
     for reader,v in zip(('linear','MLP32','MLP64','constant_entropy'),losses):add('critic_CE/'+bank+'/'+reader,v)
   for name,vs in sorted(values.items()):rows.append({'arm':item['arm'],'seed':item['seed'],'epoch':epoch,'diagnostic':name,'observations':len(vs),'mean':statistics.fmean(vs),'minimum':min(vs),'maximum':max(vs),'negative_fraction':sum(v<0 for v in vs)/len(vs),'positive_fraction':sum(v>0 for v in vs)/len(vs)})
   accounting.append({'arm':item['arm'],'seed':item['seed'],'epoch':epoch,'proposals':40,'accepted_nonzero':sum(e['status']=='ACCEPTED' for e in steps),'zero':sum(e['actual_displacement_norm']==0 for e in steps)})
  guard()
 p=Path(cfg['public_output']);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 result={'schema':'CIS1-descriptive-training-diagnostics-v1','status':'PASS','proposal_count':6400,'epoch_groups':160,'diagnostic_rows':len(rows),'scope':'stored log descriptive aggregates; no new endpoints, fitting or post-freeze selection','source_log_sha256':[e['log']['sha256'] for e in cfg['logs']],'accounting':accounting}
 from .source import write_json
 write_json(cfg['receipt_output'],result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();r=run(a.spec);print(json.dumps({'status':r['status'],'diagnostic_rows':r['diagnostic_rows']}))
