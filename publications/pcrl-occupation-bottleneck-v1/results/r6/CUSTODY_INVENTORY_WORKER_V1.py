"""Queue-owned hashes of an explicit campaign-only restore inventory."""
from pathlib import Path
import argparse,json
from .evaluation_worker import queue_config
from .custody_worker import uncached_hash
from .source import write_json

def run(spec):
 active,cfg,verify,guard=queue_config(spec,'custody')
 root=Path(cfg['repo']).resolve();private=root/'.private/occupation_bottleneck'
 entries=[]
 for item in cfg['entries']:
  p=Path(item['path']).resolve();scope=item['scope']
  if scope=='campaign':
   if not p.is_relative_to(private) or 'custody' in p.relative_to(private).parts or 'compute' in p.relative_to(private).parts:raise PermissionError('new campaign immutable artifacts only')
  elif scope=='code':
   if not p.is_relative_to(root) or '.private' in p.relative_to(root).parts:raise PermissionError('public code only')
  elif scope=='admitted_source':verify(p,item['sha256'])
  else:raise PermissionError('scope required')
  entries.append(dict(item,sha256=uncached_hash(p),size_bytes=p.stat().st_size));guard()
 write_json(cfg['private_output'],{'entries':entries},private=True)
 out={'status':'PASS','file_count':len(entries),'bytes':sum(e['size_bytes'] for e in entries),'explicit_paths_only':True,'old_assessment_or_HEAD_labels_opened':False}
 write_json(cfg['public_output'],out);return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();print(json.dumps(run(a.spec)))
