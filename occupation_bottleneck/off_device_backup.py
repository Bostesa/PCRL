"""Bounded custody-only copy of an existing snapshot; never fits or opens arrays.

This pending operation has its own wall bound and uses the existing heavy-slot
lock. It does not reopen a closed scientific queue or authorize new experiments.
"""
import argparse,fcntl,json,os,time
from pathlib import Path
from .custody_worker import copy_exact,uncached_hash,_new_target,alias_path

def main():
 p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--receipt-sha256',required=True);p.add_argument('--target',required=True);p.add_argument('--queue-root',required=True);p.add_argument('--wall-seconds',type=int,default=600);a=p.parse_args()
 if not 1<=a.wall_seconds<=1200:raise ValueError('bounded custody wall limit required')
 import psutil
 others=[]
 for proc in psutil.process_iter(['pid','name','memory_info']):
  try:
   d=proc.info;n=(d['name'] or '').lower()
   if d['pid']!=os.getpid() and ('virtualmachine' in n or n.startswith(('python','julia','matlab','rscript')) and d['memory_info'].rss>=64*1024**2):others.append(d['pid'])
  except (psutil.AccessDenied,psutil.NoSuchProcess):pass
 if len(others)>1:raise RuntimeError('shared two-heavy-process capacity unavailable')
 start=time.monotonic();src=Path(a.snapshot).resolve();receipt=src/'SNAPSHOT_RECEIPT.json'
 if uncached_hash(receipt)!=a.receipt_sha256:raise ValueError('snapshot receipt pin differs')
 r=json.loads(receipt.read_text());plan=src/'PLAN.private.json'
 if uncached_hash(plan)!=r['plan_sha256']:raise ValueError('snapshot plan pin differs')
 target=Path(a.target);ancestor=target.parent
 while not ancestor.exists():ancestor=ancestor.parent
 if ancestor.stat().st_dev==src.stat().st_dev:raise ValueError('target must be a different writable mounted filesystem')
 total=sum(e['size_bytes'] for e in r['entries'])+plan.stat().st_size+receipt.stat().st_size
 with (Path(a.queue_root)/'slot0.lock').open('a+') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB);dest=_new_target(target,total)
  for e in r['entries']:
   if time.monotonic()-start>a.wall_seconds:raise TimeoutError('custody bound reached; partial copy preserved')
   rel=alias_path(e['alias']);copy_exact(src/'files'/rel,dest/'files'/rel,e['sha256'],e['size_bytes'])
  copy_exact(plan,dest/'PLAN.private.json',r['plan_sha256'],plan.stat().st_size);copy_exact(receipt,dest/'SNAPSHOT_RECEIPT.json',a.receipt_sha256,receipt.stat().st_size)
  result={'status':'PASS','operation':'off_device_copy','file_count':len(r['entries']),'uncached_full_readback':True,'different_filesystem':True,'originals_intact':True,'scientific_fits':0,'wall_s':time.monotonic()-start,'representative_numerical_restores':'PENDING'}
  (dest/'OFF_DEVICE_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
