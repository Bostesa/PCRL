"""Exact admitted-file versioned custody, never historical directory inventory.

A plan has explicit files only. No directory traversal discovers person data,
no recursion enters earlier snapshots, and uncached verification reads every
copied byte. Numerical restoration is a separate authenticated queue unit.
"""
from __future__ import annotations
import argparse,fcntl,hashlib,json,os,shutil,sys
from pathlib import Path


def uncached_hash(path):
    from .admission import admitted_path
    p=admitted_path(path);digest=hashlib.sha256()
    with p.open('rb') as f:
        if sys.platform=='darwin':fcntl.fcntl(f.fileno(),fcntl.F_NOCACHE,1)
        for part in iter(lambda:f.read(1024**2),b''):digest.update(part)
    return digest.hexdigest()


def alias_path(alias):
    p=Path(alias)
    if not alias or p.is_absolute() or '..' in p.parts or '\\' in alias or p.as_posix()!=alias or p==Path('.'):raise PermissionError('unsafe relative custody alias')
    return p


def copy_exact(source,target,pin,size):
    from .admission import admitted_path
    source=admitted_path(source);target=Path(target)
    if source.stat().st_size!=size or uncached_hash(source)!=pin:raise PermissionError('copy source differs')
    if any(p.is_symlink() for p in (target,*target.parents)):raise PermissionError('copy target symlink refused')
    target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with source.open('rb') as src,os.fdopen(os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb') as dst:
        if sys.platform=='darwin':fcntl.fcntl(src.fileno(),fcntl.F_NOCACHE,1)
        shutil.copyfileobj(src,dst,1024**2);dst.flush();os.fsync(dst.fileno())
    if target.stat().st_size!=size or uncached_hash(target)!=pin:raise PermissionError('uncached copied bytes differ')


def _new_target(path,total):
    from .scheduler import DISK_FLOOR
    p=Path(path)
    if p.exists():raise FileExistsError('fresh versioned custody target required')
    parent=p.parent
    while not parent.exists():parent=parent.parent
    if any(q.is_symlink() for q in (parent,*parent.parents)):raise PermissionError('custody target symlink refused')
    if shutil.disk_usage(parent).free-total<DISK_FLOOR:raise OSError('custody would threaten15GiB free-disk floor')
    p.mkdir(parents=True,mode=0o700);return p


def _public_entries(entries):return [{k:e[k] for k in ('alias','sha256','size_bytes')} for e in entries]


def run(config):
    from .evaluation_worker import queue_config
    active,cfg,verify,guard=queue_config(config,'custody')
    from .source import write_json
    operation=cfg['operation']
    if operation=='plan':
        entries=[];seen=set();repo=Path(cfg['repo']).resolve();private=repo/'.private/occupation_bottleneck'
        for e in cfg['entries']:
            alias_path(e['alias']);p=Path(e['path']).resolve()
            if e['alias'] in seen:raise ValueError('duplicate custody alias')
            seen.add(e['alias']);verify(p,e['sha256'])
            scope=e['scope']
            if scope=='campaign':
                if not p.is_relative_to(private) or set(p.relative_to(private).parts)&{'custody','SNAPSHOT','RESTORED','restored'}:raise PermissionError('campaign custody excludes existing snapshots and restores')
            elif scope=='code':
                if not p.is_relative_to(repo) or '.private' in p.relative_to(repo).parts:raise PermissionError('code scope differs')
            elif scope!='admitted_source':raise PermissionError('explicit custody file scope required')
            # Already admitted exact source bytes only; no manifest recursion.
            pin=uncached_hash(p)
            if pin!=e['sha256']:raise PermissionError('uncached plan source differs')
            entries.append({'alias':e['alias'],'path':str(p),'sha256':pin,'size_bytes':p.stat().st_size,'scope':scope})
        plan={'schema':'OB1-exact-custody-plan-v1','status':'FROZEN','execution_commit':cfg['execution_commit'],'entries':entries,'scope':'only explicit new campaign artifacts, code closure, exact admitted source files; no historical walks or restore recursion','no_original_deletion':True}
        binding=write_json(cfg['plan_output'],plan,private=True)
        result={'schema':'OB1-custody-plan-receipt-v1','status':'PASS','plan_sha256':binding,'execution_commit':cfg['execution_commit'],'entries':_public_entries(entries),'file_count':len(entries),'bytes':sum(e['size_bytes'] for e in entries),'scope':plan['scope'],'uncached_source_hashes':True}
    elif operation in ('snapshot','restore','off_device'):
        if operation=='snapshot':
            plan=json.loads(verify(cfg['plan'],cfg['plan_sha256']).read_text());entries=plan['entries'];source_plan=cfg['plan'];source_plan_sha=cfg['plan_sha256'];receipt_pin=None
            sources={e['alias']:verify(e['path'],e['sha256']) for e in entries}
        else:
            root=Path(cfg['snapshot']);receipt=json.loads(verify(root/'SNAPSHOT_RECEIPT.json',cfg['snapshot_receipt_sha256']).read_text());source_plan=root/'PLAN.private.json';source_plan_sha=receipt['plan_sha256'];verify(source_plan,source_plan_sha);plan=json.loads(source_plan.read_text());entries=plan['entries'];receipt_pin=cfg['snapshot_receipt_sha256']
            sources={e['alias']:verify(root/'files'/alias_path(e['alias']),e['sha256']) for e in entries}
        if operation=='off_device':
            ancestor=Path(cfg['target']).parent
            while not ancestor.exists():ancestor=ancestor.parent
            if Path(cfg['snapshot']).stat().st_dev==ancestor.stat().st_dev:raise PermissionError('off-device target must be a different mounted filesystem')
        total=sum(e['size_bytes'] for e in entries)+Path(source_plan).stat().st_size
        target=_new_target(cfg['target'],total)
        same_device=Path(source_plan).stat().st_dev==target.stat().st_dev
        if operation=='snapshot' and not same_device:raise PermissionError('initial versioned snapshot must remain on same device')
        for e in entries:guard();copy_exact(sources[e['alias']],target/'files'/alias_path(e['alias']),e['sha256'],e['size_bytes'])
        copy_exact(source_plan,target/'PLAN.private.json',source_plan_sha,Path(source_plan).stat().st_size)
        result={'schema':'OB1-versioned-custody-receipt-v1','status':'PASS','operation':operation,'plan_sha256':source_plan_sha,'execution_commit':plan['execution_commit'],'entries':_public_entries(entries),'file_count':len(entries),'bytes':total,'uncached_full_readback':True,'numerical_replay':'SEPARATELY_QUEUED','originals_intact':True,'same_device':same_device,'separate_filesystem_verified':operation=='off_device'}
        if receipt_pin:result['source_snapshot_receipt_sha256']=receipt_pin
        if operation=='snapshot':write_json(target/'SNAPSHOT_RECEIPT.json',result,private=True)
        elif operation=='off_device':copy_exact(Path(cfg['snapshot'])/'SNAPSHOT_RECEIPT.json',target/'SNAPSHOT_RECEIPT.json',receipt_pin,(Path(cfg['snapshot'])/'SNAPSHOT_RECEIPT.json').stat().st_size)
        else:write_json(target/'RESTORE_RECEIPT.json',result,private=True)
    else:raise ValueError('unregistered custody operation')
    write_json(cfg['public_output'],result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();out=run(a.spec);print(json.dumps({'status':out['status'],'operation':out.get('operation','plan'),'file_count':out['file_count']}))
