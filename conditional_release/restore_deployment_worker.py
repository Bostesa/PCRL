"""Representative raw83 wire replay using only restored inputs and pinned coins."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
from .scheduler import assert_active_from_env
from .source import load_manifest,sha_file,pinned,write_once
from .deployment import load_p0_service,CachedRelease
from .deployment_worker import load_model
from qpc.deploy import load_permitted_input


def run(config):
    active=assert_active_from_env(stage='verification');cfg=json.loads(Path(config).read_text())
    if cfg['unit_id']!=active['unit_id']:raise PermissionError('restore unit differs')
    restored=Path(cfg['restore_root']).resolve()
    def only_restore(path):
        p=Path(path).resolve()
        if not p.is_relative_to(restored):raise PermissionError('restore replay depends on an original artifact')
        return p
    manifest_path=only_restore(cfg['source_manifest']);manifest=load_manifest(manifest_path,cfg['source_manifest_sha256'])
    def walk(x):
        if isinstance(x,dict):
            if 'path' in x and 'sha256' in x:only_restore(x['path']);pinned(x)
            else:
                for v in x.values():walk(v)
    walk(manifest)
    for k in ('raw','person_keys','sampling_root'):
        only_restore(cfg[k]['path']);pinned(cfg[k])
    secret=Path(cfg['sampling_root']['path']).read_bytes()
    if len(secret)!=32:raise PermissionError('restored private sampling root differs')
    service=load_p0_service(manifest,cfg['seed']);raw=load_permitted_input(cfg['raw']['path'],list(service.names));keys=json.loads(Path(cfg['person_keys']['path']).read_text())
    private=Path(cfg['private_output']);private.mkdir(parents=True,exist_ok=False,mode=0o700);rows=[]
    for item in cfg['models']:
        if item['kind'] not in ('P0','V_J','V_S21'):
            only_restore(item['policy_dir']);pinned({'path':str(Path(item['policy_dir'])/'POLICY_RECEIPT.json'),'sha256':item['receipt_sha256']})
        law,sampler,binding=load_model(item,manifest)
        api=CachedRelease(service,law,policy_sha256=binding,cache_path=private/(item['arm_id']+'.sqlite'),secret=secret,sampling_kind=sampler)
        wire=api.release(raw,service.names,keys);api.close()
        for recipient,view in enumerate(wire,1):
            expected=item['wire'][str(recipient)];only_restore(expected['path']);pinned(expected)
            lines=b''.join((json.dumps(w,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode() for w in view)
            if lines!=Path(expected['path']).read_bytes():raise AssertionError('restored raw83 cached wire differs')
        rows.append({'arm_id':item['arm_id'],'seed':item['seed'],'sampling_kind':sampler,'rows':len(raw),'pinned_coin_wire_byte_parity':'EXACT'})
    result={'schema':'CBR1-restored-raw83-pinned-coin-replay-v1','status':'PASS','unit_id':active['unit_id'],'original_private_artifacts_read':False,'scientific_labels_read':False,'independent_replay':False,'model_replays':rows,'source_manifest_sha256':cfg['source_manifest_sha256']}
    write_once(cfg['public_output'],result,False);print(json.dumps({'status':'PASS','models':len(rows)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();run(a.config)
