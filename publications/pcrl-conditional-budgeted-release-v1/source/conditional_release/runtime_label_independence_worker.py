"""Actual deployment with no role records and denied role/archive I/O."""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path
from .scheduler import assert_active_from_env


def run(config):
    active=assert_active_from_env(stage='verification')
    cfg=json.loads(Path(config).read_text())
    if active['unit_id']!=cfg['unit_id']:raise PermissionError('runtime guard unit differs')
    full=json.loads(Path(cfg['source_manifest']).read_text())
    if hashlib.sha256(Path(cfg['source_manifest']).read_bytes()).hexdigest()!=cfg['source_manifest_sha256']:
        raise PermissionError('source metadata differs')
    runtime={k:v for k,v in full.items() if k not in ('roles','seeds')}
    runtime['seeds']={s:{k:v for k,v in record.items() if k in ('pins','public_bank','policies')} for s,record in full['seeds'].items()}
    def leaves(value):
        if isinstance(value,dict):
            if 'path' in value and 'sha256' in value:yield value['path']
            else:
                for item in value.values():yield from leaves(item)
    allowed={str(Path(p).resolve()) for p in leaves(runtime)}
    denied={str(Path(p).resolve()) for p in leaves(full)}-allowed
    attempted=[]
    def audit(event,args):
        if event!='open' or not args or isinstance(args[0],int):return
        try:path=str(Path(os.fsdecode(args[0])).resolve())
        except (TypeError,ValueError):return
        low=path.lower()
        embargo=any(x in low for x in ('head_validation','osf_development_assessment','old_assessment','all_role'))
        archive=Path(path).name=='release.npz' and any(x in low for x in ('teacher','rel__s'))
        if path in denied or embargo or archive:
            attempted.append('denied-role-or-embargoed-path')
            raise PermissionError('deployment attempted unavailable role/archive I/O')
    # Installed before importing the actual deployment module or constructing
    # a model/policy; it also covers each runtime loader's hash checks.
    sys.addaudithook(audit)
    from .source import write_once,sha_file
    from .deployment import main
    output=Path(cfg['private_output']);output.mkdir(parents=True,exist_ok=False,mode=0o700)
    manifest=output/'RUNTIME_SOURCE_PATHS.json';manifest_sha=write_once(manifest,runtime)
    rows=[]
    for item in cfg['models']:
        stem=f"{item['arm_id']}_s{item['seed']}"
        wire=[output/(stem+f'_r{r}.jsonl') for r in (1,2)]
        argv=['--source-manifest',str(manifest),'--source-manifest-sha256',manifest_sha,
              '--seed',str(item['seed']),'--kind',item['kind'],
              '--policy-dir',item['policy_dir'],'--policy-receipt-sha256',item['receipt_sha256'],
              '--input',item['raw']['path'],'--input-sha256',item['raw']['sha256'],
              '--person-keys',item['keys']['path'],'--person-keys-sha256',item['keys']['sha256'],
              '--sampling-root',item['root']['path'],'--cache',str(output/(stem+'.sqlite')),
              '--output-r1',str(wire[0]),'--output-r2',str(wire[1])]
        main(argv)
        for path,expected in zip(wire,item['expected_wires']):
            if sha_file(expected['path'])!=expected['sha256'] or path.read_bytes()!=Path(expected['path']).read_bytes():
                raise AssertionError('role-free actual CLI wire differs')
        rows.append({'arm_id':item['arm_id'],'seed':item['seed'],'kind':item['kind'],'actual_cli_byte_parity':'EXACT'})
    if attempted:raise AssertionError('forbidden runtime dependency')
    result={'schema':'CBR1-runtime-label-independence-v1','status':'PASS','unit_id':active['unit_id'],
            'models':rows,'model_count':len(rows),'runtime_manifest_role_records':'ABSENT',
            'runtime_manifest_role_wire_reader_and_sample_records':'ABSENT',
            'blocked_original_role_and_nonruntime_source_paths':len(denied),'blocked_access_attempts':len(attempted),
            'audit_installed_before_deployment_import':True,'all_label_bytes_unavailable_to_actual_runtime':True,
            'role_or_label_array_deserialization':False,'new_scientific_fits':0,
            'queue_preflight_scope':'Runtime frozen model/router/bank/policy pins and registered campaign-private raw/key/coin/wire artifacts only; no task or SEX label archive pins',
            'earlier_queue_preflight':'Earlier comprehensive source/replay units hashed permitted label archives; that infrastructure access is distinct from this verified deployment requirement'}
    write_once(cfg['public_output'],result,False)
    print(json.dumps({'status':'PASS','models':len(rows),'blocked_paths':len(denied),'blocked_attempts':len(attempted)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args();run(args.config)
