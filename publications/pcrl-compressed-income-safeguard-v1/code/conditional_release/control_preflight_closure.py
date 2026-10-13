"""Queued closure of preserved data-free control preflight output artifacts."""
import argparse,json
from pathlib import Path
from .source import sha_file,write_once
from .scheduler import assert_active_from_env


def run(config):
    active=assert_active_from_env(stage='admission');cfg=json.loads(Path(config).read_text())
    if active['unit_id']!=cfg['unit_id']:raise PermissionError('closure unit differs')
    if sha_file(cfg['preflight_receipt'])!=cfg['preflight_receipt_sha256']:
        raise PermissionError('preserved preflight receipt changed')
    receipt=json.loads(Path(cfg['preflight_receipt']).read_text())
    if receipt['status']!='PASS' or receipt['actual_max_rss_gib']>2.:
        raise ValueError('engineering memory gate did not pass')
    for record in receipt['records']:
        if 'archive' in record:
            archive=record['archive'];path=Path(cfg['preflight_private'])/archive['archive_name']
            if sha_file(path)!=archive['archive_sha256']:raise PermissionError('full coordinate archive changed')
    wrapper=json.loads(Path(cfg['wrapper_receipt']).read_text())
    if wrapper['returncode']!=0 or wrapper['resource_limit'] is not None or wrapper['timed_out']:
        raise ValueError('preflight child did not finish successfully')
    result={'schema':'CBR1-wide-control-memory-preflight-closure-v1','status':'PASS','unit_id':cfg['unit_id'],
            'original_unit_id':receipt['unit_id'],'original_wrapper_status':wrapper['status'],
            'original_preflight_receipt_sha256':cfg['preflight_receipt_sha256'],
            'actual_max_rss_gib':receipt['actual_max_rss_gib'],'no_fit_rerun':True,
            'original_error':'queue output metadata listed directory while scheduler requires files',
            'scope':'preserved completed data-free preflight receipt and full coordinate archive hashes; no Adult claims'}
    write_once(cfg['public_output'],result,False)
    print(json.dumps({'unit_id':cfg['unit_id'],'status':'PASS'}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args();run(args.config)
