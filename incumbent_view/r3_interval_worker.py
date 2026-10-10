"""Queue-only fixed-family utility intervals from pinned private cost vectors."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from .r3_intervals import FIELDS, expected_utility_intervals
from .r3_family import K
from .r3_worker import read_exact, sha
from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env


def run(config_path):
    active=assert_active_from_env(stage='verification')
    state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text())
    sources={Path(v['path']).resolve():v['sha256'] for v in state['units'][active['unit_id']]['spec']['sources']}
    def pinned(item):
        path=Path(item['path'])
        if path.is_symlink():raise PermissionError('interval source symlink refused')
        path=path.resolve(strict=True)
        if sources.get(path)!=item['sha256'] or sha(path)!=item['sha256']:raise PermissionError('interval source pin differs')
        return path
    cfgpath=Path(config_path).resolve(strict=True)
    if sources.get(cfgpath)!=sha(cfgpath):raise PermissionError('interval config unpinned')
    cfg=json.loads(cfgpath.read_text())
    if cfg['schema']!='pcrl-incumbent-r3-interval-config-v1' or cfg['unit_id']!=active['unit_id']:raise PermissionError('interval config scope differs')
    records={};source_receipts={}
    for item in cfg['records']:
        receipt_path=pinned(item['receipt']);vector_path=pinned(item['vectors']);receipt=json.loads(receipt_path.read_text())
        key=(receipt['arm_id'],receipt['source_seed'])
        if receipt['schema']!='pcrl-incumbent-r3-utility-receipt-v1' or receipt['role']!='INNER_SELECTION' or receipt['private_vectors_sha256']!=sha(vector_path) or key in records:raise PermissionError('interval source utility scope differs')
        records[key]=read_exact(vector_path,FIELDS);source_receipts[f'{key[0]}:s{key[1]}']=sha(receipt_path)
    slots=expected_utility_intervals(records)
    out=Path(cfg['output']).resolve();owner=Path(__file__).resolve().parents[1]/'results/pcrl_incumbent_view_risk_v1/r3'
    if out.parent!=owner or out.exists() or out.is_symlink():raise PermissionError('new R3 interval output required')
    payload={'schema':'pcrl-incumbent-r3-utility-intervals-v1','K':K,'utility_slots':192,'records_observed':len(records),'source_receipts':source_receipts,'slots':slots,'queue':active,'old_assessment_read':False}
    with out.open('x') as f:json.dump(payload,f,sort_keys=True,indent=2);f.write('\n')
    return payload


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args();r=run(args.config)
    print(json.dumps({'records_observed':r['records_observed'],'utility_slots':r['utility_slots'],'K':r['K']}))

if __name__=='__main__':main()
