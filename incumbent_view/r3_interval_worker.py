"""Queue-only fixed-family utility intervals from pinned private cost vectors."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from .r3_intervals import FIELDS, expected_utility_intervals
from .r3_family import ARMS, K
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
    admission=pinned(cfg['admission_lock'])
    science=pinned(cfg['campaign_seal'])
    amendment=pinned(cfg['verification_amendment'])
    admission_sha=sha(admission);science_sha=sha(science);amendment_sha=sha(amendment)
    if admission_sha!=active['seal_sha256'] or admission_sha!=state['seal']['sha256']:
        raise PermissionError('interval queue admission seal differs')
    if science_sha not in {a['sha256'] for a in state['amendments'] if 'protection' in a['stages']}:
        raise PermissionError('interval science lock is not registered')
    verification=[a for a in state['amendments'] if 'verification' in a['stages']]
    if (not verification or verification[-1]['sha256']!=amendment_sha
        or state['units'][active['unit_id']]['spec'].get('amendment_sha256')!=amendment_sha
        or json.loads(amendment.read_text()).get('campaign_lock_sha256')!=science_sha):
        raise PermissionError('interval verification amendment does not bind science')
    if len(cfg['records'])!=2*len(ARMS):raise PermissionError('interval fixed source inventory differs')
    records={};source_receipts={}
    for item in cfg['records']:
        receipt_path=pinned(item['receipt']);vector_path=pinned(item['vectors']);receipt=json.loads(receipt_path.read_text())
        key=(receipt['arm_id'],receipt['source_seed'])
        expected_seal=admission_sha if key[0]=='P0' else science_sha
        if (receipt['schema']!='pcrl-incumbent-r3-utility-receipt-v1' or receipt['role']!='INNER_SELECTION'
            or key[0] not in ARMS or key[1] not in (0,1) or key in records
            or receipt['private_vectors_sha256']!=sha(vector_path)
            or receipt['source_pins']['seal']!=expected_seal
            or receipt['queue']['seal_sha256']!=admission_sha or receipt['queue']['stage']!='witness'
            or receipt.get('old_assessment_read') is not False or receipt.get('sensitive_labels_read') is not False):
            raise PermissionError('interval source utility scope differs')
        if key[0]!='P0' and receipt['source_pins']['admission_lock']!=admission_sha:
            raise PermissionError('protected interval source admission differs')
        records[key]=read_exact(vector_path,FIELDS);source_receipts[f'{key[0]}:s{key[1]}']=sha(receipt_path)
    if set(records)!={(arm,seed) for arm in ARMS for seed in (0,1)}:
        raise PermissionError('interval complete source inventory differs')
    slots=expected_utility_intervals(records)
    out=Path(cfg['output']).resolve();owner=Path(__file__).resolve().parents[1]/'results/pcrl_incumbent_view_risk_v1/r3'
    if out.parent!=owner or out.exists() or out.is_symlink():raise PermissionError('new R3 interval output required')
    payload={'schema':'pcrl-incumbent-r3-utility-intervals-v1','K':K,'utility_slots':192,'records_observed':len(records),'source_receipts':source_receipts,'slots':slots,'queue':active,'admission_lock_sha256':admission_sha,'campaign_seal_sha256':science_sha,'verification_amendment_sha256':amendment_sha,'old_assessment_read':False}
    with out.open('x') as f:json.dump(payload,f,sort_keys=True,indent=2);f.write('\n')
    return payload


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args();r=run(args.config)
    print(json.dumps({'records_observed':r['records_observed'],'utility_slots':r['utility_slots'],'K':r['K']}))

if __name__=='__main__':main()
