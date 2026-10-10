"""Queue-only original utility gate assembly from immutable aggregate units."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from .r3_family import ARMS, TASKS, REFERENCES, METRICS
from .r3_selection import original_inner_gate
from .r3_utility import point_clause
from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def run(config_path):
    active=assert_active_from_env()
    state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text())
    sources={Path(v['path']).resolve():v['sha256'] for v in state['units'][active['unit_id']]['spec']['sources']}
    def pinned(item):
        p=Path(item['path'])
        if p.is_symlink():raise PermissionError('gate source symlink refused')
        p=p.resolve(strict=True)
        if sources.get(p)!=item['sha256'] or sha(p)!=item['sha256']:raise PermissionError('gate source unpinned')
        return p
    cfgpath=Path(config_path).resolve(strict=True)
    if sources.get(cfgpath)!=sha(cfgpath):raise PermissionError('gate config unpinned')
    cfg=json.loads(cfgpath.read_text())
    arm=cfg['arm_id']
    if cfg.get('schema')!='pcrl-incumbent-r3-gate-config-v1' or cfg['unit_id']!=active['unit_id'] or arm not in ARMS or active['stage'] not in ('witness','protection') or (active['stage']=='witness' and arm!='P0'):
        raise PermissionError('gate queue identity differs')
    if set(cfg['source_units'])!={'s0_fit','s0_inner','s1_fit','s1_inner'}:raise PermissionError('four source units required')
    clauses=[];identities={};fit_records=[]
    for seed in (0,1):
        for role,tag in (('OSF_DEFENSE_FIT','fit'),('INNER_SELECTION','inner')):
            key=f's{seed}_{tag}';item=cfg['source_units'][key]
            receipt_path=pinned(item['receipt']);csv_path=pinned(item['csv']);receipt=json.loads(receipt_path.read_text())
            if receipt.get('schema')!='pcrl-incumbent-r3-utility-receipt-v1' or receipt['arm_id']!=arm or receipt['source_seed']!=seed or receipt['role']!=role or receipt['csv_sha256']!=sha(csv_path):raise PermissionError('source utility scope differs')
            with csv_path.open() as f:rows=list(csv.DictReader(f))
            observed=set();passed=0
            for row in rows:
                if row['primary_gate']!='True':continue
                reference=row['reference'];task=row['task'];metric=row['metric']
                if row['estimand']!='exact_expected' or int(row['realization'])!=-1 or int(row['source_seed'])!=seed or row['role']!=role or row['arm_id']!=arm:
                    raise PermissionError('primary estimand differs')
                expected_refs=('P0',) if tag=='fit' else REFERENCES
                clausekey=(task,reference,metric)
                if task not in TASKS or reference not in expected_refs or metric not in METRICS or clausekey in observed:raise PermissionError('primary clause inventory differs')
                observed.add(clausekey)
                margin=float(row['margin']);actual=point_clause(metric,margin,'fit' if tag=='fit' else 'inner')
                if actual['status']!=row['point_status'] or actual['cap']!=float(row['cap']):raise PermissionError('aggregate point gate disagrees with frozen arithmetic')
                passed+=actual['status']=='PASS'
                if tag=='inner':clauses.append({'seed':seed,'task':task,'reference':reference,'metric':metric,'status':actual['status']})
            total=6 if tag=='fit' else 12
            if len(observed)!=total or receipt['primary_clauses']!={'required':total,'passed':passed}:raise PermissionError('source primary totals differ')
            identities[key]={'unit_id':receipt['unit_id'],'receipt_sha256':sha(receipt_path),'csv_sha256':sha(csv_path)}
            if tag=='fit':fit_records.append({'seed':seed,'required':6,'passed':passed,'numeric':receipt['fit_numeric_diagnostics']})
    fit_total=sum(v['passed'] for v in fit_records)
    gate=original_inner_gate(arm,clauses,identities,{'required':12,'passed':fit_total,'status':'PASS' if fit_total==12 else 'FAIL','seed_records':fit_records})
    gate['source_seal_sha256']=active['seal_sha256'];gate['queue']=active
    out=Path(cfg['output']).resolve();owner=Path(__file__).resolve().parents[1]/'results/pcrl_incumbent_view_risk_v1/r3'
    if out.parent!=owner or out.exists() or out.is_symlink():raise PermissionError('gate output must be new in R3')
    with out.open('x') as f:json.dump(gate,f,sort_keys=True,indent=2);f.write('\n')
    return gate


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();g=run(a.config)
    print(json.dumps({'arm_id':g['arm_id'],'original_inner':g['original_inner'],'fit_proxy_status':g['fit_proxy']['status']}))

if __name__=='__main__':main()
