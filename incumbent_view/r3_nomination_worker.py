"""Queue-only FIT-reader nomination from immutable public R2 and R3 receipts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from .r3_family import ARMS, V_FAMILIES
from .r3_selection import GATE_SCHEMA, nominate_family
from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def run(config_path):
    active=assert_active_from_env(stage='protection')
    state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text())
    sources={Path(v['path']).resolve():v['sha256'] for v in state['units'][active['unit_id']]['spec']['sources']}
    def pinned(item):
        p=Path(item['path'])
        if p.is_symlink():raise PermissionError('nomination source symlink refused')
        p=p.resolve(strict=True)
        if sources.get(p)!=item['sha256'] or sha(p)!=item['sha256']:raise PermissionError('nomination source unpinned')
        return p
    cfgpath=Path(config_path).resolve(strict=True)
    if sources.get(cfgpath)!=sha(cfgpath):raise PermissionError('nomination config unpinned')
    cfg=json.loads(cfgpath.read_text())
    if cfg['schema']!='pcrl-incumbent-r3-nomination-config-v1' or cfg['unit_id']!=active['unit_id'] or set(cfg['r2_receipts'])!={f'{arm}:s{seed}' for arm in V_FAMILIES for seed in (0,1)} or not set(cfg['gates']).issubset(ARMS) or 'P0' not in cfg['gates']:
        raise PermissionError('fixed FIT nomination inventory differs')
    gates={}
    for arm,item in cfg['gates'].items():
        doc=json.loads(pinned(item).read_text())
        if doc['schema']!=GATE_SCHEMA or doc['arm_id']!=arm or doc['original_inner']['required']!=24:
            raise PermissionError('candidate original INNER point gate differs')
        gates[arm]=doc
    records={};units={}
    for key,item in cfg['r2_receipts'].items():
        arm,seedtext=key.split(':s');seed=int(seedtext)
        doc=json.loads(pinned(item).read_text())
        if doc['schema']!='IVR1-r2-protection-unit-v1' or doc['arm_id']!=arm or doc['source_seed']!=seed:
            raise PermissionError('R2 FIT selected receipt scope differs')
        units[key]={'unit_id':doc['source_unit_id'],'status':doc['status'],'receipt_sha256':sha(Path(item['path']))}
        if doc['status'] in ('COMPLETE_SELECTED','COMPLETE_IDENTITY_ALIAS','COMPLETE_WHOLE_P0_DEPENDENCY_FALLBACK'):
            nomination=doc.get('nomination')
            if (not isinstance(nomination,dict) or nomination.get('crossfit_manifest_sha256')!=doc.get('crossfit_manifest_sha256')
                or type(nomination.get('exact_p0_alias')) is not bool):
                raise PermissionError('R2 fixed OOF nomination absent or wrong fold')
            records[(arm,seed)]=nomination
        elif doc['status']!='INCOMPLETE':raise PermissionError('R2 FIT status differs')
    for arm in V_FAMILIES:
        if all(units[f'{arm}:s{s}']['status']!='INCOMPLETE' for s in (0,1)) and arm not in gates:
            raise PermissionError('completed new family lacks original point gate')
    nomination=nominate_family(gates,records)
    nomination['source_r2_units']=units
    nomination['source_gate_sha256']={arm:item['sha256'] for arm,item in cfg['gates'].items()}
    nomination['queue']=active
    out=Path(cfg['output']).resolve();owner=Path(__file__).resolve().parents[1]/'results/pcrl_incumbent_view_risk_v1/r3'
    if out.parent!=owner or out.exists() or out.is_symlink():raise PermissionError('R3 nomination output must be new')
    with out.open('x') as f:json.dump(nomination,f,sort_keys=True,indent=2);f.write('\n')
    return nomination


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();n=run(a.config)
    print(json.dumps({'nominee':n['nominee'],'status':n['status'],'FIT_only':True}))
if __name__=='__main__':main()
