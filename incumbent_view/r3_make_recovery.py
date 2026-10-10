"""Build one hash-pinned post-freeze recovery inference queue unit.

This builder reads receipts and file hashes only. R0 supplies the exact gate
map after utility gates and the R4 fixed-slot aggregate are complete.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from .r3_family import ARMS,VIEWS
from .r3_recovery_intervals import STOCHASTIC


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def pin(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise PermissionError('inference source missing or symlink')
    return {'path':str(path.absolute()),'sha256':sha(path)}


def check_pin(item):
    if set(item)!={'path','sha256'} or pin(item['path'])['sha256']!=item['sha256']:
        raise PermissionError('inference input pin differs')
    return Path(item['path'])


def build(gate_map_path,aggregate_path,science_path,amendment_path):
    root=Path(__file__).resolve().parents[1]
    owner=root/'results/pcrl_incumbent_view_risk_v1/r3';r0=root/'results/pcrl_incumbent_view_risk_v1/r0'
    r1=root/'results/pcrl_incumbent_view_risk_v1/r1';r4=root/'results/pcrl_incumbent_view_risk_v1/r4'
    queue=json.loads((r0/'.private/compute/state.json').read_text())
    admission=pin(r0/'ADMISSION_LOCK.json');science=pin(science_path);amendment=pin(amendment_path)
    if queue['seal']['sha256']!=admission['sha256']:
        raise PermissionError('inference queue admission seal differs')
    if science['sha256'] not in {a['sha256'] for a in queue['amendments'] if 'protection' in a['stages']}:
        raise PermissionError('science lock is not registered')
    verification=[a for a in queue['amendments'] if 'verification' in a['stages']]
    if not verification or verification[-1]['sha256']!=amendment['sha256'] or json.loads(Path(amendment_path).read_text()).get('campaign_lock_sha256')!=science['sha256']:
        raise PermissionError('latest inference amendment does not bind science')
    gate_map=pin(gate_map_path);mapped=json.loads(Path(gate_map_path).read_text())
    if mapped.get('schema')!='pcrl-incumbent-r3-fixed-gate-map-v1' or set(mapped['gates'])!=set(ARMS):
        raise PermissionError('all eight gate slots required')
    aggregate=pin(aggregate_path);agg=json.loads(Path(aggregate_path).read_text())
    if agg.get('schema')!='pcrl-incumbent-r4-fixed-attack-aggregate-v1' or agg['fixed_slots']!=40 or len(agg['units'])!=40:
        raise PermissionError('R4 fixed attack aggregate differs')
    source_map=pin(r1/'.private/P0_SOURCE_PATHS.json');private_map=json.loads(Path(source_map['path']).read_text())
    role_manifest=pin(r1/'DATA_ACCESS_MANIFEST.json')
    sex=pin(private_map['roles']['INNER_SELECTION']['sex']['path'])
    if sex['sha256']!=private_map['roles']['INNER_SELECTION']['sex']['sha256'] or sex['sha256']!=json.loads(Path(role_manifest['path']).read_text())['roles']['INNER_SELECTION']['source_files']['sex']['sha256']:
        raise PermissionError('role-only INNER SEX admission differs')
    status={};sources=[];all_pins=[admission,science,amendment,gate_map,aggregate,source_map,role_manifest,sex]
    by_arm={arm:[] for arm in ARMS}
    fixed={(arm,seed,draw) for arm in ARMS for seed in (0,1) for draw in (range(3) if arm in STOCHASTIC else (0,))}
    seen=set()
    for item in agg['units']:
        key=(item['arm_id'],item['source_seed'],item['realization'])
        if key not in fixed or key in seen:raise PermissionError('R4 fixed release slot inventory differs')
        seen.add(key);by_arm[key[0]].append(item)
    if seen!=fixed:raise PermissionError('R4 forty release slots incomplete')
    for arm in ARMS:
        gatepin=mapped['gates'][arm]
        gate_doc=None if gatepin is None else json.loads(check_pin(gatepin).read_text())
        if gatepin is not None:all_pins.append(gatepin)
        if gate_doc is not None and (gate_doc.get('arm_id')!=arm or gate_doc.get('original_inner',{}).get('required')!=24):
            raise PermissionError('R3 original INNER gate identity differs')
        expected=2*(3 if arm in STOCHASTIC else 1)
        items=by_arm[arm]
        if len(items)!=expected:raise PermissionError('R4 arm release count differs')
        if gate_doc is None:reason='INCOMPLETE_SEARCH'
        elif gate_doc.get('all_two_seed_task_reference_point_caps_pass') is not True:reason='POINT_INELIGIBLE'
        elif any(x['effective_status']!='COMPLETE' for x in items):reason='INCOMPLETE_ATTACK'
        elif any(x.get('all_control_gates_pass') is not True for x in items):reason='CONTROL_FAILED'
        else:reason='COMPLETE_VALID_ATTACKS'
        status[arm]={'status':reason,'gate_sha256':None if gatepin is None else gatepin['sha256'],
                     'complete_release_instances':sum(x['effective_status']=='COMPLETE' for x in items),
                     'required_release_instances':expected}
        if reason!='COMPLETE_VALID_ATTACKS':continue
        for item in items:
            uid=item['effective_unit_id'];receipt=pin(r4/f'{uid}_RECEIPT.json')
            if receipt['sha256']!=item['receipt_sha256']:raise PermissionError('R4 aggregate receipt pin differs')
            doc=json.loads(Path(receipt['path']).read_text())
            if (doc.get('arm_id')!=arm or doc.get('source_seed')!=item['source_seed'] or doc.get('realization')!=item['realization']
                or doc.get('campaign_seal_sha256')!=science['sha256'] or doc.get('queue_seal_sha256')!=admission['sha256']):
                raise PermissionError('R4 release/science receipt differs')
            custody=r4/'.private/attacks'/uid;frozen=pin(custody/'FROZEN_COMPLETE.json')
            if frozen['sha256']!=doc['attack']['freeze_receipt_sha256']:raise PermissionError('R4 freeze receipt hash differs')
            views={}
            for view in VIEWS:
                folder=custody/view/'readers'
                readers=pin(folder/'READERS_RECEIPT.json')
                if readers['sha256']!=doc['attack']['reader_custody'][view]['private_receipt_sha256']:
                    raise PermissionError('R4 reader custody hash differs')
                record=json.loads(Path(readers['path']).read_text())
                prediction=pin(folder/'predictions_inner.npz');viewfile=pin(folder/'view_inner.npz')
                if (record['files']['predictions_inner.npz']!=prediction['sha256'] or
                    record['files']['view_inner.npz']!=viewfile['sha256']):
                    raise PermissionError('R4 prediction/view hash differs')
                views[view]={'reader_receipt':readers,'predictions':prediction,'view':viewfile}
                all_pins.extend((readers,prediction,viewfile))
            sources.append({'arm_id':arm,'source_seed':item['source_seed'],'realization':item['realization'],
                            'receipt':receipt,'frozen':frozen,'views':views})
            all_pins.extend((receipt,frozen))
    unit_id='R3_RECOVERY_INFERENCE_V1'
    output={'public':str(owner/f'{unit_id}.json'),'private_vectors':str(owner/'.private'/f'{unit_id}_VECTORS.npz')}
    config={'schema':'pcrl-incumbent-r3-recovery-inference-config-v1','unit_id':unit_id,'K':948,
            'admission_lock':admission,'campaign_seal':science,'role_manifest':role_manifest,'inner_sex':sex,
            'attack_aggregate':aggregate,'gate_map':gate_map,'arm_status':status,'sources':sources,'outputs':output}
    config_path=owner/'.private'/f'{unit_id}_CONFIG.json'
    with os.fdopen(os.open(config_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as f:json.dump(config,f,sort_keys=True,indent=2);f.write('\n')
    code=[root/f'incumbent_view/{name}.py' for name in ('r3_recovery_worker','r3_recovery_intervals','r3_family','r3_make_recovery')]
    code.append(r0/'compute_queue.py')
    all_pins.extend(pin(p) for p in code);all_pins.append(pin(config_path))
    distinct={Path(x['path']):x['sha256'] for x in all_pins}
    spec={'unit_id':unit_id,'stage':'verification','amendment_sha256':amendment['sha256'],
          'argv':[str(root.parents[1]/'.venv/bin/python'),'-m','incumbent_view.r3_recovery_worker','--config',str(config_path)],
          'cwd':str(root),'max_wall_s':3600,'cpu_reserve_s':3600,'max_rss_gib':2.5,'projected_output_bytes':100000000,
          'deps':[],'outputs':list(output.values()),'sources':[{'path':str(p),'sha256':digest} for p,digest in distinct.items()],
          'env':{'PYTHONPATH':str(root),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}}
    spec_path=owner/'.private'/f'{unit_id}_QUEUE_SPEC.json'
    with os.fdopen(os.open(spec_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as f:json.dump(spec,f,sort_keys=True,indent=2);f.write('\n')
    return {'unit_id':unit_id,'config_sha256':sha(config_path),'spec_sha256':sha(spec_path),
            'eligible_arms':[a for a in ARMS if status[a]['status']=='COMPLETE_VALID_ATTACKS'],
            'source_attack_instances':len(sources),'source_hash_count':len(distinct)}


def main():
    parser=argparse.ArgumentParser()
    for name in ('gate-map','aggregate','science','amendment'):parser.add_argument('--'+name,required=True)
    a=parser.parse_args();print(json.dumps(build(a.gate_map,a.aggregate,a.science,a.amendment),sort_keys=True))


if __name__=='__main__':main()
