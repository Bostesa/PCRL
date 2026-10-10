"""Queue-only K948 recovery inference after all final R4 readers are frozen.

This worker reads the admitted INNER SEX role-only archive only after source
attacker receipts, frozen reader custody, prediction archives, role views and
control gates have all been byte checked for every included release.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from .r3_family import ARMS,VIEWS
from .r3_recovery_intervals import (STOCHASTIC,BOOTSTRAPS,GROUPS,bootstrap_plan,
                                    frozen_envelope,recovery_intervals)
from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env

VIEW_ALLOWED={'row_id','unit','x','x_defense_aware','x_metadata','x_code_bits','cell_key','x_raw_only'}


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def sha_array(value):
    value=np.ascontiguousarray(value)
    digest=hashlib.sha256();digest.update(str(value.dtype).encode());digest.update(str(tuple(value.shape)).encode());digest.update(value.tobytes())
    return digest.hexdigest()


def read_exact(path,keys):
    with ZipFile(path) as archive:
        members=[x.filename for x in archive.infolist()]
        if len(members)!=len(keys) or set(members)!={k+'.npy' for k in keys}:raise PermissionError('role-only archive members differ before deserialization')
    with np.load(path,allow_pickle=False) as source:return {k:np.array(source[k],copy=True) for k in keys}


def view_people(path):
    with ZipFile(path) as archive:
        members=[x.filename for x in archive.infolist()]
        names={x[:-4] for x in members if x.endswith('.npy')}
        if len(members)!=len(names) or not {'row_id','unit'}.issubset(names) or not names.issubset(VIEW_ALLOWED):raise PermissionError('private view member inventory differs')
    with np.load(path,allow_pickle=False) as source:return np.asarray(source['row_id']),np.asarray(source['unit'])


def inference_lock_bridge(cfg,active,state,registered):
    """Validate queue admission, science, and post-freeze inference seals."""
    def checked(record):
        path=Path(record['path'])
        if path.is_symlink():raise PermissionError('lock symlink refused')
        path=path.resolve(strict=True)
        if registered.get(path)!=record['sha256'] or sha(path)!=record['sha256']:
            raise PermissionError('inference lock byte pin differs')
        return path
    admission_sha=cfg['admission_lock']['sha256'];science_sha=cfg['campaign_seal']['sha256']
    checked(cfg['admission_lock']);checked(cfg['campaign_seal'])
    if admission_sha!=active['seal_sha256'] or admission_sha!=state['seal']['sha256']:
        raise PermissionError('inference queue admission seal differs')
    if science_sha not in {a['sha256'] for a in state['amendments'] if 'protection' in a['stages']}:
        raise PermissionError('inference science lock is not registered for protection')
    verification=[a for a in state['amendments'] if 'verification' in a['stages']]
    if not verification or state['units'][active['unit_id']]['spec'].get('amendment_sha256')!=verification[-1]['sha256']:
        raise PermissionError('inference is not under latest verification amendment')
    latest=verification[-1]
    latest_path=Path(latest['path']).resolve(strict=True)
    if registered.get(latest_path)!=latest['sha256'] or sha(latest_path)!=latest['sha256']:
        raise PermissionError('verification amendment is not pinned')
    if json.loads(latest_path.read_text()).get('campaign_lock_sha256')!=science_sha:
        raise PermissionError('verification amendment does not bind science lock')
    return admission_sha,science_sha


def run(config_path):
    active=assert_active_from_env(stage='verification')
    state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text())
    registered={Path(item['path']).resolve():item['sha256'] for item in state['units'][active['unit_id']]['spec']['sources']}
    def pinned(item):
        path=Path(item['path'])
        if path.is_symlink():raise PermissionError('inference source symlink refused')
        path=path.resolve(strict=True)
        if registered.get(path)!=item['sha256'] or sha(path)!=item['sha256']:raise PermissionError('inference source not active-queue byte-pinned')
        return path
    cfgpath=Path(config_path).resolve(strict=True)
    if registered.get(cfgpath)!=sha(cfgpath):raise PermissionError('inference config unpinned')
    cfg=json.loads(cfgpath.read_text())
    if cfg.get('schema')!='pcrl-incumbent-r3-recovery-inference-config-v1' or cfg['unit_id']!=active['unit_id'] or cfg['K']!=948:
        raise PermissionError('fixed inference config differs')
    admission_sha,science_sha=inference_lock_bridge(cfg,active,state,registered)
    manifest=json.loads(pinned(cfg['role_manifest']).read_text())
    sexpin=cfg['inner_sex']
    if sexpin['sha256']!=manifest['roles']['INNER_SELECTION']['source_files']['sex']['sha256']:
        raise PermissionError('role-only INNER SEX is not admitted')
    if set(cfg['arm_status'])!=set(ARMS):raise PermissionError('all fixed arm slots must be accounted')
    aggregate=json.loads(pinned(cfg['attack_aggregate']).read_text())
    gate_map=json.loads(pinned(cfg['gate_map']).read_text())
    if (aggregate.get('schema')!='pcrl-incumbent-r4-fixed-attack-aggregate-v1' or aggregate.get('fixed_slots')!=40
        or len(aggregate.get('units',[]))!=40 or gate_map.get('schema')!='pcrl-incumbent-r3-fixed-gate-map-v1'
        or set(gate_map.get('gates',{}))!=set(ARMS)):
        raise PermissionError('complete fixed-slot aggregate/gates differ')
    expected_slots={(arm,seed,draw) for arm in ARMS for seed in (0,1)
                    for draw in (range(3) if arm in STOCHASTIC else (0,))}
    aggregated={(x['arm_id'],x['source_seed'],x['realization']):x for x in aggregate['units']}
    if set(aggregated)!=expected_slots:raise PermissionError('fixed forty attack slot keys differ')
    for arm in ARMS:
        gatepin=gate_map['gates'][arm]
        gate=None if gatepin is None else json.loads(pinned(gatepin).read_text())
        if gate is not None and (gate.get('arm_id')!=arm or gate.get('original_inner',{}).get('required')!=24):
            raise PermissionError('original INNER utility gate differs')
        armslots=[v for k,v in aggregated.items() if k[0]==arm]
        if gate is None:reason='INCOMPLETE_SEARCH'
        elif gate.get('all_two_seed_task_reference_point_caps_pass') is not True:reason='POINT_INELIGIBLE'
        elif any(v['effective_status']!='COMPLETE' for v in armslots):reason='INCOMPLETE_ATTACK'
        elif any(v.get('all_control_gates_pass') is not True for v in armslots):reason='CONTROL_FAILED'
        else:reason='COMPLETE_VALID_ATTACKS'
        recorded=cfg['arm_status'][arm]
        if (recorded['status']!=reason or recorded['required_release_instances']!=len(armslots)
            or recorded['complete_release_instances']!=sum(v['effective_status']=='COMPLETE' for v in armslots)
            or recorded['gate_sha256']!=(None if gatepin is None else gatepin['sha256'])):
            raise PermissionError('candidate eligibility differs from frozen complete-view attack aggregate')
    eligible={arm for arm,record in cfg['arm_status'].items() if record['status']=='COMPLETE_VALID_ATTACKS'}
    expected={(arm,seed,draw) for arm in eligible for seed in (0,1) for draw in (range(3) if arm in STOCHASTIC else (0,))}
    entries={}
    for item in cfg['sources']:
        key=(item['arm_id'],item['source_seed'],item['realization'])
        if key in entries or key not in expected:raise PermissionError('final attack source key differs')
        receipt_path=pinned(item['receipt']);doc=json.loads(receipt_path.read_text())
        if (aggregated[key]['receipt_sha256']!=sha(receipt_path) or
            aggregated[key]['effective_unit_id']!=doc.get('unit_id')):
            raise PermissionError('final attack differs from fixed-slot public aggregate')
        if (doc.get('schema')!='pcrl-incumbent-r4-fresh-attack-v1' or doc.get('status')!='COMPLETE_PRIVATE_DEVELOPMENT_ONLY'
            or doc.get('arm_id')!=key[0] or doc.get('source_seed')!=key[1] or doc.get('realization')!=key[2]
            or doc.get('campaign_seal_sha256')!=science_sha or doc.get('queue_seal_sha256')!=admission_sha):raise PermissionError('R4 frozen attack receipt differs')
        frozen_path=pinned(item['frozen']);frozen=json.loads(frozen_path.read_text())
        if (frozen.get('status')!='FROZEN_COMPLETE' or frozen.get('effective_lock_sha256')!=science_sha
            or frozen.get('family')!='A' or frozen.get('seed')!=doc.get('attacker_seed')
            or sha(frozen_path)!=doc['attack']['freeze_receipt_sha256']):raise PermissionError('all R4 readers must be frozen before INNER SEX')
        viewfiles={}
        for view in VIEWS:
            attack=doc['attack']
            controls=attack['controls'][view]['scores']
            if (controls['status']!='PASS' or controls.get('gates')!={
                'null_exact':True,'planted_at_least_0.99':True,
                'shuffled_independent_inner_at_most_0.60':True}):
                raise PermissionError('final attack engineering controls fail')
            root=item['views'][view]
            reader_path=pinned(root['reader_receipt']);pred_path=pinned(root['predictions']);view_path=pinned(root['view'])
            if (sha(reader_path)!=attack['reader_custody'][view]['private_receipt_sha256']
                or sha(reader_path)!=frozen['reader_receipt_sha256'][view]):
                raise PermissionError('R4 frozen reader custody differs')
            reader=json.loads(reader_path.read_text())
            if (reader.get('status')!='FROZEN' or reader['files']['predictions_inner.npz']!=sha(pred_path)
                or reader['files']['view_inner.npz']!=sha(view_path)):
                raise PermissionError('R4 frozen reader/view SHA differs')
            viewfiles[view]={'reader':reader,'predictions':pred_path,'view':view_path,
                             'point_auc':float(attack['scores'][view]['primary_envelope_auc'])}
        entries[key]={'receipt_sha256':sha(receipt_path),'views':viewfiles}
    if set(entries)!=expected:raise PermissionError('some required final attacks or draws are missing')
    # The entire complete-slate custody above is checked BEFORE this SEX load.
    # An empty eligible set has no AUC estimand and needs no sensitive read.
    if eligible:
        truth=read_exact(pinned(sexpin),{'row_id','unit','sex'})
        row,unit,y=truth['row_id'],truth['unit'],truth['sex']
        if (row.dtype!=np.int64 or unit.dtype!=np.int64 or not np.issubdtype(y.dtype,np.integer) or row.shape!=(2235,) or unit.shape!=row.shape
            or y.shape!=row.shape or np.unique(row).size!=len(row) or np.unique(unit).size!=GROUPS or not np.isin(y,(0,1)).all()
            or sha_array(np.c_[row,unit])!=manifest['roles']['INNER_SELECTION']['row_unit_pair_sha256']):
            raise PermissionError('INNER SEX people/groups/schema differ')
        y=y.astype(np.int64,copy=False)
        inverse,weights,draw_indices=bootstrap_plan(unit)
    else:
        draw_indices=np.zeros((BOOTSTRAPS,2,3),dtype=np.int64)
    values={};point_aucs={}
    for (arm,seed,draw),entry in entries.items():
        assert_active_from_env(unit_id=active['unit_id'],stage='verification')
        for view,source in entry['views'].items():
            record_row,record_unit=view_people(source['view'])
            if not np.array_equal(record_row,row) or not np.array_equal(record_unit,unit):raise PermissionError('reader view and SEX person order differ')
            order=tuple(source['reader']['reader_order'])
            preds=read_exact(source['predictions'],set(order))
            vector=frozen_envelope(y,preds,order,inverse,weights)
            if abs(float(vector[0])-source['point_auc'])>1e-11:raise ArithmeticError('fresh R4 reader envelope point AUC did not replay')
            values[(arm,seed,draw,view)]=vector
            point_aucs[f'{arm}:s{seed}:d{draw}:{view}']=float(vector[0])
    intervals,_=recovery_intervals(values,eligible,draw_indices)
    out=cfg['outputs'];owner=Path(__file__).resolve().parents[1]/'results/pcrl_incumbent_view_risk_v1/r3'
    for key,value in out.items():
        path=Path(value).resolve();target=owner/'.private' if key=='private_vectors' else owner
        if path.parent!=target or path.exists() or path.is_symlink():raise PermissionError('new R3 inference output in owner namespace required')
    private={f'{a}__s{s}__d{d}__{v}':arr for (a,s,d,v),arr in values.items()}
    private['draw_indices']=draw_indices
    with os.fdopen(os.open(out['private_vectors'],os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb') as f:np.savez_compressed(f,**private)
    public={'schema':'pcrl-incumbent-r3-recovery-inference-v1','K':948,'recovery_slots':756,'eligible_arms':sorted(eligible,key=ARMS.index),
            'arm_status':cfg['arm_status'],'point_envelope_aucs':point_aucs,'slots':intervals,
            'R4_source_receipt_sha256':{f'{a}:s{s}:d{d}':v['receipt_sha256'] for (a,s,d),v in entries.items()},
            'private_vectors_sha256':sha(out['private_vectors']),'inner_sex_role_hash':sexpin['sha256'],
            'draws':3,'bootstrap_groups':GROUPS,'bootstrap_replicates':BOOTSTRAPS,'queue':active,
            'inner_sex_payload_read':bool(eligible),
            'not_confirmation':True,'query_contract':'one actual paired release per person; three independent virtual deployments'}
    with os.fdopen(os.open(out['public'],os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644),'w') as f:json.dump(public,f,sort_keys=True,indent=2);f.write('\n')
    return public


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();r=run(a.config)
    print(json.dumps({'K':r['K'],'recovery_slots':r['recovery_slots'],'eligible_arms':r['eligible_arms']}))
if __name__=='__main__':main()
