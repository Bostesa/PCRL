"""Queue-authenticated fitting and globally frozen reporting workers."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
from zipfile import ZipFile


def queue_config(spec_path,stage):
    from .scheduler import assert_active_from_env
    from .attacks import sha_file
    active=assert_active_from_env(stage=stage)
    state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text())
    registered={str(Path(i['path']).resolve()):i['sha256'] for i in state['units'][active['unit_id']]['spec']['sources']}
    def verify(path,expected):
        p=Path(path)
        if p.is_symlink() or registered.get(str(p.resolve()))!=expected or sha_file(p)!=expected:raise PermissionError('artifact differs from active queue exact sources')
        return p
    p=Path(spec_path);verify(p,registered.get(str(p.resolve())))
    cfg=json.loads(p.read_text())
    if cfg['unit_id']!=active['unit_id']:raise PermissionError('worker and queue unit differ')
    return active,cfg,verify


def fit_worker(spec_path):
    active,cfg,verify=queue_config(spec_path,'attacks')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from .scheduler import assert_active_from_env
    from .source import load_manifest,load_role
    from .models import load_checkpoint
    from .attacks import fit_bank,write_json,sha_file
    verify(cfg['science_lock'],cfg['science_lock_sha256'])
    attack=verify(cfg['attack_lock'],cfg['attack_lock_sha256']);lock=json.loads(attack.read_text())
    admitted=[i for i in lock.get('instances',[]) if i['arm_id']==cfg['arm_id'] and i['seed']==cfg['seed']]
    if lock.get('status')!='RELEASES_AND_ENDPOINTS_FROZEN' or len(admitted)!=1 or admitted[0]['checkpoint_sha256']!=cfg['checkpoint_sha256']:raise PermissionError('release stage lock does not admit exact arm/seed/checkpoint')
    manifest_path=verify(cfg['manifest'],cfg['manifest_sha256']);manifest=load_manifest(manifest_path,cfg['manifest_sha256'])
    checkpoint=verify(cfg['checkpoint'],cfg['checkpoint_sha256']);model,payload=load_checkpoint(checkpoint,cfg['checkpoint_sha256'])
    if int(payload['seed'])!=cfg['seed']:raise PermissionError('checkpoint source seed differs')
    att=load_role(manifest,cfg['seed'],'ATTACK_FIT_NEW',{'features','sex'});inner=load_role(manifest,cfg['seed'],'INNER_SELECTION',{'features'})
    def release(role,sex=False):
        views=model.route_raw(role['X'],role['feature_names'])
        out={k:role[k] for k in ('row_id','unit')};out.update(r1=views[0]['wire'],r2=views[1]['wire'])
        if sex:out['sex']=role['sex']
        for v in out['r1'],out['r2']:
            for group in np.unique(out['unit']):
                ix=np.flatnonzero(out['unit']==group)
                if len(ix)>1 and not np.array_equal(v[ix],np.repeat(v[ix[:1]],len(ix),axis=0)):raise PermissionError('duplicate people produce different deterministic wires')
        return out
    def guard():
        assert_active_from_env(unit_id=active['unit_id'],stage='attacks')
        verify(cfg['science_lock'],cfg['science_lock_sha256']);verify(cfg['attack_lock'],cfg['attack_lock_sha256'])
    receipt=fit_bank(release(att,True),release(inner),seed=cfg['seed'],private_output=cfg['private_output'],queue_guard=guard,
                     bindings={'arm_id':cfg['arm_id'],'checkpoint_sha256':cfg['checkpoint_sha256'],'manifest_sha256':cfg['manifest_sha256'],'science_lock_sha256':cfg['science_lock_sha256'],'attack_lock_sha256':cfg['attack_lock_sha256']})
    public={**receipt,'unit_id':active['unit_id'],'arm_id':cfg['arm_id'],'frozen_complete_sha256':sha_file(Path(cfg['private_output'])/'FROZEN_COMPLETE.json')}
    write_json(cfg['public_output'],public,False)
    return public


def read_reporting_sex(pin,verify,release):
    import numpy as np
    path=verify(pin['path'],pin['sha256'])
    if '.private' not in path.parts or path.stat().st_mode&0o077:raise PermissionError('reporting SEX requires owner-only one-role export')
    with ZipFile(path) as z:
        if len(z.infolist())!=3 or {i.filename for i in z.infolist()}!={'row_id.npy','unit.npy','sex.npy'}:raise PermissionError('reporting archive contains unadmitted fields')
    with np.load(path,allow_pickle=False) as z:data={k:z[k].copy() for k in ('row_id','unit','sex')}
    if any(a.dtype!=np.int64 for a in data.values()) or data['sex'].shape!=(2235,) or len(np.unique(data['unit']))!=2234 or not np.isin(data['sex'],[0,1]).all():raise PermissionError('reporting role schema/counts differ')
    for k in ('row_id','unit'):
        if not np.array_equal(data[k],release[k]):raise PermissionError('reporting people differ')
    return data


def reporting_lock(cfg,verify):
    lock=json.loads(verify(cfg['reporting_lock'],cfg['reporting_lock_sha256']).read_text())
    if lock.get('status')!='ALL_READERS_FROZEN' or lock.get('K_primary')!=51 or lock.get('K_secondary')!=33 or lock.get('inner_sex_sha256')!=cfg['inner_sex']['sha256']:raise PermissionError('global reporting freeze absent')
    if not lock.get('reader_freezes'):raise PermissionError('empty global reader freeze')
    for pin in lock['reader_freezes']:
        p=verify(pin['path'],pin['sha256'])
        if json.loads(p.read_text()).get('status')!='FROZEN_COMPLETE':raise PermissionError('a final reader bank is not frozen')
    return lock


def report_worker(spec_path):
    active,cfg,verify=queue_config(spec_path,'attacks')
    import numpy as np
    from .attacks import score_bank,write_json
    reporting_lock(cfg,verify)
    receipt=json.loads(verify(cfg['freeze']['path'],cfg['freeze']['sha256']).read_text())
    if receipt.get('source_seed')!=cfg['seed'] or receipt.get('bindings',{}).get('arm_id')!=cfg['arm_id']:raise PermissionError('reporting arm/seed differs from reader freeze')
    for key,field in (('scores','score_sha256'),('release','inner_release_sha256')):
        verify(cfg[key]['path'],cfg[key]['sha256'])
        if cfg[key]['sha256']!=receipt[field]:raise PermissionError('reader freeze score/release differs')
    with np.load(cfg['release']['path'],allow_pickle=False) as z:release={k:z[k].copy() for k in z.files}
    # All final banks are verified before any reporting protected-label byte is read.
    sex=read_reporting_sex(cfg['inner_sex'],verify,release)
    with np.load(cfg['scores']['path'],allow_pickle=False) as z:scores={k:z[k].copy() for k in z.files}
    out={'schema':'TCR1-fresh-continuous-report-v1','arm_id':cfg['arm_id'],'seed':cfg['seed'],'unit_id':active['unit_id'],
         'freeze_sha256':cfg['freeze']['sha256'],'scores_sha256':cfg['scores']['sha256'],'reporting_lock_sha256':cfg['reporting_lock_sha256'],
         'point_eligible':cfg['point_eligible'],'diagnostic_only':not cfg['point_eligible'],**score_bank(scores,sex['sex'],sex['unit'])}
    if cfg.get('oracle_scores'):
        from .attacks import oracle_diagnostics
        verify(cfg['oracle_scores']['path'],cfg['oracle_scores']['sha256'])
        with np.load(cfg['oracle_scores']['path'],allow_pickle=False) as z:oracle={k:z[k].copy() for k in z.files}
        if not all(np.array_equal(oracle[k],sex[k]) for k in ('row_id','unit')):raise PermissionError('oracle task-label people differ from reporting role')
        out['secondary_oracle_diagnostics']=oracle_diagnostics(scores,sex['sex'],sex['unit'],oracle,oracle)
    if cfg.get('FIT_surrogate_diagnostic') is not None:
        out['surrogate_fresh_reader_comparison']={'frozen_FIT_diagnostic':cfg['FIT_surrogate_diagnostic'],'fresh_INNER_pair_AUC':out['views']['pair']['envelope_auc'],'interpretation':'different populations/statistics; fitting penalty is not an AUC certificate; no post-recovery selection'}
    write_json(cfg['public_output'],out,False);return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('fit','report'),default='fit');p.add_argument('--spec',required=True);a=p.parse_args()
    out=fit_worker(a.spec) if a.phase=='fit' else report_worker(a.spec)
    print(json.dumps({'status':out.get('status','REPORTED'),'arm_id':out['arm_id'],'seed':out.get('source_seed',out.get('seed'))}))
if __name__=='__main__':main()
