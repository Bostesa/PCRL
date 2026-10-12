"""Metadata-only training job preparation. The coordinator alone runs the queue.

No torch/numpy imports, dataset deserialization, fitting, or scientific selection.
Generated paths and source manifests stay in private campaign custody.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
PRIVATE=BASE/'.private/task_conditioned_representation/r3'
PUBLIC=BASE/'results/pcrl_task_conditioned_representations_v1/r3'
CODE=('task_conditioned_representation/__init__.py','task_conditioned_representation/train_worker.py','task_conditioned_representation/trainer.py','task_conditioned_representation/models.py','task_conditioned_representation/useful_service.py','task_conditioned_representation/source.py','task_conditioned_representation/objectives.py','task_conditioned_representation/kernels.py','task_conditioned_representation/controls.py','task_conditioned_representation/scheduler.py')
UTILITY=('U_NEW','U_TASKPOWER')
CALIBRATION=('JOINT_CONDITIONAL','LOCAL_CONDITIONAL','RAW_JOINT')
PROTECTED=('JOINT_CONDITIONAL','LOCAL_TWIN','LOCAL_CONDITIONAL','JOINT_UNCONDITIONAL','RAW_JOINT','SEQ12','SEQ21')


def pin(path,expected=None):
    p=Path(path)
    if p.is_symlink():raise PermissionError('symlink refused')
    p=p.resolve(strict=True)
    with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    if expected and sha!=expected:raise PermissionError('metadata source pin differs: '+p.name)
    return {'path':str(p),'sha256':sha}


def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    return pin(p)


def read(item):return json.loads(Path(pin(item['path'],item['sha256'])['path']).read_text())


def recipe(lock):
    value=dict(lock['recipe'])
    if 'gradient_clip' not in value or 'clip' in value:raise ValueError('science recipe clipping alias differs')
    value['clip']=value.pop('gradient_clip')
    fields={'epochs','checkpoints','batch_size','learning_rate','weight_decay','clip','reference_size','gradient_log_every'}
    if set(value)!=fields:raise ValueError('unknown or missing recipe fields')
    if value['epochs']!=20 or value['checkpoints']!=[0,5,10,20]:raise ValueError('registered continuation opportunity differs')
    return value


def metadata_sources(manifest_pin,science_pin,amendment_pins,seed):
    manifest=read(manifest_pin);lock=read(science_pin);source=[manifest_pin,science_pin,*amendment_pins]
    source.extend(pin(BASE/name) for name in CODE)
    source.append(pin(BASE/'task_conditioned_representation/build_training_units.py'))
    source.extend(manifest['roles']['OSF_DEFENSE_FIT'][name] for name in ('features','income','occupation','sex'))
    seed_source=manifest['seeds'][str(seed)]
    source.extend(seed_source['pins'].values());source.append(seed_source['public_bank'])
    # The source helper may receive a dated amendment, but the held scientific
    # training/selection implementations must retain their science-lock hashes.
    expected={row['path']:row['sha256'] for row in lock['source_code']}
    for name in ('task_conditioned_representation/train_worker.py','task_conditioned_representation/trainer.py','task_conditioned_representation/models.py','task_conditioned_representation/controls.py','task_conditioned_representation/objectives.py','task_conditioned_representation/kernels.py'):
        pin(BASE/name,expected[name])
    unique={}
    for item in source:
        actual=pin(item['path'],item['sha256']);unique[actual['path']]=actual
    return list(unique.values())


def latest_amendment(stage,pins):
    # Ordered pins supplied by R00 represent registered amendments. Stage-specific
    # queue admission is checked again by the coordinator, never mutated here.
    if not pins:raise ValueError('registered science/latest amendment pins required')
    applicable=[]
    for item in pins:
        content=read(item)
        stages=content.get('affected_stages',content.get('stages'))
        if stages is None or stage in stages:applicable.append(item)
    if not applicable:raise ValueError('no applicable stage amendment')
    return applicable[-1]['sha256']


def unit(cfg,sources,deps,amendments,python,cap):
    uid=cfg['unit_id'];config=write(PRIVATE/'configs'/f'{uid}.json',cfg)
    by_path={p['path']:p for p in sources+[config]}
    stage='admission' if cfg['operation']=='calibrate' else 'protection'
    spec={'unit_id':uid,'stage':stage,'cwd':str(BASE),'argv':[python,'-m','task_conditioned_representation.train_worker','--spec',config['path']],
          'max_wall_s':cap,'cpu_reserve_s':cap,'max_rss_gib':2,'projected_output_bytes':32*1024**2,
          'sources':list(by_path.values()),'deps':deps,'outputs':[cfg['output']],
          'amendment_sha256':latest_amendment(stage,amendments)}
    spec_pin=write(PRIVATE/'specs'/f'{uid}.json',spec)
    return {'unit_id':uid,'family':cfg['family'],'seed':cfg['seed'],'strength':cfg.get('strength'),
            'config':config,'spec':spec_pin,'output':cfg['output'],'private_output':cfg.get('private_output'),'deps':deps}


def common_cfg(manifest,science,seed,family,operation,uid):
    return {'unit_id':uid,'operation':operation,'family':family,'seed':seed,
            'manifest':manifest['path'],'manifest_sha256':manifest['sha256'],
            'science_lock':science['path'],'science_lock_sha256':science['sha256'],
            'recipe':recipe(read(science))}


def prepare(manifest,science,amendments,output,python=sys.executable):
    lock=read(science)
    if manifest['sha256']!=lock['source_manifest_sha256']:raise PermissionError('science source manifest differs')
    units=[]
    for family in UTILITY:
        for seed in (0,1):
            uid=f'R3_FIT_{family}_S{seed}_V1';cfg=common_cfg(manifest,science,seed,family,'fit',uid)
            cfg.update(coefficient=0.,strength=0.,target_enabled=family=='U_TASKPOWER',private_output=str(PRIVATE/'fits'/f'{family}_s{seed}'),output=str(PUBLIC/f'{uid}.json'))
            units.append(unit(cfg,metadata_sources(manifest,science,amendments,seed),[],amendments,python,600))
    for family in CALIBRATION:
        for seed in (0,1):
            uid=f'R3_CAL_{family}_S{seed}_V1';cfg=common_cfg(manifest,science,seed,family,'calibrate',uid)
            cfg.update(private_output=str(PRIVATE/'calibration'/f'{family}_s{seed}'),output=str(PUBLIC/f'{uid}.json'))
            units.append(unit(cfg,metadata_sources(manifest,science,amendments,seed),[],amendments,python,300))
    return write(output,{'schema':'TCR1-prepared-initial-training-v1','manifest':manifest,'science_lock':science,'amendments':amendments,'python':python,'units':units,'executed':False,'utility_workers':4,'calibration_workers':6,'protection_workers_pending':42})


def protect(initial_index,calibration_index,amendments,output):
    initial=read(initial_index);results=read(calibration_index)
    manifest=initial['manifest'];science=initial['science_lock'];lock=read(science)
    calibrations={}
    for entry in results['calibrations']:
        payload=read(entry['result']);identity=(payload['family'],int(payload['seed']))
        if identity in calibrations or identity[0] not in CALIBRATION or identity[1] not in (0,1):raise ValueError('duplicate or unknown normalization')
        calibrations[identity]=(entry,payload)
    if set(calibrations)!={(f,s) for f in CALIBRATION for s in (0,1)}:raise ValueError('six source-seed normalizations required')
    units=[];unsupported=[]
    for family in PROTECTED:
        reference=family if family in ('LOCAL_CONDITIONAL','RAW_JOINT') else 'JOINT_CONDITIONAL'
        for seed in (0,1):
            entry,cal=calibrations[(reference,seed)]
            if not cal['supported']:
                unsupported.append({'family':family,'seed':seed,'normalization_family':reference,'result':entry['result'],'reason':cal['reason']});continue
            if cal.get('draws')!=16 or cal.get('initialization')!='exact_folded_U0_epoch0':raise ValueError('normalization scope differs')
            for strength in lock['strengths']:
                coefficient=float(strength)*float(cal['coefficient_per_ratio'])
                if not 0<coefficient<=lock['coefficient_ceiling']:raise ValueError('derived coefficient exceeds registered ceiling')
                label={.3:'R03',1.:'R1',3.:'R3'}[float(strength)];uid=f'R3_FIT_{family}_{label}_S{seed}_V1'
                cfg=common_cfg(manifest,science,seed,family,'fit',uid)
                cfg.update(coefficient=coefficient,strength=strength,target_enabled=family!='RAW_JOINT',normalization_result=entry['result'],
                           private_output=str(PRIVATE/'fits'/f'{family}_{label}_s{seed}'),output=str(PUBLIC/f'{uid}.json'))
                sources=metadata_sources(manifest,science,amendments,seed)+[entry['result'],initial_index,calibration_index]
                if family=='RAW_JOINT':
                    warm=entry['warm_checkpoint'];pin(warm['path'],warm['sha256'])
                    if warm['sha256']!=cal['warm_checkpoint_sha256']:raise ValueError('RAW warm state differs from normalization')
                    cfg.update(raw_warm_checkpoint=warm['path'],raw_warm_sha256=warm['sha256']);sources.append(warm)
                units.append(unit(cfg,sources,[entry['unit_id']],amendments,initial['python'],900 if family.startswith('SEQ') else 600))
    return write(output,{'schema':'TCR1-prepared-protection-training-v1','manifest':manifest,'science_lock':science,'amendments':amendments,
        'initial_index':initial_index,'calibration_index':calibration_index,'units':units,'unsupported':unsupported,'executed':False,
        'complete_if_normalizations_supported':len(units)==42,'maximum_protection_workers':42})


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='phase',required=True)
    first=sub.add_parser('prepare');first.add_argument('--manifest',required=True);first.add_argument('--manifest-sha256',required=True);first.add_argument('--science-lock',required=True);first.add_argument('--science-lock-sha256',required=True);first.add_argument('--python',default=sys.executable)
    second=sub.add_parser('protect');second.add_argument('--initial-index',required=True);second.add_argument('--initial-index-sha256',required=True);second.add_argument('--calibration-index',required=True);second.add_argument('--calibration-index-sha256',required=True)
    for p in (first,second):
        p.add_argument('--amendment',action='append',required=True);p.add_argument('--amendment-sha256',action='append',required=True);p.add_argument('--output',required=True)
    a=ap.parse_args()
    if len(a.amendment)!=len(a.amendment_sha256):raise ValueError('one hash per ordered amendment required')
    amendments=[pin(p,h) for p,h in zip(a.amendment,a.amendment_sha256)]
    if a.phase=='prepare':result=prepare(pin(a.manifest,a.manifest_sha256),pin(a.science_lock,a.science_lock_sha256),amendments,a.output,a.python)
    else:result=protect(pin(a.initial_index,a.initial_index_sha256),pin(a.calibration_index,a.calibration_index_sha256),amendments,a.output)
    print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
