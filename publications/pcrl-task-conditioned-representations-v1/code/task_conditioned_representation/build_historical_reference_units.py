"""Metadata-only preparation of additive historical restoration and custody pins."""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
PRIVATE=BASE/'.private/task_conditioned_representation/r1'
PUBLIC=BASE/'results/pcrl_task_conditioned_representations_v1/r1'
CODE=('task_conditioned_representation/__init__.py','task_conditioned_representation/scheduler.py','task_conditioned_representation/historical_reference_worker.py','task_conditioned_representation/source.py','task_conditioned_representation/useful_service.py','task_conditioned_representation/models.py','task_conditioned_representation/build_historical_reference_units.py')


def pin(path,expected=None):
    p=Path(path)
    if p.is_symlink():raise PermissionError('symlink refused')
    p=p.resolve(strict=True)
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    if expected and h!=expected:raise PermissionError('historical preparation pin differs')
    return {'path':str(p),'sha256':h}


def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    return pin(p)


def prepare(current_pin,historical_pin,scope_pin,science_pin,python,output):
    for p in (current_pin,historical_pin,scope_pin,science_pin):pin(p['path'],p['sha256'])
    current=json.loads(Path(current_pin['path']).read_text());hist=json.loads(Path(historical_pin['path']).read_text());scope=json.loads(Path(scope_pin['path']).read_text())
    if scope.get('status')!='FROZEN' or scope.get('arm_id')!='H_RAW_CSTAR' or scope.get('K_primary')!=51 or scope.get('primary_pool_changed') is not False or scope.get('source_manifest_sha256')!=historical_pin['sha256']:raise PermissionError('historical F scope differs')
    units=[]
    for seed in (0,1):
        uid=f'R1_HISTORICAL_CSTAR_S{seed}_RESTORE_V1';directory=PRIVATE/'historical'/f'H_RAW_CSTAR_s{seed}';receipt=PUBLIC/f'HISTORICAL_CSTAR_S{seed}_RESTORATION.json'
        cfg={'unit_id':uid,'arm_id':'H_RAW_CSTAR','seed':seed,'manifest':historical_pin['path'],'manifest_sha256':historical_pin['sha256'],'scope_lock':scope_pin['path'],'scope_lock_sha256':scope_pin['sha256'],'science_lock':science_pin['path'],'science_lock_sha256':science_pin['sha256'],'private_output':str(directory),'public_output':str(receipt)}
        cp=write(PRIVATE/'historical_configs'/f'{uid}.json',cfg)
        sources=[historical_pin,scope_pin,science_pin,cp,*[pin(BASE/n) for n in CODE]]
        record=hist['seeds'][str(seed)];sources.extend(record['pins'].values());sources.append(record['public_bank']);sources.extend(record['ucal'].values());sources.append(hist['historical_contract']['historical_role_manifest'])
        sources.extend(hist['roles'][r]['features'] for r in ('OSF_DEFENSE_FIT','ATTACK_FIT_NEW','INNER_SELECTION'))
        actual={}
        for p in sources:
            q=pin(p['path'],p['sha256']);actual[q['path']]=q
        spec={'unit_id':uid,'stage':'admission','cwd':str(BASE),'argv':[python,'-m','task_conditioned_representation.historical_reference_worker','--spec',cp['path']],'max_wall_s':300,'cpu_reserve_s':300,'max_rss_gib':2,'projected_output_bytes':4*1024**2,'sources':list(actual.values()),'deps':[],'outputs':[str(receipt),str(directory/'epoch_0.pt'),str(directory/'RESTORATION_RECEIPT.json')],'amendment_sha256':scope_pin['sha256']}
        sp=write(PRIVATE/'historical_specs'/f'{uid}.json',spec);units.append({'unit_id':uid,'seed':seed,'config':cp,'spec':sp,'checkpoint':str(directory/'epoch_0.pt'),'public_receipt':str(receipt)})
    extra=dict(current);extra['historical_sources']={'H_RAW_CSTAR':hist['seeds']};extra['historical_reference_admission']={'manifest':historical_pin,'scope_amendment':scope_pin,'source_contract_metadata':hist['historical_contract']['historical_role_manifest'],'scope':'additive descriptive reference; primary K51 unchanged'}
    custody=write(PRIVATE/'SOURCE_PATHS_WITH_HISTORICAL_REFERENCE.json',extra)
    out={'schema':'TCR1-historical-reference-restoration-unit-index-v1','scope_lock_sha256':scope_pin['sha256'],'historical_manifest_sha256':historical_pin['sha256'],'current_manifest_sha256':current_pin['sha256'],'units':units,'extended_custody_manifest':custody,'prepared_person_or_label_array_deserializations':0,'prepared_label_array_hashes':0,'numerical_jobs_launched':0,'primary_pool_changed':False}
    op=write(output,out);return {'output':op,'units':units,'extended_custody_manifest':custody}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--current-manifest',required=True);p.add_argument('--current-manifest-sha256',required=True);p.add_argument('--historical-manifest',required=True);p.add_argument('--historical-manifest-sha256',required=True);p.add_argument('--scope-lock',required=True);p.add_argument('--scope-lock-sha256',required=True);p.add_argument('--science-lock',required=True);p.add_argument('--science-lock-sha256',required=True);p.add_argument('--python',default=sys.executable);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(prepare(pin(a.current_manifest,a.current_manifest_sha256),pin(a.historical_manifest,a.historical_manifest_sha256),pin(a.scope_lock,a.scope_lock_sha256),pin(a.science_lock,a.science_lock_sha256),a.python,a.output)))
