"""Queue-only opaque reporting-label custody after every scientific freeze."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
from zipfile import ZipFile


def run(spec_path):
    from .attack_worker import queue_config,reporting_lock
    active,cfg,verify=queue_config(spec_path,'attacks')
    lock=reporting_lock(cfg,verify)
    if lock.get('fitted_service_banks')!=26 or not lock.get('four_independent_refits_frozen') or not lock.get('all_release_and_reader_parameters_frozen_before_reporting_SEX_bytes'):raise PermissionError('complete service/refit freeze absent')
    refits=json.loads(verify(lock['independent_refits']['path'],lock['independent_refits']['sha256']).read_text())
    if refits.get('status')!='FROZEN_COMPLETE' or refits.get('numerical_fit_count')!=4:raise PermissionError('four independent refits not frozen')
    expected={(r['family'],r['seed']):r['model_sha256'] for r in refits['fits']}
    actual={(r['family'],r['seed']):r['sha256'] for r in cfg['refit_models']}
    if len(actual)!=4 or expected!=actual:raise PermissionError('refit model pins differ from global aggregate')
    for item in cfg['refit_models']:verify(item['path'],item['sha256'])
    verify(cfg['public_reporting_lock']['path'],cfg['public_reporting_lock']['sha256'])
    # Explicit copy reads follow the authenticated global freeze. The root
    # queue may already hash this registered source during admission, which
    # is authorized only after R00 sealed that same global freeze.
    source_pin=cfg['inner_sex_source'];src=verify(source_pin['path'],source_pin['sha256'])
    if source_pin['sha256']!=lock['inner_sex_sha256']:raise PermissionError('protected source hash differs from freeze')
    if '.private' not in src.parts or src.stat().st_mode&0o077:raise PermissionError('source label archive must be owner-only')
    with ZipFile(src) as archive:
        if len(archive.infolist())!=3 or {i.filename for i in archive.infolist()}!={'row_id.npy','unit.npy','sex.npy'}:raise PermissionError('source must be exactly one reporting role archive')
    dst=Path(cfg['inner_sex']['path'])
    if '.private' not in dst.resolve().parts or dst.exists() or dst.is_symlink():raise PermissionError('fresh owner-only reporting archive target required')
    dst.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    digest=hashlib.sha256()
    with src.open('rb') as original,os.fdopen(os.open(dst,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb') as target:
        for block in iter(lambda:original.read(1024*1024),b''):
            digest.update(block);target.write(block)
        target.flush();os.fsync(target.fileno())
    got=digest.hexdigest()
    if got!=cfg['inner_sex']['sha256']:raise PermissionError('opaque reporting copy hash differs')
    with dst.open('rb') as copied:restored=hashlib.file_digest(copied,'sha256').hexdigest()
    if restored!=got or dst.stat().st_mode&0o077:raise PermissionError('reporting copy hash/mode verification failed')
    receipt={'schema':'TCR1-reporting-label-opaque-custody-v1','status':'COPIED_VERIFIED','unit_id':active['unit_id'],'reporting_lock_sha256':cfg['reporting_lock_sha256'],'public_reporting_lock_sha256':cfg['public_reporting_lock']['sha256'],'inner_sex_sha256':got,'source_and_copy_exact':True,'owner_only':True,'archive_fields':['row_id','unit','sex'],'arrays_deserialized':False,'all_26_banks_oracle_and_four_refits_frozen_before_copy':True,'refit_model_sha256':sorted(actual.values())}
    from .attacks import write_json
    write_json(cfg['public_output'],receipt,False)
    private=Path(cfg['private_output']);private.parent.mkdir(parents=True,exist_ok=True)
    write_json(private,{**receipt,'source':str(src),'target':str(dst)})
    return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();run(a.spec);print(json.dumps({'status':'COPIED_VERIFIED','arrays_deserialized':False}))
if __name__=='__main__':main()
