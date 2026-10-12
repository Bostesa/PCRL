"""Metadata-only private reader job preparation; R00 remains sole executor."""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
CODE=('task_conditioned_representation/attacks.py','task_conditioned_representation/inference.py','task_conditioned_representation/attack_worker.py','task_conditioned_representation/source.py','task_conditioned_representation/scheduler.py','task_conditioned_representation/models.py','task_conditioned_representation/useful_service.py')


def pin(path):
    p=Path(path)
    if p.is_symlink():raise PermissionError('symlink refused')
    p=p.resolve(strict=True)
    with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(p),'sha256':sha}


def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
    return pin(p)


def prepare_fit(index_path,output):
    index=json.loads(Path(index_path).read_text());private=BASE/'.private/task_conditioned_representation/r5';public=BASE/'results/pcrl_task_conditioned_representations_v1/r5'
    manifest_pin=index['manifest'];manifest=json.loads(Path(manifest_pin['path']).read_text());lock_pin=index['attack_lock'];lock=json.loads(Path(lock_pin['path']).read_text())
    if lock.get('status')!='RELEASES_AND_ENDPOINTS_FROZEN':raise PermissionError('release stage lock required')
    metadata=[]
    for instance in index['instances']:
        arm=instance['arm_id'];seed=instance['seed'];uid=instance.get('unit_id',f'R5_ATTACK_{arm}_S{seed}_V1')
        cfg={'unit_id':uid,'arm_id':arm,'seed':seed,'manifest':manifest_pin['path'],'manifest_sha256':manifest_pin['sha256'],
             'checkpoint':instance['checkpoint'],'checkpoint_sha256':instance['checkpoint_sha256'],'science_lock':index['science_lock']['path'],'science_lock_sha256':index['science_lock']['sha256'],
             'attack_lock':lock_pin['path'],'attack_lock_sha256':lock_pin['sha256'],'private_output':str(private/'readers'/f'{arm}_s{seed}'),'public_output':str(public/f'{uid}_FREEZE.json')}
        config=write(private/'configs'/f'{uid}.json',cfg)
        source=[config,pin(instance['checkpoint']),manifest_pin,lock_pin,index['science_lock'],*[pin(BASE/k) for k in CODE]]
        for role,names in (('ATTACK_FIT_NEW',('features','sex')),('INNER_SELECTION',('features',))):source.extend(manifest['roles'][role][k] for k in names)
        by_path={}
        for item in source:
            actual=pin(item['path'])
            if actual['sha256']!=item['sha256']:raise PermissionError('source bytes changed before queue preparation')
            by_path[actual['path']]=actual['sha256']
        spec={'unit_id':uid,'stage':'attacks','cwd':str(BASE),'argv':[sys.executable,'-m','task_conditioned_representation.attack_worker','--phase','fit','--spec',config['path']],
              'max_wall_s':1200,'cpu_reserve_s':900,'max_rss_gib':2,'projected_output_bytes':128*1024**2,'sources':[{'path':p,'sha256':h} for p,h in sorted(by_path.items())],'deps':index.get('deps',[]),'outputs':[cfg['public_output']]}
        if index.get('amendment_sha256'):spec['amendment_sha256']=index['amendment_sha256']
        spec_pin=write(private/'specs'/f'{uid}.json',spec)
        metadata.append({'unit_id':uid,'config':config,'spec':spec_pin,'private_output':cfg['private_output'],'public_output':cfg['public_output']})
    return write(output,{'schema':'TCR1-prepared-reader-units-v1','units':metadata,'executed':False})


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(prepare_fit(a.index,a.output)))
if __name__=='__main__':main()
