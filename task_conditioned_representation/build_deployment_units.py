"""Prepare metadata-only deployment jobs. Coordinator owns queue execution."""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
PYTHON=sys.executable
CODE=('deployment.py','deployment_worker.py','source.py','scheduler.py','__init__.py','models.py','useful_service.py')


def pin(p):
    p=Path(p).resolve(strict=True)
    with p.open('rb') as f:return {'path':str(p),'sha256':hashlib.file_digest(f,'sha256').hexdigest()}


def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
    return pin(p)


def prepare_live(index_path,output):
    index=json.loads(Path(index_path).read_text());manifest_pin=index['manifest']
    if pin(manifest_pin['path'])['sha256']!=manifest_pin['sha256']:raise ValueError('source manifest binding differs')
    manifest=json.loads(Path(manifest_pin['path']).read_text());r7=BASE/'.private/task_conditioned_representation/r7';public=BASE/'results/pcrl_task_conditioned_representations_v1/r7';units=[]
    for item in index['instances']:
        arm=item['arm_id'];seed=item['seed'];tag=f'{arm}_s{seed}';uid=f'R7_DEPLOY_{arm}_S{seed}_V1';role='INNER_SELECTION'
        if pin(item['checkpoint'])['sha256']!=item['checkpoint_sha256']:raise ValueError('frozen selected checkpoint differs')
        cfg={'operation':'export_live','stage':'verification','checkpoint':item['checkpoint'],'checkpoint_sha256':item['checkpoint_sha256'],'source_manifest':manifest_pin['path'],'source_manifest_sha256':manifest_pin['sha256'],'seed':seed,'role':role,'package':str(r7/'packages'/tag),'input':str(r7/'inputs'/f'{tag}.npz'),'expected':str(r7/'expected'/f'{tag}.npz'),'public':str(public/f'{uid}.json')}
        for key in ('input','expected'):Path(cfg[key]).parent.mkdir(parents=True,exist_ok=True)
        config=write(r7/'configs'/f'{uid}.json',cfg)
        sources=[config,pin(item['checkpoint']),manifest_pin,manifest['roles'][role]['features']]+[pin(BASE/'task_conditioned_representation'/name) for name in CODE]
        spec={'unit_id':uid,'stage':'verification','cwd':str(BASE),'argv':[index.get('python',PYTHON),'-m','task_conditioned_representation.deployment_worker',str(config['path'])],'max_wall_s':120,'cpu_reserve_s':120,'max_rss_gib':2,'projected_output_bytes':10000000,'sources':sources,'deps':index.get('deps',[]),'outputs':[cfg['public']]}
        if index.get('amendment_sha256'):spec['amendment_sha256']=index['amendment_sha256']
        spec_pin=write(r7/'specs'/f'{uid}.json',spec)
        units.append({'arm_id':arm,'seed':seed,'spec':spec_pin,'config':config,'package':cfg['package'],'input':cfg['input'],'expected':cfg['expected'],'public':cfg['public']})
    return write(output,{'schema':'TCR1-prepared-live-deployment-v1','units':units,'executed':False})


def prepare_restore(live_index,restore_root,output,tag='LOCAL'):
    index=json.loads(Path(live_index).read_text());root=Path(restore_root).resolve();r7=BASE/'.private/task_conditioned_representation/r7';public=BASE/'results/pcrl_task_conditioned_representations_v1/r7';units=[]
    def remap(p):return root/'files/campaign'/Path(p).resolve().relative_to(BASE/'.private/task_conditioned_representation')
    code=root/'files/code';runtime_code=[code/'task_conditioned_representation'/name for name in CODE+('attacks.py','inference.py')]
    for item in index['units']:
        arm=item['arm_id'];seed=item['seed'];uid=f'R7_RESTORE_{tag}_{arm}_S{seed}_V1'
        receipt=json.loads(Path(item['public']).read_text());package=remap(item['package']);inp=remap(item['input']);expected=remap(item['expected'])
        cfg={'operation':'restore','stage':'custody','restore_root':str(root),'package':str(package),'package_manifest_sha256':receipt['package_manifest_sha256'],'input':str(inp),'input_sha256':receipt['input_sha256'],'expected':str(expected),'expected_sha256':receipt['expected_sha256'],'public':str(public/f'{uid}.json'),'readers':[]}
        reader_dir=root/'files/campaign/r5/readers'/f'{arm}_s{seed}'
        freeze=reader_dir/'FROZEN_COMPLETE.json'
        sources=[pin(package/'DEPLOYMENT.json'),pin(package/'model_state.pt'),pin(inp),pin(expected)]+[pin(p) for p in runtime_code]
        if freeze.exists():
            frozen=json.loads(freeze.read_text());scores=reader_dir/'INNER_SCORES.npz'
            for view in ('r1','r2','pair'):
                name='LR_C1';model=reader_dir/f'MODEL_{view}__{name}.joblib'
                cfg['readers'].append({'path':str(model),'path_sha256':pin(model)['sha256'],'expected':str(scores),'expected_sha256':pin(scores)['sha256'],'view':view,'reader_name':name,'orientation':frozen['validation'][view][name]['orientation'],'score_field':view+'__'+name})
                sources.append(pin(model))
            sources.extend((pin(freeze),pin(scores)))
        else:raise FileNotFoundError('representative restored reader bank is required')
        config=write(r7/'configs'/f'{uid}.json',cfg);sources.append(config)
        spec={'unit_id':uid,'stage':'custody','cwd':str(code),'argv':[PYTHON,'-m','task_conditioned_representation.deployment_worker',str(config['path'])],'max_wall_s':120,'cpu_reserve_s':120,'max_rss_gib':2,'projected_output_bytes':1000000,'sources':sources,'deps':[],'outputs':[cfg['public']]}
        spec_pin=write(r7/'specs'/f'{uid}.json',spec);units.append({'arm_id':arm,'seed':seed,'spec':spec_pin,'public':cfg['public']})
    return write(output,{'schema':'TCR1-prepared-backup-only-numerical-restore-v1','units':units,'executed':False,'restore_code_only':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('live');a.add_argument('index');a.add_argument('output')
    a=sub.add_parser('restore');a.add_argument('index');a.add_argument('restore_root');a.add_argument('output');a.add_argument('--tag',default='LOCAL')
    a=p.parse_args();result=prepare_live(a.index,a.output) if a.action=='live' else prepare_restore(a.index,a.restore_root,a.output,a.tag);print(json.dumps(result))
