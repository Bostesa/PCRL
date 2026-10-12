"""Prepare metadata-only deployment jobs. Coordinator owns queue execution."""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
PYTHON=sys.executable
CODE=('deployment.py','deployment_worker.py','source.py','scheduler.py','__init__.py','models.py','useful_service.py','kernels.py')
LEGACY_READER_CODE=('conditional_release/__init__.py','conditional_release/attacks.py','conditional_release/metrics.py','conditional_release/inference.py')


def pin(p):
    p=Path(p).resolve(strict=True)
    with p.open('rb') as f:return {'path':str(p),'sha256':hashlib.file_digest(f,'sha256').hexdigest()}


def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
    return pin(p)


def exact_sources(items):
    """Queue sources accept exactly path and SHA; admission sizes stay metadata."""
    out={}
    for item in items:
        path=str(Path(item['path']).resolve())
        if path in out and out[path]!=item['sha256']:raise ValueError('conflicting source pins')
        out[path]=item['sha256']
    return [{'path':p,'sha256':h} for p,h in sorted(out.items())]


def prepare_live(index_path,output):
    index=json.loads(Path(index_path).read_text());manifest_pin=index['manifest']
    if pin(manifest_pin['path'])['sha256']!=manifest_pin['sha256']:raise ValueError('source manifest binding differs')
    manifest=json.loads(Path(manifest_pin['path']).read_text());r7=BASE/'.private/task_conditioned_representation/r7';public=BASE/'results/pcrl_task_conditioned_representations_v1/r7';units=[]
    for item in index['instances']:
        arm=item['arm_id'];seed=item['seed'];tag=f'{arm}_s{seed}';uid=f'R7_DEPLOY_{arm}_S{seed}_V1';role='INNER_SELECTION'
        if pin(item['checkpoint'])['sha256']!=item['checkpoint_sha256']:raise ValueError('frozen selected checkpoint differs')
        cfg={'unit_id':uid,'operation':'export_live','stage':'verification','arm_id':arm,'checkpoint':item['checkpoint'],'checkpoint_sha256':item['checkpoint_sha256'],'source_manifest':manifest_pin['path'],'source_manifest_sha256':manifest_pin['sha256'],'seed':seed,'role':role,'package':str(r7/'packages'/tag),'input':str(r7/'inputs'/f'{tag}.npz'),'expected':str(r7/'expected'/f'{tag}.npz'),'public':str(public/f'{uid}.json'),'science_lock':index['science_lock'],'attack_lock':index['attack_lock'],'actual_model_point_eligible':item['actual_model_point_eligible'],'exact_source_alias':item.get('exact_source_alias',False),'historical_diagnostic_reference':item.get('historical_scope')=='additional_descriptive_not_primary'}
        for key in ('input','expected'):Path(cfg[key]).parent.mkdir(parents=True,exist_ok=True)
        config=write(r7/'configs'/f'{uid}.json',cfg)
        sources=[config,pin(item['checkpoint']),manifest_pin,index['science_lock'],index['attack_lock'],manifest['roles'][role]['features']]+[pin(BASE/'task_conditioned_representation'/name) for name in CODE]
        spec={'unit_id':uid,'stage':'verification','cwd':str(BASE),'argv':[index.get('python',PYTHON),'-m','task_conditioned_representation.deployment_worker',str(config['path'])],'max_wall_s':120,'cpu_reserve_s':120,'max_rss_gib':2,'projected_output_bytes':10000000,'sources':exact_sources(sources),'deps':index.get('deps',[]),'outputs':[cfg['public']]}
        if index.get('amendment_sha256'):spec['amendment_sha256']=index['amendment_sha256']
        spec_pin=write(r7/'specs'/f'{uid}.json',spec)
        units.append({'arm_id':arm,'seed':seed,'spec':spec_pin,'config':config,'package':cfg['package'],'input':cfg['input'],'expected':cfg['expected'],'public':cfg['public']})
    return write(output,{'schema':'TCR1-prepared-live-deployment-v1','units':units,'executed':False})


def prepare_restore(live_index,restore_root,output,tag='LOCAL'):
    index=json.loads(Path(live_index).read_text());root=Path(restore_root).resolve();r7=BASE/'.private/task_conditioned_representation/r7';public=BASE/'results/pcrl_task_conditioned_representations_v1/r7';units=[]
    def suffix(p,prefix):
        parts=Path(p).parts
        for pos in range(len(parts)-len(prefix)+1):
            if parts[pos:pos+len(prefix)]==prefix:return Path(*parts[pos+len(prefix):])
        raise PermissionError('metadata path lacks campaign namespace')
    def remap(p):return root/'files/campaign'/suffix(p,('.private','task_conditioned_representation'))
    def remap_public(p):return root/'files/code/results/pcrl_task_conditioned_representations_v1'/suffix(p,('results','pcrl_task_conditioned_representations_v1'))
    reader_index=root/'files/campaign/r5/PREPARED_ALL_READERS_V1.json'
    reader_metadata=json.loads(reader_index.read_text())
    code=root/'files/code';runtime_code=[code/'task_conditioned_representation'/name for name in CODE+('attacks.py','inference.py')]
    runtime_code.extend(code/name for name in LEGACY_READER_CODE)
    for item in index['units']:
        arm=item['arm_id'];seed=item['seed'];uid=f'R7_RESTORE_{tag}_{arm}_S{seed}_V1'
        receipt_path=remap_public(item['public']);receipt=json.loads(receipt_path.read_text());package=remap(item['package']);inp=remap(item['input']);expected=remap(item['expected'])
        cfg={'unit_id':uid,'operation':'restore','stage':'custody','restore_root':str(root),'arm_id':arm,'seed':seed,'package':str(package),'package_manifest_sha256':receipt['package_manifest_sha256'],'input':str(inp),'input_sha256':receipt['input_sha256'],'expected':str(expected),'expected_sha256':receipt['expected_sha256'],'public':str(public/f'{uid}.json'),'readers':[],'environment_pins':receipt['environment_pins']}
        choices=[record for record in reader_metadata['units'] if record['arm_id']==arm and record['seed']==seed]
        if len(choices)!=1:raise ValueError('actual restored reader bank metadata must uniquely match arm and seed')
        reader_dir=remap(choices[0]['private_output'])
        freeze=reader_dir/'FROZEN_COMPLETE.json'
        sources=[pin(receipt_path),pin(reader_index),pin(package/'DEPLOYMENT.json'),{'path':str(package/'model_state.pt'),'sha256':receipt['model_sha256']},{'path':str(inp),'sha256':receipt['input_sha256']},{'path':str(expected),'sha256':receipt['expected_sha256']}]+[pin(p) for p in runtime_code]
        if freeze.exists():
            frozen=json.loads(freeze.read_text());scores=reader_dir/'INNER_SCORES.npz'
            if frozen.get('status')!='FROZEN_COMPLETE' or frozen['source_seed']!=seed or frozen['bindings']['arm_id']!=arm:raise ValueError('reader bank freeze identity differs')
            report_path=root/'files/code/results/pcrl_task_conditioned_representations_v1/r5'/f'R5_REPORT_{arm}_S{seed}_V1.json'
            report=json.loads(report_path.read_text());sources.append(pin(report_path))
            for view in ('r1','r2','pair'):
                for name in dict.fromkeys(('LR_C1',report['views'][view]['strongest_reader'])):
                    actual_view=view;actual_name=name
                    if name.startswith('IGNORE_'):
                        ignored,actual_name=name.split('__',1);actual_view='r2' if ignored=='IGNORE_r1' else 'r1' if ignored=='IGNORE_r2' else None
                        if view!='pair' or actual_view is None:raise ValueError('invalid frozen ignore-other reader alias')
                    model=reader_dir/f'MODEL_{actual_view}__{actual_name}.joblib';model_sha=frozen['model_files'][model.name]
                    cfg['readers'].append({'path':str(model),'path_sha256':model_sha,'expected':str(scores),'expected_sha256':frozen['score_sha256'],'view':actual_view,'reader_name':actual_name,'orientation':frozen['validation'][actual_view][actual_name]['orientation'],'score_field':view+'__'+name,'reported_view':view,'reported_reader':name,'purpose':'representative_linear' if name=='LR_C1' else 'frozen_final_envelope_strongest'})
                    sources.append({'path':str(model),'sha256':model_sha})
            sources.extend((pin(freeze),{'path':str(scores),'sha256':frozen['score_sha256']}))
        else:raise FileNotFoundError('representative restored reader bank is required')
        config=write(r7/'configs'/f'{uid}.json',cfg);sources.append(config)
        spec={'unit_id':uid,'stage':'custody','cwd':str(code),'argv':[PYTHON,'-m','task_conditioned_representation.deployment_worker',str(config['path'])],'max_wall_s':120,'cpu_reserve_s':120,'max_rss_gib':2,'projected_output_bytes':1000000,'sources':exact_sources(sources),'deps':[],'outputs':[cfg['public']]}
        spec_pin=write(r7/'specs'/f'{uid}.json',spec);units.append({'arm_id':arm,'seed':seed,'spec':spec_pin,'public':cfg['public']})
    return write(output,{'schema':'TCR1-prepared-backup-only-numerical-restore-v1','units':units,'executed':False,'restore_code_only':True,'metadata_read_from_backup_only':True,'person_arrays_opened_or_hashed_by_builder':False,'reader_scope':'representative LR_C1 and frozen final-envelope strongest reader for every actual recipient and pair; no refitting or model selection change'})


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('live');a.add_argument('index');a.add_argument('output')
    a=sub.add_parser('restore');a.add_argument('index');a.add_argument('restore_root');a.add_argument('output');a.add_argument('--tag',default='LOCAL')
    a=p.parse_args();result=prepare_live(a.index,a.output) if a.action=='live' else prepare_restore(a.index,a.restore_root,a.output,a.tag);print(json.dumps(result))
