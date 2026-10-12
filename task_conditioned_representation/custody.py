"""Versioned safe-source and campaign-private snapshot, uncached restore checks.

This module copies opaque admitted files. It never deserializes person arrays
or calls the inherited all-role source inventory. All heavy work uses queue.
"""
from __future__ import annotations
import argparse,fcntl,hashlib,json,os,shutil,subprocess,sys
from importlib.metadata import version,PackageNotFoundError
from pathlib import Path
from .source import admitted_path,load_manifest,write_json

def write_once(path,value,private=True):return write_json(path,value,private)
from .scheduler import assert_active_from_env


def uncached_hash(path):
    path=admitted_path(path)
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        if sys.platform=='darwin':fcntl.fcntl(stream.fileno(),fcntl.F_NOCACHE,1)
        for chunk in iter(lambda:stream.read(1024**2),b''):digest.update(chunk)
    return digest.hexdigest()


def safe_alias(alias):
    p=Path(alias)
    if p.is_absolute() or '..' in p.parts or '\\' in alias or str(p)!=alias:raise PermissionError('unsafe custody alias')
    return p


def create_plan(repo,source_manifest,out,public,supplemental_metadata=()):
    active=assert_active_from_env(stage='custody')
    # A running unit's stdout and final receipt are still mutable. Only
    # registered terminal unit logs can enter an immutable copy plan.
    state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text())
    terminal_units={uid for uid,unit in state['units'].items()
                    if unit.get('status') in ('done','failed','incomplete')
                    and uid!=active['unit_id']}
    repo=Path(repo).resolve();source_manifest=Path(source_manifest).resolve()
    manifest=load_manifest(source_manifest);entries={};source_alias={}
    def add(alias,path,expected=None):
        safe_alias(alias);p=admitted_path(path).absolute()
        if not p.is_file():raise FileNotFoundError('missing custody input')
        sha=uncached_hash(p)
        if expected and sha!=expected:raise PermissionError('custody source pin differs')
        entries[alias]={'alias':alias,'path':str(p),'sha256':sha,'size_bytes':p.stat().st_size}
        return alias
    def walk(value,prefix):
        if isinstance(value,dict):
            if 'path' in value and 'sha256' in value:
                p=value['path']
                # Preserve actual source sibling filenames for predecessor relative loaders.
                parent_id=hashlib.sha256(str(Path(p).parent).encode()).hexdigest()[:24]
                alias='source/parents/'+parent_id+'/'+Path(p).name
                if p not in source_alias:source_alias[p]=add(alias,p,value['sha256'])
            else:
                for name,item in value.items():walk(item,prefix+'/'+name if prefix else name)
        elif isinstance(value,list):
            for index,item in enumerate(value):walk(item,prefix+'/'+str(index))
    walk(manifest,'')
    supplements=[]
    for index,path in enumerate(supplemental_metadata):
        # A registered metadata index admits exact existing pins only. Do not
        # recurse into any referenced predecessor directory or source file.
        metadata=json.loads(admitted_path(path).read_text())
        walk(metadata,'supplement/'+str(index));add('source/SUPPLEMENTAL_METADATA_'+str(index)+'.original.json',path)
        supplements.append(metadata)
    add('source/SOURCE_PATHS.original.json',source_manifest)
    # Only this campaign's private artifacts, never predecessor directory walks.
    private=repo/'.private/task_conditioned_representation'
    for p in sorted(private.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(private)
        if rel.parts[0]=='custody' or rel.parts[:2]==('r7','custody'):continue
        if rel.parts[0]=='compute':
            if len(rel.parts)<4 or rel.parts[1]!='units' or rel.parts[2] not in terminal_units:continue
        add('campaign/'+str(rel),p)
    files=subprocess.run(['git','ls-files','-z'],cwd=repo,check=True,capture_output=True).stdout.split(b'\0')
    tracked={os.fsdecode(raw) for raw in files if raw}
    required={'conditional_release/'+name for name in ('__init__.py','attacks.py','metrics.py','inference.py')}
    required.update(('conditional_release/custody.py','task_conditioned_representation/requirements-deployment.txt'))
    for namespace in ('task_conditioned_representation','task_conditioned_representation_independent'):
        required.update(str(path.relative_to(repo)) for path in (repo/namespace).rglob('*.py'))
    if required-tracked:raise PermissionError('runtime/helper closure has uncommitted files: '+','.join(sorted(required-tracked)))
    if subprocess.run(['git','diff','--quiet','HEAD','--'],cwd=repo,check=False).returncode!=0:raise PermissionError('commit tracked changes before immutable custody capture')
    for raw in files:
        if not raw:continue
        relative=os.fsdecode(raw);p=repo/relative
        if p.is_file():add('code/'+relative,p)
    head=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,check=True,capture_output=True,text=True).stdout.strip()
    environment={}
    for package in ('numpy','torch','scipy','scikit-learn','joblib','psutil'):
        try:environment[package]=version(package)
        except PackageNotFoundError:environment[package]=None
    plan={'schema':'TCR1-versioned-custody-plan-v1','execution_commit_captured':head,'source_manifest_sha256':uncached_hash(source_manifest),
          'entries':list(entries.values()),'source_aliases':source_alias,'source_manifest':manifest,
          'supplemental_metadata':supplements,
          'environment_pins':environment,'python_version':sys.version.split()[0],
          'helper_closure':[{'path':name,'sha256':entries['code/'+name]['sha256']} for name in sorted(required)],
          'adapter':{'source':'conditional_release/custody.py','source_sha256':uncached_hash(repo/'conditional_release/custody.py'),'namespace':'task_conditioned_representation','layout':'same original parent pins remain adjacent; correction T applied prospectively'},'scope':'admitted role-only sources and this campaign private artifacts; only registered terminal queue-unit logs; opaque copy; no barred archive opened'}
    sha=write_once(out,plan)
    report={k:v for k,v in plan.items() if k not in ('entries','source_aliases','source_manifest','supplemental_metadata')}
    report['supplemental_metadata_count']=len(supplements)
    report['entries']=[{k:v for k,v in e.items() if k!='path'} for e in entries.values()]
    report['private_plan_sha256']=sha;report['bytes']=sum(x['size_bytes'] for x in entries.values());report['file_count']=len(entries)
    write_once(public,report,False)
    return {'plan_sha256':sha,'files':len(entries),'bytes':report['bytes'],'captured_commit':head}


def copy_pinned(source,target,sha,size):
    source=admitted_path(source);target=Path(target)
    if source.stat().st_size!=size or uncached_hash(source)!=sha:raise PermissionError('source changed before copy')
    target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with source.open('rb') as src,os.fdopen(os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb') as dst:
        if sys.platform=='darwin':fcntl.fcntl(src.fileno(),fcntl.F_NOCACHE,1)
        shutil.copyfileobj(src,dst,1024**2);dst.flush();os.fsync(dst.fileno())
    if target.stat().st_size!=size or uncached_hash(target)!=sha:raise PermissionError('copied artifact differs')


def snapshot(repo,plan_path,plan_sha,target,public):
    assert_active_from_env(stage='custody')
    if uncached_hash(plan_path)!=plan_sha:raise PermissionError('plan differs')
    plan=json.loads(Path(plan_path).read_text());target=Path(target)
    if target.exists():raise FileExistsError('versioned new snapshot target required')
    target.mkdir(parents=True,mode=0o700)
    for e in plan['entries']:copy_pinned(e['path'],target/'files'/safe_alias(e['alias']),e['sha256'],e['size_bytes'])
    copy_pinned(plan_path,target/'PLAN.private.json',plan_sha,Path(plan_path).stat().st_size)
    bundle=target/'execution.gitbundle'
    env=dict(os.environ);env.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    head=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,env=env,check=True,capture_output=True,text=True).stdout.strip()
    if head!=plan['execution_commit_captured']:raise PermissionError('HEAD advanced after custody plan; preserve partial and prepare a new version')
    subprocess.run(['git','-c','pack.threads=1','bundle','create',str(bundle),'HEAD'],cwd=repo,env=env,check=True,capture_output=True)
    bundle_sha=uncached_hash(bundle)
    subprocess.run(['git','bundle','verify',str(bundle)],cwd=repo,env=env,check=True,capture_output=True)
    receipt={'schema':'TCR1-versioned-snapshot-receipt-v1','status':'PASS','plan_sha256':plan_sha,
             'execution_commit_captured':plan['execution_commit_captured'],'entries':[{k:v for k,v in e.items() if k!='path'} for e in plan['entries']],
             'bundle_sha256':bundle_sha,'bundle_size_bytes':bundle.stat().st_size,'uncached_verification':True,'same_device':True}
    write_once(target/'SNAPSHOT_RECEIPT.json',receipt)
    write_once(public,receipt,False)
    return {'status':'PASS','files':len(plan['entries']),'bundle_sha256':bundle_sha}


def restore(snapshot_path,target,public):
    assert_active_from_env(stage='custody')
    snapshot_path=Path(snapshot_path);target=Path(target)
    if target.exists():raise FileExistsError('new restore target required')
    receipt=json.loads((snapshot_path/'SNAPSHOT_RECEIPT.json').read_text())
    plan_path=snapshot_path/'PLAN.private.json'
    if uncached_hash(plan_path)!=receipt['plan_sha256']:raise PermissionError('snapshot plan differs')
    plan=json.loads(plan_path.read_text());target.mkdir(parents=True,mode=0o700)
    for e in receipt['entries']:copy_pinned(snapshot_path/'files'/safe_alias(e['alias']),target/'files'/safe_alias(e['alias']),e['sha256'],e['size_bytes'])
    copy_pinned(snapshot_path/'execution.gitbundle',target/'execution.gitbundle',receipt['bundle_sha256'],receipt['bundle_size_bytes'])
    def remap(value):
        if isinstance(value,dict):
            if 'path' in value and 'sha256' in value:
                result=dict(value);result['path']=str(target/'files'/plan['source_aliases'][value['path']]);return result
            return {k:remap(v) for k,v in value.items()}
        if isinstance(value,list):return [remap(v) for v in value]
        return value
    runtime=remap(plan['source_manifest']);runtime_sha=write_once(target/'runtime/SOURCE_PATHS.json',runtime)
    supplemental_hashes=[]
    for index,metadata in enumerate(plan.get('supplemental_metadata',[])):
        supplemental_hashes.append(write_once(target/'runtime'/f'SUPPLEMENTAL_SOURCES_{index}.json',remap(metadata)))
    result={'schema':'TCR1-snapshot-only-restore-receipt-v1','status':'PASS','file_count':len(receipt['entries']),
            'source_plan_sha256':receipt['plan_sha256'],'runtime_source_manifest_sha256':runtime_sha,
            'runtime_supplemental_metadata_sha256':supplemental_hashes,
            'execution_commit_captured':receipt['execution_commit_captured'],'uncached_verification':True,
            'scope':'opaque copy and exact bytes only; numerical replay is separately queued'}
    write_once(public,result,False);return result


def off_device(snapshot_path,target,public):
    assert_active_from_env(stage='custody')
    source=Path(snapshot_path);target=Path(target)
    if target.exists():raise FileExistsError('new versioned off-device destination required')
    ancestor=target.parent
    while not ancestor.exists():ancestor=ancestor.parent
    if source.stat().st_dev==ancestor.stat().st_dev:raise PermissionError('off-device target must be a different mounted filesystem')
    target.mkdir(parents=True,mode=0o700)
    entries=[]
    for path in sorted(source.rglob('*')):
        if not path.is_file():continue
        relative=path.relative_to(source)
        sha=uncached_hash(path);size=path.stat().st_size
        copy_pinned(path,target/relative,sha,size)
        entries.append({'alias':str(relative),'sha256':sha,'size_bytes':size})
    result={'schema':'TCR1-off-device-custody-receipt-v1','status':'PASS','separate_filesystem_verified':True,
            'uncached_full_readback':True,'file_count':len(entries),'bytes':sum(x['size_bytes'] for x in entries),
            'entries':entries,'source_snapshot_receipt_sha256':uncached_hash(source/'SNAPSHOT_RECEIPT.json'),
            'destination_snapshot_receipt_sha256':uncached_hash(target/'SNAPSHOT_RECEIPT.json'),
            'scope':'separate-filesystem byte copy and uncached verification; representative numerical restore is a separate unit'}
    write_once(public,result,False);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('plan');a.add_argument('--repo',required=True);a.add_argument('--source-manifest',required=True);a.add_argument('--out',required=True);a.add_argument('--public',required=True);a.add_argument('--supplemental-metadata',action='append',default=[])
    a=sub.add_parser('snapshot');a.add_argument('--repo',required=True);a.add_argument('--plan',required=True);a.add_argument('--plan-sha256',required=True);a.add_argument('--target',required=True);a.add_argument('--public',required=True)
    a=sub.add_parser('restore');a.add_argument('--snapshot',required=True);a.add_argument('--target',required=True);a.add_argument('--public',required=True)
    a=sub.add_parser('off-device');a.add_argument('--snapshot',required=True);a.add_argument('--target',required=True);a.add_argument('--public',required=True)
    a=p.parse_args()
    if a.action=='plan':z=create_plan(a.repo,a.source_manifest,a.out,a.public,a.supplemental_metadata)
    elif a.action=='snapshot':z=snapshot(a.repo,a.plan,a.plan_sha256,a.target,a.public)
    elif a.action=='restore':z=restore(a.snapshot,a.target,a.public)
    else:z=off_device(a.snapshot,a.target,a.public)
    print(json.dumps(z))
