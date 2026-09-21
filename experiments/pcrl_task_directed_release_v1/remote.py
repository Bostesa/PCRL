"""Task-owned AWS orchestration; never selects unrelated instances/resources."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
from .config import ROOT,OUT,STUDY


def aws(*args):
    p=subprocess.run(['aws','--profile','vein','--region','us-east-1',*args,'--output','json'],
                     capture_output=True,text=True)
    if p.returncode:raise RuntimeError(p.stderr)
    return json.loads(p.stdout) if p.stdout.strip() else {}


def resources():
    record=json.loads((OUT/'private/AWS_RESOURCES.json').read_text())
    iid=record['instance_id']
    info=aws('ec2','describe-instances','--instance-ids',iid)['Reservations'][0]['Instances'][0]
    if not any(t['Key']=='study' and t['Value']==STUDY for t in info['Tags']):
        raise PermissionError('Refusing an instance not tagged to this task')
    return record


def send(commands,description):
    r=resources()
    res=aws('ssm','send-command','--instance-ids',r['instance_id'],'--document-name','AWS-RunShellScript',
            '--parameters',json.dumps({'commands':commands,'executionTimeout':['3600']}),
            '--comment',description[:100])
    cid=res['Command']['CommandId']
    p=OUT/'private/remote_commands.jsonl'
    with p.open('a') as f:f.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                            'id':cid,'description':description,'commands':commands})+'\n')
    return {'command_id':cid,'description':description}


def get(cid):
    r=resources()
    reply=aws('ssm','get-command-invocation','--command-id',cid,'--instance-id',r['instance_id'])
    return {k:reply.get(k) for k in ('Status','ResponseCode','StandardOutputContent','StandardErrorContent')}


def stage(kind):
    r=resources();private=OUT/'private';staging=private/'staging';staging.mkdir(exist_ok=True)
    if kind=='inputs':
        files=sorted((private/'inputs').rglob('*'))
        files=[p for p in files if p.is_file()]
    else:
        # Ship committed study files only; independent agents may be writing
        # unrelated, unfinished reporting/test modules in this worktree.
        tracked=subprocess.run(['git','ls-files','-z','--',f'experiments/{STUDY}',
                                f'tests/{STUDY}',f'results/{STUDY}'],
                               cwd=ROOT,capture_output=True,check=True).stdout
        files=[ROOT/p.decode() for p in tracked.split(b'\0') if p]
        files=[p for p in files if p.is_file() and p.suffix in ('.py','.md','.json')]
        dirty=subprocess.run(['git','diff','--name-only','HEAD','--',
                              *[str(p.relative_to(ROOT)) for p in files]],
                             cwd=ROOT,capture_output=True,text=True,check=True).stdout
        if dirty.strip():raise RuntimeError('Commit task code/protocol changes before staging')
    manifest=[]
    for p in files:
        relative=str(p.relative_to(ROOT))
        if p.is_symlink() or '2016' in relative or 'ss16' in relative:raise ValueError('Forbidden bundle path')
        manifest.append({'path':relative,'size':p.stat().st_size,
                         'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()})
    ident=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:20]
    path=staging/f'{kind}-{ident}.tar.zst'
    if not path.exists():
        with path.open('wb') as target:
            proc=subprocess.Popen(['zstd','-3','-c'],stdin=subprocess.PIPE,stdout=target)
            with tarfile.open(fileobj=proc.stdin,mode='w|') as tf:
                for p in files:tf.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
            proc.stdin.close()
            if proc.wait():raise RuntimeError('Bundle compression failed')
    sha=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
    (staging/f'{kind}-{ident}.json').write_text(json.dumps({'sha256':sha,'files':manifest},indent=2)+'\n')
    uri=f"s3://{r['archive_bucket']}/{STUDY}/control/{path.name}"
    subprocess.run(['aws','--profile','vein','s3','cp',str(path),uri,'--sse','AES256','--only-show-errors'],check=True)
    commands=['set -eu','mkdir -p /opt/pcrl/work /opt/pcrl/bundles',
              f'aws s3 cp {shlex.quote(uri)} /opt/pcrl/bundles/{path.name} --only-show-errors',
              f"echo '{sha}  /opt/pcrl/bundles/{path.name}' | sha256sum -c -",
              f'zstd -d -c /opt/pcrl/bundles/{path.name} | tar -xf - -C /opt/pcrl/work']
    return {**send(commands,f'Stage PCRL {kind} verified bundle'),'bundle_sha256':sha,'files':len(files),'bytes':path.stat().st_size}


if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True)
    s=sub.add_parser('stage');s.add_argument('kind',choices=('inputs','code'))
    s=sub.add_parser('get');s.add_argument('command_id')
    s=sub.add_parser('run');s.add_argument('command');s.add_argument('--description',default='PCRL task command')
    args=parser.parse_args()
    if args.action=='stage':result=stage(args.kind)
    elif args.action=='get':result=get(args.command_id)
    else:result=send(['set -eu',args.command],args.description)
    print(json.dumps(result,indent=2))
