"""Task-owned AWS orchestration (SSM only); refuses any instance not tagged to this study."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import shlex
import subprocess
import tarfile
from pathlib import Path

from .common import OUT, ROOT, STUDY

BUCKET = 'pcrl-ux-archive-ed9d21fd'


def aws(*args):
    p = subprocess.run(['aws', '--profile', 'vein', '--region', 'us-east-1', *args, '--output', 'json'],
                       capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr)
    return json.loads(p.stdout) if p.stdout.strip() else {}


def instance():
    iid = json.loads((OUT/'private/AWS_RESOURCES.json').read_text())['instance_id']
    info = aws('ec2', 'describe-instances', '--instance-ids', iid)['Reservations'][0]['Instances'][0]
    if not any(t['Key'] == 'study' and t['Value'] == STUDY for t in info.get('Tags', [])):
        raise PermissionError('Refusing an instance not tagged to this task')
    return iid


def send(commands, description, timeout=36000):
    iid = instance()
    res = aws('ssm', 'send-command', '--instance-ids', iid, '--document-name', 'AWS-RunShellScript',
              '--parameters', json.dumps({'commands': commands, 'executionTimeout': [str(timeout)]}),
              '--comment', description[:100])
    cid = res['Command']['CommandId']
    with (OUT/'private/remote_commands.jsonl').open('a') as f:
        f.write(json.dumps({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'id': cid,
                            'description': description, 'commands': commands})+'\n')
    return cid


def get(cid):
    r = aws('ssm', 'get-command-invocation', '--command-id', cid, '--instance-id', instance())
    return {k: r.get(k) for k in ('Status', 'ResponseCode', 'StandardOutputContent', 'StandardErrorContent')}


def bundle(kind, paths):
    """Tar+zstd the given repo-relative files, upload encrypted, return untar commands."""
    files = sorted({Path(p) for p in paths})
    manifest = [{'path': str(p), 'sha256': hashlib.file_digest((ROOT/p).open('rb'), 'sha256').hexdigest()} for p in files]
    ident = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    staging = OUT/'private/staging'; staging.mkdir(parents=True, exist_ok=True)
    path = staging/f'{kind}-{ident}.tar.zst'
    if not path.exists():
        with path.open('wb') as target:
            proc = subprocess.Popen(['zstd', '-3', '-T4', '-c'], stdin=subprocess.PIPE, stdout=target)
            with tarfile.open(fileobj=proc.stdin, mode='w|') as tf:
                for p in files:
                    tf.add(ROOT/p, arcname=str(p), recursive=False)
            proc.stdin.close()
            if proc.wait():
                raise RuntimeError('compression failed')
    digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    (staging/f'{kind}-{ident}.json').write_text(json.dumps({'sha256': digest, 'files': manifest}, indent=1))
    uri = f's3://{BUCKET}/{STUDY}/control/{path.name}'
    subprocess.run(['aws', '--profile', 'vein', 's3', 'cp', str(path), uri, '--sse', 'AES256', '--only-show-errors'], check=True)
    return ['set -eu', 'mkdir -p /opt/pcrl/work /opt/pcrl/bundles',
            f'aws s3 cp {shlex.quote(uri)} /opt/pcrl/bundles/{path.name} --only-show-errors',
            f"echo '{digest}  /opt/pcrl/bundles/{path.name}' | sha256sum -c -",
            f'zstd -d -c /opt/pcrl/bundles/{path.name} | tar -xf - -C /opt/pcrl/work'], digest


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='a', required=True)
    s = sub.add_parser('run'); s.add_argument('command'); s.add_argument('--description', default='PCRL final command')
    s = sub.add_parser('get'); s.add_argument('cid')
    a = ap.parse_args()
    print(send(['set -eu', a.command], a.description) if a.a == 'run' else json.dumps(get(a.cid), indent=1))
