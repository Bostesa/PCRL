"""Laptop-side AWS provisioning for pcrl_utility_extension_v1, with a live resource ledger.

Subcommands (each idempotent, each appends to the PRIVATE ledger):
  storage    private bucket (versioning, all public-access blocks, SSE-S3), no lifecycle rules
  roles      least-privilege instance role (this bucket only) + scheduler stop role (tagged instances only)
  price      live On-Demand price for the chosen type and gp3 storage
  launch     run config -> S3, security group without ingress, instance with user-data
  rehearse   CloudWatch low-CPU stop alarm + EventBridge deadline stop; harmless rehearsal of both
             stop paths against the verified instance ID; then writes run/START
  state      describe task-owned resources and billable leftovers
Private identifiers (account, bucket, instance IDs) live only in the private ledger.
"""
from __future__ import annotations

import argparse
import base64
import json
import secrets
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

HOME = Path.home()                     # the cloud host reproduces this absolute layout
COMMON = Path(subprocess.run(['git', '-C', str(Path(__file__).parent), 'rev-parse', '--path-format=absolute',
                              '--git-common-dir'], capture_output=True, text=True, check=True).stdout.strip())
PRIVATE = COMMON / 'pcrl_parallel_handoff_v5' / 'terminal_1' / 'private'
REPO_URL = subprocess.run(['git', '-C', str(Path(__file__).parent), 'remote', 'get-url', 'origin'],
                          capture_output=True, text=True, check=True).stdout.strip()
LEDGER = PRIVATE / 'AWS_RESOURCE_LEDGER.private.json'
REGION = 'us-east-1'
TAG = {'Key': 'pcrl-task', 'Value': 'utility-extension-v1'}
TYPE = 'm7i.2xlarge'
EBS_GB = 200
BRANCH = 'research/pcrl-utility-extension-aws-v1'
PREFIX = 'pcrl_utility_extension_v1'
HERE = Path(__file__).parent


def aws(*args, check=True, parse=True):
    r = subprocess.run(['aws', '--region', REGION, *args, '--output', 'json'], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f'aws {" ".join(args[:3])}: {r.stderr.strip()[:500]}')
    if not parse:
        return r
    return json.loads(r.stdout) if r.stdout.strip() else {}


def ledger() -> dict:
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else {'resources': {}, 'events': []}


def save(led, event=None):
    if event:
        led['events'].append({'utc': datetime.now(timezone.utc).isoformat(), **event})
    PRIVATE.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER.with_suffix('.tmp')
    tmp.write_text(json.dumps(led, indent=1))
    tmp.replace(LEDGER)


def storage():
    led = ledger()
    if 'bucket' in led['resources']:
        return led['resources']['bucket']
    acct = aws('sts', 'get-caller-identity')['Account']
    name = f'pcrl-ux-archive-{secrets.token_hex(4)}'
    aws('s3api', 'create-bucket', '--bucket', name, parse=False)
    aws('s3api', 'put-public-access-block', '--bucket', name, '--public-access-block-configuration',
        'BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true', parse=False)
    aws('s3api', 'put-bucket-versioning', '--bucket', name, '--versioning-configuration', 'Status=Enabled', parse=False)
    aws('s3api', 'put-bucket-encryption', '--bucket', name, '--server-side-encryption-configuration',
        json.dumps({'Rules': [{'ApplyServerSideEncryptionByDefault': {'SSEAlgorithm': 'AES256'}, 'BucketKeyEnabled': True}]}), parse=False)
    aws('s3api', 'put-bucket-tagging', '--bucket', name, '--tagging', json.dumps({'TagSet': [TAG]}), parse=False)
    led['resources']['bucket'] = {'name': name, 'account': acct, 'region': REGION, 'versioning': 'Enabled',
                                  'public_access_block': 'all four true', 'encryption': 'SSE-S3 (AES256)',
                                  'lifecycle': 'none (evidence is never auto-expired)'}
    save(led, {'action': 'create_bucket', 'bucket': name})
    return led['resources']['bucket']


def roles():
    led = ledger()
    bucket = led['resources']['bucket']['name']
    acct = led['resources']['bucket']['account']
    if 'instance_role' not in led['resources']:
        trust = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Principal': {'Service': 'ec2.amazonaws.com'},
                                                         'Action': 'sts:AssumeRole'}]}
        aws('iam', 'create-role', '--role-name', 'pcrl-ux-ec2', '--assume-role-policy-document', json.dumps(trust),
            '--tags', json.dumps([TAG]))
        policy = {'Version': '2012-10-17', 'Statement': [
            {'Effect': 'Allow', 'Action': ['s3:GetObject', 's3:GetObjectVersion', 's3:PutObject', 's3:GetObjectAttributes'],
             'Resource': f'arn:aws:s3:::{bucket}/*'},
            {'Effect': 'Allow', 'Action': ['s3:ListBucket', 's3:ListBucketVersions'], 'Resource': f'arn:aws:s3:::{bucket}'}]}
        aws('iam', 'put-role-policy', '--role-name', 'pcrl-ux-ec2', '--policy-name', 'pcrl-ux-bucket-only',
            '--policy-document', json.dumps(policy), parse=False)
        aws('iam', 'create-instance-profile', '--instance-profile-name', 'pcrl-ux-ec2')
        aws('iam', 'add-role-to-instance-profile', '--instance-profile-name', 'pcrl-ux-ec2', '--role-name', 'pcrl-ux-ec2', parse=False)
        led['resources']['instance_role'] = {'role': 'pcrl-ux-ec2', 'profile': 'pcrl-ux-ec2', 'scope': f's3 on {bucket} only'}
        save(led, {'action': 'create_instance_role'})
    if 'stop_role' not in led['resources']:
        trust = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Principal': {'Service': 'scheduler.amazonaws.com'},
                                                         'Action': 'sts:AssumeRole',
                                                         'Condition': {'StringEquals': {'aws:SourceAccount': acct}}}]}
        aws('iam', 'create-role', '--role-name', 'pcrl-ux-deadline-stop', '--assume-role-policy-document', json.dumps(trust),
            '--tags', json.dumps([TAG]))
        policy = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': 'ec2:StopInstances',
                                                          'Resource': f'arn:aws:ec2:{REGION}:{acct}:instance/*',
                                                          'Condition': {'StringEquals': {'aws:ResourceTag/pcrl-task': TAG['Value']}}}]}
        aws('iam', 'put-role-policy', '--role-name', 'pcrl-ux-deadline-stop', '--policy-name', 'stop-tagged-task-instance-only',
            '--policy-document', json.dumps(policy), parse=False)
        led['resources']['stop_role'] = {'role': 'pcrl-ux-deadline-stop', 'scope': 'ec2:StopInstances on instances tagged pcrl-task=utility-extension-v1'}
        save(led, {'action': 'create_stop_role'})
    time.sleep(12)   # IAM propagation


def price():
    flt = lambda k, v: {'Type': 'TERM_MATCH', 'Field': k, 'Value': v}
    r = subprocess.run(['aws', 'pricing', 'get-products', '--region', 'us-east-1', '--service-code', 'AmazonEC2',
                        '--filters', json.dumps([flt('instanceType', TYPE), flt('location', 'US East (N. Virginia)'),
                                                 flt('operatingSystem', 'Linux'), flt('tenancy', 'Shared'),
                                                 flt('preInstalledSw', 'NA'), flt('capacitystatus', 'Used')]),
                        '--output', 'json'], capture_output=True, text=True, check=True)
    prod = json.loads(json.loads(r.stdout)['PriceList'][0])
    od = next(iter(prod['terms']['OnDemand'].values()))
    usd = float(next(iter(od['priceDimensions'].values()))['pricePerUnit']['USD'])
    attrs = prod['product']['attributes']
    spec = aws('ec2', 'describe-instance-types', '--instance-types', TYPE)['InstanceTypes'][0]
    out = {'type': TYPE, 'on_demand_usd_per_h': usd, 'vcpu': attrs.get('vcpu'), 'memory': attrs.get('memory'),
           'live_vcpu': spec['VCpuInfo']['DefaultVCpus'], 'live_mem_mib': spec['MemoryInfo']['SizeInMiB'],
           'arch': spec['ProcessorInfo']['SupportedArchitectures'],
           'gp3_usd_per_gb_month_assumed': 0.08, 'ebs_gb': EBS_GB,
           'ebs_usd_per_h': round(EBS_GB * 0.08 / 730, 4),
           'ten_hour_estimate_usd': round(10 * (usd + EBS_GB * 0.08 / 730), 2),
           'alternatives_considered': {'c7i.4xlarge': '16 vCPU/32 GiB, ~2x price; measured workload (~8 CPU-h, ~1 GB/worker) does not need it',
                                       'c7i.2xlarge': '8 vCPU/16 GiB; tighter memory headroom for 6 workers',
                                       'GPU': 'no benefit for small tabular MLP/HistGB fits',
                                       'Spot': 'not used: checkpointed restart is unit-level only, saving < $2 does not justify interruption risk'}}
    led = ledger()
    led['resources']['price'] = out
    save(led, {'action': 'price_check', **out})
    return out


def launch(exec_chunk_ids, workers=6, hours=10.0):
    led = ledger()
    bucket = led['resources']['bucket']['name']
    pr = led['resources']['price']
    now = datetime.now(timezone.utc)
    deadline = now + timedelta(hours=hours)
    cfg = {'bucket': bucket, 'prefix': PREFIX, 'repo': str(HOME / 'PCRL-terminal-1-utility'),
           'verify_dir': '/data/verify', 'scratch': '/data/scratch', 'workers': workers,
           'exec_chunk_ids': exec_chunk_ids, 'exec_chunks': len(exec_chunk_ids),
           'remote_start_epoch': now.timestamp(), 'deadline_epoch': deadline.timestamp(),
           'deadline_systemd': deadline.strftime('%Y-%m-%d %H:%M:%S UTC'), 'closing_reserve_h': 0.75,
           'hourly_usd': pr['on_demand_usd_per_h'], 'ebs_hourly_usd': pr['ebs_usd_per_h'],
           'compute_ceiling_usd': 30.0, 'forecast_hours': 6.0, 'independent_stop_installed': True,
           'audit_portability_tol': 0.003, 'stop_on_closeout': True}
    roots = {'exec_main': str(HOME / 'PCRL'),
             'exec_rs': str(HOME / '.config/superpowers/worktrees/PCRL/residual-spectral-20260910'),
             'exec_inv': str(HOME / 'PCRL-terminal-1-invariant')}
    for name, payload in (('runcfg.json', cfg), ('roots.json', roots)):
        p = PRIVATE / f'run_{name}'
        p.write_text(json.dumps(payload, indent=1))
        subprocess.run(['aws', 's3', 'cp', str(p), f's3://{bucket}/{PREFIX}/run/{name}', '--sse', 'AES256',
                        '--only-show-errors'], check=True)
    vpc = aws('ec2', 'describe-vpcs', '--filters', 'Name=is-default,Values=true')['Vpcs'][0]['VpcId']
    if 'security_group' not in led['resources']:
        sg = aws('ec2', 'create-security-group', '--group-name', 'pcrl-ux-no-ingress', '--vpc-id', vpc,
                 '--description', 'pcrl utility extension: no inbound rules',
                 '--tag-specifications', json.dumps([{'ResourceType': 'security-group', 'Tags': [TAG]}]))['GroupId']
        led['resources']['security_group'] = {'id': sg, 'ingress': 'none'}
        save(led, {'action': 'create_sg', 'id': sg})
    ami = aws('ssm', 'get-parameter', '--name',
              '/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id')['Parameter']['Value']
    user_data = (HERE / 'bootstrap.sh').read_text().replace('__BUCKET__', bucket).replace('__PREFIX__', PREFIX) \
        .replace('__BRANCH__', BRANCH) \
        .replace('__HOME__', str(HOME)).replace('__REPO_URL__', REPO_URL)
    ud = PRIVATE / 'user_data.sh'
    ud.write_text(user_data)
    inst = aws('ec2', 'run-instances', '--image-id', ami, '--instance-type', TYPE, '--count', '1',
               '--iam-instance-profile', 'Name=pcrl-ux-ec2',
               '--security-group-ids', led['resources']['security_group']['id'],
               '--instance-initiated-shutdown-behavior', 'stop',
               '--metadata-options', 'HttpTokens=required',
               '--block-device-mappings', json.dumps([{'DeviceName': '/dev/sda1', 'Ebs': {
                   'VolumeSize': EBS_GB, 'VolumeType': 'gp3', 'Encrypted': True, 'DeleteOnTermination': True}}]),
               '--tag-specifications', json.dumps([{'ResourceType': t, 'Tags': [TAG, {'Key': 'Name', 'Value': 'pcrl-ux-runner'}]}
                                                   for t in ('instance', 'volume')]),
               '--user-data', f'file://{ud}')['Instances'][0]
    led['resources']['instance'] = {'id': inst['InstanceId'], 'type': TYPE, 'ami': ami, 'launched_utc': now.isoformat(),
                                    'deadline_utc': deadline.isoformat(), 'ebs_gb': EBS_GB,
                                    'shutdown_behavior': 'stop', 'delete_on_termination': True}
    save(led, {'action': 'run_instance', 'id': inst['InstanceId'], 'type': TYPE})
    return inst['InstanceId']


def wait_state(iid, want, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = aws('ec2', 'describe-instances', '--instance-ids', iid)['Reservations'][0]['Instances'][0]['State']['Name']
        if st == want:
            return True
        time.sleep(10)
    return False


def rehearse():
    led = ledger()
    iid = led['resources']['instance']['id']
    acct = led['resources']['bucket']['account']
    bucket = led['resources']['bucket']['name']
    alarm = 'pcrl-ux-idle-stop'
    aws('cloudwatch', 'put-metric-alarm', '--alarm-name', alarm, '--namespace', 'AWS/EC2', '--metric-name', 'CPUUtilization',
        '--dimensions', f'Name=InstanceId,Value={iid}', '--statistic', 'Average', '--period', '300',
        '--evaluation-periods', '6', '--threshold', '3', '--comparison-operator', 'LessThanThreshold',
        '--treat-missing-data', 'notBreaching', '--alarm-actions', f'arn:aws:automate:{REGION}:ec2:stop', parse=False)
    led['resources']['idle_alarm'] = {'name': alarm, 'rule': 'CPU avg < 3% for 30 min -> EC2 stop', 'target': iid}
    save(led, {'action': 'create_idle_alarm', 'target': iid})
    # Rehearsal 1: the alarm path.
    aws('cloudwatch', 'set-alarm-state', '--alarm-name', alarm, '--state-value', 'ALARM', '--state-reason', 'rehearsal', parse=False)
    ok1 = wait_state(iid, 'stopped', 420)
    save(led, {'action': 'rehearsal_alarm_stop', 'target': iid, 'stopped': ok1})
    aws('cloudwatch', 'set-alarm-state', '--alarm-name', alarm, '--state-value', 'OK', '--state-reason', 'rehearsal done', parse=False)
    aws('ec2', 'start-instances', '--instance-ids', iid)
    wait_state(iid, 'running', 420)
    # Rehearsal 2: the EventBridge Scheduler path (one-time, 3 minutes ahead), then the real deadline.
    role = f'arn:aws:iam::{acct}:role/pcrl-ux-deadline-stop'
    target = json.dumps({'Arn': 'arn:aws:scheduler:::aws-sdk:ec2:stopInstances', 'RoleArn': role,
                         'Input': json.dumps({'InstanceIds': [iid]})})
    at = (datetime.now(timezone.utc) + timedelta(minutes=3)).strftime('%Y-%m-%dT%H:%M:%S')
    aws('scheduler', 'create-schedule', '--name', 'pcrl-ux-rehearsal', '--schedule-expression', f'at({at})',
        '--schedule-expression-timezone', 'UTC', '--flexible-time-window', 'Mode=OFF', '--target', target,
        '--action-after-completion', 'DELETE')
    ok2 = wait_state(iid, 'stopped', 600)
    save(led, {'action': 'rehearsal_scheduler_stop', 'target': iid, 'stopped': ok2})
    aws('ec2', 'start-instances', '--instance-ids', iid)
    wait_state(iid, 'running', 420)
    deadline = datetime.fromisoformat(led['resources']['instance']['deadline_utc']).strftime('%Y-%m-%dT%H:%M:%S')
    aws('scheduler', 'create-schedule', '--name', 'pcrl-ux-deadline-stop', '--schedule-expression', f'at({deadline})',
        '--schedule-expression-timezone', 'UTC', '--flexible-time-window', 'Mode=OFF', '--target', target)
    led['resources']['deadline_schedule'] = {'name': 'pcrl-ux-deadline-stop', 'at_utc': deadline, 'target': iid}
    save(led, {'action': 'create_deadline_schedule', 'at_utc': deadline, 'target': iid})
    if ok1 and ok2:
        subprocess.run(['aws', 's3', 'cp', '-', f's3://{bucket}/{PREFIX}/run/START', '--sse', 'AES256'],
                       input=b'rehearsals passed\n', check=True)
        save(led, {'action': 'write_START'})
    return {'alarm_stop': ok1, 'scheduler_stop': ok2}


def state():
    led = ledger()
    out = {}
    if 'instance' in led['resources']:
        i = aws('ec2', 'describe-instances', '--instance-ids', led['resources']['instance']['id'])['Reservations'][0]['Instances'][0]
        out['instance_state'] = i['State']['Name']
        out['volumes'] = [b['Ebs']['VolumeId'] for b in i.get('BlockDeviceMappings', [])]
    out['task_tagged_volumes'] = aws('ec2', 'describe-volumes', '--filters', f'Name=tag:pcrl-task,Values={TAG["Value"]}')['Volumes']
    out['task_tagged_volumes'] = [{'id': v['VolumeId'], 'state': v['State'], 'gb': v['Size']} for v in out['task_tagged_volumes']]
    out['elastic_ips'] = aws('ec2', 'describe-addresses')['Addresses']
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['storage', 'roles', 'price', 'launch', 'rehearse', 'state'])
    ap.add_argument('--exec-chunks', nargs='*', default=[])
    a = ap.parse_args()
    fn = {'storage': storage, 'roles': roles, 'price': price, 'rehearse': rehearse, 'state': state}
    res = launch(a.exec_chunks) if a.cmd == 'launch' else fn[a.cmd]()
    print(json.dumps(res, indent=1, default=str) if not isinstance(res, str) else res)
