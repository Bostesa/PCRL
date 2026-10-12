"""Metadata-only preparation of sequential versioned custody queue units.

The coordinator prepares each next unit after its dependency's receipt lands.
No source person array is decoded and no backup is started by this helper.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from .build_deployment_units import BASE,pin,write

HELPERS=('task_conditioned_representation/custody.py','task_conditioned_representation/source.py','task_conditioned_representation/scheduler.py','task_conditioned_representation/__init__.py','conditional_release/custody.py')


def deduplicate(items):
    out={}
    for item in items:
        path=str(Path(item['path']).resolve())
        if path in out and out[path]!=item['sha256']:raise ValueError('conflicting source hashes')
        out[path]=item['sha256']
    return [{'path':p,'sha256':s} for p,s in sorted(out.items())]


def manifest_pins(value):
    """Use already admitted pins; do not hash embargoed source person bytes."""
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value:
            yield {'path':value['path'],'sha256':value['sha256']}
        else:
            for child in value.values():yield from manifest_pins(child)
    elif isinstance(value,list):
        for child in value:yield from manifest_pins(child)


def prepare(action,version,output,*,manifest=None,release_lock=None,reader_lock=None,plan=None,snapshot=None,target=None,deps=(),python=None,supplemental_metadata=()):
    if not version or any(x not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_' for x in version):raise ValueError('safe version tag required')
    r7=BASE/'.private/task_conditioned_representation/r7';public=BASE/'results/pcrl_task_conditioned_representations_v1/r7'
    custody=r7/'custody'/version
    uid=f'R7_CUSTODY_{action.upper().replace("-","_")}_{version}'
    sources=[pin(BASE/k) for k in HELPERS]
    argv=[python or sys.executable,'-m','task_conditioned_representation.custody',action]
    if action=='plan':
        if not manifest or not release_lock or not reader_lock:raise ValueError('source and final release/reader freeze pins required')
        released=json.loads(Path(release_lock).read_text());readers=json.loads(Path(reader_lock).read_text())
        if released.get('status') not in ('FROZEN','RELEASES_AND_ENDPOINTS_FROZEN') or readers.get('status')!='ALL_READERS_FROZEN':raise ValueError('final release/reader freeze required before custody')
        parsed=json.loads(Path(manifest).read_text())
        sources.extend([pin(manifest),pin(release_lock),pin(reader_lock),*manifest_pins(parsed)])
        plan_out=custody/'PLAN.private.json';receipt=public/f'{uid}.json'
        argv.extend(['--repo',str(BASE),'--source-manifest',str(manifest),'--out',str(plan_out),'--public',str(receipt)])
        for path in supplemental_metadata:
            parsed_metadata=json.loads(Path(path).read_text());sources.extend([pin(path),*manifest_pins(parsed_metadata)])
            argv.extend(['--supplemental-metadata',str(path)])
        outputs=[str(plan_out),str(receipt)]
    elif action=='snapshot':
        if not plan:raise ValueError('finished plan required')
        parsed=json.loads(Path(plan).read_text());plan_pin=pin(plan);snapshot_target=Path(target) if target else custody/'SNAPSHOT'
        # All entries are immutable captures. Queue checks exactly these bytes.
        sources.extend([plan_pin,*({'path':x['path'],'sha256':x['sha256']} for x in parsed['entries'])])
        receipt=public/f'{uid}.json'
        argv.extend(['--repo',str(BASE),'--plan',str(plan),'--plan-sha256',plan_pin['sha256'],'--target',str(snapshot_target),'--public',str(receipt)])
        outputs=[str(snapshot_target/'SNAPSHOT_RECEIPT.json'),str(receipt)]
    elif action in ('restore','off-device'):
        if not snapshot:raise ValueError('finished snapshot required')
        snapshot=Path(snapshot);receipt_pin=pin(snapshot/'SNAPSHOT_RECEIPT.json');snap=json.loads(Path(receipt_pin['path']).read_text())
        sources.extend([receipt_pin,pin(snapshot/'PLAN.private.json'),{'path':str(snapshot/'execution.gitbundle'),'sha256':snap['bundle_sha256']}])
        for item in snap['entries']:sources.append({'path':str(snapshot/'files'/item['alias']),'sha256':item['sha256']})
        if action=='off-device' and not target:raise ValueError('explicit mounted off-device target required')
        restore_target=Path(target) if target else custody/'RESTORED'
        receipt=public/f'{uid}.json'
        argv.extend(['--snapshot',str(snapshot),'--target',str(restore_target),'--public',str(receipt)])
        outputs=[str(receipt)]
    else:raise ValueError('unregistered custody action')
    spec={'unit_id':uid,'stage':'custody','cwd':str(BASE),'argv':argv,'max_wall_s':1200,'cpu_reserve_s':1200,'max_rss_gib':2,'projected_output_bytes':1 if action=='plan' else sum(int(x.get('size_bytes',0)) for x in parsed['entries']) if action=='snapshot' else sum(int(x.get('size_bytes',0)) for x in snap['entries']),'sources':deduplicate(sources),'deps':list(deps),'outputs':outputs}
    spec_pin=write(r7/'specs'/f'{uid}.json',spec)
    return write(output,{'schema':'TCR1-prepared-custody-unit-v1','unit_id':uid,'action':action,'version':version,'spec':spec_pin,'executed':False,'numeric_person_arrays_decoded':False,'sources_use_existing_admitted_pins':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('plan','snapshot','restore','off-device'));p.add_argument('version');p.add_argument('output');p.add_argument('--manifest');p.add_argument('--release-lock');p.add_argument('--reader-lock');p.add_argument('--plan');p.add_argument('--snapshot');p.add_argument('--target');p.add_argument('--deps',default='');p.add_argument('--supplemental-metadata',action='append',default=[]);a=p.parse_args()
    print(json.dumps(prepare(a.action,a.version,a.output,manifest=a.manifest,release_lock=a.release_lock,reader_lock=a.reader_lock,plan=a.plan,snapshot=a.snapshot,target=a.target,deps=tuple(filter(None,a.deps.split(','))),supplemental_metadata=a.supplemental_metadata)))
