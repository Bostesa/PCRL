"""Private, read-back-verified archive for the new 2018 development study.

Nothing here deletes originals or touches another study's resources.  The
completed prospective study's tested archive primitives are reused, while this
module inventories only the new owned study tree.  A representative fitted
unit must be restored and replayed before any separate cleanup decision.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import json
import os
from pathlib import Path
import subprocess

from experiments.pcrl_final_prospective_v1.archive_run import (
    groups, pack, put_and_readback, safe_path, sha, verify_restore,
)

STUDY = 'pcrl_task_aligned_cuts_v1'
BUCKET = 'pcrl-ux-archive-ed9d21fd'
SKIP_PARTS = {'archive_stage', '__pycache__', '.pytest_cache'}


@contextmanager
def _private_outputs(directory: Path):
    """Create archive material owner-only, restoring the caller's umask."""
    if directory.is_symlink():
        raise ValueError('archive staging directory must not be a symlink')
    previous = os.umask(0o077)
    try:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
        yield
    finally:
        os.umask(previous)


def _atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    os.replace(temporary, path)


def inventory(root: Path) -> list[dict]:
    root = root.resolve()
    prefixes = [f'experiments/{STUDY}', f'analysis/{STUDY}', f'results/{STUDY}',
                f'tests/{STUDY}']
    records = []
    for prefix in prefixes:
        base = root / prefix
        if not base.exists():
            continue
        for path in sorted(base.rglob('*')):
            rel = path.relative_to(root).as_posix()
            if set(Path(rel).parts) & SKIP_PARTS or rel.endswith('.tmp'):
                continue
            if path.is_symlink():
                raise ValueError(f'refusing archive symlink: {rel}')
            if path.is_file():
                safe_path(rel)
                records.append({'path':rel,'bytes':path.stat().st_size,
                                'sha256':sha(path)})
    if not records:
        raise ValueError('empty new-study archive inventory')
    return records


def stage(root: Path, staging: Path, *, target_bytes: int = 900_000_000) -> dict:
    """Pack and independently unpack-check all bytes; leave originals intact."""
    if staging.is_symlink():
        raise ValueError('archive staging directory must not be a symlink')
    root = root.resolve(); staging = staging.resolve()
    with _private_outputs(staging):
        return _stage_private(root, staging, target_bytes=target_bytes)


def _stage_private(root: Path, staging: Path, *, target_bytes: int) -> dict:
    manifest_path = staging / 'MANIFEST.private.json'
    if manifest_path.exists():
        raise FileExistsError('archive stage already exists; never overwrite evidence')
    records = inventory(root)
    manifest = {'study':STUDY,'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                'root_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                'files':records,'parts':[],'originals_retained':True}
    for index, batch in enumerate(groups(records, target_bytes=target_bytes), 1):
        target = staging / f'part-{index:04d}.tar.zst'
        packed = pack(root, batch, target)
        check = verify_restore(target, batch)
        manifest['parts'].append({'name':target.name, **packed,
                                  'readback_local':check['verified_files'] == len(batch),
                                  'members':[item['path'] for item in batch]})
    _atomic(manifest_path, manifest)
    return {'files':len(records),'parts':len(manifest['parts']),
            'uncompressed_bytes':sum(item['bytes'] for item in records),
            'manifest_sha256':sha(manifest_path),'staging':str(staging)}


def aws_json(*args: str) -> dict:
    result = subprocess.run(['aws', *args, '--output', 'json'],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout) if result.stdout.strip() else {}


def preflight(prefix: str) -> dict:
    if not prefix.startswith(f'{STUDY}/archive/') or '..' in prefix:
        raise ValueError('archive prefix must be unique within this study')
    block = aws_json('s3api','get-public-access-block','--bucket',BUCKET)['PublicAccessBlockConfiguration']
    if not all(block.get(name) is True for name in
               ('BlockPublicAcls','IgnorePublicAcls','BlockPublicPolicy','RestrictPublicBuckets')):
        raise ValueError('private bucket public-access controls incomplete')
    response = aws_json('s3api','list-objects-v2','--bucket',BUCKET,'--prefix',prefix,'--max-keys','1')
    if response.get('KeyCount',0):
        raise ValueError('archive destination prefix is not empty')
    return {'bucket':BUCKET,'prefix':prefix,'public_access_block':block,
            'prefix_empty':True,'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat()}


def publish(staging: Path, prefix: str, *, restore_prefix: str,
            restore_root: Path) -> dict:
    """Upload encrypted parts, read every object back, and restore one unit.

    ``restore_prefix`` must name a representative fitted scientific unit with
    model weights/predictions.  Replay of its scientific loss is a separate
    required check by the coordinator before cleanup.
    """
    if staging.is_symlink():
        raise ValueError('archive staging directory must not be a symlink')
    staging = staging.resolve(); restore_root = restore_root.resolve()
    with _private_outputs(staging):
        return _publish_private(staging, prefix, restore_prefix=restore_prefix,
                                restore_root=restore_root)


def _publish_private(staging: Path, prefix: str, *, restore_prefix: str,
                     restore_root: Path) -> dict:
    manifest_path = staging / 'MANIFEST.private.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['study'] != STUDY or not manifest['originals_retained']:
        raise ValueError('invalid private archive manifest')
    check = preflight(prefix)
    uploaded = []
    for part in manifest['parts']:
        source = staging / part['name']
        if sha(source) != part['sha256']:
            raise ValueError('staged archive part changed')
        readback = staging / ('readback-' + part['name'])
        if readback.exists():
            raise FileExistsError(readback)
        result = put_and_readback(BUCKET, prefix + '/' + part['name'], source, readback)
        if result['sha256'] != part['sha256']:
            raise ValueError('cloud part readback mismatch')
        batch = [item for item in manifest['files'] if item['path'] in set(part['members'])]
        verify_restore(readback, batch)
        if any(item['path'] == restore_prefix or item['path'].startswith(restore_prefix.rstrip('/') + '/') for item in batch):
            verify_restore(readback, batch, restore_root, prefixes=(restore_prefix,))
        uploaded.append(result)
    manifest_cloud = put_and_readback(BUCKET,prefix+'/MANIFEST.private.json',
                                      manifest_path,staging/'readback-MANIFEST.private.json')
    report = {'study':STUDY,'preflight':check,'objects':uploaded,
              'manifest_object':manifest_cloud,'restore_prefix':restore_prefix,
              'restore_root':str(restore_root),
              'restored_paths':[p.relative_to(restore_root).as_posix()
                                for p in restore_root.rglob('*') if p.is_file()],
              'originals_retained':True,
              'completed_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    _atomic(staging/'ARCHIVE_INDEX.json',report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command',required=True)
    for name in ('inventory','stage','publish'):
        p = sub.add_parser(name)
        p.add_argument('--root',type=Path,default=Path.cwd())
        if name in ('stage','publish'):
            p.add_argument('--staging',type=Path,required=True)
        if name == 'publish':
            p.add_argument('--prefix',required=True)
            p.add_argument('--restore-prefix',required=True)
            p.add_argument('--restore-root',type=Path,required=True)
    args = parser.parse_args(argv)
    if args.command == 'inventory':
        files = inventory(args.root)
        print(json.dumps({'files':len(files),'bytes':sum(x['bytes'] for x in files)}))
    elif args.command == 'stage':
        print(json.dumps(stage(args.root,args.staging),sort_keys=True))
    else:
        print(json.dumps(publish(args.staging,args.prefix,
                                 restore_prefix=args.restore_prefix,
                                 restore_root=args.restore_root),sort_keys=True))


if __name__ == '__main__':
    main()
