"""Inventory, privately archive, read back and restore the completed 2016 study.

Run on the registered Linux host after scientific writes have stopped. The source
tree is never deleted. Each compressed part and every member is read back and
hashed before the archive index is accepted.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile

from .common import OUT, ROOT, STUDY, atomic_json, now, sha

BUCKET = 'pcrl-ux-archive-ed9d21fd'
PREFIX = f'{STUDY}/archive'
EXCLUDED = {'__pycache__', '.pytest_cache', 'staging', 'archive_stage'}
RESTORE_UNIT = f'results/{STUDY}/private/runs/acs2016/units/anchor_0/Q/attack__A__SEX'


def safe_path(name):
    if not isinstance(name, str) or '\\' in name or '\x00' in name:
        raise ValueError('Unsafe archive path')
    p = PurePosixPath(name)
    if p.is_absolute() or not p.parts or '..' in p.parts or p.as_posix() != name:
        raise ValueError('Unsafe archive path')
    return p


def source_path(root, name):
    path = Path(root).resolve()
    for part in safe_path(name).parts:
        path = path/part
        if path.is_symlink():
            raise ValueError(f'Archive refuses symlink: {name}')
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Archive path escape')
    return path


def inventory(root=ROOT):
    root = Path(root).resolve()
    paths = []
    for relative in (f'experiments/{STUDY}', 'experiments/pcrl_task_directed_release_v1',
                     f'results/{STUDY}'):
        paths.extend((root/relative).rglob('*'))
    for relative in ('experiments/acs_transfer_data.py', 'experiments/acs_transfer_heads.py'):
        paths.append(root/relative)
    paths.extend((root/'tests').glob('test_pcrl_final_*.py'))
    records = []
    for p in sorted(set(paths)):
        name = p.relative_to(root).as_posix()
        if set(PurePosixPath(name).parts) & EXCLUDED or name.endswith('.tmp'):
            continue
        if p.is_symlink():
            raise ValueError(f'Archive refuses symlink: {name}')
        if p.is_file():
            safe_path(name)
            records.append({'path': name, 'bytes': p.stat().st_size, 'sha256': sha(p)})
    if not records:
        raise ValueError('Empty study inventory')
    return records


def groups(records, target_bytes=900_000_000):
    out, group, size = [], [], 0
    for item in records:
        if group and size + item['bytes'] > target_bytes:
            out.append(group)
            group, size = [], 0
        group.append(item)
        size += item['bytes']
    if group:
        out.append(group)
    return out


def pack(root, records, path):
    root, path = Path(root).resolve(), Path(path)
    if path.exists():
        raise FileExistsError(path)
    with path.open('xb') as output:
        proc = subprocess.Popen(['zstd', '-3', '-T4', '-c'], stdin=subprocess.PIPE, stdout=output)
        try:
            with tarfile.open(fileobj=proc.stdin, mode='w|') as archive:
                for item in records:
                    source = source_path(root, item['path'])
                    if (not source.is_file() or source.stat().st_size != item['bytes']
                            or sha(source) != item['sha256']):
                        raise ValueError(f"Archive source hash changed: {item['path']}")
                    archive.add(source, arcname=item['path'], recursive=False)
            proc.stdin.close()
            if proc.wait():
                raise RuntimeError('Archive compression failed')
        except BaseException:
            proc.kill()
            proc.stdin.close()
            proc.wait()
            raise
    return {'bytes': path.stat().st_size, 'sha256': sha(path),
            'files': len(records), 'uncompressed_bytes': sum(r['bytes'] for r in records)}


def verify_restore(path, records, restore_root=None, prefixes=()):
    """Hash every archive member; restore only selected files without overwrites."""
    expected = {r['path']: r for r in records}
    if len(expected) != len(records):
        raise ValueError('Duplicate manifest path')
    base = Path(restore_root).resolve() if restore_root is not None else None
    if base is not None:
        if Path(restore_root).is_symlink():
            raise ValueError('Restore root is a symlink')
        base.mkdir(parents=True, exist_ok=True)
    seen, restored = set(), []
    proc = subprocess.Popen(['zstd', '-d', '-c', str(path)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdout, mode='r|') as archive:
            for member in archive:
                safe_path(member.name)
                if member.name not in expected or member.name in seen or not member.isfile():
                    raise ValueError('Unexpected archive member')
                item = expected[member.name]
                if member.size != item['bytes']:
                    raise ValueError(f'Archive member size mismatch: {member.name}')
                restore = base is not None and any(member.name == p or member.name.startswith(p.rstrip('/')+'/')
                                                   for p in prefixes)
                target = source_path(base, member.name) if restore else None
                if target is not None:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        raise FileExistsError(target)
                digest = hashlib.sha256()
                output = target.open('xb') if target is not None else None
                try:
                    stream = archive.extractfile(member)
                    for block in iter(lambda: stream.read(4 << 20), b''):
                        digest.update(block)
                        if output is not None:
                            output.write(block)
                finally:
                    if output is not None:
                        output.close()
                if digest.hexdigest() != item['sha256']:
                    raise ValueError(f'Archive member hash mismatch: {member.name}')
                seen.add(member.name)
                if target is not None:
                    restored.append(member.name)
        proc.stdout.close()
        if proc.wait():
            raise RuntimeError('Archive decompression failed')
    except BaseException:
        proc.kill()
        proc.stdout.close()
        proc.wait()
        raise
    if seen != set(expected):
        raise ValueError('Archive omitted manifest members')
    return {'verified_files': len(seen), 'restored_files': len(restored),
            'restored_paths': restored}


def aws(*args):
    completed = subprocess.run(['aws', *args, '--output', 'json'], capture_output=True, text=True, check=True)
    return json.loads(completed.stdout) if completed.stdout.strip() else {}


def put_and_readback(bucket, key, source, readback):
    created = aws('s3api', 'put-object', '--bucket', bucket, '--key', key, '--body', str(source),
                  '--server-side-encryption', 'AES256', '--if-none-match', '*')
    version = created.get('VersionId')
    target = ['s3api', 'head-object', '--bucket', bucket, '--key', key]
    if version:
        target += ['--version-id', version]
    head = aws(*target)
    if (head.get('ServerSideEncryption') != 'AES256' or head.get('ContentLength') != source.stat().st_size
            or (version is not None and head.get('VersionId') != version)):
        raise ValueError(f'Archive object metadata differs: {key}')
    get = ['s3api', 'get-object', '--bucket', bucket, '--key', key]
    if version:
        get += ['--version-id', version]
    aws(*get, str(readback))
    if sha(readback) != sha(source):
        raise ValueError(f'Archive readback hash differs: {key}')
    return {'key': key, 'version_id': version, 'bytes': source.stat().st_size, 'sha256': sha(source),
            'encryption': 'AES256', 'readback_verified': True}


def check_preflight(path, bucket, prefix):
    preflight = json.loads(Path(path).read_text())
    required = ('BlockPublicAcls', 'IgnorePublicAcls', 'BlockPublicPolicy', 'RestrictPublicBuckets')
    checked = dt.datetime.fromisoformat(preflight['checked_utc'].replace('Z', '+00:00'))
    if (preflight['bucket'] != bucket or preflight['prefix'] != prefix
            or not all(preflight['public_access_block'].get(k) is True for k in required)
            or not preflight['prefix_empty']
            or not 0 <= (dt.datetime.now(dt.timezone.utc)-checked).total_seconds() < 3600):
        raise ValueError('Private archive destination preflight is missing, stale or mismatched')
    return preflight


def snapshot_host_records():
    target = OUT/'private/host_records'
    target.mkdir(parents=True, exist_ok=True)
    for name in ('environment.txt', 'setup_repair.sh', 'setup_repair.log', 'parity.log',
                 'prepare2016.log', 'fit2016.log', 'score2016.log', 'infer2016.log',
                 'emulated.log', 'code_hash_check.log', 'deadline.sh', 'READY',
                 'BENCHMARK_DEPRIORITIZED_UTC'):
        p = Path('/opt/pcrl')/name
        if p.is_file():
            shutil.copy2(p, target/name)
    (target/'system.txt').write_text(subprocess.run(
        ['uname', '-a'], capture_output=True, text=True, check=True).stdout)


def score_restore_replay(restore_root):
    import numpy as np
    from experiments.pcrl_task_directed_release_v1.audits import expected_token_loss
    d = Path(restore_root)/RESTORE_UNIT
    with np.load(d/'score_final.npz') as score, np.load(d/'pred_final.npz') as pred:
        replay = expected_token_loss(pred['probabilities'], pred['token_probs'], score['y'])
        error = float(np.max(np.abs(replay-score['loss'])))
        rows = len(replay)
    if error > 1e-12:
        raise ValueError(f'Restored prediction loss replay differs: {error}')
    registry_restored = (d/'registry.joblib').is_file()
    fitted_weights_restored = any(p.is_file() for p in (d/'models').rglob('*'))
    if not registry_restored or not fitted_weights_restored:
        raise ValueError('Representative fitted model did not restore')
    return {'unit': RESTORE_UNIT, 'rows': rows, 'max_abs_loss_error': error,
            'registry_restored': registry_restored, 'fitted_weights_restored': fitted_weights_restored}


def publish(preflight_path, staging_dir, source_commit, bucket=BUCKET, prefix=PREFIX):
    check_preflight(preflight_path, bucket, prefix)
    if (OUT/'ARCHIVE_INDEX.json').exists():
        raise FileExistsError('An accepted archive index already exists')
    staging = Path(staging_dir)
    staging.mkdir(parents=True, exist_ok=False)
    snapshot_host_records()
    records = inventory()
    manifest = {'created_utc': now(), 'study': STUDY, 'source_commit': source_commit,
                'evaluation_lock_sha256': sha(OUT/'EVALUATION_LOCK.json'), 'files': records}
    manifest_path = staging/'MANIFEST.private.json'
    atomic_json(manifest_path, manifest)
    restore_root = staging/'restored'
    index = {'created_utc': now(), 'study': STUDY, 'bucket': bucket, 'prefix': prefix,
             'source_commit': source_commit, 'manifest_sha256': sha(manifest_path),
             'files': len(records), 'uncompressed_bytes': sum(r['bytes'] for r in records),
             'parts': [], 'preflight': json.loads(Path(preflight_path).read_text())}
    for i, group in enumerate(groups(records)):
        part = staging/f'part-{i:04d}.tar.zst'
        info = pack(ROOT, group, part)
        key = f'{prefix}/part-{i:04d}.tar.zst'
        receipt = put_and_readback(bucket, key, part, staging/f'part-{i:04d}.readback.tar.zst')
        verification = verify_restore(staging/f'part-{i:04d}.readback.tar.zst', group, restore_root,
                                      (RESTORE_UNIT, f'results/{STUDY}/EVALUATION_LOCK.json'))
        index['parts'].append({**info, **receipt, **verification})
        atomic_json(staging/'INDEX.progress.json', index)
    index['restore_replay'] = score_restore_replay(restore_root)
    index['manifest_object'] = put_and_readback(bucket, f'{prefix}/MANIFEST.private.json', manifest_path,
                                                staging/'MANIFEST.readback.json')
    index['completed_utc'] = now()
    index['all_streams_and_files_verified'] = True
    index['restore_completed'] = True
    atomic_json(staging/'INDEX.final.json', index)
    receipt = put_and_readback(bucket, f'{prefix}/ARCHIVE_INDEX.json', staging/'INDEX.final.json',
                               staging/'ARCHIVE_INDEX.readback.json')
    atomic_json(staging/'INDEX_UPLOAD_RECEIPT.json', receipt)
    atomic_json(OUT/'ARCHIVE_INDEX.json', index)
    return index


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', required=True)
    ap.add_argument('--staging-dir', required=True)
    ap.add_argument('--source-commit', required=True)
    args = ap.parse_args()
    result = publish(args.preflight, args.staging_dir, args.source_commit)
    print(json.dumps({'files': result['files'], 'parts': len(result['parts']),
                      'restore_replay': result['restore_replay']}, indent=2))
