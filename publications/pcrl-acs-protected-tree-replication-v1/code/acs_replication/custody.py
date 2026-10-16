"""Custody: snapshot the private campaign root (source copy, fitted artifacts, private predictions, exports, receipts)
with a sha256 manifest, restore it into a separate directory from the snapshot ALONE, and verify every hash.
Off-device: ``offdevice-copy`` copies a snapshot to a mounted external drive and verifies every hash on the drive;
``restore-from`` restores from that drive copy alone. When no drive is mounted, ``offdevice`` prints the later command.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRIV = ROOT / '.private' / 'acs_replication'
CACHE = Path.home() / 'PCRL_eval_cache_private' / 'acsr_v1'


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def manifest(base):
    out = {}
    for p in sorted(Path(base).rglob('*')):
        if p.is_file() and 'pylib' not in p.parts and 'vendor' not in p.parts:
            out[str(p.relative_to(base))] = {'sha256': sha(p), 'bytes': p.stat().st_size}
    return out


def snapshot(tag):
    dst = CACHE / 'backups' / tag
    if dst.exists():
        raise PermissionError('snapshot tag exists')
    dst.mkdir(parents=True, mode=0o700)
    src_files = manifest(PRIV)
    for rel in src_files:
        tgt = dst / 'private' / rel; tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PRIV / rel, tgt)
    code = dst / 'code'; code.mkdir()
    head = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    subprocess.run(['git', '-C', str(ROOT), 'archive', '--format=tar', '-o', str(code / 'source.tar'), 'HEAD'], check=True)
    rec = {'schema': 'ACSR1-snapshot-v1', 'tag': tag, 'utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
           'source_commit': head, 'source_tar_sha256': sha(code / 'source.tar'), 'private_files': len(src_files),
           'private_bytes': sum(v['bytes'] for v in src_files.values()), 'manifest': src_files}
    (dst / 'SNAPSHOT_MANIFEST.json').write_text(json.dumps(rec, indent=2, sort_keys=True))
    return rec


def restore(tag, restore_tag):
    return restore_from(CACHE / 'backups' / tag, restore_tag)


def restore_from(src, restore_tag):
    """Restore from one snapshot directory ALONE (same-device backup or its off-device copy) into restore_tests/<tag>."""
    src = Path(src)
    man = json.loads((src / 'SNAPSHOT_MANIFEST.json').read_text())
    dst = CACHE / 'restore_tests' / restore_tag
    if dst.exists():
        raise PermissionError('restore tag exists')
    dst.mkdir(parents=True, mode=0o700)
    shutil.copytree(src / 'private', dst / 'private', copy_function=shutil.copyfile)
    (dst / 'code').mkdir()
    subprocess.run(['tar', '-xf', str(src / 'code' / 'source.tar'), '-C', str(dst / 'code')], check=True)
    bad = [rel for rel, v in man['manifest'].items() if sha(dst / 'private' / rel) != v['sha256']]
    rec = {'schema': 'ACSR1-restore-v1', 'snapshot': man['tag'], 'restore': restore_tag, 'utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
           'files_checked': len(man['manifest']), 'hash_mismatches': len(bad), 'source_tar_sha256_ok': sha(src / 'code' / 'source.tar') == man['source_tar_sha256'],
           'restored_from_snapshot_only': True,
           'source_location': 'off-device copy' if str(src).startswith('/Volumes/') else 'same-device backup'}
    (dst / 'RESTORE_RECEIPT.json').write_text(json.dumps(rec, indent=2, sort_keys=True))
    return rec


def offdevice_copy(tag, dest_root):
    """Copy a same-device snapshot to a mounted external drive and verify every private-file hash ON THE DRIVE."""
    src = CACHE / 'backups' / tag
    man = json.loads((src / 'SNAPSHOT_MANIFEST.json').read_text())
    dest_root = Path(dest_root)
    if not str(dest_root).startswith('/Volumes/') or not os.access(dest_root.parent if not dest_root.exists() else dest_root, os.W_OK):
        raise PermissionError('destination must be a writable path on a mounted external volume')
    dst = dest_root / tag
    if dst.exists():
        raise PermissionError('off-device tag exists')
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, copy_function=shutil.copyfile)
    bad = [rel for rel, v in man['manifest'].items() if sha(dst / 'private' / rel) != v['sha256']]
    rec = {'schema': 'ACSR1-offdevice-copy-v1', 'snapshot': tag, 'utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
           'volume': dest_root.parts[2], 'files_checked': len(man['manifest']), 'hash_mismatches': len(bad),
           'source_tar_sha256_ok': sha(dst / 'code' / 'source.tar') == man['source_tar_sha256'],
           'manifest_identical': sha(dst / 'SNAPSHOT_MANIFEST.json') == sha(src / 'SNAPSHOT_MANIFEST.json')}
    (dst / 'OFFDEVICE_COPY_RECEIPT.json').write_text(json.dumps(rec, indent=2, sort_keys=True))
    return rec


def off_device_status():
    vols = [v for v in os.listdir('/Volumes') if v not in ('Macintosh HD',)] if os.path.isdir('/Volumes') else []
    writable = [v for v in vols if os.access(f'/Volumes/{v}', os.W_OK) and v != 'BackgroundSyncService Setup']
    return {'mounted_volumes': vols, 'writable_external': writable,
            'later_command': f'rsync -a --checksum "{CACHE}/backups/" "/Volumes/<DRIVE>/PCRL-acsr-v1-backups/" && shasum -a 256 -c <(python3 -c "import json;m=json.load(open(\'/Volumes/<DRIVE>/PCRL-acsr-v1-backups/<TAG>/SNAPSHOT_MANIFEST.json\'));[print(v[\'sha256\']+\'  private/\'+k) for k,v in m[\'manifest\'].items()]")'}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['snapshot', 'restore', 'offdevice', 'offdevice-copy', 'restore-from'])
    ap.add_argument('--tag'); ap.add_argument('--restore-tag'); ap.add_argument('--dest'); ap.add_argument('--src')
    a = ap.parse_args()
    if a.cmd == 'snapshot':
        r = snapshot(a.tag); print(json.dumps({k: v for k, v in r.items() if k != 'manifest'}))
    elif a.cmd == 'restore':
        print(json.dumps(restore(a.tag, a.restore_tag)))
    elif a.cmd == 'offdevice-copy':
        print(json.dumps(offdevice_copy(a.tag, a.dest)))
    elif a.cmd == 'restore-from':
        print(json.dumps(restore_from(a.src, a.restore_tag)))
    else:
        print(json.dumps(off_device_status()))
