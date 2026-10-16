"""Custody for the mechanism-design campaign: versioned snapshot, restore check, and an off-device copy.

snapshot NAME
    Copies this campaign's results directory, its code (mechanism_design/, mechanism_design_independent/) and its
    private queue ledger and unit logs (.private/mechanism_design/compute/) into
    <private backup> with a sha256 manifest. Snapshots are never overwritten.
restore-check SNAPSHOT RESTORE_DIR
    Copies a snapshot into a fresh directory, verifies every hash, and, when the snapshot contains a prototype
    (code/mechanism_design/prototype/run_prototype.py and its stored results), re-runs the prototype from the
    restored code alone and requires its results to equal the stored ones exactly.
offdevice SNAPSHOT DRIVE_DIR
    Writes the snapshot as one tar archive to the external drive (exFAT-safe), reads it back, extracts it to a fresh
    directory and verifies the manifest. A missing or unwritable drive is reported, not treated as success.
No real-person data exists in this campaign; nothing here reads any.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results' / 'pcrl_mechanism_design_v1'
PRIV = ROOT / '.private' / 'mechanism_design'
BACKUPS = Path.home() / 'PCRL_eval_cache_private' / 'mechanism_design_v1' / 'backups'
CODE_DIRS = ('mechanism_design', 'mechanism_design_independent')
DEPENDENCY_FILES = ('task_conditioned_representation/__init__.py', 'task_conditioned_representation/attacks.py',
                    'task_conditioned_representation/inference.py')   # reader slate used by the prototype's continuous laws


def sha(p: Path) -> str:
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def _files():
    for f in sorted(RESULTS.rglob('*')):
        if f.is_file():
            yield f, Path('results') / f.relative_to(RESULTS)
    for d in CODE_DIRS:
        base = ROOT / d
        if base.is_dir():
            for f in sorted(base.rglob('*')):
                if f.is_file() and '__pycache__' not in f.parts:
                    yield f, Path('code') / d / f.relative_to(base)
    for rel in DEPENDENCY_FILES:
        f = ROOT / rel
        if f.is_file():
            yield f, Path('code') / rel
    comp = PRIV / 'compute'
    if comp.is_dir():
        for f in sorted(comp.rglob('*')):
            if f.is_file() and not f.name.endswith('.lock'):
                yield f, Path('private') / 'compute' / f.relative_to(comp)


def snapshot(name: str) -> dict:
    dest = BACKUPS / name
    if dest.exists():
        raise PermissionError('snapshot exists (versioned, never overwritten)')
    dest.mkdir(parents=True, mode=0o700)
    man = []
    for src, rel in _files():
        d = dest / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, d)
        man.append({'path': str(rel), 'sha256': sha(d), 'bytes': d.stat().st_size, 'source_sha256': sha(src)})
    if any(m['sha256'] != m['source_sha256'] for m in man):
        raise PermissionError('snapshot copy differs from source')
    head = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    meta = {'name': name, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'git_head': head, 'files': len(man),
            'bytes': sum(m['bytes'] for m in man)}
    (dest / 'SNAPSHOT_MANIFEST.json').write_text(json.dumps({'meta': meta, 'files': man}, indent=1))
    return {**meta, 'snapshot': str(dest).replace(str(Path.home()), '~'), 'manifest_sha256': sha(dest / 'SNAPSHOT_MANIFEST.json')}


def _verify(rest: Path) -> dict:
    man = json.loads((rest / 'SNAPSHOT_MANIFEST.json').read_text())
    bad = [m['path'] for m in man['files'] if not (rest / m['path']).is_file() or sha(rest / m['path']) != m['sha256']]
    return {'files': len(man['files']), 'hash_mismatches': len(bad), 'mismatched': bad[:20], 'git_head': man['meta']['git_head']}


def restore_check(snapshot_dir: str, restore_dir: str) -> dict:
    snap, rest = Path(snapshot_dir), Path(restore_dir)
    if rest.exists():
        raise PermissionError('restore directory must be fresh')
    shutil.copytree(snap, rest)
    out = _verify(rest)
    runner = rest / 'code' / 'mechanism_design' / 'prototype' / 'run_prototype.py'
    stored = rest / 'results' / 'prototype' / 'PROTOTYPE_RESULTS.json'
    if runner.is_file() and stored.is_file():
        rerun = rest / 'rerun_PROTOTYPE_RESULTS.json'
        env = {**os.environ, 'PYTHONPATH': str(rest / 'code'), 'OMP_NUM_THREADS': '1'}
        proc = subprocess.run([sys.executable, '-m', 'mechanism_design.prototype.run_prototype', '--out', str(rerun)],
                              cwd=str(rest), env=env, capture_output=True, text=True)
        out['prototype_rerun_returncode'] = proc.returncode
        out['prototype_rerun_stderr_tail'] = proc.stderr[-2000:]
        if proc.returncode == 0:
            a, b = json.loads(stored.read_text()), json.loads(rerun.read_text())
            a.pop('provenance', None); b.pop('provenance', None)
            out['prototype_results_identical'] = (a == b)
    else:
        out['prototype_rerun'] = 'not applicable (no prototype in snapshot)'
    return out


def offdevice(snapshot_dir: str, drive_dir: str) -> dict:
    snap, drive = Path(snapshot_dir), Path(drive_dir)
    out = {'drive_dir': str(drive)}
    if not drive.parent.exists():
        return {**out, 'status': 'PENDING_DRIVE_ABSENT'}
    try:
        drive.mkdir(parents=True, exist_ok=True)
        tar_path = drive / (snap.name + '.tar')
        if tar_path.exists():
            raise PermissionError('off-device archive exists (versioned, never overwritten)')
        with tarfile.open(tar_path, 'w') as t:
            t.add(snap, arcname=snap.name)
        os.sync()
    except OSError as e:
        return {**out, 'status': 'PENDING_DRIVE_UNWRITABLE', 'error': str(e)}
    out['tar_sha256'] = sha(tar_path)
    out['tar_bytes'] = tar_path.stat().st_size
    check = Path(str(snap) + '_DRIVE_READBACK')
    if check.exists():
        raise PermissionError('read-back directory exists')
    with tarfile.open(tar_path, 'r') as t:
        t.extractall(check, filter='data')
    out['readback'] = _verify(check / snap.name)
    out['status'] = 'PASS' if out['readback']['hash_mismatches'] == 0 else 'FAIL'
    out['note'] = 'read back immediately after writing; it may have been served from the file cache'
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('snapshot'); s.add_argument('--name', required=True)
    r = sub.add_parser('restore-check'); r.add_argument('--snapshot', required=True); r.add_argument('--restore', required=True)
    o = sub.add_parser('offdevice'); o.add_argument('--snapshot', required=True); o.add_argument('--drive-dir', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'snapshot':
        res = snapshot(a.name)
    elif a.cmd == 'restore-check':
        res = restore_check(a.snapshot, a.restore)
    else:
        res = offdevice(a.snapshot, a.drive_dir)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
