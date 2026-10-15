"""Versioned private backup and restore-from-backup-alone test.

snapshot: copy the campaign private tree and results into a new versioned
directory under the backup root, with a per-file SHA-256 manifest.
restore: using ONLY the snapshot, verify every hash into a fresh directory,
re-run a deployment package on restored real inputs against the restored
frozen INNER release, and reload representative restored reader models and
reproduce their stored INNER scores.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

SKIP_PARTS = {'__pycache__', 'compute'}  # live queue state is copied after the queue closes


def _sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def snapshot(cfg, verify):
    dest = Path(cfg['backup_dir'])
    if dest.exists():
        raise PermissionError('versioned snapshot directory exists')
    dest.mkdir(parents=True, mode=0o700)
    manifest = []
    for label, src in cfg['trees'].items():
        src = Path(src)
        for p in sorted(src.rglob('*')):
            if p.is_dir() or SKIP_PARTS & set(p.parts) or p.is_symlink():
                continue
            rel = Path(label) / p.relative_to(src)
            q = dest / rel; q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)
            manifest.append({'path': rel.as_posix(), 'sha256': _sha(q), 'bytes': q.stat().st_size, 'source_sha256': _sha(p)})
    bad = [m for m in manifest if m['sha256'] != m['source_sha256']]
    (dest / 'SNAPSHOT_MANIFEST.json').write_text(json.dumps(manifest, indent=1))
    return {'files': len(manifest), 'bytes': sum(m['bytes'] for m in manifest), 'copy_mismatches': len(bad),
            'manifest_sha256': _sha(dest / 'SNAPSHOT_MANIFEST.json'), 'backup_dir': str(dest)}


def restore(cfg, verify):
    src = Path(cfg['backup_dir']); dest = Path(cfg['restore_dir'])
    if dest.exists():
        raise PermissionError('fresh restore directory required')
    mpath = src / 'SNAPSHOT_MANIFEST.json'
    if _sha(mpath) != cfg['manifest_sha256']:
        raise PermissionError('snapshot manifest hash differs')
    manifest = json.loads(mpath.read_text()); bad = 0
    for m in manifest:
        q = dest / m['path']; q.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src / m['path'], q)
        bad += _sha(q) != m['sha256']
    out = {'files_restored': len(manifest), 'hash_mismatches': int(bad)}
    # Deployment package from the restored copy only.
    priv = dest / 'private'; env = {'PYTHONPATH': str(dest / 'code'), 'PATH': '/usr/bin:/bin', 'OMP_NUM_THREADS': '1'}
    checks = {}
    for pkg in cfg['restore_packages']:
        d = priv / pkg['package_rel']; msha = _sha(d / 'MANIFEST.json')
        inp = priv / pkg['input_rel']; o = dest / f"restored_{pkg['name']}.npz"
        r = subprocess.run([sys.executable, '-m', 'task_headroom.deploy', '--package', str(d), '--manifest-sha256', msha, '--input', str(inp),
                            '--input-sha256', _sha(inp), '--output', str(o)], cwd=str(dest / 'code'), env=env, capture_output=True, text=True)
        with np.load(priv / pkg['release_rel']) as z, np.load(o) as c:
            checks[pkg['name']] = {'returncode': r.returncode, 'bitwise': bool(np.array_equal(z['p1'], c['p_income']) and np.array_equal(z['p2'], c['p_occupation'])
                                                                              and np.array_equal(z['d1'], c['decision_income']) and np.array_equal(z['d2'], c['decision_occupation']))}
    out['package_checks'] = checks
    # Representative restored readers reproduce stored INNER scores (subprocess on the restored code only).
    r = subprocess.run([sys.executable, '-m', 'task_headroom.custody_worker', '--reader-check', json.dumps({'priv': str(priv), 'readers': cfg['restore_readers']})],
                       cwd=str(dest / 'code'), env=env, capture_output=True, text=True)
    rchecks = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {'error': {'bitwise': False, 'stderr': r.stderr[-400:]}}
    out['reader_checks'] = rchecks
    out['status'] = 'PASS' if not bad and all(c['bitwise'] and c['returncode'] == 0 for c in checks.values()) and all(c['bitwise'] for c in rchecks.values()) else 'FAIL'
    return out


def reader_check(arg):
    import joblib
    from task_headroom.reader_worker import tables, SLATE
    from task_conditioned_representation.attacks import predict_reader, orient
    a = json.loads(arg); priv = Path(a['priv']); kinds = dict(SLATE); res = {}
    for rd in a['readers']:
        bank = priv / rd['bank_rel']; rec = json.loads((bank / 'FROZEN_COMPLETE.json').read_text())
        with np.load(bank / f"INNER_RELEASE__{rd['service']}.npz") as z:
            w = [{'probability': z['p1'], 'decision': z['d1']}, {'probability': z['p2'], 'decision': z['d2']}]
        t = tables(w); view = rd['view']; name = rec['services'][rd['service']]['winner'][view]['reader']
        if name.startswith('IGNORE_'):
            other = name.split('__')[0].removeprefix('IGNORE_'); view = 'income' if other == 'occupation' else 'occupation'; name = name.split('__')[1]
        model = joblib.load(bank / f"MODEL__{rd['service']}__{view}__{name}.joblib")
        sign = rec['services'][rd['service']]['validation'][view][name]['orientation']
        sc = orient(predict_reader(model, t[view][kinds[name]]), sign)
        with np.load(bank / 'INNER_SCORES.npz') as z:
            stored = z[f"{rd['service']}__{view}__{name}"]
        res[f"{rd['bank_rel']}|{rd['service']}|{rd['view']}"] = {'reader': name, 'bitwise': bool(np.array_equal(sc, stored))}
    return res


def run(spec_path):
    from .evaluation_queue import queue_config
    active, cfg, verify, guard = queue_config(spec_path, 'custody')
    res = snapshot(cfg, verify) if cfg['operation'] == 'snapshot' else restore(cfg, verify)
    res.update({'schema': 'THP1-custody-v1', 'queue_unit_id': active['unit_id'], 'operation': cfg['operation'],
                'off_device': 'NOT PERFORMED: same device only'})
    with Path(cfg['output']).open('x') as f:
        json.dump(res, f, indent=2, sort_keys=True)
    if res.get('status') == 'FAIL':
        raise AssertionError('restore failed')
    return res


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--spec'); p.add_argument('--reader-check'); a = p.parse_args()
    if a.reader_check:
        print(json.dumps(reader_check(a.reader_check)))
    else:
        run(a.spec)
