"""Versioned same-device backup of the completion campaign and a restore-from-backup-alone test.

snapshot: copy the campaign private tree (queue state excluded while live), results, the frozen packages and
original reader banks (so restores are self-contained), and the code needed to run them, with a per-file
SHA-256 manifest. restore: from the snapshot ONLY, verify every hash into a fresh directory, re-run the frozen
deployment packages on the restored real input against the restored stored INNER releases, and reload
restored new-bank and original-bank reader models to reproduce their stored INNER scores bitwise.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

SKIP = {'__pycache__', 'compute'}


def _sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def snapshot(cfg):
    dest = Path(cfg['backup_dir'])
    if dest.exists():
        raise PermissionError('versioned snapshot directory exists')
    dest.mkdir(parents=True, mode=0o700); manifest = []
    for label, src in cfg['trees'].items():
        src = Path(src)
        for p in sorted(src.rglob('*')):
            if p.is_dir() or p.is_symlink() or SKIP & set(p.relative_to(src).parts):
                continue
            rel = Path(label) / p.relative_to(src); q = dest / rel; q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)
            manifest.append({'path': rel.as_posix(), 'sha256': _sha(q), 'bytes': q.stat().st_size, 'source_sha256': _sha(p)})
    (dest / 'SNAPSHOT_MANIFEST.json').write_text(json.dumps(manifest, indent=1))
    return {'files': len(manifest), 'bytes': sum(m['bytes'] for m in manifest), 'copy_mismatches': sum(m['sha256'] != m['source_sha256'] for m in manifest),
            'manifest_sha256': _sha(dest / 'SNAPSHOT_MANIFEST.json'), 'backup_dir': str(dest)}


def reader_check(arg):
    import joblib
    from task_headroom.reader_worker import tables, SLATE
    from task_conditioned_representation.attacks import predict_reader, orient
    a = json.loads(arg); root = Path(a['root']); kinds = dict(SLATE); res = {}
    for rd in a['readers']:
        bank = root / rd['bank_rel']; rec = json.loads((bank / 'FROZEN_COMPLETE.json').read_text())
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


def restore(cfg):
    src = Path(cfg['backup_dir']); dest = Path(cfg['restore_dir'])
    if dest.exists():
        raise PermissionError('fresh restore directory required')
    mpath = src / 'SNAPSHOT_MANIFEST.json'
    if _sha(mpath) != cfg['manifest_sha256']:
        raise PermissionError('snapshot manifest hash differs')
    manifest = json.loads(mpath.read_text()); bad = 0
    for m in manifest:
        q = dest / m['path']; q.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src / m['path'], q); bad += _sha(q) != m['sha256']
    out = {'files_restored': len(manifest), 'hash_mismatches': int(bad), 'package_checks': {}}
    env = {'PYTHONPATH': str(dest / 'code'), 'PATH': '/usr/bin:/bin', 'OMP_NUM_THREADS': '1'}
    for pkg in cfg['restore_packages']:
        d = dest / pkg['package_rel']; msha = _sha(d / 'MANIFEST.json'); inp = dest / pkg['input_rel']; o = dest / f"restored_{pkg['name']}.npz"
        r = subprocess.run([sys.executable, '-m', 'task_headroom.deploy', '--package', str(d), '--manifest-sha256', msha, '--input', str(inp),
                            '--input-sha256', _sha(inp), '--output', str(o)], cwd=str(dest / 'code'), env=env, capture_output=True, text=True)
        ok = False
        if r.returncode == 0:
            with np.load(dest / pkg['release_rel']) as z, np.load(o) as c:
                ok = bool(np.array_equal(z['p1'], c['p_income']) and np.array_equal(z['p2'], c['p_occupation'])
                          and np.array_equal(z['d1'], c['decision_income']) and np.array_equal(z['d2'], c['decision_occupation']))
        out['package_checks'][pkg['name']] = {'returncode': r.returncode, 'bitwise': ok}
    r = subprocess.run([sys.executable, '-m', 'tree_completion.custody_worker', '--reader-check', json.dumps({'root': str(dest), 'readers': cfg['restore_readers']})],
                       cwd=str(dest / 'code'), env=env, capture_output=True, text=True)
    out['reader_checks'] = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {'error': {'bitwise': False, 'stderr': r.stderr[-400:]}}
    out['status'] = 'PASS' if not bad and all(v['bitwise'] for v in out['package_checks'].values()) and all(v['bitwise'] for v in out['reader_checks'].values()) else 'FAIL'
    return out


def run(spec_path):
    from .evaluation_queue import queue_config
    active, cfg, verify, guard = queue_config(spec_path, 'custody')
    res = snapshot(cfg) if cfg['operation'] == 'snapshot' else restore(cfg)
    res.update({'schema': 'TC1-custody-v1', 'queue_unit_id': active['unit_id'], 'operation': cfg['operation'], 'off_device': 'NOT PERFORMED: no writable external drive'})
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
