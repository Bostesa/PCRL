"""Custody for the frozen-finalist validation study (private arrays and fitted artifacts stay on owned devices).

    python -m ffv.custody backup DEST_PARENT      # same-device backup + restore-and-deploy check from the backup alone
    python -m ffv.custody offdevice SRC_DIR DEST  # copy one backup elsewhere and repeat the check there (only if a drive is
                                                  # present; nothing is mounted, unmounted or forced by this tool)

A backup holds: TIP_COMMIT; the branch bundle (source tip 9e2b6c567 .. this branch); code.tar (git archive of the code and the
public records the check needs); private.tar (this campaign's private tree: admitted copies, readers, scores, private draws,
deployment detail, compute ledger, access logs, verifier outputs); MANIFEST.sha256; RESTORE_CHECK.json. The check re-hashes the
manifest, verifies the bundle, extracts both archives into a fresh directory and runs ffv.restore_check there with the restored
code. A read-back right after writing on a still-mounted drive is WARM (possibly served from cache), never a cold check.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRANCH = 'research/pcrl-frozen-finalist-validation-v1'
BASE = '9e2b6c567d7d10dbfae75133e8def490556360b6'
CODE = ['ffv', 'amd', 'scm_bench', 'acs_replication', 'acs_hybrids', 'task_conditioned_representation', 'task_headroom', 'mechanism_design',
        'results/pcrl_frozen_finalist_validation_v1', 'results/pcrl_autonomous_method_development_v1', 'results/pcrl_acs_protected_tree_replication_v1',
        'results/pcrl_acs_frozen_hybrids_v1', 'results/pcrl_scm_engineering_benchmark_v1', 'results/pcrl_scm_utility_matched_v1']
PY = os.environ.get('FFV_PYTHON', sys.executable)


def _sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _run(args, cwd=None, env=None):
    r = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def restore_check(d: Path, read_back: str) -> dict:
    files = ['TIP_COMMIT', 'branch_since_base.bundle', 'code.tar', 'private.tar']
    man = dict(l.split('  ', 1)[::-1] for l in (d / 'MANIFEST.sha256').read_text().splitlines() if l.strip())
    rehash = all(_sha(d / f) == man[f] for f in files)
    rc_b, out_b = _run(['git', '-C', str(ROOT), 'bundle', 'verify', str(d / 'branch_since_base.bundle')])
    w = Path(tempfile.mkdtemp(prefix='ffv_restore_'))
    try:
        _run(['tar', '-xf', str(d / 'code.tar'), '-C', str(w)]); _run(['tar', '-xf', str(d / 'private.tar'), '-C', str(w)])
        env = {**os.environ, 'PYTHONPATH': str(w), 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
        rc, out = _run([PY, '-m', 'ffv.restore_check', str(w / 'restore_deploy_check.json')], cwd=str(w), env=env)
        dep = json.loads((w / 'restore_deploy_check.json').read_text()) if (w / 'restore_deploy_check.json').exists() else {'all_ok': False, 'error': out[-1500:]}
    finally:
        shutil.rmtree(w, ignore_errors=True)
    rec = {'tip_commit': (d / 'TIP_COMMIT').read_text().strip(), 'manifest_rehash_ok': rehash, 'bundle_verify': rc_b == 0 and 'is okay' in out_b,
           'deploy_from_backup': dep, 'read_back': read_back, 'checked_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}
    rec['all_ok'] = rec['manifest_rehash_ok'] and rec['bundle_verify'] and bool(dep.get('all_ok'))
    (d / 'RESTORE_CHECK.json').write_text(json.dumps(rec, indent=1) + '\n')
    return rec


def backup(parent: Path) -> dict:
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    d = parent / f'pcrl_frozen_finalist_validation_v1_{stamp}'; d.mkdir(parents=True)
    tip = _run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'])[1].strip()
    (d / 'TIP_COMMIT').write_text(tip + '\n')
    _run(['git', '-C', str(ROOT), 'bundle', 'create', str(d / 'branch_since_base.bundle'), f'{BASE}..{BRANCH}'])
    _run(['git', '-C', str(ROOT), 'archive', '--format=tar', '-o', str(d / 'code.tar'), 'HEAD', *CODE])
    _run(['tar', '-cf', str(d / 'private.tar'), '-C', str(ROOT), '.private/pcrl_frozen_finalist_validation_v1'])
    (d / 'MANIFEST.sha256').write_text(''.join(f'{_sha(d / f)}  {f}\n' for f in ('TIP_COMMIT', 'branch_since_base.bundle', 'code.tar', 'private.tar')))
    rec = restore_check(d, 'same device (local disk)')
    rec['path'] = str(d)
    return rec


def offdevice(src: Path, dest: Path) -> dict:
    if not dest.parent.exists():
        return {'status': 'UNAVAILABLE', 'reason': f'destination volume not present: {dest.parent}'}
    dest.mkdir(parents=True, exist_ok=True)
    tgt = dest / src.name
    shutil.copytree(src, tgt)
    (tgt / 'RESTORE_CHECK.json').unlink(missing_ok=True)
    rec = restore_check(tgt, 'WARM: the drive stayed mounted between the copy and the check; a cold re-check after an ordinary remount is pending')
    rec['path'] = str(tgt)
    return rec


if __name__ == '__main__':
    if sys.argv[1] == 'backup':
        print(json.dumps(backup(Path(sys.argv[2])), indent=1))
    elif sys.argv[1] == 'offdevice':
        print(json.dumps(offdevice(Path(sys.argv[2]), Path(sys.argv[3])), indent=1))
    else:
        raise SystemExit(__doc__)
