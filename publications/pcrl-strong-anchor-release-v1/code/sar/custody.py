"""Custody for the PCRL strong-anchor release campaign (private arrays and fitted artifacts stay on owned devices).

    python3 -m sar.custody backup DEST_PARENT      # same-device backup + restore-and-check from the backup alone
    python3 -m sar.custody offdevice SRC_DIR DEST  # copy one backup to an already attached drive and repeat the check there (nothing is
                                                  # mounted, unmounted or forced by this tool)

Adapted from frc.custody / nrm.custody (constants, plus the disk-budget change below). A backup holds: TIP_COMMIT; the branch bundle (base = the dual-design
release research tip c577424c0 this branch started from .. this branch); code.tar (git archive of the code and the public
records the check needs); private.tar.gz (this campaign's private tree, plus the admitted source copies the code reads);
MANIFEST.sha256; RESTORE_CHECK.json.

Disk budget (this campaign, written 2026-10-22 05:45Z; PROTOCOL_DEVIATIONS item 11): the volume had about 4.4 GiB free, and a backup of the
inherited scope (about 2.5 GB) plus its full extraction for the restore check would not fit. So the private archive is gzip-compressed
(private.tar.gz; bsdtar detects the compression when extracting), and the occupation-headroom campaign's run tree is left out. The
admitted mirror and the frozen reference bank this campaign reads stay in scope (they are under the income-headroom tree's
`admitted/`), and the occupation-headroom runs keep their own same-device backup (pcrl-occupation-headroom-overnight-v1-backup). The
restore check reads nothing from the excluded tree. The check re-hashes the manifest, verifies the bundle, extracts both archives into a fresh
directory and runs sar.restore_check there with the restored code. A read-back right after writing on a still-mounted drive is WARM
(possibly served from cache), never a cold check.
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
CAMPAIGN = 'pcrl_strong_anchor_release_v1'
BRANCH = 'research/pcrl-strong-anchor-release-v1'
BASE = 'c577424c0'                                                   # the dual-design release research tip = this branch's base
CODE = ['sar', 'ddr', 'fsa', 'frc', 'nrm', 'tsz', 'cts', 'ohn', 'ihs', 'ffv', 'amd', 'scm_bench', 'acs_replication', 'acs_hybrids',
        'task_conditioned_representation', 'task_headroom', 'mechanism_design', f'results/{CAMPAIGN}',
        'results/pcrl_dual_design_release_v1',
        'results/pcrl_finite_solver_adequacy_v1', 'results/pcrl_finite_routing_comparison_v1', 'results/pcrl_new_release_mechanisms_v1',
        'results/pcrl_controlled_training_size_v1',
        'results/pcrl_occupation_headroom_overnight_v1', 'results/pcrl_income_headroom_successor_v1',
        'results/pcrl_frozen_finalist_validation_v1', 'results/pcrl_autonomous_method_development_v1',
        'results/pcrl_acs_protected_tree_replication_v1', 'results/pcrl_acs_frozen_hybrids_v1', 'results/pcrl_scm_engineering_benchmark_v1',
        'results/pcrl_scm_utility_matched_v1', 'results/pcrl_correction_transfer_stability_v1']
PRIVATE = [f'.private/{CAMPAIGN}', '.private/pcrl_dual_design_release_v1/runs/DA/DA_TWIN',
           '.private/pcrl_dual_design_release_v1/runs/DB/DB_PTC_R1', '.private/pcrl_dual_design_release_v1/runs/DB/DB_ZTILT_R1',
           '.private/pcrl_finite_routing_comparison_v1/runs/FA', '.private/pcrl_finite_routing_comparison_v1/fa_cache',
           '.private/pcrl_new_release_mechanisms_v1/runs/FA', '.private/pcrl_new_release_mechanisms_v1/fa_cache',
           '.private/pcrl_occupation_headroom_overnight_v1/admitted_logs', '.private/pcrl_income_headroom_successor_v1']
EXCLUDE = [f'.private/{CAMPAIGN}/compute/tickets']
PY = os.environ.get('SAR_PYTHON', 'python3')
TORCH_PY = '<repository>/.venv/bin/python'                 # the reader grid's interpreter (torch venv)
FILES = ('TIP_COMMIT', 'branch_since_base.bundle', 'code.tar', 'private.tar.gz')


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
    man = dict(l.split('  ', 1)[::-1] for l in (d / 'MANIFEST.sha256').read_text().splitlines() if l.strip())
    rehash = all(_sha(d / f) == man[f] for f in FILES)
    rc_b, out_b = _run(['git', '-C', str(ROOT), 'bundle', 'verify', str(d / 'branch_since_base.bundle')])
    w = Path(tempfile.mkdtemp(prefix='ipr_restore_'))
    try:
        _run(['tar', '-xf', str(d / 'code.tar'), '-C', str(w)]); _run(['tar', '-xf', str(d / 'private.tar.gz'), '-C', str(w)])
        env = {**os.environ, 'PYTHONPATH': str(w), 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}
        passes = {}
        for label, exe in (('python3', PY), ('torch_venv', TORCH_PY)):   # each family re-runs under its own fit interpreter (ENG1-04)
            outp = w / f'restore_check_{label}.json'
            rc, out = _run([exe, '-m', 'sar.restore_check', str(outp)], cwd=str(w), env=env)
            passes[label] = json.loads(outp.read_text()) if outp.exists() else {'all_ok': False, 'error': out[-1500:]}
        rerun = {}
        for p in passes.values():
            rerun.update(p.get('models_rerun_bitwise') or {})
        covered = set(rerun)
        notrun = set().union(*[set(k for k in (p.get('models_not_rerun_in_this_pass') or {}) if not k.endswith('__error'))
                              for p in passes.values()]) - covered
        seen = set().union(*[set(p.get('representative_readers_bitwise') or {}) | set(p.get('readers_judged_in_other_pass') or {})
                             for p in passes.values()])
        bitwise = {r for p in passes.values() for r, ok in (p.get('representative_readers_bitwise') or {}).items() if ok}
        dep = {'passes': passes, 'models_rerun_bitwise': rerun, 'models_never_rerun': sorted(notrun),
               'readers_never_bitwise': sorted(seen - bitwise),           # ENG2-06: every reader bitwise in at least one pass
               'all_ok': all(bool(p.get('all_ok')) for p in passes.values()) and not notrun and all(rerun.values())
               and not (seen - bitwise)}
    finally:
        shutil.rmtree(w, ignore_errors=True)
    rec = {'tip_commit': (d / 'TIP_COMMIT').read_text().strip(), 'manifest_rehash_ok': rehash, 'bundle_verify': rc_b == 0 and 'is okay' in out_b,
           'restore_check_from_backup': dep, 'read_back': read_back, 'checked_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}
    rec['all_ok'] = rec['manifest_rehash_ok'] and rec['bundle_verify'] and bool(dep.get('all_ok'))
    (d / 'RESTORE_CHECK.json').write_text(json.dumps(rec, indent=1) + '\n')
    return rec


def backup(parent: Path) -> dict:
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    d = parent / f'{CAMPAIGN}_{stamp}'; d.mkdir(parents=True)
    tip = _run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'])[1].strip()
    (d / 'TIP_COMMIT').write_text(tip + '\n')
    _run(['git', '-C', str(ROOT), 'bundle', 'create', str(d / 'branch_since_base.bundle'), f'{BASE}..{BRANCH}'])
    _run(['git', '-C', str(ROOT), 'archive', '--format=tar', '-o', str(d / 'code.tar'), 'HEAD', *CODE])
    _run(['tar', '-czf', str(d / 'private.tar.gz'), *[f'--exclude={e}' for e in EXCLUDE], '-C', str(ROOT), *PRIVATE])
    (d / 'MANIFEST.sha256').write_text(''.join(f'{_sha(d / f)}  {f}\n' for f in FILES))
    rec = restore_check(d, 'same device (local disk)')
    rec['path'] = str(d)
    rec['size_bytes'] = sum((d / f).stat().st_size for f in FILES)
    rec['manifest_sha256'] = {f: _sha(d / f) for f in FILES}
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
