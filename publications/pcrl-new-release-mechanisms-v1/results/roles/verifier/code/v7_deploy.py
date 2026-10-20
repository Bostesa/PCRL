"""V7 deployment from restored artifacts (label-free): the verifier re-hashes a custody backup's MANIFEST.sha256 itself, extracts
code.tar and private.tar into its OWN new, empty directory, runs v7_child.py there (restored code, restored model files, restored
admitted copies; the module paths are asserted to lie under the restored root), records bitwise parity and refusals per package,
and deletes the extraction afterwards. A read-back on the same device right after a backup is WARM, never a cold check.

    python3 .../v7_deploy.py --tag V1 --backup BACKUP_DIR --packages PKG1 PKG2 ...
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402

SCRATCH = Path('<session scratch>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--backup', required=True); ap.add_argument('--packages', nargs='+', required=True)
    ap.add_argument('--interpreter', default=sys.executable, help='interpreter of the child (the torch venv for torch-based families)')
    a = ap.parse_args()
    b = Path(a.backup).resolve()
    man = {}
    for line in (b / 'MANIFEST.sha256').read_text().splitlines():
        if line.strip():
            h, f = line.split('  ', 1)
            man[f] = h
    rehash = {f: V.file_sha(b / f) == h for f, h in man.items()}
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    work = SCRATCH / f'restore_{a.tag}_{stamp}'
    work.mkdir(parents=True, exist_ok=False)
    res = {'backup': str(b), 'manifest_files': sorted(man), 'manifest_rehash_ok': all(rehash.values()) and bool(rehash),
           'extraction_dir': str(work), 'warm_same_device': True}
    try:
        for t in ('code.tar', 'private.tar'):
            r = subprocess.run(['tar', '-xf', str(b / t), '-C', str(work)], capture_output=True, text=True)
            res[f'extract_{t}'] = r.returncode == 0
        out = work / 'V7_CHILD.json'
        res['child_interpreter'] = a.interpreter
        r = subprocess.run([a.interpreter, str(V.CODE_DIR / 'v7_child.py'), '--root', str(work), '--live-runs', str(V.RUNS),
                            '--packages', *a.packages, '--out', str(out)], cwd=str(work), capture_output=True, text=True,
                           env={'PATH': '/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin', 'OMP_NUM_THREADS': '1',
                                'OPENBLAS_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1', 'HOME': str(Path.home())})
        res['child_returncode'] = r.returncode
        res['child_stderr_tail'] = r.stderr[-1500:]
        child = json.loads(out.read_text()) if out.is_file() else None
        res['child'] = child
    finally:
        shutil.rmtree(work, ignore_errors=True)
        res['extraction_deleted'] = not work.exists()
    child = res.get('child') or {}
    pk = child.get('packages', {})
    ok_num = bool(pk) and all(x.get('all_bitwise') and x.get('package_release_equals_live') for x in pk.values()) and \
        len(pk) == len(a.packages)
    ok_proc = res['manifest_rehash_ok'] and child.get('modules_under_restored_root') and \
        all(x.get('all_refused') and x.get('only_probability_and_decision') for x in pk.values()) and bool(pk)
    chk = {'id': f'V7.DEPLOY_RESTORED__{a.tag}',
           'what': 'packages run from the restored backup alone (own manifest re-hash, own extraction, restored code and model files): '
                   'bitwise parity of the wrapper API, an own CSV and the underlying entry point with the restored stored wire, the '
                   'restored and the LIVE evaluated DEV_REPORT release, MODEL_SELECT parity with the evaluated wire, first-argmax '
                   'decisions, refusal probes (registered and own), exported keys; warm same-device restore',
           'inputs': {'backup_manifest': {f: man[f] for f in sorted(man)}, 'packages': a.packages},
           'result': {k: x for k, x in res.items() if k != 'child'} | {'packages': pk, 'modules': child.get('modules')},
           'numerical_status': 'PASS' if ok_num else 'FAIL', 'procedural_status': 'PASS' if ok_proc else 'FAIL'}
    print(json.dumps(V.append_checks([chk]), indent=1))
    print(json.dumps({'manifest_rehash_ok': res['manifest_rehash_ok'], 'extraction_deleted': res['extraction_deleted'],
                      'packages': {n: {'all_bitwise': x.get('all_bitwise'), 'all_refused': x.get('all_refused'),
                                       'not_refused': [k for k, y in (x.get('refusals') or {}).items() if not y.startswith('refused')],
                                       'not_bitwise': [k for k, y in (x.get('bitwise') or {}).items() if not y],
                                       'load_error': x.get('load_error')} for n, x in pk.items()},
                      'child_rc': res.get('child_returncode'), 'stderr': res.get('child_stderr_tail', '')[-600:]}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
