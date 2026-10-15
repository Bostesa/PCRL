"""Stage RESTORE: rebuild from the same-device snapshot alone, verify every manifest hash, and reproduce stored
outputs bitwise: the L_TREE_S0 and T_TREE_S1 deployment packages on the restored input_INNER.npz (restored runtime
code, plus an own numpy tree reader for comparison) and new-bank reader scores from restored joblib models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from . import core


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def own_tree_release(pkg, X, names):
    """Own oblivious-tree runtime: bit i = float32(x_f) > float32(border); leaf = sum bit_i << i; raw = scale*sum leaf
    values + bias + sum theta/T; income softmax(-d/2, d/2); occupation softmax(raw - mean)."""
    m = json.loads((pkg / 'MANIFEST.json').read_text())
    if [str(x) for x in names] != m['feature_names']:
        raise ValueError('feature order differs')
    theta = {}
    if m.get('theta'):
        with np.load(pkg / m['theta'], allow_pickle=False) as z:
            theta = {'income': z['theta_income'].copy(), 'occupation': z['theta_occupation'].copy()}
    out = {}
    X32 = np.asarray(X, dtype=np.float32)
    for task in ('income', 'occupation'):
        t = json.loads((pkg / m['teachers'][task]).read_text())
        F = np.asarray(t['split_features'], dtype=np.int64); Bd = np.asarray(t['split_borders'], dtype=np.float32)
        T, depth = F.shape
        leaves = np.zeros((len(X32), T), dtype=np.int64)
        for i in range(depth):
            leaves |= (X32[:, F[:, i]] > Bd[None, :, i]).astype(np.int64) << i
        lv = np.asarray(t['leaf_values'], dtype=np.float64)          # (T, 2^depth, dim)
        dim = lv.shape[2]
        raw = float(t['scale']) * lv[np.arange(T)[None, :], leaves].sum(1) + np.asarray(t['bias'], dtype=np.float64)[None, :]
        if task in theta:
            th = np.asarray(theta[task], dtype=np.float64).reshape(T, 2 ** depth, dim)
            raw = raw + th[np.arange(T)[None, :], leaves].sum(1) / T
        if task == 'income':
            d = raw[:, 0]; z = np.column_stack((-d / 2, d / 2))
        else:
            z = raw - raw.mean(1, keepdims=True)
        e = np.exp(z - z.max(1, keepdims=True)); p = e / e.sum(1, keepdims=True)
        out[task] = (p, p.argmax(1))
    return out


def run(spec):
    from tree_completion.evaluation_queue import queue_config
    active, cfg, verify, guard = queue_config(spec, 'verification')
    R = core.Receipt('RESTORE', active, cfg)
    cfg['_code_sha256'] = core.code_hashes(verify, cfg)
    snap = Path(cfg['snapshot_root']); dest = Path(cfg['restore_dir'])
    man_item = cfg['manifest']
    R.check('snapshot_manifest_sha_equals_announced', man_item['sha256'] == cfg['announced_manifest_sha256'])
    manifest = core.load_json(verify, man_item)
    R.check('manifest_has_1518_entries', len(manifest) == 1518 and len({e['path'] for e in manifest}) == 1518, entries=len(manifest))
    if dest.exists():
        raise SystemExit('restore directory must be fresh')
    dest.mkdir(parents=True, mode=0o700)
    # ---------------------------------------------------------------- restore every file from the snapshot alone
    bad = []; nbytes = 0
    guard()
    for e in manifest:
        src = snap / e['path']; it = cfg['files'].get(e['path'])
        if it is None or it['sha256'] != e['sha256'] or e.get('source_sha256', e['sha256']) != e['sha256']:
            bad.append(e['path']); continue
        b = verify(str(src), it['sha256']).read_bytes()
        if sha_bytes(b) != e['sha256'] or len(b) != e['bytes']:
            bad.append(e['path']); continue
        out = dest / e['path']; out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'xb') as f:
            f.write(b)
        if sha_file(out) != e['sha256']:
            bad.append(e['path'])
        nbytes += len(b)
    R.check('all_1518_files_restored_and_rehashed', not bad, bad=bad[:20], restored_bytes=nbytes)
    restored = sorted(str(p.relative_to(dest)) for p in dest.rglob('*') if p.is_file())
    R.check('restore_contains_exactly_the_manifest', restored == sorted(e['path'] for e in manifest))

    # lock bindings of the restored artifacts
    fman = core.load_json(verify, cfg['frozen_manifest']); rlock = core.load_json(verify, core.lock_item(cfg, 'reporting_lock'))
    shas = {e['path']: e['sha256'] for e in manifest}
    R.check('restored_L_TREE_S0_package_equals_frozen_manifest',
            {k: shas[f'frozen_packages/L_TREE_S0/{k}'] for k in ('teacher_income.json', 'teacher_occupation.json', 'theta.npz')} ==
            fman['per_seed']['0']['deployment_package']['files_sha256'] and shas['frozen_packages/L_TREE_S0/MANIFEST.json'] ==
            fman['per_seed']['0']['deployment_package']['manifest_sha256'])
    R.check('restored_T_TREE_S1_teachers_equal_frozen_manifest',
            shas['frozen_packages/T_TREE_S1/teacher_income.json'] == fman['per_seed']['1']['teachers']['income']['payload_sha256'] and
            shas['frozen_packages/T_TREE_S1/teacher_occupation.json'] == fman['per_seed']['1']['teachers']['occupation']['payload_sha256'])
    R.check('restored_original_INNER_releases_equal_frozen_manifest',
            shas['frozen_readers_old/S0/INNER_RELEASE__L_TREE.npz'] == fman['stored_inner_releases_sha256']['0']['L_TREE'] and
            shas['frozen_readers_old/S1/INNER_RELEASE__T_TREE.npz'] == fman['stored_inner_releases_sha256']['1']['T_TREE'])
    R.check('restored_new_bank_S0_equals_REPORTING_LOCK',
            shas['private/readers/S0/INNER_SCORES.npz'] == rlock['new_banks']['0']['scores_sha256'] and
            shas['private/readers/S0/FROZEN_COMPLETE.json'] == rlock['new_banks']['0']['receipt_sha256'])

    # ---------------------------------------------------------------- deployment packages (restored runtime + own reader)
    guard()
    inp = dest / 'frozen_packages/input_INNER.npz'
    with np.load(inp, allow_pickle=False) as z:
        X, names = z['X'].copy(), z['feature_names'].copy()
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON')}
    env['PYTHONPATH'] = str(dest / 'code')
    deploy = {}
    for svc, s in (('L_TREE', '0'), ('T_TREE', '1')):
        pkg = dest / f'frozen_packages/{svc}_S{s}'
        outp = dest / f'VERIFIER_DEPLOY__{svc}_S{s}__INNER.npz'
        cmd = [cfg['python'], '-s', '-m', 'task_headroom.deploy', '--package', str(pkg), '--manifest-sha256', shas[f'frozen_packages/{svc}_S{s}/MANIFEST.json'],
               '--input', str(inp), '--input-sha256', shas['frozen_packages/input_INNER.npz'], '--output', str(outp)]
        r = subprocess.run(cmd, cwd=str(dest / 'code'), env=env, capture_output=True, text=True, timeout=900)
        ok_run = r.returncode == 0 and '"OK"' in r.stdout
        R.check(f'restored_runtime_{svc}_S{s}_ran', ok_run, stdout=r.stdout.strip()[-300:], stderr=r.stderr.strip()[-300:])
        with np.load(dest / f'frozen_readers_old/S{s}/INNER_RELEASE__{svc}.npz', allow_pickle=False) as z:
            ref = {k: z[k].copy() for k in z.files}
        with np.load(dest / f'frozen_packages/{svc}_S{s}__INNER.npz', allow_pickle=False) as z:
            stored_dep = {k: z[k].copy() for k in z.files}
        rec = {}
        if ok_run:
            with np.load(outp, allow_pickle=False) as z:
                got = {k: z[k].copy() for k in z.files}
            pairs = (('p_income', 'p1'), ('decision_income', 'd1'), ('p_occupation', 'p2'), ('decision_occupation', 'd2'))
            bit = all(got[a].dtype == ref[b].dtype and got[a].tobytes() == ref[b].tobytes() for a, b in pairs)
            bit_dep = all(got[a].tobytes() == stored_dep[a].tobytes() for a, _ in pairs)
            rec['restored_runtime_bitwise_vs_original_release'] = bit
            rec['restored_runtime_bitwise_vs_stored_deployment_output'] = bit_dep
            R.check(f'restored_runtime_{svc}_S{s}_bitwise_equals_original_INNER_release', bit and bit_dep)
        own = own_tree_release(pkg, X, names)
        dmax = max(float(np.abs(own['income'][0] - ref['p1']).max()), float(np.abs(own['occupation'][0] - ref['p2']).max()))
        dec = np.array_equal(own['income'][1], ref['d1']) and np.array_equal(own['occupation'][1], ref['d2'])
        rec['own_reader_max_abs'] = dmax; rec['own_reader_decisions_exact'] = bool(dec)
        rec['own_reader_bitwise'] = bool(own['income'][0].tobytes() == ref['p1'].tobytes() and own['occupation'][0].tobytes() == ref['p2'].tobytes())
        R.check(f'own_tree_reader_{svc}_S{s}_matches_original_release', dmax <= 1e-12 and dec, **rec)
        deploy[f'{svc}_S{s}'] = rec
    R.numbers['deployment'] = deploy

    # ---------------------------------------------------------------- new-bank reader scores from restored joblib models
    guard()
    root = dest / 'private/readers/S0'
    rec0 = json.loads((root / 'FROZEN_COMPLETE.json').read_text())
    w = rec0['services']['RAW_J_NEQ']['winner']
    jobs = [f"pair:{w['pair']['reader']}", f"occupation:{w['occupation']['reader']}", f"income:{w['income']['reader']}"]
    cmd = [cfg['python'], '-s', str(Path(__file__).resolve().parent / 'restore_reader.py'), str(root), 'RAW_J_NEQ', ','.join(jobs)]
    r = subprocess.run(cmd, cwd=str(dest), env=env, capture_output=True, text=True, timeout=900)
    try:
        res = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        res = {}
    R.check('restored_reader_subprocess_ran', r.returncode == 0 and bool(res), stderr=r.stderr.strip()[-400:])
    for job, v in res.items():
        R.check(f'restored_new_bank_reader_S0_RAW_J_NEQ_{job.replace(":", "_")}_bitwise', v['bitwise'], **v)
    R.numbers['reader_restore'] = res
    R.numbers['restore_dir'] = str(dest)
    return R.write()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--spec', required=True)
    run(ap.parse_args().spec)
