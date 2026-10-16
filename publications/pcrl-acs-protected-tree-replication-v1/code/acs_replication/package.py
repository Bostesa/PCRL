"""Build deployment packages (raw permitted ACS fields -> released probabilities + decisions) and verify them label-free.

Packages: T_TREE and LOCAL_TREE (seed 0 and seed 1) and the strongest completed control. Verification on DEV_REPORT
raw inputs (features only, no labels): deployed probabilities equal the stored released probabilities (<= 1e-12) and
decisions are identical; an incompatible schema (reordered columns), a forbidden input column (SEX) and a hidden-output
request (leaf ids / raw scores) are each refused. Writes private packages and a public DEPLOYMENT_RESULTS.json summary.
"""
from __future__ import annotations
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from . import data as D, deploy, registry, schema

DEPLOY = D.PRIV / 'deploy'


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def build_tree_package(name, seed, theta_label=None):
    enc = D.receipt()['encoding']
    d = DEPLOY / name
    if d.exists():
        raise PermissionError('package exists')
    d.mkdir(parents=True, mode=0o700)
    meta = {'schema': 'ACSR1-deploy-package-v1', 'kind': 'TREE', 'service': name, 'seed': seed, 'encoder': enc, 'teachers': {}, 'files': {}}
    for t in D.TASKS:
        src = D.PRIV / 'runs' / f'STAGE_A_{seed}' / f'TREE_TEACHER_{t}.json'
        shutil.copy2(src, d / src.name); meta['teachers'][t] = src.name; meta['files'][src.name] = sha(d / src.name)
    if theta_label:
        src = D.PRIV / 'runs' / f'STAGE_B_{seed}' / theta_label
        shutil.copy2(src, d / 'THETA.npz'); meta['theta'] = 'THETA.npz'; meta['files']['THETA.npz'] = sha(d / 'THETA.npz')
    (d / 'PACKAGE.json').write_text(json.dumps(meta, indent=2, sort_keys=True))
    return d


def build_neural_package(name, seed, state_path):
    enc = D.receipt()['encoding']
    d = DEPLOY / name
    if d.exists():
        raise PermissionError('package exists')
    d.mkdir(parents=True, mode=0o700)
    shutil.copy2(state_path, d / 'NEURAL_STATE.pt')
    meta = {'schema': 'ACSR1-deploy-package-v1', 'kind': 'NEURAL', 'service': name, 'seed': seed, 'encoder': enc,
            'neural_state': 'NEURAL_STATE.pt', 'files': {'NEURAL_STATE.pt': sha(d / 'NEURAL_STATE.pt')}}
    (d / 'PACKAGE.json').write_text(json.dumps(meta, indent=2, sort_keys=True))
    return d


def write_raw_csv(path, role='DEV_REPORT', columns=None):
    raw = np.load(D.ROOT / D.receipt()['files'][role]['raw_inputs']['path'])
    cols = columns or deploy.PERMITTED
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(cols)
        n = len(raw['row_id'])
        for i in range(n):
            w.writerow([str(raw[c][i]) if c in raw.files else '1' for c in cols])
    return n


def verify(pkg, stored, raw_csv):
    rows = deploy.read_raw_csv(raw_csv)
    out = deploy.release(pkg, rows)
    res = {}
    for t in D.TASKS:
        p = out[t]['probability']; ref = stored[t]
        res[t] = {'max_abs_prob_diff': float(np.max(np.abs(p - ref))), 'decisions_identical': bool(np.array_equal(out[t]['decision'], ref.argmax(1)))}
    return res


def refusals(pkg, tmpdir):
    checks = {}
    bad_order = Path(tmpdir) / 'reordered.csv'; write_raw_csv(bad_order, columns=list(reversed(deploy.PERMITTED)))
    forb = Path(tmpdir) / 'forbidden.csv'; write_raw_csv(forb, columns=deploy.PERMITTED + ['SEX'])
    for label, path in (('reordered_schema', bad_order), ('forbidden_SEX_column', forb)):
        try:
            deploy.read_raw_csv(path); checks[label] = 'NOT_REFUSED'
        except (ValueError, PermissionError) as e:
            checks[label] = f'REFUSED: {type(e).__name__}'
    try:
        deploy.release(pkg, [], requested=('probability', 'leaf_ids')); checks['hidden_output_request'] = 'NOT_REFUSED'
    except PermissionError as e:
        checks['hidden_output_request'] = f'REFUSED: {type(e).__name__}'
    return checks


def main(control_service, control_seed_state):
    tmp = D.PRIV / 'deploy_tmp'; tmp.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw_csv = tmp / 'DEV_REPORT_raw_inputs.csv'; n = write_raw_csv(raw_csv)
    summary = {'schema': 'ACSR1-deployment-v1', 'raw_rows_deployed': n, 'labels_used': False, 'packages': {}}
    for seed in (0, 1):
        outs, _, _ = registry.load(seed, roles=('DEV_REPORT',))
        for name, theta in ((f'T_TREE_S{seed}', None), (f'LOCAL_TREE_S{seed}', 'LOCAL_TREE__E20.npz')):
            pkg = build_tree_package(name, seed, theta)
            svc = 'T_TREE' if theta is None else 'LOCAL_TREE'
            summary['packages'][name] = {'parity': verify(pkg, outs[svc]['DEV_REPORT'], raw_csv), 'refusals': refusals(pkg, tmp),
                                         'package_json_sha256': sha(pkg / 'PACKAGE.json')}
    if control_service:
        seed, state = control_seed_state
        outs, _, _ = registry.load(seed, roles=('DEV_REPORT',))
        pkg = build_neural_package(f'{control_service}_S{seed}', seed, state)
        summary['packages'][f'{control_service}_S{seed}'] = {'parity': verify(pkg, outs[control_service]['DEV_REPORT'], raw_csv),
                                                             'refusals': refusals(pkg, tmp), 'package_json_sha256': sha(pkg / 'PACKAGE.json')}
    summary['all_parity_ok'] = all(v['max_abs_prob_diff'] <= 1e-12 and v['decisions_identical'] for p in summary['packages'].values() for v in p['parity'].values())
    summary['all_refusals_ok'] = all(str(x).startswith('REFUSED') for p in summary['packages'].values() for x in p['refusals'].values())
    (D.RESULTS / 'DEPLOYMENT_RESULTS.json').write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--control', default=''); ap.add_argument('--control-seed', type=int, default=0); ap.add_argument('--control-state', default='')
    a = ap.parse_args()
    s = main(a.control or None, (a.control_seed, a.control_state))
    print(json.dumps({'all_parity_ok': s['all_parity_ok'], 'all_refusals_ok': s['all_refusals_ok'], 'packages': list(s['packages'])}))
