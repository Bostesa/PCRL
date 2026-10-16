"""Deploy the fixed H_LJ composition from RAW permitted ACS fields (no labels, no other service's output).

Package (private): PACKAGE.json; the seed's two frozen teacher JSONs; the two ORIGINAL frozen checkpoint files, unchanged
(LOCAL_TREE__E20.npz supplies theta_income for the income recipient, JOINT_TREE__E20.npz supplies theta_occupation for the
occupation recipient; their sha256 equal the source MODEL_MANIFEST). Input: a CSV whose header is EXACTLY the permitted raw
fields AGEP, WKHP, COW, SCHL, MAR, POBP in that order (anything else is refused). Output per person: each recipient's own
probability vector and argmax decision. Each recipient is computed by `release_task`, which reads only that task's own
component (encoder, teacher, theta); it never reads the other recipient's component or output. Hidden-output requests (leaf
ids, raw scores, teacher outputs, representations, the other recipient's output) are refused.
"""
from __future__ import annotations
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from acs_replication import deploy as base_deploy, trees

from . import common as C

ALLOWED_OUTPUTS = {'probability', 'decision'}
PERMITTED = list(base_deploy.PERMITTED)


def build(seed, name=None, dest_root=None):
    name = name or f'H_LJ_S{seed}'
    d = (dest_root or (C.PRIV / 'deploy')) / name
    if d.exists():
        raise PermissionError('package exists')
    d.mkdir(parents=True, mode=0o700)
    mm = json.loads((C.SRC_RESULTS / 'MODEL_MANIFEST.json').read_text())['models']
    comps = {}
    for task, svc in zip(C.TASKS, C.COMPONENTS[C.PRIMARY]):
        teacher = C.SRCP / f'runs/STAGE_A_{seed}/TREE_TEACHER_{task}.json'
        theta = C.SRCP / f'runs/STAGE_B_{seed}/{svc}__E20.npz'
        shutil.copy2(teacher, d / teacher.name)
        if not (d / theta.name).exists():
            shutil.copy2(theta, d / theta.name)
        comps[task] = {'source_service': svc, 'teacher': teacher.name, 'theta_file': theta.name, 'theta_key': f'theta_{task}',
                       'teacher_sha256_equals_model_manifest': C.sha(d / teacher.name) == mm[f'T_TREE_{task}_S{seed}']['sha256'],
                       'theta_sha256_equals_model_manifest': C.sha(d / theta.name) == mm[f'{svc}_E20_S{seed}']['sha256']}
    meta = {'schema': 'ACSH1-hybrid-package-v1', 'kind': 'TREE_HYBRID', 'service': C.PRIMARY, 'seed': seed, 'encoder': C.receipt()['encoding'],
            'components': comps, 'permitted_inputs': PERMITTED, 'allowed_outputs': sorted(ALLOWED_OUTPUTS),
            'files': {f.name: C.sha(f) for f in sorted(d.iterdir())}}
    (d / 'PACKAGE.json').write_text(json.dumps(meta, indent=2, sort_keys=True))
    return d


def load_package(pkg_dir, tasks=C.TASKS):
    pkg_dir = Path(pkg_dir); meta = json.loads((pkg_dir / 'PACKAGE.json').read_text())
    if meta.get('kind') != 'TREE_HYBRID':
        raise ValueError('unregistered package kind')
    needed = {meta['components'][t][k] for t in tasks for k in ('teacher', 'theta_file')}
    for name in needed:   # verify only the files this request needs
        if C.sha(pkg_dir / name) != meta['files'][name]:
            raise PermissionError(f'package file hash differs: {name}')
    return meta


def release_task(pkg_dir, X, names, task):
    """One recipient's inference: encoded inputs -> that recipient's own frozen component only."""
    pkg_dir = Path(pkg_dir); meta = load_package(pkg_dir, tasks=(task,)); comp = meta['components'][task]
    teacher = trees.TreeTeacher(json.loads((pkg_dir / comp['teacher']).read_text()))
    with np.load(pkg_dir / comp['theta_file']) as z:
        theta = z[comp['theta_key']].copy()
    return trees.release(task, teacher.raw_from_leaves(teacher.route(X, names), theta))


def release(pkg_dir, rows, requested=('probability', 'decision'), tasks=C.TASKS):
    if set(requested) - ALLOWED_OUTPUTS:
        raise PermissionError(f'hidden-output request refused: {sorted(set(requested) - ALLOWED_OUTPUTS)}')
    if set(tasks) - set(C.TASKS):
        raise PermissionError(f'unknown recipient requested: {sorted(set(tasks) - set(C.TASKS))}')
    meta = load_package(pkg_dir, tasks=tasks)
    X = base_deploy.encode_raw(rows, meta['encoder']); names = tuple(meta['encoder']['feature_names'])
    out = {t: release_task(pkg_dir, X, names, t) for t in tasks}
    return {t: {k: v for k, v in r.items() if k in requested} for t, r in out.items()}


def write_raw_csv(path, columns=None):
    raw = C.load_role('DEV_REPORT', ('features', 'raw_inputs'))['raw_inputs']
    cols = columns or PERMITTED
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(cols)
        for i in range(len(raw['row_id'])):
            w.writerow([str(raw[c][i]) if c in raw else '1' for c in cols])
    return len(raw['row_id'])


def verify(seed, pkg, tmp):
    bank, _ = C.load_bank(seed, roles=('DEV_REPORT',))
    raw_csv = tmp / 'DEV_REPORT_raw_inputs.csv'
    n = write_raw_csv(raw_csv)
    rows = base_deploy.read_raw_csv(raw_csv)
    out = release(pkg, rows)
    parity = {}
    for t in C.TASKS:
        p = out[t]['probability']; ref = bank[C.PRIMARY]['DEV_REPORT'][t]
        parity[t] = {'max_abs_prob_diff': float(np.max(np.abs(p - ref))), 'bitwise': bool(np.array_equal(p, ref)),
                     'decisions_identical': bool(np.array_equal(out[t]['decision'], ref.argmax(1)))}
    # Each recipient runs without the other recipient's component or output: split packages holding one component only.
    iso = {}
    meta = json.loads((Path(pkg) / 'PACKAGE.json').read_text())
    for t in C.TASKS:
        d = tmp / f'isolated_{t}_S{seed}'
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(mode=0o700)
        comp = meta['components'][t]
        for f in (comp['teacher'], comp['theta_file']):
            shutil.copy2(Path(pkg) / f, d / f)
        (d / 'PACKAGE.json').write_text(json.dumps({**meta, 'components': {t: comp}, 'files': {f: meta['files'][f] for f in (comp['teacher'], comp['theta_file'])}}))
        p = release(d, rows, tasks=(t,))[t]['probability']
        iso[t] = {'files_present': sorted(x.name for x in d.iterdir()), 'equals_full_package_output': bool(np.array_equal(p, out[t]['probability']))}
    # The unused halves of the two checkpoints do not affect the release (JOINT theta_income, LOCAL theta_occupation).
    unused = {}
    for t, other_file in (('income', meta['components']['occupation']['theta_file']), ('occupation', meta['components']['income']['theta_file'])):
        unused[t] = 'own component file differs from the other recipient\'s file' if meta['components'][t]['theta_file'] != other_file else 'shared file'
    refusals = {}
    bad_order = tmp / 'reordered.csv'; write_raw_csv(bad_order, columns=list(reversed(PERMITTED)))
    forb = tmp / 'forbidden.csv'; write_raw_csv(forb, columns=PERMITTED + ['SEX'])
    extra = tmp / 'extra.csv'; write_raw_csv(extra, columns=PERMITTED + ['PUMA'])
    for label, path in (('reordered_schema', bad_order), ('forbidden_SEX_column', forb), ('extra_forbidden_PUMA_column', extra)):
        try:
            base_deploy.read_raw_csv(path); refusals[label] = 'NOT_REFUSED'
        except (ValueError, PermissionError) as e:
            refusals[label] = f'REFUSED: {type(e).__name__}'
    for label, req in (('leaf_ids', ('probability', 'leaf_ids')), ('raw_scores', ('raw_score',)), ('teacher_output', ('teacher_output',)),
                       ('other_recipient_output', ('other_recipient_output',)), ('representation', ('representation',))):
        try:
            release(pkg, rows[:5], requested=req); refusals[f'hidden_output_{label}'] = 'NOT_REFUSED'
        except PermissionError as e:
            refusals[f'hidden_output_{label}'] = f'REFUSED: {type(e).__name__}'
    try:
        release(pkg, rows[:5], tasks=('sex',)); refusals['unknown_recipient'] = 'NOT_REFUSED'
    except PermissionError as e:
        refusals['unknown_recipient'] = f'REFUSED: {type(e).__name__}'
    return {'raw_rows_deployed': n, 'parity': parity, 'recipient_isolation': iso, 'unused_checkpoint_halves': unused, 'refusals': refusals,
            'labels_used': False, 'package_json_sha256': C.sha(Path(pkg) / 'PACKAGE.json'), 'components': meta['components']}


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build-verify'); b.add_argument('--out', required=True)
    r = sub.add_parser('release'); r.add_argument('--package', required=True); r.add_argument('--input', required=True); r.add_argument('--output', required=True)
    r.add_argument('--request', default='probability,decision'); r.add_argument('--tasks', default='income,occupation')
    a = ap.parse_args()
    if a.cmd == 'release':
        rows = base_deploy.read_raw_csv(a.input)
        res = release(a.package, rows, tuple(a.request.split(',')), tuple(a.tasks.split(',')))
        arrays = {f'{t}_{k}': v for t, rr in res.items() for k, v in rr.items()}
        with open(a.output, 'xb') as f:
            np.savez(f, **arrays)
        print(json.dumps({'rows': len(rows), 'fields': sorted(arrays)}))
        return
    tmp = C.PRIV / 'deploy_tmp'; tmp.mkdir(parents=True, exist_ok=True, mode=0o700)
    summary = {'schema': 'ACSH1-deployment-v1', 'utc': C.now(), 'service': C.PRIMARY, 'packages': {}}
    for s in C.SEEDS:
        pkg = build(s)
        summary['packages'][f'H_LJ_S{s}'] = verify(s, pkg, tmp)
    P = summary['packages'].values()
    summary['all_parity_ok'] = all(v['max_abs_prob_diff'] <= 1e-12 and v['decisions_identical'] for p in P for v in p['parity'].values())
    summary['all_isolation_ok'] = all(v['equals_full_package_output'] for p in P for v in p['recipient_isolation'].values())
    summary['all_refusals_ok'] = all(str(x).startswith('REFUSED') for p in P for x in p['refusals'].values())
    summary['all_components_match_model_manifest'] = all(c['teacher_sha256_equals_model_manifest'] and c['theta_sha256_equals_model_manifest']
                                                         for p in P for c in p['components'].values())
    summary['status'] = 'RUNNABLE_PRIMARY_DEPLOYED' if all(summary[k] for k in ('all_parity_ok', 'all_isolation_ok', 'all_refusals_ok', 'all_components_match_model_manifest')) else 'DEPLOYMENT_CHECK_FAILED'
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'DEPLOYMENT_RESULTS.json').write_text(json.dumps(summary, indent=2, sort_keys=True))
    man = {'schema': 'ACSH1-deployment-manifest-v1', 'packages': {k: {'package_json_sha256': v['package_json_sha256'], 'components': v['components']}
                                                                  for k, v in summary['packages'].items()}}
    (out / 'DEPLOYMENT_MANIFEST.json').write_text(json.dumps(man, indent=2, sort_keys=True))
    print(json.dumps({k: summary[k] for k in ('status', 'all_parity_ok', 'all_isolation_ok', 'all_refusals_ok', 'all_components_match_model_manifest')}))


if __name__ == '__main__':
    main()
