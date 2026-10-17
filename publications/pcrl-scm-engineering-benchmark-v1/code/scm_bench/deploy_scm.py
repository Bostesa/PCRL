"""Deployment packages for the SCM engineering benchmark: six raw permitted fields -> each recipient's probabilities + decision.

Package (private): PACKAGE.json (kind SCM, the frozen 145-column encoder admitted from the source T_TREE_S0 package whose
PACKAGE.json hash is in the public MODEL_MANIFEST, file hashes), the ten fold-teacher JSONs, CELLS.npz (fine-cell centroids,
cell-of-initial and group-of-initial maps, the stored v1 / v2 tables). Release: encode -> fold-average teacher probabilities
-> nearest fine cell by KL -> final cell and income group -> stored vectors (copied bitwise) -> argmax decisions.
Refusals: any schema other than exactly AGEP, WKHP, COW, SCHL, MAR, POBP in that order; any forbidden column (SEX among them);
any requested output other than probability and decision (cell ids, group ids, teacher probabilities, propensity).
`build` makes packages for the final-bank SCM, TWIN and the seed-0 neural reference copy; `verify` reproduces the attacked
DEV_REPORT outputs from DEV_REPORT raw inputs (no labels) and checks the refusals.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from acs_replication import deploy as SRC_DEPLOY
from acs_replication import trees
from . import common as C
from . import engine as E

ALLOWED = {'probability', 'decision'}


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def source_encoder():
    man = json.loads((C.SRC_RESULTS / 'MODEL_MANIFEST.json').read_text())
    p = C.SRC_PRIV / 'deploy' / 'T_TREE_S0' / 'PACKAGE.json'
    if sha(p) != man['deployment_packages']['T_TREE_S0']['package_json_sha256']:
        raise PermissionError('source package hash differs from the public MODEL_MANIFEST')
    return json.loads(p.read_text())['encoder']


def build(seed, arm, label, out_root):
    fit = C.PRIV / 'runs' / f'FIT_{seed}'
    d = Path(out_root) / f'{label}_S{seed}'
    if d.exists():
        raise PermissionError('package exists')
    d.mkdir(parents=True, mode=0o700)
    meta = {'schema': 'SCM1-deploy-package-v1', 'kind': 'SCM', 'service': label, 'arm': arm, 'seed': seed, 'encoder': source_encoder(),
            'teachers': {}, 'files': {}}
    for f in range(5):
        for t in C.TASKS:
            src = fit / 'TEACHERS' / f'fold{f}_{t}.json'; shutil.copy2(src, d / src.name)
            meta['teachers'].setdefault(t, []).append(src.name); meta['files'][src.name] = sha(d / src.name)
    with np.load(fit / 'PARTITION.npz') as z:
        cent1, cent2 = z['cent1'], z['cent2']
    with np.load(fit / 'ARM_TABLES.npz') as z:
        tab = {k: z[f'{arm}__{k}'] for k in ('v1', 'v2', 'cell_of_initial', 'group_of_initial')}
    np.savez(d / 'CELLS.npz', cent1=cent1, cent2=cent2, **tab); meta['cells'] = 'CELLS.npz'; meta['files']['CELLS.npz'] = sha(d / 'CELLS.npz')
    (d / 'PACKAGE.json').write_text(json.dumps(meta, indent=2, sort_keys=True))
    for p in d.iterdir():
        p.chmod(0o600)
    return d


def release(pkg_dir, rows, requested=('probability', 'decision')):
    if set(requested) - ALLOWED:
        raise PermissionError(f'hidden-output request refused: {sorted(set(requested) - ALLOWED)}')
    pkg_dir = Path(pkg_dir); meta = json.loads((pkg_dir / 'PACKAGE.json').read_text())
    for name, h in meta['files'].items():
        if sha(pkg_dir / name) != h:
            raise PermissionError(f'package file hash differs: {name}')
    X = SRC_DEPLOY.encode_raw(rows, meta['encoder']); names = tuple(meta['encoder']['feature_names'])
    t = {}
    for task in C.TASKS:
        acc = 0.0
        for fn in meta['teachers'][task]:
            tt = trees.TreeTeacher(json.loads((pkg_dir / fn).read_text()))
            acc = acc + tt.probabilities_from_raw(tt.raw_from_leaves(tt.route(X, names))) / 5.0
        t[task] = acc
    with np.load(pkg_dir / meta['cells']) as z:
        init = E.kl_assign(t['income'], t['occupation'], z['cent1'], z['cent2'])
        tab = {k: z[k] for k in ('v1', 'v2', 'cell_of_initial', 'group_of_initial')}
    rel = E.release(tab, init)
    out = {task: {'probability': rel[task], 'decision': rel[task].argmax(1).astype(np.int64)} for task in C.TASKS}
    return {task: {k: v for k, v in r.items() if k in requested} for task, r in out.items()}


def raw_rows(role):
    R = C.load_role(role, ('features', 'raw_inputs'))['raw_inputs']
    cols = list(SRC_DEPLOY.PERMITTED)
    n = len(R[cols[0]])
    return [{c: R[c][i] for c in cols} for i in range(n)]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out-root', required=True); ap.add_argument('--report', required=True)
    a = ap.parse_args()
    if not C.committed(C.RESULTS / 'REPORTING_LOCK.json'):
        raise PermissionError('packaging follows the committed REPORTING_LOCK')
    sel = json.loads((C.RESULTS / 'SELECTION_LOCK.json').read_text())
    rows = raw_rows('DEV_REPORT'); rec = {'schema': 'SCM1-deployment-v1', 'utc': C.now(), 'packages': {}}
    for s in C.SEEDS:
        with np.load(C.PRIV / 'runs' / f'FIT_{s}' / 'ARM_OUTPUTS.npz') as z:
            for label in ('SCM', 'TWIN'):
                arm = sel['bank_arm_map'][label]
                d = build(s, arm, label, a.out_root)
                res = release(d, rows)
                ok = {t: bool(np.array_equal(res[t]['probability'], z[f'{arm}__DEV_REPORT__{t}'])
                              and np.array_equal(res[t]['decision'], z[f'{arm}__DEV_REPORT__{t}'].argmax(1))) for t in C.TASKS}
                refusals = {}
                try:
                    release(d, rows[:3], ('probability', 'cell_id')); refusals['hidden_output'] = False
                except PermissionError:
                    refusals['hidden_output'] = True
                rec['packages'][f'{label}_S{s}'] = {'arm': arm, 'package_json_sha256': sha(d / 'PACKAGE.json'), 'reproduces_attacked_DEV_REPORT_wire_bitwise': ok,
                                                     'refusals': refusals}
    # schema refusals through the shared raw-CSV reader (forbidden column, reordered columns)
    import csv, tempfile
    tmp = Path(tempfile.mkdtemp())
    with open(tmp / 'bad_sex.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(list(SRC_DEPLOY.PERMITTED) + ['SEX']); w.writerow(['40', '40', '1', '21', '1', '012', '1'])
    with open(tmp / 'bad_order.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(list(reversed(SRC_DEPLOY.PERMITTED))); w.writerow(['012', '1', '21', '1', '40', '40'])
    for name, exc in (('bad_sex.csv', PermissionError), ('bad_order.csv', ValueError)):
        try:
            SRC_DEPLOY.read_raw_csv(tmp / name); rec[f'refuses_{name}'] = False
        except exc:
            rec[f'refuses_{name}'] = True
    shutil.rmtree(tmp)
    Path(a.report).write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(json.dumps(rec, indent=1))
