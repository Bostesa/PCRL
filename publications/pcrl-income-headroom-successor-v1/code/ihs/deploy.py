"""Raw-field deployment of campaign releases (six permitted ACS fields -> each recipient's probability vector and decision).

    python -m ihs.deploy build --family F --config ID --seed s --out DIR     # package directory with PACKAGE.json
    python -m ihs.deploy check --packages DIR1 DIR2 ... --report OUT.json   # bitwise parity + refusals (no label read)

Package (private): PACKAGE.json with the family module, configuration, seed, the frozen 145-column encoder of the replication
(from the admitted T_TREE package), and the SHA-256 of every file the release depends on (the run directory's parameter files and
the admitted teacher / correction files). Entry points:
- `release(pkg, rows, requested)`: the wrapper API. Every row must have exactly AGEP, WKHP, COW, SCHL, MAR, POBP in that order;
  forbidden fields (SEX, labels, ids) are refused; only 'probability' and 'decision' may be requested.
- `release_csv(pkg, path)`: the documented CSV wrapper (same checks on the header).
- `underlying(pkg, X, names)`: the family's own predict on the encoded features (no schema check: the wrapper enforces it).
`check` reproduces each package's stored DEV_REPORT release bitwise through all three paths from the DEV_REPORT raw inputs
(label-free) and records the refusal tests.
"""
from __future__ import annotations
import argparse
import csv
import importlib
import json
import tempfile
from pathlib import Path

import numpy as np

from acs_replication import deploy as SRC_DEPLOY
from acs_replication import schema
from . import common as C
from . import devrun as D

PERMITTED = list(SRC_DEPLOY.PERMITTED)
ALLOWED = {'probability', 'decision'}
MODULE = {'P': 'ihs.fam_p', 'O': 'ihs.fam_o', 'H': 'ihs.fam_h', 'CMB': 'ihs.combine'}


def build(family, config_id, seed, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    enc = json.loads(C.admitted(f'packages/T_TREE_S{seed}/PACKAGE.json').read_text())['encoder']
    run_dir = D.release_dir(family, config_id, seed)
    deps = {f'runs/{family}/{config_id}/S{seed}/{p.name}': C.sha(p) for p in sorted(run_dir.iterdir()) if p.is_file()}
    if family == 'H':                                         # family H's STRONG learner (pickled HGB and its step-1 record)
        step1 = C.RUNS / 'H' / 'STEP1' / f'S{seed}'
        deps.update({f'runs/H/STEP1/S{seed}/{p.name}': C.sha(p) for p in sorted(step1.iterdir()) if p.is_file()})
    deps.update({rel: C.sha(C.ADM / rel) for rel in (f'stage_a/S{seed}/TREE_TEACHER_income.json', f'stage_a/S{seed}/TREE_TEACHER_occupation.json',
                                                   f'stage_b/S{seed}/JOINT_TREE__E20.npz', f'stage_b/S{seed}/LOCAL_TREE__E20.npz')})
    meta = {'schema': 'IHS1-deploy-package-v1', 'family': family, 'config_id': config_id, 'seed': seed, 'module': MODULE[family],
            'encoder': enc, 'depends_sha256': deps, 'release_contract': 'probability and decision per recipient; nothing else',
            'permitted_inputs': PERMITTED}
    (out / 'PACKAGE.json').write_text(json.dumps(meta, indent=1, sort_keys=True) + '\n')
    return meta


def _load(pkg):
    meta = json.loads((Path(pkg) / 'PACKAGE.json').read_text())
    for rel, h in meta['depends_sha256'].items():
        p = (C.PRIV / rel) if rel.startswith('runs/') else (C.ADM / rel)
        if C.sha(p) != h:
            raise PermissionError(f'package dependency hash differs: {rel}')
    return meta


def check_rows(rows):
    for r in rows:
        keys = list(r.keys())
        bad = [k for k in keys if str(k).upper() in schema.FORBIDDEN]
        if bad:
            raise PermissionError(f'forbidden fields present (never accepted as inputs): {bad}')
        if keys != PERMITTED:
            raise ValueError(f'incompatible schema: expected exactly {PERMITTED} in this order, got {keys}')


def underlying(pkg, X, names):
    meta = _load(pkg)
    return importlib.import_module(meta['module']).predict(meta['seed'], meta['config_id'], X, names)


def release(pkg, rows, requested=('probability', 'decision')):
    if set(requested) - ALLOWED:
        raise PermissionError(f'hidden-output request refused: {sorted(set(requested) - ALLOWED)}')
    check_rows(rows)
    meta = _load(pkg)
    X = SRC_DEPLOY.encode_raw(rows, meta['encoder']); names = tuple(meta['encoder']['feature_names'])
    out = underlying(pkg, X, names)
    return {t: {k: v for k, v in {'probability': out[t], 'decision': out[t].argmax(1).astype(np.int64)}.items() if k in requested} for t in C.TASKS}


def release_csv(pkg, path, requested=('probability', 'decision')):
    return release(pkg, SRC_DEPLOY.read_raw_csv(path), requested)


def dev_rows():
    R = C.load_role('DEV_REPORT', ('features', 'raw_inputs'))
    raw = R['raw_inputs']
    return [{c: raw[c][i].item() if hasattr(raw[c][i], 'item') else raw[c][i] for c in PERMITTED} for i in range(len(R['row_id']))], R


def check(packages, report):
    rows, R = dev_rows()
    tmp = Path(tempfile.mkdtemp()); p = tmp / 'raw.csv'
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=PERMITTED); w.writeheader(); w.writerows(rows)
    res = {}
    for pkg in packages:
        meta = _load(pkg)
        stored, _ = D.load_release(meta['family'], meta['config_id'], meta['seed'], roles=('DEV_REPORT',))
        ref = stored['DEV_REPORT']
        a = release(pkg, rows); b = release_csv(pkg, p); u = underlying(pkg, R['X'], R['feature_names'])
        same = {name: {t: bool(np.array_equal(o[t]['probability'] if 'probability' in o[t] else o[t], ref[t])) for t in C.TASKS}
                for name, o in (('wrapper_api', a), ('wrapper_csv', b), ('underlying', {t: {'probability': u[t]} for t in C.TASKS}))}
        refusals = {}
        for label, bad in (('forbidden_SEX_field', [{**rows[0], 'SEX': 1}]), ('extra_field', [{**rows[0], 'PUMA': 1}]),
                           ('reordered_fields', [dict(reversed(list(rows[0].items())))]), ('missing_field', [{k: v for k, v in rows[0].items() if k != 'POBP'}])):
            try:
                release(pkg, bad); refusals[label] = 'NOT_REFUSED'
            except (PermissionError, ValueError) as e:
                refusals[label] = f'refused: {type(e).__name__}'
        try:
            release(pkg, rows[:3], requested=('probability', 'raw_score')); refusals['hidden_output_request'] = 'NOT_REFUSED'
        except PermissionError:
            refusals['hidden_output_request'] = 'refused: PermissionError'
        small = release(pkg, rows[:7])
        res[f"{meta['family']}:{meta['config_id']}:S{meta['seed']}"] = {
            'bitwise_equal_to_evaluated_DEV_REPORT_release': same, 'refusals': refusals,
            'small_batch_last_bit_max_abs_diff': {t: float(np.max(np.abs(small[t]['probability'] - ref[t][:7]))) for t in C.TASKS},
            'decisions_equal_small_batch': {t: bool(np.array_equal(small[t]['decision'], ref[t][:7].argmax(1))) for t in C.TASKS}}
    rec = {'schema': 'IHS1-deployment-check-v1', 'utc': C.now(), 'packages': res, 'labels_read': False,
           'note': 'schema checks are enforced by the wrapper (release / release_csv); the underlying family predict accepts encoded features only'}
    rec['all_bitwise'] = all(all(all(v.values()) for v in r['bitwise_equal_to_evaluated_DEV_REPORT_release'].values()) for r in res.values())
    C.write_json(report, rec)
    print(json.dumps({'all_bitwise': rec['all_bitwise'], 'refusals': {k: v['refusals'] for k, v in res.items()}}, indent=1))
    return rec


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build'); b.add_argument('--family', required=True); b.add_argument('--config', required=True)
    b.add_argument('--seed', type=int, required=True); b.add_argument('--out', required=True)
    c = sub.add_parser('check'); c.add_argument('--packages', nargs='+', required=True); c.add_argument('--report', required=True)
    a = ap.parse_args()
    if a.cmd == 'build':
        print(json.dumps(build(a.family, a.config, a.seed, a.out)['depends_sha256'], indent=1))
    else:
        check(a.packages, a.report)
