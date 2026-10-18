"""Raw-field deployment of complete releases of the overnight occupation-headroom campaign (six permitted ACS fields -> each
recipient's probability vector and argmax decision; nothing else is exported).

    python -m ohn.deploy build --anchor H3 --occ FAMILY:CONFIG --seed s --out DIR    # FAMILY:CONFIG or SOURCE:JOINT
    python -m ohn.deploy check --packages DIR1 DIR2 ... --report OUT.json            # bitwise parity + refusals (no label read)

A package (private) holds PACKAGE.json: the income anchor (H3 = the source's H3_STRONG_LJDIR_PB income, TW = HC_STRONG income,
both computed by the source's unmodified `ihs.fam_h.predict` from the admitted mirror), the occupation source (a family module's
`predict`, or the inherited JOINT_TREE output for the unmodified anchor package), the frozen 145-column encoder of the replication,
and the SHA-256 of every file the release depends on (anchor state, step-1 learner, teacher and correction files, the occupation
run directory, and the family module source). Entry points, both enforcing the schema:
- `release(pkg, rows, requested)` / `release_csv(pkg, path, requested)`: the documented wrapper. Every row must carry exactly AGEP,
  WKHP, COW, SCHL, MAR, POBP in that order; SEX, task targets (PINCP, OCCP, ...), identifiers and every other field are refused;
  only 'probability' and 'decision' may be requested.
- `underlying(pkg, X, names)`: the deployment function on the encoded features. It refuses any column list other than the
  package encoder's exact 145 names in order (so a missing, extra, reordered, target or SEX column is rejected here too), and
  non-finite or mis-shaped input. Intermediate quantities are never returned.
"""
from __future__ import annotations
import argparse
import csv
import importlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from acs_replication import deploy as SRC_DEPLOY
from acs_replication import schema
from . import common as C
from . import devrun as D

PERMITTED = list(SRC_DEPLOY.PERMITTED)
ALLOWED = {'probability', 'decision'}
FORBIDDEN = set(schema.FORBIDDEN) | {'INCOME', 'OCCUPATION', 'Y', 'LABEL', 'TARGET', 'UNIT', 'ROW_ID', 'HOUSEHOLD'}
FAMILY_MODULE = {'A': 'ohn.fam_a', 'B': 'ohn.fam_b', 'C': 'ohn.fam_c', 'PROT': 'ohn.protect', 'INC': 'ohn.fam_inc'}


def build(anchor, occ, seed, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    enc = json.loads(C.admitted(f'admitted/packages/T_TREE_S{seed}/PACKAGE.json').read_text())['encoder']
    fam_a, cfg_a = D.ANCHORS[anchor]
    deps = {}
    for d in (C.MIRROR / 'runs' / fam_a / cfg_a / f'S{seed}', C.MIRROR / 'runs' / 'H' / 'STEP1' / f'S{seed}'):
        deps.update({str(p.relative_to(C.MIRROR)): C.sha(p) for p in sorted(d.iterdir()) if p.is_file()})
    for rel in (f'admitted/stage_a/S{seed}/TREE_TEACHER_income.json', f'admitted/stage_a/S{seed}/TREE_TEACHER_occupation.json',
                f'admitted/stage_b/S{seed}/JOINT_TREE__E20.npz', f'admitted/stage_b/S{seed}/LOCAL_TREE__E20.npz'):
        deps[rel] = C.sha(C.MIRROR / rel)
    occ_family, occ_config = occ.split(':')
    module = None
    if occ_family != 'SOURCE':
        rd = D.occ_dir(occ_family, occ_config, seed)
        deps.update({f'OHN::{p.relative_to(C.RUNS)}': C.sha(p) for p in sorted(rd.rglob('*')) if p.is_file() and not p.name.startswith('EVAL_')})
        module = FAMILY_MODULE[occ_family]
        src = C.ROOT / (module.replace('.', '/') + '.py')
        deps[f'CODE::{module}'] = C.sha(src)
    elif occ_config != 'JOINT':
        raise ValueError('SOURCE occupation must be JOINT (the inherited output of the anchor package)')
    meta = {'schema': 'OHN1-deploy-package-v1', 'anchor': anchor, 'anchor_config': cfg_a, 'occupation': occ, 'occupation_module': module,
            'seed': seed, 'encoder': enc, 'depends_sha256': deps, 'release_contract': 'probability and decision per recipient; nothing else',
            'permitted_inputs': PERMITTED, 'utc': C.now()}
    (out / 'PACKAGE.json').write_text(json.dumps(meta, indent=1, sort_keys=True) + '\n')
    return meta


def _load(pkg):
    meta = json.loads((Path(pkg) / 'PACKAGE.json').read_text())
    for rel, h in meta['depends_sha256'].items():
        if rel.startswith('OHN::'):
            p = C.RUNS / rel[5:]
        elif rel.startswith('CODE::'):
            p = C.ROOT / (rel[6:].replace('.', '/') + '.py')
        else:
            p = C.MIRROR / rel
        if C.sha(p) != h:
            raise PermissionError(f'package dependency hash differs: {rel}')
    return meta


def check_rows(rows):
    for r in rows:
        keys = list(r.keys())
        bad = [k for k in keys if str(k).upper() in FORBIDDEN]
        if bad:
            raise PermissionError(f'forbidden fields present (never accepted as inputs): {bad}')
        if keys != PERMITTED:
            raise ValueError(f'incompatible schema: expected exactly {PERMITTED} in this order, got {keys}')


def check_encoded(meta, X, names):
    names = tuple(map(str, names))
    bad = [n for n in names if n.split('=')[0].upper() in FORBIDDEN]
    if bad:
        raise PermissionError(f'forbidden columns present in the encoded input: {bad}')
    if names != tuple(meta['encoder']['feature_names']):
        raise ValueError('encoded input columns differ from the package encoder (missing, extra or reordered columns)')
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != len(names) or not np.isfinite(X).all():
        raise ValueError('encoded input has the wrong shape or non-finite values')


def underlying(pkg, X, names):
    meta = _load(pkg)
    check_encoded(meta, X, names)
    from ihs import fam_h
    src = fam_h.predict(meta['seed'], meta['anchor_config'], X, names)
    if meta['occupation'] == 'SOURCE:JOINT':
        occ = src['occupation']
    else:
        fam, cfg = meta['occupation'].split(':')
        o = importlib.import_module(meta['occupation_module']).predict(meta['seed'], cfg, X, names)
        occ = o['occupation'] if isinstance(o, dict) else o
    return {'income': np.asarray(src['income'], dtype=np.float64), 'occupation': np.asarray(occ, dtype=np.float64)}


def release(pkg, rows, requested=('probability', 'decision')):
    if set(requested) - ALLOWED:
        raise PermissionError(f'hidden-output request refused: {sorted(set(requested) - ALLOWED)}')
    check_rows(rows)
    meta = _load(pkg)
    X = SRC_DEPLOY.encode_raw(rows, meta['encoder']); names = tuple(meta['encoder']['feature_names'])
    out = underlying(pkg, X, names)
    return {t: {k: v for k, v in {'probability': out[t], 'decision': out[t].argmax(1).astype(np.int64)}.items() if k in requested} for t in C.TASKS}


def release_csv(pkg, path, requested=('probability', 'decision')):
    with open(path, newline='') as f:
        header = csv.DictReader(f).fieldnames or []
    bad = [h for h in header if h.upper() in FORBIDDEN]
    if bad:
        raise PermissionError(f'forbidden columns present (never accepted as inputs): {bad}')
    return release(pkg, SRC_DEPLOY.read_raw_csv(path), requested)


def stored_dev(meta):
    seed = meta['seed']
    inc = D.anchor_income(meta['anchor'], seed, 'DEV_REPORT')
    if meta['occupation'] == 'SOURCE:JOINT':
        occ = D.source_occupation('JOINT', seed, 'DEV_REPORT')
    else:
        fam, cfg = meta['occupation'].split(':')
        occ = D.load_occupation(fam, cfg, seed, roles=('DEV_REPORT',))[0]['DEV_REPORT']
    return {'income': inc, 'occupation': occ}


def check(packages, report):
    R = C.load_role('DEV_REPORT', ('features', 'raw_inputs'))
    raw = R['raw_inputs']
    rows = [{c: raw[c][i].item() if hasattr(raw[c][i], 'item') else raw[c][i] for c in PERMITTED} for i in range(len(R['row_id']))]
    tmp = Path(tempfile.mkdtemp()); p = tmp / 'raw.csv'
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=PERMITTED); w.writeheader(); w.writerows(rows)
    res = {}
    for pkg in packages:
        t0 = time.process_time()
        meta = _load(pkg)
        ref = stored_dev(meta)
        a = release(pkg, rows); b = release_csv(pkg, p); u = underlying(pkg, R['X'], R['feature_names'])
        same = {name: {t: bool(np.array_equal(o[t]['probability'], ref[t])) for t in C.TASKS}
                for name, o in (('wrapper_api', a), ('wrapper_csv', b), ('underlying', {t: {'probability': u[t]} for t in C.TASKS}))}
        ref_dec = {t: bool(np.array_equal(a[t]['decision'], ref[t].argmax(1))) for t in C.TASKS}
        refusals = {}
        for label, bad in (('wrapper_SEX_field', [{**rows[0], 'SEX': 1}]), ('wrapper_target_field', [{**rows[0], 'OCCP': '0010'}]),
                           ('wrapper_extra_field', [{**rows[0], 'PUMA': 1}]), ('wrapper_reordered_fields', [dict(reversed(list(rows[0].items())))]),
                           ('wrapper_missing_field', [{k: v for k, v in rows[0].items() if k != 'POBP'}])):
            try:
                release(pkg, bad); refusals[label] = 'NOT_REFUSED'
            except (PermissionError, ValueError) as e:
                refusals[label] = f'refused: {type(e).__name__}'
        names = list(R['feature_names']); X3 = R['X'][:3]
        for label, (Xb, nb) in (('underlying_SEX_column', (np.column_stack((X3, np.ones(3))), names + ['SEX'])),
                                ('underlying_target_column', (np.column_stack((X3, np.ones(3))), names + ['OCCP'])),
                                ('underlying_extra_column', (np.column_stack((X3, np.ones(3))), names + ['EXTRA'])),
                                ('underlying_missing_column', (X3[:, 1:], names[1:])),
                                ('underlying_reordered_columns', (X3[:, ::-1], names[::-1]))):
            try:
                underlying(pkg, Xb, nb); refusals[label] = 'NOT_REFUSED'
            except (PermissionError, ValueError) as e:
                refusals[label] = f'refused: {type(e).__name__}'
        try:
            release(pkg, rows[:3], requested=('probability', 'raw_score')); refusals['hidden_output_request'] = 'NOT_REFUSED'
        except PermissionError:
            refusals['hidden_output_request'] = 'refused: PermissionError'
        small = release(pkg, rows[:7])
        res[Path(pkg).name] = {
            'anchor': meta['anchor'], 'occupation': meta['occupation'], 'seed': meta['seed'],
            'bitwise_equal_to_evaluated_DEV_REPORT_release': same, 'decisions_equal': ref_dec, 'refusals': refusals,
            'exported_keys': {t: sorted(a[t]) for t in C.TASKS},
            'small_batch_max_abs_diff': {t: float(np.max(np.abs(small[t]['probability'] - ref[t][:7]))) for t in C.TASKS},
            'cpu_s': round(time.process_time() - t0, 2), 'n_dependencies': len(meta['depends_sha256'])}
    rec = {'schema': 'OHN1-deployment-check-v1', 'utc': C.now(), 'packages': res, 'labels_read': False}
    rec['all_bitwise'] = all(all(all(v.values()) for v in r['bitwise_equal_to_evaluated_DEV_REPORT_release'].values()) for r in res.values())
    rec['all_refused'] = all(v.startswith('refused') for r in res.values() for v in r['refusals'].values())
    C.write_json(report, rec)
    print(json.dumps({'all_bitwise': rec['all_bitwise'], 'all_refused': rec['all_refused']}, indent=1))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build'); b.add_argument('--anchor', required=True, choices=tuple(D.ANCHORS))
    b.add_argument('--occ', required=True); b.add_argument('--seed', type=int, required=True); b.add_argument('--out', required=True)
    c = sub.add_parser('check'); c.add_argument('--packages', nargs='+', required=True); c.add_argument('--report', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'build':
        print(json.dumps(build(a.anchor, a.occ, a.seed, a.out)['depends_sha256'], indent=1))
    else:
        rec = check(a.packages, a.report)
        if not (rec['all_bitwise'] and rec['all_refused']):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
