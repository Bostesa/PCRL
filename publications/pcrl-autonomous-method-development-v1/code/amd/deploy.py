"""Raw-field deployment packages (six permitted ACS fields -> each recipient's probability vector and argmax decision).

Package kinds:
- TREE (family A releases and the teacher): the replication's frozen 145-column encoder and CatBoost teacher payloads (admitted by
  hash) plus the release's leaf-score corrections THETA.npz; released by the replication's own deployment code
  (acs_replication.deploy.release) behind this campaign's strict input check.
- Further kinds registered by family modules (`amd.fam_b.package_release`, `amd.fam_c.package_release`) when they exist.
- Frozen references NEURAL_RAW (seed 0; the source never built seed 1) and H_LJ (both seeds): their own source packages,
  hash-checked, run through their source deployment code behind the same strict input check.

Entry points: `release(pkg, rows)` (API: every row has exactly AGEP, WKHP, COW, SCHL, MAR, POBP in that order and no forbidden
field) and `release_csv(pkg, path)`. Requests for anything but probability and decision are refused. Every package must
reproduce its evaluated DEV_REPORT probabilities and decisions bitwise through both entry points. No label is read.
"""
from __future__ import annotations
import argparse
import csv
import importlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from acs_replication import deploy as SRC_DEPLOY
from acs_replication import schema
from . import common as C
from . import devrun as DR

PERMITTED = list(SRC_DEPLOY.PERMITTED)
ALLOWED = {'probability', 'decision'}
HIDDEN = ('leaf_ids', 'raw_score', 'teacher_probability', 'propensity', 'sex_specific_probability', 'theta')


def check_rows(rows):
    for r in rows:
        keys = list(r.keys())
        bad = [k for k in keys if str(k).upper() in schema.FORBIDDEN]
        if bad:
            raise PermissionError(f'forbidden fields present (never accepted as inputs): {bad}')
        if keys != PERMITTED:
            raise ValueError(f'incompatible schema: expected exactly {PERMITTED} in this order, got {keys}')


def _kind_release(pkg_dir, rows, requested):
    meta = json.loads((Path(pkg_dir) / 'PACKAGE.json').read_text())
    if meta['kind'] == 'TREE':
        return SRC_DEPLOY.release(pkg_dir, rows, requested)
    mod = importlib.import_module(meta['release_module'])            # family-registered kinds
    return mod.package_release(pkg_dir, rows, requested)


def release(pkg_dir, rows, requested=('probability', 'decision')):
    if set(requested) - ALLOWED:
        raise PermissionError(f'hidden-output request refused: {sorted(set(requested) - ALLOWED)}')
    check_rows(rows)
    return _kind_release(pkg_dir, rows, requested)


def release_csv(pkg_dir, path, requested=('probability', 'decision')):
    return release(pkg_dir, SRC_DEPLOY.read_raw_csv(path), requested)


def build_tree(name, seed, family, config_id, out_root):
    d = Path(out_root) / f'{name}_S{seed}'
    if d.exists():
        raise PermissionError('package exists')
    d.mkdir(parents=True, mode=0o700)
    enc = json.loads(C.admitted(f'packages/T_TREE_S{seed}/PACKAGE.json').read_text())['encoder']
    meta = {'schema': 'ACSR1-deploy-package-v1', 'kind': 'TREE', 'service': name, 'seed': seed, 'encoder': enc, 'teachers': {}, 'files': {},
            'campaign': C.CAMPAIGN, 'release': f'{family}/{config_id}'}
    for t in C.TASKS:
        src = C.admitted(f'stage_a/S{seed}/TREE_TEACHER_{t}.json'); shutil.copy2(src, d / src.name)
        meta['teachers'][t] = src.name; meta['files'][src.name] = C.sha(d / src.name)
    th = DR.release_dir(family, config_id, seed) / 'THETA.npz'
    if th.is_file():
        shutil.copy2(th, d / 'THETA.npz'); meta['theta'] = 'THETA.npz'; meta['files']['THETA.npz'] = C.sha(d / 'THETA.npz')
    (d / 'PACKAGE.json').write_text(json.dumps(meta, indent=2, sort_keys=True))
    for p in d.iterdir():
        p.chmod(0o600)
    return d


def raw_rows():
    R = C.load_role('DEV_REPORT', ('features', 'raw_inputs'))['raw_inputs']
    n = len(R[PERMITTED[0]])
    return [{c: R[c][i] for c in PERMITTED} for i in range(n)]


def write_csv(path, rows, header):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows:
            w.writerow([str(r.get(c, '1')) for c in header])


def refusals(fn_rows, fn_csv, rows, tmp):
    """Every case must raise: extra / reordered / missing / SEX columns (CSV), SEX / extra / reordered keys (API), hidden outputs."""
    out = {}
    cases_csv = {'extra_column': PERMITTED + ['PUMA'], 'reordered_columns': list(reversed(PERMITTED)), 'sex_column': PERMITTED + ['SEX'],
                 'sex_replacing_a_field': PERMITTED[:-1] + ['SEX'], 'missing_column': PERMITTED[:-1]}
    for name, header in cases_csv.items():
        p = tmp / f'{name}.csv'; write_csv(p, [dict(r, SEX='1', PUMA='00101') for r in rows[:3]], header)
        try:
            fn_csv(p); out[f'csv_{name}'] = False
        except (PermissionError, ValueError, KeyError):
            out[f'csv_{name}'] = True
    api = {'sex_key': [dict(r, SEX=1) for r in rows[:3]], 'extra_key': [dict(r, PUMA=101) for r in rows[:3]],
           'reordered_keys': [{c: r[c] for c in reversed(PERMITTED)} for r in rows[:3]]}
    for name, rr in api.items():
        try:
            fn_rows(rr); out[f'api_{name}'] = False
        except (PermissionError, ValueError, KeyError):
            out[f'api_{name}'] = True
    for h in HIDDEN:
        try:
            fn_rows(rows[:3], ('probability', h)); out[f'hidden_{h}'] = False
        except PermissionError:
            out[f'hidden_{h}'] = True
    return out


def same(res, ref):
    return {t: bool(np.array_equal(res[t]['probability'], ref[t]) and np.array_equal(res[t]['decision'], ref[t].argmax(1))) for t in C.TASKS}


def main(out_root, report, packages):
    """packages: list of 'NAME:FAMILY:CONFIG:KIND' (KIND tree | family) for this campaign's releases."""
    rows = raw_rows(); tmp = Path(tempfile.mkdtemp(dir=C.PRIV))
    csv_path = tmp / 'dev_report_raw.csv'; write_csv(csv_path, rows, PERMITTED)
    rec = {'schema': 'AMD1-deployment-v1', 'utc': C.now(), 'inputs': PERMITTED, 'labels_read': False, 'packages': {}, 'unavailable': {}}
    for spec in packages:
        name, family, config_id, kind = spec.split(':')
        for s in C.SEEDS:
            outs, relrec = DR.load_release(family, config_id, s, roles=('DEV_REPORT',))
            if kind == 'tree':
                d = build_tree(name, s, family, config_id, out_root)
            else:
                mod = importlib.import_module(f'amd.fam_{family.lower()}')
                d = mod.build_package(name, s, config_id, out_root)
            ref = outs['DEV_REPORT']
            rec['packages'][f'{name}_S{s}'] = {'release': f'{family}/{config_id}', 'release_sha256': relrec['release_sha256'],
                                              'package_json_sha256': C.sha(d / 'PACKAGE.json'),
                                              'api_reproduces_evaluated_DEV_REPORT_wire_bitwise': same(release(d, rows), ref),
                                              'csv_reproduces_evaluated_DEV_REPORT_wire_bitwise': same(release_csv(d, csv_path), ref),
                                              'refusals': refusals(lambda rr, req=('probability', 'decision'), d=d: release(d, rr, req),
                                                                   lambda p, d=d: release_csv(d, p), rows, tmp)}
    man_src = json.loads((C.REP_RESULTS / 'MODEL_MANIFEST.json').read_text())
    man_hyb = json.loads((C.HYB_RESULTS / 'DEPLOYMENT_MANIFEST.json').read_text())
    from acs_hybrids import hybrid_deploy as HYB
    from scm_bench import common as SBC
    frozen = [('NEURAL_RAW', 0, SBC.SRC_PRIV / 'deploy' / 'NEURAL_RAW_rho1.0_E2_S0', man_src['deployment_packages']['NEURAL_RAW_rho1.0_E2_S0']['package_json_sha256'], SRC_DEPLOY.release)]
    frozen += [('H_LJ', s, SBC.HYB_PRIV / 'deploy' / f'H_LJ_S{s}', man_hyb['packages'][f'H_LJ_S{s}']['package_json_sha256'], HYB.release) for s in C.SEEDS]
    rec['unavailable']['NEURAL_RAW_S1'] = 'the source study never built a seed-1 neural package (coverage gap)'
    for name, s, src, h, fn in frozen:
        if C.sha(src / 'PACKAGE.json') != h:
            raise PermissionError(f'{name} S{s} package hash differs from its public manifest')
        d = Path(out_root) / f'{name}_S{s}'
        if d.exists():
            raise PermissionError('package exists')
        shutil.copytree(src, d)
        bank, _ = C.frozen_bank(s, roles=('DEV_REPORT',))
        ref = {t: bank[name]['DEV_REPORT'][t] for t in C.TASKS}

        def api_fn(rr, req=('probability', 'decision'), d=d, fn=fn):
            if set(req) - ALLOWED:
                raise PermissionError('hidden-output request refused')
            check_rows(rr)
            return fn(d, rr, req)
        rec['packages'][f'{name}_S{s}'] = {'source_package_json_sha256': h, 'api_reproduces_evaluated_DEV_REPORT_wire_bitwise': same(api_fn(rows), ref),
                                           'csv_reproduces_evaluated_DEV_REPORT_wire_bitwise': same(api_fn(SRC_DEPLOY.read_raw_csv(csv_path)), ref),
                                           'refusals': refusals(api_fn, lambda p, api_fn=api_fn: api_fn(SRC_DEPLOY.read_raw_csv(p)), rows, tmp),
                                           'note': 'frozen reference: its own source package, run through the source deployment code behind the same strict input check'}
    shutil.rmtree(tmp)
    rec['all_reproduce_bitwise'] = all(all(p['api_reproduces_evaluated_DEV_REPORT_wire_bitwise'].values()) and all(p['csv_reproduces_evaluated_DEV_REPORT_wire_bitwise'].values())
                                       for p in rec['packages'].values())
    rec['all_refusals_hold'] = all(all(p['refusals'].values()) for p in rec['packages'].values())
    Path(report).write_text(json.dumps(rec, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'all_reproduce_bitwise': rec['all_reproduce_bitwise'], 'all_refusals_hold': rec['all_refusals_hold'], 'packages': list(rec['packages'])}))
    if not (rec['all_reproduce_bitwise'] and rec['all_refusals_hold']):
        raise SystemExit(2)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out-root', required=True); ap.add_argument('--report', required=True)
    ap.add_argument('--package', action='append', default=[])
    a = ap.parse_args(); main(a.out_root, a.report, a.package)
