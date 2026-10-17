"""Unit UM_DEPLOY: raw-field deployment packages for the selected releases, and the frozen references' existing source packages.

Package format: the source SCM package (`scm_bench.deploy_scm`): PACKAGE.json (the frozen 145-column encoder admitted by hash,
the ten fold-teacher payloads, file hashes) and CELLS.npz (fine-cell centroids, cell-of-initial and group-of-initial maps, stored
v1 / v2 tables of the selected checkpoint). Release: encode -> fold-average teacher probabilities -> KL-nearest fine cell -> final
cell and income group -> stored float64 vectors copied bitwise -> argmax decisions.

Entry points: `release(pkg, rows)` (API; every row must have exactly AGEP, WKHP, COW, SCHL, MAR, POBP in that order and no
forbidden field) and `release_csv(pkg, path)` (CSV; the shared reader refuses forbidden, extra, missing or reordered columns).
Both refuse requests for anything but probability and decision (cell or group ids, teacher probabilities, propensity).
Packaged roles: SCM, ORD_NLLM and IBSI_NLLM when selected and matched; frozen NEURAL_RAW (seed 0 only; the source never built
seed 1) and H_LJ (both seeds) because their actual source packages exist. Each must reproduce the evaluated DEV_REPORT
probabilities and decisions bitwise through both entry points. No label is read.
"""
from __future__ import annotations
import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from acs_replication import deploy as SRC_DEPLOY
from acs_replication import schema
from scm_bench import deploy_scm as SB_DEPLOY
from scm_bench import common as SB
from . import common as C
from .select import load_traj
from .readers import load_service_bank

PERMITTED = list(SRC_DEPLOY.PERMITTED)
HIDDEN = ('cell_id', 'group_id', 'teacher_probability', 'propensity', 'income_group', 'fine_cell')


def check_rows(rows):
    for r in rows:
        keys = list(r.keys())
        bad = [k for k in keys if str(k).upper() in schema.FORBIDDEN]
        if bad:
            raise PermissionError(f'forbidden fields present (never accepted as inputs): {bad}')
        if keys != PERMITTED:
            raise ValueError(f'incompatible schema: expected exactly {PERMITTED} in this order, got {keys}')


def release(pkg_dir, rows, requested=('probability', 'decision')):
    if set(requested) - SB_DEPLOY.ALLOWED:
        raise PermissionError(f'hidden-output request refused: {sorted(set(requested) - SB_DEPLOY.ALLOWED)}')
    check_rows(rows)
    return SB_DEPLOY.release(pkg_dir, rows, requested)


def release_csv(pkg_dir, path, requested=('probability', 'decision')):
    return release(pkg_dir, SRC_DEPLOY.read_raw_csv(path), requested)


def build(seed, role, rid, tabs, out_root):
    d = Path(out_root) / f'{role}_S{seed}'
    if d.exists():
        raise PermissionError('package exists')
    d.mkdir(parents=True, mode=0o700)
    enc = json.loads(C._admitted_copy('encoder/T_TREE_S0_PACKAGE.json').read_text())['encoder']
    meta = {'schema': 'SCMUM1-deploy-package-v1', 'kind': 'SCM', 'service_role': role, 'release': rid, 'seed': seed, 'encoder': enc, 'teachers': {}, 'files': {}}
    for f in range(5):
        for t in C.TASKS:
            src = C._admitted_copy(f'fit/FIT_{seed}/TEACHERS/fold{f}_{t}.json'); shutil.copy2(src, d / src.name)
            meta['teachers'].setdefault(t, []).append(src.name); meta['files'][src.name] = C.sha(d / src.name)
    P, _, _ = C.source_fit(seed)
    np.savez(d / 'CELLS.npz', cent1=P['cent1'], cent2=P['cent2'], **{k: tabs[rid][k] for k in ('v1', 'v2', 'cell_of_initial', 'group_of_initial')})
    meta['cells'] = 'CELLS.npz'; meta['files']['CELLS.npz'] = C.sha(d / 'CELLS.npz')
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
    """Every case must raise: extra column, reordered columns, SEX column (CSV and API), extra/reordered keys (API), hidden outputs."""
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


def main(out_root, report):
    for name in ('REPORTING_LOCK.json', 'SELECTION_LOCK.json', 'SERVICE_BANK_MANIFEST.json'):
        if not C.committed(C.RESULTS / name):
            raise PermissionError(f'packaging requires the committed {name}')
    sel = json.loads((C.RESULTS / 'SELECTION_LOCK.json').read_text())
    tl = json.loads((C.RESULTS / 'TRAJECTORY_COMPLETE_LOCK.json').read_text())
    rows = raw_rows(); tmp = Path(tempfile.mkdtemp())
    csv_path = tmp / 'dev_report_raw.csv'; write_csv(csv_path, rows, PERMITTED)
    rec = {'schema': 'SCMUM1-deployment-v1', 'utc': C.now(), 'inputs': PERMITTED, 'packages': {}, 'unavailable_package_roles': {}}
    for s in C.SEEDS:
        _, tabs = load_traj(s, tl)
        bank, _, man = load_service_bank(s)
        svc_of = man['per_seed'][f'S{s}']['service_of_role']
        for role in ('SCM', 'ORD_NLLM', 'IBSI_NLLM'):
            r = sel['roles'][str(s)][role]
            ok_status = r.get('status') in ('NOMINEE_SELECTED', 'INELIGIBLE_DIAGNOSTIC_FALLBACK', 'MATCHED')
            if not (ok_status and r.get('release')):
                rec['unavailable_package_roles'][f'{role}_S{s}'] = f"not packaged: role status {r.get('status')}"
                continue
            d = build(s, role, r['release'], tabs, out_root)
            ref = bank[svc_of[role]]['DEV_REPORT']
            api = release(d, rows); via_csv = release_csv(d, csv_path)
            rec['packages'][f'{role}_S{s}'] = {'release': r['release'], 'role_status': r['status'], 'package_json_sha256': C.sha(d / 'PACKAGE.json'),
                                               'api_reproduces_evaluated_DEV_REPORT_wire_bitwise': same(api, ref),
                                               'csv_reproduces_evaluated_DEV_REPORT_wire_bitwise': same(via_csv, ref),
                                               'refusals': refusals(lambda rr, req=('probability', 'decision'): release(d, rr, req), lambda p: release_csv(d, p), rows, tmp)}
    # frozen references: their actual source packages (hash-checked), run through the source deployment code
    man_src = json.loads((SB.SRC_RESULTS / 'MODEL_MANIFEST.json').read_text())
    man_hyb = json.loads((SB.HYB_RESULTS / 'DEPLOYMENT_MANIFEST.json').read_text())
    from acs_hybrids import hybrid_deploy as HYB
    frozen = [('NEURAL_RAW', 0, SB.SRC_PRIV / 'deploy' / 'NEURAL_RAW_rho1.0_E2_S0', man_src['deployment_packages']['NEURAL_RAW_rho1.0_E2_S0']['package_json_sha256'], SRC_DEPLOY.release)]
    frozen += [('H_LJ', s, SB.HYB_PRIV / 'deploy' / f'H_LJ_S{s}', man_hyb['packages'][f'H_LJ_S{s}']['package_json_sha256'], HYB.release) for s in C.SEEDS]
    rec['unavailable_package_roles']['NEURAL_RAW_S1'] = 'the source study never built a seed-1 neural package (coverage gap)'
    for name, s, src, h, fn in frozen:
        if C.sha(src / 'PACKAGE.json') != h:
            raise PermissionError(f'{name} S{s} package hash differs from its public manifest')
        d = Path(out_root) / f'{name}_S{s}'
        if d.exists():
            raise PermissionError('package exists')
        shutil.copytree(src, d)
        bank, _, man = load_service_bank(s)
        ref = bank[name]['DEV_REPORT']

        def api_fn(rr, req=('probability', 'decision'), d=d, fn=fn):
            check_rows(rr)
            return fn(d, rr, req)
        res_api = api_fn(rows); res_csv = api_fn(SRC_DEPLOY.read_raw_csv(csv_path))
        rec['packages'][f'{name}_S{s}'] = {'source_package_json_sha256': h, 'api_reproduces_evaluated_DEV_REPORT_wire_bitwise': same(res_api, ref),
                                           'csv_reproduces_evaluated_DEV_REPORT_wire_bitwise': same(res_csv, ref),
                                           'refusals': refusals(api_fn, lambda p, api_fn=api_fn: api_fn(SRC_DEPLOY.read_raw_csv(p)), rows, tmp),
                                           'note': 'frozen reference; its own source package, run through the source deployment code behind the same strict input check'}
    shutil.rmtree(tmp)
    rec['all_reproduce_bitwise'] = all(all(p['api_reproduces_evaluated_DEV_REPORT_wire_bitwise'].values()) and all(p['csv_reproduces_evaluated_DEV_REPORT_wire_bitwise'].values())
                                       for p in rec['packages'].values())
    rec['all_refusals_hold'] = all(all(p['refusals'].values()) for p in rec['packages'].values())
    Path(report).write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in rec.items() if k != 'packages'} | {'packages': {k: {kk: vv for kk, vv in v.items() if kk != 'refusals'} for k, v in rec['packages'].items()}}, indent=1))
    if not (rec['all_reproduce_bitwise'] and rec['all_refusals_hold']):
        raise SystemExit(2)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out-root', required=True); ap.add_argument('--report', required=True)
    a = ap.parse_args(); main(a.out_root, a.report)
