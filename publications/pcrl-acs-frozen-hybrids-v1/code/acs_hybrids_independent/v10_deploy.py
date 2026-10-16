"""V10 Deployment (label-free; the H_LJ packages and acs_hybrids.hybrid_deploy.release are the artifact under test).

Per model seed:
  1. package file hashes vs PACKAGE.json, DEPLOYMENT_MANIFEST and the source MODEL_MANIFEST (teacher JSONs, E20 checkpoints);
  2. the DEV_REPORT raw permitted fields (export) are written to a CSV in scratch, read back through the deployment CSV
     reader, released with hybrid_deploy.release, and compared bitwise with the bank's H_LJ DEV_REPORT outputs (= LOCAL_TREE
     income and JOINT_TREE occupation stored source outputs); decisions = argmax;
  3. recipient isolation: each recipient is released from a scratch package that holds ONLY its own component (its teacher
     JSON and its own checkpoint file), and again from a full package copy in which the other recipient's component files
     are deleted; both must equal the full-package output bitwise. A further copy replaces the unused half of the
     recipient's own checkpoint (e.g. theta_occupation inside LOCAL_TREE__E20.npz for the income recipient) with zeros:
     the output must not change.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil

import numpy as np

from acs_hybrids import common as C   # hash-checked data loading only

from . import vcommon as V


def write_csv(path, raw, cols):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(len(raw['row_id'])):
            w.writerow([str(raw[c][i]) for c in cols])


def rewrite_meta(d, meta, components, files):
    m = dict(meta)
    m['components'] = components
    m['files'] = {f: V.sha256_file(d / f) for f in files}
    (d / 'PACKAGE.json').write_text(json.dumps(m, indent=2, sort_keys=True))


def main():
    from acs_hybrids import hybrid_deploy as H
    from acs_replication import deploy as base_deploy
    ap = argparse.ArgumentParser()
    ap.add_argument('--scratch', required=True)
    ap.add_argument('--out', default=str(V.OUT / 'V10_deployment.json'))
    a = ap.parse_args()
    scratch = V.scratch_dir(a.scratch) / 'v10'
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, mode=0o700)
    mm = V.jload(V.SRC_RES / 'MODEL_MANIFEST.json')['models']
    dman = V.jload(V.RES / 'DEPLOYMENT_MANIFEST.json')['packages']
    raw = C.load_role('DEV_REPORT', ('features', 'raw_inputs'))['raw_inputs']
    R = C.load_role('DEV_REPORT', ('features',))
    cols = list(base_deploy.PERMITTED)
    raw_csv = scratch / 'DEV_REPORT_raw_permitted.csv'
    write_csv(raw_csv, raw, cols)
    rows = base_deploy.read_raw_csv(raw_csv)
    res = {'check': 'V10 Deployment', 'scope': 'release replay of the frozen H_LJ packages from raw permitted fields (label-free)',
           'permitted_fields': cols, 'raw_rows': len(rows),
           'raw_export_row_order_equals_DEV_REPORT_export': bool(np.array_equal(raw['row_id'], R['row_id'])),
           'raw_export_contains_SEX_or_PWGTP': bool({'SEX', 'sex', 'PWGTP', 'pwgtp'} & set(raw)), 'per_seed': {}}
    ok = res['raw_export_row_order_equals_DEV_REPORT_export'] and not res['raw_export_contains_SEX_or_PWGTP']
    for s in V.SEEDS:
        pkg = V.PRIV / 'deploy' / f'H_LJ_S{s}'
        meta = json.loads((pkg / 'PACKAGE.json').read_text())
        bank, ids = C.load_bank(s, roles=('DEV_REPORT',))
        if not np.array_equal(ids['row_id__DEV_REPORT'], R['row_id']):
            raise SystemExit('bank rows differ')
        rec = {'package_json_sha256_equals_DEPLOYMENT_MANIFEST': V.sha256_file(pkg / 'PACKAGE.json') == dman[f'H_LJ_S{s}']['package_json_sha256'],
               'package_files_match_PACKAGE_json': {f: V.sha256_file(pkg / f) == h for f, h in meta['files'].items()},
               'components': {}}
        for t, svc in (('income', 'LOCAL_TREE'), ('occupation', 'JOINT_TREE')):
            comp = meta['components'][t]
            rec['components'][t] = {'source_service': comp['source_service'], 'theta_key': comp['theta_key'],
                                    'teacher_sha256_equals_MODEL_MANIFEST': V.sha256_file(pkg / comp['teacher']) == mm[f'T_TREE_{t}_S{s}']['sha256'],
                                    'theta_sha256_equals_MODEL_MANIFEST': V.sha256_file(pkg / comp['theta_file']) == mm[f'{svc}_E20_S{s}']['sha256'],
                                    'expected_source': svc, 'source_matches_lock': comp['source_service'] == svc and comp['theta_key'] == f'theta_{t}'}
        full = H.release(pkg, rows)
        par = {}
        for t in V.TASKS:
            p = full[t]['probability']
            ref = bank['H_LJ']['DEV_REPORT'][t]
            par[t] = {'bitwise': bool(p.dtype == ref.dtype and np.array_equal(p, ref)), 'max_abs_diff': V.maxabs(p, ref),
                      'decisions_equal_argmax': bool(np.array_equal(full[t]['decision'], ref.argmax(1)))}
        rec['parity_vs_bank_H_LJ_DEV_REPORT'] = par
        iso = {}
        for t in V.TASKS:
            other = 'occupation' if t == 'income' else 'income'
            comp = meta['components'][t]
            ocomp = meta['components'][other]
            # (a) package holding only this recipient's component
            d1 = scratch / f'only_{t}_S{s}'
            d1.mkdir(mode=0o700)
            for f in (comp['teacher'], comp['theta_file']):
                shutil.copy2(pkg / f, d1 / f)
            rewrite_meta(d1, meta, {t: comp}, [comp['teacher'], comp['theta_file']])
            p1 = H.release(d1, rows, tasks=(t,))[t]['probability']
            # (b) full package copy with the other recipient's component files deleted (PACKAGE.json untouched)
            d2 = scratch / f'other_deleted_{t}_S{s}'
            shutil.copytree(pkg, d2)
            for f in {ocomp['teacher'], ocomp['theta_file']} - {comp['teacher'], comp['theta_file']}:
                (d2 / f).unlink()
            p2 = H.release(d2, rows, tasks=(t,))[t]['probability']
            other_refused = None
            try:
                H.release(d2, rows[:3], tasks=(other,))
                other_refused = False
            except Exception as e:
                other_refused = type(e).__name__
            # (c) own checkpoint with its unused half zeroed
            d3 = scratch / f'unused_half_zeroed_{t}_S{s}'
            d3.mkdir(mode=0o700)
            shutil.copy2(pkg / comp['teacher'], d3 / comp['teacher'])
            with np.load(pkg / comp['theta_file']) as z:
                arrs = {k: z[k].copy() for k in z.files}
            unused = f'theta_{other}'
            arrs[unused] = np.zeros_like(arrs[unused])
            with open(d3 / comp['theta_file'], 'wb') as f:
                np.savez(f, **arrs)
            rewrite_meta(d3, meta, {t: comp}, [comp['teacher'], comp['theta_file']])
            p3 = H.release(d3, rows, tasks=(t,))[t]['probability']
            iso[t] = {'own_component_only_package_equals_full': bool(np.array_equal(p1, full[t]['probability'])),
                      'files_in_own_component_package': sorted(x.name for x in d1.iterdir()),
                      'other_component_deleted_equals_full': bool(np.array_equal(p2, full[t]['probability'])),
                      'other_recipient_release_after_deletion': f'refused ({other_refused})' if other_refused else 'NOT refused',
                      'unused_checkpoint_half_zeroed_equals_full': bool(np.array_equal(p3, full[t]['probability']))}
        rec['recipient_isolation'] = iso
        # all four bank roles recomputed from the admitted (encoded) features through each recipient's own component; the
        # STAGE_B output file carries no row ids of its own, so this also re-checks its row order for the H_LJ arrays
        allroles = {}
        for role in V.ROLES:
            Xr = C.load_role(role, ('features',))
            names = tuple(Xr['feature_names'])
            bk, ids_r = C.load_bank(s, roles=(role,))
            for t in V.TASKS:
                p = H.release_task(pkg, Xr['X'], names, t)['probability']
                allroles[f'{role}|{t}'] = {'bitwise': bool(np.array_equal(p, bk['H_LJ'][role][t])), 'max_abs_diff': V.maxabs(p, bk['H_LJ'][role][t]),
                                           'feature_names_equal_package_encoder': names == tuple(meta['encoder']['feature_names']),
                                           'bank_rows_equal_export': bool(np.array_equal(ids_r[f'row_id__{role}'], Xr['row_id']))}
        rec['all_roles_recompute_from_admitted_features'] = allroles
        seed_ok = (all(v['bitwise'] and v['feature_names_equal_package_encoder'] and v['bank_rows_equal_export'] for v in allroles.values()) and rec['package_json_sha256_equals_DEPLOYMENT_MANIFEST'] and all(rec['package_files_match_PACKAGE_json'].values())
                   and all(c['teacher_sha256_equals_MODEL_MANIFEST'] and c['theta_sha256_equals_MODEL_MANIFEST'] and c['source_matches_lock'] for c in rec['components'].values())
                   and all(v['bitwise'] and v['decisions_equal_argmax'] for v in par.values())
                   and all(v['own_component_only_package_equals_full'] and v['other_component_deleted_equals_full'] and v['unused_checkpoint_half_zeroed_equals_full'] for v in iso.values()))
        rec['pass'] = bool(seed_ok)
        ok &= seed_ok
        res['per_seed'][f'S{s}'] = rec
    res['status'] = 'PASS' if ok else 'FAIL'
    V.write_json(a.out, res)
    print({'status': res['status'], **{k: v['pass'] for k, v in res['per_seed'].items()}})


if __name__ == '__main__':
    main()
