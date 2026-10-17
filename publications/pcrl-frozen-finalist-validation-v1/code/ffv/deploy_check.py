"""Unit FFV_DEPLOY: raw six-field deployment checks of the admitted source packages (PROTOCOL.md section 12). No label is read.

1. Reproduction: each admitted package (FINALIST and TASKONLY both seeds; NEURAL_RAW seed 0; H_LJ both seeds) must reproduce its
   evaluated DEV_REPORT outputs (the frozen bank) bitwise through (a) the documented wrapper API (amd.deploy.release, or the
   source campaign's strict check_rows + allowed-output gate around the frozen references' own code), (b) the wrapper CSV path,
   and (c) the underlying source release function called directly. NEURAL_RAW seed 1 has no package (coverage gap).
2. Refusals: the source refusal suite (amd.deploy.refusals) run separately against the wrapper and against the underlying
   function, so each refusal is attributed to the layer that actually enforces it.
3. Batch-size characterization: single-row requests and 8-row batches for a fixed random subset of DEV_REPORT rows versus the
   evaluated batch outputs (bitwise), for FINALIST (both seeds) and H_LJ (seed 0).
4. Threshold-tie characterization (source D3), FINALIST both seeds: the income statistic d recomputed with the package arithmetic
   on every role's encoded inputs; people exactly at / within 1e-12 of a frozen threshold; for them, d by an exactly rounded sum
   (math.fsum of per-tree terms) and by reversed summation, and the level each would take; nearest distance on evaluated roles.
Nothing is repaired: thresholds, calibration and summation stay as frozen.
"""
from __future__ import annotations
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from acs_replication import deploy as SRC_DEPLOY
from acs_replication import trees
from acs_hybrids import hybrid_deploy as HYB
from amd import deploy as AD
from amd import fam_b
from . import common as C

PKGS = {'FINALIST_S0': fam_b.package_release, 'FINALIST_S1': fam_b.package_release, 'TASKONLY_S0': fam_b.package_release,
        'TASKONLY_S1': fam_b.package_release, 'NEURAL_RAW_S0': SRC_DEPLOY.release, 'H_LJ_S0': HYB.release, 'H_LJ_S1': HYB.release}


def dev_rows():
    raw = C.load_role('DEV_REPORT', ('features', 'raw_inputs'))['raw_inputs']
    return [{c: str(raw[c][i]) for c in AD.PERMITTED} for i in range(len(raw['row_id']))]


def wrapper(name, d):
    if name.startswith(('FINALIST', 'TASKONLY')):
        return (lambda rr, req=('probability', 'decision'): AD.release(d, rr, req)), (lambda p: AD.release_csv(d, p))
    fn = PKGS[name]
    def api(rr, req=('probability', 'decision')):
        if set(req) - AD.ALLOWED:
            raise PermissionError('hidden-output request refused')
        AD.check_rows(rr)
        return fn(d, rr, req)
    return api, (lambda p: api(SRC_DEPLOY.read_raw_csv(p)))


def underlying(name, d):
    fn = PKGS[name]
    return (lambda rr, req=('probability', 'decision'): fn(d, rr, req)), (lambda p: fn(d, SRC_DEPLOY.read_raw_csv(p)))


def same(res, ref):
    return {t: bool(np.array_equal(res[t]['probability'], ref[t]) and np.array_equal(res[t]['decision'], ref[t].argmax(1))) for t in C.TASKS}


def tie_analysis(seed):
    d_pkg = C.ADM / f'deploy/FINALIST_S{seed}'; meta = json.loads((d_pkg / 'PACKAGE.json').read_text())
    names = tuple(meta['encoder']['feature_names'])
    te = trees.TreeTeacher(json.loads((d_pkg / meta['teachers']['income']).read_text()))
    th = {}
    for arm, f in meta['thetas'].items():
        with np.load(d_pkg / f) as z:
            th[arm] = z['theta_income'].copy()
    with np.load(d_pkg / meta['h']) as z:
        thr = [float(x) for x in z['thresholds']]
    a = meta['config']['alpha']
    with np.load(C.admitted(f'runs/B/B3CTL_UNCOUPLED_MIX50_C6A/S{seed}/RELEASE.npz')) as z:
        rel = {k: z[k].copy() for k in z.files}
    out, nearest = {}, {}
    for r in C.ROLES6:
        X = C.load_role(r, ('features',))['X'].astype(np.float32)
        leaves = te.route(X, names)
        rl, rj = te.raw_from_leaves(leaves, th['LOCAL_TREE']).reshape(-1), te.raw_from_leaves(leaves, th['JOINT_TREE']).reshape(-1)
        d = (1 - a) * rl + a * rj
        lev = fam_b.coarse_bins(d, thr)
        dist = np.min(np.abs(d[:, None] - np.array(thr)[None, :]), axis=1)
        nearest[r] = float(dist.min())
        tied = np.flatnonzero(dist <= 1e-12)
        rows = []
        for i in tied:
            t_idx = np.arange(te.T)
            def raw_exact(theta):
                terms = [te.scale * te.leaf_values[t, leaves[i, t], 0] for t in t_idx]
                cor = [float(theta.reshape(te.T, te.L, te.dim)[t, leaves[i, t], 0]) for t in t_idx]
                return math.fsum(terms) + float(te.bias[0]) + math.fsum(cor) / te.T
            def raw_rev(theta):
                lv = te.leaf_values[t_idx[::-1], leaves[i, ::-1], 0]
                cor = theta.reshape(te.T, te.L, te.dim)[t_idx[::-1], leaves[i, ::-1], 0]
                return te.scale * np.add.reduce(lv) + float(te.bias[0]) + np.add.reduce(cor) / te.T
            d_ex = (1 - a) * raw_exact(th['LOCAL_TREE']) + a * raw_exact(th['JOINT_TREE'])
            d_rv = (1 - a) * raw_rev(th['LOCAL_TREE']) + a * raw_rev(th['JOINT_TREE'])
            p_rel = rel[f'{r}__income'][i]
            rows.append({'exactly_on_threshold': bool(dist[i] == 0.0), 'distance_package': float(dist[i]),
                         'level_package_arithmetic': int(lev[i]), 'level_exactly_rounded_sum': int(fam_b.coarse_bins([d_ex], thr)[0]),
                         'level_reversed_sum': int(fam_b.coarse_bins([d_rv], thr)[0]),
                         'd_package_minus_exact': float(d[i] - d_ex), 'stored_release_income_decision': int(p_rel.argmax())})
        out[r] = {'people_within_1e-12_of_a_threshold': int(len(tied)), 'detail': rows}
    return {'per_role': out, 'nearest_distance_to_a_threshold_by_role': nearest, 'thresholds_repr': [repr(x) for x in thr]}


def main():
    rows = dev_rows(); tmp = Path(tempfile.mkdtemp(dir=C.PRIV)); csv_path = tmp / 'dev_report_raw.csv'; AD.write_csv(csv_path, rows, AD.PERMITTED)
    rec = {'schema': 'FFV1-deployment-checks-v1', 'utc': C.now(), 'labels_read': False, 'inputs': AD.PERMITTED, 'packages': {},
           'unavailable': {'NEURAL_RAW_S1': 'no raw-input package exists in the source (deployment coverage gap); its output is admitted by hash'}}
    rng = np.random.default_rng(2026101804); sub = sorted(rng.choice(len(rows), 200, replace=False).tolist())
    for name in PKGS:
        d = C.ADM / f'deploy/{name}'; base, s = name.rsplit('_S', 1); s = int(s)
        bank, _, _ = C.service_bank(s); ref = {t: bank[base]['DEV_REPORT'][t] for t in C.TASKS}
        w_api, w_csv = wrapper(name, d); u_api, u_csv = underlying(name, d)
        p = {'wrapper_api_bitwise': same(w_api(rows), ref), 'wrapper_csv_bitwise': same(w_csv(csv_path), ref),
             'underlying_function_bitwise': same(u_api(rows), ref),
             'refusals_wrapper': AD.refusals(w_api, w_csv, rows, tmp), 'refusals_underlying_function': AD.refusals(u_api, u_csv, rows, tmp),
             'wrapper': 'amd.deploy.release (check_rows + allowed outputs)' if base in ('FINALIST', 'TASKONLY') else 'the source campaign strict check (amd.deploy.check_rows + allowed outputs) around the frozen code',
             'underlying': PKGS[name].__module__ + '.' + PKGS[name].__name__,
             'package_json_sha256': C.sha(d / 'PACKAGE.json')}
        if name in ('FINALIST_S0', 'FINALIST_S1', 'H_LJ_S0'):
            single = [u_api([rows[i]]) for i in sub]
            batches = [u_api([rows[i] for i in sub[k:k + 8]]) for k in range(0, len(sub), 8)]
            diff = {}
            for t in C.TASKS:
                one = np.vstack([x[t]['probability'] for x in single]); eight = np.vstack([x[t]['probability'] for x in batches])
                r_ = ref[t][sub]
                diff[t] = {'single_row_rows_differing_bitwise': int((~(one == r_).all(1)).sum()), 'single_row_max_abs_diff': float(np.abs(one - r_).max()),
                           'eight_row_rows_differing_bitwise': int((~(eight == r_).all(1)).sum()), 'eight_row_max_abs_diff': float(np.abs(eight - r_).max()),
                           'decisions_equal': bool((one.argmax(1) == r_.argmax(1)).all() and (eight.argmax(1) == r_.argmax(1)).all())}
            p['batch_size_characterization'] = {'rows_tested': len(sub), 'subset_seed': 2026101804, 'by_task': diff}
        rec['packages'][name] = p
    rec['threshold_ties'] = {f'S{s}': tie_analysis(s) for s in C.SEEDS}
    rec['all_reproduce_bitwise'] = all(all(v.values()) for p in rec['packages'].values() for k, v in p.items() if k.endswith('_bitwise'))
    rec['all_wrapper_refusals_hold'] = all(all(p['refusals_wrapper'].values()) for p in rec['packages'].values())
    rec['refusals_enforced_only_by_wrapper'] = sorted({k for p in rec['packages'].values() for k, v in p['refusals_underlying_function'].items() if not v and p['refusals_wrapper'][k]})
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    C.write_json_x(C.RESULTS / 'DEPLOYMENT_CHECKS.json', rec)
    print(json.dumps({k: rec[k] for k in ('all_reproduce_bitwise', 'all_wrapper_refusals_hold', 'refusals_enforced_only_by_wrapper')}, indent=1))


if __name__ == '__main__':
    main()
