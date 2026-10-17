"""Verifier check 6: raw-field packages and input validation.

For each package in .private/pcrl_scm_utility_matched_v1/deploy (SCM, ORD_NLLM, IBSI_NLLM both seeds in the SCM format; frozen
NEURAL_RAW seed 0 and H_LJ both seeds as copied source packages):
- integrity: package file hashes; SCM-format teachers equal the admitted fold teachers, the encoder equals the admitted encoder
  record, CELLS.npz centroids equal PARTITION and the tables equal the locked release tables of the role's selected release;
  frozen packages equal their public manifests.
- own path (SCM format): own raw-field encoder (from the encoder record) -> compared bitwise with the DEV_REPORT feature export;
  own fold-average teacher evaluation and KL routing (check 1 code) -> compared with the admitted routing; own table lookup ->
  compared bitwise with the evaluated DEV_REPORT wire of the final bank.
- black box (objects under test): scm_um.deploy.release / release_csv for the SCM-format packages; the frozen references through
  their source entry points (acs_replication.deploy.release, acs_hybrids.hybrid_deploy.release) directly AND behind the campaign's
  strict row check (scm_um.deploy.check_rows), API and CSV (acs_replication.deploy.read_raw_csv) paths; outputs compared bitwise.
- refusals: extra / reordered / missing / duplicated / lower-case 'sex' / SEX columns and keys, SEX replacing a field, hidden-output
  requests. A refusal counts only if an exception is raised; silent acceptance is recorded.
The DEV_REPORT raw-input CSV is written to the session scratchpad (mode 600) and deleted at the end. No label is read.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402
sys.path.insert(1, str(V.ROOT))

SCRATCH = Path('<session scratch>')
PERMITTED = ['AGEP', 'WKHP', 'COW', 'SCHL', 'MAR', 'POBP']
DEP = V.PRIV / 'deploy'


def own_encode(rows, enc):
    names = list(enc['feature_names']); col = {n: i for i, n in enumerate(names)}
    X = np.zeros((len(rows), len(names)))
    for i, r in enumerate(rows):
        for c in ('AGEP', 'WKHP'):
            X[i, col[c]] = (float(r[c]) - enc['numeric'][c]['mean']) / enc['numeric'][c]['sd']
        for c in ('COW', 'SCHL', 'MAR', 'POBP'):
            k = f'{c}={str(r[c]).strip()}'
            X[i, col[k] if k in col else col[f'{c}=__OTHER__']] = 1.0
    return X.astype(np.float32)


def own_teacher(payload, X):
    feats = np.asarray(payload['split_features'], dtype=np.int64); borders = np.asarray(payload['split_borders'], dtype=np.float32)
    leaves = np.asarray(payload['leaf_values'], dtype=np.float64); T, depth = feats.shape
    Xf = np.ascontiguousarray(X, dtype=np.float32); raw = np.zeros((len(X), leaves.shape[2]))
    for t in range(T):
        leaf = np.zeros(len(X), dtype=np.int64)
        for i in range(depth):
            leaf += (Xf[:, feats[t, i]] > borders[t, i]).astype(np.int64) << i
        raw += leaves[t, leaf, :]
    raw = float(payload['scale']) * raw + np.asarray(payload['bias'], dtype=np.float64)
    lg = np.column_stack((-raw[:, 0] / 2.0, raw[:, 0] / 2.0)) if payload['task'] == 'income' else raw - raw.mean(1, keepdims=True)
    lg = lg - lg.max(1, keepdims=True); e = np.exp(lg)
    return e / e.sum(1, keepdims=True)


def own_route(t1, t2, c1, c2):
    l1 = np.log(np.maximum(c1, 1e-300)); l2 = np.log(np.maximum(c2, 1e-300)); out = np.empty(len(t1), dtype=np.int64)
    for a in range(0, len(t1), 1000):
        sc = (t1[a:a + 1000, None, :] * l1[None]).sum(-1) + (t2[a:a + 1000, None, :] * l2[None]).sum(-1)
        out[a:a + 1000] = np.argmax(sc, 1)
    return out


def write_csv(path, rows, header, extra=None):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows:
            rr = dict(r, **(extra or {}))
            w.writerow([str(rr.get(c, '1')) for c in header])


def same(res, ref):
    return {t: bool(np.array_equal(res[t]['probability'], ref[t]) and res[t]['probability'].dtype == np.float64
                    and np.array_equal(res[t]['decision'], np.argmax(ref[t], 1))) for t in V.TASKS}


def attempt(fn):
    try:
        fn(); return 'NOT_REFUSED (silently accepted)'
    except (PermissionError, ValueError, KeyError) as e:
        return f'refused: {type(e).__name__}'


def refusal_suite(api, csvfn, rows):
    """api(rows, requested) and csvfn(path) under test; returns case -> outcome."""
    out = {}
    few = rows[:5]
    csv_cases = {'extra_column_PUMA': (PERMITTED + ['PUMA'], {'PUMA': '00101'}), 'reordered_columns': (list(reversed(PERMITTED)), None),
                 'SEX_column_appended': (PERMITTED + ['SEX'], {'SEX': '1'}), 'SEX_replacing_POBP': (PERMITTED[:-1] + ['SEX'], {'SEX': '1'}),
                 'lowercase_sex_column': (PERMITTED + ['sex'], {'sex': '1'}), 'missing_column': (PERMITTED[:-1], None),
                 'duplicated_column': (PERMITTED + ['AGEP'], None), 'RAC1P_column': (PERMITTED + ['RAC1P'], {'RAC1P': '1'})}
    for name, (hdr, extra) in csv_cases.items():
        p = SCRATCH / f'case_{name}.csv'; write_csv(p, few, hdr, extra)
        out[f'csv|{name}'] = attempt(lambda p=p: csvfn(p))
        p.unlink()
    api_cases = {'SEX_key': [dict(r, SEX='1') for r in few], 'lowercase_sex_key': [dict(r, sex='1') for r in few], 'extra_key_PUMA': [dict(r, PUMA='00101') for r in few],
                 'reordered_keys': [{c: r[c] for c in reversed(PERMITTED)} for r in few], 'missing_key': [{c: r[c] for c in PERMITTED[:-1]} for r in few],
                 'SEX_replacing_POBP': [{**{c: r[c] for c in PERMITTED[:-1]}, 'SEX': '1'} for r in few]}
    for name, rr in api_cases.items():
        out[f'api|{name}'] = attempt(lambda rr=rr: api(rr, ('probability', 'decision')))
    for h in ('cell_id', 'group_id', 'teacher_probability', 'propensity', 'income_group', 'fine_cell', 'raw_score', 'leaf_ids', 'representation'):
        out[f'hidden|{h}'] = attempt(lambda h=h: api(few, ('probability', h)))
    out['hidden|decision_only_allowed'] = 'allowed' if attempt(lambda: api(few, ('decision',))).startswith('NOT_REFUSED') else 'refused'
    return out


def main():
    t0 = time.process_time()
    SCRATCH.mkdir(mode=0o700, exist_ok=True)
    from scm_um import deploy as UD                       # objects under test
    from acs_replication import deploy as SD
    from acs_hybrids import hybrid_deploy as HD
    sel = V.jload(V.RES / 'SELECTION_LOCK.json'); sbm = V.jload(V.RES / 'SERVICE_BANK_MANIFEST.json'); adm = V.jload(V.RES / 'SOURCE_ADMISSION.json')
    dres = V.jload(V.RES / 'DEPLOYMENT_RESULTS.json')
    raw = V.export('DEV_REPORT', 'raw_inputs'); F = V.export('DEV_REPORT', 'features')
    assert np.array_equal(raw['row_id'], F['row_id'])
    rows = [{c: raw[c][i] for c in PERMITTED} for i in range(len(raw['row_id']))]
    csv_path = SCRATCH / 'dev_report_raw.csv'; write_csv(csv_path, rows, PERMITTED)
    enc_adm = V.jload(V.SRC / 'encoder' / 'T_TREE_S0_PACKAGE.json')['encoder']
    X_own = own_encode(rows, enc_adm)
    out = {'check': 6, 'title': 'raw-field packages and input validation', 'packages': {},
           'own_encoder': {'feature_names_equal_export': list(enc_adm['feature_names']) == [str(x) for x in F['feature_names']],
                           'encoded_equals_DEV_REPORT_feature_export_bitwise': bool(np.array_equal(X_own, F['X']) and X_own.dtype == F['X'].dtype),
                           'rows': len(rows)}}
    for s in V.SEEDS:
        SB = V.npz(V.PRIV / 'bank' / f'SERVICE_BANK_S{s}.npz'); assert V.sha_file(V.PRIV / 'bank' / f'SERVICE_BANK_S{s}.npz') == sbm['bank_files_sha256'][f'S{s}']
        RT = V.npz(V.SRC / 'routing' / f'ROUTING_S{s}.npz'); RTAB = V.npz(V.PRIV / 'runs' / f'TRAJ_{s}' / 'RELEASE_TABLES.npz')
        P = V.npz(V.SRC / 'fit' / f'FIT_{s}' / 'PARTITION.npz')
        svc_of = sel['services'][str(s)]['service_of_role']
        teach = {t: [V.jload(V.SRC / 'fit' / f'FIT_{s}' / 'TEACHERS' / f'fold{f}_{t}.json') for f in range(5)] for t in V.TASKS}
        prob = {t: sum((own_teacher(teach[t][f], X_own) / 5.0 for f in range(5)), start=np.zeros((len(rows), V.KCLS[t]))) for t in V.TASKS}
        # the start=0 + sequential adds reproduce acc = acc + p_f / 5 in fold order
        init_own = own_route(prob['income'], prob['occupation'], P['cent1'], P['cent2'])
        for role in ('SCM', 'ORD_NLLM', 'IBSI_NLLM'):
            name = f'{role}_S{s}'; d = DEP / name; meta = V.jload(d / 'PACKAGE.json')
            rid = sel['roles'][str(s)][role]['release']
            ref = {t: SB[f'{svc_of[role]}__DEV_REPORT__{t}'] for t in V.TASKS}
            C = V.npz(d / 'CELLS.npz')
            integ = {'file_hashes': all(V.sha_file(d / fn) == h for fn, h in meta['files'].items()),
                     'files_on_disk_equal_manifest': sorted(p.name for p in d.iterdir() if p.name != 'PACKAGE.json') == sorted(meta['files']),
                     'teachers_equal_admitted': all(V.sha_file(d / f'fold{f}_{t}.json') == adm['copies'][f'fit/FIT_{s}/TEACHERS/fold{f}_{t}.json']['sha256'] for f in range(5) for t in V.TASKS),
                     'encoder_equals_admitted_record': meta['encoder'] == enc_adm, 'release_id_equals_selection_lock': meta['release'] == rid,
                     'centroids_equal_partition': bool(np.array_equal(C['cent1'], P['cent1']) and np.array_equal(C['cent2'], P['cent2'])),
                     'tables_equal_locked_release': all(np.array_equal(C[k], RTAB[f'{rid}__{k}']) for k in ('v1', 'v2', 'cell_of_initial', 'group_of_initial')),
                     'package_json_sha_equals_deployment_results': V.sha_file(d / 'PACKAGE.json') == dres['packages'][name]['package_json_sha256']}
            own_out = {'income': C['v1'][C['group_of_initial'][init_own]], 'occupation': C['v2'][C['cell_of_initial'][init_own]]}
            own = {'routing_equals_admitted_DEV_REPORT_routing': bool(np.array_equal(init_own, RT['init__DEV_REPORT'])),
                   'own_lookup_equals_bank_wire': {t: bool(np.array_equal(own_out[t], ref[t])) for t in V.TASKS}}
            api = UD.release(d, rows); via = UD.release_csv(d, csv_path)
            bb = {'api_equals_bank_wire': same(api, ref), 'csv_equals_bank_wire': same(via, ref),
                  'api_equals_own_path': {t: bool(np.array_equal(api[t]['probability'], own_out[t])) for t in V.TASKS}}
            ref_suite = refusal_suite(lambda rr, req, d=d: UD.release(d, rr, req), lambda p, d=d: UD.release_csv(d, p), rows)
            out['packages'][name] = {'integrity': integ, 'own_path': own, 'black_box': bb, 'refusals_scm_um_deploy': ref_suite}
        # frozen references
        frozen = []
        if s == 0:
            frozen.append(('NEURAL_RAW', SD.release, V.jload(V.REPL_RES / 'MODEL_MANIFEST.json')['deployment_packages']['NEURAL_RAW_rho1.0_E2_S0']['package_json_sha256']))
        frozen.append(('H_LJ', HD.release, V.jload(V.HYB_RES / 'DEPLOYMENT_MANIFEST.json')['packages'][f'H_LJ_S{s}']['package_json_sha256']))
        for role, fn, h in frozen:
            name = f'{role}_S{s}'; d = DEP / name; meta = V.jload(d / 'PACKAGE.json')
            ref = {t: SB[f'{role}__DEV_REPORT__{t}'] for t in V.TASKS}
            integ = {'package_json_equals_public_manifest': V.sha_file(d / 'PACKAGE.json') == h,
                     'file_hashes': all(V.sha_file(d / fn_) == hh for fn_, hh in meta['files'].items())}
            direct = fn(d, rows); wrapped_csv = fn(d, SD.read_raw_csv(csv_path))
            bb = {'direct_api_equals_bank_wire': same(direct, ref), 'csv_equals_bank_wire': same(wrapped_csv, ref)}

            def wrapped(rr, req, d=d, fn=fn):
                UD.check_rows(rr)
                return fn(d, rr, req)
            out['packages'][name] = {'integrity': integ, 'black_box': bb,
                                     'refusals_behind_campaign_check_rows': refusal_suite(wrapped, lambda p, d=d, fn=fn: fn(d, SD.read_raw_csv(p)), rows),
                                     'refusals_direct_source_entry_point': refusal_suite(lambda rr, req, d=d, fn=fn: fn(d, rr, req), lambda p, d=d, fn=fn: fn(d, SD.read_raw_csv(p)), rows)}
    os.remove(csv_path)
    try:
        SCRATCH.rmdir()
    except OSError:
        pass
    # summary
    pk = out['packages']
    scm_fmt = [k for k in pk if not k.startswith(('NEURAL', 'H_LJ'))]
    out['summary'] = {
        'packages_tested': sorted(pk), 'unavailable_recorded': dres.get('unavailable_package_roles'),
        'own_encoder_bitwise': out['own_encoder']['encoded_equals_DEV_REPORT_feature_export_bitwise'],
        'scm_format_integrity_all': all(all(pk[k]['integrity'].values()) for k in scm_fmt),
        'scm_format_own_path_all': all(pk[k]['own_path']['routing_equals_admitted_DEV_REPORT_routing'] and all(pk[k]['own_path']['own_lookup_equals_bank_wire'].values()) for k in scm_fmt),
        'black_box_bitwise_all': all(all(all(v.values()) for v in pk[k]['black_box'].values()) for k in pk),
        'scm_format_refusals_all_refused': all(v.startswith('refused') for k in scm_fmt for kk, v in pk[k]['refusals_scm_um_deploy'].items() if kk != 'hidden|decision_only_allowed'),
        'frozen_refusals_behind_check_rows_all_refused': all(v.startswith('refused') for k in pk if k not in scm_fmt
                                                             for kk, v in pk[k]['refusals_behind_campaign_check_rows'].items() if kk != 'hidden|decision_only_allowed'),
        'frozen_direct_entry_point_not_refused': {k: sorted(kk for kk, v in pk[k]['refusals_direct_source_entry_point'].items() if v.startswith('NOT_REFUSED') and kk != 'hidden|decision_only_allowed')
                                                  for k in pk if k not in scm_fmt},
        'frozen_integrity_all': all(all(pk[k]['integrity'].values()) for k in pk if k not in scm_fmt)}
    sm = out['summary']
    out['status'] = 'PASS' if (sm['own_encoder_bitwise'] and sm['scm_format_integrity_all'] and sm['scm_format_own_path_all'] and sm['black_box_bitwise_all']
                               and sm['scm_format_refusals_all_refused'] and sm['frozen_refusals_behind_check_rows_all_refused'] and sm['frozen_integrity_all']) else 'FAIL_OR_REVIEW'
    out['cpu_s'] = round(time.process_time() - t0, 2)
    out['code_under_test_imported'] = ['scm_um.deploy.release / release_csv / check_rows', 'acs_replication.deploy.release / read_raw_csv', 'acs_hybrids.hybrid_deploy.release']
    V.write_json('check6_deploy.json', out)
    print(json.dumps({'status': out['status'], 'summary': sm, 'cpu_s': out['cpu_s']}, indent=1, default=str))


if __name__ == '__main__':
    main()
