"""Unit VER_IDENTITY_V1 (check 1: hashes and identity). Label-free: no SEX or task label is read.

Public-use ACS 2019 Florida person records (development roles). Re-hashes every admitted copy against SOURCE_ADMISSION.json and
against the source's own public records; checks the frozen release hash against the source FINALIST_LOCK; checks that the
released income vectors are exactly the six per-level vectors; checks that FINALIST, H_LJ and B2TWIN share one occupation wire
on every role and seed (bank roles directly; all six roles through the release records and an own recomputation of the
JOINT_TREE E20 occupation output from the encoded inputs and the admitted teacher and correction files).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import vcommon as V
from ffv import common as C        # admitted loaders only (load_role, service_bank)

CFG = {'FINALIST': 'B3CTL_UNCOUPLED_MIX50_C6A', 'B2TWIN': 'B2CTL_UNCOUPLED_C6A', 'TASKONLY': 'B3CTL_TASKONLY_TR_C6A'}
REP_RES = V.ROOT / 'results' / 'pcrl_acs_protected_tree_replication_v1'
HYB_RES = V.ROOT / 'results' / 'pcrl_acs_frozen_hybrids_v1'


def load_npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def teacher(seed, task):
    t = V.jload(V.ADM / 'admitted' / 'stage_a' / f'S{seed}' / f'TREE_TEACHER_{task}.json')
    return {'f': np.asarray(t['split_features'], dtype=np.int64), 'b': np.asarray(t['split_borders'], dtype=np.float32),
            'leaf': np.asarray(t['leaf_values'], dtype=np.float64), 'scale': float(t['scale']),
            'bias': np.asarray(t['bias'], dtype=np.float64).reshape(-1), 'names': tuple(t['feature_names']), 'depth': int(t['depth'])}


def raw_scores(te, theta, X, names, chunk=2000):
    """Teacher raw scores plus theta/T (own implementation of the registered oblivious-tree arithmetic)."""
    if tuple(names) != te['names']:
        raise AssertionError('feature order differs from the teacher')
    X = np.ascontiguousarray(X, dtype=np.float32)
    T = te['f'].shape[0]
    dim = te['leaf'].shape[2]
    th = None if theta is None else np.asarray(theta, dtype=np.float64).reshape(T, -1, dim)
    out = np.empty((len(X), dim))
    t = np.arange(T)[None, :]
    for a in range(0, len(X), chunk):
        Xc = X[a:a + chunk]
        leaves = np.zeros((len(Xc), T), dtype=np.int64)
        for i in range(te['depth']):
            leaves |= (Xc[:, te['f'][:, i]] > te['b'][None, :, i]).astype(np.int64) << i
        r = te['scale'] * te['leaf'][t, leaves].sum(1) + te['bias']
        if th is not None:
            r = r + th[t, leaves].sum(1) / T
        out[a:a + chunk] = r
    return out


def softmax_occ(raw):
    lg = raw - raw.mean(1, keepdims=True)
    lg = lg - lg.max(1, keepdims=True)
    e = np.exp(lg)
    return e / e.sum(1, keepdims=True)


def main():
    out = {'unit': 'VER_IDENTITY_V1', 'labels_read': 'none'}
    # ---------------------------------------------------------------- 1a. every admitted copy re-hashed
    adm = V.jload(V.RES / 'SOURCE_ADMISSION.json')
    copies = adm['copies']
    mism, exp_mism, missing, n_exp = [], [], [], 0
    groups = {}
    for rel, rec in sorted(copies.items()):
        p = V.ADM / rel
        g = rel.split('/')[0] + ('/' + rel.split('/')[1] if rel.count('/') >= 2 else '')
        groups[g] = groups.get(g, 0) + 1
        if not p.is_file():
            missing.append(rel)
            continue
        h = V.sha_file(p)
        if h != rec['sha256']:
            mism.append(rel)
        if rec.get('expected_sha256'):
            n_exp += 1
            if h != rec['expected_sha256']:
                exp_mism.append(rel)
    out['admission_rehash'] = {'n_copies_listed': len(copies), 'n_copies_field': adm.get('n_copies'), 'n_rehashed': len(copies) - len(missing),
                               'missing': missing, 'sha256_mismatch': mism, 'n_with_expected_sha256': n_exp,
                               'expected_sha256_mismatch': exp_mism, 'by_group': groups, 'problems_field': adm.get('problems')}

    # ---------------------------------------------------------------- 1b. admitted copies against the source's own public records
    xc = {}
    sbm = V.jload(V.SRC_RES / 'SERVICE_BANK_MANIFEST.json')
    fbm = V.jload(HYB_RES / 'FROZEN_BANK_MANIFEST.json')
    srl = V.jload(V.SRC_RES / 'REPORTING_LOCK.json')
    fl = V.jload(V.SRC_RES / 'FINALIST_LOCK.json')
    ext = V.jload(REP_RES / 'EXTRACTION_SUMMARY.json')['export_file_sha256']
    for s in V.SEEDS:
        xc[f'service_bank_S{s}'] = V.sha_file(V.ADM / 'bank' / f'SERVICE_BANK_S{s}.npz') == sbm['bank_files_sha256'][f'S{s}']
        xc[f'frozen_bank_S{s}'] = V.sha_file(V.ADM / 'admitted' / 'bank' / f'FROZEN_BANK_S{s}.npz') == fbm['bank_files_sha256'][f'S{s}']
        rd = V.ADM / 'runs' / f'READERS_{s}'
        xc[f'source_readers_record_S{s}'] = V.sha_file(rd / 'READERS_SEED.json') == srl['readers'][f'S{s}']['readers_seed_sha256']
        xc[f'source_reader_scores_S{s}'] = V.sha_file(rd / 'DEV_REPORT_SCORES.npz') == srl['readers'][f'S{s}']['dev_report_scores_sha256']
        rr = V.jload(rd / 'READERS_SEED.json')
        bad = [k for k, h in rr['model_files_sha256'].items() if V.sha_file(rd / f'MODEL__{k}.joblib') != h]
        xc[f'source_reader_model_files_S{s}'] = {'n': len(rr['model_files_sha256']), 'mismatch': bad}
        for svc, cfg in CFG.items():
            p = V.ADM / 'runs' / 'B' / cfg / f'S{s}' / 'RELEASE.npz'
            h = V.sha_file(p)
            rj = V.jload(p.with_suffix('.json'))
            xc[f'release_{svc}_S{s}'] = {'file_equals_FINALIST_LOCK': h == fl['release_sha256'][f'B/{cfg}'][f'S{s}'],
                                         'file_equals_RELEASE_json': h == rj['release_sha256']}
    for role, d in ext.items():
        for field, h in d.items():
            p = V.ADM / 'admitted' / 'exports' / role / f'{field}.npz'
            if p.is_file():
                xc[f'export_{role}_{field}'] = V.sha_file(p) == h
    pk = {}
    for d in sorted((V.ADM / 'deploy').iterdir()):
        meta = V.jload(d / 'PACKAGE.json')
        pk[d.name] = {f: V.sha_file(d / f) == h for f, h in meta.get('files', {}).items()}
    xc['deploy_package_file_maps'] = pk
    # this campaign's enhanced reader files against REPORTING_LOCK (fitted readers frozen before the DEV_REPORT re-read)
    rl = V.jload(V.RES / 'REPORTING_LOCK.json')
    for s in V.SEEDS:
        ed = V.RUNS / f'READERS_ENH_S{s}'
        e = rl['enhanced_readers'][f'S{s}']
        rec = V.jload(ed / 'READERS_ENH_SEED.json')
        bad = [k for k, h in rec['model_files_sha256'].items() if V.sha_file(ed / f'MODEL__{k}.joblib') != h]
        xc[f'enhanced_readers_S{s}'] = {'AV_SCORES': V.sha_file(ed / 'AV_SCORES.npz') == e['av_scores_sha256'],
                                        'DEV_REPORT_SCORES': V.sha_file(ed / 'DEV_REPORT_SCORES.npz') == e['dev_report_scores_sha256'],
                                        'record': V.sha_file(ed / 'READERS_ENH_SEED.json') == e['record_sha256'],
                                        'n_model_files': len(rec['model_files_sha256']), 'n_model_files_lock': e['n_model_files'],
                                        'model_file_mismatch': bad}
    for n, h in rl['locks_sha256'].items():
        xc[f'lock_{n}'] = V.sha_file(V.RES / n) == h
    out['cross_checks_against_source_records'] = xc

    # ---------------------------------------------------------------- 1c. frozen candidate manifest
    man = V.jload(V.RES / 'FROZEN_CANDIDATE_MANIFEST.json')
    mc = {}
    for s in V.SEEDS:
        ps = man['per_seed'][f'S{s}']
        rel_p = V.ADM / 'runs' / 'B' / CFG['FINALIST'] / f'S{s}' / 'RELEASE.npz'
        rel = load_npz(rel_p)
        rj = V.jload(rel_p.with_suffix('.json'))
        pkg = V.ADM / 'deploy' / f'FINALIST_S{s}'
        H = load_npz(pkg / 'H.npz')
        lv_hex = np.array([[float.fromhex(x) for x in row] for row in ps['per_level_income_probability_hex']])
        lv_rep = np.array([[float(x) for x in row] for row in ps['per_level_income_probability']])
        level_bytes = {lv_hex[j].tobytes(): j for j in range(6)}
        per_role = {}
        for r in V.ROLES6:
            inc = np.ascontiguousarray(rel[f'{r}__income'])
            lev = np.array([level_bytes.get(row.tobytes(), -1) for row in inc])
            per_role[r] = {'rows': int(len(inc)), 'rows_not_equal_to_a_level_vector': int((lev < 0).sum()),
                           'levels_observed': sorted(int(x) for x in np.unique(lev)),
                           'level_counts_equal_release_record_coarse_cell_counts': np.bincount(lev[lev >= 0], minlength=6).tolist() == rj['coarse_cell_counts'][r],
                           'wire_sha256_equals_manifest': {t: V.wire_hash(rel[f'{r}__{t}']) == ps['wire_sha256_by_role'][f'{r}__{t}'] for t in V.TASKS}}
        # per-level vectors recomputed from H.npz (own arithmetic, 6-row batch): softmax(-s/2, s/2), s = ((Z - mu)/sd) w + b
        Z = np.eye(6)[:, 1:]
        sc = ((Z - H['mu']) / H['sd']) @ H['w'] + float(H['b'][0])
        lg = np.column_stack((-sc / 2, sc / 2))
        lg = lg - lg.max(1, keepdims=True)
        e = np.exp(lg)
        own = e / e.sum(1, keepdims=True)
        mc[f'S{s}'] = {
            'release_sha256_manifest_equals_FINALIST_LOCK': ps['release_sha256'] == fl['release_sha256'][f'B/{CFG["FINALIST"]}'][f'S{s}'],
            'release_file_sha256_equals_manifest': V.sha_file(rel_p) == ps['release_sha256'],
            'package_files_equal_manifest': {f: V.sha_file(pkg / f) == h for f, h in ps['package_files_sha256'].items()},
            'package_json_equals_manifest': V.sha_file(pkg / 'PACKAGE.json') == ps['package_json_sha256'],
            'thresholds_hex_equal_package_H': [float.fromhex(x) for x in ps['thresholds_hex']] == [float(x) for x in H['thresholds']],
            'calibration_h_equals_package_H': all(np.array_equal(np.array([float(x) for x in ps['calibration_h'][k]]), np.asarray(H[k], dtype=np.float64).reshape(-1))
                                                  for k in ('mu', 'sd', 'w', 'b')),
            'per_level_hex_equals_repr_strings': bool(np.array_equal(lv_hex, lv_rep)),
            'per_level_decisions': lv_hex.argmax(1).tolist(), 'per_level_decisions_equal_manifest': lv_hex.argmax(1).tolist() == ps['per_level_decision'],
            'released_income_rows_all_equal_a_per_level_vector_bitwise': all(v['rows_not_equal_to_a_level_vector'] == 0 for v in per_role.values()),
            'all_six_levels_observed_every_role': all(v['levels_observed'] == list(range(6)) for v in per_role.values()),
            'per_role': per_role,
            'own_6row_recompute_from_H': {'bitwise_equal_by_level': [bool(np.array_equal(own[j], lv_hex[j])) for j in range(6)],
                                          'max_abs_diff_by_level': [float(np.max(np.abs(own[j] - lv_hex[j]))) for j in range(6)]}}

    # ---------------------------------------------------------------- 1d. occupation wire identity (FINALIST = H_LJ = B2TWIN)
    occ = {}
    for s in V.SEEDS:
        bank, ids, _ = C.service_bank(s)
        man_w = man['bank']['per_seed'][f'S{s}']['wire_sha256']
        bank_roles = ('MODEL_SELECT',) + V.ROLES3
        occ[f'S{s}|bank_roles'] = {r: bool(np.array_equal(bank['FINALIST'][r]['occupation'], bank['H_LJ'][r]['occupation'])
                                           and np.array_equal(bank['FINALIST'][r]['occupation'], bank['B2TWIN'][r]['occupation'])
                                           and bank['FINALIST'][r]['occupation'].tobytes() == bank['H_LJ'][r]['occupation'].tobytes())
                                   for r in bank_roles}
        occ[f'S{s}|bank_wire_hashes_equal_manifest'] = all(V.wire_hash(bank[svc][r][t]) == man_w[f'{svc}|{t}'][r]
                                                           for svc in V.BANK for t in V.TASKS for r in bank_roles)
        uniq = {t: len({tuple(V.wire_hash(bank[svc][r][t]) for r in V.ROLES3) for svc in V.BANK}) for t in V.TASKS}
        occ[f'S{s}|unique_single_recipient_wires'] = uniq
        rel_f = load_npz(V.ADM / 'runs' / 'B' / CFG['FINALIST'] / f'S{s}' / 'RELEASE.npz')
        rel_b = load_npz(V.ADM / 'runs' / 'B' / CFG['B2TWIN'] / f'S{s}' / 'RELEASE.npz')
        occ[f'S{s}|release_FINALIST_vs_B2TWIN_all_roles'] = {r: bool(np.array_equal(rel_f[f'{r}__occupation'], rel_b[f'{r}__occupation'])) for r in V.ROLES6}
        occ[f'S{s}|release_vs_bank_FINALIST'] = {r: {t: bool(np.array_equal(rel_f[f'{r}__{t}'], bank['FINALIST'][r][t])) for t in V.TASKS} for r in bank_roles}
        # own recomputation from the encoded inputs: JOINT_TREE E20 occupation (= H_LJ's occupation by definition) and the
        # FINALIST income level, level = #thresholds strictly below d, d = 0.5 LOCAL_TREE + 0.5 JOINT_TREE E20 income raw score
        te_o, te_i = teacher(s, 'occupation'), teacher(s, 'income')
        thJ = load_npz(V.ADM / 'admitted' / 'stage_b' / f'S{s}' / 'JOINT_TREE__E20.npz')
        thL = load_npz(V.ADM / 'admitted' / 'stage_b' / f'S{s}' / 'LOCAL_TREE__E20.npz')
        H = load_npz(V.ADM / 'deploy' / f'FINALIST_S{s}' / 'H.npz')
        lv = np.array([[float.fromhex(x) for x in row] for row in man['per_seed'][f'S{s}']['per_level_income_probability_hex']])
        lvb = {lv[j].tobytes(): j for j in range(6)}
        rec = {}
        for r in V.ROLES6:
            F = C.load_role(r, ('features',))
            p_occ = softmax_occ(raw_scores(te_o, thJ['theta_occupation'], F['X'], F['feature_names']))
            d = (1 - 0.5) * raw_scores(te_i, thL['theta_income'], F['X'], F['feature_names']).reshape(-1) + \
                0.5 * raw_scores(te_i, thJ['theta_income'], F['X'], F['feature_names']).reshape(-1)
            lev_own = np.searchsorted(np.asarray(H['thresholds'], dtype=np.float64), d, side='left')
            inc = np.ascontiguousarray(rel_f[f'{r}__income'])
            lev_rel = np.array([lvb.get(row.tobytes(), -1) for row in inc])
            rec[r] = {'own_joint_tree_occupation_bitwise_equal_release': bool(np.array_equal(p_occ, rel_f[f'{r}__occupation'])),
                      'own_joint_tree_occupation_max_abs_diff': float(np.max(np.abs(p_occ - rel_f[f'{r}__occupation']))),
                      'own_decisions_equal': bool(np.array_equal(p_occ.argmax(1), rel_f[f'{r}__occupation'].argmax(1))),
                      'own_income_level_equals_released_level': int((lev_own == lev_rel).sum()), 'rows': int(len(lev_rel)),
                      'own_income_level_differs': int((lev_own != lev_rel).sum())}
        occ[f'S{s}|own_recompute_all_roles'] = rec
    out['frozen_candidate_manifest'] = mc
    out['occupation_wire_identity'] = occ
    V.jdump(V.VPRIV / 'check1_identity.json', out)
    print('admission', {k: (len(v) if isinstance(v, list) else v) for k, v in out['admission_rehash'].items() if k != 'by_group'})
    print('manifest', {s: {k: v for k, v in m.items() if k not in ('per_role', 'package_files_equal_manifest')} for s, m in mc.items()})
    print('occ', {k: v for k, v in occ.items() if 'own' not in k})
    print('own', {k: {r: (x['own_joint_tree_occupation_bitwise_equal_release'], x['own_income_level_differs']) for r, x in v.items()} for k, v in occ.items() if 'own' in k})


if __name__ == '__main__':
    main()
