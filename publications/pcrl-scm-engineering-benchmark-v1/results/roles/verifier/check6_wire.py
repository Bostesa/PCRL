"""Independent verifier, Check 6: the transmitted wire and routing.

(a) Bank = fit outputs. SERVICE_BANK_S{s}.npz wires for SCM/TWIN/IBSI/SHUF/C0 are byte-equal to the nominated arms in
    runs/FIT_{s}/ARM_OUTPUTS.npz on all four roles; NEURAL_RAW/H_LJ/U0/Ucal are byte-equal to source FROZEN_BANK_S{s}.npz;
    row ids aligned with the exports; the bank holds nothing else; file hashes match SERVICE_BANK_MANIFEST, REPORTING_LOCK,
    SELECTION_LOCK, TRAINING_COMPLETE_LOCK and the frozen-hybrids manifest; per-wire hashes (own code: sha256 of float64
    bytes then int64 argmax bytes) match the manifest.
(b) Bit-identical within a final cell. From ARM_TABLES: the income group is a function of the final cell; every released
    income / occupation row (all four roles) is byte-equal to a used table row; every released pair is realisable by one
    initial cell; distinct released vectors <= final cells / groups. On DEV_REPORT, with this script's OWN routing from raw
    inputs (own encoder, own oblivious-tree evaluator, own KL assignment), every person in the same final cell receives
    byte-identical vectors and the result equals the bank wire.
(c) Deployment packages (deploy/SCM_S{s}, TWIN_S{s}) run as black boxes through their package interface
    (scm_bench.deploy_scm.release; the package kind 'SCM' has no other runner) on DEV_REPORT raw inputs reproduce the bank's
    attacked DEV_REPORT wires bitwise; refusal behaviour is probed for SEX, reordered columns and hidden-output requests,
    both at the shared CSV reader and at the package API. The seed-0 NEURAL_RAW reference package is re-run too.
Writes aggregates only (check6_wire.json).
"""
import csv
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

W = Path('<repository>/.claude/worktrees/pcrl-scm-engineering-benchmark-v1')
P = W / '.private' / 'pcrl_scm_engineering_benchmark_v1'
R = W / 'results' / 'pcrl_scm_engineering_benchmark_v1'
OUT = R / 'roles' / 'verifier' / 'check6_wire.json'
SCRATCH = Path('<session scratch>')
ROLES4 = ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT', 'MODEL_SELECT')
TASKS = ('income', 'occupation')
FITTED = ('C0', 'SCM', 'TWIN', 'IBSI', 'SHUF')
FROZEN = ('NEURAL_RAW', 'H_LJ', 'U0', 'Ucal')
PERMITTED = ['AGEP', 'WKHP', 'COW', 'SCHL', 'MAR', 'POBP']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def wire_hash(p):
    p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
    h = hashlib.sha256(p.tobytes()); h.update(np.ascontiguousarray(p.argmax(1).astype(np.int64)).tobytes())
    return h.hexdigest()


def same_bytes(a, b):
    return bool(a.dtype == b.dtype and a.shape == b.shape and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes())


def npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def row_bytes(a):
    a = np.ascontiguousarray(a, dtype=np.float64)
    return [r.tobytes() for r in a]


# ---------------- own runtime map (encoder, oblivious trees, KL routing) ----------------
def own_encode(raw, enc):
    names = list(enc['feature_names']); col = {n: i for i, n in enumerate(names)}
    n = len(raw['AGEP']); X = np.zeros((n, len(names)), dtype=np.float64)
    for c in ('AGEP', 'WKHP'):
        v = raw[c].astype(np.float64); st = enc['numeric'][c]
        X[:, col[c]] = (v - st['mean']) / st['sd']
    for c in ('COW', 'SCHL', 'MAR', 'POBP'):
        for i, val in enumerate(raw[c]):
            key = f'{c}={str(val).strip()}'
            X[i, col[key] if key in col else col[f'{c}=__OTHER__']] = 1.0
    return X.astype(np.float32), tuple(names)


def own_teacher_prob(payload, X32, names):
    if tuple(payload['feature_names']) != names:
        raise SystemExit('teacher feature order differs from encoder')
    f = np.asarray(payload['split_features'], dtype=np.int64); b = np.asarray(payload['split_borders'], dtype=np.float32)
    lv = np.asarray(payload['leaf_values'], dtype=np.float64); T, depth = f.shape
    leaf = np.zeros((len(X32), T), dtype=np.int64)
    for i in range(depth):
        leaf += (X32[:, f[:, i]] > b[None, :, i]).astype(np.int64) * (1 << i)
    raw = float(payload['scale']) * lv[np.arange(T)[None, :], leaf].sum(1) + np.asarray(payload['bias'], dtype=np.float64).reshape(-1)
    if payload['task'] == 'income':
        d = raw.reshape(-1); z = np.column_stack((-d / 2, d / 2))
    else:
        z = raw - raw.mean(1, keepdims=True)
    z = z - z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def own_kl_cell(t1, t2, cent1, cent2):
    s = t1 @ np.log(np.maximum(cent1, 1e-300)).T + t2 @ np.log(np.maximum(cent2, 1e-300)).T
    return s.argmax(1)


def main():
    t_start = time.process_time()
    res = {'schema': 'SCM1-verifier-check6-v1', 'a_bank': {}, 'b_within_cell': {}, 'c_deploy': {}}
    man = json.loads((R / 'SERVICE_BANK_MANIFEST.json').read_text())
    sel = json.loads((R / 'SELECTION_LOCK.json').read_text())
    rl = json.loads((R / 'REPORTING_LOCK.json').read_text())
    tl = json.loads((R / 'TRAINING_COMPLETE_LOCK.json').read_text())
    hm = json.loads((W / 'results' / 'pcrl_acs_frozen_hybrids_v1' / 'FROZEN_BANK_MANIFEST.json').read_text())
    pub = json.loads((W / 'results' / 'pcrl_acs_protected_tree_replication_v1' / 'EXTRACTION_SUMMARY.json').read_text())['export_file_sha256']
    mm = json.loads((W / 'results' / 'pcrl_acs_protected_tree_replication_v1' / 'MODEL_MANIFEST.json').read_text())
    arm_map = sel['bank_arm_map']
    res['arm_map'] = arm_map; res['manifest_arm_map_equals_selection_lock'] = man['arm_map'] == arm_map
    exp_ids = {}
    for r in ROLES4:
        fp = P / 'source' / 'exports' / r / 'features.npz'
        if sha(fp) != pub[r]['features']:
            raise SystemExit('export hash differs ' + r)
        with np.load(fp) as z:
            exp_ids[r] = (z['row_id'].copy(), z['unit'].copy())
    raw_p = P / 'source' / 'exports' / 'DEV_REPORT' / 'raw_inputs.npz'
    res['dev_report_raw_inputs_sha_matches_public'] = sha(raw_p) == pub['DEV_REPORT']['raw_inputs']
    raw = npz(raw_p)
    if not (np.array_equal(raw['row_id'], exp_ids['DEV_REPORT'][0]) and np.array_equal(raw['unit'], exp_ids['DEV_REPORT'][1])):
        raise SystemExit('raw inputs pairing differs')
    banks = {}
    all_ok = True
    for s in (0, 1):
        bp = P / 'bank' / f'SERVICE_BANK_S{s}.npz'; ap = P / 'runs' / f'FIT_{s}' / 'ARM_OUTPUTS.npz'; fp = P / 'source' / 'bank' / f'FROZEN_BANK_S{s}.npz'
        rec = {'bank_sha_vs_manifest': sha(bp) == man['bank_files_sha256'][f'S{s}'],
               'bank_sha_vs_reporting_lock': sha(bp) == rl['bank_files_sha256'][f'S{s}'],
               'arm_outputs_sha_vs_selection_lock': sha(ap) == sel['arm_output_files_sha256'][f'runs/FIT_{s}/ARM_OUTPUTS.npz'],
               'arm_outputs_sha_vs_training_lock': sha(ap) == tl['arm_outputs_sha256'][f'S{s}'],
               'frozen_bank_sha_vs_hybrids_manifest': sha(fp) == hm['bank_files_sha256'][f'S{s}']}
        bank = npz(bp); arms = npz(ap); froz = npz(fp); banks[s] = bank
        expected = {f'{svc}__{r}__{t}' for svc in list(arm_map) + list(FROZEN) for r in ROLES4 for t in TASKS} | \
                   {f'{k}__{r}' for k in ('row_id', 'unit') for r in ROLES4}
        rec['bank_keys_exactly_expected'] = set(bank) == expected
        rec['bank_extra_keys'] = sorted(set(bank) - expected); rec['bank_missing_keys'] = sorted(expected - set(bank))
        ids_ok = True
        for r in ROLES4:
            for src in (bank, arms, froz):
                ids_ok &= bool(np.array_equal(src[f'row_id__{r}'], exp_ids[r][0]) and np.array_equal(src[f'unit__{r}'], exp_ids[r][1]))
        rec['row_ids_aligned_bank_arms_frozen_exports'] = ids_ok
        fitted_eq = {svc: all(same_bytes(bank[f'{svc}__{r}__{t}'], arms[f'{arm_map[svc]}__{r}__{t}']) for r in ROLES4 for t in TASKS) for svc in arm_map}
        frozen_eq = {svc: all(same_bytes(bank[f'{svc}__{r}__{t}'], froz[f'{svc}__{r}__{t}']) for r in ROLES4 for t in TASKS) for svc in FROZEN}
        rec['fitted_bytes_equal_arm_outputs'] = fitted_eq; rec['frozen_bytes_equal_hybrids_bank'] = frozen_eq
        # per-wire hashes (3 attacked roles) vs manifest
        wm = man['per_seed'][f'S{s}']['wire_sha256']; n_w = 0; n_ok = 0
        for key, roles in wm.items():
            svc, t = key.split('|')
            for r, h in roles.items():
                n_w += 1; n_ok += int(wire_hash(bank[f'{svc}__{r}__{t}']) == h)
        rec['manifest_wire_hashes'] = {'n': n_w, 'match': n_ok, 'services_covered': sorted({k.split('|')[0] for k in wm})}
        # probability conventions
        conv = True
        for k, a in bank.items():
            if '__' in k and not k.startswith(('row_id', 'unit')):
                conv &= bool(a.dtype == np.float64 and np.isfinite(a).all() and (a >= 0).all() and np.allclose(a.sum(1), 1, atol=1e-9, rtol=0))
        rec['float64_probability_vectors'] = conv
        ok = all(v for k, v in rec.items() if isinstance(v, bool)) and all(fitted_eq.values()) and all(frozen_eq.values()) and n_ok == n_w
        rec['PASS'] = bool(ok); all_ok &= ok
        res['a_bank'][f'S{s}'] = rec
        print('a', s, json.dumps(rec), flush=True)

    # ---------------- (b) within-cell identity ----------------
    b_ok = True
    for s in (0, 1):
        tabs = npz(P / 'runs' / f'FIT_{s}' / 'ARM_TABLES.npz'); part = npz(P / 'runs' / f'FIT_{s}' / 'PARTITION.npz')
        bank = banks[s]; srec = {}
        for svc in FITTED:
            arm = arm_map[svc]
            v1, v2 = tabs[f'{arm}__v1'], tabs[f'{arm}__v2']; coi, goi = tabs[f'{arm}__cell_of_initial'], tabs[f'{arm}__group_of_initial']
            cells = np.unique(coi); groups = np.unique(goi)
            grp_fn = all(len(np.unique(goi[coi == c])) == 1 for c in cells)
            v2_rows = {}
            for c in cells:
                v2_rows.setdefault(v2[c].tobytes(), set()).add(int(c))
            v1_rows = {}
            for g in groups:
                v1_rows.setdefault(v1[g].tobytes(), set()).add(int(g))
            realisable = {(int(goi[i]), int(coi[i])) for i in range(len(coi))}
            arec = {'final_cells': int(len(cells)), 'income_groups_in_use': int(len(groups)), 'income_group_is_function_of_final_cell': bool(grp_fn),
                    'v2_used_rows_distinct': len(v2_rows) == len(cells), 'v1_used_rows_distinct': len(v1_rows) == len(groups), 'roles': {}}
            rok = grp_fn
            for r in ROLES4:
                inc, occ = bank[f'{svc}__{r}__income'], bank[f'{svc}__{r}__occupation']
                ib, ob = row_bytes(inc), row_bytes(occ)
                in_v1 = all(x in v1_rows for x in ib); in_v2 = all(x in v2_rows for x in ob)
                pairs_ok = True; pair_set = set()
                for x, y in zip(ib, ob):
                    pair_set.add((x, y))
                for x, y in pair_set:
                    gs, cs = v1_rows.get(x, set()), v2_rows.get(y, set())
                    pairs_ok &= any((g, c) in realisable for g in gs for c in cs)
                rr = {'every_income_row_is_a_v1_table_row': in_v1, 'every_occupation_row_is_a_v2_table_row': in_v2,
                      'every_pair_realisable_by_one_initial_cell': bool(pairs_ok), 'distinct_income': len(set(ib)),
                      'distinct_occupation': len(set(ob)), 'distinct_pair': len(pair_set),
                      'distinct_pair_le_final_cells': len(pair_set) <= len(cells), 'distinct_income_le_groups': len(set(ib)) <= len(groups)}
                rok &= in_v1 and in_v2 and pairs_ok and rr['distinct_pair_le_final_cells'] and rr['distinct_income_le_groups']
                arec['roles'][r] = rr
            arec['PASS'] = bool(rok); b_ok &= rok
            srec[svc] = arec
        res['b_within_cell'][f'S{s}'] = srec
        print('b', s, json.dumps({k: (v['final_cells'], v['PASS']) for k, v in srec.items()}), flush=True)

    # own routing from raw inputs on DEV_REPORT (SCM and TWIN, both seeds, plus C0/IBSI/SHUF since they share the teachers)
    enc_pkg = json.loads((P / 'deploy' / 'SCM_S0' / 'PACKAGE.json').read_text())['encoder']
    X32, names = own_encode(raw, enc_pkg)
    own = {}
    for s in (0, 1):
        tdir = P / 'runs' / f'FIT_{s}' / 'TEACHERS'
        tp = {}
        for t in TASKS:
            acc = None
            for f in range(5):
                p = own_teacher_prob(json.loads((tdir / f'fold{f}_{t}.json').read_text()), X32, names)
                acc = p if acc is None else acc + p
            tp[t] = acc / 5.0
        part = npz(P / 'runs' / f'FIT_{s}' / 'PARTITION.npz'); tabs = npz(P / 'runs' / f'FIT_{s}' / 'ARM_TABLES.npz')
        init = own_kl_cell(tp['income'], tp['occupation'], part['cent1'], part['cent2'])
        # alternative mean order (sequential p/5 accumulation) to expose any rounding-order sensitivity of the routing
        tp2 = {}
        for t in TASKS:
            acc = 0.0
            for f in range(5):
                acc = acc + own_teacher_prob(json.loads((tdir / f'fold{f}_{t}.json').read_text()), X32, names) / 5.0
            tp2[t] = acc
        init2 = own_kl_cell(tp2['income'], tp2['occupation'], part['cent1'], part['cent2'])
        orec = {'initial_cells_occupied_DEV_REPORT': int(len(np.unique(init))), 'routing_mean_order_disagreements': int((init != init2).sum())}
        for svc in FITTED:
            arm = arm_map[svc]
            c = tabs[f'{arm}__cell_of_initial'][init]; g = tabs[f'{arm}__group_of_initial'][init]
            inc = tabs[f'{arm}__v1'][g]; occ = tabs[f'{arm}__v2'][c]
            bi, bo = banks[s][f'{svc}__DEV_REPORT__income'], banks[s][f'{svc}__DEV_REPORT__occupation']
            mism = int((~(np.all(inc == bi, 1) & np.all(occ == bo, 1))).sum())
            within = True
            ib, ob = row_bytes(bi), row_bytes(bo)
            for cc in np.unique(c):
                idx = np.flatnonzero(c == cc)
                within &= len({ib[i] for i in idx}) == 1 and len({ob[i] for i in idx}) == 1
            orec[svc] = {'own_raw_route_equals_bank_bitwise': bool(same_bytes(inc, bi) and same_bytes(occ, bo)), 'rows_mismatched': mism,
                         'bank_bit_identical_within_each_final_cell': bool(within), 'final_cells_occupied_DEV_REPORT': int(len(np.unique(c)))}
            b_ok &= orec[svc]['own_raw_route_equals_bank_bitwise'] and within
        own[f'S{s}'] = orec
        print('own', s, json.dumps(orec), flush=True)
    res['b_within_cell']['own_raw_routing_DEV_REPORT'] = own
    res['b_within_cell']['PASS'] = bool(b_ok); all_ok &= b_ok

    # ---------------- (c) deployment packages ----------------
    import sys
    sys.path[:0] = [str(W), '<worktrees>/pcrl-acs-protected-tree-replication-v1/.private/acs_replication/pylib']
    from scm_bench import deploy_scm as DS          # the package interface (black box runner for kind 'SCM')
    from acs_replication import deploy as SRC        # shared raw CSV reader and the NEURAL/TREE runner
    dres = json.loads((R / 'DEPLOYMENT_RESULTS.json').read_text())
    rows = [{c: raw[c][i] for c in PERMITTED} for i in range(len(raw['AGEP']))]
    src_t = Path('<worktrees>/pcrl-acs-protected-tree-replication-v1/.private/acs_replication/deploy/T_TREE_S0/PACKAGE.json')
    src_enc_ok = sha(src_t) == mm['deployment_packages']['T_TREE_S0']['package_json_sha256']
    src_enc = json.loads(src_t.read_text())['encoder']
    c_ok = True
    for s in (0, 1):
        tabs = npz(P / 'runs' / f'FIT_{s}' / 'ARM_TABLES.npz'); part = npz(P / 'runs' / f'FIT_{s}' / 'PARTITION.npz')
        for label in ('SCM', 'TWIN'):
            d = P / 'deploy' / f'{label}_S{s}'; meta = json.loads((d / 'PACKAGE.json').read_text()); arm = arm_map[label]
            prec = {'arm': meta['arm'], 'arm_matches_selection': meta['arm'] == arm, 'kind': meta['kind'], 'seed_matches': meta['seed'] == s,
                    'package_json_sha_matches_DEPLOYMENT_RESULTS': sha(d / 'PACKAGE.json') == dres['packages'][f'{label}_S{s}']['package_json_sha256'],
                    'file_hashes_match_PACKAGE_json': all(sha(d / n) == h for n, h in meta['files'].items()),
                    'files_listed_exactly': sorted(meta['files']) == sorted(x.name for x in d.iterdir() if x.name != 'PACKAGE.json'),
                    'encoder_equals_source_T_TREE_S0_encoder': meta['encoder'] == src_enc and src_enc_ok,
                    'teachers_byte_equal_fit_teachers': all((d / f'fold{f}_{t}.json').read_bytes() == (P / 'runs' / f'FIT_{s}' / 'TEACHERS' / f'fold{f}_{t}.json').read_bytes()
                                                            for f in range(5) for t in TASKS)}
            cz = npz(d / 'CELLS.npz')
            prec['cells_equal_partition_and_arm_tables'] = bool(same_bytes(cz['cent1'], part['cent1']) and same_bytes(cz['cent2'], part['cent2'])
                                                               and all(same_bytes(cz[k], tabs[f'{arm}__{k}']) for k in ('v1', 'v2', 'cell_of_initial', 'group_of_initial'))
                                                               and set(cz) == {'cent1', 'cent2', 'v1', 'v2', 'cell_of_initial', 'group_of_initial'})
            t0 = time.process_time()
            out = DS.release(d, rows)
            prec['release_cpu_s'] = round(time.process_time() - t0, 2)
            prec['reproduces_bank_DEV_REPORT_bitwise'] = {t: bool(same_bytes(out[t]['probability'], banks[s][f'{label}__DEV_REPORT__{t}'])
                                                              and np.array_equal(out[t]['decision'], banks[s][f'{label}__DEV_REPORT__{t}'].argmax(1))
                                                              and out[t]['decision'].dtype == np.int64) for t in TASKS}
            prec['output_fields'] = {t: sorted(out[t]) for t in out}
            # refusals at the package API
            ref = {}
            for req in (('probability', 'cell_id'), ('group_id',), ('teacher_probabilities',), ('propensity',), ('pi_hat',), ('fine_cell',)):
                try:
                    DS.release(d, rows[:3], req); ref['hidden:' + ','.join(req)] = False
                except PermissionError:
                    ref['hidden:' + ','.join(req)] = True
            base3 = DS.release(d, rows[:3])
            for sexval in ('1', '2'):
                rows_sex = [dict(r, SEX=sexval) for r in rows[:3]]
                try:
                    o = DS.release(d, rows_sex)
                    ref[f'api_dict_rows_with_SEX={sexval}_refused'] = False
                    ref[f'api_dict_rows_with_SEX={sexval}_output_unchanged'] = all(same_bytes(o[t]['probability'], base3[t]['probability']) for t in TASKS)
                except (PermissionError, ValueError):
                    ref[f'api_dict_rows_with_SEX={sexval}_refused'] = True
            rows_rev = [{c: r[c] for c in reversed(PERMITTED)} for r in rows[:3]]
            try:
                o = DS.release(d, rows_rev); ref['api_dict_rows_reordered_refused'] = False
                ref['api_dict_rows_reordered_output_unchanged'] = all(same_bytes(o[t]['probability'], base3[t]['probability']) for t in TASKS)
            except (PermissionError, ValueError):
                ref['api_dict_rows_reordered_refused'] = True
            try:
                DS.release(d, [{c: r[c] for c in PERMITTED[:-1]} for r in rows[:3]]); ref['api_missing_POBP_refused'] = False
            except (KeyError, ValueError, PermissionError):
                ref['api_missing_POBP_refused'] = True
            try:
                SRC.release(d, rows[:3]); ref['source_cli_runner_accepts_SCM_kind'] = True
            except ValueError:
                ref['source_cli_runner_accepts_SCM_kind'] = False
            prec['refusals'] = ref
            ok = (prec['arm_matches_selection'] and prec['seed_matches'] and prec['package_json_sha_matches_DEPLOYMENT_RESULTS'] and prec['file_hashes_match_PACKAGE_json']
                  and prec['encoder_equals_source_T_TREE_S0_encoder'] and prec['teachers_byte_equal_fit_teachers'] and prec['cells_equal_partition_and_arm_tables']
                  and all(prec['reproduces_bank_DEV_REPORT_bitwise'].values()) and all(v for k, v in ref.items() if k.startswith('hidden:')))
            prec['PASS_core'] = bool(ok); c_ok &= ok
            res['c_deploy'][f'{label}_S{s}'] = prec
            print('c', label, s, json.dumps(prec), flush=True)
    # shared CSV reader refusals (synthetic rows only)
    tmp = Path(tempfile.mkdtemp(dir=SCRATCH))
    csvs = {'with_SEX': (PERMITTED + ['SEX'], ['40', '40', '1', '21', '1', '012', '1']),
            'reordered': (list(reversed(PERMITTED)), ['012', '1', '21', '1', '40', '40']),
            'swapped_two': (['WKHP', 'AGEP'] + PERMITTED[2:], ['40', '40', '1', '21', '1', '012']),
            'extra_non_forbidden': (PERMITTED + ['ST'], ['40', '40', '1', '21', '1', '012', '12']),
            'missing_one': (PERMITTED[:-1], ['40', '40', '1', '21', '1']),
            'lowercase_sex': (PERMITTED + ['sex'], ['40', '40', '1', '21', '1', '012', '1']),
            'valid': (PERMITTED, ['40', '40', '1', '21', '1', '012'])}
    cres = {}
    for name, (hdr, row) in csvs.items():
        with open(tmp / f'{name}.csv', 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(hdr); w.writerow(row)
        try:
            SRC.read_raw_csv(tmp / f'{name}.csv'); cres[name] = 'accepted'
        except PermissionError:
            cres[name] = 'refused_PermissionError'
        except ValueError:
            cres[name] = 'refused_ValueError'
    shutil.rmtree(tmp)
    res['c_deploy']['shared_csv_reader'] = cres
    csv_ok = cres['valid'] == 'accepted' and all(v.startswith('refused') for k, v in cres.items() if k != 'valid')
    res['c_deploy']['shared_csv_reader_PASS'] = bool(csv_ok); c_ok &= csv_ok
    # neural reference (seed 0 only)
    nd = P / 'deploy' / 'NEURAL_RAW_rho1.0_E2_S0'
    nrec = {'package_json_sha_matches_public_MODEL_MANIFEST': sha(nd / 'PACKAGE.json') == mm['deployment_packages']['NEURAL_RAW_rho1.0_E2_S0']['package_json_sha256']}
    try:
        t0 = time.process_time()
        no = SRC.release(nd, rows)
        nrec['release_cpu_s'] = round(time.process_time() - t0, 2)
        nrec['reproduces_bank_NEURAL_RAW_S0_DEV_REPORT_bitwise'] = {t: bool(same_bytes(np.asarray(no[t]['probability'], dtype=np.float64), banks[0][f'NEURAL_RAW__DEV_REPORT__{t}'])) for t in TASKS}
        nrec['max_abs_diff'] = {t: float(np.max(np.abs(no[t]['probability'] - banks[0][f'NEURAL_RAW__DEV_REPORT__{t}']))) for t in TASKS}
        nrec['output_dtype'] = {t: str(np.asarray(no[t]['probability']).dtype) for t in TASKS}
        try:
            SRC.release(nd, rows[:3], ('probability', 'representation')); nrec['hidden_output_refused'] = False
        except PermissionError:
            nrec['hidden_output_refused'] = True
    except Exception as e:  # recorded, not hidden
        nrec['error'] = repr(e)
    nrec['seed1_package'] = 'absent (source study packaged seed 0 only; DEPLOYMENT_RESULTS notes the coverage gap)'
    res['c_deploy']['NEURAL_RAW_rho1.0_E2_S0'] = nrec
    print('neural', json.dumps(nrec), flush=True)
    res['c_deploy']['PASS_core'] = bool(c_ok); all_ok &= c_ok
    res['PASS'] = bool(all_ok)
    res['cpu_s_total'] = round(time.process_time() - t_start, 2)
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True))
    print('PASS', all_ok, 'cpu', res['cpu_s_total'])


if __name__ == '__main__':
    main()
