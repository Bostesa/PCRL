"""Independent verifier, checks 1 and 2 (teacher and cell identities; matched capacity).

Own code only (numpy); no scm_bench / acs_replication import. Reads private arrays under
.private/pcrl_scm_engineering_benchmark_v1/ and writes aggregates only to check1_2_identities_capacity.json.

Check 1: numpy oblivious-tree evaluation of the fold teachers (route float32 features > float32 borders, bit i = depth level i,
sum leaf values, scale + bias; income = binary logit d -> softmax(-d/2, d/2); occupation = softmax of centred raw scores).
  (a) MODEL_FIT out-of-fold probabilities (each person by the teacher of their fold) vs PARTITION.npz oof_* (bitwise).
  (b) runtime fold-average on MODEL_SELECT, ATTACK_FIT, ATTACK_VALIDATION, DEV_REPORT (and MODEL_FIT for diagnostics);
      KL-nearest cell vs cent1/cent2 (log clipped at 1e-300, argmax of sum_t sum_k t_k log mu_ck, ties lowest index);
      released vectors via ARM_TABLES (v1[group_of_initial[c]], v2[cell_of_initial[c]]) vs ARM_OUTPUTS.npz (bitwise), all arms.
Check 2: per arm, final cells, occupied cells per role, distinct transmitted vectors per role, income groups in use;
  matched capacity of TWIN/IBSI/SHUF vs SCM at the same beta; comparison with FIT_SEED.json capacity records.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('<repository>/.claude/worktrees/pcrl-scm-engineering-benchmark-v1')
PRIV = ROOT / '.private' / 'pcrl_scm_engineering_benchmark_v1'
OUT = ROOT / 'results' / 'pcrl_scm_engineering_benchmark_v1' / 'roles' / 'verifier'
ROLES = ('MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
TINY = 1e-300


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_features(role, pub):
    p = PRIV / 'source' / 'exports' / role / 'features.npz'
    assert sha_file(p) == pub[role]['features'], f'{role} features hash differs from public extraction summary'
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


class Teacher:
    def __init__(self, d):
        self.task = d['task']
        self.depth = int(d['depth'])
        self.feat = np.asarray(d['split_features'], dtype=np.int64)
        self.bord = np.asarray(d['split_borders'], dtype=np.float32)
        self.leaf = np.asarray(d['leaf_values'], dtype=np.float64)
        self.scale = float(d['scale'])
        self.bias = np.asarray(d['bias'], dtype=np.float64).reshape(-1)
        self.names = tuple(d['feature_names'])
        self.T = self.feat.shape[0]

    def raw(self, X, chunk=2000):
        X32 = np.ascontiguousarray(X, dtype=np.float32)
        out = np.empty((len(X32), self.leaf.shape[2]))
        tix = np.arange(self.T)[None, :]
        for s in range(0, len(X32), chunk):
            xb = X32[s:s + chunk]
            idx = np.zeros((len(xb), self.T), dtype=np.int64)
            for lvl in range(self.depth):
                bit = xb[:, self.feat[:, lvl]] > self.bord[None, :, lvl]
                idx += bit.astype(np.int64) * (1 << lvl)
            out[s:s + chunk] = self.scale * self.leaf[tix, idx].sum(1) + self.bias
        return out

    def probs(self, X):
        r = self.raw(X)
        if self.task == 'income':
            d = r[:, 0]
            z = np.stack((-d / 2.0, d / 2.0), axis=1)
        else:
            z = r - r.mean(axis=1, keepdims=True)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


def kl_nearest(T1, T2, c1, c2):
    L1 = np.log(np.maximum(c1, TINY)); L2 = np.log(np.maximum(c2, TINY))
    out = np.empty(len(T1), dtype=np.int64); gap = np.empty(len(T1))
    for s in range(0, len(T1), 4096):
        sc = T1[s:s + 4096] @ L1.T + T2[s:s + 4096] @ L2.T
        out[s:s + 4096] = sc.argmax(1)   # numpy argmax returns the first (lowest) index on ties
        part = np.sort(sc, axis=1)
        gap[s:s + 4096] = part[:, -1] - part[:, -2]
    return out, gap


def kl_nearest_elementwise(T1, T2, c1, c2):
    """Alternative evaluation order (explicit sum over classes) to measure tie/rounding sensitivity."""
    L1 = np.log(np.maximum(c1, TINY)); L2 = np.log(np.maximum(c2, TINY))
    out = np.empty(len(T1), dtype=np.int64)
    for s in range(0, len(T1), 1024):
        a = (T1[s:s + 1024, None, :] * L1[None]).sum(2) + (T2[s:s + 1024, None, :] * L2[None]).sum(2)
        out[s:s + 1024] = a.argmax(1)
    return out


def ndistinct(a):
    a = np.ascontiguousarray(a)
    return int(len(np.unique(a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))))))


def main():
    pub = json.loads((ROOT / 'results' / 'pcrl_acs_protected_tree_replication_v1' / 'EXTRACTION_SUMMARY.json').read_text())['export_file_sha256']
    feats = {r: load_features(r, pub) for r in ROLES + ('MODEL_FIT',)}
    res = {'check1': {}, 'check2': {}}
    for s in (0, 1):
        fit = PRIV / 'runs' / f'FIT_{s}'
        seedrec = json.loads((fit / 'FIT_SEED.json').read_text())
        # file hashes recorded in FIT_SEED vs on disk
        fh = {k: (sha_file(fit / k) == v) for k, v in seedrec['files_sha256'].items()}
        part = dict(np.load(fit / 'PARTITION.npz'))
        tabs = dict(np.load(fit / 'ARM_TABLES.npz'))
        outs = np.load(fit / 'ARM_OUTPUTS.npz')
        arms = sorted({k.split('__')[0] for k in tabs})
        names_ok = True
        teachers = {}
        for f in range(5):
            for task in ('income', 'occupation'):
                d = json.loads((fit / 'TEACHERS' / f'fold{f}_{task}.json').read_text())
                t = Teacher(d)
                names_ok &= (t.names == tuple(map(str, feats['MODEL_FIT']['feature_names'])))
                teachers[(f, task)] = t
        # row alignment
        align = {r: bool(np.array_equal(outs[f'row_id__{r}'], feats[r]['row_id']) and np.array_equal(outs[f'unit__{r}'], feats[r]['unit']))
                 for r in ROLES}
        for r in ROLES:
            names_ok &= tuple(map(str, feats[r]['feature_names'])) == tuple(map(str, feats['MODEL_FIT']['feature_names']))
        # (a) OOF on MODEL_FIT
        fold = part['fold']
        XF = feats['MODEL_FIT']['X']
        oof = {'income': np.zeros((len(XF), 2)), 'occupation': np.zeros((len(XF), 5))}
        runtime = {r: {'income': 0.0, 'occupation': 0.0} for r in ROLES + ('MODEL_FIT',)}
        runtime_mean = {r: {'income': [], 'occupation': []} for r in ROLES + ('MODEL_FIT',)}
        for f in range(5):
            te = fold == f
            for task in ('income', 'occupation'):
                t = teachers[(f, task)]
                for r in runtime:
                    Xr = XF if r == 'MODEL_FIT' else feats[r]['X']
                    p = t.probs(Xr)
                    if r == 'MODEL_FIT':
                        oof[task][te] = p[te]
                    runtime[r][task] = runtime[r][task] + p / 5.0       # registered accumulation order
                    runtime_mean[r][task].append(p)
        c1a = {'oof_income_bitwise': bool(np.array_equal(oof['income'], part['oof_income'])),
               'oof_occupation_bitwise': bool(np.array_equal(oof['occupation'], part['oof_occupation'])),
               'oof_income_max_abs_diff': float(np.abs(oof['income'] - part['oof_income']).max()),
               'oof_occupation_max_abs_diff': float(np.abs(oof['occupation'] - part['oof_occupation']).max()),
               'feature_names_identical_all_roles': bool(names_ok), 'row_alignment': align,
               'fit_seed_file_hashes_match': fh}
        # centroids: pooled OOF mean per fine cell (protocol section 3/4) vs PARTITION cent1/cent2
        fa = part['fine_assign_MODEL_FIT']; k = part['cent1'].shape[0]
        n = np.bincount(fa, minlength=k).astype(float)
        S1 = np.zeros((k, 2)); np.add.at(S1, fa, part['oof_income']); S2 = np.zeros((k, 5)); np.add.at(S2, fa, part['oof_occupation'])
        c1a['cent_equals_pooled_oof_mean_max_abs'] = float(max(np.abs(S1 / n[:, None] - part['cent1']).max(), np.abs(S2 / n[:, None] - part['cent2']).max()))
        c1a['occupied_fine_cells'] = int((n > 0).sum())
        # (b) runtime cells and releases
        cells, gaps, flips_alt, flips_mean = {}, {}, {}, {}
        for r in runtime:
            T1, T2 = runtime[r]['income'], runtime[r]['occupation']
            cells[r], g = kl_nearest(T1, T2, part['cent1'], part['cent2'])
            gaps[r] = {'min_top2_gap': float(g.min()), 'n_gap_below_1e-12': int((g < 1e-12).sum())}
            flips_alt[r] = int((kl_nearest_elementwise(T1, T2, part['cent1'], part['cent2']) != cells[r]).sum())
            m1 = np.mean(np.stack(runtime_mean[r]['income']), 0); m2 = np.mean(np.stack(runtime_mean[r]['occupation']), 0)
            flips_mean[r] = int((kl_nearest(m1, m2, part['cent1'], part['cent2'])[0] != cells[r]).sum())
        diag = {'share_MODEL_FIT_runtime_cell_equals_oof_cell': float(np.mean(cells['MODEL_FIT'] == fa)),
                'tv_MODEL_FIT_oof_vs_MODEL_SELECT_runtime': float(0.5 * np.abs(np.bincount(fa, minlength=k) / len(fa)
                                                                              - np.bincount(cells['MODEL_SELECT'], minlength=k) / len(cells['MODEL_SELECT'])).sum())}
        diag_rec = {kk: seedrec['diagnostics'][kk] for kk in ('share_MODEL_FIT_runtime_cell_equals_oof_cell',)}
        diag_rec['tv_MODEL_FIT_oof_vs_MODEL_SELECT_runtime'] = seedrec['diagnostics']['tv_distance_MODEL_FIT_oof_vs_MODEL_SELECT_runtime_occupancy']
        rel_eq = {}
        cap = {}
        for arm in arms:
            v1, v2 = tabs[f'{arm}__v1'], tabs[f'{arm}__v2']
            coi, goi = tabs[f'{arm}__cell_of_initial'], tabs[f'{arm}__group_of_initial']
            ok = True
            capa = {'final_cells': int(len(np.unique(coi))), 'nominal_cells_record': seedrec['capacity'][arm]['nominal_cells'],
                    'income_groups_used': int(len(np.unique(goi))), 'nonzero_v2_rows': int((v2.sum(1) > 0).sum()),
                    'nonzero_v1_rows': int((v1.sum(1) > 0).sum())}
            for r in ROLES:
                c = cells[r]
                inc = v1[goi[c]]; occ = v2[coi[c]]
                e1 = np.array_equal(inc.view(np.uint64), outs[f'{arm}__{r}__income'].view(np.uint64))
                e2 = np.array_equal(occ.view(np.uint64), outs[f'{arm}__{r}__occupation'].view(np.uint64))
                ok &= bool(e1 and e2)
                if not (e1 and e2):
                    rel_eq.setdefault('mismatch_rows', {})[f'{arm}|{r}'] = int((~np.all(inc == outs[f'{arm}__{r}__income'], 1) | ~np.all(occ == outs[f'{arm}__{r}__occupation'], 1)).sum())
                capa[r] = {'occupied_cells': int(len(np.unique(coi[c]))), 'distinct_income_vectors': ndistinct(inc),
                           'distinct_occupation_vectors': ndistinct(occ), 'distinct_pair_vectors': ndistinct(np.column_stack((inc, occ))),
                           'income_groups_in_use': int(len(np.unique(goi[c])))}
                recd = seedrec['capacity'][arm][r]
                capa[r]['matches_FIT_SEED_record'] = all(capa[r][kk] == recd[kk] for kk in recd)
            rel_eq[arm] = bool(ok)
            capa['record_income_groups_used'] = seedrec['capacity'][arm]['income_groups_used']
            capa['record_final_cells_arms'] = seedrec['arms'][arm]['final_cells']
            capa['stop_reason'] = seedrec['arms'][arm].get('stop_reason')
            capa['target'] = seedrec['arms'][arm].get('target')
            cap[arm] = capa
        res['check1'][f'S{s}'] = {'oof_and_alignment': c1a, 'runtime_cell_sensitivity': {'top2_gap': gaps,
                                  'flips_vs_elementwise_score_order': flips_alt, 'flips_vs_np_mean_fold_average': flips_mean},
                                  'mismatch_diagnostics_recomputed': diag, 'mismatch_diagnostics_recorded': diag_rec,
                                  'released_vectors_bitwise_equal_ARM_OUTPUTS': rel_eq,
                                  'all_31_arms_bitwise': bool(all(v for kk, v in rel_eq.items() if kk != 'mismatch_rows')),
                                  'n_arms': len(arms)}
        # matched capacity
        mc = {}
        for b in ('0.5', '1.0', '1.5'):
            K = cap[f'SCM_b{b}']['final_cells']
            others = [a for a in arms if (a.startswith(f'TWIN_b{b}') or a.startswith(f'IBSI_b{b}_') or a.startswith(f'SHUF_b{b}_'))]
            mc[b] = {'SCM_K': K, 'equal': {a: cap[a]['final_cells'] == K for a in others},
                     'counts': {a: cap[a]['final_cells'] for a in others}}
        res['check2'][f'S{s}'] = {'capacity': cap, 'matched': mc,
                                  'all_capacity_records_match': bool(all(cap[a][r]['matches_FIT_SEED_record'] for a in arms for r in ROLES)
                                                                     and all(cap[a]['final_cells'] == cap[a]['nominal_cells_record'] == cap[a]['record_final_cells_arms'] for a in arms)
                                                                     and all(cap[a]['income_groups_used'] == cap[a]['record_income_groups_used'] for a in arms))}
        print(f'seed {s} done', file=sys.stderr)
    (OUT / 'check1_2_identities_capacity.json').write_text(json.dumps(res, indent=1, sort_keys=True))
    s1 = {s: {'oof_bitwise': (res['check1'][s]['oof_and_alignment']['oof_income_bitwise'], res['check1'][s]['oof_and_alignment']['oof_occupation_bitwise']),
              'all_arms_bitwise': res['check1'][s]['all_31_arms_bitwise']} for s in res['check1']}
    print(json.dumps(s1))


if __name__ == '__main__':
    main()
