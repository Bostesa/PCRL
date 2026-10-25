"""Diagnostic for the CHECK6a D_nested mismatch (kept beside the failed output): for the same own samples as check6a part (e),
compare the study's float LOOKUP scores with the verifier's exact-rational scores. Hypothesis: distinct wires whose shrunk scores
(s1 + a rbar) / (n + a) are EXACTLY equal as rationals get float values that differ in the last bit (or the reverse), so the
float reader orders pairs the exact idealization ties. Test: (1) count exact-tie groups split in float and float ties that are not
exact ties; (2) recompute the pooled AUC and the fold population AUCs by brute force on the study's FLOAT scores - those must
equal the study's values to 1e-12 if the hypothesis holds."""
from fractions import Fraction as Fr

import numpy as np

import vlib as V

V.study_imports()
from edg import laws as L             # noqa: E402
from edg import estimators as E       # noqa: E402
from edg import source_prc as PRC     # noqa: E402

T0 = V.utc_now()
VSEED = 202610251840                  # the check6a seed root: identical samples
rec = {'diagnostic': 'D_nested exact-vs-float tie structure', 'views': []}


def exact_table(train_w, train_y, K, a):
    rbar = Fr(int(np.sum(train_y)), len(train_y))
    n = np.bincount(train_w, minlength=K)
    n1 = np.bincount(train_w, weights=train_y, minlength=K).astype(np.int64)
    return [(int(n1[w]) + a * rbar) / (int(n[w]) + a) if n[w] > 0 else rbar for w in range(K)]


worst = {'value': 0.0, 'pfm': 0.0}
max_float_route = {'value': 0.0, 'pop_fold_mean': 0.0}
tot_split, tot_merge = 0, 0
for fi, fam in enumerate(L.FAMILIES):
    for ip, pi in enumerate(L.PIS):
        for io, occ in enumerate(L.OCCS):
            law = L.build(fam, pi, occ)
            for iv, (name, Vw) in enumerate(law['views'].items()):
                rng = np.random.default_rng(np.random.SeedSequence(VSEED, spawn_key=(4, fi, ip, io, iv)))
                n = 2000
                c, y = V.draw_people(law['p'], law['q'], n, rng)
                ids, K = Vw['inv'][c], Vw['K']
                folds = rng.permutation(n) % 5
                dn = E.d_nested(ids, y, folds, K, Vw['m1'], Vw['m0'])
                wid, K2, m1, m0 = V.study_law_exact_view(law, name)
                own2st = np.full(K2, -1)
                own2st[wid] = Vw['inv']
                m1x, m0x = [None] * K, [None] * K
                for w in range(K):
                    m1x[own2st[w]], m0x[own2st[w]] = m1[w], m0[w]
                look_f = np.zeros(n)
                keys_ex = [None] * n
                split = merge = 0
                pops_fl, pops_ex, shares = [], [], []
                for f in range(5):
                    tr, te = folds != f, folds == f
                    s_fl = PRC.lookup_scores(ids[tr], y[tr], np.arange(K), a=1.0)
                    s_ex = exact_table(ids[tr], y[tr], K, Fr(1))
                    look_f[te] = PRC.lookup_scores(ids[tr], y[tr], ids[te], a=1.0)
                    for i in np.flatnonzero(te):
                        keys_ex[i] = s_ex[ids[i]]
                    # tie structure within the fold's table (over wires)
                    grp_ex, grp_fl = {}, {}
                    for w in range(K):
                        grp_ex.setdefault(s_ex[w], set()).add(float(s_fl[w]))
                        grp_fl.setdefault(float(s_fl[w]), set()).add(s_ex[w])
                    split += sum(len(v) - 1 for v in grp_ex.values())        # exact ties split into several floats
                    merge += sum(len(v) - 1 for v in grp_fl.values())        # float ties joining distinct exact values
                    pops_fl.append(V.pop_auc_brute([float(x) for x in s_fl], m1x, m0x))
                    pops_ex.append(V.pop_auc_brute(s_ex, m1x, m0x))
                    shares.append(Fr(int(te.sum()), n))
                tot_split += split
                tot_merge += merge
                v_float = V.sample_auc_brute(y, score=look_f)
                v_exact = V.sample_auc_brute(y, exact_keys=keys_ex)
                pfm_fl = sum(s_ * p_ for s_, p_ in zip(shares, pops_fl))
                pfm_ex = sum(s_ * p_ for s_, p_ in zip(shares, pops_ex))
                r = {'view': f'{fam}|{pi}|{occ}|{name}', 'K': K, 'exact_ties_split_in_float': split, 'float_ties_of_distinct_exact': merge,
                     'study_value': dn['value'], 'bruteforce_on_float_scores_minus_study': float(v_float) - dn['value'],
                     'exact_idealization_minus_study': float(v_exact) - dn['value'],
                     'pop_fold_mean_bruteforce_on_float_minus_study': float(pfm_fl) - dn['pop_fold_mean'],
                     'pop_fold_mean_exact_idealization_minus_study': float(pfm_ex) - dn['pop_fold_mean']}
                max_float_route['value'] = max(max_float_route['value'], abs(r['bruteforce_on_float_scores_minus_study']))
                max_float_route['pop_fold_mean'] = max(max_float_route['pop_fold_mean'],
                                                       abs(r['pop_fold_mean_bruteforce_on_float_minus_study']))
                worst['value'] = max(worst['value'], abs(r['exact_idealization_minus_study']))
                worst['pfm'] = max(worst['pfm'], abs(r['pop_fold_mean_exact_idealization_minus_study']))
                rec['views'].append(r)
                if abs(r['exact_idealization_minus_study']) > 1e-12 or split or merge:
                    print(r)
rec['max_abs_bruteforce_on_float_scores_minus_study'] = max_float_route
rec['max_abs_exact_idealization_minus_study'] = worst
rec['total_exact_ties_split_in_float'] = tot_split
rec['total_float_ties_of_distinct_exact_values'] = tot_merge
rec['views_with_any_exact_vs_float_difference'] = sum(1 for r in rec['views'] if abs(r['exact_idealization_minus_study']) > 1e-12
                                                       or abs(r['pop_fold_mean_exact_idealization_minus_study']) > 1e-12)
print('float-route max', max_float_route, 'exact-idealization max', worst, 'split', tot_split, 'merge', tot_merge)
V.write_out('DIAG6a_dnested_ties.json', rec, started_utc=T0)
