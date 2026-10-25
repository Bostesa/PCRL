"""CHECK3 (part 1): identical emitted wires are merged, ties score 1/2, tied likelihood ratios are handled.

Method (own merging by exact bytes, own exact masses, brute-force exact AUC):
  3a  every view of all 16 study laws: own observable support vs the study's K vs the protocol table; the own partition of the
      internal cells equals the study's (a bijection between own wire ids and the study's inverse); the study's merged masses
      equal the own exact masses (relative 1e-12); emitted vectors are valid probability vectors.
  3b  COMP (4 study laws + 4 own exact laws): occupation cells 14 and 15 emit bitwise-identical vectors and no other occupation
      vectors coincide; supports 15 / 240; A on UNMERGED internal ids (256 cells, exact) differs from (exceeds) merged A (exact),
      and the merged cells (i, 14), (i, 15) carry different posteriors.
  3c  REF coarse atoms (4 study laws + 4 own laws): 8 posterior levels, each shared by atoms 2k and 2k + 1 (bitwise in the study's
      q); population A with EXACT ties (own exact masses: tied atoms have exactly equal posteriors) vs the study's float A.
  3d  tie structure of the study's merged-view float posterior V['post'] (the score used by B_same / B_indep) and float
      likelihood ratios V['lr'] (used by A): number of distinct float values vs the number of distinct EXACT posteriors of the
      same law. The sampling consequence is quantified in part 2 (check3b_known_score_ties.py)."""
from fractions import Fraction as Fr

import numpy as np

import vlib as V

V.study_imports()
from edg import laws as L             # noqa: E402

T0 = V.utc_now()
VSEED = 202610251750
EXPECT = {'IND': {'income': 16, 'occupation': 64, 'pair': 1024}, 'REF': {'L16': 16, 'L64': 64, 'L256': 256, 'L1024': 1024},
          'RED': {'income': 16, 'occupation': 64, 'pair': 256}, 'COMP': {'income': 16, 'occupation': 15, 'pair': 240}}
rec = {'check': 3, 'name': 'ties_and_merging_part1', '3a_supports_partition': {}, '3b_comp_merge': {}, '3c_ref_ties': {},
       '3d_float_tie_structure': {}}
fails = {'3a': [], '3b': [], '3c': []}

for fam in L.FAMILIES:
    for pi in L.PIS:
        for occ in L.OCCS:
            law = L.build(fam, pi, occ)
            key = f'{fam}|{pi}|{occ}'
            # ---------------- 3a
            ra = {}
            comps = [law['vectors']['levels'][k] for k in law['vectors']['levels']] if fam == 'REF' else \
                [law['vectors']['income'], law['vectors']['occupation']]
            valid = all(bool(np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all() and np.abs(a.sum(1) - 1).max() <= 1e-12)
                        for a in comps)
            for name, Vw in law['views'].items():
                wid, K, m1, m0 = V.study_law_exact_view(law, name)
                pairs = set(zip(wid.tolist(), Vw['inv'].tolist()))
                bij = len(pairs) == K == Vw['K']
                own2st = np.full(K, -1)
                own2st[wid] = Vw['inv']
                m1s, m0s = np.asarray(Vw['m1']), np.asarray(Vw['m0'])
                rel = max(max(abs(float(m1[w]) - m1s[own2st[w]]) / max(float(m1[w]), 1e-300) for w in range(K)),
                          max(abs(float(m0[w]) - m0s[own2st[w]]) / max(float(m0[w]), 1e-300) for w in range(K)))
                ok = bij and K == EXPECT[fam][name] and rel <= 1e-12
                ra[name] = {'K_own': K, 'K_study': Vw['K'], 'K_protocol': EXPECT[fam][name], 'partition_bijection': bij,
                            'max_rel_mass_diff': rel}
                if not ok:
                    fails['3a'].append(f'{key} {name}')
                # ---------------- 3d (tie structure of the float posterior / LR vs the exact posterior)
                post_ex = V.posterior_exact(m1, m0)
                rec['3d_float_tie_structure'][f'{key}|{name}'] = {
                    'distinct_exact_posteriors': len(set(post_ex)), 'distinct_float_post_V_post': int(len(np.unique(Vw['post']))),
                    'distinct_float_lr_V_lr': int(len(np.unique(Vw['lr']))),
                    'max_abs_float_post_minus_exact': max(abs(float(Vw['post'][own2st[w]]) - float(post_ex[w])) for w in range(K))}
            ra['_vectors_valid_probability'] = valid
            if not valid:
                fails['3a'].append(f'{key} vectors')
            rec['3a_supports_partition'][key] = ra
            # ---------------- 3b COMP
            if fam == 'COMP':
                occv = law['vectors']['occupation']
                jj = np.tile(np.arange(16), 16)
                rows = {j: {occv[c].tobytes() for c in np.flatnonzero(jj == j)} for j in range(16)}
                same_1415 = all(len(rows[j]) == 1 for j in range(16)) and rows[14] == rows[15]
                distinct_j = len({next(iter(rows[j])) for j in range(16)})
                Vp = law['views']['pair']
                wid, K, m1, m0 = V.study_law_exact_view(law, 'pair')
                A_merged = V.pop_optimal_auc_exact(m1, m0)
                p, q = law['p'], law['q']
                m1i = [V.F(p[c]) * V.F(q[c]) for c in range(len(p))]
                m0i = [V.F(p[c]) * (1 - V.F(q[c])) for c in range(len(p))]
                A_internal = V.pop_optimal_auc_exact(m1i, m0i)
                ii = np.repeat(np.arange(16), 16)
                q_differs = all(q[i * 16 + 14] != q[i * 16 + 15] for i in range(16) if law['comp']['u'][i] != 0)
                rb = {'occ_cells_14_15_identical_bytes': bool(same_1415), 'distinct_occupation_vectors': distinct_j,
                      'K_occupation_own': V.study_law_exact_view(law, 'occupation')[1], 'K_pair_own': K,
                      'A_pair_merged_exact': A_merged, 'A_pair_unmerged_internal_exact': A_internal,
                      'unmerged_minus_merged': float(A_internal - A_merged),
                      'study_internal_id_auc_pair': law['internal_id_auc_pair'], 'study_A_pair': Vp['A'],
                      'study_vs_own_internal_diff': abs(law['internal_id_auc_pair'] - float(A_internal)),
                      'study_vs_own_merged_diff': abs(Vp['A'] - float(A_merged)),
                      'merged_cells_have_different_q': bool(q_differs)}
                ok = (same_1415 and distinct_j == 15 and rb['K_occupation_own'] == 15 and K == 240 and A_internal > A_merged
                      and rb['study_vs_own_internal_diff'] <= 1e-12 and rb['study_vs_own_merged_diff'] <= 1e-12 and q_differs)
                rb['ok'] = bool(ok)
                if not ok:
                    fails['3b'].append(key)
                rec['3b_comp_merge'][key] = rb
            # ---------------- 3c REF ties
            if fam == 'REF':
                qc = np.asarray(law['ref_struct']['q_coarse'])
                pairs_tied = all(qc[2 * k].tobytes() == qc[2 * k + 1].tobytes() for k in range(8))
                n_levels = int(len(np.unique(qc)))
                n_internal_q = int(len(np.unique(law['q'])))
                rc = {'coarse_distinct_q': n_levels, 'atoms_2k_2k1_bitwise_tied': bool(pairs_tied),
                      'internal_distinct_q': n_internal_q}
                for name in law['views']:
                    wid, K, m1, m0 = V.study_law_exact_view(law, name)
                    A_ex = V.pop_optimal_auc_exact(m1, m0)
                    rc[f'{name}_A_exact_ties'] = A_ex
                    rc[f'{name}_A_study_minus_exact'] = law['views'][name]['A'] - float(A_ex)
                    rc[f'{name}_distinct_exact_posteriors'] = len(set(V.posterior_exact(m1, m0)))
                ok = (n_levels == 8 and pairs_tied and n_internal_q == 8
                      and all(abs(rc[f'{n}_A_study_minus_exact']) <= 1e-12 for n in law['views'])
                      and all(rc[f'{n}_distinct_exact_posteriors'] == 8 for n in law['views']))
                rc['ok'] = bool(ok)
                if not ok:
                    fails['3c'].append(key)
                rec['3c_ref_ties'][key] = rc

# ---------------- own exact constructions (COMP merge, REF ties)
for ip, pi in enumerate((Fr(1, 2), Fr(1, 5))):
    for io, occ in enumerate(('uniform', 'skewed')):
        own = V.own_law('COMP', pi, occ, seed=VSEED + 10 * ip + io)
        Vp = own['views']['pair']
        A_m = V.pop_optimal_auc_exact(Vp['m1'], Vp['m0'])
        m1i = [pc * qc for pc, qc in zip(own['p'], own['q'])]
        m0i = [pc * (1 - qc) for pc, qc in zip(own['p'], own['q'])]
        A_i = V.pop_optimal_auc_exact(m1i, m0i)
        rb = {'K_income': own['views']['income']['K'], 'K_occupation': own['views']['occupation']['K'], 'K_pair': Vp['K'],
              'A_pair_merged': A_m, 'A_pair_unmerged_internal': A_i, 'weighted_mean_u': own['weighted_mean_u'],
              'weighted_mean_v': own['weighted_mean_v'], 'q_range': [float(min(own['q'])), float(max(own['q']))]}
        ok = rb['K_occupation'] == 15 and Vp['K'] == 240 and A_i > A_m and own['weighted_mean_u'] == 0 and own['weighted_mean_v'] == 0
        rb['ok'] = bool(ok)
        if not ok:
            fails['3b'].append(f'OWN-COMP|{pi}|{occ}')
        rec['3b_comp_merge'][f'OWN-COMP|{pi}|{occ}'] = rb
        own = V.own_law('REF', pi, occ, seed=VSEED + 100 + 10 * ip + io)
        qc = own['q_coarse']
        rc = {'coarse_distinct_q': len(set(qc)), 'atoms_2k_2k1_tied': all(qc[2 * k] == qc[2 * k + 1] for k in range(8)),
              'marginal_error_abs': float(abs(own['marginal_error'])),
              'L16_distinct_posteriors': len(set(V.posterior_exact(own['views']['L16']['m1'], own['views']['L16']['m0'])))}
        ok = rc['coarse_distinct_q'] == 8 and rc['atoms_2k_2k1_tied'] and rc['L16_distinct_posteriors'] == 8
        rc['ok'] = bool(ok)
        if not ok:
            fails['3c'].append(f'OWN-REF|{pi}|{occ}')
        rec['3c_ref_ties'][f'OWN-REF|{pi}|{occ}'] = rc

# 3d summary: which study views carry MORE distinct float posteriors than exact posteriors (ties split by float rounding)
split = {k: v for k, v in rec['3d_float_tie_structure'].items() if v['distinct_float_post_V_post'] > v['distinct_exact_posteriors']}
rec['3d_views_with_float_split_ties_in_V_post'] = {k: [v['distinct_exact_posteriors'], v['distinct_float_post_V_post']]
                                                   for k, v in split.items()}
rec['3d_max_abs_float_post_minus_exact_over_all_views'] = max(v['max_abs_float_post_minus_exact']
                                                             for v in rec['3d_float_tie_structure'].values())
rec['failures'] = fails
rec['status'] = {k: ('PASS' if not v else 'FAIL') for k, v in fails.items()}
rec['status']['3d'] = 'FINDING' if split else 'PASS'
print(rec['status'], {k: v[:4] for k, v in fails.items()}, len(split))
for k, v in rec['3d_views_with_float_split_ties_in_V_post'].items():
    print('  split', k, v)
V.write_out('CHECK3_ties_merging.json', rec, started_utc=T0)
