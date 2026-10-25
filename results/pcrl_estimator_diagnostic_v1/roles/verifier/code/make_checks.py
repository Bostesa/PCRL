"""Build D/roles/verifier/CHECKS.json (round 1) from the verifier's output files: one entry per check (and sub-check), status
PASS / FAIL / NOTE, a detail line built from the recorded values, and the output file. Also totals the recorded CPU seconds."""
import json

import vlib as V

O = V.OUT


def J(name):
    return json.load(open(O / name))


c0, c1, c2, c3, c3b, c3c = (J(f) for f in ('CHECK0_vlib_selftest.json', 'CHECK1_independence.json', 'CHECK2_known_score.json',
                                           'CHECK3_ties_merging.json', 'CHECK3b_known_score_ties.json', 'CHECK3c_b_ties_vs_spec.json'))
c4, c5, c6r, c6a, d6a, c6a2, c6b, c7 = (J(f) for f in ('CHECK4_refinement.json', 'CHECK5_red_comp.json', 'CHECK6_code_review.json',
                                                      'CHECK6a_estimators.json', 'DIAG6a_dnested_ties.json',
                                                      'CHECK6a_estimators_r2.json', 'CHECK6b_bank.json', 'CHECK7_provenance.json'))
ab = c6a2['a_b_c_same_and_inequality']
dfx = c6a2['d_e_f_nested_and_bank']
# B tie-split effect sizes, computed from the recorded part 2 / part 3 values (v2: v1 quoted an eyeballed R2SE range)
rows = []
for k, cc in c3b['cases'].items():
    for n, o in cc.items():
        for m, z in o['comparisons'].items():
            rows.append((n, z['R2SE_resolved_count_float'], z['R'], round(z['M20_share_float'] * z['R']), z['d_float_sd']))
for k, cc in c3c['cells'].items():
    for n, o in cc.items():
        for m, z in o.items():
            if not z['d_identically_zero']:
                rows.append((n, z['R2SE_resolved'], z['R'], z['M20'], z['d_sd']))
eff = {}
for n in ('2000', '20000'):
    rr = [r for r in rows if r[0] == n]
    eff[n] = {'R2SE_pooled': f'{sum(r[1] for r in rr)}/{sum(r[2] for r in rr)}',
              'R2SE_pooled_rate': sum(r[1] for r in rr) / sum(r[2] for r in rr),
              'R2SE_per_comparison_max': max(r[1] / r[2] for r in rr), 'M20_max_share': max(r[3] / r[2] for r in rr),
              'd_sd_max': max(r[4] for r in rr)}
E = [
    {'check': '0 (verifier tooling)', 'status': 'PASS' if c0['results']['all_pass'] else 'FAIL',
     'detail': 'Own exact-AUC library validated on hand-computable cases (3/5, 11/20, pure wires, ties, enumeration maximum, merging, '
               '60-digit coarse-signal solve) before judging the study.', 'out_file': 'out/CHECK0_vlib_selftest.json'},
    {'check': '1 independence: population AUC exactly 1/2', 'status': c1['status'],
     'detail': 'All 12 views of the 4 study IND laws and of 4 own exact-rational IND laws: P1 == P0 exactly; exact AUC = 1/2 '
               '(Fraction equality) for the optimal ranking, random tied / untied fixed scores, the wire index, the study float '
               'posterior and LOOKUP tables fitted on own samples; the study float A, drf auc_of_score and the d_nested / d_bank '
               f'fitted-reader pops are within 1e-12 of 1/2. Failures: {len(c1["failures"])}.', 'out_file': 'out/CHECK1_independence.json'},
    {'check': '2 known-score sample AUC vs brute-force enumeration', 'status': c2['status'],
     'detail': f'{c2["cases"]} sample x score cases (all 16 laws, every view, n = 20-1,000, own seeds): max |auc_if - exact| '
               f'{c2["max_abs_diff"]["auc_if"]:.1e}, max |b_known - exact| {c2["max_abs_diff"]["b_known"]:.1e}; IF equals brute-force '
               f'DeLong placements (max {c2["max_abs_diff"]["IF_vs_placements"]:.1e}); person-unit SE relative error '
               f'{c2["max_abs_diff"]["cluster_se_rel"]:.1e}; UNDEFINED (one class) -> NaN, never 1/2; ties 1/2.',
     'out_file': 'out/CHECK2_known_score.json'},
    {'check': '3a merging: observable supports and partitions', 'status': c3['status']['3a'],
     'detail': 'Own byte-level merging of every view of all 16 study laws: supports equal the study K and the protocol table '
               '(IND 16/64/1024, REF 16/64/256/1024, RED 16/64/256, COMP 16/15/240); own partition = study partition (bijection); '
               'merged masses equal (relative <= 1e-12); all emitted vectors valid.', 'out_file': 'out/CHECK3_ties_merging.json'},
    {'check': '3b COMP identical wires merged; unmerged internal ids overstate A', 'status': c3['status']['3b'],
     'detail': 'All 4 study COMP laws and 4 own exact COMP laws: occupation cells 14 and 15 emit bitwise-identical vectors (15 '
               'distinct, no other coincidence); supports 15 / 240; exact A on 256 unmerged internal ids exceeds merged A (study '
               'laws by 0.0009-0.0019); merged cells carry different q. Own uniform COMP reproduces the study exactly '
               '(A(pair) 0.5828515625 / 0.603564453125).', 'out_file': 'out/CHECK3_ties_merging.json'},
    {'check': '3c tied likelihood ratios in A (REF 8 posterior levels)', 'status': c3['status']['3c'],
     'detail': 'REF coarse q: 8 distinct values, atoms 2k / 2k+1 bitwise tied, 8 internal levels (study and own laws); the study '
               'float A equals the exact tied-ratio A within 1.5e-15 at every level, so A handles the ties correctly.',
     'out_file': 'out/CHECK3_ties_merging.json'},
    {'check': '3d ties 1/2 in the B known score (B_same / B_indep) - STUDY DEFECT', 'status': 'FAIL',
     'detail': 'laws.build sets post = m1 / (m1 + m0) from float masses, and b_known ranks that float. Wires whose posteriors '
               'are exactly equal (in the float law and in the specification) get posteriors differing in the last bits, so b_known '
               f'orders pairs that must tie: {len(c3["3d_views_with_float_split_ties_in_V_post"])} of 52 views split float-law '
               f'ties (e.g. REF|0.5|skewed L16: 8 exact levels -> 15 floats). Against the specification, '
               f'{len(c3c["null_comparisons_where_B_d_is_not_identically_zero"])} of {c3c["n_null_comparisons_total"]} dA = 0 '
               'comparisons give B differences that are not identically 0 (the exact-tie known score gives d = 0 on every sample): '
               'IND|0.2|skewed (3), REF|*|skewed (6), RED|*|skewed pair-income (2), COMP occupation-income (all 4 laws). Over the '
               f'affected comparisons (parts 2 and 3, own samples): at n = 2,000, SD of d up to {eff["2000"]["d_sd_max"]:.3f}, '
               f'|d| >= 0.020 in up to {eff["2000"]["M20_max_share"]:.0%} of samples, R2SE resolved {eff["2000"]["R2SE_pooled"]} '
               f'pooled ({eff["2000"]["R2SE_pooled_rate"]:.1%}; at most {eff["2000"]["R2SE_per_comparison_max"]:.0%} per comparison); '
               f'at n = 20,000, SD up to {eff["20000"]["d_sd_max"]:.4f}, M20 up to {eff["20000"]["M20_max_share"]:.0%}, R2SE '
               f'{eff["20000"]["R2SE_pooled"]} pooled ({eff["20000"]["R2SE_pooled_rate"]:.1%}; at most '
               f'{eff["20000"]["R2SE_per_comparison_max"]:.0%} per comparison, R = 20-40). The exact-tie known score gives 0 for all '
               'of these. B stays unbiased (mean deviations within about 2 MCSE of 0) and A, C, D are unaffected. Fix suggestion: '
               'rank B by the exact tie structure (e.g. group wires by the internal q, which is bitwise tied in REF / RED / IND, or '
               'use rationals; COMP needs exactly centred u, v), then recompute B rows; C and D rows need no change.',
     'out_file': 'out/CHECK3_ties_merging.json; out/CHECK3b_known_score_ties.json; out/CHECK3c_b_ties_vs_spec.json'},
    {'check': '4 irrelevant refinement preserves the population optimum', 'status': c4['status'],
     'detail': 'All 4 study REF laws and 4 own exact REF laws: exact A identical (Fraction equality) at L16, L64, L256, L1024; '
               'nested partitions with 4 children per parent; every child exact posterior equals its parent exactly; study float A '
               'spread across levels <= 1.8e-15; study A range 0.6708-0.6814 as stated; own uniform laws and an exact rebuild of the '
               'study skewed laws (recovered Zipf permutation, own 60-digit alpha) reproduce the study A within 3.3e-16.',
     'out_file': 'out/CHECK4_refinement.json'},
    {'check': '5 redundant and complementary views; null / non-null classification', 'status': c5['status'],
     'detail': 'RED (study + own): A(pair) == A(income) exactly, A(income) > A(occupation), every pair-wire posterior equals its '
               'income-wire posterior exactly (S independent of occupation given income); ranges 0.642-0.677. COMP (study + own): '
               'A(income) = A(occupation) = 1/2 (own exact; study float law within 1.1e-17), A(pair) 0.563-0.604 > 1/2. Own uniform '
               'laws equal the study A. All 32 + 16 registered comparisons classified as the protocol states; largest null |dA| '
               f'{c5["max_abs_dA_null"]:.1e}, smallest non-null |dA| {c5["min_abs_dA_nonnull"]:.4f}.', 'out_file': 'out/CHECK5_red_comp.json'},
    {'check': '6 static review: label access in estimators.py, simulate.py, bank.py', 'status': 'PASS',
     'detail': f'{len(c6r["citations"])} cited lines, each verified programmatically to contain the quoted code: LOOKUP tables use '
               'training-fold / FIT labels only; every bank recipe, its feature design and scaler fit on FIT rows; orientation and '
               'winner selection use VALIDATION only; SCORE labels score the D_bank table and the selected winner once; support '
               'rows have False FIT / VALIDATION masks; C_same / C_nested are same-sample by design. The reviewed files equal the '
               'registration lock.', 'out_file': 'out/CHECK6_code_review.json'},
    {'check': '6 static review: lock comparison recorded inside the review file', 'status': 'NOTE',
     'detail': 'The file\'s own status field reads FAIL because its lock list also included selftest.py, which differs from the '
               'GRID.json registration lock: disclosed amendment A3 (2 lines, tooling only, after outcomes). No estimator, law, '
               'simulate, bank or analysis file changed (CHECK7 7d). The file is kept unchanged.', 'out_file': 'out/CHECK6_code_review.json'},
    {'check': '6a estimator behaviour - ORIGINAL run (kept)', 'status': 'FAIL',
     'detail': 'Verifier-check defect, not a study defect: D_nested was compared with an exact-rational idealization of the DRF1 float '
               'LOOKUP reader and differed by up to 2.2e-5 in 4 of 52 views. Diagnosed in DIAG6a (brute force on the reader\'s own float '
               'scores equals the study value exactly; 10 rational ties split by float rounding). Superseded by r2.',
     'out_file': 'out/CHECK6a_estimators.json'},
    {'check': '6a diagnostic of the original failure', 'status': 'NOTE',
     'detail': f'{d6a["total_exact_ties_split_in_float"]} exact ties of shrunk LOOKUP rates split by float rounding across 52 own-sample '
               f'views; brute force on float scores minus study: value {d6a["max_abs_bruteforce_on_float_scores_minus_study"]["value"]:.1e}, '
               f'pop_fold_mean {d6a["max_abs_bruteforce_on_float_scores_minus_study"]["pop_fold_mean"]:.1e}.',
     'out_file': 'out/DIAG6a_dnested_ties.json'},
    {'check': '6a estimator behaviour - CORRECTED run r2', 'status': c6a2['status'],
     'detail': f'{ab["samples"]} own samples (all 16 laws, every view, n = 200 / 2,000 / 20,000): C_same equals drf.solver.auc_star, '
               f'drf.finite_law.auc_star and the exact brute force (max {ab["max_abs_c_same_minus_exact"]:.1e}); C_same >= B_same on every '
               f'sample (0 violations exact, both float and exact-tie B; min C - B {ab["min_C_minus_B_float"]:.1e}); C_same equals '
               'the maximum over ALL rankings on 93 tiny tables (k <= 6). C_nested exact (7.8e-16). D_nested and D_bank equal '
               f'brute force on the readers\' float scores (max {max(dfx["max_abs"].values()):.1e}); every lookup_scores call receives '
               'exactly ids / labels of folds != f (0 violations); relabelling a fold leaves its held-out scores unchanged; relabelling '
               'SCORE leaves the D_bank table, orientation, VALIDATION AUC and pop bitwise unchanged (52/52); sample_rep never splits a '
               'household across folds or parts (cell 48) and the parts are 50 / 20 / 30% of units. NOTE: exact-idealization '
               f'differences kept, max {max(dfx["note_exact_idealization_max_abs"].values()):.1e} (source reader behaviour).',
     'out_file': 'out/CHECK6a_estimators_r2.json'},
    {'check': '6b reader bank: no SCORE label or support row enters a fit, orientation or selection', 'status': c6b['status'],
     'detail': 'RED|0.5|skewed (income, occupation, pair) and REF|0.5|skewed L16, n = 2,000, 16 recipes, non-study seed: SCORE-label '
               'perturbation leaves every recipe score, VALIDATION AUC, orientation, convergence flag, population AUC and winner '
               'bitwise unchanged; refits without the appended support rows give bitwise-identical scores; recipe pops equal brute '
               'force (<= 3.3e-16); winners equal an independent implementation of the rule; VALIDATION-label perturbation changes '
               'all 16 VALIDATION AUCs (the test has power); _src_tab / _fit_one are text-equal to drf/devreaders.py.',
     'out_file': 'out/CHECK6b_bank.json'},
    {'check': '7 provenance', 'status': c7['status'],
     'detail': 'All 10 vendored files equal SOURCE_PROVENANCE.json and the 913eb5c91 blobs; the 5 lock-listed recipe files (and '
               'drf/devreaders.py) equal DEVGRID_CODE_LOCK.json; nrm/fa/prc.py blob hash equals provenance and both copied '
               'functions are text-equal; drf publication files equal provenance and the c4896479e blobs; LAWS.json p / q hashes, '
               'supports and A (within 1.6e-15 of the exact A) match. Since registration: edg_run.py and selftest.py changed; '
               'custody.py, publish.py and posthoc.py were added. All are disclosed (A1-A4); estimator / analysis files and GRID.json '
               'are unchanged.', 'out_file': 'out/CHECK7_provenance.json'},
]
cpu = {}
for p in sorted(O.glob('*.json')):
    cpu[p.name] = json.load(open(p)).get('cpu_s_rusage_user_plus_sys')
rec = {'schema': 'EDG1-verifier-checks-round1', 'role': 'independent verifier', 'checks': E, 'cpu_s_per_output': cpu,
       'cpu_s_total_recorded': round(sum(v for v in cpu.values() if v), 3), 'B_tie_split_effect_sizes': eff,
       'supersedes': 'CHECKS_v1_superseded.json (kept): its 3d detail quoted an eyeballed R2SE range "about 2-10%"; v2 computes '
                     'the ranges from the recorded values. No status changed.'}
V.write_out('CHECKS_build_record_v2.json', {'built': 'CHECKS.json (v2)', 'n_entries': len(E)})
with open(V.VER / 'CHECKS.json', 'x') as fh:
    json.dump(rec, fh, indent=1)
print(json.dumps({e['check']: e['status'] for e in E}, indent=1), rec['cpu_s_total_recorded'])
