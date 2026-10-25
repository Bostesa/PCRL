"""Append the round-2 entries to D/roles/verifier/CHECKS.json. Round-1 entries are not rewritten: the file is loaded, new entries
are appended, and the round-1 entries are asserted unchanged before writing."""
import json

import vlib as V

O = V.OUT
path = V.VER / 'CHECKS.json'
old = json.load(open(path))
r1 = json.dumps(old['checks'], sort_keys=True)
c8, c9, c10a, c10b = (json.load(open(O / f)) for f in ('CHECK8_reproduce_summaries.json', 'CHECK9_rerun_repetitions.json',
                                                      'CHECK10_report_numbers.json', 'CHECK10_report_numbers_r2.json'))
own9 = c9['own_recompute_cell17_rep0']
new = [
    {'check': '8 (round 2) reproduce SUMMARY / POST_HOC entries from saved outputs with own code', 'status': c8['status'],
     'detail': f'{c8["n_compared"]["a"]} (cell, view, estimator) entries (cells 15, 17 REF|0.5|skewed L1024 at n = 2,000 / 20,000; 5 IND pair; '
               '29 RED pair; 41 COMP pair; 48 clustered): R, mean, SD, MCSE, bias against the own exact A and RMSE all equal SUMMARY.json '
               f'(1e-12); {c8["n_compared"]["b"]} comparisons from the saved [d, se_unit, se_person] triples (cell 17 L1024-L16 C_same / '
               'C_nested / D_bank, cell 29 pair-income, cell 28 income-occupation wrong-direction and power, cell 48 household and person '
               f'SE) equal; {c8["n_compared"]["c_bank"]} bank entries (winner SCORE / pop means, selection loss, A - best pop) equal; POST_HOC '
               'dbank_target REF|L1024-L16 and REF|L64-L16 (coverage, resolved rate, mean dpop) re-implemented and equal.',
     'out_file': 'out/CHECK8_reproduce_summaries.json'},
    {'check': '9 (round 2) re-run repetitions from their seeds', 'status': c9['status'],
     'detail': 'edg.simulate.one_rep re-run for (cell 17, rep 0, source check on), (cell 48, rep 7) and (cell 41, rep 499): the JSON lines '
               'are byte-identical to the saved lines. Own recomputation for (17, 0) from sample_rep: C_same (exact, own merging) and '
               f'B_indep (float posterior keys) equal the saved values (max {max(abs(x["C_same_diff"]) for x in own9.values()):.1e} / '
               f'{max(abs(x["B_indep_diff"]) for x in own9.values()):.1e}); with exact ties B_indep would differ by '
               f'{own9["L16"]["B_indep_exact_ties_minus_saved"]:+.1e} (L16) and {own9["L1024"]["B_indep_exact_ties_minus_saved"]:+.1e} '
               '(L1024) in this repetition (D1).', 'out_file': 'out/CHECK9_rerun_repetitions.json'},
    {'check': '10 (round 2) report numbers - ORIGINAL run (kept)', 'status': c10a['status'],
     'detail': f'{c10a["n_items"]} items against the report version of commit 6ceaa0b40: {c10a["n_mismatch_items"]} flagged. Four were the '
               'verifier\'s own criterion (more decimals than printed; an upper-bound claim judged as equality). Five were genuine in that '
               'version; three of those were already revised in the current report (RED pair test variation -0.000 -> -0.001, cross-fit '
               'range -0.0006 -> -0.0005, RED ignore-one now scoped to pi = 0.5; RED|0.2|skewed had 4 of 10). Superseded by r2.',
     'out_file': 'out/CHECK10_report_numbers.json'},
    {'check': '10 (round 2) report numbers - CORRECTED run r2 (current report)', 'status': 'FAIL',
     'detail': f'{c10b["n_items"]} items / {c10b["n_scalar_values"]} values against the current report (SHA-256 '
               f'{c10b["report_sha256_checked"][:12]}..., HEAD 523f0bec6), prioritising answers 1-5: {c10b["n_items"] - c10b["n_mismatch_items"]} '
               'match. Two mismatches remain, both last-digit rounding: (1) lines 48 and 124 "94.8-95.9%" - the POST_HOC dbank_target '
               'coverage range is 94.78-95.85%, i.e. 94.8-95.8%; (2) line 100 COMP pair row "0.092 | 0.075" - SUMMARY cell 41 gives gap '
               '0.09149 and optimism 0.07449, i.e. 0.091 and 0.074 (the 81% / 18% shares and the other entries match). No mismatch '
               'changes a conclusion.', 'out_file': 'out/CHECK10_report_numbers_r2.json'},
]
assert json.dumps(old['checks'], sort_keys=True) == r1
old['checks'] = old['checks'] + new
cpu2 = {f: json.load(open(O / f))['cpu_s_rusage_user_plus_sys'] for f in ('CHECK8_reproduce_summaries.json', 'CHECK10_report_numbers.json',
                                                                           'CHECK10_report_numbers_r2.json')}
old['round2'] = {'appended_utc': V.utc_now(), 'entries_appended': len(new), 'cpu_s_per_run': cpu2,
                 'cpu_s_total_recorded': round(sum(cpu2.values()), 3),
                 'note': 'CHECK8 and CHECK9 come from one process (its CPU is recorded in both files); round-1 entries unchanged'}
assert json.dumps(old['checks'][:len(old['checks']) - len(new)], sort_keys=True) == r1
with open(path, 'w') as fh:
    json.dump(old, fh, indent=1)
print('appended', len(new), 'round-1 entries unchanged', [e['status'] for e in new], old['round2']['cpu_s_total_recorded'])
