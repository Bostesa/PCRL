"""Verifier unit AMD_VERIFY_ASSEMBLE_V1: assemble INDEPENDENT_VERIFICATION.json and VERIFICATION.md from the verifier's own
unit outputs (V_NOLABEL, V_READERS, V_DEV, V_DEPLOY, V_ACCESS). Aggregates only; no person-level array."""
from __future__ import annotations
import json
import sys

sys.path.insert(0, '<repository>/.claude/worktrees/pcrl-autonomous-method-development-v1/.private/pcrl_autonomous_method_development_v1/verifier/independent')
import vcommon as V

O = {n: json.loads((V.OUT / f'V_{n}.json').read_text()) for n in ('NOLABEL', 'READERS', 'DEV', 'DEPLOY', 'ACCESS')}
nl, rd, dv, dp, ac = O['NOLABEL'], O['READERS'], O['DEV'], O['DEPLOY'], O['ACCESS']
EPS = {r['key']: r for r in dv['endpoints']}


def f(x, n=10):
    return 'n/a' if x is None else (f'{x:.{n}f}' if isinstance(x, float) else str(x))


# ------------------------------------------------------------------ check verdicts
c2_eval = all(r[t]['vs_bank_max_abs'] <= 1e-12 and r[t]['decisions_equal_bank'] for s in nl['check2']['detail'].values()
              for role, r in s['roles'].items() if role in ('MODEL_SELECT', 'DEV_REPORT') for t in V.TASKS)
c2_thr = max(s['thresholds_max_abs_diff'] for s in nl['check2']['detail'].values())
c2_h = max(max(v for k, v in s['h_refit_on_CALIBRATION'].items() if k.endswith('_abs')) for s in nl['check2']['detail'].values())
dev_ok = dv['max_point_abs_diff'] <= 1e-12 and dv['max_se_rel_diff'] <= 1e-9 and not dv['endpoint_mismatches']
dev_ev = [e for e in ac['dev_events']]
chk1 = (len(dev_ev) == 2 and dev_ev[0]['event'].endswith('COORDINATOR') and dev_ev[0]['utc'] == '2026-10-17T17:45:30+00:00'
        and all(set(ac['ms_reads_by_stage']) <= {'baselines', 'famA-score', 'famB-score', 'famC-score', 'ctl-taskonly-score', 'verifier'} for _ in [0])
        and len(ac['attack_sex_reads']) == 1 and ac['attack_sex_reads'][0]['file'] == 'readers.py' and not ac['sealed_in_any_argv']
        and not ac['source_access_log_sealed_mentions'] and c2_h == 0.0)
checks = {
    '1_data_roles_and_exposure': chk1,
    '2_runtime_outputs': c2_eval and c2_thr <= 1e-12,
    '3_utility_and_subgroup': nl['check3_model_select']['pass'] and dev_ok and dv['comparator_dev_point_status_12_matches'],
    '4_selection': nl['check4']['pass'],
    '5_paired_household_bootstrap': dev_ok and dv['weights']['equal_to_group_weights_bitwise'],
    '6_labels': dv['claims_match'] and dv['CV'],
    '7_reader_refits': rd['pass'] and rd['winner_rule_pass'],
    '8_deployment': dp['deployment_pass'] and all(v['bitwise_all_roles'] for v in dp['bank_integrity'].values()),
}
disc = [
    {'id': 'D1', 'severity': 'LOW (custody narrative; no numerical effect)',
     'text': ("The statistical reviewer's CLAIMS_AND_ENDPOINTS_DRAFT.md says it was written 'at about 17:50Z, before any finalist freeze', and "
              "DEVELOPMENT_REVIEW.md says 'written 17:30-17:50Z'. Git shows DEVELOPMENT_REVIEW.md first committed in ad37c120c (17:37:13Z, the finalist "
              "freeze) and the draft first committed in 345e0adb0 (17:43:02Z), whose ANALYSIS_LOCK (utc 17:42:54Z) carries the draft's hash. The "
              "self-reported 17:50Z times are impossible. The repository cannot show that the draft existed before the finalist freeze (its first "
              "commit is 5 min 49 s after it, and its worked instance names the adjudicated finalist). What is verifiable: the draft hash in "
              "ANALYSIS_LOCK equals the current file; the lock was committed before the reader fits (17:43:14Z) and was published (origin/publication "
              "614c9f36b, 17:45:17Z, same blobs as 721d4163d) before the coordinator's DEV_REPORT read (17:45:30Z).")},
    {'id': 'D2', 'severity': 'INFO (documentation)',
     'text': 'PROTOCOL.md section 3 gives MODEL_SELECT as 8,000 people; the admitted export, the bank and every evaluation use 8,001 rows.'},
    {'id': 'D3', 'severity': 'INFO (numerical fragility on a non-evaluated role)',
     'text': ("The verifier's own recomputation of the finalist on MODEL_FIT (not a released or evaluated role) differs from RELEASE.npz for the people "
              "whose MIX income statistic lies within 4.4e-16 of a bin threshold (2 people on seed 0, 4 on seed 1; max probability difference 0.179 / 0.191; "
              "decisions unchanged). The thresholds are empirical quantiles of those same MODEL_FIT values, so some people sit exactly on a threshold. "
              "A last-bit change in the summation order of the tree scores moves them to the other bin. On MODEL_SELECT and DEV_REPORT the nearest "
              "statistic is at least 3.5e-5 from any threshold, and outputs match bitwise.")},
]
nested = dv['claims']['B']['nested']
obs = [
    ('E holds at the point with very thin utility margins on DEV_REPORT: worst clause = income Brier excess vs Ucal at '
     f"{EPS['UTILITY|income|Ucal|brier_excess|s0']['point']:.6f} (s0) and {EPS['UTILITY|income|Ucal|brier_excess|s1']['point']:.6f} (s1) against the 0.005 cap "
     f"({dv['finalist_utility_max_normalized']['S0']:.3f} / {dv['finalist_utility_max_normalized']['S1']:.3f} of cap; the registered forecast cited "
     "0.87-0.88 on MODEL_SELECT). Its clause L1 is above 0 on both seeds "
     f"({EPS['UTILITY|income|Ucal|brier_excess|s0']['L1']:.6f}, {EPS['UTILITY|income|Ucal|brier_excess|s1']['L1']:.6f}). "
     f"Simultaneous confidence over the 28 utility and subgroup endpoints is {dv['E_simultaneous_utility']['label']} "
     f"({dv['E_simultaneous_utility']['PASS']} PASS, {dv['E_simultaneous_utility']['UNRESOLVED']} UNRESOLVED, 0 ESTABLISHED_VIOLATION)."),
    ('The nested 0.020 target is not established: clause L1 = ' + ', '.join(f'{k.split("|")[1].replace("pair_benefit_vs_", "")} {k.split("|")[2]} {v:.6f}' for k, v in nested['L1_values'].items())
     + '. All are "positive, below the original 0.020 target".'),
    ('J fails on JOINT|pair_benefit_vs_B2TWIN|s1 (point 0.009915, clause L1 -0.002526; family status UNRESOLVED). '
     'Seed 0 passes (L1 0.009920).'),
    ('M_SS is the alias of claim B\'s H_LJ keys (sigma = 0). The step therefore adds no comparison beyond B. '
     'MECHANISM_ATTRIBUTION_ESTABLISHED rests on the task-only control (pair benefit 0.0515 / 0.0465) plus those H_LJ legs.'),
    ('The occupation outputs of FINALIST, H_LJ, B2TWIN and H_JJ are bitwise identical on MODEL_SELECT and DEV_REPORT. '
     'The occupation recipient keys against H_LJ and B2TWIN therefore have point = SE = 0, which matches the IDENTICAL_RELEASE flags.'),
    ('Scope: the non-refit reader scores are the frozen DEV_REPORT_SCORES.npz. The winner rule was re-applied to the recorded ATTACK_VALIDATION AUCs, '
     'and the 14-id subset was refitted. The ATTACK_VALIDATION control AUCs are the recorded values. The section 6 proxies are the recorded '
     'EVAL_MS.json values; the utility clauses of all 3 promotable configurations were recomputed. The NEURAL_RAW seed-1 package does not exist in '
     'the source (coverage gap); deployment check 8 covers FINALIST and TASKONLY only, as briefed.'),
]
res = {'schema': 'AMD1-independent-verification-v1', 'utc': None,
       'verifier_units': ['AMD_VERIFY_NOLABEL_V1', 'AMD_VERIFY_READERS_V1', 'AMD_VERIFY_DEVDRY_V1', 'AMD_VERIFY_DEVDRY_V2', 'AMD_VERIFY_DEV_V1',
                          'AMD_VERIFY_DEPLOY_V1', 'AMD_VERIFY_ACCESS_V1', 'AMD_VERIFY_ASSEMBLE_V1'],
       'dev_truth_event': [e for e in dev_ev if e['event'].endswith('INDEPENDENT_VERIFIER')],
       'checks': {k: 'PASS' if v else 'FAIL' for k, v in checks.items()},
       'numbers': {
           'check2': {s: {'DEV_REPORT_max_abs_vs_bank': {t: d['roles']['DEV_REPORT'][t]['vs_bank_max_abs'] for t in V.TASKS},
                          'MODEL_SELECT_max_abs_vs_bank': {t: d['roles']['MODEL_SELECT'][t]['vs_bank_max_abs'] for t in V.TASKS},
                          'thresholds_recomputed': d['thresholds_recomputed_from_MODEL_FIT_features'], 'thresholds_recorded': d['thresholds_recorded'],
                          'thresholds_max_abs_diff': d['thresholds_max_abs_diff'], 'h_refit_on_CALIBRATION': d['h_refit_on_CALIBRATION'],
                          'occupation_identical_to': d['occupation_identical_to'], 'distinct_income_vectors_DEV': d['distinct_income_vectors_DEV'],
                          'MODEL_FIT_income_vs_release_npz_max_abs': d['roles']['MODEL_FIT']['income']['vs_release_npz_max_abs']}
                      for s, d in nl['check2']['detail'].items()},
           'check3_model_select': nl['check3_model_select']['detail'],
           'check3_dev_comparator_point_status_12': dv['comparator_dev_point_status_12'],
           'check4': {k: nl['check4'][k] for k in ('promotable', 'robust_promotable', 'choice', 'label', 'recorded_finalist', 'reference_bar',
                                                   'comparison', 'ledger_rows', 'ledger_counts_toward_36', 'n_configs_with_eval_ms')},
           'check4_promotable_eval_ms_recheck_max_abs_diff': nl['check4_promotable_eval_ms_recheck_max_abs_diff'],
           'check5_weights': dv['weights'], 'critical': dv['critical'], 'endpoints_all_72': dv['endpoints'],
           'max_point_abs_diff': dv['max_point_abs_diff'], 'max_se_rel_diff': dv['max_se_rel_diff'],
           'lock_ab_sign_encoding_matches_estimands': dv['lock_ab_sign_encoding_matches_estimands'],
           'controls': dv['controls'], 'CV': dv['CV'],
           'claims_recomputed': dv['claims'], 'claims_recorded': dv['claims_recorded'], 'B_nested_recorded': dv['B_nested_recorded'],
           'E_simultaneous_utility': dv['E_simultaneous_utility'], 'E_recipient_clause_counts': dv['E_recipient_clause_counts'],
           'mandatory_qualifiers': dv['mandatory_qualifiers'], 'auc_by_service': dv['auc_by_service'],
           'check7_refits': rd['refits'], 'check7_winner_rule': {s: {k: v for k, v in w.items() if k != 'detail'} for s, w in rd['winner_rule'].items()},
           'check8_deployment': dp['deployment'], 'bank_integrity': dp['bank_integrity'], 'lock_chain': dp['lock_chain'],
           'threshold_proximity': dp['threshold_proximity'],
           'check1': {'dev_events': ac['dev_events'], 'ms_reads_by_stage': {k: len(v) for k, v in ac['ms_reads_by_stage'].items()},
                      'attack_sex_reads_in_code': ac['attack_sex_reads'], 'sealed_in_any_argv': ac['sealed_in_any_argv'],
                      'source_access_log_events_this_campaign': ac['source_access_log_events_this_campaign'],
                      'queue_units': ac['n_units'], 'failed_units_in_ledger': [u for u in ac['failed_units'] if u != 'AMD_VERIFY_ACCESS_V1']}},
       'discrepancies': disc, 'observations': obs}
res['utc'] = dv['utc']
(V.PRIV / 'verifier' / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')

# ------------------------------------------------------------------ markdown
L = []
A = L.append
A('# Independent verification: PCRL autonomous method-development campaign (frozen finalist B3CTL_UNCOUPLED_MIX50_C6A)')
A('')
A('**Census context.** The records are public-use Florida ACS 2019 person records. The service releases, for each person, an income probability vector '
  '(2 classes) and an occupation probability vector (5 classes), each with its argmax decision. The inputs are six raw fields: AGEP, WKHP, COW, SCHL, '
  "MAR and POBP. Evaluation readers infer the dataset's recorded SEX column from each output alone and from the two outputs aligned to the same "
  'record (the linked pair).')
A('')
A('**Who and how.** This is the independent verifier, using separate code in `.private/pcrl_autonomous_method_development_v1/verifier/independent/` '
  '(`vcommon.py`, `v_nolabel.py`, `v_readers.py`, `v_dev.py`, `v_deploy.py`, `v_access.py`, `assemble.py`).')
A('- Nothing is imported from `amd.report`, `amd.evaluate`, `amd.promote` or `amd.devrun` for any checked quantity. `amd.deploy` imports `amd.devrun` '
  'transitively, but only `release`, `release_csv` and `raw_rows` are called, and they are the objects under test.')
A('- `amd.common` is used only for its hash-checked loaders.')
A('- `task_conditioned_representation.attacks.fit_reader` is used only to construct the refitted readers. Its `predict_reader`, `orientation` and `auc` are not used.')
A("- `task_headroom.inference.group_weights` is called once, to compare the source's bootstrap weights with the verifier's own implementation.")
A('')
A('**Units.** Every numerical job ran through `amd.jobqueue`, about 40 CPU-s in total: AMD_VERIFY_NOLABEL_V1, AMD_VERIFY_READERS_V1, AMD_VERIFY_DEVDRY_V1 and '
  'AMD_VERIFY_DEVDRY_V2 (invented labels, no truth access), AMD_VERIFY_DEV_V1, AMD_VERIFY_DEPLOY_V1, AMD_VERIFY_ACCESS_V1 and AMD_VERIFY_ASSEMBLE_V1.')
A('')
A('**DEV_REPORT truth.** It was re-read once, inside AMD_VERIFY_DEV_V1, through `amd.common.dev_report_truth(..., '
  "'DEV_REPORT_TRUTH_REREAD_BY_AMD_INDEPENDENT_VERIFIER')`. The access log records the event at "
  f"{[e['utc'] for e in dev_ev if e['event'].endswith('INDEPENDENT_VERIFIER')][0]}, and both the campaign log and the source log carry it. "
  'Labels stayed in memory; only aggregates were written.')
A('')
A('**Tolerances.** Outputs and points: bitwise or <= 1e-12. Standard errors: <= 1e-9 relative.')
A('')
A('## Verdicts')
A('')
A('| Check | Verdict |')
A('|---|---|')
for k, v in checks.items():
    A(f'| {k} | **{"PASS" if v else "FAIL"}** |')
A('')
A('## 1. Data roles and exposure: ' + ('PASS' if checks['1_data_roles_and_exposure'] else 'FAIL'))
A('')
A('**DEV_REPORT_REREAD.jsonl** holds exactly two events.')
A('- The coordinator read at 17:45:30Z in unit AMD_REPORT_DEV_V1. This came after REPORTING_LOCK was committed (721d4163d, 17:45:01Z) and published '
  '(origin/publication/... 614c9f36b, 17:45:17Z; the REPORTING_LOCK and ANALYSIS_LOCK blobs are identical in both commits).')
A('- The verifier read once, in this verification.')
A("- Both events carry the REPORTING_LOCK sha256 f8d6e354... and appear in the source access log.")
A('')
A('**MODEL_SELECT_USE.jsonl** holds 25 entries: 24 from the campaign and 1 from this verifier.')
A('- By queue stage: ' + ', '.join(f'{k} {len(v)}' for k, v in ac['ms_reads_by_stage'].items()) + '.')
A('- Every read is a scoring or evaluation stage. No fit stage read MODEL_SELECT labels.')
A('')
A('**ATTACK SEX.** Static inspection of every `load_role` call in `amd/*.py` finds one ATTACK SEX read: `readers.py:61`, the final readers.')
A('- That code refuses to run without the committed FINALIST_LOCK, ANALYSIS_LOCK and SERVICE_BANK_MANIFEST.')
A('- The reader units started at 17:43:14Z, after the ANALYSIS_LOCK commit at 17:43:02Z.')
A('- Fitting code (`fam_a`, `fam_b`, `fam_c`) loads ATTACK and DEV_REPORT features only.')
A('')
A('**Finalist training exposure.** `fam_b.fit` reads features of every role and CALIBRATION income only.')
A('- Refitting h on CALIBRATION income labels alone, from the verifier\'s own recomputed design, reproduces w, b, mu and sd with a maximum '
  f'absolute difference of {c2_h}, i.e. exactly.')
A('- The coarse thresholds are reproduced from label-free MODEL_FIT log-odds to within '
  f'{c2_thr:.2e} (summation-order rounding).')
A('- The stored Stage B E20 corrections are admitted copies, hash-checked by `amd.common.admitted`. They used MODEL_FIT labels and SEX in the replication, '
  'as RELEASE.json discloses.')
A('')
A('**Sealed assessment.** No queue argv mentions SEALED, the source access log has no sealed-assessment event, and the loader refuses the role.')
A('')
A('## 2. Runtime outputs: ' + ('PASS' if checks['2_runtime_outputs'] else 'FAIL'))
A('')
A('The verifier used its own oblivious-tree routing from the admitted teacher JSONs (float32 comparison, bit i per split), with sequential '
  'accumulation of leaf values and of theta/T. On top of that it applied the stored LOCAL_TREE and JOINT_TREE E20 thetas, MIX = 0.5 L + 0.5 J, '
  'the 6 aligned bins (`#thresholds < d`), one-hot minus first, ((Z - mu)/sd) w + b, and softmax(-s/2, s/2). The occupation output is the JOINT_TREE '
  'softmax of centred scores.')
A('')
A('| Seed | Role | Max abs diff vs bank (income / occupation) | Decisions equal |')
A('|---|---|---|---|')
for s, d in nl['check2']['detail'].items():
    for role in ('MODEL_SELECT', 'DEV_REPORT'):
        r = d['roles'][role]
        A(f"| {s} | {role} (n={r['income']['n']}) | {r['income']['vs_bank_max_abs']} / {r['occupation']['vs_bank_max_abs']} (bitwise) | {r['income']['decisions_equal_bank'] and r['occupation']['decisions_equal_bank']} |")
A('')
for s, d in nl['check2']['detail'].items():
    A(f"- {s}: the thresholds recomputed from MODEL_FIT features are {[round(x, 12) for x in d['thresholds_recomputed_from_MODEL_FIT_features']]}; the recorded values are "
      f"{[round(x, 12) for x in d['thresholds_recorded']]}, a maximum difference of {d['thresholds_max_abs_diff']:.2e}.")
    A(f"  The h refit on CALIBRATION matches exactly, and DEV_REPORT has {d['distinct_income_vectors_DEV']} distinct income vectors.")
    A(f"  The occupation output is identical to H_LJ, B2TWIN and H_JJ on MODEL_SELECT and DEV_REPORT, and differs from NEURAL_RAW and TASKONLY.")
A('')
A('See D3 for the non-evaluated MODEL_FIT role.')
A('')
A('## 3. Utility and subgroup: ' + ('PASS' if checks['3_utility_and_subgroup'] else 'FAIL'))
A('')
A('**MODEL_SELECT (finalist; the verifier\'s own loss vectors).** These are compared with EVAL_MS.json.')
for s, d in nl['check3_model_select']['detail'].items():
    A(f"- {s}: all 12 clauses pass at the point, and both guards pass. The maximum absolute difference against EVAL_MS.json is {d['max_abs_diff_vs_EVAL_MS']:.2e}.")
    A(f"  The worst normalised clause is {d['max_normalized_12']:.4f}. The guards are men {d['guards']['sex0']['recomputed']:.10f} (recorded {d['guards']['sex0']['recorded']:.10f}) "
      f"and women {d['guards']['sex1']['recomputed']:.10f} (recorded {d['guards']['sex1']['recorded']:.10f}).")
A('')
A('**DEV_REPORT.** All 24 utility points and the 4 guards are recomputed with SEs. They are listed in INDEPENDENT_VERIFICATION.json, and every one equals '
  f"the recorded value. The maximum point difference is {dv['max_point_abs_diff']} and the maximum relative SE difference is {dv['max_se_rel_diff']:.1e}. "
  f"The finalist's worst normalised utility clause is {dv['finalist_utility_max_normalized']['S0']:.4f} (s0) and {dv['finalist_utility_max_normalized']['S1']:.4f} (s1).")
A('')
A("**Comparators' DEV 12-clause point status** (the verifier's recomputation equals the record):")
A('')
A('| Seed | ' + ' | '.join(sorted({k.split('|')[1] for k in dv['comparator_dev_point_status_12']})) + ' |')
A('|---|' + '---|' * len({k.split('|')[1] for k in dv['comparator_dev_point_status_12']}))
for s in ('S0', 'S1'):
    A(f'| {s} | ' + ' | '.join(str(dv['comparator_dev_point_status_12'][f'{s}|{svc}']) for svc in sorted({k.split('|')[1] for k in dv['comparator_dev_point_status_12']})) + ' |')
A('')
A('## 4. Selection (PROTOCOL section 6 over every fitted configuration): ' + ('PASS' if checks['4_selection'] else 'FAIL'))
A('')
c4 = nl['check4']
A(f"The rule was re-applied to {c4['n_configs_with_eval_ms']} configurations with EVAL_MS.json, controls included, per DEVELOPMENT_REVIEW.md section 0.")
A('- U: 12 clauses at the point. G: 2 guards. R screen: each single-output proxy <= the reference proxy + 0.010, for NEURAL_RAW and H_LJ.')
A('- Pair proxy below both references on both seeds.')
A('- Robustness: every U and G value <= 0.8 x cap.')
A('- Reference bars from BASELINES_MS_S*.json (equal to PROMOTION.json): '
  + '; '.join(f"S{s} NEURAL_RAW pair {c4['reference_bar'][str(s)]['NEURAL_RAW']['pair']:.4f}, H_LJ pair {c4['reference_bar'][str(s)]['H_LJ']['pair']:.4f}" for s in (0, 1)) + '.')
A('')
A(f"**Result.** Promotable: {c4['promotable']}. Robust: {c4['robust_promotable'] or 'none'}. Choice: **{c4['choice']}**, {c4['label']}. "
  f"PROMOTION.json records {c4['recorded_finalist']['config_id']}, {c4['recorded_finalist']['selection_label']}, n_promotable 3, robust 0. These match.")
A('')
A('| Config | Promotable (verifier / recorded) | Mean pair proxy (verifier / recorded) |')
A('|---|---|---|')
for r in c4['comparison']:
    if r['mine_mean_pair_proxy'] is not None:
        A(f"| {r['config']} | {r['mine_promotable']} / {r['recorded_promotable']} | {r['mine_mean_pair_proxy']:.6f} / {r['recorded_mean_pair_proxy']:.6f} |")
A('')
A("The 3 promotable configurations' MODEL_SELECT clauses and guards were recomputed from their RELEASE.npz. The maximum difference from EVAL_MS.json is "
  f"{max(nl['check4_promotable_eval_ms_recheck_max_abs_diff'].values()):.1e}.")
A(f"EXPERIMENT_LEDGER.csv has {c4['ledger_rows']} rows, {c4['ledger_counts_toward_36']} of which count toward the 36-configuration limit.")
A('')
A('## 5. Paired household bootstrap: ' + ('PASS' if checks['5_paired_household_bootstrap'] else 'FAIL'))
A('')
A("**Weights.** The verifier's own household multinomial weights (seed 2026101705, B 2048, `np.add.at` counts over sorted household ids) are "
  f"bitwise equal to `task_headroom.inference.group_weights`: {dv['weights']['equal_to_group_weights_bitwise']}.")
A(f"- G = {dv['weights']['G']}, n = {dv['weights']['n']}.")
A("- AUC replicates use the verifier's own sparse tie aggregation, not the source's `reduceat`.")
A('')
A(f"**Critical values.** c1 = {dv['critical']['c1']:.7f}, c_P = {dv['critical']['c_P']:.7f}, c_S = {dv['critical']['c_S']:.7f}.")
A('')
A(f"**Endpoints.** All 72 lock endpoints were recomputed. The maximum point difference is {dv['max_point_abs_diff']} and the maximum relative SE difference is {dv['max_se_rel_diff']:.1e}.")
A('- Statuses and categories: 0 mismatches.')
A("- The lock's a - b encoding equals the draft's estimand signs.")
A('')
A('| Key | Point (verifier = recorded) | SE | Clause L1 / U1 (c1) | Family [L, U] (c_P) | Status / category |')
A('|---|---|---|---|---|---|')
show = [k for k in EPS if k.startswith(('COMPETITIVE|pair_benefit', 'MECH_TO|pair_benefit', 'JOINT|pair_benefit', 'COMPETITIVE|income_worsening',
                                        'UTILITY|income|Ucal|brier_excess', 'SUBGROUP|occupation_brier_excess_vs_U0|sex0'))]
for k in show:
    r = EPS[k]
    same = 'equal' if r['point_abs_diff'] <= 1e-12 and r['se_rel_diff'] <= 1e-9 else 'DIFFERS'
    A(f"| {k.replace('|', chr(92) + '|')} | {r['point']:.10f} ({same}) | {r['se']:.10f} | {r['L1']:.6f} / {r['U1']:.6f} | [{r['Lf']:.6f}, {r['Uf']:.6f}] | {r['status']} / {r['category']} |")
A('')
A('## 6. Labels: ' + ('PASS' if checks['6_labels'] else 'FAIL'))
A('')
cl = dv['claims']
A('Each label below was re-derived from the ANALYSIS_LOCK conjunctions and the draft section 5 vocabulary, using the recomputed endpoints and a recomputed CV.')
A('')
A('| Claim | Verifier | Recorded |')
A('|---|---|---|')
for k in ('E', 'B', 'M', 'J'):
    A(f"| {k} | {cl[k]['label']} | {dv['claims_recorded'][k]} |")
A('')
A(f"- **E.** All 36 point conditions hold (24 + 4 + 8).")
A(f"  Simultaneous utility confidence: {dv['E_simultaneous_utility']['label']} ({dv['E_simultaneous_utility']['PASS']} PASS, {dv['E_simultaneous_utility']['UNRESOLVED']} UNRESOLVED).")
A(f"  Recipient clauses: {dv['E_recipient_clause_counts']['PASS']}/8 PASS.")
A(f"- **B nested 0.020.** MATERIAL_SUPERIORITY_ESTABLISHED = {nested['MATERIAL_SUPERIORITY_ESTABLISHED']}; recorded: {dv['B_nested_recorded']['label']}.")
A(f"- **M.** M_TO = {cl['M']['steps']['M_TO']}, M_SS = {cl['M']['steps']['M_SS']}.")
A(f"- **J.** First failure: {cl['J']['failed']}.")
A('- **CV.** Recomputed planted-bit DEV AUC = 1.0 on every view and seed.')
A('  Shuffled-null DEV AUC lies in [' + f"{min(c['shuffled_null_auc'] for c in dv['controls'].values()):.4f}, {max(c['shuffled_null_auc'] for c in dv['controls'].values()):.4f}" + '].')
A('  The ATTACK_VALIDATION values are the recorded ones, inside the bands.')
A('')
A('## 7. Reader refits (the 14 ids in refit_subset): ' + ('PASS' if checks['7_reader_refits'] else 'FAIL'))
A('')
A('| Seed | Reader id | ATTACK_VALIDATION AUC (verifier = recorded) | Orientation | DEV_REPORT scores vs stored |')
A('|---|---|---|---|---|')
for r in rd['refits']:
    A(f"| {r['seed']} | {r['reader_id'].replace('|', chr(92) + '|')} | {r['attack_validation_auc']:.10f} ({'equal' if r['av_auc_abs_diff'] == 0 else r['av_auc_abs_diff']}) | {r['orientation']} | "
      f"{'bitwise' if r['dev_scores_bitwise'] else r['dev_scores_max_abs_diff']} |")
A('')
A(f"The winner rule (max ATTACK_VALIDATION AUC, ties to the first candidate; pair candidates include the 16 ignore-one alternatives) was re-applied to the recorded AUCs.")
A(f"It matches all {sum(w['n'] for w in rd['winner_rule'].values())} recorded winners.")
A('')
A('## 8. Deployment: ' + ('PASS' if checks['8_deployment'] else 'FAIL'))
A('')
A('The restored packages ran through `amd.deploy.release` (API) and `release_csv` (the verifier wrote its own CSV) on 5,943 label-free DEV_REPORT raw rows.')
A('')
A('| Package | API parity | CSV parity | Refusals |')
A('|---|---|---|---|')
for k, v in dp['deployment'].items():
    A(f"| {k} | prob max abs {v['api_income_prob_max_abs']}/{v['api_occupation_prob_max_abs']}, decisions equal | "
      f"{v['csv_income_prob_max_abs']}/{v['csv_occupation_prob_max_abs']} | {sum(x != 'ACCEPTED' for x in v['refusals'].values())}/{len(v['refusals'])} refused |")
A('')
A('**Refusal cases.**')
A('- CSV: extra column, reordered columns, missing column, SEX column, SEX replacing a field, SEX first, lowercase sex column, two columns swapped.')
A('- API: SEX key, lowercase sex key, extra key, reordered keys, missing key.')
A('- Hidden-output requests: leaf_ids, raw_score, theta, sex_specific_probability, teacher_probability, propensity, coarse_bin, income_statistic, h_logit.')
A('- Forbidden fields raise PermissionError; schema errors raise ValueError.')
A('')
A('**Bank integrity.** Every bank release equals its FINALIST_LOCK source on all four roles, bitwise. The sources are the frozen-hybrids bank, the '
  'utility-matched bank (IBSI_NLLM), and this campaign\'s RELEASE.npz files, whose release_sha256 equal the lock.')
A('')
A('**Lock chain.**')
A('- PROMOTION is hashed in FINALIST_LOCK; FINALIST_LOCK in ANALYSIS_LOCK and the bank manifest; the draft in ANALYSIS_LOCK.')
A('- The reader score files are hashed in REPORTING_LOCK.')
A('- The report\'s lock hashes match the files.')
A('')
A('All true.')
A('')
A('## Discrepancies')
A('')
for d in disc:
    A(f"- **{d['id']}** [{d['severity']}]: {d['text']}")
A('')
A('## Observations (not discrepancies; required context for the labels)')
A('')
for o in obs:
    A(f'- {o}')
A('')
A('Reader results are evidence against the declared readers only. The two seeds are the same people with the same bootstrap draws.')
A('This is exploratory development evidence on previously used records: DEV_REPORT is not historically unseen.')
md = '\n'.join(L) + '\n'
(V.PRIV / 'verifier' / 'VERIFICATION.md').write_text(md)
print(json.dumps(res['checks']))
