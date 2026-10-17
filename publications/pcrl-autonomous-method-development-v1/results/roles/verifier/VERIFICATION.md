# Independent verification: PCRL autonomous method-development campaign (frozen finalist B3CTL_UNCOUPLED_MIX50_C6A)

**Census context.** The records are public-use Florida ACS 2019 person records. The service releases, for each person, an income probability vector (2 classes) and an occupation probability vector (5 classes), each with its argmax decision. The inputs are six raw fields: AGEP, WKHP, COW, SCHL, MAR and POBP. Evaluation readers infer the dataset's recorded SEX column from each output alone and from the two outputs aligned to the same record (the linked pair).

**Who and how.** This is the independent verifier, using separate code in `.private/pcrl_autonomous_method_development_v1/verifier/independent/` (`vcommon.py`, `v_nolabel.py`, `v_readers.py`, `v_dev.py`, `v_deploy.py`, `v_access.py`, `assemble.py`).
- Nothing is imported from `amd.report`, `amd.evaluate`, `amd.promote` or `amd.devrun` for any checked quantity. `amd.deploy` imports `amd.devrun` transitively, but only `release`, `release_csv` and `raw_rows` are called, and they are the objects under test.
- `amd.common` is used only for its hash-checked loaders.
- `task_conditioned_representation.attacks.fit_reader` is used only to construct the refitted readers. Its `predict_reader`, `orientation` and `auc` are not used.
- `task_headroom.inference.group_weights` is called once, to compare the source's bootstrap weights with the verifier's own implementation.

**Units.** Every numerical job ran through `amd.jobqueue`, about 40 CPU-s in total: AMD_VERIFY_NOLABEL_V1, AMD_VERIFY_READERS_V1, AMD_VERIFY_DEVDRY_V1 and AMD_VERIFY_DEVDRY_V2 (invented labels, no truth access), AMD_VERIFY_DEV_V1, AMD_VERIFY_DEPLOY_V1, AMD_VERIFY_ACCESS_V1 and AMD_VERIFY_ASSEMBLE_V1.

**DEV_REPORT truth.** It was re-read once, inside AMD_VERIFY_DEV_V1, through `amd.common.dev_report_truth(..., 'DEV_REPORT_TRUTH_REREAD_BY_AMD_INDEPENDENT_VERIFIER')`. The access log records the event at 2026-10-17T17:56:47+00:00, and both the campaign log and the source log carry it. Labels stayed in memory; only aggregates were written.

**Tolerances.** Outputs and points: bitwise or <= 1e-12. Standard errors: <= 1e-9 relative.

## Verdicts

| Check | Verdict |
|---|---|
| 1_data_roles_and_exposure | **PASS** |
| 2_runtime_outputs | **PASS** |
| 3_utility_and_subgroup | **PASS** |
| 4_selection | **PASS** |
| 5_paired_household_bootstrap | **PASS** |
| 6_labels | **PASS** |
| 7_reader_refits | **PASS** |
| 8_deployment | **PASS** |

## 1. Data roles and exposure: PASS

**DEV_REPORT_REREAD.jsonl** holds exactly two events.
- The coordinator read at 17:45:30Z in unit AMD_REPORT_DEV_V1. This came after REPORTING_LOCK was committed (721d4163d, 17:45:01Z) and published (origin/publication/... 614c9f36b, 17:45:17Z; the REPORTING_LOCK and ANALYSIS_LOCK blobs are identical in both commits).
- The verifier read once, in this verification.
- Both events carry the REPORTING_LOCK sha256 f8d6e354... and appear in the source access log.

**MODEL_SELECT_USE.jsonl** holds 25 entries: 24 from the campaign and 1 from this verifier.
- By queue stage: baselines 2, ctl-taskonly-score 2, famA-score 4, famB-score 10, famC-score 6, verifier 1.
- Every read is a scoring or evaluation stage. No fit stage read MODEL_SELECT labels.

**ATTACK SEX.** Static inspection of every `load_role` call in `amd/*.py` finds one ATTACK SEX read: `readers.py:61`, the final readers.
- That code refuses to run without the committed FINALIST_LOCK, ANALYSIS_LOCK and SERVICE_BANK_MANIFEST.
- The reader units started at 17:43:14Z, after the ANALYSIS_LOCK commit at 17:43:02Z.
- Fitting code (`fam_a`, `fam_b`, `fam_c`) loads ATTACK and DEV_REPORT features only.

**Finalist training exposure.** `fam_b.fit` reads features of every role and CALIBRATION income only.
- Refitting h on CALIBRATION income labels alone, from the verifier's own recomputed design, reproduces w, b, mu and sd with a maximum absolute difference of 0.0, i.e. exactly.
- The coarse thresholds are reproduced from label-free MODEL_FIT log-odds to within 1.11e-15 (summation-order rounding).
- The stored Stage B E20 corrections are admitted copies, hash-checked by `amd.common.admitted`. They used MODEL_FIT labels and SEX in the replication, as RELEASE.json discloses.

**Sealed assessment.** No queue argv mentions SEALED, the source access log has no sealed-assessment event, and the loader refuses the role.

## 2. Runtime outputs: PASS

The verifier used its own oblivious-tree routing from the admitted teacher JSONs (float32 comparison, bit i per split), with sequential accumulation of leaf values and of theta/T. On top of that it applied the stored LOCAL_TREE and JOINT_TREE E20 thetas, MIX = 0.5 L + 0.5 J, the 6 aligned bins (`#thresholds < d`), one-hot minus first, ((Z - mu)/sd) w + b, and softmax(-s/2, s/2). The occupation output is the JOINT_TREE softmax of centred scores.

| Seed | Role | Max abs diff vs bank (income / occupation) | Decisions equal |
|---|---|---|---|
| S0 | MODEL_SELECT (n=8001) | 0.0 / 0.0 (bitwise) | True |
| S0 | DEV_REPORT (n=5943) | 0.0 / 0.0 (bitwise) | True |
| S1 | MODEL_SELECT (n=8001) | 0.0 / 0.0 (bitwise) | True |
| S1 | DEV_REPORT (n=5943) | 0.0 / 0.0 (bitwise) | True |

- S0: the thresholds recomputed from MODEL_FIT features are [-1.631672813114, -0.742758134503, 0.0, 0.513837978684, 1.142964986524]; the recorded values are [-1.631672813114, -0.742758134503, 0.0, 0.513837978684, 1.142964986524], a maximum difference of 3.33e-16.
  The h refit on CALIBRATION matches exactly, and DEV_REPORT has 6 distinct income vectors.
  The occupation output is identical to H_LJ, B2TWIN and H_JJ on MODEL_SELECT and DEV_REPORT, and differs from NEURAL_RAW and TASKONLY.
- S1: the thresholds recomputed from MODEL_FIT features are [-1.607190802667, -0.735047639733, 0.0, 0.51754854877, 1.13838538839]; the recorded values are [-1.607190802667, -0.735047639733, 0.0, 0.51754854877, 1.13838538839], a maximum difference of 1.11e-15.
  The h refit on CALIBRATION matches exactly, and DEV_REPORT has 6 distinct income vectors.
  The occupation output is identical to H_LJ, B2TWIN and H_JJ on MODEL_SELECT and DEV_REPORT, and differs from NEURAL_RAW and TASKONLY.

See D3 for the non-evaluated MODEL_FIT role.

## 3. Utility and subgroup: PASS

**MODEL_SELECT (finalist; the verifier's own loss vectors).** These are compared with EVAL_MS.json.
- S0: all 12 clauses pass at the point, and both guards pass. The maximum absolute difference against EVAL_MS.json is 1.98e-16.
  The worst normalised clause is 0.8731. The guards are men 0.0044173657 (recorded 0.0044173657) and women -0.0049383200 (recorded -0.0049383200).
- S1: all 12 clauses pass at the point, and both guards pass. The maximum absolute difference against EVAL_MS.json is 9.15e-17.
  The worst normalised clause is 0.8835. The guards are men 0.0039406258 (recorded 0.0039406258) and women -0.0030386966 (recorded -0.0030386966).

**DEV_REPORT.** All 24 utility points and the 4 guards are recomputed with SEs. They are listed in INDEPENDENT_VERIFICATION.json, and every one equals the recorded value. The maximum point difference is 0.0 and the maximum relative SE difference is 2.0e-16. The finalist's worst normalised utility clause is 0.9838 (s0) and 0.9888 (s1).

**Comparators' DEV 12-clause point status** (the verifier's recomputation equals the record):

| Seed | B2TWIN | FINALIST | H_JJ | H_LJ | NEURAL_RAW | PUB_IBSI | TASKONLY | T_TREE |
|---|---|---|---|---|---|---|---|---|
| S0 | True | True | False | True | True | True | True | True |
| S1 | True | True | False | True | True | False | True | True |

## 4. Selection (PROTOCOL section 6 over every fitted configuration): PASS

The rule was re-applied to 14 configurations with EVAL_MS.json, controls included, per DEVELOPMENT_REVIEW.md section 0.
- U: 12 clauses at the point. G: 2 guards. R screen: each single-output proxy <= the reference proxy + 0.010, for NEURAL_RAW and H_LJ.
- Pair proxy below both references on both seeds.
- Robustness: every U and G value <= 0.8 x cap.
- Reference bars from BASELINES_MS_S*.json (equal to PROMOTION.json): S0 NEURAL_RAW pair 0.6238, H_LJ pair 0.6224; S1 NEURAL_RAW pair 0.6269, H_LJ pair 0.6233.

**Result.** Promotable: ['B2CTL_UNCOUPLED_C6A', 'B2_CSR_JT_C6A', 'B3CTL_UNCOUPLED_MIX50_C6A']. Robust: none. Choice: **B3CTL_UNCOUPLED_MIX50_C6A**, EDGE_SELECTED. PROMOTION.json records B3CTL_UNCOUPLED_MIX50_C6A, EDGE_SELECTED, n_promotable 3, robust 0. These match.

| Config | Promotable (verifier / recorded) | Mean pair proxy (verifier / recorded) |
|---|---|---|
| CCLC_a2 | False / False | 0.619940 / 0.619940 |
| B2CTL_TASKONLY_TT_C6A | False / False | 0.645858 / 0.645858 |
| B2CTL_UNCOUPLED_C6A | True / True | 0.615333 / 0.615333 |
| B2_CSR_JT_C6A | True / True | 0.613145 / 0.613145 |
| B3CTL_TASKONLY_TR_C6A | False / False | 0.646157 / 0.646157 |
| B3CTL_UNCOUPLED_MIX50_C6A | True / True | 0.600020 / 0.600020 |
| C1_CTRL_TEMP | False / False | 0.649226 / 0.649226 |
| C1_MIX_FULL | False / False | 0.631552 / 0.631552 |

The 3 promotable configurations' MODEL_SELECT clauses and guards were recomputed from their RELEASE.npz. The maximum difference from EVAL_MS.json is 2.0e-16.
EXPERIMENT_LEDGER.csv has 48 rows, 21 of which count toward the 36-configuration limit.

## 5. Paired household bootstrap: PASS

**Weights.** The verifier's own household multinomial weights (seed 2026101705, B 2048, `np.add.at` counts over sorted household ids) are bitwise equal to `task_headroom.inference.group_weights`: True.
- G = 3955, n = 5943.
- AUC replicates use the verifier's own sparse tie aggregation, not the source's `reduceat`.

**Critical values.** c1 = 2.3949991, c_P = 3.3040310, c_S = 3.0252810.

**Endpoints.** All 72 lock endpoints were recomputed. The maximum point difference is 0.0 and the maximum relative SE difference is 2.0e-16.
- Statuses and categories: 0 mismatches.
- The lock's a - b encoding equals the draft's estimand signs.

| Key | Point (verifier = recorded) | SE | Clause L1 / U1 (c1) | Family [L, U] (c_P) | Status / category |
|---|---|---|---|---|---|
| UTILITY\|income\|Ucal\|brier_excess\|s0 | 0.0049189255 (equal) | 0.0019250463 | 0.000308 / 0.009529 | [-0.001441, 0.011279] | UNRESOLVED / UNRESOLVED |
| SUBGROUP\|occupation_brier_excess_vs_U0\|sex0\|s0 | 0.0032351392 (equal) | 0.0020463980 | -0.001666 / 0.008136 | [-0.003526, 0.009997] | UNRESOLVED / UNRESOLVED |
| COMPETITIVE\|pair_benefit_vs_NEURAL_RAW\|s0 | 0.0270257530 (equal) | 0.0056310387 | 0.013539 / 0.040512 | [0.008421, 0.045631] | ESTABLISHED / PASS |
| COMPETITIVE\|income_worsening_vs_NEURAL_RAW\|s0 | -0.0049245641 (equal) | 0.0029598546 | -0.012013 / 0.002164 | [-0.014704, 0.004855] | PASS / PASS |
| COMPETITIVE\|pair_benefit_vs_H_LJ\|s0 | 0.0283917831 (equal) | 0.0038936040 | 0.019067 / 0.037717 | [0.015527, 0.041256] | ESTABLISHED / PASS |
| COMPETITIVE\|income_worsening_vs_H_LJ\|s0 | -0.0168449475 (equal) | 0.0025461691 | -0.022943 / -0.010747 | [-0.025258, -0.008432] | PASS / PASS |
| MECH_TO\|pair_benefit_vs_TASKONLY\|s0 | 0.0515285146 (equal) | 0.0051944466 | 0.039088 / 0.063969 | [0.034366, 0.068691] | ESTABLISHED / PASS |
| JOINT\|pair_benefit_vs_B2TWIN\|s0 | 0.0163974607 (equal) | 0.0027046595 | 0.009920 / 0.022875 | [0.007461, 0.025334] | ESTABLISHED / PASS |
| UTILITY\|income\|Ucal\|brier_excess\|s1 | 0.0049437578 (equal) | 0.0020470216 | 0.000041 / 0.009846 | [-0.001820, 0.011707] | UNRESOLVED / UNRESOLVED |
| SUBGROUP\|occupation_brier_excess_vs_U0\|sex0\|s1 | 0.0039511562 (equal) | 0.0019078875 | -0.000618 / 0.008521 | [-0.002353, 0.010255] | UNRESOLVED / UNRESOLVED |
| COMPETITIVE\|pair_benefit_vs_NEURAL_RAW\|s1 | 0.0198884014 (equal) | 0.0058333652 | 0.005917 / 0.033859 | [0.000615, 0.039162] | ESTABLISHED / PASS |
| COMPETITIVE\|income_worsening_vs_NEURAL_RAW\|s1 | -0.0055523374 (equal) | 0.0032606831 | -0.013362 / 0.002257 | [-0.016326, 0.005221] | PASS / PASS |
| COMPETITIVE\|pair_benefit_vs_H_LJ\|s1 | 0.0249857165 (equal) | 0.0050467876 | 0.012899 / 0.037073 | [0.008311, 0.041660] | ESTABLISHED / PASS |
| COMPETITIVE\|income_worsening_vs_H_LJ\|s1 | -0.0035216495 (equal) | 0.0039070518 | -0.012879 / 0.005836 | [-0.016431, 0.009387] | PASS / PASS |
| MECH_TO\|pair_benefit_vs_TASKONLY\|s1 | 0.0465118809 (equal) | 0.0053982070 | 0.033583 / 0.059441 | [0.028676, 0.064348] | ESTABLISHED / PASS |
| JOINT\|pair_benefit_vs_B2TWIN\|s1 | 0.0099147384 (equal) | 0.0051945838 | -0.002526 / 0.022356 | [-0.007248, 0.027078] | UNRESOLVED / UNRESOLVED |

## 6. Labels: PASS

Each label below was re-derived from the ANALYSIS_LOCK conjunctions and the draft section 5 vocabulary, using the recomputed endpoints and a recomputed CV.

| Claim | Verifier | Recorded |
|---|---|---|
| E | DEV_POINT_ELIGIBLE | DEV_POINT_ELIGIBLE |
| B | SUPERIOR_TO_ALL_ELIGIBLE | SUPERIOR_TO_ALL_ELIGIBLE |
| M | MECHANISM_ATTRIBUTION_ESTABLISHED | MECHANISM_ATTRIBUTION_ESTABLISHED |
| J | JOINT_ADVANTAGE_NOT_ESTABLISHED | JOINT_ADVANTAGE_NOT_ESTABLISHED |

- **E.** All 36 point conditions hold (24 + 4 + 8).
  Simultaneous utility confidence: CONFIDENCE_UNRESOLVED (3 PASS, 25 UNRESOLVED).
  Recipient clauses: 8/8 PASS.
- **B nested 0.020.** MATERIAL_SUPERIORITY_ESTABLISHED = False; recorded: NOT_ESTABLISHED.
- **M.** M_TO = True, M_SS = True.
- **J.** First failure: ['JOINT|pair_benefit_vs_B2TWIN|s1'].
- **CV.** Recomputed planted-bit DEV AUC = 1.0 on every view and seed.
  Shuffled-null DEV AUC lies in [0.5001, 0.5208].
  The ATTACK_VALIDATION values are the recorded ones, inside the bands.

## 7. Reader refits (the 14 ids in refit_subset): PASS

| Seed | Reader id | ATTACK_VALIDATION AUC (verifier = recorded) | Orientation | DEV_REPORT scores vs stored |
|---|---|---|---|---|
| 0 | PAIR__FINALIST__OUT_HGB | 0.5832416566 (equal) | 1 | bitwise |
| 0 | PAIR__FINALIST__LOG_MLP | 0.5928032358 (equal) | 1 | bitwise |
| 0 | PAIR__NEURAL_RAW__OUT_MLP | 0.6216891879 (equal) | 1 | bitwise |
| 0 | PAIR__H_LJ__OUT_MLP | 0.6202786854 (equal) | 1 | bitwise |
| 0 | PAIR__TASKONLY__OUT_HGB | 0.6408085665 (equal) | 1 | bitwise |
| 0 | PAIR__B2TWIN__OUT_HGB | 0.5995324709 (equal) | 1 | bitwise |
| 0 | MARG__FINALIST\|income__LOOKUP | 0.5276315298 (equal) | 1 | bitwise |
| 1 | PAIR__FINALIST__OUT_HGB | 0.6019241441 (equal) | 1 | bitwise |
| 1 | PAIR__FINALIST__LOG_MLP | 0.5873150828 (equal) | 1 | bitwise |
| 1 | PAIR__NEURAL_RAW__OUT_MLP | 0.6233260288 (equal) | 1 | bitwise |
| 1 | PAIR__H_LJ__OUT_MLP | 0.6247089616 (equal) | 1 | bitwise |
| 1 | PAIR__TASKONLY__OUT_HGB | 0.6406518193 (equal) | 1 | bitwise |
| 1 | PAIR__B2TWIN__OUT_HGB | 0.6141944483 (equal) | 1 | bitwise |
| 1 | MARG__FINALIST\|income__LOOKUP | 0.5317042890 (equal) | 1 | bitwise |

The winner rule (max ATTACK_VALIDATION AUC, ties to the first candidate; pair candidates include the 16 ignore-one alternatives) was re-applied to the recorded AUCs.
It matches all 48 recorded winners.

## 8. Deployment: PASS

The restored packages ran through `amd.deploy.release` (API) and `release_csv` (the verifier wrote its own CSV) on 5,943 label-free DEV_REPORT raw rows.

| Package | API parity | CSV parity | Refusals |
|---|---|---|---|
| FINALIST_S0 | prob max abs 0.0/0.0, decisions equal | 0.0/0.0 | 22/22 refused |
| FINALIST_S1 | prob max abs 0.0/0.0, decisions equal | 0.0/0.0 | 22/22 refused |
| TASKONLY_S0 | prob max abs 0.0/0.0, decisions equal | 0.0/0.0 | 22/22 refused |
| TASKONLY_S1 | prob max abs 0.0/0.0, decisions equal | 0.0/0.0 | 22/22 refused |

**Refusal cases.**
- CSV: extra column, reordered columns, missing column, SEX column, SEX replacing a field, SEX first, lowercase sex column, two columns swapped.
- API: SEX key, lowercase sex key, extra key, reordered keys, missing key.
- Hidden-output requests: leaf_ids, raw_score, theta, sex_specific_probability, teacher_probability, propensity, coarse_bin, income_statistic, h_logit.
- Forbidden fields raise PermissionError; schema errors raise ValueError.

**Bank integrity.** Every bank release equals its FINALIST_LOCK source on all four roles, bitwise. The sources are the frozen-hybrids bank, the utility-matched bank (IBSI_NLLM), and this campaign's RELEASE.npz files, whose release_sha256 equal the lock.

**Lock chain.**
- PROMOTION is hashed in FINALIST_LOCK; FINALIST_LOCK in ANALYSIS_LOCK and the bank manifest; the draft in ANALYSIS_LOCK.
- The reader score files are hashed in REPORTING_LOCK.
- The report's lock hashes match the files.

All true.

## Discrepancies

- **D1** [LOW (custody narrative; no numerical effect)]: The statistical reviewer's CLAIMS_AND_ENDPOINTS_DRAFT.md says it was written 'at about 17:50Z, before any finalist freeze', and DEVELOPMENT_REVIEW.md says 'written 17:30-17:50Z'. Git shows DEVELOPMENT_REVIEW.md first committed in ad37c120c (17:37:13Z, the finalist freeze) and the draft first committed in 345e0adb0 (17:43:02Z), whose ANALYSIS_LOCK (utc 17:42:54Z) carries the draft's hash. The self-reported 17:50Z times are impossible. The repository cannot show that the draft existed before the finalist freeze (its first commit is 5 min 49 s after it, and its worked instance names the adjudicated finalist). What is verifiable: the draft hash in ANALYSIS_LOCK equals the current file; the lock was committed before the reader fits (17:43:14Z) and was published (origin/publication 614c9f36b, 17:45:17Z, same blobs as 721d4163d) before the coordinator's DEV_REPORT read (17:45:30Z).
- **D2** [INFO (documentation)]: PROTOCOL.md section 3 gives MODEL_SELECT as 8,000 people; the admitted export, the bank and every evaluation use 8,001 rows.
- **D3** [INFO (numerical fragility on a non-evaluated role)]: The verifier's own recomputation of the finalist on MODEL_FIT (not a released or evaluated role) differs from RELEASE.npz for the people whose MIX income statistic lies within 4.4e-16 of a bin threshold (2 people on seed 0, 4 on seed 1; max probability difference 0.179 / 0.191; decisions unchanged). The thresholds are empirical quantiles of those same MODEL_FIT values, so some people sit exactly on a threshold. A last-bit change in the summation order of the tree scores moves them to the other bin. On MODEL_SELECT and DEV_REPORT the nearest statistic is at least 3.5e-5 from any threshold, and outputs match bitwise.

## Observations (not discrepancies; required context for the labels)

- E holds at the point with very thin utility margins on DEV_REPORT: worst clause = income Brier excess vs Ucal at 0.004919 (s0) and 0.004944 (s1) against the 0.005 cap (0.984 / 0.989 of cap; the registered forecast cited 0.87-0.88 on MODEL_SELECT). Its clause L1 is above 0 on both seeds (0.000308, 0.000041). Simultaneous confidence over the 28 utility and subgroup endpoints is CONFIDENCE_UNRESOLVED (3 PASS, 25 UNRESOLVED, 0 ESTABLISHED_VIOLATION).
- The nested 0.020 target is not established: clause L1 = H_LJ s0 0.019067, H_LJ s1 0.012899, NEURAL_RAW s0 0.013539, NEURAL_RAW s1 0.005917. All are "positive, below the original 0.020 target".
- J fails on JOINT|pair_benefit_vs_B2TWIN|s1 (point 0.009915, clause L1 -0.002526; family status UNRESOLVED). Seed 0 passes (L1 0.009920).
- M_SS is the alias of claim B's H_LJ keys (sigma = 0). The step therefore adds no comparison beyond B. MECHANISM_ATTRIBUTION_ESTABLISHED rests on the task-only control (pair benefit 0.0515 / 0.0465) plus those H_LJ legs.
- The occupation outputs of FINALIST, H_LJ, B2TWIN and H_JJ are bitwise identical on MODEL_SELECT and DEV_REPORT. The occupation recipient keys against H_LJ and B2TWIN therefore have point = SE = 0, which matches the IDENTICAL_RELEASE flags.
- Scope: the non-refit reader scores are the frozen DEV_REPORT_SCORES.npz. The winner rule was re-applied to the recorded ATTACK_VALIDATION AUCs, and the 14-id subset was refitted. The ATTACK_VALIDATION control AUCs are the recorded values. The section 6 proxies are the recorded EVAL_MS.json values; the utility clauses of all 3 promotable configurations were recomputed. The NEURAL_RAW seed-1 package does not exist in the source (coverage gap); deployment check 8 covers FINALIST and TASKONLY only, as briefed.

Reader results are evidence against the declared readers only. The two seeds are the same people with the same bootstrap draws.
This is exploratory development evidence on previously used records: DEV_REPORT is not historically unseen.
