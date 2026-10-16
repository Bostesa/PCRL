# Claim-to-source check for the revised claims

**Scope.** Every substantive claim whose wording changed in `papers/combined_empirical_v2/main.tex`, and every
number in the abstract, the tables and their captions. Sources are the existing claim ledger
(`results/combined_analysis_paper_v1/CLAIM_LEDGER.csv`, ids C01–C26), stored aggregate tables and pinned reports. No
statistic was recomputed from per-record files.

**Abbreviations.**
- `MRB` = `results/combined_matched_removal_benchmark_v1/` @ `70f978ffc` (Experiment 1).
- `OAR` = `results/combined_output_aware_removal_v1/` @ `f7425b15c` (Experiment 2).
- `ODX` = `results/combined_output_diagnosis_v1/` @ `dc6a2e961` (Experiment 3).
- `CAP` = `results/combined_analysis_paper_v1/` (Experiment 4; lock `a0de44953`, results `90253ed39` / `5150f42e8`).

**Status vocabulary.** No claim in the paper is fresh confirmation, because every assessment row was used before.
- *registered*: a pre-registered endpoint with its decision. This is development evidence, not confirmation.
- *exploratory*: a committed value outside a registered family, unadjusted.
- *post hoc*: computed after the results, descriptive.
- *descriptive*: a count, a level or a definition.

**Verification.** The claims and provenance reviewer checked the previous version of this table (SHA-256 prefix `7b5e6f71`) against the committed sources on 2026-10-16 (`review/CLAIMS_REVIEW_DRAFT.md` §C). It confirmed the rows now numbered 2, 4, 7–9, 11, 12, 14, 15, 19–21, 23, 24, 27, 28–37, 39, 41 and 44–46, and flagged the rows now numbered 1, 3, 5, 6, 10, 13, 16–18, 22, 25, 26, 38, 40, 42 and 43 for correction. The owner applied those corrections here; the corrected rows and rows 47–51 were not re-reviewed.

## A. Numbers and empirical claims

| # | Location (revised) | Claim as worded | Value | Source and commit | Surface | Attacker access and selection | Utility comparison | Status | Limitation that controls the wording |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Abstract; §1 ¶5; §6.1; Table 1 | LEACE meets its linear bound, yet nonlinear recovery ≈ untreated | B 0.817 / 0.863 vs A 0.815 / 0.870; bound on all 60 maps (42 target, 18 for an attribute set) | MRB `PRIMARY_ENDPOINTS.csv` l.3–4, l.15–16 (levels); MRB `MATCHED_COMPARISONS.csv` l.396, l.588 (B−A); MRB `VALIDATION.md` l.22 (60 maps) | representation | slate refit on `attacker_fit` (7,571 / 6,794), selected on `attacker_val` | probe accuracy (Exp. 1) | levels registered (family of 24); B−A exploratory (unadjusted 90%) | scope of a linear guarantee, not a violation; HMDA −0.007, unadjusted interval excludes 0 |
| 2 | §6.1 | held-out ρ₁² small | 0.0006 / 0.0001; unadjusted 90% upper ends 0.013 / 0.033 | MRB `RECOVERY.csv` l.6706, l.9634 (C03) | representation | linear statistic fitted on `attacker_fit`, scored on assessment | — | exploratory | one already-used split; "small", not "zero" |
| 3 | §1 ¶5; §6.1; Table 1; §7 | encoders' stored check fails on **11 of 42** | 11/42 (**corrected from 12**) | MRB `NATIVE_AND_HELDOUT_CHECKS.csv` (`native_status`: 11 "above tau", 31 "passes"); MRB `RESEARCH_DECISION.md:55-61` itemizes 11 and states "12" (C01 inherited the 12) | representation | in-sample ridge on all test rows (15,060 / 13,661), own labels | — | descriptive (historical value reproduced 42/42) | failure of the method's own test; largest pass 0.0438, smallest fail 0.0546 |
| 4 | §6.1 | HMDA underwriting/race fails on two of three seeds | 0.090, 0.081 (s1, s2) | MRB `RESEARCH_DECISION.md:49-52` | representation | as row 3 | — | descriptive | — |
| 5 | §6.1 | FARE certificate unavailable, vacuous or weak; only informative bound 0.175 on a single-cell HMDA configuration | Adult ≈ 0.50; 0.175 | OAR `NATIVE_TEST_VS_RECOVERY.md` l.26–39 | FARE cell | certification role 1,500 / 1,385 | — | descriptive | certificates exist only after post-run amendment A1; 0.175 is the secondary race {0, 1, 2} certificate (the primary all-pairs bound on that seed is 3.18, vacuous) |
| 6 | Abstract; §6.2 | clean output beside noisy or FARE features restores recovery | 0.778 / 0.758 (noise); 0.788 / 0.768 (FARE) | MRB `PRIMARY_ENDPOINTS.csv` l.10, l.22; OAR `RESEARCH_DECISION.md` §3 P5; OAR `EXPLORATORY.csv` (C04) | features + clean full logits | slate + ignore-one candidates on `attacker_val` | — | registered (Exp. 1 levels; Exp. 2 contrast P5); FARE-plus-clean levels exploratory | evaluated treatments only; expected, since the output alone recovers 0.776 / 0.768 |
| 7 | §6.2 | output-only exceeds the label-only reference | +0.177 / +0.236 | MRB `RESEARCH_DECISION.md` §4 (C05) | full logits | slate; label-only reference reads the true label | — | exploratory | reference is a comparator, not a recipient |
| 8 | §3; Table 3 | exploratory noise-aware likelihood-ratio attackers | Adult 0.546 / 0.563 vs 0.528; HMDA 0.526 / 0.575 vs 0.513 | MRB `RECOVERY.csv` rows `…D_sigma2…LRT_A2/LRT_A4/NL`, `…D_sigma4…` (lines 7057, 7101, 7105, 10155, 10199, 10203) | noisy representation | know σ; A4 fitted on clean auxiliary representations; one draw per record | — | exploratory (unadjusted 90% intervals; no paired test) | not insider, repeated-query or white-box |
| 9 | Table 1 | full logits vs hard decision (frozen head) | +0.263 (0.249), +0.262 (0.254) | OAR `PRIMARY_ENDPOINTS.csv` P1 (C06) | frozen-head output | slate | decisions weakly useful (row 12) | registered (P1) | one-sided Bonferroni percentile bounds |
| 10 | Table 1; §6.3 | withholding the offset lowers recovery | 0.072 [0.062, …] / 0.067 [0.060, …]; per seed 0.079 / 0.064 / 0.072 and 0.058 / −0.0003 / 0.143 | ODX `PRIMARY_ENDPOINTS.csv` FC (C07); per-seed points from CAP `report/corrections_recomputed.json` l.127 | full bank vs offset-free bank | bank chosen per encoder seed on `attacker_val` | — | contrast registered (family of 30); per-seed points descriptive (Stage A) | 5 of 6 seeds; 5 of 14 pairs; not for multiclass Adult |
| 11 | §6.4 | 5 of 14 pairs; HMDA pricing passes; refitted head gain 0.083, margin recovery 0.773 | as stated | ODX `S3_ENDPOINTS.csv`; `FROZEN_HEAD_UTILITY.csv`; `OUTPUT_SURFACE_RESULTS.csv` (C10, C11) | outputs | as row 10 | head gain 0.083 | secondary / descriptive | zero by construction for log-probability heads |
| 12 | Table 1; §6.3 | frozen-head usefulness (gain over constant) | 0.039 (0.031) pass; 0.016 (0.0098) not established | ODX `PRIMARY_ENDPOINTS.csv` U-frozen (C08) | frozen head | — | gain target 0.01 | registered | — |
| 13 | §6.4 | hard decisions 0.513 / 0.505; Adult head gains 3–5 points, recalls 14–22%; fair-lending constant decision, centred recovery 0.646 / 0.655 | as stated | OAR `EXPLORATORY.csv` l.4, l.56 (hard levels; P1 is the contrast); ODX `FROZEN_HEAD_UTILITY.csv` (utility); ODX `OUTPUT_SURFACE_RESULTS.csv` l.170, l.181 (fair-lending recovery) | outputs | slate | as stated | hard levels exploratory; utility registered / descriptive | low usefulness, not protection |
| 14 | Table 1 | FARE vs LEACE features; zero-fairness tree | 0.550 vs 0.812; 0.501 vs **0.863**; F0 0.645 / **0.608** (rounding corrected) | OAR `EXPLORATORY.csv` l.63, l.78, l.87; `PRIMARY_ENDPOINTS.csv` l.9 (C21) | features | slate (+ cell-conditional, descriptive) | probe accuracy | registered (P2) and descriptive | HMDA s1 configuration is a single cell |
| 15 | Abstract; §6.5; Table 2; §7 | useful heads: A **0.773** (corrected from 0.774; stored 0.773489), B 0.759 (difference 0.015 [0.011, 0.019], not established, excludes 0) | as stated | CAP `PRIMARY_USEFUL_HEAD_ENDPOINTS.csv` l.11 and `report/posthoc_diagnostics.json` l.7–18 @ `90253ed39` (C12; the ledger's `a0de44953` is the lock, not the results) | own-head centred output | slate; banks on `attacker_val` | B non-inferior (−0.0006 [−0.0016, 0.0003]) | registered (19) | carried by seed 0; below target |
| 16 | Abstract; §6.5; Table 2 | FARE head 0.539; R(A)−R(F) 0.234 [0.217, 0.252] | as stated | CAP `PRIMARY_USEFUL_HEAD_ENDPOINTS.csv` @ `90253ed39` (C13) | own-head output | as row 15 | FARE non-inferiority not established (row 17) | registered | not at matched utility |
| 17 | Abstract; §6.5; Table 2; §7 | FARE accuracy non-inferiority not established | −0.0099 [−0.0152, −0.0046] | CAP `PRIMARY_USEFUL_HEAD_ENDPOINTS.csv` @ `90253ed39` (C15) | — | — | margin −0.01 | registered | seed 2 loses 2.0 points; gain and retention pass |
| 18 | Abstract; §1; §6.5; §6.6 | compression ≈ half of FARE's reduction; fairness term 0.109 beyond F0 on the score and 0.096 [0.076, 0.116] on the complete release | 0.125 of 0.234; 0.109 [0.092, 0.126]; 0.0959 [0.0757, 0.1161] (simultaneous over 33 secondary endpoints, z = 3.172) | CAP `PRIMARY_USEFUL_HEAD_ENDPOINTS.csv`; `SECONDARY_COMPLETE_RELEASE_ENDPOINTS.csv` l.11 @ `90253ed39` (C14) | own-head output; complete release | as row 15 | F0 non-inferior (−0.0040 [−0.0086, 0.0005]) | registered (primary; complete release secondary) | setting-specific (row 20) |
| 19 | §6.5 | utility levels and the retention and gain rows of Table 2 | 0.832 / 0.831 / 0.822 / 0.828 vs 0.749; balanced 0.72–0.73 | CAP `ACTUAL_HEAD_UTILITY.csv` | — | — | as stated | registered / descriptive | — |
| 20 | §6.6 | easier task: compression reproduces most; additional improvement 0.0074 [0.0019, 0.0129] (after amendment S1) | as stated | ODX `S5_ENDPOINTS.csv`; `AMENDMENT_S1_2026-10-03.md` (C17) | complete release | slate + ignore-one; FARE configuration selected on `attacker_val` | probe accuracy FARE 0.974 vs 0.975, constant 0.277 (`S5_SUMMARY.json`) | registered (S5) | smaller difference below target, not "no difference" |
| 21 | §6.5 | post hoc: linear vs nonlinear on the margin; decisions vs FARE scores | 0.515 / 0.773; +0.005 [−0.003, 0.013] (90%) | CAP `report/posthoc_diagnostics.json` (C18, C19) | margin; decision | L family; slate | +0.0099 accuracy | post hoc | consistent with, not establishing |
| 22 | §6.5 | hard decisions 0.536–0.546; complete release A 0.816, B 0.817, F 0.550, F0 0.646; bypass 0.788 / 0.789 | as stated | CAP `USEFUL_HEAD_COMPARISON.md` l.77–80; `report/posthoc_diagnostics.json`; `SECONDARY_COMPLETE_RELEASE_ENDPOINTS.csv`; OAR `EXPLORATORY.csv` l.17, l.19, l.29–33 (aliased units) (C16) | outputs; features + output | as row 15 | — | registered / secondary | complete release close to features alone by construction |
| 23 | §6.5 | four of ten predictions correct; registration after exploratory full-logit levels disclosed | 4/10 | CAP `USEFUL_HEAD_COMPARISON.md` | — | — | — | descriptive disclosure | — |
| 24 | Abstract; §1; §6.7 | one linked pair beats both singles in every format | 0.917 vs 0.845; 0.883 vs 0.792; 0.611 vs 0.576 (+0.034, lower 0.023) | ODX `S4_ENDPOINTS.csv`, `MULTI_RECIPIENT_RESULTS.md` (C20) | concatenated pair outputs | perfect linkage by aligned rows; bank of pair + single attackers | — | registered (S4) | one coalition; perfect linkage; no synergy claim |
| 25 | §7 | measured gap: F0 score 0.648 vs equally accurate decisions 0.544 (untreated) and 0.546 (F0's own) | as stated | CAP `report/posthoc_diagnostics.json` l.38 (0.6482); ODX `OUTPUT_SURFACE_RESULTS.csv` l.21 (0.5438); CAP `USEFUL_HEAD_COMPARISON.md` l.80 (F0 decision 0.546) (C22) | own-head output / decision | as row 15 | accuracy within one point (established) | post hoc, descriptive | no claim that a method exists or is impossible |
| 26 | §5 Inference; §8 | post-run changes and replay scope | two lock amendments before the affected fits (Exp. 2 map-identifier fix; Exp. 3 easier-task code); two changes after results (Exp. 2 A1 certification guard; Exp. 3 S1 scoring repair, lower bound 0.0001 → 0.0019); Exp. 2 report-script fix and reporting repairs (amendment R, made at the start of Exp. 3); post hoc, Stage-A and Figure 2 values outside the replay | OAR `ACTIVITY_LOG.md` l.25–34; OAR `RESEARCH_DECISION.md` l.12–13; OAR `NATIVE_TEST_VS_RECOVERY.md:26-32`; ODX `VALIDATION.md` l.10; ODX `RUN_STATUS.json` l.13; ODX `AMENDMENT_S1_2026-10-03.md`, `AMENDMENT_R_2026-10-03.md`; CAP `CLAIM_LEDGER.csv` C09, C11, C18, C19 | — | — | — | descriptive disclosure | — |
| 27 | §8 | about 0.25 CPU-hours | 0.25 (corrected from 0.2) | CAP `COST_AND_STORAGE.md` l.16 | — | — | — | descriptive | — |

## B. Access statements (Section 3, Table 3, Section 5)

| # | Statement | Source | Status |
|---|---|---|---|
| 28 | attacker_fit / attacker_val rows: Exp. 1 Adult 7,571 / 2,239, HMDA 6,794 / 2,089 | MRB `PROTOCOL.md:82-88` | descriptive |
| 29 | Exp. 2–4: Adult 6,065 / 2,235, HMDA 5,388 / 2,082; certification 1,500 / 1,385; Exp. 4 inherits the Exp. 2 roles | OAR `ROLES_AND_SUPPORT.json`; ODX `repaired/ROLES_EXCLUSIONS_REPAIRED.json`; CAP `PROTOCOL_USEFUL_HEADS.md:19` | descriptive |
| 30 | tree and MLP attackers stop early on an internal 10% of attacker_fit | MRB `EFFECTIVE_PROTOCOL.json:489-496,522-525`; `stored_model_eval/recipes.py:71-90` | descriptive |
| 31 | σ\* = smallest σ with attacker_val NL AUC ≤ 0.55 (σ\* = 2 Adult, 4 HMDA) | MRB `EFFECTIVE_PROTOCOL.json:645-652`; `SIGMA_STAR_SUMMARY.json` | descriptive |
| 32 | FARE configuration = admissible (probe accuracy within one point) with the lowest attacker_val NL AUC; the probe is fitted on attacker_fit task labels | OAR `EXECUTION_LOCK.json:291`; `oar/fare_run.py:25-47`; `oar/study.py:325-344` | descriptive |
| 33 | noise is drawn once per record per release seed; three realizations attacked separately, never pooled | MRB `EFFECTIVE_PROTOCOL.json:255-272`; OAR `RELEASE_CONTRACTS.md:5-7` | descriptive |
| 34 | the label-only reference reads the scored record's true task label | `stored_model_eval/bench.py:1045-1051`; MRB `RELEASE_ACCESS_TABLE.csv:26` | descriptive |
| 35 | no orientation step; selection by attacker_val log loss | MRB `EFFECTIVE_PROTOCOL.json:351-565`; ODX `PROTOCOL.md:111-112` | descriptive |
| 36 | roles assigned by a hash of a canonical record key; no household identifiers in Adult or HMDA | ODX `PROTOCOL.md:130-132`; MRB `PROTOCOL.md:95` | descriptive |
| 37 | the pair's linkage is granted by asserted row alignment; the bank includes single-recipient attackers | `odx/run.py:199-214`; ODX `PROTOCOL.md:168-176` | descriptive |
| 38 | the cell-conditional attacker is fitted on some finite releases; in Experiment 2, where both were fitted, it is within 0.003 of NL; Experiment 3–4 values are not committed | OAR `EXPLORATORY.csv` l.4–5, l.26–27, l.35–36, l.78–79, l.87–88; `oar/fare_run.py:91-97`; CAP `INDEPENDENT_VERIFICATION.json` l.295 | descriptive |
| 39 | training constraint: per-minibatch one-hot ridge R²; stored check: in-sample ridge on the test split | `pcrl/training/losses.py:475-518`; `pcrl/purposes/verification.py:62-109` | descriptive |

| 40 | data: UCI Adult; HMDA 2023 California LAR (home purchase, first lien, site-built); six purposes, 14 pairs; training split 24,145 / 63,747, of which `defense_fit` keeps 24,127 / 63,704 after removing the 18 / 43 records that also occur in the test split; test split 15,060 / 13,661 | `pcrl/data/adult.py:68`; `pcrl/data/hmda.py:1,35`; MRB `notes/admission/purposes.json`; MRB `notes/admission/rows_provenance.json` (`n_train`, `n_test`, `train_rows_whose_key_occurs_in_test`); MRB `PROTOCOL.md` l.85, l.89 | descriptive |
| 41 | supported classes: at least 100 / 30 / 100 per class (attacker_fit / attacker_val / assessment); HMDA race classes 3 and 4 not estimable | OAR `PROTOCOL.md:14-19,44` | descriptive |
| 42 | HMDA FARE configuration (Exp. 2 nominees): 4 / 1 / 4 cells; seed 1 is a constant release that keeps none of the gain; the untreated probe's HMDA gain is 0.6–2.3 points | OAR `RESEARCH_DECISION.md` l.100–106; ODX `FARE_USEFUL_TASK_RESULTS.md` l.9–23 and `FARE_REPLAY_EXISTING.csv` l.14–24 (Exp. 3's replay of the Exp. 2 nominees; gain = probe accuracy minus the constant) | descriptive |
| 43 | one-sided Bonferroni families of 24 (Exp. 1) and 12 (Exp. 2), α = 0.05/24 and 0.05/12 | MRB `PROTOCOL.md` l.137; OAR `PROTOCOL.md` l.146–150 | descriptive |
| 44 | registered predictions: 4 correct, 6 wrong, with the breakdown in §6.5 | CAP `PROTOCOL_USEFUL_HEADS.md:107-118`; CAP `PRIMARY_USEFUL_HEAD_ENDPOINTS.csv` | descriptive |
| 45 | the HMDA seed-1 encoder is collapsed (many distinct records share a representation vector) | OAR `NATIVE_TEST_VS_RECOVERY.md:26-28` | descriptive |
| 46 | LRT attribution: the first (noisy releases) 0.546 / 0.526; the second (clean auxiliary representations) 0.563 / 0.575 | MRB `RECOVERY.csv` rows `LRT_A2`, `LRT_A4` | exploratory |
| 47 | the frozen heads of the two primary settings are weak; the employment, education and pricing heads gain 0.69, 0.63 and 0.37 over the constant | ODX `S3_ENDPOINTS.csv` l.19, l.20, l.34; ODX `RESEARCH_DECISION.md` l.29–31 | registered (S3) |
| 48 | easier task: the registered non-inferiority of FARE's probe accuracy passed, −0.0004 [−0.0025, 0.0016] | ODX `S5_ENDPOINTS.csv` l.4 | registered (S5) |
| 49 | the zero-fairness tree's own decisions recover 0.546 | CAP `USEFUL_HEAD_COMPARISON.md` l.80 | registered level |
| 50 | noise ran on three secondary Adult pairs in full and a fourth in part (201 of 648 units) before a budget rule stopped it; eight pairs were not run; none is reported | MRB `RESEARCH_DECISION.md` l.20, l.29–34 | descriptive |
| 51 | Experiment 1's level endpoints are three-way (established above, established below, unresolved) | MRB `PRIMARY_ENDPOINTS.csv` (`decision`) | descriptive |

Access facts and line citations: `review/ACCESS_FACTS.md`. The owner re-checked rows 3, 8, 28, 29, 31 and 32 against the committed files.

## C. Citation claims changed or newly relied on

See `review/CITATION_PRECHECK.md` (claims reviewer, primary sources) for the verdicts and the exact supporting passages.

## D. Distinctions preserved

| Distinction | Where the manuscript keeps it |
|---|---|
| point vs confidence-bound feasibility | utility endpoints are pass / not established by lower bounds (Table 2); FARE "not shown to stay within one point" |
| unresolved benefit vs established shortfall | R(A)−R(B) is "a smaller difference below the registered target", not "no effect"; the hard contrasts are "not established" |
| no demonstrated difference vs equivalence | no formal equivalence claim; the "about as well" statements are descriptive and give their numbers; "excl. 0" is defined in Table 3's caption |
| scope-limited guarantee vs attack-based evidence | §6.1 separates the three outcomes; LEACE's result is described as scope, not violation |
| development evidence vs fresh confirmation | abstract, §1 and Limitations: every assessment row was used before |
