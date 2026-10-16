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

**Verification.** PENDING: the claims reviewer's independent check of these rows will be recorded here.

## A. Numbers and empirical claims

| # | Location (revised) | Claim as worded | Value | Source and commit | Surface | Attacker access and selection | Utility comparison | Status | Limitation that controls the wording |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Abstract; §1 ¶6; §6.1; Table 1 | LEACE meets its linear bound, yet nonlinear recovery ≈ untreated | B 0.817 / 0.863 vs A 0.815 / 0.870; bound on all 60 maps | MRB `RESEARCH_DECISION.md` §3, `RECOVERY.csv` (C02) | representation | slate refit on `attacker_fit` (7,571 / 6,794), selected on `attacker_val` | probe accuracy (Exp. 1) | registered (primary family of 24) | scope of a linear guarantee, not a violation; HMDA −0.007, interval excludes 0 |
| 2 | §6.1 | held-out ρ₁² near zero | 0.0006 / 0.0001 | MRB `RESEARCH_DECISION.md` §2 (C03) | representation | linear statistic fitted on `attacker_fit`, scored on assessment | — | exploratory | one already-used split |
| 3 | §1 ¶6; §6.1; Table 1; §7 | encoders' stored check fails on **11 of 42** | 11/42 (**corrected from 12**) | MRB `NATIVE_AND_HELDOUT_CHECKS.csv` (`native_status`: 11 "above tau", 31 "passes"); MRB `RESEARCH_DECISION.md:55-61` itemizes 11 and states "12" (C01 inherited the 12) | representation | in-sample ridge on all test rows (15,060 / 13,661), own labels | — | descriptive (historical value reproduced 42/42) | failure of the method's own test; largest pass 0.0438, smallest fail 0.0546 |
| 4 | §6.1 | HMDA underwriting/race fails on two of three seeds | 0.090, 0.081 (s1, s2) | MRB `RESEARCH_DECISION.md:49-52` | representation | as row 3 | — | descriptive | — |
| 5 | §6.1 | FARE certificate unavailable, vacuous or weak; only informative bound 0.175 on a single-cell HMDA configuration | Adult ≈ 0.50; 0.175 | OAR `NATIVE_TEST_VS_RECOVERY.md`; `RESEARCH_DECISION.md` | FARE cell | certification role 1,500 / 1,385 | — | descriptive | certification sizes |
| 6 | Abstract; §6.2 | clean output beside noisy or FARE features restores recovery | 0.778 / 0.758 (noise); 0.788 / 0.768 (FARE) | MRB `RESEARCH_DECISION.md` §4; OAR `RESEARCH_DECISION.md` §3 P5 (C04) | features + untreated full logits | slate + ignore-one candidates on `attacker_val` | — | registered (P5) and exploratory | evaluated treatments only |
| 7 | §6.2 | output-only exceeds the label-only reference | +0.177 / +0.236 | MRB `RESEARCH_DECISION.md` §4 (C05) | full logits | slate; label-only reference reads the true label | — | exploratory | reference is a comparator, not a recipient |
| 8 | §3; Table 3 | exploratory noise-aware likelihood-ratio attackers | Adult 0.546 / 0.563 vs 0.528; HMDA 0.526 / 0.575 vs 0.513 | MRB `RECOVERY.csv` rows `…D_sigma2…LRT_A2/LRT_A4/NL`, `…D_sigma4…` (lines 7057, 7101, 7105, 10155, 10199, 10203) | noisy representation | know σ; A4 fitted on clean auxiliary representations; one draw per record | — | exploratory (unadjusted 90% intervals; no paired test) | not insider, repeated-query or white-box |
| 9 | Table 1 | full logits vs hard decision (frozen head) | +0.263 (0.249), +0.262 (0.254) | OAR `PRIMARY_ENDPOINTS.csv` P1 (C06) | frozen-head output | slate | decisions weakly useful (row 12) | registered (P1) | one-sided Bonferroni percentile bounds |
| 10 | Table 1; §6.4 | withholding the offset lowers recovery | 0.072 [0.062, …] / 0.067 [0.060, …]; per seed 0.079 / 0.064 / 0.072 and 0.058 / −0.000 / 0.143 | ODX `PRIMARY_ENDPOINTS.csv` FC (C07) | full bank vs offset-free bank | bank chosen per encoder seed on `attacker_val` | — | registered (30) | 5 of 6 seeds; 5 of 14 pairs; not for multiclass Adult |
| 11 | §6.4 | 5 of 14 pairs; HMDA pricing passes; refitted head gain 0.083, margin recovery 0.773 | as stated | ODX `S3_ENDPOINTS.csv`; `FROZEN_HEAD_UTILITY.csv`; `OUTPUT_SURFACE_RESULTS.csv` (C10, C11) | outputs | as row 10 | head gain 0.083 | secondary / descriptive | zero by construction for log-probability heads |
| 12 | Table 1; §6.3 | frozen-head usefulness (gain over constant) | 0.039 (0.031) pass; 0.016 (0.0098) not established | ODX `PRIMARY_ENDPOINTS.csv` U-frozen (C08) | frozen head | — | gain target 0.01 | registered | — |
| 13 | §6.3 | hard decisions 0.513 / 0.505; Adult head gains 3–5 points, recalls 14–22%; fair_lending constant decision, centred recovery 0.646 / 0.655 | as stated | OAR P1; ODX `FROZEN_HEAD_UTILITY.csv` (C08, C09) | outputs | slate | as stated | registered / descriptive | low usefulness, not protection |
| 14 | Table 1 | FARE vs LEACE features; zero-fairness tree | 0.550 vs 0.812; 0.501 vs 0.864; F0 0.645 / 0.607 | OAR `RESEARCH_DECISION.md` §3 (C21) | features | slate (+ cell-conditional, descriptive) | probe accuracy | registered (P2) and descriptive | HMDA s1 configuration is a single cell |
| 15 | Abstract; §6.5; Table 2 | useful heads: A 0.774, B 0.759 (difference 0.015 [0.011, 0.019], not established, excludes 0) | as stated | CAP `PRIMARY_USEFUL_HEAD_ENDPOINTS.csv` (C12) | own-head centred output | slate; banks on `attacker_val` | B non-inferior (−0.0006 [−0.0016, 0.0003]) | registered (19) | carried by seed 0; below target |
| 16 | Abstract; §6.5; Table 2 | FARE head 0.539; R(A)−R(F) 0.234 [0.217, 0.252] | as stated | CAP (C13) | own-head output | as row 15 | FARE non-inferiority not established (row 17) | registered | not at matched utility |
| 17 | Abstract; §6.5; Table 2; §7 | FARE accuracy non-inferiority not established | −0.0099 [−0.0152, −0.0046] | CAP (C15) | — | — | margin −0.01 | registered | seed 2 loses 2.0 points; gain and retention pass |
| 18 | Abstract; §1; §6.5; §6.6 | compression ≈ half of FARE's reduction; fairness term 0.109 beyond F0 | 0.125 of 0.234; 0.109 [0.092, 0.126] | CAP (C14) | own-head output | as row 15 | F0 non-inferior (−0.0040 [−0.0086, 0.0005]) | registered | cell-specific (row 20) |
| 19 | §6.5 | utility levels and the retention and gain rows of Table 2 | 0.832 / 0.831 / 0.822 / 0.828 vs 0.749; balanced 0.72–0.73 | CAP `ACTUAL_HEAD_UTILITY.csv` | — | — | as stated | registered / descriptive | — |
| 20 | §6.6 | easier task: compression reproduces most; additional improvement 0.0074 [0.0019, 0.0129] | as stated | ODX `S5_ENDPOINTS.csv` (C17) | complete release | slate + ignore-one; FARE configuration selected on `attacker_val` | FARE 0.974 vs 0.975, constant 0.277 | registered (S5) | smaller difference below target, not "no difference" |
| 21 | §6.5 | post hoc: linear vs nonlinear on the margin; decisions vs FARE scores | 0.515 / 0.773; +0.005 [−0.003, 0.013] (90%) | CAP `report/posthoc_diagnostics.json` (C18, C19) | margin; decision | L family; slate | +0.0099 accuracy | post hoc | consistent with, not establishing |
| 22 | §6.5 | hard decisions 0.536–0.546; complete release A 0.816, B 0.817, F 0.550, F0 0.646; bypass 0.788 / 0.789 | as stated | CAP inference levels; `SECONDARY_COMPLETE_RELEASE_ENDPOINTS.csv` (C16) | outputs; features + output | as row 15 | — | registered / secondary | — |
| 23 | §6.5 | four of ten predictions correct; registration after exploratory full-logit levels disclosed | 4/10 | CAP `USEFUL_HEAD_COMPARISON.md` | — | — | — | descriptive disclosure | — |
| 24 | Abstract; §1; §6.7 | one linked pair beats both singles in every format | 0.917 vs 0.845; 0.883 vs 0.792; 0.611 vs 0.576 (+0.034, lower 0.023) | ODX `S4_ENDPOINTS.csv`, `MULTI_RECIPIENT_RESULTS.md` (C20) | concatenated pair outputs | perfect linkage by aligned rows; bank of pair + single attackers | — | registered (S4) | one coalition; perfect linkage; no synergy claim |
| 25 | §7 | measured gap F0 0.648 vs untreated decision 0.544 | as stated | CAP (C22) | own-head output / decision | as row 15 | accuracy within one point (established) | post hoc, descriptive | no claim that a method exists or is impossible |

## B. Access statements (Section 3, Table 3, Section 5)

| # | Statement | Source | Status |
|---|---|---|---|
| 26 | attacker_fit / attacker_val rows: Exp. 1 Adult 7,571 / 2,239, HMDA 6,794 / 2,089 | MRB `PROTOCOL.md:82-88` | descriptive |
| 27 | Exp. 2–4: Adult 6,065 / 2,235, HMDA 5,388 / 2,082; certification 1,500 / 1,385; Exp. 4 inherits the Exp. 2 roles | OAR `ROLES_AND_SUPPORT.json`; ODX `repaired/ROLES_EXCLUSIONS_REPAIRED.json`; CAP `PROTOCOL_USEFUL_HEADS.md:19` | descriptive |
| 28 | tree and MLP attackers stop early on an internal 10% of attacker_fit | MRB `EFFECTIVE_PROTOCOL.json:489-496,522-525`; `stored_model_eval/recipes.py:71-90` | descriptive |
| 29 | σ\* = smallest σ with attacker_val NL AUC ≤ 0.55 (σ\* = 2 Adult, 4 HMDA) | MRB `EFFECTIVE_PROTOCOL.json:645-652`; `SIGMA_STAR_SUMMARY.json` | descriptive |
| 30 | FARE configuration = admissible (probe accuracy within one point) with the lowest attacker_val NL AUC; the probe is fitted on attacker_fit task labels | OAR `EXECUTION_LOCK.json:291`; `oar/fare_run.py:25-47`; `oar/study.py:325-344` | descriptive |
| 31 | noise is drawn once per record per release seed; three realizations attacked separately, never pooled | MRB `EFFECTIVE_PROTOCOL.json:255-272`; OAR `RELEASE_CONTRACTS.md:5-7` | descriptive |
| 32 | the label-only reference reads the scored record's true task label | `stored_model_eval/bench.py:1045-1051`; MRB `RELEASE_ACCESS_TABLE.csv:26` | descriptive |
| 33 | no orientation step; selection by attacker_val log loss | MRB `EFFECTIVE_PROTOCOL.json:351-565`; ODX `PROTOCOL.md:111-112` | descriptive |
| 34 | roles assigned by a hash of a canonical record key; no household identifiers in Adult or HMDA | ODX `PROTOCOL.md:130-132`; MRB `PROTOCOL.md:95` | descriptive |
| 35 | the pair's linkage is granted by asserted row alignment; the bank includes single-recipient attackers | `odx/run.py:199-214`; ODX `PROTOCOL.md:168-176` | descriptive |
| 36 | the cell-conditional attacker is fitted on some finite releases and equals NL where both were fitted | OAR `EXPLORATORY.csv`; `oar/fare_run.py:91-97` | descriptive |
| 37 | training constraint: per-minibatch one-hot ridge R²; stored check: in-sample ridge on the test split | `pcrl/training/losses.py:475-518`; `pcrl/purposes/verification.py:62-109` | descriptive |

Access facts and line citations: `review/ACCESS_FACTS.md`. The owner re-checked rows 3, 8, 26, 27, 29 and 30 against
the committed files.

## C. Citation claims changed or newly relied on

See `review/CITATION_PRECHECK.md` (claims reviewer, primary sources) for the verdicts and the exact supporting passages.

## D. Distinctions preserved

| Distinction | Where the manuscript keeps it |
|---|---|
| point vs confidence-bound feasibility | utility endpoints are pass / not established by lower bounds (Table 2); FARE "not shown to stay within one point" |
| unresolved benefit vs established shortfall | R(A)−R(B) is "a smaller difference below the registered target", not "no effect"; the hard contrasts are "not established" |
| no demonstrated difference vs equivalence | no equivalence claim anywhere; "excl. 0" is labelled in Table 2 |
| scope-limited guarantee vs attack-based evidence | §6.1 separates the three outcomes; LEACE's result is described as scope, not violation |
| development evidence vs fresh confirmation | abstract, §1 and Limitations: every assessment row was used before |
