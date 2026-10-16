# Claim source index: combined empirical manuscript (advisor revision, phase 1)

**Role.** Claims and provenance reviewer, 2026-10-16. Read-only on the manuscript and on every result package. This file
suggests; it edits nothing. No priority or novelty claim is suggested anywhere.

**What is indexed.** `papers/combined_empirical_v2/main.tex` as committed at `133d2c740` (459 lines, SHA-256 prefix
`cf0b25f2`): the admitted manuscript of `3b5065805` plus adversary patch v2. All `l.` numbers below refer to that file
(`git show 133d2c740:papers/combined_empirical_v2/main.tex`). While this review ran, the owner committed a revision
(`726ea2907`, "Revise the combined manuscript for review"). It is **not** covered here and is left for phase 2.

**Rules followed.** Only committed reports, aggregate endpoint tables, locks, protocols and code were read. No per-record
file, array, private cache or private inference file was opened; nothing was fitted or recomputed beyond rounding and
simple sums of stored aggregates. Where a check needed a private file, that is stated.

## 1. Source keys

| Key | Path | Commit used |
|---|---|---|
| MRB | `results/combined_matched_removal_benchmark_v1/` (matched removal benchmark) | `70f978ffc` |
| OAR | `results/combined_output_aware_removal_v1/` (output-aware study) | `f7425b15c`; its `RESEARCH_DECISION.md` as repaired at `351116705` (contained in `dc6a2e961`) |
| ODX | `results/combined_output_diagnosis_v1/` (output diagnosis) | `dc6a2e961` |
| CAP-L | `results/combined_analysis_paper_v1/PROTOCOL_USEFUL_HEADS.md`, `LOCK.json` | `a0de44953` (lock) |
| CAP-R | same directory: `PRIMARY_USEFUL_HEAD_ENDPOINTS.csv`, `SECONDARY_COMPLETE_RELEASE_ENDPOINTS.csv`, `ACTUAL_HEAD_UTILITY.csv`, `report/posthoc_diagnostics.json`, `report/corrections_recomputed.json` | `90253ed39` (first commit of these files; unchanged since) |
| CAP-F | same directory: `USEFUL_HEAD_COMPARISON.md`, `VALIDATION.md`, `INDEPENDENT_VERIFICATION.json`, `COST_AND_STORAGE.md`, `CORRECTIONS_AND_SCOPE.md`, `CLAIM_LEDGER.csv` | `5150f42e8` (unchanged at `3b5065805` except backup lines) |
| COR | `publications/pcrl-mechanism-design-v1/results/CORRECTIONS.md`, `CLOSEST_PRIOR_WORK.md` | `c9709a298` |
| CODE | `experiments/run_v2_dataset.py`, `pcrl/models/lora.py` | `b96c412` |

**CSV line convention.** `file.csv l.N` is the physical line (header = line 1); the column is named where it matters.

**Registration labels.** P-MRB = matched-benchmark primary family (24 endpoints, bars 0.55 AUC / τ 0.05 / −1 pp);
P-OAR = output-aware primary (12); P-ODX = output-diagnosis primary (30); S3 (34), S4 (6), S5 (4) = output-diagnosis
secondary families; P-CAP = useful-head primary (19); S-CAP = useful-head secondary (33); EXPL = exploratory,
unadjusted; DESC = descriptive; POST = post hoc; NATIVE = a method's own check; COUNT = a count over stored rows.

**Uncertainty labels.** 1S-PCT = one-sided Bonferroni percentile bound, B = 20,000 (MRB α = 0.05/24; OAR α = 0.05/12);
2S-NORM(z) = simultaneous two-sided 95 % Bonferroni normal interval, B = 1,999 (z = 3.144 P-ODX, 3.180 S3, 2.638 S4,
2.498 S5, 3.008 P-CAP, 3.172 S-CAP); M90 = marginal (unadjusted) 90 % interval; none.

**Discrepancy classes.** N = a printed number or count differs from the stored value; I = interval type, registration
or point-versus-bound; S = evaluated surface or generality; W = wording or disclosure; P = provenance.

## 2. Discrepancies, in order of consequence

| # | Lines | Class | Finding | Evidence |
|---|---|---|---|---|
| D1 | 37, 265, 282 | N | **"12 of 42" is not supported by the stored tables: they show 11 of 42.** | MRB `NATIVE_AND_HELDOUT_CHECKS.csv`, column `native_status`: 11 rows "above tau (historical check fails)" (l.74, 122, 188, 287, 332, 371, 377, 380, 404, 407, 416), 31 "historical check passes". OAR `EXPOSURE_ENDPOINTS.csv` l.124–165: 11 "fails". MRB `RESEARCH_DECISION.md` l.55 says 12, but its own list (l.56–61: 3 + 1 + 2 + 2 + 1 + 2) sums to 11. The 12 propagated to MRB `HANDOFF.json` l.29, `ADVISOR_BRIEF.md` l.44, OAR `RESEARCH_DECISION.md` l.43 and l.48, `EXPOSURE_SENSITIVITY.md` l.60, CAP `CLAIM_LEDGER.csv` C01 and `TITLE_AND_ABSTRACT.md` l.31. The replay checked the 42 values (MRB `VALIDATION.md` l.21), not the count. |
| D2 | 42, 355, 412 | N | Untreated useful-head score recovery printed **0.774**; the stored value is **0.773489 → 0.773**. The manuscript prints 0.773 for the same quantity at l.304 and l.359. | CAP-R `posthoc_diagnostics.json` l.7–11 (nonlinear, arm A, output-only centred: 0.7734894); `PRIMARY_USEFUL_HEAD_ENDPOINTS.csv` l.11 (R(A)−R(B) 0.014811) plus `posthoc_diagnostics.json` l.18 (R(B) 0.758679) gives 0.773490; ODX `OUTPUT_SURFACE_RESULTS.csv` l.18 (refitted head, centred, 0.7734894), a bitwise-identical unit (CAP-F `VALIDATION.md` l.33). CAP-F `USEFUL_HEAD_COMPARISON.md` l.77 prints 0.774 (double rounding 0.77349 → 0.7735 → 0.774); CAP-F `CORRECTIONS_AND_SCOPE.md` l.13 prints 0.773. |
| D3 | 273 | N | Table 1 HMDA LEACE features **0.864**; stored **0.863469 → 0.863**. | OAR `EXPLORATORY.csv` l.63; check: R(F*,rep) 0.500657 (l.78) + P2 0.362812 (OAR `PRIMARY_ENDPOINTS.csv` l.9) = 0.863469. The 0.864 comes from OAR `NATIVE_TEST_VS_RECOVERY.md` l.15. |
| D4 | 274 | N | Table 1 HMDA zero-fairness features **0.607**; stored **0.607524 → 0.608**. | OAR `EXPLORATORY.csv` l.87 (0.6075235). 0.607 is the truncation; the cell-conditional value (l.88, 0.6074) is not the reported attacker. OAR `RESEARCH_DECISION.md` l.111 also prints 0.607. |
| D5 | 271 | N | Table 1 per-seed HMDA s1 printed "−0.000" (negative zero). | CAP-R `corrections_recomputed.json` l.127: −0.000277; print −0.0003. |
| D6 | 447 | N/P | "about 0.2 CPU-hours": the committed figure is **about 0.25 CPU-h** in total. | CAP-F `COST_AND_STORAGE.md` l.16. Fits 501 s user + 73 s sys and controls 40 s (l.11) give ≈ 0.17 CPU-h; 0.2 matches fitting only. |
| D7 | 259–260 | I | Table 1 caption: "simultaneous lower bounds in parentheses where an endpoint was registered" does not describe the table. | Registered endpoints appear without bounds: MRB rows l.266–268 are P-MRB endpoints (level > 0.55, 1S-PCT; lower bounds 0.804 / 0.866, 0.806 / 0.859, noise-plus-outputs 0.764 / 0.749; MRB `PRIMARY_ENDPOINTS.csv` l.3, 15, 4, 16, 10, 22); l.273 rests on P-OAR P2 (0.262 / 0.363, lower 0.247 / 0.356) and l.275 on P5 (0.238 / 0.267, lower 0.223 / 0.258) (OAR `PRIMARY_ENDPOINTS.csv` l.3, 9, 6, 12). The bounds shown are of two kinds: 1S-PCT α = 0.05/12 (l.269) and lower ends of 2S-NORM(3.144) (l.270, 272). Noise rows average 9 encoder × release seed combinations, not three seeds. |
| D8 | 284–285 | I | "the HMDA interval excludes zero": the interval is an **exploratory, unadjusted 90 %** percentile interval, a type §4 never introduces. | MRB `MATCHED_COMPARISONS.csv` l.588 (−0.007006 [−0.007998, −0.005943], label "exploratory (unadjusted)"); l.396 (Adult +0.001963 [−0.000016, 0.003902]). The Adult call is MC-borderline in the replay (MRB `INDEPENDENT_VERIFICATION.json`, non-pass item for Adult B−A rep NL) and its lower bound changes sign under exposure removal (OAR `RESEARCH_DECISION.md` l.44–45). |
| D9 | 283–284 | I | ρ₁² "near zero" rests on points only. The exploratory 90 % upper ends are 0.013 (Adult) and 0.033 (HMDA); the Adult upper end is about half the untreated 0.025. | MRB `RECOVERY.csv` l.6706 (0.000559 [0.000189, 0.012931]), l.9634 (0.000140 [0.0000495, 0.033121]); untreated l.6642 (0.0249), l.9558 (0.2122). |
| D10 | 247 | I | "recovery targets are 0.02 AUC" does not hold for the matched benchmark. Its registered recovery endpoints are levels against a 0.55 bar, and G1 against τ = 0.05. These are the decisions behind Table 1 rows l.266–268. | MRB `PRIMARY_ENDPOINTS.csv` column `bar` (0.55 for recovery, 0.05 for G1, −0.01 for U2); MRB `RESEARCH_DECISION.md` l.94. |
| D11 | 320–322 | I/P | Fig. 2 caption: "90% intervals" are **marginal** ±1.645·SE on recovery only (`report/figures.py` l.34), whereas Table 2 gives simultaneous 95 % intervals. The figure reads an uncommitted private inference file (`figures.py` l.23), contradicting its docstring (l.1, "committed aggregate tables only"). The caption does not name the two panels (centred margin; hard decision) or the x-axis (deployed-head gain over constant). | The centred-panel intervals should equal the M90 rows of CAP-R `posthoc_diagnostics.json` (same bootstrap, z = 1.645); this cannot be confirmed without the private file. Hard-panel intervals for B, F and F0 are in no committed file. |
| D12 | 381–382 | S | The accuracies 0.974 / 0.975 / 0.277 are **U2 refitted-probe** accuracies, not deployed-head accuracies. They sit beside §5.1's deployed-head utility. | ODX `PROTOCOL.md` l.210 ("Acc = U2 probe accuracy on assessment"); ODX `S5_SUMMARY.json` l.77, 87, 97. |
| D13 | 43–44, 379–385, 417 | S | The cross-cell comparison of FARE's compression share mixes surfaces and reference arms. The useful-head cell uses an output-only centred score with A as reference (0.109 of 0.234); the S5 cell uses the complete release (features + own head) with LEACE as reference (0.0074 of 0.151). A like-for-like stored complete-release contrast exists for Adult income/sex. | CAP-R `SECONDARY_COMPLETE_RELEASE_ENDPOINTS.csv` l.11: S-feat+out-full-F0-minus-F 0.0959 [0.0757, 0.1161], 2S-NORM(3.172), PASS, comparable to ODX `S5_ENDPOINTS.csv` l.3. |
| D14 | 368 | S/W | "withholding the offset changes nothing for these heads" sits in the features-plus-output paragraph, but the registered offset endpoints are **output-only**. For B the point is +0.0030 [−0.0009, 0.0068]; for F and F0 it is 0 by construction. | CAP-R `SECONDARY_COMPLETE_RELEASE_ENDPOINTS.csv` l.27–30; CAP-L `PROTOCOL_USEFUL_HEADS.md` l.82; CAP-F `VALIDATION.md` l.35 (WARN 2). The features-plus-output equality is visible only in the private inference file. Safer: "no offset effect was established (none is possible by construction)". |
| D15 | 314–315 | W | "the 19 primary endpoints use the centred and hard formats": 9 of the 19 are format-free utility endpoints. The 10 recovery endpoints use those formats. | CAP-L `PROTOCOL_USEFUL_HEADS.md` l.70–76. |
| D16 | 39, 290–291, 428–429; 45–46 | S | Generality. "bypasses any feature defense" / "Whatever is done to the features" / "any defense is bypassed" was measured for noise, FARE and F0 on two cells. "two recipients recover more than one" rests on **one** Adult pair (income + occupation group), one attribute (race) and three seeds (S4, secondary). | MRB `PRIMARY_ENDPOINTS.csv` l.10, 22; OAR `EXPLORATORY.csv` l.29, 81; CAP-R secondary l.33–34; ODX `MULTI_RECIPIENT_RESULTS.md` l.3–9. |
| D17 | 249–250, 444–445 | W | Pre-fit locks and replay are stated correctly, but the **post-run amendments are not disclosed**: (a) OAR A1, a wrapper duplicate guard; every FARE certificate was UNAVAILABLE in the original run (OAR `NATIVE_TEST_VS_RECOVERY.md` l.26–32). (b) ODX S1, an S5 scoring repair found by the replay that moved the S5 lower bound from 0.0001 to 0.0019 (ODX `AMENDMENT_S1_2026-10-03.md` l.1–24). (c) ODX R, reporting repairs (`AMENDMENT_R_2026-10-03.md`). The pre-fit amendments OAR L1 and ODX L1 are documented in each package. | As listed. |
| D18 | 453–454 | W/I | "All numbers were ... checked by a replay that does not import the analysis code" overclaims. Outside replay scope: the post hoc numbers (0.515, 0.773, +0.005 [−0.003, 0.013], +0.0099); the Stage-A recomputations (per-seed offset points, refitted-head gain); the Fig. 2 intervals. The 12-of-42 count is narrative only. | CAP-F `CLAIM_LEDGER.csv` C18–C19 ("not in replay scope"), C09 and C11 ("recomputed (Stage A)"); CAP-R `corrections_recomputed.json`; `figures.py` l.23. |
| D19 | 139 | W | "Record and **household** identifiers assign roles": Adult has no household IDs and HMDA no applicant IDs. Roles are hash partitions of record keys. | ODX `PROTOCOL.md` l.131–132; OAR `PROTOCOL.md` l.25; MRB `EFFECTIVE_PROTOCOL.json` `roles.hmda_rule`. |
| D20 | 125–129, 223–224 | W | The attacker data budget is said to be fixed by `attacker_fit` / `attacker_val`, but the sizes are never stated. | MRB 7,571 / 2,239 (Adult) and 6,794 / 2,089 (HMDA) (MRB `PROTOCOL.md` l.84, 88); later studies 6,065 / 2,235 and 5,388 / 2,082 (OAR `PROTOCOL.md` l.29–35). Flagged for the owner in COR `CORRECTIONS.md` l.194. |
| D21 | 156–157, 213–214, 238 | W | `defense_fit` is the encoders' own training split (24,127 Adult, 63,704 HMDA rows). LEACE maps, its native bound, FARE trees and release heads are all fitted there. Not stated. | MRB `PROTOCOL.md` l.85, 89, 92; OAR `RESEARCH_DECISION.md` l.181; OAR `PROTOCOL.md` l.31. |
| D22 | 60 | W | "A recipient usually knows whom an output describes": unsourced generalisation; "may know" would match the stated assumptions. | No source in any package; COR `CORRECTIONS.md` l.172 treats linkage as an assumption. |
| D23 | 381, 393, 397 | W | "employment group" vs the task's actual name "occupation group" (Introduction l.57 uses "occupation group"). | MRB `notes/admission/purposes.json` (employment_analysis task `occupation_group`); ODX `MULTI_RECIPIENT_RESULTS.md` l.38. |
| D24 | 37, 282 vs 265 | W | "attribute–seed combinations" vs "pair×seed": the 42 are 14 purpose–attribute pairs × 3 seeds. | MRB `RESEARCH_DECISION.md` l.54. |
| D25 | 283 | W | "all 60 maps" = 42 target + 18 policy-set maps; the paper defines only target LEACE (arm B). | MRB `VALIDATION.md` l.22. |
| D26 | 286–287 | W | The informative 0.175 bound is the **secondary** race {0,1,2} certificate (declared before the run). The primary all-pairs bound on that seed is 3.18 (vacuous). Every value exists only after amendment A1 (D17). | OAR `NATIVE_TEST_VS_RECOVERY.md` l.36–39; OAR `PROTOCOL.md` l.178–179. |
| D27 | 159–160 | W | F0 "a tree of the same budget that only compresses": the γ = 0 tree is still grown on the task label (a task-supervised compression control). | OAR `PROTOCOL.md` l.57. |
| D28 | 222 | S | "Every surface is attacked by the same slate": MRB noise arms also had two Gaussian likelihood-ratio attackers. Table 1's noise values are the NL attacker, so they are unaffected. | MRB `EFFECTIVE_PROTOCOL.json` `attackers.LRT_A2` / `LRT_A4` ("applies_to: noise"); MRB `PROTOCOL.md` l.131. |
| D29 | 150–152 | W | Fig. 1 draws output path (1) only from the LR head on arm features. The frozen-head outputs used in Table 1 (l.269–272) and Fig. 3 (left) are not drawn. The dashed label says "features + clean logits" but the arrow carries logits only. | `papers/combined_empirical_v2/fig_recipient.tex` l.12–24. |
| D30 | 245 | W | Typo: "In all studies and an endpoint passes only if". | — |
| D31 | 188–189 | W (citation) | Stadler et al. are over-applied, and the paper's own stored data contradict the sentence for decisions: hard decisions reveal **less** than the label-only reference. | OAR `EXPLORATORY.csv` l.40 (R(O_hard)−R(LO), Adult −0.0885 [−0.0975, −0.0798], M90) and l.92 (HMDA −0.0162 [−0.0246, −0.0092]); outputs reveal more (l.41, l.93). See `CITATION_PRECHECK.md` item a. |
| D32 | 196–198 | W (citation) | AttriGuard's attacker reads users' public data, not "what a model exposes". | `CITATION_PRECHECK.md` item d. |
| D33 | 170–172 | W (citation) | Song and Shmatikov do not report the in-training adversary's accuracy, so "could not" is shown by Elazar and Goldberg, not by them. | `CITATION_PRECHECK.md` item e. |
| D34 | 199–201 | W (citation) | Li et al.'s guarantee is for linear least-squares (LLSE) reconstruction. | `CITATION_PRECHECK.md` item d. |
| D35 | bib | P | `tian2025fairnessprivacy` is cited as an arXiv preprint (published in IEEE TDSC 22(5):4654–4670); `jia2018attriguard` lacks pages 513–529. | `CITATION_PRECHECK.md` items b, d. |
| D36 | — | P | CAP `CLAIM_LEDGER.csv` has stale or wrong provenance. C12–C16 give the source commit as `a0de44953` (the lock) for `PRIMARY_USEFUL_HEAD_ENDPOINTS.csv`, first committed at `90253ed39`. C02 (0.81–0.87), C06 (HMDA +0.263) and C21 (0.864, 0.607) carry pre-correction or mis-rounded values; C01 says 12/42; C22 cites private "inference levels" although 0.648 and 0.544 follow from committed files. | CAP-F `CLAIM_LEDGER.csv`; `git show --stat a0de44953`. |

**No discrepancy** was found in: Table 2 (all 11 rows); the useful-head utility, recovery-contrast and prediction-count
statements; the offset, coalition, S5 point, refitted-head and fair_lending numbers; the role sizes and exclusions; the
bootstrap and interval descriptions for the four studies (targets aside, D10); and the backbone, retired-bound and
unmerged-fix statements.

## 3. Full index

Columns: lines; claim or value; supporting source(s) (file:line or CSV line/column, commit key); registration and
decision; uncertainty actually used; evaluated surface; discrepancy (see §2) or "none".

### 3.1 Abstract (l.28–49)

| ID | Lines | Claim / value | Source | Reg. / decision | Uncertainty | Surface | Discrepancy |
|---|---|---|---|---|---|---|---|
| A01 | 29 | A multi-purpose model "may release" features and a prediction | COR `CORRECTIONS.md` l.186 (optional v2 hunk) | framing | none | — | none |
| A02 | 32–34 | Methods evaluated; one attacker slate, held-out roles, paired group-bootstrap intervals | OAR `PROTOCOL.md` l.46–57, 97–107; ODX `PROTOCOL.md` l.130–136; CAP-L l.85–89; MRB `PROTOCOL.md` l.133–137 | design | — | — | minor: MRB adds LRT attackers for noise (D28); bootstrap differs by study (stated in §4) |
| A03 | 36–37 | Under official LEACE, nonlinear AUC about 0.81–0.86, against 0.82–0.87 untreated | MRB `PRIMARY_ENDPOINTS.csv` l.4 (0.8174), l.16 (0.8632), l.3 (0.8154), l.15 (0.8702); OAR `EXPLORATORY.csv` l.11 (0.8119), l.63 (0.8635), l.8, l.60 | P-MRB, EST-ABOVE 0.55; OAR levels EXPL | none printed (sources 1S-PCT / M90) | features only, NL | none |
| A04 | 37–38 | Encoders fail their own stored check on **12 of 42** attribute–seed combinations | MRB `NATIVE_AND_HELDOUT_CHECKS.csv` `native_status` (11 fail, 31 pass); OAR `EXPOSURE_ENDPOINTS.csv` l.124–165 | COUNT (native check); not an endpoint | none (in-sample, deterministic) | untreated rep; historical in-sample one-hot ridge R² ≤ 0.05 on the test split | **D1** (11 of 42), D24 |
| A05 | 39 | An unchanged task output bypasses any feature defense; recovery returns to 0.76–0.79 | MRB `PRIMARY_ENDPOINTS.csv` l.10 (0.7784), l.22 (0.7577); OAR `EXPLORATORY.csv` l.29 (0.7878), l.81 (0.7678) | P-MRB EST-ABOVE (noise + outputs); levels EXPL; contrast P-OAR P5 PASS | none printed | features + historical clean logits | D16 ("any") |
| A06 | 40–41 | Frozen heads' hard decisions leak little and are weakly useful | ODX `PRIMARY_ENDPOINTS.csv` l.2 (Adult gain 0.039, PASS), l.8 (HMDA 0.0156, NE); OAR `EXPLORATORY.csv` l.4, 56 (0.513 / 0.505) | P-ODX; levels EXPL | 2S-NORM(3.144) for gains | frozen head, hard | none |
| A07 | 41–42 | LEACE head's score recovers 0.759 (untreated **0.774**); FARE's 0.539; FARE non-inferiority within one point not established | CAP-R `posthoc_diagnostics.json` l.18 (0.7587), l.8 (0.7735), l.28 (0.5390); `PRIMARY_USEFUL_HEAD_ENDPOINTS.csv` l.5 (−0.0099 [−0.0152, −0.0046], NE) | levels DESC; NI P-CAP NE | levels none; NI 2S-NORM(3.008) | output-only centred margin of the deployed LR head | **D2** (0.773) |
| A08 | 43–44 | Same-size tree without the fairness term recovers 0.648; the fairness term removes 0.109 AUC beyond compression; on an easier task compression reproduced most of FARE's effect | CAP-R `posthoc_diagnostics.json` l.38 (0.6482); `PRIMARY_USEFUL_HEAD_ENDPOINTS.csv` l.15 (0.1092 [0.0923, 0.1261], PASS); ODX `S5_ENDPOINTS.csv` l.3 | P-CAP PASS; S5 NE-excl0 | 2S-NORM(3.008); 2S-NORM(2.498) | output-only centred vs complete release | D13; D17 (S5 after amendment S1) |
| A09 | 45–46 | When two recipients link outputs, our attackers recover more from the pair than from either output alone, including decisions only | ODX `S4_ENDPOINTS.csv` l.2–7 (6 PASS; weakest lower bound 0.0234) | S4, 6/6 PASS | 2S-NORM(2.638) | frozen-head outputs of two purposes, linked | D16 (one pair) |
| A10 | 47 | All results are development evidence on repeatedly used data | ODX `PROTOCOL.md` l.6–8; CAP-L l.5–6; OAR `PROTOCOL.md` l.6–7 | scope | — | — | none |
| A11 | 47–48 | No new algorithm, no population certificate, no composition theorem | ODX `MULTI_RECIPIENT_RESULTS.md` l.45–46; OAR `RESEARCH_DECISION.md` l.205 | scope | — | — | none |

### 3.2 Introduction (l.51–102)

| ID | Lines | Claim / value | Source | Reg. / decision | Uncertainty | Surface | Discrepancy |
|---|---|---|---|---|---|---|---|
| I01 | 53–54 | Representation defenses are usually evaluated on the representation | CAP `CLOSEST_PRIOR_ANALYSES.md` §2 table, columns D1/D3 (`5150f42e8`) | literature | — | — | none |
| I02 | 55–59 | Illustration, explicitly not a documented deployment | COR `CORRECTIONS.md` l.174 | framing | — | — | none |
| I03 | 60 | A recipient "usually knows whom an output describes" | none | assertion | — | — | D22 |
| I04 | 64–65 | No source documents a deployment with all these properties | COR `CORRECTIONS.md` l.186 | framing | — | — | none |
| I05 | 72–77 | Four bodies of work, no connecting pipeline; the bodies listed | CAP-F `CORRECTIONS_AND_SCOPE.md` l.20–26; `BOTH_REPOS_EVIDENCE_MAP.csv` | scope | — | — | minor: the source's body (iv) also includes the stored-model pilot |
| I06 | 77–78 | Frozen, seeded, randomly initialised backbone; trained per-purpose adapters and heads; not pretrained | CODE `run_v2_dataset.py` l.161 (`torch.manual_seed`), l.176–180 (`repr_dim = 64`, `StandardEncoder([128,128])`); `lora.py` l.174–176 (freeze); CAP `review/CLAIMS_SCOPE_REVIEW.md` item 4 | code inspection | — | — | none |
| I07 | 85–88 | Common protocol; roles and bootstrap "differ slightly" | MRB `PROTOCOL.md` l.84, 88, 137 vs OAR `PROTOCOL.md` l.29–35 and ODX `PROTOCOL.md` l.133–135 | design | — | — | minor: attacker_fit 7,571 vs 6,065 (Adult); B 20,000 percentile vs 1,999 normal |
| I08 | 89–90 | Every arm releases a deployable head of one common family | CAP-L l.29–38 | design | — | — | none |
| I09 | 91–93 | Offset effect lowered recovery for particular frozen heads, not in general; compression share differs between two cells; decision-level coalition gains | ODX `S3_ENDPOINTS.csv` (5/14 PASS); CAP-R l.15 vs ODX `S5_ENDPOINTS.csv` l.3; ODX `S4_ENDPOINTS.csv` l.6–7 | S3; P-CAP / S5; S4 | as A08/A09 | — | D13 (cell comparison mixes surfaces) |
| I10 | 93–96 | The individual observations are established in prior work | CAP `CLOSEST_PRIOR_ANALYSES.md` §3 items 1–3, 7 | literature | — | — | none |
| I11 | 98–100 | Rows reused; locks do not make them fresh; intervals conditional; recovery is an attacker property | CAP-L l.5–6, 92; ODX `PROTOCOL.md` l.156–158 | scope | — | — | none |
| I12 | 100–102 | No retired R²-to-accuracy bound used; public default interface not repaired; fix on an unmerged branch | CAP `review/CLAIMS_SCOPE_REVIEW.md` item 5; re-checked 2026-10-16 on local remote-tracking refs (not refetched): `origin/main` = `55e4cb1d1` still defines `certified_accuracy_bound` (`pcrl/purposes/verification.py:208`); `origin/fix/retire-accuracy-guarantee` = `5d4eda046` is not an ancestor | scope | — | — | none (as of local refs) |

### 3.3 Setting, threat model and arms (l.104–161)

| ID | Lines | Claim / value | Source | Reg. / decision | Uncertainty | Surface | Discrepancy |
|---|---|---|---|---|---|---|---|
| S01 | 106–107 | Six stored encoders, Round 4, Adult/HMDA × seeds 0–2 | MRB `PROTOCOL.md` l.43–44; `MODEL_MANIFEST.json` | inputs | — | — | none |
| S02 | 107–108 | 64-dim representation per purpose; one frozen task head per purpose | CODE `run_v2_dataset.py` l.176; MRB `PROTOCOL.md` l.117 | inputs | — | — | none |
| S03 | 108–109 | Trained with a linear (one-hot ridge R² ≤ 0.05) constraint | MRB `RESEARCH_DECISION.md` l.46 | inputs | — | — | none |
| S04 | 109–110 | Primary cells Adult income/sex, HMDA underwriting/race; 14 stored purpose–attribute pairs | MRB `notes/admission/purposes.json`; `PERMISSION_TABLE.csv` | design | — | — | none |
| S05 | 112–115 | Three release contracts, (3) as adverse control | OAR `PROTOCOL.md` l.79–85; CAP-L l.52–59 | design | — | — | none |
| S06 | 115–119 | Offset c, margin d; probabilities and decisions do not depend on c | ODX `SCORE_DECOMPOSITION.md` l.14–16; ODX `EXACTNESS.json` (softmax difference 0.0, argmax = 1[d > 0]) | exact check | none | — | none |
| S07 | 121–124 | Recipient receives only the release; no inputs, labels, attribute, parameters or internals | OAR `RELEASE_CONTRACTS.md`; CAP-L l.52–59 | design | — | — | none |
| S08 | 124–125 | Each record released once; no chosen inputs | MRB `PROTOCOL.md` l.53 (one persistent noise draw per person) | design | — | — | none |
| S09 | 125–129 | Supervised attackers use `attacker_fit` (fit) and `attacker_val` (choose and orient), same population; these roles fix the data budget | OAR `PROTOCOL.md` l.29–35, 107; MRB `PROTOCOL.md` l.84, 88 | design | — | — | D20 (sizes not stated) |
| S10 | 130–132 | Models a recipient who knows the attribute for some people, as in Song & Shmatikov and AttriGuard | Song §3.1; AttriGuard §2.1, §3.2 | literature | — | — | none (see `CITATION_PRECHECK.md` d, e) |
| S11 | 133–137 | Sample-free attacks not evaluated; coalition = perfectly linked pair; attackers refitted to defended releases | ODX `PROTOCOL.md` l.168–176; OAR `PROTOCOL.md` l.107 | design | — | — | none |
| S12 | 137–139 | Attackers use no published-objective knowledge and no fitted parameters or gradients | OAR `PROTOCOL.md` l.97–114 | design | — | — | none |
| S13 | 139–140 | "Record and household identifiers assign roles and align releases; never attacker inputs" | ODX `PROTOCOL.md` l.131–132; OAR `PROTOCOL.md` l.25 | design | — | — | **D19** |
| S14 | 156–157 | B = official target LEACE (concept-erasure 0.2.4), fitted on `defense_fit` | MRB `PROTOCOL.md` l.110–113; OAR `PROTOCOL.md` l.53 | design | — | — | D21 (defense_fit = encoder training rows) |
| S15 | 157–159 | F = official FARE; nominee from a frozen grid, chosen on validation within one point of untreated validation accuracy; released as the one-hot cell | OAR `PROTOCOL.md` l.56, 61–77 (admissible if U2 attacker_val accuracy ≥ untreated U2 − 0.01) | design | — | — | minor: "validation accuracy" is the U2 probe's |
| S16 | 159–161 | F0 = same configuration, γ = 0; noise at a validation-selected σ*; no grid expanded, no re-selection | OAR `PROTOCOL.md` l.55, 57; MRB `RESEARCH_DECISION.md` l.37 (σ* 2.0 / 4.0 on attacker_val); CAP-L l.25 | design | — | — | D27 (wording) |

### 3.4 Related work (l.163–209): citation claims

| ID | Lines | Claim | Source | Verdict | Discrepancy |
|---|---|---|---|---|---|
| R01 | 165–168 | INLP, R-LACE aim at linear erasure; LEACE guarantees it in closed form with minimal change | LEACE Def. 2.3, §3, Thm 4.2 (`CITATION_PRECHECK.md` f); INLP/R-LACE per CAP `CLOSEST_PRIOR_ANALYSES.md` §2 | LEACE supported with qualification; INLP/R-LACE not re-verified | none |
| R02 | 168–170 | Guarantees do not extend to nonlinear recipients; kernel erasure does not transfer; LEACE restricts its claim to linear adversaries | LEACE §7 p.10; others per `CLOSEST_PRIOR_ANALYSES.md` §2 | supported | none |
| R03 | 170–173 | Post-hoc attackers recover what in-training adversaries could not (Elazar; Song); post-hoc adversaries are a standard check (Moyer) | Song §4.2, Tables 2–3; Elazar Table 3 per `CLOSEST_PRIOR_ANALYSES.md` | Song qualified | D33 |
| R04 | 175–178 | Confidence-based inversion; label-only matches confidence-based; masking confidences does not stop label-only MIA | `CLOSEST_PRIOR_ANALYSES.md` §2 | not re-verified (outside brief) | none |
| R05 | 178–179 | Wang et al. compare prediction leakage with label leakage at matched performance; we report a label-only reference in that spirit | Wang §3 (`CITATION_PRECHECK.md` c) | supported | none |
| R06 | 180–181 | Multiclass log-linear head can reveal; binary cannot | `CLOSEST_PRIOR_ANALYSES.md` §2; LEACE footnote 5 p.8 | not re-verified | none |
| R07 | 182–183 | Many black-box attribute-inference results do not exceed imputation | Jayaraman abstract, §2.3 (`CITATION_PRECHECK.md` h) | supported with qualification | none |
| R08 | 183–185 | Rounding confidences is an old countermeasure; probabilities leave calibration and proper losses unchanged | Fredrikson 2015 §6 per `CLOSEST_PRIOR_ANALYSES.md`; ODX `EXACTNESS.json` | supported | none |
| R09 | 187–188 | Fair representations trade accuracy for protection when the label depends on the attribute | Zhao & Gordon per `CLOSEST_PRIOR_ANALYSES.md` (bound concerns demographic parity) | loose (prior review O8) | none new |
| R10 | 188–189 | "a release useful for a task reveals what the task's label reveals" (Stadler) | Stadler §1, §3.4, Thm 2 (`CITATION_PRECHECK.md` a) | **not supported as stated** | **D31** |
| R11 | 189–191 | LAFTR lacks guarantees against unseen classifiers; FCRL common protocol | `CLOSEST_PRIOR_ANALYSES.md` §2 (no such LAFTR statement; prior review O7) | not re-verified; previously flagged | none new |
| R12 | 191–194 | FARE: finite partition; certifies DP distance of any classifier on the cell; authors note information loss and compare k-means | FARE Def. 4.1 p.4, §5 p.6, §7 p.7, App. C p.14 (`CITATION_PRECHECK.md` g) | supported with qualification | none |
| R13 | 196–198 | The recipient is the auxiliary-data attacker that reads an attribute "from what a model exposes" (Song; AttriGuard) | Song §3.1; AttriGuard abstract, §3.2 | Song supported; AttriGuard not | **D32** |
| R14 | 198–199 | Combining independent releases is a classical route to breaches | Ganta abstract (`CITATION_PRECHECK.md` i) | supported | none |
| R15 | 199–201 | Differently perturbed copies can be pooled; mechanisms keep the pool no better than the best single copy | Li abstract, §3.2 (`CITATION_PRECHECK.md` d) | supported with qualification (LLSE) | D34 |
| R16 | 201–202 | Combining a biased and a debiased model raises attribute inference | Tian abstract, §6.3 Eq. 12, Table 5 (`CITATION_PRECHECK.md` b) | supported with qualification; bib outdated | D35 |
| R17 | 202–203 | Collusion-aware mechanisms for sequential finite-alphabet releases | COR `CLOSEST_PRIOR_WORK.md` §1 item 3 | not re-verified | none |
| R18 | 205–209 | Bounded-search position; no priority | CAP `CLOSEST_PRIOR_ANALYSES.md` §4 l.160–175 | scope (no change suggested) | none |

### 3.5 Evaluation protocol (l.211–250)

| ID | Lines | Claim / value | Source | Reg. / decision | Uncertainty | Surface | Discrepancy |
|---|---|---|---|---|---|---|---|
| P01 | 213–215 | OAR, ODX and CAP share roles; 20 % head holdout; assessment 5,243 Adult / 4,764 HMDA | OAR `PROTOCOL.md` l.21–35, 42; `RELEASE_CONTRACTS.md` l.39; `ROLES_AND_SUPPORT.json` (head_fit 19,230 / head_val 4,897; assessment 5,243 rows, 5,243 groups); ODX `PROTOCOL.md` l.24–30; CAP-L l.19 | design | — | — | D21 |
| P02 | 216–217 | Training-overlap records (17 Adult, 42 HMDA) excluded from every scored role; 20 % of attacker_fit groups reserved for certification | OAR `PROTOCOL.md` l.23–27, 32; ODX `AMENDMENT_R_2026-10-03.md` l.11 (R1: 17/17 and 42/42) | design | — | — | none |
| P03 | 217–218 | No sensitive assessment label selects a release, grouping, penalty, attacker or probe | OAR `PROTOCOL.md` l.72–77, 107; ODX `PROTOCOL.md` l.188; MRB `RESEARCH_DECISION.md` l.37 | design | — | — | none |
| P04 | 218–220 | MRB used uncleaned roles (5,250 / 4,778), overlap retained and disclosed; decisions stable when removed | MRB `PROTOCOL.md` l.84, 88, 93–94; OAR `RESEARCH_DECISION.md` l.40–45 (24/24 primary STABLE; 42/42 native categories STABLE; 97/98 exploratory rows) | sensitivity | — | — | none |
| P05 | 222–224 | Slate: LR (5 C values), GBT (20), MLP (18), fitted on attacker_fit | MRB `EFFECTIVE_PROTOCOL.json` `attackers` (C list 5, GBT configs 20, MLP 18); OAR `PROTOCOL.md` l.97–100; CAP-L l.61 | design | — | — | D28 |
| P06 | 224–226 | NL = better of GBT/MLP on attacker_val log loss; refitted at three attacker seeds | OAR `PROTOCOL.md` l.103–104; MRB `EFFECTIVE_PROTOCOL.json` `attackers.NL`, `attacker_seeds` | design | — | — | none |
| P07 | 226–227 | Recovery = macro OvR AUC on assessment over supported classes, averaged over attacker then encoder seeds | OAR `PROTOCOL.md` l.132–133; MRB `VALIDATION.md` l.85 | design | — | — | minor: noise arms also average release seeds |
| P08 | 227–229 | Plus-surfaces admit ignore-one-part candidates; the pair bank contains each single recipient's attacker | OAR `PROTOCOL.md` l.105; ODX `PROTOCOL.md` l.173 | design | — | — | none |
| P09 | 229–231 | Cell-conditional attacker for finite releases, descriptive; format banks select on validation | OAR `PROTOCOL.md` l.109–114; CAP-L l.65–66; ODX `PROTOCOL.md` l.85–89 | DESC | — | — | none |
| P10 | 233–235 | Utility of the deployed head (accuracy, balanced accuracy, minority recall, log loss, gain over the attacker_fit majority constant); probe reported separately | CAP-L l.46–50; CAP-R `ACTUAL_HEAD_UTILITY.csv` | design | — | — | note: S5 accuracy is the probe's (D12) |
| P11 | 237–239 | Native checks: in-sample ridge R² ≤ 0.05; LEACE whitened residual cross-covariance below tolerance on defense_fit; FARE DP certificate on the certification role | MRB `RESEARCH_DECISION.md` l.46, 66 (svd_tol 0.01); OAR `PROTOCOL.md` l.170–182 | NATIVE | — | — | D21 |
| P12 | 241 | Each endpoint is a seed mean of a per-seed statistic | ODX `PROTOCOL.md` l.106–107; CAP-L l.70 | design | — | — | none |
| P13 | 241–245 | ODX/CAP: B = 1,999 paired multinomial group bootstrap, simultaneous two-sided 95 % Bonferroni normal; MRB/OAR: one-sided Bonferroni percentile, B = 20,000 | ODX `PROTOCOL.md` l.130–136; CAP-L l.85–90; MRB `PROTOCOL.md` l.133–137; OAR `PROTOCOL.md` l.146–150 | design | as stated | — | none; D30 (typo) |
| P14 | 245–247 | Utility targets −0.01, 0.03, retention > 0; recovery targets 0.02 AUC; earlier frozen-head usefulness target 0.01 | CAP-L l.71–76; ODX `PROTOCOL.md` l.117–118 | design | — | — | D10 (MRB bars 0.55 / τ) |
| P15 | 247–248 | Excludes-zero-not-target is reported as "a smaller difference below the target" | CAP-L l.91 | wording rule | — | — | none (the paper complies; see D14 for "changes nothing") |
| P16 | 249–250 | Protocol, code, inputs, unit list and families committed and pushed before first fit; independent replay recomputed every primary result | MRB `VALIDATION.md` l.9, 44–48 (183/183 primary PASS; 185 exploratory FAILs diagnosed); OAR `INDEPENDENT_VERIFICATION.json` (458 PASS / 4 reporting FAILs); ODX `INDEPENDENT_VERIFICATION.json` (30 primary slots); CAP-F `VALIDATION.md` l.11, 25 (35 PASS / 0 FAIL) | provenance | — | — | D17 (post-run amendments undisclosed) |

### 3.6 Table 1 (l.257–278)

| ID | Line | Row / value | Source | Reg. / decision | Uncertainty | Surface | Discrepancy |
|---|---|---|---|---|---|---|---|
| T00 | 259–260 | Caption | — | — | — | — | **D7** |
| T01 | 265 | Own linear check fails, **12 of 42** | as A04 | COUNT | none | untreated rep, in-sample | **D1** |
| T02 | 266 | Untreated NL 0.815 / 0.870 | MRB `PRIMARY_ENDPOINTS.csv` l.3 (0.81540, lower 0.80396), l.15 (0.87024, lower 0.86638) | P-MRB EST-ABOVE 0.55 | 1S-PCT (not shown) | rep, MRB roles | D7 |
| T03 | 267 | LEACE NL 0.817 / 0.863 | l.4 (0.81736, lower 0.80557), l.16 (0.86323, lower 0.85898) | P-MRB EST-ABOVE | 1S-PCT (not shown) | rep, MRB roles | D7 |
| T04 | 268 | Noise σ*: features 0.528 / 0.513; plus clean outputs 0.778 / 0.758 | l.6 (0.52840, EST-BELOW), l.18 (0.51304, EST-BELOW), l.10 (0.77842, lower 0.76381), l.22 (0.75770, lower 0.74945) | P-MRB | 1S-PCT (not shown) | rep; rep + clean logits | D7 (9 seed combinations) |
| T05 | 269 | Full vs hard: 0.776 vs 0.513, +0.263 (0.249); 0.768 vs 0.505, +0.262 (0.254) | OAR `PRIMARY_ENDPOINTS.csv` l.2 (0.26278, lower 0.24924), l.8 (0.26247, lower 0.25395); levels `EXPLORATORY.csv` l.2, 4, 54, 56 | P-OAR P1 PASS | 1S-PCT α = 0.05/12 | frozen head, output-only | none (rounded levels differ by 0.263 on HMDA; the endpoint is 0.262) |
| T06 | 270 | Offset: +0.072 (0.062) / +0.067 (0.060) | ODX `PRIMARY_ENDPOINTS.csv` l.4 (0.07185, lower 0.06227), l.10 (0.06702, lower 0.05954) | P-ODX PASS | lower end of 2S-NORM(3.144) | frozen head; full bank vs ignore-offset bank | none |
| T07 | 271 | Per seed 0.079 / 0.064 / 0.072; 0.058 / −0.000 / 0.143 | CAP-R `corrections_recomputed.json` l.105, 110, 115, 122, 127, 132 | DESC (Stage-A points from private predictions) | none | as T06 | **D5**; not replayed (D18) |
| T08 | 272 | Frozen gain (target 0.01): 0.039 (0.031); 0.016 (0.0098), not established | ODX `PRIMARY_ENDPOINTS.csv` l.2 (0.03929, lower 0.03058, PASS), l.8 (0.01560, lower 0.00978, NE) | P-ODX | lower end of 2S-NORM(3.144) | frozen head accuracy | none (Adult PASS not labelled) |
| T09 | 273 | FARE vs LEACE features: 0.550 vs 0.812; 0.501 vs **0.864** | OAR `EXPLORATORY.csv` l.26 (0.55009), l.11 (0.81190), l.78 (0.50066), l.63 (0.86347); contrast P2 `PRIMARY_ENDPOINTS.csv` l.3, 9 | levels EXPL; P2 PASS | M90 (levels); 1S-PCT (P2, not shown) | features only, OAR roles | **D3**; D7 |
| T10 | 274 | Zero-fairness tree features 0.645 / **0.607** | OAR `EXPLORATORY.csv` l.35 (0.64496), l.87 (0.60752) | EXPL / secondary control | M90 | features only | **D4** |
| T11 | 275 | FARE + historical clean outputs 0.788 / 0.768 | OAR `EXPLORATORY.csv` l.29 (0.78780), l.81 (0.76779); P5 `PRIMARY_ENDPOINTS.csv` l.6, 12 | levels EXPL; P5 PASS | M90 (levels) | FARE cell + clean logits | D7 |

### 3.7 Results text before §5.1 (l.280–304)

| ID | Lines | Claim / value | Source | Reg. / decision | Uncertainty | Surface | Discrepancy |
|---|---|---|---|---|---|---|---|
| N01 | 280–281 | Three outcomes kept apart | OAR `RESEARCH_DECISION.md` l.16–19 | framing | — | — | none |
| N02 | 281–283 | **12 of 42** untreated combinations fail, including HMDA underwriting/race on 2 of 3 seeds | as A04; HMDA s1 0.0900, s2 0.0807 (MRB `NATIVE_AND_HELDOUT_CHECKS.csv` l.380, 416) | COUNT | none | untreated rep | **D1**, D24 |
| N03 | 283 | Official LEACE meets its bound on all 60 maps | MRB `VALIDATION.md` l.22; `RESEARCH_DECISION.md` l.66–70 | NATIVE, all pass | none (tolerance 0.01) | defense_fit | D25 |
| N04 | 283–284 | Held-out ρ₁² near zero (0.0006, 0.0001), one already-used split | MRB `RECOVERY.csv` l.6706, 9634 | EXPL | M90 (not stated) | rep, assessment | D9 |
| N05 | 284–285 | NL recovery changes < 0.01 AUC (Adult +0.002, HMDA −0.007); HMDA interval excludes zero | MRB `MATCHED_COMPARISONS.csv` l.396, 588 | EXPL (decision vs 0: Adult UNRESOLVED, HMDA ESTABLISHED_BELOW) | M90, unadjusted | rep, NL, B−A | **D8** |
| N06 | 285–287 | FARE certificate unavailable, vacuous or weak (Adult ≈ 0.50); only informative bound 0.175 on a degenerate single-cell HMDA nominee | OAR `NATIVE_TEST_VS_RECOVERY.md` l.36–45; `RESEARCH_DECISION.md` l.123–133 | NATIVE | certificate δ = 0.05 | FARE cells, certification rows | D17, D26 |
| O01 | 289–290 | Output-only recovery exceeds label-only by 0.177 / 0.236 | MRB `MATCHED_COMPARISONS.csv` l.468 (0.17703), l.660 (0.23628 [0.22712, 0.24534]); `RESEARCH_DECISION.md` l.127–129 | EXPL | M90 (not printed) | clean frozen-head logits vs true-label reference, MRB roles | none (cleaned-role analogue 0.174 / 0.246: OAR `EXPLORATORY.csv` l.41, 93) |
| O02 | 290–291 | Releasing the clean output restores recovery to 0.76–0.79 whatever is done to features (noise, FARE) | as A05 | as A05 | none | as A05 | D16 |
| O03 | 291 | Changing features cannot remove what an unchanged output carries | logic; OAR `RESEARCH_DECISION.md` l.118–119 | — | — | — | none |
| U01 | 293–294 | Frozen hard decisions leak little (0.513 / 0.505) | OAR `EXPLORATORY.csv` l.4, 56; ODX `OUTPUT_SURFACE_RESULTS.csv` l.7, 109 | EXPL levels | none | frozen head, hard | none |
| U02 | 294–295 | Adult frozen head gains 3–5 accuracy points; recalls 14–22 % of high earners | ODX `FROZEN_HEAD_UTILITY.csv` l.2, 5, 8 (gain 0.0366 / 0.0334 / 0.0479; minority recall 0.2228 / 0.1399 / 0.2122) | DESC per seed; mean P-ODX PASS | none | frozen head | none |
| U03 | 295 | HMDA underwriting usefulness not established | ODX `PRIMARY_ENDPOINTS.csv` l.8 | P-ODX NE (lower 0.0098 < 0.01) | 2S-NORM(3.144) | frozen head | none |
| U04 | 295–297 | fair_lending predicts one class for everyone; scores vary; centred recovery race 0.646, sex 0.655 | ODX `FROZEN_HEAD_UTILITY.csv` l.29–31 (`is_constant_prediction` True); `OUTPUT_SURFACE_RESULTS.csv` l.170 (0.64588), l.181 (0.65542) | DESC / EXPL | none (M90 in source) | frozen head, centred logits | none |
| U05 | 297 | Low decision-level recovery coincides with decisions barely beating a constant | U01–U03 | interpretation | — | — | none |
| C01 | 299–300 | Withholding the offset changes no probability, decision or task loss | ODX `EXACTNESS.json`; `SCORE_DECOMPOSITION.md` l.14–16 | exact check | none | — | none |
| C02 | 300 | Lowered recovery by 0.072 / 0.067 AUC | as T06 | P-ODX PASS | 2S-NORM(3.144) | frozen head banks | none |
| C03 | 300–301 | On 5 of 6 primary seeds, not on collapsed HMDA s1 | CAP-R `corrections_recomputed.json` l.105–132; CAP-F `CORRECTIONS_AND_SCOPE.md` l.11 | DESC (points) | none | as T06 | none |
| C04 | 301 | On 5 of 14 stored pairs | ODX `S3_ENDPOINTS.csv` l.2, 4, 21, 23, 25 PASS; CAP-R `corrections_recomputed.json` l.76–78 | S3, 5 PASS / 9 NE | 2S-NORM(3.180) | frozen heads | none |
| C05 | 301–302 | Not established for the six multiclass Adult pairs; passed for 5-class HMDA pricing/race | ODX `S3_ENDPOINTS.csv` l.6, 8, 10, 12, 14, 16 (NE), l.25 (0.0473 [0.0411, 0.0536], PASS); `FROZEN_HEAD_UTILITY.csv` l.26–28 (K = 5) | S3 | 2S-NORM(3.180) | frozen heads | none (employment/race −0.0172 excludes zero on the negative side: an attacker-selection effect, CAP `review/CLAIMS_SCOPE_REVIEW.md` §C) |
| C06 | 302–303 | For a refitted LR head the offset is a function of the margin; effect zero by construction | CAP-L l.40–44; ODX `PROTOCOL.md` l.93–94; measured: ODX `OUTPUT_SURFACE_RESULTS.csv` l.22–23 (0.77363 vs 0.77346); CAP-R secondary l.27 (−0.000165 [−0.001268, 0.000937]) | secondary stratum, DESC | none | refitted head HEAD__A | none |
| C07 | 303–304 | That head is about twice as useful (gain 0.083); its margin alone recovers 0.773 | CAP-R `corrections_recomputed.json` l.55 (0.08265), vs frozen 0.03929 (ODX `PRIMARY_ENDPOINTS.csv` l.2): ratio 2.10; ODX `OUTPUT_SURFACE_RESULTS.csv` l.18 (0.77349) | DESC | none | refitted head, centred margin | none (but see D2: same value printed 0.774 elsewhere) |

### 3.8 §5.1 Matched useful-head comparison (l.306–375) and Table 2

| ID | Lines | Claim / value | Source | Reg. / decision | Uncertainty | Surface | Discrepancy |
|---|---|---|---|---|---|---|---|
| UD01 | 308–310 | Common LR family; grid fitted on defense_fit minus holdout, selected on holdout log loss; runtime input = arm's features | CAP-L l.29–38 | design | — | — | none |
| UD02 | 310 | Heads, LEACE maps, FARE trees reused unchanged | CAP-L l.21–25; CAP-F `VALIDATION.md` l.7 | custody | — | — | none |
| UD03 | 310–312 | Full-logit output-only, features-only, features-plus-full-output and bypass surfaces reused and hash-verified; levels already reported | CAP-F `USEFUL_HEAD_COMPARISON.md` l.16–25; OAR `EXPLORATORY.csv` l.17–20, 29–34 (committed at `4bdd93043`, before the lock) | disclosure | — | — | none |
| UD04 | 312–313 | New: 84 attacker units (3,924 fits), after the lock | CAP-F `USEFUL_HEAD_COMPARISON.md` l.14; `COST_AND_STORAGE.md` l.17; `VALIDATION.md` l.11 | provenance | — | — | none |
| UD05 | 313–314 | Full logits, centred margin and probabilities carry identical information (10⁻¹²) | CAP-L l.40–44; CAP-F `VALIDATION.md` l.8 | exact check | none | — | none |
| UD06 | 314–315 | "the 19 primary endpoints use the centred and hard formats" | CAP-L l.70–76 | — | — | — | D15 |
| TT01 | 333 | Acc(B)−Acc(A) −0.0006 [−0.0016, 0.0003] pass | CAP-R `PRIMARY_USEFUL_HEAD_ENDPOINTS.csv` l.2 | P-CAP PASS (NI) | 2S-NORM(3.008) | deployed head accuracy | none |
| TT02 | 334 | Acc(F)−Acc(A) −0.0099 [−0.0152, −0.0046] n.e. | l.5 | P-CAP NE | 2S-NORM(3.008) | deployed head | none |
| TT03 | 335 | Acc(F0)−Acc(A) −0.0040 [−0.0086, 0.0005] pass | l.8 | P-CAP PASS | 2S-NORM(3.008) | deployed head | none |
| TT04 | 336 | Gain B/F/F0 0.082 / 0.073 / 0.079; ≥ 0.067 / 0.057 / 0.063 | l.3, 6, 9 | P-CAP PASS ×3 | lower ends of 2S-NORM | deployed head | none |
| TT05 | 337 | Retention 0.016 / 0.007 / 0.013; ≥ 0.013 / 0.001 / 0.007 | l.4, 7, 10 | P-CAP PASS ×3 | lower ends | deployed head | none |
| TT06 | 339 | Centred R(A)−R(B) 0.015 [0.011, 0.019], n.e. (excl. 0) | l.11 (0.014811 [0.010540, 0.019081]) | P-CAP NE-excl0 | 2S-NORM(3.008) | output-only centred | none |
| TT07 | 340 | R(A)−R(F) 0.234 [0.217, 0.252] pass | l.12 | P-CAP PASS | 2S-NORM | output-only centred | none |
| TT08 | 341 | R(A)−R(F0) 0.125 [0.112, 0.139] pass | l.13 | P-CAP PASS | 2S-NORM | output-only centred | none |
| TT09 | 342 | R(B)−R(F) 0.220 [0.201, 0.238] pass | l.14 | P-CAP PASS | 2S-NORM | output-only centred | none |
| TT10 | 343 | R(F0)−R(F) 0.109 [0.092, 0.126] pass | l.15 | P-CAP PASS | 2S-NORM | output-only centred | none |
| TT11 | 344 | Hard, all five: −0.002 to 0.010, lower bounds < 0, n.e. | l.16–20 (points −0.002045 … 0.009670; lowers −0.006317 … −0.001209) | P-CAP NE ×5 | 2S-NORM | output-only hard | none |
| TT00 | 327–328 | Table 2 caption (19 endpoints; z = 3.008; output-only; n.e.; excl. 0) | CAP-L l.88 (z = 3.007787) | — | — | — | minor: the caption does not say "simultaneous two-sided 95 %", and gain and retention rows show only lower ends |
| UT01 | 349–351 | Accuracy 0.832 / 0.831 / 0.822 / 0.828; constant 0.749; balanced 0.72–0.73; about half of high earners recalled | CAP-R `ACTUAL_HEAD_UTILITY.csv` l.2–13 (seed means 0.8318 / 0.8312 / 0.8219 / 0.8278; constant 0.749189; balanced 0.723 / 0.722 / 0.721 / 0.728; recall 0.506 / 0.503 / 0.519 / 0.527); CAP-F `USEFUL_HEAD_COMPARISON.md` l.42–45 | DESC | none | deployed head | none |
| UT02 | 351–353 | B and F0 non-inferior; FARE not (lower −0.015) because seed 2 loses 2.0 points; gain and retention pass, the latter narrowly | `PRIMARY_USEFUL_HEAD_ENDPOINTS.csv` l.2, 5 (seed 2 −0.020408), 7 (lower 0.000641), 8 | P-CAP | 2S-NORM(3.008) | deployed head | none |
| RS01 | 355–356 | Untreated score recovers **0.774**; LEACE head 0.759 | as A07 | DESC levels | none | output-only centred | **D2** |
| RS02 | 356 | Smaller difference 0.015 below target, carried by seed 0 alone | l.11 (seeds 0.0486 / −0.0002 / −0.0039) | P-CAP NE-excl0 | 2S-NORM | output-only centred | none |
| RS03 | 357 | FARE head 0.539; zero-fairness tree 0.648 | CAP-R `posthoc_diagnostics.json` l.28, 38 | DESC | none | output-only centred | none |
| RS04 | 357–358 | Of 0.234, compression 0.125 and fairness term 0.109, on every seed | l.12, 13, 15 (F0−F seeds 0.113 / 0.106 / 0.109) | P-CAP PASS | 2S-NORM; seeds are points | output-only centred | none |
| RS05 | 358–359 | Post hoc: linear attacker on the untreated margin 0.515; nonlinear 0.773 | CAP-R `posthoc_diagnostics.json` l.2–11 (0.51509 [0.51067, 0.51951]; 0.77349) | POST, DESC | M90 (not printed) | output-only centred | D2 (0.773 vs 0.774 four lines earlier); outside replay (D18) |
| RS06 | 359–360 | Consistent with, not establishing, nonlinear residual outside a linear eraser's scope | CAP-F `USEFUL_HEAD_COMPARISON.md` l.106 | interpretation | — | — | none |
| RD01 | 362 | Decision-only: every arm 0.536–0.546; no contrast established | CAP-F `USEFUL_HEAD_COMPARISON.md` l.77–80; A 0.54375 (ODX `OUTPUT_SURFACE_RESULTS.csv` l.21; CAP-R posthoc l.43 + l.28); B, F, F0 from the hard contrasts (l.16–18): 0.5441 / 0.5361 / 0.5458 | levels DESC; contrasts P-CAP NE | — | output-only hard | none |
| RD02 | 363 | The decision has the score's accuracy | ODX `PROTOCOL.md` l.55–56; OAR `PROTOCOL.md` l.124–125 | by construction | — | — | none |
| RD03 | 363–365 | Post hoc: untreated decisions ≈ FARE scores (+0.005, 90 % [−0.003, 0.013]); about one point more accurate | CAP-R `posthoc_diagnostics.json` l.42–51 (0.00474 [−0.00335, 0.01283]; accuracy +0.00992 [0.00703, 0.01281]) | POST, DESC | M90 (stated) | hard (A) vs centred (F) | outside replay (D18) |
| CR01 | 367–368 | Features plus own output: A 0.816, B 0.817, F 0.550, F0 0.646 | OAR `EXPLORATORY.csv` l.17, 19, 30, 33 (aliased units); CAP-R secondary l.7–11 | levels EXPL; contrasts S-CAP | none | features + own full output | none |
| CR02 | 368 | Withholding the offset changes nothing for these heads | CAP-R secondary l.27–30 (output-only) | S-CAP NE ×4 | 2S-NORM(3.172) | output-only (not features + output) | **D14** |
| CR03 | 368–369 | Clean logits on F or F0 features restore recovery to 0.788 / 0.789 | OAR `EXPLORATORY.csv` l.29, 32; CAP-R secondary l.33–34 (+0.238 / +0.143 PASS) | levels EXPL; S-CAP PASS | none printed | features + clean logits | none |
| PR01 | 371 | 4 of 10 primary recovery predictions correct | CAP-F `USEFUL_HEAD_COMPARISON.md` l.116–128; CAP-L l.107–119 | scoring of registered predictions | — | — | none |
| PR02 | 371–373 | Predictions registered after full-logit recovery was reported as exploratory; not disclosed in the protocol | CAP-F `USEFUL_HEAD_COMPARISON.md` l.16–25, 128; CAP-L l.96–105 | disclosure | — | — | none |
| PR03 | 373–375 | Three predictions wrong (LEACE effect, fairness-term effect, hard contrasts) | CAP-F `USEFUL_HEAD_COMPARISON.md` l.118–122, 129–131 | — | — | — | none |

### 3.9 §5.2 FARE and compression (l.377–385)

| ID | Lines | Claim / value | Source | Reg. / decision | Uncertainty | Surface | Discrepancy |
|---|---|---|---|---|---|---|---|
| FA01 | 379–380 | Adult features (OAR): zero-fairness 0.645 vs FARE 0.550 vs LEACE 0.812 | OAR `EXPLORATORY.csv` l.35, 26, 11 | EXPL levels | M90 (not printed) | features only | none |
| FA02 | 380 | On head outputs, fairness term removes 0.109 beyond compression | as TT10 | P-CAP PASS | 2S-NORM(3.008) | output-only centred | D13 |
| FA03 | 381–382 | Easier task chosen by a validation-only screen (Adult employment group, age group protected): accuracy 0.974 FARE vs 0.975 untreated; constant 0.277 | ODX `S5_SUMMARY.json` l.77 (0.97476), l.87 (0.97431), l.97 (0.27675); `PROTOCOL.md` l.187–199 (screen); `S5_ENDPOINTS.csv` l.4 (accuracy NI PASS) | S5 PASS (NI) | levels none; NI 2S-NORM(2.498) | **U2 probe**, not deployed head | **D12**, D23 |
| FA04 | 382 | FARE's complete-release recovery 0.151 below LEACE's | ODX `S5_ENDPOINTS.csv` l.2 (0.1512 [0.1373, 0.1651], PASS) | S5 PASS | 2S-NORM(2.498) | features + own head | D17 (after S1) |
| FA05 | 383–384 | Zero-fairness tree reproduces most of it; FARE's smaller additional improvement 0.0074 [0.0019, 0.0129], below 0.02 | ODX `S5_ENDPOINTS.csv` l.3; `S5_SUMMARY.json` l.57, 67 (0.5616 / 0.5689) | S5 NE-excl0 | 2S-NORM(2.498) | features + own head | D17 (amendment S1: lower bound was 0.0001) |
| FA06 | 384 | Easy task, but no single input column determines it | ODX `INDEPENDENT_VERIFICATION.json` l.123–128 (`task_is_function_of_input_columns`: []); `VALIDATION.md` l.59 | verifier INFO | none | — | none |
| FA07 | 384–385 | A fairness-term claim needs its capacity-matched twin | interpretation of FA02 vs FA05 | — | — | — | D13 |

### 3.10 §5.3 Combining recipients (l.387–406) and Figure 3

| ID | Lines | Claim / value | Source | Reg. / decision | Uncertainty | Surface | Discrepancy |
|---|---|---|---|---|---|---|---|
| PA01 | 397–398 | Two Adult recipients (income; employment group) link and pool outputs to recover race, disallowed for both | MRB `PERMISSION_TABLE.csv` (Adult income_prediction and employment_analysis disallow race); ODX `MULTI_RECIPIENT_RESULTS.md` l.3–9 | S4 design | — | linked frozen-head outputs | D23 |
| PA02 | 398–400 | Pair beats both singles by > 0.02 under every format: full 0.917 vs 0.845; centred 0.883 vs 0.792; hard 0.611 vs 0.576 (+0.034, lower 0.023, weakest of six) | ODX `S4_ENDPOINTS.csv` l.2–7 (hard pair−employment 0.03440, lower 0.02338); `MULTI_RECIPIENT_RESULTS.md` l.25–27 | S4 6/6 PASS | lower ends of 2S-NORM(2.638) | full / centred / hard | none (generality: D16) |
| PA03 | 400–401 | Absolute pair recovery and increment reported separately | `MULTI_RECIPIENT_RESULTS.md` l.23–27 | wording | — | — | none |
| PA04 | 401–403 | Seed 2, full logits: validation chose the income-only attacker; that seed's pair−income difference is 0 | `S4_ENDPOINTS.csv` l.2 (`pair_bank_selected_by_seed`[2] = income-only); `MULTI_RECIPIENT_RESULTS.md` l.29–31 | S4 | — | full logits | none |
| PA05 | 403–406 | Measurements with a fixed slate; not a joint-carriage, information or composition claim | `MULTI_RECIPIENT_RESULTS.md` l.42–47 | scope | — | — | none |
| FG3 | 392–393 | Fig. 3 caption: left = frozen-head recovery by format (Adult income/sex, HMDA underwriting/race); right = race recovery, income vs employment vs pair, by format | `report/figures.py` l.46–74: left bars from ODX `OUTPUT_SURFACE_RESULTS.csv` l.2, 5, 4, 7, 104, 107, 106, 109; right bars hard-coded at l.62, matching `MULTI_RECIPIENT_RESULTS.md` l.25–27 | — | none drawn | — | minor: no intervals; three-seed means not stated; y-axis starts at 0.45; D23 |

### 3.11 Figures 1 and 2

| ID | Lines | Caption claim | Source | Discrepancy |
|---|---|---|---|---|
| FG1 | 150–152 | Paths (1) output only in four formats, (2) arm features with own-head output, (3) dashed clean-output bypass | `fig_recipient.tex` l.8–24 | D29 |
| FG2 | 320–322 | Each arm releases its own LR head's output; large markers = three-seed means with 90 % intervals; small = seeds; dotted = chance | `report/figures.py` l.22–43 | **D11** |

### 3.12 Discussion (l.408–429)

| ID | Lines | Claim / value | Source | Reg. / decision | Uncertainty | Surface | Discrepancy |
|---|---|---|---|---|---|---|---|
| DS01 | 410–412 | LEACE kept accuracy (−0.0006, lower −0.0016; NI), passed its bound, barely changed score leakage (0.759 vs **0.774**) | as TT01, N03, A07 | P-CAP PASS; NATIVE; DESC | 2S-NORM(3.008) | deployed head; centred | **D2** |
| DS02 | 412–414 | Zero-fairness tree within one point, 0.648; FARE 0.539 with accuracy cost not shown within one point | TT03, RS03, TT02 | P-CAP | 2S-NORM | — | none |
| DS03 | 414–415 | Decision-only preserved accuracy by construction and gave the lowest recovery for every arm | CAP-F `USEFUL_HEAD_COMPARISON.md` l.77–80 | DESC | none | hard vs score formats | none (for F the gap is 0.003, untested) |
| DS04 | 417–419 | Part of FARE's effect is compression, by a cell-dependent amount; useful decisions still recover only 0.536–0.546 | as A08, RD01 | — | — | — | D13 |
| DS05 | 421–424 | Post hoc: at matched accuracy, lowest score-level recovery is F0's 0.648; untreated decision 0.544; gap about 0.10 AUC | CAP-R `posthoc_diagnostics.json` l.38 (0.64824); R(A, hard) 0.54375 (RD01); difference 0.1045; NI for B and F0 (TT01, TT03) | POST, DESC | none | centred (F0) vs hard (A) | none |
| DS06 | 426–428 | Favourable: fairness term removes 0.109 beyond a capacity-matched tree on every seed; post hoc, FARE's score ≈ untreated decision | TT10, RS04, RD03 | P-CAP PASS; POST | 2S-NORM; M90 | — | none |
| DS07 | 428–429 | Adverse: FARE NI not shown; "any defense is bypassed by an unchanged output"; two recipients recover more than one, even from decisions | TT02; A05; PA02 | as cited | — | — | D16 |

### 3.13 Limitations, reproducibility, ethics, LLM assistance (l.431–454)

| ID | Lines | Claim / value | Source | Reg. / decision | Uncertainty | Surface | Discrepancy |
|---|---|---|---|---|---|---|---|
| L01 | 433–434 | Two datasets, few primary cells, three seeds; one HMDA encoder heavily collapsed | OAR `RESEARCH_DECISION.md` l.54–58 (HMDA s1: 3,836 assessment rows share representations with training rows); ODX `AMENDMENT_R_2026-10-03.md` l.12 (R2: 1,138 / 1,385 certification rows share one vector) | scope | — | — | none |
| L02 | 434–435 | All rows used before; no fresh confirmation | as A10 | scope | — | — | none |
| L03 | 435 | Intervals conditional on fitted heads and attackers; no retraining variance | ODX `PROTOCOL.md` l.156–158; CAP-L l.92 | scope | — | — | none |
| L04 | 435–436 | FARE applied to stored representations | OAR `RESEARCH_DECISION.md` l.185 | scope | — | — | none |
| L05 | 436 | Rare HMDA race groups not estimable | OAR `PROTOCOL.md` l.17; ODX `PRIMARY_ENDPOINTS.csv` l.14–21 (NOT_ESTIMABLE) | scope | — | — | none |
| L06 | 436–442 | Untested threat models listed | COR `CORRECTIONS.md` l.180 | scope | — | — | none |
| RP01 | 444–446 | Protocol, code pin, input hashes, unit list, families and budget committed before fitting; tables, inference code, replay and hash indexes in the repository | MRB `LOCK.json` (lock `3274fe1`), OAR `EXECUTION_LOCK.json` (`3130c8d`), ODX `LOCK.json` (`01ef342`), CAP `LOCK.json` (`a0de449`) | provenance | — | — | D17 |
| RP02 | 446–447 | Per-record outputs, weights and checkpoints kept privately, with hash indexes and a verified offline copy | CAP-F `VALIDATION.md` l.63–67; `PRIVATE_BACKUP_INDEX.json`; MRB `VALIDATION.md` §6; ODX `VALIDATION.md` (backup) | provenance | — | — | none |
| RP03 | 447 | Useful-head comparison used about 0.2 CPU-hours | CAP-F `COST_AND_STORAGE.md` l.11, 16 | — | — | — | **D6** |
| E01 | 449–450 | Public benchmark data; standard attack code; no per-person predictions released | CAP-F `COST_AND_STORAGE.md` l.34–37 | scope | — | — | none |
| LL01 | 452–454 | All numbers produced by committed code and checked by a replay that does not import the analysis code | as D18 | provenance | — | — | **D18** |

## 4. Numbers repeated in several places

| Value | Places | Consistent? |
|---|---|---|
| Untreated useful-head score recovery | 0.774 at l.42, 355, 412; 0.773 at l.304, 359 (same stored 0.773489) | **No** (D2) |
| Own-check failures | 12 of 42 at l.37, 265, 282 (stored 11) | Internally yes; **unsupported** (D1) |
| LEACE head score 0.759; FARE 0.539; F0 0.648; fairness term 0.109; 0.234 / 0.125 | abstract, Table 2, §5.1, §5.2, §6 | Yes |
| Decision-only recovery 0.536–0.546; untreated decision 0.544 | l.362, 419, 423 | Yes |
| Bypass range 0.76–0.79 | l.39, 291 (from 0.758, 0.778, 0.768, 0.788) | Yes |
| Offset effect 0.072 / 0.067 | Table 1 l.270, l.300 | Yes |
| Frozen hard decisions 0.513 / 0.505 | Table 1 l.269, l.293–294 | Yes |
| Adult features: F0 0.645, FARE 0.550, LEACE 0.812 | Table 1 l.273–274, l.379–380 | Yes |
| S5: 0.151; 0.0074 [0.0019, 0.0129]; 0.974 / 0.975 / 0.277 | l.381–384 | Yes (single place) |
| Coalition 0.917 / 0.845; 0.883 / 0.792; 0.611 / 0.576 | l.399–400, Fig. 3 data | Yes |
| Assessment sizes 5,243 / 4,764 and 5,250 / 4,778; exclusions 17 / 42 | l.215–219 | Yes |
| Refitted head gain 0.083 vs useful-head gains 0.082 / 0.073 / 0.079 | l.303, Table 2 | Yes (A's gain 0.0826 is not in Table 2) |
| HMDA full-minus-hard 0.262 vs printed levels 0.768 − 0.505 = 0.263 | Table 1 l.269 | Rounding only; the endpoint is 0.262 |

## 5. Caption checks

- **Table 1 (l.259–260).** It contains levels from the MRB (benchmark roles) and the OAR (output-aware roles), contrasts
  from the OAR and ODX, a descriptive per-seed row and a count. The caption's "simultaneous lower bounds ... where an
  endpoint was registered" is inaccurate (D7). The count row is wrong (D1). Two cells are mis-rounded (D3, D4) and one
  prints negative zero (D5).
- **Table 2 (l.327–328).** The contents match `PRIMARY_USEFUL_HEAD_ENDPOINTS.csv` l.2–20 exactly at the printed
  precision. The caption omits the interval type; the inference paragraph supplies it.
- **Figure 1 (l.150–152).** See D29.
- **Figure 2 (l.320–322).** See D11.
- **Figure 3 (l.392–393).** The values match their sources. There are no intervals and the seeds are not stated (FG3).

## 6. Notes for the owner

- Printing 11 instead of 12 (D1) and fixing the three-decimal roundings (D2–D5) would align the manuscript with the
  stored aggregates. It would not change any stored value or registered decision.
- Where a sentence relies on an exploratory interval (D8, D9), naming it as an unadjusted 90 % interval would keep it
  distinct from the simultaneous intervals of §4.
- The S5 numbers (amendment S1) and the FARE certificate values (amendment A1) are outputs of post-run repairs (D17).
  Disclosing that in one sentence would match the record.
- Entries indexed: 174 (claims, table rows and captions), of which 36 discrepancies are listed in §2.
