# Independent numerical verification: PCRL frozen-finalist validation study

**Role.** Independent numerical verifier (a separate agent), 2026-10-17, about 22:32–23:07Z. Brief: `roles/coordinator/VERIFIER_BRIEF.md`.

**Data.** Public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida, development roles only.
- The service predicts income class (2) and occupation group (5) from AGEP, WKHP, COW, SCHL, MAR and POBP.
- Each recipient receives its own float64 probability vector and argmax decision.
- SEX is the dataset's recorded demographic column. Evaluation readers try to infer it from the released predictions: each output alone and the linked pair.

**Method.** Every number below comes from the verifier's own code in `roles/verifier/code/`:
- Nothing from `ffv/report.py`, `ffv/feasibility.py`, `ffv/feasibility_v2.py` or `ffv/mapping.py` is reused.
- Data came through the admitted loaders (`ffv.common.load_role`, `service_bank`, `frozen_bank`, `source_readers`).
- Bootstrap weights came from `task_headroom.inference.group_weights`. AUC, ratio-mean replicates, endpoint bounds, the claim engine, selection, features, scaling and power calculations are the verifier's own.
- `amd.fam_b.package_release` was used only to confirm invented counterexamples, as the brief allows.
- The statistical reviewer's feasibility files were not read.

**Label access.**
- ATTACK_FIT and ATTACK_VALIDATION SEX were read through the logging loader, in units VER_REFIT_S0_V1 and VER_REFIT_S1_V1.
- DEV_REPORT truth was read once: event `DEV_REPORT_TRUTH_REREAD_BY_FFV_INDEPENDENT_VERIFIER` at **2026-10-17T22:50:37Z**, unit VER_DEVREPORT_V1.
- Every label-dependent number was computed inside that unit. Later steps used only its saved aggregates and per-replicate endpoint draws (private). No truth file was written.
- No MODEL_FIT, CALIBRATION or MODEL_SELECT label was read. The sealed assessment was not opened.

## Summary

| # | Check | Status | Key numbers |
|---|---|---|---|
| 1 | Hashes and identity | **PASS** | 414 / 414 admitted copies re-hashed, 0 mismatches. Release hashes equal FINALIST_LOCK. Every released income vector equals one of the 6 level vectors bitwise. One occupation wire for FINALIST, H_LJ and B2TWIN on all 6 roles and both seeds, confirmed by an own bitwise recomputation. |
| 2 | Exposure ledger | **PASS** | 15 / 15 spot checks match the cited records. Own overlap check: 15 role pairs, 0 shared households or persons. |
| 3 | Output mapping | **PASS** | Every per-role, union and routing-cell count reproduced exactly. 8 / 8 reported counterexamples and 4 / 4 shared-vector examples confirmed through the package. |
| 4 | Reader feature access | **PASS** | 256 design widths match the access rules. Aliased occupation winners share one reader. The SEX-read log holds ATTACK roles only. |
| 5 | Roster refits | **PASS** | 29 / 29 refits bitwise on DEV_REPORT, with ATTACK_VALIDATION AUC differences of 0.0. |
| 6 | Selection and primary endpoints | **PASS** | 36 / 36 winners re-derived. 316 / 316 new-candidate AV AUCs exact. 70 + 70 endpoints: maximum point difference 0.0, maximum SE difference 2.2e-18. Every claim label and interpretation equal. |
| 7 | Utility and subgroup headroom | **PASS** | 28 / 28 point gates pass; 7 / 28 upper bounds sit at or below their caps. Binding clauses confirmed at 0.027, 0.042 and 0.110 SE (the last twice). |
| 8 | Power and sample size (V2) | **PASS** | S1 and S2 × 0.75 n80 for D1 and D2 match V2 at 0 grid steps, by both independent methods. No n80 disagreement exceeds 1 grid step in any scenario. |
| 9 | Reporting labels and limitations | **PARTIAL** | The first review found 7 minor statements imprecise or unsupported (V-D1 to V-D7). The coordinator corrected all 7. One residual sentence (V-D13) is open at the final re-check. No claim label, endpoint or recommendation depends on any of them. |

**No major discrepancy was found.** The verifier records no unresolved discrepancy in any primary endpoint, claim label or feasibility number. That is the condition the recommendation in `RESEARCH_DECISION.md` places on this verification.

## 1. Hashes and identity: PASS

**Admitted copies.** All 414 copies in `SOURCE_ADMISSION.json` were re-hashed, not a sample: 0 mismatches. The 406 copies that carry an expected hash also match it.

**Cross-checks against the source's own public records, all true:**
- the service banks against `SERVICE_BANK_MANIFEST`;
- the frozen banks against `FROZEN_BANK_MANIFEST`;
- the source reader record and scores against the source `REPORTING_LOCK`;
- 168 + 168 source reader model files against `READERS_SEED.json`;
- the 10 exports against `EXTRACTION_SUMMARY`;
- the three release files against `FINALIST_LOCK` and `RELEASE.json`;
- every deployment `PACKAGE.json` file map;
- this campaign's enhanced reader files (AV and DEV scores, records, and 156 + 148 model files) against `REPORTING_LOCK`;
- the lock hashes.

**Release identity.** The release hash of each seed equals `FINALIST_LOCK.release_sha256` and the manifest (S0 `04e6a1b3…`, S1 `b7b72562…`). The package `H.npz` equals the manifest's thresholds and calibration values bitwise.

**Per-level income vectors.**
- On each seed, every released income row in all six roles (47,927 per seed) equals one of the 6 manifest level vectors bitwise.
- All 6 levels occur in every role.
- The per-level counts equal the release record's `coarse_cell_counts` for every role.
- An own 6-row recomputation from `H.npz` reproduces the manifest's small-batch last-bit observation exactly: levels 2 and 4 on S0, and 3 and 4 on S1, differ by at most 1.1e-16.

**Occupation wire.**
- FINALIST = H_LJ = B2TWIN bitwise on the 4 bank roles of both seeds.
- FINALIST = B2TWIN on all 6 release roles.
- An own recomputation of the JOINT_TREE E20 occupation output, from the encoded inputs and the admitted teacher and theta files, equals the released occupation output bitwise on all 6 roles and both seeds. This covers H_LJ's definition on MODEL_FIT and CALIBRATION, which are not in the bank.
- An own recomputation of the income level, as the number of thresholds strictly below d = 0.5 LOCAL_TREE + 0.5 JOINT_TREE, equals the released level for every row (0 differ), including the documented threshold-tie people.
- Unique single-recipient wires: 6 income and 4 occupation per seed.

## 2. Exposure ledger: PASS

**Spot checks.** 15 entries were checked against the records they cite, and all 15 match:
- F8: the seed-0 bins 4,114 / 4,113 / 4,114 / 2,553 / 2,554 / 2,552;
- F9: the `h_fit_role` and CALIBRATION income-only fields; 2,982 people in 1,974 households;
- S7: PROMOTION.json gives 17 trials, 3 promotable and 0 robust; proxies 0.60002 / 0.61314 / 0.61533; 17:35:23Z; designation control;
- S8: the superseded interim record, 17:23:54Z, naming B2;
- S9: the eligibility map, and ledger entry 22 with PUB_IBSI seed-1 men's guard 0.00501 and a failed guard;
- S2 and S3: the CALIBRATION screens (ledger 31 and 40) and their unit times, 17:13:21Z and 17:21:01Z;
- S4: ledger entries 41 and 42;
- F6: SELECTION_LOCK at 2026-10-16T00:18:37Z with DEV_REPORT unopened;
- S10: the hybrids ANALYSIS_LOCK at 02:46:48Z, primary H_LJ;
- §6: FINALIST_LOCK at 17:37:02Z, and ANALYSIS_LOCK at 17:42:54Z with draft hash `8389e591`;
- §4: 24 MODEL_SELECT label reads before promotion (2 baseline plus 22 development) and a 25th at 17:52:58Z;
- §4: 10 lineage DEV_REPORT truth reads before this campaign, the first at 2026-10-16T00:20:20Z, AMD coordinator at 17:45:30Z and AMD verifier at 17:56:47Z;
- §6: the admitted AMD access snapshot at 17:45:30Z.

**Household overlap, own code.**
- Each role's persons, households and largest household equal the audit table and `EXTRACTION_SUMMARY`.
- All 15 role pairs share 0 households and 0 persons.
- The union is 47,927 distinct persons in 31,865 distinct households.
- Overlap with the sealed assessment stays UNKNOWN; it was not opened.

## 3. Output mapping: PASS

**Empirical counts.** Recomputed from the admitted release arrays and encoded inputs, using exact bytes:
- every per-role count (records, distinct inputs and raw vectors, occupation and income vectors, pairs, conflicts, shared vectors, singleton shares, maximum group);
- every union count;

on both seeds. There are 0 differences from `OUTPUT_MAPPING_RESULTS.json`.

**Seed-1 cross-role conflicts.** There are 2:
- one between DEV_REPORT and MODEL_FIT; one between ATTACK_FIT and MODEL_SELECT;
- in both, the income levels are {2, 3}, so the decisions are {0, 1};
- the encoded inputs differ only in POBP columns.

The same encoded input never received two different output pairs across roles.

**Routing cells.** Own encoder and own split signatures, from the frozen teacher JSONs:

| | Domain | Occupation cells | Income cells | Joint cells | Singleton share |
|---|---|---|---|---|---|
| S0 | 369,388,800 | 264,668,800 | 182,307,840 | 299,376,000 | 0.5735 |
| S1 | 369,388,800 | 255,974,400 | 142,560,000 | 290,048,000 | 0.5241 |

All of these equal the reported values, as does every per-variable class count.

**Counterexamples.**
- All 8 reported same-occupation, different-income examples were confirmed through `amd.fam_b.package_release`: occupation vectors bitwise equal, income levels as reported. So were all 4 distinct-input, same-vector examples.
- The own routing gives the same occupation vector and the same levels.
- The reported income strings of some examples differ in the last bit under a 2-row request. That is the characterized batch-size dependence; levels and decisions agree.
- An own bounded search with a different seed tested 800 invented pairs per seed. It found 63 (S0) and 156 (S1) counterexamples, 16 and 37 of which flip the income decision. They arise only through WKHP and POBP, and the first 3 per seed were confirmed through the package.

**Part 3 reconstruction.** Reproduced exactly with own code: level accuracy 0.6277 / 0.6303 on ATTACK_VALIDATION and 0.6246 / 0.6397 on DEV_REPORT.

## 4. Reader feature access: PASS

**From the code** (`ffv/readers.py`, `ffv/features.py`):
- `X` and `feature_names` are deleted from the three exports before any fit.
- Every design is built from the released probability vectors only: a single view from its own output, a pair from the two aligned outputs.
- `design_for` (spline knots, quantile cuts, categories) and `FiniteScaler` are fitted on ATTACK_FIT rows only.
- Identical wires, by exact hashes over the three reader roles, are fitted once and aliased.
- Source candidates enter with `SRC::` keys, stored AV AUCs and stored DEV scores.
- The planted SEX bit appears only in `CONTROL__` readers.

**From the records:**
- All 128 + 128 design widths equal the widths the access rules imply: rich features give income 6, occupation 15 and pair 21; CATI gives income 6, 18 or 32, occupation 48, and pair 69, 73 or 329.
- No income vector at scoring time was unseen.
- The `FINALIST|occupation` bank aliases FINALIST, H_LJ and B2TWIN on both seeds, and their original and enhanced occupation winners share one reader key.
- The 7 (S0) and 12 (S1) source-origin enhanced winners carry exactly the source record's AV AUC.
- The SEX-read log contains only ATTACK_FIT and ATTACK_VALIDATION `sex` reads, by the two reader units and the verifier's two refit units.
- Check 5's bitwise refits, made with ATTACK_FIT-only fitting and the CATI state equal to the lock, independently confirm the fitting roles.

## 5. Representative reader refits: PASS

**Roster.** Per seed, the ET5, HGB31, MLP6432 and CATI1 pair readers of FINALIST, NEURAL_RAW and H_LJ, plus the enhanced pair winners. The winners were resolved by the verifier's own selection:
- S0: LOG_MLP, LOG_MLP and OUT_MLP, all source readers;
- S1: HGB15 [new], CATI1 [new] (already in the roster) and OUT_MLP.

**How they were refitted.** Own feature construction, own standardization and own estimator instantiation. The full `get_params()` of all 8 recipes and the library versions equal `READER_BANK_LOCK.json`.

**Results.**
- **ATTACK_VALIDATION AUC:** 29 / 29 identical to the stored values (maximum difference 0.0).
- **AV scores:** 25 / 25 new-reader score vectors bitwise equal.
- **DEV_REPORT scores:** 29 / 29 bitwise equal to the stored scores (maximum absolute difference 0.0), so the DEV AUC differences are 0.0.
- **CATI fitted state:** the FINALIST categories and the NEURAL_RAW / H_LJ quantile cuts equal the lock, 6 / 6.
- **Source pair winners:** 4 of the 8 reproduction-subset readers were also refitted independently and are bitwise: S0 FINALIST LOG_MLP, NEURAL_RAW LOG_MLP and H_LJ OUT_MLP, and S1 H_LJ OUT_MLP. The other 4 were not refitted by the verifier.

## 6. Selection and primary endpoints: PASS

**Winners.**
- All 316 (2 × 158) stored new-candidate and control ATTACK_VALIDATION AUCs were recomputed with own AUC code and equal the records exactly.
- All 36 winners (6 releases × 3 views × 2 seeds) were re-derived in the source-first order, with exact ties going to the earlier candidate. They equal `READERS_ENH_SEED.json` and `REPORTING_LOCK.json`, with the same gaps and flags. The original winners re-derive to the source records.
- 11 (S0) and 6 (S1) enhanced winners differ from the originals.
- All 36 carry AV_NEAR_TIE (gap below 0.010).

**DEV_REPORT replay.** Household bootstrap with B = 2048, seed 2026101705, G = 3,955, and own AUC and ratio-mean code. The critical values equal the registered ones: c1 2.3949991, c_P 3.3040310, c_S 3.0252810, marginal 1.9605641.
- All 72 winner AUCs and SEs equal the coordinator's (maximum difference 0.0).
- Every endpoint of both scorecards, 70 + 70, matches: maximum point difference 0.0; SE and bound differences at most 7e-18; every status and category equal (see V-D9).
- The ORIGINAL scorecard reproduces the source report: 70 / 70 points exact, every label equal.

| Endpoint (seed 0 / seed 1) | ORIGINAL point; clause bound | ENHANCED point; clause bound |
|---|---|---|
| Pair benefit vs NEURAL_RAW | +0.02703 / +0.01989; L1 0.01354 / 0.00592 | +0.02703 / +0.02174; L1 0.01354 / 0.00691 |
| Pair benefit vs H_LJ | +0.02839 / +0.02499; L1 0.01907 / 0.01290 | +0.02839 / +0.02600; L1 0.01907 / 0.01404 |
| Income recipient vs H_LJ, s1 | −0.00352; U1 0.00584 | **+0.00760; U1 0.02352 > 0.010** |
| Pair benefit vs TASKONLY | +0.05153 / +0.04651; L1 0.03909 / 0.03358 | +0.05083 / +0.04752; L1 0.03739 / 0.03455 |
| Pair benefit vs B2TWIN | +0.01640 / +0.00991; L1 0.00992 / −0.00253 | +0.01844 / +0.01093; L1 0.00800 / −0.00156 |
| IBSI − FINALIST, pair (descriptive) | +0.01667 / +0.00470 | +0.01632 / +0.00571 |

**Claim labels**, equal to the coordinator's:

| Claim | ORIGINAL | ENHANCED |
|---|---|---|
| E | DEV_POINT_ELIGIBLE | DEV_POINT_ELIGIBLE |
| B | SUPERIOR_TO_ALL_ELIGIBLE | NOT_SUPERIOR_TO_ALL_ELIGIBLE; first failing condition `COMPETITIVE|income_worsening_vs_H_LJ|s1` |
| 0.020 nested test | NOT_ESTABLISHED | NOT_ESTABLISHED |
| M | MECHANISM_ATTRIBUTION_ESTABLISHED | TASK_ONLY_ATTRIBUTION_ONLY |
| J | JOINT_ADVANTAGE_NOT_ESTABLISHED; first failing `JOINT|pair_benefit_vs_B2TWIN|s1` | same |

**Interpretations**, equal to the coordinator's:
- B: READER_SENSITIVE, side COMPARATOR_READ_LOWER (FINALIST change 0.0; H_LJ income winner −0.01112);
- 0.020: NOT_ESTABLISHED_EITHER_BANK;
- M: READER_SENSITIVE (M_TO SURVIVED_THIS_READER_BANK; M_SS READER_SENSITIVE);
- J: NOT_ESTABLISHED_EITHER_BANK.

**Control validity.**
- ORIGINAL and ENHANCED CV both hold.
- 36 control AUCs were recomputed with a maximum difference of 0.0.
- Minimum planted AUC is 1.0. The maximum |shuffled − 0.5| is 0.0353, inside the 0.05 band.

**Everything else matches at 0.0 maximum difference:**
- reader-class changes (36);
- benefit changes (8);
- reader-matched benefits (160);
- refit-variation AUCs and SDs (20);
- utility points of every bank release (180).

## 7. Utility and subgroup headroom: PASS

**Point gates.** All 24 utility excesses against U0 and Ucal, and all 4 within-SEX guards, pass. Family statuses: 3 PASS, 25 UNRESOLVED. 7 of 28 one-sided c1 bounds are at or below their caps.

**Binding clauses**, smallest slack / SE (verifier):

| Clause | Point | SE | Cap | Slack | Slack / SE | U1 |
|---|---|---|---|---|---|---|
| income Brier vs Ucal, s1 | 0.004944 | 0.002047 | 0.005 | 5.6e-5 | 0.0275 | 0.00985 |
| income Brier vs Ucal, s0 | 0.004919 | 0.001925 | 0.005 | 8.1e-5 | 0.0421 | 0.00953 |
| income accuracy loss, s1 (U0 = Ucal; identical draws) | 0.009591 | 0.003719 | 0.010 | 4.09e-4 | 0.1099 | 0.01850 |

The next clause is occupation NLL vs Ucal, s0, at 0.40 SE.

**Guards** (sex code 0 = men): men 0.003235 / 0.003951 (0.86 / 0.55 SE of slack); women −0.000868 / −0.002275.

These are the binding clauses named in PROTOCOL §10 and in both feasibility files.

## 8. Power and sample size: PASS

**What was checked.** `CONFIRMATION_FEASIBILITY_RESULTS_V2.json`, as the coordinator asked; V1 was compared too.

**Methods.** Three, independent of the campaign code:
- **A.** The analytic formula n = 3955 × ((z + c) × SE / slack)², and analytic boundary limits (orthant probability; MVN CDF at −c1).
- **B.** Nonparametric bootstrap-deviation rescaling of the verifier's own 2,048 replicates.
- **C.** An own MVN Monte Carlo with seed 61017 and 20,000 draws. It uses no factorization: X = Z D_c / √(B − 1).

**Inputs.** The same 38 conditions per scorecard (28 utility, 6 recipient, 4 benefit). The SEs equal the replicate SDs, and the refit τ values equal V2's (difference 0.0).

**Governing scorecard (ENHANCED; ORIGINAL identical):**

| Scenario | V2 n80 / n90 | Bootstrap (B) | MVN (C) |
|---|---|---|---|
| S1 plug-in, D1 | 4,632,356 / 9,486,693 | same | same |
| S1 plug-in, D2 | 59,928,535 / 73,549,580 | same | same |
| S2 × 0.75, D1 | 4,632,356 / 9,486,693 | same | same |
| S2 × 0.75, D2 | 59,928,535 / 73,549,580 | same | same |

**Analytic single-clause n80** (method A):
- income Brier vs Ucal s1: 3.71 M (D1), 54.9 M (D2);
- income Brier vs Ucal s0: 1.58 M (D1), 23.4 M (D2);
- accuracy loss s1: 0.23 M (D1), 3.4 M (D2).

The joint values are at or above the worst single clause, as they must be: for D1, 2 grid steps above 3.71 M (the correlated s0 Brier clause also binds); for D2, the first grid point above 54.9 M. Two further values match the feasibility text:
- the ENHANCED H_LJ s1 recipient clause needs 317,958 (text: "about 318,000");
- the NEURAL_RAW s1 benefit needs 522,255 for the 0.020 bound (text: "about 522,000").

**All 39 scenario × definition cells per scorecard:**
- n80 agrees within at most 1 grid step (10.8%).
- "Not reached" agrees in every cell.
- The 1-step cases are S4 (refit τ) for the bootstrap method and S5 slack 0.001 D1 for the MVN.

**Limits:**
- At the cap, D1: the analytic orthant probability is 0.2647 (V2: 0.2657 ENHANCED, 0.2681 ORIGINAL; V1's 0.0625 was wrong).
- At the cap, D2: 0.00034 (V2 0.0004 / 0.0003).
- Assurance limits: 0.102 / 0.102 / 0.051 (ENHANCED) and 0.163 / 0.163 / 0.068 (ORIGINAL), against V2's 0.102 / 0.102 / 0.053 and 0.161 / 0.161 / 0.064.
- Same-size replication, D1: 0.0026–0.0034 (ENHANCED) and 0.104–0.113 (ORIGINAL), against V2's 0.0027 and 0.115.

**V1 against V2.**
- Identical n80 and n90 in every cell.
- V2 corrects the boundary limits and adds the assurance limits (V-D12).
- The S6 status labels changed, and 8 governing assignments changed (V-D4).
- The governing scorecard differs between the verifier's methods and V2 only in Monte Carlo near-ties (V-D11).

## 9. Reporting labels and limitations: PARTIAL

**Documents read:**
- `CONFIRMATION_FEASIBILITY.md`, `OUTPUT_MAPPING_CHECK.md`, `RESEARCH_DECISION.md` and `HANDOFF_DECISION.json`, in full;
- `PROTOCOL_DEVIATIONS.md`;
- `START_HERE.md`, skimmed.

**Timeline.**
- The first review was at about 22:58Z.
- The coordinator then revised the documents, about 22:59–23:05Z, and added a disposition of these items as PROTOCOL_DEVIATIONS.md item 14.
- A final label-free re-check (VER_RECHECK_V3, 23:06:40Z) tested each flagged phrase in the current text and recorded the document hashes in `INDEPENDENT_VERIFICATION.json`. Any later edit of those documents is outside this verification.
- **At the re-check, V-D1 to V-D8 are corrected in the current text**, apart from one residual sentence, V-D13 below.
- The newly added statements were checked against the verifier's numbers:
  - the H_LJ seed-1 change interval, −0.024 to +0.001 (verifier: −0.0236 to +0.0014);
  - the H_LJ seed-0 income change, −0.0085;
  - the exact AV tie under the FINALIST seed-1 income winner (gap 0.0);
  - the bootstrap-refit benefit range, −0.001 to +0.031;
  - "2 and 13 within-role shared vectors";
  - power 0.013 [0.252] at about 17,500 households;
  - the τ ranges;
  - the governing-scorecard description;
  - `MAPPING_CONFLICTS.json`: 5 / 46 shared vectors, excess 6 / 49, maximum 3 / 4 inputs, and both seed-1 conflicts with their roles, levels and POBP-only difference. All agree.

**Verified as supported:**
- the binding-slack table, including the flat-prior P(slack > 0) values 0.51 / 0.52 / 0.54;
- every n80 table cell and the assurance and same-size figures;
- the pair-benefit and recipient numbers;
- the H_LJ seed-1 income winner change: LOG_MLP 0.5517 to HGB15 0.5523 on ATTACK_VALIDATION, and 0.549 to 0.538 on DEV_REPORT;
- the reader-matched summary: 14 of the 15 non-LOOKUP recipes positive for each seed and reference, range +0.005 to +0.034; CONF_HGB −0.015 to −0.002; LOOKUP about 0;
- all 24 bootstrap refits reading lower, and the benefit shrinking in 3 of 4 cells;
- the refit SD range;
- the 43–48% routing-cell share and the counterexample statements;
- the Part 3 figures;
- the floating-point disclosures, against `DEPLOYMENT_CHECKS.json`.

**Not supported or imprecise at the first review** (all minor; no label, endpoint or recommendation depends on them). V-D1 to V-D7 are **corrected in the current text**:

| Id | Where | Issue |
|---|---|---|
| V-D1 | OUTPUT_MAPPING_CHECK.md, Part 1 | "6 / 49 such vectors": there are **5 / 46** occupation vectors shared by distinct inputs. The 6 / 49 are excess inputs. |
| V-D2 | OUTPUT_MAPPING_CHECK.md, Part 2 | "587 income trees and 800 occupation trees per seed": seed 1 has **529 / 799**. |
| V-D3 | CONFIRMATION_FEASIBILITY.md conclusion 1; RESEARCH_DECISION.md §4 | "At about 17,500 households joint power is at most about 0.2 (D1)": the ORIGINAL plug-in D1 power at 17,510 is **0.23–0.25**. 0.20 holds at about 10,000. ENHANCED is about 0.01. |
| V-D4 | CONFIRMATION_FEASIBILITY.md; RESEARCH_DECISION.md §4; PROTOCOL_DEVIATIONS.md item 9 | "No status label changed" (V1 to V2): the 6 S6 assurance labels changed from NOT_REACHED_BY_1E8 to LIMIT_BELOW_TARGET, and 8 governing assignments changed. No n80 changed. |
| V-D5 | CONFIRMATION_FEASIBILITY.md results table header | "ENHANCED governs" is inaccurate for the D3 column (slack 0.0005–0.002 and plug-in) and the slack-0 D2 and D3 cells, where ORIGINAL governs by V2's rule. The bracketed values are correct. |
| V-D6 | RESEARCH_DECISION.md §4 "What binds" | "the recovery clauses alone pass at DEV's size": only the 4 benefit clauses do. The ENHANCED s1 recipient clause against H_LJ fails at DEV size and needs about 318,000 households. Utility still binds. |
| V-D7 | RESEARCH_DECISION.md, "Output structure" | "two occupation vectors carry different income codes, one on each side of the income decision" is ambiguous. Each vector carries levels 2 and 3, which straddle the decision. |

**Still open at the final re-check:**

| Id | Where | Issue |
|---|---|---|
| V-D13 (residual of V-D4) | RESEARCH_DECISION.md, "Why validation is not incomplete (option 4)" | "through a disclosed V2 that changed no label". V2 changed no claim label and no n80 or n90 value, but it did change 6 feasibility status labels and 8 governing assignments, as §4 of the same document and PROTOCOL_DEVIATIONS.md item 9 now say. This is wording only. |

**Informational:**
- **V-D8.** The decision text predated the verifier's logged replay at 22:50:37Z. It is now stated.
- **V-D9.** Cosmetic: `clause_pass` is false rather than null on 2 identical family-S keys.
- **V-D10.** The side attribution is applied to a failing recipient clause, while the protocol text says "failing benefit". The label meaning carries over.
- **V-D11.** Governing-scorecard Monte Carlo near-ties.
- **V-D12.** The V1 limits are resolved in V2.

## Units and resources

| Unit | CPU s | Note |
|---|---|---|
| VER_IDENTITY_V1 | 6.5 | check 1 |
| VER_REFIT_S0_V1 / S1_V1 | 17.7 / 15.7 | check 5 and the AV half of check 6; ATTACK SEX reads logged |
| VER_MAPPING_V1 | 1.3 | check 3 |
| VER_EXPOSURE_V1 | 0.1 | check 2 |
| VER_DEVREPORT_DRYRUN_V1 | 18.7 | full pipeline on invented labels; no truth read |
| VER_DEVREPORT_V1 | 18.4 | **the one DEV_REPORT truth read**, 22:50:37Z |
| VER_COMPARE_V1 | 0.1 | label-free comparison |
| VER_POWER_V1 | 26.3 | check 8 |
| VER_FOLLOWUP_V1 | 4.4 | check 4 records; check 9 figures |
| VER_ASSEMBLE_V1 to V4 | 0.3 | this record, assembled 4 times; V4 is final |
| VER_RECHECK_V1 to V3 | 0.8 | re-check of the revised documents and `MAPPING_CONFLICTS.json`. V1's test for V-D1 matched "46 such vectors" as a substring; corrected from V2 on. V3 is final. |

No unit failed. Queue units total 110 CPU seconds. At most 2 heavy jobs ran at once (the two refit units). The agent effort is charged separately: label `verifier`, 900 CPU seconds, an agent upper estimate, at 23:06:29Z.

## Not covered by this verification

- The deployment and backup checks, beyond the 8 + 4 + 6 package confirmations.
- The population inventory.
- The statistical reviewer's feasibility check (deliberately not read).
- The exact counts of the reported bounded search (689 / 1,610). An own search with another seed was run instead.
- 4 of the 8 source-reproduction refits: S1 FINALIST pair OUT_HGB, S1 NEURAL_RAW pair LOG_MLP, and FINALIST occupation OUT_HGB on both seeds. These rest on the coordinator's bitwise records.
- Forecast scoring.
