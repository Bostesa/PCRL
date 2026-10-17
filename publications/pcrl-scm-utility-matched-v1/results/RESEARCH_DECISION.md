# Research decision: quality-matched comparison of grouped prediction tables (SCM utility-matched follow-up)

**What this is.** An exploratory engineering comparison of existing methods, run on Florida ACS 2019 development records that earlier work has already used:
- Shared-Cell Merging (SCM);
- ordinary task-only merging. Its merge objective is task-only, but it still uses SEX through the shared within-SEX guard, and its comparator was selected with the SEX-labelled MODEL_SELECT reader proxy;
- the published IBSI objective (PUB_IBSI, adapted);
- shuffled-signal controls.

The design was informed by the source benchmark's published results (`6fc06021d`); those results and labels are unchanged. No novelty, new-algorithm or joint-design claim is made. The service is a controlled benchmark resting on three experimental assumptions: the readers' SEX-labelled auxiliary records, accurate linkage of the two outputs, and distributional similarity. The sealed assessment stayed closed.

All numbers are seed 0 / seed 1 on DEV_REPORT unless marked MODEL_SELECT. Intervals are conditional on the fitted releases and readers and exclude the development and selection history. The two seeds share the same 5,943 people in 3,955 households.

## Decision

**`COMPARISON_CLOSED_INCONCLUSIVE_AT_REGISTERED_TOLERANCE`** (from claim M = `MATCH_DID_NOT_CARRY_OVER`), with its four mandatory qualifiers:
- DEV utility confidence: `CONFIDENCE_UNRESOLVED` (3 PASS, 21 UNRESOLVED, 0 violations);
- claim C: `SUBGROUP_UNRESOLVED`;
- the ORD_NLLM match on DEV_REPORT:
  - seed 0: `POINT_MISMATCH` and `MISMATCH_ESTABLISHED`;
  - seed 1: `POINT_MISMATCH` and `EQUIVALENCE_UNRESOLVED`;
  - direction `SCM_SPENT_MORE` on both seeds;
- claim B: `NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD`.

Two related statuses:
- nested 0.020 target for claim M: `NOT_ESTABLISHED` (seed 0 clause bound 0.0225 > 0.020; seed 1 0.0144, UNRESOLVED). This is comparison-only and not cost-matched on DEV_REPORT;
- comparison-only status: `BENEFIT_AND_NON_WORSENING_CLAUSES_PASS (not cost-matched on DEV_REPORT)`.

No release is frozen as an engineering reference. The fixed comparison is closed, and its measured limitations are identified in §6. This does not generalise to every possible shared partition.

| Claim (α/4 each) | Label | First failing condition (category) | Other false conditions in the conjunction (category) | Comparison-only status |
|---|---|---|---|---|
| M, matched ordinary (primary; MODEL_SELECT match) | `MATCH_DID_NOT_CARRY_OVER` | DEV occupation-NLL point match vs ORD_NLLM, both seeds (POINT_FAILURE: seed 0 `MISMATCH_ESTABLISHED`, seed 1 `EQUIVALENCE_UNRESOLVED`; both `SCM_SPENT_MORE`) | four DEV subgroup point guards (POINT_FAILURE, men, both seeds); nested 0.020 `NOT_ESTABLISHED` (seed 1 clause bound 0.0144, UNRESOLVED) | `BENEFIT_AND_NON_WORSENING_CLAUSES_PASS (not cost-matched on DEV_REPORT)` |
| M components (IBSI → SHUF) | `NOT_TESTED_M_NOT_ESTABLISHED` | M not established (UNAVAILABLE) | descriptive status `MATCH_DID_NOT_CARRY_OVER` for both | — |
| K, same capacity | `NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD` | men's occupation Brier point guard (POINT_FAILURE) | income worsening vs ORD_CNT, both seeds (UNRESOLVED; clause bounds 0.0136 / 0.0174); nested 0.020 `NOT_ESTABLISHED` (seed 1 clause bound 0.0169, UNRESOLVED); IBSI_CNT `NOT_TESTED_K_NOT_ESTABLISHED` (UNAVAILABLE) | `COMPARISON_CLAUSES_NOT_ALL_PASS` |
| B, competitive | `NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD` | men's occupation Brier point guard (POINT_FAILURE) | benefit vs NEURAL_RAW and vs H_LJ, seed 1 (UNRESOLVED); income worsening vs NEURAL_RAW both seeds and vs H_LJ seed 1 (POINT_EXCESS), vs H_LJ seed 0 (UNRESOLVED); occupation worsening vs H_LJ seed 0 (ESTABLISHED_VIOLATION), seed 1 (POINT_EXCESS), vs NEURAL_RAW seed 1 (UNRESOLVED); nested 0.020: all four POINT_SHORTFALL | `COMPARISON_CLAUSES_NOT_ALL_PASS` |
| C, subgroup | `SUBGROUP_UNRESOLVED` (point status `SUBGROUP_POINT_GUARD_FAILED`) | men's guard clause bounds 0.0104 / 0.0113 > 0.005 (UNRESOLVED) | — (women's guards PASS) | — |
| D, joint component | `NOT_CLAIMED` | | | |
| E, novelty | `NOT_CLAIMED (existing-method engineering comparison on previously used development records)` | | | |

Controls are valid on all six views: planted SEX bit AUC 1.0 and shuffled-label null 0.487–0.517 across DEV_REPORT and ATTACK_VALIDATION.

## 1. Did a conservative SCM endpoint keep the original utility and subgroup limits?

**Utility: yes at the point. Subgroup: no at the point, for men.**

**Nomination (MODEL_SELECT).** Common threshold t* = 0.008: 298 / 300 cells, held-out occupation-NLL expenditure 0.00825 / 0.00843 nats over C0.
- Thresholds 0–0.008 were eligible under the 24 original clauses, the occupation NLL ≤ 0.008 slack and both within-SEX guards.
- 0.009 and 0.010 failed only the men's occupation Brier guard on seed 1. 0.010 is the historical nominee: 261 / 266 cells, MODEL_SELECT seed-1 men's excess 0.0059.
- The 0.008 slack did not bind; the men's guard did.

**DEV_REPORT utility.** All 24 original point clauses pass (`POINT_FEASIBLE_24`).
- Occupation NLL excess: 0.0068 / 0.0059 against U0 and 0.0085 / 0.0062 against Ucal. The historical nominee had 0.0106 against Ucal on seed 0; it is reproduced here and still fails that clause.
- Simultaneous confidence (K = 58): 3 PASS, 21 UNRESOLVED, 0 violations.

**Subgroup.**
- Men's within-SEX occupation Brier excess against U0 is 0.0054 / 0.0064, above the 0.005 guard at the point on both seeds. The family intervals are [−0.0014, 0.0121] and [−0.0003, 0.0130], so the status is UNRESOLVED, not established harm.
- Women's is −0.0027 / −0.0052 (PASS).
- Men's occupation NLL rose by 0.0148 / 0.0094 over C0, women's by 0.0076 / 0.0049. The cost of SCM's merges falls mostly on men, as in the source.

## 2. Did the reader-recovery advantage survive similar occupation-NLL expenditure?

**The selection-role match did not carry over to DEV_REPORT, so the comparison is inconclusive at the registered tolerance.**

**MODEL_SELECT match.**
- The strongest ordinary control within ±0.001 nats was found on both seeds: 129 / 105 cells, MODEL_SELECT mismatch (SCM − control) −0.00003 / +0.00015 nats.
- Its proxy was 0.645 / 0.643, against SCM's 0.617 / 0.616.
- 31 / 30 candidate releases were examined.

**DEV_REPORT recovery.**
- SCM's linked-pair AUC was 0.6234 / 0.6326, against 0.6570 / 0.6579 for that control.
- Benefit: 0.034 / 0.025. Clause bounds: 0.0225 / 0.0144. Family intervals: [0.0187, 0.0485] and [0.0107, 0.0397].
- Both recipients read lower for SCM: income −0.029 / −0.023 and occupation −0.039 / −0.045. All non-worsening clauses pass.
- The original 0.020 target at the clause bound holds on seed 0 (0.0225) but not on seed 1 (0.0144, UNRESOLVED; nested status `NOT_ESTABLISHED`). It is not cost-matched on DEV_REPORT either way.
- Reader budget: SCM's pair vectors hold a median of 16 / 17 ATTACK_FIT people, against 53 / 64 for this control. The fixed budget can under-read SCM's finer partition, which favours SCM. For these grouped releases the pair is redundant with the occupation output (§7), and the lookup reader is not population-optimal.

**But SCM spent more.** On DEV_REPORT, SCM's occupation NLL exceeded the control's by **+0.0070 / +0.0017 nats** (secondary family intervals [0.0012, 0.0128] and [−0.0044, 0.0079]).
- This is beyond the 0.001 tolerance on both seeds.
- The direction sub-status is `SCM_SPENT_MORE` on both seeds. It states only who spent more; because SCM did, the lower pair AUC is not separated from the extra occupation loss.
- SCM also spent more income NLL: +0.0064 / +0.0035 on DEV_REPORT (secondary family: ABOVE_ZERO on seed 0, [0.0015, 0.0114]; UNRESOLVED on seed 1, [−0.0020, 0.0090]) and +0.0025 / +0.0034 on MODEL_SELECT (`INCOME_EXPENDITURE_NOT_MATCHED`).
- Under the registered rule this is `MATCH_DID_NOT_CARRY_OVER`. The benefit and non-worsening clauses all pass, but only as a comparison-only status that is not cost-matched on DEV_REPORT. It is not a matched result.

**Descriptive context (not a claim).** Ordinary merging barely moved recovery at the loss levels measured.
- On MODEL_SELECT (cross-fitted proxies, not fresh readers), the ordinary trajectory's proxy stays within 0.645–0.649 from 0 to 0.0145 nats on seed 0, and within 0.643–0.648 from 0 to 0.0125 nats on seed 1.
- On DEV_REPORT, C0 minus the MODEL_SELECT-matched ordinary control is +0.0049 / +0.0047 (UNRESOLVED), and the same-count ordinary control minus the MODEL_SELECT-matched one is +0.0024 / +0.0023 (UNRESOLVED).
- No ordinary release in the final bank spent as much DEV_REPORT occupation NLL as SCM. The highest was 0.0043 / 0.0054 nats over C0, against SCM's 0.0113 / 0.0072. How an ordinary control at SCM's DEV_REPORT spend would read is therefore not measured, and the failed match is not repaired by extrapolating from the MODEL_SELECT proxies.

## 3. Did the same-capacity comparison give a different answer?

**No: it has the same confound in a larger form, and claim K is `NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD`.**
- At exactly SCM's cell count (298 / 300), the ordinary control read 0.6595 / 0.6602.
- Benefit 0.036 / 0.028: clause bounds 0.0243 / 0.0169, family-established above 0.
- SCM spent +0.0102 / +0.0055 nats more occupation NLL on DEV_REPORT (above zero on both seeds) and +0.0072 / +0.0063 more income NLL.
- Occupation non-worsening passes. The income worsening clause bounds (0.0136 / 0.0174) exceed 0.010, which is UNRESOLVED, not a violation. The comparison-only status is `COMPARISON_CLAUSES_NOT_ALL_PASS`.
- The label is decided first by the men's guard (POINT_FAILURE).
- Reader budget: ORD_CNT occupies 298 / 300 ATTACK_FIT vectors, against SCM's 297 / 299, but its people are spread more evenly (median 21 against 16 / 17). The budget bias is therefore smaller here, but it still favours SCM.
- The PUB_IBSI same-count control (γ = 1.0) reads 0.6331 / 0.6352. Its benefit of 0.010 / 0.003 is UNRESOLVED on the family interval and was not tested (`NOT_TESTED_K_NOT_ESTABLISHED`), so seed 0's clause bound (0.0022 > 0) is not a test.

## 4. Did ordinary grouping or PUB_IBSI account for the earlier difference?

**Ordinary grouping did not. The published IBSI objective read about as low as SCM, but neither comparison is cost-matched on DEV_REPORT** (descriptive: the components are `NOT_TESTED_M_NOT_ESTABLISHED`, with descriptive status `MATCH_DID_NOT_CARRY_OVER`).
- **Ordinary merging** reduced recovery very little at the expenditures measured (§2). The earlier SCM-versus-twin gap is therefore not ordinary grouping's doing.
- **PUB_IBSI (γ = 2.0)**, selected at a MODEL_SELECT match (473 / 475 cells):
  - SCM's pair benefit over PUB_IBSI (PUB_IBSI AUC − SCM AUC) is +0.006 / −0.010, UNRESOLVED on both seeds; the family intervals are [−0.008, 0.020] and [−0.025, 0.005]. This is not equivalence.
  - On seed 1, PUB_IBSI read lower than SCM while spending 0.0039 nats more occupation NLL. That seed-1 PUB_IBSI release fails three DEV utility point clauses: income accuracy loss against U0 and Ucal, and occupation NLL excess against Ucal. This is a comparator qualifier; it does not change the release's MODEL_SELECT eligibility.
  - Its DEV_REPORT match drifted too (+0.0011 / −0.0039). SCM spent more on seed 0, PUB_IBSI on seed 1.
  - SCM spent more income NLL than PUB_IBSI on both seeds (+0.0026 / +0.0011; `INCOME_EXPENDITURE_NOT_MATCHED`).
  - Reader budget: PUB_IBSI keeps 471 / 471 ATTACK_FIT pair vectors (median 13 people), against SCM's 297 / 299 (median 16 / 17). In this comparison the fixed budget can under-read PUB_IBSI more than SCM.
  - On MODEL_SELECT, PUB_IBSI at γ = 2 traced about the same or a lower proxy-versus-expenditure curve than SCM, with far fewer merges (figure, top row).
- **SEX-guided releases.** SCM and PUB_IBSI read lower than the ordinary and shuffled controls. In each of those comparisons the SEX-guided side spent more DEV_REPORT occupation and income NLL, so the lower recovery is not separated from that extra loss. SCM's specific ordering shows no measured advantage over the published objective.
- **Shuffled control (r2)** at a MODEL_SELECT match: SCM benefit 0.020 / 0.020, family-ESTABLISHED. It is not cost-matched on DEV_REPORT:
  - the match drifted, with SCM spending +0.0042 / +0.0017 more occupation NLL;
  - SCM also spent +0.0041 / +0.0026 more income NLL.

  The result is consistent with a role for the SEX-rate signal in the merge order, but it is not separated from SCM's extra loss. Shuffled controls are component comparisons, not permutation tests.

## 5. How did it compare with the strongest eligible neural and tree references?

**Not competitive: claim B is `NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD`.** The first failing condition is the men's occupation Brier point guard (POINT_FAILURE), and the comparison-only status is `COMPARISON_CLAUSES_NOT_ALL_PASS`. The recipient conditions fail as well:
- Against NEURAL_RAW, benefit +0.017 / +0.005:
  - seed 0: clause bound 0.0024 > 0, so the condition holds, but the family interval [−0.002, 0.036] is UNRESOLVED;
  - seed 1: the clause fails (bound −0.0079; UNRESOLVED).
- Against H_LJ:
  - seed 0: +0.018, clause bound 0.0046 > 0, family-ESTABLISHED (lower bound 0.0001);
  - seed 1: +0.010, clause bound −0.0030 (UNRESOLVED).
- All four reference benefits fall below the original 0.020 target at the point (POINT_SHORTFALL; MATERIAL_SUPERIORITY not established).
- **Per recipient, SCM reads higher.**
  - Income recipient: +0.021 / +0.023 against NEURAL_RAW and +0.009 / +0.025 against H_LJ. All four non-worsening clauses fail at the 0.010 margin, in two categories:
    - POINT_EXCESS against NEURAL_RAW on both seeds and against H_LJ on seed 1 (the point is above 0.010);
    - UNRESOLVED against H_LJ on seed 0 (point 0.009, clause bound 0.022).

    None is an established violation: every family interval contains 0.010.
  - Occupation recipient:
    - against H_LJ: +0.048 on seed 0 (ESTABLISHED_VIOLATION; family interval [0.021, 0.074]) and +0.034 on seed 1 (POINT_EXCESS; family interval [0.0085, 0.059] contains 0.010);
    - against NEURAL_RAW: −0.014 on seed 0 (PASS) and −0.005 on seed 1 (UNRESOLVED; clause bound 0.0105 > 0.010).
- Reader budget: no occupancy diagnostic exists for NEURAL_RAW or H_LJ. The fixed-budget and lookup-reader caveats apply to every AUC here.
- **Utility.** NEURAL_RAW spent less occupation NLL (excess against U0 0.0000 / 0.0021) and put a smaller burden on men (0.0013 / 0.0022). H_LJ's men's excess is 0.0032 / 0.0040.

## 6. Did selection-role matching carry over? The measured limitation

**No MODEL_SELECT-matched comparator carried over to DEV_REPORT.** All are point mismatches; equivalence was never established, as the statistical reviewer expected.

| Comparator | DEV occupation-NLL difference (SCM − comparator) | Direction |
|---|---|---|
| ORD_NLLM | +0.0070 / +0.0017 | SCM spent more (both seeds) |
| IBSI_NLLM | +0.0011 / −0.0039 | SCM more / comparator more |
| SHUF_NLLM | +0.0042 / +0.0017 | SCM spent more (both seeds) |

**Measured limitation.** The 0.001-nat tolerance is below the sampling resolution of held-out expenditure on these sample sizes: paired DEV standard errors are 0.0016–0.0020. Matching a comparator on its MODEL_SELECT estimate therefore selects partly on noise. The drift seen on DEV_REPORT is consistent with that: the control's spend fell by 0.0040 / 0.0028, while SCM's rose by 0.0030 on seed 0 and fell by 0.0012 on seed 1. This record does not separate noise from a real difference between the selection and evaluation people.

| Release | Fitted (plug-in) | MODEL_SELECT | DEV_REPORT |
|---|---|---|---|
| MODEL_SELECT-matched ordinary control (ORD_NLLM) | 0.0060 / 0.0070 | 0.0083 / 0.0083 | 0.0043 / 0.0054 |
| SCM | 0.0079 / 0.0080 | 0.0082 / 0.0084 | 0.0113 / 0.0072 |

Two further measured limitations:
- **Men's occupation Brier burden.** It passed on MODEL_SELECT because the nomination required it, then rose above 0.005 at the point on DEV_REPORT.
- **Recipient-level reads against the references.** SCM's income recipient read higher than both references at the point (POINT_EXCESS against NEURAL_RAW on both seeds and against H_LJ on seed 1; UNRESOLVED against H_LJ on seed 0). Its occupation recipient read higher than H_LJ's (ESTABLISHED_VIOLATION on seed 0, POINT_EXCESS on seed 1).

## 7. What is established, a point result, unresolved, unavailable or diagnostic

No claim is established, so everything in this section is descriptive under the labels above.

**Established on the family interval:**
- Pair benefit (lower SCM pair AUC) against ORD_NLLM (MODEL_SELECT match), ORD_CNT (same count) and SHUF_NLLM (MODEL_SELECT match), on both seeds. None of these is cost-matched on DEV_REPORT, because SCM spent more:

  | Comparator | Extra occupation NLL | Extra income NLL |
  |---|---|---|
  | ORD_NLLM | +0.0070 / +0.0017 | +0.0064 / +0.0035 |
  | ORD_CNT | +0.0102 / +0.0055 | +0.0072 / +0.0063 |
  | SHUF_NLLM | +0.0042 / +0.0017 | +0.0041 / +0.0026 |
- Pair benefit against H_LJ, seed 0 (lower bound 0.0001).
- SCM spent more occupation NLL than ORD_CNT on both seeds, and more income NLL than ORD_NLLM on seed 0 (ABOVE_ZERO).
- The ORD_NLLM match mismatch on seed 0 (`MISMATCH_ESTABLISHED`; SCM spent more).
- C0 minus SCM pair AUC (+0.039 / +0.030) and the 7-reader-slate ORD_NLLM minus SCM (+0.034 / +0.025), above zero.
- SCM's pair reader above its occupation-only reader (+0.015 / +0.015). This is a difference between fitted readers; see Diagnostic.
- Recipient non-worsening (PASS):
  - income and occupation against ORD_NLLM;
  - occupation against ORD_CNT;
  - occupation against NEURAL_RAW, seed 0.
- One occupation-recipient violation against H_LJ, seed 0 (`ESTABLISHED_VIOLATION`).
- 3 of the 24 utility clauses (PASS), and the women's subgroup guard on both seeds (PASS).
- Controls.

**Point results (neither established nor refuted by an interval):**
- SCM's 24 DEV utility clauses all pass at the point (`POINT_FEASIBLE_24`). The family-interval status is `CONFIDENCE_UNRESOLVED`.
- POINT_FAILURE:
  - the DEV occupation-NLL match against ORD_NLLM, both seeds;
  - the men's subgroup point guard, both seeds;
  - the IBSI_NLLM and SHUF_NLLM matches are also point mismatches (descriptive; the components were not tested).
- POINT_EXCESS:
  - income recipient against NEURAL_RAW (both seeds) and H_LJ (seed 1);
  - occupation recipient against H_LJ (seed 1).
- POINT_SHORTFALL: all four reference pair benefits are below the 0.020 target.

**Unresolved:**
- Every equivalence. All five other match endpoints are `EQUIVALENCE_UNRESOLVED`.
- The PUB_IBSI pair benefits (MODEL_SELECT-matched and same-count) and the NEURAL_RAW pair benefits; the seed-1 benefit against H_LJ.
- The men's guard. Its family intervals contain 0.005.
- 21 of the 24 utility confidence statuses.
- Recipient clauses: income against ORD_CNT (both seeds) and H_LJ (seed 0); occupation against NEURAL_RAW (seed 1).
- The secondary differences:
  - pair AUC: C0 − ORD_NLLM, ORD_CNT − ORD_NLLM, HIST_SCM − SCM;
  - income NLL against ORD_NLLM, seed 1;
  - occupation NLL against IBSI_CNT.

**Unavailable (UNAVAILABLE category):**
- Not tested, so evidence neither way:
  - the IBSI and SHUF matched components (`NOT_TESTED_M_NOT_ESTABLISHED`);
  - IBSI_CNT inside K (`NOT_TESTED_K_NOT_ESTABLISHED`).
- No final-bank role is missing: all ten roles were filled on both seeds, and every endpoint is AVAILABLE.
- **Checkpoint coverage (missing coverage, not a result).**
  - SCM's trajectory stops at 261 / 266 cells: counts below that are unreached, and `N0.010` is unreached by a hair (terminal spend 0.0099989 / 0.0099999).
  - The ordinary trajectory stops at 60 / 58 cells (C32 unreached).
  - The shuffled trajectories stop at 108–131 cells (C96, C64 and C32 unreached; C128 also unreached for one seed-1 run).
  - `N0.010` is unreached on the capped trajectories.
  - PUB_IBSI reached every registered checkpoint.
- **Deployment.** Packages exist for SCM, ORD_NLLM and IBSI_NLLM (both seeds), the frozen NEURAL_RAW (seed 0) and H_LJ (both seeds). Each reproduces its evaluated DEV_REPORT wire bitwise through the API and CSV paths, and all tested refusals hold through the documented entry points. The frozen references' raw source functions, called directly, ignore extra keys instead of refusing them (verifier D6; `PROTOCOL_POSTRUN_CLARIFICATIONS.md`). No seed-1 NEURAL_RAW package exists (the source never built one); that is a coverage gap, neither a demographic-inference result nor a method win. ORD_CNT, IBSI_CNT, SHUF_NLLM, C0 and HIST_SCM were not registered for packaging.

**Diagnostic:**
- **Historical nominee (HIST_SCM).** It reads 0.006 / 0.011 lower than the new nominee (unresolved) and still fails one DEV point clause on seed 0. Its outputs and reader scores reproduce the source bitwise.
- **Common-cell check.** For every grouped release in the bank (SCM, ORD_NLLM, ORD_CNT, IBSI_NLLM, IBSI_CNT, SHUF_NLLM, C0 and HIST_SCM, both seeds), on every role checked (ATTACK_FIT, ATTACK_VALIDATION, MODEL_SELECT and DEV_REPORT), the occupation output identifies the shared cell and income is a function of it (distinct occupation vectors = occupied cells = distinct pair vectors). The check was not computed for NEURAL_RAW or H_LJ. For these grouped releases a Bayes-optimal reader therefore gains nothing from the linked pair beyond the occupation output for these people. The fitted readers still differ: SCM's pair reader exceeds its occupation-only reader by 0.015 / 0.015. That is a difference between fitted readers, not new information. The lookup reader is not population-optimal.
- **Reader budget.**
  - With 8,001 ATTACK_FIT people, SCM's 297 / 299 occupied pair vectors hold a median of 16 / 17 people each. The 129 / 105 vectors of the MODEL_SELECT-matched ordinary control hold a median of 53 / 64.
  - 87 % of ATTACK_FIT people sit in SCM vectors with at least 15 people, against 99 % for that control.
  - Other roles: ORD_CNT median 21, SHUF_NLLM 33 / 30, PUB_IBSI (IBSI_NLLM) 13.
  - The fixed budget can under-read finer partitions. Against the ordinary and shuffled controls this favours SCM; against PUB_IBSI it favours PUB_IBSI. The bias was disclosed before fitting.
- **Class-wise occupation NLL.** SCM changes contributions little. It predicts occupation class 3 less often (0.026 / 0.020 against C0's 0.036 / 0.028; true 0.090). It predicts class 0 slightly more often on seed 0 (0.594 against 0.589) and slightly less often on seed 1 (0.587 against 0.589).
- **Routing and teacher-prediction differences.** These are the source's out-of-fold versus runtime diagnostics, admitted unchanged; they are in `report/DEV_REPORT_RESULTS.json` → `diagnostics`.

## 8. Registered bets (scored; they decided nothing)

| Bet | Outcome |
|---|---|
| P1 (0.95) | held |
| P2 (0.70) | held |
| P3 (0.70) | held |
| P4 (0.35) | not held |
| P5 (0.55) | held (scored on ORD_NLLM's MODEL_SELECT role status MATCHED; the DEV_REPORT match did not carry over, so the comparison it scores is not cost-matched) |
| P6 (0.75) | held |
| P7 (0.50) | not held |
| P8 (0.10) | not held |

Mean Brier score over non-void bets: 0.104. Under the alternative reading of P5 (the DEV_REPORT point match instead of the MODEL_SELECT role status), P5 would be not held and the mean Brier 0.116 (verifier open issue).

## 9. Selection effort

| Selection | Candidates examined |
|---|---|
| SCM | 11 thresholds (11 releases per seed) |
| Ordinary (MODEL_SELECT match) | 31 / 30 releases |
| PUB_IBSI (MODEL_SELECT match) | 3 γ × 29–32 releases per seed |
| Shuffled (MODEL_SELECT match) | 5 r × 26–29 releases per seed |
| PUB_IBSI same count | 3 γ |

Selecting the most protective eligible comparator is conservative for SCM. It is outside the intervals. ORD_CNT is one fixed label (the same-count companion). The ORD_NLLM pick rests on thin MODEL_SELECT proxy margins (0.00014 / 0.00029 over the runner-up); this is the registered strongest-proxy rule, applied correctly (verifier check 4).

**Supersession note.** The commit message of `b9299804f` summarised an earlier draft as "PUB_IBSI accounts for most of the earlier difference". This text supersedes it after the claims review: the PUB_IBSI comparison is unresolved and not cost-matched (§4).
