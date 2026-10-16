# Advisor brief: ACS frozen hybrids (exploratory development follow-up)

**Decision: `H_LJ_FIXED_COMPOSITION_NOT_SUPPORTED`.**
- **Claim A:** `RECIPE_BENEFIT_NOT_ESTABLISHED`.
- **Claim B:** `NOT_SUPERIOR_TO_ELIGIBLE_NEURAL_CONTROL`. This is the limiting comparison.
- **Claim C:** `SUBGROUP_UNRESOLVED`. This was settled by published values before any reader ran.
- **Controls:** valid.

> Exploratory development result on already-opened DEV_REPORT data; the composition was fixed after its components' DEV_REPORT results were public, so error rates are nominal; conditional on the two fitted seed pairs (shared people, not independent samples), the frozen defense models and the fitted readers, in the unweighted DEV_REPORT household superpopulation; not confirmation; source labels unchanged.

**What was tested.** H_LJ joins LOCAL_TREE's income output to JOINT_TREE's occupation output within each of the two existing model seeds. It was fixed before any attacker was fit to its linked outputs, but after its components' DEV_REPORT results were public, including each recipient's recovery.
- **Combinations:** all four were attacked, H_LL (= LOCAL_TREE), H_LJ, H_JL and H_JJ (= JOINT_TREE).
- **Controls:** the eligible neural control NEURAL_RAW (ρ1.0 E2), the matched task-only continuation TASK_REFIT E20, and the teacher.
- **What was fitted:** no defense model was trained. The only new fits were 250 reader fits: 238 recovery attackers and 12 pipeline-control readers, all trained on ATTACK_FIT. Winners were selected on ATTACK_VALIDATION.
- **Where it was measured:** the already-exposed DEV_REPORT people (5,943 persons in 3,955 households).

## 1. Did H_LJ inherit the expected task usefulness?

**Yes, at the point.**
- **Point clauses:** all 24 DEV_REPORT utility clauses pass. These are 2 tasks × 2 references × 3 metrics × 2 seeds, with caps of 0.010 for accuracy loss, 0.010 for NLL excess and 0.005 for summed Brier excess.
- **Inheritance is exact:** H_LJ's per-person income metrics are LOCAL_TREE's and its occupation metrics are JOINT_TREE's, so every value equals the published one.
- **Eligibility rule:** H_LJ also meets the source's MODEL_SELECT eligibility rule.
- **Settled in advance:** all of this follows from the components' published results and was recorded before readers ran. It is not a finding of this campaign.

**Overall confidence (a separate field) is unresolved: `CONFIDENCE_UNRESOLVED`.** Under the 38-interval simultaneous family, 10 utility clauses pass, 14 are unresolved and none is violated. 15 of 24 pass at their one-sided clause bounds. The tightest clauses are:
- income accuracy loss on seed 1 (identical vs U0 and Ucal): 0.0089, with simultaneous upper bound 0.020 against the 0.010 cap;
- occupation NLL excess vs Ucal on seed 0: 0.0087, upper bound 0.019.

## 2. Did the combined outputs reveal less, and by how much?

Coalition SEX recovery is the DEV_REPORT AUC of the linked-pair attacker chosen on ATTACK_VALIDATION from the fixed reader slate. Lower means the outputs reveal less to these attackers.

| seed | H_LJ | TASK_REFIT | benefit vs TASK_REFIT (K = 38 interval) | clause bound vs 0.020 target |
|---|---|---|---|---|
| 0 | 0.641 | 0.664 | 0.022 [0.008, 0.037] | 0.012 (fails) |
| 1 | 0.643 | 0.666 | 0.024 [0.011, 0.036] | 0.014 (fails) |

- **Against TASK_REFIT.** H_LJ reveals less than continued task fitting, established above zero on both seeds. The required 0.020 margin is not established. Both points sit just above 0.020, but the clause bounds fall below it, so the margin fails on precision: unresolved, not a shortfall.
- **Against LOCAL_TREE (H_LL).** Swapping in JOINT's occupation output lowers coalition recovery by 0.015 [0.006, 0.024] on seed 0, which is established. On seed 1 the drop is 0.006 [−0.004, 0.016], which is unresolved.
- **Each recipient alone** (already published; not findings of this campaign). H_LJ's income output is LOCAL's: 0.014 and 0.018 below TASK_REFIT. This is the source's own endpoint, settled before readers. Its occupation output is JOINT's: 0.089 and 0.061 below TASK_REFIT. Those points were published; only the standard errors are new. No recipient worsens.

## 3. Did the neural control still do better?

**Not established either way.**
- **Coalition AUC:** at the point the neural control is slightly lower, by 0.001 and 0.005 (NEURAL_RAW 0.640 and 0.638). H_LJ's benefit over it (NEURAL_RAW AUC minus H_LJ AUC) is −0.001 [−0.018, 0.015] and −0.005 [−0.021, 0.011]. Both intervals include zero, and equivalence was not tested.
- **Superiority:** H_LJ is not superior.
- **The 0.020 target:** it is an established shortfall on both seeds.
- **Recipient level** (descriptive, unadjusted):
  - H_LJ's occupation output alone reveals less than NEURAL_RAW's: 0.560 vs 0.622, and 0.583 vs 0.623.
  - Its income output reveals more on seed 0 (0.558 vs 0.546) and about the same on seed 1 (0.549 vs 0.551).
  - The linked pair is no better.
- **Utility:** both services pass all 24 utility clauses at the point, so H_LJ shows no advantage over the neural control here.

## 4. Does a benefit require both JOINT outputs, including the income output that failed utility?

**Not both, but the larger benefit needs JOINT's income output, and that output fails utility.**
- **JOINT occupation alone (H_LJ):** a benefit over TASK_REFIT that is established above zero but not at 0.020, and no benefit over NEURAL_RAW.
- **JOINT income alone (H_JL):** this already gives the larger, descriptive reductions.
- **Both (H_JJ):** adding JOINT's occupation output lowers recovery further.

Coalition AUC drop from H_LL, at the point:

| swap | seed 0 | seed 1 |
|---|---|---|
| occupation only (H_LJ) | 0.015 | 0.006 |
| income only (H_JL) | 0.032 | 0.024 |
| both (H_JJ = JOINT_TREE) | 0.053 | 0.042 |

The table shows differences of full-precision pair AUCs, rounded to 3 decimals. Differences of the rounded AUCs in START_HERE can differ by 0.001. Only the occupation-only row is a registered endpoint (the increment, with K = 38 intervals in question 2); the other rows are point differences without intervals.

- **Post-hoc observation, not a nominee.** Only the combinations with JOINT's income output show the larger reductions.
  - **vs TASK_REFIT:** H_JL and H_JJ are lower by more than 0.020 (H_JL by 0.039 and 0.041; H_JJ by 0.060 and 0.060), with secondary-family intervals wholly above 0.020.
  - **vs NEURAL_RAW:** at the point they also lie below it (H_JL by 0.015 and 0.013; H_JJ by 0.037 and 0.031). H_JL's seed-1 interval includes zero, and neither reaches the 0.020 target.
  - **Status:** all of these figures are secondary-family and descriptive. H_JJ's values are JOINT_TREE's, already published by the source; H_JL's pair result is new. Neither combination was a nominee, neither replaces H_LJ, and both are utility-ineligible.
- **Both are utility-ineligible.** JOINT's income output fails income utility:
  - on MODEL_SELECT (the eligibility rule): summed Brier excess on seed 0, accuracy loss on seed 1;
  - on DEV_REPORT, seed 0: summed Brier excess, 0.0054 vs U0 and 0.0070 vs Ucal (cap 0.005);
  - on DEV_REPORT, seed 1: accuracy loss 0.0140 (cap 0.010) and summed Brier excess, 0.0067 vs U0 and 0.0077 vs Ucal.
- **The four-way interaction is unresolved.** It is (AUC_LL − AUC_LJ) − (AUC_JL − AUC_JJ): −0.006 [−0.015, 0.003] on seed 0 and −0.013 [−0.031, 0.006] on seed 1 (secondary family).
  - The data do not establish that the occupation swap's effect depends on the income output it is paired with, and the intervals are wide.
  - At the point, the swap helped more next to JOINT income (0.021 and 0.019) than next to LOCAL income (0.015 and 0.006).
- **What this is not.** The interaction is descriptive.
  - It is not a theorem about information synergy.
  - It is not causal evidence about the training objective.
  - It is not a population privacy guarantee. Every value is the recovery AUC of the fitted reader slate, conditional on the frozen models, in the unweighted DEV_REPORT household superpopulation.

## 5. Are subgroup usefulness and overall confidence established or unresolved?

**Both are unresolved. The three status fields are reported separately and never merged.**
- **Subgroups: `SUBGROUP_UNRESOLVED`.**
  - The male group (sex0, ACS SEX 1) has occupation Brier excess vs U0 of 0.0032 and 0.0040. Both its one-sided clause bounds (0.0081 and 0.0087), which decide claim C, and its K = 38 upper bounds (0.0098 and 0.0103) exceed the 0.005 cap.
  - For the female group (sex1, SEX 2), seed 1 passes. Seed 0 is unresolved at K = 38, though it passes its clause bound.
  - No harm is established. These guards are exactly JOINT_TREE's published values.
- **Overall confidence: `CONFIDENCE_UNRESOLVED`** (10 PASS, 14 UNRESOLVED, 0 violations).
- **Overall point feasibility: `POINT_FEASIBLE_24`** on DEV_REPORT and MODEL_SELECT.

## 6. What runnable artifact exists?

**Two private deployment packages, H_LJ_S0 and H_LJ_S1.**
- **Inputs and outputs:** they take the six permitted raw ACS fields (AGEP, WKHP, COW, SCHL, MAR, POBP) and return each recipient's own probabilities and decision.
- **No cross-recipient dependence:** each recipient is computed only from its own frozen component: the original LOCAL_TREE or JOINT_TREE checkpoint file and that task's teacher-tree file, whose hashes match the source model manifest. Neither needs labels or the other recipient's output.
- **Reproduction:** on the 5,943 DEV_REPORT rows they reproduce the attacked outputs bitwise, probabilities and decisions.
- **Refusals:** they refuse a wrong column order, a SEX or PUMA column, and hidden-output requests.
- **Restore:** both were restored from the backup alone and re-deployed exactly (`LOCK_NOTES.md`, "Custody V1"; `BACKUP_VERIFICATION.json`).

## The limiting comparison, and what to do

The limiting comparison is the eligible neural control (claim B).
- **Utility:** H_LJ keeps the trees' point utility. Simultaneous confidence and subgroup suitability are unresolved.
- **vs TASK_REFIT:** it lowers coalition recovery by 0.022 and 0.024 at the point, just above the 0.020 margin. The clause bounds (0.012 and 0.014) fall below it, so the margin is unresolved.
- **vs NEURAL_RAW:** even if that margin had been established, claim B would still fail. H_LJ's coalition AUC is 0.001 and 0.005 above NEURAL_RAW's at the point, and the 0.020 superiority target is an established shortfall.

At the point, coalition recovery falls below NEURAL_RAW's only in the combinations that use JOINT_TREE's income output (H_JL and H_JJ; descriptive, secondary family). That output fails income utility, and those combinations are a post-hoc observation, not nominees.

**This fixed hybrid test is closed.** No nearby sweep is proposed, and nothing here justifies opening the sealed assessment.

If described at all, H_LJ is an exploratory composition of existing models whose measured coalition recovery could not be distinguished from the neural control's, and whose subgroup suitability is unresolved. It is not a novel algorithm, not evidence that joint training is better, and not a population privacy guarantee.

## Verification, custody and process

- **Independent verification: 10/10 PASS** (`independent/INDEPENDENT_VERIFICATION.json`).
  - **Method:** the verifier used separate metric and inference code.
  - **Refits:** it refit 17 of the 250 reader fits from scratch: the five declared winning pair readers and the 12 control readers. All reproduced bitwise.
  - **Replays:** it replayed the other 233 saved models.
  - **Re-derivations:** it re-derived every endpoint, status, distance, claim and the conjunction, with no numerical discrepancy.
  - **Truth access:** it re-read DEV_REPORT truth once, logged.
- **Source records:** the source's verification stays 10/11 (D1 was a transport-annotation failure, not a numerical failure). D2 applies: the source extractor computed sealed records transiently in memory, and the sealed assessment stays closed. This study is an exploratory development follow-up, not confirmation, independent replication or rescue. The source labels stand: `RECIPE_EFFECT_NOT_ESTABLISHED`, `NOT_SUPERIOR_TO_ALL_ELIGIBLE` and `SUBGROUP_UNRESOLVED`.
- **Process notes** (`LOCK_NOTES.md`):
  - one forecast timestamp correction;
  - rehearsal-only engineering amendment E1, made before truth was read;
  - three synthetic-label rehearsals for E1. V1 stopped at the source-reproduction halt by design. V2 completed the report, but its summary script used two stale field names (a script-only fix). V3 ran end to end. None read DEV_REPORT truth, and all three are charged in the queue ledger.
