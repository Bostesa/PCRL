# Research decision: ACS development replication of the protected-tree recipe

**Population:** Florida, ACS 2019 1-Year PUMS (person file `psam_p12.csv`). This is a custom two-task, shared-worker benchmark.

**Estimand:** a recipe replication with new models, inputs and task definitions. This is a development study. It is **not** confirmation of the frozen Adult packages, and no Adult number was transferred.

**Decision labels (pre-registered claim rules, `SCIENCE_LOCK.json`):**

| Claim | Decision |
|---|---|
| RECIPE_EFFECT | **RECIPE_EFFECT_NOT_ESTABLISHED** |
| COMPETITIVE_COMPARISON | **NOT_SUPERIOR_TO_ALL_ELIGIBLE** (comparison complete; NEURAL_RAW is the only eligible baseline) |
| SUBGROUP_SUITABILITY | **SUBGROUP_UNRESOLVED** (no established harm) |
| Benchmark | **SUPPORTED** (usefulness gate SUPPORTED_CONTINUE: the teacher beats constant prediction, and class support passed) |

**One-sentence answer.** The fixed local leaf-score recipe transferred in one sense and not the other:
- **Transferred:** it kept overall utility at the point. Against matched continued task fitting, it cut each recipient's own SEX recovery by 0.014–0.018 AUC for income and 0.061–0.067 for occupation.
- **Did not transfer:** it barely reduced what a linked coalition of the two recipients recovers. That coalition reduction was +0.007 and +0.0175 AUC against matched continued task fitting, short of the 0.020 target. The shortfall is established on seed 0 and unresolved on seed 1.
- **Comparison:** the eligible neural adversarial control had lower coalition recovery at the point on both seeds (established on seed 0, unresolved on seed 1).

---

## 1. What was run

**Population and roles.** The admission order was frozen before any data were read. Florida was first in that order and was admitted after a bounded prior-use audit with manual review of every hit (`POPULATION_DECISION.json`).
- 74,091 eligible civilian workers.
- Households were assigned to roles by a published salt.
- 20% of households were sealed (14,693 people, 9,853 households). They stayed closed: no features, labels or predictions were written or used, and no model saw them. The extraction's eligibility pass parsed their records in memory only (disclosure D2; `SEALED_ASSESSMENT_STATUS.md`).
- Development roles (people): MODEL_FIT 20,000, CALIBRATION 2,982, MODEL_SELECT 8,001, ATTACK_FIT 8,001, ATTACK_VALIDATION 3,000, DEV_REPORT 5,943 (3,955 households).

**Tasks.**
- Income: adjusted personal income above $50,000 in 2019 dollars (PINCP × ADJINC).
- Occupation: five official broad civilian occupation groups, mapped code by code from the official 2018 ACS PUMS occupation code list (525 codes; military and 9920 excluded).

**Inputs** (145 columns) are AGEP, WKHP, COW, SCHL, MAR and POBP only. SEX is the protected attribute.

**Models,** each fitted for two paired seeds on MODEL_FIT only:
- **U0:** the validated source neural network, with input width and occupation output count changed.
- **Ucal:** one CALIBRATION temperature per task.
- **T_TREE:** CatBoost depth 4, learning rate 0.05, at most 800 trees, l2 3, internal early stopping.
- **LOCAL_TREE:** the transported L_TREE procedure at ρ = 1, E20 (20 stages of 40 updates per recipient), with the FIT-only coefficient rule. The zero-correction witness was exact.
- **TASK_REFIT:** the identical loop with privacy terms disabled.
- **NEURAL_RAW:** neural adversarial protection from U0. Grid: ρ ∈ {1, 3, 10} × E{2, 5, 10, 20}.
- **FAGTB:** the faithful binary income port (depth 9, 300 trees, learning rate 0.01, max_features 20), plus the documented FAGTB_MC_ADAPTATION for occupation. λ ∈ {0.03, 0.06, 0.15}.
- **Official FARE:** settings F1–F3, plus zero-fairness compression twins Z1–Z3.
- **JOINT_TREE:** the optional diagnostic (adds the pair term). It is not a nominee.

**Selection, readers and reporting.**
- Controls were selected on MODEL_SELECT under the frozen rule: utility feasibility first, then a recovery proxy.
- The reader slate is 7 readers × 3 views, fitted on ATTACK_FIT and selected on ATTACK_VALIDATION.
- DEV_REPORT was opened **once** for analysis, at 2026-10-16T00:20:20Z. That was after the REPORTING_LOCK, and after the public receipt (`7d23e0e`). The verifier's later re-read, for recomputation only, is disclosed in `LOCK_NOTES.md`.

## 2. Did the stronger teacher create headroom?

**Yes, modest, at the point** (descriptive, no interval; forecast F1). On MODEL_SELECT, T_TREE had lower NLL and Brier and higher accuracy than U0 on both tasks and both seeds; balanced accuracy was mixed. The pre-registered gate is a different test: the teacher must beat constant prediction, and class support must pass. That gate was SUPPORTED_CONTINUE.

| Task | NLL lower by | Brier lower by | Accuracy gain |
|---|---|---|---|
| Income | 0.0066 / 0.0068 | 0.0041 / 0.0045 | +0.12 / +0.06 pp |
| Occupation | 0.0121 / 0.0082 | 0.0051 / 0.0036 | +0.64 / +0.61 pp |

`STAGE_A_HEADROOM.json` gives the full table: NLL, summed Brier, accuracy, gain over constant prediction, balanced accuracy and per-class recall.

The descriptive DEV_REPORT contrast is the same: for occupation against U0, T_TREE has NLL excess −0.011 / −0.009 and accuracy loss −0.008 / −0.004.

## 3. Did the fixed local recipe keep utility and reduce recovery?

**Utility: kept at the point.**
- All **24/24** DEV_REPORT clauses pass at the point (both tasks × U0/Ucal × accuracy/NLL/Brier × two seeds).
- At the one-sided α/3 bounds, 17/24 pass. Most of the 7 that do not are accuracy-loss clauses, whose bounds are wide. With the independent verifier's bootstrap seed the count is 18/24: the occupation Ucal NLL clause on seed 0 sits within Monte Carlo error of its cap.
- In the K = 40 simultaneous intervals: 14 PASS, 10 UNRESOLVED, **0 established violations**.
- Most of the teacher's occupation headroom was spent on privacy. Against U0, LOCAL_TREE's occupation NLL excess is +0.0007 / −0.0003, where TASK_REFIT's is −0.012 / −0.010.

**Recovery: each recipient's own recovery fell; the coalition's barely did.** Values are DEV_REPORT AUCs, seed 0 / seed 1.

| Service | Income recipient | Occupation recipient | Linked coalition |
|---|---|---|---|
| TASK_REFIT (matched control) | 0.572 / 0.567 | 0.650 / 0.645 | 0.664 / 0.666 |
| **LOCAL_TREE** | 0.558 / 0.549 | **0.582 / 0.584** | **0.657 / 0.649** |
| T_TREE (teacher) | 0.569 / 0.568 | 0.650 / 0.649 | 0.664 / 0.664 |
| U0 (neural reference) | 0.560 / 0.567 | 0.641 / 0.638 | 0.654 / 0.649 |

**Recipient non-worsening vs TASK_REFIT** (all PASS): LOCAL_TREE minus TASK_REFIT is −0.014 / −0.018 (income) and **−0.067 / −0.061 (occupation)**. The simultaneous intervals lie wholly below zero. The reductions are therefore established in the K = 40 reporting family (approximate simultaneous 95% coverage, conditional on the fitted models and frozen readers). The pre-registered clauses test only non-worsening (≤ +0.010), which all four PASS.

**Coalition benefit vs TASK_REFIT** (target: one-sided lower bound > 0.020):

| Seed | Point | One-sided α/3 bound | K = 40 simultaneous interval | Status |
|---|---|---|---|---|
| 0 | +0.0070 | lower −0.0020 | [−0.0051, +0.0191] | **ESTABLISHED_SHORTFALL** against 0.020 |
| 1 | +0.0175 | lower +0.0092 | [+0.0063, +0.0286] | UNRESOLVED against 0.020 |

**This is not an artefact of the winning reader.** Across all seven readers, the same-reader coalition differences between TASK_REFIT and LOCAL_TREE are 0.006–0.019. In the occupation view the picture differs by reader type (descriptive slate matrix):
- the five probability-input readers (OUT_LR, OUT_MLP, OUT_HGB, LOG_LR, LOG_MLP) show differences of 0.048–0.082;
- CONF_HGB (maximum probability only) and DEC_LR (decisions only) show −0.008 to +0.023.

**Interpretation, descriptive.**
- The local objective makes each recipient's output less informative on its own.
- The aligned pair still carries most of the teacher's SEX signal: LOCAL_TREE coalition AUC is 0.657 / 0.649, against the teacher's 0.664 / 0.664. That is 91–96% of the teacher's above-chance coalition AUC.
- JOINT_TREE, which adds only the pair term, cut the coalition AUC to 0.603 / 0.606. It fails 2–4 income clauses (Brier and accuracy against U0/Ucal), and it was a diagnostic, never nominated.
- Descriptively, in Florida the coalition is where the local recipe's reduction falls short. One unselected, utility-infeasible diagnostic (JOINT_TREE) is consistent with this, but it does not establish the mechanism.

## 4. Did it beat continued task fitting and the closest published tree approach?

**Continued task fitting:** the 0.020 coalition margin was **not reached**.

**Competitive comparison** (coalition only; margin 0; eligible baselines only). The comparison is complete:
- FAGTB is two-task valid.
- LOCAL_TREE meets the same utility requirements: MODEL_SELECT 24/24 at the point and DEV_REPORT 24/24 at the point.

Coalition benefit is baseline AUC minus LOCAL_TREE AUC, seed 0 / seed 1:

| Baseline | Selection status | Coalition benefit | Status | Utility |
|---|---|---|---|---|
| NEURAL_RAW ρ1 E2 | **ELIGIBLE** | −0.0167 / −0.0112 | seed 0 **ESTABLISHED_SHORTFALL**; seed 1 unresolved | 24/24 points pass |
| FAGTB λ0.03 (binary + MC adaptation) | INELIGIBLE (no MODEL_SELECT-feasible λ) | −0.0088 / +0.0069 | descriptive only | fails 12/12 clauses per seed (income 1.1 / 2.8 pp less accurate than U0, occupation 5.5 / 5.7 pp less accurate) |
| FARE F1 | INELIGIBLE (no feasible setting) | −0.0171 / −0.0157 | descriptive only | fails 12/12 clauses per seed (occupation accuracy 0.469) |
| FARE Z1 (compression twin of F1) | — | descriptive | — | coalition AUC 0.667 / 0.666 |

- **NEURAL_RAW (eligible):** LOCAL_TREE leaks more. The eligible neural control has lower coalition recovery at feasible utility (established on seed 0, unresolved on seed 1).
- **FAGTB:** coalition recovery is about the same as LOCAL_TREE, at much worse utility.
  - One interpretation is that the faithful published configuration is underfit here. At λ = 0, 300 trees × 0.01 shrinkage caps the score within 3 logits of its initialisation.
  - But all runs used λ > 0, where leaves can exceed 1 (`baselines/FAGTB_AUDIT.md` §8), so "underfit" is an interpretation, not a measured cause.
- **FARE:** the fair tree removed more coalition signal than its no-fairness twin (0.640 / 0.633 vs 0.667 / 0.666).
  - But F1 and its twin both fail every utility clause, so most of the cost is shared with compression.
  - F1's extra cost over Z1 is occupation accuracy +0.5 / +0.6 pp and occupation NLL +0.005 / +0.007.

**Decision: NOT_SUPERIOR_TO_ALL_ELIGIBLE.** The nested material-superiority decision is NOT_ESTABLISHED.

**What separates LOCAL_TREE from the published tree method** (descriptive; growing new trees under an adversary versus keeping the teacher's routing and changing leaf scores):
- Under matched MODEL_FIT access and a bounded three-point search, the routing-preserving recipe keeps utility where faithful FAGTB does not.
  - This contrast is confounded with the faithful configuration's fixed shrinkage schedule, so it is not evidence that routing preservation causes it.
  - FAGTB was not beaten; it was ineligible.
- On the protection goal both are weak.
- This replication does not establish algorithmic novelty.

## 5. Was usefulness kept within both SEX groups?

**Unresolved.** The guard is occupation summed-Brier excess vs U0, cap +0.005, for both seeds. PASS and HARM labels are read only from the K = 40 simultaneous intervals. The α/3 clause bounds are marginal, and they are not counted after the failed conjunction.

| SEX group | Point (seed 0 / seed 1) | α/3 clause upper bound (marginal) | K = 40 simultaneous interval and status (seed 0 / seed 1) |
|---|---|---|---|
| Female | −0.0013 / −0.0038 | 0.0040 / 0.0012 (both ≤ 0.005) | [−0.0084, +0.0058] UNRESOLVED / [−0.0106, +0.0030] PASS |
| Male | +0.0002 / +0.0025 | 0.0051 / 0.0071 (both > 0.005) | [−0.0065, +0.0068] UNRESOLVED / [−0.0038, +0.0087] UNRESOLVED |

- **Decision: SUBGROUP_UNRESOLVED.** SUBGROUP_QUALIFIED fails because the male clauses fail. The simultaneous intervals cross the guard for men on both seeds and for women on seed 0. Harm is not established, and suitability is not established either.
- In this population, the group with the positive point excess is male. That differs from the Adult diagnostic, as the handoff anticipated.
- Under the lock's separation rule, any recommendation must state that subgroup suitability is unresolved.
- Per-group NLL, accuracy and recall, and the teacher and TASK_REFIT contrasts, are in `SUBGROUP_TABLE.csv`.
- **Monte Carlo sensitivity (independent verifier).** With a different bootstrap seed, the male seed-0 upper bound is 0.00492, against 0.00508 under the registered seed, so it is within Monte Carlo error of the 0.005 cap. The decision does not change, because the male seed-1 clause fails under both seeds. The reported values use the registered seed.

## 6. Status ledger

| Item | Status |
|---|---|
| Teacher headroom over U0 (MODEL_SELECT, both tasks and seeds) | **observed at the point** (descriptive, no interval; forecast F1 occurred). The pre-registered gate (teacher useful over constant prediction, plus class support) is SUPPORTED_CONTINUE |
| LOCAL_TREE overall utility | **24/24 at the point** (no error control); 17/24 at α/3 bounds; 0 violations; 10 unresolved simultaneous intervals |
| Recipient-level recovery reduction vs TASK_REFIT | **established**: all four simultaneous intervals lie wholly below 0 (reporting family; not a pre-registered claim decision) |
| Coalition benefit lower bound > 0.020 vs TASK_REFIT | **not established**: seed 0 established shortfall; seed 1 unresolved (positive reduction, interval [+0.006, +0.029]) |
| Superiority vs eligible NEURAL_RAW | **not established**: seed 0 established shortfall (NEURAL_RAW lower); seed 1 unresolved |
| FAGTB, FARE comparisons | **descriptive** (both ineligible on utility) |
| Subgroup usefulness | **unresolved**: men on both seeds, women on seed 0 (simultaneous intervals); no harm established |
| Sealed assessment | **closed**; nothing written or used beyond counts and a key-set hash. The extraction's in-memory parse of sealed records is disclosed as D2 |
| Documentation disclosures | D1 transport labelling, D2 sealed in-memory parse, D3 FAGTB seeding text (`LOCK_AMENDMENTS.md`); claims review applied (`CLAIMS_REVIEW.md`) |
| Benchmark support | **supported** |
| Required baselines | **complete**: FAGTB occupation is the documented adaptation, not an official implementation |
| JOINT_TREE | diagnostic only: lower coalition recovery, utility-infeasible on income clauses |
| Controls | shuffled null 0.481–0.519; planted bit 1.000 (both seeds, all views) |
| Sensitivity (percentile bounds) | one disagreement (occupation Ucal NLL, seed 0), disclosed; no decision depends on it |
| Independent verification | **10 of 11 checks PASS** with separate code (`INDEPENDENT_VERIFICATION.json`). Every point estimate, clause, nomination, bound and decision it recomputed from the stored outputs reproduces; reader refits are bitwise; deployment parity is 10/10. Document numbers were spot-checked only. The claims review later corrected three rounding errors in the documents, one of which (JOINT_TREE 0.604 / 0.607) the spot-check had accepted. Check 1 FAILS on a code-labelling defect, not behaviour (disclosure D1 in `LOCK_AMENDMENTS.md`) |

## 7. Forecasts (registered at the protocol seal, before any label)

Six of the 14 registered events occurred:
- F1: teacher headroom.
- F2: LOCAL_TREE utility at the point.
- F7: NEURAL_RAW eligible.
- F9: FAGTB two-task valid.
- F13: null controls in range.
- F14: planted controls above 0.95.

The other eight did not occur: the coalition point target, the recipe effect, competitive superiority, an incomplete comparison, FARE or FAGTB eligibility, subgroup qualification and established subgroup harm. Mean Brier score: 0.099 (`FORECAST_SCORES.json`).

## 8. Is a later sealed assessment justified?

**No, not for the claims as specified.**
- At the development points, the recipe effect is FUTILE: both seeds' coalition benefits lie below the 0.020 margin, and no assessment size can establish the claim if those points are accurate. Planning treats the development points as the true values for the same frozen models and readers; the seed-1 coalition interval [+0.006, +0.029] still includes values above 0.020.
- The competitive claim against the eligible neural control is also FUTILE (negative points). `DEV_REPORT_RESULTS.planning` has per-clause and conjunction power at 1–32 × the DEV_REPORT household count.
- Opening the sealed 20% would spend Florida 2019 to confirm a failure.

**Runnable model:** yes. There are raw-field deployment packages for T_TREE and LOCAL_TREE (both seeds) and NEURAL_RAW ρ1 E2 (seed 0), verified label-free (`DEPLOYMENT_RESULTS.json`). The 743 private files were restored byte-identically from the same-device backup and, separately, from an off-device copy on an external drive. A LOCAL_TREE_S0 redeploy and a reader re-score from the off-device restore matched exactly (`BACKUP_VERIFICATION.json`). The immediate on-drive read-back may have been served partly from the file cache, so a cold re-verification command is recorded there.

## 9. Next decision (follows the results; not a sweep)

The informative finding is descriptive: in this benchmark, local per-recipient protection lowered the linked coalition's recovery by less than the pre-registered 0.020 margin. If the programme continues, the candidate is a new, separately registered development study of coalition-aware objectives. The JOINT_TREE pair term is the obvious starting point; it would have to meet the income utility clauses it failed here. That study must be registered fresh. This campaign nominated nothing new, and every Florida 2019 sealed household stays closed.

## 10. Scope and limits

- **Intervals:** conditional on the fitted models and frozen readers. The two seeds share people and are not independent samples.
- **Variance:** the household bootstrap is not ACS survey-design variance. PWGTP-weighted values are point summaries only. Weighted coalition AUCs show the same ordering, TASK_REFIT > LOCAL_TREE > NEURAL_RAW, on both seeds: TASK_REFIT 0.660 / 0.661, LOCAL_TREE 0.647 / 0.642, NEURAL_RAW 0.635 / 0.629. Weighted U0 (0.647 / 0.636) sits marginally above LOCAL_TREE on seed 0 (0.6471 vs 0.6466), unlike the unweighted values.
- **Readers:** finite readers bound recovery from below; they are not certificates.
- **Adversary:** auxiliary labelled releases plus one target release, with perfect linkage. This is the auxiliary-data adversary of Song & Shmatikov (ICLR 2020) applied to released outputs, and the downstream-consumer setting of FARE (Jovanović et al., ICML 2023). It is not a new adversary.
- **Population independence:** Florida 2019 had no use found in the searched records. That is not certified non-use or person-level independence (`POPULATION_DECISION.json`).
- **Out of scope:** race protection was never claimed or tested.
- **Teacher capacity:** the occupation teacher used 800 and 799 trees against the registered 800-tree cap, so internal early stopping barely bound. Its edge over U0 is thin (NLL 0.008–0.012). A larger cap was not tried, because it would be a post-hoc retune.
- **Verification scope:** the registered sealed household-key-set hash covers eligible sealed households only. It cannot be reproduced from identifiers alone without reading eligibility variables for sealed households, which the access rule forbids after extraction. The extraction's own in-memory parse is disclosure D2. The verifier confirmed the role assignment for every exported row and the disjointness of roles.
