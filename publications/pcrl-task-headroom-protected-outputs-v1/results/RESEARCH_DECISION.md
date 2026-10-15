# Research decision: task headroom and protected prediction outputs

**Completion label:** COMPLETE (see the status ledger; off-device custody is pending).
**Scientific label:** `POINT_FEASIBLE_NO_COMPETITIVE_CRITERION`
**Joint label:** `JOINT_COMPONENT_NOT_ESTABLISHED` (the local arm was the FIT2-selected primary; J_TREE vs L_TREE on the coalition view is unresolved)
**Stage A:** `TASK_MODEL_PREREQUISITE_MET`

All people are repeatedly used Adult development people. FIT2 and INNER are exposed development roles, not confirmation. The design came after every predecessor result. Intervals are conditional on the fitted models and readers, and do not cover the adaptive research history. Both model seeds share the same INNER people. The pair view assumes perfect person-level linkage between the two recipients.

## 1. Did the boosted-tree task learner create headroom?

Yes.

**On FIT2, against the equally exposed neural reference N_EQ** (unadjusted 95% percentile intervals, labelled diagnostic in `STAGE_A_REPORT.json`) (same fitting people, same internal fold, validated Recipe-A protocol):
- **Selected teachers:** CatBoost depth 4, chosen by internal-fold NLL for both tasks, with 747/492 income trees and 494/751 occupation trees (seed 0 / seed 1).
- **Income NLL:** 0.027 nats better, 95% interval [0.022, 0.032].
- **Occupation NLL:** 0.014 nats better, [0.009, 0.019]. That meets the prospective 0.010 practical target at the point estimate but not at the lower bound.
- **Occupation Brier:** 0.0045 better, [0.0021, 0.0070].
- **Accuracy:** income +0.016 [0.011, 0.021]; occupation +0.003, unresolved.
- **Prerequisite:** all 12 internal clauses pass (`STAGE_A_REPORT.json`).

**Why U0/Ucal were not the internal reference:** they were trained on all of FIT, FIT2 included, so on FIT2 they look much better (in-sample). They were kept only for the final INNER requirement.

**On INNER, against U0** (seed means, descriptive): the teacher T_TREE has
- income NLL 0.292 vs 0.334 (−0.042);
- occupation NLL 1.254 vs 1.266 (−0.013);
- occupation accuracy 0.495 vs 0.491.

The headroom is real on the common reporting role too.

**The teacher leaks more than the incumbent.** T_TREE coalition SEX AUC is 0.877 / 0.868, against RAW_PROJ 0.843 / 0.828. The stronger task model starts from a worse privacy position.

## 2. Did privacy training spend that headroom?

Largely yes, for the local arm.

**Primary protected service: L_TREE** (local per-recipient critics only, strength ρ=1, epoch 20). It was nominated on FIT2 by the frozen rule: internally point-feasible in both seeds, then lowest worst-seed FIT2 coalition proxy.

On INNER, against its own teacher:
- **Coalition AUC:** falls from 0.872 to 0.785 (seed mean), a reduction of 0.087 [0.063, 0.111].
- **Occupation NLL:** rises from 1.254 to 1.268. That spends about the whole occupation headroom over U0 (ending +0.002 above U0).
- **Income NLL:** rises from 0.292 to 0.303, still 0.031 below U0.

**Utility (24 original clauses vs U0 and Ucal):**
- every point estimate passes;
- 12 pass under simultaneous bounds (all income clauses);
- 12 occupation clauses are unresolved (largest upper bound 0.032, accuracy loss vs U0, seed 0);
- 0 established violations.

**Privacy-driven movement:** in J_TREE at ρ=3, all 8 of the first recorded updates per seed changed the transmitted probe probabilities when the privacy gradient was removed.

**Strength:** every family nominee is at ρ=1. Stronger strengths were point-feasible only at the earliest checkpoint (ρ=3 E2 in J, L, L_STRONG and S12; ρ=10 E2 in J and L), and their later checkpoints were infeasible. At ρ=1 the lower FIT2 proxy came later in training, so the frozen rule picked ρ=1. The diagnostic J_TREE ρ=10 E20 checkpoint reaches coalition AUC 0.659 but fails most clauses (occupation accuracy 0.452).

## 3. Did the outputs become harder for fresh recipients to exploit?

**Yes against the incumbent; not established against the strongest historical codes.**

Primary family: K = 78, Student-t Bonferroni critical value 3.418, 2,048 paired duplicate-group bootstrap draws.

| L_TREE vs (required comparator) | coalition benefit (needs lower > 0.020) | income excess (upper ≤ 0.010) | occupation excess (upper ≤ 0.010) |
|---|---|---|---|
| RAW_PROJ (incumbent; also the RAW-J baseline) | **0.0505 [0.0257, 0.0754] ESTABLISHED** | −0.036 PASS | −0.082 PASS |
| N_EQ (equally exposed neural task model) | **0.0800 [0.0541, 0.1060] ESTABLISHED** | −0.049 PASS | −0.115 PASS |
| T_TREE (its own teacher) | **0.0871 [0.0630, 0.1111] ESTABLISHED** | −0.056 PASS | −0.135 PASS |
| H_JOINT_CODE (JOINT i8o64 λ0.1 score code) | 0.0177 [−0.0064, 0.0419] UNRESOLVED | −0.047 PASS | −0.030 PASS |
| H_SEQ21_CODE (SEQ-21 i8o64 λ0.1 code) | 0.0150 [−0.0098, 0.0399] UNRESOLVED | −0.048 PASS | −0.025 PASS |
| H_DPC_JOINT (decision-preserving JOINT m8 λ0.1) | 0.0138 [−0.0119, 0.0396] UNRESOLVED | −0.053 PASS | −0.010 UNRESOLVED |

- **Per-seed benefit vs RAW_PROJ:** 0.0595 and 0.0416.
- **Per-seed benefit vs H_JOINT_CODE:** 0.0176 and 0.0178.
- **Individual recipients vs RAW_PROJ:** income AUC 0.651 vs 0.687, occupation 0.737 vs 0.819. No recipient shows a positive point excess against any required comparator. 11 of 12 recipient no-excess bounds PASS, and one (occupation vs H_DPC_JOINT, upper 0.0216) is UNRESOLVED.
- **Controls:** shuffled-label readers ≤ 0.521, planted leak 1.000, constant 0.5.
- **Contract effect (descriptive, no interval):** under this narrowed output contract the historical codes H_JOINT_CODE and H_SEQ21_CODE are also INNER point-feasible on all 24 clauses, and their coalition AUC (about 0.803 / 0.800) is already about 0.033 below RAW_PROJ's 0.836. Part of any margin over RAW_PROJ is therefore available to non-tree releases under this contract, and is not specific to the tree recipe.

**Why the competitive criterion fails.** The three historical score codes were point-feasible on FIT2, so they were frozen as required comparators before reporting. L_TREE beats each of them by 0.014–0.018 at the point estimate, but every interval includes values below 0.020 (and below 0).

**Distance to the criterion:** against H_JOINT_CODE the point estimate is 0.0023 short of 0.020, and the lower bound is 0.0264 short.

**Asymmetry, disclosed and not used to change the label:** the historical codes were trained with SEX on all of FIT, and their nominees were historically selected on INNER. Both favour them on INNER.

**Not counted as defeated:**
- **FARE** (stored official FARE c1 with its own heads) reaches coalition AUC 0.684, but it failed FIT2 utility and is ineligible; it is reported descriptively. Its occupation accuracy is 0.462 and its INNER points pass 2–3 of 12.
- **FARE_Z1**, its matched no-fairness compression control, reads 0.861.
- **Confidence-constrained mechanism:** unavailable; it was never fitted.

## 4. Did joint design add anything?

No. The local arm was the FIT2-selected primary; on INNER, J_TREE was not distinguishable from L_TREE on the coalition view and leaked more occupation.

J_TREE was nominated at ρ=1, epoch 5, against its matched controls:

| J_TREE vs | coalition benefit | status |
|---|---|---|
| L_TREE (equally trained local twin) | −0.0155 [−0.0311, 0.0002] | unresolved at the 0 criterion (established shortfall at 0.020). J's occupation recovery is 0.036 higher [0.019, 0.052] (established violation of the 0.010 no-excess bound); its income excess 0.018 [0.004, 0.032] is unresolved. |
| L_STRONG (stronger local control) | +0.0047 [−0.0041, 0.0135] | unresolved |
| S12_TREE | −0.0045 [−0.0166, 0.0076] | unresolved (J's income excess 0.019 [0.005, 0.033] also unresolved) |
| S21_TREE | −0.0369 [−0.0546, −0.0192] | established shortfall (S21 is an infeasible fallback, ineligible) |

The joint (coalition) critic term did not improve on the local or sequential schedules here. The FIT2-selected primary used local penalties alone. S12/S21 are learned sequential controls, not Taylor et al.'s finite-law algorithm.

## 5. Status ledger

- **Established:**
  - the trees create task headroom over N_EQ (FIT2) and over U0 (INNER, descriptive);
  - the L_TREE coalition and recipient reductions vs RAW_PROJ, N_EQ and T_TREE beyond the 0.020 margin;
  - L_TREE is INNER point-feasible on all 24 original clauses;
  - privacy updates change transmitted outputs.
- **Unresolved:**
  - L_TREE vs the three historical score codes (all below-margin point estimates);
  - 12 occupation utility clauses under simultaneous bounds;
  - the occupation no-excess bound vs H_DPC_JOINT.
- **Not established:**
  - the competitive criterion against all required references;
  - any joint-design component.
- **Untested:**
  - other sensitive attributes;
  - adversaries with model access, repeated queries or insider access;
  - imperfect linkage;
  - transfer to another population;
  - reusable-feature-service protection;
  - any unexposed confirmation population (still closed).
- **Incomplete or unavailable:**
  - the confidence-constrained mechanism (never fitted);
  - combined-study FARE (different split and inputs);
  - FARE was not newly fitted on this pool (GUARD SEX is not exported); its stored release was attacked under the same contract.

## 6. Forecasts (registered in SCIENCE_LOCK)

| id | p | claim | outcome |
|---|---|---|---|
| F1 | 0.30 | occupation FIT2 NLL improvement over N_EQ ≥ 0.010 (point) | **true** (0.0139) |
| F2 | 0.75 | prerequisite met | **true** |
| F3 | 0.60 | some protected family point-feasible on FIT2 | **true** (4 of 5) |
| F4 | 0.05 | competitive protection established | false |
| F5 | 0.05 | joint component established | false |
| F6 | 0.40 | primary coalition benefit vs RAW_PROJ > 0 (point) | **true** (0.0505) |

## 7. Decision

Among the studies summarized in `PRIOR_METHOD_DIFFERENCES.md`, this is the first fixed recipe to clear the 0.020 coalition margin against its incumbent while every original point utility clause passes. This needs three qualifications. (a) Predecessors measured RAW_PROJ under full-wire contracts, while here RAW_PROJ is narrowed to probabilities plus decision. (b) Under this narrowed contract the historical score codes are also 24/24 INNER point-feasible and already read about 0.033 below RAW_PROJ (descriptive), so part of the margin reflects the contract. (c) L_TREE's worst occupation upper bound (0.032) is wider than RAW_PROJ's (0.013); earlier studies rejected candidates on simultaneous occupation bounds near 0.012, a stricter criterion than this campaign's point gate. It does not establish a competitive method, because the strongest historical score codes are within the margin. It does not support joint design: local leaf-score penalties on a stronger tree teacher were best.

**Evidence-based next decision:** keep this fixed recipe (CatBoost depth-4 teachers + local leaf-score correction at ρ=1, epoch 20) and the frozen comparator set unchanged. Resolving the 0.014–0.018 gap to the historical codes needs new, unexposed people and a pre-registered head-to-head, not more tuning on these roles. Whether to authorize such a population is the owner's decision. No nearby sweep (strengths, epochs, depths, critics) is scheduled.
