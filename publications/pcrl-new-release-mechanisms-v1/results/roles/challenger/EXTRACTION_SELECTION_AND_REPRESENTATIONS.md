# Failure-map extraction pass: selection and representations (saved verbatim with attribution)

Author: an extraction sub-agent that the challenger spawned for its failure map, titled "Failure map: selection and representations".
It started at about 14:40Z, and its final message reached the coordinator at about 15:00Z on 2026-10-20. Saved verbatim by the
coordinator; nothing below was edited. The challenger's own synthesis is FAILURE_MAP.md. This pass is one of its inputs.

---

FAILURE MAP: 7 assigned campaigns. Paired values a/b are seed (version) 0/1; brackets are registered intervals; sources are public records only. I checked every fork-supplied number used here against the documents.

Data caveat: entries 1–3 ran on repeatedly used Adult people, not ACS FL. Entries 2–3 also release 16-d features, so their AUCs (about 0.85) are not comparable with the ACS entries 4–7 (about 0.61–0.66).

**1. capacity_matched_feature_selection_v1 (Adult; 2026-10-08/09)**
- **Label:** INCOMPLETE_MATCHED_COMPARISON. The FIT gate gave NO_UTILITY_FEASIBLE_T / NO_FEASIBLE_J on both seeds.
- **Operation:** each recipient gets a fixed raw-group mask plus its own calibrated probabilities and decision. A mask keeps 4 numeric and 4 categorical groups. Five masks each delete one categorical group: education (66 cols), marital (76), native country (42), relationship (77), workclass (75). Heads are 64×64 with global temperature calibration.
- **Best:** 0/5 masks feasible. All 20 mask-task-seed rows fail U0 and Ucal, with NLL binding.
  - Best U0 margin: income −0.031630, occupation −0.044394 (Ucal −0.023884 / −0.038326).
  - The full-input F head itself fails U0: NLL excess 0.044753–0.060306, accuracy loss 0.020731–0.030052, Brier 0.023905–0.030091.
  - Deleting native country or marital passes all F-relative caps.
  - No recovery was measured. The run stopped before SEX access, on the FIT validation fold.
- **Failure:** infeasible (no config passes the gates), rooted in a weak full-input baseline.
- **Control lesson:** the F control explains it, because the gap exists before any deletion. Every off-diagonal pair exposes all 83 columns to the linked pair, so a union-matched control is needed.
- **Files:** RESEARCH_DECISION.md ¶2–3; m2/FIT_SCREEN_SUMMARY.md ("Full-input F reference gap").

**2. occupation_bottleneck_v1 (Adult; closed 2026-10-13)**
- **Label:** PREDICTION_LEAKAGE + EXTRA_FEATURE_LEAKAGE; cause MIXED_OR_UNRESOLVED; pilot NOT_TRIGGERED_DIAGNOSIS_UNRESOLVED (zero defense fits).
- **Operation:** post-hoc head-row-space projection with no training. Release is Pz, with P = A⁺A from the recipient's own head (effective dims: 1 income, 5 occupation), plus centred logits, probabilities and decision. Applied to U0, RAW (adversarial, .3, epoch 5) and JOINT (.3, epoch 5).
- **Best:** INNER, seed means, base9+extended9 readers. SEX AUC, income / occupation / pair:

| Release | Income | Occupation | Pair |
|---|---|---|---|
| U0 | 0.8212 | 0.8581 | 0.8701 |
| RAW | 0.7519 | 0.8381 | 0.8568 |
| RAW projected | 0.6895 | 0.8283 | 0.8461 |
| JOINT projected | 0.6990 | 0.8309 | 0.8482 |

  - Utility: 24/24 at the point; K864 5 PASS / 19 UNRESOLVED.
  - Pair benefit of RAW projected vs RAW: 0.010686 [0.000388, 0.020984]. JOINT projected: 0.008585 [−0.001563, 0.018732].
  - Full JOINT income view is 0.063252 [0.045004, 0.081501] worse than RAW, a VIOLATION.
  - RAW outputs-only readers: pair 0.841485. Decisions alone: 0.723088.
- **Failure:** benefit below 0.020, and the mechanism gate was not met.
  - Gradient trigger: 2/4 batches adverse, against a 3/4 rule.
  - Interface trigger: ATT pair reductions 0.005787/0.017258, against ≥0.015 required in both seeds.
- **Control lesson:** outputs-only readers show predictions carry most of the pair recovery. Against full RAW, the projection is just ordinary compression.
- **Files:** START_HERE.md "Important releases"; BOTTLENECK_DECISION.md; RESEARCH_DECISION.md l.45.

**3. task_conditioned_representations_v1 (Adult; 2026-10-12)**
- **Label:** COMPLETE_EXPLORATORY_NEGATIVE_COMPARISON; competitive method SHORTFALL; utility POINT_PASS_PRECISION_UNRESOLVED; conditioning/coupling/joint NOT_ESTABLISHED.
- **Operation:** two 83→64→64→16 encoders plus logistic heads, trained jointly with an MMD-B-Fair conditional kernel penalty. It matches SEX within task labels and within aligned pair labels. Release is 16-d z, centred logits, probabilities and decision.
  - Strengths .3/1/3 × epochs 0/5/10/20; 46 two-seed configs; nominee .3, epoch 5.
- **Best:** INNER, nine readers per view, trained on ATT.
  - JOINT pair 0.861891 vs U0 0.870143: benefit 0.008252 [0.000111, 0.016393].
  - vs LOCAL_TWIN: 0.004826 [−0.002662, 0.012315].
  - vs unconditional: −0.000635.
  - vs SEQ12/SEQ21: 0.001114/0.005394.
  - vs RAW_JOINT (ordinary adversary, 0.856824): −0.005067, with income worsening 0.063252 [0.043198, 0.083306], a VIOLATION.
  - Utility 24/24 at the point; K51 13 PASS / 11 UNRESOLVED.
  - H_RAW_CSTAR pair is 0.781964, but it fails 8/24 clauses.
- **Failure:** benefit below 0.020, and recovery not reduced vs the matched adversarial control.
- **Control lesson:** RAW_JOINT, the local twin, the unconditional control and both sequential controls each explain away a component.
- **Files:** RESEARCH_DECISION.md "Separate verdicts"; COMPLETE_VIEW_RECOVERY.md l.15–27, 55–65.

**4. autonomous_method_development_v1 (ACS; 2026-10-17)**
- **Label:** B3CTL_UNCOUPLED_MIX50_C6A EDGE_SELECTED. Claims: E DEV_POINT_ELIGIBLE; B SUPERIOR_TO_ALL_ELIGIBLE (0.020 not established); M MECHANISM_ATTRIBUTION_ESTABLISHED; J JOINT_ADVANTAGE_NOT_ESTABLISHED.
- **Operation:** occupation is the stored JOINT_TREE output (bitwise H_LJ's). Income is a 6-level code of 0.5·LOCAL_TREE + 0.5·JOINT_TREE log-odds, with level probabilities fitted on CALIBRATION. α=0.5 was the largest value within the income caps.
- **Other arms (MODEL_SELECT pair proxy):**
  - B1, income computed from occupation: 0.5666, but income NLL +0.053.
  - B2, LOCAL-only code: 0.6131/0.6132.
  - Coupled B3: failed s0 income Brier.
  - C, SEX-mixture: 0.6263/0.6368, and the s1 men's guard fails.
- **Best:** DEV_REPORT, 8 readers. Linked pair 0.613/0.618.
  - vs NEURAL_RAW 0.640/0.638: +0.027/+0.0199 (bounds 0.0135/0.0059).
  - vs H_LJ 0.641/0.643: +0.028/+0.025 (bounds 0.0191/0.0129).
  - vs task-only control: +0.052/+0.047.
  - vs LOCAL-only twin: +0.016/+0.010 (s1 bound −0.0025).
  - At the point, 24/24 clauses, 4/4 guards and 8/8 recipient conditions pass, but at the edge: income Brier vs Ucal 0.0049/0.0049, s1 accuracy loss 0.0096, men's guard 0.0032/0.0040.
  - Confidence: 25/28 UNRESOLVED.
- **Failure:** benefit below 0.020, bought with about the whole income Brier budget.
- **Control lesson:** the task-only control differs in both outputs. No continuous-mixture control was run, so coarsening and mixing are not separated.
- **Files:** START_HERE.md; RESEARCH_DECISION.md l.114, §8 l.239–247.

**5. frozen_finalist_validation_v1 (ACS; 2026-10-17)**
- **Label:** "retire as a competitive candidate". B READER_SENSITIVE (side COMPARATOR_READ_LOWER); J and 0.020 NOT_ESTABLISHED_EITHER_BANK.
- **Operation:** the release was frozen. Only the readers changed: the source's 8 plus 8 new recipes (ExtraTrees, HGB, MLPs, categorical-interaction), plus household-bootstrap refits.
- **Best:**
  - 70/70 endpoints reproduced.
  - DEV_REPORT re-read, ENHANCED bank: +0.027/+0.022 vs NEURAL_RAW, +0.028/+0.026 vs H_LJ.
  - Claim B fails on the s1 income recipient clause vs H_LJ: +0.0076, bound 0.0235 > 0.010. H_LJ's income reader was picked on a near tie (0.5523 vs 0.5517) and read 0.011 lower on DEV.
  - Bootstrap-trained readers shrank the benefit in 3/4 cells.
  - 80% power needs about 4.6M households (plug-in) or about 60M (confirmatory). The recipient clause alone needs about 318,000. About 17,500 exist, giving power about 0.01 (ENHANCED) or 0.25 (ORIGINAL).
- **Failure:** infeasible (no realistic confirmation) and reader-sensitive.
- **Control lesson:** the enhanced readers exposed instability in how the comparator's reader is selected. Reader-refit SD of 0.003–0.011 is not covered by the intervals.
- **Files:** START_HERE.md results table; RESEARCH_DECISION.md l.28–58, 104–131; CONFIRMATION_FEASIBILITY.md l.54–74.

**6. income_headroom_successor_v1 (ACS; 2026-10-18)**
- **Label:** INCOME_BRIER_NLL_HEADROOM_KEPT_NOT_COMPETITIVE. Claims: P NOT_SUPERIOR_TO_ALL_ELIGIBLE [READER_SENSITIVE]; R RESERVE_ERODED; T TARGET_NOT_ESTABLISHED; O INHERITED_CONSTRAINTS_BIND.
- **Operation (income only):** HGB stacked with T_TREE ("STRONG"), plus a budgeted transplant d = a·(s_STRONG + D_LOCAL + β·D_JOINT) + b, then a Platt map.
  - β values: 0 (H1), 0.85 (H2), 0.25/0.20 (H3, the nominee).
  - Occupation is the frozen JOINT_TREE output.
  - Families P (pair-critic income corrections) and O (SEX-balanced occupation recalibration) also ran; 34/36 configs used.
- **Best:** DEV_REPORT, ENHANCED readers.

| Release | Pair AUC |
|---|---|
| Nominee H3 | 0.6272/0.6263 |
| H_LJ | 0.6414/0.6428 |
| NEURAL_RAW | 0.6400/0.6386 |
| Twin HC_STRONG | 0.6557/0.6578 |
| Component control | 0.6293/0.6236 |
| RETIRED_B3CTL | 0.6130/0.6168 |

  - Benefit vs H_LJ: +0.0141 (bound +0.0092) / +0.0166 (+0.0057).
  - Benefit vs NEURAL_RAW: +0.0128/+0.0123 (s1 bound −0.0015).
  - Income Brier vs Ucal −0.00164/−0.00124.
  - s1 accuracy loss +0.00707: 0.83 SE of margin against 1.7267 SE required.
  - s1 income recipient vs H_LJ: +0.0061, bound 0.0217.
  - 3/4 bootstrap refits reverse the s1 benefit.
  - Confirmation needs about 129,000–292,000 households, against about 17,510 available.
- **Failure:** reader-sensitive, and the benefit of 0.012–0.017 is below 0.020.
- **Control lesson:** the component control (the same transplant on T_TREE) gives C NOT_ESTABLISHED (+0.0020/−0.0026). The twin gain of +0.0285/+0.0316 counts only as task-only attribution. Missing: a matched temperature control, and T_TREE was absent from the DEV bank.
- **Files:** RESEARCH_DECISION.md l.81–105, 145; MECHANISM_CARDS.md l.13–30.

**7. occupation_headroom_overnight_v1 (ACS; 2026-10-18)**
- **Label:** NO_QUALIFYING_CONFIGURATION. All 18 admissible configs stop at RECIPIENT_POINT, and all are BENEFIT_NOT_STABLE. DEV_REPORT was never opened.
- **Operation:** occupation HGB with POBP coded as an ordinal value instead of a 106-level category (A1_PO_16, B2_SINGLE).
  - Protection is a leaf-score correction on the frozen T_TREE routing, trained against SEX critics on the pair (PAIR) or on occupation alone (LOC).
  - Budgets: B50/B80 spend 50/80% of the reserve slack; M90 spends 90% of the point margin.
  - Variants P1/P2/P3; income is frozen H3.
- **Best:** MODEL_SELECT 8-cell grid.
  - The witnesses beat T_TREE NLL by 0.0026–0.0046. B2_SINGLE men's guard −0.0041/−0.0025.
  - P2A_PAIR_B80 pair 0.5813/0.6118:
    - vs continuation twin +0.040 STABLE;
    - vs K-means code COMP_B80 +0.026;
    - vs SUCC +0.004 (+0.018/−0.010);
    - vs RETIRED_B3CTL −0.003;
    - s0 men's reserve 0.85; the 0.020 target is met at the point only.
  - Under fixed readers it beats SUCC 18/18 on s0 but 3/18 on s1. A third split gives −0.0025.
  - The reserve-keeping P2_LOC_B80 (2.20) is −0.010 vs SUCC.
  - H3's own income view is +0.0309 over H_LJ (limit 0.010).
- **Failure:** infeasible (no config passes the gates). Underneath is a utility cost: reaching SUCC spends the s0 men's reserve (ratios 0.57–1.39).
- **Control lesson:** the matched twin and the cost-matched code were beaten. The inherited JOINT_TREE releases (SUCC, B3CTL) were not. The comparators themselves fail the reserve (SUCC 0.321/0.624).
- **Files:** ADVISOR_BRIEF.md; RESEARCH_DECISION.md l.103–110, 195–230, 305–318.

**Proposed but not run, or untried**
- **1:** SEX critics, J/T/S12/S21/SHARED and union-matched selection, attacks and INNER were never run. The record proposes a full-input F feasibility study first.
- **2:** occupation-only safeguarded RAW refinement (actual-displacement QP, income frozen), compared against RAW, continuation, local twin and task-only.
- **3:** CFair NOT_REGISTERED_NOT_RUN. The record says not to launch another strength/kernel/critic sweep.
- **4:** continuous-mixture control; coupled B3 on s1; JOINT-only code α=1 (over the cap on the screen).
- **5:** the sealed 20% (9,853 households) and 7,657 capped-out households were not opened.
- **6:** the best P+O combination, CMB_P3JR_O1SEXBAL, has no DEV evidence.
- **7:** P3 on s1 and the P3 compression twin; fixed-recipe checks at B50/M90. Proposed: held-out budget calibration, a per-SEX cost balance between P2 and P3, and a per-cell recipient gate on two splits.

**Flags**
- In entry 2, identifying "RAW .3 epoch 5" as RAW_JOINT_R03_E5 is inferred from the roster.
- Entry 2's s1 ATT reduction is 0.017257 in BOTTLENECK_DECISION.md but 0.017258 in ADVISOR_BRIEF.md.
