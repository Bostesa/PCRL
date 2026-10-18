# Research decision: overnight occupation-headroom and robust-release campaign (PCRL, 2026-10-18)

**Question.** Can a stronger occupation predictor create enough task-quality margin to support a useful reduction in demographic
recovery from both released outputs, while preserving the income improvement from H3?

**Decision.** Partly, and not as a whole release.
- A stronger occupation learner creates occupation headroom on both seeds. Both SEX-group guards are met with the development reserve,
  and the guard that bound every earlier task-only release, the men's, improves.
- A coalition-aware occupation correction spends part of that headroom to lower linked-pair recovery by the declared development
  readers. At the B80 level it beats matched continuation and compression controls in every grid cell.
- No configuration lowers recovery below the strongest inherited releases on both seeds, and none passes the frozen nomination gates.
  Every grid candidate fails the per-cell recipient gate, none has a stable benefit against all six required comparators, and six
  also miss the reserve on one seed.
- The frozen nomination rule therefore found **no qualifying configuration**. DEV_REPORT stayed closed (only its label-free raw inputs
  were used, for deployment checks), no ATTACK label was read, and no final evaluation ran.
- The campaign delivers development-stage components (section 6), the frontier and the measured reasons promotion failed, and a
  negative answer to the whole-release question.

**What the evidence is.**
- It is development evidence on previously used Florida ACS 2019 records. MODEL_SELECT (8,001 people, 5,306 households) was used
  adaptively for selection and diagnosis.
- The "seeds" are not replications, and for the witnesses they are not independent fits either.
  - Family A: the seed-1 model is bitwise the seed-0 boosting path truncated at the seed-1 tree count. A1_PO_16 uses 154 of 168
    trees; MODEL_SELECT decisions agree on 98.9% of people (label-free check, `checks/A_SEED_IDENTITY.json`).
  - Family B: the deployed base learners are bitwise identical on both seeds. B2_SINGLE, B2_STACK and B3_CALTR differ by seed only in
    layer weights and temperature (family B card, "Seed caveat").
  - So seed-to-seed agreement for the witnesses mostly re-evaluates nearly the same model against seed-specific references.
  - The protection corrections and their projection scales are separate fits per seed, on these near-identical witnesses.
- The 8 reader-grid cells contain only two household splits, shared by both seeds.
- Nothing here is a confirmation or a population guarantee.

## 1. Did any new occupation learner create useful headroom, including both subgroup guards?

Yes, in the development sense. The frozen witness rule requires:
- both seeds point-feasible on all 28 conditions;
- all 16 occupation-side conditions, both guards included, at least 2 SE inside their limits.

It found 10 promising task-only learners and selected two by the occupation-side leximin ranking:

| Occupation learner (with H3 income) | Occupation NLL vs U0 (s0 / s1) | vs teacher T_TREE (SE) | Men's guard (limit +0.005) | Women's guard | Occupation-side min ratio (s0 / s1) |
|---|---|---|---|---|---|
| B/B2_SINGLE (witness 1) | −0.0147 / −0.0114 | −0.0026 (0.0017) / −0.0031 (0.0017) | −0.0041 / −0.0025 | −0.0091 / −0.0082 | 5.000 / 4.418 |
| A/A1_PO_16 (witness 2) | −0.0161 / −0.0128 | −0.0040 (0.0017) / −0.0046 (0.0017) | −0.0039 / −0.0023 | −0.0097 / −0.0087 | 4.881 / 4.299 |
| B/B3_CALTR (best NLL) | −0.0174 / −0.0140 | −0.0053 (0.0014) / −0.0058 (0.0014) | −0.0025 / −0.0012 | −0.0122 / −0.0111 | 4.153 / 3.634 |
| teacher T_TREE with H3 (reference) | −0.0121 / −0.0082 | 0 | −0.0010 / +0.0006 | −0.0093 / −0.0079 | 3.33 / 2.60 |
| SUCC (inherited JOINT_TREE with H3) | +0.0025 / +0.0029 | +0.0147 / +0.0111 | +0.0044 / +0.0039 | −0.0049 / −0.0030 | 0.32 / 0.62 |

**The men's guard.** Every witness improves it; it is the condition that binds every task-only reference.
- For B2_SINGLE it moves from the teacher's −0.0010 / +0.0006 to −0.0041 / −0.0025, i.e. 0.0091 / 0.0075 below the +0.005 limit.
- The paired cost decompositions put the gain on men on both seeds (B2_SINGLE vs teacher, NLL: men −0.0045 / −0.0048, women −0.0006 /
  −0.0014).

**The women's guard** stays at the teacher's level for B2_SINGLE, improves slightly for A1_PO_16, and more for B3_CALTR.

**Size of the gain.**
- Over the equally exposed task-only refit TASK_REFIT the gain is small: B2_SINGLE −0.0009 / −0.0010 (about 0.5-0.6 SE); A1_PO_16
  −0.0024 / −0.0025 (about 1.4-1.5 SE).
- Witness 1 (B2_SINGLE) is family B's registered control, the single learner. Family B's own mechanism, the stack B2_STACK, ranked
  third because stacking weakened the men's guard (−0.0035 / −0.0021).
- With the witnesses beside H3, the occupation side no longer binds: the whole-release minimum becomes H3's income accuracy
  (section 5).

**Headroom is not protection.** The witnesses' linked-pair recovery (0.636 to 0.643 SCORE AUC) is NOT_POSITIVE against all six
required comparators. Their occupation views exceed the JOINT_TREE references by up to +0.080 per cell (up to +0.082 for A1_PO_8).

## 2. Which model operation produced that improvement?

The improvement comes from coding place of birth (POBP, 106 levels) as its ordinal census code instead of as a native 106-level
category in histogram gradient boosting.
- **With native categories** the boosted learner is worse than the teacher. The seed-0 NLL difference vs T_TREE is +0.0086 for
  A0_HGB_S and +0.0122 for A0_HGB_L; no seed-1 fit was made.
- **With the code**, A1_PO_16 beats the teacher on NLL, Brier and accuracy on both seeds. It is scikit-learn HistGradientBoosting
  (16 leaves, min leaf 100, learning rate 0.025, l2 3, tree count from seed-specific household folds) on AGEP and WKHP as z-scores,
  COW and MAR as native categories, and SCHL ordinal.
- **Why it helps.** The code orders US states, territories, Europe, Asia, the Americas and Africa in blocks, so trees can only split
  contiguous ranges. This pools related places of birth and reduces, though does not remove, a variance problem: family A's
  committee-minus-refit cross-fitted NLL gain falls from −0.0013 to −0.0031 for the native-category learners to about −0.0005 for
  A1_PO_16.
- **Family B** uses the same coding in its repaired tree (TREE_V2), and its temperature-scaled single learner is witness 1. Stacking
  TREE_V2 with an additive logistic basis improves NLL out of fold by −0.0025 / −0.0024 (B2_STACK minus B2_SINGLE; −0.0020 on
  MODEL_SELECT).
- **Family C** (residual learning on the teacher) did not beat TASK_REFIT: best NLL difference +0.0001 / +0.0003. The teacher's
  in-sample optimism on MODEL_FIT (+0.034 NLL) made in-sample residual selection blind.
- **Income learner.** The same repair applied to H3's income learner improved it in household cross-fitting (Brier −0.0029 / −0.0028,
  NLL −0.0042 / −0.0041) but did not change seed-1 income accuracy.

## 3. Did a protection step reduce linked-pair recovery beyond strong controls at useful task quality?

Beyond the matched controls, yes, at the B80 level. Beyond the strongest inherited references, no.

**What the protection step is.** A coalition-aware occupation correction:
- a leaf-score correction on the frozen T_TREE routing;
- trained on MODEL_FIT against SEX critics that read the linked pair (H3's frozen income vector with the occupation vector) and the
  occupation vector (PAIR), or the occupation vector alone (LOC);
- scaled down until its out-of-fold MODEL_FIT costs fit a budget. B50 / B80 spend 50% / 80% of the witness's MODEL_SELECT reserve
  slack (a logged design input). M90 spends 90% of its point margin, a frontier point.

Only occupation is optimized, so this is not a two-output joint design.

**Controls.** The continuation twin (CONT) runs the same pipeline with critic weight zero; its fits still read MODEL_FIT SEX for the
critics and the projection. The compression code (COMP_B80) is a label-free K-means code at the B80 arm's out-of-fold cost.

| Candidate | vs continuation twin | vs compression code COMP_B80 | vs SUCC | vs RETIRED_B3CTL | Reserve (both seeds) |
|---|---|---|---|---|---|
| P2_PAIR_B80 (B2_SINGLE) | +0.032 (+0.038 / +0.026), 8 of 8 cells | +0.018 (+0.027 / +0.009), 8 of 8 cells | −0.004 (+0.002 / −0.010) | −0.011 (−0.001 / −0.021) | seed 1 only (seed-0 men's ratio 1.39) |
| P2A_PAIR_B80 (A1_PO_16) | +0.040 STABLE (+0.058 / +0.021) | +0.026 (+0.043 / +0.010), 8 of 8 cells | +0.004 (+0.018 / −0.010) | −0.003 (+0.015 / −0.020) | seed 1 only (seed-0 men's ratio 0.85) |
| P2_PAIR_M90 (B2_SINGLE) | +0.039 STABLE (+0.043 / +0.036) | +0.025 (+0.033 / +0.018); a cost-matched M90 code exists on seed 0 only | +0.004 (+0.008 / −0.0005) | −0.003 (+0.004 / −0.011) | seed 1 only (seed-0 men's ratio 0.57) |
| P2_LOC_B80 (B2_SINGLE, local critic) | +0.026 (+0.021 / +0.031), 8 of 8 cells | +0.012 (+0.010 / +0.013), 8 of 8 cells | −0.010 (−0.015 / −0.006) | −0.017 | MET (2.20) |
| P2_PAIR_B50 (B2_SINGLE) | +0.020, sign changes (+0.027 / +0.013) | +0.007, sign changes (+0.017 / −0.004) | −0.015 (−0.008 / −0.023) | −0.022 | MET (2.62) |
| P2A_PAIR_B50 (A1_PO_16) | +0.025 (+0.035 / +0.014) | +0.011, sign changes (+0.020 / +0.002) | −0.011 (−0.005 / −0.017) | −0.018 | MET (2.15) |

Values are 8-cell means of comparator AUC minus candidate AUC, with seed means in parentheses.

**What the controls show.**
- The continuation twin barely moves recovery: 8-cell means +0.004 over B2_SINGLE and +0.001 over A1_PO_16. The critic, not extra
  training, does the work.
- The compression code lowers recovery by about 0.015-0.018 at a large cost and fails the seed-1 reserve (1.92 / 1.80).
- At B80 the pair critic reads lower than the local critic in all 8 cells on A1_PO_16 and in 6 of 8 on B2_SINGLE. That budget is the
  same only nominally: PAIR_B80 breaks the seed-0 reserve, while P2_LOC_B80 keeps it on both seeds.

**The reserve-keeping levels stay above SUCC.**
- The reserve-keeping PAIR arms (B50) stay 0.011-0.015 above SUCC, and their margin over compression changes sign.
- The closest reserve-keeping configuration is the local-critic P2_LOC_B80. It is 0.010 above SUCC (seed means 0.015 / 0.006) and
  ahead of its continuation twin and compression code in every cell.
- The levels that reach SUCC (PAIR B80, M90) spend the binding men's margin on seed 0.

**The two allowed revisions did not remove this trade-off.**
- Removing the in-training guard Lagrangian (P2) strengthened the critic.
- Static per-SEX weights on the task loss (P3; seed 0 only) protected the men's guard. Over the witness, men's cost was −0.0024 instead
  of +0.0064 on B2_SINGLE and −0.0025 instead of +0.0071 on A1_PO_16. But the binding constraint moved to women and protection fell:
  pair AUC 0.616 (P3, B2_SINGLE) and 0.618 (P3A, A1_PO_16) against 0.597 and 0.581 for P2. That loses about 40-49% of P2_PAIR_B80's
  margin on B2_SINGLE and about 67% of P2A_PAIR_B80's on A1_PO_16.
- The lowest recorded pair AUC, 0.5766 (P2A_PAIR_M90, seed 0 only), comes with men's ratio 0.38.

**Frontier (SELECTION_RULE §6, NOMINATION_FRONTIER.csv).** The score is the minimum over the six required comparators of the 8-cell
mean benefit. Every score is bound by RETIRED_B3CTL.

| Configuration | Score | Min ratio |
|---|---|---|
| P2A_PAIR_B80 | −0.0027 | 0.85 |
| P2_PAIR_B80 | −0.0109 | 1.39 |
| P2_LOC_B80 | −0.0170 | 2.20 |
| P2_PAIR_B50 | −0.0223 | 2.62 |
| B2_STACK | −0.0401 | 2.862 |
| INC_TW_PO | −0.0479 | 3.144 |

All other configurations are dominated.

## 4. Did that benefit survive repeated reader fitting?

At B80 the benefit over each arm's own controls did, for the declared readers. The benefit over the inherited references did not.

**Over its controls, at B80.**
- P2A_PAIR_B80 beats P2A_CONT under 18 of 18 fixed reader recipes on each seed (medians +0.048 / +0.033).
- P2_PAIR_B80 beats P2_CONT under 18 of 18 on each seed (medians +0.038 / +0.028).
- Against the B80 compression codes it is positive under 17-18 of 18 recipes per seed. One probe reader (long gradient boosting) gives
  sign changes.
- A third household split (R3; previously used people, not fresh data) agrees for P2_PAIR_B80 on seed 0 (+0.047 vs P2_CONT).
- Learning curves at 1/4, 1/2 and full reader data are flat under fixed recipes (median change −0.003 to +0.005). The frozen-winner
  "shrinks" flags come from VALIDATION-winner selection noise and do not survive fixed recipes.
- No fixed-recipe check was run for the B50 or M90 levels.

**Over the references, reader- and seed-dependent.**
- P2A_PAIR_B80 vs SUCC is positive under 18 of 18 fixed recipes on seed 0 (median +0.013) and 3 of 18 on seed 1 (median −0.005).
- On R3 the frozen winner reverses it (−0.0025), while 14 of 15 fixed recipes stay positive (median +0.008).
- P2_PAIR_B80 vs SUCC is positive under 9 of 18 recipes on seed 0 (median −0.0001) and 3 of 18 on seed 1 (median −0.012).

**The seed difference is a release effect, not a reader effect.**
- Every PAIR arm reads 0.010-0.031 higher on seed 1, while controls, witnesses and references move by at most 0.007.
- The out-of-fold projection chose smaller seed-1 scales (B80 0.72 vs 0.86 / 0.92), for two reasons:
  - The witnesses' seed-1 men's reserve slack is smaller (0.0041 vs 0.0054 on B2_SINGLE).
  - The projection's transfer error on the men's guard changes sign across seeds: +0.0016 to +0.0029 summed Brier above the
    projection on seed 0, −0.0008 to −0.0012 below it on seed 1. On seed 0 that is 30-54% of the slack, and more than the roughly
    0.0011 a B80 level leaves unspent.
- On seed 0 the correction spends mostly men's task quality (P2_PAIR_B80 NLL: men +0.0193, women +0.0049). On seed 1 it is balanced and
  half as large (+0.0060 / +0.0068).

Lower recovery for these declared readers is not protection against every possible reader.

## 5. Did the complete income-plus-occupation release retain margin, or does income accuracy still bind?

Income accuracy still binds.
- **H3 with a witness.** The complete release meets the 2-SE reserve on both seeds. Its minimum is H3's seed-1 income accuracy:
  accuracy loss +0.0018 against the 0.010 limit, SE 0.0029, ratio 2.862 (seed 0: 3.511). That leaves 0.0025 above the 2-SE
  requirement. The Bonferroni check leaves 2 of 28 conditions unresolved, the seed-1 income accuracy clauses against U0 and Ucal.
- **The income branch.** It was triggered by the source anchor's published DEV seed-1 income ratio of 0.830 (DEV-guided, disclosed).
  It applied the POBP repair to H3's income learner and kept H3's transplant step.
  - INC_H3_PO stays at 2.840.
  - The task-only twin INC_TW_PO reaches 3.144. That is below the unrepaired task-only HC_STRONG's 3.296; its gain over 2.862 comes
    from dropping H3's transplant step, not from the repair.
  - INC_TW_PO's income view fails the recipient gate against all six required comparators and the matched component.
  - Neither gives a pair benefit.
- **The protection levels.** Those that approach SUCC also push the occupation side below the reserve on seed 0 (men's ratios
  0.57-1.39 against 2.0).

## 6. Is there a whole competitive nominee, an incremental component or a negative result?

There is no whole competitive nominee. The formal nomination (NOMINATION.json, 07:45:00Z, frozen rule sha e18f1c68, replayed 34 / 34
by the independent verifier) returned NO_QUALIFYING_CONFIGURATION over 34 configurations: 18 admissible two-seed configurations, 5
NOT_ADMISSIBLE (no grid entries, no package) and 11 with one seed only. The gates are checked in order and the first failure is the
label; all 18 admissible configurations stop at RECIPIENT_POINT, and each would fail later gates too.
- **Recipient point.**
  - Every H3-anchored release fails through its income view. In the seed-0 cells S0R1k0 / S0R1k1 (one household split read by two
    reader seeds), it reads +0.0309 above H_LJ and +0.0147 above TWJ against a 0.010 limit; seed 1 passes (maximum +0.0087). This was
    recorded before any candidate grid.
  - The failure is consistent with split noise: under a normal-noise model, a true zero difference fails against one comparator with
    probability about 0.81, and about 0.99 against five other income wires.
  - Every candidate's occupation view also exceeds the JOINT_TREE comparators, by 0.019 to 0.082 per cell.
- **Stability.** All 18 are BENEFIT_NOT_STABLE. The best score is −0.0027 (P2A_PAIR_B80), bound by RETIRED_B3CTL.
- **Reserve.** Six miss it on one seed: P2_PAIR_B80 1.39, P2_PAIR_M90 0.57, P2A_PAIR_B80 0.85, P2_COMP_B80 1.92, P2A_COMP_B80 1.80,
  P2A_LOC_B80 1.41.
- **Without the recipient gate the outcome would be unchanged.** The six required comparators themselves fail this campaign's reserve:
  SUCC, H_LJ, B2TWIN and TWJ at 0.321 / 0.624, RETIRED_B3CTL at 0.321 / 0.349, NEURAL_RAW at 1.947 / 1.558. The reserve constrains
  candidates only.

**Development-stage labels (SELECTION_RULE §7).**

| Label | Status |
|---|---|
| (1) Useful task-only component | holds for the selected witnesses B2_SINGLE and A1_PO_16 (SELECTED_WITNESS) |
| (2) Incremental reader-relative protection against the matched component | STABLE_POSITIVE only for P2A_PAIR_B80 vs A1_PO_16 (+0.041) and P2_PAIR_M90 vs B2_SINGLE (+0.044); both break the seed-0 reserve. The reserve-keeping levels are POSITIVE_NOT_STABLE (+0.025 to +0.030) |
| (3) Positive superiority over all six | none |
| (4) The 0.020 target | DEV_GRID_TARGET_POINT_ONLY for P2A_PAIR_B80 (both target references). P2_PAIR_M90 reaches point-only against H_LJ alone, so its overall label is NOT_MET. Never stable |
| (5) Joint design | NOT_CLAIMABLE |
| (6) Novelty | NOT_CLAIMED |

The final claims registered for a finalist (W, E, R, P, N, C, M, T, O in CLAIMS_DRAFT.md) are **NOT_EVALUATED**, because there was
no finalist.

**What can be kept.** Two development-stage components:
1. the repaired occupation learner (POBP ordinal coding), packaged as B2_SINGLE and A1_PO_16 beside H3;
2. the coalition-aware occupation correction with its matched controls and frontier.

The whole-release question has a negative answer in this campaign.

### Strongest favourable and adverse comparisons

**Favourable: P2A_PAIR_B80 on seed 0.**
- Pair AUC 0.5813 is below all six required comparators in every seed-0 cell (smallest cell margins: RETIRED_B3CTL +0.003, SUCC
  +0.016).
- It has better occupation NLL (+0.00002 vs SUCC's +0.0025), Brier and men's guard (+0.0033 vs +0.0044) than SUCC. Its accuracy
  (−0.0013 vs −0.0020) and women's guard (−0.0048 vs −0.0049) are slightly worse.
- Its seed-0 means against the target references are +0.040 (H_LJ) and +0.038 (NEURAL_RAW).
- Its seed-0 occupation-view maximum (+0.007) passes the recipient limit.

**Adverse: the same configuration on seed 1, and on R3.**
- On seed 1 its pair AUC is 0.6118, behind SUCC (−0.010; 3 of 18 recipes) and RETIRED_B3CTL (−0.020). It is only +0.003 / +0.0025
  ahead of H_LJ / NEURAL_RAW.
- On the third household split the frozen winner puts it behind SUCC (−0.0025).
- Its seed-0 men's guard misses the reserve (ratio 0.85 against 2.0).
- Recipient maxima: income +0.031 on seed 0 (seed 1 +0.009); occupation +0.007 on seed 0 and +0.033 on seed 1.
- Its nomination score is −0.0027. Eight of its 28 utility conditions are unresolved at the Bonferroni level.
- Over the 8 cells its means against the target references (+0.0217 / +0.0201) exceed 0.020 only as a point result.

The 0.020 venue target is reported separately and changes no criterion.

### Distance to each governing requirement (closest candidates)

Utility, guard and reserve rows are per seed (seed 0 / seed 1). Recipient rows give the per-seed maximum over the required
comparators. Score and target rows are 8-cell means.

| Requirement (limit) | B2_SINGLE + H3 | P2_LOC_B80 | P2_PAIR_B50 | P2A_PAIR_B80 | INC_H3_PO |
|---|---|---|---|---|---|
| 24 utility clauses (acc ≤ 0.010, NLL ≤ 0.010, Brier ≤ 0.005) | all pass; binding income accuracy +0.0001 / +0.0018 | all pass | all pass | all pass; occ NLL +0.00002 / −0.0050 | all pass; income accuracy +0.0000 / +0.0018 |
| Men's guard (≤ +0.005) | −0.0041 / −0.0025 | +0.0007 / +0.0001 | +0.0002 / −0.0012 | +0.0033 / −0.0001 | B2_SINGLE's |
| Women's guard (≤ +0.005) | −0.0091 / −0.0082 | −0.0052 / −0.0062 | −0.0083 / −0.0070 | −0.0048 / −0.0054 | B2_SINGLE's |
| 2-SE reserve (min ratio ≥ 2) | 3.511 / 2.862, MET | 2.20 / 2.76, MET | 2.62 / 2.86, MET | 0.85 / 2.82, seed 0 not met | 3.556 / 2.840, MET |
| Recipient, income view (each cell ≤ +0.010) | +0.031 / +0.009 | +0.031 / +0.009 | +0.031 / +0.009 | +0.031 / +0.009 | +0.027 / +0.055 |
| Recipient, occupation view (each cell ≤ +0.010) | +0.080 / +0.078 | +0.025 / +0.029 | +0.038 / +0.051 | +0.007 / +0.033 | +0.080 / +0.078 |
| Pair benefit stable vs all six (score > 0) | −0.047 | −0.017 | −0.022 | −0.003 | −0.048 |
| 0.020 target vs H_LJ / NEURAL_RAW (separate) | −0.023 / −0.024 | +0.007 / +0.006 | +0.002 / +0.001 | +0.022 / +0.020 (point only) | not met |

## 7. What exactly ran, what did not run, and why

**Ran** (start 04:36:11Z; the research stage closed after the protection stage and the nomination, about 07:45Z):
- Admission of 201 source copies. 61 have no expected hash, and 59 of those are not pinned by any committed source record; the
  verifier compared every copy bitwise with the source's private files.
- The registration commit (b1fd591d0) and its public receipt on GitHub (435fe44e5, verified 05:11:09Z), before any label read.
- Families A, B and C: 28 seed-specific task-only configurations (OCC36). 10 were promising; 2 were selected by the frozen witness
  rule (06:27Z).
- The conditional income branch: 4 configurations (INC8).
- The protection stage: P1, revisions P2 and P3, and a seed-1 LOC arm. 23 privacy configurations (PRIV24) and 10 matched controls
  (CTRL12).
- In total 65 of 80 seed-specific configurations. 57 of them make up the 34 configurations in the nomination table (18 × 2 + 5 × 2 +
  11). The other 8 are one-seed task-only configurations that were not listed in the nomination call; the rule would have labelled
  them NOT_CONSIDERED_ONE_SEED_ONLY. They are A0_HGB_S, A0_HGB_L, C1_RB_D1, C1_RB_D2, C2_XF_D2, CC_CONT_D4, CC_XF_CONT_D4 and
  CC_VSCAL.
- The development reader grid: 15 banks.
- Reader-robustness diagnostics: probe readers, a third household split and fixed-recipe learning curves.
- The formal nomination and its frontier.
- Fallback cost decompositions on both seeds.
- The closest-work evidence table.
- Deployment packages for every admissible configuration and the unmodified anchor: 38 packages, batch 3, final. They run bitwise from
  the six raw fields with every refusal at both entry points, and also from a restored backup.
- Independent verification with separate code. Every recomputed number matches (largest gap 2.3e-13), and the verifier replays the
  nomination 34 / 34. It also reports:
  - its access check as FAIL under the registered rule: authorized, logged diagnostic MODEL_SELECT reads went outside the two named
    entry points (deviations 9, 13, 19);
  - provenance as ATTENTION: the P2 card was committed after its first fit (deviation 10);
  - open documentation notes D3, D4 and D6.
- Outside the verifier's checks: the fixed-recipe, probe, R3, learning-curve and income-noise numbers of the reader-robustness
  diagnostics were not independently recomputed.
- The engineering review and the claims review.

**Did not run, and why:**
- No final reader evaluation, no ATTACK SEX read and no DEV_REPORT truth read: no configuration qualified.
- No P3 seed 1 and no P3 compression twin: they were conditional on predictions that P3 refuted.
- Unused budget: 15 seed-specific slots, none with a justified use under the protocol.
  - 8 OCC36: family A exhausted its line; B and C used their two revisions.
  - 1 PRIV24: the two protection revisions were used.
  - 4 INC8: the branch failed, and no revision was allowed except for a defect.
  - 2 CTRL12: reserved for a P3 compression twin that its conditions did not trigger.
- No learning curve for P2_PAIR_M90 and no fixed-recipe check at B50 / M90.
- The reader-robustness agent's closing text was cut short by a platform safeguard (deviation 20). Its tables were complete; it was not
  relaunched.
- No off-device backup: no external drive is attached (see BACKUP_VERIFICATION.json).

**Deviations.** Items 1-21 are in PROTOCOL_DEVIATIONS.md. They include:
- the P2 card committed 71 s after its first fit (10);
- fits on an uncommitted file whose P2 path was unchanged (14);
- the verifier's widening of its own access-check rule after seeing results, flagged by the platform's safety monitor and restored in
  its records (15-16);
- logged diagnostic MODEL_SELECT reads outside the two named entry points (9, 13, 19);
- code-provenance gaps for families B and C (17).

## 8. Packages, public commit and backup

- **Packages:**
  - Location: `.private/pcrl_occupation_headroom_overnight_v1/runs/DEPLOY/` (private), checked in
    `deploy/DEPLOYMENT_CHECKS_BATCH3.json`.
  - Contents:
    - the strongest witnesses: `B__B2_SINGLE__H3_S{0,1}` and `A__A1_PO_16__H3_S{0,1}`;
    - the unmodified anchor: `SOURCE__JOINT__H3_S{0,1}`;
    - the reserve-keeping frontier point `PROT__P2_LOC_B80__H3_*`;
    - the decisive matched controls: `PROT__P2A_CONT__H3_*` and `PROT__P2A_COMP_B80__H3_*`.
  - Usage: QUICKSTART.md.
- **Public commit:** see PUBLICATION_RECORD.json. The branch is `publication/pcrl-occupation-headroom-overnight-v1` on
  https://github.com/Bostesa/PCRL; its parent is the registration receipt 435fe44e5.
- **Backup:** same-device backups with a restore check from the backup alone, recorded in BACKUP_VERIFICATION.json. The independent
  verifier also redeployed all 38 packages from a restored copy. The off-device copy is pending, with the exact command recorded.
- **Elapsed work:** the actual figures are in RESOURCE_ACCOUNTING.json (about 3.2 h and 4.8 of 14 CPU-h when the research stage
  closed; closeout figures there).

## Failed-mechanism explanation and closest existing methods (fallback programme)

- **Task learner (kept as a component).**
  - Native-category splits on the 106-level POBP are worse than the teacher; the ordinal code reduces that variance. Handling
    categoricals in gradient boosting is established practice (evidence table, family 1: ESTABLISHED), so no novelty is claimed.
  - Stacking with household-grouped cross-fitting is established (family 2).
  - Residual learning on the frozen teacher (family C) failed for a measured reason: in-sample optimism of the teacher.
- **Protection (failed as a whole-release method).** The coalition-aware correction is a standard adaptation of conditional
  adversarial debiasing (Zhang, Lemoine & Mitchell 2018; evidence table row 4.1) and of adversarial gradient tree boosting (Grari et
  al. 2019; row 4.3). Its adversary reads the linked pair, and the income output is frozen. Three measured reasons stop it:
  1. The budget projection does not control the binding men's-guard cost across refits (transfer error of +0.0016 to +0.0029 vs
     −0.0008 to −0.0012, against a slack of 0.0041-0.0054).
  2. The in-training guard either oscillated (P1 Lagrangian) or moved the binding constraint to women (P3 static weights). The
     balance between the two was not located within the two allowed revisions.
  3. The occupation view alone stays more SEX-readable than the inherited JOINT_TREE occupation views (+0.019 to +0.051 per cell for
     the PAIR and LOC configurations).
- **Selection (not a mechanism failure).**
  - The per-cell recipient gate on two household splits fails for H3's own income view in two seed-0 cells. On this grid that is
    consistent with split noise, and it applies to every H3-anchored release alike.
  - Beyond that gate, no configuration has a stable benefit against all six required comparators.
- **No impossibility claim.** Exhausted lines were stopped; no impossibility result is claimed from this finite search.

## Interpretation limits

- Deterministic does not imply one-to-one.
- An invertible recalibration may change usefulness or fitted-reader behaviour without removing information.
- A point pass is not a population guarantee.
- An unresolved condition is neither an established violation nor equivalence.
- Every MODEL_SELECT result is adaptive. The P2 / P3 budgets and weights were set from logged MODEL_SELECT reserve slacks (a design
  input, not fitting), and family B chose its revisions after MODEL_SELECT feedback.
- Rotating previously used people into new splits (R3) is not fresh confirmation.
