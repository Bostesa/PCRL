# Coordinator's running findings notes (for the final documents; append-only)

## Stage 1 (task-only occupation learning), as of 06:15Z

- Every learner that kept POBP (place of birth, 106 admitted levels) as a native category was no better than, or worse than, the
  CatBoost teacher; coding POBP as its ordinal place-of-birth code (US states, territories, then countries grouped by continent in
  code order) created real occupation headroom in two independent families:
  - family A (HistGradientBoosting): A1_PO_16 NLL vs U0 -0.0161 / -0.0128 (seed 0 / 1), 0.0040 / 0.0046 nats better than the teacher
    (2.4 / 2.7 paired SE), men's guard -0.0039 / -0.0023 (teacher -0.0010 / +0.0006);
  - family B (household-cross-fitted stack of a repaired tree learner and a regularised additive model): B2_STACK -0.0167 / -0.0134,
    men -0.0035 / -0.0021; its equally tuned single tree learner B2_SINGLE -0.0147 / -0.0114, men -0.0041 / -0.0025;
  - family C (capacity-limited residual on the frozen teacher margin, then on cross-fitted teacher margins): no gain over the teacher;
    continuation = the teacher; vector-scaling calibration worsens the men's guard (+0.0017).
- The binding utility condition for every task-only output is the men's occupation Brier guard. The stronger learners improve it by
  about 0.0024-0.0031 Brier, the first direct widening of the room a protection step must spend.
- Where the gain comes from (A1_PO_16 vs the teacher): management and service classes; men's NLL vs U0 -0.0081 / -0.0047 (teacher
  -0.0034 / +0.0006).

## Witness recovery (WITNESS_BANK, seed 0, after the utility-only selection)

- Stronger task-only occupation learning did NOT change SEX recovery noticeably: occupation-view SCORE means 0.630-0.636 (A1_PO_16
  0.635, B2_SINGLE 0.635, B2_STACK 0.630, A1_PO_8 0.636, B3_CALTR 0.632) vs the teacher 0.633; pair beside H3 0.634-0.643 (B2_SINGLE
  0.643, A1_PO_16 0.636, B2_STACK 0.634) vs H3+T_TREE 0.636 and H3+TASK_REFIT 0.633. The utility headroom came without extra
  leakage and without any reduction; to qualify, protection must take the pair from about 0.64 to about 0.575 (JOINT_TREE's step
  took the teacher's 0.636 to 0.600).

## Protection, first attempt P1 (witness B2_SINGLE, seed 0; SCORE pair means, H3 income)

- Witness 0.6426; matched continuation twin P1_CONT 0.6327 (occupation view 0.6323); LOC_B80 0.6240 (0.5978); PAIR_B50 0.6233
  (0.6095); PAIR_B80 0.6190 (0.5936); PAIR_M90 0.6170 (0.5871). Comparators: SUCC 0.5997, RETIRED_B3CTL 0.5964, B2TWIN 0.6096,
  NEURAL_RAW 0.6191, H_LJ 0.6217, TWJ 0.6337.
- The linked-pair critic lowers recovery about 0.014-0.016 beyond the matched continuation and more than the local critic at the
  same budget (0.6240), at an occupation NLL cost of about 0.003-0.007 nats over the witness and men's guard -0.0028 / -0.0019 /
  -0.0014 (still >= 2 SE); but the best P1 point is about 0.017 above SUCC and 0.021 above RETIRED_B3CTL.
- P1's in-training per-SEX guard Lagrangian oscillated (mu 0 <-> 5 between epochs), so its out-of-fold projection did not transfer
  (fold-average label-free change 0.0029 vs 0.0015 deployed on MODEL_SELECT); revision P2 removes the in-training guard term.

## Protection, revision P2 (no in-training guard term; witness B2_SINGLE, seed 0; SCORE pair means)

- P2_CONT 0.6351 (occupation view 0.6289); P2_COMP_B80 0.6246 (0.6025); P2_COMP_M90 0.6233 (0.5871); P2_LOC_B80 0.6142 (0.5773);
  P2_PAIR_B50 0.6076 (0.5933; reserve met, min ratio 2.62); P2_PAIR_B80 0.5972 (0.5812; ratio 1.39, men's guard +0.0023);
  P2_PAIR_M90 0.5919 (0.5732; ratio 0.57). SUCC 0.5997, RETIRED_B3CTL 0.5964.
- At matched cost the pair critic beats the continuation twin by 0.028-0.043 and the compression control by about 0.027; P2_PAIR_B80
  matches SUCC's recovery with better occupation utility (NLL vs U0 -0.0025 vs SUCC +0.0025; men's guard +0.0023 vs +0.0044): on seed
  0 the new frontier dominates the inherited JOINT_TREE point, but outside the 2-SE reserve; within the reserve the best point is
  P2_PAIR_B50 at 0.6076 (0.008 above SUCC).
- Without the in-training guard the critic spends far more of the men's margin (P2_PAIR_B80 men +0.0023 vs P1_PAIR_B80 -0.0019).

## Protection benefit vs reader data; grid structure (reader robustness, 06:57-06:58Z)

- P1_PAIR_B80's benefit SHRINKS with reader data (seed 0, 4 cells per fraction): vs its matched continuation twin +0.024 -> +0.023 ->
  +0.014 at 1/4 -> 1/2 -> full FIT; vs H_LJ +0.011 -> +0.009 -> +0.003; vs NEURAL_RAW +0.007 -> +0.003 -> +0.000; vs SUCC -0.010 ->
  -0.018 -> -0.019. Its pair recovery rises 0.599 -> 0.619 (+0.020, the steepest rise; CONT +0.010, SUCC +0.011). The critic-trained
  correction mostly removes what small-sample readers extract; larger readers recover a good part of it; the 8,001-person final readers
  would probably shrink it further.
- CORRECTION (07:02Z, reader-robustness investigator; the bullet above is WITHDRAWN): with candidate and comparator read by the SAME
  frozen recipe in each cell, P1_PAIR_B80's benefit over P1_CONT does not shrink (LOG_LR +0.035 -> +0.029, CATI1 +0.029 -> +0.030,
  ET25 +0.020 -> +0.023, MLP64 +0.021 -> +0.027, HGB31 +0.024 -> +0.029 at 1/4 -> full FIT; 2 of 15 recipes flagged; median change
  +0.003); release-level recovery rises about equally (P1_PAIR_B80 +0.017, P1_CONT +0.020, SUCC +0.019, H_LJ +0.017, NEURAL_RAW +0.019).
  The frozen BENEFIT_SHRINKS_WITH_READER_DATA flag inherited VALIDATION-winner selection noise (P1_CONT's full-FIT S0R1 winner had
  VALIDATION AUC 0.654 but SCORE AUC 0.620: a winner's curse of 0.034). The diagnostic probe agrees (+0.025 vs CONT with MLP_BIG and
  GB_LONG). Lesson: read a shrink flag as established only when fixed recipes agree. The witness flags are only weakly supported by
  fixed recipes (4-7 of 15 recipes flagged; median change -0.002 to -0.003).
- The 8 grid cells contain only 2 independent household splits (R1, R2; shared by both release seeds); the R1-minus-R2 contrast of pair
  benefits correlates 0.73 between seeds, the reader-seed contrast -0.005. A STABLE_POSITIVE label means stable over 2 household splits
  x 2 reader seeds x 2 release refits, not over 4 independent splits.

## Paired cost decomposition (seed 0; candidate minus base; decomposition/DECOMP_S0.json)

- JOINT_TREE vs teacher: NLL +0.0147 (SE 0.0018), class 3 +0.0157, men +0.0203 / women +0.0088.
- B2_SINGLE vs teacher -0.0026 (men -0.0045, women -0.0006); A1_PO_16 vs teacher -0.0040 (men -0.0047, women -0.0033).
- P1_PAIR_B80 vs its witness +0.0053 (SE 0.0011): classes 0 / 1 / 3 +0.0026 / +0.0037 / +0.0038, class 2 -0.0046; men +0.0062, women
  +0.0045 (balanced); P1_PAIR_M90 +0.0070; P1_LOC_B80 +0.0053 but men +0.0114 / women -0.0010 (spends the binding men's margin);
  P1_CONT -0.0012 (task continuation improves the witness).

## Income branch (INC_BANK, both seeds)

- The POBP-ordinal repair improves the source's STRONG income learner in household cross-fitting (Brier -0.00288, NLL -0.00422,
  accuracy loss -0.00225 on seed 0); on MODEL_SELECT the task-only twin INC_TW_PO raises the whole-release minimum ratio from H3's
  3.511 / 2.862 to 4.889 / 3.144, but the branch configuration INC_H3_PO (the same base with the source's H3 transplant step
  unchanged) stays at 3.556 / 2.840: the transplanted critic-trained corrections spend the accuracy the repair gains on seed 1.
- Its income view still fails the frozen per-cell recipient gate (INC_H3_PO minus H_LJ up to +0.018 / +0.030; minus SUCC up to +0.026 /
  +0.055 in single cells), and its pair recovery equals the witness's (about 0.640), as expected for a task-only occupation output.
- Consequence: no configuration of this campaign can pass the frozen nomination rule; even the comparators' own income views differ
  from each other by about 0.03 per cell, so the rule's per-cell recipient gate is unpassable on this grid. DEV_REPORT stays closed.

## Reader robustness, interim (06:50Z)

- Witness benefits SHRINK with reader data (BENEFIT_SHRINKS_WITH_READER_DATA in 5 of 7 comparisons): B2_SINGLE vs SUCC -0.032 ->
  -0.035 -> -0.040 at 1/4, 1/2, full FIT; witness pair recovery rises +0.017 / +0.018 from 1/4 to full FIT vs +0.010 for SUCC / H_LJ /
  NEURAL_RAW: the 4,000-person grid probably understates the gaps that 8,001-person readers would see.
- QUALIFICATION (07:16Z, reader-robustness investigator; supersedes the inference in the bullet above): under fixed recipes (each
  recipe reading candidate and comparator alike) only 2-7 of 15 recipes flag against each comparator (median change -0.003 to
  +0.001); the witnesses' recovery gain from 1/4 to full FIT matches SUCC's under linear and MLP readers (median +0.017 to +0.019
  overall) and exceeds it only under tree readers (+0.025 to +0.030 vs +0.019). "The grid understates the gaps" holds for tree
  readers only and is not established in general.
- A diagnostic probe beyond the frozen slate (MLP 256-128, long HGB, kNN on log p; never selection) does not change the ordering of
  pair recovery (Kendall tau 0.94-1.0; 0.83 for kNN); the SCORE-max oracle sits 0.003-0.008 above the frozen winner.
- Recipe confound: candidate and comparator winners differ in recipe family in 67% of cells, but reading both with the same recipe
  keeps the sign of the mean benefit (SUCC ahead of H_LJ / NEURAL_RAW / TWJ under 18 of 18 informative recipes; ahead of RETIRED_B3CTL
  under 2 of 18). The pair LOOKUP reader is non-informative on continuous outputs (AUC about 0.4997).
- Income gate noise: the split SD of an income-view difference is 0.024 (bootstrap SE 0.009, reader-seed SD 0.002); a true difference
  of exactly 0 fails the per-cell gate against one comparator with p = 0.83 and against five other income wires with p = 0.99; a true
  0.010 deficit still fails with p = 0.62 per comparator. On the 8-cell mean H3's income view reads LOWER than every other income wire.
- Sign stability: median sd_total 0.012 = reader / split spread 0.0084 and bootstrap SE 0.0088 in quadrature.

## Reader-grid facts (references; before any candidate)

- Seed-0 SCORE means: H3+T_TREE pair 0.636 (occupation view 0.633), SUCC 0.600 (0.561), RETIRED_B3CTL 0.596, H_LJ 0.622, NEURAL_RAW
  0.619, B2TWIN 0.610, TWJ 0.634; H3's income view 0.526 (the floor of any H3-anchored occupation-only step).
- Stability spread sd_total 0.009-0.015 per comparator: STABLE_POSITIVE needs a per-seed mean benefit of about 0.02-0.03 over every
  required comparator; SUCC itself is POSITIVE_NOT_STABLE vs H_LJ / NEURAL_RAW, SIGN_CHANGES vs B2TWIN, NOT_POSITIVE vs RETIRED_B3CTL.
- Income views are near chance (SCORE AUC 0.50-0.57) with large cell-to-cell swings: SUCC minus H_LJ +0.031 in R1 cells, -0.038 in
  R2 cells (seed 0). The frozen per-cell recipient point gate (every cell <= 0.010, 6 comparators x 8 cells x 2 views) therefore fails
  every H3-anchored release through reader noise in near-chance income views (verifier finding 05:55Z). Methodological lesson for
  future rules: a per-cell point gate on near-chance views has a high false-failure rate; a pooled-cell condition with a stability
  bound would be the natural design. Not changed here (frozen).

## JOINT_TREE cost decomposition (references)

- Relative to H3+T_TREE, SUCC's extra occupation NLL sits almost entirely on one SEX-typed class and on men: seed 0 class 3 (natural
  resources / construction / maintenance) +0.0144 vs -0.0012; men's NLL excess +0.0170 vs -0.0034; seed 1 class 1 (service), men
  +0.0094 vs +0.0006.

## Protection P2 / P2A over both seeds (8-cell grid summaries PROT_P2, PROT_P2A; comparator minus candidate pair AUC; 07:24Z)

- Against its own matched controls the linked-pair critic lowers recovery on BOTH seeds, more on seed 0: P2_PAIR_B80 vs P2_CONT
  +0.032 (seed means +0.038 / +0.026), vs the cost-matched compression control P2_COMP_B80 +0.018 (+0.027 / +0.009); P2A_PAIR_B80
  vs P2A_CONT +0.040 STABLE_POSITIVE (+0.058 / +0.021), vs P2A_COMP_B80 +0.026 (+0.043 / +0.010); P2_PAIR_M90 vs P2_CONT +0.040
  STABLE_POSITIVE. At B50 the margin over the compression control is small (P2 +0.007, P2A +0.011; SIGN_CHANGES).
- Against the strongest frozen references it does not hold over both seeds: vs SUCC P2_PAIR_B50 -0.015, P2_PAIR_B80 -0.004
  (+0.002 / -0.010), P2_PAIR_M90 +0.004 (+0.008 / -0.0005), P2A_PAIR_B50 -0.011, P2A_PAIR_B80 +0.004 (+0.018 / -0.010); vs
  RETIRED_B3CTL every level is negative on the 8-cell mean (-0.003 to -0.022). Against H_LJ / NEURAL_RAW the B80 / M90 levels are
  +0.012 to +0.022 on the mean but change sign across cells (seed-1 means +0.002 to +0.012). The seed-0 statement that P2A_PAIR_B80
  sits below all six required comparators in every seed-0 cell does not replicate on seed 1.
- Reserve (MODEL_SELECT, both seeds): only the B50 levels meet it on both seeds (P2_PAIR_B50, P2A_PAIR_B50). The B80 / M90 levels
  meet it on seed 1 (men's ratios P2 3.10 / 2.03, P2A 2.82) but not on seed 0 (P2_PAIR_B80 1.39, P2_PAIR_M90 0.57, P2A_PAIR_B80
  0.85); the compression controls miss it on seed 1 (1.92 / 1.80). So the levels that approach SUCC are the ones that spend the
  binding men's margin on seed 0, and the levels that keep the reserve on both seeds stay 0.011-0.015 above SUCC.
- Recipient gate: every protection candidate's occupation view is more readable than the JOINT_TREE occupation views of B2TWIN,
  H_LJ, RETIRED_B3CTL, SUCC and TWJ (cell maxima +0.019 to +0.051, 4-8 cells above 0.010 per comparator); against NEURAL_RAW it
  passes. The H3 income view exceeds 0.010 against H_LJ (+0.031) and TWJ (+0.015) in 2 cells. The critic lowers linked-pair
  recovery but leaves the occupation view alone more readable than JOINT_TREE's at this task quality.
- Reading: a coalition-aware occupation correction that is real against matched continuation and compression controls on both
  seeds, but whose two-seed frontier reaches the inherited JOINT_TREE releases only at levels that break the seed-0 reserve.

## Reader robustness of the P2 benefit (07:10-07:16Z; seed 0; diagnostics, never selection)

- Extra household split R3 (OHN1_devgrid_R3_diag; frozen slate refitted with the locked code, reproducing the cached R1 readers
  bitwise on 48 of 48 recipe-views): R3 behaves like one more split draw (median |R3 - mean(R1,R2)| 0.0067 vs median |R1 - R2| 0.0074;
  outside the R1-R2 range in 30 of 41 pairs, about 2/3 expected). R3 flips the sign of the 2-split mean in 5 of 41 pairs, all with
  |mean| < 0.010. P2_PAIR_B80 (R1 / R2 / R3): vs SUCC +0.0016 / +0.0033 / +0.0021; vs H_LJ +0.027 / +0.022 / +0.030; vs P2_CONT +0.040 /
  +0.036 / +0.047; vs P2_COMP_B80 +0.033 / +0.022 / +0.043.
- Reader probe (MLP_BIG, GB_LONG, kNN and all 18 frozen recipes as fixed readers): P2_PAIR_B80 beats P2_CONT under 18 of 18 fixed
  recipes (median +0.038), P2_COMP_B80 under 17 of 18 (median +0.026), the witness under 18 of 18, H_LJ under 17 of 18 (median
  +0.016), NEURAL_RAW under 18 of 18 (median +0.023). Against SUCC the sign depends on the reader (9 of 18 recipes positive, median
  -0.0001; frozen winner +0.0024, GB_LONG -0.0005, kNN -0.003); against RETIRED_B3CTL 4 of 18 (median -0.007).
- Winner's curse: the S0R1 frozen pair winner of P2_PAIR_B80 (ET5; VALIDATION 0.632-0.633, SCORE 0.583-0.584) has the largest
  VALIDATION-SCORE gap of all 184 pair winners in the grid (0.049); it inflates the R1 cells (vs NEURAL_RAW R1 +0.030 / +0.036 vs R2
  +0.011). With fixed recipes or the SCORE-max oracle the benefit vs H_LJ is about +0.016 to +0.018 instead of +0.024.
- Learning curves under fixed recipes (15 informative recipes): the benefit over P2_CONT is large at every reader size and shrinks
  only mildly (median change -0.003; 3 of 15 flagged); against the references it is flat (vs H_LJ median +0.0006, NEURAL_RAW +0.003,
  SUCC +0.0006); the frozen-winner shrink flags come from selection (P2_PAIR_B80's 1/4-FIT winner reads low). Rule kept from P1: a
  frozen BENEFIT_SHRINKS flag counts only when fixed recipes agree.
- Seed shift is a release effect, not a reader effect (reader-robustness, 07:26Z): from seed 0 to seed 1 the PAIR arms lose
  protection (P2A_PAIR_B80 frozen pair 0.581 -> 0.612, P2_PAIR_B80 0.597 -> 0.612; B50 levels +0.015 / +0.017; M90 +0.011) while CONT,
  COMP, the witnesses and the references move by at most 0.007. The seed-1 projection scales are smaller (B80 0.72 / 0.72 vs 0.86 / 0.92
  on seed 0; B50 0.53 / 0.55 vs 0.64 / 0.70); the investigator's MODEL_FIT out-of-fold readers agree (P2A_PAIR_B80 0.604 -> 0.617).
  Fixed-recipe probe: seed-0 P2A_PAIR_B80 beats SUCC under 18 of 18 recipes (median +0.013) and RETIRED_B3CTL under 16 of 18 (+0.005);
  on seed 1 under 3 of 18 (median -0.005) and 1 of 18 (-0.017). On both seeds and under every reader, only the benefit over the
  arm's own CONT / COMP twins holds (P2A seed 1: +0.033 / +0.016).
- Budget control is the limiting engineering fact: the out-of-fold projection sets the level, but its transfer error on the
  MODEL_SELECT men's guard (seed 0: about +0.002 nats over the projection; seed 1: reversed) is of the order of the witness's
  men's reserve slack itself (B2_SINGLE seed 0: 0.0054). The same nominal level therefore over-spends on seed 0 (more protection,
  reserve broken) and under-spends on seed 1 (less protection, reserve kept).
- CORRECTION (08:05Z, coordinator): in the "budget control" bullet above, the transfer error is about 0.002 in the men's guard's
  own unit (occupation summed-Brier excess for the SEX group), not 0.002 nats; the seed-0 / seed-1 directions are unchanged (protection
  investigator: +0.0016 to +0.0029 above the projection on seed 0, -0.0008 to -0.0012 on seed 1).

## Cost decomposition on both seeds (08:00Z; decomposition/DECOMP_P2_S0.json, DECOMP_S1.json; candidate minus base, occupation NLL)

- The witnesses' gain over the teacher is on men on both seeds (B2_SINGLE: men -0.0045 / -0.0048, women -0.0006 / -0.0014; A1_PO_16:
  men -0.0047 / -0.0053, women -0.0033 / -0.0038). JOINT_TREE vs the teacher costs +0.0147 / +0.0111 (seed 0 men-heavy +0.0203 /
  +0.0088; seed 1 women-heavy +0.0088 / +0.0135).
- P2_PAIR_B80 over its witness: seed 0 +0.0122 (men +0.0193, women +0.0049); seed 1 +0.0064 (men +0.0060, women +0.0068).
  P2A_PAIR_B80: seed 0 +0.0162 (men +0.0188, women +0.0135); seed 1 +0.0078 (+0.0057 / +0.0098). The seed-0 corrections spend the
  binding men's margin; the seed-1 corrections are about half as large and balanced, which matches their smaller scales and higher
  recovery.
- Controls: CONT improves the witness slightly on both seeds (P2_CONT -0.0011 / -0.0018); COMP costs +0.0103 / +0.0095 (B2_SINGLE)
  and spends men (+0.0162 / +0.0147 vs women +0.0042 / +0.0043), i.e. compression takes task quality from the binding group without
  matching the critic's protection.

## Seed identity of the witnesses (08:33Z; checks/A_SEED_IDENTITY.json; label-free)

- Family A's seed-1 models are bitwise the seed-0 boosting path truncated at the seed-1 tree count (A1_PO_16: 154 of 168 trees;
  A1_PO_8: 240 of 279); only max_iter and the unused random_state differ. Seed-specific inner folds (05:07Z design change) moved only
  the stopping point. With family B's bases identical across seeds, neither witness has independent seed refits; seed-to-seed
  agreement of the witnesses mostly re-evaluates nearly the same model against seed-specific references. MODEL_SELECT decisions of
  the two A1_PO_16 seeds agree on 98.9% of people (max |dp| 0.033).
