# Family H card: headroom-first income learner (investigator H)

Exploratory development on previously used records (PCRL income-headroom successor development campaign). Not confirmation.
Public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida. The service predicts an income class (2 classes)
and an occupation group (5 classes) from six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP; admitted 145-column encoding `X`).
Each recipient receives only its own float64 probability vector and argmax decision. SEX is the dataset's recorded demographic
column. Readers are separate evaluation classifiers that estimate SEX from the released probability vectors (each output alone
and the linked pair).

Written 2026-10-18 02:27Z, before any real-data fit of this family and before the registration commit; the initial version
was refined at about 02:32Z, still before any fit and before registration (the draft nomination rule's c3 bar and the income-view
recipient screen made the full LOCAL_TREE correction part of H1, so that its control differs only in the base). Code:
`ihs/fam_h.py`.
Unit ids `IHS_H_<config>_S<s>_V<n>`; configurations `H1_*`, `H2_*`, `H3_*` (versions) and controls `HC_*`.

## Measured bottleneck

- The retired release started from tree income scores with little held-out headroom: on MODEL_SELECT, H_LJ (LOCAL_TREE income)
  has income summed-Brier excess vs Ucal -0.00306 / -0.00394 (seed 0 / seed 1), T_TREE -0.00367 / -0.00446 and TASK_REFIT
  -0.00396 / -0.00488 (`REFERENCE_MS_TABLE.csv`).
- Its 6-level code spent about 0.0075 of summed Brier and ended 0.35-0.40 SE from the 0.005 cap (RETIRED_B3CTL +0.00437 /
  +0.00442). B2TWIN shows the code alone costs about 0.004 for a pair drop of only about 0.006 against H_LJ.
- The continuous JOINT_TREE income score is the most efficient privacy lever on record (H_JJ pair 0.5952 / 0.5968 against H_LJ
  0.6241 / 0.6239) but spends the whole income room (Brier +0.00571 / +0.00499; seed-1 accuracy loss 0.01025 > 0.010).
- The reviewer's reserve (`RESERVE_RULE.json`, read as a draft before the registration commit; k = 2 x floor SE) needs income
  Brier excess vs Ucal about <= 0.0023-0.0025, accuracy loss <= about 0.0042-0.0044 and NLL excess <= about 0.0063-0.0069 at
  the floor SEs. From T_TREE's registered values the room is about 0.0062 / 0.0067 (Brier), 0.0056 / 0.0049 (accuracy) and
  0.0124 / 0.0130 (NLL).

## Exact operation that changes the release

Occupation output: the frozen, accepted JOINT_TREE E20 occupation output (`ihs.bases.joint_occupation`), unchanged.

Income output (log-odds `d`, released as `trees.probabilities('income', d)`, decision = argmax):

1. **Step 1, task learning (no privacy, no SEX).** `s_HGB(x)`: scikit-learn HistGradientBoosting income log-odds, on a
   deterministic compact re-coding of `X` (AGEP, WKHP as admitted; COW, MAR, POBP as native categories; SCHL as its ordinal
   value) or on `X` itself. A fixed grid of 7 settings (learning rate 0.05, up to 600 trees, leaves 8/16/32 x min leaf 20/80 on
   the compact coding, plus leaves 16 / min leaf 20 on `X`; L2 1.0) and the tree count are chosen by household-separated 5-fold
   cross-validation on MODEL_FIT (minimum out-of-fold summed Brier), then refitted on all MODEL_FIT. The STRONG base is the
   pre-declared 3-coefficient logistic stack `s_B = w0 + w1 s_HGB + w2 s_T` of `s_HGB` and the frozen T_TREE income log-odds
   `s_T`, fitted on CALIBRATION (both scores are out-of-sample there). Honest quality: out-of-fold on MODEL_FIT (HGB grid only:
   T_TREE is in-sample on MODEL_FIT), then household cross-fitted maps on CALIBRATION against raw T_TREE, Platt-mapped T_TREE,
   TASK_REFIT, U0 and Ucal (in-sample there), then one development score on MODEL_SELECT (`HC_STRONG`). Diagnostic stacks
   (HGB alone, HGB + TASK_REFIT, HGB + T_TREE + ExtraTrees) are reported, not selected.
2. **Step 2, privacy (LJDIR: budgeted transplant of the replication's privacy-trained corrections as a residual).**
   `d(x) = a * (s_B(x) + D_L(x) + beta * D_J(x)) + b`, with `D_L = s_L - s_T` and `D_J = s_J - s_T` the stored LOCAL_TREE and
   JOINT_TREE E20 income leaf corrections on the frozen T_TREE routing (Stage B: LOCAL against online SEX critics on each single
   wire, which lowered H_LJ's income view from T_TREE's 0.5577 / 0.5585 to 0.5427 / 0.5421 while improving its accuracy; JOINT
   adding the linked-pair wire, trained jointly with the very occupation output released here). `D_L` is kept at full strength
   (it serves the income-view recipient screen, which binds near 0.547 against NEURAL_RAW seed 1). `beta` is the largest value
   on the grid 0, 0.05, ..., 2.0 for which it and every smaller grid value stay inside budget **B1**: household 5-fold
   cross-fitted CALIBRATION costs relative to the task-only base (no residual; paired per person) of at most **0.0045 summed
   Brier, 0.0080 nats NLL and 0.0035 accuracy**. B1 is about 0.65-0.72 of the room above (fixed now, before any fit; never tuned
   on MODEL_SELECT). `(a, b)` is a 2-parameter Platt map refitted on CALIBRATION at that beta: one slope for the whole score, so
   it cannot undo the residual selectively.

What changes in the release relative to H_LJ: only the income probability vector (a continuous score from a different learner
plus the LOCAL_TREE correction and a scaled copy of the JOINT_TREE income correction, recalibrated); the occupation vector is
bitwise H_LJ's. The income output does not depend on the released occupation output. No new fit in this module uses SEX; SEX
entered only through the stored LOCAL_TREE / JOINT_TREE corrections (replication Stage B, MODEL_FIT SEX as a training-time
label). MODEL_SELECT is read only through `python3 -m ihs.devrun score`.

Relation to the old mixtures: the source's B3 code used 0.5 s_L + 0.5 s_J = s_T + 0.5 D_L + 0.5 D_J, coded to 6 levels. LJDIR
differs in the base (the stronger learner), keeps D_L at full strength, sets the JOINT strength by a pre-declared CALIBRATION
budget rather than a sweep, and releases a continuous score (no code). The learner-isolation control below is exactly "the
same step on the old base", so any difference between H1 and it belongs to the learner.

## Why it could improve the observed trade-off

- If a stronger learner has genuine held-out income headroom, the same privacy spend ends further from the caps, or a larger
  privacy spend fits inside the reserve (the B3CTL failure was lack of room, not lack of pair benefit).
- The continuous JOINT direction was the most efficient lever on record; a fractional strength should be more efficient still,
  because the task cost of a perturbation grows roughly quadratically in its size while the critic-trained privacy effect
  starts roughly linearly. A coarse code is not needed (it cost about 0.004 of Brier on its own in B2TWIN / B3CTL).

## Controls that could explain away the benefit

- **HC_STRONG** (matched task-only control): the STRONG learner's own released output beside the JOINT_TREE occupation output
  (beta = 0). Measures the held-out headroom and the learner's own leakage.
- **HC_TTREE_LJDIR** (learner-isolation control): the identical step on the old base T_TREE (Platt-mapped), which is
  `a * (s_L + beta * D_J) + b`, i.e. the same privacy step applied to the old LOCAL_TREE income score, with the same B1 rule
  measured against Platt-mapped T_TREE. If it matches H1's pair and utility, the stronger learner contributes nothing and the
  effect is ordinary LOCAL/JOINT-type mixing.
- Reference context (no new fit): H_LJ, H_JJ, RETIRED_B3CTL, B2TWIN, T_TREE, TASK_REFIT.

## Revise / abandon observations

- Step 1: if the held-out summed-Brier gain of STRONG over T_TREE is below about 0.0005 (CALIBRATION cross-fitted, and
  HC_STRONG vs T_TREE's registered MODEL_SELECT value), report that there is no real task headroom to create; Step 2 then
  proceeds as an attribution test only (its effect would belong to the transplanted direction, not to new headroom).
- H1 fails the reserve on seed 0: the CALIBRATION budget did not transfer; revise once with a smaller or decision-preserving
  spend, or stop.
- H1's pair proxy is not below both eligible references and B2TWIN (the draft nomination rule's c3 bar, two-seed mean 0.6179)
  in every development cell, or a recipient screen fails (single view > reference + 0.010 in a cell): revise or abandon.
- H1 within about 0.003 of HC_TTREE_LJDIR on pair and utility: the learner adds nothing; investigate that rather than adding
  complexity.
- Promising on seed 0 (reserve met, pair below both references and B2TWIN in all four cells, recipient screens pass): run seed 1.

## Closest known methods

- Stacked generalization / model averaging (Wolpert 1992; Breiman 1996) for Step 1; Platt scaling (Platt 1999) for the map.
- Adding a fine-tuning correction learned on one base to another base with a scaling coefficient: task arithmetic / task vectors
  (Ilharco et al. 2023). Interpolating a task-only and an adversarially debiased predictor to trade accuracy for fairness
  (adversarial debiasing, Zhang et al. 2018; JOINT_TREE / LOCAL_TREE and the B3 mixture in this project).

## Invented-data check (no real data)

`python3 -m ihs.jobqueue run --unit IHS_H_SELFTEST_V2 ... python3 -m ihs.fam_h selftest` (02:25Z): PASS, 14 checks: compact
coding shape, ranges and rejection of an incomplete one-hot block; household-separated folds; CV curve improves; summed Brier
convention (2 (p - y)^2 for the binary vector); Newton logistic map equals scikit-learn's unpenalized fit; budget rule (zero
cost at beta = 0, every chosen beta and all smaller ones inside budget, cost increases, a cheap invented direction gets beta >
0, a costly one gets beta = 0); release is a valid probability vector with decision = sign of the log-odds; the stored
JOINT_TREE income direction is computed from invented `X` alone. (V1 at 02:25Z: PASS on the first 13 checks; V3 at 02:29Z
after generalizing the residual to "fixed stored corrections + beta x D_J", costs always against the task-only base: PASS, 15
checks, adding the flag for a fixed residual that alone exceeds the budget.)

End-to-end dry run on invented data (unit IHS_H_DRYRUN_V1, 02:30Z; scratch script outside the repository; role loaders, the
JOINT occupation output and the stored U0 / Ucal references replaced by invented arrays; all outputs in scratch): `step1` (small
grid) -> `fit` for HC_STRONG, H1_STRONG_JDIR, HC_LOCAL_JDIR and a fixed-residual test configuration -> `verify` (`predict` from
features alone equals the saved release, <= 1e-12, decisions identical): PASS for all four. Re-run after the 02:32Z refinement
(unit IHS_H_DRYRUN_V2, 02:33Z queue time): HC_STRONG, H1_STRONG_LJDIR, HC_TTREE_LJDIR and a LOCAL-base test configuration all
PASS, plus the identity s_T + D_L = s_L of the stored scores on invented features (so the control is exactly the same step on the
old LOCAL_TREE income score).

## Allotment

At most 10 seed-specific fitted configurations (controls included), an initial version plus at most two revisions, about 1.2
CPU-hours including scoring.

## Record of fits and revisions (appended as they happen)

### Step 1, seed 0 (unit IHS_H_STEP1_S0_V1, 02:36-02:37Z, 68 CPU-s, 72 fits; no SEX, no MODEL_SELECT)

- MODEL_FIT out-of-fold summed Brier is flat across the 7 settings (0.34914-0.34969); chosen: compact coding, 16 leaves, min
  leaf 20, 94 trees (0.34914). T_TREE has no out-of-fold value on MODEL_FIT (in-sample there).
- CALIBRATION (household 5-fold cross-fitted maps; summed Brier difference vs raw T_TREE, positive = worse, household-clustered
  SE): HGB raw +0.00191 (0.00186); HGB mapped +0.00277 (0.00189); **STRONG stack (pre-declared) +0.00028 (0.00082)**; HGB +
  TASK_REFIT stack -0.00036 (0.00076); HGB + T_TREE + ExtraTrees +0.00085 (0.00085); Platt-mapped T_TREE +0.00083 (0.00038);
  TASK_REFIT raw -0.00070 (0.00035); LOCAL raw -0.00073 (0.00088); JOINT raw +0.00425 (0.0023); U0 +0.00060; Ucal (in-sample)
  -0.00052.
- Reading: on CALIBRATION the new learner creates no held-out headroom over T_TREE (gain below 0.0005; the only stored score
  that beats T_TREE there is TASK_REFIT).

### Initial version H1 and controls, seed 0 (02:36-02:44Z)

- HC_TTREE_LJDIR S0 (fit IHS_H_HC_TTREE_LJDIR_S0_V1): T_TREE Platt slope w1 = 0.980 (so the release is a*(0.980 s_T + 0.039 +
  D_L + beta D_J) + b, about the old LOCAL score + beta D_J). B1 chose beta = 0.10: the realized-accuracy term bound (CALIBRATION
  accuracy cost +0.0034, SE 0.0029, already at beta = 0, i.e. from D_L alone; Brier alone would have allowed beta of about
  0.6-0.7: beta 0.5 +0.0021, 0.75 +0.0052).
- H1_STRONG_LJDIR S0 (fit IHS_H_H1_STRONG_LJDIR_S0_V1): STRONG stack 0.043 + 0.384 s_HGB + 0.615 s_T. **Outside budget B1 even
  at beta = 0**: D_L alone costs +0.0044 (SE 0.0030) realized accuracy against the STRONG base on CALIBRATION (> 0.0035), so the
  release is STRONG + D_L (beta = 0), Platt-mapped. Brier column: beta 0.5 +0.0026, 0.75 +0.0057.
- MODEL_SELECT seed 0 (`ihs.devrun score`; values vs Ucal, household-clustered SE):

  | Config | inc. Brier | inc. NLL | acc. loss | reserve (min ratio) | pair cells | income view | pair benefit vs NEURAL_RAW / H_LJ / B2TWIN (min..max) |
  |---|---|---|---|---|---|---|---|
  | HC_STRONG | -0.005271 (0.001141) | -0.007263 | -0.002875 | MET (4.578) | 0.6363-0.6388 | 0.5561-0.5607 | -0.0110..-0.0052 / -0.0155..-0.0114 / -0.0195..-0.0183 |
  | H1_STRONG_LJDIR (beta 0) | -0.004542 (0.001097) | -0.006342 | -0.002875 | MET (4.578) | 0.6229-0.6241 | 0.5396-0.5441 | +0.0042..+0.0082 / -0.0008..+0.0020 / -0.0061..-0.0036 |
  | HC_TTREE_LJDIR (beta 0.10) | -0.002341 (0.001216) | -0.003858 | -0.003000 | MET (4.607) | 0.6180-0.6195 | 0.5374-0.5408 | +0.0094..+0.0123 / +0.0053..+0.0054 / -0.0027..+0.0025 |

- Readings. (1) On MODEL_SELECT the STRONG learner is 0.0016 better than T_TREE's registered value (-0.00367) and 0.0013 better
  than TASK_REFIT's (-0.00396): the headroom shows on MODEL_SELECT but not on CALIBRATION (mixed evidence; a precision-weighted
  pooling of the two held-out roles is about +0.0009). (2) H1 spent nothing: the noise-dominated accuracy term stopped it, and
  H1's realized MODEL_SELECT accuracy equals HC_STRONG's exactly (-0.002875), so the CALIBRATION accuracy cost of D_L was noise.
  (3) Even beta = 0.10 of the JOINT direction (control) lowered pair recovery by 0.0053 in every cell against H_LJ for about
  0.0007 of Brier: the direction is efficient; B1's accuracy term, not the Brier room, is what prevented spending.

### Revision 1: H2_STRONG_LJDIR_DP (decided 02:46Z, before its fit)

- Reason: readings (2)-(3) above. The reserve's accuracy condition (loss <= about 0.0042-0.0044) is also the known weak point
  of the JOINT direction (H_JJ seed-1 accuracy loss 0.01025), and a 2,982-person accuracy estimate cannot control it.
- One change from H1: **decision-preserving residual**. The released income decision is held equal to the task-only base's own
  decision sign(w . Z_B) (the STRONG stack): where a * (s_B + D_L + beta D_J) + b would cross, the log-odds is set to +-0.01 on
  the kept side. The realized accuracy cost against the task-only base is then exactly 0 on every role, so beta is set by B1's
  Brier and NLL terms (same B1, same grid, same STRONG base, D_L, Platt map). HC_STRONG is its matched task-only twin with
  identical decisions; HC_TTREE_LJDIR_DP (same step on the old base) is the learner-isolation control, fitted only if needed.
- Recorded expectation: beta about 0.6-0.7 (H1's Brier column); the release keeps HC_STRONG's MODEL_SELECT accuracy
  (-0.002875 on seed 0). Revise or abandon if the reserve fails, if the pair is not below both references and B2TWIN in every
  cell, or if holding the decisions removes most of the pair benefit (the decision itself is then the leak).
- Invented-data checks for the change: selftest V5 PASS (18 checks; adds exact-zero accuracy cost, decisions equal to the base,
  only crossing scores changed and held at +-0.01). V4 failed on an invented-budget test design (the costly invented direction
  exceeded the invented Brier budget at beta 0.4), kept in the record.
- Result, seed 0 (fit IHS_H_H2_STRONG_LJDIR_DP_S0_V1, score ..._SCORE, 02:47-02:48Z): beta = 0.85 (CALIBRATION cost vs base:
  Brier +0.00419 (SE 0.00233), NLL +0.00637, accuracy exactly 0); 8.5% of MODEL_SELECT people held at +-0.01; decisions equal
  HC_STRONG's for 100%. MODEL_SELECT: income Brier vs Ucal **+0.004970 (SE 0.001497)**, vs U0 +0.004522 (SE 0.001658); NLL vs
  Ucal +0.006192; accuracy loss -0.002875 (as HC_STRONG); **RESERVE_NOT_MET** (both Brier conditions; min ratio 0.020). Pair cells
  **0.6006-0.6079** (winner ET25 in all four), below NEURAL_RAW by +0.0206..+0.0300, H_LJ by +0.0154..+0.0243 and **B2TWIN by
  +0.0126..+0.0162 in every cell**; income view 0.5507-0.5513: **recipient screen fails against NEURAL_RAW (+0.0113 > 0.010)**,
  +0.0090 against H_LJ.
- Reading. (a) The privacy direction delivers a large pair benefit, but the CALIBRATION budget did not transfer: the realized
  MODEL_SELECT Brier cost vs HC_STRONG was +0.0102 against the CALIBRATION estimate +0.0042. The CALIBRATION cost SE grows with
  beta (0.0023 at 0.85) and "largest beta inside the budget" selects where the noisy estimate is low (winner's curse); at beta 0
  the two roles already disagreed by 0.0015 (CALIBRATION -0.0007, MODEL_SELECT +0.0007). (b) Holding decisions creates a mass of
  identical vectors at p = 0.5 +- 0.0025 whose membership (the crossing set of the JOINT direction) is itself readable: the income
  view rose from H1's 0.540-0.544 to 0.551 while the JOINT direction alone lowers it (H_JJ 0.526). Decision preservation is
  therefore abandoned.

### Revision 2 (last): H3_STRONG_LJDIR_PB (decided 02:51Z, before its fit)

- Reason: reading (a) and (b), and H1's noisy accuracy term.
- Change from H1 (the initial version's release form; decisions may change): a **precision-aware budget rule** on the same B1
  values: beta is the largest grid value (with every smaller one) whose cross-fitted CALIBRATION cost **plus one household-
  clustered SE** is within B1 for summed Brier (0.0045) and NLL (0.0080), and whose **expected accuracy cost** under the
  cross-fitted task-only base probabilities (mean of |2 p_B - 1| over changed decisions; low-variance, label-free given the
  map) is within 0.0035. The realized accuracy cost is reported, not gated. Same STRONG base, D_L, grid and Platt map; matched
  task-only twin HC_STRONG; learner-isolation control HC_TTREE_LJDIR_PB (same rule on the old base) if the allotment allows.
- This revision is adaptive: it was chosen after seeing MODEL_SELECT feedback for H1 and H2 (through `ihs.devrun score` only);
  beta itself is still set on CALIBRATION by the rule. The one-SE margin is a fixed choice, not tuned on MODEL_SELECT.
- Recorded expectation (from H1's CALIBRATION column, cost + 1 SE: beta 0.25 0.0019, beta 0.5 0.0045): beta about 0.45-0.5;
  MODEL_SELECT Brier vs Ucal about -0.000 to +0.001 if the H2 transfer gap scales with the spend; pair about 0.61; income view
  falling with beta. Selftest V6 PASS (20 checks; the precision-aware rule is never looser than the plain rule; expected accuracy
  cost is 0 without decision changes).
- Result. The expected-accuracy term bound, not Brier (seed 0: 0.00372 > 0.0035 at beta 0.30; seed 1 at 0.25), so beta = 0.25
  (seed 0) and 0.20 (seed 1), smaller than expected. Matched task-only twin HC_STRONG (seed 1 refit: compact coding, 8 leaves,
  min leaf 80, 185 trees; stack 0.042 + 0.304 s_HGB + 0.688 s_T). Learner-isolation (component) control HC_TTREE_LJDIR_PB: same
  rule on the Platt-mapped T_TREE base, which chose the same beta on each seed. All MODEL_SELECT values vs Ucal (SE):

  | Config | seed | beta | inc. Brier | inc. NLL | acc. loss | reserve (min ratio) | pair cells A0/A1/B0/B1 | income view | pair benefit min..max vs NEURAL_RAW / H_LJ / B2TWIN |
  |---|---|---|---|---|---|---|---|---|---|
  | H3_STRONG_LJDIR_PB | 0 | 0.25 | -0.002342 (0.001179) | -0.003592 | +0.000125 (0.002637) | MET (3.511) | 0.6099/0.6098/0.6103/0.6103 | 0.5307-0.5343 | +0.0179..+0.0208 / +0.0134..+0.0146 / +0.0065..+0.0106 |
  | H3_STRONG_LJDIR_PB | 1 | 0.20 | -0.003254 (0.001284) | -0.004879 | +0.001750 (0.002861) | MET (2.862) | 0.6127/0.6129/0.6144/0.6144 | 0.5324-0.5359 | +0.0211..+0.0230 / +0.0091..+0.0118 / +0.0012..+0.0064 |
  | HC_STRONG | 0 | - | -0.005271 (0.001141) | -0.007263 | -0.002875 | MET (4.578) | 0.6388/0.6387/0.6363/0.6363 | 0.5561-0.5607 | -0.0110..-0.0052 / -0.0155..-0.0114 / -0.0195..-0.0183 |
  | HC_STRONG | 1 | - | -0.005783 (0.001269) | -0.008079 | +0.000500 | MET (3.296) | 0.6360/0.6359/0.6365/0.6365 | 0.5582-0.5606 | -0.0020..+0.0009 / -0.0130..-0.0115 / -0.0209..-0.0169 |
  | HC_TTREE_LJDIR_PB | 0 | 0.25 | -0.000901 (0.001270) | -0.002063 | -0.001250 | MET (3.902) | 0.6119/0.6106/0.6121/0.6121 | 0.5321-0.5356 | +0.0168..+0.0190 / +0.0114..+0.0128 / +0.0047..+0.0092 |
  | HC_TTREE_LJDIR_PB | 1 | 0.20 | -0.002149 (0.001389) | -0.003652 | +0.002750 | MET (2.400) | 0.6179/0.6183/0.6156/0.6156 | 0.5362-0.5365 | +0.0157..+0.0218 / +0.0056..+0.0079 / -0.0000..+0.0012 |

- Readings. (1) H3 meets, on both seeds, point eligibility, RESERVE_MET, pair below both eligible references in all 16 cells,
  the recipient screens (income view at most -0.0017 above NEURAL_RAW; occupation output inherited) and pair below B2TWIN in all
  8 cells (narrowest: seed 1, cells B0/B1, +0.0012). Two-seed mean pair proxy 0.61185; min income ratio 2.862 (seed-1 accuracy
  loss). (2) What the STRONG learner is worth at the matched privacy step (H3 minus HC_TTREE_LJDIR_PB, same beta): income Brier
  -0.00144 / -0.00111 (more room), NLL -0.00153 / -0.00123, pair mean -0.0016 / -0.0033 (lower), accuracy +0.0014 / -0.0010;
  without it the seed-1 pair only ties B2TWIN (B cells 0.0000) and the min income ratio falls to 2.400. (3) The CALIBRATION budget
  still under-predicted the MODEL_SELECT cost: H3's Brier cost against HC_STRONG was +0.00293 / +0.00253 against CALIBRATION
  estimates +0.00047 / +0.00029; the reserve was kept because the expected-accuracy term stopped beta early, not because the
  Brier term transferred. (4) The headroom itself: HC_STRONG reads 0.0016 / 0.0013 below T_TREE's registered income Brier on
  MODEL_SELECT but +0.0003 above it on CALIBRATION (both seeds); alone it is UTILITY_HEADROOM_ONLY (pair above both references,
  income view +0.020 / +0.023 above NEURAL_RAW).
- Deployment: `python3 -m ihs.fam_h verifyall` (unit IHS_H_VERIFYALL_V1, 02:56Z, final code sha256 c890263d...e176):
  `predict(seed, config_id, X, names)` recomputes all 9 family-H releases bitwise (every role, both outputs) from features alone.
- Family H stops here (initial version plus two revisions; 9 of 10 configurations used). No further revision.
