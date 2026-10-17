# Family B (CSR, coupled shared-statistic release): card revisions, appended as they happen

Investigator B. Florida ACS 2019 public-use person records; readers infer the recorded SEX column from the released income and
occupation probability vectors. Every MODEL_SELECT number below comes from `amd.devrun score` (ledger rows); every CALIBRATION number
from `amd.fam_b diag|diag2` (CALIBRATION income labels only, household 5-fold cross-fitting, no SEX, no MODEL_SELECT).

## B1, initial version (17:15Z), seed 0

- **Operation.** Occupation = JOINT_TREE E20 occupation output (frozen T_TREE routing + stored corrections; recomputed and checked
  against the stored outputs, max abs diff 0). Income = softmax(-s/2, s/2), s = L2 logistic h on the 4 occupation log-ratios, fitted on
  all of CALIBRATION (income labels).
- **Pre-fit screen (CALIBRATION only, `runs/B/DIAG/DIAG_S0.json`).** Income NLL excess vs U0 of h(occupation vector): T_TREE base
  +0.021 (lin) / +0.020 (quad); LOCAL_TREE base +0.045 / +0.043; JOINT_TREE base +0.044 / +0.041. Even the teacher's occupation vector
  does not carry the income information within the 0.010 cap.
- **MODEL_SELECT seed 0 (ledger entry 28).** Income NLL excess vs U0 +0.0530, Brier +0.0437, accuracy loss +0.0379 (worst normalized
  8.82, income|Ucal|Brier); occupation and guard as JOINT_TREE (men +0.00442, women -0.00494). Pair proxy 0.5666 (below the occupation
  proxy 0.5724: the pair is read no better than the occupation output alone), income proxy 0.5199.
- **Diagnosis.** The coupling does what the information identity says at the proxy level, but the occupation vector does not contain the
  income information: the card's revise condition ("income from the occupation vector exceeds the caps on MODEL_SELECT") is met by a
  factor of 5 to 9.

## B2, revision 1 (decided 17:17Z): add a coarse income-specific statistic with few levels

- **Why not the card's shared-statistic training.** B1 says the occupation output would have to absorb about 0.05 nats of income
  information. That information is SEX-informative, and the occupation output's own recipient non-worsening margin (H_LJ 0.5724 + 0.010)
  would then bind. Family A's A2 failed exactly there (occupation proxy 0.597). The CALIBRATION screen (`DIAG2_S0.json`) also shows that
  coupling the JOINT_TREE occupation vector into any income statistic adds at most about 0.0005 nats. The occupation vector holds almost no
  income information beyond the statistic.
- **Change.** Income = h(occupation log-ratios, one-hot c), where c = 6 decision-aligned bins of the LOCAL_TREE E20 income log-odds. The
  thresholds are label-free MODEL_FIT quantiles, 3 on each side of 0, so no bin straddles LOCAL_TREE's decision. h is fitted on
  CALIBRATION. The occupation output is unchanged (JOINT_TREE).
- **Information bound.** The pair is a deterministic function of (occupation output, c). Its information beyond the occupation output is
  at most log 6 (2.6 bits).
- **Why m = 6.** On the CALIBRATION screen it is the smallest m inside the income caps: NLL +0.0014 vs U0, Brier +0.0024 / +0.0036
  (U0 / Ucal), accuracy loss +0.0070. m = 4 fails Brier (+0.0081). MODEL_SELECT is more favourable to LOCAL_TREE income than
  CALIBRATION by about 0.002 nats.
- **Revise or abandon.** Abandon if the pair proxy is not below both references on seed 0 at feasible utility.

### B2 results (17:17-17:22Z, MODEL_SELECT, ledger entries 31, 32, 35, 37, 38, 39)

- **B2_CSR_JT_C6A (coupled).**
  - Eligible on both seeds: 12/12 clauses (worst normalized 0.55 / 0.31) and both guards. The men's guard is +0.00442 / +0.00394, inherited from JOINT_TREE.
  - Pair proxy 0.6131 / 0.6132 (H_LJ 0.6224 / 0.6233; NEURAL_RAW 0.6238 / 0.6269).
  - Income proxy 0.5404 / 0.5394; occupation proxy 0.5724 / 0.5853.
  - Promotable under §6, as EDGE (seed-0 men's guard is above 0.8 x cap).
- **Card control: coarse-only twin B2CTL_UNCOUPLED_C6A.** Same occupation output and same 6 levels, but income does not depend on the occupation output. The two releases carry the same pair information.
  - Eligible on both seeds; pair proxy 0.6153 / 0.6154.
  - Better income utility than the coupled B2: NLL +0.0013 / +0.0006 vs U0, against +0.0033 / +0.0026.
  - Coupling therefore moves the pair proxy by -0.002 (a finite-reader effect) and costs about 0.002 nats of income NLL on MODEL_SELECT. On CALIBRATION it had looked +0.001 favourable.
- **Matched task-only control B2CTL_TASKONLY_TT_C6A** (coordinator request). The identical construction on the teacher's outputs.
  - Pair proxy 0.6454 / 0.6463, against T_TREE's 0.6487 / 0.6493.
  - The occupation (0.643 / 0.639) and income (0.552 / 0.553) proxies fail the recipient bars.
  - The construction itself removes almost nothing. B2's gain is coarsening the income channel next to the privacy-trained JOINT_TREE occupation output.

## B3, revision 2 (decided 17:22Z): make the binned statistic partly pair-aware

- **Evidence.** B2's pair reading is income-occupation synergy: pair OUT_LR 0.613 versus single-output proxies of at most 0.545. Coarsening LOCAL_TREE income to 6 levels removed only about 0.008 of it.
- **Change.** Bin c6 (decision-aligned, label-free MODEL_FIT thresholds) of d_mix = 0.5 d_LOCAL_TREE + 0.5 d_JOINT_TREE income log-odds. Everything else is as in B2 (coupled h fitted on CALIBRATION).
- **Choice of alpha.** alpha = 0.5 came from the CALIBRATION-only screen `DIAG3_S0.json` (alpha in {0, .25, .5, .75, 1} x m {6, 8} x coupled / uncoupled).
- **Coordinator's suggestion (alpha = 1, a JOINT-only code).** Not run. The same screen puts it outside the caps: m = 6 Brier +0.0073 / +0.0084 coupled and +0.0068 / +0.0080 bins-only (U0 / Ucal); accuracy loss 0.012-0.017.
- **B3_CSR_JT_MIX50_C6A, seed 0 (entry 40).** Pair proxy 0.6018, but the income Brier clause fails: +0.0054 vs U0, +0.0058 vs Ucal (normalized 1.17). Not run on seed 1.
- **Coarse-only twin B3CTL_UNCOUPLED_MIX50_C6A (entries 41, 42).** The informative comparison under the controls logic. It was run after B3's failure: an adaptive step, logged.
  - Eligible on both seeds: 12/12 clauses (worst normalized 0.873 / 0.883, income Brier vs Ucal) and both guards.
  - Pair proxy **0.6000 / 0.6001**: 0.0224 / 0.0232 below H_LJ and 0.0238 / 0.0268 below NEURAL_RAW.
  - Income proxy 0.5277 / 0.5286; occupation proxy 0.5724 / 0.5853.
- **Budget.** 10 of 10 family-B configurations used. Family B stops here.
