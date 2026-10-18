# Family B card: complementary occupation learners and a household-cross-fitted stack (investigator B)

Exploratory development on previously used records (PCRL overnight occupation-headroom campaign). Not confirmation.
Public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida (housing-unit residents aged 25-64, civilian
employed). The service predicts an income class (2) and an occupation group (5 official broad civilian groups) from six raw
fields (AGEP, WKHP, COW, SCHL, MAR, POBP; admitted 145-column encoding `X`). Each recipient receives only its own float64
probability vector and argmax decision. SEX is the dataset's recorded demographic column; readers are separate evaluation
classifiers that estimate SEX from the released probability vectors. Family B is a TASK-ONLY occupation stage: no SEX enters any
fit, input or output of this family.

Written 2026-10-18 05:01-05:30Z, before any real-data fit of this family and before the registration commit. Before writing it
I read only label-free MODEL_FIT feature summaries (value counts of AGEP / WKHP and of the one-hot blocks), the stored Stage A
records (tree counts, internal curves) and the source campaign's reference table, and ran one invented-data timing probe
(unit OHN_B_TIMING_PROBE_INVENTED_V1, 4.2 CPU-s; no real data). Code: `ohn/fam_b.py`. Unit ids `OHN_B_<step>_<config>_S<s>_V<n>`.

## 1. Measured failure this addresses

- **The teacher is capacity-capped.** T_TREE's occupation model (CatBoost depth-4 oblivious trees on the one-hot `X`, learning
  rate 0.05, at most 800 trees, tree count on one 15% household-disjoint internal split) hit the cap on both seeds (800 / 799
  trees) with its internal-split loss still falling (seed 0: 1.21290 at 701 trees, 1.21227 at 751, 1.21192 at 800; seed 1:
  1.23808, 1.23792, 1.23780 at 799) (`STAGE_A_SEED.json`). TASK_REFIT (task-only leaf corrections on the same routing) is better
  than T_TREE on MODEL_SELECT (occupation NLL excess vs U0 -0.01379 / -0.01033 against T_TREE's -0.01213 / -0.00825;
  `REFERENCE_MS_TABLE.csv`), so task signal is left on the table by the teacher itself.
- **The headroom is narrow and concentrated in women.** T_TREE vs U0 on MODEL_SELECT: NLL -0.01213 (SE 0.00246) / -0.00825
  (0.00229), summed Brier -0.00512 / -0.00360, accuracy loss -0.0064 / -0.0061, men's Brier guard -0.00103 (0.00151) / +0.00059
  (0.00144), women's -0.00931 / -0.00788. The tree predicts men's occupation no better than U0.
- **The inherited output spends it.** JOINT_TREE (the inherited occupation output) sits at NLL +0.00253 / +0.00290 vs U0 and men's
  guard +0.00442 / +0.00394 (0.32 / 0.62 SE of margin): the binding constraints of the source campaign.

## 2. Exact operation and deployed output (initial design B1)

Runtime input: the admitted 145-column `X` only, through deterministic re-codings below. No SEX, label, identifier, household id
or other recipient quantity enters any runtime output.

**Base learner TREE (the equally tuned tree learner).** scikit-learn `HistGradientBoostingClassifier` (multinomial log-loss) on
the compact re-coding of `X` used by `ihs.fam_h.compact` (AGEP z, WKHP z, COW code [categorical], SCHL ordinal value, MAR code
[categorical], POBP code [categorical]). Grid G_T, 6 settings: learning rate 0.1, `max_iter` 600, `l2_regularization` 1.0,
`max_bins` 255, no early stopping, `max_leaf_nodes` in {15, 31, 63} x `min_samples_leaf` in {20, 100}; the tree count n in
{10, 20, ..., 600} is chosen together with the setting on the inner cross-validation curve (staged predictions). Output:
centered raw class scores z_T(x) in R^5.

**Base learner ADD (different inductive bias: smooth and additive in the class log-odds, a fixed small set of pairwise
interactions, L2).** Multinomial logistic regression (scikit-learn `LogisticRegression`, lbfgs, L2, unpenalized intercept,
tol 1e-6, max_iter 5000) on a deterministic label-free basis Phi(x):
- main effects: cubic B-spline of AGEP z (6 uniform knots over the MODEL_FIT range, 7 columns), cubic B-spline of WKHP z (6
  uniform knots over the MODEL_FIT range, 7 columns), indicators of the 13 WKHP values held by at least 200 MODEL_FIT people
  (20, 25, 30, 32, 35, 36, 38, 40, 45, 48, 50, 55, 60 hours), AGEP z and WKHP z (linear), and the COW, SCHL, MAR, POBP one-hot
  blocks of `X` as admitted;
- capacity **S** interactions: EDU6 x COW, EDU6 x AGEP-spline, COW x WKHP z, EDU6 x WKHP z, AGEP-spline x WKHP z, MAR x AGEP z,
  POB6 x EDU6;
- capacity **L** = S plus: SCHL x COW (full one-hot product), EDU6 x WKHP-spline, COW x WKHP-spline, AGEP-spline x WKHP-spline
  (tensor), POB6 x COW, POB6 x AGEP z, POB6 x WKHP z, MAR x EDU6, MAR x COW;
- EDU6 from the SCHL value: <= 15 (and SCHL=__OTHER__, the merged rare pre-grade-3 codes) no diploma; 16-17 high-school diploma
  / GED; 18-19 some college; 20 associate; 21 bachelor; >= 22 graduate or professional. POB6 from the POBP code: 001-056 US
  states and DC; 060-099 US territories; 100-199 Europe; 200-299 Asia; 300-399 the Americas outside the US; >= 400 and
  POBP=__OTHER__ Africa, Oceania and other;
- knots and the WKHP indicator list are computed once from MODEL_FIT features (label-free) and frozen;
- grid G_A, 6 settings: capacity in {S, L} x C in {0.03, 0.15, 0.75}. Output: centered logits z_A(x).

**Equal tuning effort.** Both learners: 6 settings; the same household-separated 5-fold inner cross-validation inside each
training set; the same criterion (mean out-of-fold NLL of the softmax of the raw scores); the same number of fits per training
set (6 settings x 5 inner folds + 1 refit of the chosen setting on the whole training set = 31). TREE reads its tree count off
the staged curve (checkpoints every 10 iterations); ADD's inner fits are warm-started along increasing C within a capacity (its
regularization path). The refit of the chosen ADD setting is a cold start.

**Final layers** (each fitted on the household-separated inner out-of-fold logits of the same training set; never on
CALIBRATION, MODEL_SELECT or any other role):
- **B1_STACK (candidate):** log-linear (geometric) pool `p(x) = softmax(a_T z_T(x) + a_A z_A(x))`, `a_T, a_A >= 0` (L-BFGS-B,
  bounds [0, 20], start (0.5, 0.5), minimizing mean NLL). Two parameters; the pool's free scale absorbs the temperature.
- **B1_SINGLE (matched control, the best equally tuned single learner):** L* = the learner (TREE or ADD) whose temperature-scaled
  inner out-of-fold NLL is lower; `p(x) = softmax(tau z_L*(x))`, tau in [0.05, 20] by bounded scalar minimization of the mean
  out-of-fold NLL (the Ucal solver settings). One parameter, the same data as the stack's layer.
- **Released:** float64 occupation probability vectors (centered scores, softmax; `acs_replication.trees.probabilities`
  convention) for all six roles from features; decision = argmax. The occupation output does not depend on the income output.
  Complete releases pair it with the frozen H3 income anchor (`--anchor H3`).

**Seeds.** Seed s changes the inner fold assignment (household folds, tag `OHN_B_inner_S{s}` for the main pipeline and
`OHN_B_inner_S{s}_out{k}` inside outer fold k) and the HGB `random_state`; the fits are otherwise deterministic, so the two seeds'
releases differ through their tuning folds.

**Household cross-fitting (full).** For each of the 5 outer folds k of `ohn.common.crossfit_folds` (tag OHN1_crossfit) the
ENTIRE label-using pipeline is rerun on MODEL_FIT without fold k: both learners' tuning (6 settings x 5 inner folds inside the
remaining people; setting and tree count chosen on their own inner curve), the refit at the chosen setting, the choice of L*,
and the final layer (tau or a) fitted on the NESTED inner out-of-fold logits of the remaining people. Fold-k predictions ->
`MODEL_FIT__oof`; the average of the 5 fold pipelines' MODEL_SELECT probability vectors -> `MODEL_SELECT__foldavg`. The
label-free steps (the admitted `X` dictionary, spline knots, WKHP indicator list, EDU6 / POB6 maps) come from all MODEL_FIT
features and are not refitted. No frozen historical checkpoint (T_TREE, U0, TASK_REFIT, Stage B corrections) enters any B1
layer, so B1's cross-fitted MODEL_FIT predictions are out of fold for every label-using step. Every outer fit asserts that its
training rows exclude fold k. Fits per seed: 2 learners x 6 training sets (main + 5 outer) x 31 = 372 model fits, shared by
B1_STACK and B1_SINGLE (counted once, sharing recorded), plus 6 final-layer fits per configuration.

## 3. Why it might create occupation headroom

- A learner that is not capped (tree count chosen on a 5-fold curve, leaf-wise trees with native category-subset splits for the
  106 POBP and 9 COW codes) may recover the signal the capped teacher leaves (the TASK_REFIT gap is 0.0017-0.0021 nats).
- Smooth age and hours effects and education x class-of-worker structure are represented by the additive model directly, but
  only piecewise by depth-4 oblivious trees on one-hot inputs. If the two learners err differently, a pool of their scores
  lowers NLL below the better one (stacked generalization).
- Every 0.001 nats or 0.0005 summed Brier of task margin below the original caps is margin a later privacy operation can spend.
  This family does NOT claim that better utility lowers SEX recovery; it only measures task margin and its decomposition by SEX.

## 4. Closest established methods

Stacked generalization (Wolpert 1992), stacked regressions with non-negative weights (Breiman 1996), the super learner with
cross-validated risk (van der Laan, Polley and Hubbard 2007); log-linear (geometric) opinion pools / products of experts
(Genest and Zidek 1986; Hinton 2002); additive models with selected pairwise interactions (Hastie and Tibshirani 1990; GA2M,
Lou et al. 2013) fitted as penalized multinomial logistic regression; histogram gradient boosting (Ke et al. 2017); nested
cross-validation for honest performance of tuned pipelines (Varma and Simon 2006; Cawley and Talbot 2010). Nothing here is novel.

## 5. Matched controls

- **B1_SINGLE**: the best equally tuned single learner with the same final-layer treatment (one temperature on the same
  out-of-fold logits). The stack's added value is B1_STACK minus B1_SINGLE.
- **Calibration and combination controls** (fully cross-fitted on MODEL_FIT, recorded in B1_STACK's OCC.json, not released):
  TREE raw, TREE + tau, ADD raw, ADD + tau, the log-linear pool, a linear pool `p = b p_T + (1 - b) p_A` (b in [0, 1]) and a
  multinomial-logistic stacker on [z_T, z_A] (C = 1, diagnostic): each with mean NLL, summed Brier, accuracy and paired
  differences against B1_SINGLE with household-clustered SEs.
- **Task-only references** (no fit): T_TREE and TASK_REFIT (equally exposed task-only; in-sample on MODEL_FIT, so compared on
  MODEL_SELECT only), U0, Ucal, and the inherited JOINT_TREE output (SUCC).

## 6. Primary correctness checks

1. Full household cross-fitting as above, with an in-code assertion that every fit used for fold k excludes fold k's people
   and households, and the scorer's deployed-versus-fold-average comparison on MODEL_SELECT.
2. Deployment parity: `verify` (`predict(seed, config, X, names)` from features and the saved, hash-recorded parameters equals the
   saved release bitwise on all six roles).
3. `selftest` (invented data only): compact and basis re-codings (shapes, completeness of one-hot blocks, label-free freezing),
   staged HGB predictions at n equal a refit with `max_iter` = n, nested fold integrity, the pool recovers known weights and the
   temperature solver agrees with a grid search, probability contract, predict = saved release on invented roles.

## 7. Decision rules, triggers and recorded expectations (fixed now)

- **Stack adds task value** if, on seed 0 (and seed 1 if run), the fully cross-fitted MODEL_FIT NLL difference
  d = STACK - SINGLE satisfies d <= -0.0005 nats and d + 1.96 SE < 0 (paired, household-clustered), AND B1_STACK's MODEL_SELECT
  occupation NLL excess vs U0 is below B1_SINGLE's (point). Otherwise: no added value beyond the best equally tuned single learner.
- **Headroom** (secondary, descriptive): B1 occupation NLL, Brier and accuracy vs U0, Ucal, T_TREE and TASK_REFIT on MODEL_SELECT
  (scorer, paired SEs) and both guards.
- **Promising -> seed 1:** a B1 configuration whose seed-0 MODEL_SELECT occupation NLL excess vs U0 is below T_TREE's (-0.0121),
  summed Brier excess below T_TREE's (-0.0051), men's guard at most T_TREE's + 0.001, with a sound cross-fit (next item).
  B1_SINGLE is then run on seed 1 with it, as its matched control.
- **Investigate, do not revise:** deployed minus fold-average MODEL_SELECT NLL beyond 2 SE or decision agreement < 0.95; any
  selftest or verify failure (fix, new unit version, report).
- **Revision triggers** (appended, dated, before the revision's fit; at most two revisions, at most two predefined capacity
  settings each):
  (R-learner) the stack adds no value and a_A < 0.1 or ADD's cross-fitted NLL is > 0.010 worse than TREE's: the additive model is
  too weak to complement; replace it by a compact neural learner (one-hot input, two small hidden layers, weight decay, epoch
  count by the same inner folds; torch 2.10 in the .venv interpreter) under the same nested protocol and equal-effort rule;
  (R-capacity) a chosen setting lies on a grid edge (600 trees at the largest leaves, or C = 0.75 / 0.03 at the edge capacity):
  extend that one dimension once, as one of two predefined capacity settings;
  (R-frozen) the B1 learners do not beat TASK_REFIT on MODEL_SELECT NLL (point): test a CALIBRATION-fitted pool layer of B1 with
  the frozen TASK_REFIT output (out of sample there; the MODEL_FIT cross-fit is then NOT fully out of sample for that layer,
  recorded as such).
- **Abandon** when, after at most two revisions, no family-B configuration reaches MODEL_SELECT occupation NLL excess vs U0 below
  TASK_REFIT's (-0.0138 / -0.0103) and the stack adds no value.
- **Recorded expectations (bets, before any fit):** TREE cross-fitted MODEL_FIT NLL about 1.20-1.22 nats; ADD 0.003-0.010 worse;
  STACK 0.0005-0.002 better than SINGLE (I give the "adds value" rule about a 40% chance); B1 MODEL_SELECT occupation NLL excess
  vs U0 within +-0.003 of T_TREE / TASK_REFIT (no more than about 0.002 better than TASK_REFIT); men's guard near T_TREE's.

## 8. Capacity settings and configuration budget

Initial design B1: one capacity design (the grids above, chosen by inner cross-validation) released as two configurations per
seed, B1_STACK and its matched control B1_SINGLE (seed 0 first; seed 1 only if promising). Revisions B2 and B3 (at most two),
each with at most two predefined capacity settings, appended before their fits. At most 12 seed-specific configurations, controls
included. Shared base-learner fits are counted once.

## 9. Exposure and roles read

MODEL_FIT features and occupation labels (tuning, fitting, cross-fitting; logged by `ohn.common.load_role`); no SEX, no income,
no CALIBRATION labels in B1; MODEL_SELECT only through `python3 -m ohn.devrun score`; features of all six roles (label-free).
Deployment needs python3 (Homebrew 3.14, scikit-learn 1.8.0): fitted models are pickled with their sha256 recorded.

## Record of fits and revisions (appended as they happen)

### Before registration (invented data only; no real-data label read)

- 04:57Z timing probe (unit OHN_B_TIMING_PROBE_INVENTED_V1, 4.2 CPU-s; invented data, scratch script outside the repository):
  HGB multinomial on 16,000 invented rows, 6 columns: about 3.3 / 5.0 / 8.2 ms per iteration at 15 / 31 / 63 leaves; sparse
  multinomial logistic with 700 invented columns: 0.14-0.29 s. Planning estimate: about 15-25 CPU-minutes per seed for both
  B1 base learners (2 x 186 fits), shared by B1_STACK and B1_SINGLE.
- 05:07Z `python3 -m ohn.fam_b selftest` (unit OHN_B_SELFTEST_V1, 9.5 CPU-s): PASS, 21 checks (compact coding and rejection of
  an incomplete one-hot block; additive basis shapes S / L, determinism, EDU6 / POB6 maps; nested outer and inner folds
  household-separated; staged HGB scores at n bitwise equal to a refit with max_iter = n; the pool recovers invented weights
  (1.29, 0.41 for true 1.3, 0.4) and keeps a weight at the 0 bound for a reversed input; temperature equals a grid search; the
  whole two-learner nested pipeline and both configurations on invented data: probability contract, cross-fitting arrays,
  diagnostics, out-of-fold NLL above in-sample NLL, append-only base storage).
- Count clarification (05:08Z, before registration): B1_SINGLE's final layer fits one temperature per learner per training set
  (2 x 6 = 12 per seed); B1_STACK fits one pool per training set (6 per seed). Cross-fitted diagnostics add 5 x 9 small
  layer fits per configuration (recorded as n_diagnostic_layer_fits; not released). The record-only field was added to the code
  after V1; selftest V2 re-runs on the final code.
- 05:09Z selftest V2 on the final pre-registration code (unit OHN_B_SELFTEST_V2, 9.6 CPU-s; `ohn/fam_b.py` sha256
  bed62600d2b1b6299ca628af1d8d65c2144ee793f2629d38afe4fab0e806655b): PASS, the same 21 checks.
