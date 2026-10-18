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
- 05:10Z code addition before registration (no change to B1's release, layers or cross-fitting): each fold pipeline also
  stores its base logits on CALIBRATION features (label-free), so that a possible CALIBRATION-fitted layer (trigger R-frozen)
  could be refitted per outer fold; a label-free helper computes the frozen T_TREE / TASK_REFIT occupation scores on the teacher
  routing (not used by B1). Selftest V3 (unit OHN_B_SELFTEST_V3, 9.5 CPU-s; sha256
  d38f6b16ecdc4b217f681a966f26ed522d462d23048816eccf2d4efc3800a3b2): PASS, 21 checks. This is the code for B1's first fits.

### B1 seed 0, base learners (MODEL_FIT only; no MODEL_SELECT read yet)

- 05:14-05:25Z TREE_V1 S0 (unit OHN_B_BASES_TREE_V1_S0_V1, 669 CPU-s, 31 main + 155 cross-fitting fits): **the tree learner
  overfits after about 20 iterations in every setting.** Main inner-CV NLL minimum 1.22586 at 20 iterations (15 leaves, min leaf
  100); at 600 iterations 1.350 (15/20) to 1.703 (63/100); every outer pipeline also chose 15 leaves and 20-30 iterations
  (OOF NLL 1.2269-1.2299). For scale (different splits, not a comparison): the teacher's 15% internal NLL was 1.2119 and U0's
  1.2273. Family A's HGB (31 leaves, lr 0.05) independently chose about 40 iterations (its unit log), so this is a property of
  the learner and encoding, not a defect of this code. Suspected cause: native category-subset splits on the 106 POBP codes
  (many with 10-20 MODEL_FIT people) behave like unregularized target encoding.
- 05:27Z diagnosis started (unit OHN_B_DIAG_TREE_ENCODING_S0_V1; MODEL_FIT inner CV only, the main pipeline's seed-0 inner
  folds, TREE_V1's chosen setting 15 leaves / min leaf 100 / lr 0.1, 600 iterations): the same learner with POBP as its ordinal
  code, with no native categories, and on the admitted one-hot X. Code additions for it (`cv_curve`, `diag_tree`, an optional
  categorical mask in `make_hgb`, flushed progress lines) do not change any B1 path.
- 05:25-05:28Z ADD_V1 S0 (unit OHN_B_BASES_ADD_V1_S0_V1, 151 CPU-s, 31 main + 155 cross-fitting fits; L design 840 columns,
  S 332; 13 WKHP indicators): main inner-CV NLL by setting S/0.03 1.21806, S/0.15 1.21929, S/0.75 1.22750, **L/0.03 1.21470**,
  L/0.15 1.22043, L/0.75 1.23537; every pipeline (main and 5 outer) chose L with C = 0.03, the strongest-regularization EDGE of
  the grid (trigger R-capacity); lbfgs converged (126-149 iterations at the choice, none at the 5000 cap).
- 05:27-05:37Z diagnosis (unit OHN_B_DIAG_TREE_ENCODING_S0_V1, 213.5 CPU-s, 15 fits): inner-CV NLL minimum of TREE_V1's chosen
  setting under other encodings: TREE_V1 encoding 1.22586 at 20 iterations; **POBP as ordinal code 1.21382 at 30**; no native
  categories 1.21466 at 40; admitted one-hot X 1.22146 at 50 (14x the CPU). All four still rise steadily after their minimum
  (1.280-1.367 at 600). Reading: the native category-subset splits on POBP cost about 0.012 nats; even repaired, the 15-leaf,
  lr 0.1 trees reach their best within 30-50 iterations, so a tree grid with smaller trees, larger leaves and a smaller learning
  rate is the natural repair. These MODEL_FIT-only results are the basis of revision B2 (appended below before its fit).
- 05:39-05:43Z code: registry entries for the B2 learners/configurations (TREE_V2, ADD_V2, B2_STACK, B2_SINGLE; see the B2
  section below, appended before any B2 fit) and a per-learner categorical mask (TREE_V1 unchanged). Selftest V4 FAILED (unit
  OHN_B_SELFTEST_V4: the invented-data test iterated over every registered configuration, including the new B2 ones whose
  invented bases it had not built; KeyError 'TREE_V2'; a test-harness defect, kept in the record). Selftest V5 (unit
  OHN_B_SELFTEST_V5, 12.8 CPU-s; sha256 1537d8c61c37d9f31532bfdd0221de2005ec666739c7621b2729686d50dcf51b): PASS, 23 checks
  (adds: the B2 tree's POBP-ordinal mask, and invented B2 bases through the whole pipeline). B1_STACK / B1_SINGLE seed-0 fits
  run on this code.

### B1 seed 0, configurations (05:43-05:45Z)

- Fits (units OHN_B_FIT_B1_STACK_S0_V1, OHN_B_FIT_B1_SINGLE_S0_V1; 3.2 / 2.9 CPU-s; verify PASS, bitwise on all six roles).
  B1_STACK pool weights a_T = 0.2874 (TREE_V1), a_A = 0.7392 (ADD_V1). B1_SINGLE chose ADD_V1 (tau 1.0070).
- Fully cross-fitted MODEL_FIT (20,000 people; difference vs the cross-fitted B1_SINGLE, household-clustered SE): TREE_V1 raw
  NLL 1.22521 (+0.01036, SE 0.00156); ADD_V1 raw 1.21487; SINGLE 1.21486; **POOL (B1_STACK) 1.21279 (-0.00206, SE 0.00047)**,
  Brier 0.61440 vs 0.61545, accuracy 0.49575 vs 0.49555; linear pool -0.00168 (0.00047); multinomial-logistic stacker
  (diagnostic) -0.00279 (0.00061).
- MODEL_SELECT (scores OHN_B_SCORE_B1_STACK_S0_V1 / ..._SINGLE_S0_V1; ledger entries 50 and the next; anchor H3):

  | config | occ NLL vs U0 (SE) | Brier vs U0 (SE) | acc. loss vs U0 (SE) | NLL vs Ucal | men's guard (SE) | women's guard (SE) | NLL vs T_TREE (SE) | NLL vs TASK_REFIT (SE) | deployed - foldavg NLL (SE) |
  |---|---|---|---|---|---|---|---|---|---|
  | B1_STACK | -0.013102 (0.002377) | -0.005516 (0.001055) | -0.004999 (0.003127) | -0.012222 | -0.000873 (0.00149) | -0.010264 (0.001485) | -0.000975 (0.001556) | +0.000688 (0.001581) | +0.000312 (0.000227) |
  | B1_SINGLE | -0.011811 (0.002561) | -0.004823 (0.001166) | -0.002250 (0.003235) | -0.010930 | -0.000450 (0.001628) | -0.009295 (0.001663) | +0.000316 (0.001758) | +0.001979 (0.00177) | -0.000109 (0.000189) |

- Readings. (1) Stack-value rule, seed 0: MET (cross-fitted d = -0.00206, d + 1.96 SE = -0.00114 < 0; MODEL_SELECT NLL
  -0.01310 < -0.01181). The best equally tuned single learner here is ADD_V1 because TREE_V1 is defective (above). (2) Headroom:
  B1_STACK sits between T_TREE (-0.0121) and TASK_REFIT (-0.0138): -0.0010 vs T_TREE and +0.0007 vs TASK_REFIT, both well inside
  1 SE. (3) The whole gain over U0 is among women (NLL -0.0245, Brier -0.0103; men -0.0019 / -0.0009), and by class it comes
  from service (1) and sales/office (2), while management/business/science/arts (0) and production/transportation (4) are worse
  than U0 (+0.0058 and +0.0069 NLL contribution). (4) Cross-fit sound (decision agreement 0.986; max |dp| 0.050).
- Pre-registered "promising -> seed 1" rule: B1_STACK satisfies it (NLL -0.0131 < -0.0121; Brier -0.0055 < -0.0051; men's
  guard -0.00087 <= -0.00103 + 0.001; cross-fit sound). B1_STACK and B1_SINGLE will be run on seed 1.

### Revision B2 (decided 05:44Z, before any B2 fit)

- **Reason** (MODEL_FIT evidence, plus B1's seed-0 MODEL_SELECT feedback above):
  (a) TREE_V1 is a defective tree learner: its inner-CV NLL peaks at 20 iterations and rises steeply; the diagnosis attributes
  about 0.012 nats to native category-subset splits on the 106 POBP codes (POBP as its ordinal code: 1.2138; one-hot: 1.2215),
  and even repaired, 15-leaf lr-0.1 trees reach their best within 30-50 iterations. The B1 "best equally tuned single learner"
  comparison was therefore against ADD only, and the pool's tree input was weak.
  (b) ADD_V1 chose the strongest-regularization edge (C = 0.03) in all six pipelines (card trigger R-capacity).
  (c) B1_STACK added value over its single learner but did not move past TASK_REFIT on MODEL_SELECT.
- **Change** (layers, folds, nested protocol, scoring and the equal-effort rule unchanged):
  TREE_V2 = HGB on the compact coding with POBP as its ordinal code (COW, MAR native categories; SCHL ordinal), learning rate
  0.05, max_iter 500 (tree count by the inner-CV curve, checkpoints every 10), max_leaf_nodes {4, 8, 15} x min_samples_leaf
  {50, 200}, l2 1.0 (6 settings). ADD_V2 = the same basis, C in {0.003, 0.01, 0.03} x capacity {S, L} (6 settings; keeps the B1
  choice 0.03 and extends below it). B2_STACK = the same log-linear pool; B2_SINGLE = the same single-learner rule (matched
  control). Two configurations per seed (B2_STACK and its control B2_SINGLE); seed 0 first, seed 1 if B2_STACK is promising by
  the B1 rule (NLL, Brier, men's guard vs T_TREE; sound cross-fit). Fits: 2 x 186 base fits per seed plus the layers.
- **Recorded expectation:** TREE_V2 inner-CV NLL about 1.211-1.214 (from 1.2259); ADD_V2 about 1.213-1.2147; with two learners
  of comparable strength the pool gains about 0.002-0.004 over the single; B2_STACK MODEL_SELECT occupation NLL vs U0 about
  -0.014 to -0.017 (at most about 0.003 better than TASK_REFIT), men's guard still near 0 (the women-only pattern is expected to
  persist, since neither learner sees SEX).
- **What would end the line:** B2_STACK not better than B2_SINGLE (the rule) on seed 0 -> stacking adds no value once the tree
  is repaired; B2_STACK not below TASK_REFIT's NLL on MODEL_SELECT -> no headroom beyond the frozen task-only references from
  this family's learners (B3, if any, would then be the R-frozen CALIBRATION layer or the multinomial-logistic stacker; decided
  and appended before its fit).

### B2 seed 0 and B1 / B2 seed 1 (05:46-06:08Z; all fits verify bitwise on all six roles)

- Bases: TREE_V2 S0 (262 CPU-s) main 8 leaves / min leaf 50 / 130 iterations, inner-CV NLL 1.21224 (no edge; outer pipelines
  70-120 iterations); ADD_V2 S0 (48 CPU-s) L / C 0.03 again (0.01: 1.21898, 0.003: 1.23507; with B1's 0.15: 1.22043 the
  optimum is bracketed); TREE_V1 S1 (637 CPU-s) 15 leaves, 20-30 iterations, 1.22579; ADD_V1 S1 (144 CPU-s) L / 0.03, 1.21420;
  TREE_V2 S1 (263 CPU-s) 8 / 50 / 130, 1.21065; ADD_V2 S1 (48 CPU-s) L / 0.03.
- Pool weights (a_T, a_A): B1 S1 0.2776 / 0.7350; B2 S0 0.5925 / 0.4221; B2 S1 0.6335 / 0.3815. Single learner: B1 ADD (tau
  1.0069 on S1); B2 TREE_V2 (tau 1.0034 / 1.0054).
- Fully cross-fitted MODEL_FIT, STACK minus SINGLE NLL (SE): B1 S1 -0.00210 (0.00046); B2 S0 -0.00253 (0.00047); B2 S1 -0.00237
  (0.00046). Multinomial-logistic stacker (diagnostic) vs SINGLE: B2 S0 -0.00231, S1 -0.00188, i.e. WORSE than the pool for B2.
- MODEL_SELECT (anchor H3; ledger entries 59, 60, 64-67):

  | config | seed | occ NLL vs U0 (SE) | Brier vs U0 (SE) | acc. loss vs U0 (SE) | men's guard (SE) | women's guard (SE) | NLL vs TASK_REFIT (SE) | Brier vs TASK_REFIT (SE) | NLL vs T_TREE (SE) |
  |---|---|---|---|---|---|---|---|---|---|
  | B2_STACK | 0 | -0.016730 (0.002461) | -0.007134 (0.001122) | -0.007124 (0.003322) | -0.003494 (0.001575) | -0.010857 (0.001578) | -0.002940 (0.00147) | -0.001491 (0.000673) | -0.004603 (0.001467) |
  | B2_SINGLE | 0 | -0.014695 (0.002663) | -0.006545 (0.001219) | -0.008749 (0.003436) | -0.004069 (0.001709) | -0.009077 (0.001711) | -0.000905 (0.001678) | -0.000901 (0.000767) | -0.002567 (0.001668) |
  | B2_STACK | 1 | -0.013391 (0.00236) | -0.005914 (0.001081) | -0.007374 (0.003274) | -0.002055 (0.001513) | -0.009859 (0.001527) | -0.003064 (0.001477) | -0.001678 (0.000689) | -0.005142 (0.001456) |
  | B2_SINGLE | 1 | -0.011365 (0.002532) | -0.005304 (0.001164) | -0.008749 (0.003421) | -0.002495 (0.00162) | -0.008176 (0.001648) | -0.001038 (0.001687) | -0.001068 (0.000783) | -0.003116 (0.001667) |
  | B1_STACK | 1 | -0.010314 (0.00228) | -0.004445 (0.001022) | -0.005124 (0.003203) | +0.000023 (0.001445) | -0.009013 (0.001438) | +0.000013 (0.001579) | -0.000209 (0.000733) | -0.002065 (0.001551) |
  | B1_SINGLE | 1 | -0.008457 (0.00248) | -0.003575 (0.001129) | -0.002250 (0.00334) | +0.001078 (0.001587) | -0.008333 (0.001602) | +0.001870 (0.001744) | +0.000661 (0.000815) | -0.000209 (0.001726) |

  Deployed minus fold-average NLL on MODEL_SELECT: B2_STACK -0.00023 (0.00035) / +0.00013 (0.00037); decision agreement 0.977 /
  0.976. **Stack-value rule MET on both seeds for B1 and for B2.** B2_STACK beats TASK_REFIT on both seeds by about 2 SE in NLL
  and Brier, and the men's guard improves (-0.0035 / -0.0021 against T_TREE's -0.0010 / +0.0006).

### Revision B3 (decided 06:16Z, before any B3 fit; the last revision)

- **Reason.** (a) The stack question is answered for the two family-B learners (MET, both seeds, both designs). (b) A more
  flexible stacker of the same two learners is not the next lever: the multinomial-logistic stacker is worse than the pool in
  B2's cross-fitted diagnostics on both seeds. (c) The remaining unexplored complementary source is the frozen task-only teacher
  TASK_REFIT (CatBoost depth-4 oblivious trees on the one-hot X with stored task-only leaf corrections: a different tree
  algorithm and encoding). It is in-sample on every MODEL_FIT person, so per the brief and trigger R-frozen it can enter only
  through a layer fitted on CALIBRATION, where it is out of sample. R-frozen's literal condition (no B learner beats TASK_REFIT)
  held for B1 seed 0 but not for B2; B3 asks the different question whether the frozen teacher COMPLEMENTS the B2 pool. This
  choice was made after MODEL_SELECT feedback on B1 and B2 (adaptive; disclosed).
- **Change.** B3_CALTR: p(x) = softmax(c_1 s_B2(x) + c_2 z_TR(x)), s_B2 = a_T z_T2(x) + a_A z_A2(x) exactly as B2_STACK (weights
  on MODEL_FIT inner OOF logits; the B2 bases are reused, no new base fit), z_TR = centered TASK_REFIT occupation scores
  recomputed from features (teacher routing + stored E20 leaf corrections; checked against the frozen bank on MODEL_SELECT to
  1e-12), c >= 0 (2 parameters, L-BFGS-B) fitted on CALIBRATION occupation labels (2,982 people / 1,974 households). One
  capacity setting: B3_CALTR on seed 0 and seed 1 (2 configurations; 10 of 12 used, no further revision). Matched control:
  B2_STACK (the same pool without the layer and the frozen input). Cross-fitting: per OHN1 fold the B2 pool and the CALIBRATION
  weights are refitted from that fold pipeline's own scores; the TASK_REFIT component is in-sample on MODEL_FIT, so B3's
  MODEL_FIT__oof is NOT fully out of sample (recorded in OCC.json). Honest CALIBRATION value: a household 5-fold cross-fitted
  check on CALIBRATION of the layer against the unrefitted pool and against a CALIBRATION temperature-only control.
- **Decision rule (fixed now).** B3 adds value if on BOTH seeds the cross-fitted CALIBRATION check gives CALPOOL minus
  POOL_ONLY NLL < 0 with its upper 95% bound < 0, AND B3_CALTR's MODEL_SELECT occupation NLL vs U0 is below B2_STACK's (point).
  Otherwise the frozen teacher's complement is not established (with 2,982 CALIBRATION people a gain of about 0.001 nats is
  within noise).
- **Recorded expectation.** c_2 about 0.2-0.5; MODEL_SELECT gain over B2_STACK about 0.0005-0.0015 nats; the CALIBRATION check
  probably not significant (SE about 0.001), so the rule will likely say "not established" even if MODEL_SELECT improves.

### B3 results (06:20-06:21Z) and family-B close (06:22Z)

- Fits (units OHN_B_FIT_B3_CALTR_S0_V1 / _S1_V1, 11.1 / 10.9 CPU-s; verify PASS). Recomputed TASK_REFIT scores equal the frozen
  bank on MODEL_SELECT (max |dp| 3.3e-16 / 2.2e-16). CALIBRATION weights (c_1 on the B2 pool, c_2 on TASK_REFIT): seed 0
  0.8794 / 0.1591; seed 1 0.9171 / 0.1230 (fold pipelines: c_2 0.23-0.44, larger because their pools are trained on 80%).
- Household cross-fitted CALIBRATION check, CALPOOL minus comparison NLL (SE): vs the unrefitted B2 pool +0.00044 (0.00068) /
  +0.00004 (0.00059); vs the pool with a CALIBRATION temperature +0.00039 (0.00045) / -0.00005 (0.00031); vs TASK_REFIT alone
  -0.00597 (0.00220) / -0.00753 (0.00230).
- MODEL_SELECT (ledger 73-74): occupation NLL vs U0 -0.017430 (0.002397) / -0.014022 (0.002345); Brier -0.007314 (0.001095) /
  -0.006074 (0.001076); accuracy loss -0.007749 / -0.007249; men's guard -0.002532 (0.001544) / -0.001165 (0.001514); women's
  -0.012204 / -0.011093; vs TASK_REFIT NLL -0.003640 (0.001307) / -0.003695 (0.001398). Against B2_STACK (point): NLL -0.00070 /
  -0.00063, Brier -0.00018 / -0.00016, men's guard +0.00096 / +0.00089 (worse), women's -0.00135 / -0.00123.
- **B3 rule: NOT MET** (the cross-fitted CALIBRATION gain is not below 0 on either seed). The MODEL_SELECT point gain is small,
  comes from women and costs men's margin (TASK_REFIT's women-weighted headroom pattern). B3's MODEL_FIT__oof (NLL 1.1943 /
  1.1946) is optimistic because TASK_REFIT is in-sample there (recorded).
- `python3 -m ohn.fam_b verifyall` (unit OHN_B_VERIFYALL_V1, 12.6 CPU-s; code sha256 a3a605f1...9549): PASS, all 10 releases
  recomputed bitwise from features on all six roles.
- **Family B stops** (10 of 12 configurations; no further revision). Conclusions: (1) the convex log-linear stack adds task value
  beyond the best equally tuned single learner on both seeds in both designs (cross-fitted MODEL_FIT -0.0021 to -0.0025 nats,
  4.6-5.4 SE; MODEL_SELECT -0.0013 to -0.0020); (2) the main source of new headroom is repairing the tree learner (POBP as an
  ordinal code), plus the stack; (3) B2_STACK is the promising witness: fully cross-fitted, no frozen layer, better than
  TASK_REFIT by about 2 SE in NLL and Brier on both seeds, with the largest men's margin of the family's stacks.
- **Seed caveat (checked 06:24Z, label-free):** the DEPLOYED base models of TREE_V2, ADD_V2 and ADD_V1 are bitwise identical on
  seed 0 and seed 1 (MODEL_SELECT logits max |diff| 0.0): the seed-specific inner folds selected the same setting and tree
  count (8 leaves / min leaf 50 / 130 iterations; L / C 0.03), and these fits are deterministic (the HGB random_state is unused
  without early stopping or feature subsampling). ADD_V2 equals ADD_V1 as deployed. So B2's (and B3's) seed-0 and seed-1
  releases differ only through the layer weights fitted on seed-specific inner out-of-fold logits (pool 0.5925 / 0.4221 vs
  0.6335 / 0.3815; tau 1.0034 vs 1.0054) and through the cross-fitting pipelines; the seed-to-seed differences of their
  MODEL_SELECT results mainly reflect the seed-specific references (U0, Ucal, T_TREE, TASK_REFIT, H3 anchor). Only TREE_V1
  differed by seed. The second seed is therefore a replication of the evaluation, not an independent refit of the bases.
