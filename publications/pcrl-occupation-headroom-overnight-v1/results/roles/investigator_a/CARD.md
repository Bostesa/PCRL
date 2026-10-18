# Family A card: stronger categorical boosted learning for the OCCUPATION output (investigator A)

Exploratory development on previously used records (PCRL overnight occupation-headroom campaign); not confirmation. Public-use
U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida (housing-unit residents aged 25-64, civilian employed). The
service predicts an income class (2) and an occupation group (5 official broad civilian groups) from six raw fields (AGEP, WKHP,
COW, SCHL, MAR, POBP; admitted 145-column encoding `X`). Each recipient receives only its own float64 probability vector and
argmax decision. SEX is the dataset's recorded demographic column; readers are separate evaluation classifiers that estimate SEX
from the released probability vectors. This family is TASK-ONLY: no fit in it uses SEX.

Written 2026-10-18 about 04:57-05:05Z, before any real-data fit of this family and before the registration commit (label reads
are refused until then); refined 05:06-05:12Z, still before any fit: the T\* selection uses seed-specific household inner folds
(step 3) because with the seed-independent OHN1 partition this deterministic learner would have produced bitwise-identical
releases on seed 0 and seed 1 (a second seed would then be an equivalent refit, not a replication). Code: `ohn/fam_a.py`
(selftest `OHN_A_SELFTEST_V2` PASS, 25 invented-data checks; V1 PASS on the earlier shared-fold version). Queue units `OHN_A_<config>_S<s>_V<n>` (fits), `..._SCORE_V<n>` (scores),
`..._VERIFY_V<n>`, `OHN_A_SELFTEST_V<n>`. Configurations: `A0_*` initial design, `A1_*` / `A2_*` revisions, `AC_*` controls.
Revisions are appended below as dated sections before their fits; nothing above them is rewritten.

## 1. Measured failure addressed

- The inherited occupation output (JOINT_TREE E20; used by H_LJ and the source successor) spends the occupation headroom of its
  teacher: on MODEL_SELECT NLL excess vs U0 +0.0025 / +0.0029 (seed 0 / 1) and the men's Brier guard +0.0044 / +0.0039 against the
  0.005 cap (0.32 / 0.62 SE of margin) (`results/pcrl_income_headroom_successor_v1/REFERENCE_MS_TABLE.csv`).
- The headroom it spends is T_TREE's: NLL excess vs U0 -0.0121 / -0.0083, Brier -0.0051 / -0.0036, accuracy loss -0.0064 / -0.0061,
  men's guard -0.0010 / +0.0006, women's -0.0093 / -0.0079; TASK_REFIT (task-only leaf corrections on the teacher routing) reaches
  NLL -0.0138 / -0.0103. The occupation gain over U0 is concentrated among women: for men T_TREE is no better than U0.
- Measured weakness of the teacher AS A TASK LEARNER (admitted `stage_a/S<s>/STAGE_A_SEED.json`, label-free record): the
  occupation CatBoost teacher hit its 800-tree cap (800 trees on seed 0, 799 on seed 1) with its 15% household-disjoint internal
  loss still falling at the cap (trees 750 -> 800: 1.21290 -> 1.21227 on seed 0, 1.23808 -> 1.23792 on seed 1), while the income
  teachers stopped at 587 / 529 trees. TASK_REFIT, i.e. more task fitting on the same frozen routing, gains another 0.0017 / 0.0021
  nats on MODEL_SELECT. The teacher's form is restrictive for this task: depth-4 OBLIVIOUS trees, one shared 5-vector of leaf values
  per tree, on one-hot inputs (one level per split for POBP's 106, COW's 9, MAR's 6 admitted levels), learning rate 0.05, <= 800
  trees, tree count from a single 15% split.

## 2. Exact operation and deployed output (initial design)

For configuration c and seed s the deployed occupation output is computed from X alone:

1. **Deterministic compact re-coding of X** (no learned dictionary beyond the admitted one): AGEP, WKHP as admitted (MODEL_FIT
   z-scores); COW, MAR, POBP = index of the active column inside the admitted one-hot block (native categories with 9 / 6 / 106
   levels, the admitted `__OTHER__` level included); SCHL = the ordinal value of its admitted code (`__OTHER__`, the merged rare
   pre-grade-3 codes, = 3). Incomplete or non-binary blocks are rejected. Identical to `ihs.fam_h.compact` (checked on invented data).
2. **Learner g**: scikit-learn 1.8.0 `HistGradientBoostingClassifier` (python3), multinomial log loss (one tree per class per
   iteration, leaf-wise growth), native categorical splits for COW, MAR and POBP, learning rate 0.05, l2_regularization 1.0,
   max_bins 255, no internal early stopping (`early_stopping=False`, so the fit is deterministic), random_state from the seed.
3. **Iteration count T\***: first argmin over t = 1..T_max of the pooled out-of-fold mean occupation NLL over 5 SEED-SPECIFIC
   household folds of the training rows (`ohn.common.household_folds(unit, 5, 'OHN_A_cv_S<seed>')`; label-free, SHA-256 order of
   household ids; the analogue of the teacher's seed-specific internal split). The deployed model is then refitted on all 20,000
   MODEL_FIT people with exactly T\* iterations.
4. **Calibrator**: none in the initial design (the release is the raw learner, compared raw-against-raw with T_TREE). A global
   temperature tau on CALIBRATION occupation labels (bounded 1-D NLL minimisation, tau in [0.5, 2]; p = softmax(log p_g / tau))
   is fitted and assessed by household 5-fold cross-fitting on CALIBRATION as a recorded calibrator DIAGNOSTIC only (counted as
   calibrator fits); it is not applied to the initial releases.
5. **Release**: p_g(x) (n, 5) float64 for every person of all six roles, decision = argmax. Inputs: the six permitted fields only.
   No SEX, no income label, no household id, no other output enters. Labels used: MODEL_FIT occupation (training and internal
   folds); CALIBRATION occupation (calibrator diagnostic only). MODEL_SELECT labels only through `ohn.devrun score` (logged).

**Initial design: two predefined capacity settings** (learning rate 0.05, l2 1.0 for both):

| Config | max_leaf_nodes | min_samples_leaf | T_max | intent |
|---|---|---|---|---|
| `A0_HGB_S` | 8 | 50 | 1500 | small, strongly regularised trees, many iterations |
| `A0_HGB_L` | 31 | 20 | 800 | large trees (scikit-learn's default size), fewer iterations |

**Household cross-fitting check (every configuration).** For each registered crossfit fold k (`ohn.common.crossfit_folds`, 5
folds, tag OHN1_crossfit) the ENTIRE label-using pipeline is rerun on MODEL_FIT minus fold k: T\*_k by the same seed-specific
inner household CV restricted to those rows (5 inner models), a refit on those rows with T\*_k, and any CALIBRATION-fitted
calibrator refitted from that pipeline's own CALIBRATION predictions; it predicts fold k -> `MODEL_FIT__oof`; the mean of the 5
fold pipelines' probability vectors on MODEL_SELECT -> `MODEL_SELECT__foldavg`. 36 physical fits per configuration (deployed
pipeline 5 + 1, cross-fitting 5 x (5 + 1)), recorded as n_model_fits = 6 and n_crossfit_fits = 30. The same run records, as an
honest internal variance diagnostic, the cross-fitted MODEL_FIT value of a committee (each fold pipeline's 5 inner models,
truncated at T\*_k, averaged); a model truncated at t iterations equals a fresh t-iteration fit (checked bitwise in the
selftest). No layer of this family is in-sample on MODEL_FIT for the cross-fitted predictions (no T_TREE / U0 / TASK_REFIT input).

## 3. Why it might create occupation headroom

- It is not iteration-capped where the teacher was: T\* is chosen by 5-fold household CV over a range that extends well past
  the teacher's effective capacity.
- Native categorical splits group POBP / COW / MAR levels by gradient statistics in one split, instead of one admitted level per
  oblivious split.
- One tree per class per iteration with leaf-wise growth can represent class-specific interactions (e.g. SCHL x WKHP x AGEP for
  management versus office work; COW x SCHL for construction / production), where an oblivious tree imposes one split per level
  shared by all classes and all leaves.
- Expected (recorded before any fit): MODEL_SELECT occupation NLL excess vs U0 below T_TREE's by about 0.002-0.005 nats and summed
  Brier by about 0.001-0.002; accuracy changes small. Whether the men's guard improves is open (the teacher's gain is for women).
- Known risk: in the source campaign an HGB income learner ALONE was not better than T_TREE on CALIBRATION (+0.0019 Brier);
  CatBoost's defaults (Bayesian bagging, random split strength, l2 3) regularise well and a single deterministic HGB can have higher
  variance. The fold-average versus deployed comparison and the cross-fitted MODEL_FIT values are the diagnostics for this.

## 4. Closest established methods

Gradient boosted decision trees (Friedman 2001) with histogram binning and native categorical splits (LightGBM, Ke et al. 2017,
as implemented in scikit-learn's HistGradientBoosting); tree count by K-fold cross-validation; nested cross-validation for the
held-out check; temperature scaling (Guo et al. 2017) for the optional calibrator; CatBoost (Prokhorenkova et al. 2018), the
teacher compared against. Nothing new is claimed; the question is only whether this standard learner creates more occupation
margin here.

## 5. Matched task-only and calibration controls

- **Equally exposed task-only references** (MODEL_FIT labels only; frozen bank, no fit): T_TREE (the teacher) and TASK_REFIT; the
  utility references U0 and Ucal.
- **Within-configuration controls**: the fold-average of the 5 cross-fitting pipelines against the deployed refit (variance /
  bagging diagnostic) and the cross-fitted MODEL_FIT utility (honest internal estimate, no in-sample layer).
- **Calibration control** (`AC_TTREE_TEMP`): T_TREE occupation with the identical CALIBRATION temperature calibrator. It is fitted
  if and only if a released configuration of this family applies a CALIBRATION calibrator, so that any gain claimed for the learner
  exceeds what the same calibrator gives the teacher. In the initial design (no calibrator) the comparison is raw learner versus raw
  teacher, so no calibration control is needed.
- **Representation control** (only in a revision whose diagnosis points to the categorical handling): the same learner on the
  admitted 145-column X (no native categories).

## 6. Primary correctness check

1. Deployment: `predict(seed, config, X, names)` recomputes the saved release from features alone, bitwise, on all six roles
   (`verify`), with the model file's SHA-256 checked against OCC.json.
2. Pipeline (invented data, `selftest`): compact coding equals `ihs.fam_h.compact` and rejects incomplete blocks; cross-fitting
   and inner folds are household-separated and the inner partition differs by seed; truncation at t iterations equals a fresh
   t-iteration fit bitwise; T\* and every T\*_k equal an independent brute-force nested CV with fresh fits; every cross-fitted
   prediction equals a fresh refit of that fold's pipeline (training rows exclude the fold) bitwise; the fold-average equals the
   mean of the 5 fresh pipelines; the deployed model equals a fresh refit; committee members equal truncated inner models; fit
   counts; the temperature fit recovers an invented temperature; releases are valid probability vectors; a pickled model
   reproduces the release bitwise; the loader guard refuses SEX and MODEL_SELECT labels.
3. Data access: only MODEL_FIT and CALIBRATION occupation labels are read (through `ohn.common.load_role`, logged); no SEX; no
   MODEL_SELECT label outside `ohn.devrun score`.
4. Real-data plausibility (before scoring): cross-fitted MODEL_FIT NLL in a plausible range (about 1.18-1.26 nats; on their own
   15% split U0 reached 1.227 / 1.250 and T_TREE 1.212 / 1.238) and every pipeline's T\* strictly inside 1..T_max (else flagged).

## 7. Revise / replace / abandon triggers (seed 0 first)

- **Promising** (-> seed 1): seed-0 MODEL_SELECT occupation NLL excess and summed-Brier excess vs U0 both below T_TREE's registered
  values, both SEX-group guards no worse than T_TREE's + 0.001, accuracy loss no worse than T_TREE's + 0.002, and the cross-fitted
  MODEL_FIT values consistent with the ordering.
- **Capacity**: T\* at T_max, or `A0_HGB_L` better than `A0_HGB_S` on cross-fitted NLL by more than one household SE -> revise
  with more capacity (higher T_max / larger trees).
- **Variance / overfitting**: the fold-average better than the deployed refit on MODEL_SELECT NLL by more than about 0.001, or the
  cross-fitted MODEL_FIT gain over the deployed MODEL_SELECT gain large -> revise with variance reduction (deploy the cross-validation
  committee, stronger regularisation, or row-subsampled boosting).
- **Calibration**: cross-fitted CALIBRATION temperature gain above 0.0005 nats and two SEs (or |log tau| > 0.05) -> revise adding
  the temperature calibrator, together with the calibration control `AC_TTREE_TEMP`.
- **Cross-fitted vs deployed mismatch**: cross-fitted MODEL_FIT and deployed MODEL_SELECT disagree on the ordering of the settings
  -> investigate before fitting more.
- **Subgroup burden**: aggregate NLL / Brier better than T_TREE but a guard worse than T_TREE's -> report it; no SEX-aware repair in
  this task-only family; consider a class-level change only if the scorer's per-class decomposition points to it.
- **Replacement**: both initial settings worse than T_TREE on MODEL_SELECT NLL and Brier -> diagnose (representation, variance,
  calibration) and write a replacement section (e.g. row-subsampled XGBoost hist with native categories, or the encoded X) before
  its fit.
- **Abandonment**: after the second revision, if no configuration improves on TASK_REFIT's seed-0 MODEL_SELECT occupation NLL and
  Brier excess (point values), report that stronger categorical boosting creates no extra occupation headroom here and stop.

## 8. Allotment

At most 12 seed-specific fitted configurations for this family (controls included): the initial design and at most two justified
revisions, at most two predefined capacity settings per revision, both seeds. Aim: about 1 CPU-hour including scoring; finish the
task-model work by about 08:00Z.

## Record of fits and revisions (appended as they happen)
