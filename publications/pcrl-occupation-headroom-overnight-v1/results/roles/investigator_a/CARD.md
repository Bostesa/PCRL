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

### Initial design, seed 0 (fits 05:11:36-05:13:48Z; scores 05:25Z and 05:42Z)

- Units `OHN_A_A0_HGB_S_S0_V1` (131.6 CPU-s) and `OHN_A_A0_HGB_L_S0_V1` (127.2 CPU-s): 36 HGB fits each (6 deployed pipeline + 30
  cross-fitting) and 7 temperature diagnostic fits each. T\* = 71 / 40 (cross-fitting pipelines 58-64 / 37-41); no flag.
- Internal (no MODEL_SELECT): cross-fitted MODEL_FIT NLL 1.22342 / 1.22649, Brier 0.61822 / 0.61907, accuracy 0.49365 / 0.49245;
  pooled inner-CV minimum 1.22166 / 1.22470 followed by a sharp rise (S 1.2347 at 300 iterations; L 1.2762 at 300); committee of
  the 5 inner models minus refit, cross-fitted NLL -0.00130 (SE 0.00030) / -0.00306 (SE 0.00049); CALIBRATION temperature 0.956 /
  0.964 with cross-fitted NLL change -0.00029 (SE 0.00059) / -0.00015 (SE 0.00049): no calibration problem.
- MODEL_SELECT (anchor H3), S / L: occupation NLL excess vs U0 -0.00349 (SE 0.00251) / +0.00010 (SE 0.00278); Brier -0.00220
  (0.00108) / -0.00096 (0.00118); accuracy loss -0.00212 (0.00322) / +0.00100 (0.00325); men's guard +0.00086 (0.00155) / +0.00251
  (0.00170); women's -0.00533 (0.00151) / -0.00452 (0.00167). Against T_TREE: NLL +0.00864 (0.00196) / +0.01223 (0.00220), Brier
  +0.00292 (0.00086) / +0.00416 (0.00099); against TASK_REFIT NLL +0.01030 / +0.01389. The deficit is spread over all five classes
  and both SEX groups. Deployed minus fold-average NLL -0.00061 (0.00054) / +0.00327 (0.00070). L fails the reserve on the men's guard.
- Trigger met: **replacement** (both initial settings worse than T_TREE on MODEL_SELECT NLL and Brier); also the variance trigger
  for L. Not capacity (T\* far from T_max; smaller trees better) and not calibration.

### Diagnostic unit, seed 0 (MODEL_FIT only; no release, no MODEL_SELECT; decided ~05:17-05:27Z)

Pooled 5-fold inner-CV occupation NLL on MODEL_FIT, the same seed-0 inner folds as A0 (so directly comparable with A0_HGB_S's
1.22166); 5 HGB fits per setting (25 fits; units `OHN_A_DIAG_D1D4_S0_V1` 98 CPU-s, `OHN_A_DIAG_D5_S0_V1` 548 CPU-s; settings in
`ohn/fam_a.py` DIAG_SETTINGS, fixed before the A0 MODEL_SELECT score was read except D5's iteration cap, raised to 2000 after it):

| Setting | change from A0_HGB_S | T\* | OOF NLL min | Brier | accuracy |
|---|---|---|---|---|---|
| A0_HGB_S | (lr 0.05, 8 leaves, min leaf 50, l2 1, native COW/MAR/POBP) | 71 | 1.22166 | | 0.4929 |
| D1 | lr 0.025, min leaf 100, l2 3 | 155 | 1.22167 | 0.61782 | 0.4931 |
| D2 | D1 with 4 leaves | 273 | 1.22183 | 0.61833 | 0.4921 |
| D3 | D1 with 16 leaves, min leaf 200 | 96 | 1.22349 | 0.61844 | 0.4929 |
| **D4** | **D1 with POBP as its ordinal place-of-birth code** | 279 | **1.21073** | 0.61351 | 0.4962 |
| D5 | D1 on the admitted one-hot X (no native categories; SCHL one-hot) | 397 | 1.22219 | 0.61799 | 0.4938 |

Diagnosis: the failure is the native-category POBP split (106 admitted levels, many rare). scikit-learn's categorical split
takes, in every node, the best binary partition of the levels ordered by their gradient statistics; on 20,000 people this fits
noise in level means, and neither leaf size, L2, learning rate nor tree size touches it (D1-D3). Coding POBP as its numeric
place-of-birth code (US states 001-056, territories 072/078, Europe 100-164, Asia 207-247, the Americas 301-373, Africa 440/449,
`__OTHER__` = 999) lets trees split only contiguous code ranges, i.e. pool related countries; the OOF NLL falls by 0.0109 and the
curve is flat after its minimum. One level per split (D5) does not help: it cannot pool related countries (and loses SCHL's order).

### Revision 1 (replacement of the POBP coding): `A1_PO_8`, `A1_PO_16` (decided 05:43Z, before their fits)

- One change from the initial design: the re-coding `compact_po` (deterministic from X): AGEP, WKHP as admitted; COW, MAR native
  categories; SCHL ordinal; **POBP = its numeric place-of-birth code** (`__OTHER__` = 999), at the diagnostic's neutral
  regularisation (learning rate 0.025, min leaf 100, l2 3). Everything else as the initial design: seed-specific inner CV for T\*,
  refit on all MODEL_FIT, full nested household cross-fitting, no calibrator (temperature diagnostic only), no SEX.
- Two predefined capacity settings: `A1_PO_8` = 8 leaves, T_max 1200 (the diagnostic's D4); `A1_PO_16` = 16 leaves, T_max 1000
  (more interaction capacity now that the POBP overfitting is removed).
- Matched controls: unchanged (T_TREE, TASK_REFIT, U0, Ucal; the fold-average and the cross-fitted committee diagnostic; no
  calibrator, so no calibration control).
- Recorded expectation: A1_PO_8 inner OOF about 1.211; MODEL_SELECT NLL about 0.008-0.011 better than A0_HGB_S, i.e. about T_TREE's
  level (-0.012) or slightly better; Brier about -0.004 to -0.006; men's guard unknown.
- Selection caveat: the coding was chosen with the seed-0 inner CV on all of MODEL_FIT (5 diagnostic settings), so the pipeline's
  cross-fitted check does not cover that choice (mild optimism); MODEL_SELECT remains the arbiter, and seed 1 uses its own folds.
- Next: promising (MODEL_SELECT NLL and Brier below T_TREE's, both guards no worse than T_TREE's + 0.001) -> seed 1 for the better
  setting(s); below T_TREE but not TASK_REFIT, or a committee gain above about 0.001 in the cross-fitted diagnostic -> Revision 2
  (variance reduction: committee deployment); no gain over A0 -> stop the representation line and report.

### Revision 1 results, both seeds (fits 05:43-06:00Z; scores 05:46-06:01Z; commit cb89a8b51 before the first fit)

Units `OHN_A_A1_PO_8_S0_V1` 111.6, `OHN_A_A1_PO_16_S0_V1` 125.9, `OHN_A_A1_PO_8_S1_V1` 110.8, `OHN_A_A1_PO_16_S1_V1` 128.4 CPU-s; 36
HGB fits and 7 temperature diagnostic fits each; no flag. A1_PO_8's pooled seed-0 inner-CV minimum (1.21073) equals diagnostic D4
exactly (same setting and folds). MODEL_SELECT, anchor H3, occupation against U0 (SE), and paired against the task-only references:

| Config / seed | T\* | OOF NLL (MODEL_FIT, cross-fitted) | NLL excess | Brier excess | accuracy loss | men's guard | women's guard | NLL vs T_TREE | Brier vs T_TREE | NLL vs TASK_REFIT | Brier vs TASK_REFIT |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1_PO_8 / 0 | 279 | 1.21253 | -0.01443 (0.00265) | -0.00628 (0.00121) | -0.00962 (0.00341) | -0.00344 (0.00170) | -0.00917 (0.00171) | -0.00231 (0.00166) | -0.00115 (0.00074) | -0.00064 (0.00166) | -0.00063 (0.00075) |
| A1_PO_16 / 0 | 168 | 1.21231 | -0.01614 (0.00271) | -0.00673 (0.00124) | -0.00987 (0.00343) | -0.00385 (0.00174) | -0.00968 (0.00174) | -0.00402 (0.00171) | -0.00161 (0.00078) | -0.00235 (0.00171) | -0.00109 (0.00079) |
| A1_PO_8 / 1 | 240 | 1.21249 | -0.01118 (0.00249) | -0.00502 (0.00115) | -0.00975 (0.00342) | -0.00195 (0.00160) | -0.00816 (0.00162) | -0.00293 (0.00161) | -0.00142 (0.00074) | -0.00086 (0.00164) | -0.00078 (0.00076) |
| A1_PO_16 / 1 | 154 | 1.21215 | -0.01280 (0.00257) | -0.00546 (0.00118) | -0.00975 (0.00353) | -0.00229 (0.00165) | -0.00870 (0.00166) | -0.00456 (0.00168) | -0.00186 (0.00078) | -0.00248 (0.00170) | -0.00123 (0.00080) |
| T_TREE / 0, 1 (registered) | - | in-sample | -0.0121 / -0.0082 | -0.0051 / -0.0036 | -0.0064 / -0.0061 | -0.0010 / +0.0006 | -0.0093 / -0.0079 | | | | |
| TASK_REFIT / 0, 1 | - | in-sample | -0.0138 / -0.0103 | -0.0056 / -0.0042 | -0.0044 / -0.0035 | -0.0012 / -0.0004 | -0.0102 / -0.0082 | | | | |

- Both settings meet the pre-declared "promising" trigger on both seeds; A1_PO_16 is ahead of A1_PO_8 on both seeds. The men's
  guard (the binding condition of every task-only reference) improves on T_TREE's registered value by 0.0028 / 0.0029 (A1_PO_16,
  seed 0 / 1) and 0.0024 / 0.0025 (A1_PO_8), point differences of the unpaired guard values; the women's guard is at T_TREE's level. All 12 utility clauses and both guards pass on both seeds; every per-seed reserve condition is met (binding:
  H3's income accuracy, ratio 3.51 / 2.86; occupation-side minimum ratio 4.65-4.88 / 4.09-4.30).
- Cross-fitted versus deployed: deployed minus fold-average NLL +0.00021 (0.00050) / -0.00011 (0.00058) on seed 0 and -0.00000
  (0.00042) / +0.00020 (0.00049) on seed 1: no variance gap on MODEL_SELECT. Cross-fitted committee minus refit on MODEL_FIT: -0.00097
  / -0.00058 (seed 0), -0.00086 / -0.00045 (seed 1), SEs about 0.0003. CALIBRATION temperatures 0.950-0.960 with cross-fitted NLL
  changes -0.00015 to -0.00038 (SEs 0.00055-0.00068): no calibration trigger.

### Further MODEL_FIT-only diagnostics on the revision-1 coding (no release, no MODEL_SELECT; seed-0 inner folds)

Units `OHN_A_DIAG_D6D8_S0_V1` (fixed ~05:57Z after the A1 seed-0 scores) and `OHN_A_DIAG_D9D10_S0_V1` (fixed ~06:03Z after the A1
seed-1 scores); 5 HGB fits per setting. Reference pooled minima: A1_PO_8 (= D4) 1.21073, A1_PO_16 1.21141.

| Setting | change from A1_PO_16 | T\* | OOF NLL min |
|---|---|---|---|
| D6 | 32 leaves (lr 0.025) | 103 | 1.21512 |
| D7 | learning rate 0.01 | 440 | 1.21124 |
| D8 | min leaf 200, l2 10 | 168 | 1.21194 |
| D9 | COW as its numeric code (not native) | 176 | 1.21312 |
| D10 | COW and MAR as numeric codes | 169 | 1.21233 |

### Decision (06:05Z): no Revision 2; the line is exhausted

- The only lever that moved the held-out quality was the POBP coding (0.011 nats internally; 0.008-0.012 nats on MODEL_SELECT
  against A0). On the new coding, capacity (8 = 16 leaves; 32 worse), learning rate, leaf regularisation and the remaining category
  codings are flat or worse (D6-D10, within about +-0.0005 or worse), the committee gain of the better setting (A1_PO_16) is 0.0005-
  0.0006 nats internally (below the card's 0.001 trigger) and invisible on MODEL_SELECT, and no calibration trigger fired. A
  revision with no diagnostic support would only spend configurations on differences below the MODEL_SELECT resolution (paired NLL
  SE about 0.0017), so none is fitted. Configurations used: 6 of 12 (A0_HGB_S S0, A0_HGB_L S0, A1_PO_8 S0/S1, A1_PO_16 S0/S1).
- Promising task-only witnesses (both seeds scored before 07:15Z): A1_PO_16 first, A1_PO_8 second (same coding, smaller trees).
- Deployment: `python3 -m ohn.fam_a verify` recomputes all six family-A releases (A0_HGB_S S0, A0_HGB_L S0, A1_PO_8 S0/S1, A1_PO_16
  S0/S1) bitwise on all six roles from features alone, with each model file's SHA-256 checked against OCC.json (units
  `OHN_A_<config>_S<s>_VERIFY_V1`, 05:43-06:07Z). Final code `ohn/fam_a.py` passes selftest `OHN_A_SELFTEST_V4` (30 invented-data
  checks; the changes since V3 are registry entries and diagnostic-only codings). Family-A queue use: 26 units, 0 failed,
  1,583.5 CPU-s (0.44 CPU-h; the one-hot diagnostic D5 alone 548 CPU-s).
