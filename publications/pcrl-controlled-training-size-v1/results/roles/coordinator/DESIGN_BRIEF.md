# Coordinator design brief: controlled training-size test with one conditional protection experiment (PCRL)

Campaign `pcrl_controlled_training_size_v1`; code package `tsz/`; results `results/pcrl_controlled_training_size_v1/`; private tree
`.private/pcrl_controlled_training_size_v1/` (never committed). Branch `research/pcrl-controlled-training-size-v1` (local; created
from the correction-transfer research tip 507024649 in the worktree directory `.claude/worktrees/pcrl-correction-transfer-stability-v1`,
which this session is bound to; the previous branch is unchanged). Campaign start 2026-10-18T23:45:15Z (date -u).

This brief is the coordinator's design proposal. The statistical reviewer owns the rules (RULES.md / RULES.json) and may change any
threshold, contrast, trigger detail, strength set or selection rule BEFORE any new real-label computation, with a written reason.

## 0. Research setting

Already admitted public-use U.S. Census Bureau ACS 2019 1-Year PUMS Florida person records (development roles of the protected-tree
replication). Income (2 classes) and occupation (5 official broad civilian groups) are the useful tasks, predicted from six raw fields
(AGEP, WKHP, COW, SCHL, MAR, POBP; admitted 145-column encoding). Each recipient receives only its own float64 probability vector and
argmax decision. SEX is the recorded demographic column: an authorized training label on MODEL_FIT and an evaluation label on
MODEL_SELECT for development readers (separate evaluation classifiers estimating SEX from released probability vectors). SEX is never
a runtime input or output. "Recovery" always means recovery by the declared development readers.

## 1. Sources (pinned; preserved verdicts)

- Correction-transfer campaign: publication e32a185b4 (`publications/pcrl-correction-transfer-stability-v1`), research tip
  507024649. Verdict preserved: a replicated, independently replayed diagnosis plus a negative method result (NO_RECIPE, rung 0).
  Its TRAINING_SIZE_BET / TRAINING_SIZE_CURVE are ADAPTIVE diagnostics (P70, P70A).
- Occupation-headroom campaign: publication 1435c395d (research ca8725e72). Verdict preserved: NO_QUALIFYING_CONFIGURATION.
  Source of the A1_PO_16 recipe (ohn.fam_a), the P2 correction recipe (ohn.protect), the reserve rule (ohn.rules, RULES.json
  sha e18f1c68...), the development reader grid (ohn.devreaders).
- H3 income: publication 89fd2f6e6 (income-headroom successor), admitted only as its manifests require, through the verified
  mirror `.private/pcrl_income_headroom_successor_v1/` (ohn.common.admitted re-checks every file against SOURCE_ADMISSION.json).
- Frozen references: U0, Ucal (utility references), T_TREE, TASK_REFIT, H_LJ, NEURAL_RAW, etc. from the admitted frozen bank
  (ohn.common.frozen_bank, hash-checked against FROZEN_BANK_MANIFEST).

## 2. Roles and exposure

| Role | People / households | Use here |
|---|---|---|
| MODEL_FIT | 20,000 / 13,337 | all fitting: size subsets; full pool; nested cross-fitting (occupation, income for the T_TREE support repair, SEX as a training label and for out-of-fold subgroup costs in the strength rule) |
| CALIBRATION | 2,982 / 1,974 (no SEX admitted) | A1_PO_16 diagnostic temperature (recorded, never applied, as in the source); nothing else |
| MODEL_SELECT | 8,001 / 5,306 | common development evaluation: task-stage benchmark and trigger; fresh development readers on new household splits. Already used adaptively by every earlier campaign; never confirmation |
| ATTACK_FIT, ATTACK_VALIDATION, DEV_REPORT | - | labels closed throughout (features only for deployment parity) |
| sealed assessment, capped-out records | - | refused by every loader |

## 3. Controlled training-size design

- **Orderings.** Four label-free household orderings O0..O3 of MODEL_FIT: households sorted by SHA-256('TSZ1_order_O<o>|<household
  id>'). Nested subsets per ordering: S50(o) = first ceil(0.5 x 13,337) = 6,669 households; S70(o) = first ceil(0.7 x 13,337) =
  9,336 households; S100 = all 13,337 households (identical for every ordering). Whole households always. Person counts and pairwise
  overlaps recorded label-free before any fit.
- **Versions.** Two source versions v = 0, 1, exactly as in the source: the recipe's seed-specific parts (A1_PO_16 inner-CV fold tag
  OHN_A_cv_S<v>; T_TREE CatBoost seed and internal split of seed v; Stage B streams of seed v), H3 income S<v>, references U0_S<v> /
  Ucal_S<v>, H_LJ floor of seed v. The two versions share every training row; they differ only by those procedure seeds, so they are
  procedure repeats, never independent replications. Fit cells: (o, f, v) for f in {50, 70}, plus (100, v), which is one fit per
  version, aliased to all four orderings (one fit, one result).
- **Primary recipe A1_PO_16** (fixed from ohn.fam_a.CONFIGS['A1_PO_16'] and the source fit code): HistGradientBoosting multinomial,
  lr 0.025, 16 leaves, min leaf 100, l2 3, max_iter 1000, early_stopping off; compact_po coding (AGEP, WKHP numeric; COW, MAR native
  categories; SCHL ordinal admitted code; POBP its numeric place-of-birth code, __OTHER__ = 999); T* = first argmin of the pooled
  inner-CV NLL over household_folds(all MODEL_FIT units, 5, 'OHN_A_cv_S<v>') restricted to the subset (the source's convention for
  sub-pipelines, run_pipeline); refit on the subset with T* iterations; no calibrator applied; the source's diagnostic CALIBRATION
  temperature recorded only. At S100 this is the source fit procedure exactly, so W100_v must reproduce the source A1_PO_16 S<v>
  release bitwise (checked; any difference is reported, never tuned away).
- **Primary preprocessing:** FIT-only dictionary (cts.trackf.fit_dictionary: a COW / SCHL / MAR / POBP level with fewer than 10
  people in the training subset moves to its block's __OTHER__ column), applied to every role. At S100 it equals the admitted
  dictionary. Recorded per fit: collapsed levels per block; MODEL_SELECT share with a collapsed level per block; MODEL_SELECT share
  with a level unseen in the training subset ("unknown").
- **Encoding sensitivity (bounded, label-free preprocessing):** A1_PO_16 at S50(o), v = 0, o = O0..O3, with the admitted
  full-MODEL_FIT feature dictionary instead of the FIT-only one (same rows, same folds). Four fits.
- **Equally exposed control:** the replication's task-only reference pipeline refitted on exactly the same households: T_TREE
  (acs_replication Stage A income and occupation CatBoost teachers, depth 4, lr 0.05, l2 3, <= 800 trees, seed_for('ACSR1catboost',
  v, task, 4), 15% household internal split with the two-task class-support repair of seed v, refit on the subset) followed by
  TASK_REFIT (Stage B arm T_REFIT, lambda 0, fixed 20 x 40 x 256 updates, AdamW lr 0.01 / wd 0.01, on the subset). Primary control
  output: TASK_REFIT occupation; T_TREE occupation reported as its first stage. Neither uses CALIBRATION labels in its source recipe.
  At S100 the control must reproduce the admitted T_TREE / TASK_REFIT of seed v (checked; reported).
- **Class support:** fixed before drawing: no support repair is added to A1_PO_16 (its fit asserts all five occupation classes in
  every training set; a failure stops the unit, is recorded, and the cell is reported missing; no redraw); T_TREE keeps its own
  internal-split repair. Minimum class counts per subset recorded.
- **Effort:** capacity, max_iter, learning rates and the Stage B update count are fixed; T* and tree counts are chosen inside each
  subset by its own household-internal procedure and reported. No size gets extra tuning or extra validation labels.

Fit count (planned): A1_PO_16 2 x (8 + 1) = 18; controls 18 (each T_TREE + TASK_REFIT); encoding 4.

## 4. Evaluation and analysis (MODEL_SELECT, common people)

Every occupation output is paired with the frozen H3 income of its version and scored by the unchanged source definitions
(ohn.evaluate.complete_eval via cts.evaluate.eval_release): 12 clauses per version (income and occupation x U0 and Ucal x accuracy loss
<= 0.010, NLL excess <= 0.010, summed Brier excess <= 0.005; household-clustered paired SEs) + men's and women's occupation summed-Brier
excess vs U0 (<= 0.005); reserve = ohn.rules.reserve (k = 2, floor = H_LJ's MODEL_SELECT SE of the same condition and seed, rescaled).
Reported per fit: occupation accuracy, NLL, Brier; guards; point status; margin in metric units and in SE units; reserve status
(occupation side; income separately). Paired contrasts on the common MODEL_SELECT people (reference-free): 50->70, 70->100, 50->100 per
ordering and version; mean over orderings; between-ordering spread; A1_PO_16 minus control at every cell; source reproduction.

## 5. Conditional trigger (the statistical reviewer defines it; proposal)

Fires iff, for BOTH versions v: W100_v meets all 6 occupation point caps and both subgroup guards against the original references;
meets the occupation-side reserve (all 8 occupation-side conditions of version v, ratio >= 2 with the source H_LJ floor) on both
versions; and its deployed implementation reproduces the stored release bitwise from the six raw fields with every refusal. Smaller
subsets are never substituted; nothing is relaxed after outcomes. Not a declaration that the H3-anchored service qualifies.

## 6. Conditional correction (only if triggered)

Per version v on the full pool (all MODEL_FIT): W = W100_v; correction base = W100_v's 5-fold household cross-fitting out-of-fold
logits on MODEL_FIT (OHN1_crossfit folds, each fold pipeline with its own inner CV, as the source); routing = the T_TREE occupation
teacher refitted on all MODEL_FIT with seed v (the control pipeline's S100 teacher); arms of the source P2 recipe (ohn.protect.
train_correction, guard 'none', one full pipeline per arm): CONT (zero-critic continuation), LOC (occupation-only critic; PRIMARY,
declared before fitting), PAIR (coalition-aware critic reading the frozen H3 income wire S<v>; secondary). Identical routing,
epochs, task loss and effort except the critic term. Release r_s = softmax(log p_W + s x Delta_arm).

Strength set proposal: {0, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0}.

Selection information: nested 10-fold household cross-fitting of MODEL_FIT (outer folds household_folds(MODEL_FIT units, 10,
'TSZ1_ncv10')). For outer fold k (training = the other 90%, about 18,000 people): FIT-only dictionary, A1_PO_16 witness (inner CV
restricted to the 90%), its own 5-fold household cross-fitting base, T_TREE routing (seed v), LOC and PAIR corrections, all refitted
on the 90%; applied to fold k with fold k's witness and routing; per-person incremental costs against s = 0 at every strength; pooled
over the 10 folds (all 20,000 people). Every submodel's training size recorded. Disclosure: pooled cross-fitted costs target
90%-size submodels and do not guarantee the deployed 100% model's costs. MODEL_SELECT never enters strength selection.

Selection rule: the statistical reviewer fixes a single documented rule from fitting information only (proposal: the largest s whose
every smaller-or-equal strength has upper bound estimate + 2 SE <= a fixed allowance on men's Brier, women's Brier, summed Brier,
NLL and accuracy loss). Identity is always admissible and is stored when selected. CONT is evaluated at the selected strength of the
arm it matches (no own selection).

## 7. Readers (frozen before fitting)

Fresh development readers (no cache reuse): MODEL_SELECT households ordered by SHA-256('TSZ1_devgrid_<R>|<household>') for R in
{R1, R2}, dealt round-robin into 10 parts (0-4 FIT 50%, 5-6 VALIDATION 20%, 7-9 SCORE 30%); reader seeds k0, k1 from a new seed base
(effective seeds checked per recipe); the source 16-recipe bank, selection on VALIDATION (max oriented AUC), scored on SCORE,
household bootstrap of SCORE (B = 400, paired across releases in a cell). Views: income-only, occupation-only and pair. Cells per
release: 2 versions x 2 repetitions x 2 reader seeds = 8. Releases: W100_v, the selected LOC, PAIR (identity where selected),
the matched CONT, and the inherited strong references (NEURAL_RAW, H_LJ, B2TWIN, RETIRED_B3CTL, SUCC, TWJ).

## 8. Packages and verification

Packages (deployable from the six raw fields): W100_v, the S100 TASK_REFIT control, a representative smaller fit (A1_PO_16 S50(O0),
v = 0), and, if the stage runs, the selected correction and matched continuation releases. Independent verifier: separate numerical
code for pins, nesting, membership and exposure, identities and aliases, metrics and contrasts, trigger and reserve, selection,
recovery at the selected strengths, claim labels and raw-field deployment.

## 9. Resources

Eight research hours from 23:45:15Z (to 07:45:15Z) plus 30 minutes of closeout; 14 CPU-hours; at most two heavy jobs, one thread each,
through one shared queue (tsz.jobqueue); no paid or cloud compute; at most six live agents including the coordinator.
