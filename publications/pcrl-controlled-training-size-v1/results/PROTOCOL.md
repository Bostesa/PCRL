# Protocol: controlled training-size test with one conditional protection experiment (PCRL)

Registered before any new real-label computation or fit of this campaign. EXPLORATORY DEVELOPMENT on previously used public-use
ACS records; nothing here is confirmation. Thresholds, labels and the exact rules are in `roles/statistical_reviewer/RULES.md` and
`RULES.json` (part of this registration); the label-free subset and split manifests are `SUBSETS.json` and `READER_SPLITS.json`;
exposure is in `EXPOSURE_MAP.md`; access in `ACCESS_POLICY.json`; source pins in `SOURCE_ADMISSION.json`; the reader code lock in
`DEVGRID_CODE_LOCK.json`. Campaign start: 2026-10-18T23:45:15Z.

## 1. Setting

Already admitted public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida (housing-unit residents aged 25-64,
civilian employed), in the development roles of the protected-tree replication. Income (2 classes) and occupation (5 official broad
civilian groups) are the useful tasks, predicted from six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP). Each recipient receives only
its own probability vector and argmax decision; that pair is the complete release. Separate evaluation classifiers ("development
readers") estimate the recorded SEX column from the released probability vectors; "recovery" always means recovery by these declared
readers. SEX is a training label on MODEL_FIT and an evaluation label on MODEL_SELECT; it is never a runtime input or output.

## 2. Questions

- **Primary.** Does a fixed occupation-learning recipe (A1_PO_16) retain useful task and subgroup margin when trained at 50%, 70% and
  100% of the permitted training pool (MODEL_FIT), under common evaluation (MODEL_SELECT) and equally trained controls?
- **Conditional.** If the full-data model retains that margin (the frozen trigger, section 7), does one fixed correction recipe use it
  to lower declared-reader recovery while keeping usefulness?

No correction-strength controller search and no broad learner sweep is run.

## 3. Sources and preserved verdicts

Pinned (SOURCE_ADMISSION.json): the correction-transfer publication e32a185b4 (research tip 507024649; verdict: a replicated,
independently replayed diagnosis plus a negative method result, NO_RECIPE, rung 0); through its admitted chain, the occupation-headroom
publication 1435c395d (verdict NO_QUALIFYING_CONFIGURATION, never reinterpreted) and the H3 income publication 89fd2f6e6 (admitted only
as its manifests require). Every earlier verdict is preserved. The correction-transfer TRAINING_SIZE_BET / TRAINING_SIZE_CURVE are
adaptive diagnostics; they motivate this study and are not re-labelled by it.

## 4. Roles

MODEL_FIT (20,000 people, 13,337 households): all fitting. CALIBRATION (2,982 people; no SEX admitted): only the A1_PO_16 diagnostic
temperature (recorded, never applied, as in the source). MODEL_SELECT (8,001 people, 5,306 households): the common development
evaluation role (benchmark, reserve, size contrasts, trigger, fresh development readers); it was used adaptively by earlier campaigns
and neither these results nor its new reader splits are confirmation. ATTACK_FIT, ATTACK_VALIDATION, DEV_REPORT labels, the sealed
assessment and capped-out records stay closed throughout. No new people, population or microdata download.

## 5. Controlled training-size design

- **Subsets.** Four label-free household orderings O0..O3 of MODEL_FIT (SHA-256('TSZ1_order_<o>|<household>')); nested subsets per
  ordering: S50 (6,669 households, 9,934-10,046 people), S70 (9,336 households, 13,954-14,040 people), S100 (all 13,337 households,
  20,000 people; identical for every ordering). Whole households always; pairwise overlaps in SUBSETS.json.
- **Versions.** Two source versions v = 0, 1: the recipes' seed-specific parts (A1_PO_16 inner folds 'OHN_A_cv_S<v>'; T_TREE CatBoost
  seed and internal split of seed v; Stage B streams of seed v), H3 income S<v>, references U0_S<v> / Ucal_S<v>, the H_LJ floor of seed
  v. The two versions share every training row: they are procedure repeats, never independent replications, and their difference is
  reported separately from subset-to-subset variation.
- **Fits.** A1_PO_16 at (S50_O<o>, v) and (S70_O<o>, v) for every ordering and version, and (S100, v): 18 fits; S100 is one fit per
  version, aliased to all four orderings (one fit, one result).
- **Recipe A1_PO_16** (ohn.fam_a.CONFIGS['A1_PO_16'], fixed from source records): HistGradientBoosting multinomial, lr 0.025, 16 leaves,
  min leaf 100, l2 3, max_iter 1000, early stopping off; compact_po coding (AGEP, WKHP numeric; COW, MAR native categories; SCHL its
  ordinal admitted code; POBP its numeric Census place-of-birth code, __OTHER__ = 999); T* = first argmin of the pooled inner-CV NLL
  over household_folds(all MODEL_FIT units, 5, 'OHN_A_cv_S<v>') restricted to the subset (the source's sub-pipeline convention);
  refit on the subset with T* iterations; no calibrator applied; diagnostic CALIBRATION temperature recorded only. The ordinal POBP
  code is the source's implementation detail: the Census numeric codes are not claimed to be a universal meaningful ordering and the
  coding is not claimed as algorithmic novelty.
- **Preprocessing.** Primary: FIT-only dictionary (a COW / SCHL / MAR / POBP level with fewer than 10 people in the training subset
  moves to its block's __OTHER__ column), applied to every role; at S100 it equals the admitted dictionary. Recorded per fit: collapsed
  levels, and the MODEL_SELECT shares with a collapsed or an unseen ("unknown") level.
- **Encoding sensitivity** (one bounded, label-free preprocessing sensitivity at fixed training rows): A1_PO_16 at S50_O<o>, v = 0,
  o = O0..O3, with the admitted full-MODEL_FIT feature dictionary instead (same rows, same folds). Four fits. No target-label leakage
  and no feature search.
- **Equally exposed control.** The replication's task-only reference pipeline refitted on exactly the same households and version:
  T_TREE (Stage A income and occupation CatBoost teachers on the subset's FIT-only dictionary; source recipe) followed by TASK_REFIT
  (Stage B arm T_REFIT, task loss only, fixed update count). Primary control output: TASK_REFIT occupation; T_TREE occupation is
  reported as its first stage. No CALIBRATION labels enter either (none in their source recipe). The Stage B loop assumes at least
  40 x 256 = 10,240 training people per epoch; for the S50 subsets (fewer people) the same epoch permutation is extended cyclically
  so that every update has 256 rows (PROTOCOL_DEVIATIONS.md item 3; the unchanged loop runs at 10,240 or more people).
- **Class support** (fixed before drawing): no repair added to A1_PO_16 (a training set lacking an occupation class stops the unit,
  is recorded and the cell is reported missing; no redraw); T_TREE keeps its own internal-split repair. Minimum class counts recorded.
- **Effort.** Capacity, max_iter, learning rates and the Stage B update count are fixed; T* and tree counts are chosen inside each
  subset by its own household-internal procedure and reported. No size receives extra tuning or extra validation labels. The Stage B
  update count is fixed in updates, so smaller subsets are passed over more often (recorded).

## 6. Evaluation and analysis

Every occupation output is paired with the frozen H3 income of its version and scored on the same MODEL_SELECT people by the unchanged
source definitions (ohn.evaluate.complete_eval): 12 clauses per version (income and occupation x U0 and Ucal x accuracy loss <= 0.010,
NLL excess <= 0.010 nats, summed Brier excess <= 0.005; household-clustered paired SEs) and the men's and women's occupation summed-Brier
excess vs U0 (<= 0.005); two versions give 24 clauses and 4 guards. The two-SE reserve (ohn.rules.reserve: margin >= 2 x max(own SE,
H_LJ floor SE)) is a development heuristic, not a new cap or a population guarantee; it is reported separately from the original
limits. Reported per fit: occupation accuracy, NLL, Brier; men's and women's Brier excess; point-cap status; margins in metric and SE
units; reserve status; the H3 income conditions separately. Paired 50->70, 70->100 and 50->100 contrasts on the common MODEL_SELECT
households per ordering and version; subset-to-subset variation; A1_PO_16 minus the control at every cell; reproduction of the source
full-data package (W100_v vs the source A1_PO_16 S<v>, bitwise); the encoding sensitivity. Interpretation labels, estimators and the
four interpretation rules are fixed in RULES.md. Not claimed: a population learning curve, independent replication, or proof that
size caused every historical difference.

## 7. Conditional trigger

Defined exactly in RULES.md before fitting. In short: it fires only if the newly refitted S100 occupation predictor of BOTH versions
meets every occupation point cap and both subgroup guards against the original references, meets the predefined occupation reserve on
both source-reference versions, and has a verified deployed implementation (bitwise from the six raw fields, with every refusal). It is
an occupation-component trigger, not a declaration that the H3-anchored service qualifies. A smaller subset is never substituted and
the reserve is never relaxed after outcomes. If it fails, the size analysis, independent replay, deployment and closeout are completed
and the correction recipe is closed without another protection sweep.

## 8. One conditional correction experiment (only if triggered)

Per version v on the full permitted pool (S100):
- **Arms.** W (unchanged full-data witness W100_v); CONT (matched zero-critic continuation); LOC (occupation-only critic; the PRIMARY
  corrected arm, declared here); PAIR (coalition-aware critic reading the frozen H3 income S<v>). The source P2 recipe
  (ohn.protect.train_correction, guard 'none', one full pipeline per arm) on the frozen-T_TREE-routing parametrization: identical
  routing capacity, epochs, task loss and fitting effort except the declared critic term. Correction base = W100_v's 5-fold household
  cross-fitted out-of-fold logits (as the source); routing = the T_TREE occupation teacher refitted on S100 with seed v.
- **Strengths.** The finite set in RULES.json (identity included). Release r_s = softmax(log p_W + s x Delta_arm).
- **Selection information.** Nested 10-fold household cross-fitting of MODEL_FIT ('TSZ1_ncv10'): for every outer fold every upstream
  layer (dictionary, witness and its cross-fitting base, routing, correction) is refitted on the other 90% (about 18,000 people) and the
  incremental costs against s = 0 are measured on the held-out fold; pooled over the 10 folds. Every submodel's training size is
  recorded. Pooled cross-fitted cost estimates target 90%-size submodels and do not guarantee the deployed 100% model's costs. The
  compute is benchmarked first; if the honest calculation does not fit the budget, ancillary experiments are cut, not the exclusion.
- **Selection rule.** The single fixed rule of RULES.md, applied to the nested estimates only. MODEL_SELECT supplies only the task-stage
  trigger and never enters strength selection. CALIBRATION has no SEX column and is not used. Identity is always admissible; the
  selected release is always stored, identity included; there is no requirement that both versions select a nonzero strength. CONT
  is evaluated at the strength selected for the arm it matches.
- **No revision.** No correction revision follows the reader outcomes; negative and zero-strength results are preserved.

## 9. Development readers

Fresh readers (READER_SPLITS.json, DEVGRID_CODE_LOCK.json), frozen here before any new fit: MODEL_SELECT households split per
repetition R1, R2 (new tags) into FIT 50% / VALIDATION 20% / SCORE 30%; reader seeds k0, k1 from a new seed base; the source 16-recipe
bank with identical sizes and effort for every release; selection on VALIDATION, scoring on SCORE, paired household bootstrap of SCORE.
Views: income-only, occupation-only and pair readers, each given only the transmitted probabilities and decisions. Cells: 2 versions x
2 repetitions x 2 reader seeds. Evaluated (if the stage ran): the actual selected corrected release of each version (identity where
selected), the unchanged witness, the actual selected LOC and PAIR counterparts, the matched CONT, and the strong inherited references
(NEURAL_RAW, H_LJ, B2TWIN, RETIRED_B3CTL, SUCC, TWJ). Nearby ladder strengths are never substituted for the selected outputs. Reader
recipes and overlapping splits are not independent datasets. No compression control is run.

## 10. Claims

Kept separate (RULES.md): task-quality margin restored by training size; nonzero correction preserves utility; recovery falls beyond
continuation; recovery falls beyond the local correction; strongest-reference competitiveness; the original 0.020 material criterion;
joint-design advantage; novelty. Original limits and the reserve are reported separately for every candidate and reference; all
originally point-eligible strong references stay in the original-benchmark table even if they fail the reserve, and a separate table
applies the same reserve to every method. Occupation-only optimization is not a new two-output joint method; lower recovery for declared
readers is not arbitrary-reader protection. If no selected nonzero release preserves utility and improves recovery beyond its strongest
relevant controls, this correction recipe is closed and no nearby strength adjustment is proposed.

## 11. Packages, deployment and verification

Packages deployable from the six raw fields: the refitted full-data task witness (W100_v), the equally exposed S100 TASK_REFIT control,
a representative smaller fit (A1_PO_16 S50_O0, v = 0), and, if the stage ran, the selected correction and matched continuation
releases. Each must reproduce its stored outputs bitwise and refuse SEX, target, extra, missing and reordered fields at both the
underlying function and the documented entry point; no hidden outputs. An independent verifier replays, with separate numerical code:
source pins and household nesting; subset membership and fitting exposure; model identities and deterministic aliases; task and
subgroup metrics and paired size contrasts; trigger and reserve; strength selection; recovery at the selected strengths; claim labels
and their scope; raw-field deployment from restored artifacts. Numerical reproduction and procedural compliance are reported
separately; a corrected verifier check keeps its original result, cause and adjudication; acceptance rules are never widened.

## 12. Resources, process and amendments

Eight elapsed research hours (to 2026-10-19T07:45:15Z) plus thirty minutes for safe closeout; fourteen CPU-hours including agents,
failures and reader refits; at most two heavy jobs at once, one numerical thread each, through one shared queue (tsz.jobqueue) that
hashes imported code at launch; zero paid or cloud compute; at most six live agents including the coordinator. Code is committed before
the jobs it governs. Locks are immutable; any change is a dated amendment before the stage it affects (PROTOCOL_DEVIATIONS.md,
ACCESS_AMENDMENTS.json); every attempt is preserved and its timing disclosed; this protocol is never edited to insert retrospective
explanations.

## 13. Publication and custody

Sanitized source and aggregate results are published on the dedicated branch publication/pcrl-controlled-training-size-v1 of the
GitHub remote (a local push is not a public receipt; the receipt is verified with git ls-remote). No raw person records, fitted private
checkpoints, identifying local paths or unreviewed inherited history are published; the research branch, whose history contains
private paths, stays local. main, manuscripts, earlier branches, old locks and source packages are not changed. Models and readers are
backed up and restored from the backup alone; an attached writable external drive is used if available, otherwise off-device custody
is reported as pending; no mount is forced and a warm read is never called a cold check. No messages are sent to people; repository
handoff files are the only handoff.
