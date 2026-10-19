# Research decision: controlled training-size test with one conditional protection experiment (PCRL, 2026-10-19)

**Scope.**
- Exploratory development on previously used public-use U.S. Census Bureau ACS 2019 1-Year PUMS Florida records, using only the
  MODEL_FIT (20,000 people), CALIBRATION and MODEL_SELECT (8,001 people) roles.
- Income and occupation are the useful tasks. "Recovery" means estimation of the recorded SEX column by the declared development
  readers: separate evaluation classifiers given only the released probability vectors and decisions. It is never protection against
  every reader.
- MODEL_SELECT was used adaptively by every earlier campaign and was the common evaluation role here, so nothing here is
  confirmation. The ATTACK_FIT, ATTACK_VALIDATION and DEV_REPORT labels, the sealed assessment and capped-out records stayed closed.
- Every earlier verdict is preserved:
  - NO_QUALIFYING_CONFIGURATION (occupation headroom);
  - the correction-transfer diagnosis and negative method result (rung 0);
  - H3's frozen income package.
- Numbers come from the records named in START_HERE.md; "x 1e-3" means thousandths. The two source versions (v0, v1) share every
  training row: they are procedure repeats, never independent replications.
- The independent claims review (roles/claims_review/REPORT.md) was applied to this text (adjudication A5).

## Decision in brief

- **Training size (the primary question): the registered size explanation is supported on this pool, but the size gain is not shown
  to be specific to the recipe.**
  - Under the fixed A1_PO_16 recipe the men's subgroup margin improves with training size, resolved on both versions.
  - The occupation reserve holds in all 8 cells at 70% and in both versions at 100%, but in only 4 of 8 cells at 50%.
  - The full-data refit reproduces the source package bitwise.
  - The equally exposed task-only control (T_TREE then TASK_REFIT) shows a resolved size gain of its own on NLL, Brier and men's Brier,
    and the difference between the two gains is not resolved on those metrics (registered reading IR3, CONTROL_EXPLAINS_GAIN).
- **The conditional correction ran** (the registered trigger fired). It selected s = 0.3 for both the primary occupation-only arm
  (LOC) and the coalition-aware arm (PAIR), in both versions.
- **Classification: a closed negative correction recipe** (CORRECTION_RECIPE_CLOSED; reader labels under amendment A-SC1, disclosure
  in section 5).
  - At s = 0.3 both arms keep utility on MODEL_SELECT in both versions: all 12 clauses and both within-SEX guards per version at the
    original point limits, and, reported separately, the occupation-side two-SE reserve (a development heuristic).
  - Both fail the original per-cell recipient gate in the occupation view against every strong reference (section 6).
  - Neither lowers the declared readers' recovery stably beyond the unchanged witness or the matched zero-critic continuation. Both
    stay far from the 0.020 material target.
  - Not a useful component; not competitive.

## 1. Did the controlled training-size comparison explain the lost margin?

**Short answer.** On this development pool, within the fixed recipe, the registered size explanation is supported: the men's-guard
margin rises with training size, resolved on both versions, and the margin pattern matches the earlier loss. But the ordinary equally
exposed control shows a resolved size gain of its own: NLL 7.8 / 8.1 against the recipe's 11.5 / 11.4 x 1e-3, and men's Brier 2.9 / 4.1
against 4.3 / 4.2. The difference between the gains is not resolved on those metrics, so the gain is not shown to be specific to the
recipe (IR3). This is not a population learning curve and not causal proof.

Design: four household orderings of MODEL_FIT, with nested subsets of 6,669 (50%), 9,336 (70%) and all 13,337 households (100%). Each
subset was fitted with A1_PO_16 and with the control, in both versions (SUBSETS.json; size/SIZE_TABLE.csv; size/SIZE_RESULTS.json;
figures/SIZE_CURVE.png).

| A1_PO_16, mean over orderings | 50% | 70% | 100% |
|---|---|---|---|
| Men's Brier excess vs U0 (x 1e-3), v0 / v1 (cap 5) | +0.47 / +1.92 | -1.51 / -0.09 | -3.85 / -2.29 |
| Occupation NLL excess vs U0 (x 1e-3), v0 / v1 | -4.67 / -1.45 | -10.47 / -7.40 | -16.14 / -12.80 |
| Cells meeting the occupation-side reserve (k = 2, H_LJ floor) | 4 of 8 | 8 of 8 | 2 of 2 |
| Minimum occupation-side ratio (range over cells) | 1.10-2.87 | 2.55-3.73 | 4.88 / 4.30 |

The v0 / v1 gap in the excess rows mostly reflects each version's own reference U0_S<v>. On the same rows, the two versions' fits
differ by at most 0.43e-3 in men's Brier.

- **Paired size contrasts** (same MODEL_SELECT households; improvement, x 1e-3; mean over the four orderings, v0 / v1, with the
  combined SE; size/PAIRED_CONTRASTS.csv):
  - 50 -> 100 men's Brier, the registered primary endpoint: +4.32 +/- 1.13 / +4.22 +/- 1.12, every ordering positive (3.2-5.3).
    Label SIZE_GAIN_RESOLVED.
  - 50 -> 100 NLL +11.5 / +11.4 (+/- 1.9), Brier +4.8 / +4.7 (+/- 0.9), women's Brier +5.3 / +5.2 (+/- 1.5): all resolved on both
    versions. Accuracy +5.9 / +5.8 (+/- 3.0) is not resolved.
  - Both steps contribute: 50 -> 70 and 70 -> 100 are each resolved on both versions for NLL, Brier and both SEX groups.
- **Margin pattern:** FULL_ONLY, with the sub-flag FULL_ONLY_70_RETAINS.
  - Registered reading IR1: "Full-data refitting reproduces the occupation-side margin while smaller fits lose it: the previous
    reduced-training experiment did not settle full-data feasibility."
  - Size explanation: SUPPORTED_ON_THIS_POOL. By the registered definition this means only that, on this pool and under this recipe,
    the larger nested training sets had resolved lower loss on MODEL_SELECT and the margin pattern matched.
- **The earlier campaign.** Its refits on 42-53% of MODEL_FIT had men's excess +0.65 / +2.15 / +1.11 / +1.89 (x 1e-3). That is
  descriptively in line with the 50% cells here, but it comes from different partitions.
- **Subset-to-subset variation** (size/SUBSET_VARIATION.csv): at 50% the men's excess spans -0.44 to +1.40 (v0) and +0.94 to +2.99
  (v1) x 1e-3 across the four orderings, a spread of about 2e-3. The version difference on the same rows (different inner folds only)
  is at most 0.43e-3.
- **Encoding sensitivity** (FIT-only vs the admitted full-MODEL_FIT dictionary at the same S50 rows, v0; size/ENCODING.csv):
  unresolved on every metric (NLL +0.64 +/- 0.66 x 1e-3 in favour of the admitted dictionary). Its point estimate is about 5-6% of the
  50 -> 100 change (version 0 only; descriptive, not a decomposition).
- **Not claimed:** a population learning curve, independent replication (the orderings are nested draws from one pool and the
  versions share rows), or that size caused every historical difference.

## 2. Did the 100% recipe reproduce usable occupation headroom?

**Short answer.** Yes for reproduction and for the registered margin: the refits are bitwise (REPRODUCED_BITWISE, MARGIN_REPRODUCED),
keep the occupation reserve with minimum ratios 4.88 (v0) and 4.30 (v1), and the trigger fired (MARGIN_AVAILABLE). Whether the slack
is usable for protection is answered in section 6.

- **REPRODUCED_BITWISE.** The refits have the same T* (168 / 154), identical model files, and every role's release equal to the source
  bit for bit (runs/SIZE/A1__S100__V<v>/REPRO.json). The independent verifier refitted all 22 A1_PO_16 fits with its own code, both
  full-data fits included, and matched them bitwise. MARGIN_REPRODUCED.
- **Trigger** (trigger/TRIGGER.json, evaluated once): T0-T3 pass on both versions. That covers:
  - all 8 occupation point conditions per version;
  - the 8 occupation-side reserve conditions (binding: the men's guard);
  - the verifier's deployment record: both packages bitwise from the six raw fields, 24 of 24 refusals.

  The verifier's independent replay agrees within 1.7e-16.
- **Usable slack at 100%** (margin - 2 x se_used on the men's guard): about 5.2e-3 (v0) and 3.9e-3 (v1).
- The two-SE reserve is a development heuristic, not a new scientific cap or population guarantee; it is reported separately from
  the original limits. The trigger is an occupation-component trigger and does not declare that the H3-anchored service qualifies.

## 3. Was that better than an equally trained ordinary control?

**Short answer.** Only partly.
- At 100% A1_PO_16 is resolved better than T_TREE on NLL, Brier and men's Brier (not on accuracy or women's Brier).
- It is not resolved better than TASK_REFIT, the registered primary control, which itself keeps the occupation reserve at 100%.
- At 50% and 70% no difference is resolved.
- The control shows a resolved size gain of its own.

Details:
- **Level differences at 100%** (A1_PO_16 improvement over the control, x 1e-3, v0 / v1; size/CONTROL_COMPARISON.csv):
  - vs TASK_REFIT: NLL +2.4 / +2.5 (+/- 1.7, not resolved); men's Brier +2.7 / +1.9 (+/- 1.1, version-dependent); accuracy loss
    +5.5 / +6.3 (+/- 2.8, version-dependent).
  - vs T_TREE: NLL +4.0 / +4.6, Brier +1.6 / +1.9, men's Brier +2.8 / +2.9: RECIPE_BETTER_RESOLVED.
- **The controls' reserve:**
  - TASK_REFIT meets the occupation-side reserve at 100% (3.41 / 3.19), in 7 of 8 cells at 70% and in 1 of 8 at 50%;
  - T_TREE meets it in 2 of 2, 4 of 8 and 1 of 8 cells;
  - so CONTROL_ALSO_MEETS_RESERVE holds at 100% only.
- **The control's size gain.** TASK_REFIT's 50 -> 100 gain is resolved for NLL, Brier and men's Brier.
  - The difference between the two gains (per-person double difference) is not resolved for NLL, Brier, men's Brier or accuracy.
  - For women's Brier it is version-dependent: +2.44 +/- 1.14 x 1e-3 in favour of A1_PO_16 on v1, unresolved on v0 (label
    RECIPE_GAIN_CONTROL_UNRESOLVED).
  - The registered label is CONTROL_EXPLAINS_GAIN for NLL, Brier and men's Brier (reading IR3).
- **Identity.** The S100 controls reproduce the admitted T_TREE and TASK_REFIT bitwise (control repro; verifier).

## 4. Did the conditional correction run, and what strengths did it actually select?

**Short answer.** Yes. The single fixed rule selected s = 0.3 for LOC (primary) and PAIR (secondary) in both versions, from fitting
information only.

- **Recipe.** The source P2 correction, fitted on the full pool (all MODEL_FIT):
  - on the full-data witness W100_v, with its 5-fold cross-fitted base (bitwise equal to the source's) and the S100 T_TREE routing;
  - CONT, LOC and PAIR are identical except for the critic term.
- **Selection information.** Nested 10-fold household cross-fitting of MODEL_FIT: every upstream layer was refitted on about 18,000
  people. The frozen H3 income is the one input not refitted per fold; it is used only by PAIR's training critic.
- **Rule.** The largest registered strength s such that s and every smaller nonzero strength keep estimate + 2 SE at or below one
  H_LJ-floor SE for each of men's Brier, women's Brier, Brier, NLL and accuracy loss. The floor used is the smallest among the
  conditions mapped to each statistic. If 0.01 fails, identity is selected.
- **Selected.** LOC 0.3 and PAIR 0.3 in both versions; the first failing strength is 0.5 in every case
  (conditional/SELECTION_V<v>.json). The selections were committed at 01:18Z, before any MODEL_SELECT read of a corrected release.
  The verifier's independent replay, and its end-to-end recomputation of one outer fold, agree.
- **In point estimates the realized costs exceeded the nested predictions** (conditional/COND_UTILITY.json, realized_minus_predicted;
  recorded after selection, never gating).
  - Men's Brier at s = 0.3, realized (on MODEL_SELECT) against predicted (out-of-fold on MODEL_FIT), x 1e-3:
    - LOC: +1.90 / +1.13 against +0.69 / +0.52;
    - PAIR: +1.46 / +0.89 against +0.76 / +0.47.
  - NLL: +1.6 to +2.2 against +1.0 to +1.1.
  - Only LOC v0's men's gap (1.21e-3) exceeds two combined SEs (realized SE 0.33-0.40e-3, predicted SE 0.19-0.22e-3); the other three
    gaps are about 1.0-1.6 SE.
  - As registered, pooled cross-fitted costs target 90%-size submodels and do not guarantee the deployed model's costs.

## 5. Did readers evaluate those exact selected releases?

**Short answer.** Yes. Fresh development readers on new MODEL_SELECT household splits (grid_specs/COND.json; the grid outputs are
private) evaluated:
- LOC and PAIR at their selected strength 0.3, and the matched continuation at 0.3;
- the witness, the full-pool task-only controls and the six inherited references.

They used 8 cells (2 versions x 2 repetitions x 2 reader seeds) and the same 16-recipe bank for every release.

- No nearby strength was substituted, and no version selected identity.
- **Reader-seed check.** The registered effective-reader-seed check, read literally over all 14 checked bank members, returned
  READER_SEEDS_NOT_EFFECTIVE: ExtraTrees readers (ET5, ET25) are seed-invariant on the income views of B2TWIN and RETIRED_B3CTL, which
  take 6 distinct values. On the registered minimum set (W100_v and every selected release) there was no equality. Amendment A-SC1,
  made after this check's outcome and before any reader-stability label was computed or read, limited grid invalidation to
  unexplained equalities. The grid passed the amended check. No stability formula changed.
  - Qualification (PROTOCOL_DEVIATIONS items 11-13): "before any reader-stability label was computed or read" holds for the
    coordinator and the statistical reviewer. The independent verifier had already computed its own reader labels (unit
    VER_READERS_V1, 01:32:59-01:33:09Z, after the seed-check record and before the ruling at 01:36:00Z) and stored them unread until
    the claims were computed.
  - Under the literal reading no stability label would exist and neither arm could qualify, so the outcome below is the same under
    both readings.
- **Seed-deterministic winners.** Many registered comparisons have them in some cells (qualifier READER_SEEDS_PARTLY_EFFECTIVE); the
  registered spread rule counts duplicated cells once. Reader recipes and overlapping splits are not independent datasets.

## 6. Did useful protection survive, and against which controls?

**Short answer.** No. Point utility survived at s = 0.3, but the declared readers' recovery did not fall stably beyond the witness or the
matched continuation, and the coalition-aware arm did no better than the occupation-only one. Reader labels are under amendment A-SC1
(disclosure in section 5).

Pair benefit = comparator SCORE AUC minus candidate SCORE AUC (positive = less recovery), x 1e-3, over 8 cells (conditional/CLAIMS.json,
MATCHED_CONTROLS.csv):

| Comparison | Mean | Cells (min / max) | Label |
|---|---|---|---|
| LOC 0.3 vs witness W | +4.6 | +1.4 / +7.3 (all positive) | POSITIVE_NOT_STABLE |
| LOC 0.3 vs continuation CONT 0.3 | +5.0 | +1.2 / +8.1 (all positive) | POSITIVE_NOT_STABLE |
| PAIR 0.3 vs W | +3.7 | -3.1 / +8.4 | SIGN_CHANGES |
| PAIR 0.3 vs CONT 0.3 | +4.1 | -2.2 / +9.2 | SIGN_CHANGES |
| PAIR 0.3 vs LOC 0.3 (its own selected strength) | -0.9 | -6.6 / +2.8 | NOT_POSITIVE |
| CONT 0.3 vs W | -0.4 | -0.9 / +0.2 | NOT_POSITIVE |

The reader seed changed the result in 3 of the 4 (version, repetition) pairs for every comparison in the table except CONT vs W (2 of
4); identical cells are counted once (n_eff 7 and 6).

- **Utility** (conditional/COND_UTILITY.json): LOC, PAIR and CONT at 0.3 pass all 12 clauses and both guards at the point in both
  versions. Reported separately, they keep the occupation-side reserve: minimum ratios LOC 3.83 / 3.64, PAIR 4.08 / 3.78, CONT
  4.95 / 4.52, W 4.88 / 4.30. Claims C2 (point and reserve) and C2p (point only): UTILITY_PRESERVED_BOTH_VERSIONS for both arms.
- **Protection claims.** C3 (beyond the continuation) and C3w (beyond the witness) are NOT_MET for both arms; C4 (PAIR beyond LOC) is
  NOT_MET. Neither arm qualifies, and the registered closure rule gives CORRECTION_RECIPE_CLOSED. No nearby strength is proposed.
- **Strong references (original benchmark):**
  - Against each of the six references (NEURAL_RAW, H_LJ, B2TWIN, RETIRED_B3CTL, SUCC, TWJ), every candidate's 8-cell mean pair
    benefit is negative: -2.8 to -45.2 x 1e-3, all NOT_POSITIVE. Against TWJ the cells take both signs for LOC, W and CONT.
  - Every reference passes the original point limits (24 clauses + 4 guards) but fails the reserve heuristic.
  - The occupation-view recipient gate fails against every reference.
  - The 0.020 target against NEURAL_RAW and H_LJ is not met.
  - Not competitive: C5 NOT_MET, C6 DEV_GRID_TARGET_NOT_MET.
- **Equal reserve** (EQUAL_RESERVE.csv):
  - Every reference fails the reserve: whole-release minimum ratio over both versions 0.32-1.56, per version 0.32-1.95.
  - Every H3-anchored candidate and control qualifies.
  - Among qualified points LOC 0.3 has the lowest mean pair AUC (0.644 against the witness's 0.649); that is the same not-stable
    difference as above.
  - A positive equal-reserve position is not superiority over the original references.
- The flag COARSE_REFERENCE_INCOME_WIRE (never gating) marks the income-view gates and pair comparisons against B2TWIN and
  RETIRED_B3CTL.

## 7. What income-side limitations remain?

**Short answer.** The income side is H3's frozen release throughout, and it was not refitted here.
- It passes its income clauses and reserve (minimum ratio 3.51 / 2.86).
- It caps the whole-release reserve of every H3-anchored release at 2.86 (v1 income accuracy). That is the binding condition for every
  candidate, the witness and TASK_REFIT; T_TREE binds lower on its own occupation side (2.60).

Details:
- **The income in every release.** Every complete release in this study carries H3_STRONG_LJDIR_PB income of its version, frozen. Per
  its manifests it was fitted by its source on all MODEL_FIT and CALIBRATION, and its RELEASE.json also lists stored replication tree
  parameters, including a JOINT_TREE income correction trained with MODEL_FIT SEX. The correction stage changes occupation only;
  PAIR's critic reads H3's income during training.
- **Untested here.** The earlier campaign found that H3's income headroom did not survive a refit of its recipe on about half of
  MODEL_FIT. This study did not refit the income recipe at 100%, so whether that headroom is a training-size effect too is untested.
- **Recipient gates** are recorded beside every comparison. Against the references, the occupation view fails the gate for every
  candidate; the H3 income view fails it only against RETIRED_B3CTL, whose income wire is a 6-level code.

## 8. Classification

**A controlled training-size result on this development pool, and a closed negative correction recipe.**
- **The training-size result.**
  - Within the fixed recipe, larger nested training sets have resolved lower occupation and subgroup loss on MODEL_SELECT.
  - The full-data refit reproduces the source package bitwise and, with it, the occupation-side margin (MARGIN_REPRODUCED).
  - The reduced-training refits of the earlier campaign did not settle full-data feasibility (IR1).
  - The ordinary equally exposed control shows a resolved size gain of its own on NLL, Brier and men's Brier, and the difference is
    not resolved there (IR3).
  - Claim C1: MARGIN_RESTORED_BY_TRAINING_SIZE_ON_THIS_POOL. Registered scope: within the fixed recipe, this MODEL_FIT pool, adaptively
    used MODEL_SELECT; not a population learning curve; not causal for every historical difference.
- **The correction: CORRECTION_RECIPE_CLOSED.**
  - It is not a useful protection component and not a competitive result.
  - C7 (joint-design advantage) is NOT_CLAIMABLE; C8 (novelty) is NOT_CLAIMED.
  - Occupation-only optimization is not a new two-output joint method. Lower recovery for the declared readers would not be
    arbitrary-reader protection in any case.

## What remains unresolved

- Whether A1_PO_16 is better than the ordinary TASK_REFIT control at full size (version-dependent on men's Brier and accuracy).
- Whether the size gain is anything more than a property of this finite pool and these learners; no fresh population exists.
- Whether LOC's small recovery reduction at 0.3 (about +0.005 AUC, all cells positive, not stable) is real. It would be far below the
  0.020 target in any case.
- Why the nested 90% estimate falls below the deployed model's subgroup costs in point estimates (by 0.4-1.2e-3 men's Brier; only one
  of four gaps beyond two SEs).
- Whether H3's income headroom depends on training size; its recipe was not refitted at 100%.

## Resources, failures and unrun stages

Final figures are in RESOURCE_ACCOUNTING.json, EXPERIMENT_LEDGER.csv and HANDOFF.json, written at closeout. Every failed attempt and
departure is in PROTOCOL_DEVIATIONS.md. The principal ones:
- two selftest label reads (MODEL_FIT occupation) seconds after the receipt, discarded unused (item 4);
- two failed coordinator deployment checks before the trigger (a shadowed name, then a stale code pin), rebuilt and passed (item 6);
- five label-free benchmark units that ran uncommitted code before the registration (item 7; the verifier's procedural FAIL);
- code amendments from the engineering review before the size analysis (item 5);
- the reader-seed amendment A-SC1 (items 10-13);
- one coordinator time stamp written ahead of the clock (item 9);
- the verifier's reader labels computed before the coordinator had examined the seed check (items 11-12);
- the first custody dry run, which failed on the superseded packages (item 14).

**Verifier.**
- Every numerical replay passes.
- Three passes followed adjudicated verifier implementation bugs, with the original results kept:
  - A1: V0.PINS.PUBLICATION_TREE, FAIL then PASS (CRLF handling);
  - A4: V5.CORRECTION.TARGETS_INVARIANCE, ATTENTION then PASS;
  - A4: V6.TABLES, FAIL then PASS.
- Procedural compliance:
  - V6.PROCESS.CODE_PROVENANCE FAIL (item 7);
  - ATTENTION on the two null-unit selftest reads (item 4) and on the literal seed check (A-SC1).

**Unrun.**
- The optional compression control (not registered as run).
- `tsz.correct realized`; its numbers come from `tsz.evaluate conditional`.

Nothing registered as required was skipped.
