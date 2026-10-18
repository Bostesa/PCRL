# Coordinator design brief: correction transfer, genuine refits and a stable method comparison (PCRL, 2026-10-18)

Written by the coordinator at about 18:50Z, before any registration, fit or label read of this campaign. It is the working
design that the registration (PROTOCOL.md, RULES, EXPOSURE_MAP.md, PARTITIONS.json) freezes. Agents read it as their shared
context. Campaign id `pcrl_correction_transfer_stability_v1`, package `cts/`, branch `research/pcrl-correction-transfer-stability-v1`
(local), publication branch `publication/pcrl-correction-transfer-stability-v1` on https://github.com/Bostesa/PCRL.git.

## 0. Research context (terminology fixed)

Statistical machine-learning research on already admitted public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for
Florida (development roles of the protected-tree replication). Useful tasks: income class (2) and occupation group (5 official
broad civilian groups). Each recipient receives only its own float64 probability vector and argmax decision. Runtime fields:
AGEP, WKHP, COW, SCHL, MAR, POBP only. SEX is the recorded demographic column: a training-time label (critics, subgroup costs) on
authorized training households and an evaluation label for development readers; never a runtime input or output. Evaluation
classifiers ("readers") estimate SEX from released probability vectors (each output alone and the linked pair). The
recipient-access adversary (granted linked outputs, labelled auxiliary examples, one release per person) is standard, not new.

Source result kept as is: the occupation-headroom campaign (publication 1435c395d) ended NO_QUALIFYING_CONFIGURATION. Nothing
here reinterprets it.

## 1. Questions

- A. Cost estimation: can incremental occupation and subgroup costs be predicted on excluded households accurately enough to
  choose a useful correction strength?
- B. Method stability: does an improved correction rule keep actual utility while delivering a repeatable recovery benefit
  beyond its strongest matched controls?
- C. Comparison design: how much do the old recipient screen and unequal reserve requirements affect interpretation?

## 2. Roles and partitions (MODEL_FIT households only; label-free assignment)

Admitted roles: MODEL_FIT 20,000 people / 13,337 households (features, raw inputs, income, occupation, SEX); CALIBRATION
2,982 / 1,974 (features, income, occupation; no SEX); MODEL_SELECT 8,001 / 5,306 (features, income, occupation, SEX; adaptively
used by every earlier campaign). ATTACK_FIT, ATTACK_VALIDATION and DEV_REPORT labels stay closed for the whole campaign (their
features may be used label-free for deployment parity). Sealed assessment and capped-out records: closed.

Partition scheme (frozen before any fit; assignment by SHA-256 of household ids, `ohn.common.household_folds` convention):
- blocks: every MODEL_FIT household -> one of 20 blocks, tag `CTS1_blocks` (5% each).
- repair CHECK panels, reserved now: `C0` = blocks 0-2, `C1` = blocks 3-5 (about 15% of households each). Excluded from every new
  diagnostic and repair fit, calibration and preprocessing fit. Their labels are refused by the access layer until the committed
  REPAIR_FREEZE lock.
- pool = blocks 6-19 (about 70%).
- D0, D1 (diagnosis): pool households ordered by tag `CTS1_D0` / `CTS1_D1` into 10 parts: FIT parts 0-5 (60%), GUARD 6-7 (20%),
  CHECK 8-9 (20%).
- R0, R1 (repair comparison): pool households ordered by tag `CTS1_R0` / `CTS1_R1` into 4 parts: FIT 0-2 (75% of the pool), GUARD 3
  (25%); CHECK = `C0` for R0 and `C1` for R1.
- associations: D0, R0 -> source reference seed 0 (U0 s0, Ucal s0, H3 s0, frozen comparators s0); D1, R1 -> seed 1.
- calibration: the task learner of the main study (A1_PO_16 recipe) has no calibrator; any calibration uses the admitted
  CALIBRATION role (disjoint from every CHECK) or an internal split of FIT.
- support check (recorded before any fit, after the receipt): each SEX group >= 500 people in every GUARD / CHECK; every
  occupation class x SEX cell >= 3 people in every role and >= 10 in every FIT. On failure the partition takes the next tag suffix
  (`b`, `c`, ...), recorded; no outcome is computed before the check.
- overlaps are disclosed: D0 / D1 FIT sets share about 36% of the pool, R0 / R1 FIT sets about 56%. These are different training
  partitions of the same people, not independent population replications.

Within a repetition: FIT = task learning, preprocessing dictionary, routing, correction directions, training critics, cross-fitting;
GUARD = choosing a strength from the fixed ladder (and nothing else); CHECK = realized cost transfer after that choice.

Original-benchmark reporting: for each stage the pair (seed-0 association, seed-1 association) is evaluated on MODEL_SELECT with
its frozen references: 24 utility clauses (2 tasks x U0 / Ucal x accuracy / NLL / Brier x 2 source seeds) plus the 4 subgroup
guards. D pairs and R pairs are reported separately. Repetitions are never pooled into extra seeds.

## 3. Track F (fixed H3 income, newly fitted occupation pipeline)

Per partition P (associated seed s):
1. Dictionary: the admitted 145-column encoding re-collapsed on FIT (an admitted COW / SCHL / MAR / POBP level with fewer than 10
   FIT people -> its block's OTHER column). Because the admitted dictionary kept levels with >= 10 MODEL_FIT people, the composite
   equals a dictionary built from FIT raw codes alone (any level with >= 10 FIT people has >= 10 MODEL_FIT people). AGEP / WKHP
   enter through the admitted strictly increasing affine z-map, to which the tree learners are invariant.
2. Witness W_P: the A1_PO_16 recipe (scikit-learn HistGradientBoosting, compact_po coding, 16 leaves, min leaf 100, lr 0.025, l2 3,
   T* <= 1000 by household 5-fold inner CV on FIT, refit on FIT with T*). Household cross-fitting on FIT (5 outer folds, full
   nested inner CV inside each) gives the out-of-fold base logits used to train corrections and the fold models used for the
   deployed-ensemble variant. Sensitivity witness (bounded): the B2_SINGLE recipe on one partition.
3. Routing for the leaf-score correction: REFIT = the replication's T_TREE occupation recipe (CatBoost 1.2.8, depth 4, lr 0.05,
   <= 800 trees, l2 3, tree count by its 15% household-disjoint internal split, refit) on FIT only; FROZEN = the admitted T_TREE of
   seed s (fitted on all MODEL_FIT: saw GUARD and CHECK labels). CatBoost 1.2.8 is installed in a private virtual environment for
   fitting only; routing at inference is numpy-only (acs_replication.trees.TreeTeacher).
4. Directions (source P2 recipe, `ohn.protect.train_correction`, guard 'none'): PAIR (critics on the occupation wire and on the
   linked pair with H3's frozen income wire), LOC (occupation-wire critics), CONT (critic weight zero, matched effort). Trained on
   FIT from the cross-fitted witness logits; 5 fold pipelines (on the witness's outer folds) plus the full-FIT pipeline.
5. Releases on the fixed strength ladder S = {0, 0.1, ..., 1.0}: deployed r_s(x) = softmax(log p_W(x) + s * Delta_full(x)); the
   deployed-ensemble variant e_s(x) = mean_k softmax(log p_Wk(x) + s * Delta_k(x)) over the 5 fold pipelines.

Cost estimators of the incremental cost against the uncorrected release on the same people (men's / women's / all summed-Brier
increase, NLL increase, accuracy loss; household-clustered SEs):
- OOF (source estimator): fold pipeline k on fold k of FIT, cross-fitted witness base (exactly `ohn.protect.project` costs);
- GUARD: deployed release on GUARD; also the deployed-ensemble on GUARD;
- CHECK: realized on CHECK (D: diagnostic; R: reserved panel, after REPAIR_FREEZE);
- MS: realized on MODEL_SELECT (benchmark role).
Baseline slack: the witness's MODEL_SELECT margin to each occupation-side cap (accuracy, NLL, Brier vs U0 and Ucal; men's and
women's guards vs U0) with its household-clustered SE; the reserve slack is margin - 2 SE (the frozen source reserve rule). The
baseline slack and the incremental cost are reported separately.

Diagnosis factorial on D0 and D1: routing (FROZEN, REFIT) x arm (PAIR, LOC, CONT) x deployment (full, ensemble), all ladder points,
estimators OOF / GUARD / CHECK / MS. A nested variant (every upstream layer refitted inside each outer fold) is compared with the
non-nested OOF at the same strengths on at least one partition. LOC versus PAIR isolates the frozen H3 income anchor (the only
place H3 enters the occupation pipeline is the PAIR critic).

## 4. Strength rules (statistical reviewer specifies uncertainty before outcomes)

- OLD: the source projection reproduced (OOF costs within FIT, largest s on the 101-point grid refined by bisection with every
  smaller grid point feasible, budget f x reserve slack, f = 0.8 (B80) primary and 0.5 (B50) secondary).
- SIMPLE: conservative scalar shrinkage selected on GUARD from the ladder: accept s if every GUARD point cost <= gamma x reserve
  slack (gamma fixed before outcomes), fixed-sequence ascending, select the last accepted.
- UNCERTAINTY: household-aware uncertainty in GUARD costs and in the baseline slack, explicit subgroup SEs, finite ladder
  selection (fixed-sequence over the ladder; intersection-union over constraints), stated assumptions. A bootstrap upper bound is
  not presented as a distribution-free certificate.
- Identity is always a valid outcome; it earns no protection-component claim. Rules are compared at common strengths and across
  their observed frontiers on the same fitted directions.
- At most one direction redesign (card first, with a falsification condition and the simplest alternative explanation), only if
  the diagnosis shows that the direction, not its scale estimate, drives unequal group cost.

## 5. Readers, comparators and screens

- Development readers on MODEL_SELECT with the source grid (DEV_GRID: repetitions R1, R2 x reader seeds k0, k1 per source seed;
  FIT 50% / VALIDATION 20% / SCORE 30% of households; the validated enhanced slate of 16 recipes with ignore-recipient
  alternatives); cached frozen-reference results reused as cache hits; a third repetition for finalists if affordable; nested
  learning curves for finalists.
- Original-benchmark comparison keeps every original point-eligible strong reference (NEURAL_RAW, H_LJ, B2TWIN, RETIRED_B3CTL, SUCC,
  TWJ). The reserve-qualified frontier applies the same reserve to candidates and controls and shows which references fail it.
- Screen study (independent of candidate results, frozen before new outcomes are seen): the old per-cell recipient gate versus a
  predeclared aggregate paired rule with household-aware uncertainty and between-reader variability, on frozen source releases,
  exact aliases, identical releases under separately fitted readers, and known zero / nonzero shifts under specified synthetic
  laws.

## 6. Track R (whole-service refit), if the recipe can be rebuilt within budget

On R0 and R1 (FIT, CALIBRATION): T_TREE income and occupation (CatBoost recipe), Stage B LOCAL and JOINT corrections (replication
recipe, SEX critics on FIT), H3's HGB income learner (ihs.fam_h Step 1 recipe on FIT), its CALIBRATION stack, beta budget and
Platt map, then the occupation pipeline with the refitted income wire. A recipe test on different training subsets, not
confirmation of the frozen H3 package. If it cannot be completed, Track F finishes and whole-service stability is marked untested.

## 7. Budget and accounting

Elapsed: research window 8 h from 2026-10-18T18:32:40Z (first action) to 2026-10-19T02:32:40Z; closeout allowance 30 min to
03:02:40Z. CPU: 14 CPU-hours in total (agents' numerical work, failed attempts and refits included). At most two heavy numerical
jobs at once, one numerical thread each, through one shared queue (`cts.jobqueue`). Up to 72 new fitted release specifications
across training repetitions; routing, critic, calibration, cross-fitting and reader fits counted separately; cache hits,
people, model seeds, training repetitions and reader seeds counted separately. No paid or cloud compute.
