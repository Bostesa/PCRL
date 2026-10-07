# Research decision — hcal (held-out, shared calibration of frozen releases)

**Label: NO_COMPETITIVE_RELEASE_CRITERION_ESTABLISHED.**

| Claim or diagnostic | Status |
|---|---|
| OriginalCriterion | NOT_ESTABLISHED (10 of 11 clauses pass; P07 precision) |
| CalibrationMatchedCriterion | NOT_ESTABLISHED (P07 and P13 precision) |
| FittingRole | MIXED |
| ParameterSharing | MIXED |
| ExactDecisionPreservation | receipt: every release, every row, every seed |

This is exploratory development evidence on repeatedly used Adult rows, not confirmation (EXPOSURE_LEDGER.md).

## Timeline (UTC, 2026-10-07; every lock committed, pushed and verified before its stage)

| Step | Commit | Time |
|---|---|---|
| Start (worktree from lra tip 9762025; lra evidence 1dee332 = 1dee33253af03bb5efcaa652774cb222df642aa3) | — | 12:22:22 |
| SOURCE_ADMISSION_LOCK; admission ADMITTED (540 units, 171 frozen-bank tables, bitwise parity) | f259db6 | 12:39 / 12:40 |
| ENGINEERING_LOCK; ENGINEERING_READY (130/130 registered synthetic checks) | f1b2d81 | 13:58 |
| SCIENCE_LOCK | 05cebd6 | 13:59 |
| calibrate, utility (Ucal\* and the audit plan), audit, compose | — | 13:59–14:18 |
| First control run stopped on a technical defect; attempt 1 preserved | — | 14:30 |
| AMENDMENT_A1 (bitwise serialisation receipt; NaN teacher means) | 7fba038 | 14:32 |
| Controls rerun (all_ok) | — | 14:32–14:52 |
| select, replay (90/90 bitwise) | — | 14:53–14:58 |
| Independent inner replay (role E): PASS | — | — |
| EVALUATION_LOCK pushed | b23df38 | 15:00:31 |
| Single assessment | — | 15:00:43–15:04:30 |
| Inference evidence | 5b2d8ef | — |
| Independent phase 3: PASS | — | — |

## What was tested

Everything was frozen except the probability decoder: teachers, heads, the 57 fine partitions and assignments, and the
privacy weights.
- **H-TOKEN32.** The lra per-token κ = 32 objective, with the fixed teacher-mean prior, refitted on 2,000 held-out
  calibration representatives (held out from teacher fitting and head selection).
- **T-TOKEN32.** The same objective on 2,000 matched teacher-training representatives; diagnostic.
- **H-GLOBAL-TEMP.** One inverse temperature per task, map and seed. This is temperature scaling, prior work.
- **H-CLASS-TEMP.** One temperature per predicted class.
- **Controls.** All task-only controls and continuous U were calibrated equally.
- **Common attack bank.** One bank per partition: the admitted legacy readers plus fresh readers on the remaining
  attack rows. Every decoder variant of a partition faces identical attack predictions.

## Results (assessment means over seeds; 95% family-wise intervals, z = 3.0654)

**Inner selection.**

| Role | Selected | Basis |
|---|---|---|
| Ucal\* | H-GLOBAL-TEMP | summed inner NLL 1.5943 vs identity 1.5984 |
| T\* | U\|FINE-TASK\|i8o64\|H-GLOBAL-TEMP | inner pair AUC 0.8468 |
| P\* | U\|JOINT\|i8o64\|l0.1\|H-CLASS-TEMP | inner pair AUC 0.8171; inner benefit 0.0297 ≥ 0.02; local guards hold |

P\* is the existing privacy-trained JOINT λ0.1 partition (SEX used in its historical assignment search) with a newly
registered held-out class-temperature decoder. It is the same frozen partition as lra's P\*, with a different decoder and
a distinct ID.

**Primary slots.**

| Slot | Point [interval] | Outcome |
|---|---|---|
| P01 pair benefit vs T\* | 0.0319 [0.0272, 0.0366] | PASS |
| P02 / P03 local | −0.0014 / −0.0496 | PASS |
| P04 / P05 accuracy | 0 / 0 | PASS (decisions identical to U) |
| P06 income LL vs U0 | −0.0019 | PASS |
| **P07 occupation LL vs U0** | 0.0075 [0.0031, 0.0120] | **NOT_ESTABLISHED_PRECISION** |
| P08 / P09 Brier vs U0 | — | PASS |
| P10 / P11 retention | — | PASS |
| P12 income LL vs Ucal\* | 0.0016 | PASS |
| **P13 occupation LL vs Ucal\*** | 0.0092 [0.0060, 0.0124] | **NOT_ESTABLISHED_PRECISION** |
| P14 / P15 Brier vs Ucal\* | — | PASS |

Neither failing clause is a measured violation. Both are unresolved preservation, close to the pattern lra and qpc found
for this partition's mean decoder (lra P07 0.0081, upper bound 0.0121).

**FittingRole (D01–D04; JOINT λ0.1; T-TOKEN32 minus H-TOKEN32; positive = held-out fitting helped).** MIXED.
- Income LL improved: 0.0099 [0.0044, 0.0154].
- Occupation LL (0.0015 [−0.0027, 0.0057]) and both Brier scores are unresolved.

**ParameterSharing (D05–D08; H-TOKEN32 minus H-GLOBAL-TEMP; positive = sharing helped).** MIXED.
- Occupation LL: +0.0113 [0.0075, 0.0150].
- Occupation Brier: +0.0039 [0.0024, 0.0054].
- Both income contrasts are unresolved.

**Supplementary** (nominal 95%, descriptive; DIRECT-TASK, FINE-TASK and CLASS repeat the pattern).
- The per-token held-out calibrator is worse than the original mean decoder on occupation LL. JOINT λ0.1: 1.2866 vs
  1.2766.
- It equals the in-sample learned decoder D1 (D1 minus H-TOKEN32 −0.0000 [−0.0028, 0.0027]).
- The shared temperature ties the original mean decoder: 1.2754 vs 1.2766, contrast 0.0013 [−0.0002, 0.0027],
  unresolved.

**Losses together** (CALIBRATION_GENERALIZATION.csv; JOINT λ0.1, occupation LL).

| Decoder | Own fitting rows | Inner | Assessment |
|---|---|---|---|
| H-TOKEN32 | 1.2212 (calibration rows) | 1.2832 | 1.2866 |
| T-TOKEN32 | 1.1570 (matched teacher rows) | 1.2839 | 1.2882 |
| Original mean decoder | — | 1.2747 | 1.2766 |
| H-GLOBAL-TEMP | — | 1.2736 | 1.2754 |

- Both per-token fits overfit their own rows by a similar margin and generalise equally badly. The fitting role was not
  the binding problem for the occupation tokens.
- Parameter counts: about 1,070–1,330 free coordinates per seed (occupation) and 13–16 (income) for the per-token
  calibrator, from 2,000 rows, against 2 (global) or 6 (class) parameters.

**Calibrated references.** The fitted inverse temperatures are 0.80–0.89, which softens U and codes alike. Calibrated U
gained about 0.0017 nats on occupation log loss; calibrating P\* gained about 0.0005. The calibration-matched comparison
is therefore slightly harder than the original one.

## Decision (prompt section 15)

- **What was measured.** Pooling helps relative to per-token calibration. The held-out role helps income only. Neither
  repairs the occupation-confidence gap of the frozen privacy-trained code. The best calibrated release keeps the
  earlier privacy benefit (pair AUC −0.032 vs the strongest calibrated task-only code, the same frozen map as before)
  but does not establish occupation log-loss preservation against either U0 or calibrated U.
- **Close this frozen-decoder rescue line.** Do not append a temperature-bound, κ, λ, sample-split or capacity sweep.
  Do not rescore these rows.
- **Next step.** An evidence-based research decision about another mechanism, supported by the empirical paper. No
  confirmation population is opened.

**What this is not.**
- It is not evidence that calibration can never help.
- It is not a new calibration algorithm; temperature scaling is Guo et al. 2017.
- It is not information removal; decoder changes on unchanged tokens remove nothing.
- It is not a joint-versus-sequential result.
- It is not a population privacy guarantee.

## Validity and custody

- **Controls.** Every required real-data control passed after AMENDMENT_A1 (common bank; null threshold equal to the
  source's).
- **Independent verification (role E).** PASS: inner replay of all six checks and phase 3 (all 23 slots exact).
- **Deployment.** Six releases deploy bitwise from the 83-column input, including P\*, T\*, an H-TOKEN32 release
  re-solved from its counts, D1 and D0. Ten refusals exit with code 2.
- **Same-device copy.** Verified (4,098 store files). Every restore from it passes: teachers, frozen bank, every
  calibrator family, P\*/T\* deployment, and P\*'s selected reader.
- **Off-device backup and predecessor custody.** PENDING (drive absent; BACKUP_VERIFICATION.json).
- **Cost.** $0 cloud; CPU in COST_AND_CLOSEOUT.md.
