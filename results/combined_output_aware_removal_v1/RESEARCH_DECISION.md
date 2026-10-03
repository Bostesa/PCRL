# Research decision: corrected benchmark and protection of the complete release

**Status:** development evidence, 2026-10-03. Every row has been used before, so none of this is fresh confirmation.

**Encoders and cells.** Six frozen PCRL Round-4 encoders were used: Adult and HMDA × seeds 0, 1, 2. The mandatory
cells are Adult income_prediction × sex and HMDA underwriting × race.

**Pins:**
- Base: 70f978ffc0afff55ecdf49cf1a74908cc3db5f49.
- Execution lock: pushed at 3130c8d4e48e4446854d0744e34b22997d985430 before any new fit. Amendment L1 at
  72445377230fd92aa47ce325c4cf10cbea1115f7 is a technical map-id fix, made before any HMDA fit.
- Independent replay: see `INDEPENDENT_VERIFICATION.json` and `VALIDATION.md`.

Three different conclusions are kept apart throughout:
- **(i)** a method fails its own intended test;
- **(ii)** its test holds, but recovery remains outside its stated scope;
- **(iii)** a separately released output bypasses protection of the features.

## 1. Which original headlines survive the corrections and the exposure sensitivity?

**Corrections** (`CORRECTION_ADDENDUM.md`, `ORIGINAL_VS_CORRECTED.csv`).
- **Custody.** The locked runner re-run from saved predictions reproduces all 24 original primary endpoints to 1e-12,
  with identical decisions.
- **Deterministic errors, both exploratory, both outside the primary family:**
  - unpaired encoder seeds in the budget-truncated marital_status noise group. 50 rows were corrected; the spurious
    "+0.014 rep+outputs beyond outputs" becomes 0.000.
  - precision loss in 9 ρ₁² rows. Values below 0.015 changed by up to 1.2 %, and the original intervals had excluded
    their own points.
- **Monte Carlo spread.** 5 exploratory bounds are simulation spread, not errors.
- **Reporting fixes:**
  - noise *retains* about 40 % of the Adult probe lift; it does not cost 40 %;
  - "the linear guarantee transfers" is replaced by the measured held-out ρ₁²;
  - "information" is replaced by "measured AUC";
  - the claim that more seeds or pairs could not change the conclusion is removed;
  - the affine-lookup claim is replaced by an observed collision-free partition;
  - the overlap rows are test-split rows across all three roles, not "assessment rows".

**Exposure sensitivity** (`EXPOSURE_SENSITIVITY.md`). The training-overlapping assessment groups were removed under a
rule fixed in advance, with every predictor held fixed.
- **24/24 primary decisions STABLE.** The largest point shift is 0.00044.
- **42/42 native-check categories STABLE**, including the 12 failures of PCRL's own check.
- **97/98 exploratory headline rows STABLE.** The exception is Adult B−A nonlinear AUC, +0.002, whose lower bound moves
  from just below 0 to just above.

**Surviving headlines:**
- the PCRL linear check fails on 12/42 pair × seed (category i);
- LEACE passes its own bound but leaves nonlinear recovery about 0.81–0.87 (category ii);
- clean outputs keep recovery at about 0.76–0.78 when the features are protected (category iii);
- noise trades away most utility;
- encoder seeds differ a lot.

**One new exposure caveat** was found after the rule was fixed. Some test rows (for example 34 Adult and 25 HMDA
assessment rows in the new roles) share a feature vector with an encoder-training row while having a different record
key. These are reported, not removed.

## 2. What changes when only the prediction decision is released?

This is primary P1, outputs-only recipient, comparing the full output with the hard prediction.

| | Adult | HMDA |
|---|---|---|
| Recovery from the full logit vector | 0.776 | 0.768 |
| Recovery from the hard prediction | 0.513 | 0.505 |
| Δ (lower bound) | +0.263 (0.249) | +0.263 (0.254) |
| P1 | **PASS** | **PASS** |

- Half-way between the two, the probability vector gives 0.705 / 0.702. The logit vector's extra per-row offset (the
  logit sum) adds about 0.07 AUC that no decision needs.
- Hard decisions are even below the Adult label-only reference (0.601).
- The task accuracy of the hard decision is *identical* to the full output's: it is the same decision. That parity
  says nothing about log loss, calibration or ranking.
- With noisy features at σ\*, the hard decision keeps recovery at 0.527 / 0.510, against 0.776 / 0.768 with full
  logits.

**Scope.** This is not a claim that hard labels solve privacy. It is a measured property of these heads and this
attacker slate.

## 3. Does FARE beat LEACE at the same task cap and under the same release contract?

The FARE nominee is chosen on validation only, among configurations within 1 point of untreated validation accuracy.

| Primary | Δ | Simultaneous lower bound (α = 0.05/12) | Adult | HMDA |
|---|---|---|---|---|
| P2 recovery, LEACE − FARE (features) | Adult +0.262, HMDA +0.363 | 0.248 / 0.356 | PASS | PASS |
| P3 accuracy, FARE − untreated | −0.0069 / −0.0036 | −0.0118 / −0.0010 | **NOT_ESTABLISHED** | PASS |
| P4 accuracy, FARE − LEACE | −0.0067 / −0.0029 | −0.0116 / −0.0001 | **NOT_ESTABLISHED** | PASS |
| P5 FARE + clean outputs − FARE + own head | +0.238 / +0.267 | 0.223 / 0.258 | PASS | PASS |
| P6 same contract (features + own head), LEACE − FARE | +0.267 / +0.362 | 0.253 / 0.355 | PASS | PASS |

**Adult: not competitive.**
- FARE lowers feature recovery from 0.812 (LEACE) to 0.550 and keeps about 92 % of the probe lift (accuracy 0.824
  against 0.831).
- Non-inferiority within 1 point is **not established**: the lower bound is −0.0118.
- Seed 2 drives this. Its validation-admissible nominee lost 2.0 points on assessment; seeds 0 and 1 lose 0.04 points.

**HMDA: formally competitive (P2 ∧ P3 ∧ P4 PASS, admissible nominees on all 3 seeds), with a material qualification.**
- HMDA's whole task gain over a constant predictor is small: untreated 0.903 against constant 0.886.
- On **seed 1** the validation-selected nominee is a **single cell**, a constant release. Its recovery is exactly 0.5
  and its accuracy equals the constant predictor (0.886). That encoder's untreated gain is only 0.6 points, so the
  1-point cap admits it.
- On **seeds 0 and 2** the nominees have 4 cells. They keep the untreated accuracy to within 0.2–0.3 points (0.905 /
  0.907) at chance-level recovery (0.504 / 0.498). That part is a genuine, favourable trade-off on this task.
- So the HMDA pass reflects a task with little lift to protect, partly met by withholding everything. It is not
  evidence of a generally competitive method.

**The fairness weight matters.** The zero-fairness twin uses the same tree budget with γ = 0. It keeps recovery at
0.645 (Adult) and 0.607 (HMDA) at untreated accuracy. Compression alone therefore removes part of the recovery, and
the fairness weight removes the rest down to about 0.55 / 0.50.

**Same contract.** With features plus a head fitted only on those features (P6), FARE's complete release is lower
than LEACE's by 0.27 (Adult) and 0.36 (HMDA). LEACE's own-head outputs alone are recovered at 0.755 (Adult) and
0.693 (HMDA); FARE's own-head outputs alone at 0.548 / 0.501.

**Bypass (category iii).** FARE features plus the historical clean outputs recover at 0.788 / 0.768, the same as
outputs alone. Changing the features cannot remove what an unchanged output carries.

## 4. Is FARE's native certificate meaningful and applicable?

**Mostly not, in this study** (`NATIVE_TEST_VS_RECOVERY.md`).
- In the original run every certificate was UNAVAILABLE, because of an over-strict duplicate guard in the wrapper.
- After amendment A1:
  - Adult bounds are 0.50 / unavailable / 0.50;
  - HMDA all-pairs bounds are unavailable or vacuous (3.18);
  - the declared race {0, 1, 2} secondary gives 1.05 / **0.175** / 0.95. The only informative value (0.175) belongs to
    the degenerate single-cell nominee.
- The cause is about 1,400–1,500 certification rows split in half. This is a limitation of this study's roles, not
  evidence that FARE is unsafe.
- Even where available, the certificate bounds demographic-parity distance for cell-only classifiers. It is not an
  attack-AUC bound and does not cover the outputs bypass.

## Strongest comparisons

**Strongest favourable.** On HMDA seeds 0 and 2, official FARE with 4 cells keeps untreated task accuracy to within
0.3 points while the measured sensitive recovery falls to chance (0.50). Target-only LEACE stays at 0.86 at the same
accuracy. Under the same output contract (P6) FARE is lower by 0.36.

**Strongest adverse.** Any defense applied to the features is bypassed by the historical clean output: recovery
returns to 0.77–0.79 for FARE and for noise. On Adult, FARE's accuracy is not shown to be within one point. FARE's
certificate is unavailable or vacuous at this sample size.

## Counts

| Item | Count |
|---|---|
| Units planned | 378: 63 per encoder seed × 6. Per seed: reference 1, outputs-only 3, A/B/C 9, heads + head outputs + view 3 for A/B 6, noise 15, FARE fits 7, FARE grid 12, zero-fairness twin 6, nominee views 4 |
| Completed | 372 units (private run, hash-complete), plus 14 real-data control runs |
| Failed | 0 scientific units. One HMDA worker attempt failed before any defense or attacker fit (map-id bug); it was repaired (amendment L1) and resumed. |
| Aliased | 3 FARE configurations (HMDA s0 6→5; s1 5→4, 6→4) = 6 attack/probe units correctly not refitted (378 − 6 = 372) |
| Budget-unrun | 0 |
| Technically unavailable | FARE certificates as tabulated (native test only) |

**Primary family.** 12 rows, 12 decided:
- 10 PASS;
- 2 NOT_ESTABLISHED (Adult P3, P4);
- 0 UNRESOLVED.

## Registered predictions

These were registered before any fit and are scored here.

| # | Adult | HMDA |
|---|---|---|
| H1 hard lowers ≥ 0.02 (P = 0.85) | true | true |
| H2 FARE < LEACE by ≥ 0.02 (0.80 / 0.75) | true | true |
| H3 accuracy within 0.01 of untreated (0.35 / 0.40) | false | true |
| H4 within 0.01 of LEACE (0.35 / 0.40) | false | true |
| H5 own head < clean by ≥ 0.02 (0.85 / 0.80) | true | true |
| H6 same contract, FARE < LEACE (0.75 / 0.70) | true | true |
| H7 clean outputs bypass FARE, > 0.55 (0.9) | true | true |
| H8 competitive conjunction (0.25) | false | true, formally (see the qualification) |
| H9 defense-aware attacker stronger (0.30) | false: cell-conditional = standard slate | false |
| H10 exposure changes no primary decision (0.95) | true | true |

## Limits

- **Data.** Development data that has already been used.
- **Exposure.** defense_fit = the encoders' training rows. There are feature-identical rows across roles (reported).
- **Scope.** One purpose and protected attribute per dataset is mandatory.
- **HMDA race.** Classes 3 and 4 are not estimable.
- **Inference.** Intervals are conditional on the fitted predictors.
- **FARE.** It is applied to stored features, not to raw inputs as in the paper.
- **Kernelized adversarial concept erasure.** Optional and not run.
- **Repeated queries.** Outside the fixed-release protocol.

## Next decision

The results support one specific, already-published direction:

> **Release contracts that do not ship the clean task output.** Hard decisions, or heads computed only from the
> protected features, combined with a coarse fair-partition representation.

This should be tested where the task has real lift to protect. On HMDA underwriting the lift is too small to
discriminate between methods.

**Concretely.** Before any method paper, decide whether to:
- (a) evaluate the same matched protocol on a cell with larger task lift and larger certification samples, so the FARE
  certificate can be non-vacuous; and
- (b) treat the output contract (full logits vs probability vs decision) as a first-class variable in the combined
  paper.

No new algorithm is suggested by these results.
