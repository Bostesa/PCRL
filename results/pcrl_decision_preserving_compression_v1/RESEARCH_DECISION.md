# Research decision: decision-preserving joint compression

## Plain-language answer

**What was shared.** Each recipient received a public code instead of the teacher's continuous scores:
- a token, from at most m = 2, 4 or 8 states per predicted class;
- that token's public decoded probability vector;
- the decision.

**Did the prediction classes stay identical? Yes, exactly.** All 258 fitted policies reproduce their own teacher's predicted class for every person in every role. That is 20.2 million row checks, verified independently. Accuracy, confusion matrices and every class recall are therefore unchanged.

**How much useful confidence was lost? Too much for occupation.** Even the finest code (8 states per class, U teacher) adds about +0.020 nats of occupation log loss and +0.007 Brier on the assessment. The registered allowances are 0.01 and 0.005. Income confidence is nearly free: +0.001 nats for that task-only code and +0.002 for the joint code.

No finite code met the registered score contract on inner selection on every seed, and the class-only (decision-only) code fails it by about 0.1 nats. The only release meeting the contract is U's continuous output.

**Did the joint design beat the strongest non-joint control? No.**
- No joint policy was a valid nominee.
- Its descriptive fallback (U, JOINT, m 8, λ 0.1) leaked 0.018 less on the pair than the matched task-only code, under the required 0.02 margin. It was only marginally better than the two sequential orders (pair 0.807 vs 0.807 and 0.809).
- It leaked 0.051 less than the continuous scores. Those scores, however, keep the confidence quality the code loses.

**What is runnable.**
- The truthful baseline is U's continuous output, the only release meeting the contract.
- The best compact code, U JOINT m8 λ 0.1, is packaged as a runnable policy. It is labelled NOT ELIGIBLE: it preserves every decision but fails the occupation confidence allowance.
- `python -m dpc.deploy` takes the 83 permitted columns and returns only tokens, decoded probabilities and decisions.

**Label: EXPERIMENTAL_NO_ADVANTAGE.** The study is complete and valid; this is a negative result.

This is an exploratory, locked benchmark on 13,936 previously used Adult rows. Its intervals condition on the fitted artifacts and do not correct for the adaptive research history. It is not fresh confirmation or a population guarantee.

## Results (assessment means over seeds 0–2; FINAL attack slate on every row; lower AUC = less SEX recovery)

| Release | Pair AUC | Income-recipient AUC | Occupation-recipient AUC | Log loss (income / occupation) | Brier (income / occupation) | Accuracy (income / occupation) |
|---|---:|---:|---:|---|---|---|
| U: features + scores (old full view) | 0.883 | 0.859 | 0.878 | — | — | 0.844 / 0.475 |
| U: continuous scores (T\* = C_global; composed code readers) | 0.858 | 0.697 | 0.856 | 0.338 / 1.269 | 0.214 / 0.652 | 0.844 / 0.475 |
| U FINE-TASK m8 (task-only code; C_match fallback) | 0.825 | 0.697 | 0.786 | 0.339 / 1.289 | 0.215 / 0.660 | identical to U |
| U LOCAL m8 λ 0.1 (P\* fallback) | 0.820 | 0.695 | 0.770 | 0.339 / 1.289 | 0.215 / 0.660 | identical to U |
| U SEQ-12 / SEQ-21 m8 λ 0.1 | 0.809 / 0.807 | 0.695 / 0.695 | 0.763 / 0.752 | 0.339 / 1.291 and 0.340 / 1.292 | 0.215 / 0.660 and 0.215 / 0.661 | identical to U |
| **U JOINT m8 λ 0.1 (J\* fallback)** | **0.807** | 0.696 | 0.764 | 0.340 / 1.290 | 0.215 / 0.660 | identical to U |
| U class-only (decisions alone) | 0.739 | 0.587 | 0.687 | 0.423 / 1.379 | 0.257 / 0.693 | identical to U |
| RAW-J β 0.3 continuous scores | 0.788 | 0.685 | 0.774 | 0.335 / 1.282 | 0.212 / 0.659 | 0.845 / 0.466 |
| RAW-J FINE-TASK m8 | 0.752 | 0.684 | 0.704 | 0.336 / 1.300 | 0.214 / 0.667 | identical to RAW-J |
| FARE (official) | 0.704 | 0.685 | 0.636 | 0.352 / 1.315 | 0.224 / 0.676 | 0.844 / 0.451 |
| F0 (no-fairness FARE) | 0.865 | 0.803 | 0.851 | 0.327 / 1.298 | 0.209 / 0.665 | 0.849 / 0.460 |
| LEACE (official) | 0.808 | 0.554 | 0.801 | 0.454 / 1.320 | 0.291 / 0.680 | 0.794 / 0.436 |

Full levels, bootstrap SEs and every view family are in `ALL_LEVELS.csv` and `COALITION_AND_RATE_RESULTS.csv`. The figure is `figures/fig_tradeoff.pdf`.

## Primary claims (33 slots, z = 3.1717657833516224, B = 1,999 exact-record-group bootstrap)

All three claims are NOT_ESTABLISHED: J\* and P\* had no feasible nominee, so every clause is DESCRIPTIVE_ONLY. Read on the numbers alone, each fallback fails on occupation confidence:

| Claim | Fallback | Coalition gain (interval) | Locals | Occupation log loss vs U | Occupation Brier vs U | Clauses passing (numeric) |
|---|---|---|---|---|---|---|
| A: J\* vs C_match (U FINE-TASK m8) | U JOINT m8 λ 0.1 | 0.0176 [0.014, 0.021]; fails the 0.02 margin | pass | +0.0214 [0.015, 0.028] | +0.0079 [0.0055, 0.0103] | 8/11 |
| B: J\* vs C_global (U continuous) | U JOINT m8 λ 0.1 | 0.0515 [0.045, 0.058] | pass | +0.0214 | +0.0079 | 9/11 |
| C: P\* vs T\* (U continuous) | U LOCAL m8 λ 0.1 | 0.0389 [0.033, 0.045] | pass | +0.0209 [0.014, 0.027] | +0.0075 [0.005, 0.010] | 9/11 |

## Explanatory findings (descriptive)

**Views under equal attack strength (U).**

| View | Pair AUC |
|---|---:|
| Features + scores | 0.883 |
| Scores alone | 0.858–0.864 (interface / scores family) |
| Compact token + decoded score | 0.789–0.825 (task-only m2–m8), 0.807 (joint m8) |
| Decisions alone | 0.739 |

The joint decision vector alone already reveals SEX at 0.74, and no confidence coarsening can remove that.

**Individual versus pair recovery.**
- Coding mostly protects the occupation recipient: its AUC falls from 0.856 (scores) to 0.786 (task-only m8) and 0.764 (joint m8).
- The income recipient's score view is already low (0.697) and barely changes.
- The pair follows the occupation recipient.

**Plain compression versus privacy-trained compression.**
- At m 8, privacy training lowers pair AUC by 0.018 (joint) or 0.005 (local) for about 0.001 nats extra occupation log loss.
- Joint m8 dominates task-only m4 on both pair AUC (0.807 vs 0.811) and occupation log loss (1.290 vs 1.303).
- So the sensitive objective adds something beyond compression. It is small, below the 0.02 margin, and inside a contract every code fails.

**Joint versus local versus sequential (U, m8, λ 0.1).**

| Family | Pair AUC | Occupation log loss |
|---|---:|---:|
| JOINT | 0.807 | ≈ 1.290 |
| SEQ-21 | 0.807 | ≈ 1.292 |
| SEQ-12 | 0.809 | ≈ 1.291 |
| LOCAL | 0.820 | ≈ 1.289 |

Joint fitting is marginally below both sequential orders: by 0.0004 and 0.0016 on the assessment. On inner selection it is lowest in 11 of 18 matched teacher/rate/λ cells, by less than 0.007 in every one. This is a tie for practical purposes, not a demonstrated advantage. Coordinating the two maps (joint or sequential) helps relative to independent local maps: joint is below local in 14 of 18 inner cells.

**Fitted objective versus held-out recovery.**
- At U m8, JOINT λ 0.1 lowers fitted I(S; C1, C2) from 0.201 to 0.178 nats, and inner pair AUC falls 0.825 → 0.805.
- λ 10 lowers I12 to 0.112 and inner pair AUC to 0.757, but distortion rises sharply (income D1 0.002 → 0.068, occupation D2 0.034 → 0.074) and the score contract fails further.
- The plug-in MI and the attacks move together here. Neither is a privacy guarantee.

**Confidence cost of m = 1.** +0.085 nats (income) and +0.110 nats (occupation). The decision-only release is not confidence-preserving.

**Teacher weak classes (unchanged by construction).**
- U's occupation recalls (range over seeds) are 0.495–0.499 / 0.466–0.483 / 0.390–0.400 / **0.020–0.024** / 0.703–0.709 / **0.000** for classes 0–5, with balanced accuracy 0.417–0.421. Class 5 is never predicted by any teacher.
- RAW-J is similar (class 3 recall 0.015–0.028) and is 0.9 occupation-accuracy points below U.
- A code that preserves predictions cannot repair these recalls.

**RAW-J teacher codes.** They leak less (pair 0.737–0.755 for m 2–8; 0.693 class-only), but they inherit RAW-J's lower occupation accuracy and its +0.013-nat occupation log-loss gap. Every one fails the U-relative contract.

**References.**
- FARE leaks least (pair 0.704) but loses 2.4 occupation points and 0.046 nats.
- LEACE loses 5 income points.
- F0 keeps utility and leaks like U.
- All three are task-ineligible.

## Registered predictions (`PREDICTIONS.json`, committed at `325cdbb` before any fit)

| ID | Prediction | Registered probability | Outcome |
|---|---|---|---|
| PR1 | A compact code meets the full inner contract on every seed | 0.55 | **No.** All fail occupation log loss and/or Brier on some seed |
| PR2 | An eligible privacy-trained code beats T\* by > 0.02 pair AUC | 0.25 | **No.** None is eligible |
| PR3 | JOINT beats both sequential orders at matched teacher/rate (inner pair AUC, any margin) | 0.45 | **Yes, negligibly.** At the J\* fallback cell, 0.805 vs 0.812 / 0.811; in 11 of 18 cells overall, every margin < 0.007 |
| PR4 | The strongest control is a simple compression, not a continuous source or reference | 0.75 | **No.** T\* = C_global = U continuous, because no code was eligible |
| PR5 | Label EXPERIMENTAL_NO_ADVANTAGE | 0.70 | **Yes** |

The main miss was PR1 and PR4: we expected 8 cells per class to keep occupation confidence within 0.01 nats, and it did not.

## Process and validity

| Lock | Commit |
|---|---|
| ENGINEERING_LOCK | `0b4753b` |
| TRAINING_LOCK (full bank from synthetic timing) | `01f97f2` |
| SELECTION_AND_AUDIT_LOCK | `31e2f90` |
| AMENDMENT_A1 (controls-stage plant bits; the first controls run crashed before writing anything) | `7049c5d` |
| AMENDMENT_A2 (evaluation-lock writer; failed before writing) | `919e524` |
| EVALUATION_LOCK | `3140d4c`, committed 01:18:31Z and on origin before the assessment opened at 01:18:45Z |

**Ordering.** Predictions were registered at `325cdbb` before any fit, and every lock was on origin before its stage.

**Review.** The math review found 0 REQUIRED defects; its three recommendations (R1–R3) were applied before the selection lock, and 52 of 52 non-equivalent injected defects were caught.

**Controls.** All pass. Real-data nulls show 0 of 15 exceedances (held-out mean AUC 0.500), and the within-class confidence, decoder-collision and coalition XOR plants are detected. A decisions-only audit misses the confidence plant, as designed.

**Independent verification.** Two phases, own code with no study imports: 0 FAIL. The single WARN was 11 rounding or transcription slips (each ≤ 0.0025) in these documents, now corrected. See `VALIDATION.md` and `INDEPENDENT_VERIFICATION.json`.

**Matched-control coverage failure (claim A).** C_match is NO_FEASIBLE_CONTROL at all six teacher/rate cells. Every matched non-joint code fails the score contract, for the same occupation-confidence reason as the joint codes. The locked per-claim rule (review R3, `dpc/infer.py`, locked before selection) counts only an INVALID_* (uncomputable) comparator as missing. So claim A is a valid NOT_ESTABLISHED and the computed label is EXPERIMENTAL_NO_ADVANTAGE. PROTOCOL §8's sentence "A missing comparator invalidates the dependent claim" can also be read to cover NO_FEASIBLE_CONTROL. On that stricter reading, claim A is invalid and the overall label would be INCOMPLETE_OR_INVALID. Neither reading changes the outcome. No JOINT policy is task-eligible, so claim A could not pass against any comparator. Claims B and C have a valid, eligible comparator (U continuous) and are NOT_ESTABLISHED. The locked label is kept, not re-chosen after the outcome. J\*'s guard against C_match is recorded as missing (`SELECTION.json` `missing_guards`), not silently dropped.

**Process deviation (disclosed).** On two occasions, for about 2 and 4 minutes, a third heavy process (the verifier) overlapped the two study workers.

**Custody.** The external drive was absent, so there is a same-device copy with full restore checks only. Off-device backup and the predecessor (osf/smf) custody repair are pending, with exact commands recorded.

## Prior art and claims

Agglomerative and multivariate information bottleneck, the privacy funnel, output transformation, data processing and multi-recipient release are established (`PRIOR_ART_AND_BASELINE_GAPS.md`).
- The official PURIFIER repository is empty.
- No Taylor et al. solver code exists.
- Neither could be added as a correctly scoped comparison, so no claim of beating them is made.
- The SEQ arms are matched adaptations, not Taylor's algorithm.
- No novelty is claimed.

## Decision

1. **Record a valid negative.** Under this prediction-preserving contract, compressing the teachers' confidence into public class-preserving codes reduces SEX recovery (pair −0.04 to −0.05 vs continuous scores). It cannot do so while keeping occupation confidence within 0.01 nats and 0.005 Brier of U: every finite code costs at least about 0.02 nats.
2. **No component established a development advantage.** Joint fitting is at most negligibly better than sequential fitting, and privacy training adds little beyond plain compression.
3. **No confirmation population is spent.** Any follow-up would need a registered change of contract, for example a richer per-class code for occupation or a different occupation teacher. It must not relax this study's allowances after the fact.
