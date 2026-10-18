# Review: proposed training-size diagnostic (P70 / P70B witness fit)

Statistical reviewer, 2026-10-18 about 20:05Z.
- This review comes before the proposed fit, and it changes no frozen rule, gate, partition or outcome.
- It covers MODEL_SELECT development evidence only (adaptively used).
- **The D-stage outcome `NO_RECIPE` stands whatever P70 shows.**

## Verdict

**Justified, as a disclosed adaptive diagnostic with no method claim.** It addresses the question that decides how `NO_RECIPE` should
be read: is the missing occupation-side headroom a design consequence of the smaller training sets, or was the source witness's
margin partly post-selection optimism?
- It costs one fitted release specification (P70 and P70B are one fit evaluated against both seeds' references).
- It uses no closed role and keeps the repair panels C0 / C1 excluded.
- It needs the five conditions below.

## One correction to the framing

"The selection effect is present only at 100%" holds only for the **fit-specific** part of the selection effect. A1_PO_16 was chosen
as a *recipe* on MODEL_SELECT, and every refit is scored on the same MODEL_SELECT people. A recipe × MODEL_SELECT interaction
(optimism shared by every fit of the recipe) is therefore present at 42%, 53% and 70% as well. The curve can speak only to the part of
the source's margin that belongs to its particular fit.

## Explanations still in play

| Explanation | Kind | Can P70 separate it? |
|---|---|---|
| (i) More training data for the trees | size | partly, through the curve shape |
| (ii) Fit-specific regression to the mean from selecting among ten learners | selection | **not directly**; no clean held-out set exists for the source fit (it saw C0 / C1; DEV_REPORT and ATTACK labels are closed) |
| (iii) Dictionary collapse: with fewer FIT people, more COW / SCHL / MAR / POBP levels (POBP: 106) fall below 10 people and become OTHER, which erodes the POBP-ordinal benefit | size, but step-like | only if recorded (condition 3) |
| (iv) Other differences of the source fit: the panels' households were in its training set, its T* folds differed, and its seed-1 model is the seed-0 path truncated | pipeline | no |

## What the existing numbers already say

These are descriptive, with seed-0 / seed-1 references shown as s0 / s1.

**From 42% to 53% the curve is flat.**
- Men's guard: +0.0004 (s0) and -0.0003 (s1).
- Occupation NLL: -0.0002 and -0.0016.

**From 53% to 100% it jumps.**
- Men's guard: -0.0050 and -0.0042.
- NLL: -0.0101 and -0.0107.

**What a smooth size effect would predict.** The 53% → 100% step is about 3 times the log-size distance of the 42% → 53% step. A
curve linear in log training size would therefore predict a 53% → 100% change of about three times a near-zero change, not
-0.005 / -0.010. This already argues against (i) alone, but only weakly:
- each size has one fit per seed;
- the fit-to-fit spread at a fixed size is unknown;
- the seed comparison is confounded with the seed-specific references.

## Conditions

**1. Register the bet before the fit, in the commit that precedes it.** Record the predicted P70 values under each reading. P70 sits
about 45% of the log-size distance from 53% to 100%.

| Reading | P70 prediction |
|---|---|
| (i) alone, log-linear between the 53% and 100% points | men's guard about -0.0011 (s0) / 0.0000 (s1); NLL about -0.0105 / -0.0069 |
| "Flat beyond 53%" (ii, iii or iv) | about the 53% values: +0.0011 / +0.0019; NLL -0.0060 / -0.0021 |

Register the reading rule with these predictions:
- *Near the log-linear prediction* (within the fit-to-fit spread of condition 2): consistent with a size effect being sufficient. It
  **cannot exclude** a fit-specific selection component at 100%.
- *Near the 53% values*: the 100% advantage is not explained by a smooth size effect over 42-70%. It is consistent with (ii), with a
  step-like size effect between 70% and 100% such as (iii), or with (iv). **It cannot distinguish among them.**
- *At or beyond the source values*: selection is not needed to explain the source margin at 70% or more. The 42-53% shortfall is a
  training-size effect of the partition design.

**2. Measure differences between fits on the same people.**
- Compute paired differences on MODEL_SELECT with household-clustered paired SEs: P70 − R0, P70 − D0 and source − P70, per seed
  reference, for the 8 occupation-side conditions.
- Do not compare absolute values with their roughly 0.0018 SEs, because most MODEL_SELECT noise is shared across the fits.
- Use P70B − P70 (the same fit) to estimate the pure reference contrast.
- Estimate the fit-to-fit spread at a fixed size from (D1 − D0) and (R1 − R0), each minus that contrast (one degree of freedom each).
- Name the spread as the yardstick for "near" in condition 1.

**3. Record the dictionary label-free at each size (42%, 53%, 70%, admitted 100%).**
- Record the levels collapsed to OTHER per block, and the share of MODEL_SELECT people whose POBP level is collapsed.
- If POBP collapse differs sharply between 70% and 100%, explanation (iii) stays live.
- A second P70 fit using the admitted (frozen, label-free) dictionary would separate (iii) from (i). It is optional, and it is
  acceptable only if registered together with P70, never added after seeing P70.

**4. Keep the diagnostic out of selection.**
- No P70 output may become a witness, a baseline slack, a recipe input or a rule input.
- It may not trigger a rerun of the D or R stages, and it may not re-label `NO_RECIPE`.
- Any follow-up built on a larger training set needs its own registered amendment, disclosed as adaptive and motivated by D-stage
  outcomes.
- The pool's GUARD and CHECK households are inside P70's training set, so no within-pool held-out set would remain.

**5. Log it and count it.**
- Use a new partition purpose in the access log ("training-size diagnostic"), with the panels refused.
- Count one fitted specification; P70B is an evaluation alias, not a second fit.
- Label it ADAPTIVE (motivated by observed D / R values) in every table.

## What may and may not be concluded

**May.** Descriptive statements of the form: "On MODEL_SELECT, refits of the fixed A1_PO_16 recipe at 42%, 53% and 70% of
MODEL_FIT had occupation-side margins X, Y and Z, against W for the selected source fit at 100%." The shape and its paired
differences may be reported, with the reading rule of condition 1 applied as registered.

**May not.**
- A population learning curve or a causal attribution to training size.
- A statement that selection explains, or does not explain, the source margin (only "consistent with").
- Any claim that a 100%-size refit would restore the reserve (no such refit is permitted while the panels are excluded).
- Any change to the rules, the partitions or the `NO_RECIPE` outcome.
- Treating a monotone curve as excluding (ii), or a jump only at 100% as proving it. A jump is also consistent with (iii) or (iv).
