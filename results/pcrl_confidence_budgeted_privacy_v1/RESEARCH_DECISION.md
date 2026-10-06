# Research decision: confidence-budgeted privacy compression (cbp)

**Label: CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION.** This is an exploratory, locked development comparison
on reused Adult rows. It is not confirmation, and it is not a population privacy guarantee. See the exposure statement in
EXPOSURE_LEDGER.md.

| Claim | Status | Root cause |
|---|---|---|
| A: J\* vs C_rate | NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE | LOCAL_GUARD_FAILURE (no JOINT configuration met headroom plus its guards) |
| B: J\* vs C_global | NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE | LOCAL_GUARD_FAILURE |
| C: P\* vs T\* | NOT_ESTABLISHED | MEASURED_VIOLATION_SUPPORTED_BY_BOUND on P23 (the pair benefit); the other 10 clauses PASS |
| Q: DIRECT-TASK i8o64 | PASS | All four confidence bounds pass |

Sources: LABEL_RESULT.json, PRIMARY_ENDPOINTS.csv, SELECTION.json. Every claim was technically valid (controls all_ok,
endpoint parity 33/33, inner validation 129/129, all assessment arrays finite).

## 1. In plain words

- **The confidence budget held. The privacy benefit did not survive it.**
  - The headroom rule did what it was designed to do. The selected privacy code kept occupation confidence inside the
    original allowance on the assessment: the upper bound was 0.00674 nats against the 0.01 limit, which the previous
    study could not establish.
  - The only code that met the headroom rule and the local guards was the weakest one on the grid, JOINT λ 0.01. Its
    combined-view SEX AUC was 0.0022 below strong task-only compression (FINE-TASK). The registered criterion needs a
    lower bound above 0.02.
- **No intermediate λ kept confidence while reducing recovery by a useful margin.**
  - Inner rows: at every λ ≥ 0.025, every family's worst-seed occupation log-loss excess exceeded the 0.006
    headroom. The only exception was JOINT λ 0.04 (Figure 1). Privacy was bought with occupation confidence almost linearly.
- **No family won the method criterion, and joint added nothing.**
  - No JOINT configuration passed headroom plus its local guards against the strongest nonjoint controls.
  - JOINT λ 0.1 and SEQ-21 λ 0.1 remain within 0.003 pair AUC of each other.
- **Decision (prompt §17): close this interpolation and selection line.** Headroom removed the benefit. We do not recommend
  another slightly adjusted λ on these rows.

## 2. Inner selection (INNER_SELECTION only; SELECTION.json, INNER_SELECTION_TABLE.csv, Figure 1)

**Roles**

- **Ordinary eligibility:** holds for every privacy code except SEQ-21 λ 0.04, LOCAL λ 0.08 and LOCAL λ 0.1. Each of
  those fails on some seed.
- **Headroom (log-loss excess ≤ 0.006 and Brier excess ≤ 0.0035 for every task and seed):** passed only by the four
  λ 0.01 maps and by JOINT λ 0.04.
- **T\* = U|FINE-TASK|i8o64.** RAW-J is excluded from T\* by the registered correction. C_rate = C_global =
  U|SEQ-21|i8o64|l0.1. Q = U|DIRECT-TASK|i8o64.
- **P\* = U|JOINT|i8o64|l0.01.**
  - Its inner mean pair AUC was 0.8446, against 0.8459 for T\*.
  - Its alias set is itself. It equals SEQ-12 λ 0.01 on seed 0 only, so the alias is partial.
- **Why the stronger headroom-eligible candidates were not nominated.** The guard is local AUC ≤ T\* + 0.005 for each
  recipient and seed.
  - **JOINT λ 0.04** (inner pair 0.8350): recipient 1 on seed 2 was +0.0119 above T\*.
  - **SEQ-21 λ 0.01** (inner pair 0.8420): recipient 1 on seed 2 was +0.0052, missing the guard by 0.0002.
  - **LOCAL and SEQ-12 λ 0.01:** recipient 1 was +0.006 to +0.008 above T\* on seeds 1 and 2.
- **J\* has no eligible nominee (LOCAL_GUARD_FAILURE).**
  - The only headroom-eligible JOINT codes are λ 0.01 and λ 0.04.
  - Their occupation-recipient AUC is about 0.03 to 0.06 above C_rate, which is SEQ-21 λ 0.1, on every seed.
  - The fixed fallback, JOINT λ 0.04, was scored DESCRIPTIVE_ONLY.

**Prespecified diagnostics (HEADROOM_VS_STANDARD_SELECTION.csv)**

- Without headroom, the ordinary privacy winner under the T\* guard is JOINT λ 0.1. Without any guard it is SEQ-21 λ 0.1.
- Headroom changed the winner. It gave up 0.0280 mean inner pair AUC: 0.0355, 0.0308 and 0.0176 on seeds 0, 1 and 2.
- Family headroom winners:
  - JOINT: λ 0.01.
  - LOCAL, SEQ-12 and SEQ-21: none eligible (LOCAL_GUARD_FAILURE). Their fallbacks are at λ 0.01.

**Interpretation note (statistics reviewer, SEL-N4)**

- The inner estimates use 2,235 rows, so the SE of the inner log-loss excess is about 0.003. The seeds share those rows.
  The per-seed rule therefore does not average the noise away, and per-seed local guards at +0.005 are noisy.
- This is a property of the registered rule, not a reason to relax it.

## 3. Locked assessment (ASSESSMENT_COMPARISON.csv, PRIMARY_ENDPOINTS.csv; 13,936 rows, 13,929 groups)

**Setup**

- Absolute AUCs are seed means of SEX AUC from the inner-AUC-selected attacker, with the score's orientation fixed.
- Accuracy is identical to U for every U-derived code: income 0.843953, occupation 0.475387. Decisions are preserved
  exactly.
- Excess means the excess over the U continuous teacher.

| Release | Role | Inner ord / head | Pair AUC | Recipient 1 AUC (income) | Recipient 2 AUC (occupation) | Occ. LL excess | Inc. LL excess | Occ. Brier excess |
|---|---|---|---|---|---|---|---|---|
| U continuous | baseline | yes / yes | 0.8585 | 0.6972 | 0.8563 | 0 | 0 | 0 |
| DIRECT-TASK i8o64 | Q | yes / yes | 0.8491 | 0.6957 | 0.8342 | 0.00295 | 0.00083 | 0.00166 |
| FINE-TASK i8o64 | T\* | yes / yes | 0.8462 | 0.6957 | 0.8310 | 0.00308 | 0.00110 | 0.00144 |
| JOINT λ 0.01 | **P\*** | yes / yes | 0.8440 | 0.6952 | 0.8274 | 0.00337 | 0.00046 | 0.00151 |
| LOCAL λ 0.01 | LOCAL fallback | yes / yes | 0.8439 | 0.6977 | 0.8296 | 0.00307 | 0.00101 | 0.00151 |
| SEQ-12 λ 0.01 | SEQ-12 fallback | yes / yes | 0.8435 | 0.6956 | 0.8265 | 0.00343 | 0.00122 | 0.00153 |
| SEQ-21 λ 0.01 | SEQ-21 fallback | yes / yes | 0.8432 | 0.6977 | 0.8250 | 0.00361 | 0.00057 | 0.00163 |
| JOINT λ 0.04 | J\* fallback (descriptive) | yes / yes | 0.8329 | 0.6969 | 0.8109 | 0.00422 | 0.00099 | 0.00174 |
| SEQ-21 λ 0.1 | C_rate = C_global | yes / no | 0.8157 | 0.6945 | 0.7711 | 0.00838 | 0.00180 | 0.00327 |
| SEQ-12 λ 0.1 | source control | yes / no | 0.8160 | 0.6963 | 0.7824 | 0.00825 | 0.00109 | 0.00320 |
| JOINT λ 0.1 | ordinary winner, source control | yes / no | 0.8125 | 0.6948 | 0.7823 | 0.00807 | 0.00137 | 0.00292 |
| CLASS-ONLY | decisions only | no / no | 0.7391 | 0.5871 | 0.6866 | 0.1102 | 0.0846 | 0.0411 |
| RAW-J β 0.3 | historical privacy-trained | no / no | 0.7884 | 0.6846 | 0.7743 | 0.0134 | −0.0034 | 0.0065 |
| FARE (REF\|F) | reference | no / no | 0.7042 | 0.6851 | 0.6356 | 0.0460 | 0.0133 | 0.0233 |
| no-fairness FARE (REF\|F0) | reference | no / no | 0.8647 | 0.8034 | 0.8509 | 0.0299 | −0.0115 | 0.0122 |
| LEACE (REF\|E) | reference | no / no | 0.8078 | 0.5538 | 0.8011 | 0.0518 | 0.1155 | 0.0274 |

**Claim C (P\* vs T\*), point [lower, upper]** with z = 3.2048, B = 1999 and exact-record-group bootstrap seed 20261008:

| Slot | Quantity | Target | Point [lower, upper] | Outcome |
|---|---|---|---|---|
| P23 | pair benefit | lower > 0.02 | 0.00217 [0.00019, 0.00415] | **MEASURED_VIOLATION** |
| P24 | recipient-1 AUC change | upper < 0.01 | −0.00056 [−0.00260, 0.00149] | PASS |
| P25 | recipient-2 AUC change | upper < 0.01 | −0.00362 [−0.00559, −0.00164] | PASS |
| P26–P27 | accuracy identity | lower > −0.01 | 0, SE 0 | PASS |
| P28 | income LL excess | upper < 0.01 | 0.00046 [−0.00101, 0.00192] | PASS |
| P29 | occupation LL excess | upper < 0.01 | 0.00337 [−0.00000, 0.00674] | PASS |
| P30 | income Brier excess | upper < 0.005 | 0.00045 [−0.00002, 0.00092] | PASS |
| P31 | occupation Brier excess | upper < 0.005 | 0.00151 [0.00048, 0.00254] | PASS |
| P32–P33 | retention | lower > 0 | 0.01964, 0.03823 | PASS |

**Q (DIRECT-TASK i8o64)**

| Slot | Quantity | Point | Upper bound |
|---|---|---|---|
| P34 | income LL excess | 0.00083 | 0.00234 |
| P35 | occupation LL excess | 0.00295 | 0.00614 |
| P36 | income Brier excess | 0.00053 | 0.00103 |
| P37 | occupation Brier excess | 0.00166 | 0.00266 |

All four pass.

**Seeds (ALL_LEVELS.csv)**

| Release | Pair AUC, seeds 0, 1, 2 | Occupation LL excess, seeds 0, 1, 2 |
|---|---|---|
| P\* | 0.8476, 0.8461, 0.8382 | 0.0018, 0.0044, 0.0039 |
| T\* | 0.8515, 0.8454, 0.8415 | — |

The pair benefit (T\* minus P\*) is not consistent in sign across seeds: 0.0039, −0.0007 and 0.0033 on seeds
0, 1 and 2. On seed 1, P\* is slightly more recoverable than T\*.

**Consistency with the source.** The assessment reproduces the qpc context values:
- U continuous 0.8585 (source 0.858);
- JOINT λ 0.1: pair 0.8125 (source 0.813) and occupation LL excess 0.00807 (source 0.008072);
- FINE-TASK pair minus JOINT λ 0.1 pair = 0.0336 (source 0.033627).

## 4. Strongest favourable and strongest adverse results

- **Favourable.**
  - Confidence-budgeted selection delivered what it promised on confidence. P\* passed every confidence and utility
    clause (P24–P33), including the occupation log-loss bound that the source's JOINT λ 0.1 failed.
  - Q again passed all four confidence bounds.
  - The descriptive JOINT λ 0.04 code also stayed inside the confidence allowance, with occupation LL excess 0.00422.
    It was 0.0133 below T\* in pair AUC.
- **Adverse.**
  - The privacy benefit is essentially gone. P23 = 0.00217, with an upper bound of 0.00415, so a violation of the 0.02
    target is supported by the interval.
  - Even the most private headroom-eligible code, JOINT λ 0.04, which is descriptive only, is 0.0133 below T\*. That is
    below the 0.02 point target.
  - On these rows, the useful privacy benefit seen at λ 0.1 (about 0.034) costs about 0.008 nats of occupation
    confidence, and the 0.006 budget does not leave room for it.

## 5. Bottleneck (prompt §13)

- **Headroom passed, but the privacy margin disappeared.** P\* exists and its confidence bound passes, but its pair
  benefit is 0.002.
- **Local guards decided the nominee.** Without the per-seed recipient-1 guard, JOINT λ 0.04 (inner pair 0.835) would
  have been P\*. Its assessment pair benefit, 0.013 descriptive, still falls short of 0.02.
- **Joint remained tied with sequential.**
  - At λ 0.1, JOINT and SEQ-12/SEQ-21 differ by 0.003 pair AUC.
  - Under headroom, no JOINT code passed the guards against SEQ-21 λ 0.1.
- **What this is not.** It is not a capacity impossibility and not a universal privacy floor. It is the measured
  trade-off of this code family, this frozen head, this attacker slate and these rows.

## 6. Structural properties (separate from the empirical results)

- **Decision preservation.** It holds on every row of every role for all 114 policy units: 8,930,760 row checks, 0 failed
  (CLASS_PRESERVATION.json).
- **What guarantees it.** The guarantee is Theorem 1 in MATH_REVIEW.md, under its assumptions A1–A7. The row checks are
  receipts, not the proof.
- **What it does not imply.** It gives no SEX AUC bound, no population MI bound and no training-data privacy.
- **The decision floor.** The decision-only release (CLASS-ONLY) has pair AUC 0.739. No class-preserving code can go
  below what its decisions disclose.

## 7. Predictions registered before any fit (PREDICTIONS.json, commit 357b300)

| ID | Prediction | Probability | Outcome |
|---|---|---|---|
| CP1 | P\* exists | 0.80 | TRUE |
| CP2 | Claim C passes | 0.25 | FALSE |
| CP3 | Winning family is sequential | 0.45 | FALSE (JOINT λ 0.01; SEQ-21 λ 0.01 missed the guard by 0.0002) |
| CP4 | Joint criterion met | 0.03 | FALSE |
| CP5 | Headroom changes the winner | 0.75 | TRUE |
| CP6 | Q passes | 0.95 | TRUE |
| CP7 | Label | 0.66 on this label | CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION (the modal forecast) |

- **Scores.** The mean Brier score over CP1–CP6 is 0.062. The multiclass Brier score of CP7 is 0.177.
- **Where the reasoning was wrong.** CP2's reasoning expected P\* at an intermediate λ (0.025–0.06). The guard and
  headroom rules left only λ 0.01.

## 8. Decision and next step

- **Close this line.** The interpolation and headroom-selection line for i8o64 privacy compression is closed. On these
  rows, headroom removes the privacy benefit, and another λ tweak on the same rows is not the recommended next
  experiment (prompt §17).
- **What is usable now.**
  - Q, the confidence-feasible task-only code. Its status is confidence feasibility established, with no privacy
    claim. Pair AUC is 0.849, against 0.858 for U.
  - P\* is packaged with its status: inner-eligible, confidence preserved on the assessment, privacy criterion NOT
    established. It is practically equivalent to T\*.
- **If privacy compression is pursued again,** it needs a genuinely different mechanism. That would need its own
  protocol and data. Two directions, neither of which this study tested:
  - a code that is not confidence-limited on the occupation head;
  - a different disclosure contract.
- **Confirmation data.** None was opened, and none is recommended for this line.

Other evidence for this study:
- Verification: VALIDATION.md and INDEPENDENT_VERIFICATION.json.
- Claim scope: PRIOR_ART_AND_CLAIM_SCOPE.md.
- Custody: COST_AND_CLOSEOUT.md.
