# Research decision: confidence-budgeted privacy compression (cbp)

**Label: CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION.**

This is an exploratory, locked development comparison on reused Adult rows. The design was motivated by opened Adult
development results. Its nominal intervals condition on fitted artifacts and do not account for the adaptive research
history. It is not confirmation and not a population privacy guarantee (EXPOSURE_LEDGER.md).

| Claim | Status | Root cause |
|---|---|---|
| A: J\* vs C_rate | NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE | LOCAL_GUARD_FAILURE (no JOINT configuration met headroom plus its guards); not a scored head-to-head result |
| B: J\* vs C_global | NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE | LOCAL_GUARD_FAILURE |
| C: P\* vs T\* | NOT_ESTABLISHED | MEASURED_VIOLATION_SUPPORTED_BY_BOUND on P23. This is the pair privacy benefit, not a confidence clause. The other 10 clauses PASS. |
| Q: DIRECT-TASK i8o64 | PASS | All four confidence bounds pass |

Sources: LABEL_RESULT.json, PRIMARY_ENDPOINTS.csv, SELECTION.json.

Every claim was technically valid:
- controls all_ok;
- endpoint parity 33/33;
- inner validation 129/129;
- all assessment arrays finite;
- independent replay: overall WARN (26 PASS, 2 WARN, 0 FAIL), with no disagreement on any endpoint, selection or
  label (VALIDATION.md). The WARNs:
  - 5 seed-1 assessment units were computed twice, within the single opening and under the same lock, by an
    overlapping load-balancing worker. The copies are bitwise identical, and the first copies were quarantined, not
    deleted (ASSESSMENT_ATTEMPTS.json).
  - 3 attacker refits agree within 1e-9 but are not bitwise.

## 1. In plain words

- **The selected privacy code kept confidence, but its privacy benefit was negligible.**
  - P\* (JOINT λ 0.01) kept both tasks' confidence within the original allowances on the assessment (P28–P31 PASS).
    Its occupation log-loss upper bound was 0.00674 nats against the 0.01 limit; the previous study's privacy nominee
    could not establish this.
  - Headroom is an inner selection margin, not a confidence guarantee. P\*'s occupation excess (0.00337) is close to
    that of the privacy-untrained T\* (0.00308).
  - P\* was the only code that met the headroom rule and the local guards, and it is the weakest privacy weight on the
    grid. Its combined-view SEX AUC was only 0.0022 below strong task-only compression (FINE-TASK). The registered
    criterion needs a lower bound above 0.02.
- **No intermediate λ passed the registered every-seed headroom rule, except JOINT λ 0.04, which failed a local guard.**
  - Inner rows: at every λ ≥ 0.025, each family's worst-seed occupation log-loss excess exceeded the 0.006 headroom. The
    only exception was JOINT λ 0.04 (Figures 1 and 1b).
  - Most of these codes still met the original 0.01 allowance on every seed (ordinary eligibility).
  - Example: SEQ-12 and JOINT at λ 0.06 and 0.08 were ordinarily eligible and passed the T\* guards, with inner mean pair
    AUC 0.022–0.025 below T\* (point estimates). Headroom excluded them, and none was nominated or assessed as a claim.
  - Larger pair reductions came with larger occupation log-loss excess, with family-dependent slopes (Figure 1b).
- **No family met the method criterion, and no joint contribution was earned.**
  - No JOINT configuration qualified for J\*: none passed headroom plus its local guards against the strongest nonjoint
    controls.
  - JOINT λ 0.1 and SEQ-21 λ 0.1 differ by only 0.0031 pair AUC on the assessment (0.8125 vs 0.8157), and by 0.0004 on
    inner rows.
- **Decision (prompt §17): close this λ-interpolation and headroom-selection line.** Headroom removed the privacy
  benefit. We do not recommend another slightly adjusted λ on these rows.

## 2. Inner selection (INNER_SELECTION only; SELECTION.json, INNER_SELECTION_TABLE.csv, Figures 1 and 1b)

**Eligibility and roles**

- **Ordinary eligibility** holds for every privacy code except SEQ-21 λ 0.04, LOCAL λ 0.08 and LOCAL λ 0.1. Each of
  those fails on some seed.
- **Headroom** (log-loss excess ≤ 0.006 and Brier excess ≤ 0.0035 for every task and seed) was passed only by the four
  λ 0.01 maps and by JOINT λ 0.04.
- **Comparators.** T\* = U|FINE-TASK|i8o64; RAW-J is excluded from T\* by the registered correction. C_rate = C_global
  = U|SEQ-21|i8o64|l0.1. Q = U|DIRECT-TASK|i8o64.
- **P\* = U|JOINT|i8o64|l0.01.**
  - Its inner mean pair AUC was 0.8446, against 0.8459 for T\*.
  - Its alias set is itself. Its map equals SEQ-12 λ 0.01 on seed 0 only, so the alias is partial.

**Why stronger headroom-eligible candidates were not nominated.** The guard is: local AUC ≤ T\* + 0.005 for each
recipient and seed.
- **JOINT λ 0.04** (inner pair 0.8350): recipient 1 on seed 2 was +0.0119 above T\*.
- **SEQ-21 λ 0.01** (inner pair 0.8420): recipient 1 on seed 2 was +0.0052, missing the guard by 0.0002.
- **SEQ-12 λ 0.01** (inner pair 0.8440): recipient 1 was +0.0068 and +0.0082 above T\* on seeds 1 and 2.
- **LOCAL λ 0.01** (inner pair 0.8464) was weaker than both P\* and T\*, so it was not a stronger candidate. It also
  failed the guard, at +0.0061 and +0.0070 on seeds 1 and 2.

**J\* has no eligible nominee (LOCAL_GUARD_FAILURE).**
- The only headroom-eligible JOINT codes are λ 0.01 and λ 0.04.
- On every seed, their occupation-recipient AUC is about 0.03 to 0.06 above C_rate (SEQ-21 λ 0.1).
- The fixed fallback, JOINT λ 0.04, was scored DESCRIPTIVE_ONLY.

**Prespecified diagnostics (HEADROOM_VS_STANDARD_SELECTION.csv)**
- Without headroom, the ordinary privacy winner under the T\* guard is JOINT λ 0.1. Without any guard it is SEQ-21 λ 0.1.
- Headroom changed the winner. It gave up 0.0280 mean inner pair AUC: 0.0355, 0.0308 and 0.0176 on seeds 0, 1 and 2.
- Family headroom winners: JOINT λ 0.01. LOCAL, SEQ-12 and SEQ-21 have none eligible (LOCAL_GUARD_FAILURE); their
  fixed fallbacks are at λ 0.01.

**Interpretation note (statistics reviewer, SEL-N4).**
- The inner estimates use 2,235 rows, so the SE of the inner log-loss excess is about 0.003.
- The seeds share those rows, so the per-seed rule does not average the noise away. Per-seed local guards at +0.005 are
  noisy.
- This is a property of the registered rule, not a reason to relax it.

## 3. Locked assessment (ASSESSMENT_COMPARISON.csv, PRIMARY_ENDPOINTS.csv; 13,936 rows, 13,929 groups)

Notes on the table below:
- Absolute AUCs are seed means of the SEX AUC of the inner-AUC-selected attacker, with fixed orientation.
- Excesses are relative to the U continuous teacher.
- Every U-derived code has exactly U's accuracy, because decisions are preserved.

| Release | Role | Inner ord / head | Pair AUC | Recipient 1 AUC (income) | Recipient 2 AUC (occupation) | Inc. acc | Occ. acc | Inc. LL excess | Occ. LL excess | Inc. Brier excess | Occ. Brier excess |
|---|---|---|---|---|---|---|---|---|---|---|---|
| U continuous | baseline | yes / yes | 0.8585 | 0.6972 | 0.8563 | 0.8440 | 0.4754 | 0 | 0 | 0 | 0 |
| DIRECT-TASK i8o64 | Q | yes / yes | 0.8491 | 0.6957 | 0.8342 | 0.8440 | 0.4754 | 0.00083 | 0.00295 | 0.00053 | 0.00166 |
| FINE-TASK i8o64 | T\* | yes / yes | 0.8462 | 0.6957 | 0.8310 | 0.8440 | 0.4754 | 0.00110 | 0.00308 | 0.00087 | 0.00144 |
| JOINT λ 0.01 | **P\*** | yes / yes | 0.8440 | 0.6952 | 0.8274 | 0.8440 | 0.4754 | 0.00046 | 0.00337 | 0.00045 | 0.00151 |
| LOCAL λ 0.01 | LOCAL family fallback | yes / yes | 0.8439 | 0.6977 | 0.8296 | 0.8440 | 0.4754 | 0.00101 | 0.00307 | 0.00075 | 0.00151 |
| SEQ-12 λ 0.01 | SEQ-12 family fallback | yes / yes | 0.8435 | 0.6956 | 0.8265 | 0.8440 | 0.4754 | 0.00122 | 0.00343 | 0.00091 | 0.00153 |
| SEQ-21 λ 0.01 | SEQ-21 family fallback | yes / yes | 0.8432 | 0.6977 | 0.8250 | 0.8440 | 0.4754 | 0.00057 | 0.00361 | 0.00043 | 0.00163 |
| JOINT λ 0.04 | J\* fallback (descriptive) | yes / yes | 0.8329 | 0.6969 | 0.8109 | 0.8440 | 0.4754 | 0.00099 | 0.00422 | 0.00068 | 0.00174 |
| SEQ-21 λ 0.1 | C_rate = C_global | yes / no | 0.8157 | 0.6945 | 0.7711 | 0.8440 | 0.4754 | 0.00180 | 0.00838 | 0.00090 | 0.00327 |
| SEQ-12 λ 0.1 | source control | yes / no | 0.8160 | 0.6963 | 0.7824 | 0.8440 | 0.4754 | 0.00109 | 0.00825 | 0.00077 | 0.00320 |
| JOINT λ 0.1 | ordinary winner; source control | yes / no | 0.8125 | 0.6948 | 0.7823 | 0.8440 | 0.4754 | 0.00137 | 0.00807 | 0.00067 | 0.00292 |
| CLASS-ONLY | decisions only | no / no | 0.7391 | 0.5871 | 0.6866 | 0.8440 | 0.4754 | 0.0846 | 0.1102 | 0.0432 | 0.0411 |
| RAW-J β 0.3 | reference (privacy-trained) | no / no | 0.7884 | 0.6846 | 0.7743 | 0.8452 | 0.4665 | -0.00344 | 0.0134 | -0.00156 | 0.00653 |
| FARE (REF\|F) | reference | no / no | 0.7042 | 0.6851 | 0.6356 | 0.8441 | 0.4509 | 0.0133 | 0.0460 | 0.00995 | 0.0233 |
| no-fairness FARE (REF\|F0) | reference | no / no | 0.8647 | 0.8034 | 0.8509 | 0.8492 | 0.4604 | -0.0115 | 0.0299 | -0.00512 | 0.0122 |
| LEACE (REF\|E) | reference | no / no | 0.8078 | 0.5538 | 0.8011 | 0.7940 | 0.4358 | 0.1155 | 0.0518 | 0.0769 | 0.0274 |

**References** work under different contracts and are descriptive only. None meets the task/confidence contract, so their
lower recovery is not a usable privacy gain (prompt §17):
- **RAW-J β 0.3:** keeps accuracy within 0.01 (occupation −0.0089), but its occupation log-loss excess (0.0134) and Brier
  excess (0.00653) exceed the 0.01 and 0.005 allowances.
- **FARE:** loses 0.0245 occupation accuracy.
- **No-fairness FARE (F0):** loses 0.0150 occupation accuracy and is more recoverable than U.
- **LEACE:** loses 0.0499 income and 0.0396 occupation accuracy.
- **CLASS-ONLY:** keeps decisions but has an occupation log-loss excess of 0.110.

**Claim C (P\* vs T\*), point [lower, upper].** z = 3.2048; B = 1999; exact-record-group bootstrap with seed 20261008.

| Slot | Quantity | Target | Point [lower, upper] | Outcome |
|---|---|---|---|---|
| P23 | pair benefit | lower > 0.02 | 0.00217 [0.00019, 0.00415] | **MEASURED_VIOLATION** |
| P24 | recipient-1 AUC change | upper < 0.01 | −0.00056 [−0.00260, 0.00149] | PASS |
| P25 | recipient-2 AUC change | upper < 0.01 | −0.00362 [−0.00559, −0.00164] | PASS |
| P26–P27 | accuracy identity | lower > −0.01 | 0 (SE 0) | PASS |
| P28 | income LL excess | upper < 0.01 | 0.00046 [−0.00101, 0.00192] | PASS |
| P29 | occupation LL excess | upper < 0.01 | 0.00337 [−0.00000, 0.00674] | PASS |
| P30 | income Brier excess | upper < 0.005 | 0.00045 [−0.00002, 0.00092] | PASS |
| P31 | occupation Brier excess | upper < 0.005 | 0.00151 [0.00048, 0.00254] | PASS |
| P32–P33 | retention | lower > 0 | 0.01964, 0.03823 | PASS |

**Q (DIRECT-TASK i8o64), point / upper bound.** All four pass.

| Slot | Quantity | Point | Upper bound |
|---|---|---|---|
| P34 | income LL excess | 0.00083 | 0.00234 |
| P35 | occupation LL excess | 0.00295 | 0.00614 |
| P36 | income Brier excess | 0.00053 | 0.00103 |
| P37 | occupation Brier excess | 0.00166 | 0.00266 |

**Seeds (ALL_LEVELS.csv).**

| Code | Pair AUC, seeds 0 / 1 / 2 | Occupation LL excess, seeds 0 / 1 / 2 |
|---|---|---|
| P\* | 0.8476 / 0.8461 / 0.8382 | 0.0018 / 0.0044 / 0.0039 |
| T\* | 0.8515 / 0.8454 / 0.8415 | 0.0017 / 0.0039 / 0.0036 |

The pair benefit (T\* minus P\*) is not consistent in sign across seeds: 0.0039, −0.0007 and 0.0033 on seeds 0, 1 and
2. On seed 1, P\* is slightly more recoverable than T\*.

**Consistency with the source.** The assessment reproduces the qpc context values:
- U continuous: 0.8585 (source 0.858).
- JOINT λ 0.1: pair 0.8125 (source 0.813) and occupation LL excess 0.00807 (source 0.008072).
- FINE-TASK minus JOINT λ 0.1 pair: 0.0336 (source 0.033627).

## 4. Strongest favourable and strongest adverse results

**Favourable**
- P\* passed every clause except P23. That is both local AUC clauses (P24–P25) and every accuracy, confidence and
  retention clause (P26–P33).
- This includes the occupation log-loss clause that the source's JOINT λ 0.1 could not establish. There the point was
  0.0081, inside the 0.01 allowance, but the upper bound was 0.0121: a precision failure, not an established violation.
- Q again passed all four confidence bounds.

**Adverse**
- **The privacy benefit is essentially gone.** P23 = 0.00217 with upper bound 0.00415, so a violation of the 0.02 target
  is supported by the interval.
- **The best headroom-eligible JOINT code** (descriptive and post hoc; no registered slot). JOINT λ 0.04, the J\*
  fallback, had a pair AUC 0.0133 below T\* (point estimate). This is context only, not a claim-C outcome.
- **Joint vs the strongest nonjoint control** (descriptive only; J\* is the fixed fallback JOINT λ 0.04).
  - C_rate = C_global (SEQ-21 λ 0.1) is less recoverable than the J\* fallback: P01 = −0.0172 [−0.0209, −0.0136]
    against the +0.02 target.
  - The fallback's occupation-recipient AUC is 0.0398 [0.0346, 0.0450] higher (P03; P12 and P14 are role aliases).
  - The comparator is not held to headroom and spends about twice the occupation confidence (log-loss excess 0.0084 vs
    0.0042). So this shows joint added nothing under headroom; it does not show joint is worse at matched confidence.
- **The λ 0.1 cost** (descriptive). At λ 0.1, the larger pair reduction (JOINT λ 0.1 vs T\*, about 0.034) came with
  about 0.008 nats of occupation log-loss excess over U, about 0.005 more than T\*.
  - Its confidence preservation would not be established at the registered bound: upper bound about 0.0119 > 0.01. This
    is a precision failure, not an established violation.
  - On inner rows, its worst-seed excess (0.0083) was above the 0.006 headroom.

## 5. Bottleneck and diagnostics (prompt §13)

**Bottlenecks**
- **Headroom passed, but the privacy margin disappeared.** P\* exists and its confidence clauses pass, but its pair
  benefit is 0.002.
- **The local guards decided the nominee.** Without the per-seed recipient-1 guard, JOINT λ 0.04 (inner pair 0.835)
  would have been nominated. Its assessment pair difference from T\* (0.013) is a descriptive point estimate, post hoc,
  context only.
- **Joint remained tied with sequential.**
  - At λ 0.1, JOINT's pair AUC is 0.0031 below SEQ-21 and 0.0034 below SEQ-12.
  - Under headroom, no JOINT code passed the guards against SEQ-21 λ 0.1.
- **What this is not.** It is not a capacity impossibility and not a universal privacy floor. It is the measured
  trade-off of this code family, this frozen head, this attacker slate and these rows.

**Diagnostics**
- **Where the diagnostics are.**
  - OPTIMIZATION_RECEIPTS.json (72 privacy units) and ENDPOINT_RECEIPTS.json (9 task-only references):
    - per-code alphabets and states per predicted class;
    - entropy, support, singletons and unseen-pair fraction;
    - objective terms (teacher KL distortion and plug-in MI);
    - merges, exchanges, sweeps and unresolved optima;
    - fitted MI against the qpc fit-receipt permutation null (100 permutations, seed 20261006), for all 72 privacy
      units and the FINE-TASK and CLASS-ONLY references. The 3 DIRECT-TASK (Q) units carry fitted MI terms but no
      recorded permutation null.
    - A fitted MI near or below its null is not protection.
  - Per-seed true-label log loss and Brier: INNER_SELECTION_TABLE.csv. Every assessment seed: ALL_LEVELS.csv.
- **Fewer token states do not explain the gain beyond task-only compression** (Figure 4, INNER_STATES_VS_RECOVERY_v2.csv).
  - Every code at λ ≤ 0.06 occupies 334–336 fitting states, the same as FINE-TASK and DIRECT-TASK (336). Yet their
    inner pair AUC falls as low as 0.824.
  - Only at λ 0.08–0.1 do SEQ-12 and JOINT drop to about 301–325 states.
- **Optimiser.**
  - All 18 JOINT units have unresolved local optima across their starts.
  - 8 JOINT units have a start that hit the 5-sweep cap, including the J\* fallback (λ 0.04, seed 0, SEQ-21 start). These
    are recorded, not rescued.
  - One non-JOINT unit, the C_rate = C_global map (SEQ-21 λ 0.1, seed 0, reused after parity), also stopped at the
    stage-1 5-sweep cap without converging. It is recorded, not rescued.
- **Figures.**
  - 1 and 1b: the inner λ curve and trade-off.
  - 2: the locked list's pair recovery vs occupation log-loss excess.
  - 3: individual vs pair recovery.
  - 4: states vs recovery.
  - The `_annotated` versions add legends, recipient names and a zoom inset (post-lock presentation only).

## 6. Structural properties (separate from the empirical results)

- **Decision preservation holds on every row of every role** for all 114 policy units: 8,930,760 row checks, 0 failed
  (CLASS_PRESERVATION.json).
- **The guarantee is Theorem 1 in MATH_REVIEW.md, under assumptions A1–A7.** The row checks are receipts, not the
  proof. It implies no SEX AUC bound, no population MI bound and no training-data privacy.
- **The decision floor.**
  - Every class-preserving code carries at least the SEX information in its decisions: I(S; C1, C2) ≥ I(S; d1, d2)
    (MATH_REVIEW DP3).
  - The corresponding AUC ordering (DP5) holds only for Bayes-optimal population readers.
  - With this slate, the decision-only release (CLASS-ONLY) had a measured pair AUC of 0.739. That is a reference
    level, not a bound on any fitted attacker's AUC against another code.

## 7. Predictions registered before any fit (PREDICTIONS.json, commit 357b300)

| ID | Prediction | Probability | Outcome |
|---|---|---|---|
| CP1 | P\* exists | 0.80 | TRUE |
| CP2 | Claim C passes | 0.25 | FALSE |
| CP3 | Winning family is sequential | 0.45 | FALSE. JOINT λ 0.01 was nominated narrowly; SEQ-21 λ 0.01 missed the guard by 0.0002. JOINT is not compute-matched (§9). |
| CP4 | Joint criterion met | 0.03 | FALSE |
| CP5 | Headroom changes the winner | 0.75 | TRUE |
| CP6 | Q passes | 0.95 | TRUE |
| CP7 | Label | 0.66 on this label | CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION (the modal forecast) |

- **Brier scores:** mean 0.062 over CP1–CP6; multiclass 0.177 for CP7.
- **Where the reasoning was wrong:** CP2's reasoning expected P\* at an intermediate λ (0.025–0.06). The guard and
  headroom rules left only λ 0.01.

## 8. Decision and what is runnable

- **Close the line.** The λ-interpolation and headroom-selection line for i8o64 privacy compression is closed. On these
  rows headroom removes the privacy benefit, so another λ tweak on the same rows is not the recommended next experiment
  (prompt §17).
- **Usable now.**
  - **Q** is the confidence-feasible task-only code. Its status is confidence feasibility established, with no privacy
    claim. Pair AUC 0.849, against 0.858 for U.
  - **P\*** is packaged with its status: inner-eligible; accuracy, confidence and retention clauses PASS; privacy
    criterion NOT established. It is practically equivalent to T\*.
  - **The J\* fallback** (JOINT λ 0.04) is packaged as DESCRIPTIVE_ONLY. There is no eligible JOINT nominee, and this is
    the fixed minimum-shortfall fallback.
  - **The U continuous baseline** is the admitted teacher itself, with no protection.
- **How to run them.**
  - Q, P\* and the J\* fallback run with `python -m cbp.deploy` (QUICKSTART.md §2). MODEL_MANIFEST.json gives the Q and
    P\* deploy lines and the fallback's file hashes.
  - Q and P\* were deployed on seed 1 and are bitwise equal to the stored releases.
  - The J\* fallback was re-applied from the same-device copy on seed 1 (BACKUP_VERIFICATION.json).
  - The schema, teacher and flag refusals were tested; each exits 2.
- **If privacy compression is pursued again**, it needs a genuinely different mechanism under its own protocol and data,
  for example a code not confidence-limited on the occupation head, or a different disclosure contract. Neither was
  tested here. No confirmation data was opened, and none is recommended for this line.

## 9. Novelty and claim scope (PRIOR_ART_AND_CLAIM_SCOPE.md)

- **No novelty claim.** KL clustering, privacy-funnel penalties, λ sweeps, margin ("headroom") selection and output
  compression are all prior work. Headroom selection is a design choice of this protocol, not a method.
- **The sequential arms.** SEQ-12 and SEQ-21 are matched, deterministic adaptations of the source's corrected sequential
  design. They are not the official Taylor, Vippathalla and Coon solver, for which no code exists.
- **JOINT is not compute-matched.**
  - It refines five starts and keeps the four same-λ witnesses (FINE-TASK, LOCAL, SEQ-12, SEQ-21) as candidates, so its
    fitting-objective advantage holds by construction (MATH_REVIEW Theorem 2).
  - P\*'s JOINT label is therefore not evidence for joint design: its map equals SEQ-12 λ 0.01 on seed 0, and its
    assessment pair AUC (0.8440) is within 0.001 of the λ 0.01 LOCAL, SEQ-12 and SEQ-21 maps.
  - A tie with sequential despite more search favours the simpler sequential story.
- **The release is a fixed-task output** (token, decoded probabilities, unchanged decision). It is not reusable or
  privacy-preserving representation learning.
- **No method claim (A, B or C) passed.** Even a C pass would not have meant a new algorithm, joint superiority,
  dominance over published defences, chance-level privacy or a population certificate.

## 10. What remains unknown and what was backed up

- **Backed up.** A versioned same-device copy (`cbp_v1_local_copy_20261006`):
  - 1,144 files: 1,143 study files plus the bundled pinned input; 635,850,300 bytes after the final refresh at 21:41:03Z.
  - Every file was re-read uncached, and all 1,144 match. No physical cold-disk read was performed.
  - Restores from the copy alone PASS on seed 1 for teacher U, teacher RAW-J β 0.3, Q, P\*, the J\* fallback and the
    selected pair attacker (BACKUP_VERIFICATION.json, RESTORE_INDEX.json).
- **Not backed up.**
  - The off-device copy is PENDING, because no drive matching the content rule was mounted.
  - The inherited qpc/dpc/osf/smf off-device custody is also still PENDING (provenance/qpc_custody/STATUS.json).
  - The commands to finish both are in COST_AND_CLOSEOUT.md §3 and QUICKSTART.md §4.
- **Unknown.**
  - Whether a mechanism not tested here gives a useful benefit.
  - Recovery by attackers outside the declared matched slate. Weak recovery by this slate is not proof that all attacks
    fail.
  - Behaviour on new data. No confirmation data was opened, and the intervals do not account for historical adaptive
    reuse of these rows.

Verification: VALIDATION.md and INDEPENDENT_VERIFICATION.json. Cost and custody: COST_AND_CLOSEOUT.md.
