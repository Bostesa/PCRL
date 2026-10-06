# Advisor brief: confidence-budgeted privacy compression (2026-10-06)

**Result: CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION.**

| Claim | Status |
|---|---|
| A (J\* vs C_rate), B (J\* vs C_global) | NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE (LOCAL_GUARD_FAILURE); never scored head to head. The J\* fallback, JOINT λ 0.04, is DESCRIPTIVE_ONLY. |
| C (P\* vs T\*) | NOT_ESTABLISHED: MEASURED_VIOLATION_SUPPORTED_BY_BOUND on P23, the pair privacy benefit. The other 10 clauses, including every confidence clause, PASS. |
| Q (DIRECT-TASK i8o64) | PASS, 4/4 |

- **Confidence held, but the privacy gain disappeared.** The selected privacy code kept both tasks' confidence within
  the original allowances, and every decision was unchanged. Its privacy gain was 0.002 SEX AUC beyond family-matched
  task-only compression, against a registered bar of 0.02.
- **No joint contribution was earned.** No JOINT code qualified for J\*.
- **The λ-interpolation and headroom-selection line is closed.**

## The question

The previous study used a code size that preserves confidence: income 8 / occupation 64 states per predicted class.

- At λ 0.1, privacy training cut combined-view SEX recovery by 0.034 AUC versus family-matched task-only compression
  (FINE-TASK, T\*).
- Its occupation confidence could not be established at the bound: the upper bound was 0.0121 against the 0.01 limit.

This study asked whether a smaller privacy weight keeps confidence while keeping enough of that gain.

## What we did

- **Bank.** 6 λ values (0.01–0.1) × 4 families (LOCAL, SEQ-12, SEQ-21, JOINT) × 3 seeds. 24 units were reused after
  bitwise parity and 48 were new fits with the unchanged qpc optimiser.
- **Selection rule, fixed before fitting and applied to inner rows only:**
  - ordinary eligibility on every seed and task;
  - for privacy nominees, headroom: log-loss excess ≤ 0.006 nats and Brier excess ≤ 0.0035;
  - local guards: each recipient's inner AUC at most 0.005 above the comparator's, on every seed;
  - strong comparators, not subject to headroom.
- **Assessment.** One locked assessment with the unchanged 37-clause family: z = 3.205, B = 1999, exact-record groups.
- **Verification.** An independent replay reproduced admission, all fits, inner audits, selection and every endpoint.
  Separate reviewers checked the statistics and the mathematics and claims.
  The replay's overall verdict is WARN (26 PASS, 2 WARN, 0 FAIL):
  - 5 assessment units were computed twice by overlapping workers; the copies are bitwise identical.
  - 3 refits agree within 1e-9 but are not bitwise.

## What happened

| | Pair SEX AUC | Occ. log-loss excess (upper bound) | Status |
|---|---|---|---|
| U continuous (no protection) | 0.858 | 0 | baseline |
| Task-only code Q (DIRECT-TASK) | 0.849 | 0.0030 (0.0061) | all 4 confidence bounds PASS |
| Strong compression T\* (FINE-TASK) | 0.846 | 0.0031 | comparator |
| **Privacy nominee P\* (JOINT λ 0.01)** | **0.844** | **0.0034 (0.0067)** | 10/11 PASS; **pair benefit 0.0022 [0.0002, 0.0042], needed > 0.02** |
| JOINT λ 0.04 (J\* fallback; descriptive) | 0.833 | 0.0042 | failed a per-seed local guard on inner rows |
| JOINT λ 0.1 (source nominee; no headroom) | 0.813 | 0.0081 (about 0.012) | over the headroom on inner rows |
| Decisions only | 0.739 | 0.110 | far outside the allowance |

- **Headroom was the binding rule on inner rows.**
  - Every code above λ 0.01 except JOINT λ 0.04 exceeds 0.006 nats of occupation log-loss excess on at least one seed,
    which fails the every-seed headroom rule. Most remain within the original 0.01 allowance.
  - At λ 0.1, JOINT's 0.034 pair reduction relative to T\* came with 0.008 nats of occupation log-loss excess over U,
    0.005 more than T\*'s own 0.003. Its confidence was not established at the bound (upper bound about 0.012); this is
    a precision failure, not a measured violation.
  - Headroom gave up 0.028 inner pair AUC relative to the standard winner.
- **No JOINT code earned the J\* role.** None satisfied headroom plus guards against the strongest sequential control.
  - JOINT λ 0.01 holds the P\* role, but that is not a joint contribution. Its pair AUC is within 0.001 of the λ 0.01
    LOCAL, SEQ-12 and SEQ-21 maps.
  - JOINT also used more search (it is not compute-matched).
  - At λ 0.1, JOINT and sequential differ by 0.003 pair AUC.

## What this means

- **Usable now.**
  - **Q**, a confidence-feasible task code. It offers little protection: 0.009 pair AUC below U.
  - **P\***, packaged with an explicit "privacy criterion not established" status.
  - Every decision is unchanged. This is guaranteed by Theorem 1 in MATH_REVIEW.md under assumptions A1–A7; the
    8,930,760 row checks are receipts, not the proof. It implies no SEX AUC bound, no population MI bound and no
    training-data privacy.
- **Not shown.**
  - Not a privacy-method win and not a joint-design contribution. No novelty is claimed.
  - The sequential arms are matched adaptations, not the official Taylor, Vippathalla and Coon solver.
  - Not evidence that privacy compression can never work. It covers this code family, frozen head, attacker slate and
    reused rows.
- **Recommendation.** Do not run another slightly adjusted λ on these rows. A future attempt needs a genuinely different
  mechanism under its own protocol and fresh, independently planned data.
- **Exposure.**
  - All rows were used historically, and the design was motivated by opened results.
  - The nominal intervals condition on fitted artifacts and do not account for the adaptive research history.
  - This is an exploratory, locked development comparison: not confirmation and not a population privacy guarantee.

## Predictions registered before fitting

| Prediction | Forecast | Outcome |
|---|---|---|
| P\* exists | 0.80 | yes |
| Claim C passes | 0.25 | no |
| Sequential family wins | 0.45 | no: JOINT λ 0.01 was nominated narrowly, and SEQ-21 λ 0.01 missed a guard by 0.0002. JOINT keeps the same-λ sequential and local maps as candidates, so this is not a joint contribution. |
| Headroom changes the winner | 0.75 | yes |
| Q passes | 0.95 | yes |
| This label | 0.66 | yes, the modal forecast |

## Custody and cost

- **Cost.** $0 in cloud spend. 4.0 CPU-h measured by the shared semaphore ledger (4.010 CPU-h over 104 holds, all six
  roles, including independent verification), against a 20 CPU-h ceiling.
- **Backup.** A same-device copy was made and restored from the copy alone. Off-device backup is pending, because the
  drive was absent.
- **Details:** RESEARCH_DECISION.md, VALIDATION.md and COST_AND_CLOSEOUT.md.
