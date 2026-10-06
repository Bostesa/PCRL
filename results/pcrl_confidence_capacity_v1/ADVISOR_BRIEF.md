# Advisor brief: confidence capacity and privacy

**Label: CONFIDENCE_FEASIBILITY_ESTABLISHED.** No method advantage was established.

This is an exploratory locked comparison on 13,936 already-used Adult rows. No new population was opened, and cloud cost was $0.

## The question

The previous study's decision-preserving codes, at 8 states per predicted class, lost too much occupation confidence. Two questions follow:
- Was that because the clustering never finished, or because there were too few states?
- If confidence can be kept, does privacy training still help against an equally capable ordinary code?

## What happened

**Convergence did not help.** Letting the old 8-state code converge to a fixed point changed occupation fit distortion by at most 0.0003 nats, and held-out confidence did not improve.

**Capacity did.** With 64 occupation states per predicted class (income 8), the task-only code is the only rate eligible on every inner seed. On the assessment it passes all four confidence bounds:
- occupation log loss +0.0030 nats, upper bound 0.0061;
- occupation Brier +0.0017, upper bound 0.0027.

Every decision is preserved on every row.

**But that code barely protects.** Pair SEX AUC is 0.849, against 0.858 for U's continuous scores.

**Privacy training at the same capacity (JOINT, λ 0.1).** Against the task-only code:
- it lowers pair SEX AUC by 0.034 [0.029, 0.039], clearing the 0.02 margin;
- it costs +0.0081 nats of occupation log loss, upper bound 0.0121, over the 0.01 allowance.

So the privacy claim passes 10 of 11 clauses and is NOT_ESTABLISHED.

**Joint did not beat sequential.**
- Joint and both sequential orders are within 0.003.
- No joint code passed its guards against the sequential control.
- The corrected sequential baseline (stage one fitted against the other recipient's class-only release) differs from the old surrogate in every sequential unit.

**Decision floor.** The decisions alone still reveal SEX at pair AUC 0.739.

## What it means

1. The occupation confidence gap in dpc was a capacity problem. On this benchmark it can be closed without changing any decision.
2. Once confidence is kept, little protection remains: about 0.009 pair AUC below the continuous scores.
3. Privacy training buys about 0.03 pair AUC for about +0.005 nats. That is a better trade than lowering capacity (16 states gives pair 0.834 at +0.011 nats), but it is still just outside the registered allowance.
4. Joint optimisation earns nothing over sequential.

## Runnable, custody and verification

**Runnable** (`python -m qpc.deploy`, tested bitwise with the required refusals):
- U continuous;
- the confidence-feasible task-only code Q*;
- the privacy code P*, marked "confidence not established".

**Custody:**
- A same-device verified copy, with restores from the copy alone all passing.
- Off-device backup: PENDING (drive absent).
- dpc and osf/smf custody: PENDING, with exact commands in `QUICKSTART.md`.

**Verification:** `VALIDATION.md`.

## Process notes (disclosed)

- **Locked-file edit:** a reviewer edited a locked test file after the audit lock. The next stage refused before loading data, and the file was restored.
- **Semaphore:** a killed reviewer process left no release record; it ran about 40 CPU-s with no orphan. The semaphore was hardened (AMENDMENT_A1).
- **Label rule:** amended after the Stage A gate but before any recovery data. The label is reported under both rules, and they agree.

## Recommendation

**Record the result as is.**
- Confidence-feasible decision-preserving codes exist at 64 occupation states.
- Privacy training gives a measurable recovery reduction at a confidence cost just over the allowance.
- Joint adds nothing.

**Any follow-up** must be newly registered on inner data before any population is spent, for example a privacy weight between 0.01 and 0.1, or an occupation-only privacy term. None may be chosen from this assessment.
