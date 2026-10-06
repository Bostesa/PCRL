# Advisor brief: decision-preserving joint compression

**Label: EXPERIMENTAL_NO_ADVANTAGE.** The study is complete and valid, and the result is negative. It is a locked development study on 13,936 already-used Adult rows, not a confirmation. No new population was opened, and cloud cost was $0.

## The question

Replace each recipient's continuous scores with a small public code: a token, its decoded probability vector and the decision. The code is built so that the decision never changes.

Can such a code, especially one fitted jointly against the two-recipient coalition, reveal less SEX than the strongest non-joint release? And can it do so while keeping both tasks' confidence within 0.01 nats log loss and 0.005 Brier of the task-only teacher U?

## What happened

- **Decisions:** identical to the teacher's on every row of every role, for all 258 fitted codes (independently verified).
- **Confidence:** income is nearly free (+0.001 to +0.002 nats). Occupation is not: the finest code (8 states per class) costs +0.020 to +0.021 nats and +0.007 to +0.008 Brier, about twice the allowance.
  - No code met the contract on every inner seed.
  - The class-only code costs +0.11 nats.
- **Selection:**
  - No joint nominee, and no privacy-trained nominee.
  - The strongest eligible release is U's continuous output.
  - Every comparison is therefore descriptive only.
- **Leakage** (pair SEX AUC; lower is better):

  | Release | Pair AUC |
  |---|---:|
  | Continuous U scores | 0.858 |
  | Task-only code (m 8) | 0.825 |
  | Joint code (m 8, λ 0.1) | 0.807 |
  | Sequential codes | 0.807–0.809 |
  | Decisions alone | 0.739 |
  | FARE | 0.704 (but loses 2.4 occupation-accuracy points) |

- **Joint versus matched task-only:** 0.018 [0.014, 0.021], short of the 0.02 margin. Joint versus sequential: a tie.

## What it means

1. The decision vector alone already reveals SEX at about 0.74. Coarsening confidence can remove only part of the 0.86 that the scores carry.
2. The removable part is paid for in occupation confidence, at a rate above the registered allowance.
3. Joint fitting and privacy training help a little (about 0.018 pair AUC at matched rate). That help is not established, and not distinct from sequential fitting.

## Runnable and custody

**Runnable:**
- U's continuous output (the only eligible release).
- The joint m8 code, marked NOT ELIGIBLE, through `python -m dpc.deploy`. Deployment takes the 83 permitted columns, returns only tokens, decoded probabilities and decisions, and refuses extra or reordered columns and fine-ID or raw-score exports.

**Custody:**
- Backup: a verified same-device copy only. Off-device backup is pending (drive absent).
- Predecessor custody repair (osf/smf drive copy and smf drive restore): still pending, with exact commands recorded in `QUICKSTART.md`.

**Validity caveat:** every matched task-only control was also ineligible. The locked rule treats that as a coverage failure, giving a valid negative. A strict reading of the protocol's missing-comparator sentence would instead give INCOMPLETE_OR_INVALID. No claim passes either way.

**Disclosed process deviation:** a third heavy process overlapped twice, for about 2 and 4 minutes.

## Recommendation

Record the negative. Do not relax the allowances. A follow-up would need a newly registered contract, for example a separately budgeted occupation code or a better-calibrated occupation teacher, and should be developed on inner data before any population is spent.
