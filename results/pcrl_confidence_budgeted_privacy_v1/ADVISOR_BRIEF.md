# Advisor brief: confidence-budgeted privacy compression (2026-10-06)

**Result: CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION.**

- **Confidence held.** Spending only a small, fixed confidence budget on privacy kept both tasks' confidence within the
  original allowances. Every decision was unchanged.
- **The privacy gain disappeared.** The privacy nominee is 0.002 SEX AUC below strong task-only compression, and the
  registered bar is 0.02.
- **Joint added nothing.** This line is closed.

## The question

The previous study found two things at a code size that preserves confidence (income 8 / occupation 64 states per
predicted class):

- Privacy training at λ 0.1 cut combined-view SEX recovery by 0.034 AUC versus task-only compression.
- It came too close to the occupation confidence limit to be established: the upper bound was 0.0121 nats against a limit
  of 0.01.

This study asked whether a smaller privacy weight keeps confidence while keeping enough of that privacy gain.

## What we did

- **Bank.**
  - Six λ (0.01–0.1) × four families (LOCAL, SEQ-12, SEQ-21, JOINT) × three seeds.
  - 24 earlier units were reused after bitwise parity, and 48 new units were fitted with the unchanged qpc optimiser.
- **Selection rule, fixed before fitting.**
  - Inner rows only.
  - Ordinary eligibility on every seed and task.
  - For privacy nominees, a headroom of at most 0.006 nats log-loss excess and at most 0.0035 Brier excess.
  - Local guards (each recipient within 0.005 AUC of the comparator on every seed).
  - Strong comparators that are not subject to headroom.
- **Assessment.** One locked assessment with the unchanged 37-clause family (z = 3.205, B = 1999, exact-record groups).
- **Verification and review.** An independent replay reproduced admission, all fits, all inner audits, the selection and
  every endpoint. Separate reviewers checked the statistics/selection rules and the mathematics/claims.

## What happened

| | Pair SEX AUC | Occupation log-loss excess (upper bound) | Status |
|---|---|---|---|
| U continuous (no protection) | 0.858 | 0 | baseline |
| Task-only code Q (DIRECT-TASK) | 0.849 | 0.0030 (0.0061) | all 4 confidence bounds PASS |
| Strong compression T\* (FINE-TASK) | 0.846 | 0.0031 | comparator |
| **Privacy nominee P\* (JOINT λ 0.01)** | **0.844** | **0.0034 (0.0067)** | 10/11 clauses PASS; **pair benefit 0.0022, bounds [0.0002, 0.0042], needed > 0.02** |
| JOINT λ 0.04 (best headroom-eligible JOINT; descriptive) | 0.833 | 0.0042 | failed a per-seed local guard on inner rows |
| JOINT λ 0.1 (source nominee; no headroom) | 0.813 | 0.0081 | over the headroom budget on inner rows |
| Decisions only | 0.739 | 0.110 | far outside the allowance |

- **The trade-off is steep on these rows.**
  - On inner rows, every code above λ 0.01 except JOINT λ 0.04 already spends more than 0.006 nats of occupation
    confidence.
  - The useful 0.03 pair reduction costs about 0.008 nats.
  - The headroom rule therefore gives up 0.028 inner pair AUC relative to the standard winner.
- **No JOINT code earned a role.** No JOINT code satisfied headroom plus guards against the strongest sequential
  control. At λ 0.1, JOINT and sequential differ by 0.003 pair AUC.

## What this means

- **Usable now.**
  - Q, a confidence-feasible, decision-preserving task code. It offers little protection: 0.009 pair AUC below U.
  - P\*, packaged with an explicit "privacy criterion not established" status.
- **Not shown.** This is not a privacy method win and not a joint-design contribution. It is not evidence that privacy
  compression can never work: it is this code family, frozen head, attacker slate and set of reused rows.
- **Recommendation.**
  - Close the λ-interpolation and headroom-selection line.
  - Do not run another slightly adjusted λ on these rows.
  - A future attempt needs a genuinely different mechanism under its own protocol and fresh, independently planned data.
- **Exposure.** All rows were used historically, so these nominal intervals are exploratory and condition on the fitted
  artifacts.

## Predictions registered before fitting

| Prediction | Forecast | Outcome |
|---|---|---|
| P\* exists | 0.80 | yes |
| Claim C passes | 0.25 | no |
| Sequential family wins | 0.45 | no: JOINT λ 0.01 won narrowly, and SEQ-21 λ 0.01 missed a guard by 0.0002 |
| Headroom changes the winner | 0.75 | yes |
| Q passes | 0.95 | yes |
| This label | 0.66 | yes, the modal forecast |

## Custody and cost

- **Cost.** $0 in cloud spend and about 3 CPU-h in total. Results are in RESEARCH_DECISION.md and COST_AND_CLOSEOUT.md.
- **Backup.** A same-device backup was made and restored from the copy alone. The off-device backup is pending because
  the drive was absent.
