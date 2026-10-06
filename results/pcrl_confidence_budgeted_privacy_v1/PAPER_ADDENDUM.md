# Paper addendum: confidence-budgeted privacy compression

Scope: an exploratory, locked development comparison on reused Adult rows. The fitting, inner-selection and assessment
data were all used historically, so the nominal intervals condition on fitted artifacts and do not account for the
adaptive research history. This is not confirmation and not a population privacy guarantee.

## Suggested paragraph

> At the code capacity that preserves confidence (income 8 / occupation 64 states per teacher-predicted class), we asked
> whether a smaller privacy weight keeps both tasks' confidence within the original allowances while still reducing
> combined-view sensitive-attribute recovery beyond strong task-only compression. Before fitting, we registered a
> six-value λ grid (0.01 to 0.1) for local, both sequential orders and joint objectives, and an inner selection rule. The
> rule required privacy nominees to leave confidence headroom: at most 0.006 nats of log-loss excess and at most 0.0035
> Brier excess on every task and seed. Comparators were not subject to headroom. Only λ 0.01 codes and one joint
> λ 0.04 code met headroom; after per-seed local guards, the nominee was joint λ 0.01. On the locked assessment
> (13,936 rows; 37-clause family; z = 3.205), the nominee passed every confidence and utility clause. Its occupation
> log-loss excess was 0.0034 nats with an upper bound of 0.0067. It reduced pair SEX AUC by only 0.0022 relative to
> family-matched task-only compression (bounds [0.0002, 0.0042]), far short of the registered 0.02. No joint
> configuration qualified against the strongest sequential control. The registered outcome was confidence feasibility
> without a method criterion. On this benchmark, the privacy gain of class-preserving compression was bought with
> occupation confidence almost linearly: the 0.034 AUC gain at λ 0.1 costs about 0.008 nats. A budget that leaves
> headroom under the 0.01 allowance removes it.

## Numbers and their sources

| Quantity | Value | Source |
|---|---|---|
| P\* | U\|JOINT\|i8o64\|l0.01 | SELECTION.json, EVALUATION_LOCK.json |
| T\*, C_rate = C_global, Q | FINE-TASK i8o64, SEQ-21 λ 0.1, DIRECT-TASK i8o64 | SELECTION.json |
| P23 (pair benefit vs T\*) | 0.00217 [0.00019, 0.00415], MEASURED_VIOLATION of > 0.02 | PRIMARY_ENDPOINTS.csv |
| P29 (P\* occupation LL excess) | 0.00337, upper 0.00674 (PASS) | PRIMARY_ENDPOINTS.csv |
| P34–P37 (Q) | all PASS; occupation LL upper 0.00614 | PRIMARY_ENDPOINTS.csv |
| Headroom give-up (inner pair AUC vs the ordinary winner) | 0.0280 (seeds 0.0355 / 0.0308 / 0.0176) | SELECTION.json diagnostics |
| JOINT λ 0.1: pair AUC and occupation LL excess (assessment) | 0.8125, 0.00807 | ASSESSMENT_COMPARISON.csv |
| U continuous pair AUC | 0.8585 | ASSESSMENT_COMPARISON.csv |
| Decision preservation | 114 policy units, 8,930,760 row checks, 0 failed | CLASS_PRESERVATION.json |

## Wording constraints (PRIOR_ART_AND_CLAIM_SCOPE.md §5.3)

- **Allowed:** "A decision-preserving code at i8o64 keeps confidence within the allowances; no privacy-training
  criterion passed."
- **Not allowed:**
  - "privacy compression works";
  - "capacity impossibility";
  - "a universal privacy floor";
  - any joint-design advantage;
  - "certified", "private" or "chance-level".
- **Sequential arms:** they are matched adaptations of the source's corrected design, not the official Taylor,
  Vippathalla and Coon solver. JOINT used more search starts than the sequential arms, so it was not compute-matched.
- **Decision preservation:** cite it as Theorem 1 in MATH_REVIEW.md, under assumptions A1–A7. Row checks are receipts,
  not the proof. It implies no SEX AUC bound and no population MI bound.
