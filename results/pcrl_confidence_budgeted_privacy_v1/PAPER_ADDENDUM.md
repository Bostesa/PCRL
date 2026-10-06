# Paper addendum: confidence-budgeted privacy compression

**Scope.** The design was motivated by opened Adult development results, including the completed qpc
confidence-capacity study. This is an exploratory, locked development comparison on reused Adult rows. The fitting, inner-selection and
assessment data were all used historically. The nominal intervals therefore condition on fitted artifacts and do not
account for the adaptive research history. It is not confirmation and not a population privacy guarantee.

## Suggested paragraph

> At the code capacity that preserves confidence (income 8 / occupation 64 states per teacher-predicted class), we asked
> whether a smaller privacy weight keeps both tasks' confidence within the original allowances while still reducing
> combined-view sensitive-attribute recovery beyond strong task-only compression. Before fitting, we registered a six-value
> λ grid (0.01 to 0.1) for local, both sequential orders and joint objectives, and an inner selection rule. The rule
> required privacy nominees to leave confidence headroom of at most 0.006 nats of log-loss excess and at most 0.0035 Brier
> excess on every task and seed; comparators were not subject to headroom. Only the λ 0.01 codes and one joint λ 0.04 code
> met headroom. After per-seed local guards, the nominee was joint λ 0.01. Its pair recovery was within 0.001 of the
> λ 0.01 local and sequential codes, and the joint search used more starts than the sequential arms (it was not
> compute-matched), so the label is not evidence for joint design. The design was motivated by opened Adult development
> results, and the fitting, inner-selection and assessment rows had all been used before. This is an exploratory, locked
> development comparison: its nominal intervals condition on fitted artifacts and do not account for the adaptive
> research history, and it is neither confirmation nor a population privacy guarantee. On the locked assessment (13,936 rows; 37-clause family; z = 3.205), the
> nominee passed every confidence and utility clause, with an occupation log-loss excess of 0.0034 nats and an upper
> bound of 0.0067. It reduced pair SEX AUC by only 0.0022 relative to family-matched task-only compression (bounds
> [0.0002, 0.0042]), far short of the registered 0.02. No joint configuration qualified against the strongest sequential
> control. The registered outcome was confidence feasibility without a method criterion. On this benchmark, larger
> reductions from privacy training came with larger occupation confidence costs. At λ 0.1, the joint code's 0.034
> pair-AUC reduction relative to family-matched task-only compression came with 0.008 nats of occupation log-loss excess
> over the continuous teacher, 0.005 more than the task-only code's own 0.003, and its confidence preservation was not
> established at the registered bound. Under the registered every-seed headroom rule and per-seed local guards, the
> selected nominee retained only 0.002 of that reduction.

## Numbers and their sources

| Quantity | Value | Source |
|---|---|---|
| Claim statuses | A, B: NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE (LOCAL_GUARD_FAILURE; the J\* fallback JOINT λ 0.04 is DESCRIPTIVE_ONLY). C: NOT_ESTABLISHED, MEASURED_VIOLATION_SUPPORTED_BY_BOUND on P23 (the pair privacy benefit); the other 10 clauses, including every confidence clause, pass. Q: PASS 4/4. | LABEL_RESULT.json |
| P\* | U\|JOINT\|i8o64\|l0.01 | SELECTION.json, EVALUATION_LOCK.json |
| T\*, C_rate = C_global, Q | FINE-TASK i8o64; SEQ-21 λ 0.1; DIRECT-TASK i8o64 | SELECTION.json |
| P23 (pair benefit vs T\*) | 0.00217 [0.00019, 0.00415]; MEASURED_VIOLATION of > 0.02 | PRIMARY_ENDPOINTS.csv |
| P29 (P\* occupation LL excess) | 0.00337, upper 0.00674 (PASS) | PRIMARY_ENDPOINTS.csv |
| P34–P37 (Q) | all PASS; occupation LL upper 0.00614 | PRIMARY_ENDPOINTS.csv |
| Headroom give-up (inner pair AUC vs the ordinary winner) | 0.0280 (seeds 0.0355 / 0.0308 / 0.0176) | SELECTION.json diagnostics |
| T\* (FINE-TASK): pair AUC and occupation LL excess; T\* minus JOINT λ 0.1 pair | 0.8462, 0.00308; 0.0336 | ASSESSMENT_COMPARISON.csv |
| JOINT λ 0.1: pair AUC and occupation LL excess (assessment) | 0.8125, 0.00807 | ASSESSMENT_COMPARISON.csv |
| U continuous pair AUC | 0.8585 | ASSESSMENT_COMPARISON.csv |
| Decision preservation | 114 policy units, 8,930,760 row checks, 0 failed | CLASS_PRESERVATION.json |

## Wording constraints (PRIOR_ART_AND_CLAIM_SCOPE.md §5.3)

**Allowed:** "A decision-preserving code at i8o64 keeps confidence within the allowances; no privacy-training criterion
passed."

**Not allowed:**
- "privacy compression works";
- "capacity impossibility";
- "a universal privacy floor";
- any joint-design advantage;
- "certified", "private" or "chance-level";
- "privacy-preserving (or reusable) representation learning". The release is a fixed-task output: a token, decoded
  probabilities and the unchanged decision.
- novelty of KL clustering, privacy-funnel penalties, λ sweeps, margin selection or output compression;
- "beats PURIFIER", "beats Taylor" or "a Taylor baseline";
- "JOINT dominates DIRECT-TASK", or any dominance beyond the fitting objective over the unchanged witnesses.

**Required caveats:**
- The sequential arms are matched adaptations of the source's corrected design, not the official Taylor, Vippathalla and
  Coon solver.
- JOINT used more search starts than the sequential arms, so it was not compute-matched.
- Cite decision preservation as Theorem 1 in MATH_REVIEW.md, under assumptions A1–A7. Row checks are receipts, not the
  proof. It implies no SEX AUC bound, no population MI bound and no training-data privacy.
