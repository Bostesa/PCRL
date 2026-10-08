# Research decision: ccm sprint

**Label:** PREMISE_NOT_SUPPORTED. The feasibility premise of the selected mechanism failed under its registered
contract.

**Exposure (verbatim, registered).** This sprint is motivated by repeatedly used Adult development results and by an
inner comparison observed after those results. Existing models, partitions, thresholds and past outcomes are known. Any
real-data prototype result is exploratory development evidence. New procedural locks do not undo historical exposure.
No old assessment is reopened and no new confirmation population is opened.

## Decision

1. **Do not build the confidence-constrained score-channel prototype.** The locked go rule failed in 6 of 6
   seed-recipient cells under the pointwise guard G(0.005, 0.0025) at i8o64. Every coverage failure is intrinsic:
   - occupation: at most 36.6% coverage is possible and 85% of new inputs fall back to raw scores;
   - income: the exact optimum is 37–39% coverage, with 61–63% fallback.

   No PILOT_LOCK was written and Stage F was skipped. No SEX or task label was read. INNER_SELECTION and ATTACK_FIT_NEW
   were not used. No assessment was opened.
2. **Mechanism table** (MECHANISM_TABLE.md):
   - The stochastic channel under the same pointwise guard inherits the same admissible support, so it is not a way
     around the result. It was kept only as an exact oracle arm.
   - Retraining purpose heads with a confidence objective re-enters closed lines and offers no guarantee.
   - Neither is recommended in place of (1).
3. **Incumbent triage** (INCUMBENT_TRIAGE.md):
   - The hcal inner gaps are small and cannot be resolved by precision without an unexposed population, which this
     sprint may not open.
   - A larger sample would not create a better algorithm.

## What this changes in the program

**Where the binding constraint sits.** Across the confidence lines (hcal average-loss allowances, earlier
capacity/budget studies, and this sprint's pointwise and teacher-expected guards), the binding constraint is the
OCCUPATION recipient's confidence at the registered capacity:
- Codes that keep occupation confidence barely protect: confidence-capacity and confidence-budgeted-privacy studies.
- Privacy-aware codes break the occupation log-loss bound: hcal P* upper bound 0.0120 / 0.0124 against 0.010.
- Under a per-person guarantee, occupation is essentially an identity release (this sprint).
- Even under the weaker teacher-expected guarantee, 44–48% of new occupation inputs fall back (this sprint,
  diagnostic).

**What remains open.** The only cell with room under any guarantee tested here is income under the teacher-expected
guarantee: 16 tokens, 95–96% coverage, 4–5% fallback. But the coalition sees the occupation release too, and that
release stays near-identity. Income-side gains would therefore be bounded by what the occupation scores already
disclose.

## One recommended next step

**Settle the occupation requirement before funding another algorithm.** The claims and manuscript owners should decide
what confidence the occupation recipient is actually owed. Until that requirement changes, no further Adult
release-mechanism study is recommended. The available evidence says the obstacle is this requirement at this
capacity, not the search procedure.

## Unresolved

- Privacy was not measured on Adult.
- The smooth-law bound (MATH_REVIEW.md R6) is an expectation, not a measurement.
- The G_exp diagnostic's occupation bounds do not prove its shortfall intrinsic.
- Off-device custody is pending: no external drive was attached.
