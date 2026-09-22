# Independent selection and uncertainty review

Reviewed by the weighted-audits/replay subagent on 2026-09-21, before inspecting any comparative ACS outcomes. Review inputs were source, the prospective protocol/amendments, and synthetic summaries and household arrays. No ACS outcome, current-run evaluation record, or live comparative result was read or used for this review.

## Reviewed contracts

Sensitive recovery increments use `L_comparator - L_candidate`. The common H baseline cancels pointwise: `(L_H-L_candidate)-(L_H-L_comparator) = L_comparator-L_candidate`. This cancellation requires common H predictions and aligned person masks, IDs and weights; the evaluation and independent replay modules check these conditions. Validation selects one predictor by the balanced unweighted/PWGTP objective; both reported weightings use that same predictor. Configuration selection similarly freezes one three-anchor configuration for both weightings.

The household uncertainty engine draws one multiplicity vector on the union of households and reuses it across all anchors and contrasts. Each replicate recomputes each anchor's actual weighted numerator/denominator ratio and then averages the three ratios equally. A zero denominator rejects the entire common draw. Exact paired zeros preserve zero variance. The Bonferroni divisor is the explicit endpoint-list length, with `z = norm.isf(.05/(2*M))`. A separate synthetic ratio oracle with unequal masks and weights matched the engine to numerical precision, including whole-draw rejection bookkeeping.

The main historical-J and competitive claim formulas retain their existing all/any semantics. Protection-first requires all stated non-inferiority checks and at least one strict privacy improvement in each required comparison. Empty required baseline eligibility blocks competitive claims. Eligible optional family nominees also enter the conjunction. Default control families are disjoint; overlapping custom nominees are explicitly refused rather than silently deduplicated.

## Prospective defects resolved

The initial coalition attribution compared only the same-budget local setting. The implementation now freezes the complete three-budget local frontier with matching input, action count and conditioning family, and selects its strongest validation-eligible nominee separately for each route. Incomplete or empty frontiers block that attribution claim. The same-budget comparison remains a descriptive mechanistic diagnostic with endpoints counted in M.

The initial attribution metadata did not define an adjusted success conjunction, and its utility threshold was not route-specific. Amendment 3 and the implementation now provide explicit `attribution_claim_formulas` in both frozen selection and contrast artifacts. Utility-first requires strict utility improvement under both weightings and all eight sensitive non-inferiority checks. Protection-first requires both utility non-inferiority checks, all eight sensitive non-inferiority checks and an any-strict-improvement clause; coalition attribution restricts that strict clause to the four AB endpoints. Candidate and comparator evaluated common operating caps are included in the formula and M. Risk attribution conjoins the comparisons against both T0 and Ttask. Availability flags are not claim success, and narrow attribution is not gated on the separate main historical-J validation screen.

Registered branch-A U33 and matching controls join the existing baseline families. Fine-C conditioning remains separate from primary conditioning for matched local and refinement comparisons. Missing matched fine-C T0/Ttask controls block risk attribution.

## Verification and reporting scope

Fresh synthetic verification: **80 tests passed in 3.13 seconds** across `test_audits.py`, `test_evaluation.py`, `test_replay.py`, `test_selection.py`, and `test_uncertainty.py`. The selection/uncertainty subset passed 35 tests. Tests cover complete and incomplete local frontiers, route-specific nomination, PWGTP-only eligibility failures, lexical ties, adjusted all/any behavior, AB-specific strict improvement, direct non-inferiority and common-cap failure, both risk comparators, extension provenance, frozen formula persistence, independent artifact reconstruction, and household uncertainty.

Report any supported coalition result as a **selected local frontier comparison** under the registered common caps and validation selection. It does not establish superiority to a global population frontier or to every conceivable local mechanism. Unselected local settings remain on the descriptive frontier; adjusted inferential claims use the frozen selected nominee. Bounds remain conditional on the fitted models and do not include training-seed uncertainty.

No comparative study result is asserted here. Selection source and tests are handed back to the parent for integration; this subagent will not continue editing them during that integration.
