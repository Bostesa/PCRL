# Claims review (independent read-only role) and resolutions

The review checked every listed figure against the evidence files. All figures matched except where noted, and all items were applied before publication.

| item | finding | resolution |
|---|---|---|
| M1 | C1 lower bound rounded twice (0.0655 → 0.066) | now [0.065, 0.105] in START_HERE, ADVISOR_BRIEF and MANUSCRIPT_CLAIMS_HANDOFF |
| M2 | Manuscript feasibility bullet did not say which claim it sizes | now names claim B and states that claim A cannot be confirmed at the observed code margins |
| M3 | FARE called "neither beaten nor beating", although C4 is an established shortfall for L_TREE (−0.091 [−0.121, −0.061]) | reworded: lower recovery than L_TREE, utility-infeasible, reported descriptively, not counted as beaten |
| M4 | "No compatible unexposed population exists" | now "identified" (non-use outside searched records is UNKNOWN) |
| S1 | Flat-prior assurance figures (0.37 / 0.29 / 0.35) had no source file | recomputed and published: claim B 0.38 and average power 0.31 / 0.34 / 0.36 at 0.32M / 1M / 5M groups; claim A 0.05 (`power/ASSURANCE_CLAIM_{A,B}.json`, `tree_completion.power.assurance`) |
| S2 | RAW_J_NEQ INNER utility not disclosed | 20/24 points (0/24 simultaneous) disclosed |
| S3 | "T_REFIT E20 equals the teacher" | "indistinguishable from the teacher", difference 0.0019 [−0.0024, 0.0065], descriptive |
| S4 | "Reproduced exactly" | 30/30 within 1e-12 (28 bitwise) |
| S5 | False precision of sample sizes | Monte Carlo 95% interval reported (312k–321k), "round up for planning" stated |
| S6 | Prevalence threshold missing | claim B becomes infeasible above a minority share of about 0.362 under the label-shift model |
| S7 | NY 2018 described as reserved | now "conditional fallback of the reserved TX 2018 cohort" |
| S8 | 0.975^30 coverage stated without its assumption | now "under independence" |
| S9 | Claim B's comparator choice used exposed INNER outcomes | disclosed in CONFIRMATION_PROTOCOL_DRAFT, POWER_AND_FEASIBILITY and ADVISOR_BRIEF |
| S10 | Claim B worded "not worse than the best code" | now "better than the best code control by a positive margin" (superiority > 0) |
| S11 | Advisor sample-size section unlabelled | labelled "claim B" |
| S12 | Report label strings drift from the lock | documented in LOCK_NOTES (wording only) |

Confirmed without change:
- C1/C2/C3 values and statuses; K=11, critical value 2.8405, 2,048 draws, 2,234 groups.
- FARE_TC: 4/24 points, occupation accuracy 0.444.
- RAW_J_NEQ: feasible only at ρ1 E2.
- All power-table cells, privacy-only results and Brier-shift rows; the resampling check; the Adult 48,842 accounting.
- Source-study figures, and 46/46 admission checks.
- No causal contract wording remains, and claim B is never treated as passage of A.
