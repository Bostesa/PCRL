# Disposition of the claims review (coordinator, 2026-10-17)

Review: `roles/claims_reviewer/CLAIMS_REVIEW.md` (10 MUST-FIX, 14 SHOULD-FIX, notes), written against commit `b9299804f`. Every item was checked against the frozen records before it was applied (the extra numbers it introduces were re-read from `report/DEV_REPORT_RESULTS.json` and `TRAJECTORY_SUMMARY.json`). No number, label or decision in the frozen records changed; only the prose documents and the presentation generator did.

| Item | Disposition |
|---|---|
| MF-1 qualifiers, scope caveats and recovery qualifiers in START_HERE | Applied as proposed |
| MF-2 "matched" only for the DEV_REPORT point match | Applied in START_HERE, RESEARCH_DECISION and the handoff ("MODEL_SELECT-matched", "MODEL_SELECT match") |
| MF-3 failure categories and full conjunctions | Applied: claim table with first failing and other false conditions; §5 categories; §7 restructured (family-established, point results, unresolved, unavailable, diagnostic) |
| MF-4 sign of the MODEL_SELECT mismatch | Applied: SCM − control −0.00003 / +0.00015 |
| MF-5 no extrapolation past the failed match | Applied: no ordinary release reached SCM's DEV spend; ranges corrected (0.645–0.649 / 0.643–0.648) |
| MF-6 PUB_IBSI and SEX-guidance attributions | Applied: unresolved, not equivalence, not cost-matched; comparator DEV clause qualifier; income qualifier; reader budget favours PUB_IBSI |
| MF-7 common-cell check scope | Applied: grouped releases only; not computed for NEURAL_RAW or H_LJ |
| MF-8 reader-budget statistics and placement | Applied: medians, 2.3 / 2.8 times, beside §2, §3, §4 and §5 |
| MF-9 verifier replay wording | Applied: registered; result recorded in the verifier files (updated when they land) |
| MF-10 class-0 frequency | Applied: seed 0 more, seed 1 less |
| SF-1 to SF-9, SF-12 to SF-14 | Applied as proposed |
| SF-10 QUICKSTART refusals, step 9, custody | Applied as proposed |
| SF-11 SELECTION_AND_MATCHING | Applied in the generator (`scm_um/tables.py`) and regenerated (signed Δ columns, MODEL_SELECT role-status note, no dict syntax) |
| N-1 rounding | Applied (0.0049; SCM MODEL_SELECT 0.0082) |
| N-3 receipt times | Commit times added; push times cited from the push log in HANDOFF.json |
| N-4 gate count | Applied: "51 checks plus the summary flag" |
| N-6 superseded commit message | Supersession note added at the end of RESEARCH_DECISION.md |
