# Brief: deployment, reporting and closeout reviewer (SCM utility-matched follow-up)

You review the finished record for accuracy of every claim and for completeness against the commission. You do not recompute
numbers (the independent verifier does); you check that every sentence in `START_HERE.md`, `RESEARCH_DECISION.md`,
`SELECTION_AND_MATCHING.md`, `QUICKSTART.md` and the handoffs is supported by the frozen records (`report/DEV_REPORT_RESULTS.json`,
`SELECTION_LOCK.json`, `TRAJECTORY_SUMMARY.json`, `DEPLOYMENT_RESULTS.json`, `BACKUP_VERIFICATION.json`, `INDEPENDENT_VERIFICATION.json`,
`RESOURCE_ACCOUNTING.json`), with the right qualifiers. Write `roles/claims_reviewer/CLAIMS_REVIEW.md` with MUST-FIX / SHOULD-FIX / NOTES,
each with the exact replacement text. Do not edit other files and do not commit.

Check in particular:
1. Labels are quoted exactly from the report, as explicit conjunctions, with the first failing condition and its failure category
   (UNAVAILABLE, POINT_FAILURE, ESTABLISHED_SHORTFALL / VIOLATION, POINT_SHORTFALL / EXCESS, UNRESOLVED) kept distinct.
2. "Matched" is used only for the DEV_REPORT point match; a MODEL_SELECT match that did not carry over is said so; a point match is never
   called equivalence; the direction sub-status says only who spent more.
3. Matching occupation NLL is never described as matching the whole utility vector; income, Brier, accuracy and subgroup differences are shown.
4. Recovery advantages are qualified by the fixed reader budget (finer partitions can be under-read), the redundancy of the linked pair
   when the occupation output identifies the shared cell, and the fact that the lookup reader is not population-optimal.
5. Every result is described as exploratory engineering evidence on previously used development records; no novelty, new-algorithm or
   joint-design claim; the sealed assessment stayed closed; intervals are conditional on fitted releases and readers and exclude the
   development and selection history; seeds are not separate populations.
6. The historical results and labels of the source benchmark are unchanged and are never re-labelled.
7. Deployment claims match `DEPLOYMENT_RESULTS.json` (both entry points, refusals, which roles are packaged or unavailable).
8. Custody claims distinguish same-device, warm off-device read-back and a cold read after remounting.
9. Resource figures match the ledger; failed attempts and stopped work are counted.
10. The manuscript and claims handoffs change no other branch and state only what the record supports.
