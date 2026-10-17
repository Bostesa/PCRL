# Claims review disposition (coordinator)

All 14 must-fix items were applied before publication. No endpoint, label or result file of the logged evaluation was changed. One supplementary label-free unit was added (FFV_MAPPING_CONFLICTS_V2, for items M12 and M13).

| Item | Done |
|---|---|
| M1 | RESEARCH_DECISION item 4 and CONFIRMATION_FEASIBILITY now read: at about 17,500 households, D1 power is about 0.01 under the governing ENHANCED scorecard (about 0.25 under ORIGINAL), and D2 about 0. The feasibility study-size table gains a 17,500-household column. |
| M2 | "Recovery is not binding" removed from RESEARCH_DECISION, START_HERE and CONFIRMATION_FEASIBILITY. The ENHANCED seed-1 income recipient clause against H_LJ is stated as UNRESOLVED at DEV's size, needing about 318,000 households whatever the utility headroom, and setting the ENHANCED requirement at slack 0.002. It is also used in the argument against option 1. |
| M3 | The item-4 table now gives slack 0.002 as "the largest on the registered grid", with D2 values and the ENHANCED governing value "set by its recipient clause". |
| M4 | The feasibility results header states which scorecard governs where. The slack-0 row gains its ORIGINAL brackets. |
| M5 | START_HERE gains a status block: development robustness evidence on already-used records; 10 prior DEV_REPORT reads plus this campaign's two; not confirmation; selection history not erased; seeds share people. |
| M6 | RESEARCH_DECISION "Status of the evidence" now says the two seeds share people and are robustness to the release seed, not replication. |
| M7 | START_HERE and RESEARCH_DECISION list the 7 existing packages, say "same-device backup", and state the NEURAL_RAW seed-1 coverage gap. |
| M8 | QUICKSTART: full-role batches (and the reviewer's 600-row batch) reproduce bitwise; 1-, 6- and 8-row requests do not. Affected levels named. Threshold batch size not determined. |
| M9 | QUICKSTART: the wrapper checks field names and order only. Values are not validated, and the valid domain is not enforced at deployment. |
| M10 | QUICKSTART: Census and SEX context line added; the backup row is marked research-branch-only, with the local environment noted. |
| M11 | The recommendation is stated as conditional on `INDEPENDENT_VERIFICATION.json` (otherwise option 4). The option-4 paragraph mentions the post-lock feasibility correction. Publication waits for the verifier and for `BACKUP_VERIFICATION.json`. The verifier was told to check feasibility V2. |
| M12 | OUTPUT_MAPPING_CHECK now gives the exact union counts, 5 and 46 shared vectors (excess 6 and 49; at most 3 and 4 inputs per vector), from the new `MAPPING_CONFLICTS.json`. |
| M13 | The conflict script is in the repository (`ffv/mapping_conflicts.py`), its identifier-free result is saved (`MAPPING_CONFLICTS.json`, unit FFV_MAPPING_CONFLICTS_V2), and the sourcing sentence cites it. |
| M14 | RESEARCH_DECISION now calls the within-role relation "near-vacuous": 93–97% of vectors occur once, and only 2 and 13 within-role vectors are shared. It is not evidence about information content. |

## Optional items

**Applied:**

| Item | Done |
|---|---|
| O1 | "From backup alone" qualified in QUICKSTART: the backup's own code and data, with the local interpreter and libraries. |
| O2 | "Not added exposure" scoped to the change between banks. The failing clause is UNRESOLVED, with an upper bound of +0.0235. Selection instability is two-sided: the finalist's exact income tie, and H_LJ's seed-0 change. |
| O3 | "Against these frozen fitted readers", plus the bootstrap-trained-reader shrinkage (3 of 4 cells; −0.001 to +0.031). |
| O4 | M's frozen interpretation is stated as READER_SENSITIVE through step 2. |
| O5 | Threshold precision: closest at 4.5e-6 (ATTACK_FIT, seed 0); neither seed-1 move crosses the decision; 2.2e-16 and 4.4e-16. |
| O6 | Batch-size coverage: H_LJ tested on seed 0 only; TASKONLY untested. |
| O7 | KeyError and non-finite-value nuance. |
| O8 | "Replication disclosure D2" and "source deviation D3" prefixes where both meanings meet. |
| O9 | Full Census wording in CONFIRMATION_FEASIBILITY and OUTPUT_MAPPING_CHECK. |
| O10 | Conflict wording: "two occupation vectors each carry two income codes (levels 2 and 3), which straddle the income decision". |
| O11 | The stale CV condition string is disclosed in PROTOCOL_DEVIATIONS item 11. |
| O12 | The τ values used in S4 are stated. |
| O13 | "Not achievable in any study" replaced by "exceeds the number of households in Florida". |
| O14 | DEV_REPORT read counts stated. |
| O15 | The aggregation script is committed (`ffv/aggregate_ties.py`). |

**Not applied:** none.
- Privacy redaction: the home-directory path in this review's appendix was replaced by `/Users/<you>` (no other change).
