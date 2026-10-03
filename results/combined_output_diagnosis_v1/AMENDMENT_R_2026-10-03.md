# Amendment R: reporting repairs, 2026-10-03

**Scope.** Repairs made on branch `research/combined-output-diagnosis-v1`, cut from f7425b15. The original files remain
at f7425b15; their hashes are in `repaired/ORIGINAL_HASHES.json`. The per-item ledger is `ORIGINAL_VS_REPAIRED.csv`.

**What was not changed.** No estimand, threshold, selection rule, role row set or decision changed. Every role row-set
hash is identical before and after; this is asserted in `scripts/s0_repairs.py`.

| ID | Defect | Repair | Regression test |
|---|---|---|---|
| R1 | `oar/study.py` stored roles as `<U16`, truncating `excluded_exposure` (17 characters). The exclusion counts in ROLES_AND_SUPPORT.json were reported as 0. The exclusion itself was always applied. | The roles array is now `<U32`. Exclusions are recomputed from record identities: `canon_key` ∈ encoder-training split. Adult: 17 rows / 17 groups (6 / 4 / 7). HMDA: 42 / 42 (21 / 7 / 14). | `odx/tests/test_repairs.py::test_role_labels_not_truncated`, `::test_load_world_counts_exclusions_from_identities` |
| R2 | Amendment A1's field `cert_rows_with_feature_vector_equal_to_a_fit_row` counted *distinct vectors* (the wrapper's `row_hashes` deduplicates). | `odx/equality.feature_equality_counts` reports affected rows and distinct vectors separately, for cert, attacker and assessment rows, every seed (`repaired/FEATURE_EQUALITY_COUNTS.json`). HMDA s1: **1,138 / 1,385** cert rows share **one** vector with fit rows, a collapsed representation. | `::test_feature_equality_counts_rows_and_distinct_vectors` |
| R3 | RESEARCH_DECISION.md (output-aware) §3 listed the HMDA P3 / P4 *upper* endpoints (−0.0010, −0.0001) as lower bounds. | Corrected from PRIMARY_ENDPOINTS.csv: lower bounds −0.0062 and −0.0057. The decisions (PASS) are unchanged. | Values read from the committed endpoint arrays |
