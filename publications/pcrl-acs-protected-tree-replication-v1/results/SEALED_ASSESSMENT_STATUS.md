# Sealed assessment status: CLOSED

**What it is.** SEALED_ASSESSMENT is 20% of eligible Florida 2019 households: 9,853 households and 14,693 people.

**What was computed for it (disclosure D2, `LOCK_AMENDMENTS.md`).**
- **The parse.** The extraction made a single eligibility pass that parsed every eligible record before household roles were assigned (`acs_replication/extract.py`, `eligibility()`). For sealed people it therefore read the eligibility and target-validity variables. In process memory it computed their inputs, SEX code, PWGTP, RELSHIPP and both task labels.
- **The discard.** Those values were discarded when the extraction process ended. Apart from the counts, the key-set hash and the one data-quality count below, no file, table, statistic or model input was derived from them.
- **What was written.** The household and person counts and a sha256 of the sorted household-key set (`EXTRACTION_SUMMARY.json → sealed_assessment.household_key_set_sha256`). The counts depend only on eligibility, which uses the validity of income and occupation, never their values.
- **One data-quality count.** `household_resident_rows_with_GQ_relationship_code` (= 0) was computed over all eligible people, sealed people included.
- **The docstrings were wrong.** The extractor's docstring and this file's earlier text said no label was computed beyond the eligibility predicate. That was false; the claims review found it (`CLAIMS_REVIEW.md`, R11).

**What does not exist.** No feature matrix, label file, support table or weight file was written for these households, and no model prediction was computed. No person and no model saw any sealed feature, label or weight value. Only identifiers were read, by the metadata admission and by the verifier (below).

**How the closure is enforced.** These are operational controls, not cryptographic isolation on a fully privileged host:
- The exporter has no SEALED branch.
- The loader (`acs_replication/data.py`) refuses the role.
- The deployment code never receives sealed feature, label or weight values. The independent verifier read only the raw SERIALNO column, which includes sealed serials, to re-derive roles.

**Is opening it justified now? No.**
- At the development values, both the recipe effect (coalition benefit lower bound > 0.020 vs TASK_REFIT) and the competitive claim (superiority over the eligible neural control) are FUTILE: their points lie on the failing side of their margins (`DEV_REPORT_RESULTS.planning`).
- A favourable DEV_REPORT result would not have authorised opening it either.
- Opening it is a later, separately specified action, which this campaign does not take.

**Reuse.** If a new, separately registered method is ever developed and frozen, these households remain unused by any analysis and can serve it. The new study must:
- re-run the bounded prior-use audit;
- record that this campaign's extraction parsed these households' records in memory (eligibility and target-validity variables, inputs, SEX and both labels), wrote only counts and the key-set hash, and used nothing else.
