# Claims and closeout reviewer brief (SCM engineering benchmark)

**Context.** This is statistical machine-learning research on public ACS Census tabular records. It asks whether income and occupation probability releases reveal the recorded field SEX, and whether Shared-Cell Merging (SCM, an existing-method combination) preserves usefulness while reducing that recovery. It is an engineering benchmark: no novelty claim is permitted.

**Your job.** Check that every statement in the coordinator's closeout documents matches the computed records and stays inside its scope:
- `START_HERE.md`;
- `DECISION.md`;
- `RESULTS_TABLES.md`;
- `HANDOFF.json`.

Write `results/pcrl_scm_engineering_benchmark_v1/roles/claims_reviewer/CLAIMS_REVIEW.md` with MUST-FIX, SHOULD-FIX and NOTE items, each with exact replacement text. Edit nothing else, and open no private arrays.

**Sources of truth:**
- `report/DEV_REPORT_RESULTS.json`;
- `report/PRIMARY_ENDPOINTS.csv` and `report/SECONDARY_ENDPOINTS.csv`;
- `report/RECOVERY_AND_UTILITY_TABLE.csv`;
- `SELECTION_LOCK.json`;
- `FIT_SUMMARY.json`;
- `READERS_SUMMARY.json`;
- `ANALYSIS_LOCK.json`;
- `PROTOCOL.md`;
- `ENGINEERING_REPAIRS.md`;
- `roles/verifier/VERIFICATION.md`, once present.

**Check in particular:**
1. Every number quoted, with its sign and seed, against the records. Rounding must be correct.
2. No method win is claimed from an ineligible release, a missing comparator, low recovery caused by an unusable task, or a comparison outside the frozen claims.
3. Claim labels match the report:
   - recipe versus the matched twin, with the 0.020 margin;
   - attribution, tested in sequence inside A's slice;
   - competitive versus NEURAL_RAW and H_LJ;
   - subgroup;
   - point feasibility kept separate from confidence status.
4. The linked-pair statement is checked against actual ties and serialised outputs. A fitted lookup table is not called a Bayes oracle. Finite-cell merging is not called a guarantee against arbitrary readers.
5. The failure or success statement is scoped to this fixed recipe and its capacity, and the benchmark is called an existing-method engineering benchmark.
6. Budgets, failed attempts, engineering repairs, custody (on-device and off-device) and publication receipts are described accurately.
7. The algorithmic-bottleneck statement is supported by the measured evidence, not by speculation.
