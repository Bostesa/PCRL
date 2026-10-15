# Start here: frozen-candidate control completion and confirmation feasibility

**Answer in three lines.**
1. The frozen L_TREE candidate's advantage over the incumbent is **not explained by the missing controls**:
   - vs the task-only continuation at the same epoch: 0.085 [0.065, 0.105];
   - vs a fresh equally exposed neural RAW-J: 0.066 [0.045, 0.088];
   - official FARE has lower recovery than L_TREE (an established shortfall for L_TREE) but is utility-infeasible, so it is reported descriptively and not counted as beaten.
2. The original confirmation criterion (claim A) is **not practically feasible**: it is impossible at the observed code margins. The narrower, new claim B needs about 317,000 independent groups (MC 95% 312k–321k), set by the occupation Brier slack.
3. **No compatible unexposed population was identified** for the frozen Adult release. All 48,842 Adult records are used or excluded; non-use elsewhere is UNKNOWN.

**Recommendation:** `ORIGINAL_CONFIRMATION_CRITERION_NOT_PRACTICALLY_FEASIBLE` + `NO_COMPATIBLE_UNEXPOSED_POPULATION_IDENTIFIED`. All new results are exploratory, on already-used development people. The source study's labels are unchanged.

**Read in this order:**
1. `ADVISOR_BRIEF.md`: plain answers.
2. `CONTROL_GAPS_AND_CORRECTIONS.md`: gaps A–F, resolutions, and corrected wording.
3. Control results:
   - `CONTROL_COMPLETION_RESULTS.csv` and `COMPLETION_REPORT.json` (K=11 family, labels, diagnostics);
   - `COMPLETION_SERVICE_TABLE.csv` and `COMPLETION_UTILITY_BOUNDS.csv`;
   - `COMPARATOR_EXPOSURE_MATRIX.csv`.
4. Power and design:
   - `POWER_AND_FEASIBILITY.md`, plus `power/` (joint-power tables, binding clauses, privacy-only and beyond-cap results, resampling check, aggregate inputs, and runnable `tree_completion/power.py`);
   - `STATISTICAL_DESIGN_IUT.md`.
5. `POPULATION_AUDIT.md` and `POPULATION_COMPATIBILITY_AND_USAGE.csv`.
6. `TARGETED_METHOD_NOVELTY.md`.
7. Confirmation preparation:
   - `CONFIRMATION_PROTOCOL_DRAFT.md`;
   - `confirmation/` (public protocol bindings, development and synthetic rehearsals);
   - `QUICKSTART.md` (runner; fresh execution hard-disabled).
8. `FROZEN_CANDIDATE_MANIFEST.json` and `FROZEN_ADMISSION.json`.
9. Locks, in order:
   - `PROTOCOL_SEAL.json`
   - `COMPLETION_LOCK.json` (public receipt `ba54ecc` on GitHub before any fit; scrub `4db5534`)
   - `SCIENCE_AUTHORIZATION.json`
   - `COMPLETION_TRAINING_LOCK.json`
   - `COMPLETION_SELECTION_LOCK.json` and `COMPLETION_SELECTION_DETAIL.json`
   - `ATTACKS_AUTHORIZATION.json`
   - `REPORTING_LOCK.json`
   - `LOCK_NOTES.md`
10. Verification and custody: `INDEPENDENT_VERIFICATION.json`, `independent/`, `BACKUP_VERIFICATION.json`, `RESOURCE_INVENTORY.json` and `AGENT_ROSTER.json`.
11. Handoffs: `MANUSCRIPT_CLAIMS_HANDOFF.md` and `CLAIMS_REVIEW.md`.
