# Start here: ACS development replication of the protected-tree recipe

**Answer:** relative to matched continued task fitting, the local recipe lowered each recipient's own SEX recovery, but lowered the linked coalition's recovery by less than the pre-registered 0.020 margin.

**Key results** (Florida ACS 2019; seed 0 / seed 1):
- **Utility:** all 24 utility clauses pass at the point. This point gate has no error control. In the simultaneous intervals, 14 clauses pass, 10 are unresolved and none is an established violation.
- **Individual recipients:** against matched continued task fitting, each recipient's own SEX recovery fell by 0.014–0.018 AUC for income and 0.061–0.067 for occupation. All four simultaneous intervals lie wholly below zero.
- **Linked coalition:** recovery fell only 0.007 and 0.0175 AUC against matched continued task fitting, below the pre-registered 0.020 margin at the point. That is an established shortfall on seed 0 and unresolved on seed 1. The seed-1 reduction is positive: its simultaneous interval [+0.006, +0.029] lies above zero.
- **Neural control:** the eligible neural adversarial control had lower coalition recovery on both seeds at the point, by 0.017 and 0.011 AUC. For the recipe, this is an established shortfall on seed 0 and unresolved on seed 1.
- **Subgroups:** unresolved. In the K = 40 simultaneous intervals, the women's occupation-Brier guard passes on seed 1 and is unresolved on seed 0. The men's is unresolved on both seeds. No harm is established.
- **Verification:** an independent verifier with separate code reproduced every decision and every result value it recomputed from the stored outputs; it only spot-checked the numbers in the documents. Ten of 11 checks pass; check 1 fails on code labelling only (disclosure D1 in `LOCK_AMENDMENTS.md`). A separate claims review then corrected the wording of the documents (`CLAIMS_REVIEW.md`).

**Labels:**
- `RECIPE_EFFECT_NOT_ESTABLISHED`
- `NOT_SUPERIOR_TO_ALL_ELIGIBLE` (complete)
- `SUBGROUP_UNRESOLVED`
- Benchmark `SUPPORTED` (the usefulness gate: the teacher beats constant prediction, and class support passed)

**Scope.** This is a recipe replication with new models, inputs and tasks, i.e. development evidence. It is not confirmation of the frozen Adult packages. DEV_REPORT was opened once for analysis. The independent verifier later re-read it only to recompute the reported numbers (`LOCK_NOTES.md`).

**Sealed 20% of households: closed.**
- Nothing about these households was written or used beyond counts and a key-set hash, and no model saw them.
- The extraction's single eligibility pass did parse their records in memory before roles were assigned (disclosure D2; `SEALED_ASSESSMENT_STATUS.md`).
- Opening them is not justified: the claims are FUTILE at the development values.

**Read in this order:**
1. `ADVISOR_BRIEF.md`: the six questions, answered plainly.
2. `RESEARCH_DECISION.md`: full decision, tables, status ledger, forecasts, next decision and limits.
3. `POPULATION_AND_SCHEMA.md`: sources, population choice, schema and task mapping, household roles. Also `POPULATION_DECISION.json`, `SOURCE_ADMISSION.json`, `OCCUPATION_MAPPING.json`, `EXTRACTION_SUMMARY.json`.
4. Tables:
   - `PRIMARY_ENDPOINTS.csv` (K = 40)
   - `SERVICE_TABLE.csv` (all services, seeds, views and utility)
   - `SUBGROUP_TABLE.csv` (`sex0` = male, SEX 1; `sex1` = female, SEX 2)
   - `DEV_REPORT_RESULTS.json` (everything, including planning and sensitivity)
5. Figures:
   - `figures/UTILITY_VS_COALITION_RECOVERY.png`
   - `figures/RECIPIENT_RECOVERY.png`
   - `figures/SUBGROUP_OCCUPATION_BRIER.png`
6. Baseline audit: `baselines/FAGTB_AUDIT.md`.
7. Verification and runnable outputs:
   - `INDEPENDENT_VERIFICATION.json` and `independent/`
   - `CLAIMS_REVIEW.md`
   - `DEPLOYMENT_RESULTS.json`
   - `MODEL_MANIFEST.json`
   - `QUICKSTART.md`
   - `BACKUP_VERIFICATION.json`
   - `SEALED_ASSESSMENT_STATUS.md`
8. Locks, in order:
   - `PROTOCOL_SEAL.json`
   - `FORECASTS.json`
   - `STATISTICAL_REVIEW_PRELOCK.md`
   - `SCIENCE_LOCK.json` (publicly registered before any label: `85b2a41`)
   - `LOCK_AMENDMENTS.md` (amendment A1; post-report disclosures D1, D2 and D3)
   - `STAGE_A_HEADROOM.json`
   - `TRAINING_COMPLETE_LOCK.json`
   - `SELECTION_LOCK.json` and `SELECTION_DETAIL.json`
   - `REPORTING_LOCK.json` (public before DEV_REPORT opened: `7d23e0e`)
   - `LOCK_NOTES.md`
   - `FORECAST_SCORES.json`
9. Handoffs:
   - `MANUSCRIPT_CLAIMS_HANDOFF.md`
   - `RESOURCE_INVENTORY.json`
   - `AGENT_ROSTER.json`
