# Start here: task headroom and protected prediction outputs

**Answer:**
- **Headroom:** CatBoost task teachers created real task headroom.
- **Spending it:** local privacy training of their leaf scores (L_TREE) cut linked-coalition SEX recovery by **0.0505 AUC [0.0257, 0.0754]** against the incumbent RAW_PROJ. Neither recipient got worse, and all 24 original point utility clauses pass.
- **Not established:** against three historical score-code releases, the gain is 0.014–0.018 (unresolved), so competitive protection is not established. Joint design added nothing.

**Labels:**
- `POINT_FEASIBLE_NO_COMPETITIVE_CRITERION`
- `JOINT_COMPONENT_NOT_ESTABLISHED`
- `TASK_MODEL_PREREQUISITE_MET`

**Read in this order:**
1. `ADVISOR_BRIEF.md`: the six questions, answered plainly.
2. `RESEARCH_DECISION.md`: the full decision, tables, status ledger, forecasts and next decision.
3. `ADVERSARY_AND_MOTIVATION.md`: who the attacker is, what it can and cannot access, and a replacement methodology passage.
4. `PRIOR_METHOD_DIFFERENCES.md` and `ORIGINAL_VS_NEW_METHOD.md`: what changed and what was reused.
5. `figures/UTILITY_VS_COALITION_RECOVERY.png` and `figures/RECIPIENT_RECOVERY.png`: every attacked service, including controls and failed nominees.
6. Endpoints:
   - `PRIMARY_ENDPOINTS.csv` (K = 78)
   - `DIAGNOSTIC_ENDPOINTS.csv`
   - `INNER_SERVICE_TABLE.csv` (per seed)
   - `UTILITY_BOUNDS_ALL_SERVICES.csv`
   - `INNER_REPORT.json`
7. Verification and runnable outputs:
   - `INDEPENDENT_VERIFICATION.json` and `independent/`
   - `DEPLOYMENT_RESULTS.json`
   - `QUICKSTART.md`
   - `MODEL_MANIFEST.json`
   - `BACKUP_VERIFICATION.json`
8. Locks, in order:
   - `PROTOCOL_SEAL.json`
   - `ADMISSION.json`
   - `SCIENCE_LOCK.json` (pushed prefit, `27f888e`)
   - `STAGE_A_REPORT.json`
   - `STAGE_B_LOCK.json` (pushed before protected fits, `4ba4c58`)
   - `TRAINING_COMPLETE_LOCK.json`
   - `SELECTION_LOCK.json` and `SELECTION_DETAIL.json`
   - `ATTACKS_AUTHORIZATION.json`
   - `REPORTING_LOCK.json`
   - `LOCK_NOTES.md` (disclosed deviations)
9. Handoffs:
   - `CLAIMS_HANDOFF.json`
   - `PAPER_ADDENDUM.md`
   - `CLAIMS_REVIEW.md`
   - `RESOURCE_INVENTORY.json`
   - `AGENT_ROSTER.json`

**Scope:** exploratory development on repeatedly used Adult people; prediction outputs only.
- **Release contract:** each recipient sees only its own probability vector and decision.
- **Adversary:** the coalition is two linked recipients, and linkage is assumed.
- **Not established:** feature services, other attributes, stronger adversaries, other populations, and any confirmation.
