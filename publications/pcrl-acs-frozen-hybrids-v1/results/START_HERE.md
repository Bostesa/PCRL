# Start here: ACS frozen hybrids (exploratory development follow-up)

**Answer.** The fixed composition H_LJ joins LOCAL_TREE's income output to JOINT_TREE's occupation output. It keeps the trees' point utility; simultaneous utility confidence and subgroup suitability are unresolved.
- **Against continued task fitting:** it lowers coalition SEX recovery by 0.022 and 0.024 AUC. Both points sit just above the 0.020 margin, but the clause bounds (0.012 and 0.014) fall below it. The margin is therefore not established: unresolved, not a shortfall.
- **Against the neural control:** H_LJ is not superior to the eligible neural control, and that is the limiting comparison. Its coalition AUC is 0.001 and 0.005 higher than NEURAL_RAW's at the point, and the 0.020 superiority target is an established shortfall.

Overall label: **`H_LJ_FIXED_COMPOSITION_NOT_SUPPORTED`**. This fixed hybrid test is closed; no nearby sweep is proposed.

> Exploratory development result on already-opened DEV_REPORT data; the composition was fixed after its components' DEV_REPORT results were public, so error rates are nominal; conditional on the two fitted seed pairs (shared people, not independent samples), the frozen defense models and the fitted readers, in the unweighted DEV_REPORT household superpopulation; not confirmation; source labels unchanged.

**Registration.** This is an exploratory development follow-up of the already-opened ACS DEV_REPORT results. It is not confirmation, not an independent replication and not a rescue of the source claim. The source study's labels stand unchanged: `RECIPE_EFFECT_NOT_ESTABLISHED`, `NOT_SUPERIOR_TO_ALL_ELIGIBLE` (complete) and `SUBGROUP_UNRESOLVED` (benchmark `SUPPORTED`).

**Labels.**
- Claim A: `RECIPE_BENEFIT_NOT_ESTABLISHED`.
- Claim B: `NOT_SUPERIOR_TO_ELIGIBLE_NEURAL_CONTROL`, with eligible set {NEURAL_RAW}. FAGTB and FARE had no MODEL_SELECT-feasible configuration and were not compared. The nested 0.020 target is `NOT_ESTABLISHED`; its K = 38 status is `ESTABLISHED_SHORTFALL` on both seeds.
- Claim C: `SUBGROUP_UNRESOLVED` (settled before readers).
- Controls: `CONTROLS_VALID`.
- Separately reported:
  - overall point feasibility: `POINT_FEASIBLE_24` on DEV_REPORT and MODEL_SELECT;
  - overall confidence: `CONFIDENCE_UNRESOLVED` (10 PASS, 14 UNRESOLVED, 0 violations).
- Independent verification: **10/10 PASS**, using separate code; 17 of the 250 reader fits were refit from scratch and reproduced bitwise.

**Key numbers** (coalition SEX-recovery AUC on DEV_REPORT; seed 0 / seed 1; lower reveals less to the fitted attackers).

| service | coalition AUC | role |
|---|---|---|
| H_LL = LOCAL_TREE | 0.657 / 0.649 | source primary |
| **H_LJ** | **0.641 / 0.643** | preselected primary |
| H_JL | 0.625 / 0.625 | diagnostic; utility-ineligible (JOINT income fails); post-hoc, not a nominee |
| H_JJ = JOINT_TREE | 0.603 / 0.606 | diagnostic; utility-ineligible (JOINT income fails); post-hoc, not a nominee |
| NEURAL_RAW ρ1.0 E2 | 0.640 / 0.638 | eligible neural control |
| TASK_REFIT E20 | 0.664 / 0.666 | matched control |
| T_TREE | 0.664 / 0.664 | teacher (diagnostic) |

Only the H_LJ and H_JL pair AUCs are new. Every other value reproduces a published source value exactly.

- **vs TASK_REFIT:** H_LJ is lower by 0.022 [0.008, 0.037] and 0.024 [0.011, 0.036] (K = 38 intervals). Its one-sided clause bounds, 0.012 and 0.014, fall below 0.020.
- **vs NEURAL_RAW:** H_LJ's benefit (NEURAL_RAW AUC minus H_LJ AUC) is −0.001 [−0.018, 0.015] and −0.005 [−0.021, 0.011]. H_LJ is slightly higher at the point, and neither difference is established.
- **vs H_LL:** H_LJ is lower by 0.015 [0.006, 0.024] and 0.006 [−0.004, 0.016].
- **Where the large reductions come from** (descriptive, post-hoc): the combinations that include JOINT_TREE's income output (H_JL and H_JJ). That output fails income utility, so these combinations are utility-ineligible diagnostics, a post-hoc observation and not new nominees.

**Scope notes.**
- No defense model was trained. Only recovery attackers and pipeline-control readers were fitted.
- The Florida sealed assessment stayed closed, and nothing here justifies opening it. The source's D2 disclosure still applies: the source extractor computed sealed people's inputs, SEX, weights and both task labels transiently in memory and discarded them unwritten, keeping only counts, a key-set hash and one data-quality count. The sealed assessment is not described as never accessed.
- The source's independent verification remains 10/11. Check 1 failed on D1, a transport-annotation (code-labelling) failure, not a numerical failure.

**Read in this order.**
1. `ADVISOR_BRIEF.md`: the six questions, answered plainly.
2. `PROTOCOL.md` (a summary) and `ANALYSIS_LOCK.json` (the binding record), both committed and publicly receipted before any reader. The lock also fixes the favourable-alternative rule, the three separate status fields and the conjunction. See also `STATISTICAL_REVIEW_PRELOCK.md` and `FORECASTS.json`.
3. Results:
   - `DEV_REPORT_RESULTS.json` (everything);
   - `PRIMARY_ENDPOINTS.csv` (K = 38);
   - `SECONDARY_ENDPOINTS.csv` (K = 22);
   - `DISTANCES.csv` (signed distances from 0, 0.020 and the caps);
   - `FOUR_HYBRID_TABLE.csv`, `SERVICE_TABLE.csv`, `SUBGROUP_TABLE.csv`;
   - `figures/RECOVERY_VS_UTILITY.png`. Its bars are marginal 95% intervals for absolute AUCs. Paired comparisons are much tighter, so use the family intervals in the endpoint tables. H_JL and H_JJ are utility-ineligible diagnostics and not nominees. In the occupation panel, H_JL's seed-0 marker is hidden behind its seed-1 marker.
4. Provenance:
   - `SOURCE_ADMISSION.json` (source pinned to `f4ea199`; 112 outputs recomputed from frozen models);
   - `FROZEN_BANK_MANIFEST.json` (the hashed four-combination bank);
   - `READER_ACCOUNTING.json` / `.csv` (250 effective fits, aliases, replay against the source).
5. Locks and notes: `REPORTING_LOCK.json` (readers frozen and Stage-1 controls passed before truth), then `LOCK_NOTES.md`.
6. Verification and artifacts:
   - `independent/INDEPENDENT_VERIFICATION.json` and the per-check files in `independent/`;
   - `DEPLOYMENT_RESULTS.json`, `DEPLOYMENT_MANIFEST.json`;
   - `QUICKSTART.md`;
   - `BACKUP_VERIFICATION.json`, `RESOURCE_INVENTORY.json`, `FORECAST_SCORES.json`, `AGENT_ROSTER.json`.
7. Handoffs: `MANUSCRIPT_CLAIMS_HANDOFF.md`, `CLAIMS_REVIEW.md`.

**Kept private.** People, labels, predictions, reader models, deployment packages and the frozen bank all live under `.private/acs_frozen_hybrids/` and the private backups.
