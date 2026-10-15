# Start here: two-recipient task-wire adapter (PCRL, Adult development data)

**Question:** does adapting **both** recipients' transmitted predictions allow a useful coalition improvement that occupation-only adaptation could not achieve?

**Answer:** no.
- **Completion:** `COMPLETE`. That covers execution, independent verification (8/8 checks PASS), deployment, same-host backup and restore, and publication. The off-device copy is pending.
- **Scientific label:** `POINT_FEASIBLE_NO_COMPETITIVE_CRITERION`, recomputed independently with the same result.
- **Recipe:** closed.

## In plain language
1. **Did changing income help?** No. Income-only adaptation moved pair recovery by +0.0004 (more recovery, not less). Adding J_BOTH's changed income to its changed occupation raised pair recovery by 0.0024. Both are unresolved.
2. **Did both useful tasks survive?** At the point level, yes: all 24 original point clauses pass for J_BOTH on FIT2 and INNER. Under simultaneous bounds, J_BOTH resolves only 5 of 24 clauses, against 10 of 24 for RAW_PROJ.
3. **Did each recipient and their combination reveal less?** Only slightly, and nothing is established. Relative to RAW_PROJ, J_BOTH's point reductions are:

   | view | reduction (point) |
   |---|---|
   | income | 0.0037 |
   | occupation | 0.0055 |
   | pair | 0.0036 |

   The pair benefit interval is [−0.0025, +0.0097], an established shortfall against 0.020.
4. **Did joint training beat local training and both sequential orders?** No. The pair differences are −0.0007 vs L_BOTH, −0.0009 vs S12 and −0.0000 vs S21.
5. **Did income uncertainty improve?** It remained unresolved for 10 of 12 income clauses (upper bounds up to 0.0226). The 2 new passes are also obtained by the task-only arm.
6. **What earned evidence, and what failed?**
   - **Earned:** an exactly matched, verified six-arm bank whose privacy term demonstrably changes both wires; established occupation-only reductions for the local arm L_BOTH (0.0099) and the inherited occupation adapter (0.0100).
   - **Failed:** the coalition criterion, any joint-over-local or joint-over-sequential advantage, any coalition benefit from changing income, and income utility resolution.

## Every attacked service, both seeds (INNER)
AUC columns are complete-envelope SEX recovery from fresh readers. The last column counts simultaneous-bound PASS clauses across both seeds (shown on the seed-0 row).

| service | status | seed | income AUC | occupation AUC | pair AUC | income NLL excess vs U0 | occupation NLL excess vs U0 | point clauses (seed) | simultaneous PASS (both seeds, /24) |
|---|---|---|---|---|---|---|---|---|---|
| RAW_PROJ | REFERENCE_POINT_ELIGIBLE | 0 | 0.6861 | 0.8302 | 0.8514 | 0.0059 | 0.0037 | 12/12 | 10 |
| RAW_PROJ | REFERENCE_POINT_ELIGIBLE | 1 | 0.6928 | 0.8264 | 0.8408 | 0.0061 | 0.0016 | 12/12 |  |
| O_PREVIOUS | INHERITED_COMPONENT_REFERENCE_POINT_ELIGIBLE | 0 | 0.6861 | 0.8192 | 0.8468 | 0.0059 | 0.0055 | 12/12 | 5 |
| O_PREVIOUS | INHERITED_COMPONENT_REFERENCE_POINT_ELIGIBLE | 1 | 0.6928 | 0.8175 | 0.8357 | 0.0061 | 0.0027 | 12/12 |  |
| T_BOTH | SELECTED_POINT_ELIGIBLE | 0 | 0.6862 | 0.8303 | 0.8475 | 0.0060 | 0.0033 | 12/12 | 10 |
| T_BOTH | SELECTED_POINT_ELIGIBLE | 1 | 0.6944 | 0.8270 | 0.8429 | 0.0061 | 0.0021 | 12/12 |  |
| L_BOTH | SELECTED_POINT_ELIGIBLE | 0 | 0.6821 | 0.8184 | 0.8453 | 0.0072 | 0.0056 | 12/12 | 3 |
| L_BOTH | SELECTED_POINT_ELIGIBLE | 1 | 0.6898 | 0.8184 | 0.8383 | 0.0063 | 0.0030 | 12/12 |  |
| J_BOTH | SELECTED_POINT_ELIGIBLE | 0 | 0.6818 | 0.8245 | 0.8460 | 0.0075 | 0.0057 | 12/12 | 5 |
| J_BOTH | SELECTED_POINT_ELIGIBLE | 1 | 0.6898 | 0.8212 | 0.8390 | 0.0062 | 0.0024 | 12/12 |  |
| S12 | SELECTED_POINT_ELIGIBLE | 0 | 0.6829 | 0.8183 | 0.8466 | 0.0072 | 0.0046 | 12/12 | 5 |
| S12 | SELECTED_POINT_ELIGIBLE | 1 | 0.6902 | 0.8224 | 0.8366 | 0.0065 | 0.0021 | 11/12 |  |
| S21 | SELECTED_POINT_ELIGIBLE | 0 | 0.6838 | 0.8194 | 0.8455 | 0.0079 | 0.0058 | 11/12 | 5 |
| S21 | SELECTED_POINT_ELIGIBLE | 1 | 0.6899 | 0.8178 | 0.8395 | 0.0078 | 0.0027 | 11/12 |  |
| I_ONLY | SELECTED_POINT_ELIGIBLE | 0 | 0.6829 | 0.8302 | 0.8492 | 0.0072 | 0.0037 | 12/12 | 10 |
| I_ONLY | SELECTED_POINT_ELIGIBLE | 1 | 0.6902 | 0.8264 | 0.8439 | 0.0065 | 0.0016 | 11/12 |  |
| HYB_JI_RO | FIXED_WIRE_SUBSTITUTION_DIAGNOSTIC | 0 | 0.6818 | 0.8302 | 0.8488 | 0.0075 | 0.0037 | 12/12 | 12 |
| HYB_JI_RO | FIXED_WIRE_SUBSTITUTION_DIAGNOSTIC | 1 | 0.6898 | 0.8264 | 0.8439 | 0.0062 | 0.0016 | 12/12 |  |
| HYB_RI_JO | FIXED_WIRE_SUBSTITUTION_DIAGNOSTIC | 0 | 0.6861 | 0.8245 | 0.8442 | 0.0059 | 0.0057 | 12/12 | 3 |
| HYB_RI_JO | FIXED_WIRE_SUBSTITUTION_DIAGNOSTIC | 1 | 0.6928 | 0.8212 | 0.8361 | 0.0061 | 0.0024 | 12/12 |  |

## Primary family (K = 39)
In every row the income and occupation no-excess bounds pass, except two occupation bounds, which are unresolved.

| J_BOTH vs | pair benefit | status |
|---|---|---|
| RAW_PROJ | +0.0036 [−0.0025, +0.0097] | established shortfall |
| T_BOTH | +0.0027 [−0.0041, +0.0095] | established shortfall |
| L_BOTH | −0.0007 [−0.0067, +0.0052] | established shortfall (occupation bound unresolved) |
| S12 | −0.0009 [−0.0064, +0.0046] | established shortfall |
| S21 | −0.0000 [−0.0058, +0.0058] | established shortfall (occupation bound unresolved) |

## Figures (attacked services only)
- `figures/RECOVERY_BY_SERVICE.png`
- `figures/PRIMARY_PAIR_BENEFIT.png`
- `figures/UTILITY_VS_PAIR_RECOVERY.png`

## Files
- **Decision and answers:** `RESEARCH_DECISION.md`.
- **Exposure, roles and lock chronology:** `SOURCE_AND_EXPOSURE.md`.
- **Mechanism, bank and accounting:** `RECIPE_AND_CONTROLS.md`.
- **Endpoints:** `PRIMARY_ENDPOINTS.csv`, `DIAGNOSTIC_ENDPOINTS.csv`, `INNER_SERVICE_TABLE.csv`, `UTILITY_BOUNDS_ALL_SERVICES.csv`, and `INNER_REPORT.json`.
- **Verification:** `INDEPENDENT_VERIFICATION.json` and `independent/`.
- **Reviews:** `MATH_REVIEW_PREFIT.md` and `CLAIMS_REVIEW.md`.
- **Locks:** `PROTOCOL_SEAL.json`, `SCIENCE_LOCK.json`, `TRAINING_COMPLETE_LOCK.json`, `SELECTION_LOCK.json`, `ATTACKS_AUTHORIZATION.json` and `REPORTING_LOCK.json`.
- **Deployment:** `QUICKSTART.md` and `DEPLOYMENT_RESULTS.json`.
- **Backup:** `BACKUP_VERIFICATION.json` (same host only; off-device pending).
- **Manifests:** `UNIT_MANIFEST.json`, `MODEL_MANIFEST.json`, `RESOURCE_INVENTORY.json` and `AGENT_ROSTER.json`.
- **Handoffs:** `HANDOFF.json`, `CLAIMS_HANDOFF.json` and `PAPER_ADDENDUM.md` (manuscripts not edited).

## Limits
- **Exposure:** exploratory development on exposed people; the design came after the previous results.
- **Conditioning:** intervals are conditional on the fitted models and readers, and the seeds share people.
- **Hybrids:** they are wire substitutions.
- **O_PREVIOUS:** an inherited component.
- **Sequential arms:** S12/S21 are learned sequential controls, not Taylor et al.'s algorithm.
- **What is not claimed:** novelty or a population privacy guarantee.
