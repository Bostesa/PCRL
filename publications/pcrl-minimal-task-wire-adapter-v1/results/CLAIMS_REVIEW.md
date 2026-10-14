# Claims review: pcrl_minimal_task_wire_adapter_v1 (2026-10-14)
Scope: START_HERE, RESEARCH_DECISION, RECIPE_AND_CONTROLS, SOURCE_AND_EXPOSURE, QUICKSTART vs committed evidence. About 120 numbers spot-checked. No .npz/.pt files loaded.

## MUST-FIX
1. INDEPENDENT_VERIFICATION.json does not exist in the results directory. These statements depend on it and must be marked PENDING until it exists and passes:
   - START_HERE: "Completion: COMPLETE (see INDEPENDENT_VERIFICATION.json)" and the chronology line "...independent verification, and publication".
   - RESEARCH_DECISION: "COMPLETE (execution and independent verification...)", Q1 "The verifier re-solved these independently", and Earned "privacy term verifiably changes".
   - RECIPE: "Independent check. The verifier independently re-solved these counterfactuals".
   - START_HERE: "verifiably changed".
   The SCIENCE_LOCK selection_rule.nominee_also_requires ">=1 independently verified privacy-driven accepted update". So the nominee status, and with it POINT_FEASIBLE_NO_COMPETITIVE_CRITERION, is itself conditional on verification. State this.
2. RESEARCH_DECISION Q5: "For the global guard, the drop decomposes as follows" / "Changed predictions account for the established part". This is wrong. D04 (0.0209) and D03 (0.0111) are not components of D02 (0.0116); they sum to 0.032. The registered contrasts give D02 = D05 − D03 = (D04 + D06) − D03 = 0.0209 + 0.0019 − 0.0111 ≈ 0.0116. GLOBAL keeps the hybrid residual, so residual removal is no part of its drop. Suggested rewrite: "The own-projected variant decomposes as D04 changed predictions under the head-row-space contract (pair 0.0209 [0.0108, 0.0310], established) plus D03 residual removal at fixed changed predictions (occupation 0.0285 established; pair 0.0111 [−0.0003, 0.0226] unresolved); combined D05 pair 0.0228 [0.0120, 0.0335]." Do not attribute D02 to changed predictions: no registered contrast isolates that at the hybrid contract.

## SHOULD-FIX
3. RECIPE, Controls: "(18 readers × 3 views × 2 seeds each)" should read 2 banks × 18 readers × 3 views = 108 per seed (REPORTING_LOCK CONTROLS__S0/S1 = 108), 216 total. The 1188→972 note itself is correct (2×(18+360+108)=972; lock per_seed 594 / controls_per_seed 216). The discrepancy appears only in RECIPE. Add one line to START_HERE or RESEARCH_DECISION.
4. The occupation-only reduction is not specific to J_TRUST. D09 L_TRUST vs RAW_PROJ r2 = 0.0100 [0.0023, 0.0178] REDUCTION_ESTABLISHED (identical to J_TRUST's 0.0100), and D08 J_FREE gives 0.0121. START_HERE ("about 0.010 AUC less... established") and RESEARCH_DECISION "What the adapter did earn" should say the local arm got the same reduction. The coalition term contributed nothing measurable to it.
5. Occupation-excess PASS rows (RESEARCH_DECISION primary table): label them "no-excess bound (upper ≤ 0.010), not a protection criterion". The "simultaneously established" occupation reduction should cite the diagnostic statuses D07 [+0.0025, +0.0174] and D15 [+0.0008, +0.0155] REDUCTION_ESTABLISHED (K=45), not the primary PASS.
6. RESEARCH_DECISION Q6: "Under fresh readers, very little in this setting". The D06 occupation point estimate of 0.0098 [−0.0016, 0.0212] is as large as the adapter's own 0.0100. Say instead: "not established; point estimate comparable to the adapter's occupation effect".
7. RESEARCH_DECISION Earned: "The first fresh-reader evidence that the earlier guards' privacy-bearing updates did lower independent recovery" makes two overclaims:
   - D02 compares the whole guard checkpoint with its start, with no task-only control, so the drop cannot be attributed to the privacy-bearing updates.
   - "first" reads as a priority claim.
   Rewrite as "the earlier guard checkpoints lowered fresh-reader recovery vs their start".
8. RESEARCH_DECISION Q5: "fail the original utility caps on FIT2 and INNER (occupation NLL excess 0.011–0.0175)". The range is INNER-only. FIT2 is 0.0142–0.0213 (RECIPE correctly says 0.014–0.021). Give both.
9. RESEARCH_DECISION Q3: "0.0049 lower vs RAW_PROJ, unresolved and an established shortfall" mixes two families. Say "primary: ESTABLISHED_VIOLATION vs 0.020; diagnostic D07: reduction UNRESOLVED [−0.0018, +0.0115]".
10. D04 is labelled "changed predictions at fixed projection". OWNPROJ uses its own head's basis, not RAW_PROJ's B. Use the registered wording "fixed head-row-space projection contract".
11. START_HERE has no caveat line. Add: intervals are conditional on the fitted models and readers; both seeds reuse the same people; exploratory, with exposed history. The column header "24 INNER point clauses" shows 12/12 per seed; rename it "INNER point clauses (per seed)".
12. QUICKSTART: "a manifest whose checkpoint or seed does not match". DEPLOYMENT_RESULTS tests only mismatched_checkpoint_manifest; there is no separate seed check. Drop "or seed" or point to the test that covers it.
13. The completion-label decision rules are not in SCIENCE_LOCK, which lists only the 5 labels. Cite where the rule text is locked.

## OK (verified)
- Scientific label follows: J_TRUST e10 point-feasible (FIT2 24/24, INNER 24/24 points); roster = RAW_PROJ + 4 trainable (all eligible, none excluded); 5/5 pair endpoints ESTABLISHED_VIOLATION; status_counts 15 PASS / 19 UNRESOLVED / 5 EV, K=39, t=3.224. No stronger label is implied; QUICKSTART/DEPLOYMENT status strings say the criterion is NOT ESTABLISHED.
- All 20 service-table rows match INNER_REPORT exactly: accuracy/NLL/Brier, r1/r2/pair AUC, epochs, 12/12, and 6/12 and 8/12 for the prior guards.
- All primary intervals match, e.g. pair vs RAW_PROJ 0.0049 [−0.0017, 0.0115], gap 0.0151, upper short by 0.0085. Also matched: utility 5 PASS (seed-1 occupation, as listed), largest upper 0.0223; INNER occupation NLL excess 0.0055/0.0027 vs J_FREE 0.0070/0.0038.
- All 9 START_HERE diagnostic contrasts, with per-arm values and statuses, match DIAGNOSTIC_ENDPOINTS (K=45, t=3.265).
- Training table: 291/29, 309/11, 315/5, 320/0, 314/6, 313/7, 318/2. NLL deltas and K values match. λ 0.934/0.788/1.539/1.086 match. Critic steps 10080 = (80+5×320)×6; qp_fallbacks 0.
- Probe movement: J_TRUST 0.2125/0.2173 with 21/24 decisions; TASK_TRUST 0.0446/0.0341 with 4/5. Counterfactual 8/8 per seed at epoch 1.
- FIT2: J_TRUST NLL +0.0028/+0.0015; L_FREE e10 0.0102 fails; FREE arms >0.004 in seed 0 (0.0043/0.0046). Figure is consistent.
- Control gates: null 0.5, shuffled 0.509–0.527, planted 1.000.
- Admission facts match: rank residual 2.2e-16/1.0e-15, ‖h64‖² 21.38/16.12, RMS 0.58/0.50, s=1, roles and duplicates, class 5 counts 0/2/2, ownproj logits 1.3e-14.
- Forecast probabilities match SCIENCE_LOCK exactly (0.55/0.75/0.04/0.05/0.30), and the outcomes are defensible.
- Exposure statement is quoted verbatim. Diagnostic prior-guard results are framed as diagnosis. "Established shortfall" = ESTABLISHED_VIOLATION is explained correctly. No "guarantee", "novel", "joint design" or "decision preservation" claim appears.
- No "/Users", username or email in any public md/json/csv (tokens use $PRIV).
