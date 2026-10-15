# Claims review: two-recipient task-wire adapter drafts

Role: CLAIMS REVIEW (adversarial). Scope: RESEARCH_DECISION.md, RECIPE_AND_CONTROLS.md, SOURCE_AND_EXPOSURE.md and QUICKSTART.md, checked against INNER_REPORT.json, FIT2_UTILITY.json, SELECTION_LOCK.json, SCIENCE_LOCK.json, TRAINING_COMPLETE_LOCK.json, REPORTING_LOCK.json, DEPLOYMENT_RESULTS.json, BACKUP_VERIFICATION.json, PROTOCOL_SEAL.json, ADMISSION.json, MODEL_MANIFEST.json, the private S0/S1 training SUMMARY.json files and git history (tip 377ecfe). The placeholders `__COMPLETION__`, `__LABEL_CHECK__` and `__CF_CHECK__` were ignored as instructed. No file other than this one was edited.

Sign convention checked: a diagnostic reduction is control AUC minus candidate AUC (for example, D14 seed 0: HYB_RI_JO pair 0.8442 − J_BOTH 0.8460 = −0.0018, so J_BOTH recovers *more*). A primary pair benefit is AUC_control − AUC_candidate, and r1/r2 excess is AUC_candidate − AUC_control.

---

## MUST-FIX

### M1. Sign wording inverted for D08 and D14 (RESEARCH_DECISION, answer 1)
- **Claims:** "income-only adaptation changed pair recovery by −0.0004 relative to RAW_PROJ (D08 …)" and "…changed pair recovery by −0.0024 (D14 …). That is slightly more recovery, not less."
- **Data:** D08 pair `recovery_reduction_point` = −0.00041. D14 pair = −0.00236 (per seed −0.0018 / −0.0029). These are *reductions*, so negative values mean recovery went **up**: I_ONLY by 0.0004 on average, and J_BOTH by 0.0024 over HYB_RI_JO.
- **Problem:** "changed recovery by −x" says recovery fell. The D14 sentence then says the opposite, so the paragraph contradicts itself.
- **Fix:** Write "pair recovery *reduction* −0.0004 (recovery 0.0004 higher, unresolved)" and "reduction −0.0024 [−0.0073, +0.0025] (recovery 0.0024 higher, unresolved)". Apply the same wording anywhere else a reduction is phrased as a change in recovery.

### M2. "Strongest across the diagnostic family" names the wrong endpoint (RESEARCH_DECISION, strongest favorable)
- **Claim:** "Across the diagnostic family, the strongest is L_BOTH vs RAW_PROJ, occupation view, 0.0099 [0.0012, 0.0186], established."
- **Data:** D15 r2 (O_PREVIOUS vs RAW_PROJ) is 0.00996 [0.0025, 0.0174], REDUCTION_ESTABLISHED. D09 r2 is 0.00990 [0.0012, 0.0186]. D15 is larger at the point estimate and has the larger lower bound.
- **Fix:** "The strongest diagnostic reductions are O_PREVIOUS's occupation view (D15, 0.0100 [0.0025, 0.0174]; the inherited component, not equally trained) and L_BOTH's occupation view (D09, 0.0099 [0.0012, 0.0186]; a local-objective arm). Neither is the candidate, and neither is a coalition or pair result."

### M3. Adverse occupation-utility change omitted from direct answers 2 and 5 (RESEARCH_DECISION)
- **Claims:** answer 2, "At the point level, yes … 19 of 24 are unresolved"; answer 5, which discusses only income resolution (2/12 now PASS).
- **Data (`utility_bounds_all_services`):**
  - Occupation: RAW_PROJ has **10/12 PASS** (max upper 0.0129); J_BOTH has **3/12 PASS** (max upper 0.0162); T_BOTH has 8/12.
  - Income: J_BOTH 2/12 PASS vs RAW_PROJ 0/12.
  - Net: J_BOTH resolves **5/24** clauses vs RAW_PROJ's 10/24.
- **Problem:** The drafts present the income gain (+2, also reproduced by task-only T_BOTH) and stay silent on the larger occupation loss (−7), most of which is not explained by task-only adaptation. That makes the utility picture look better than it is.
- **Fix:** In answer 2 add: "Relative to the start, simultaneous resolution got worse. RAW_PROJ resolves 10 of 24 clauses (0 income, 10 occupation); J_BOTH resolves 5 (2 income, 3 occupation). The occupation upper bounds widened (max 0.0129 → 0.0162). The task-only T_BOTH keeps 8 occupation PASSes." Reference this in answer 5.

### M4. The decision overgeneralizes and attributes the null to the wrong constraint (RESEARCH_DECISION, Decision)
- **Claims:** "…does not move coalition recovery; the coalition signal is not removable by small per-recipient prediction corrections of this kind." Also "Adapting a rank-1 income head inside a KL ≤ 0.001 trust region…".
- **Data:**
  1. The pair point estimates do move: J_BOTH 0.0036, L_BOTH 0.0044 (D09), HYB_RI_JO 0.0060 [−0.0002, 0.0122] (D12). All are unresolved, not zero.
  2. "Not removable" is a generalization beyond one fixed recipe, two seeds and one 18-reader slate on exposed people, which the handoff forbids.
  3. The J_BOTH income adapter had **0 active-budget cases** in both seeds (SUMMARY `active_budget_cases.income = 0`). Its fitting-pool KL at E10 was 0.00074 / 0.00036, so the KL trust region never bound the income adapter. The limit came from the step cap (≤ 0.01 per step, 320 steps) and the fixed coefficient, not from KL.
  4. The heads are frozen. The adapter is a rank-1 residual on the income wire's representation, not an adaptation "of the head".
- **Fix:** "Close this recipe. Under this fixed recipe (rank-1 income and rank-5 occupation residuals over frozen trunks and heads; per-step displacement ≤ 0.01; KL ≤ 0.001, which never bound the income adapter), no arm established a pair-recovery reduction. The candidate's pair benefit is an established shortfall against the 0.020 margin, and all pair point estimates are ≤ 0.0060. This does not show that the coalition signal cannot be reduced by other prediction-level mechanisms. No nearby parameter adjustment is scheduled."

### M5. QUICKSTART misstates the packages' `scientific_status` field
- **Claim:** "every package's `scientific_status` field says no competitive coalition criterion was established."
- **Data (DEPLOYMENT_RESULTS / MODEL_MANIFEST):** Only J_BOTH_S0/S1 say "POINT_FEASIBLE_NO_COMPETITIVE_CRITERION: …". The others say:
  - "matched control arm (SELECTED_POINT_ELIGIBLE); not a competitive privacy result" (T_BOTH, L_BOTH, S12, S21, I_ONLY);
  - "immutable reference (zero residuals)" (RAW_PROJ);
  - "inherited occupation-only component reference (previous J_TRUST epoch 10)" (O_PREVIOUS).
- **Fix:** "J_BOTH packages carry `POINT_FEASIBLE_NO_COMPETITIVE_CRITERION`. Control, reference and inherited packages state their role and that they are not a competitive privacy result."

---

## SHOULD-FIX

### S1. The headline reads as an achieved reduction (RESEARCH_DECISION, answer paragraph)
- **Claim:** "J_BOTH (epoch 10) reduces aligned-pair recovery relative to RAW_PROJ by **0.0036**".
- **Data:** 0.0036 [−0.0025, +0.0097]; in the diagnostic family D01 it is UNRESOLVED.
- **Fix:** "has a point pair-recovery reduction of 0.0036 (not established; interval [−0.0025, +0.0097])".

### S2. "Smaller than the previous occupation-only 0.0049" needs the direct test (RESEARCH_DECISION)
- **Data:** D06 (J_BOTH vs O_PREVIOUS) pair −0.0012 [−0.0065, +0.0041], unresolved. D06 r2 is −0.0045 [−0.0105, +0.0015]: at the point estimate, J_BOTH's occupation wire reveals more than O_PREVIOUS's. Neither draft cites D06.
- **Fix:** Add "(point comparison; the direct contrast D06 is −0.0012, unresolved; on the occupation view J_BOTH reveals 0.0045 more than O_PREVIOUS at the point estimate, unresolved; O_PREVIOUS is an inherited component, not an equally trained control)".

### S3. The complementarity sentence is stated as a finding (RESEARCH_DECISION, answer 3)
- **Claim:** "Combining the two changed wires does not reduce recovery beyond each substitution alone."
- **Data:** Interaction −0.0021, 95% [−0.0063, +0.0020], defined against the **sum** of the two substitutions. The interval includes positive values. The r1/r2 interactions are identically 0 by construction.
- **Fix:** "The point interaction is sub-additive (−0.0021) but the interval includes zero, so complementarity is neither shown nor excluded. Descriptively, J_BOTH's occupation wire alone (D12, 0.0060) gives a larger point pair reduction than both changed wires together (0.0036)."

### S4. "Most adverse" needs a view qualifier (RESEARCH_DECISION)
- **Claim:** "Most adverse: J_BOTH vs HYB_RI_JO (D14), pair −0.0024".
- **Data:** Across views the most adverse diagnostic points are D06 r2 −0.0045, D03 r2 −0.0044 and D05 r2 −0.0042, all unresolved. D14 is the most adverse on the **pair** view.
- **Fix:** Write "Most adverse on the pair view", and add "on the occupation view, J_BOTH vs O_PREVIOUS −0.0045 (D06)".

### S5. Answer 1 conflates two different income wires
- **Claim:** "the changed income lowered income-only recovery by about 0.003–0.004 (D01/D08/D11 r1)".
- **Data:** D01 and D11 r1 are 0.0037 (J_BOTH's income wire). D08 r1 is 0.0029 and uses **I_ONLY's** wire (identical to S12's, per REPORTING_LOCK aliases), not J_BOTH's.
- **Fix:** "J_BOTH's income wire lowered income-view recovery by 0.0037 (D01/D11 r1, unresolved); I_ONLY's separately trained income wire by 0.0029 (D08 r1, unresolved)". Use "income-view" rather than "income-only" to avoid confusion with I_ONLY.

### S6. The task-loss claim needs scope (RESEARCH_DECISION, answer 1)
- **Claim:** "the income change did not improve income task loss either."
- **Data:**
  - INNER point (no interval): J_BOTH 0.0075 / 0.0062 vs RAW_PROJ 0.0059 / 0.0061. Verified.
  - Fitting pool: income NLL rose 0.29483 → 0.29554 (s0) and 0.28917 → 0.28966 (s1).
  - FIT2: seed 1 is slightly better (0.0007 vs 0.0010, U0).
- **Fix:** "At the point estimate on INNER and on the fitting pool, the income NLL did not improve (it rose slightly; FIT2 seed 1 is marginally lower). No interval is computed for the difference." This keeps to the rule that changed income "may improve task loss but does not automatically resolve uncertainty".

### S7. A separate status ledger is missing (RESEARCH_DECISION)
- **Issue:** The handoff requires point feasibility, established benefit, established shortfall, unresolved and unavailable to be reported separately. The draft spreads them across sections and never states "unavailable".
- **Fix:** Add a five-line ledger:
  - point-feasible: J_BOTH E10, both seeds, FIT2 and INNER points;
  - established benefit: none in the primary family (diagnostic only: D09 r2, D15 r2, neither the candidate);
  - established shortfall: 5 of 5 pair endpoints;
  - unresolved: 21 primary endpoints (19 utility, 2 occupation excess: L_BOTH, S21);
  - unavailable: 0 NOT_ESTIMABLE, 0 INCOMPLETE; independent verification and off-device custody pending.

### S8. Off-device custody is not disclosed; restore wording is ambiguous (QUICKSTART, RESEARCH_DECISION Limits)
- **Claim (QUICKSTART):** "…and it did so again after restoring from the backup alone."
- **Data:** BACKUP_VERIFICATION has `same_host_snapshot` SNAPSHOT_V1 (1,739 files), `off_device.status = PENDING`, `same_host_is_not_off_device_custody = true`, and the predecessor off-device backup is also PENDING.
- **Fix:** In QUICKSTART, write "after restoring from the same-host snapshot alone (off-device custody pending)". Add the same limit to RESEARCH_DECISION Limits.

### S9. The counterfactual claim in RECIPE lacks its verification caveat
- **Claim:** "changed the deployed probe wire when its privacy gradient was removed (8/8 each…)".
- **Data:** The training SUMMARY counterfactual_scan agrees (8/8 `privacy_changes_deployed_wire_beyond_tolerance=true`, all epoch 1, both seeds and recipients). However, SELECTION_LOCK `movement_precheck.note` = "campaign record; independent re-solve pending".
- **Fix:** Add "(campaign record; independent re-solve pending)", or the `__CF_CHECK__` placeholder, to match RESEARCH_DECISION.

### S10. Rounding of the control-gate value (RECIPE)
- **Claim:** "shuffled envelope ≤ 0.527".
- **Data:** The maximum is S1 pair 0.52703, which is greater than 0.527. The gate is ≤ 0.60. The planted envelope is 1.000 in all six; the weakest planted reader is 0.99999.
- **Fix:** "shuffled envelope max 0.5270 (gate ≤ 0.60); planted envelope 1.000".

### S11. Referenced files do not exist yet (RECIPE, SOURCE)
- **Issue:** `INDEPENDENT_VERIFICATION.json` (RECIPE fit counts) and `UNIT_MANIFEST.json` (SOURCE last paragraph) are absent from the campaign directory.
- **Fix:** Make sure both are written before publication, or mark them as pending. SELECTION_LOCK caps independent refits at 36.

### S12. The source-training overlap is not disclosed (SOURCE_AND_EXPOSURE)
- **Data:** PROTOCOL_SEAL and SCIENCE_LOCK `exposure_statement` say "Source models were trained on all FIT". ADMISSION `group_overlap` shows OSF_DEFENSE_FIT containing all UPDATE (8,176), GUARD (2,043) and VALIDATION/FIT2 (5,209) groups.
- **Problem:** The draft's "role groups are disjoint" is true only among the study roles.
- **Fix:** Add "The source models were trained on all FIT, including the UPDATE, GUARD and FIT2 rows; disjointness refers to the study roles."

### S13. The class-5 limitation is missing (SOURCE_AND_EXPOSURE or RECIPE)
- **Data:** PROTOCOL_SEAL `class5_limitation`: occupation class 5 has 0 GUARD, 2 UPDATE and 2 FIT2 rows. Class-5 statistics are not estimable, and there is no protection claim for unsupported classes.
- **Fix:** Add one line.

### S14. The "Earned" list should keep local and inherited results separate from the candidate (RESEARCH_DECISION, answer 6)
- **Issue:** The L_BOTH 0.0099 and O_PREVIOUS 0.0100 occupation-view reductions are diagnostic-family results for a local arm and an inherited component. They are neither coalition evidence nor the candidate's.
- **Fix:** Tag them "(diagnostic family; local / inherited; not coalition or candidate evidence)". Phrase the "Failed" items as "not established" rather than as proven absence, e.g. "no coalition benefit from changing income was established".

### S15. Minor chronology detail (SOURCE_AND_EXPOSURE)
- **Data:** Commit `27625c5` is 14:27:28Z; the seal's `campaign_start_utc` is 14:26:40Z.
- **Fix:** Either time is acceptable, but say which one is given ("seal dated 14:26:40Z; committed 14:27Z").

---

## VERIFIED

### Primary family (INNER_REPORT.primary)
- K = 39; critical value 3.2241 (draft 3.224); 2,048 replicates; candidate J_BOTH E10 SELECTED_POINT_ELIGIBLE.
- All five comparators are point-eligible; status counts are PASS 13, UNRESOLVED 21, ESTABLISHED_VIOLATION 5, NOT_ESTIMABLE 0, INCOMPLETE 0.
- Every cell of the primary table matches to 4 decimals:

  | row | pair | r1 | r2 |
  |---|---|---|---|
  | RAW_PROJ | +0.0036 [−0.0025, +0.0097] | −0.0037 [−0.0089, +0.0016] | −0.0055 [−0.0128, +0.0018] |
  | T_BOTH | +0.0027 [−0.0041, +0.0095] | −0.0045 [−0.0092, +0.0001] | −0.0058 [−0.0125, +0.0009] |
  | L_BOTH | −0.0007 [−0.0067, +0.0052] | −0.0002 [−0.0051, +0.0048] | +0.0044 [−0.0035, +0.0124] UNRESOLVED |
  | S12 | −0.0009 [−0.0064, +0.0046] | −0.0008 [−0.0051, +0.0036] | +0.0025 [−0.0041, +0.0090] |
  | S21 | −0.0000 [−0.0058, +0.0058] | −0.0011 [−0.0056, +0.0034] | +0.0042 [−0.0032, +0.0116] UNRESOLVED |

  All five pair endpoints are ESTABLISHED_VIOLATION (upper < 0.020).
- Gap to the failed clause: 0.020 − 0.00363 = 0.0164; 0.020 − 0.00972 = 0.0103.

### J_BOTH INNER utility
- 5 PASS / 19 UNRESOLVED / 0 violations, and all 24 point estimates pass.
- The PASS set matches exactly: S1 income accuracy vs U0 and Ucal; S1 occupation NLL vs U0; S1 occupation Brier vs U0 and Ucal.
- Largest upper bound is S1 income NLL vs Ucal, 0.0226.

### FIT2 (FIT2_UTILITY, SELECTION_LOCK)
- All 24 point clauses pass for J_BOTH E10 in both seeds.
- Every arm is selected at E10 or P2E10, with feasible common labels E1/E3/E5/E10.
- RAW_PROJ and O_PREVIOUS are point-eligible.

### Held-out drift (answer 2)
- INNER income NLL vs Ucal point fails: I_ONLY s1 0.0101, S12 s1 0.0101, S21 s0 0.0102, S21 s1 0.0114.
- These are the only point fails in the service table, and each of these arms passed FIT2.

### Diagnostic family (K = 45, critical 3.2650)
- D01: r1 0.0037, r2 0.0055, pair 0.0036.
- D03/D04/D05 pair: −0.0007, −0.0009, −0.0000.
- D03/D04/D05 r2: −0.0044, −0.0025, −0.0042 (the "−0.0025 to −0.0044" range is correct).
- D08 pair: −0.0004 [−0.0060, +0.0051].
- D09 r2: 0.0099 [0.0012, 0.0186], established.
- D14 pair: −0.0024 [−0.0073, +0.0025].
- D15 r2: 0.0100, established; D15 pair 0.0049 (replicates the previous 0.0049).
- All pair contrasts are UNRESOLVED.

### Interaction and income resolution
- Descriptive interaction (pair): −0.0021 [−0.0063, +0.0020].
- Income resolution: J_BOTH 2/12 PASS (max upper 0.0226); RAW_PROJ 0/12 (max upper 0.0223); T_BOTH passes the same two S1 accuracy clauses.
- Answer 1 RAW_PROJ income NLL excess vs U0: 0.0059 / 0.0061.

### Forecasts (SCIENCE_LOCK)
- Probabilities match: F1 0.55, F2 0.04, F3 0.03, F4 0.20, F5 0.05, F6 0.20.
- Outcomes are correct:
  - F1 true (E10);
  - F2 false (lower bound −0.0025);
  - F3 false (no D03–D05 pair established);
  - F4 false (D08 −0.0004, unresolved);
  - F5 false (2/12);
  - F6 false (D14 −0.0024).

### Scientific label
- `POINT_FEASIBLE_NO_COMPETITIVE_CRITERION` is consistent with a point-feasible candidate, a complete roster and a failed coalition criterion. It remains subject to `__LABEL_CHECK__`, because SCIENCE_LOCK also requires complete independent verification.

### Recipe (SCIENCE_LOCK recipe/measured, training SUMMARY)
- Ranks are 1 and 5; parameters 65 + 325 = 390; scales are 1.0 everywhere.
- Optimizer and steps: SGD at lr 0.01, batch 256, maximum step norm 0.01, ladder 1, ½, ¼, ⅛, checkpoints 0/1/3/5/10.
- Task weights are 8180/10223 and 2043/10223.
- Budgets: NLL +0.004, Brier +0.002, correct ≥ start − 51, KL 0.001, never reset.
- Coefficients: joint income 1.537 / 1.315, joint occupation 0.740 / 0.749, local income 3.111 / 2.228, local occupation 1.438 / 1.617.
- Critic rounds: 3,280 (simultaneous and sequential) and 1,680 (I_ONLY).
- All income opportunities were accepted (320 in every arm and seed).
- Rejected occupation steps: J 7/6, L 7/10, S12 20/18, S21 14/10, T 0/0.
- J_BOTH active-budget cases: income 0/0, occupation 88/79.
- The 8/8 counterfactuals, all in epoch 1, match the campaign record.
- Wire widths are 21/29/50.

### Readers and fits (REPORTING_LOCK, control gates)
- 18 readers; 1,080 direct + 216 control = 1,296 fits; cap 1,500.
- Aliases are recorded and not deduplicated.
- All six control gates passed.

### Source and roles (ADMISSION, INCOME_EXPORT)
- Role sizes: UPDATE 8,180 rows / 8,176 groups; GUARD 2,043; FIT2 5,211; ATT 4,064 / 4,061; INNER 2,235 / 2,234.
- Income class counts: 6,187/1,993 and 1,550/493.
- Geometry: max ‖ABᵀB − A‖ = 1.04e-15 (≤ 1.1e-15).
- O_PREVIOUS hashes and scale 1.0 match, and it was reproduced bitwise.

### Chronology (git)
- Lock order verified: 6ee0864 → 405aa12 → dae14cc → f00c6b4 → 9c49b84.
- The prefit receipt `3f094fa` exists on the remote publication branch (14:34:42Z) and predates the bank checkpoints (14:35Z).
- Only deployment_worker.py, custody_worker.py and figures.py changed after SCIENCE_LOCK.

### Deployment (DEPLOYMENT_RESULTS)
- 16 packages, each with 15/15 refusal checks true.
- INNER live vs stored maximum absolute difference is 0 for all.
- The J_BOTH_S0 manifest SHA matches the QUICKSTART pin.
- The CLI flags and the {X, feature_names} input contract match `deployment.py`.
- Snapshot restore reproduced all 16 packages with difference 0.

### Handoff rules respected
- O_PREVIOUS is labeled an inherited component.
- S12/S21 are labeled not Taylor et al.'s finite-law algorithm.
- Hybrids are labeled fixed substitutions, not proofs.
- No claim that the previous result proved income caused the shortfall.
- Novelty is kept separate.
- Excess rows are described as no-excess bounds, not protection criteria.
- Intervals are described as conditional and on exposed people.
