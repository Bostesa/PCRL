# Closing-review brief (coordinator, campaign `pcrl_input_and_pair_release_v1`)

You are the statistical reviewer, returning for the closing review (you wrote RULES.md and REPORT_PREREGISTRATION.md). Read (all under
`results/pcrl_input_and_pair_release_v1/`): PROTOCOL.md, PROTOCOL_AMENDMENT_1.md, PROTOCOL_DEVIATIONS.md, the cards (cards/IA_SFG_v1.md,
IB_CSF_v1.md, CTL_TWIN_v1.md, CTL_PATHS_v1.md, and IA's revision card IA_OSG_R1.md), the challenger's review (roles/challenger/),
roles/coordinator/FS_SWAP_DECISION.md, the decision records claims/IA_REG, claims/IB_REG and the revision reading claims/IA_R1
(DECISIONS.json; tag ADAPTIVE_R1) and the D2R records (claims/*_D2R/D2R.json, when present), the
plans claims/PLAN_*.json, claims/STRONGEST.json, claims/BESIDE_IPR1_MAIN.json, claims/CAMPAIGN_SUMMARY.json, the role members
paths/SHARED_ROLES.json, the data-rich reader record rich/RICH_*.json, the fit records ia/ and ib/, the utility ledger
dev/UTILITY_LEDGER.csv, the coordinator's draft RESEARCH_DECISION.md / TABLES.md (when present), and the verifier's report
(roles/verifier/, when present).

Write (or return as text if the harness refuses report files) `roles/statistical_reviewer/CLOSING_STATEMENT.md`:
1. Whether each decision label follows from the frozen rules and the records (check the inputs the claims builder passed: the card
   declarations operation_sex_blind / ro / joint, brackets, paths, the both-versions membership of control points (IA_SBC_LO exists in
   version 0 only), aliases, grid validity, registration status, adaptive tags and the fineness record).
2. The **required headline**: one paragraph the documents must quote verbatim, stating the outcome per family (registered readings,
   with the number of D2 looks), the decisive numbers with their uncertainty, the material criterion, and the scope limits
   (development evidence on adaptively used MODEL_SELECT records, declared readers only).
3. Required corrections (numbered C1, C2, ...) to the coordinator's drafts.
4. What the evidence does and does not support:
   - the input-side reading: IA's registered candidate generalizes almost nothing at S3, and leakage moved to the income view;
   - IA's adaptive revision IA_OSG (occupation-only generalization; controls restricted to the occupation output), which is
     never IA's registered result: its trigger and diagnosis, why it is numbered R1 although IA_SFG_v1 declared a different R1,
     and whether its controls are matched;
   - the SEX-blind reading of IB (no SEX-targeting claim possible);
   - the declared RO fallback of IA;
   - the D5 collapse declaration of IB;
   - the replacement family not run, and IB's revisions not run;
   - novelty.
Read no label and run no real-data computation; invented-data checks only. Report CPU seconds. Any time you write must come from
`date -u` read first, never ahead of the clock.
