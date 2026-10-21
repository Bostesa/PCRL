# Closing-review brief (coordinator, campaign `pcrl_dual_design_release_v1`)

You are the statistical reviewer, returning for the closing review (you wrote RULES.md and REPORT_PREREGISTRATION.md). Read:
PROTOCOL.md, PROTOCOL_AMENDMENT_1.md, PROTOCOL_DEVIATIONS.md, the cards (cards/DA_STD_v1.md, DB_PTC_v1.md, DB_PTC_R1.md and any
later card), the decision records claims/*/DECISIONS.json and D2R records, the data-rich reader record rich/RICH_*.json, the fit
records da/ and db/, the utility ledger dev/UTILITY_LEDGER.csv, the coordinator's draft RESEARCH_DECISION.md / TABLES.md (when
present), and the verifier's report (roles/verifier/).

Write (or return as text if the harness refuses report files) `roles/statistical_reviewer/CLOSING_STATEMENT.md`:
1. Whether each decision label follows from the frozen rules and the records (check the inputs the claims builder passed: brackets,
   paths, aliases, grid validity, registration status and adaptive tags).
2. The **required headline**: one paragraph the documents must quote verbatim, stating the outcome per family (registered reading
   first, adaptive readings beside, with the number of D2 looks), the decisive numbers with their uncertainty, the material criterion,
   and the scope limits (development evidence on adaptively used MODEL_SELECT records, declared readers only).
3. Required corrections (numbered C1, C2, ...) to the coordinator's drafts.
4. What the evidence does and does not support (the information-removal versus reader-budget reading of each family; the shared idea
   family; the joint-design and novelty labels).
Read no label and run no real-data computation; invented-data checks only. Report CPU seconds.
