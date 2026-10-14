# Validation record

| check | result | where |
|---|---|---|
| Source hashes, roles, schema, folded heads, geometry, zero-residual witness (all roles, both seeds) | PASS | ADMISSION.json (MTA_ADMISSION_V2; V1 failed before reading data because it imported the predecessor's queue guard) |
| Engineering tests (synthetic): zero witness, schema refusal, scale rule, planted residual bypass caught, 3-halfspace solver vs SLSQP incl. degenerate rows, zero/round-off KL row, chunked full-pool reduction and finite-difference gradients, anchored budgets, backtracking/rollback/norm, held-out export refusal, task weights, five-arm end-to-end with frozen-source check | 13 passed | engineering/ENGINEERING_TESTS_V4.txt |
| Mathematical review | 0 blockers; 3 should-fix applied before SCIENCE_LOCK | MATH_REVIEW.md |
| Reporting arithmetic fixtures (tie-aware weighted AUC, envelope reselection, duplicate-group multiplicity, shared draws, Bonferroni-t, classification, ignore aliases) | PASS before INNER | REPORTING_FIXTURES.json |
| Matched critic budget per arm ((80+5x320)x6) and shared warm critic state | asserted in trainer | TRAINING_COMPLETE_LOCK.json |
| Privacy-bearing movement (J_TRUST updates 1-8 per seed) | 16/16 beyond tolerance in displacement and deployed wire | TRAINING_COMPLETE_LOCK.json; independently re-solved (INDEPENDENT_VERIFICATION.json) |
| Epoch-0 adapter = RAW_PROJ on FIT2 | asserted bitwise in FIT2 unit | FIT2_UTILITY.json |
| Income wire identical in all 10 services (ATT and INNER hashes) | yes, both seeds | REPORTING_LOCK.json r1_alias_and_distinct_wires |
| Reader control gates (null 0.5, shuffle <= 0.60, planted >= 0.99) | all 6 view-seed gates PASS | INNER_REPORT.json control_gates |
| Deployment raw83: live == CLI == stored INNER release (10 packages), 15 refusal checks each | PASS, max abs 0 | DEPLOYMENT_RESULTS.json |
| Same-device snapshot and backup-alone restore | PASS (1,301 files) | BACKUP_VERIFICATION.json |
| Independent verification | see INDEPENDENT_VERIFICATION.json | |
| Claims review | see CLAIMS_REVIEW.md | |

Failed or superseded units are preserved in the queue state and UNIT_MANIFEST.json: MTA_ADMISSION_V1 (technical), MTA_ENGINEERING_TESTS_V1 (test assumption), and V2/V3 (superseded by review fixes). None is a scientific outcome.
