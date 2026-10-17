# Brief: independent numerical verifier (frozen-finalist validation study)

**Context.**
- Statistical machine-learning research on public-use U.S. Census ACS 2019 1-Year PUMS person records for Florida, development roles only.
- The service predicts income class (2) and occupation group (5) from AGEP, WKHP, COW, SCHL, MAR and POBP. Each recipient receives its own float64 probability vector and argmax decision.
- SEX is the dataset's recorded demographic column. Evaluation readers infer it from the released predictions: each output alone and the linked pair.
- This is defensive evaluation of a privacy-protecting release. Keep the variable names explicit.

**Your role.** Recompute the campaign's results with your own code, and say what does and does not match. Do not reuse `ffv/report.py`, `ffv/feasibility.py` or `ffv/mapping.py` for the quantities you verify. You may use:
- the admitted data loaders (`ffv.common.load_role`, `ffv.common.service_bank`, `ffv.common.frozen_bank`, `ffv.common.source_readers`);
- the source statistics functions only where noted.

**Your one DEV_REPORT access.**
- Event name: `DEV_REPORT_TRUTH_REREAD_BY_FFV_INDEPENDENT_VERIFIER`.
- Call `ffv.common.dev_report_truth(purpose, 'DEV_REPORT_TRUTH_REREAD_BY_FFV_INDEPENDENT_VERIFIER')` exactly once, inside one queue unit. It refuses a second use.
- Write the truth nowhere except private memory and your private outputs under `.private/pcrl_frozen_finalist_validation_v1/verifier/`.

## Checks

Report each as PASS / FAIL / PARTIAL, with numbers.

1. **Hashes and identity.**
   - `SOURCE_ADMISSION.json`: re-hash the admitted copies, all or a stratified sample of at least 60 including every bank, reader, release and package file.
   - `FROZEN_CANDIDATE_MANIFEST.json`: the release hash equals the source `FINALIST_LOCK`.
   - The per-level income vectors equal the released vectors bitwise.
   - FINALIST, H_LJ and B2TWIN occupation wires are identical on every role and seed.
2. **Exposure ledger.** Spot-check at least 6 entries of `EXPOSURE_AND_SELECTION_AUDIT.md` against the cited source records, and re-run the household-overlap check with your own code.
3. **Output mapping.**
   - Recompute with your own code, from the admitted release arrays and features: per-role distinct counts, the union counts, and the seed-1 cross-role conflicts.
   - Recompute the occupation and income routing-cell counts from the teacher JSONs.
   - Confirm at least 3 reported invented counterexamples through the frozen package (`amd.fam_b.package_release` on `.private/.../admitted/amd/deploy/FINALIST_S{s}`).
4. **Reader feature access.** Confirm from the code and the records that:
   - single-view readers see only their own output;
   - pair readers see only the two outputs;
   - no reader sees raw inputs, internals or non-ATTACK labels;
   - transforms are fitted on ATTACK_FIT only;
   - identical occupation wires share fits;
   - the source candidates use their stored ATTACK_VALIDATION AUCs and DEV scores.
5. **Representative reader refits.** The `READER_BANK_LOCK.json` `verifier_refit_roster`, per seed:
   - Refit, with your own code following the lock text, the ET5, HGB31, MLP6432 and CATI1 pair readers of FINALIST, NEURAL_RAW and H_LJ, plus the enhanced pair winners of those three.
   - Compare their ATTACK_VALIDATION AUCs and DEV_REPORT scores with the stored ones (`runs/READERS_ENH_S{s}/AV_SCORES.npz`, `DEV_REPORT_SCORES.npz`). Bitwise if possible; otherwise give the max abs difference and the AUC difference.
6. **Selection and primary endpoints.**
   - Re-derive every enhanced winner from the candidate AUCs, using the source-first order and exact ties to the earlier candidate.
   - Recompute, with your own bootstrap code, AUC point values and paired differences for both scorecards and both seeds. The bootstrap is household-level, B = 2048, seed 2026101705, weights = `task_headroom.inference.group_weights`; you may import that function and `task_conditioned_representation.inference.bootstrap_auc`.
   - Endpoints: the claim-B pair benefits (NEURAL_RAW, H_LJ), the recipient clauses, the TASKONLY and B2TWIN benefits, the IBSI descriptive difference, and the claim labels (E, B and its nested 0.020 test, M, J) under the frozen conjunctions.
   - Compare with `report/DEV_REPORT_RESULTS.json`.
7. **Utility and subgroup headroom.** Recompute the 24 utility excesses against U0/Ucal and the 4 within-SEX guards for FINALIST, with their bootstrap SEs. Confirm the binding clauses and their slack / SE.
8. **Power and sample size.** Independently check `CONFIRMATION_FEASIBILITY_RESULTS.json`, with a different method:
   - For the binding single clauses: the analytic normal formula n = 3955 × ((z + c) × SE / slack)².
   - For a joint check: your own Monte Carlo with a different seed, or the nonparametric bootstrap-deviation rescaling.
   - Do at least the S1 plug-in and the S2 0.75-shrink scenarios for D1 and D2, on the governing scorecard.
   - Report agreement, and any disagreement in n80 beyond one grid step.
9. **Reporting labels and limitations.** Read `CONFIRMATION_FEASIBILITY.md`, `OUTPUT_MAPPING_CHECK.md` and the decision text, if present, and flag any statement not supported by the numbers.

**Rules.**
- Run every computation longer than a few seconds through the queue: `python3 -m ffv.jobqueue run --unit VER_<name>_V1 --stage verification --rss-gib 3 -- /Users/<you>/PCRL/.venv/bin/python <script>`. At most 2 heavy jobs at once.
- Put your code under `results/pcrl_frozen_finalist_validation_v1/roles/verifier/code/` and your report in `roles/verifier/VERIFICATION.md`.
- Write a machine-readable `INDEPENDENT_VERIFICATION.json` in the results root with:
  - `checks` (name → status, numbers);
  - `discrepancies` (id, severity, description);
  - `dev_truth_event` (utc, purpose);
  - `verifier_units`.
- Do not modify any other file, do not commit or push, do not open the sealed assessment, and do not read MODEL_FIT, CALIBRATION or MODEL_SELECT labels.
- Charge your effort at the end: `python3 -m ffv.jobqueue charge --label verifier --cpu-s <upper estimate> --note "agent upper estimate"`.
