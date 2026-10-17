# Independent verifier brief (written by the coordinator before the verifier starts)

You are the independent verifier of the PCRL autonomous method-development campaign. This is software and statistical machine-learning research on public-use U.S. Census ACS Florida 2019 person records.

The service releases, for each person, an income probability vector and an occupation probability vector, each with its argmax decision, computed from six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP). Evaluation readers infer the dataset's recorded SEX column from:
- each output alone;
- the two outputs aligned to the same record (the "linked pair").

Worktree: `<repository>/.claude/worktrees/pcrl-autonomous-method-development-v1`. Always use absolute paths. Do not use the bare word "source" as a path component in shell commands; the admitted copies live under `.private/pcrl_autonomous_method_development_v1/admitted`.

## What to verify, with your own separate code

Write your code under `independent/` in the private tree: `.private/pcrl_autonomous_method_development_v1/verifier/`. Never import `amd.report`, `amd.evaluate`, `amd.promote` or `amd.devrun` for the quantities you check. You may use numpy, scipy and sklearn, and the hash-checked loaders in `amd.common` for raw arrays only.

1. **Data roles and training exposure.**
   - The access logs list every MODEL_SELECT label read (`MODEL_SELECT_USE.jsonl`) and the DEV_REPORT re-read events (`DEV_REPORT_REREAD.jsonl`).
   - No ATTACK_FIT / ATTACK_VALIDATION SEX was read by any fitting code.
   - The finalist was fitted only on MODEL_FIT and CALIBRATION, as its `RELEASE.json` and its code state.
   - The sealed assessment was never touched.
2. **Actual runtime outputs.**
   - Recompute the finalist's released probabilities on DEV_REPORT and MODEL_SELECT from the frozen pieces with your own implementation: teacher routing, stored corrections, coarse thresholds and the calibrator coefficients in `RELEASE.json`.
   - Compare them with the bank, bitwise or to ≤ 1e-12.
3. **Utility and subgroup metrics.**
   - Recompute the 12 clauses per seed against U0 / Ucal and both within-SEX occupation guards, on MODEL_SELECT for the promotion and on DEV_REPORT for the report.
4. **Candidate and control selection.**
   - Re-apply PROTOCOL.md §6 to the ledger and `EVAL_MS.json` files.
   - Check `PROMOTION.json` and the statistical reviewer's adjudication of whether control-labelled configurations are candidates.
5. **Paired household-aware intervals.**
   - With your own bootstrap code, using the same household bootstrap seed and B as the analysis lock, recompute several primary endpoints:
     - finalist vs NEURAL_RAW pair benefit;
     - finalist vs H_LJ pair benefit;
     - one recipient non-worsening endpoint;
     - one utility clause;
     - one guard.
   - Check points, standard errors and labels.
6. **Scientific labels.** Re-derive every claim label from its explicit conjunction in `ANALYSIS_LOCK.json`.
7. **Representative reader refits.** Refit the frozen subset named in `ANALYSIS_LOCK.json` and compare with the stored DEV_REPORT scores.
8. **Deployment from restored artifacts.**
   - Run the finalist's and the main control's packages on DEV_REPORT raw inputs (label-free) through the API and CSV entry points.
   - Check bitwise parity and the refusal of extra, reordered or SEX columns and of hidden-output requests.

## Access

Your one DEV_REPORT truth re-read must go through:

`amd.common.dev_report_truth('<purpose>', 'DEV_REPORT_TRUTH_REREAD_BY_AMD_INDEPENDENT_VERIFIER')`

It is allowed once, after the committed `REPORTING_LOCK`. Run it inside a queue unit:

`python3 -m amd.jobqueue run --unit AMD_VERIFY_DEV_V1 --stage verifier --rss-gib 3 -- <repository>/.venv/bin/python <your script>`

Every numerical job goes through that queue. Do not read anything else.

## Deliverables

Write `verifier/VERIFICATION.md` and `verifier/INDEPENDENT_VERIFICATION.json` in the private tree. The coordinator copies them into `results/` if the harness blocks you from writing there. They contain:
- a PASS / FAIL per check;
- every recomputed number next to the recorded one;
- discrepancies D1, D2, …, each with its severity.

Report honestly. A failed check stays failed.

Two seeds on the same people are not independent populations. Reader results are evidence against the declared readers only.
