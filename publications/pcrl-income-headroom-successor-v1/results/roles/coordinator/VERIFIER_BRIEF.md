# Brief: independent numerical verifier (PCRL income-headroom successor development campaign)

**Context.** Statistical machine-learning research on public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for
Florida, development roles only. The service predicts income class (2) and occupation group (5) from AGEP, WKHP, COW, SCHL, MAR
and POBP; each recipient receives its own float64 probability vector and argmax decision. SEX is the dataset's recorded
demographic column; evaluation readers estimate it from the released predictions (each output alone and the linked pair). This
is defensive evaluation of releases designed to lower that recovery. Keep the variable names explicit.

**Your role.** Recompute the campaign's substantive results with your OWN code and say exactly what does and does not match.
Do not reuse `ihs/evaluate.py`, `ihs/reserve.py`, `ihs/nominate.py`, `ihs/report.py`, `ihs/claims.py` or `ihs/feasibility.py`
for the quantities you verify. You may use the admitted, hash-checked loaders (`ihs.common.load_role`, `frozen_bank`,
`amd_service_bank`, `stage_a_outputs`, `stage_b_theta`, `teacher`), `ihs.devrun.load_release`, `ihs.bases.base_outputs`,
`task_conditioned_representation.inference.auc / bootstrap_auc`, `task_headroom.inference.group_weights`, and the reader
constructors (`task_conditioned_representation.attacks.fit_reader`, `ffv.readers.make_model / fit_new`, `ffv.features`) where
noted. Work in `<repository>/.claude/worktrees/pcrl-income-headroom-successor-v1` only; keep shell commands simple.

**Role access for you.** You may read MODEL_SELECT labels and SEX through `ihs.common.load_role('MODEL_SELECT', (...),
purpose='independent verification ...')` (each read is logged), MODEL_FIT labels only to check training exposure statements,
CALIBRATION labels only to check calibrator statements. ATTACK_FIT / ATTACK_VALIDATION SEX and DEV_REPORT truth only in the
finalist branch below. Never open the sealed assessment.

## Checks (report each as PASS / FAIL / PARTIAL with numbers)

1. **Source admission and role integrity.** Re-hash at least 60 admitted copies (`SOURCE_ADMISSION.json`), including every
   bank, stage file and run release; confirm the loaders' gates (MODEL_SELECT logging; ATTACK gate on FINALIST_LOCK; DEV gate on
   REPORTING_LOCK; sealed refusal) from the code; confirm the MODEL_SELECT use log has a purpose on every line.
2. **Training and calibration exposure.** For each family's best configurations (and the finalist, if any): from each release
   record and the family code, which roles and fields were used for fitting, calibration and selection. Flag any MODEL_SELECT,
   ATTACK or DEV_REPORT label used for fitting or calibration, any internal MODEL_FIT fold described as unseen data, and any
   runtime use of SEX.
3. **Task and subgroup metrics and SEs.** For at least 6 configurations (including the best of each family, the nominee and its
   twin if any) and 3 references, recompute on MODEL_SELECT with your own code the 12 utility clauses per seed and the 2 guards
   with household-clustered SEs; compare with the `EVAL_MS.json` records (tolerance 1e-12 for points, 1e-9 for SEs).
4. **Reserve calculations.** Re-derive each of those configurations' reserve status from the registered rule text
   (`roles/statistical_reviewer/UTILITY_RESERVE_RULE.md`, `RESERVE_RULE.json`) with your own code.
5. **Development proxies.** For at least 3 configurations and 2 references, recompute at least 2 cells (one per repetition) of
   the development reader grid with your own code from the protocol text (repetition A / B halves, reader seeds), and compare.
6. **Candidate and comparator selection.** Re-derive every configuration's nomination label and the choice from
   `NOMINATION_RULE.md` with your own code from the `EVAL_MS.json` records; compare with `NOMINATION.json`.
7. **Actual transmitted outputs.** For the finalist (or the best diagnostic package if none): redeploy from the six raw fields
   through the package wrapper and compare with the stored release bitwise on DEV_REPORT inputs (label-free); confirm the
   release uses no runtime field other than the six.
8. **Finalist branch only (if `FINALIST_LOCK.json` exists):**
   - **Final reader refits.** Refit with your own code the READER_PLAN refit roster's representative readers (at least the
     ENHANCED pair winners of SUCCESSOR, NEURAL_RAW and H_LJ on both seeds, plus ET5, HGB31, MLP6432, CATI1 pair readers of
     SUCCESSOR); compare AV AUCs and DEV_REPORT scores with the stored ones (bitwise if possible).
   - **Winners.** Re-derive every ENHANCED and ORIGINAL winner from the candidate AUCs.
   - **DEV_REPORT replay.** Your ONE DEV_REPORT access: event `DEV_REPORT_TRUTH_REREAD_BY_IHS_INDEPENDENT_VERIFIER` through
     `ihs.common.dev_report_truth(purpose, event)`, exactly once, inside one queue unit. Recompute with your own bootstrap code
     (household, B = 2048, seed 2026101705) the utility clauses, guards, pair benefits, recipient differences, twin and component
     contrasts, the 0.020 target and the claim labels E, R2, R3, P, N, M, C, T, O under `CLAIMS_DRAFT.md`; compare with
     `report/DEV_REPORT_RESULTS.json`.
9. **Planning feasibility.** Independently check the feasibility numbers (`CONFIRMATION_FEASIBILITY.md` and its JSON) with a
   different method (analytic normal formula for the binding clauses; your own Monte Carlo or rescaling for the joint result).
10. **Labels and limitations.** Read `RESEARCH_DECISION.md`, `START_HERE.md` and the comparison tables; flag any statement not
    supported by the numbers (two seeds are not replications; conditional intervals do not cover the adaptive search; no
    injectivity or information-equality inference; reader results are evidence against the declared readers only).

**Rules.** Run every computation longer than a few seconds through the queue:
`python3 -m ihs.jobqueue run --unit VER_<name>_V1 --stage verification --rss-gib 3 -- python3 <script>` (for final-reader
refits use `<repository>/.venv/bin/python`, the READER_PLAN interpreter). At most 2 heavy jobs at once; the queue
enforces it. Code under `results/pcrl_income_headroom_successor_v1/roles/verifier/code/`; report `roles/verifier/VERIFICATION.md`;
machine-readable `INDEPENDENT_VERIFICATION.json` in the results root with `checks`, `discrepancies` (id, severity, description),
`dev_truth_event` (if any) and `verifier_units`. Do not modify any other file; do not commit or push. Charge your effort at the
end: `python3 -m ihs.jobqueue charge --label verifier --cpu-s <upper estimate> --note "agent upper estimate"`.
