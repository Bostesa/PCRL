# Brief: independent numerical verifier (SCM utility-matched follow-up)

You are the independent verifier. Recompute every checked quantity with **your own code** (numpy, scipy, scikit-learn), written in
`results/pcrl_scm_utility_matched_v1/roles/verifier/` (scripts `checkN_*.py` and their `checkN_*.json` outputs). Import campaign code only
where a check's object under test *is* that code (say so per check). Never write person-level arrays into the repository; a crash copy,
if you make one, goes to your session scratchpad (mode 600) and is deleted after use.

Environment: run from the worktree root with
`PYTHONPATH=.:<source private pylib>` and `<repository>/.venv/bin/python` (Python 3.13). Every numerical script runs
through the shared queue, one thread, e.g.
`python -m scm_um.jobqueue run --unit VERIFY_CHECK3_V1 --stage verify --rss-gib 2 -- <repository>/.venv/bin/python results/pcrl_scm_utility_matched_v1/roles/verifier/check3_trajectories.py`
(at most two heavy jobs run at once across the whole campaign; use a new unit id for every attempt). Do not commit; the coordinator commits.

Read first: `PROTOCOL.md`, `UNIT_INVENTORY.json`, `ANALYSIS_LOCK.json`, `SOURCE_ADMISSION.json`, `TRAJECTORY_COMPLETE_LOCK.json`,
`TRAJECTORY_SUMMARY.json`, `SELECTION_LOCK.json`, and the code in `scm_um/` and `scm_bench/`.

## Checks (PASS / FAIL with evidence; list every discrepancy however small, numbered D1, D2, ...)

1. **Source admission and household-role integrity.** Hashes of the admitted copies against their manifests (source TRAINING_COMPLETE_LOCK,
   FIT_SEED files_sha256, public EXTRACTION_SUMMARY, FROZEN_BANK_MANIFEST); household disjointness between roles (MODEL_FIT, MODEL_SELECT,
   ATTACK_FIT, ATTACK_VALIDATION, DEV_REPORT units); the routing arrays reproduce the historical outputs from your own teacher evaluation
   and KL routing; no SEALED export exists and `load_role` refuses it.
2. **Trajectory checkpoints and aliases.** With your own merge implementation (or an independent re-derivation of the merge sequence's
   costs), confirm for both seeds: the historical endpoints (SCM_b1.0, TWIN_b1.0, IBSI_b1.0_g*, SHUF_b1.0_r*) equal the recorded prefixes at
   261 / 266 cells; the recorded running fitted occupation-NLL F_m equals your recomputation; every registered checkpoint label maps to the
   prefix the protocol rule defines (NLL grid with the 1e-15 slack, counts, TERM, S{t} companions, HIST), reached flags, and that effective
   releases are deduplicated exactly when per-initial-cell tables are bit-identical. Capacity fields (nominal, occupied per role, distinct vectors).
3. **Utility calculations and nomination.** Recompute all MODEL_SELECT clauses, subgroup guards, expenditures and the cross-fitted proxy
   (own readers with the same settings) for every effective release; re-apply the nomination rule and fallback.
4. **NLL and cell-count matching; comparator selection.** Re-apply the matching tolerance, candidate sets, strongest-proxy selection,
   common-γ / common-r rules, nearest diagnostics and same-count roles; confirm the selection effort counts.
5. **Primary results, intervals and labels** (after the coordinator's report exists). One logged DEV_REPORT replay through
   `scm_um.common.dev_report_truth(<purpose>, event_name='DEV_REPORT_TRUTH_REREAD_BY_INDEPENDENT_VERIFIER')` — the only permitted route
   (the coordinator will tell you when the reporting lock is committed). Dry-run your pipeline on invented labels first. Recompute the
   household bootstrap, AUCs, every primary (58) and secondary (24) endpoint, availabilities, statuses, matching carry-over and direction,
   the strict M → IBSI → SHUF sequence, and every claim label from the registered conjunctions in the statistical reviewer's
   `FROZEN DESIGN` block (`roles/statistical_reviewer/PREFIT_STATISTICAL_REVIEW.md`), which is authoritative where the protocol is ambiguous.
6. **Raw-field packages and input validation.** Run every package through the API and CSV paths on DEV_REPORT raw inputs and compare with
   the evaluated wires; try the refusals (extra, reordered, SEX column or key, hidden outputs).
7. **Predefined reader refits.** Refit, with your own code and the source settings, the pair winners of SCM, ORD_NLLM, IBSI_NLLM and
   NEURAL_RAW on both seeds; compare DEV_REPORT scores and ATTACK_VALIDATION AUCs with the stored ones.
8. **Access.** No early DEV_REPORT access (event times vs REPORTING_LOCK commit), no sealed access, ledger order.

Write `VERIFICATION.md` (table of checks, discrepancies, favourable and unfavourable readings, open issues) and
`INDEPENDENT_VERIFICATION.json` (machine-readable: per check status, key numbers, discrepancies). Charge your own CPU estimate in your report.
