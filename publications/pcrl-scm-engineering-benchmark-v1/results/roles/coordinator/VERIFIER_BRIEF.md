# Independent verifier brief (SCM engineering benchmark)

**Context.** This is statistical machine-learning research on public ACS Census tabular records. It asks whether income and occupation probability releases reveal the recorded field SEX, and whether Shared-Cell Merging (SCM, an existing-method combination) preserves usefulness while reducing that recovery.

**Your job.** Recompute the benchmark's results with your own code. Do not import `scm_bench` modules for the quantities you verify: write separate code. You may read private arrays under `.private/pcrl_scm_engineering_benchmark_v1/`, because this verification is authorised. The rules:
- DEV_REPORT truth must be read **only** through `scm_bench.common.dev_report_truth(<purpose>, event_name='DEV_REPORT_TRUTH_REREAD_BY_INDEPENDENT_VERIFIER')`, which logs the re-read.
- Never touch the sealed assessment.
- Write only inside `results/pcrl_scm_engineering_benchmark_v1/roles/verifier/`.

**Checks** (PASS or FAIL each, with evidence):
1. **Teacher and cell identities.** From `runs/FIT_{s}/TEACHERS/*.json` (numpy oblivious trees: route with float32 features > float32 borders, then sum leaf values; income is the binary logit d giving softmax(−d/2, d/2), occupation is the softmax of centred raw scores), recompute the fold-average probabilities on MODEL_SELECT, ATTACK_FIT, ATTACK_VALIDATION and DEV_REPORT features. Then recompute the KL-nearest fine cell against `PARTITION.npz` cent1 and cent2, and the released vectors through `ARM_TABLES.npz`. Confirm they equal `ARM_OUTPUTS.npz` bitwise, for at least the final-bank arms.
2. **Matched capacity.** Final cell counts per arm, from the tables. TWIN, IBSI and SHUF must equal the SCM count at the same β on both seeds. Report nominal cells, occupied cells per role and distinct transmitted vectors.
3. **Merge cost formulas.** Check them on invented data with your own brute force: Ward Brier, KL NLL, accuracy, grouped income costs, the fixed-law within-SEX guard, SEX variance and SEX information.
4. **MODEL_SELECT utility and nomination.** Recompute the 24 clauses for every SCM, TWIN and IBSI arm and the nominated SHUF, from `ARM_OUTPUTS.npz`, the frozen U0/Ucal MODEL_SELECT outputs in `source/bank/FROZEN_BANK_S{s}.npz` and the MODEL_SELECT labels. Recompute the cross-fitted proxy for the nominee, its twin, the IBSI nominee and the SHUF nominee. Confirm the nominations follow `PROTOCOL.md` §7.
5. **Primary endpoints, intervals and labels.** Recompute all 42 primary endpoints from `runs/READERS_{s}/DEV_REPORT_SCORES.npz` (winner reader ids from `READERS_SEED.json`), the bank DEV_REPORT outputs and the logged truth re-read: the household bootstrap with B = 2048 and seed 2026101705 (resampling households with replacement; check the implementation in `task_headroom.inference.group_weights`), Wald t(G − 1) bounds, and the clause and family critical values. Confirm the statuses and the claim decisions in `report/DEV_REPORT_RESULTS.json`.
6. **The transmitted wire and routing.** The bank's wires equal the fit outputs. Outputs are bit-identical within a final cell. The deployment packages (`deploy/`) reproduce the attacked DEV_REPORT wires from raw inputs and refuse SEX, reordered columns and hidden-output requests.
7. **Reader refits.** Refit the pair winners of SCM, TWIN and NEURAL_RAW on both seeds from ATTACK_FIT, with the same reader definitions and the same seed rule, MODEL_SEED_BASE + s. Confirm that the DEV_REPORT scores match the stored ones.
8. **No early or sealed access.** The access logs contain exactly one coordinator re-read and your verifier re-read for this campaign, and nothing sealed.

Write `VERIFICATION.md` (a checks table plus discrepancies) and your scripts. Report every discrepancy, however small.
