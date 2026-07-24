# Registered predictions — FAccT resubmission ablations (2026-07-24)

Registered BEFORE any smoke or full run of these configurations.
Branch: `ablations-facct-2026-07-24` (cut from `rebuttal-evidence` @ 5739d3b).
Baseline being ablated: erase-layer pilot (`--use-erase-layer --lora-target repr_proj_only`,
LoRA rank per `LORA_BY_DATASET` = Adult/HMDA 8, Diabetes 24, λ_vicreg=1.0,
λ_min=5.0, warmup skipped via leace_init), strict R² 60/60
(`results/rebuttal/erase_layer_pilot_aws/`).

Local CPU smokes (Diabetes seed 0, 5 epochs) are code-path checks only; their
numbers are not evidence for or against these predictions.

## Ablation 1 — LoRA vs plain linear adapter (reviewers AC, AkJK Q1, NY7k Q2)

Arm: identical to erase-layer pilot but each per-purpose rank-8/24 LoRA on
repr_proj is replaced by a full-rank per-purpose linear delta
(ΔW ∈ R^{64×128} + bias, zero-init), 60-cell grid (Adult 24 + HMDA 18 +
Diabetes 18, seeds 0/1/2).

**Prediction (~70% confidence):** the linear adapter MATCHES the LoRA arm —
strict R² pass 60/60, eff_rank within ±2, per_dim_std within ±0.05, task
accuracy within ±2pp on all tasks except possibly Diabetes
`medication_change_outcome` (which the rank-8 ablation already showed is
rank-sensitive; full-rank should match or exceed rank-24's 99.9%).
Rationale: the frozen LEACE erase layer carries compliance (rank-8 vs rank-24
Diabetes ablation showed compliance insensitive to adapter capacity); a
full-rank delta strictly contains the rank-8 hypothesis class.
If this holds, the method simplifies: LoRA is removed by deletion.
Falsifier: any dataset where linear drops below 100% strict pass or task acc
drops >2pp → LoRA (low-rank regularisation) is justified by the ablation.

## Ablation 2 — remove the frozen LEACE erase layer (AkJK Q2)

Arm: `--lora-target repr_proj_only`, NO erase layer, NO LoRA-side LEACE
warm-start (`--no-leace-init`), warmup forced to 0 epochs so the schedule is
identical to the pilot (single-variable change: the frozen LEACE layer),
adapter = LoRA at published ranks, 60-cell grid, seeds 0/1/2.

**Prediction (~85% confidence):** compliance collapses substantially without
the structural component. Registered bet: strict R² pass ≤ 30/60 overall, and
the surviving passes concentrate on low-cardinality attrs (sex-type binaries);
Diabetes age_bucket (K=10) and HMDA race fail on all seeds. Duals saturate
(λ → λ_max) on failing cells. Task accuracy stays roughly at pilot levels.
This is the on-thesis outcome (structure carries the load; prior evidence:
best_epoch=0 on Diabetes, Cotter n_feasible 0/200, cross-purpose constraint
adding little over erase baseline). We report per-cell numbers either way,
including if the optimizer alone does better than predicted (which would
weaken the "structural component is necessary" framing and will be reported
as such).

## Ablation 3 — held-out hyperparameter selection (LgnK, AC)

Protocol: for each dataset, train every candidate config on train, evaluate
the compliance grid on the VALIDATION split only (`--eval-split val`), select
one config per dataset by the pre-registered rule below, then report that
config's TEST-grid numbers (via `--eval-only --eval-split test` on the saved
checkpoints). Test-grid numbers of non-selected configs are not consulted for
selection (they may be computed afterwards for the delta table).

Candidate grids (registered):
- Adult:    λ_min ∈ {0, 5} × warmup_epochs ∈ {0, 5}, rank fixed 8   → 4 configs
- HMDA:     λ_min ∈ {0, 5} × warmup_epochs ∈ {0, 5}, rank fixed 8   → 4 configs
- Diabetes: λ_min ∈ {0, 5} × warmup_epochs ∈ {0, 5} × rank ∈ {8, 24} → 8 configs
All under the erase-layer architecture, 3 seeds, 200 epochs, otherwise pilot
settings. (λ_min=1 midpoint dropped for cost; the reviewer complaint is about
selection protocol, not grid density.)

Selection rule (deterministic, registered):
1. Maximise total strict-pass count (linear R² < 0.05) on the val grid,
   summed over the dataset's 3 seeds.
2. Tie-break 1: maximise clean-compliance count on val (strict pass AND
   per_dim_std_mean ≥ 0.5 AND eff_rank ≥ 2.0).
3. Tie-break 2: maximise mean task accuracy on val (mean over tasks & seeds).
4. Tie-break 3 (simplicity): lower rank, then fewer warmup epochs, then
   lower λ_min.

**Prediction (~75% confidence):** the erase layer makes compliance largely
insensitive to λ_min/warmup, so candidates tie at (or near) full val strict
pass; tie-breaks fall through to task accuracy and simplicity. Selected
configs will NOT necessarily reproduce the published (λ_min=5, warmup-skip,
Diabetes rank 24) — plausibly λ_min=0 gets picked via the simplicity
tie-break — but the selected configs' TEST grid stays 60/60 strict, i.e.
delta vs eval-grid selection = 0 cells. Clean-compliance stays 0/60 in both
(architectural per_dim_std floor, per the 2026-05-30 diagnostic), so the
"clean-compliance rate stable" condition holds trivially and we will say so
explicitly rather than claim it as a win.
Falsifier: selected config loses ≥1 test cell vs published → we report the
delta and which hyperparameter drove it.

## Honesty clauses

- If any arm beats the published pilot on any metric, that is reported, not
  buried (including if it argues for removing components we published).
- Diabetes seed-order effects, saturated duals, or checkpoint-selector
  anomalies (Cotter picking early epochs) get reported per-cell.
- No paper prose from these runs; numbers only.
