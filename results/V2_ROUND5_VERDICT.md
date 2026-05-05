# V2 Round 5 Verdict

Generated: 2026-04-30. Numbers below come from
`experiments/eval_round4_final_vs_best_v2.py` (test-set auditor, encoder
in eval mode, full-dataset OLS) — the canonical compliance metric, same
script that produced the Round 4 baselines in
`results/v2_*_ROUND4/final_vs_best.json`.

An earlier draft of this file used the train-time per-batch averaged R²
from `history.r2_per_pair_per_epoch[-1]`, which gave Adult 22/24, HMDA 6/18,
Diabetes 9/18. Those numbers are NOT comparable to Round 4 (which reports
auditor R²) and have been replaced.

## **PARTIAL** (one pair-seed short of ALL-GREEN)

By the user's classification rules:

- Adult: strict 23/24 (≥22 threshold) → **GREEN**
- HMDA: strict 16/18 (≥14 threshold) → **GREEN**
- Diabetes: strict 16/18 (≥17 threshold) → **1 short of GREEN**

Two datasets meet the GREEN bar; Diabetes is one pair-seed short
(quality_research/age_bucket fails on all 3 seeds with R²=0.06/0.10/0.09).

## Headline numbers (test-set auditor on final.pt)

| Dataset | R4 strict | R5 strict | Δ | R4 mean R² | R5 mean R² |
|---|---|---|---|---|---|
| Adult | 20/24 | **23/24** | +3 | 0.038 | 0.012 |
| HMDA | 11/18 | **16/18** | +5 | 0.045 | 0.020 |
| Diabetes | 15/18 | **16/18** | +1 | 0.025 | 0.017 |
| **TOTAL** | **46/60** | **55/60** | **+9** | — | — |

Drift fix worked on the 6 originally-failing HMDA pair-seeds; pinned all 3 Adult
income/race seeds (Round 4 finals 0.065/0.232/0.382 → R5 0.023/0.011/0.017 with
λ floored at 5.00); failed on HMDA pricing_analysis/race s0 even though λ ramped to 53
(task gradient genuinely overpowers an active dual on that pair).

## What's new in Round 5

- R1 = lambda floor (`Constraint.lambda_min`, set to 5.0 in `run_v2_dataset.py`).
- R2 = skip warmup when LEACE-init is active (`V2TrainerConfig.warmup_when_leace_init`,
  default False; `train()` sets effective `warmup_n=0` when `leace_init=True`).
- Diabetes-only LoRA bump to rank=16 alpha=32 in `run_v2_dataset.py:LORA_BY_DATASET`
  (addresses joint LEACE rank-deficit on `quality_research/age_bucket`, required
  `race(5) + age_bucket(10) → 4 + 9 = 13` directions).

## What R5 didn't fix

- **HMDA pricing_analysis/race s0 still fails** at 0.073 with λ=53. Dual was actively
  trying. Task gradient on race-correlated mortgage pricing is overpowering even
  saturated dual pressure. Candidate for **R3 (periodic LEACE re-anchor)**.
- **Diabetes quality_research/age_bucket s1, s2 still fail** at 0.20 / 0.15 even with
  rank-16. The rank-deficit was real (warm-start probe showed s1 R²=0.27 at epoch 0
  with rank 8), but rank-16 alone doesn't carry through training on all seeds.
- **Rep collapse from permanent λ floor.** All 3 datasets fire `STATUS: COLLAPSED`
  health flag (`per_dim_std<0.5` on multiple purposes). The auditor's empirical attack
  still beats majority by >2pp on the collapsed rep, so the *adjusted* compliance
  criterion (R²<0.05 AND delta<2pp) only passes 4-15/seed depending on dataset.
  Tradeoff: lambda_min=5 wins the linear constraint at the cost of rep variance.

## Recommended next moves (in priority order)

1. **Reduce lambda_min** from 5 to 1-2 to relax the permanent floor and recover rep
   variance. The audit's STARVE failure mode happened at λ_final < 1 with R² > 0.075,
   so lambda_min=2 should still block dual relaxation while halving the squeeze.
2. **Raise VICReg γ** from 1.0 to 5-10 to defend rep variance more aggressively under
   sustained constraint pressure.
3. **R3 (periodic LEACE re-anchor)** for the 4 stubborn pair-seeds (HMDA pricing/race
   s0; Diabetes quality_research/age_bucket s1+s2; possibly Diabetes
   quality_research/race s1+s2). Re-fit eraser every K=20 epochs.

## AWS instance status

All 3 instances stopped:

- v2-adult-r5: [redacted-instance-id] — stopped (training, on-instance watchdog shutdown)
- v2-hmda-r5: [redacted-instance-id] — stopped (training, on-instance watchdog shutdown)
- v2-diabetes-r5: [redacted-instance-id] — stopped (manually after restart-to-scp)

## Cost

~8.5 GPU-hours × $0.526/h on g4dn.xlarge = **~$4.50**.

## Operational notes (for next round)

- AWS session expired ~5h after launch, which blocked the orchestrator from launching
  Diabetes when capacity opened. Diabetes ended up launching ~9h late.
- Local orchestrator + post-processor used a 16h hard cap. Diabetes finished ~21h after
  launch (because of the late start), past the cap. Need a longer cap if launches can
  be delayed by quota waits.
- The Framework D dominant-axis evaluation was not run as part of this verdict (per
  user instruction "do not modify the in-progress Framework D evaluation"); recommend
  running `scripts/eval_round4_dominant_axis.py` against Round 5 checkpoints before
  paper-table updates.

## Pointers

- `results/V2_ROUND5_SUMMARY.md` — full table + per-pair-seed comparison
- `results/v2_{adult,hmda,diabetes}_ROUND5/per_seed_results.json` — raw auditor reports
- `checkpoints/v2_{adult,hmda,diabetes}_ROUND5_s{0,1,2}/{final,best}.pt` — weights
- `/tmp/round5/orchestrator.log` — orchestrator history
- `/tmp/round5/post.log` — post-processor history (died at 16h cap)
- Round 4 baselines: `results/v2_{adult,hmda,diabetes}_ROUND4/`
