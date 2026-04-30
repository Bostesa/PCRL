# V2 Round 5 Summary

Generated: 2026-04-30 (manual, post-processor hit 16h cap before Diabetes finished).

R1 (lambda floor=5.0) + R2 (skip warmup with LEACE init) + Diabetes rank-16
LoRA. 3 seeds × 200 epochs × 3 datasets on AWS g4dn.xlarge.

Commits: `dbe0fdc` (R1+R2) + `b3f43c5` (Diabetes rank-16).

## Headline — strict R²<0.05 on final.pt

| Dataset | R4 strict | R5 strict | Δ | R5 mean R² (final) | Status |
|---|---|---|---|---|---|
| Adult | 20/24 | **22/24** | +2 | 0.024 | GREEN (≥22 threshold met) |
| HMDA | 6/18 | **16/18** | +10 | 0.030 | GREEN (≥14 threshold met) |
| Diabetes | 6/18 | **9/18** | +3 | 0.072 | below 17/18 GREEN threshold |
| **TOTAL** | **32/60** | **47/60** | **+15** | — | — |

## Per-pair-seed R² (final.pt) — previously failing pairs

### Adult income_prediction/race  (was the optimizer-drift target)

| seed | R4 final | R5 final | R4 λ | R5 λ |
|---|---|---|---|---|
| s0 | 0.065 ✗ | **0.023 ✓** | 0.11 | 5.00 |
| s1 | 0.232 ✗ | **0.011 ✓** | 0.51 | 5.00 |
| s2 | 0.382 ✗ | **0.017 ✓** | 0.30 | 5.00 |

R1 lambda floor pinned λ at 5.0 across all 3 seeds, killing the post-feasibility drift.

### HMDA underwriting/race

| seed | R4 final | R5 final | R4 λ | R5 λ |
|---|---|---|---|---|
| s0 | 0.080 ✗ | **0.014 ✓** | 3.07 | 5.00 |
| s1 | 0.094 ✗ | **0.040 ✓** | 50.5 | 5.00 |
| s2 | 0.123 ✗ | **0.031 ✓** | 2.98 | 5.00 |

### HMDA pricing_analysis/race

| seed | R4 final | R5 final | R4 λ | R5 λ |
|---|---|---|---|---|
| s0 | 0.070 ✗ | 0.073 ✗ | 1.35 | 53.0 |
| s1 | 0.089 ✗ | **0.040 ✓** | 1.08 | 5.00 |
| s2 | 0.106 ✗ | **0.020 ✓** | 0.66 | 8.4 |

s0 still narrowly fails despite λ ramping to 53 — task gradient on this pair is genuinely
overpowering. Candidate for R3 (periodic LEACE re-anchor) in a future round.

### Diabetes quality_research/age_bucket  (rank-16 LoRA target)

| seed | R4 final | R5 final | R4 λ | R5 λ | rank |
|---|---|---|---|---|---|
| s0 | 0.163 ✗ | **0.061** ✗ | — | 5.9 | 16 |
| s1 | 0.234 ✗ | 0.203 ✗ | — | 75.2 | 16 |
| s2 | 0.128 ✗ | 0.146 ✗ | — | 66.5 | 16 |

Rank-16 helped seed 0 substantially (0.163 → 0.061, just above τ) but seeds 1+2 still fail.
The rank-deficit diagnosis was correct but not sufficient on its own — the optimizer also
struggles even when the warm-start is rank-feasible (high λ values 75/66 confirm dual was
trying hard).

## Adjusted compliance criterion (R²<0.05 AND auditor delta<2pp)

| Dataset | R5 adj pass | Note |
|---|---|---|
| Adult | 12/24 (4/8 mean) | rep collapse from permanent λ floor leaks nonlinear signal |
| HMDA | 3/18 (1.0/6 mean) | same rep-collapse problem amplified |
| Diabetes | 15/18 (5.0/6 mean) | rank-16 prevented worst collapse on Diabetes |

Auditor's empirical attack still beats majority by >2pp on collapsed reps even when linear R²
is satisfied. **STATUS: COLLAPSED** flagged on all 3 datasets via `per_dim_std<0.5` health check —
the lambda_min=5 floor + VICReg γ=1 wasn't enough to maintain rep variance under permanent
constraint pressure.

## Cost

- Adult: 3 seeds × ~50min = 2.5h, finished ~05:50Z 2026-04-30
- HMDA: 3 seeds × ~60min = 3h, finished ~06:50Z 2026-04-30
- Diabetes: 3 seeds × ~58min = 3h, finished ~19:46Z 2026-04-30 (launched 8h late after
  AWS session expired blocked the queued launch until ~11:43Z)
- Total: ~8.5h compute × $0.526/h on g4dn.xlarge ≈ **$4.50**

## Files

- `results/v2_{adult,hmda,diabetes}_ROUND5/per_seed_results.json` — auditor reports
- `checkpoints/v2_{adult,hmda,diabetes}_ROUND5_s{0,1,2}/{final,best}.pt` — weights
- `results/v2_R1R2_probe_vs_round4.md` — pre-launch CPU probe motivating R1+R2
- `results/v2_optimizer_drift_audit.md` — Round 4 audit that diagnosed the drift
- `results/v2_fix1_audit.md` — joint LEACE audit that diagnosed Diabetes rank deficit
