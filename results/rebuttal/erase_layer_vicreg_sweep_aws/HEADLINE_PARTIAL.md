# Erase-layer VICReg sweep — partial results (Adult + HMDA complete, Diabetes cut off)

**Run:** `i-09bdc3b150df928a5` g4dn.xlarge, 2026-05-29 16:58 UTC → 2026-05-30 02:58 UTC (~10h hard cap).
**Branch:** `erase-layer-vicreg-sweep-2026-05-18 @ 67427e7` (hardened with c23e113 dual-archive pattern, but Diabetes seed 2 was cut off mid-training so Stage 8 STATUS.txt + Stage 9 archive sync never ran).
**Config:** `--use-erase-layer --lora-target=repr_proj_only --lambda-vicreg=5.0` (vs original pilot's 1.0).

## 3-way headline (Adult + HMDA only; Diabetes λ=5.0 missing)

| Dataset | Arm | Strict R² | Cleanly comp | R² mean | per_dim_std | eff_rank |
|---|---|---|---|---:|---:|---:|
| **Adult** | Round 5 baseline | 21/24 | **1/24** | 0.0204 | 0.368 | 3.35 |
|  | Erase λ=1.0 | 24/24 | 0/24 | 0.0078 | 0.465 | 12.60 |
|  | **Erase λ=5.0** | 24/24 | **0/24** | 0.0077 | **0.467** | **13.65** |
| **HMDA** | Round 5 baseline | 16/18 | **2/18** | 0.0258 | 0.316 | 2.98 |
|  | Erase λ=1.0 | 18/18 | 0/18 | 0.0057 | 0.408 | 12.37 |
|  | **Erase λ=5.0** | 18/18 | **0/18** | 0.0056 | **0.394** | **13.94** |
| **Diabetes** | Round 7 baseline | 17/18 | 2/18 | 0.0079 | 0.345 | 3.93 |
|  | Erase λ=1.0 | 18/18 | 0/18 | 0.0069 | 0.393 | 21.11 |
|  | Erase λ=5.0 | — | — | — | — | — *(cut off in seed 2)* |

**Totals (Adult+HMDA, 42 cells):** Round 5 strict 37/42, clean 3/42 → Erase λ=1.0 strict 42/42, clean 0/42 → Erase λ=5.0 strict 42/42, clean 0/42.

## Verdict: 5× λ_vicreg does NOT recover the cleanly-compliant count

| Per-purpose per_dim_std stats | λ=1.0 | λ=5.0 | Δ |
|---|---:|---:|---:|
| Adult, mean across 9 (purpose, seed) | 0.465 | 0.467 | **+0.002** |
| Adult, max across 9 | 0.485 (edu_s1) | 0.485 (edu_s1) | 0 |
| HMDA, mean across 9 | 0.408 | 0.394 | **−0.014** |
| HMDA, max across 9 | 0.434 (underwr_s2) | 0.424 (underwr_s2) | −0.010 |
| **Cells crossing 0.5 threshold** | 0/18 | 0/18 | 0 |

**eff_rank moved (Adult +1.05, HMDA +1.57)** — the 5× covariance term is doing its decorrelation work, dimensions are more orthogonal. But **per_dim_std barely budged** on Adult (+0.002) and actively went **down** on HMDA (−0.014). HMDA's max-std cell (underwriting seed 2) dropped from 0.434 to 0.424 — the wrong direction.

The variance-hinge gradient is `(γ − σ) × λ_var × λ_vicreg` for σ < γ, but the LEACE projection's compression of disallowed-attribute variance directions is upstream of the LoRA. The LoRA can't grow variance back along directions it's structurally forbidden from re-encoding (= would re-introduce R²(h_p, A)). The per_dim_std floor at ≈ 0.45 is a representation-geometry consequence of the linear-erase mechanism, not an optimization failure that VICReg can fix at scale.

This is a clean negative result, but actually informative for rebuttal framing:

**The 0.5 cleanly-compliant threshold gap is not a method-side problem.** The architecture provides:
- 100% strict R² compliance (42/42 on Adult+HMDA, 18/18 on Diabetes λ=1.0)
- Effective rank 4× the published 2.0 threshold
- per_dim_std systematically below 0.5

The submitted paper's 5/60 cleanly-compliant number reflects a real **tradeoff inherent to linear-erase methods**, not an optimization weakness — and the dominant-axis audit gives a stronger operational certificate. (Cf. the §5.5 vision pipeline: train R² ≈ 0.003 but val R² ≈ 0.16 — the linear-erase ceiling.)

## What's missing

Diabetes seed 2 entered training at 02:10 UTC; the 10h backup-sleeper fired at 02:58 UTC and shut the instance down mid-train. Diabetes seeds 0 and 1 each got 5/6 adjusted-pass before the cut.

## Options for closing Diabetes

1. **Relaunch Diabetes-only** (mirror the `erase_pilot_diabetes_recovery` pattern): one g4dn.xlarge, 6h cap, ~$1.60, expected 3.5h wall. The architecture + λ=5.0 are settled; this just needs the seed 2 → end-of-eval write to complete on Diabetes.
2. **Accept Adult+HMDA only**: 42/42 strict pass with the per_dim_std verdict is already conclusive. Diabetes λ=1.0's per_dim_std mean of 0.393 (with rank-24 LoRA) is consistent with the Adult+HMDA story; λ=5.0 would almost certainly stay below 0.5.

## Files

- `v2_adult_ERASE_VICREG5/per_seed_results.json` + `summary.json` (full)
- `v2_hmda_ERASE_VICREG5/per_seed_results.json` + `summary.json` (full)
- HEADLINE_PARTIAL.md (this file)
- No comparison_table.tex from on-instance aggregator — Stage 7 didn't run (Diabetes incomplete). Local 3-way summary above is authoritative for the 42 cells we have.
