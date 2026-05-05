# CelebA Multi-Seed Variance Note (n=3)

## Per-seed numbers

| Seed | Train R²(M) | Train R²(Y) | Val R²(M) | Val R²(Y) | task_acc final | task_acc best |
|---|---|---|---|---|---|---|
| 0 | 0.00286 | 0.00311 | 0.1558 | 0.1705 | 0.7556 | 0.7563 |
| 1 | 0.00303 | 0.00280 | 0.1569 | 0.1561 | 0.7400 | 0.7478 |
| 2 | 0.00358 | 0.00294 | 0.1636 | 0.1672 | 0.7595 | 0.7617 |

## Mean ± std

- **Train-set R²(Male)**:  0.00316 ± 0.00037
- **Train-set R²(Young)**: 0.00295 ± 0.00016
- **Val-partition R²(Male)**:  0.1587 ± 0.0042
- **Val-partition R²(Young)**: 0.1646 ± 0.0075
- **Smiling task acc (final)**: 0.7517 ± 0.0103
- **Smiling task acc (best)**:  0.7553 ± 0.0070

## Variance interpretation

- **Train-set R² spread across seeds**: max(max-min) = 0.00072 (well below the 0.05 PCRL constraint and the <0.01 expected LEACE finite-sample band)
- **Val-partition R² spread across seeds**: max(max-min) = 0.0144 (slightly above the <0.01 prediction; spread driven by per-seed val-loader subsamples rather than training instability)

Within each seed, R² drift across all 25 epochs is < 1e-3 — the decoupled architecture *structurally* prevents the LoRA from moving the projection. Across-seed variance comes from the random val subsample (different `seed` produces different stratified 4096-sample draws of the CelebA val partition), not from training instability.

## Cross-purpose attack across seeds

Worst-case R5 vs. baseline Δ across 6 attackers, per seed:

| Seed | Male Δ (worst-case) | Young Δ (worst-case) |
|---|---|---|
| 0 | -4.5pp | -6.8pp |
| 1 | -6.3pp | -6.8pp |
| 2 | -2.8pp | -8.2pp |

All seeds: worst-case R5 R² ≤ baseline (negative Δ). LoRA training *reduces* non-linear concept recovery rather than adding leakage. OPLoRA is **not triggered** under the spec threshold.