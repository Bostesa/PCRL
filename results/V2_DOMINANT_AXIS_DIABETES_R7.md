# Framework D — Dominant-Axis Auditing — Cross-Dataset Summary

Re-evaluation of Round 4 `final.pt` checkpoints with R²_DA = max_k R²_OvR_k.

## Per-dataset gap (mean across seed × pair)

| Dataset | Pair-seeds | Mean R²_onehot | Mean R²_DA | Gap (R²_DA − R²_onehot) | # hidden by R²_onehot ≤ 0.05 (R²_DA > 0.05) |
|---------|------------|----------------|------------|--------------------------|---------------------------------------------|
| diabetes | 18 | 0.0092 | 0.0128 | +0.0036 | 0 |
| **Total** | 18 | — | — | — | **0** |

## Multi-class subset only (where R²_DA can disagree with R²_onehot)

| Dataset | MC pair-seeds | Mean R²_onehot | Mean R²_DA | Gap | Mean MLP-DA Δ |
|---------|---------------|----------------|------------|-----|----------------|
| diabetes | 12 | 0.0125 | 0.0179 | +0.0054 | +0.0035 |

## Convex-combination identity validation

The Convex-Combination Identity predicts `R²_onehot = Σ_k w_k · R²_OvR_k` with `w_k = π_k(1-π_k)/Σ_j π_j(1-π_j)`. Per multi-class pair-seed we compare the predicted vs observed one-hot R²:

| Dataset | MC pair-seeds | Match (residual < 0.01) | Max abs residual |
|---------|---------------|--------------------------|-------------------|
| diabetes | 12 | 12/12 | 0.0000 |
| **Total** | 12 | **12/12 (100.0%)** | — |

