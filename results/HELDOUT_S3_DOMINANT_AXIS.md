# Framework D — Dominant-Axis Auditing — Cross-Dataset Summary

Re-evaluation of Round 4 `final.pt` checkpoints with R²_DA = max_k R²_OvR_k.

## Per-dataset gap (mean across seed × pair)

| Dataset | Pair-seeds | Mean R²_onehot | Mean R²_DA | Gap (R²_DA − R²_onehot) | # hidden by R²_onehot ≤ 0.05 (R²_DA > 0.05) |
|---------|------------|----------------|------------|--------------------------|---------------------------------------------|
| adult | 8 | 0.0273 | 0.0317 | +0.0044 | 1 |
| **Total** | 8 | — | — | — | **1** |

## Multi-class subset only (where R²_DA can disagree with R²_onehot)

| Dataset | MC pair-seeds | Mean R²_onehot | Mean R²_DA | Gap | Mean MLP-DA Δ |
|---------|---------------|----------------|------------|-----|----------------|
| adult | 4 | 0.0239 | 0.0328 | +0.0089 | +0.0377 |

## Convex-combination identity validation

The Convex-Combination Identity predicts `R²_onehot = Σ_k w_k · R²_OvR_k` with `w_k = π_k(1-π_k)/Σ_j π_j(1-π_j)`. Per multi-class pair-seed we compare the predicted vs observed one-hot R²:

| Dataset | MC pair-seeds | Match (residual < 0.01) | Max abs residual |
|---------|---------------|--------------------------|-------------------|
| adult | 4 | 4/4 | 0.0000 |
| **Total** | 4 | **4/4 (100.0%)** | — |

