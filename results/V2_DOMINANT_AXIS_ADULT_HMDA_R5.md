# Framework D — Dominant-Axis Auditing — Cross-Dataset Summary

Re-evaluation of Round 4 `final.pt` checkpoints with R²_DA = max_k R²_OvR_k.

## Per-dataset gap (mean across seed × pair)

| Dataset | Pair-seeds | Mean R²_onehot | Mean R²_DA | Gap (R²_DA − R²_onehot) | # hidden by R²_onehot ≤ 0.05 (R²_DA > 0.05) |
|---------|------------|----------------|------------|--------------------------|---------------------------------------------|
| adult | 24 | 0.0119 | 0.0138 | +0.0018 | 0 |
| hmda | 18 | 0.0201 | 0.0410 | +0.0209 | 1 |
| **Total** | 42 | — | — | — | **1** |

## Multi-class subset only (where R²_DA can disagree with R²_onehot)

| Dataset | MC pair-seeds | Mean R²_onehot | Mean R²_DA | Gap | Mean MLP-DA Δ |
|---------|---------------|----------------|------------|-----|----------------|
| adult | 12 | 0.0080 | 0.0114 | +0.0034 | +0.0398 |
| hmda | 9 | 0.0192 | 0.0609 | +0.0417 | +0.1691 |

## Convex-combination identity validation

The Convex-Combination Identity predicts `R²_onehot = Σ_k w_k · R²_OvR_k` with `w_k = π_k(1-π_k)/Σ_j π_j(1-π_j)`. Per multi-class pair-seed we compare the predicted vs observed one-hot R²:

| Dataset | MC pair-seeds | Match (residual < 0.01) | Max abs residual |
|---------|---------------|--------------------------|-------------------|
| adult | 12 | 12/12 | 0.0021 |
| hmda | 9 | 9/9 | 0.0000 |
| **Total** | 21 | **21/21 (100.0%)** | — |

## Worked example — Adult `<income-purpose>` × race (seed 0)

- Purpose: `income_prediction`
- Empirical priors π = [0.0099, 0.0271, 0.0937, 0.0081, 0.8612]
- Per-class R²_OvR = [0.001, 0.013, 0.0105, 0.0087, 0.0186]
- Predicted R²_onehot from convex combo = `0.0142`
- Observed R²_onehot = `0.0142`
- Predicted / observed ratio = `1.0005` (1.00 = exact identity)
- R²_DA = `0.0186` at argmax class = `4`

