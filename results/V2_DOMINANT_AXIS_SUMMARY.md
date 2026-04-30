# Framework D — Dominant-Axis Auditing — Cross-Dataset Summary

Re-evaluation of Round 4 `final.pt` checkpoints with R²_DA = max_k R²_OvR_k.

## Per-dataset gap (mean across seed × pair)

| Dataset | Pair-seeds | Mean R²_onehot | Mean R²_DA | Gap (R²_DA − R²_onehot) | # hidden by R²_onehot ≤ 0.05 (R²_DA > 0.05) |
|---------|------------|----------------|------------|--------------------------|---------------------------------------------|
| adult | 24 | 0.0384 | 0.0563 | +0.0179 | 0 |
| diabetes | 18 | 0.0253 | 0.0382 | +0.0129 | 0 |
| hmda | 18 | 0.0455 | 0.0756 | +0.0302 | 4 |
| **Total** | 60 | — | — | — | **4** |

## Multi-class subset only (where R²_DA can disagree with R²_onehot)

| Dataset | MC pair-seeds | Mean R²_onehot | Mean R²_DA | Gap | Mean MLP-DA Δ |
|---------|---------------|----------------|------------|-----|----------------|
| adult | 12 | 0.0572 | 0.0920 | +0.0348 | +0.0437 |
| diabetes | 12 | 0.0346 | 0.0540 | +0.0194 | +0.0158 |
| hmda | 9 | 0.0534 | 0.1138 | +0.0603 | +0.1580 |

## Convex-combination identity validation

The Convex-Combination Identity predicts `R²_onehot = Σ_k w_k · R²_OvR_k` with `w_k = π_k(1-π_k)/Σ_j π_j(1-π_j)`. Per multi-class pair-seed we compare the predicted vs observed one-hot R²:

| Dataset | MC pair-seeds | Match (residual < 0.01) | Max abs residual |
|---------|---------------|--------------------------|-------------------|
| adult | 12 | 12/12 | 0.0097 |
| diabetes | 12 | 12/12 | 0.0000 |
| hmda | 9 | 9/9 | 0.0000 |
| **Total** | 33 | **33/33 (100.0%)** | — |

## Worked example — Adult `<income-purpose>` × race (seed 0)

- Purpose: `income_prediction`
- Empirical priors π = [0.0099, 0.0271, 0.0937, 0.0081, 0.8612]
- Per-class R²_OvR = [0.0188, 0.1107, 0.0124, 0.0072, 0.0844]
- Predicted R²_onehot from convex combo = `0.0575`
- Observed R²_onehot = `0.0575`
- Predicted / observed ratio = `1.0000` (1.00 = exact identity)
- R²_DA = `0.1107` at argmax class = `1`

