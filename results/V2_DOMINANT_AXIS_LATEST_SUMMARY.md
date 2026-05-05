# Framework D — Dominant-Axis Auditing — Latest (R5/R7) Summary

Re-evaluation of the **canonical** ``final.pt`` checkpoints reported in Section 5: Adult Round 5, HMDA Round 5, Diabetes Round 7 (rank-24 LoRA + per-class OvR).

## Per-dataset gap (mean across seed × pair)

| Dataset | Tag | Pair-seeds | Mean R²_onehot | Mean R²_DA | Gap (R²_DA − R²_onehot) | Hidden cases (R²_onehot ≤ 0.05 & R²_DA > 0.05) |
|---|---|---|---|---|---|---|
| adult | ROUND5 | 24 | 0.0119 | 0.0138 | +0.0018 | 0 |
| hmda | ROUND5 | 18 | 0.0201 | 0.0410 | +0.0209 | 1 |
| diabetes | ROUND7 | 18 | 0.0092 | 0.0128 | +0.0036 | 0 |
| **Total** | — | 60 | — | — | — | **1** |

## Multi-class subset only (where R²_DA can disagree with R²_onehot)

| Dataset | Tag | MC pair-seeds | Mean R²_onehot | Mean R²_DA | Gap | Mean MLP-DA Δ |
|---|---|---|---|---|---|---|
| adult | ROUND5 | 12 | 0.0080 | 0.0114 | +0.0034 | +0.0398 |
| hmda | ROUND5 | 9 | 0.0192 | 0.0609 | +0.0417 | +0.1691 |
| diabetes | ROUND7 | 12 | 0.0125 | 0.0179 | +0.0054 | +0.0035 |

## Convex-combination identity validation

Predicts ``R²_onehot = Σ_k w_k · R²_OvR_k`` with ``w_k = π_k(1−π_k) / Σ_j π_j(1−π_j)``. Per multi-class pair-seed we compare predicted vs observed R²_onehot.

| Dataset | Tag | MC pair-seeds | Match (residual < 0.01) | Max abs residual |
|---|---|---|---|---|
| adult | ROUND5 | 12 | 12/12 | 0.0021 |
| hmda | ROUND5 | 9 | 9/9 | 0.0000 |
| diabetes | ROUND7 | 12 | 12/12 | 0.0000 |
| **Total** | — | 33 | **33/33 (100.0%)** | — |

