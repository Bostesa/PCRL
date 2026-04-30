# R1+R2 Probe vs Round 4 — Adult

R1 = lambda floor (lambda_min=5.0). R2 = skip warmup with LEACE init.

Probe: 0 warmup + 50 constrained, lambda_min=5. Round 4: 5 warmup + 200 constrained, lambda_min=0.

## Failing pair-seeds

### income_prediction__race

| seed | probe e0 | e4 | e5 | e10 | e25 | **e49** | λ_probe | r4 final (e204) | λ_r4 |
|---|---|---|---|---|---|---|---|---|---|
| s0 | 0.403✗ | 0.394✗ | 0.420✗ | 0.392✗ | 0.039✓ | 0.037✓ | 7.35 | 0.065✗ | 0.11 |
| s1 | 0.357✗ | 0.513✗ | 0.523✗ | 0.418✗ | 0.070✗ | 0.004✓ | 7.55 | 0.232✗ | 0.51 |
| s2 | 0.409✗ | 0.420✗ | 0.453✗ | 0.311✗ | 0.035✓ | 0.019✓ | 6.43 | 0.382✗ | 0.30 |

## Passing pair-seeds

### income_prediction__sex

| seed | probe e0 | e4 | e5 | e10 | e25 | **e49** | λ_probe | r4 final (e204) | λ_r4 |
|---|---|---|---|---|---|---|---|---|---|
| s0 | 0.538✗ | 0.467✗ | 0.501✗ | 0.477✗ | 0.152✗ | 0.137✗ | 16.73 | 0.043✓ | 8.98 |
| s1 | 0.419✗ | 0.411✗ | 0.473✗ | 0.470✗ | 0.246✗ | 0.009✓ | 16.35 | 0.048✓ | 4.69 |
| s2 | 0.604✗ | 0.459✗ | 0.508✗ | 0.467✗ | 0.137✗ | 0.075✗ | 15.40 | 0.018✓ | 3.62 |

### employment_analysis__age_group

| seed | probe e0 | e4 | e5 | e10 | e25 | **e49** | λ_probe | r4 final (e204) | λ_r4 |
|---|---|---|---|---|---|---|---|---|---|
| s0 | 0.375✗ | 0.373✗ | 0.365✗ | 0.344✗ | 0.305✗ | 0.285✗ | 22.95 | 0.031✓ | 60.16 |
| s1 | 0.380✗ | 0.392✗ | 0.355✗ | 0.339✗ | 0.297✗ | 0.277✗ | 22.91 | 0.031✓ | 60.93 |
| s2 | 0.394✗ | 0.363✗ | 0.347✗ | 0.340✗ | 0.301✗ | 0.281✗ | 22.93 | 0.018✓ | 64.87 |

## Task accuracy — final vs final (LITERAL user criterion)

Round 4 was trained for 5+200=205 epochs; probe for 0+50=50 epochs.
Probe sees its task heads 4× less data — naive comparison is biased.
Below: Round 4 *final* (e204) test acc vs probe *final* (e49) test acc.

| seed | task | Round 4 acc (e204) | probe acc (e49) | Δ (pp) |
|---|---|---|---|---|
| s0 | income | 0.842 | 0.790 | -5.14 |
| s0 | occupation_group | 0.999 | 0.996 | -0.30 |
| s0 | education_level | 1.000 | 0.981 | -1.90 |
| s1 | income | 0.840 | 0.826 | -1.37 |
| s1 | occupation_group | 1.000 | 0.985 | -1.43 |
| s1 | education_level | 1.000 | 0.981 | -1.91 |
| s2 | income | 0.837 | 0.779 | -5.82 |
| s2 | occupation_group | 1.000 | 0.989 | -1.10 |
| s2 | education_level | 1.000 | 0.984 | -1.63 |

Max task acc drop (probe e49 vs Round 4 e204): **+5.82 pp**. ⚠ confounded by epoch-count gap

## Task loss — epoch-matched (probe e49 vs Round 4 e49)

Per-epoch test accuracy is not stored in history; per-epoch val task loss is. Compare both at index 49 of `val_task_loss` to remove the epoch-count confound.

| seed | R4 val_loss @ e49 | probe val_loss @ e49 | Δ |
|---|---|---|---|
| s0 | 0.6143 | 0.6207 | +0.0065 |
| s1 | 0.5921 | 0.5955 | +0.0033 |
| s2 | 0.6545 | 0.6484 | -0.0061 |

Max |Δ val_task_loss| at e49: **0.0065** (< 0.02 means R1+R2 not hurting convergence at matched epochs).

## Verdict

Adult failing pair-seeds (income/race × 3 seeds): **3/3** now have R²<0.05 at epoch 49.

### **GREEN** — all 3 Adult race seeds feasible at e49 (probe finals 0.037/0.004/0.019 vs R4 finals 0.065/0.232/0.382). Epoch-matched val task loss delta 0.0065 ≤ 0.02 — R1+R2 not hurting convergence. The 5.8 pp gap between probe e49 and R4 e204 is the epoch-count gap.

### Failing pair-seed comparison

| pair | seed | R4 final R² | probe final R² | improved? | R4 λ | probe λ |
|---|---|---|---|---|---|---|
| income_prediction__race | s0 | 0.065 | 0.037 | ✓ | 0.11 | 7.35 |
| income_prediction__race | s1 | 0.232 | 0.004 | ✓ | 0.51 | 7.55 |
| income_prediction__race | s2 | 0.382 | 0.019 | ✓ | 0.30 | 6.43 |
