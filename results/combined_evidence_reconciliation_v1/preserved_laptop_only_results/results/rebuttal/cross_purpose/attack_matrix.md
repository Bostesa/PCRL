# Per-cell cross-purpose attack matrix

FLAG = mean Δ_pp over seeds > +1.0pp on h_concat. Cells matching the §5.4 / §5.5 audit pool.

## ADULT (tag=CROSS_PURPOSE_AB, n_seeds=3)
| attr | LR | MLP | XGB |
|---|---|---|---|
| age_group | +0.00pp (ok) | +32.72pp (**FLAG**) | +26.23pp (**FLAG**) |
| income | +0.00pp (ok) | +8.50pp (**FLAG**) | +6.97pp (**FLAG**) |
| marital_status | +0.00pp (ok) | +46.06pp (**FLAG**) | +42.78pp (**FLAG**) |
| race | +0.00pp (ok) | +10.13pp (**FLAG**) | +5.01pp (**FLAG**) |
| sex | +0.00pp (ok) | +31.20pp (**FLAG**) | +27.33pp (**FLAG**) |

## HMDA (tag=CROSS_PURPOSE_AB, n_seeds=3)
| attr | LR | MLP | XGB |
|---|---|---|---|
| ethnicity | +0.00pp (ok) | +26.91pp (**FLAG**) | +26.16pp (**FLAG**) |
| race | +0.00pp (ok) | +34.79pp (**FLAG**) | +32.09pp (**FLAG**) |
| sex | +0.00pp (ok) | +38.39pp (**FLAG**) | +37.43pp (**FLAG**) |

## DIABETES (tag=CROSS_PURPOSE_DIABETES, n_seeds=3)
| attr | LR | MLP | XGB |
|---|---|---|---|
| age_bucket | +0.00pp (ok) | +59.51pp (**FLAG**) | +38.52pp (**FLAG**) |
| gender | +0.00pp (ok) | +1.67pp (**FLAG**) | -0.25pp (ok) |
| race | +0.00pp (ok) | +0.02pp (ok) | -0.26pp (ok) |
