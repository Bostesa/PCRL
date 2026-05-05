# Round 4 Adult — binarized race re-evaluation (FFB ICLR 2024)

Re-evaluates `final.pt` for all 3 seeds with binarized race (1=White, 0=non-White).
No retraining — encoders are loaded from disk and only the audit metric changes.

## Per-seed race-pair table

| Pair | seed | 5-class R² | binarized R² | binarized passes (R²<0.05)? | binarized post-hoc delta | binarized combined passes? |
|---|---|---|---|---|---|---|
| income_prediction/race | 0 | 0.0575 | 0.0844 | NO | +0.0047 | NO |
| employment_analysis/race | 0 | 0.0130 | 0.0175 | YES | +0.0269 | NO |
| education_assessment/race | 0 | 0.0000 | 0.0000 | YES | -0.0001 | YES |
| income_prediction/race | 1 | 0.1996 | 0.3213 | NO | +0.0623 | NO |
| employment_analysis/race | 1 | 0.0085 | 0.0111 | YES | +0.0126 | YES |
| education_assessment/race | 1 | 0.0201 | 0.0267 | YES | +0.0103 | YES |
| income_prediction/race | 2 | 0.3617 | 0.5669 | NO | +0.1038 | NO |
| employment_analysis/race | 2 | 0.0058 | 0.0079 | YES | +0.0213 | NO |
| education_assessment/race | 2 | 0.0007 | 0.0007 | YES | -0.0001 | YES |

## Race-pair aggregates (n=9 across 3 seeds × 3 race pairs)

- 5-class: mean R² = **0.0741**, n<0.05 = **6/9**
- Binarized: mean R² = **0.1152**, n<0.05 = **6/9**

## Combined-convention 24-cell aggregate

Race pairs use binarized R² + binarized post-hoc delta; non-race pairs unchanged from `final_vs_best.json`.

- Pass count under combined convention (R²<0.05 AND delta<0.02): **8/24**
- Cells with R² < 0.05 under combined convention: **20/24**
- Mean R² under combined convention: **0.0538**

## Per-cell breakdown (combined convention)

| seed | purpose | attribute | convention | R² | delta | combined pass |
|---|---|---|---|---|---|---|
| 0 | income_prediction | race | binarized | 0.0844 | +0.0047 | NO |
| 0 | income_prediction | sex | binary | 0.0348 | -0.0020 | YES |
| 0 | employment_analysis | race | binarized | 0.0175 | +0.0269 | NO |
| 0 | employment_analysis | age_group | multinomial | 0.0094 | +0.1322 | NO |
| 0 | employment_analysis | marital_status | multinomial | 0.0189 | +0.2225 | NO |
| 0 | education_assessment | sex | binary | 0.0000 | +0.0493 | NO |
| 0 | education_assessment | race | binarized | 0.0000 | -0.0001 | YES |
| 0 | education_assessment | income | multinomial | 0.0000 | +0.0015 | YES |
| 1 | income_prediction | race | binarized | 0.3213 | +0.0623 | NO |
| 1 | income_prediction | sex | binary | 0.0340 | +0.1717 | NO |
| 1 | employment_analysis | race | binarized | 0.0111 | +0.0126 | YES |
| 1 | employment_analysis | age_group | multinomial | 0.0060 | +0.0532 | NO |
| 1 | employment_analysis | marital_status | multinomial | 0.0028 | +0.2144 | NO |
| 1 | education_assessment | sex | binary | 0.0779 | +0.1016 | NO |
| 1 | education_assessment | race | binarized | 0.0267 | +0.0103 | YES |
| 1 | education_assessment | income | multinomial | 0.0346 | +0.0058 | YES |
| 2 | income_prediction | race | binarized | 0.5669 | +0.1038 | NO |
| 2 | income_prediction | sex | binary | 0.0103 | +0.1364 | NO |
| 2 | employment_analysis | race | binarized | 0.0079 | +0.0213 | NO |
| 2 | employment_analysis | age_group | multinomial | 0.0046 | +0.0495 | NO |
| 2 | employment_analysis | marital_status | multinomial | 0.0062 | +0.1627 | NO |
| 2 | education_assessment | sex | binary | 0.0007 | +0.0452 | NO |
| 2 | education_assessment | race | binarized | 0.0007 | -0.0001 | YES |
| 2 | education_assessment | income | multinomial | 0.0138 | +0.0015 | YES |

## Verdict: **RED**

Metric artifact wasn't dominant; commit to Fix 1 implementation.

## Diagnosis: binarization made R² *worse*, not better

Across every cell with non-trivial signal, the binarized R² is **higher** than
the 5-class one-hot R². The income/race outliers move in the wrong direction
under FFB convention:

| seed | income/race 5-class | income/race binarized | Δ |
|---|---|---|---|
| 0 | 0.0575 | 0.0844 | **+0.0269** |
| 1 | 0.1996 | 0.3213 | **+0.1217** |
| 2 | 0.3617 | 0.5669 | **+0.2052** |

The same direction holds for employment/race (0.013→0.018, 0.009→0.011,
0.006→0.008) and education/race (no change). On the post-hoc auditor side,
binarized delta is also uniformly higher than 5-class delta; one previously
combined-passing cell (employment/race seed 2) flipped to combined-fail
because binarized delta crept just over 0.02 (0.0158→0.0213).

Headline shift under FFB convention:

|  | mean R² (24) | n<0.05 (24) | combined-pass (24) |
|---|---|---|---|
| Original (multinomial) | 0.0384 | 20/24 | 10/24 |
| FFB combined            | 0.0538 | 20/24 | **8/24** |

`n<0.05` is unchanged at 20/24 — binarization moves cells *within* the failing
set, not across the threshold. Combined-pass count drops by two.

### Why this happened (interpretation)

The 5-class one-hot R² spreads variance across 4 minority columns; the
binarized target concentrates all variance on the single White-vs-rest axis
that the encoder retains. The 5-class metric was therefore **under-estimating
linear leakage on the dominant axis**, not over-estimating it. Binarization
gives a more honest read.

This is the opposite of the metric-artifact hypothesis: the income/race
encoder has not retained a generic "race-direction" that gets randomly split
across one-hot columns; it has retained a specific White-vs-rest direction
that the multinomial R² partially obscures.

### Recommended next step

Commit to Fix 1 (joint multivariate LEACE on race + the other disallowed
attributes for the same purpose), since this is a representational issue, not
a metric artifact. The income_prediction encoder retains a strong
White-vs-rest direction even under the proxy-Lagrangian R²<0.05 constraint —
the constraint is satisfied in expectation across the multinomial target but
not on the dominant binary axis. A LEACE-style projection on the binarized
target (or a joint LEACE across all disallowed attributes per purpose) is the
direct fix; lambda tuning alone will not move the encoder off this axis.
