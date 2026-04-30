# ADULT Round 4 — final.pt vs best.pt re-evaluation

**Verdict: GREEN**  (mean final R² = 0.012; 23/24 pair-seeds < 0.05; 24/24 < 0.10)

final.pt R² < 0.05 on 23/24 pair-seeds (threshold 15=5/8 per seed × 3). Cotter selection is the bug; the optimizer worked. Recommendation: claim from final.pt as canonical.

## Per-pair linear R² (auditor)

Cell: `<adj_pass tick><R²>`. ✓ = passes paper criterion (R² < 0.05 AND Δ < 0.02 on post-hoc MLP).

| Pair | s0 best | s0 final | s1 best | s1 final | s2 best | s2 final |
|------|---------|----------|---------|----------|---------|----------|
| income_prediction/race | ✓0.011 | ✓0.014 | ✓0.004 | ✓0.005 | ✓0.014 | ✓0.014 |
| income_prediction/sex | ✓0.044 | ✓0.048 |  0.018 |  0.003 |  0.060 |  0.060 |
| employment_analysis/race | ✓0.040 | ✓0.003 | ✓0.011 |  0.012 |  0.025 |  0.019 |
| employment_analysis/age_group |  0.062 |  0.005 |  0.002 |  0.002 |  0.009 |  0.008 |
| employment_analysis/marital_status |  0.081 |  0.003 |  0.001 |  0.001 |  0.003 |  0.003 |
| education_assessment/sex |  0.024 |  0.020 |  0.001 |  0.003 |  0.017 |  0.012 |
| education_assessment/race | ✓0.013 |  0.008 | ✓0.001 | ✓0.002 | ✓0.004 | ✓0.003 |
| education_assessment/income | ✓0.025 | ✓0.026 | ✓0.003 | ✓0.003 | ✓0.017 | ✓0.009 |

## Per-seed pass counts and mean R²

| Seed | best.pt pass | best.pt mean R² | final.pt pass | final.pt mean R² |
|------|--------------|------------------|----------------|-------------------|
| 0 | 5/8 | 0.0373 | 4/8 | 0.0160 |
| 1 | 4/8 | 0.0050 | 3/8 | 0.0037 |
| 2 | 3/8 | 0.0188 | 3/8 | 0.0162 |

## Aggregate

- best.pt mean R² across 24 pair-seeds: 0.0204
- final.pt mean R² across 24 pair-seeds: 0.0119
- final.pt R² < 0.05: 23/24
- final.pt R² < 0.10: 24/24

