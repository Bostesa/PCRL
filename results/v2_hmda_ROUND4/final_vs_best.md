# HMDA Round 4 — final.pt vs best.pt re-evaluation

**Verdict: YELLOW**  (mean final R² = 0.045; 11/18 pair-seeds < 0.05; 18/18 < 0.10)

mean final.pt R² = 0.045 ∈ [0.05, 0.10). 11/18 pair-seeds < 0.05. Recommendation: honest SOTA at realigned R²<0.10 threshold.

## Per-pair linear R² (auditor)

Cell: `<adj_pass tick><R²>`. ✓ = passes paper criterion (R² < 0.05 AND Δ < 0.02 on post-hoc MLP).

| Pair | s0 best | s0 final | s1 best | s1 final | s2 best | s2 final |
|------|---------|----------|---------|----------|---------|----------|
| underwriting/race | ✓0.002 |  0.044 |  0.066 |  0.090 |  0.005 |  0.081 |
| underwriting/ethnicity | ✓0.000 |  0.013 |  0.063 |  0.086 |  0.001 |  0.028 |
| pricing_analysis/race |  0.004 |  0.037 |  0.006 |  0.055 |  0.011 |  0.071 |
| pricing_analysis/sex |  0.002 |  0.004 |  0.016 |  0.011 |  0.024 |  0.027 |
| fair_lending_audit/race | ✓0.007 | ✓0.038 | ✓0.024 | ✓0.026 | ✓0.015 |  0.041 |
| fair_lending_audit/sex | ✓0.011 |  0.061 | ✓0.041 | ✓0.042 | ✓0.022 |  0.065 |

## Per-seed pass counts and mean R²

| Seed | best.pt pass | best.pt mean R² | final.pt pass | final.pt mean R² |
|------|--------------|------------------|----------------|-------------------|
| 0 | 4/6 | 0.0043 | 1/6 | 0.0327 |
| 1 | 2/6 | 0.0359 | 2/6 | 0.0516 |
| 2 | 2/6 | 0.0131 | 0/6 | 0.0521 |

## Aggregate

- best.pt mean R² across 18 pair-seeds: 0.0178
- final.pt mean R² across 18 pair-seeds: 0.0455
- final.pt R² < 0.05: 11/18
- final.pt R² < 0.10: 18/18

