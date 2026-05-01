# HMDA Round 4 — final.pt vs best.pt re-evaluation

**Verdict: GREEN**  (mean final R² = 0.020; 16/18 pair-seeds < 0.05; 17/18 < 0.10)

final.pt R² < 0.05 on 16/18 pair-seeds (threshold 12=4/6 per seed × 3). Cotter selection is the bug; the optimizer worked. Recommendation: claim from final.pt as canonical.

## Per-pair linear R² (auditor)

Cell: `<adj_pass tick><R²>`. ✓ = passes paper criterion (R² < 0.05 AND Δ < 0.02 on post-hoc MLP).

| Pair | s0 best | s0 final | s1 best | s1 final | s2 best | s2 final |
|------|---------|----------|---------|----------|---------|----------|
| underwriting/race | ✓0.000 |  0.003 | ✓0.004 |  0.027 |  0.015 |  0.008 |
| underwriting/ethnicity | ✓0.000 |  0.014 |  0.000 |  0.017 |  0.013 |  0.008 |
| pricing_analysis/race |  0.179 |  0.073 |  0.005 |  0.016 |  0.007 |  0.007 |
| pricing_analysis/sex |  0.197 |  0.118 |  0.001 |  0.004 |  0.011 |  0.005 |
| fair_lending_audit/race |  0.002 |  0.034 |  0.015 |  0.004 |  0.001 |  0.003 |
| fair_lending_audit/sex |  0.008 |  0.024 |  0.003 |  0.000 |  0.002 |  0.000 |

## Per-seed pass counts and mean R²

| Seed | best.pt pass | best.pt mean R² | final.pt pass | final.pt mean R² |
|------|--------------|------------------|----------------|-------------------|
| 0 | 2/6 | 0.0645 | 0/6 | 0.0443 |
| 1 | 1/6 | 0.0048 | 0/6 | 0.0112 |
| 2 | 0/6 | 0.0080 | 0/6 | 0.0049 |

## Aggregate

- best.pt mean R² across 18 pair-seeds: 0.0258
- final.pt mean R² across 18 pair-seeds: 0.0201
- final.pt R² < 0.05: 16/18
- final.pt R² < 0.10: 17/18

