# DIABETES Round 4 — final.pt vs best.pt re-evaluation

**Verdict: GREEN**  (mean final R² = 0.017; 16/18 pair-seeds < 0.05; 17/18 < 0.10)

final.pt R² < 0.05 on 16/18 pair-seeds (threshold 12=4/6 per seed × 3). Cotter selection is the bug; the optimizer worked. Recommendation: claim from final.pt as canonical.

## Per-pair linear R² (auditor)

Cell: `<adj_pass tick><R²>`. ✓ = passes paper criterion (R² < 0.05 AND Δ < 0.02 on post-hoc MLP).

| Pair | s0 best | s0 final | s1 best | s1 final | s2 best | s2 final |
|------|---------|----------|---------|----------|---------|----------|
| billing_audit/race | ✓0.007 | ✓0.010 | ✓0.020 | ✓0.009 | ✓0.007 | ✓0.009 |
| billing_audit/gender | ✓0.001 | ✓0.004 | ✓0.002 | ✓0.002 | ✓0.007 | ✓0.006 |
| quality_research/race | ✓0.002 | ✓0.004 | ✓0.007 | ✓0.015 | ✓0.007 | ✓0.012 |
| quality_research/age_bucket |  0.018 |  0.011 |  0.023 |  0.107 |  0.026 |  0.092 |
| clinical_decision_support/race | ✓0.007 | ✓0.010 | ✓0.012 | ✓0.006 | ✓0.008 | ✓0.009 |
| clinical_decision_support/gender | ✓0.001 | ✓0.002 | ✓0.003 | ✓0.002 | ✓0.002 | ✓0.001 |

## Per-seed pass counts and mean R²

| Seed | best.pt pass | best.pt mean R² | final.pt pass | final.pt mean R² |
|------|--------------|------------------|----------------|-------------------|
| 0 | 5/6 | 0.0061 | 5/6 | 0.0069 |
| 1 | 5/6 | 0.0112 | 5/6 | 0.0236 |
| 2 | 5/6 | 0.0096 | 5/6 | 0.0217 |

## Aggregate

- best.pt mean R² across 18 pair-seeds: 0.0090
- final.pt mean R² across 18 pair-seeds: 0.0174
- final.pt R² < 0.05: 16/18
- final.pt R² < 0.10: 17/18

