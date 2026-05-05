# DIABETES Round 4 — final.pt vs best.pt re-evaluation

**Verdict: GREEN**  (mean final R² = 0.025; 15/18 pair-seeds < 0.05; 16/18 < 0.10)

final.pt R² < 0.05 on 15/18 pair-seeds (threshold 12=4/6 per seed × 3). Cotter selection is the bug; the optimizer worked. Recommendation: claim from final.pt as canonical.

## Per-pair linear R² (auditor)

Cell: `<adj_pass tick><R²>`. ✓ = passes paper criterion (R² < 0.05 AND Δ < 0.02 on post-hoc MLP).

| Pair | s0 best | s0 final | s1 best | s1 final | s2 best | s2 final |
|------|---------|----------|---------|----------|---------|----------|
| billing_audit/race | ✓0.014 | ✓0.012 | ✓0.000 | ✓0.010 | ✓0.002 | ✓0.016 |
| billing_audit/gender | ✓0.008 | ✓0.015 | ✓0.000 | ✓0.014 | ✓0.000 | ✓0.001 |
| quality_research/race | ✓0.011 | ✓0.015 | ✓0.007 | ✓0.017 | ✓0.007 | ✓0.011 |
| quality_research/age_bucket |  0.066 |  0.107 |  0.023 |  0.122 | ✓0.023 |  0.082 |
| clinical_decision_support/race | ✓0.015 | ✓0.011 | ✓0.010 | ✓0.007 | ✓0.017 | ✓0.005 |
| clinical_decision_support/gender | ✓0.003 | ✓0.001 | ✓0.005 | ✓0.008 | ✓0.004 | ✓0.000 |

## Per-seed pass counts and mean R²

| Seed | best.pt pass | best.pt mean R² | final.pt pass | final.pt mean R² |
|------|--------------|------------------|----------------|-------------------|
| 0 | 5/6 | 0.0194 | 5/6 | 0.0267 |
| 1 | 5/6 | 0.0074 | 5/6 | 0.0298 |
| 2 | 6/6 | 0.0087 | 5/6 | 0.0194 |

## Aggregate

- best.pt mean R² across 18 pair-seeds: 0.0118
- final.pt mean R² across 18 pair-seeds: 0.0253
- final.pt R² < 0.05: 15/18
- final.pt R² < 0.10: 16/18

