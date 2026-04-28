# Round 4 — final.pt vs best.pt re-evaluation

**Verdict: GREEN**  (mean final R² = 0.038; 20/24 pair-seeds < 0.05; 22/24 < 0.10)

final.pt R² < 0.05 on 20/24 pair-seeds (threshold: 15). Cotter selection is the bug; the optimizer worked. Recommendation: change selection rule and re-evaluate from existing checkpoints.

## Per-pair linear R² (auditor)

Cell: `<adj_pass tick><R²>`. ✓ = passes paper criterion (R² < 0.05 AND Δ < 0.02 on post-hoc MLP).

| Pair | s0 best | s0 final | s1 best | s1 final | s2 best | s2 final |
|------|---------|----------|---------|----------|---------|----------|
| income_prediction/race |  0.299 |  0.058 |  0.231 |  0.200 |  0.242 |  0.362 |
| income_prediction/sex |  0.245 | ✓0.035 |  0.254 |  0.034 |  0.242 |  0.010 |
| employment_analysis/race |  0.122 | ✓0.013 |  0.123 | ✓0.009 |  0.138 | ✓0.006 |
| employment_analysis/age_group |  0.088 |  0.009 |  0.103 |  0.006 |  0.105 |  0.005 |
| employment_analysis/marital_status |  0.157 |  0.019 |  0.190 |  0.003 |  0.098 |  0.006 |
| education_assessment/sex |  0.159 |  0.000 |  0.151 |  0.078 |  0.169 |  0.001 |
| education_assessment/race |  0.203 | ✓0.000 |  0.172 | ✓0.020 |  0.173 | ✓0.001 |
| education_assessment/income |  0.100 | ✓0.000 |  0.148 | ✓0.035 |  0.143 | ✓0.014 |

## Per-seed pass counts and mean R²

| Seed | best.pt pass | best.pt mean R² | final.pt pass | final.pt mean R² |
|------|--------------|------------------|----------------|-------------------|
| 0 | 0/8 | 0.1716 | 4/8 | 0.0167 |
| 1 | 0/8 | 0.1717 | 3/8 | 0.0479 |
| 2 | 0/8 | 0.1638 | 3/8 | 0.0505 |

## Aggregate

- best.pt mean R² across 24 pair-seeds: 0.1690
- final.pt mean R² across 24 pair-seeds: 0.0384
- final.pt R² < 0.05: 20/24 pair-seeds
- final.pt R² < 0.10: 22/24 pair-seeds

