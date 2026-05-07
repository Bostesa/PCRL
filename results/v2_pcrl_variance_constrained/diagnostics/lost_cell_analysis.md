# Lost-compliance cell analysis

## adult s0 — income_prediction / sex

- **R²(h, A)**: baseline 0.0475 → post 0.0800 (Δ +0.0324, crossed τ=0.05 from below)
- **purpose eff_rank**: 6.33 → 9.01 (Δ +2.68)
- **purpose per_dim_std_min**: 0.142 → 0.189
- **purpose per_dim_std_mean**: 0.319 → 0.318
- **task acc in this cell**: {'income': 0.801926, 'occupation_group': 0.999535, 'education_level': 0.999934}

### Other (purpose, attribute) pairs in the same purpose (for context)
  - race: R²=0.0358 (passes)


## Cross-cell comparison: was the lost cell's eff_rank delta the largest in its seed?

In adult s0, sorted eff_rank deltas:
  1. income_prediction: Δ eff_rank = +2.68 ← LOST
  2. employment_analysis: Δ eff_rank = +1.11
  3. education_assessment: Δ eff_rank = -0.01

**Finding: the lost cell (adult s0 income_prediction/sex) corresponds to the purpose with the LARGEST eff_rank improvement in its seed.** Consistent with the 'trading collapse-compliance for actual leakage' hypothesis — the constraint pushed the representation to use more dimensions, and one of those new dimensions encoded enough of attribute `sex` to push R² above τ.
