# eff_rank pre vs post (40 originally-collapse-compliant pair-seeds in 7 retrained cells)

## Headline

The variance constraint achieved **partial structural improvement on effective rank** even though it did not reach the σ_d ≥ 0.5 floor. Across 40 pair-seeds: 21 (52%) showed eff_rank delta > 0; mean delta = +0.28. The number of pair-seeds with eff_rank ≥ 2.0 went from 25/40 (baseline) to 30/40 (post-retrain) — a structural improvement of 5 pair-seeds clearing the audit's eff_rank threshold.


In contrast, the **per_dim_std_min** trace is essentially flat: only 21/40 pair-seeds saw any improvement, and 0 of 40 reached the 0.5 floor. The variance Lagrangian's gradient cannot push the worst dim past ~0.3 because the mean-of-clamped-slack formulation discounts the worst-dim signal as soon as any dim crosses the floor.


**Interpretation.** The constraint architecture is doing *some* useful work — the soft eff_rank sigmoid penalty meaningfully widens the rank, and the variance dual narrows the worst-dim gap. But the chosen mean-of-clamped-slack aggregation can't close the worst-dim gap entirely. A per-dimension Lagrangian (one dual per of the 64 representation dimensions, totaling 64 × n_purposes additional duals) would directly address this.


## Per-pair-seed table (sorted by Δ eff_rank, descending)

| dataset | seed | purpose | attribute | baseline eff_rank | post eff_rank | Δ eff_rank | baseline σ_min | post σ_min | post R² | now clean? |
|---|---|---|---|---|---|---|---|---|---|---|
| adult | 0 | income_prediction | race | 6.33 | 9.01 | +2.68 | 0.142 | 0.189 | 0.0358 | N |
| adult | 0 | income_prediction | sex | 6.33 | 9.01 | +2.68 | 0.142 | 0.189 | 0.0800 | N |
| hmda | 2 | fair_lending_audit | race | 1.45 | 3.77 | +2.31 | 0.169 | 0.270 | 0.0091 | N |
| hmda | 2 | fair_lending_audit | sex | 1.45 | 3.77 | +2.31 | 0.169 | 0.270 | 0.0124 | N |
| hmda | 1 | fair_lending_audit | race | 1.41 | 2.47 | +1.05 | 0.175 | 0.149 | 0.0124 | N |
| hmda | 1 | fair_lending_audit | sex | 1.41 | 2.47 | +1.05 | 0.175 | 0.149 | 0.0095 | N |
| adult | 1 | education_assessment | sex | 1.88 | 2.93 | +1.05 | 0.376 | 0.164 | 0.0085 | N |
| adult | 1 | education_assessment | race | 1.88 | 2.93 | +1.05 | 0.376 | 0.164 | 0.0094 | N |
| adult | 1 | education_assessment | income | 1.88 | 2.93 | +1.05 | 0.376 | 0.164 | 0.0043 | N |
| diabetes | 0 | billing_audit | race | 3.28 | 4.33 | +1.05 | 0.093 | 0.070 | 0.0120 | N |
| diabetes | 0 | billing_audit | gender | 3.28 | 4.33 | +1.05 | 0.093 | 0.070 | 0.0032 | N |
| hmda | 0 | underwriting | race | 1.96 | 2.83 | +0.87 | 0.098 | 0.143 | 0.0089 | N |
| hmda | 0 | underwriting | ethnicity | 1.96 | 2.83 | +0.87 | 0.098 | 0.143 | 0.0054 | N |
| hmda | 1 | underwriting | race | 4.20 | 4.55 | +0.35 | 0.148 | 0.155 | 0.0123 | N |
| hmda | 1 | underwriting | ethnicity | 4.20 | 4.55 | +0.35 | 0.148 | 0.155 | 0.0089 | N |
| adult | 1 | employment_analysis | race | 2.10 | 2.31 | +0.21 | 0.090 | 0.062 | 0.0203 | N |
| adult | 1 | employment_analysis | age_group | 2.10 | 2.31 | +0.21 | 0.090 | 0.062 | 0.0135 | N |
| adult | 1 | employment_analysis | marital_status | 2.10 | 2.31 | +0.21 | 0.090 | 0.062 | 0.0064 | N |
| adult | 2 | education_assessment | sex | 1.10 | 1.27 | +0.17 | 0.061 | 0.062 | 0.0150 | N |
| adult | 2 | education_assessment | race | 1.10 | 1.27 | +0.17 | 0.061 | 0.062 | 0.0050 | N |
| adult | 2 | education_assessment | income | 1.10 | 1.27 | +0.17 | 0.061 | 0.062 | 0.0162 | N |
| hmda | 0 | fair_lending_audit | race | 3.16 | 3.16 | -0.00 | 0.141 | 0.206 | 0.0305 | N |
| hmda | 0 | fair_lending_audit | sex | 3.16 | 3.16 | -0.00 | 0.141 | 0.206 | 0.0153 | N |
| adult | 0 | education_assessment | sex | 1.17 | 1.16 | -0.01 | 0.035 | 0.060 | 0.0079 | N |
| adult | 0 | education_assessment | race | 1.17 | 1.16 | -0.01 | 0.035 | 0.060 | 0.0055 | N |
| adult | 0 | education_assessment | income | 1.17 | 1.16 | -0.01 | 0.035 | 0.060 | 0.0298 | N |
| adult | 2 | employment_analysis | race | 3.21 | 3.15 | -0.06 | 0.112 | 0.070 | 0.0052 | N |
| adult | 2 | employment_analysis | age_group | 3.21 | 3.15 | -0.06 | 0.112 | 0.070 | 0.0047 | N |
| adult | 2 | employment_analysis | marital_status | 3.21 | 3.15 | -0.06 | 0.112 | 0.070 | 0.0019 | N |
| hmda | 2 | pricing_analysis | race | 3.74 | 3.59 | -0.15 | 0.231 | 0.243 | 0.0014 | N |
| hmda | 2 | pricing_analysis | sex | 3.74 | 3.59 | -0.15 | 0.231 | 0.243 | 0.0013 | N |
| adult | 1 | income_prediction | race | 2.06 | 1.83 | -0.23 | 0.079 | 0.069 | 0.0079 | N |
| adult | 1 | income_prediction | sex | 2.06 | 1.83 | -0.23 | 0.079 | 0.069 | 0.0344 | N |
| adult | 2 | income_prediction | race | 5.20 | 4.75 | -0.45 | 0.125 | 0.162 | 0.0146 | N |
| diabetes | 0 | quality_research | race | 2.08 | 1.28 | -0.79 | 0.097 | 0.131 | 0.0015 | N |
| diabetes | 0 | quality_research | age_bucket | 2.08 | 1.28 | -0.79 | 0.097 | 0.131 | 0.0050 | N |
| hmda | 1 | pricing_analysis | race | 4.35 | 3.14 | -1.21 | 0.259 | 0.187 | 0.0138 | N |
| hmda | 1 | pricing_analysis | sex | 4.35 | 3.14 | -1.21 | 0.259 | 0.187 | 0.0036 | N |
| diabetes | 0 | clinical_decision_support | race | 4.69 | 2.49 | -2.19 | 0.213 | 0.090 | 0.0074 | N |
| diabetes | 0 | clinical_decision_support | gender | 4.69 | 2.49 | -2.19 | 0.213 | 0.090 | 0.0030 | N |