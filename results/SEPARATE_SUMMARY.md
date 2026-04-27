# Separate-Encoders Threat Experiment — Summary

Trains |P| separate encoders (no FiLM, no purpose embedding) and compares to PCRL paper compliance + task accuracy.

| dataset | separate pass | PCRL paper pass | separate task | PCRL task | wall (s) | rep health |
|---------|---------------|-----------------|---------------|-----------|----------|------------|
| adult | 0/8 | 6/8 | 79.0% | 76.3% | 1814 | OK |
| diabetes | 5/6 | 5/6 | 31.3% | — | 3960 | COLLAPSED |
| hmda | 0/6 | 5.3/6 | 89.3% | — | 2317 | OK |

## Per-purpose detail

### adult
| purpose | pass | task acc | per_dim_std | l2 std | eff rank | per-attr Δ (R²) |
|---------|------|----------|-------------|--------|----------|-----------------|
| income_prediction | 0/2 | income=79.0% | 1.02 | 7.67 | 3.0 | race Δ+3.1% (R²0.06); sex Δ+17.0% (R²0.08) |
| employment_analysis | 0/3 | occupation_group=71.4% | 0.81 | 5.36 | 4.0 | race Δ+4.4% (R²0.07); age_group Δ+17.4% (R²0.07); marital_status Δ+33.4% (R²0.10) |
| education_assessment | 0/3 | education_level=79.3% | 0.66 | 4.07 | 5.6 | sex Δ+16.8% (R²0.09); race Δ+3.6% (R²0.06); income Δ+5.1% (R²0.12) |

### diabetes
| purpose | pass | task acc | per_dim_std | l2 std | eff rank | per-attr Δ (R²) |
|---------|------|----------|-------------|--------|----------|-----------------|
| billing_audit | 2/2 | primary_diagnosis_category=31.3% | 1.42 | 15.02 | 1.2 | race Δ+0.3% (R²0.02); gender Δ+1.3% (R²0.01) |
| quality_research | 1/2 | readmission_outcome=91.2% | 2.86 | 27.97 | 1.0 | race Δ+0.3% (R²0.02); age_bucket Δ+7.7% (R²0.10) |
| clinical_decision_support | 2/2 | medication_change_outcome=100.0% | 2.52 | 24.77 | 1.0 | race Δ+0.3% (R²0.00); gender Δ+0.8% (R²0.00) |

### hmda
| purpose | pass | task acc | per_dim_std | l2 std | eff rank | per-attr Δ (R²) |
|---------|------|----------|-------------|--------|----------|-----------------|
| underwriting | 0/2 | loan_decision=89.3% | 0.83 | 5.78 | 2.0 | race Δ+18.4% (R²0.14); ethnicity Δ+15.7% (R²0.17) |
| pricing_analysis | 0/2 | loan_amount_band=47.0% | 0.85 | 6.36 | 2.3 | race Δ+20.4% (R²0.13); sex Δ+20.5% (R²0.17) |
| fair_lending_audit | 0/2 | tract_denial_high=63.4% | 0.74 | 5.76 | 2.5 | race Δ+21.1% (R²0.15); sex Δ+20.5% (R²0.16) |

## Per-dataset verdict

- **adult** — Conditioning has real compliance value — separate encoders fail more disallowed-attribute audits.
- **diabetes** — Pass count is degenerate — encoder collapsed (effective rank < 1.5). Compliance 'passes' because the rep carries near-zero information about ANYTHING, not because conditioning was unnecessary.
- **hmda** — Conditioning has real compliance value — separate encoders fail more disallowed-attribute audits.

## Overall

- Total pass: separate=5/20 vs PCRL paper=16.3/20
- **Direction:** PCRL conditioning has measurably better compliance than separate encoders.
