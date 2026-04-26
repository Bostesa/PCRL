# HMDA 2023 California — Purpose Specifications

Three purposes, six (purpose, sensitive-attribute) compliance pairs.
All sensitive attributes are derived per the HMDA 2023 schema.

| Purpose              | Allowed task          | Disallowed attributes  | Why                                                          |
|----------------------|-----------------------|------------------------|--------------------------------------------------------------|
| underwriting         | loan_decision         | race, ethnicity        | ECOA prohibits race and national origin from credit decisions. |
| pricing_analysis     | loan_amount_band      | race, sex              | Disparate-pricing analysis must not depend on protected class. |
| fair_lending_audit   | tract_denial_high     | race, sex              | Tract-level audit needs aggregate signal, not individual demographics. |

Filters applied to the public LAR snapshot:

- `state_code = CA`, `activity_year = 2023`
- `loan_purpose = 1` (home purchase)
- `lien_status = 1` (first lien)
- `construction_method = 1` (site-built)
- `action_taken in {1, 3}` (originated or denied)
- known race / ethnicity / sex / age / debt-to-income (drops `Joint`,
  `Sex Not Available`, `Race Not Available`, `8888`, `9999`, `Exempt`, `NA`)

Tasks:

- **loan_decision**: 1 if `action_taken = 1`, 0 if `action_taken = 3`.
- **loan_amount_band**: 5-class quintile of `loan_amount` fitted on
  training data only (cutoffs persisted in `metadata.json`).
- **tract_denial_high**: 1 if the applicant's `census_tract` has a denial
  rate above the median per-tract denial rate (median computed over
  filtered records before splitting).

Sensitive attribute encodings:

- **race** (5): 0=White, 1=Black, 2=Asian, 3=AIAN/NHPI/2+, 4=Joint.
- **ethnicity** (2): 0=Not Hispanic, 1=Hispanic.
- **sex** (2): 0=Female, 1=Male.
