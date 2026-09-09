# Fixed guarded-method utility matching

The complete comparison contains 9,216 per-seed rows and 3,072 fixed-pair aggregates: both fixed guarded local comparators, all four utility panels, four δ values, both attributes, three audit scopes, both budgets, both weightings and validation/development separately. [UTILITY_MATCHES.csv.gz](UTILITY_MATCHES.csv.gz) retains every task-specific exclusion; [UTILITY_MATCHES_AGGREGATE.csv.gz](UTILITY_MATCHES_AGGREGATE.csv.gz) includes every seed and all-three means/SD without filtering by eligibility.

Tables below use development AB SEX and expanded catch-up at 360 epochs. Each entry remains a fixed method pair. Validation and other audit scopes/budgets remain separate in the complete exports. Downstream validation is the independent probe-selection pool and is diagnostic, not an independent confirmation. It is distinct from the native source-validation pool.

## F; unweighted

| Panel | δ | Local | Close eligible | Close lower SEX | Directional eligible | Directional lower SEX |
| --- | --- | --- | --- | --- | --- | --- |
| source_only | 0 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_only | 0 | G_L20 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_only | 0.0005 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_only | 0.0005 | G_L20 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_only | 0.001 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_only | 0.001 | G_L20 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_only | 0.002 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_only | 0.002 | G_L20 | 1/3 | 1/3 | 1/3 | 1/3 |
| residential_transfer | 0 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| residential_transfer | 0 | G_L20 | 0/3 | 0/3 | 0/3 | 0/3 |
| residential_transfer | 0.0005 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| residential_transfer | 0.0005 | G_L20 | 0/3 | 0/3 | 0/3 | 0/3 |
| residential_transfer | 0.001 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| residential_transfer | 0.001 | G_L20 | 0/3 | 0/3 | 0/3 | 0/3 |
| residential_transfer | 0.002 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| residential_transfer | 0.002 | G_L20 | 0/3 | 0/3 | 1/3 | 1/3 |
| source_commute | 0 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_commute | 0 | G_L20 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_commute | 0.0005 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_commute | 0.0005 | G_L20 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_commute | 0.001 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_commute | 0.001 | G_L20 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_commute | 0.002 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_commute | 0.002 | G_L20 | 0/3 | 0/3 | 0/3 | 0/3 |
| full_authorized | 0 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| full_authorized | 0 | G_L20 | 0/3 | 0/3 | 0/3 | 0/3 |
| full_authorized | 0.0005 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| full_authorized | 0.0005 | G_L20 | 0/3 | 0/3 | 0/3 | 0/3 |
| full_authorized | 0.001 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| full_authorized | 0.001 | G_L20 | 0/3 | 0/3 | 0/3 | 0/3 |
| full_authorized | 0.002 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| full_authorized | 0.002 | G_L20 | 0/3 | 0/3 | 0/3 | 0/3 |

## F; person_weighted

| Panel | δ | Local | Close eligible | Close lower SEX | Directional eligible | Directional lower SEX |
| --- | --- | --- | --- | --- | --- | --- |
| source_only | 0 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_only | 0 | G_L20 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_only | 0.0005 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_only | 0.0005 | G_L20 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_only | 0.001 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_only | 0.001 | G_L20 | 0/3 | 0/3 | 1/3 | 1/3 |
| source_only | 0.002 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_only | 0.002 | G_L20 | 1/3 | 1/3 | 1/3 | 1/3 |
| residential_transfer | 0 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| residential_transfer | 0 | G_L20 | 0/3 | 0/3 | 0/3 | 0/3 |
| residential_transfer | 0.0005 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| residential_transfer | 0.0005 | G_L20 | 0/3 | 0/3 | 0/3 | 0/3 |
| residential_transfer | 0.001 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| residential_transfer | 0.001 | G_L20 | 0/3 | 0/3 | 0/3 | 0/3 |
| residential_transfer | 0.002 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| residential_transfer | 0.002 | G_L20 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_commute | 0 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_commute | 0 | G_L20 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_commute | 0.0005 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_commute | 0.0005 | G_L20 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_commute | 0.001 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_commute | 0.001 | G_L20 | 0/3 | 0/3 | 1/3 | 1/3 |
| source_commute | 0.002 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_commute | 0.002 | G_L20 | 1/3 | 1/3 | 1/3 | 1/3 |
| full_authorized | 0 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| full_authorized | 0 | G_L20 | 0/3 | 0/3 | 0/3 | 0/3 |
| full_authorized | 0.0005 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| full_authorized | 0.0005 | G_L20 | 0/3 | 0/3 | 0/3 | 0/3 |
| full_authorized | 0.001 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| full_authorized | 0.001 | G_L20 | 0/3 | 0/3 | 0/3 | 0/3 |
| full_authorized | 0.002 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| full_authorized | 0.002 | G_L20 | 0/3 | 0/3 | 0/3 | 0/3 |

## P; unweighted

| Panel | δ | Local | Close eligible | Close lower SEX | Directional eligible | Directional lower SEX |
| --- | --- | --- | --- | --- | --- | --- |
| source_only | 0 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_only | 0 | G_L20 | 0/3 | 0/3 | 1/3 | 0/3 |
| source_only | 0.0005 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_only | 0.0005 | G_L20 | 0/3 | 0/3 | 1/3 | 0/3 |
| source_only | 0.001 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_only | 0.001 | G_L20 | 0/3 | 0/3 | 1/3 | 0/3 |
| source_only | 0.002 | G_L025 | 1/3 | 1/3 | 1/3 | 1/3 |
| source_only | 0.002 | G_L20 | 1/3 | 0/3 | 1/3 | 0/3 |
| residential_transfer | 0 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| residential_transfer | 0 | G_L20 | 0/3 | 0/3 | 1/3 | 0/3 |
| residential_transfer | 0.0005 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| residential_transfer | 0.0005 | G_L20 | 0/3 | 0/3 | 1/3 | 0/3 |
| residential_transfer | 0.001 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| residential_transfer | 0.001 | G_L20 | 0/3 | 0/3 | 1/3 | 0/3 |
| residential_transfer | 0.002 | G_L025 | 1/3 | 1/3 | 1/3 | 1/3 |
| residential_transfer | 0.002 | G_L20 | 1/3 | 0/3 | 1/3 | 0/3 |
| source_commute | 0 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_commute | 0 | G_L20 | 0/3 | 0/3 | 1/3 | 0/3 |
| source_commute | 0.0005 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_commute | 0.0005 | G_L20 | 0/3 | 0/3 | 1/3 | 0/3 |
| source_commute | 0.001 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_commute | 0.001 | G_L20 | 0/3 | 0/3 | 1/3 | 0/3 |
| source_commute | 0.002 | G_L025 | 1/3 | 1/3 | 1/3 | 1/3 |
| source_commute | 0.002 | G_L20 | 1/3 | 0/3 | 1/3 | 0/3 |
| full_authorized | 0 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| full_authorized | 0 | G_L20 | 0/3 | 0/3 | 1/3 | 0/3 |
| full_authorized | 0.0005 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| full_authorized | 0.0005 | G_L20 | 0/3 | 0/3 | 1/3 | 0/3 |
| full_authorized | 0.001 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| full_authorized | 0.001 | G_L20 | 0/3 | 0/3 | 1/3 | 0/3 |
| full_authorized | 0.002 | G_L025 | 1/3 | 1/3 | 1/3 | 1/3 |
| full_authorized | 0.002 | G_L20 | 1/3 | 0/3 | 1/3 | 0/3 |

## P; person_weighted

| Panel | δ | Local | Close eligible | Close lower SEX | Directional eligible | Directional lower SEX |
| --- | --- | --- | --- | --- | --- | --- |
| source_only | 0 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_only | 0 | G_L20 | 0/3 | 0/3 | 1/3 | 0/3 |
| source_only | 0.0005 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_only | 0.0005 | G_L20 | 0/3 | 0/3 | 1/3 | 0/3 |
| source_only | 0.001 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_only | 0.001 | G_L20 | 0/3 | 0/3 | 1/3 | 0/3 |
| source_only | 0.002 | G_L025 | 1/3 | 1/3 | 1/3 | 1/3 |
| source_only | 0.002 | G_L20 | 1/3 | 0/3 | 1/3 | 0/3 |
| residential_transfer | 0 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| residential_transfer | 0 | G_L20 | 0/3 | 0/3 | 1/3 | 0/3 |
| residential_transfer | 0.0005 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| residential_transfer | 0.0005 | G_L20 | 0/3 | 0/3 | 1/3 | 0/3 |
| residential_transfer | 0.001 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| residential_transfer | 0.001 | G_L20 | 0/3 | 0/3 | 1/3 | 0/3 |
| residential_transfer | 0.002 | G_L025 | 1/3 | 1/3 | 1/3 | 1/3 |
| residential_transfer | 0.002 | G_L20 | 1/3 | 0/3 | 1/3 | 0/3 |
| source_commute | 0 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_commute | 0 | G_L20 | 0/3 | 0/3 | 1/3 | 0/3 |
| source_commute | 0.0005 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_commute | 0.0005 | G_L20 | 0/3 | 0/3 | 1/3 | 0/3 |
| source_commute | 0.001 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| source_commute | 0.001 | G_L20 | 0/3 | 0/3 | 1/3 | 0/3 |
| source_commute | 0.002 | G_L025 | 1/3 | 1/3 | 1/3 | 1/3 |
| source_commute | 0.002 | G_L20 | 1/3 | 0/3 | 1/3 | 0/3 |
| full_authorized | 0 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| full_authorized | 0 | G_L20 | 0/3 | 0/3 | 1/3 | 0/3 |
| full_authorized | 0.0005 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| full_authorized | 0.0005 | G_L20 | 0/3 | 0/3 | 1/3 | 0/3 |
| full_authorized | 0.001 | G_L025 | 0/3 | 0/3 | 0/3 | 0/3 |
| full_authorized | 0.001 | G_L20 | 0/3 | 0/3 | 1/3 | 0/3 |
| full_authorized | 0.002 | G_L025 | 1/3 | 1/3 | 1/3 | 1/3 |
| full_authorized | 0.002 | G_L20 | 1/3 | 0/3 | 1/3 | 0/3 |

## Primary fixed-pair exclusions, every seed

| Interface | Weighting | Local | Seed | Close | Directional | Close exclusions | Directional exclusions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F | unweighted | G_L025 | 0 | False | False | G_L025/source/public_coverage; utility/income_binary; utility/public_coverage; no_strict_SEX_gain_improvement | G_L025/source/public_coverage; utility/income_binary; no_strict_SEX_gain_improvement |
| F | person_weighted | G_L025 | 0 | False | False | utility/income_binary; utility/civilian_at_work; utility/public_coverage; no_strict_SEX_gain_improvement | utility/income_binary; no_strict_SEX_gain_improvement |
| F | unweighted | G_L20 | 0 | False | False | utility/income_binary; utility/public_coverage; utility/same_residence | utility/public_coverage |
| F | person_weighted | G_L20 | 0 | False | False | utility/income_binary; utility/public_coverage; utility/same_residence | utility/public_coverage |
| P | unweighted | G_L025 | 0 | False | False | utility/public_coverage | utility/public_coverage |
| P | person_weighted | G_L025 | 0 | False | False | utility/income_binary; utility/public_coverage | utility/income_binary; utility/public_coverage |
| P | unweighted | G_L20 | 0 | False | False | utility/income_binary; no_strict_SEX_gain_improvement | no_strict_SEX_gain_improvement |
| P | person_weighted | G_L20 | 0 | False | False | utility/income_binary; no_strict_SEX_gain_improvement | no_strict_SEX_gain_improvement |
| F | unweighted | G_L025 | 1 | False | False | G_J/source/public_coverage; G_L025/source/public_coverage; utility/public_coverage; utility/same_residence | G_J/source/public_coverage; G_L025/source/public_coverage; utility/public_coverage; utility/same_residence |
| F | person_weighted | G_L025 | 1 | False | False | G_J/source/public_coverage; G_L025/source/public_coverage; utility/income_binary; utility/public_coverage; utility/same_residence | G_J/source/public_coverage; G_L025/source/public_coverage; utility/public_coverage; utility/same_residence |
| F | unweighted | G_L20 | 1 | False | False | G_J/source/public_coverage; utility/civilian_at_work; utility/public_coverage; utility/same_residence | G_J/source/public_coverage; utility/civilian_at_work; utility/public_coverage; utility/same_residence |
| F | person_weighted | G_L20 | 1 | False | False | G_J/source/public_coverage; G_L20/source/public_coverage; utility/income_binary; utility/civilian_at_work; utility/public_coverage; utility/same_residence | G_J/source/public_coverage; G_L20/source/public_coverage; utility/civilian_at_work; utility/public_coverage; utility/same_residence |
| P | unweighted | G_L025 | 1 | False | False | G_J/source/public_coverage; G_L025/source/public_coverage; utility/civilian_at_work | G_J/source/public_coverage; G_L025/source/public_coverage; utility/civilian_at_work |
| P | person_weighted | G_L025 | 1 | False | False | G_J/source/public_coverage; G_L025/source/public_coverage; utility/income_binary; no_strict_SEX_gain_improvement | G_J/source/public_coverage; G_L025/source/public_coverage; utility/income_binary; no_strict_SEX_gain_improvement |
| P | unweighted | G_L20 | 1 | False | False | G_J/source/public_coverage; G_L20/source/public_coverage; utility/income_binary; utility/same_residence | G_J/source/public_coverage; G_L20/source/public_coverage |
| P | person_weighted | G_L20 | 1 | False | False | G_J/source/public_coverage; G_L20/source/public_coverage; utility/income_binary; no_strict_SEX_gain_improvement | G_J/source/public_coverage; G_L20/source/public_coverage; no_strict_SEX_gain_improvement |
| F | unweighted | G_L025 | 2 | False | False | utility/income_binary; utility/civilian_at_work; utility/same_residence; no_strict_SEX_gain_improvement | utility/income_binary; utility/civilian_at_work; no_strict_SEX_gain_improvement |
| F | person_weighted | G_L025 | 2 | False | False | utility/income_binary; utility/civilian_at_work; utility/public_coverage; utility/same_residence; no_strict_SEX_gain_improvement | utility/income_binary; utility/civilian_at_work; utility/public_coverage; no_strict_SEX_gain_improvement |
| F | unweighted | G_L20 | 2 | False | False | G_L20/source/public_coverage; utility/civilian_at_work; utility/public_coverage | G_L20/source/public_coverage |
| F | person_weighted | G_L20 | 2 | False | False | utility/civilian_at_work; utility/same_residence | utility/same_residence |
| P | unweighted | G_L025 | 2 | False | False | G_J/source/public_coverage; G_L025/source/public_coverage | G_J/source/public_coverage; G_L025/source/public_coverage |
| P | person_weighted | G_L025 | 2 | False | False | G_J/source/income_binary; G_L025/source/income_binary | G_J/source/income_binary; G_L025/source/income_binary |
| P | unweighted | G_L20 | 2 | False | False | G_J/source/public_coverage; G_L20/source/public_coverage; utility/income_binary; utility/public_coverage; no_strict_SEX_gain_improvement | G_J/source/public_coverage; G_L20/source/public_coverage; utility/income_binary; no_strict_SEX_gain_improvement |
| P | person_weighted | G_L20 | 2 | False | False | G_J/source/income_binary; utility/income_binary; no_strict_SEX_gain_improvement | G_J/source/income_binary; utility/income_binary; no_strict_SEX_gain_improvement |

[Full vector tradeoffs](VECTOR_TRADEOFF.csv) keep all five authorized utility losses and eleven forbidden recovery gains. [Pareto evidence](NONDOMINATED_POINTS.csv.gz) uses ordinary componentwise dominance with 1e−12 roundoff, never δ; per-seed and three-seed-mean results are separate and missing components remain unassessable. A numerical race vector is not a full-support certificate.
