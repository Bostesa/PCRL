# Reused direct teacher and audit controls

These are matching historical controls, reused without new fitting. The direct E teacher retains its original genuine nested 120/360-epoch audits and original utility heads. Priors have no fitting-epoch budget; repeating their score beside both audit budgets is an alias. Exposed-label controls retain the selected checkpoint from their declared original budget. Original PCA32 and other named references keep their actual historical audit budgets in [CONTEXTUAL_REFERENCES.md](CONTEXTUAL_REFERENCES.md).

All cells are mean ± descriptive sample SD across the same three cohort-sharing seeds. PWGTP uses exactly the predictors selected by unweighted validation. These are development-evaluation inputs previously used in method development.

## Direct E utility and coalition recovery

| Weighting | income_binary | civilian_at_work | public_coverage | same_residence | commute_over20 |
| --- | --- | --- | --- | --- | --- |
| unweighted | 0.35666 ± 0.01755 | 0.41451 ± 0.05529 | 0.52916 ± 0.02563 | 0.50838 ± 0.00204 | 0.68724 ± 0.00406 |
| person_weighted | 0.36938 ± 0.01938 | 0.39921 ± 0.05715 | 0.53528 ± 0.02223 | 0.48616 ± 0.00584 | 0.68969 ± 0.00264 |

| Weighting | Actual nested budget | Scope | AB SEX signed gain | AB race signed gain |
| --- | --- | --- | --- | --- |
| unweighted | 120 | standard_independent | 0.00820 ± 0.00419 | 0.05271 ± 0.00259 |
| unweighted | 120 | expanded_independent | 0.00820 ± 0.00419 | 0.05271 ± 0.00259 |
| unweighted | 120 | expanded_catchup | 0.00820 ± 0.00419 | 0.05271 ± 0.00259 |
| unweighted | 360 | standard_independent | 0.00820 ± 0.00419 | 0.05271 ± 0.00259 |
| unweighted | 360 | expanded_independent | 0.00820 ± 0.00419 | 0.05271 ± 0.00259 |
| unweighted | 360 | expanded_catchup | 0.00820 ± 0.00419 | 0.05271 ± 0.00259 |
| person_weighted | 120 | standard_independent | 0.00449 ± 0.00353 | 0.04623 ± 0.00430 |
| person_weighted | 120 | expanded_independent | 0.00449 ± 0.00353 | 0.04623 ± 0.00430 |
| person_weighted | 120 | expanded_catchup | 0.00449 ± 0.00353 | 0.04623 ± 0.00430 |
| person_weighted | 360 | standard_independent | 0.00449 ± 0.00353 | 0.04623 ± 0.00430 |
| person_weighted | 360 | expanded_independent | 0.00449 ± 0.00353 | 0.04623 ± 0.00430 |
| person_weighted | 360 | expanded_catchup | 0.00449 ± 0.00353 | 0.04623 ± 0.00430 |

## Prior and exposed-target control losses

Lower exposed-label loss shows that the declared candidate/data recipe can recover an explicitly supplied target. A failure on a class remains a limitation; it is not evidence that a learned release protects that class. Full race assessment remains unassessable because code 4 is absent from independent fitting and validation. Inherited representation observer exposure is a separate route.

### Budget 120; unweighted

| Target | Validation prior | Validation exposed | Development prior | Development exposed |
| --- | --- | --- | --- | --- |
| SEX | 0.69239 ± 0.00041 | 0.00002 ± 0.00001 | 0.69269 ± 0.00057 | 0.00002 ± 0.00001 |
| RAC1P | 1.28350 ± 0.00596 | 0.00012 ± 0.00003 | 1.29004 ± 0.02560 | 0.00067 ± 0.00093 |
| income_binary | 0.47262 ± 0.01276 | 0.00003 ± 0.00002 | 0.48172 ± 0.00619 | 0.00003 ± 0.00002 |
| civilian_at_work | 0.63073 ± 0.00338 | 0.00003 ± 0.00002 | 0.63211 ± 0.00622 | 0.00003 ± 0.00002 |
| public_coverage | 0.54733 ± 0.01097 | 0.00004 ± 0.00003 | 0.56616 ± 0.01035 | 0.00004 ± 0.00003 |
| same_residence | 0.54344 ± 0.00847 | 0.00005 ± 0.00001 | 0.53654 ± 0.00646 | 0.00005 ± 0.00001 |
| commute_over20 | 0.69365 ± 0.00052 | 0.00010 ± 0.00004 | 0.69396 ± 0.00054 | 0.00010 ± 0.00004 |

### Budget 120; person_weighted

| Target | Validation prior | Validation exposed | Development prior | Development exposed |
| --- | --- | --- | --- | --- |
| SEX | 0.69233 ± 0.00024 | 0.00002 ± 0.00001 | 0.69211 ± 0.00077 | 0.00002 ± 0.00001 |
| RAC1P | 1.28470 ± 0.00448 | 0.00011 ± 0.00003 | 1.28724 ± 0.03910 | 0.00051 ± 0.00065 |
| income_binary | 0.47853 ± 0.01658 | 0.00003 ± 0.00002 | 0.48485 ± 0.00979 | 0.00003 ± 0.00002 |
| civilian_at_work | 0.61062 ± 0.00558 | 0.00003 ± 0.00003 | 0.61503 ± 0.00188 | 0.00003 ± 0.00003 |
| public_coverage | 0.54293 ± 0.01512 | 0.00004 ± 0.00003 | 0.57098 ± 0.01250 | 0.00004 ± 0.00003 |
| same_residence | 0.52034 ± 0.00632 | 0.00005 ± 0.00001 | 0.51099 ± 0.00364 | 0.00005 ± 0.00001 |
| commute_over20 | 0.69337 ± 0.00088 | 0.00010 ± 0.00003 | 0.69389 ± 0.00074 | 0.00010 ± 0.00004 |

### Budget 360; unweighted

| Target | Validation prior | Validation exposed | Development prior | Development exposed |
| --- | --- | --- | --- | --- |
| SEX | 0.69239 ± 0.00041 | 0.00000 ± 0.00000 | 0.69269 ± 0.00057 | 0.00000 ± 0.00000 |
| RAC1P | 1.28350 ± 0.00596 | 0.00001 ± 0.00000 | 1.29004 ± 0.02560 | 0.00062 ± 0.00107 |
| income_binary | 0.47262 ± 0.01276 | 0.00000 ± 0.00000 | 0.48172 ± 0.00619 | 0.00000 ± 0.00000 |
| civilian_at_work | 0.63073 ± 0.00338 | 0.00000 ± 0.00000 | 0.63211 ± 0.00622 | 0.00000 ± 0.00000 |
| public_coverage | 0.54733 ± 0.01097 | 0.00000 ± 0.00000 | 0.56616 ± 0.01035 | 0.00000 ± 0.00000 |
| same_residence | 0.54344 ± 0.00847 | 0.00000 ± 0.00000 | 0.53654 ± 0.00646 | 0.00000 ± 0.00000 |
| commute_over20 | 0.69365 ± 0.00052 | 0.00001 ± 0.00000 | 0.69396 ± 0.00054 | 0.00001 ± 0.00000 |

### Budget 360; person_weighted

| Target | Validation prior | Validation exposed | Development prior | Development exposed |
| --- | --- | --- | --- | --- |
| SEX | 0.69233 ± 0.00024 | 0.00000 ± 0.00000 | 0.69211 ± 0.00077 | 0.00000 ± 0.00000 |
| RAC1P | 1.28470 ± 0.00448 | 0.00001 ± 0.00000 | 1.28724 ± 0.03910 | 0.00044 ± 0.00075 |
| income_binary | 0.47853 ± 0.01658 | 0.00000 ± 0.00000 | 0.48485 ± 0.00979 | 0.00000 ± 0.00000 |
| civilian_at_work | 0.61062 ± 0.00558 | 0.00000 ± 0.00000 | 0.61503 ± 0.00188 | 0.00000 ± 0.00000 |
| public_coverage | 0.54293 ± 0.01512 | 0.00000 ± 0.00000 | 0.57098 ± 0.01250 | 0.00000 ± 0.00000 |
| same_residence | 0.52034 ± 0.00632 | 0.00000 ± 0.00000 | 0.51099 ± 0.00364 | 0.00000 ± 0.00000 |
| commute_over20 | 0.69337 ± 0.00088 | 0.00001 ± 0.00000 | 0.69389 ± 0.00074 | 0.00001 ± 0.00000 |

[All control rows and original predictor identities](CONTROL_RESULTS.csv) · [All direct E selected scores](PER_SEED.csv) · [All candidate and class evidence](PER_CANDIDATE.csv.gz) · [Category support](PER_CLASS.csv.gz)
