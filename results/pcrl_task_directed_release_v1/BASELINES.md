# Baseline coverage

Source: EVIDENCE.json; native accepted summaries: EVALUATION_GRID.json. Selection SHA-256: `061467a38e7e40721777fb0c25f7348eb0a510541c6a6525bb8eb4f9cb2bdf70`. Grid digest: `4259cdd23b36254f2d5b2e81dfd625ef66ec6ebacc91f5ca0346e77e8fa2b143`. Every complete configuration stores its three summary/receipt hashes.

Machine-rendered descriptive aggregates. Means require all three anchors; unweighted and PWGTP metrics use the same frozen predictors. Missing diagnostics are unavailable, never imputed. These are dependent analyses of the same inherited 2018 cohort; people are not globally unseen across anchors. The anchor-specific test seal does not imply globally untouched labels. Household bounds are conditional on fitted models; they do not include retraining or validation-selection uncertainty. See DATED_SPLIT_CLARIFICATION.md.

Availability is not an eligibility or superiority verdict. A complete scientifically ineligible family supplies no comparator and does not veto other eligible families. Missing mandatory settings block the required frontier. At least one eligible comparator family is required. Stricter all-family results are separate diagnostics.

| Family | Configuration | Evaluation status |
| --- | --- | --- |
| historical | J | complete |
| historical | leace_A0 | complete |
| historical | splince_A0 | complete |
| historical | optnet16_L1 | complete |
| historical | optnet16_L2 | complete |
| historical | optnet16_C1 | complete |
| continuous_task | continuous_task | complete |
| deterministic_actions | T0_U_unconstrained_a17 | complete |
| deterministic_actions | Ttask_U_unconstrained_a17 | complete |
| deterministic_actions | Trisk_U_unconstrained_a17 | complete |
| deterministic_actions | T0_U_unconstrained_a33 | complete |
| deterministic_actions | Trisk_U_unconstrained_a33 | complete |
| deterministic_actions | Ttask_U_unconstrained_a33 | complete |
| direct_code | T0_code | complete |
| direct_code | Ttask_code | complete |
| direct_code | Trisk_code | complete |
| withholding | T0_withhold_0.25 | complete |
| withholding | T0_withhold_0.5 | complete |
| withholding | T0_withhold_0.75 | complete |
| withholding | Ttask_withhold_0.25 | complete |
| withholding | Ttask_withhold_0.5 | complete |
| withholding | Ttask_withhold_0.75 | complete |
| withholding | Trisk_withhold_0.25 | complete |
| withholding | Trisk_withhold_0.5 | complete |
| withholding | Trisk_withhold_0.75 | complete |
| withholding | T0_withhold_0.25_a33 | complete |
| withholding | T0_withhold_0.5_a33 | complete |
| withholding | T0_withhold_0.75_a33 | complete |
| withholding | Trisk_withhold_0.25_a33 | complete |
| withholding | Trisk_withhold_0.5_a33 | complete |
| withholding | Trisk_withhold_0.75_a33 | complete |
| withholding | Ttask_withhold_0.25_a33 | complete |
| withholding | Ttask_withhold_0.5_a33 | complete |
| withholding | Ttask_withhold_0.75_a33 | complete |
| randomized_response | T0_rr_0.25 | complete |
| randomized_response | T0_rr_0.5 | complete |
| randomized_response | T0_rr_0.75 | complete |
| randomized_response | Ttask_rr_0.25 | complete |
| randomized_response | Ttask_rr_0.5 | complete |
| randomized_response | Ttask_rr_0.75 | complete |
| randomized_response | Trisk_rr_0.25 | complete |
| randomized_response | Trisk_rr_0.5 | complete |
| randomized_response | Trisk_rr_0.75 | complete |
| randomized_response | T0_rr_0.25_a33 | complete |
| randomized_response | T0_rr_0.5_a33 | complete |
| randomized_response | T0_rr_0.75_a33 | complete |
| randomized_response | Trisk_rr_0.25_a33 | complete |
| randomized_response | Trisk_rr_0.5_a33 | complete |
| randomized_response | Trisk_rr_0.75_a33 | complete |
| randomized_response | Ttask_rr_0.25_a33 | complete |
| randomized_response | Ttask_rr_0.5_a33 | complete |
| randomized_response | Ttask_rr_0.75_a33 | complete |
| supervised_LEACE | leace_supervised | complete |
| supervised_LEACE | leace_supervised_mechanism40 | complete |
| supervised_LEACE | leace_supervised_union88 | complete |
| supervised_SPLINCE | splince_supervised | complete |
| supervised_SPLINCE | splince_supervised_mechanism40 | complete |
| supervised_SPLINCE | splince_supervised_union88 | complete |
| constant_null | constant_best | complete |
| constant_null | independent_token | complete |
| constant_null | constant_best_a33 | complete |
| constant_null | independent_token_a33 | complete |

| Route | Family | Required frontier complete | Missing mandatory settings | Scientifically ineligible | Frozen nominee |
| --- | --- | --- | --- | --- | --- |
| protection_first | constant_null | True | none | True | none |
| protection_first | continuous_task | True | none | False | continuous_task |
| protection_first | deterministic_actions | True | none | False | T0_U_unconstrained_a33 |
| protection_first | direct_code | True | none | False | T0_code |
| protection_first | randomized_response | True | none | True | none |
| protection_first | supervised_LEACE | True | none | False | leace_supervised_mechanism40 |
| protection_first | supervised_SPLINCE | True | none | False | splince_supervised_mechanism40 |
| protection_first | withholding | True | none | False | T0_withhold_0.75_a33 |
| utility_first | constant_null | True | none | False | independent_token |
| utility_first | continuous_task | True | none | True | none |
| utility_first | deterministic_actions | True | none | False | T0_U_unconstrained_a33 |
| utility_first | direct_code | True | none | False | T0_code |
| utility_first | randomized_response | True | none | False | Ttask_rr_0.75 |
| utility_first | supervised_LEACE | True | none | True | none |
| utility_first | supervised_SPLINCE | True | none | True | none |
| utility_first | withholding | True | none | False | T0_withhold_0.75_a33 |
