# Validation evidence

Everything below was produced by code that consumes the locked artifacts; no check re-fits a model or re-selects a candidate.

## 1. Numerical and software checks

| Suite | Result |
|---|---|
| Focused tests (`tests/test_acs_spectral_transport_eval.py`) | 8 passed: sealed final partition without a lock, tamper detection, dry-run guard, scope membership and selection rule, coalition inheritance columns, person-loss vs the scorer and the bootstrap ratio estimator (including that both rows of a household share a resample count), simultaneous-interval coverage and degenerate endpoints, withholding linearity |
| Inherited suites (`test_acs_residual_spectral.py`, `test_acs_spectral_audits.py`, `test_acs_spectral_transport.py`) | 22 passed, unchanged |
| 2018 pipeline identity | bitwise equal for every saved PCA array, anchor vector, bank, historical wire/derived release and spectral release, all 3 seeds and all 7 pools (57–59 arrays per pool); `seed_*/REPLAY_2018.json` |
| Score replay | 9660 selected losses recomputed from saved predictions and labels; max difference 4.44e-16 |
| Prediction replay | 59,874 stored final predictions reproduced exactly in independent processes; 0 mismatches (`PREDICTION_REPLAY.json`) |
| Validation-score audit | 37,122 Mode B validation predictions recomputed: 0 stored-score mismatches, **0 selection changes**, 0 utility mismatches (`VALIDATION_SCORE_AUDIT.json`) |

## 2. Independent verification (`scripts/verify_acs_spectral_transport.py`)

Separate code paths; the report's computation functions are not reused. All 15 checks pass.

| Check | Result | Detail |
|---|---|---|
| `household_disjointness` | pass | shared_households: 0; duplicate_person_keys: 0; rows: 79869; households: 53504 |
| `lock_inputs_unchanged_or_amended` | pass | files: 17639; changed: []; objects_data_and_selections_under_original_hashes: True |
| `final_access_after_lock` | pass | accesses: 10; lock_created: 2026-09-17T16:46:04.539079+00:00; first_access: 2026-09-17T16:46:33.908902+00:00 |
| `source_output_identity` | pass | release_views_checked: 210; violations: 0 |
| `b_only_identical` | pass |  |
| `projection_rules` | pass | violations: [] |
| `b_view_aliases_H` | pass | violations: [] |
| `mode_B_selection_identity` | pass | violations: [] |
| `score_dictionaries` | pass | selected_rows: 3116; max_abs_difference: 6.661338147750939e-16 |
| `incremental_baselines` | pass | rows: 18480 |
| `paired_differences` | pass | family_rows: 90; max_abs_difference: 1.101440987028024e-15 |
| `withholding_arithmetic` | pass | sampled_rows: 400; max_abs_difference: 1.1102230246251565e-16; per_person_mixture_vs_linear: 2.220446049250313e-16 |
| `bootstrap_replay` | pass | estimate: -0.006876572306322325; reported: -0.0068765723063223; se: 0.0012440974121320522; reported_se: 0.001244097412132 |
| `adjusted_interval_arithmetic` | pass | critical: 2.930105743069395 |
| `service_quality` | pass | max_abs_difference: 5.551115123125783e-17 |

## 3. Selection and sealing evidence

* `TRANSPORT_LOCK.json` hashes 17,639 inputs (protocol, comparison spec, evaluation code, all 42 interface objects and their inputs, 2017 fitting/validation releases, every Mode B fitted object and selection record, the development audit records, and all five partition files). It was committed and pushed as `3e31f28` **before** the first final-partition read.
* `FINAL_ACCESS_LOG.json` records every final read with the lock hash; the earliest is after the lock timestamp.
* Mode B selections were recomputed from stored 2017 validation scores by the verifier and match exactly; scope membership follows the declared origin/space predicates.
* Both lock amendments are code-only and are validated as such by the verifier (an amendment naming any non-code locked input is rejected).

## 4. Distinctions the reports preserve

* Expected outcomes of the stochastic withholding mechanism are separated from realized routings; both are reported.
* Frozen 2018 attacks (Mode A) are never pooled with fresh 2017 attacks (Mode B).
* Development (2018, exhausted cohort) is never pooled with transport (2017).
* Inference over people (household bootstrap on the final partition) is separated from variability over training seeds; the three seeds are three fixed fitted systems, shown individually.
* Absolute recovery, incremental recovery over the per-view H baseline, and each fixed ancestor attack's own score are reported separately; negative increments are never clipped.

## 5. Known limitations of this validation

* The verification confirms arithmetic, identity, selection and reproducibility. It cannot establish that the attack families are strong, that the moments certify independence, or that 2017 respondents are different people from 2018 respondents.
* Two lock amendments were needed after the final partition was opened (a write race and rare load-dependent numerical corruption under extreme memory pressure). Both are documented with diffs; all objects, data and selections kept their original hashes; every final unit was rescored with confirmed predictions and the originals are preserved.
* Class-level race conclusions are limited by support; see `TRANSPORT_RESULTS.md` §13.
