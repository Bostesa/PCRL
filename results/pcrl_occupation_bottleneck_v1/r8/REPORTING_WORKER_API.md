# R8 fixed reporting worker

Only ROOT executes numerical jobs, each as an `attacks` queue unit. Importing `occupation_bottleneck.reporting` uses standard-library modules and metadata-only reader definitions. Registry construction uses package metadata and does not touch person arrays or truth hashes.

`python -m occupation_bottleneck.reporting --phase report --spec PRIVATE_CONFIG.json`

Report config:

- `unit_id`, `manifest:{path,sha256}`, `reporting_seal` operational pin/supplement, `endpoint_registries:{utility:{path,sha256},privacy:{path,sha256}}`.
- Fresh `private_output` directory containing `.private`, `public_output` directory (R8 namespace).
- `label_table:{path,sha256}` and `label_table_freeze:{path,sha256}` bind the preassessment C3 FIT01 table and FROZEN_COMPLETE receipt.
- Exactly72 `packages` records, each `arm_id`, `seed`, `release_variant` (`full` or `head_row_space`), optional `source_arm` for projections, and `freeze`, `release`, `scores`, `fit_release`, `fit_scores` pins. The pins bind the original frozen INNER/FIT-selection wires and fixed scores. Both source seeds must occur for each of36 configurations.
- `control_dependencies` maps every arm to its registered representative control arms: RAW_JOINT_R03_E5 universally, also U0 for U0/U0_PROJ, also JOINT_CONDITIONAL_R03_E5 for JOINT_CONDITIONAL arms. Both ATT and INNER gates for both source seeds are required. Static registry failure_dependencies name the person-aligned constant bool `control_gates` table; any missing/false gate preserves the endpoint as MISSING at K108. Malformed gates fail closed.
- Operational seal `roles.INNER_SELECTION` maps income/occupation/sex to private pins. Shared `require_reporting` validates these against the committed public lock. Every scientific source and code/config/registry dependency must appear in the active queue manifest.

Canonical public lock `endpoint_registries` stores utility/privacy `{sha256}` objects without private paths. Registry definitions must equal metadata regeneration from the full72package roster before truth opens. C3 frozen tables/counts are replayed on FIT01 and checked before opening INNER. Both seed roles must have the identical people, groups and labels.

Actual probabilities and decisions come from each bank's frozen released wire, including projected outputs. The metric vectors carry row_id/unit plus `own__TASK__accuracy|nll|brier` and `REF_U0|REF_Ucal__TASK__accuracy|nll|brier`. The actual reference namespace never replaces candidate U0. Scores are copied to new archives with row_id/unit attached; original archives remain intact.

Public outputs:

- `TRAJECTORY_METRICS.csv`: REPORT/INNER and FIT_DIAGNOSTIC/FIT2 are explicit. FIT2 has no assessment gate.
- `RELEASE_SURFACE_COMPARISONS.csv`: primary, representation, output, logits_output, decision and confidence for both recipients and pair; absent fitted contracts remain MISSING.
- `REPORT_PACKAGE_METRICS.json`: top-level `packages` array; each identity contains `roles.FIT2` and `roles.INNER`, whose `tasks.TASK` contains `metrics` and `references.REF.{metrics,difference}`; roles also contain `clauses`, `recovery`, `controls`, `point_eligible_seed`, `assessment_gate_applied`.
- `REPORT_SUMMARY.json`: aggregate paired configurations, with all24 original point utility clauses required for eligibility.
- `C3_TASK_ASSOCIATION.json`: `task_association.FIT2|INNER.r1|r2|pair`, plus `reader_banks` true-task supported strata; interpretation-only, never a universal privacy floor.
- `DECISION_READER_ALIASES.json`: exact full/projection decision score aliases across all anchor seeds/roles; duplicates give no extra scientific evidence.

Private `REPORT_INPUT_INDEX.json` gives `unit_input` pin and named `arrays` pins. Names are `truth`, `control_gates`, `metrics__ARM__S0|1`, `scores__ARM__S0|1`. Tables are person-aligned in the canonical units order. It also preserves all package metrics and their private vector/score pins.

`python -m occupation_bottleneck.reporting --phase inference --spec PRIVATE_CONFIG.json`

Inference config contains `unit_id`, the same operational `reporting_seal` and frozen `endpoint_registries`, `unit_input` and named `arrays` pins from the report index, fresh `private_output` and R8 `public_output`. Every pin is an active queue source.

The runner independently implements weighted tie-aware AUC, metric contrasts, duplicate-group draws and bounds. B2048 seed2026101301 draws are shared by all arms and both seeds, preserving row multiplicities. One utility part retains source seed identity; privacy averages two seed envelopes within each draw. Each whole base9/local-ignore envelope is maximized anew inside each draw. Oriented reader curves and envelopes are cached once for the shared draw bank. Utility K864 and privacy K108 remain distinct, full families; missing dependencies retain MISSING slots. Intervals use bootstrap SD(ddof1) times Bonferroni Student-t quantile with df2233. These are conditional approximate exploratory bounds.

Inference emits `UTILITY_INFERENCE.json`, `PRIVACY_INFERENCE.json`, `GAP_TO_TARGET.csv` and private `ENDPOINT_BOOTSTRAP_DRAWS.npz`. Positive gap_to_target is the cap/benefit shortfall; the endpoints themselves use the original signed contrasts and original thresholds.

The independent API consumes one family registry, the same unit_input and named arrays, plus its independently authenticated reporting lock. Neither runner module imports independent scientific code.

`python -m occupation_bottleneck.reporting --phase fixture --spec PRIVATE_CONFIG.json` runs only as a root-owned `attacks` queue unit. Config needs unit_id and a fresh public_output file. Synthetic checks cover tie-aware weighted AUC, envelope reselection within draws, NLL clipping, Brier summation, input immutability, missing classes, duplicate groups, paired seed draws and threshold classifications. It reads no scientific arrays. CSV outputs use LF newlines.
