# Objective-diagnosis review index

This branch is post hoc method analysis based on completed prospective evidence commit `5e154e5c4fdaeb23d327a0ebefe838525f1a19cb`. It leaves that branch, the original inference, and all private person records unchanged.

| File | Review purpose |
|---|---|
| [OBJECTIVE_DIAGNOSIS.md](OBJECTIVE_DIAGNOSIS.md) | Main answer, four-way mismatch decomposition, limitations, and procedural deviation. |
| [CONSTRAINT_SLACKS.json](CONSTRAINT_SLACKS.json) | Three anchors' exact fixed objectives, D17 row minima, Q and D17 fitted-CMI values and slacks. |
| [RANDOMIZATION_PROFILE.json](RANDOMIZATION_PROFILE.json) | Aggregate channel entropy, TV, row distortion, support, information-radius bound, and 2018 coarse/fine/held-out laws. No person rows. |
| [SOLVER_CERTIFICATE.md](SOLVER_CERTIFICATE.md), [SOLVER_CERTIFICATE.json](SOLVER_CERTIFICATE.json) | Derivation, nonnegative multipliers, supporting-hyperplane lower bounds, residuals, and numerical limits. |
| [EXACT_DIAGNOSTIC_FIXTURES.json](EXACT_DIAGNOSTIC_FIXTURES.json) | Object/channel hashes, exact task-cost replays, encoder dimensions and role counts. |
| [DECODER_COMPARISON_2018.json](DECODER_COMPARISON_2018.json) | Existing supervised 2018 fixed-decoder, independent-probe, and selected-deployment aggregates. |
| [PROSPECTIVE_AGGREGATE_REPLAY.json](PROSPECTIVE_AGGREGATE_REPLAY.json) | Completed 2016 decision reproduced solely from the published aggregate, with no final-row access. |
| [REUSABLE_INPUTS.json](REUSABLE_INPUTS.json) | SHA-256-pinned private input and archive member map, frozen source and Linux x86 environment. |
| [RESEARCH_RECOMMENDATION.md](RESEARCH_RECOMMENDATION.md) | One small matched 2018-only discriminating test and Terminal 4 design constraints. |
| [DIAGNOSTIC_ACTIVITY_LOG.md](DIAGNOSTIC_ACTIVITY_LOG.md) | Exact scope and the preliminary partition-fit deviation. |

Reproduction on the already restored 2018 objects: use the compatible environment in `REUSABLE_INPUTS.json`, then run `python -m analysis.pcrl_objective_diagnosis_v1.run --private-root <historical-restored-run> --finec-root <directory-of-archived-fineC_anchor_{0,1,2}.joblib> --output-root results/pcrl_objective_diagnosis_v1`. Re-run `archive_map.py` with both verified private manifests to restore archive member metadata to `REUSABLE_INPUTS.json`; run `replay_aggregate.py` and `decoder_compare.py` against their pinned aggregate JSON sources. No step fits a release or audit model. The six targeted tests run with `python -m pytest -q tests/pcrl_objective_diagnosis_v1`.

The three anchors share households. The 2018 supervised development test influenced original selection and is not a new holdout. Empirical conditional information is sparse and neither the fitted two-cell law nor the four-cell diagnostic identifies full-H conditional information. The solver bound is numerical rather than interval certified. No new confirmatory p-value is reported.
