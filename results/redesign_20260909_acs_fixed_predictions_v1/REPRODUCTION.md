# Reproduction and exact replay

Use the existing PCRL `.venv` on the recorded M4 Pro with one numerical thread. No installation, download or historical refitting was needed. [environment.json](environment.json) records the environment as observed after execution, not a retrospectively invented prefit record. [config.json](config.json), [PROTOCOL.md](PROTOCOL.md), [protocol_freeze.json](protocol_freeze.json), [REUSE_MANIFEST.json](REUSE_MANIFEST.json) and [PREFIT_IDENTITY.json](PREFIT_IDENTITY.json) bind the original sources and all new scientific inputs.

From repository root, restore any compact metric exports before replay:

```sh
.venv/bin/python scripts/package_acs_fixed_predictions.py --restore
```

This only decompresses hash-verified derived evidence. It does not recreate omitted fitted models or person-level arrays. Exact saved-model replay additionally requires the original ACS file, historical PCA/context caches, the nine selected PCA32 predictor objects and standardizers, the original initialization checkpoints, the three frozen E maps, and this study's local fitted objects listed in [LOCAL_ARTIFACTS.md](LOCAL_ARTIFACTS.md). Without those objects, compact comparison arithmetic remains reproducible; model-level exact replay does not.

The actual bounded sequence was:

```sh
.venv/bin/python scripts/run_acs_fixed_predictions_bounded.py --phase prepare --max-units 1
.venv/bin/python scripts/run_acs_fixed_predictions_bounded.py --phase train --max-units 1
.venv/bin/python scripts/run_acs_fixed_predictions_bounded.py --phase train --max-units 2
.venv/bin/python scripts/run_acs_fixed_predictions_bounded.py --phase evaluate --max-units 1
.venv/bin/python scripts/run_acs_fixed_predictions_bounded.py --phase evaluate --max-units 17
.venv/bin/python scripts/run_acs_fixed_predictions_checks.py --step tests
.venv/bin/python scripts/run_acs_fixed_predictions_checks.py --step replay
.venv/bin/python scripts/run_acs_fixed_predictions_checks.py --step summary
.venv/bin/python scripts/run_acs_fixed_predictions_checks.py --step enrich
.venv/bin/python scripts/run_acs_fixed_predictions_checks.py --step comparisons
```

The first prepare attempt failed before fitting due to a namespace error; the correction preceded the protocol freeze. Process records preserve both attempts. All representations were globally frozen before new reserved evaluation. Evaluation proceeded H, E, A0, L025, L20, J per seed. Completed-unit manifests prevent overwriting or refitting completed units; representation fitting is closed after the global release manifest exists.

For read-only replay after the study's original wall-clock deadline, use the modules directly with the same one-thread environment; the bounded execution wrappers intentionally refuse to restart scientific work past that deadline:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m scripts.replay_acs_fixed_predictions
.venv/bin/python -m scripts.verify_acs_fixed_predictions_comparisons
```

A new end-to-end regeneration must use a new output directory, prospectively record its actual start and regenerate its frozen manifest from the same pinned configuration and historical inputs. Do not overwrite this study or falsify the old clock. The `--out` option is available on the runner and check tools. Preparing input manifests is not authorization to fit a changed matrix. Historical regeneration instructions must be followed if a pinned original object is absent; an unverifiable replacement anchor cannot preserve the claimed source function.

The additional read-only `python -m scripts.analyze_acs_fixed_predictions_controls` command reproduces the fixed-control table and per-sensitive-view .005 reference flags; no threshold is assigned to opposing task targets.

The historical `audit_mlp_epochs=120` and `catchup_epochs=120` fields define nested-prefix reporting, while `extended_audit_epochs=360` is the actual single trajectory length. No second prefix was fitted. Candidate paths distinguish last optimizer/RNG states from best-validation weights. Static H/E observers receive the260-pass fitting exposure and own catch-up; B trajectories and all matching B candidates are aliases. [FITTING_COUNTS.json](FITTING_COUNTS.json) reports actual unique work.
