# Reproduction and evidence availability

This is DEVELOPMENT EVALUATION on the original ACS households. The [protocol](PROTOCOL.md), [configuration](config.json), and [source freeze](protocol_freeze.json) specify the finite study. The original uniform-preservation study is complete and reused. Its source identities remain in its own freeze; edited training code is not attributed to historical runs.

Use the existing M4 Pro environment in [environment.json](environment.json), with one numerical thread. No infrastructure, package, model or data download was needed. Raw input stays local at `data/folktables/2018/1-Year/psam_p06.csv`, SHA256 `dc2187fc90df2c5f6b546ee89a2b41c9a97379c9e7136461b6a6c8de871b43e0`. Historical source PCA preprocessing, split arrays, fitted models and prediction caches are prerequisites identified by configuration and each static seed's `historical_reuse.json`.

The six R/rho=.1/beta1-persistent C/D models and their task predictions come from `../redesign_20260908_acs_preservation_v1/beta_1/seed_{0,1,2}/`; matching nested120/360 audits come from that study's `extended/seed_{0,1,2}/`. I comes from the initialization study after exact initial tensors, preprocessing, schedules and release verification. Raw PCA16, beta0, warmup-only, PCA32/LEACE, neural/tree-bank and prior/exposed controls retain their original provenance. No historical scientific model was regenerated or refitted here.

E/S map files, original teacher targets and the once-only paired permutation are under `static/seed_N/teachers/`. Their prefit freeze precedes both map fits and binds original fitting rows, real/permuted labels, masks, scales and permutation. Each learned unit loads those exact maps; it does not refit them. Static E/S direct releases have no saved observer or catch-up candidate. W and I have utility/geometry diagnostics, not newly fitted attribute audits.

New learned-unit directories are `E_rho0p1/seed_N`, `R_rho0/seed_N`, `E_rho0/seed_N`, `S_rho0p1/seed_N` and `S_rho0/seed_N`. A directory is completed only with its `metrics.json`, `local_artifacts.json` and `completion.json`. The [progress manifest](progress.json) records scientific wall times and exact ordered units. Resume verifies and skips completed units and never overwrites them. An incomplete directory requires preserving and diagnosing the attempt; it is not automatically reused as valid evidence.

The exact execution commands were:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python -m experiments.run_acs_selective --prepare
/usr/bin/time -p .venv/bin/python -m experiments.run_acs_selective --run --max-units 1
/usr/bin/time -p .venv/bin/python -m experiments.run_acs_selective --run
```

The first-unit stop is a predeclared timing check, not a result-selection gate. The continuation follows the configuration's fixed order and budget guards. Re-running `--prepare` in a frozen directory is intentionally rejected. A reproduction must use a fresh directory, copy the protocol/configuration, set the new operational `started_utc` before preparing, preserve the scientific settings, and pass `--out <fresh-directory>` to both commands. Changing the timestamp changes the configuration hash and must be reported as a new execution identity. Current completed artifacts are never overwritten. Do not rerun this matrix just to regenerate reports.

To reconstruct missing historical prerequisites, follow the published [transfer](../redesign_20260907_acs_transfer_v1/REPRODUCTION.md), [protection](../redesign_20260908_acs_protection_v1/REPRODUCTION.md), [bottleneck](../redesign_20260908_acs_bottleneck_v1/REPRODUCTION.md), [PCA16](../redesign_20260908_acs_pca16_v1/REPRODUCTION.md), [initialization](../redesign_20260908_acs_pca16_init_v1/REPRODUCTION.md), and [uniform-preservation](../redesign_20260908_acs_preservation_v1/REPRODUCTION.md) instructions. Such regeneration is separate compute and must not be described as loading the original artifacts. No regeneration was necessary for this execution.

Fitted weights/Adam/RNG, selected probability arrays, PCA/teacher/release arrays, affine coefficients and raw-person rows stay local. Each unit's local artifact manifest records paths, bytes and SHA256. The scientific source, compact scores/curves, selection records, support, protocol, teacher diagnostics and reports are published. The public evidence supports read-only table/figure regeneration; exact network inference replay additionally requires the declared local artifacts. There is no claim that private local checkpoints can be downloaded from GitHub.

Utility candidates are saved at `fitted/transfer/<release>/<task>/<candidate>/`. Audit candidates are at `fitted/audit/<release>/<attribute>/fresh/nested{120,360}/<candidate>/` and, for learned releases, `saved_start/nested{120,360}/catchup/` plus `saved_start/saved/`. Actual terminal model/Adam/RNG files `last_training_*_epoch{120,360}.pt` remain distinct from validation-selected weights. The nested prefixes are from one360-epoch trajectory; no best-checkpoint rewind or repeated120-epoch fit is claimed.

See [validation](VALIDATION.md) for focused checks and independent replay commands, [runtime](runtime.json) for compute versus read-only/total elapsed work, [executed matrix](EXECUTED_MATRIX.json) for completeness, and [the decision](RESEARCH_DECISION.md) for the evidence-based interpretation. The next experiment is a recommendation only.
