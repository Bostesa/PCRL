# Reproduction and local artifact boundary

All evaluation fields named `test` are **DEVELOPMENT EVALUATION** on the same
households used in previous stages. Read the frozen [protocol](PROTOCOL.md) and
[configuration](config.json) before interpreting scores.

Use [requirements-redesign.txt](../../requirements-redesign.txt) and the recorded
[environment](environment.json). The existing M4 Pro CPU environment was reused,
with one numerical thread and no new package/model/data downloads.

The [runner](../../experiments/run_acs_bottleneck.py) now accepts the optional
`mapper_initialization: pca16` configuration, saving I/W alongside renamed final
C_init/D_init releases. The default random initialization remains available.
[Training](../../experiments/acs_bottleneck_training.py) changes only mapper
initialization and diagnostic snapshots; [catch-up](../../experiments/acs_bottleneck_catchup.py),
[independent audits](../../experiments/acs_protection_audits.py),
[heads/scoring](../../experiments/acs_transfer_heads.py),
[data/masks/splits](../../experiments/acs_transfer_data.py),
[source maps](../../experiments/acs_transfer_models.py),
[original runner helpers](../../experiments/run_acs_transfer.py),
[saved erasure maps](../../experiments/acs_protection_maps.py), and
[reference loader/scorer](../../experiments/run_acs_protection.py) are reused.
Actual source hashes are frozen in [protocol_freeze.json](protocol_freeze.json).

Historical execution hashes remain unchanged in historical files. To replay an
old runner against its original freeze, use that stage's original pinned commit;
the current runner/training source intentionally adds an option. An artificial
default-mode regression verified old/new checkpoint and optimizer equivalence.
Do not replace historical source hashes with this later publication commit.

## Exact local replay versus numerical regeneration

Raw local CSV: `data/folktables/2018/1-Year/psam_p06.csv`, SHA256
`dc2187fc90df2c5f6b546ee89a2b41c9a97379c9e7136461b6a6c8de871b43e0`.
Official retrieval and initial source-generation instructions are in the
[original ACS reproduction notes](../redesign_20260907_acs_transfer_v1/REPRODUCTION.md).
No raw person records or fitted objects are downloadable from this publication.

This run reused exact original PCA/source/bank maps, saved PCA32/LEACE/PCA16
arrays, old initial C/D checkpoint, old reference predictions/selections and
verified score records. Necessary original files are identified in every
`seed_*/parent_provenance.json`; original manifests retain their original hashes.
No necessary artifact was regenerated. New `seed_*/local_artifacts.json` records
all checkpoint/Adam, fitted head/auditor, prediction, release and split-array paths,
bytes and SHA256 values. They stay local. Compact metrics and selections permit
review/report regeneration through GitHub; exact frozen-model/prediction replay
requires the hashed local files.

Without those objects, regenerate the original transfer, fixed LEACE,
random-initialized bottleneck and PCA16 stages in dependency order using their
pinned source and recipes, always into fresh directories. Configure this run with
those regenerated parents before creating a new freeze. This is numerical recipe
reproduction, not byte-identical historical replay. A source/version mismatch or
missing object must be diagnosed, never silently replaced in an existing run.

## Commands

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
# Focused checks, including artificial miniature training/head/audit execution.
.venv/bin/python -m pytest -q tests/test_acs_bottleneck_initialization.py tests/test_acs_pca16_init_runner.py tests/test_acs_pca16_init_comparisons.py
mkdir results/acs_pca16_init_reproduction
cp results/redesign_20260908_acs_pca16_init_v1/config.json results/acs_pca16_init_reproduction/
cp results/redesign_20260908_acs_pca16_init_v1/PROTOCOL.md results/acs_pca16_init_reproduction/
# If necessary, change only fresh parent paths/provenance before freezing.
.venv/bin/python -m experiments.run_acs_bottleneck --out results/acs_pca16_init_reproduction --prepare
.venv/bin/python -m experiments.run_acs_bottleneck --out results/acs_pca16_init_reproduction --seeds 0
# Measure complete seed; expand unchanged only when total estimate <=900 seconds.
.venv/bin/python -m experiments.run_acs_bottleneck --out results/acs_pca16_init_reproduction --seeds 1 2
.venv/bin/python scripts/summarize_acs_pca16_init.py --out results/acs_pca16_init_reproduction
.venv/bin/python scripts/verify_acs_pca16_init.py --out results/acs_pca16_init_reproduction
```

Seed directories and freezes are created exclusively; no scientific overwrite.
Reporting reads serialized metrics and predictions without fitting. All release
snapshots complete training before utility fitting; final evaluation is gated
on saved selections. Published checks and runtime distinguish new scientific
execution, miniature validation, score replay and total implementation work.
