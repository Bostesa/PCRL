# Reproduction and local artifact boundary

This diagnostic adds only PCA16 utility heads and independent auditors. Start
with the [frozen protocol](PROTOCOL.md), [configuration](config.json),
[decision](RESEARCH_DECISION.md) and [validation](VALIDATION.md). All `test`
fields mean DEVELOPMENT EVALUATION on the same previously used households.

## Actual dependencies and identities

- [Runner](../../experiments/run_acs_pca16.py): original PCA identity/order,
  immutable first16 slicing, reference reuse, fitting boundaries and selection gate.
- Unchanged [utility heads/scorer](../../experiments/acs_transfer_heads.py),
  [five auditors](../../experiments/acs_protection_audits.py),
  [labels/splits/data](../../experiments/acs_transfer_data.py), and helpers in
  [protection runner](../../experiments/run_acs_protection.py) and
  [original runner](../../experiments/run_acs_transfer.py). The frozen execution
  dependency hashes are in [protocol_freeze.json](protocol_freeze.json).
- [Reporter](../../scripts/summarize_acs_pca16.py) reuses pure
  [protection comparison](../../scripts/summarize_acs_protection.py) and
  [bottleneck indexing](../../scripts/summarize_acs_bottleneck.py) helpers.
  [Verifier](../../scripts/verify_acs_pca16.py) reuses published
  [independent scores/inference](../../scripts/verify_acs_bottleneck_scores.py).
  Neither fits models. Reporting source hashes are in [summary.json](summary.json).
- [Environment](environment.json) is the unchanged preceding-stage environment
  record, copied for reference; its capture date remains original. Existing
  [requirements](../../requirements-redesign.txt) and local environment were
  reused with no installations/downloads. Actual timing is in [runtime.json](runtime.json).

Each seed's `release_freeze.json` identifies all original component rows, order,
mean, variance and first16 output hashes. `provenance.json` hashes actually used
original maps, cached releases, split rows and prior predictions. Each new
`selection_before_test.json` records choices, fit/validation data hashes,
preprocessing, schedules, actual fitting exposures/updates and saved models.
No original source hashes are replaced by the later publication commit.

## Commands

Run from the repository root. Use a fresh directory; seed directories and the
protocol freeze refuse overwrites. Preserve historical paths as read-only inputs.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python -m pytest -q tests/test_acs_pca16.py tests/test_acs_pca16_comparisons.py
mkdir -p results/acs_pca16_reproduction
cp results/redesign_20260908_acs_pca16_v1/config.json results/acs_pca16_reproduction/
cp results/redesign_20260908_acs_pca16_v1/PROTOCOL.md results/acs_pca16_reproduction/
# Record the actual starting commit and any regenerated reference paths in the fresh config.
.venv/bin/python -m experiments.run_acs_pca16 --out results/acs_pca16_reproduction --prepare
.venv/bin/python -m experiments.run_acs_pca16 --out results/acs_pca16_reproduction --seeds 0
# Time the complete seed; expand only if the total estimate fits600seconds.
.venv/bin/python -m experiments.run_acs_pca16 --out results/acs_pca16_reproduction --seeds 1 2
.venv/bin/python scripts/summarize_acs_pca16.py --out results/acs_pca16_reproduction
.venv/bin/python scripts/verify_acs_pca16.py --out results/acs_pca16_reproduction
```

The actual commands and complete-seed timings are in [runtime.json](runtime.json),
[seed0 log](seed_0_execution.log) and [remaining logs](seeds_1_2_execution.log).
Allthree seeds ran unchanged. No scientific error required a correction/rerun.

## Available through GitHub versus local only

Published: source/tests and required helper dependencies, protocols/configuration,
small tables, new per-candidate/per-class/weighted scores, curves/exposures,
selections, component/provenance hashes, paired comparisons, runtime and checks.
Historical reference scores remain linked rather than duplicated.

Raw CSV remains local at `data/folktables/2018/1-Year/psam_p06.csv`, SHA256
`dc2187fc90df2c5f6b546ee89a2b41c9a97379c9e7136461b6a6c8de871b43e0`.
Original maps/releases/reference fitted objects and probabilities remain local.
The original [ACS reproduction recipe](../redesign_20260907_acs_transfer_v1/REPRODUCTION.md)
provides public data retrieval; [protection](../redesign_20260908_acs_protection_v1/REPRODUCTION.md)
and [bottleneck](../redesign_20260908_acs_bottleneck_v1/REPRODUCTION.md) give the
subsequent recipes and omitted-artifact identities. This experiment found all
required artifacts already present and verified: **none were regenerated**.

New `seed_*/local_artifacts.json` records hashes/bytes/paths of PCA16 cached
releases, predictions, row indices and fitted models; these binaries and
duplicate estimator metadata are excluded. No claim is made that local fitted
objects can be downloaded from GitHub. Exact saved-object/row-score replay needs
them. Numerical reproduction without those objects requires regenerating the
earlier stages into new directories in dependency order, including their saved
verification records, then freezing the new PCA16 protocol against those actual
identities. That is recipe reproduction, not byte-identical historical replay.
The current runner checks its inputs and never silently regenerates a reference.
