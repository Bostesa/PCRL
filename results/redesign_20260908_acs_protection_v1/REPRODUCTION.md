# Reproduction and availability

Run from the repository root on CPU with one numerical thread. The experiment
reuses the original household pools; its `test` JSON/array keys mean
**DEVELOPMENT EVALUATION**, not a fresh confirmatory test. Read [PROTOCOL.md](PROTOCOL.md).

## Dependencies and source reuse

The runner is [run_acs_protection.py](../../experiments/run_acs_protection.py),
with [joint affine maps](../../experiments/acs_protection_maps.py) and
[stronger auditors](../../experiments/acs_protection_audits.py). It imports the
unchanged [ACS data](../../experiments/acs_transfer_data.py),
[source models](../../experiments/acs_transfer_models.py),
[head/scorer code](../../experiments/acs_transfer_heads.py), and two small helpers
from the [original runner](../../experiments/run_acs_transfer.py).
The external LEACE implementation is `concept-erasure==0.2.4`; exact source
hashes are in [protocol_freeze.json](protocol_freeze.json) and versions in
[environment.json](environment.json). Existing [requirements](../../requirements-redesign.txt)
cover the environment. No dependency, dataset, or source model was downloaded
or installed for this run.

Raw file remains local at `data/folktables/2018/1-Year/psam_p06.csv`, SHA256
`dc2187fc90df2c5f6b546ee89a2b41c9a97379c9e7136461b6a6c8de871b43e0`.
Official retrieval and complete source regeneration commands are in the
[parent reproduction instructions](../redesign_20260907_acs_transfer_v1/REPRODUCTION.md).
This new runner requires the original selected source checkpoints, tree bank,
PCA/compression maps, cached development arrays, split rows and published parent
metadata. It checks original hashes and does **not** silently regenerate a
missing source object. Exact saved-object reuse needs those local files; no
public checkpoint download is claimed. For independent reproduction, regenerate
the parent experiment into a fresh directory using its original recipe, then
point the new config's `parent_results` at that directory and freeze a fresh
protocol. New numerical reproduction differs from exact historical byte replay.

## Commands

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python -m pytest -q tests/test_acs_protection_maps.py tests/test_acs_protection_audits.py tests/test_acs_protection_comparisons.py tests/test_acs_protection_runner.py
mkdir -p results/acs_protection_reproduction
cp results/redesign_20260908_acs_protection_v1/PROTOCOL.md results/acs_protection_reproduction/
cp results/redesign_20260908_acs_protection_v1/config.json results/acs_protection_reproduction/
.venv/bin/python -m experiments.run_acs_protection --out results/acs_protection_reproduction --prepare
.venv/bin/python -m experiments.run_acs_protection --out results/acs_protection_reproduction --seeds 0
# Measure complete seed wall time; retain one seed if the declared total exceeds 1800 seconds.
.venv/bin/python -m experiments.run_acs_protection --out results/acs_protection_reproduction --seeds 1 2
.venv/bin/python scripts/summarize_acs_protection.py --out results/acs_protection_reproduction
```

The pipeline fixture is part of the focused runner tests: artificial rows and
small common budgets, with no ACS outcomes or schema search. Scientific output
directories are created exclusively and never overwritten. The actual timed
commands, process exit statuses and logs are saved alongside the evidence.

Reporting uses serialized JSON only and never fits, selects, or accesses raw
records. Large local prediction arrays support score replay without refitting.
Fitted model/map checks are independent inference or array replay only.

```sh
.venv/bin/python scripts/verify_acs_protection_artifacts.py --out results/acs_protection_reproduction --report results/acs_protection_reproduction/new_artifact_replay.json
.venv/bin/python scripts/verify_acs_protection_scores.py --out results/acs_protection_reproduction --report results/acs_protection_reproduction/new_score_replay.json
```

Choose fresh report paths. These commands require local fitted arrays/objects
and the raw CSV; GitHub's compact JSON supports table regeneration but does not
contain the row-level probabilities or checkpoints needed for exact replay.
The preparation command records the M4 hardware with macOS `sysctl`; porting
to a different operating system requires adapting that metadata query before
freezing a fresh reproduction. The scientific fitting code is CPU based.

## GitHub boundary

Published: protocol/config, source dependencies and tests, per-seed candidate
metrics/support, release-map spectra and numerical diagnostics, source identity,
selection records and learning curves, tables/paired comparisons/plots, runtime,
verification and research decision. Fitted estimators, map arrays, cached
releases, row indices, probabilities, raw records and duplicate per-estimator
metadata remain local. Per-seed `local_artifacts.json` records binary paths,
sizes, SHA256 and regeneration instructions; the publication manifest also
accounts for omitted duplicate files. Recorded starting/source hashes retain
execution provenance and are not replaced by the later packaging commit.

The [publication manifest](publication_manifest.json) records the exact included
and local-only boundary. It accounts for about36.1MB of compact text/plots and
216.1MB of omitted new fitted/cache/duplicate artifacts, in addition to the raw
CSV and reused parent objects. [Packaging checks](PUBLICATION_CHECK.json) are
separate from scientific verification. GitHub contains no model download archive.
