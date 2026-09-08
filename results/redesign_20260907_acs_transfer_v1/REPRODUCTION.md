# Reproduction and GitHub availability

Read [PROTOCOL.md](PROTOCOL.md), [config.json](config.json), and
[the schema review](../../docs/ACS_2018_SCHEMA_NOTES.md). Numerical fitting used
the versions in [environment.json](environment.json); the repository's
[redesign requirements](../../requirements-redesign.txt) cover these dependencies.
No dependency or data was installed/downloaded for this run.

The runner and its complete new helper closure are:

- [run_acs_transfer.py](../../experiments/run_acs_transfer.py): source selection,
  fixed release extraction, shared downstream rows, saved selections and test gate.
- [acs_transfer_data.py](../../experiments/acs_transfer_data.py): raw cohort,
  household sampling/splits, target masks, strict allowlist, fit-only preprocessing.
- [acs_transfer_models.py](../../experiments/acs_transfer_models.py): source-only
  encoder and tree banks, full categorical schemas, masked loss and checkpoints.
- [acs_transfer_heads.py](../../experiments/acs_transfer_heads.py): independent
  downstream/audit families, validation selection, fixed-schema metrics and weights.
- [summarize_acs_transfer.py](../../scripts/summarize_acs_transfer.py): reporting
  only; no refitting, label access or selection.

## Data retrieval

Local raw file: `data/folktables/2018/1-Year/psam_p06.csv`.
Bytes: 267297811. SHA256:
`dc2187fc90df2c5f6b546ee89a2b41c9a97379c9e7136461b6a6c8de871b43e0`.
The official [California 2018 one-year person archive](https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/csv_pca.zip)
was verified reachable by HTTP HEAD (200, 67233615-byte ZIP); it was **not**
downloaded during this experiment. Extract the person CSV to the path above and
verify its hash. If a different official revision yields different bytes, record
that explicitly and run in a new result directory; do not represent it as exact
replay. Do not use the historical processed parquet.

## Checks and fresh fitting

Run from the repository root. A working virtual environment is assumed; adapt
the Python executable if necessary. Use a **fresh** directory, preserving this
published evidence. The following commands regenerate models and outcomes;
they are distinct from replaying the original saved checkpoints.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python -m pytest -q tests/test_acs_transfer_data.py tests/test_acs_transfer_models.py tests/test_acs_transfer_heads.py tests/test_acs_transfer_runner.py
mkdir -p results/acs_transfer_reproduction
cp results/redesign_20260907_acs_transfer_v1/PROTOCOL.md results/acs_transfer_reproduction/
cp results/redesign_20260907_acs_transfer_v1/config.json results/acs_transfer_reproduction/
.venv/bin/python -m experiments.run_acs_transfer --out results/acs_transfer_reproduction --mini-check
.venv/bin/python -m experiments.run_acs_transfer --out results/acs_transfer_reproduction --prepare
.venv/bin/python -m experiments.run_acs_transfer --out results/acs_transfer_reproduction --seeds 0
# Inspect complete-seed timing, then run the unchanged remaining declared seeds.
.venv/bin/python -m experiments.run_acs_transfer --out results/acs_transfer_reproduction --seeds 1 2
.venv/bin/python scripts/summarize_acs_transfer.py --out results/acs_transfer_reproduction
```

These exact module commands were executed for the original results using the
published directory, with process wrappers recording wall time and stdout logs.
The miniature check uses temporary artificial records solely for full pipeline
plumbing, deletes its fitted objects afterward, and reports no scientific result.
Prepare freezes protocol/config/source hashes, data identity and support counts;
each seed verifies them before fitting. An existing seed directory is never
overwritten. Source test scoring opens only after all 34 selections per seed are
saved. No frozen source was changed after the original protocol preparation.

Reporting can be regenerated from GitHub's compact evidence alone:

```sh
.venv/bin/python scripts/summarize_acs_transfer.py --out results/redesign_20260907_acs_transfer_v1
```

`RESEARCH_DECISION.md` is the human interpretation; the generated `ANALYSIS.md`
contains all paired/weighted summaries. Regenerating plots can change nonnumeric
file metadata; numerical records remain the evidence.

Portable inference-only verification commands and the relationship to the
original verification-script hashes are documented in
[PORTABLE_VERIFIERS.md](PORTABLE_VERIFIERS.md). They require the local fitted
arrays/objects or freshly regenerated equivalents and never retrain a model.
They print a report or exclusively create a new destination, preserving the
original verification records.

## Included and omitted objects

GitHub includes protocol/config/schema, original raw per-seed metrics,
per-target/per-class summaries, all validation selection records, preprocessing
parameters, source-candidate learning/selection diagnostics, support and split
hashes, selected checkpoint identities, runtime/logs, verification, and small
plots. `source_selection.json` embeds the two neural and two tree candidate
metadata records; `selection_before_test.json` embeds every downstream/audit
candidate's fitting and validation metadata. Duplicate per-model copies are
unnecessary for review.

Raw Census records, source checkpoints, tree estimators, probe models, PCA maps,
cached releases, split-row arrays and prediction arrays remain **local only**.
Every seed's `local_artifacts.json` records paths, bytes and SHA256 hashes for
these fitted/binary artifacts. The [publication manifest](publication_manifest.json)
also identifies omitted duplicate metadata. No public model download is claimed.
Exact saved-object score replay requires those local arrays/models; independent
fresh reproduction can regenerate them with the commands above and verified raw
CSV. Tests and result summarization require neither raw data nor checkpoints.

The experiment's historical starting commit and frozen execution-source hashes
are retained. The later publication commit packages the evidence and does not
replace its original provenance. Public rows/household identifiers are not
included in this commit; deterministic seeds plus the exact CSV recreate splits.
