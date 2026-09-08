# Reproduction and artifact availability

Read [PROTOCOL.md](PROTOCOL.md) first. All `test` fields refer to **DEVELOPMENT
EVALUATION on the original households**, not untouched confirmation. The fixed
configuration is one comparison, with no release or protection-weight search.

## Source and prerequisites

Run from the repository root using the existing environment and
[requirements-redesign.txt](../../requirements-redesign.txt). The execution
closure and installed versions are recorded in [protocol_freeze.json](protocol_freeze.json)
and [environment.json](environment.json). No dependencies, datasets or models
were downloaded for this run.

- [Runner, immutable reference reuse, label boundaries and selection gate](../../experiments/run_acs_bottleneck.py).
- [Matched training, optimizer clones, schedules and gradients](../../experiments/acs_bottleneck_training.py).
- [Direct-coordinate saved adversary and catch-up](../../experiments/acs_bottleneck_catchup.py).
- Unchanged [independent auditors](../../experiments/acs_protection_audits.py),
  [utility heads and scorer](../../experiments/acs_transfer_heads.py),
  [data/masks/splits](../../experiments/acs_transfer_data.py),
  [original source maps](../../experiments/acs_transfer_models.py),
  [saved affine map loader](../../experiments/acs_protection_maps.py), and helper
  dependencies in [the previous runner](../../experiments/run_acs_protection.py)
  and [original runner](../../experiments/run_acs_transfer.py).
- [Reporting without refitting](../../scripts/summarize_acs_bottleneck.py), which
  reuses pure comparison helpers from [the prior report](../../scripts/summarize_acs_protection.py).
- [Artifact replay](../../scripts/verify_acs_bottleneck_artifacts.py) reuses read,
  label and array helpers from [the prior artifact checker](../../scripts/verify_acs_protection_artifacts.py);
  [score/model replay](../../scripts/verify_acs_bottleneck_scores.py) checks saved
  predictions and selections without retraining. Commands are in [VALIDATION.md](VALIDATION.md).

Raw CSV remains local at `data/folktables/2018/1-Year/psam_p06.csv`, SHA256
`dc2187fc90df2c5f6b546ee89a2b41c9a97379c9e7136461b6a6c8de871b43e0`.
The [original reproduction notes](../redesign_20260907_acs_transfer_v1/REPRODUCTION.md)
give the official retrieval and source-generation recipe. Exact replay needs
the original PCA/source/bank maps, cached releases, split rows, and the preceding
pilot's saved LEACE maps, utility/audit objects and predictions. None are claimed
to be downloadable from this GitHub repository. Their original SHA256 identities
remain in the parent manifests; new per-seed provenance records identify those
actually used. This runner checks them and never silently retrains a reference.

For independent numerical reproduction without those local objects, regenerate
the original transfer stage into a new directory, then the preceding fixed LEACE
stage into another new directory using their published recipes. Point the new
configuration at these two regenerated parents before its new protocol freeze.
That is recipe reproduction, distinct from exact historical byte replay. Do not
overwrite the published historical directories or substitute a newly fitted
eraser during the learned-arm audit.

## Commands

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python -m pytest -q tests/test_acs_bottleneck_training.py tests/test_acs_bottleneck_catchup.py tests/test_acs_bottleneck_runner.py tests/test_acs_bottleneck_comparisons.py
mkdir -p results/acs_bottleneck_reproduction
cp results/redesign_20260908_acs_bottleneck_v1/config.json results/acs_bottleneck_reproduction/
cp results/redesign_20260908_acs_bottleneck_v1/PROTOCOL.md results/acs_bottleneck_reproduction/
# If needed, edit only the fresh copy's parent paths and record its actual starting commit.
.venv/bin/python -m experiments.run_acs_bottleneck --out results/acs_bottleneck_reproduction --prepare
.venv/bin/python -m experiments.run_acs_bottleneck --out results/acs_bottleneck_reproduction --seeds 0
# Measure the complete seed before expanding. If total estimate exceeds1800s, retain one seed.
.venv/bin/python -m experiments.run_acs_bottleneck --out results/acs_bottleneck_reproduction --seeds 1 2
.venv/bin/python scripts/summarize_acs_bottleneck.py --out results/acs_bottleneck_reproduction
```

The miniature test fixture uses artificial rows, not ACS outcomes, with reduced
common fixture-only epochs. Scientific entry points enforce the full fixed
configuration. Every seed directory is created exclusively. Reports read saved
JSON/CSV and never refit models to reconstruct a table. Seed runtime logs include
reference verification, training, independent and catch-up attackers, and scoring.

## GitHub versus local-only evidence

Published: execution dependencies/tests, frozen configuration/protocol, per-seed
and per-candidate/class metrics, training and audit curves, fixed-final checkpoint
identities, clone/gradient/exposure records, support/split hashes, saved selections,
paired comparisons, descriptive criteria, plots, runtime and independent checks.
Per-seed `local_artifacts.json` and the publication manifest record local paths,
sizes and hashes for fitted models/Adam states, arrays, row indices and saved
probabilities; those objects and duplicate per-estimator metadata are omitted.
Raw person records and environments are omitted. Compact scores support review
and reporting through GitHub, while exact fitted-object/row-score replay needs
the local artifacts. Original execution source hashes are retained unchanged and
are not replaced by the later publication commit.
