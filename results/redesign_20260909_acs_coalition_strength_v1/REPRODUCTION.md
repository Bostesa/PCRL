# Reproduction and exact replay

Read [the frozen protocol](PROTOCOL.md), [configuration](config.json), [comparison rules](comparison_rules.json), [source freeze](protocol_freeze.json), [historical reuse manifest](REUSE_MANIFEST.json), and [executed matrix](EXECUTED_MATRIX.json). This study varies only the additional local/coalition coefficient on the existing research branch. It does not run the deferred refreshed-version experiment.

Use the existing `.venv` and local ACS data, original PCA32 maps, historical coalition continuation forks, final models, utilities, auditors and controls. [LOCAL_ARTIFACTS.md](LOCAL_ARTIFACTS.md) distinguishes compact public evidence from local fitted objects. There are no package installations, downloads or paid services. Historical evidence retains its original fitting source identities and prior operational amendment; current source is not substituted into old records.

The original cohort file is `data/folktables/2018/1-Year/psam_p06.csv`, SHA256 `dc2187fc90df2c5f6b546ee89a2b41c9a97379c9e7136461b6a6c8de871b43e0`. Reuse checks bind all historical artifacts and original independent replay certificates. New continuations start from `results/redesign_20260908_acs_coalition_v1/seed_N/training/F_I/fork.pt` or `P_I/fork.pt`. These are the complete post-observer-warmup states with the exact continuation-start cursor; the earlier `warm_adversary.pt` has the same warmed tensors but a different saved schedule position and is not silently substituted.

For a prospective exact rerun, choose a new unused directory, set its actual start timestamp, and copy the protocol/configuration/comparison definitions before preparing. Do not overwrite this completed study or claim the rerun loaded historical new-arm fits. The source and configuration closure must match the intended protocol. Commands below document the original finite execution; reusing this directory skips complete hash-bound units and rejects unrecorded partial artifacts.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python -m experiments.run_acs_coalition_strength --out NEW_DIRECTORY --phase prepare
# First complete training block, then remaining fixed continuations.
.venv/bin/python scripts/run_acs_coalition_strength_bounded.py --out NEW_DIRECTORY --phase train --max-units 12
.venv/bin/python scripts/run_acs_coalition_strength_bounded.py --out NEW_DIRECTORY --phase train --max-units 24
# The runner requires all 54 systems frozen before any new reserved labels.
.venv/bin/python -m experiments.run_acs_coalition_strength --out NEW_DIRECTORY --phase freeze
.venv/bin/python scripts/run_acs_coalition_strength_bounded.py --out NEW_DIRECTORY --phase evaluate --max-units 1
.venv/bin/python scripts/run_acs_coalition_strength_bounded.py --out NEW_DIRECTORY --phase evaluate --max-units 35
```

The outer wrapper counts whole scientific-process wall time, including loading, fitting, checkpointing, selection, scoring and failures. It enforces the smaller of the remaining 10,800-second scientific ceiling and 21,600-second total-work deadline after reserving the final 2,700 seconds for reporting/publication. It does not promise that a stopped or sleeping session can continue. Completion records prevent overwrites; a transient operational recovery needs a preserved failure and explicit record. Scientifically invalid evidence must be labeled and source amendments preserved before affected work is rerun.

Each new evaluation unit includes all five utility tasks, eleven audit roles, F public probability views, all legal singleton projections, and nine own-observer catch-ups. Fresh/catch-up MLPs use one 360-epoch trajectory with nested 120/360 validation-best results and distinct actual terminal Adam/RNG checkpoints. The legacy `audit_mlp_epochs=120` and `catchup_epochs=120` configuration fields describe the nested base budget; `extended_audit_epochs=360` specifies the executed trajectory. P's probability view is identical to its wire and is deduplicated. No cross-condition auditor is fitted or transferred.

Exact score/model replay needs the saved local person-level predictions, releases, fitted candidate preprocessing/weights, terminal states, original row identities and raw labels. Read-only table/figure regeneration needs the compact metric/selection/training evidence and referenced historical records, without raw data or models. The final validation report records the exact commands, hashes, counts and numerical tolerances used. A model/score replay is verification, not another scientific fit.

All outcomes remain DEVELOPMENT EVALUATION. PWGTP scores the same unweighted-validation-selected predictions. Finite utility matching, coefficient contrasts and Pareto sets do not select a deployable release, create confirmatory evidence, or repair absent RAC1P category support. The shared zero anchor is one fitted object per interface and seed, even when referenced by both comparison families.

## Read-only verification and report regeneration

Use a new output filename for each replay report; completed reports are not overwritten. These commands do not refit models or choose a new coefficient.

In a fresh checkout, first restore the exact non-person-level per-unit JSON evidence from its published gzip encoding. This verifies both encoded and original-byte hashes and refuses to overwrite differing files. It does not restore models or person-level arrays. The aggregate CSV exports may be read directly with gzip-aware tools.

```sh
.venv/bin/python scripts/compact_acs_coalition_strength_evidence.py --out results/redesign_20260909_acs_coalition_strength_v1 --restore
```

```sh
.venv/bin/python -m pytest -q tests/test_acs_coalition_strength_training.py tests/test_acs_coalition_strength_runner.py tests/test_acs_coalition_strength_comparisons.py tests/test_acs_coalition_strength_replay.py tests/test_acs_coalition_strength_reporting.py
.venv/bin/python scripts/verify_acs_coalition_strength_training.py --out results/redesign_20260909_acs_coalition_strength_v1 --report /tmp/strength_training_replay.json
.venv/bin/python scripts/verify_acs_coalition_strength.py --out results/redesign_20260909_acs_coalition_strength_v1 --report /tmp/strength_score_replay.json
.venv/bin/python scripts/summarize_acs_coalition_strength.py --out results/redesign_20260909_acs_coalition_strength_v1
.venv/bin/python scripts/verify_acs_coalition_strength_comparisons.py --out results/redesign_20260909_acs_coalition_strength_v1 --report /tmp/strength_comparison_replay.json
```

The publication uses deterministic gzip for large CSV/JSON score exports. For example, `gzip -dc UTILITY_MATCHES.csv.gz` recovers the complete fixed-pair CSV; `NONDOMINATED_POINTS.csv.gz` names every vector and observed configuration. Plain local versions are excluded from Git to avoid duplicate archives. Canonical per-condition compact scores remain public, while all fitted objects and person-level predictions stay local. The final source identity and validation records distinguish scientific execution code frozen before fitting from later read-only reporting/replay code.

[COMPACT_UNIT_EVIDENCE.json](COMPACT_UNIT_EVIDENCE.json) binds the 72 per-unit score/selection gzip files to their unchanged original completion hashes. Recompression is an exact publication encoding, not a scientific artifact regeneration. All original JSON files remain local and authoritative for replay.
