# Quickstart: SCM utility-matched follow-up

Run from the repository root with

```
PYTHONPATH=.:<source private pylib>
```

Environment: Python 3.13, numpy 2.4, scipy 1.17, scikit-learn 1.8, and CatBoost 1.2.8 from the source study's private library (imported
by the source modules; nothing is retrained here). Every numerical unit runs through the shared queue (two slots, one thread per job,
10 CPU-hours and 6 elapsed hours):

```
python -m scm_um.jobqueue run --unit <ID> --stage <stage> --rss-gib <r> -- <command>
```

## Reproduce (development roles only; the sealed assessment is never exported)

The source benchmark's private tree (`.private/pcrl_scm_engineering_benchmark_v1`) and the protected-tree replication's role exports
must be present; they are admitted by hash, never modified.

1. `python -m scm_um.admit`: verifies the source tip, the public package and 12 manifests; copies 51 artifacts by hash; recomputes the
   runtime routing and checks that it reproduces all 31 historical arm outputs bitwise. Writes `SOURCE_ADMISSION.json`.
2. `python -m scm_um.make_lock`: writes `UNIT_INVENTORY.json` and the pre-fit `ANALYSIS_LOCK.json`. Commit them.
3. `python -m scm_um.trajectory --seed {0,1} --out .private/pcrl_scm_utility_matched_v1/runs/TRAJ_{s}`: the ten fixed trajectories per
   seed, the historical reproduction gate, every registered checkpoint and the effective-release tables.
4. `python -m scm_um.locks trajectory`: commit `TRAJECTORY_COMPLETE_LOCK.json` and `TRAJECTORY_SUMMARY.json` before any MODEL_SELECT label.
5. `python -m scm_um.select --out results/pcrl_scm_utility_matched_v1/SELECTION_LOCK.json`: MODEL_SELECT clauses, guards, expenditures and
   proxies for every effective release; nomination, matching and comparator selection. Commit.
6. `python -m scm_um.bank --out results/pcrl_scm_utility_matched_v1/SERVICE_BANK_MANIFEST.json`. Commit.
7. `python -m scm_um.readers --seed {0,1} --out .private/pcrl_scm_utility_matched_v1/runs/READERS_{s}`.
8. `python -m scm_um.locks reporting`: commit `REPORTING_LOCK.json` and `READERS_SUMMARY.json`; publish the reporting receipt.
9. `python -m scm_um.report --out results/pcrl_scm_utility_matched_v1/report`: the one logged coordinator re-read of DEV_REPORT truth.
10. `python -m scm_um.deploy --out-root .private/pcrl_scm_utility_matched_v1/deploy --report results/pcrl_scm_utility_matched_v1/DEPLOYMENT_RESULTS.json`.
11. `python -m scm_um.tables`: `UTILITY_TABLE.csv`, `RECOVERY_TABLE.csv`, `PRIMARY_ENDPOINTS.csv`, `SELECTION_AND_MATCHING.md`, `UTILITY_RECOVERY_FRONTIER.png`.

Dry runs on invented labels (no real label file is opened): `python -m scm_um.dryrun select <out.json>` and
`python -m scm_um.dryrun report <out_dir>`.

## Deploy a package on raw ACS fields (no labels)

Input: a CSV whose header is exactly `AGEP,WKHP,COW,SCHL,MAR,POBP`, in that order, coded as in the 2019 PUMS dictionary.

```
python -c "from scm_um import deploy as D; import numpy as np; \
r = D.release_csv('<package dir>', 'people.csv'); \
np.savez('released.npz', **{f'{t}_{k}': v for t, o in r.items() for k, v in o.items()})"
```

or, from Python rows (each row a dict with exactly those six keys, in that order): `D.release('<package dir>', rows)`.

Output: `income_probability` (n × 2), `income_decision`, `occupation_probability` (n × 5) and `occupation_decision`. In service each
task's output goes only to its own recipient.

**Refusals** (both entry points): extra, missing or reordered columns or keys; any forbidden column (SEX, for example); any request for
anything but probability and decision (cell or group ids, teacher probabilities, propensity).

## Tests (invented data only)

```
python -m scm_um.test_trajectory_synthetic
python -m scm_bench.test_engine_synthetic
python -m structural_pilot.test_lookup_synthetic
```

## Custody

`scm_um/custody.sh backup <dest parent>` writes a branch bundle, a source archive and the private tree, then restores them into a fresh
directory and re-runs the packages and the synthetic tests from the backup alone; `scm_um/custody.sh offdevice <backup dir> <dest>`
repeats the check on another device. Set `SCM_PYLIB` to the private CatBoost library and `SCM_PYTHON` to the interpreter.
