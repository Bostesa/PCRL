# Quickstart

Run everything from the repository root with `PYTHONPATH=.:<private>/pylib`.

**Environment:** Python 3.13 with torch 2.10, scikit-learn 1.8, numpy 2.4 and scipy 1.17, plus CatBoost 1.2.8 and openpyxl installed into the private pylib.

**Private files.** The study-specific private files are the raw ACS copy, the role exports, fitted weights and per-person predictions. They live under the git-ignored `.private/acs_replication/` and are never committed or published.

## Deploy a frozen service on raw ACS fields (no labels)

**Input:** a CSV whose header is exactly `AGEP,WKHP,COW,SCHL,MAR,POBP`, in that order. Values are coded as in the 2019 PUMS dictionary, e.g. SCHL `21`, POBP `012`.

```
PYTHONPATH=. python -m acs_replication.deploy --package <private>/deploy/LOCAL_TREE_S0 --input people.csv --output released.npz
```

**Output:** `income_probability` (n × 2), `income_decision`, `occupation_probability` (n × 5), `occupation_decision`. Nothing else is released.

**Refusals:**
- reordered or extra columns;
- a forbidden column (for example SEX, RAC1P, OCCP, PINCP);
- any request for hidden outputs (leaf ids, raw scores).

**Available packages:** `T_TREE_S0`, `T_TREE_S1`, `LOCAL_TREE_S0`, `LOCAL_TREE_S1`, `NEURAL_RAW_rho1.0_E2_S0`. They are listed with hashes in `MODEL_MANIFEST.json` and verified in `DEPLOYMENT_RESULTS.json`.

## Re-derive the public tables from the frozen report

```
PYTHONPATH=. python -m acs_replication.tables results/pcrl_acs_protected_tree_replication_v1
```

## Reproduce the study end to end (development roles only)

Each step runs through the campaign queue, `python -m acs_replication.jobqueue run --unit <ID> --stage <stage> --rss-gib <r> -- <command>`.

1. `python -m acs_replication.admission --out …/METADATA_ADMISSION.json`: downloads the official zip and checks hashes. It reads identifiers only.
2. Commit `SCIENCE_LOCK.json`. Then run `python -m acs_replication.extract --lock … --lock-sha … --admission … --occupation-map … --out …`.
3. `python -m acs_replication.stage_a --seed {0,1}`, then `python -m acs_replication.stage_a_report`.
4. `python -m acs_replication.stage_b --seed {0,1} --arms TASK_REFIT,LOCAL_TREE,JOINT_TREE`.
5. Run the controls: `neural_raw_worker`, `fagtb_worker --grid FAGTB_GRID.json` and `fare_worker`, each with `--seed {0,1}`. FARE needs the pinned official FARE environment used by `oar.fare_official`.
6. `lock_stage training`, commit, then `selection`.
7. Commit `SELECTION_LOCK.json`, then run `readers --seed {0,1}`.
8. `lock_stage reporting`, commit, then `report`. The report opens DEV_REPORT truth once; a second opening is refused.

**The sealed assessment is not reproducible from this package by design.** It is never exported.

## Custody

```
python -m acs_replication.custody snapshot --tag <TAG>
python -m acs_replication.custody restore --tag <TAG> --restore-tag <R>
python -m acs_replication.custody offdevice-copy --tag <TAG> --dest /Volumes/<DRIVE>/PCRL_private_backups/pcrl_acs_protected_tree_replication_v1
python -m acs_replication.custody restore-from --src /Volumes/<DRIVE>/PCRL_private_backups/pcrl_acs_protected_tree_replication_v1/<TAG> --restore-tag <R2>
python -I <restore>/code/acs_replication/custody_check.py <restore> <restore>/RESTORE_USE_CHECK.json
```

The last command is the label-free restore-use check. All five steps were done for `SNAPSHOT_V2`, including the off-device copy and the restore from it (`BACKUP_VERIFICATION.json`).
