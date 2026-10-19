# Quickstart: controlled training-size campaign (PCRL, 2026-10-19)

## Interpreters

| Name | Path (relative to the repository) | Versions | Used for |
|---|---|---|---|
| PY | `python3` (Homebrew 3.14.3) | numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0 | A1_PO_16 fits, correction base, nested witnesses, rules, evaluation, selection, deployment |
| TORCH | `.venv/bin/python` | numpy 2.4.2, scikit-learn 1.8.0, torch | Stage B TASK_REFIT control, correction directions, nested corrections, development readers (grid lock) |
| CATBOOST | `.private/pcrl_correction_transfer_stability_v1/venv_catboost/bin/python` | catboost 1.2.8 | T_TREE teachers (control and routing; inference is numpy) |

Every real-data numerical step runs through the shared queue: `python3 -m tsz.jobqueue run --unit <ID> --stage <stage> --rss-gib <g>
-- <interpreter> -m <module> ...` (two heavy jobs at most, one thread each, job-start code hashes, append-only ledger).
`python3 -m tsz.lane LANE.json` runs a list of units in order (interpreter aliases PY / TORCH / CATBOOST). Labels are read only through
`tsz.common.load` (role x field x operation x purpose; authorized entry points in `ACCESS_POLICY.json` and `ACCESS_AMENDMENTS.json`);
nothing is readable before the registration receipt (`tsz.receipt verify`).

## Stage order

1. Registration (label-free): `tsz.admit run`, `tsz.subsets manifest`, `<TORCH> -m tsz.devreaders manifest` and `lock`, the rules
   (`python3 -m tsz.test_rules`), commit, `tsz.publish_build build / commit`, push, `tsz.receipt verify --kind registration`.
2. Size fits (per subset SID in S50_O0..O3, S70_O0..O3, S100 and version V in 0, 1):
   `tsz.sizefit fit --subset SID --version V` (PY), `verify`, `repro --version V` (S100 only);
   control: `<CATBOOST> -m tsz.control teachers --subset SID --version V`, `<TORCH> -m tsz.control stage_b ...`, `tsz.control verify`,
   `repro`; encoding sensitivity: `tsz.sizefit fit --subset S50_O<o> --version 0 --dictionary admitted`.
3. Trigger: W100 packages (`tsz.deploy store / build / check`), the verifier's `roles/verifier/DEPLOY_W100.json`, then
   `tsz.evaluate trigger` (once).
4. Size analysis: `tsz.evaluate size` (SIZE_TABLE.csv, SIZE_RESULTS.json).
5. Conditional stage (per version V): `tsz.correct base`, `<TORCH> -m tsz.correct direction --arm CONT|LOC|PAIR`, `tsz.correct releases`,
   `verify`; `tsz.nested witness --folds 0-4 / 5-9`, `<CATBOOST> -m tsz.nested routing`, `<TORCH> -m tsz.nested correct --arm LOC|PAIR`;
   `tsz.select run` (commit the selections before any MODEL_SELECT read of a corrected release); descriptive `tsz.nested correct --arm CONT`.
6. Readers and labels: `tsz.banks conditional`; `<TORCH> -m tsz.devreaders run --spec grid_specs/COND.json --seed 0|1`;
   `<TORCH> -m tsz.seedcheck run --spec grid_specs/COND.json`; `tsz.evaluate conditional`; `tsz.claims run`.
7. Deployment of the final packages: `tsz.deploy store / build / check` (packages under `.private/<campaign>/deploy/`).
8. Closeout: `tsz.ledger`, `tsz.closeout`; backup with `tsz.custody backup <parent>` and, on an already attached drive,
   `tsz.custody offdevice <backup> <dest>`; `tsz.publish_build build / commit`.

## Deployment from the six raw fields

```python
from tsz import deploy as D
out = D.release('<package dir>', rows)          # rows: list of dicts with exactly AGEP, WKHP, COW, SCHL, MAR, POBP (in this order)
out['income']['probability'], out['income']['decision'], out['occupation']['probability'], out['occupation']['decision']
```

`D.release_csv(pkg, path)` reads a CSV with exactly those six columns. Missing, extra, reordered, target (PINCP, OCCP, ...) and SEX
fields are refused at the wrapper and at `D.underlying`. Only probability vectors and argmax decisions are returned. Income is always the
frozen H3 income of the package's version; the occupation output is the package's (`A1:<fit id>`, `CTRL:<SID>__V<v>`,
`CORR:<ARM>@s<x>`). Packages pin the hashes of the code they execute; an edited module refuses to load until the package is rebuilt.
