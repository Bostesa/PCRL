# Quickstart: correction-transfer campaign (PCRL, 2026-10-18)

## Interpreters

| Name | Path (relative to the repository) | Versions | Used for |
|---|---|---|---|
| PY | `python3` (Homebrew 3.14.3) | numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0 | witnesses, ladders, rules, evaluation, deployment |
| TORCH | `.venv/bin/python` (3.13.7) | numpy 2.4.2, scikit-learn 1.8.0, torch 2.10.0 | correction directions, Track R Stage B, development readers (grid lock) |
| CATBOOST | `.private/pcrl_correction_transfer_stability_v1/venv_catboost/bin/python` (3.13.7) | numpy 2.4.2, catboost 1.2.8 | routing teachers only (inference is numpy) |

Every real-data numerical step runs through the shared queue: `python3 -m cts.jobqueue run --unit <ID> --stage <stage> --rss-gib <g>
-- <interpreter> -m <module> ...` (two heavy jobs at most, one thread each, job-start code hashes). `python3 -m cts.lane LANE.json`
runs a list of units in order. Labels are read only through `cts.common.load` (role x field x operation x purpose; authorized entry
points in `ACCESS_POLICY.json` and `ACCESS_AMENDMENTS.json`).

## Stage order

1. `cts.partitions support`; `cts.preflight replay`; `cts.trackf routing_repro --seed 0`.
2. Per partition P in D0, D1, R0, R1 (and the sensitivity alias D0B): `cts.trackf witness`, `cts.trackf routing` (CATBOOST),
   `cts.evaluate witness`, `cts.trackf direction --routing REFIT|FROZEN --arm PAIR|LOC|CONT` (TORCH), `cts.trackf ladder`,
   `cts.select run`.
3. Nested variant on D0: `cts.nested witness` (PY), `routing` (CATBOOST), `correct --arm PAIR|LOC` (TORCH).
4. Track R on R0, R1: `cts.track_r teachers` (CATBOOST), `stage_b` (TORCH), `income` (TORCH); `cts.evaluate trackr`;
   `cts.trackf direction --income TRACKR`.
5. Recipe selection: `cts.dstage gates`, `bank`, reader grid, `select`. Then `cts.freeze write` and its receipt.
6. After the freeze: `cts.select check`, `cts.trackf ladder --check`, `cts.questiona run` (default `--scheme registered`, one household
   resample per set; `--scheme superseded` reproduces the first records only), `cts.compression fit / realized` on R0, R1.
7. Readers: `cts.devreaders run --spec grid_specs/<bank>.json --seed 0|1` (TORCH); screen study `cts.screen_study`; whole-release
   utility and reserve of every bank member: `cts.evaluate bank --bank-id <bank>` (PY).
8. Tables and figures (label-free): `cts.diagnosis` (cost transfer and factors), `cts.diagnosis qa` (question-A labels, primary and
   superseded), `cts.tables stability --bank-id <bank>`, `cts.claims all` (repair rule comparison, matched controls, reserve-qualified
   frontier, original benchmark, claim ladder, frontier drivers), `<TORCH> -m cts.figures cost | frontier`.
9. Deployment: `cts.deploy store`, `build`, `check` (packages under `.private/<campaign>/deploy/`; the P70 task witness is packaged the
   same way with `--partition P70 --name W`).
10. Closeout: `cts.ledger`, `cts.closeout`; backup with `cts.custody backup <parent>` and, on an already attached drive,
    `cts.custody offdevice <backup> <dest>`; `cts.publish_build build / commit`.

## Deployment from the six raw fields

```python
from cts import deploy as D
out = D.release('<package dir>', rows)          # rows: list of dicts with exactly AGEP, WKHP, COW, SCHL, MAR, POBP (in this order)
out['income']['probability'], out['income']['decision'], out['occupation']['probability'], out['occupation']['decision']
```

`D.release_csv(pkg, path)` reads a CSV with exactly those six columns. Missing, extra, reordered, target (PINCP, OCCP, ...) and SEX
fields are refused at the wrapper and at `D.underlying`. Only probability vectors and argmax decisions are returned.
