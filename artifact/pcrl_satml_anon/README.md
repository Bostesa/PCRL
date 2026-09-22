# Anonymous artifact — task-directed releases beside a fixed prediction service

Contents: the generators and verification scripts behind every table and figure, the claim ledger, the
correction log, the scope statement, and synthetic fixtures. All scripts are arithmetic or rendering only:
none fits a model, reads a survey pool or contacts a network.

```
code/       generators and the independent verification of the reported numbers
evidence/   claim ledger, corrections, scope, machine-readable verification output, asset hashes
tables/     the LaTeX tables used in the paper, as generated
figures/    the figures used in the paper, as generated
fixtures/   self-contained counterexample fixtures; each asserts its own conclusion and exits nonzero on failure
```

## Data

The study uses **public US Census American Community Survey (ACS) one-year PUMS microdata** for
California, ages 19–34. No individual-level records, fitted weights or private archives are included here.
To obtain the data: download the ACS 1-year PUMS person file for the state and year from the Census
Bureau's public PUMS distribution, or use the Folktables package, which wraps the same public files.

## Reproducing

```
python code/verify_task_directed.py   # recomputes every reported number from the pinned evidence
python code/build_final_assets.py     # regenerates the tables and figures
python fixtures/b1_label_leak_fixture.py
python fixtures/replacement_vs_append_fixtures.py
```

The verification and generator scripts resolve committed evidence by commit hash through a Git object
store. Given only this archive, the recorded SHA-256 values in `evidence/FINAL_ASSET_HASHES.json` and
`evidence/TASK_DIRECTED_VERIFICATION.json` let a reader check that the shipped tables and figures match the
evidence they were generated from; running them against the source repository requires that repository.
The fixtures are fully self-contained and run anywhere.

## What is not here, and why

Fitted model weights, per-person predictions and internal coordination records are excluded: they are
individual-level or operational rather than evidentiary, and none is needed to check a reported number.
This archive is not a claim that every private archive has been released.
