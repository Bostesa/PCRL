# RUN_STATUS — locked 2017 transport of the fixed 14-interface slate

Updated: 2026-09-17, end of run. Machine: local Apple CPU (14 cores, 24 GB), single numerical thread per worker. No paid or remote compute. Unrelated user workloads (a VM, Docker, two other Python jobs) shared the machine throughout; swap was near capacity (29.8/30.7 GB).

## State: complete

| Stage | Status | Measured time |
|---|---|---|
| Prospective protocol + comparison spec | committed `9be345a` before any 2017 fit | — |
| 2018 pipeline identity replay | bitwise equal, 3/3 seeds, 7 pools, 57–59 arrays each | 8 s/seed |
| 2017 fit/validation releases (4 partitions x 3 seeds x 14 interfaces) | complete | 65 s |
| Mode B fitting + selection (45 units: 42 interfaces + 3 reference) | 45/45 complete, 0 failures | ~50 CPU-min (8 workers, ~12 min wall) |
| Dry run of scoring/report on a validation partition | complete (code exercise; caught 2 defects) | ~12 min |
| TRANSPORT_LOCK.json (17,639 hashed inputs) | committed and pushed `3e31f28` before the first final read | 8 s |
| Final scoring, Modes A and B (45 units) | first pass 42/45 then rescored 45/45 with confirmed predictions | ~10 min then ~35 min (4 workers) |
| Prediction replay (independent processes) | 59,874 stored predictions, 0 mismatches | 20 min |
| Validation-score audit | 37,122 predictions, 0 selection changes | 6 min |
| Report (bootstrap, families, decisions, 10 figures) | complete | 6.5 min |
| Independent verification (15 checks) | all pass | 23 s |

## Unique fits (this study, Mode B only; no representation, service or historical channel was refitted)

| Unit | Count |
|---|---:|
| Five-candidate attacker roles (logistic, 2 restarted MLPs, 2 boosted trees) | 339 |
| Random-feature kernel-ridge fits (3 regularizations each) | 339 |
| Saved-observer catch-up trajectories | 102 |
| Utility probe pairs (logistic + MLP) | 132 |
| Reference probe pairs (PCA32, rich bank, tree bank) | 21 |

Counts are read from completion files by `scripts/report_acs_spectral_transport.py` (`unique_fits` in `TRANSPORT_DECISION.json`); resumed units are counted once.

## Failures and repairs (all documented, no completed unit lost)

1. Reference-probe hash check compared float32 against the fitter's float64 view — check bug, fixed before fitting.
2. Spectral arms have no saved observer; missing guard — fixed, unit re-run.
3. Three seed-2 Mode B units aborted during the first final scoring with an invalid-probability error (concurrent writes/reads of one final-release file). Fix: atomic writes, releases built once in the parent. Lock amendment 1.
4. An independent replay then found 5 stored predictions (of 56,836) wrong, and a later validation-only audit produced one corrupted array, under extreme machine memory pressure. Fix: every prediction is computed twice and must agree; all final units rescored with 4 workers; originals preserved under `superseded_nondeterminism_20260917/`. Lock amendment 2. Post-fix replay: 0 mismatches in 59,874 predictions; validation audit: 0 selection changes.

## Reproduce

See `REPRODUCE.md` for the exact command sequence, required local objects and software versions.
