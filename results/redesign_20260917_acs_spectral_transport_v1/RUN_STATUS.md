# RUN_STATUS — locked 2017 transport of the fixed 14-interface slate

Updated: 2026-09-17T16:08Z. Machine: local Apple CPU (14 cores, 24 GB; two unrelated user jobs occupy ~1.3 cores). No paid compute.

## Completed (reused, not rerun)
- Development study `redesign_20260910_acs_residual_spectral_v1` @ ad2c088: 24/24 spectral maps, 42/42 units, independent replay passed. Map hashes: `seed_*/maps_complete.json` of that study.
- 2017 admission and household partition (same commit).

## This study
| Stage | Status |
|---|---|
| Prospective protocol + COMPARISONS.json | written, PROTOCOL_FREEZE.json |
| Pipeline identity replay on 2018 | done: bitwise equal, 3/3 seeds, all 7 pools (`seed_*/REPLAY_2018.json`) |
| 2017 fit/validation releases | done (4 partitions x 3 seeds x 14 interfaces), ~65 s |
| Mode B fitting + selection | running: 45 units (42 interfaces + 3 reference), 8 workers; calibration H unit 90 s single-threaded, projection ~50 CPU-min |
| TRANSPORT_LOCK.json commit | pending |
| Final scoring (Modes A, B) | pending |
| Bootstrap + reports | pending |

Failures (repaired, no completed unit lost): reference-probe hash check compared float32 instead of the fitter's float64 view (check bug); spectral arms lacked an observer-dictionary guard (first attempt fitted one role, reused on resume after hash verification).

Next executable commands (from the worktree):
```
python -m experiments.run_acs_spectral_transport --phase fit --workers 8      # resumable
python -m scripts.dry_run_acs_spectral_transport --dest <scratch>              # code exercise on a validation partition
python -m experiments.run_acs_spectral_transport --phase lock
python -m experiments.run_acs_spectral_transport --phase score --workers 8   # only after the lock commit is pushed
python -m scripts.report_acs_spectral_transport
```

Compute accounting: 0 new-year fits, 0 new-year scores.
