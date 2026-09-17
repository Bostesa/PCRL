# RUN_STATUS

Machine: local Apple CPU, 14 cores, 24 GB. **One worker and one BLAS/OpenMP thread
throughout.** Unrelated user workloads (a VM, Docker, a browser and several other
processes) shared the machine and swap was at or near capacity for the whole run;
parallelism was deliberately never raised. No paid or remote compute.

## Measured time

| Stage | Measured |
|---|---|
| Rank diagnostic (reads saved eigenvalues only) | < 1 s |
| Fit 36 conditions x 3 seeds (27 unique new fits) | 1324 s |
| Mechanism diagnostics (rotation + nuisance calibration) | 610 s |
| 2018 development audits and scores (27 new units) | 1106 s |
| Independent prediction replay (27540 arrays) | 211 s |
| Report, bootstrap and decisions | 54 s |

Peak resident set size during fitting: **647 MB** per worker.

## Fit ledger

| Quantity | Count |
|---|---:|
| Nominal interface records | 54 |
| Unique new fits | 27 |
| Reused historical conditions (never refitted) | 9 |
| Void by alias (r_plus == 16, neither fitted nor audited twice) | 18 |
| Deterministic starts per new nonlinear condition | 2 |
| Eligible checkpoints per new nonlinear condition | 4 |

A cache read is never counted as a new fit. Resumed units are counted once.

## 2018 audit fit ledger

| Unit | Count |
|---|---:|
| fresh_mlp_trajectories | 324 |
| kernel_feature_fits | 162 |
| kernel_ridge_solves | 486 |
| new_five_candidate_roles | 162 |
| own_catchup_trajectories | 0 |
| static_fits | 486 |

## Failures and repairs

Recorded so the run is auditable:

1. The first timing calibration showed ~5 min per optimiser start. Profiling found
   `chi_r(VW)` being rebuilt once per protected role although it depends only on
   `W`, and the role row-weights being rebuilt inside every row chunk. Both were
   hoisted; the fixtures were re-run and still pass, and one start now costs ~1 min
   at rank 16. This changed cost only, not any number: chunk-size invariance is a
   standing fixture.
2. **A concurrency incident, caused by this session's orchestration and not by the
   pipeline.** Two queued waiters fired on the same sentinel file and ran
   `seed_0/spectral_nlr16_C1` simultaneously. One process completed in 43.0 s; the
   other crashed on `FileExistsError` while creating
   `fitted/utility/A/income_binary/logistic`. The unit was **quarantined**
   unchanged (`spectral_nlr16_C1__concurrent_write_*/QUARANTINE.json`) and re-run by
   a single process in 43.1 s, because the completed transport study had seen concurrent
   writes produce invalid probabilities. An independent comparison of the two then
   showed **0 of 1020 prediction arrays and 0 of 2040 log losses differing** — the
   historical `mkdir(exist_ok=False)` guard did its job, the crashed process wrote
   nothing conflicting, and the quarantine cost 43 s and changed no number. Only the
   clean unit is used in any table; nothing from the quarantined copy is merged,
   averaged or voted with it. Fixed by never duplicating a waiter on one sentinel.
3. A rotation-diagnostic test initially asserted a principal angle below 1e-8 and
   measured 2.1e-8. `arccos` of a singular value near 1 has unbounded derivative,
   so a 1e-16 rounding error becomes ~1e-8 in the angle. The well-conditioned
   projector Frobenius distance is now the assertion (< 1e-10) and the angle keeps a
   descriptive 1e-6 tolerance. A tolerance was corrected, not a measure.

## Amendments to the registered protocol

Recorded with their timing, per the protocol:

* **After** the fit phase began and **before** any 2018 score was read, Terminal B
  delivered an independent mathematical review. Its finding A2 (a penalty nonlinear
  in `Z` is provably not a trace form, because a trace form is invariant under
  `W -> W Q` and the nonlinear feature map is not) prompted one **addition**: the
  rotation decomposition in `diagnostics.py`, plus the held-out nuisance calibration
  diagnostic its finding A6 asked for. Neither changes the frozen matrix, any
  endpoint, any decision rule or any selection; both are diagnostics computed from
  already-fitted maps. No registered prediction was altered.
