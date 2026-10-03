# Runtime and storage

**Machine:** the laptop only (Apple M4 Pro, 14 cores, 24 GB). No cloud resources were used.

**Execution settings:** single-threaded, with `OMP/MKL/OPENBLAS/VECLIB_MAXIMUM_THREADS=1` and `PYTHONHASHSEED=0`.

**How CPU was measured:**
- CPU-s is user + sys, from `/usr/bin/time -l` for each stage process.
- "Ledger" CPU is what the budget rule counted: the sum of per-unit `time.process_time`.

## Stages

| Stage | Wall | CPU (process) | Ledger CPU | Peak RSS | Notes |
|---|---|---|---|---|---|
| Lock build (final) | ~20 s | — | — | — | 2026-10-02 20:57:48Z, commit 3274fe1, pushed before any fit |
| Shuffled-label sanity (6 arm-A units) | 47 s | 46 s | 44 s | — | fit/val only; AUC 0.47–0.51, no flag |
| Tier 1, 126 units | 4,786 s (1 h 20 m) | 4,765 s (1.32 h) | | 0.74 GB | 20:59:19Z → 22:19:05Z |
| σ\* | 2.3 s | 1.8 s | 0.1 s | — | attacker_val only |
| Tier 2, 309 units (E1 72, E2 36, E3 201) | 11,026 s (3 h 04 m) | 10,974 s (3.05 h) | | 0.79 GB | 22:19:24Z → 01:23:11Z; budget stop |
| Units total (Tier 1 + Tier 2) | | | 15,698 s (4.36 h) | | |
| Inference (all 435 units) | 2,304 s (38 m) | 2,290 s (0.64 h) | 2,289 s | 2.86 GB | B = 20,000 primary, 2,000 exploratory |
| Report + plots | ~10 s | | | | 29 plots |
| **Science total** | **≈ 5.1 h** | | **18,031 s = 5.01 CPU-h** | | ceiling 8 CPU-h |

**Budget stop** (from `logs/BUDGET_LEDGER.json`): spent 15,743 s + next unit 27 s + inference reserve 13,080 s
(30 s × 436 units) = 28,850 s, which exceeds the budget of 28,800 s.

The realised inference cost (2,289 s) was far below the reserve. The reserve was a frozen ceiling, set before the
run, and was not revised afterwards.

**Calibration vs actual:**

| | Projected | Actual |
|---|---|---|
| Tier 1 | 1.57 CPU-h | 1.32 CPU-h (plus its share of inference) |
| E1 + E2 | 1.33 CPU-h | about 1.6 CPU-h |

HMDA units ran slower than calibrated: arm A took 70–110 s.

**Resume test:** `STAGE=tier1 RESUME=1` skipped 126/126 hash-verified units with 0 fits and the lock check OK, in
5.2 s.

## Fits

| Item | Count |
|---|---|
| Official LEACE maps (concept-erasure 0.2.4) | 60 (42 target maps "B", 18 policy-set maps "C"), each fitted once on defense_fit and pinned in `defenses/MAP_PINS.json` |
| Model fits logged by units | 44,824 |
| Shared fits, counted once at first use | 42 outputs-only stores, 240 U2 probe stores |
| Units completed | 435 of 882 registered (126 Tier 1, 309 Tier 2); 447 E3 units not run (budget); 0 aliased |

The model-fit count includes:
- every hyper-parameter grid point (L 5, GBT 20, MLP 18);
- attacker-seed refits;
- the closed-form G1/G2/ρ₁ checks;
- label-only references;
- shared outputs-only and U2 fits.

## Storage

| Location | Size | Files |
|---|---|---|
| `~/PCRL_eval_cache_private/bench_v1/` (laptop, private) | 5.4 GB | 9,764 (at backup time) |
| ├ `units/` (per-unit predictions, fitted attackers and probes) | 4.6 GB | |
| ├ `inputs/` (rows, roles, labels, forward caches) | 665 MB | |
| ├ `shared/` (outputs-only, U2) | 189 MB | |
| ├ `infer/` (BENCH_INFER.json, SIGMA_STAR.json) | 15 MB | |
| ├ `defenses/` (LEACE maps) | 3.3 MB | |
| └ `logs/`, `support/`, `sanity/` | < 1 MB | |
| Drive copy `<drive>/private_bench_v1_20261003/` | 5.4 GB | 9,764 + SHA256SUMS + README |
| Public package `results/combined_matched_removal_benchmark_v1/` | 11 MB | aggregates, code, plots; no rows, predictions or models |

**Backup:**
- The drive copy was made with `ditto` (5 min 41 s).
- Every file was re-read with F_NOCACHE: **9,764/9,764 sha256 match at backup; 9,765/9,765 after the log re-sync**.
- Restoration of a LEACE map and two attackers from the drive reproduces the saved outputs exactly. See
  `notes/execution/BACKUP_AND_RESTORE.json`.
- Nothing was deleted: the laptop copy is kept for replay.
- After the backup, the resume test and the replay added small log files under `logs/`. These are regenerable and
  were re-synced to the drive before handoff; see VALIDATION.md.
