# Runtime and storage

Laptop only: 14-core Mac, 24 GB. No cloud and no GPU. OMP_NUM_THREADS = 1, one scheduler.

## Runtime

| Stage | Wall | CPU (user + sys) | Peak RSS |
|---|---|---|---|
| Execute: 26 units, 2,902 model fits | 684 s | 676 s (0.19 CPU-h) | 0.51 GB |
| Inference: all bootstraps incl. primary B = 20,000 | 124 s | 122 s | 0.89 GB |
| Resume check: 26 units hash-verified and skipped | 2.6 s | — | — |
| Pre-fit synthetic calibration estimate | — | ≈ 0.26 CPU-h | — |

The total scientific CPU time was about 0.22 CPU-h, against the 2 CPU-h allowance. Repairs and synthetic validation
are not counted against it.

## Model fits by recipe

The counts come from `n_model_fits` in each unit's `fit_records.json`.

| Fits | Count |
|---|---|
| rep L / GBT / MLP | 130 / 520 / 468 |
| outputs L / GBT / MLP | 40 / 160 / 144 (8 untreated units; reused for 18 noise units) |
| rep+outputs L / GBT / MLP | 130 / 520 / 468 |
| LO | 26 |
| U2 | 130 |
| G1 | 26 |
| G2 | 78 |
| ρ₁² | 26 |
| LRT-A2 / LRT-A4 | 18 / 18 |
| **Total** | **2,902** |

## Storage

All private. Raw rows, per-person predictions and fitted models are never committed.

| Location | Contents | Size |
|---|---|---|
| `~/PCRL_eval_cache_private/pilot_adult_s0/run_v1/` | inputs, 26 unit dirs (`preds.npz`, `models/`, `fit_records.json`, `supported.json`, `COMPLETE.json`), inference, logs | 178 MB, 551 files |
| Drive copy `YOTUO:NathanSamson-Mac-relocated-2026-09-30/private_pilot_run_v1_20261002/` | `run_v1` + features, labels, forward cache, all noise releases, checkpoints | 417 MB, 612 files |

The drive copy was verified file by file with uncached reads (F_NOCACHE): **612/612 sha256 match**. Its
`SHA256SUMS` is on the drive.

`PRIVATE_ARCHIVE_INDEX.json` in this directory lists paths, sizes and sha256 only, with no values.
