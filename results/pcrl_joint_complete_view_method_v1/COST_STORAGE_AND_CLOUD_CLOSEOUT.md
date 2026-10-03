# Cost, storage and cloud closeout

## Compute (laptop only)

| Item | Value |
|---|---|
| Start | 2026-10-03 19:09:09Z (`~/PCRL_eval_cache_private/jcv_v1/START.txt`) |
| Protocol lock pushed | 19:45Z (`b67664f`), before any real fit |
| Real fits | Warm starts 19:46Z; training and FARE 19:47–20:01Z; inner audit 20:12–20:14Z; selection and F0 20:14Z |
| Selection lock pushed | `197f323` |
| Outer scoring | 20:15–20:33Z, including the A2 rerun; inference 20:41Z |
| Worker CPU (logged by `/usr/bin/time`) | About 0.65 CPU-h for training, FARE, inner, selection, outer and controls |
| Other CPU | Fixtures, timing pilot, tests, inference, backup and the independent replay: about 0.3 CPU-h |
| **Total** | **About 1.0 CPU-h** of the 20 CPU-h ceiling; about 2 h elapsed of the 12 h ceiling |
| Workers / memory | At most 2 heavy workers (`OMP_NUM_THREADS=1`, torch 1 thread); peak RSS 0.73 GB (ceiling 8 GiB) |
| Model fits | 54 neural runs (3 warm + 51 arm runs, incl. E finalisation); 42 official FARE trees (36 grid + 6 F0); about 3,300 attacker / probe fits (inner + outer + controls) |

## Storage

| Location | Content | Size | Files |
|---|---|---|---|
| `~/PCRL_eval_cache_private/jcv_v1` | Inputs (corrected-contract npz), units (models, maps, heads, releases, inner/outer records, per-row assessment predictions), FARE cache, logs, inference | 1.4 GB | 2,085 |
| `<drive>/private_jcv_v1_20261003/jcv_v1` | Identical copy + `SHA256SUMS` + `BACKUP_RECORD.json`; verified 2,085/2,085 by uncached re-read (F_NOCACHE; not a cold-disk unmount) | 1.4 GB | 2,087 |
| Repository | Aggregate tables, code, tests, locks, replay script, figures | Small | — |

Laptop free space: 125 GiB (≥ 5 GiB floor). Nothing was deleted. The 87 pre-amendment unit versions are kept as `*.quarantined`.

## Cloud closeout

- **No cloud resource was launched.** Local compute was sufficient, so no AWS profile was used and no spending was incurred ($0 of the $50 cap).
- No task-created process is left running: the training, inner and outer workers all exited. The independent verifier ran locally and finished.
