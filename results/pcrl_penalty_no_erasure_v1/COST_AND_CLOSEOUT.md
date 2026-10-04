# Cost and closeout

## Compute (laptop only; $0 cloud)

| Item | Value |
|---|---|
| Start | 2026-10-03 23:34:59Z (`~/PCRL_eval_cache_private/pnx_v1/START.txt`) |
| Protocol lock pushed | `f1d1b76` (before any fit) |
| Parity (engineering) | About 95 s: U, PN and LN at β = 0 on 3 seeds, plus the JP replication |
| 18 new fits | About 3 min wall on 2 workers |
| Inner audit and selection | About 1 min |
| Selection lock pushed | `b0b9256` |
| Outer scoring (39 units) | About 12 min on 2 workers |
| Controls and critic-gap diagnosis | About 5 min |
| Inference | 83 s |
| Worker CPU (`/usr/bin/time`) | 0.40 CPU-h |
| **Total CPU** | **About 0.7 CPU-h** of the 12 CPU-h ceiling, including tests, inference, backup and three runs of the independent replay (about 137 s each); the elapsed work took about 1.5 h of the 8 h ceiling (start 23:35Z, closeout about 01:00Z) |
| Workers / memory | At most 2 heavy workers, `OMP_NUM_THREADS=1`; peak RSS 0.94 GB (< 8 GiB) |
| Disk | 125 GiB free (≥ 5 GiB) |
| Fits | 18 new encoder fits (plus 10 parity-check re-trainings — U, PN and LN at β = 0 on 3 seeds and one JP — not kept as units). Several thousand attacker and probe fits (inner + 39 outer units + controls; each outer unit fits the full slate on 12 views, with refits). Fresh critics for the diagnostic: 2 × 2 sets per view per unit. |

## Storage

| Location | Content | Size | Files |
|---|---|---|---|
| `~/PCRL_eval_cache_private/pnx_v1` | New units, critic snapshots, inner, outer and critic-gap records, per-row assessment predictions, logs, inference | 257 MB | 561 |
| `<drive>/private_pnx_v1_20261003/pnx_v1` | Identical copy + `SHA256SUMS` + `BACKUP_RECORD.json`; 561/561 verified by uncached re-read (not a cold unmount) | 257 MB | 563 |
| Reused predecessor data | `~/PCRL_eval_cache_private/jcv_v1` and its verified drive copy `private_jcv_v1_20261003` | — | — |

Nothing was deleted, and no older branch, result or lock was modified.

## Closeout

- No cloud resource was used, and no Docker or unrelated service was started.
- All owned worker processes have exited. The independent verifier ran locally and finished before the final commit (final run: 40 PASS, 0 FAIL, 0 WARN, 5 INFO). Four stale waiter shells left over from the predecessor study were stopped; no other process was touched.
