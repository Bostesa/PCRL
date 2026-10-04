# Cost and closeout

## Compute (laptop only; $0 cloud; no cloud resource created; no Docker or unrelated service started)

| Item | Value |
|---|---|
| Start | 2026-10-04 04:46:16Z (`~/PCRL_eval_cache_private/rgj_v1/START.txt`) |
| Ceilings | 10 h elapsed (watchdog armed for 14:46Z, stopping only this study's `rgj` processes), 20 CPU-h, 2 heavy workers, < 8 GiB, ≥ 5 GiB free, $0 |
| Pre-fit lock pushed | `d7367e0` at 05:22Z, after the math review (R1 and R2 fixed) and two engineering parity rounds whose receipts are kept and quarantined |
| Parity (final, under the lock) | About 3 min on 2 workers; 21/21 bitwise |
| Task line, Stage B, inner audit, Stage B selection | 05:25–05:31Z |
| Stage B freeze pushed | `2be8cea` at 05:31:45Z |
| Calibration, Stage C, baselines, inner audit, Stage C selection | 05:31–05:45Z |
| Tracking, whitening, optional ablation | 05:46–05:49Z |
| EVALUATION_LOCK pushed | `3affa49` at 05:49:43Z |
| Assessment (39 units), controls, inference | 05:49:56–06:03:50Z |
| Backup (drive) with uncached read-back and restore checks | About 06:06–06:10Z |
| Timed worker CPU (`/usr/bin/time`, 28 processes) | **1.23 CPU-h**; peak RSS 1.21 GB |
| Independent replay | 3 runs of about 4 min each (one by the verifier agent, two by the lead with the verifier's unchanged script); about 0.2 CPU-h; peak RSS 2.7 GB |
| Untimed work | Tests, engineering checks and agents' synthetic fixtures; estimated at under 0.5 CPU-h |
| **Totals** | **About 2 CPU-h of the 20 CPU-h ceiling; about 1 h 50 min elapsed of 10 h** (04:46Z to closeout at about 06:35Z). The final 2-hour reserve was never needed. |
| Disk | 134 GiB free on the system volume throughout |

## Storage

| Location | Content | Size | Files |
|---|---|---|---|
| `~/PCRL_eval_cache_private/rgj_v1` | Units (models, critics, transforms, releases, heads, inner, tracking and outer records, per-row assessment predictions), logs, selections, inference | 2.25 GB | 3,062 at backup |
| `<drive>/<relocation folder>/private_rgj_v1_20261004/rgj_v1` | Versioned copy, `SHA256SUMS`, `BACKUP_RECORD.json`. 3,062/3,062 files re-read uncached; U, J-G and L-G restored bitwise from the drive copy; one recorded attacker refit reproduces its saved predictions exactly | 2.25 GB | 3,064 |
| Reused predecessor data | `~/PCRL_eval_cache_private/jcv_v1` (inputs, warm starts, FARE trees) and its verified drive copy `private_jcv_v1_20261003` | — | — |

Working units were kept on the internal disk for reliability during unattended runs, and the versioned, verified drive copy was made at closeout. Nothing was deleted. Superseded parity receipts were renamed `*.quarantined_*`.

## Closeout

- **Owned processes.** All study workers have exited. The watchdog is stopped at closeout (see the final report).
- **Private files that changed after the backup.** Two files changed: the activity log, appended by the tested QUICKSTART commands, and the backup log. Both were synced to `<drive>/<relocation folder>/private_rgj_v1_20261004/post_backup_sync/20261004T062830Z/` with checksums and an uncached read-back (2/2 match).
- **Privacy note.**
  - The first backup manifests and SOURCE_INDEX.json named the drive's local folder, which contains a personal name.
  - The files were corrected in place, and the closeout code fixed (amendment A4).
  - Earlier pushed commits (`2be8cea`, `1350bfb`, `3affa49`, `3f0db2b`) still contain that folder name. No history was rewritten.
