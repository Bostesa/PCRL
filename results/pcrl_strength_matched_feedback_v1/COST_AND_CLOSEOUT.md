# Cost and closeout

## Compute (laptop only; $0 cloud; no cloud resource; no Docker or unrelated service)

| Item | Value |
|---|---|
| Start | 2026-10-05 00:42:09Z (`<PRIVATE_CACHE>/smf_v1/START.txt`) |
| Ceilings | 10 h elapsed (watchdog armed for 10:42Z, stopping only `smf` processes), 20 CPU-h, 2 heavy workers, < 8 GiB, ≥ 5 GiB free, $0 |
| Warm starts, parity, U, raw controls | 00:53–00:57Z (DATA_AND_ENGINEERING_LOCK `6f307ee`) |
| Review fixes R1–R4, parity rerun, Phase A (54 runs), inner audit, selection | 01:06–01:17Z (PHASE_A_PROTOCOL_LOCK `65a6eeb`; freeze `15a9db9`) |
| Calibration and controller preflight | about 01:17–01:20Z (amendments A2, A3) |
| Phase B (36 runs), baselines, inner audit, selection, tracking | 01:20–01:33Z (PHASE_B_PROTOCOL_LOCK `90270f0`) |
| Single assessment (51 units), controls, inference, reports | 01:33:59–01:50:50Z (EVALUATION_LOCK `74af248`) |
| Drive backup with uncached read-back and restore checks | about 01:52Z |
| Timed worker CPU (`/usr/bin/time`, 27 processes) | **1.32 CPU-h**; peak RSS 1.21 GB |
| Untimed work | Tests, engineering checks, agents' fixtures and the independent replay; about 0.7 CPU-h, so **≈ 2 CPU-h in total of 20** |
| Elapsed | About 1 h 15 min to the assessment; the reserve was not needed |
| Disk | ≥ 132 GiB free on the system volume |

## Storage

| Location | Content | Size | Files |
|---|---|---|---|
| `<PRIVATE_CACHE>/smf_v1` | Units (warm starts, checkpoints, critics with optimizer state, probe receipts, inner, tracking and outer records, per-row assessment predictions, FARE cache), logs, selections, inference | 1.1 GB | 2,412 at backup |
| `<DRIVE_ROOT>/private_smf_v1_20261005/smf_v1` | Versioned copy, `SHA256SUMS`, `BACKUP_RECORD.json`; 2,412/2,412 re-read uncached; U, J-F and L-F restored bitwise; attacker refit exact | 1.1 GB | 2,414 |
| Reused input | `<PRIVATE_CACHE>/jcv_v1/inputs/adult_jcv.npz` (hash-verified); no predecessor checkpoint reused | — | — |

Nothing was deleted. Superseded receipts were renamed `*.quarantined_*`: 36 parity receipts and 3 preflight receipts.

## Closeout

- All study workers have exited. The watchdog is stopped at closeout.
- **Custody gap.** The drive was detached after the backup and restore checks.
  - The only private file changed since then is `run/closeout_backup.log`; the watchdog stop line in `run/watchdog.log` is added at closeout. Neither is on the drive.
  - The independent verifier's own drive restore is PENDING until the drive is reconnected (see VALIDATION.md).
- **Privacy.**
  - Public files use `<PRIVATE_CACHE>`, `<WORKTREE>` and `<DRIVE_ROOT>`.
  - Every push was preceded by a scan of the staged files that blocks the push on a hit.
  - The previous branch's four earlier commits that contain a drive-folder name are unchanged. No history was rewritten, and that exposure is not claimed to be removed.
