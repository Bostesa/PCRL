# Cost and closeout

## Compute (laptop only; $0 cloud; no cloud resource)

| Item | Value |
|---|---|
| Start | 2026-10-05 03:11:22Z (`<PRIVATE_CACHE>/osf_v1/START.txt`) |
| Ceilings | 10 h elapsed (watchdog armed for 13:11Z, stopping only `osf` processes); 20 CPU-h; 2 heavy workers; < 8 GiB; ≥ 5 GiB free; $0; final 2 h reserved |
| DATA_AND_ENGINEERING_LOCK (`0179033`) | 03:54:59Z; admission, parity, fidelity (amendments A1 `ad5d867` and A2 `cbe72dd`), 15 replays and timing, 03:55–04:02Z |
| TRAINING_PROTOCOL_LOCK (`b8b0e9e`) | 04:02:33Z; 48 bank fits, 04:02–04:13Z |
| SELECTION_AND_AUDIT_LOCK (`a72664f`) | 04:14:45Z; references, inner audits, selection and tracking, 04:14–04:19Z |
| EVALUATION_LOCK (`3554235`) | Pushed 04:19:24Z; first outer unit complete 04:20:23Z; assessment 04:19–04:51Z (72 units); inference about 04:53–05:03Z |
| Timed worker CPU (`/usr/bin/time -l`, 24 processes) | **1.51 CPU-h**; peak RSS 3.14 GB (the inference process) |
| Untimed work | Tests, agents' fixtures, pre-lock audit checks (0.17 CPU-h), admission copies and the independent replay: under 1 CPU-h, so **about 2.5 CPU-h in total of 20** |
| Elapsed to the assessment | About 1 h 08 min; the final-two-hour reserve was not needed for fitting |
| Disk | ≥ 127 GiB free on the system volume |

## Storage

| Location | Content | Size | Files |
|---|---|---|---|
| `<PRIVATE_CACHE>/osf_v1` | Units (releases, run receipts with per-step strength records, checkpoints, critics and optimiser states, inner/track/outer records, per-row assessment predictions), admitted smf copies, logs | 2.3 GB | 2,583 at backup |
| `<PRIVATE_CACHE>/osf_v1_local_copy_20261005/osf_v1` | Same-device versioned copy with `SHA256SUMS`; all 2,583 files re-read uncached and matching. U, the incumbent, the N\*/R\* fallbacks and L\* restore bitwise from it alone; the final attacker refit reproduces its saved predictions exactly | 2.3 GB | 2,583 |
| External drive | **Not mounted at any check**, so no off-device copy exists for this study | — | — |
| Quarantine (kept) | `quarantine_prelock_references_20261005` (96 pre-lock reference units and 5 run files); `quarantine_amendment_A1`, `quarantine_amendment_A2` (vacuous equivalence receipts) | — | — |

Nothing was deleted. The smf private store and the closed smf worktree were not modified.

## Closeout

- **Custody gap.**
  - The external drive was absent, so `BACKUP_VERIFICATION.json` is `LOCAL_SAME_DEVICE_COPY_VERIFIED_OFF_DEVICE_PENDING`. The local copy proves restorability, not off-device custody.
  - Pending: `python -m osf.closeout backup --dest <DRIVE_ROOT>` and `python -m osf.closeout predecessor --restore`. The second is the smf custody repair; `PREDECESSOR_CUSTODY_REPAIR.json` is PENDING with the two later smf logs hashed into a local supplement.
- **Privacy.**
  - Public files use `<PRIVATE_CACHE>`, `<WORKTREE>` and `<DRIVE_ROOT>`.
  - Every push was preceded by a scan of the staged files that blocks the push on a hit.
  - The predecessor branch's four earlier commits that contain a drive-folder name are unchanged. No history was rewritten, and that exposure is not claimed to be removed.
