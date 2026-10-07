# Cost and closeout — lcr

## Budget against use

| Resource | Cap | Used | Source |
|---|---|---|---|
| Elapsed | 10 h from 2026-10-06T23:42:46Z (ceiling 2026-10-07T09:42:46Z) | closed at about 02:30Z (about 2.8 h) | START.txt, git log |
| CPU | 20 CPU-h (4 reserved for closeout) | about 1.6 CPU-h measured through the semaphore, all roles (§2) | SEMA_LOG.jsonl |
| Heavy processes | at most 2 at once, one thread each | at most 2 slots used; every numerical job went through `lcr.sema` | SEMA_LOG.jsonl |
| Memory | 8 GiB aggregate | largest single child about 1.3 GiB | SEMA_LOG.jsonl |
| Disk | keep ≥ 5 GiB free | about 120 GiB free throughout | work logs |
| Cloud | $0 | $0; no AWS instance, no new data | — |

The registered Adult plan was projected at about 0.61 CPU-h for fitting, 3.7 CPU-h for inner audits (calibrated) and
1.15 CPU-h for assessment (TIMING.json, AUDIT_COMPUTE.json). None of it ran: the gate failed before any Adult fit.

## Measured CPU by role (`lcr.sema` releases; CPU-h)

| Role | What | CPU-h |
|---|---|---|
| A (lead) | admission, both fixture attempts, report replays, deployment checks, test runs | about 0.10 |
| B | decoder and fixture tests | 0.04 |
| C | mapper tests and synthetic timing | 0.59 |
| D | audit tests and synthetic timing | 0.71 |
| E | verifier self-tests and fixture oracle | see VALIDATION.md |
| F | custody status, backup and restore | < 0.01 |
| R | review-workflow test runs | 0.04 |

The final figures are recomputed at closeout from SEMA_LOG.jsonl (HANDOFF.json `compute`).

## Closeout state

- **Worktree.** It is left clean. Private outputs live only under `<PRIVATE_CACHE>/lcr_v1/`. Review-only scratch tests
  were moved out of the tree into the private store; nothing was deleted.
- **Custody.**
  - Versioned same-device copy `<PRIVATE_CACHE>/lcr_v1_local_copy_20261007/` of the store as of 01:49Z, verified
    uncached and restored from (BACKUP_VERIFICATION.json, RESTORE_INDEX.json). Refreshed at closeout: see
    BACKUP_VERIFICATION.json for the refresh record.
  - Status: LOCAL_SAME_DEVICE_COPY_VERIFIED_OFF_DEVICE_BACKUP_PENDING. The off-device drive was absent. Pending
    command: QUICKSTART.md §6.
- **cbp custody: PENDING** (provenance/cbp_custody/STATUS.json). It carries the pending qpc, dpc, osf and smf
  off-device custody.
  - It opens osf's assessment rows, which are the same rows as this study's never-opened assessment, so it is the row
    owner's decision. The exact command is recorded there.
- **Other worktrees, branches and main.** Untouched. No merge, no history rewrite, no manuscript edit, no outside
  message.
- **Processes.** Only this study's own processes were run; none remain. The "BackgroundSyncService Setup" image was
  skipped by name and never touched.
- **Exposure.**
  - The material label-use change (OSF_DEFENSE_FIT task labels and SEX for supervised fitting) was authorized but never
    exercised: no Adult fit ran.
  - AUDIT_FIT and INNER_SELECTION were never read by an lcr attacker.
  - lcr never unsealed the assessment rows (they were opened historically by qpc and cbp).
