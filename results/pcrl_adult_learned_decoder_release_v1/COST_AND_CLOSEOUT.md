# Cost and closeout — lra

## Budget against use

| Resource | Cap | Used | Source |
|---|---|---|---|
| Elapsed | 10 h from 2026-10-07T04:03:43Z (ceiling 14:03:43Z; closeout reserve from 12:03:43Z) | assessment closed at 08:32Z; closeout recorded in HANDOFF.json `closed_utc` | START.txt, git log |
| CPU | 20 CPU-h (4 reserved for closeout) | about 6.8 CPU-h at inference (§2); final total in HANDOFF.json `compute` | SEMA_LOG.jsonl |
| Concurrency | at most 2 heavy processes, one thread each | at most 2 semaphore slots ever held | SEMA_LOG.jsonl |
| Memory | 8 GiB aggregate | largest single process 1.7 GiB | SEMA_LOG.jsonl |
| Disk | keep ≥ 5 GiB free | about 116 GiB free; private store about 1.5 GB | df, du |
| Cloud | $0 | $0; no AWS instance, no new data | — |

## CPU by stage (lead holds, CPU-h, at inference)

| Stage | CPU-h | Stage | CPU-h |
|---|---|---|---|
| inner | 3.14 | fit | 0.59 |
| assess | 1.01 | controls | 0.33 |
| select | 0.03 | correctness | 0.02 |
| infer | 0.02 | | |

These lead stages total about 5.2 CPU-h (A). The other roles add about 1.6: B 0.14, C 0.32, D 0.24, E 0.86 and
F < 0.01.

## Closeout state

- **Custody.** A versioned same-device copy, verified uncached and restored from: teacher, 3 learned decoders, P\*, Q,
  the constrained fallback, the decoder-only baseline, the sequential winner and P\*'s attacker (BACKUP_VERIFICATION.json).
  - Status: LOCAL_SAME_DEVICE_COPY_VERIFIED_OFF_DEVICE_BACKUP_PENDING; the drive was absent.
  - The predecessor custody (cbp → qpc/dpc/osf/smf) is PENDING because the drive is absent. This study's opening, which
    it had been waiting for, has happened.
  - The exact commands are recorded in BACKUP_VERIFICATION.json and provenance/cbp_custody/STATUS.json.
- **Repository.**
  - Untouched: the lcr, cbp and qpc registrations, their code and results; main and the manuscript branches; other
    worktrees.
  - No merge, history rewrite, manuscript edit or outside message.
- **Processes.** Only this study's own processes ran, and none remain. The "BackgroundSyncService Setup" image was
  skipped by name and never touched.
- **Exposure.** The realized use is appended to EXPOSURE_LEDGER.md.
