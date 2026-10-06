# Cost and closeout

## Limits (prompt) and use

Everything ran on the laptop: $0 cloud, and no cloud resource was created.

| Limit | Ceiling | Used |
|---|---|---|
| Elapsed | 10 h | 2026-10-05T23:37:49Z start (`<PRIVATE_CACHE>/dpc_v1/START.txt`); about 3.2 h at closeout. Watchdog pid armed for 2026-10-06T09:37Z, stopped at closeout |
| Aggregate CPU | 20 CPU-h | about 2.1 CPU-h measured (study workers 1.27 h, independent verifier 0.82 h over all runs, lead tests/reports/closeout ~0.05 h) plus ≤ 1 CPU-h estimated for agent-run synthetic tests and mutation runs (not per-process instrumented). Total < 3.5 CPU-h |
| Heavy workers | 2 | **Exceeded twice** (see deviations) |
| Working memory | 8 GiB combined | Peak single process 1.38 GB (inference); two assessment workers 0.99 + 1.02 GB; combined peak < 3.5 GB |
| Free disk | ≥ 5 GiB | 126 GiB free throughout. Private store 648 MB; same-device copy 648 MB |
| Final reserve | 2 h / 4 CPU-h | Kept: the assessment finished about 01:34Z, about 2 h into the run |

## Study workers (`/usr/bin/time -l`, `<PRIVATE_CACHE>/dpc_v1/run/work_*.log`)

| Stage | Wall (s) | CPU (s) | Peak RSS |
|---|---:|---:|---:|
| admit | 8 | 8 | 0.73 GB |
| partition (2 shards) | 3 + 3 | 2 + 2 | 0.61 GB |
| fit (2 shards; 258 units) | 35 + 35 | 35 + 34 | 0.68 GB |
| inner audits (2 shards; 273 units) | 1,170 + 1,133 | 1,170 + 1,133 | 0.64 GB |
| controls, crashed run (before A1) | 11 | 10 | 0.56 GB |
| controls (after A1) | 363 | 363 | 0.79 GB |
| select | 2 | 2 | 0.52 GB |
| assessment (2 shards; 69 units) | 827 + 917 | 823 + 913 | 1.02 GB |
| inference (33 primary slots + 1,070 levels) | 55 | 55 | 1.38 GB |
| post-completion resume tests (QUICKSTART) | 3 + 2 | 2 + 2 | 0.68 GB |
| **Total** | **4,567** | **4,554 (1.27 CPU-h)** | |

**Other lead processes:**
- `dpc.report`, the backup with restore checks (4 s), the backup dry-run (4 s), the deploy tests (a few seconds each), and closeout status/dry-runs.
- The 158-test suite: 50 s.

**Verifier.** 0.82 CPU-h over all runs: Phase 1, the first Phase 2 run, and two full re-runs after document corrections. Every run used one process and is recorded in `INDEPENDENT_VERIFICATION.json` → `verifier_compute_ledger`. The final three Phase 2 runs had 0 lead workers at start and end.

## Deviations (disclosed)

- **Two-heavy-worker limit exceeded twice.** Both times the lead started two study workers while an independent-verifier heavy run was already in progress:
  - about 120 s (00:49:45–00:51:45Z, inner audits);
  - about 239 s (01:18:45–01:22:44Z, assessment start).

  The verifier had started each run when at most one lead worker was active. Combined memory stayed under 3.5 GB, and no result depends on scheduling.
- **The controls stage ran twice.** The first run crashed before writing any result (AMENDMENT_A1). Its log is kept, with a copy quarantined.

## Backup and custody

| Item | Status |
|---|---|
| New-study private store | **LOCAL_SAME_DEVICE_COPY_VERIFIED_OFF_DEVICE_BACKUP_PENDING**: `<PRIVATE_CACHE>/dpc_v1_local_copy_20261006/` holds 2,364 files and 673,460,066 bytes, all re-read uncached and matching. Restore checks from the copy alone: 7/7 PASS (`BACKUP_VERIFICATION.json`, `RESTORE_INDEX.json`). A same-device copy is not a drive restore |
| External drive | Not mounted (probe by content: 0 matching volumes). Nothing was copied to it, and nothing is claimed |
| Pending new-study command | `OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m dpc.closeout backup --dest <DRIVE_ROOT> --targets <PRIVATE_CACHE>/dpc_v1/run/closeout_targets.json` |
| Predecessor custody (osf/smf) | **PENDING**: the osf off-device copy, the smf drive restore and the two smf internal-disk-only logs. Command: `OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m dpc.closeout predecessor --restore` (`provenance/predecessor_custody/STATUS.json`) |
| Historical exposure | An identifying drive-folder name in earlier committed history is not removed; no history rewrite is authorised. Public files of this study use placeholders |
| Private data in git | None: no checkpoints, per-person predictions, private paths or raw reviews. The pre-push scan is recorded in the closeout commit |

## Closeout steps

1. Commit and push all public files; verify the remote SHA with `git ls-remote`.
2. Write the atomic handoffs `TO_MANUSCRIPT_DECISION_PRESERVING_COMPRESSION.json` and `TO_CLAIMS_DECISION_PRESERVING_COMPRESSION.json` in the shared evaluation directory.
3. Stop the watchdog, and confirm no `dpc` worker or verifier process remains.
4. Remove the scratch deploy inputs. They were person-level features in the session scratchpad only, never in git.
