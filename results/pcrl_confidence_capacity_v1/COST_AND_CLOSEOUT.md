# Cost and closeout

## Limits (prompt §4) and use

Everything ran on the laptop: $0 cloud, and no cloud resource was created.

| Limit | Ceiling | Used |
|---|---|---|
| Elapsed | 10 h from 2026-10-06T03:48:42Z (hard stop 13:48Z) | About 3.7 h (closeout ~07:32Z) at closeout |
| Aggregate CPU | 20 CPU-h | 3.68 CPU-h measured in heavy semaphore holds, plus ≤ 1.0 CPU-h estimated for light unsemaphored agent work (tests, reviews, light checks; not per-process instrumented). Total < 4.7 CPU-h |
| Heavy processes | 2 in total, verifier included | **Never more than 2 concurrent holds** (per-slot reconstruction of SEMA_LOG.jsonl); see deviations |
| Heavy memory | 8 GiB aggregate | Peak single process 1.41 GB (assessment shard); two heavy processes < 3 GB combined |
| Free disk | ≥ 5 GiB | 127 GiB free throughout. Private store 599 MB; each same-device copy about 0.6 GB |
| Final reserve | 2 h / 4 CPU-h | Kept: the science finished at 06:03Z, 2.2 h into the run |
| Stage allocations | Preparation + Stage A ≤ 3 h / 6 CPU-h; Stage B + audits ≤ 5 h / 10 CPU-h | Stage A done at 04:21Z (0.5 h); Stage B, audits and assessment done at 06:03Z |

## Measured CPU by role (SEMA_LOG.jsonl; child user + sys)

| Role | Heavy holds | CPU-s | Note |
|---|---:|---:|---|
| A, lead | 23 | 7,207 | Study workers 7,100 CPU-s (1.97 CPU-h: admit through inference), plus the 246-test suite run, the final closeout copy and two semaphore tests |
| B, compressor (synthetic timing) | 3 | 201 | |
| C, math review (mutation testing) | 3 (+1 stopped) | 1,599 + about 40 | The stopped 05:11Z run (about 40 CPU-s) has no release record |
| D, attacks (synthetic timing) | 4 | 753 | |
| E, custody (closeout runs) | 1 | 13 | |
| F, verifier | 18 | 3,426 | |

## Study workers (`/usr/bin/time -l`, `<PRIVATE_CACHE>/qpc_v1/run/work_*.log`)

| Stage | Wall (s) | CPU (s) | Peak RSS |
|---|---:|---:|---:|
| admit | 11 | 11 | 0.99 GB |
| stagea (A1 + 18 recipient fits + 24 pairs) | 16 | 15 | 0.65 GB |
| gate | 3 | 3 | 0.62 GB |
| partition (2 shards) | 7 + 5 | 7 + 5 | 0.77 GB |
| fit (2 shards; 42 units) | 49 + 25 | 48 + 25 | 0.84 GB |
| inner (2 shards; 75 units) | 1,169 + 1,225 | 1,173 + 1,229 | 1.18 GB |
| inner_src (6 composed source units) | 209 | 209 | 0.67 GB |
| controls (2 shards; plus 2 refused starts of about 2 s each) | 510 + 201 | 512 + 202 | 1.24 GB |
| select | 2 | 2 | 0.51 GB |
| assessment (2 shards; 57 outer units) | 1,550 + 2,039 | 1,549 + 2,039 | 1.41 GB |
| inference (37 slots, B = 1,999) | 67 | 67 | 1.22 GB |

## Deviations (disclosed)

- **Locked test file edited after AUDIT_AND_SELECTION_LOCK** (math review role, 05:13–05:20Z). Both controls shards refused at 05:16:07Z ("locked file changed") before loading any data. The locked file was restored from git, and the edits moved to the unlocked `qpc/tests/test_math_review_postlock.py`. The controls were rerun from 05:16:58Z.
- **Killed wrapper without a release record.** A queued review wrapper ("C:mutation") was stopped by SIGTERM at about 05:12:26Z. The reviewer confirmed that no child outlived it (about 40 CPU-s; slot reacquired at 05:12:31Z). The semaphore now forwards signals and always logs the release (AMENDMENT_A1, committed 05:26:55Z and pushed about 05:26:56Z, before selection started at 05:27:06Z).
  - The first version of that fix deadlocked in a trivial `sleep` test wrapper, which was killed. It never wrapped a study process.
- **Light unsemaphored checks.** The math review role ran about 62 CPU-s of checks, each under the one-CPU-minute "light" threshold of TEAM_PLAN, while two heavy processes held the slots. Other agents' unit-test runs were also light and unsemaphored.
- **Label-rule clarifications after STAGE_A_LOCK** (SEL-R1, SEL-C2, SEL-R2): made after the Stage A gate was known, before any recovery data, and recorded in STAGE_B_LOCK. The label is reported under both rules, and they agree.

## Backup and custody

| Item | Status |
|---|---|
| This study's private store | **LOCAL_SAME_DEVICE_COPY_VERIFIED_OFF_DEVICE_BACKUP_PENDING.** Final versioned copy `<PRIVATE_CACHE>/qpc_v1_local_copy_20261006_v2/qpc_v1`: 926 files, 625,761,008 bytes, all re-read uncached and matching. Restores from the copy alone all PASS: teachers U and RAW-J by own forward pass, Q* and P* re-encoded and deployed (BOUND), and the selected pair attacker refit (max diff 0). The earlier copy `..._20261006` (925 files) is kept; it was independently re-verified by the verifier. A same-device copy is not a drive restore |
| External drive | Not mounted: probe by content found 0 matching volumes, 2 skipped by name. Nothing was copied to it, and nothing is claimed |
| Pending: this study, dpc, osf and smf | `OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m qpc.closeout all --targets <PRIVATE_CACHE>/qpc_v1/run/closeout_targets.json`, run with the drive mounted. It completes the dpc off-device backup, the osf backup with the smf drive restore and two late logs, and a new qpc drive copy with restores |
| Predecessor custody | PENDING (`provenance/dpc_custody/STATUS.json`, `provenance/predecessor_custody/STATUS.json`) |
| Historical exposure | An identifying drive-folder name in earlier committed history is not removed; no history rewrite is authorised |
| Private data in git | None: no checkpoints, per-person arrays, private paths or raw reviews. A privacy scan ran before every commit |

## Closeout steps

1. Final commit and push; remote SHA verified with `git ls-remote`.
2. Atomic handoffs `TO_MANUSCRIPT_CONFIDENCE_CAPACITY.json` and `TO_CLAIMS_CONFIDENCE_CAPACITY.json` in the shared evaluation directory, plus `HANDOFF.json` here.
3. Owned wakeups cancelled; no owned fit, assessment or verifier process remains (checked with `pgrep`). No watchdog was used in this study: elapsed time was tracked by the lead.
