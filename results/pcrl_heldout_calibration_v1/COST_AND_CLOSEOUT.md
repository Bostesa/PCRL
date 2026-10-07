# Cost and closeout — hcal

Role F, drafted 2026-10-07 at about 15:15Z, after the locked assessment and the inference. Accounting comes from
`hcal.closeout.accounting()` on the private ledgers (`SEMA_LOG.jsonl`, `COMPUTE_LEDGER.jsonl`, `START.txt`), read at
15:12:57Z. Custody facts come from `BACKUP_VERIFICATION.json` and `RESTORE_INDEX.json`. Integrated by role A at closeout; the final push SHA is in HANDOFF.json.

**Study label:** NO_COMPETITIVE_RELEASE_CRITERION_ESTABLISHED. The single assessment opening ran 15:00:43–15:04:30Z,
and the inference is done.

## 1. Cost

| Item | Used | Budget (PROTOCOL §9) |
|---|---|---|
| Cloud | **$0** (nothing cloud-side) | $0 |
| CPU, captured by the semaphore | **2.12 CPU-h** (61 released holds) | 20 CPU-h (17.88 left) |
| CPU, not captured (upper bound) | ≤ 0.43 CPU-h: the first control attempt, see below | — |
| Wall, since START 12:22:22Z | **2.84 h** at 15:12:57Z | 10 h (7.16 left) |
| Peak concurrent heavy processes | **2** (limit respected) | 2 |
| Peak child RSS | 2.88 GiB (D), 1.65 GiB (A) | 8 GiB aggregate |
| Free disk | 113–115 GiB at every stage start and F check; 111.8 GiB now, after the copy | ≥ 5 GiB |

**CPU by role** (semaphore children, seconds): A 5,164.6 (23 holds); D 2,317.4 (17); E 103.9 (8); F 30.3 (5);
C 19.1 (7); B 3.6 (1).

**CPU by stage** (stage-process CPU from COMPUTE_LEDGER; a subset of the semaphore children, not added twice), in
seconds:

| admit | engineering | calibrate | utility | audit | compose | controls | select | replay | assess |
|---|---|---|---|---|---|---|---|---|---|
| 40.3 | 0.0 (44.1 s wall) | 14.4 | 11.7 | 2,152.3 | 0.1 | 2,062.8 | 0.4 | 304.8 | 348.1 |

- Engineering shows 0.0 s because its work ran in child processes; that CPU is in the semaphore log.
- Controls covers attempt 2 only.
- The inference has no ledger line.

**Not captured.**
- **AMENDMENT_A1 control rerun.** The first control run (pre-A1, two shards, 14:18:19–14:31:15Z) was stopped by
  signalling the semaphore wrappers. The wrappers logged cpu_s 0.0, because the signal reached `/usr/bin/time` and not
  the python workers. The upper bound is 2 × 776.2 s = 0.43 CPU-h, if both workers were CPU-bound.
- **OPS-1** (REVIEW_FINDINGS_DISPOSITION.json). The workers ran for about 2 s more as orphans (PPID 1) and were then
  stopped directly. No third heavy process started in that window: the next acquire was at 14:32:42Z.
- **Outside the semaphore.** The agents' own processes and light commands are in no ledger.
- **Total.** About 2.1–2.6 CPU-h of the 20, plus the uncaptured agent work.

## 2. Work counts

Counted from unit names in `<PRIVATE_CACHE>/hcal_v1/run/units`. F opened no unit.

| Units | Count | Rule (QUEUE_MANIFEST.json) |
|---|---|---|
| cal__ | 171 | 57 partitions × 3 seeds (calibrate) |
| calU__ | 3 | continuous-U calibrations, one per seed |
| util__ | 3 | utility tables, one per seed |
| fam__ / com__ | 75 / 75 | 2 × audited × 3 seeds (audit), i.e. 25 audited partitions per seed |
| com__ SRC_U | 3 | continuous-U composed records (compose) |
| oprob__ | 81 | scored releases × 3 seeds (assess) |
| oatt__ | 15 | scored partitions × 3 seeds (assess) |
| controls parts | 6 + 2 | attempt 2 (after A1): 6/6 parts. Attempt 1 (pre-A1): stopped after 2 of 6 parts, retained in `run/controls_parts_attempt1_preA1/` |

## 3. Custody

**Status: LOCAL_SAME_DEVICE_COPY_VERIFIED_OFF_DEVICE_BACKUP_PENDING. No off-device custody is claimed.**

- **Same-device copy.** `<PRIVATE_CACHE>/hcal_v1_local_copy_20261007` holds 4,098 store files plus the pinned input
  (4,099 files, 1.37 GB). SHA256SUMS sha256 is e9b78c91…; the uncached (F_NOCACHE) re-read matched 4,099/4,099. It is
  labelled SAME_DEVICE_COPY.
- **Restores from the copy.** All required classes PASS:
  - the U teacher, seeds 0–2, bitwise;
  - 12 representative frozen-bank tables;
  - every calibrator family, including U's;
  - the P\* and T\* deployments from the copy, bitwise;
  - the selected reader refit, which was exact.

  The sealed D came from the pinned input path, which is byte-identical to the copy's bundle.
- **Drive.** Re-checked by content at 15:12:57Z: not mounted (0 candidate volumes, 2 skipped by name, no physical
  external disk).
- **Pending; each item needs the drive.** The exact commands are in BACKUP_VERIFICATION.json:
  - the hcal off-device copy (`hcal.closeout offdevice`);
  - the lra off-device backup;
  - the lcr off-device backup;
  - the cbp chain, which covers the cbp, qpc and osf drive copies, the dpc off-device backup, and the smf drive restore
    with its two later logs.
- **Studies without off-device custody:** hcal, lra, lcr, cbp, qpc, dpc, osf. smf holds a verified drive copy.
- **Historical identifying-path commits.** The four historical commits with identifying paths were **not scrubbed and
  not rewritten here**. No history rewrite or force-push is authorised.

## 4. Identity scan (`hcal.closeout.identity_scan`, re-run at 15:14Z)

- **Scope:** 85 files under `results/pcrl_heldout_calibration_v1/`, `hcal/` and `tests/pcrl_heldout_calibration_v1/`.
  That is 70 tracked files (git ls-files) and 15 untracked new files, including these receipts.
- **Identifying findings: 0.** No home or volume path, local user name, email, credential-like string or per-person
  record pattern.
- **Review findings: 14.** These are home-relative command lines in usage docstrings: a `~`-prefixed interpreter path,
  and in E's commands a `HOME`-prefixed private-cache path.
  - Files: E's `verify_inner.py`, `verify_phase3.py`, `verify_phase3b.py`, `selftest_verify_inner.py` and
    `E_ENGINEERING_REVIEW.md`; C's `test_calib.py`; D's `test_bank.py` and `test_controls.py`.
  - None names a user. The owners may switch them to `<python>` and `<PRIVATE_CACHE>`.

## 5. Processes and deletions

- **Processes.** `owned_processes()` at 15:12:57Z matched only F's own closeout wrapper; at 15:14:45Z it matched
  nothing (no hcal.run, hcal.sema, hcal.assess or hcal.verify process running). Nothing was killed.
- **Deletions: none.** Nothing was deleted or moved. The pre-A1 control attempt and every receipt are retained.
