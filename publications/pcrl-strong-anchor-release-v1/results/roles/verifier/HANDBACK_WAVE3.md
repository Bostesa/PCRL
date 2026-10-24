# Independent verifier hand-back, wave 3 (saved verbatim by the coordinator at 07:13Z on 2026-10-24, `date -u`)

Wave 3 is done (finished 07:09:44Z, `date -u`): **109 checks, 102 PASS, 7 FAIL.** Across all waves: **365 checks, 355 PASS, 10 FAIL**. The 3 wave-1 FAILs stand; no wave-2 FAILs.

The 7 new FAILs are 6 FB packages that are not bitwise on a 50-row batch (ULP-level, decisions identical), plus one publication finding: your name appears in three published files.

**CPU.** Wave 3 was label-free: no queue unit, no git, new code files only, and no edit while any unit of mine ran. The checks ran under the torch venv (Python 3.13.7) and `python3 -I`, about 5 CPU-s outside the queue. All waves: 57.9 CPU-s in the queue plus at most about 35 CPU-s outside it, under 100 of the 2,400 cap.

**1. Deployment packages (all 13)**
- **(a) Dependencies:** every hash re-checks, 21–31 per package, using my own key resolution.
- **(b) Stored wire:** equals RELEASE.npz bitwise for both tasks on all six roles, and the release hash chain holds.
- **(c) MODEL_SELECT wires:** equal `wire_sha256_eval` of the evaluating record (the ROLES records for the role points).
- **(d) Validity:** all values finite and in [0, 1]; worst row-sum error 4.4e-16, against my 1e-9 tolerance.
- **(f) Refusals:** all three probes refused in every package: extra SEX field → PermissionError, missing POBP → ValueError, `raw_score` request → PermissionError.
- **(e) Black-box, 50 DEV_REPORT raw rows** (seed 20261026, same rows for every package):
  - The 7 FA packages are bitwise.
  - **The 6 FB packages FAIL the bitwise test, and the FAILs are kept.** Their occupation probabilities differ from the stored wire by up to the amounts below; income is bitwise and every decision matches the stored argmax.

    | Package | V0 | V1 |
    |---|---|---|
    | FB_PFO | 5.6e-16 | 2.2e-16 |
    | FB_PFO_DQ_HI | 1.1e-16 | 1.1e-16 |
    | FB_PFO_RAWJ_HI | 3.3e-16 | 2.2e-16 |

  - Separate entries (the FAILs still stand): the 50-row outputs are within 1e-15 with identical decisions.
  - A diagnostic shows `sar.deploy.release` on all 5,943 DEV_REPORT rows reproduces the stored wire bitwise for all six FB packages and for FA_TFRB_R2_V0. Smaller batches differ: 50 rows and single rows differ at the ULP level, and even FA_TFRB_R2_V0's single-row occupation is off by 5.6e-17.
  - So the deployed output depends on how a request is batched, at floating-point level; no decision changes.
- **Against your records** (read only after my checks): your full-batch bitwise parity, validity and refusal verdicts all agree with mine. Your `all_bitwise: true` holds only for the full DEV_REPORT batch, not for arbitrary request batches.

**2. Publication dry build (237 files)**
- **No hits for:** absolute `/Users/` paths, e-mail addresses, tokens or keys, session temp paths, binary or array files, or record-level data. The only CSV is the aggregate `results/dev/UTILITY_LEDGER.csv` (44 rows).
- **FAIL, W3_PUB_user_name:** your name appears in the drive-relocation folder name "<owner>-Mac-relocated-2026-10-22". It is a personal identifier in an anonymized publication, so I'd redact it (for example `<relocation run>`) before the final build. Locations:
  - `results/PROTOCOL.md:59`
  - `results/STORAGE_ADMISSION.json:6`
  - `results/STORAGE_ADMISSION.json:47`
- **Hits I judged harmless:**
  - **`/Volumes/` (3):** `code/sar/admit.py:366` and `code/sar/storage.py:24` name the backup-drive label YOTUO; `storage.py:33` is the system path /System/Volumes/Data.
  - **`~/storage-relocation-20261022/` (5):** `storage.py:7` and `:64`, `PROTOCOL.md:55`, `STORAGE_ADMISSION.json:39`, `COORDINATOR_LOG.md:7`. Home-relative, no user name.
  - **Worktree paths (295):** all sanitized `<repository>/...` forms or the project's own worktree names. Most are in my published CHECKS.json and work files. The sanitizer also rewrote a path literal in `code/v_decide.py:656`, so that published copy no longer runs as-is; this is cosmetic.
  - **"Bostesa" (6):** the repository's public remote URL, which is public by your standing decision.
  - **"scratchpad" (3):** a word in the statistical reviewer's hand-backs, not a path.
- Every hit, with file and line, is in `work/pubscan3.json`.

**Not checked**
- The final publication build (this was the dry build) and the final backup.
- Request batches other than the 50-row sample, single rows and the full DEV_REPORT batch.
- Deployment on other roles' raw inputs (only DEV_REPORT has raw inputs).
- Randomized-release key handling (no package is randomized).

Files are in <repository>/.claude/worktrees/pcrl-strong-anchor-release-v1/results/pcrl_strong_anchor_release_v1/roles/verifier/:
- CHECKS.json (wave-3 entries appended)
- VERIFICATION_REPORT.md ("Wave 3" section added)
- work/deploy3.json
- work/deploy3_diag.json
- work/deploy3_diag_b.json
- work/pubscan3.json
- code/v3_deploy.py
- code/v3_deploy_diag.py
- code/v3_pubscan.py
- code/v3_assemble.py
- code/v3_append_b.py
