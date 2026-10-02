# Backup status: laptop-only launch configurations (2026-10-02, updated after drive remount)

**Status: PRESERVED and verified (2026-10-02T15:33:36Z).** All 102 files (26,510,823 bytes) were copied to a private directory on the YOTUO drive and checked by reading them back.

**Timeline.** At first the drive was not mounted, and the morning dry-run and real run both refused with exit code 2. The drive was then disconnected and remounted on 2026-10-02, and the backup ran after the remount.

**Verification.**
- Before copying, all 102 source files matched the hashes recorded in the manifest.
- Every copy was re-read from the drive in a separate process. All 102 sha256 values and sizes match the originals.
- The SHA256 manifest on the drive passes `shasum -c`.
- The private manifest holds the per-file result: 102 of 102 `PRESERVED_sha256_match`.
- Originals were not modified or deleted. Nothing else on the drive was touched.

**File-system caveats.** The drive is exFAT, mounted with `noowners`.
- exFAT cannot store POSIX permissions, so `chmod 700` has no effect there.
- The original file modes (755 for scripts, 644 or 600 for the rest) are recorded in `ORIGINAL_MODES.txt` on the drive and in the manifest. Restore them with `chmod`.
- Modification times were kept by `ditto`. Extended attributes, resource forks and ACLs were not copied, because exFAT rejects them.
- macOS created AppleDouble `._*` side files inside the new backup directory.
- The read-back ran right after the write on a live mount, so it may have been served from the OS cache rather than from the disk itself. The cache could not be flushed without root. Re-run the script after the next remount: it skips identical files and re-verifies everything, which gives a read-back from the disk.

## Scope enumerated
- PCRL main checkout: everything git-ignored or untracked by `git status --porcelain --ignored`. This excludes results data, data/, .venv, caches, `__pycache__`, `.claude/` harness settings and the private mock-review notes.
- Every worktree under `PCRL/.worktrees/*`.
- durable-guarantees: no laptop checkout exists, because it was relocated. The scratch clone is a clean copy of the GitHub main branch and has no untracked files.

## Counts (102 files, 26,510,823 bytes)
| category | files | bytes | identical blob in a PCRL git ref | single copy (no git blob, absent from the 2026-09-30 drive inventories) |
|---|---|---|---|---|
| infra launchers (8 dirs: erase pilot, erase pilot diabetes, VICReg sweep, rank-8, INLP, SPLINCE, varconstraint, per-dim Lagrangian; launch.sh / user_data.sh / README / aggregate.py) | 19 | 163,433 | 5 | 14 |
| root launch README | 1 | 8,968 | 1 | 0 |
| root launch script (BIOS layer-12) | 1 | 17,700 | 0 | 1 |
| run logs (held-out seed-3 train/audit logs) | 3 | 2,818 | 0 | 3 |
| **single-copy checkpoints** (cross-purpose constrained retrains, best.pt + final.pt × 9) | 18 | 20,246,619 | 0 | 18 |
| worktree private launch state (Sept ACS-lineage: user_data, SSM command logs, resource/cost ledgers, staging source tarballs, private aggregates) | 60 | 6,071,285 | 0 | 60 |

- **96 of the 102 files existed only on the laptop until this backup.** The 6 git-backed files are the INLP/SPLINCE launchers and the root README; their blobs survive in history at d09c5f79f. None of the 102 sha256 values appears in any 2026-09-30 drive inventory.
- All 102 files are git-ignored in their checkout. No file is tracked.

## Secret-pattern scan
The scan matched patterns only. No values were printed, and binary checkpoints were not scanned. Tarballs were scanned inside.
- IAM / instance-profile reference: 9 files (all launch.sh-type launchers)
- EC2 key-pair name option: 9 files (the same launchers)
- generic "token" word: 32 files. These are mostly JSON/aggregate files and tarballs, and the matches are likely tokenizer or field names. This is unconfirmed.
- No AWS access-key IDs or values, AWS secret names, session tokens, passwords, private-key blocks, account-ID-bearing ARNs or GitHub tokens were found.

The backup therefore goes to a private directory on the drive (exFAT cannot enforce mode 700; the private manifest directory on the laptop is mode 700). Do not commit these files to any repository.

## How to re-run / re-verify (drive mounted)
Use the private script in `~/Documents/PCRL_private_review_sources_20261001/backup_manifest_private/`:
1. `./backup_launch_configs.sh --dry-run` re-verifies the sources and checks the destination for collisions.
2. `./backup_launch_configs.sh` runs the backup. It copies each file with `ditto`, which preserves bytes, mode and mtime. It refuses to overwrite a differing file and writes `SHA256SUMS` plus the manifest on the drive. It then re-reads every copy and checks sha256, size and mode, runs `shasum -c`, and exits nonzero on any mismatch.

Originals are never deleted or modified. The destination is a private directory under the 2026-09-30 relocation folder on YOTUO. If any source has changed since the manifest was generated today, the script refuses until the manifest is regenerated with `build_backup_manifest.py` (in the same private dir).

## Coordinator re-verification (2026-10-02)

The coordinator independently re-hashed all 102 drive copies, using macOS uncached reads (`fcntl F_NOCACHE`), and
compared each against the private manifest and the original file: **102 of 102 match**.

A cold read after an unmount/remount was attempted. The unmount was refused by a system process holding the
volume, and it was not forced. Re-run the private verification script after the next remount to complete a
guaranteed on-disk read.
