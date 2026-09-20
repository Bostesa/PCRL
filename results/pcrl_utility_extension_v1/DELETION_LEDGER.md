# DELETION_LEDGER — `pcrl_utility_extension_v1` (public summary)

Local copies were unlinked **only** after the chunk containing them had been read back from S3 on
a different machine, its stream hash checked, extracted and every file re-hashed against the
manifest. At deletion time each file was re-checked: `lstat`/`realpath` containment, unchanged
size, mtime and inode, not open by any process, and a **freshly recomputed SHA-256** equal to the
manifest value. Anything failing a check is skipped and re-inventoried, never removed.

| group | chunks | files | freed | skipped |
|---|---|---|---|---|
| `competitive` | 7 | 85,541 | 13.73 GiB | 0 |
| `direct_adversarial` | 6 | 24,351 | 9.98 GiB | 0 |
| `invariant` | 2 | 6,761 | 2.85 GiB | 0 |
| `nonlinear_rank` | 2 | 11,171 | 2.93 GiB | 0 |
| `rs_residual_spectral` | 4 | 8,555 | 7.26 GiB | 0 |
| `rs_spectral_transport` | 8 | 17,981 | 14.93 GiB | 0 |
| **total** | **29** | **154,360** | **51.69 GiB** | **0** |

Measured filesystem free space: **49.42 GiB before, 101.12 GiB after** (+51.70 GiB), consistent
with the allocated bytes freed. No hardlinks were involved, so logical and allocated sizes agree.

## What was NOT deleted

* Source code, git metadata, raw data, the virtualenv, current reports and manifests.
* The main checkout and Terminal 2's manuscript worktree (another session owns them).
* The execution inputs (`exec_*` groups): archived for the cloud host and for backup, kept locally.
* Symlinks (archived as links, never removed), and the sealed 2016 data, which was excluded from
  every plan.
* Fitted attacker weights: archived in full. The previous cleanup deleted attacker weights that a
  later transport needed; this study keeps them and never compacts them away.

Per-chunk records with absolute paths, archive URIs and object version IDs are in the private
ledger under the git common directory. Each record carries the exact restore command.

## Disposable caches removed without archiving

47 `__pycache__` / `.pytest_cache` directories in Terminal-1-owned worktrees, **10.1 MiB**,
regenerated automatically on the next import or test run. Claim registry entry:
`t1__pycache_historical_t1_worktrees`. Nothing else was deleted without a verified archive.
