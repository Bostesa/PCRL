# ARCHIVE_MANIFEST — `pcrl_utility_extension_v1`

Chunked, streamed tar+zstd archives in a private, versioned, encrypted S3 bucket with **no**
lifecycle rules. Each chunk has a manifest (relative path, byte count and SHA-256 per file, the
archive stream SHA-256, object key and version ID, source commit, required readers) and an
independent AWS-side verification record (read the stored version back, check the stream hash,
extract to a disposable directory, re-hash every file).

| group | chunks | files | source bytes | stored bytes | verified | source commit |
|---|---|---|---|---|---|---|
| `competitive` | 7 | 85,541 | 13.55 GiB | 7.58 GiB | 7/7 | `7f961d5c7` |
| `direct_adversarial` | 6 | 24,351 | 9.92 GiB | 9.33 GiB | 6/6 | `106de9afa` |
| `exec_inv` | 1 | 39 | 0.02 GiB | 0.02 GiB | 1/1 | `73903b7f2` |
| `exec_main` | 1 | 2,545 | 0.57 GiB | 0.30 GiB | 1/1 | `ad2c08872` |
| `exec_rs` | 1 | 264 | 0.30 GiB | 0.20 GiB | 1/1 | `349efa454` |
| `invariant` | 2 | 6,761 | 2.83 GiB | 2.68 GiB | 2/2 | `73903b7f2` |
| `nonlinear_rank` | 2 | 11,171 | 2.90 GiB | 2.63 GiB | 2/2 | `c37807e4f` |
| `rs_residual_spectral` | 4 | 8,555 | 7.25 GiB | 5.97 GiB | 4/4 | `349efa454` |
| `rs_spectral_transport` | 8 | 17,981 | 14.90 GiB | 13.60 GiB | 8/8 | `349efa454` |
| **total** | **32** | **157,208** | **52.23 GiB** | **42.30 GiB** | **32/32** | |

## What each group is

* **`competitive`** — 372-slot competitive-method study 7f961d5c: releases, audits, predictions, fits, 2017
* **`direct_adversarial`** — direct adversarial study 106de9af/69e790af
* **`exec_inv`** — H utility probes + leace_A0 release
* **`exec_main`** — execution inputs from main checkout (KEPT locally)
* **`exec_rs`** — H ancestor slate + historical A0/J/H metrics (KEPT locally until rs_residual_spectral archive verified)
* **`invariant`** — invariant baselines 73903b7f: leace_A0, splince_A0, optnet releases and audits
* **`nonlinear_rank`** — nonlinear/rank study c37807e4
* **`rs_residual_spectral`** — residual spectral development study; H/A0/J historical audits and H ancestor slate
* **`rs_spectral_transport`** — locked 2017 spectral transport study 349efa45 (historical status retained)

Restore: see `RESTORE.md`. The bucket name and object version IDs are held in the private
ledger under the git common directory; they are deliberately not published here.

## Integrity notes

* An upload returning success is not a verification: every chunk was re-read from S3 **by a
  different machine** and re-hashed file by file before any local copy was removed.
* Multipart ETags were never treated as file SHA-256 values; the per-file hashes are computed
  from the exact bytes written into the tar, and the stream hash from the exact bytes sent.
* Symlinks are archived as links and never deleted locally.
* Year-2016 paths are rejected at plan time and appear in no manifest.
