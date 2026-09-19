# STORAGE_INVENTORY — Terminal 1, 2026-09-19

Read-only inventory of UNTRACKED files in every PCRL worktree (tracked files live in git and are not cleanup candidates). File contents were not opened; unrelated personal documents were not scanned.

* Filesystem free at start: 15 GiB of 460 GiB (97% used). Later snapshot 22.6 GiB (other activity; Terminal 2 cleaned its own build outputs).
* Untracked PCRL files: 493972 files, logical 72.76 GiB, allocated 73.75 GiB. No hardlinks (nlink = 1 everywhere), so logical and allocated differ only by block rounding and sparse/compressed files.
* Sealed 2016 files found only under the evidence-paper worktree `data/acs_2016_admission/` (6 files, 0.43 GB). Excluded from every candidate list, archive plan and cloud bundle.

## By category (untracked)

| category | logical GiB | allocated GiB | files |
|---|---|---|---|
| arrays | 45.22 | 45.47 | 76896 |
| weights_pt | 8.53 | 8.63 | 47606 |
| records | 7.43 | 7.57 | 85159 |
| fitted_joblib | 6.72 | 6.81 | 39051 |
| other | 2.50 | 2.92 | 234234 |
| tabular | 1.99 | 1.99 | 93 |
| pycache | 0.18 | 0.21 | 10489 |
| archive | 0.17 | 0.17 | 126 |
| logs | 0.00 | 0.00 | 134 |
| reports | 0.00 | 0.00 | 52 |
| image | 0.00 | 0.00 | 36 |
| pdf | 0.00 | 0.00 | 12 |
| symlink_or_special | 0.00 | 0.00 | 84 |

## By worktree (untracked)

| worktree | logical GiB | allocated GiB |
|---|---|---|
| residual-spectral-20260910 | 22.55 | 22.61 |
| PCRL | 20.60 | 21.24 |
| PCRL-terminal-1-competitive | 13.55 | 13.73 |
| PCRL-terminal-1-adversarial | 9.92 | 9.98 |
| PCRL-terminal-a | 2.90 | 2.93 |
| PCRL-terminal-1-invariant | 2.83 | 2.85 |
| PCRL-terminal-b | 0.40 | 0.40 |
| laftr-hard-r2-2026-05-17 | 0.00 | 0.00 |
| PCRL-terminal-2-manuscript | 0.00 | 0.00 |
| PCRL-terminal-1-utility | 0.00 | 0.00 |

## Classification and plan

| group | GiB | class | action |
|---|---|---|---|
| rs_spectral_transport | 14.90 | essential evidence (locked 2017 transport study: fitted maps, attackers, predictions) | archive -> verify on AWS -> unlink exact manifest-listed copies |
| rs_residual_spectral | 7.25 | essential evidence (H/A0/J historical audits incl. the H ancestor slate read by every audit) | archive -> verify on AWS -> unlink exact manifest-listed copies |
| competitive | 13.55 | essential evidence (372-slot study: releases, fitted audit attackers, predictions, fits, 2017) | archive -> verify on AWS -> unlink exact manifest-listed copies |
| direct_adversarial | 9.92 | essential evidence (direct adversarial study) | archive -> verify on AWS -> unlink exact manifest-listed copies |
| invariant | 2.83 | essential evidence (leace_A0 / splince_A0 / optnet releases and audits) | archive -> verify on AWS -> unlink exact manifest-listed copies |
| nonlinear_rank | 2.90 | essential evidence (nonlinear/rank study) | archive -> verify on AWS -> unlink exact manifest-listed copies |

Everything scientific is treated as essential evidence and archived before any local removal, including fitted attacker weights (a previous cleanup deleted attacker weights that transport later needed). Nothing is deleted on the basis of a reproducibility argument.

Kept locally regardless: source code, git metadata, raw data (`data/`), `.venv`, the main checkout (another session works there), Terminal 2's manuscript worktree, current reports/PDFs, manifests, restore instructions, and the execution inputs (also archived as `exec_*` chunks for the cloud host).

Disposable caches removed without archive (claim `t1__pycache_historical_t1_worktrees`): 47 `__pycache__`/`.pytest_cache` directories in Terminal-1 historical worktrees, 10.1 MiB; regenerated automatically on import. Measured free space before/after: 23,744,868 KB -> 23,755,924 KB.

Status: archive **not yet uploaded** — the AWS CLI session is expired; no local evidence has been removed.
