# MANUSCRIPT_STORAGE_LEDGER — Terminal 2, 2026-09-19, updated 2026-09-20

Public ledger; no local absolute paths. The path-level private inventory and the request to Terminal 1 are
in the shared local handoff (`pcrl_parallel_handoff_v5/terminal_2/`). Nothing was moved to Trash; every figure
below is a real unlink. No Git history, snapshot, branch, worktree, private AAAI material or unrelated file was
touched. Filesystem free space moved during this session for reasons outside Terminal 2 (other terminals'
cleanup), so it is **not** used as the measure of what Terminal 2 freed; sizes below are summed per file.

## Deleted (Terminal 2 owned, reproducible)

| category | paths | bytes | surviving source / regeneration |
|---|---|---:|---|
| LaTeX auxiliaries, manuscripts v2–v4 (`.aux .bbl .blg .fdb_latexmk .fls .log .out`) | 18 files | 299,531 | `main.tex` + `references.bib` committed; `latexmk -pdf main.tex` regenerates; each version's `main.pdf` is committed and kept |
| Python bytecode / test cache (`__pycache__`, `.pytest_cache`) in the manuscript worktree | 8 dirs | 524,288 (du, 4 KB blocks) | regenerated on import / test run |
| **Pre-existing space reclaimed** | | **823,819 (~0.8 MB)** | |
| Session intermediates created and removed by this session (evidence extraction for reading, page-render previews, v5 LaTeX auxiliaries) | 3 groups | ~5.2 MB | evidence at `7f961d5c` in Git; renders reproducible with `pdftoppm`; LaTeX as above |

Copies of v4 figures/tables that the v5 draft stopped using were removed from the **v5** directory only; the
committed v4 originals are untouched.

## Added by this session
`papers/pcrl_manuscript_v5/` (~3.4 MB, including the committed PDF) and `results/pcrl_manuscript_review_v5/`
(~0.1 MB). Net effect of Terminal 2 on disk: roughly −0.8 MB freed, +3.5 MB committed deliverables.

## Preserved exceptions
All manuscript sources, bibliography, figures, every committed compiled PDF (v2–v5, documenting reviewed
versions), scripts, claim ledgers, review notes, source manifests, private material, Git metadata.

## Pending on Terminal 1 — resolved 2026-09-20, no action needed

Terminal 2 had asked to be assigned a `git sparse-checkout` of the ~1.9 GB of **Git-tracked** experimental
results in this worktree (largest: `results/pcrl_nonlinear_rank_v1`, ~0.9 GB with 1,539 tracked `.joblib`
files). No assignment was issued, and Terminal 2 never acted on them.

**The request is withdrawn as unnecessary.** Terminal 1's own cleanup archived 29 chunks / 154,360
*untracked* experiment files to private S3 — reading each chunk back on a different machine, checking the
stream hash, extracting it and re-hashing every file, then re-verifying size, inode, mtime and a freshly
computed SHA-256 immediately before each unlink (0 skipped, 0 failures) — and freed **51.69 GiB**, taking
filesystem free space from about 15 GiB at session start to **97.2 GiB**. It explicitly did not touch this
worktree, the main checkout, symlinks, fitted attacker weights (archived in full, deliberately not compacted
after the earlier incident where needed weights were lost) or the sealed 2016 data.

With the disk pressure gone, hiding 1.9 GB of tracked files that are also on `origin` buys little and costs
reviewers a `git sparse-checkout disable` to read the evidence. The option stays documented here if space is
ever needed again; nothing about it is blocking.

## Net position

| | |
|---|---|
| Freed by Terminal 2 (pre-existing, real unlinks) | 823,819 B (~0.8 MB) |
| Added by Terminal 2 (v5 + v6 deliverables, incl. committed PDFs) | ~7 MB |
| Freed by Terminal 1 (archive-verified) | 51.69 GiB |
| Filesystem free now | 97.2 GiB |

Terminal 2 claims only the first row. The large figure is Terminal 1's work, recorded here for context and
not as its own.
