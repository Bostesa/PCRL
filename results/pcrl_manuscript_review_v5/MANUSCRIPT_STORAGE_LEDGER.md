# MANUSCRIPT_STORAGE_LEDGER — Terminal 2, 2026-09-19

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

## Pending on Terminal 1 (archive-dependent; NOT done)
The manuscript worktree holds ~1.9 GB of **Git-tracked** experimental results (largest:
`results/pcrl_nonlinear_rank_v1`, ~0.9 GB, including 1,539 tracked `.joblib` model files). They are byte-identical
to blobs in the shared object store and on `origin` (the worktree HEAD is contained in
`origin/research/pcrl-manuscript-integrated-v4`). Proposed action: exclude them from this worktree with
`git sparse-checkout` (restore: `git sparse-checkout disable`); no object is deleted. Per the storage rules,
Terminal 2 does not act on experimental artefacts until Terminal 1 assigns the path in the cleanup-claim
registry. No assignment had arrived at publication. Nothing untracked of scientific value exists in this worktree.
