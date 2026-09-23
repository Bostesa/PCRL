# Cleanup ledger — manuscript worktree

Owner: manuscript assignment, worktree `PCRL-t2-finish`, branch `research/pcrl-submission-finish-v1`.
Date 2026-09-23. **Nothing outside this worktree was touched. No Git history was rewritten. No object was
deleted. No other assignment's directory, credential, untracked unique file or review was removed.**

## Why cleanup happened at all

Free space was **12.2 GiB (98% used)** at the start of this session, against 97 GiB a day earlier. Storage
pressure is real again, so acting was justified; it would not have been if space were adequate.

## What was actually freed

| action | bytes freed | recoverability |
|---|---:|---|
| `git sparse-checkout` of duplicate experimental result trees in **this worktree only** | **1,995,648 KB = 1.90 GB** | `git sparse-checkout disable` restores every file; no object was deleted |
| LaTeX auxiliaries rebuilt by `latexmk` (13 files) | 145,276 B = 0.14 MB | `latexmk -pdf main.tex` |

Measured as `du -sk` of the worktree before (2,020,524 KB) and after (24,876 KB). **This is a measurement
of the worktree, not of free disk space**, which also moves for reasons outside this assignment.

## Classification of this worktree's large paths

| class | paths | disposition |
|---|---|---|
| Unique scientific evidence owned here | `results/pcrl_submission_review_v1/` (verification JSONs, ledger, corrections) | **kept and committed** |
| My review packages from earlier rounds | `results/pcrl_manuscript_review_v5,v6/`, `results/pcrl_manuscript_overnight_v1/` | **kept** |
| Sources and deliverables | `papers/**`, `artifact/**` including every final PDF | **kept** |
| Verified duplicates | `results/pcrl_nonlinear_rank_v1/` (905 MB), `results/redesign_*` (≈1 GB total), other historical study trees | **hidden from the working tree, not deleted** |
| Unknown ownership | none found in this worktree | — |

## How recoverability was verified before hiding anything

1. All 8,745 files under `results/` are **tracked**; only 3 files were untracked, and all three are my own
   new deliverables, which were kept and committed.
2. `HEAD` is contained in `origin/research/pcrl-submission-finish-v1`, so every blob is on the remote.
3. Sampled blob identity against the remote with `git hash-object` versus `git rev-parse origin/…:path`
   — exact match.
4. After the change: **zero** deletions staged, tracked file count unchanged at 8,745, and all four
   verification scripts still pass (29 + 24 + 15 + 12 checks). They read evidence through
   `git show <sha>:<path>`, never the working tree, which is why hiding working copies is safe here.

## Coordination

The hidden trees are duplicates of committed evidence that other assignments hold in their own worktrees;
nothing was removed from the shared object store, the remote, or the study owners' checkouts. The
prospective study's private archive was **not** re-archived and **not** touched. If any owner needs these
paths materialised in this worktree, `git sparse-checkout disable` is a one-line restore.

**Large deletion was not treated as a goal.** The 1.90 GB is duplicate checkout, and the only thing
genuinely destroyed anywhere was 0.14 MB of rebuildable LaTeX output.
