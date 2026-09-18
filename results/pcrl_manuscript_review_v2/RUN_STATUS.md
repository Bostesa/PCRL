# RUN_STATUS — Terminal 2 (manuscript integration)

Branch `research/pcrl-manuscript-integrated-v2`, worktree `/Users/nathansamson/PCRL-terminal-2-manuscript`.
Restartable: each stage below is idempotent and its outputs are committed.

## Integrated sources

| Role | Branch | SHA |
|---|---|---|
| Common baseline | -- | `349efa454afd907389760fd1f59fd8806a215efd` |
| Completed evidence/manuscript work | `research/pcrl-evidence-paper-v1` | `0d8f4b67b6d4961dfa133289d0167c874d2f4794` |
| Completed nonlinear/rank method work | `research/pcrl-nonlinear-rank-v1` | `c37807e4f568ef38e5528fc09c1506083278bf4d` |

Both descend from the baseline and touch disjoint paths; the merge was clean with no conflicts.
Neither original worktree, nor `main`, was touched.

## Stages

| # | Stage | State |
|---|---|---|
| S1 | Worktree + branch, merge both completed studies | done |
| S2 | Read all evidence; verify headline numbers from machine-readable artifacts | done |
| S3 | Asset generator `experiments/pcrl_manuscript_v2/make_assets_v2.py` | done |
| S4 | Claim ledger and claim changes | done |
| S5 | Integrated manuscript source + PDF | done |
| S6 | Verification-record repair (stale `INDEPENDENT_VERIFICATION.json`) | done |
| S7 | Related-work scope, reviewer risks, corrected decision, reproduce, review index | done |
| S8 | Terminal 1 incorporation | pending — no committed Terminal 1 result exists |

## Measured runtime

`make_assets_v2.py` runs in well under a minute on one CPU worker with one BLAS thread.
No model fitting, scoring or refitting is performed by this terminal.

## Concurrency and memory

One CPU worker; `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` set in the generator.
No unrelated process, VM, container or research job was touched. The `removal-pricing` jobs
running on this machine belong to a different project and were left alone.

## Final state

| Item | Result |
|---|---|
| Manuscript | `papers/pcrl_manuscript_v2/main.pdf`, **22 pages** |
| Build | `latexmk -pdf`, exit 0, **0 overfull/underfull boxes, 0 undefined references or citations, 0 LaTeX warnings**; every page rendered and inspected |
| Generated assets | 13 tables, 7 figures, each with its source files sha256-recorded in `papers/pcrl_manuscript_v2/MANIFEST.json` |
| Claim ledger | 25 claims, values recomputed from evidence |
| Tests | `44 passed` (`tests/pcrl_evidence_review_v1` + `tests/pcrl_nonlinear_rank_v1`), 4.5 s |
| Corrected lock verification | 17,639 inputs, 1.80 GiB, all three amendments, **0 changed / 0 missing / 0 invalid**, 5.1 s |
| Unique model fits performed by this terminal | **0** |
| 2016 touched | **no** — not scored, not fitted, no outcome distribution inspected |

## Measured runtime, this terminal

| Step | Wall time |
|---|---|
| `make_assets_v2.py` | ~6 s |
| `build_claim_ledger.py` | ~1 s |
| `recheck_verification.py` (1.80 GiB hashed) | 5.1 s |
| `build_integration_manifest.py` | ~2 s |
| `pytest` (44 tests) | 4.5 s |
| `latexmk -pdf` (full run with bibtex) | ~20 s |

No scientific fit, score or refit was run here, so there is no fit count to report beyond zero.

## S8 — Terminal 1

Checked at every stage boundary. Terminal 1's worktree `/Users/nathansamson/PCRL-terminal-a` is
clean at `c37807e`, which is the nonlinear/rank study **already integrated here**. No new
experiment directory, protocol, handoff file or running job belonging to that terminal was found;
the Python processes on this machine belong to the unrelated `removal-pricing` project and were
left alone. Nothing of Terminal 1's is incorporated and nothing is claimed. The exact integration
specification is in [PENDING_ADDENDUM.md](PENDING_ADDENDUM.md); the handoff is
[HANDOFF.json](HANDOFF.json), mirrored to
`<git common dir>/pcrl_parallel_handoff_v2/terminal_2/`.
