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
