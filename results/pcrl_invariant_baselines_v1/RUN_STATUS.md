# RUN_STATUS — Terminal 1, rotation-invariant objective + external baselines

Restartable ledger. Every stage records its own resume command. A cache read is
never counted as a new fit.

## Identity

| Field | Value |
|---|---|
| Branch | `research/pcrl-invariant-baselines-v1` |
| Worktree | `/Users/nathansamson/PCRL-terminal-1-invariant` |
| Base commit | `c37807e4f568ef38e5528fc09c1506083278bf4d` (completed method work) |
| Historical common baseline | `349efa454afd907389760fd1f59fd8806a215efd` |
| Evidence/manuscript branch (read-only) | `0d8f4b67b6d4961dfa133289d0167c874d2f4794` |
| Owned paths | `experiments/pcrl_invariant_baselines_v1/`, `tests/pcrl_invariant_baselines_v1/`, `results/pcrl_invariant_baselines_v1/` |
| Handoff root | `/Users/nathansamson/PCRL/.git/pcrl_parallel_handoff_v2/terminal_1/` |

## Machine and concurrency policy

Measured at session start (2026-09-18T02:20Z):

* Apple CPU, 14 cores, 24 GB RAM.
* **Swap at capacity**: `total = 29696 MB, used = 28164 MB, free = 1531 MB`.
  System-wide free memory 23%. Load average 6.24.
* Unrelated research jobs are running and are **not** touched: `tq.pilot` /
  `tq.supervisor` under `~/removal-pricing` (PIDs 67516/67501/67498), deadline
  `2026-09-18T18:16:34Z`, plus one long-running Python process (PID 89913).

Policy, in force for every stage below:

* **One fitting worker.** `OMP/OPENBLAS/MKL/VECLIB_MAXIMUM_THREADS=1`.
* Audit suites never run concurrently with a model-fitting job.
* Exact kernel blocks and prediction batches are chunked; no `n`-by-`n` matrix over
  the full representation pool is ever formed.
* Peak RSS is measured and recorded per stage.
* Memory pressure was *observed*. Its causal role in the predecessor's rare
  numerical faults is **suspected, not demonstrated**, and nothing here upgrades it.

## Stage ledger

Status values: `PENDING`, `RUNNING`, `DONE`, `BLOCKED`, `SCOPED OUT (evidence)`.

| # | Stage | Status | Artifact | Resume |
|---|---|---|---|---|
| 0 | Worktree, artifact recovery, required reading | DONE | this file | — |
| 1 | `PROTOCOL.md` + `METHOD.md` frozen and committed | PENDING | `PROTOCOL.md`, `PROTOCOL_FREEZE.json` | — |
| 2 | Rotation-invariant objective implemented | PENDING | `experiments/pcrl_invariant_baselines_v1/invariant_moment.py` | — |
| 3 | Invariance + gradient validation (the gate) | PENDING | `INVARIANCE_VALIDATION.md` | `pytest tests/pcrl_invariant_baselines_v1` |
| 4 | Repaired matrix: 18 mapper fits | PENDING | `MATRIX.json` | `python -m experiments.pcrl_invariant_baselines_v1.run_fit` |
| 5a | Spectral/SARL alias audit | PENDING | `BASELINE_ADAPTATIONS.md` | — |
| 5b | LEACE auxiliary-channel baseline | PENDING | `BASELINE_ADAPTATIONS.md` | — |
| 5c | SPLINCE auxiliary-channel baseline | PENDING | `BASELINE_ADAPTATIONS.md` | — |
| 5d | OptNet-ARL adaptation: 9 mapper fits | PENDING | `BASELINE_ADAPTATIONS.md` | — |
| 6 | 2018 development evaluation | PENDING | `DEVELOPMENT_2018.md` | `python -m experiments.pcrl_invariant_baselines_v1.run_dev_2018` |
| 7 | Exploratory 2017 evaluation | PENDING | `EXPLORATORY_2017.md` | `python -m experiments.pcrl_invariant_baselines_v1.run_exploratory_2017` |
| 8 | Reports, decision, handoff | PENDING | `RESEARCH_DECISION.md`, `PAPER_ADDENDUM.md`, `HANDOFF.json` | — |

## Recovered read-only inputs

Resolved and hashed at stage 0; none is refitted, moved or modified.

| Input | Location |
|---|---|
| Frozen 2018 `SpectralModel` (V, U, bases, 15 OOF nuisances), seeds 0-2 | `~/.config/superpowers/worktrees/PCRL/residual-spectral-20260910/results/redesign_20260910_acs_residual_spectral_v1/seed_N/maps.joblib` |
| Pools, PCA teacher, anchors, subset indices | `~/PCRL/results/redesign_20260909_acs_fixed_predictions_v1/seed_N/` |
| Frozen interfaces `H`, `E`, `A0`, `L025`, `L20`, `J` | same, `seed_N/training/<arm>/releases.npz` |
| Defective nonlinear maps at ranks 8 and 16 | `~/PCRL-terminal-a/results/pcrl_nonlinear_rank_v1/seed_N/maps.joblib` |
| Original locked 2017 transport study | `~/PCRL-terminal-a/results/redesign_20260917_acs_spectral_transport_v1/` |
| 2018 raw ACS | `~/PCRL/data/folktables/2018/1-Year/psam_p06.csv` |

`A0`'s A-wire is 20 columns = `H_A` (4, bitwise preserved) + the 16-coordinate
auxiliary channel. That channel is the input to the erasure baselines.

## Measured runtime

Filled in as stages complete. Predecessor reference on the same machine: 27 unique
fits in 1324 s, 27 dev-2018 audit units in 1106 s, peak RSS 647 MB.

| Stage | Measured |
|---|---|
| (pending) | |

## Fit ledger

| Quantity | Planned | Actual |
|---|---:|---:|
| Repaired rotation-invariant mapper fits (3 policies x 2 ranks x 3 seeds) | 18 | — |
| Linear-erasure baseline fits (LEACE, SPLINCE x 3 seeds) | 6 | — |
| OptNet-ARL adaptation fits (3 policies x rank 16 x 3 seeds) | 9 | — |
| **Nominal total new mapper fits** | **33** | — |
| Reused matched-rank original controls (never refitted) | 9 | — |
| Reused defective-nonlinear controls (never refitted) | 18 | — |

Counts fall below nominal where an alias is proved or a formulation is documented
infeasible; each such reduction is recorded with its evidence.

## Failures and repairs

Recorded as they happen so the run stays auditable.

*(none yet)*

## Amendments to the registered protocol

Recorded with timing and affected units.

*(none yet)*

## Scope boundaries in force

* **2016 is not scored, fitted, transformed, tuned on, or inspected for outcome
  distributions in this assignment.** Its admission (Terminal 2's completed work) is
  provenance/schema only; the year remains UNSCORED.
* 2018 is development. 2017 is *also* development for these new methods and is
  labelled so everywhere; the original locked 2017 result keeps its historical
  status and is not relabelled.
* No paid compute, no message sending, no paper submission, no raw survey upload.
