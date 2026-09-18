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
| 1 | `PROTOCOL.md` + `METHOD.md` frozen and committed | DONE | `PROTOCOL.md`, `PROTOCOL_FREEZE.json` | — |
| 2 | Rotation-invariant objective implemented | DONE | `experiments/pcrl_invariant_baselines_v1/invariant_moment.py` | — |
| 3 | Invariance + gradient validation (the gate) | DONE | `INVARIANCE_VALIDATION.md` | `pytest tests/pcrl_invariant_baselines_v1` |
| 4 | Repaired matrix: 18 mapper fits | DONE | `MATRIX.json` | `python -m experiments.pcrl_invariant_baselines_v1.run_fit` |
| 5a | Spectral/SARL alias audit | DONE (0 fits: alias proved) | `BASELINE_ADAPTATIONS.md` | — |
| 5b | LEACE auxiliary-channel baseline | DONE | `BASELINE_ADAPTATIONS.md` | — |
| 5c | SPLINCE auxiliary-channel baseline | DONE (feasible) | `BASELINE_ADAPTATIONS.md` | — |
| 5d | OptNet-ARL adaptation: 9 mapper fits | DONE | `BASELINE_ADAPTATIONS.md` | — |
| 6 | 2018 development evaluation | DONE (33 units) | `DEVELOPMENT_2018.md` | `python -m experiments.pcrl_invariant_baselines_v1.run_dev_2018` |
| 7 | Exploratory 2017 evaluation | DONE (repaired arms only) | `EXPLORATORY_2017.md` | `python -m experiments.pcrl_invariant_baselines_v1.run_exploratory_2017` |
| 8 | Reports, decision, handoff | DONE | `RESEARCH_DECISION.md`, `PAPER_ADDENDUM.md`, `HANDOFF.json` | — |

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
| Validation suite (40 fixtures) | 10 s |
| Fit 18 repaired conditions (3 seeds) | 949 s |
| Mechanism gate + cross-family objective matrix | 12 s |
| SARL alias audit (fits nothing) | 3 s |
| LEACE + SPLINCE, 6 fits | 11 s |
| OptNet-ARL, 9 fits (1200 steps x 2 starts each) | 1362 s |
| 2018 development evaluation, 33 audit units | 1345 s |
| 2018 endpoints, bootstrap and decisions | 34 s |
| Exploratory 2017 fit, 18 units | ~1500 s |
| Exploratory 2017 scoring, 18 units | ~1400 s |
| **Total new compute** | **~1.9 hours at one worker** |

Peak resident set size during fitting: **369 MB**.

## Fit ledger

| Quantity | Planned | Actual |
|---|---:|---:|
| Repaired rotation-invariant mapper fits (3 policies x 2 ranks x 3 seeds) | 18 | **18** |
| Linear-erasure baseline fits (LEACE, SPLINCE x 3 seeds) | 6 | **6** |
| OptNet-ARL adaptation fits (3 policies x rank 16 x 3 seeds) | 9 | **9** |
| **Nominal total new mapper fits** | **33** | **33** |
| Reused matched-rank original controls (never refitted) | 9 | **9** |
| Reused defective-nonlinear controls (never refitted) | 18 | **18** |
| SARL duplicate fits avoided by the alias audit | — | **6** |

Counts fall below nominal where an alias is proved or a formulation is documented
infeasible; each such reduction is recorded with its evidence.

## Failures and repairs

Recorded as they happen so the run stays auditable.

1. **A validation fixture asserted the wrong thing; the code was correct.** The
   conditional-null fixture was written to assert that *both* penalty blocks
   concentrate towards zero as the representation pool `n` grows. The quadratic block
   does (4.9e-4 at n=2000 -> 3.3e-6 at n=128000). The kernel block does **not**, and
   should not: it averages over the frozen subset of at most `m = 512` rows, so its
   sample size is `m` regardless of `n`. The fixture was split into two -- quadratic
   concentration in `n`, kernel concentration in `m` -- and the underlying property was
   measured and recorded as a limitation (`METHOD.md` section 3.7). **A fixture
   expectation was corrected; no measure, constant or datum was changed.** Timing:
   before any fit and before any outcome.
2. **A reporting bug in an OptNet diagnostic, found and fixed; no fitted value affected.**
   `utility_signal_ratio` was written without its `1/n` factor, so it recorded `10513.0`
   where the quantity is an OLS R-squared on the `[0, 1]` scale. Because `V` is whitened
   to `V'V/n = I`, the explained sum of squares is `||V'R||_F^2 / n`. **The corrected
   value is 1.000**, i.e. the residualised teacher lies entirely in the span of the
   whitened features, which is the healthy case the source review's warning was about
   (a value near 0 would mean residualisation had annihilated the utility signal and the
   encoder would be empty). The code is fixed for future runs; this run's stored field is
   unnormalised and is reported as such. **The diagnostic is a report-only field: it
   enters no objective, no selection and no fitted map.** Timing: during the OptNet fits,
   before any OptNet score was read.
3. `KernelBlock.build` derived `role_index` only from the frozen `ROLE_ORDER`, which
   synthetic fixtures cannot satisfy. `role_index` is now an explicit optional
   parameter defaulting to the `ROLE_ORDER` position, so every production fit is
   unchanged. Timing: before any fit.

4. **Sporadic corruption of in-memory prediction output, captured and characterised.**
   *(This entry supersedes two earlier versions. The first called the fault "transient"
   on a single occurrence; the second claimed it struck at a fixed position in the unit
   sequence. Both were wrong and both are corrected here. The evidence below is the
   captured artefact, not an inference.)*

   **What happened.** The exploratory 2017 stage aborted four times with "Probabilities
   must be finite, normalized and aligned with the full class schema", raised by
   `acs_transfer_heads.metrics`:

   | # | Phase | Unit | Position in that process |
   |---|---|---|---|
   | 1 | fit | `seed_0/spectral_riv8_C1` | 4th computed |
   | 2 | fit | `seed_2/spectral_riv8_C1` | 13th computed |
   | 3 | fit | `seed_2/spectral_riv8_L2` | 3rd computed |
   | 4 | score | `seed_0` (2nd scored unit) | 2nd computed |

   **The captured signature.** Wrapping every `metrics` call in a pass-through observer
   caught the offending array:

   ```
   BAD ARRAY: rowsum [0, 1.00000017]  worst deviation 1.000e+00
   ```

   At least one row of a probability matrix summed to **exactly zero** — an all-zero
   block — while the rest of the array was well-formed (max row sum 1.00000017, i.e.
   normal floating-point noise). **This is a zeroed memory block, not a modelling,
   schema or support failure.**

   **Hypotheses tested and rejected**, before any re-run:

   * *Systematic support/schema failure of the rank-8 wire.* `RAC1P` class 3 has support
     2 in the 2017 `attacker_fit` partition and 0 in `attacker_validation`. **Rejected:**
     the rank-16 arms score cleanly against identical labels; the rank-8 arms themselves
     complete cleanly on recompute; and a support failure would produce a wrong **width**,
     not a zeroed row inside a correct-width array.
   * *Cross-condition cache collision.* **Rejected by reading the code:** `Loaded` is
     per-unit and its key includes `array_hash(x)`, so a width-20 wire's prediction
     cannot be served to a width-12 wire.
   * *Fixed position in the unit sequence.* **Rejected by the table above:** 4th, 13th,
     3rd and 2nd. The fault is **sporadic**, not positional.
   * *The double-compute guard firing.* `stable_predict` computes each array twice and
     raises `ArithmeticError` on persistent disagreement. It did not raise, so the
     corruption occurred **after** the guard — consistent with the cached, read-only
     array being damaged in memory between storage and use.

   **What this does and does not establish.** It establishes that a well-formed
   prediction array acquired an all-zero row in memory, sporadically, on a machine that
   was at **swap capacity for the entire run**, and that the identical computation
   succeeds on recompute every time. It matches the transport study's documented
   precedent — `stable_predict`'s own docstring records "rare, load-dependent corrupted
   blocks ... in CPU prediction outputs" — and the predecessor's instance. **It does not
   establish causation.** Memory pressure remains a **SUSPECTED** condition. Four
   occurrences on one machine in one session is a pattern, not a mechanism, and no claim
   is made about the hardware.

   **The integrity system worked.** Every occurrence was **detected** by the scorer's
   full-schema validation and **aborted the unit before it wrote a metrics file**. No
   corrupted array was ever scored, cached into a table, averaged, or silently
   tolerated. This is the guard doing precisely what it exists for.

   **Operational response, bounded and recorded as a workaround rather than a fix:** the
   stage is driven by its resume markers, which skip completed units, inside a retry
   loop with a **hard cap of 12 attempts**. Progress is monotone because a unit that
   aborts writes no `complete.json`. This is not an unbounded sweep and it changes no
   number — a unit either completes cleanly or is not used.

   **Disposition.** Aborted units wrote no metrics file. Every unit computed under the
   diagnostic wrapper was **quarantined** with a `QUARANTINE.json` and recomputed by the
   **unpatched** pipeline, even though the wrapper returns `real(...)` unconditionally,
   so **no unit used in any table was produced under a patched scorer**. Nothing is
   merged, averaged or voted between copies.

## Amendments to the registered protocol

Recorded with timing and affected units.

1. **`METHOD.md` section 3.7 added** -- the kernel block's conditional-null floor is
   governed by the frozen subset size `m`, not by the representation pool, with the
   measured `1/m` decay table. Timing: **before any fit and before any ACS outcome was
   opened.** Affected units: none fitted yet. This **adds a documented limitation**; it
   changes no equation, constant, subset, seed, endpoint, decision rule or registered
   forecast. `m = 512` was deliberately **not** changed in response to the
   measurement.
2. **OptNet-ARL optimiser budget set to 1200 Adam steps per start, two starts**, in place
   of the 200 full-objective updates used by every other arm. Timing: **after a
   training-only calibration probe and before any OptNet fit or any ACS outcome.**
   Affected units: the 9 OptNet conditions only.

   Justification, which the protocol explicitly permits ("a justified prospectively
   documented comparable budget after training-only calibration"): 200 mini-batch Adam
   steps at batch 512 is about 10 epochs of gradient signal over a ~12k-parameter
   encoder, and is not comparable to 200 full-objective Riemannian updates on a
   128x16 Stiefel point. A comparable budget should equalise optimisation *progress*,
   not step count. A single-start training-only probe (`OPTNET_CALIBRATION.json`) shows
   the training objective improving by 0.030 between steps 800 and 1000 but only 0.003
   between 1000 and 1200 -- a tenfold deceleration, leaving the run within ~0.4% of its
   plateau. Measured cost 0.085 s/step, so 9 conditions x 2 starts x 1200 steps is
   ~31 minutes at one worker. **The probe read the training objective and the wall
   clock only; no attacker, residence, commute or development outcome is reachable
   from it.**

## Scope boundaries in force

* **2016 is not scored, fitted, transformed, tuned on, or inspected for outcome
  distributions in this assignment.** Its admission (Terminal 2's completed work) is
  provenance/schema only; the year remains UNSCORED.
* 2018 is development. 2017 is *also* development for these new methods and is
  labelled so everywhere; the original locked 2017 result keeps its historical
  status and is not relabelled.
* No paid compute, no message sending, no paper submission, no raw survey upload.
