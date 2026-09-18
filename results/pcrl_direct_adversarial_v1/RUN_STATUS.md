# RUN_STATUS — direct adversarial channel refinement

Session start `2026-09-18T12:26:53Z`. Nine-hour execution ceiling, last 45 minutes
reserved for accounting and publication.

## Amendments

### Amendment 1 — equal-budget fresh-probe checkpoint selection

**Declared `2026-09-18T13:05Z`, after the training-only timing pilot and BEFORE any
2018 or 2017 outcome for any new arm. No fit made under a superseded rule is carried
into any table.**

`METHOD.md` §5.2 originally scored each checkpoint with the attacker slate
*contemporaneous* with it. **That is biased**, and so is the obvious repair, and both
biases point the same way — toward never moving the channel at all.

* **Contemporaneous slate.** The slate at step 0 has had only the 100-update warmup;
  the slate at step 600 has had 3000 updates plus three refreshes. The measured penalty
  therefore rises with training **even when the channel does not move**. On the pilot's
  `beta = 0` arm, which is under no penalty pressure at all, the monitor penalty rose
  `0.059 -> 0.094 -> 0.083` across checkpoints.
* **Final slate (the first repair, also rejected).** Re-scoring every checkpoint
  against the single final slate is worse in a different way: that slate was trained
  against the *final* channel and generalises poorly backwards, so it **understates**
  what is recoverable from an early checkpoint. Measured on `dax16_C1_b100` seed 0, the
  step-0 penalty read `0.0593` contemporaneously but only `0.0419` under the final
  slate — a 29% understatement of the starting point's recoverability. Under that rule
  **every one of the first 11 arms fitted selected step 0**, i.e. the unmoved `A0`
  channel, which would have made the whole study vacuous by construction.

**The rule is therefore:** each checkpoint is scored against **its own freshly
initialised attacker slate**, using the same slot schedule and families, trained for the
**same fixed budget of 300 attacker minibatch updates** against that frozen channel on
`mapper_fit`, and read on `monitor`. Selection is `argmin` of that score, ties to the
earlier step. Equal budget across checkpoints is what makes it a yardstick. On the same
`dax16_C1_b100` cell the fresh probe reads the step-0 penalty as `0.094` — above both
biased slates — and selection moves to step 300.

The contemporaneous and final-slate scores are still computed and stored
(`contemporaneous_monitor_scores`, `final_slate_monitor_scores`) and are used for the
training-attacker-versus-fresh-auditor analysis.

Nothing about the objective, the beta grid, the update budget, the folds or the endpoint
set changes. The rule still uses only `U` and the policy penalty on an internal
representation-training fold: **no residence, no commute, no downstream pool, no test
pool.** This is a **selection** yardstick and not a protection measurement; 300 fresh
updates on three differentiable families is far weaker than the 2018 audit slate, and
every protection claim in this study rests on the audit.

The change was made because the original rule was methodologically wrong, not because a
number came out a particular way: **no outcome had been opened.** Launch 1 (PID 50964,
3 arms) and launch 2 (PID 51166/51472, 11 arms) were stopped and their partial output
deleted.

### Amendment 2 — checkpoint grid corrected to the frozen value

Launch 2 ran with `checkpoint_every = 50` (the dataclass default) against the frozen
`METHOD.md` §5.2 value of **100**. The code was corrected to the frozen value rather
than the document to the code. 13 checkpoints would have been strictly more evidence
than 7, but honouring a just-frozen protocol matters more than a discretionary grid.

### Amendment 3 — the candidate-wide correction is studentized, not percentile

**Declared `2026-09-18T13:00Z`, before any 2018 or 2017 result was read.**

`PROTOCOL.md` §5 registered the candidate-wide adjustment as "per-comparison two-sided
bootstrap quantiles at level `alpha / m`". The realised family is **86 contrasts x 5
family endpoints x 2 weightings = 860**, so `alpha / m = 5.8e-5` and the required tail
quantile sits at `2.9e-5`. With 2000 bootstrap replicates that quantile is **not
estimable**: it lies beyond the most extreme replicate (`2000 x 2.9e-5 = 0.06` of one
observation).

The correction used is therefore the conservative alternative the same section
authorises: a **studentized Bonferroni bound**, `estimate +/- z_{1 - alpha/(2m)} x
bootstrap_SE`, with `z = 4.02` at `m = 860`. Every row carries a `percentile_estimable`
flag recording that the registered percentile form was unavailable, so the substitution
is visible rather than silent. The reused within-contrast studentized max-|t| bounds
remain in the same table under their own column names, so the predecessor's tables stay
directly comparable.

### Amendment 4 — the primary 2018 scope is matched-exposure, not catch-up

**Declared `2026-09-18T13:25Z`, after the first two audit units of seed 0 and before
any comparison was computed. The bias it removes ran in this study's own favour.**

`METHOD.md` §7 reuses the predecessors' primary 2018 scope, `kernel_expanded_catchup`.
Under that scope the historical arms carry **saved-observer catch-up candidates** that
no arm fitted in this study has — this study has no saved observer, which `METHOD.md`
§7 itself states. The two sides of every new-versus-historical contrast are therefore
attacked with **different budgets**, and the new arms are the ones attacked less.

The size of the effect was measured directly, on the cleanest possible case.
`dax16_C1_b010` selected its unmoved initial checkpoint, so its released channel is
**bitwise identical to `A0`** (release hash `d0e97e9dcdf0` on `wire/A/test`, equal to
`dax16_none` and to the frozen `A0` channel). Additional recovery over `H`, seed 0,
unweighted, budget 360:

| scope | arm | `A/SEX` | `AB/SEX` | `A/RAC1P` | `AB/RAC1P` |
|---|---|---|---|---|---|
| `kernel_expanded_independent` | `A0` | +0.0340 | +0.0314 | +0.0548 | +0.0359 |
| `kernel_expanded_independent` | `dax16_C1_b010` | +0.0340 | +0.0314 | **+0.0548** | +0.0359 |
| `kernel_expanded_catchup` | `A0` | +0.0305 | +0.0255 | +0.0664 | +0.0384 |
| `kernel_expanded_catchup` | `dax16_C1_b010` | +0.0305 | +0.0255 | **+0.0509** | +0.0279 |

Under the matched-exposure scope the two rows are **exactly equal, as they must be for
a bitwise-identical channel**. Under the catch-up scope the identical channel appears
**0.0155 better** on `A/RAC1P` and 0.0105 better on `AB/RAC1P`, purely because `A0`'s
selected attacker there is a `catchup` candidate and the new arm has none.

**The primary 2018 scope is therefore `kernel_expanded_independent`.** The catch-up
scope remains in `PER_SEED.csv` and is reported alongside, explicitly labelled as giving
the historical arms exposure the new arms never had. No equal-total-history claim is
made in either direction.

The exact-equality row is also the strongest end-to-end correctness check this study
has: an independently re-audited channel that is bitwise identical to a historical one
reproduces its endpoints exactly. It is recorded as a validation result, not just as a
scope note.

## Resource decisions

* Machine: 14 CPUs, 24 GB, shared. Load average at session start `8.20 / 7.72 / 7.01`,
  with a Docker VM holding 14 CPUs and 8 GB and other interactive sessions present.
* One experimental worker per phase, one BLAS/OpenMP thread, `torch.set_num_threads(1)`.

**Concurrency was measured and then declined.** At `13:19Z`:

```
vm.swapusage: total = 29696.00M  used = 28636.19M  free = 1059.81M
System-wide memory free percentage: 22%
audit worker RSS: 332 MB
```

**Swap is 96% full with 1.0 GB free.** That is precisely the condition under which this
project previously observed zero-sum probability rows, whose cause was never
established. CPU headroom exists (14 cores, load ~8, most of it a mostly-idle Docker
VM), but CPU headroom is not the binding resource here. The study therefore runs on
**one experimental worker throughout**, which is the conservative reading of the
instruction to increase only after confirming available memory.

Two short-lived transport workers were launched at `13:17Z` before this measurement and
both aborted within 20 seconds on a path-ordering bug (`ev.DEV` was overwritten before
`FrozenSeed.object_files()` was read). They wrote no scientific output. The bug is
fixed; the incident is recorded because it happened, not because it mattered.

Cost of the decision: the 2017 panel runs serially at roughly 161 s per interface rather
than in parallel across seeds. The schedule absorbs it.

## Phase log

| phase | status |
|---|---|
| 0 recovery, protocol, implementation, timing pilot | complete, commit `f5976c28` |
| 1 baseline transport completion (15 interfaces, 2017) | see below |
| 2 matrix fitting (126) | see below |
| 3 2018 audits (126) | see below |
| 4 2017 panel (69) and stress attacks | see below |
| 5 analysis, verification, publication | see below |

## Failures, quarantines and omitted units

Recorded here as they occur. Corrupted scores are never treated as data.
