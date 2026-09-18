# Research decision

## The verdict in one line

**A directly adversarial refinement of the strongest developed protection channel does
not improve the frontier.** The best arm this study produced is significantly *worse*
than frozen `J` on `AB/RAC1P` and significantly worse than `leace_A0` on `AB/SEX`, and
is significantly better than **no** comparator on **no** sensitive endpoint. The
declared nominee conjunction is **UNSUPPORTED**. Two things inside the study do work,
and neither is the headline: coalition conditioning genuinely beats strong local
controls, and it pays for that in measured utility.

All numbers are **2018 DEVELOPMENT** on pools this project has used repeatedly, under
the matched-exposure primary scope `kernel_expanded_independent`, budget 360,
unweighted unless stated.

## 1. Disclosure: what was registered, and how it turned out

Five directional expectations were registered in `PROTOCOL.md` §3 and hashed into
`PROTOCOL_FREEZE.json` **before** the fit stage started. Their evaluation is
**generated**, not read off by eye (`run_report.registered_expectations`), so it cannot
drift toward whatever happened.

| # | Registered expectation | Outcome |
|---|---|---|
| **D1** | `C1` reduces `AB/*` recovery relative to `L1` at matched width and beta in ≥2 of 3 seeds | **CONFIRMED**, 7 of 8 *informative* cells. 8 of the 16 cells are **degenerate** — at `beta` 0.1 and 0.3 the checkpoint rule returned the unmoved channel, so `C1` and `L1` are the same object there and every difference is exactly zero. Counting those as refutations would have been an artifact, and they are reported rather than dropped |
| **D2** | at least one `C1` arm shows a **local race cost** relative to `L2` | **CONFIRMED**, 7 instances. This is the failure mode §8 forbids concealing, and it occurred |
| **D3** | the ensemble arm's recovery is **not lower** than the single-attacker arm's in most cells | **CONFIRMED**, 15 of 24 informative cells. The one significant ensemble-vs-single row in the whole study has the **ensemble worse**: `dax16_L2_b100` leaks `+0.0110` more on `A/RAC1P` than its single-`mlp64` counterpart |
| **D4** | the nominee conjunction is **not** satisfied | **CONFIRMED.** All 16 registered coordination cells are NOT SUPPORTED |
| **D5** | width 8 shows lower recovery **and** lower residence gain than width 16 | **CONFIRMED**, 10 of 12 cells |

I was right about all five, which is not a compliment to the method: four of the five
predicted that it would not work.

## 2. The three prespecified questions

### Q1 — does refreshed multi-attacker training beat one attacker? **No.**

Across the four ablation cells, the ensemble's measured recovery is not lower than the
single-`mlp64` arm's in 15 of 24 informative cells, and the only contrast whose
adjusted interval excludes zero has the **ensemble worse**. The mechanism diagnostics
say why: over 1350 refresh decisions the refreshed attacker was kept only **84 times
(6%)**, with mean monitor recovery **−0.0130**. The incumbents were not being fooled by
a moving channel — they were genuinely strong, so refreshing had almost nothing to
recover. Extending the audit budget tells the same story: mean catch-up gain from
budget 120 to 360 is **2.1e-5**.

Three differentiable families at matched per-attacker optimiser exposure buy nothing
here. **Compute equivalence is not expressive equivalence**, and the single arm has one
third of the ensemble's parameter updates; that asymmetry is disclosed and is in the
direction that would favour the ensemble, not the single arm.

### Q2 — does coalition conditioning help beyond stronger local training? **Yes, and it is not free.**

This is the one clear positive. At `beta = 1.0`, with architecture, initialisation,
utility objective, attacker slot count and optimiser exposure all controlled, and with
the local controls deliberately strengthened by spending their two coalition-equivalent
slots on independently initialised local replicas:

| contrast | endpoints significantly better | endpoints worse | residence difference (adjusted) |
|---|---|---|---|
| `dax16_C1_b100` vs `dax16_L1_b100` | **all four** sensitive | none | **+0.00717** [+0.0031, +0.0113] |
| `dax16_C1_b100` vs `dax16_L2_b100` | `AB/RAC1P` | none | **+0.00346** [+0.0001, +0.0068] |
| `dax8_C1_b100` vs `dax8_L1_b100` | **all four** sensitive, both weightings | none | — |

`advantage_all` is **true** for `dax16_C1_b100` against both local controls. The
coordination rule still fails, and it fails on the right thing: **both residence
intervals lie entirely above zero and above the `.001` reference.** Coalition
conditioning is buying protection with utility, not getting it for free.

`L2` matches `C1`'s nominal total group weight, not its gradients and not its
difficulty; that limitation stands wherever this contrast appears.

### Q3 — does any candidate improve the useful frontier? **No.**

Seed means, matched-exposure scope, budget 360, unweighted. Lower recovery is less
disclosure; higher residence gain is more capability.

| condition | `A/SEX` | `AB/SEX` | `A/RAC1P` | `AB/RAC1P` | residence gain |
|---|---|---|---|---|---|
| `A0` (no protection) | +0.0317 | +0.0294 | +0.0525 | +0.0469 | +0.0318 |
| **`J`** | **+0.0015** | +0.0080 | **+0.0000** | **−0.0006** | +0.0215 |
| `leace_A0` | +0.0039 | **+0.0038** | +0.0097 | +0.0096 | **+0.0220** |
| `splince_A0` | −0.0002 | +0.0024 | +0.0081 | +0.0073 | +0.0167 |
| `optnet16_C1` | +0.0070 | +0.0046 | +0.0060 | +0.0116 | +0.0170 |
| `spectral_C1` (previous line of work) | +0.0191 | +0.0177 | +0.0294 | +0.0285 | +0.0265 |
| `dax8_C1_b300` (**best new arm**) | +0.0072 | +0.0096 | +0.0047 | +0.0079 | +0.0220 |
| `dax16_C1_b100` | +0.0114 | +0.0162 | +0.0201 | +0.0182 | +0.0226 |
| `dax16_L2_b300` | +0.0104 | +0.0173 | +0.0118 | +0.0165 | +0.0173 |
| `leace_dax8_none` | +0.0000 | +0.0032 | +0.0001 | −0.0011 | **−0.0005** |

The best new arm, `dax8_C1_b300`, is **significantly worse than `J` on `AB/RAC1P`**
under both weightings and **significantly worse than `leace_A0` on `AB/SEX`**, with
**no endpoint significantly better than any comparator**. Its residence gain is
statistically indistinguishable from both. Across the whole `frontier_vs_comparator`
family there are **189 candidate-wide-significant WORSE cells and 18 BETTER cells, and
every one of the 18 is on residence utility, not on a sensitive endpoint.**

**The nominee conjunction is UNSUPPORTED.** No arm improves a declared sensitive
endpoint against both local controls *and* a strong external comparator. It is not
loosened after the near miss, and there is no nominee.

## 3. The most consequential row is not ours

`leace_dax8_none` deserves its own line. LEACE on the width-8 channel has **realised
projection rank 0** in all three seeds: the joint `{SEX, RAC1P, public_coverage}`
schema needs 11 dimensions of rank and the channel has 8, so erasure annihilates the
channel entirely. The result is a **constant** auxiliary output — recovery at zero on
every sensitive endpoint, and residence gain of **−0.0005**, i.e. no capability at all.
That is the degenerate corner of the frontier, reached by a closed-form 2023 method
applied to a channel that was too narrow for it, and it is worth stating plainly
because it is the only point in this study with recovery indistinguishable from zero.

The previous study's central finding survives this one: **a closed-form linear eraser
applied to a channel this project already had still matches `J` and still beats four
studies of developed mechanism, now including this one.**

## 4. Mechanism: why it failed, stated carefully

* **The training family was not the binding problem.** The training probe's gain and
  the independent auditor's additional recovery correlate at Pearson **0.58** / Spearman
  **0.44** across 900 role-cells, with the auditor finding slightly *more* on average
  (0.0233 against 0.0196). The mapper was not merely fooling its own differentiable
  family and then being caught by trees and kernels it had never seen.
* **The refresh machinery found almost nothing to fix** (6% kept, mean recovery
  −0.0130), and the extended catch-up found essentially nothing (2.1e-5). The inner
  player was already close to its ceiling.
* **The utility objective is the binding constraint.** `A0` is very nearly the minimiser
  of `U` by construction — it was trained for source cross-entropy, and teacher
  distortion is zero there — so **every unit of penalty reduction must be paid for**.
  At `beta` 0.1 and 0.3 the preregistered checkpoint rule returns the **unmoved
  channel** in 3 of 3 seeds for every policy and width. **39 of 72 main arms are bitwise
  identical to their no-protection continuation.**
* **Teacher fidelity does predict residence transfer, weakly and in the right
  direction**: distortion against residence gain, Pearson **−0.33**, Spearman −0.30.
  More distortion, less residence. It is a usable proxy and a weak one; this study does
  not treat it as a guarantee, and §9 of the protocol is why it was measured.
* **Compression is doing much of the work.** Width 8 beats width 16 on race recovery in
  10 candidate-wide-significant cells, and D5 confirms it also gives up residence. That
  is a compression effect, available without any adversarial objective at all.
* **The mechanism is not stable under a different optimiser draw at the same anchor.**
  `dax16_L2_b100_r1` is **significantly better than its own anchor** on `A/RAC1P`
  (−0.0111) and `AB/RAC1P` (−0.0075). An optimiser seed moves the measured endpoint by
  as much as the effects being claimed.

**None of this is causal.** These are observational associations among diagnostics of a
bounded search, on three anchor seeds.

## 5. Corrections this study made to itself

Four amendments, all declared before the outcome they could have affected, in
`RUN_STATUS.md`. Two were material and both ran **against** this study's interest:

* **Amendment 1.** Neither the contemporaneous attacker slate nor the single final
  slate is a neutral yardstick for checkpoint selection; both bias toward never moving
  the channel. Under the final-slate rule **every one of the first 11 arms fitted
  selected step 0**, which would have made the study vacuous by construction. Replaced
  with an equal-budget fresh probe per checkpoint.
* **Amendment 4.** The inherited primary scope `kernel_expanded_catchup` gives the
  historical arms saved-observer catch-up attacks that **no arm in this study
  received**. Measured on the cleanest possible case: `dax16_C1_b010`, whose channel is
  **bitwise identical to `A0`**, scored `A/RAC1P` `+0.0509` against `A0`'s `+0.0664`
  under that scope — a 0.0155 advantage for an identical object — and `+0.0548` against
  `+0.0548`, exactly equal, under the matched-exposure scope. The primary scope was
  changed. **The bias this removed favoured this study's own arms.**

The exact-equality row is also the strongest end-to-end correctness evidence here: an
independently re-audited channel that is bitwise identical to a historical one
reproduces its endpoints exactly.

## 6. Exact fit accounting

126 transform fits were planned. **123 produced a release**; the 3 missing are
`splince_dax8_none` in each seed, **SCOPED INFEASIBLE** and recorded as such rather
than omitted. Of the 123, **57 are bitwise duplicates of another unit in the same
seed** and **6 are bitwise duplicates of a historical arm** (`leace_dax16_none` and
`splince_dax16_none` are exactly `leace_A0` and `splince_A0`, because their input
channel is the no-protection continuation, which is exactly `A0`). **The number of
distinct, new released channels this study contributes is 60**, not 126.

Those duplicates are **outcomes of the preregistered checkpoint rule, not failed or
wasted fits**: every unit ran its full 600-update budget and its selection returned the
unmoved initial checkpoint. They are counted as completed units that are not unique
systems, and they are reused with proof rather than presented as new.

## 7. A descriptive observation that is NOT in the registered family

On seed 0, `dax16_L1_b100` reaches `A/RAC1P` `−0.0014` and `AB/RAC1P` `+0.0000` with a
residence gain of `+0.0217`, which reads better than `leace_A0` on race at more
residence. **This is disclosed as post-hoc.** The registered contrast list compares
**`C1` arms only** against `J` and the external comparators; `L1`-versus-`J` is not in
it, adding it now would enlarge the multiplicity family, and the arm's seed means
(`+0.0225 / +0.0226 / +0.0338 / +0.0349`) show the seed-0 row is not representative —
`dax16_L1_b100` selects the unmoved channel in seeds 1 and 2. It is recorded because
suppressing it would be worse, not because it supports anything.

## 8. Boundaries and uncertainty this study cannot remove

* These are **development** numbers on pools used repeatedly. The paired household
  bootstrap (2000 replicates, cohort `SERIALNO` clusters) quantifies sampling
  variability for **fixed fitted systems**. It cannot undo repeated use and is neither
  Census replicate-weight variance nor retraining variability.
* **Three anchor seeds do not establish broad training-population robustness.** The 24
  repeat fits are optimisation repetitions *conditional on those anchors* — not six or
  nine independent population seeds.
* Finite alternating optimisation is **not** a solved minimax problem. Low measured
  recovery means the **declared finite family** did not find a gain in its budget. It is
  **not** independence, **not** differential privacy, **not** a bound on `I(S;Z|H)`, and
  **not** protection against arbitrary attackers.
* `RAC1P` has classes with **no support in the actual fit folds** (7 of 27 fold-attribute
  cells). That is a property of the fold, and population absence is not inferred from it.
* No withholding mixture is used anywhere, so no privacy property here depends on hidden
  randomisation.
* **2016 remains SEALED and UNUSED.** Nothing in this study touched it, and nothing in
  this result opens it.

## 9. One next scientific decision

**Stop refining this interface's objective, and test whether the utility term — not the
penalty — is what is limiting every arm in this family.**

Four studies have now varied the penalty (moment, nonlinear, rotation-invariant,
directly adversarial) against a fixed utility objective anchored at a channel that is
already its own minimiser. The measurement that finally isolates it is cheap and does
not need a new year: **hold the penalty fixed at the one setting that demonstrably
works (`C1`, `beta = 1.0`) and vary the teacher-distortion coefficient**, which is
pinned at `1.0` here by specification and never moved. If the frontier is
coefficient-limited, that sweep will show it in one afternoon on the existing 2018
pools; if it is not, this family is closed on evidence rather than on fatigue.

**2016 is not authorised by this decision and was not touched. It remains UNSCORED.** A
confirmation would need its own prospective protocol, its own registered predictions and
an outcome-free admission pass, and **a development result is not a reason to spend it —
including this one, and including the one positive finding in §2.**
