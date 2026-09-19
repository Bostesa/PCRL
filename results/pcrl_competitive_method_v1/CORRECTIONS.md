# CORRECTIONS — additive, to the direct-adversarial study's interpretation

**Nothing in the historical evidence is edited.** Every predecessor document keeps its
published bytes, including where this note says it overreaches. This file records what
*this* study is correcting and carries each correction forward into its own claims.

Sources corrected, resolved to full SHAs:

| Study | Branch | Commit |
|---|---|---|
| Direct adversarial channel refinement | `research/pcrl-direct-adversarial-v1` | `106de9afa58cebbc26e34fb782e539e2a0881108` (evidence `69e790af36c5ca53203dab17b757a8e3415ee934`) |
| Invariant repair + external baselines | `research/pcrl-invariant-baselines-v1` | `73903b7f28df68284285f0610a4036beb32b208f` |
| Nonlinear / rank | `research/pcrl-nonlinear-rank-v1` | `c37807e4f568ef38e5528fc09c1506083278bf4d` |
| ACS 2017 transport (locked) | `residual-spectral-20260910` | `349efa454afd907389760fd1f59fd8806a215efd` |
| Manuscript (Terminal 2) | `research/pcrl-manuscript-integrated-v3` | `ce5ba24a0a0ed1848c4029a922ac66fa629d71fc` |

The predecessor's own `CORRECTIONS.md` at `69e790af` remains in force and is **not**
restated here. This file is additive to it.

---

## 1. The previous study never tested J-initialised fine-tuning

`METHOD.md` §1 and §1.1 at `69e790af` are explicit that width 16 "**is** the frozen
`A0` mapper, parameter for parameter", and width 8 is a deterministic PCA compression
of *the same `A0` output*. `run_fit.seed_context` builds both widths from `a0` alone.

**Every one of the 114 fitted trajectories therefore started from `A0`.** `J` — the
strongest developed protection channel in this line of work, and the channel the study
was trying to beat — appears only as a frozen comparator. The sentence in
`METHOD.md` §0, "whether a **directly adversarial refinement** of that same channel …
improves the measured frontier beyond `J`", describes an experiment that was not run:
the refinement was applied to `A0`, not to `J`.

This is not a defect in the previous execution, which documented its initialisation
accurately. It is a gap in the evidence, and it is the reason Track N of this study
carries initialisation as an explicit factor.

## 2. 39/72 unchanged channels is a selection outcome, not evidence of a stationary optimiser

`RUN_STATUS.md` amendment 4 and the fit records show that 39 of the 72 main-block
trajectories published their step-0 checkpoint, i.e. released a channel bitwise
identical to `A0`.

`SELECTION_DIAGNOSIS.md` §2 establishes, from the saved parameter states, that
**all 39 of those trajectories had moved**: median relative parameter displacement
`0.0991` at the final checkpoint, median final teacher distortion `0.0119`, and **no
trajectory anywhere in the run was stationary**. Across all 114 fitted trajectories the
same holds: 63 selected step 0, and all 63 had moved.

The correct statement is: *optimisation moved the channel and the registered selection
objective preferred the starting point.* The following inferences are **not** licensed
and are not made here:

* that optimisation never moved;
* that teacher distortion alone caused the failure (§3 below);
* that the discarded checkpoints would have audited better — **none of them has been
  audited**, and the monitor score is a deliberately weak selection yardstick.

## 3. Only the distortion term is automatically minimised at the teacher

The selection objective is `C_t + gamma*D_t + beta*P_t`. At `theta = teacher`, `D = 0`,
which is `D`'s global minimum. **`C` and `P` are not minimised there.** The Z-only
source heads were warmed at the teacher but not trained to optimality, and the
protection penalty at the teacher is whatever the fresh attacker slate recovers from
the unmodified channel — on `dax16_C1_b100` seed 0 that is `0.094`, the *largest*
penalty anywhere on that trajectory.

So the claim "the objective is automatically minimised at its own teacher" is false for
the objective as a whole, and the step-0 preference is a *net* outcome of three terms.
`SELECTION_DIAGNOSIS.md` §3 quantifies it: across the 51 trajectories that did select a
moved checkpoint, movement cost `+0.014` in source loss and `+0.028` in distortion and
bought `−0.084` in `beta * penalty`. The distortion cost is about **2x** the source
cost — a large share, but **not the whole story**, and a one-factor reading is wrong.

Relatedly: **it is not correct that all four prior studies used an identical utility
objective.** The `A0`/`J` training objective of
`redesign_20260909_acs_fixed_predictions_v1` is a source term plus signed
individual/coalition observer terms at coefficients `-0.1`; the spectral line used a
moment penalty with its own `lambda` grid; the invariant-baselines erasure arms have no
training utility objective at all (they are closed-form post-processing); and the
direct-adversarial study used `source + 1.0 * teacher_distortion`. These are different
objectives on different scales and their coefficients are not comparable.

## 4. The distortion coefficient 1.0 was specified, never shown optimal

`METHOD.md` §5 at `69e790af` states "Coefficient **1.0**, fixed." It was fixed by
specification, and `RESEARCH_DECISION.md` §9 defers a sweep of it as the study's own
outstanding work. No evidence at `69e790af` bears on whether `1.0` is a good value.
Track N of this study varies it prospectively as `gamma in {0, .01, .1, 1}`.

`SELECTION_DIAGNOSIS.md` §4 additionally shows that **rescoring the stored trajectories
at `gamma = 0` changes 50 of 114 selections**. That is a fact about selection on fixed
trajectories and is **explicitly not a utility ablation**: a different `gamma` would
have produced a different trajectory, so the counterfactual cannot predict what
retraining does. Track N retrains.

## 5. The identity check and its correction are preserved

`RUN_STATUS.md` amendment 4 found that `dax16_C1_b010`, whose released channel is
bitwise identical to `A0`, scored `0.0155` *better* on `A/RAC1P` than `A0` itself under
the inherited `kernel_expanded_catchup` scope — purely because the historical arm's
selected attacker there was a catch-up candidate the new arm had no equivalent of.
Under the matched-exposure scope `kernel_expanded_independent` the two rows are exactly
equal, as they must be.

**Both the identity check and the correction are preserved in this study.** The primary
2018 scope here is `kernel_expanded_independent`; the catch-up scope is reported
alongside and explicitly labelled as giving historical arms exposure that new arms
never had. An unchanged `A0` or `J` candidate must reproduce its matched primary audit
exactly, and that reproduction is a registered validation check, not a footnote.

## 6. C1 versus L2 on residence: the interval crosses .001

`RESEARCH_DECISION.md` §Q2 at `69e790af` reports:

| contrast | residence difference (adjusted) |
|---|---|
| `dax16_C1_b100` vs `dax16_L1_b100` | `+0.00717` `[+0.0031, +0.0113]` |
| `dax16_C1_b100` vs `dax16_L2_b100` | `+0.00346` `[+0.0001, +0.0068]` |

and then says "**both residence intervals lie entirely above zero and above the `.001`
reference.**"

That is correct for the `L1` contrast and **incorrect for the `L2` contrast**. The `L2`
interval `[+0.0001, +0.0068]` excludes zero but **crosses `.001`**: its lower bound is
an order of magnitude below the reference. The accurate statement has three parts and
this study uses all three:

* the **point estimate** is `+0.00346`, above the `.001` reference;
* the cost is **significant** — the interval excludes zero;
* **noninferiority at `.001` is not established** in either direction, because a
  one-sided claim needs the relevant adjusted bound inside the margin and this interval
  straddles it.

## 7. J's profile is a tradeoff, not Pareto dominance

`RESEARCH_DECISION.md` §Q3 reports, matched-exposure scope, budget 360, unweighted seed
means:

| condition | `A/SEX` | `AB/SEX` | `A/RAC1P` | `AB/RAC1P` | residence gain |
|---|---|---|---|---|---|
| `J` | +0.0015 | +0.0080 | +0.0000 | −0.0006 | +0.0215 |
| `leace_A0` | +0.0039 | +0.0038 | +0.0097 | +0.0096 | +0.0220 |

`J` leaks less on three of four sensitive endpoints and **more** on `AB/SEX`, at a
slightly lower residence gain. That is a **tradeoff**. It is not Pareto dominance, and
neither of the following follows from it or from any nonsignificant contrast in that
table:

* that `LEACE` and `J` are **equivalent** — nonsignificance with three seeds, one
  cohort and a simultaneous adjustment is largely a statement about power, and
  equivalence needs its own two-sided margin procedure, which was not run;
* that either has **zero utility cost** — absence of a detectable cost is not absence
  of a cost.

This study reports equivalence and noninferiority separately and by name, and never
infers either from a failure to reject.

## 8. Realised eraser rank depends on the cross-moment matrix, not on class counts

The previous fitting log states:

> `LEACE` on the width-8 channel has realised projection rank 0 in all 3 seeds: the
> joint `{SEX, RAC1P, public_coverage}` schema needs 11 rank and the channel has 8, so
> erasure annihilates it

The **observation** — realised rank 0 in all three seeds — stands and is not disputed.
The **explanation** does not. LEACE's erasing projection is determined by the rank and
row space of the feature/target **cross-moment matrix** `Cov(Z, onehot(S))`, not by the
nominal number of target classes. A one-hot schema with 11 columns has at most rank 10
after centering, and its cross-moment with an 8-dimensional feature has rank at most 8;
whether that forces the projection to annihilate everything depends on the **realised**
cross-moment spectrum on the actual rows with the actual valid-label masks, including
classes with near-zero support (one `RAC1P` class has a support of exactly 1 row in seed 0).

A categorical target having more coordinates than the feature space **does not by
itself** prove the eraser must annihilate all features. This study therefore reports
realised rank as a measured quantity for every projection, against a declared relative
tolerance, and never derives it from a class count.

## 9. Poor fresh-attack results have more than one possible cause

`ATTACK_STRENGTH.md` and the stress suite at `69e790af` observe that freshly fitted
attackers recover little from some arms. At least four mechanisms produce that
observation:

1. **finite optimisation** — the attacker's budget or family did not find the signal;
2. **representation loss** — the signal is genuinely not linearly-or-otherwise present;
3. **utility/transfer mismatch** — the channel retained capability for the *source*
   tasks and not for the transfer task actually being probed;
4. **selection** — the released checkpoint was chosen by a rule correlated with the
   measured quantity.

`SELECTION_DIAGNOSIS.md` §5 removes one adjacent hypothesis — the training-time zero-gain
clamp was essentially inactive wherever an attacker existed (measured rate `0.23%` on
the fresh probe, `0.43%` at a 10x budget), so a saturating `max(0, ·)` is not what
shaped the previous run. It does **not** identify which of the four mechanisms above is
operating. **No cause is claimed from a single diagnostic in this study.**

---

## 10. Corrections this study makes to its own conduct

Recorded in `RUN_STATUS.md` as amendments, with timing, what outcomes had been seen,
affected units, and whether refitting is required. Any change motivated by a
development outcome is carried as an explicitly labelled exploratory branch and cannot
rewrite the registered primary study.

---

## Addendum (2026-09-19, after Terminal 2 review of commit `4f09e63c`) — correcting §6 of this file

§6 above says the `dax16_C1_b100` vs `dax16_L2_b100` residence cost is "**significant** —
the interval excludes zero". That holds **only at the predecessor's family-adjusted
(within-contrast max-|t|) level**, `[+0.0001, +0.0068]`. At the predecessor's
**candidate-wide** level the same contrast reads `[-0.0018, +0.0087]` and **contains
zero**: the cost is then not significant. Two further points this file omitted:

* the predecessor reported its one positive family (coalition vs local) at the
  family-adjusted level and its negative frontier family at the candidate-wide level —
  **different correction levels for the positive and negative results**;
* the predecessor's Q2 table omitted the fourth, strength-matched contrast
  `dax8_C1_b100` vs `dax8_L2_b100`, all of whose rows are null, with several point
  estimates favouring `L2`.

The accurate statement: coalition conditioning beat the weak local control `L1`; against
the strength-matched `L2` it showed one sensitive endpoint at width 16 at the
family-adjusted level only, nothing candidate-wide, and nothing at width 8. **This
study's decisions use ONE declared level** — the candidate-wide studentized Bonferroni
bound within each registered family (P for decisions, X exploratory) — for positive and
negative results alike; both that level and the within-contrast level are reported for
every contrast in `INTERVALS_P.csv` / `INTERVALS_X.csv`.
