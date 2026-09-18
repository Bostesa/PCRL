# CORRECTIONS — additive, to the starting interpretation

**Nothing in the historical evidence is edited.** Every correction below is recorded
here and carried forward into this study's own claims. The predecessors' reports keep
their published bytes, including where this note says they overreach.

Sources corrected: `results/pcrl_invariant_baselines_v1/{RESEARCH_DECISION,METHOD,
BASELINE_ADAPTATIONS,DEVELOPMENT_2018,EXPLORATORY_2017}.md` at commit
`73903b7f28df68284285f0610a4036beb32b208f`, and the reports it inherits from
`c37807e4f568ef38e5528fc09c1506083278bf4d` and
`349efa454afd907389760fd1f59fd8806a215efd`.

---

## 1. LEACE does not dominate on utility, and "not significant" is not "no cost"

`RESEARCH_DECISION.md` §3 at `73903b7f` reads:

> **LEACE dominates the repaired mechanism**: significantly less recovery on **all
> four** sensitive endpoints with **no statistically detectable residence cost**.

The recovery half is supported. The utility half is **not a dominance result**. The
measured residence-loss difference is `+0.0036` with adjusted interval
`[-0.0028, +0.0100]` unweighted. That interval:

* contains zero — hence "no statistically detectable cost";
* but also contains `+0.0100`, **ten times** the `.001` utility reference the same
  study uses elsewhere.

**Absence of a detectable cost is not absence of a cost, and it is certainly not
noninferiority.** A one-sided noninferiority claim at margin `.001` would require the
adjusted upper bound to be at most `.001`; the observed upper bound is `+0.0100`, an
order of magnitude outside it. The word "dominates" is therefore withdrawn in this
study's usage: `leace_A0` **leaks less on all four sensitive endpoints, at a residence
cost this evidence cannot bound below the utility reference.**

## 2. Nonsignificance versus `J` is not equivalence to `J`

`RESEARCH_DECISION.md` §3 reads:

> Against `J`, the frozen neural channel: `leace_A0`, `splince_A0` and `optnet16_C1`
> are **statistically indistinguishable on all five family endpoints**.

That statement is accurate as written and is **not** an equivalence result. With three
seeds, one cohort and a simultaneous max-|t| adjustment, the intervals are wide enough
that indistinguishability is largely a statement about power. Equivalence requires its
own two-sided margin procedure, which was not run. This study reports equivalence and
noninferiority **separately and by name** (`PROTOCOL.md` §5) and never infers either
from a failure to reject.

## 3. Coordinate dependence was not isolated as the cause

`RESEARCH_DECISION.md` §2 offers:

> the coordinate dependence was acting as an **implicit regulariser** on an
> under-determined surrogate

as "the mechanism reading, stated carefully". It is carefully stated, and it is still
**not identified**. The repair changed four things at once — Frobenius weighting,
per-feature standardisation, exact kernel in place of Fourier features, and the frozen
kernel subset (`METHOD.md` §0 at `73903b7f` says so itself: "any measured ACS effect is
attributable to the **package**"). No single-factor ablation was run. Implicit
regularisation is therefore a **possible explanation**, not a cause, and this study does
not build on it.

## 4. The original coordination success used a point-estimate rule

The `.001` coordination rule is a **one-sided point-estimate** comparison of residence
losses. It is not an equivalence band and not an interval-based noninferiority result.
The predecessors label it correctly in their rule definitions, but the prose around the
original coordination success reads as though a utility guarantee had been established.
It had not. This study keeps the `.001` point rule **for continuity only**, under its
own name, and states a separate interval-based noninferiority rule for any new claim
(`PROTOCOL.md` §5).

## 5. The external baselines were never evaluated on 2017

`EXPLORATORY_2017.md` at `73903b7f` scopes out `leace_A0`, `splince_A0` and
`optnet16_*` with these reasons:

> `leace_A0` — acts on the frozen A0 neural auxiliary channel; its 2017 construction
> belongs to a different pipeline. […] `optnet16_*` — encoders were not persisted by
> this run's fit stage.

Both are **engineering boundaries, not scientific ones**, and both are resolved here:

* the frozen `A0` inference pipeline *already exists* on 2017 inside the transport
  study's own `FrozenSeed.interface`, so "a different pipeline" was an availability
  problem in that worktree, not a property of the method;
* the OptNet stage is deterministic and its budget is recorded, so the encoders are
  reconstructible.

Until this study, therefore, the sentence "**the external-baseline comparison rests
entirely on the 2018 development evaluation**" was correct — and it meant the strongest
finding of the previous study had never been looked at on a second year.
`BASELINE_TRANSPORT_COMPLETION.md` reports the completion, with every reconstruction
carrying an explicit bitwise identity proof against the recorded 2018 artifacts.

## 6. Corrections this study makes to itself

Recorded in `RUN_STATUS.md` rather than here, because they concern this study's own
conduct and all three were declared before any outcome was opened:

* **Amendment 1** — checkpoint selection moved from a contemporaneous attacker slate,
  to a single final slate, to an equal-budget fresh probe per checkpoint. The first two
  are biased in the same direction and the second would have handed every arm to the
  unmoved starting point.
* **Amendment 2** — the checkpoint grid was corrected to the frozen protocol value.
* **Amendment 3** — the candidate-wide correction is a studentized Bonferroni bound,
  because the registered percentile at `alpha/m` is not estimable from 2000 replicates.
