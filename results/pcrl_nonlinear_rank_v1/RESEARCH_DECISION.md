# Research decision

## Disclosure: what was registered, and how it turned out

Six directional predictions were registered in `PROTOCOL.md` §2 and hashed into
`PROTOCOL_FREEZE.json` before the fit phase started. **Three of the six were wrong or
ambiguous.** That is reported first, because the registration is what makes the rest of this
document worth anything.

| # | Registered prediction | Outcome |
|---|---|---|
| P1 | `r_plus = 16` in all three seeds, so eigenvalue-sign rank selection changes nothing | **Confirmed** (structural, checkable without a fit) |
| P2 | The refinement lowers its own training objective below the original-moment solution at rank 16 | **Confirmed**, 18/18 conditions, gains 0.017–0.146 |
| P3 | The best new candidate keeps additional A/RAC1P recovery **above** J's in ≥2 of 3 seeds | **Ambiguous / not confirmed**: 1 of 3 seeds unweighted, 2 of 3 person-weighted. Seed-mean 0.0055 against J's 0.0071, and the paired interval on the difference contains zero |
| P4 | Training-objective improvement will **not transfer** to independent-attack improvement (surrogate failure) | **Refuted in its qualitative form.** The penalty produced significant attack reductions at matched rank: A/SEX −0.0064 (r16) and −0.0065 (r8), A/RAC1P −0.0073 (r16), AB/RAC1P −0.0116 (r16), AB/SEX −0.0055 (r8), all with adjusted upper bounds below zero |
| P5 | Rank-8 compression costs >0.002 nats of residence gain in ≥2 of 3 seeds | **Literally met in 4 of 6 policy×family cells, but the substantive claim fails.** The seed-mean cost is ≤0.002 everywhere and the sign flips across seeds (e.g. nonlinear C1: +0.0044, −0.0041, +0.0053). Compression did not cost measurable residence capability |
| P6 | No new candidate satisfies the primary coordination rule against both L1 and L2 | **Confirmed**, all 8 registered coordination cells NOT SUPPORTED |

I predicted the mechanism would fail for the wrong reason. It failed, but not by being inert.

## Conclusion

**The new method did not improve the intended tradeoff against both local controls and the
frozen neural channel J, so it is not a candidate for confirmation.** But the honest description
is not "it did not work": the tradeoff moved substantially in the right direction, the movement
is attributable to two separable causes, and a specific fixable defect in the construction was
identified and measured.

### Why it is a no-go

* **Against J.** The best new candidate, `spectral_nlr8_C1`, is **significantly worse than J on
  `A/SEX`** under both weightings (+0.0106 unweighted, +0.0098 person-weighted; adjusted
  intervals exclude zero). `A/SEX` is a forbidden role and is not set aside. On the other three
  family sensitive endpoints it is statistically indistinguishable from J, and its residence gain
  is higher (0.0259 against 0.0215) but not significantly so. It also costs more service-probe
  loss: it clears the legacy source allowance in 1–2 of 3 seeds where J clears it in 3 of 3.
* **Against the local controls.** All eight registered coordination cells fail. Advantage over
  the equal-total-mass control `L2` is not established in at least one weighting for every
  candidate, which is the same pattern the predecessor study saw on development and could only
  resolve on a five-times-larger locked partition. This is a reproduction of a known limitation,
  not a new one.
* No candidate is a strict Pareto improvement over J, and none was assumed.
* **The exploratory cross-year look makes the gap wider, not narrower.** On the already-spent
  2017 partitions the two attribution effects reproduce in direction at the seed mean, but J has
  lower additional recovery than the best new candidate on **all four** family sensitive endpoints
  (0.0013/0.0039/0.0069/0.0015 against 0.0105/0.0155/0.0127/0.0154), while the candidate keeps its
  residence advantage (0.0267 against 0.0189). The single place the 2018 result came closest to J
  — parity on both race endpoints — does not carry over. Per-seed direction agreement with 2018 is
  only moderate (188 of 288 seed-level comparisons). This is exploratory development evidence with
  no intervals and it cannot confirm anything; it is reported because it cuts against the
  candidate, and omitting it would be the selective reading. See `EXPLORATORY_2017.md`.

### What did improve, and what caused it

The answer to "penalty, dimension, both, or neither" is **both, and they are complementary** —
the opposite of what I registered.

| | effect at matched rank/policy, adjusted intervals excluding zero |
|---|---|
| **Nonlinear penalty** | reduces **sex** recovery most: `A/SEX` −0.0064 (r16), −0.0065 (r8); `AB/SEX` −0.0055 (r8); also `A/RAC1P` −0.0073 and `AB/RAC1P` −0.0116 at r16. No endpoint significantly worse; no residence cost |
| **Rank-8 compression** | reduces **race** recovery most: `A/RAC1P` −0.0217 (original penalty), −0.0146 (nonlinear); `AB/RAC1P` −0.0149 (original). No endpoint significantly worse; no residence cost |

Composed, they take additional `A/RAC1P` recovery from the predecessor's 0.0274 down to 0.0055
— indistinguishable from J's 0.0071 — while holding residence gain at 0.0259 against J's 0.0215.
Every new arm clears the `.01` residence reference in every seed and both weightings.

### The mechanism finding, which is the most transferable result

A penalty nonlinear in `Z` is not invariant under `W → W Q`, while the utility term, the original
linear penalty and **the information content of the release** all are. So part of the optimiser's
progress is a re-basing that provably changes nothing an attacker can recover. Measured share of
the training gain reachable by rotation alone: **mean 0.628, range 0.367–0.907** across 18
conditions.

Root cause, localised exactly and reproducible on a synthetic fixture: summing the degree-two
monomials `a ≤ b` with equal weight computes `Σ_a M_aa² + Σ_{a<b} M_ab²` instead of `‖M‖_F²`, and
only the latter survives `M → Q'MQ`. Weighting off-diagonals by `√2` restores exact invariance
(`< 1e-12` against ~10% slack). Per-feature standardisation adds ~1%; the Fourier block's residual
slack is Monte-Carlo error that shrinks with feature count.

This makes the fix **better** motivated than a pure negative would have: roughly 37% of the
surrogate movement bought statistically significant reductions in measured recovery, so removing
the provably inert 63% should let the same compute buy more.

### A second structural finding, relevant to anyone porting a conditional criterion here

The frozen nuisances barely beat a constant prior on the protected attributes: sex by 0.2–0.7%
of the prior loss, race by 1.0–2.4%. The protected residual is therefore close to the *marginal*
residual, so any `H`-conditioned penalty has limited room to differ from the marginal one **on
this interface, whatever function class it uses in `Z`**. That is a property of the interface.

### Controls that came out clean

* **Not a withholding artifact.** Across 360 cells (6 sources × 5 values of `p` × 12 candidates),
  randomised withholding of a simpler channel dominates a new candidate in **0** cells across all
  seeds and weightings; the maximum is 2 of 6 cells. The branch-routed expected-loss identity
  holds to 2.2e-16.
* **Not a numerical artifact.** 27,540 stored predictions were recomputed in a separate process
  and matched **bitwise, 0 mismatches**, with the machine at swap capacity throughout — the
  condition associated with the predecessor's corruption. Score replay agrees to 4.4e-16 over
  2112 checks. Memory pressure remains a *suspected* contributing condition there, not a
  demonstrated cause.
* **Not an optimisation artifact.** The refinement never returned its initial point in any of the
  18 conditions; restart spread tracks the gain; feasibility `max|W'W − I| ≤ 1.4e-15`.

## One next decision

**Do not run a confirmation on this design.** Fix the rotation defect first, verify it with the
outcome-free acceptance gate, and only then decide whether anything deserves the one unused year.
`NEXT_CONFIRMATION_SPEC.md` specifies the recipe (three closed-form changes to the feature
family), its matched controls, and a gate that requires the measured rotation share below 0.10
**before** any outcome may be read. If the gate fails, the construction is still misspecified and
that is the reportable result.

**2016 is not authorised by this decision and was not touched.** Its admission has never been run
and is Terminal B's scope; an admission report is admission preparation, never performance
evidence. A 2016 confirmation would need its own prospective protocol and its own registered
predictions, and must not be started on the strength of a development result — including this one.

## Uncertainty that this study cannot remove

* These are **development** numbers on pools used repeatedly. The paired household bootstrap
  quantifies sampling variability for fixed fitted systems; it cannot undo repeated use, and it
  is not Census replicate-weight variance or retraining variability.
* Candidate selection was made: 12 new conditions were inspected and `nlr8_C1` is reported as
  best. Its `A/SEX` deficit against J is the reason it is not nominated, and it is the endpoint a
  selective reading would have omitted.
* Seed-to-seed variability on race recovery is large relative to the effects (`nlr8_C1` additional
  `A/RAC1P`: 0.000, −0.002, +0.018 unweighted). Seed 2 drives the mean. Three seeds cannot resolve
  this.
* The `.001` residence condition is a point-difference rule reused for comparability. A pass under
  it is **not** equivalence, and the simultaneous intervals do not establish equivalence within
  that band. Noninferiority and equivalence interval outcomes are reported separately per contrast
  in `DEVELOPMENT_2018.json`.
* Nuisances were held fixed by design. Residual moments can reflect nuisance error as well as
  disclosure, and the nuisance-error fixture shows a misspecified `m(H)` manufacturing a penalty
  more than ten times the oracle value under *exact* conditional independence.
* Nothing here is a certificate. Vanishing fitted finite moments imply neither `Z ⊥ S | H` nor any
  bound on `I(S;Z|H)`, and no calibration is inherited from KCI or RCoT.
* Two arms failing, or improving, is evidence about two arms. It is **not** evidence about the
  closed-form spectral family as a whole — and the refined objective is not in that family at all,
  since it is not a trace form.
