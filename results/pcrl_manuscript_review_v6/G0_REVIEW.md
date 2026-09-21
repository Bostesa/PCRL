# G0_REVIEW — independent review of the stochastic study's precision gate

Reviewed: `results/pcrl_stochastic_channel_v1/gates/G0.json` and `REPORT.md` §5, on branch
`research/pcrl-stochastic-channel-v1` at `4de62f318` (local; not on `origin` at review time).
Source data re-read: `results/pcrl_nonlinear_rank_v1/PAIRED_INTERVALS.csv`.

**This is a correction to an interpretation, not a scientific result.** Nothing below is evidence for or
against any mechanism, and nothing in it improves any measured outcome. It narrows a claim that was stated
more broadly than the evidence supports. It must not be reported as a finding, a contribution, or a reason
the programme is in better shape than it was.

## 1. The arithmetic is right

Recomputed from the source file, independently of the gate:

| quantity | G0 | recomputed |
|---|---|---|
| sensitive rows | 544 | 544 |
| fraction with adjusted one-sided half-width > .001 | 1.00 | 1.00 |
| smallest adjusted half-width, any endpoint | 0.0021988 | 0.0021988 |
| smallest unadjusted one-sided 95% half-width | 0.0014115 | 0.0014115 |
| median critical value | 2.5403 | 2.5403 |

The sub-claim that **multiplicity is not the cause** also holds: the margin fails unadjusted, by a factor of
about 1.4 at the best row. No arithmetic objection.

## 2. The over-general claim

G0 concludes, and `REPORT.md` §5 repeats in bold:

> a `.001`-nat confirmation claim is not demonstrable at this sample size … **for any mechanism, not just
> this one.** More fitted channels do not supply more independent households. This is a property of the
> evidence base.

The second sentence is true. **The universal quantifier in the first does not follow from it.** The
half-width is `z × SE`, and the paired-bootstrap `SE` is not a function of the household count alone: it is
the dispersion of the *per-household paired loss differences* for the specific pair being compared, divided by
roughly the square root of the effective cluster count. Dispersion is a property of **how similar the release
is to its comparator**, which is a property of the mechanism. The household count fixes one factor, not the
product.

**Counterexample from this project's own committed evidence.** In Study 5's decision family
(`results/pcrl_competitive_method_v1/INTERVALS_P.csv`, commit `7f961d5`), **60 of 320 sensitive rows have a
bootstrap SE of exactly 0** and are flagged `degenerate`, because the J-initialised neural nominees are
bitwise identical to `J`: the paired difference is zero for every household. Example row:
`N_J_g000_C1_b100` vs `J`, `recovery/A/SEX`, estimate 0.0, adjusted upper bound 0.0. For those pairs a .001
one-sided bound — indeed any margin — *is* demonstrable at exactly the household count G0 says forbids it.

That case is scientifically vacuous: a release identical to J demonstrates the margin by being the thing it
is compared with. But it is a valid counterexample to a universally quantified claim, and it shows what the
binding constraint really is.

**How close the gap is.** The required SE for a .001 bound is `.001/z`:

| criterion | required SE | smallest SE observed in the source family | ratio |
|---|---|---|---|
| unadjusted one-sided 95% (z = 1.645) | 0.000608 | 0.000858 | **1.41×** |
| adjusted at the median critical value (z = 2.540) | 0.000394 | 0.000858 | **2.18×** |

Observed SEs in that family span 0.00086–0.00677, about an 8× range. So the conclusion rests on a factor of
1.4–2.2 in a mechanism-dependent quantity that already varies 8× across the rows being averaged over. It is
an empirical statement about the dispersion of the comparisons measured so far, not a limit implied by the
sample size.

**Selection of the reference family matters too.** The source family contains **zero** degenerate rows,
because the nonlinear-rank study compared channels that genuinely differ. G0 therefore estimates the
precision available *conditional on comparing substantially different channels* and then generalises to all
mechanisms.

## 3. The defensible version

> At this cluster count, a one-sided .001-nat bound requires the paired per-household loss-difference SE to
> be at most about 0.0006 (unadjusted) or 0.0004 (adjusted). Every comparison in the reference family is
> 1.4× to 8× above that, so a .001 confirmation is out of reach for releases whose per-person differences
> behave like the ones measured so far. The only committed cases that achieve the required precision are
> releases bitwise identical to their comparator, where the margin is demonstrated trivially and means
> nothing. A .001 confirmation therefore requires a release close enough to J that the demonstration is
> uninteresting — which is a constraint on what a confirmation could be worth, not a proof that no mechanism
> can reach the precision.

This keeps everything G0 is actually used for: the budget ladder is still set in advance from measured
precision, no stage may make a .001 confirmation claim, and the point-estimate screen remains the runnable
instrument. The practical consequence is unchanged.

## 4. Consequences for the registered predictions

P8 predicted `z × SE > .001` on at least one sensitive endpoint. That is **confirmed as written**, on every
endpoint and both weightings; this review does not touch it. What it touches is the *inference drawn beyond*
P8 in `REPORT.md` §5 and §8 ("a result about the whole research programme", "for any mechanism"). Suggested
handling: keep P8 confirmed, restate the consequence in the form of §3 above, and drop the universal
quantifier wherever it appears.

## 5. What this does not change

* No mechanism is more likely to succeed because of this review.
* The stochastic study still has **no ACS result**: no pool read, no release fitted, no compute spent.
* The .001 operating point still failed as a *screen* in the deterministic pilot, which is a measured
  outcome and is untouched.
* Nothing here is a reason to relax a margin after an outcome, which the registration correctly forbids.
