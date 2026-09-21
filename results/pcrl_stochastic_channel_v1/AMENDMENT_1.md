# AMENDMENT 1 — narrowing the G0 conclusion

**Date: 2026-09-20.** Published **before any new ACS outcome exists** for this study. No ACS pool has
been read by `pcrl_stochastic_channel_v1` at the time of writing; no release has been fitted.

**Status of G0 is unchanged.** It was registered in `REGISTRATION.md` §6 as a gate that "cannot fail
the study; it can only constrain what may be claimed" — i.e. **not** an operational stopping gate.
This amendment does not change that status. It narrows what G0 concluded.

**G0's measurements are unchanged and are not restated as corrected.** `gates/G0.json` keeps every
number it recorded. Nothing in this amendment alters a measurement, a margin, or any other
scientific criterion.

**The `.001`-nat margin is unchanged.** So is every other criterion, gate, selection rule, stopping
rule and registered prediction.

---

## 1. What was claimed, and why it was wrong

`REPORT.md` §5 and `gates/G0.json` concluded:

> "A `.001`-nat confirmation claim is not demonstrable at this sample size even with a point estimate
> of exactly zero, on any endpoint, under any weighting" — and, extrapolating, "for any mechanism,
> not just this one."

**The extrapolation is withdrawn.** Two errors.

### Error 1 — the gate is one-sided, so a negative estimate passes

The criterion is an upper bound on *additional* sensitive recovery over J:

```
estimate + z * SE  <=  .001
```

A half-width above `.001` does **not** rule out passing. It only means the estimate cannot be at or
near zero. A sufficiently **negative** estimate — a mechanism that *reduces* measured sensitive
recovery relative to J — passes at any half-width observed in the historical record. The threshold
estimate at each observed half-width:

| endpoint / weighting | observed adjusted half-width | passes if estimate ≤ |
|---|---|---|
| smallest anywhere in the record | 0.00220 | **−0.00120** |
| `AB/SEX` unweighted (median) | 0.00693 | −0.00593 |
| `A/SEX` unweighted (median) | 0.00720 | −0.00620 |
| `AB/RAC1P` unweighted (median) | 0.01034 | −0.00934 |
| `A/RAC1P` person-weighted (median) | 0.01359 | −0.01259 |

These are not unreachable magnitudes in this project. The predecessor's own `pcrl_nonlinear_rank_v1`
recorded significant attack *reductions* with adjusted upper bounds below zero at exactly this scale
(A/SEX −0.0064, A/RAC1P −0.0073, AB/RAC1P −0.0116). My §5 reasoning implicitly assumed the estimate
would be ≈ 0, which is the regime where the pilot's mechanism landed — not a property of the gate.

### Error 2 — historical half-widths do not lower-bound a new mechanism's SE

`SE` is the paired-household bootstrap standard error of the *per-household loss difference for the
specific pair being compared*. It is a property of that pair, not of the pool. A mechanism whose
release agrees with J on most households produces small paired differences and can have a materially
smaller `SE` than any pair in the historical record. Nothing in G0 measured, or could measure, the
paired-loss variance of a channel that does not yet exist.

## 2. The narrowed conclusion

G0 establishes:

> **For the historical comparisons on the 2018 development pools, precision is limited near
> equality.** On these four sensitive endpoints, an adjusted one-sided upper bound at or below
> `.001` was not achievable for a comparison whose point estimate sat near zero: the smallest
> adjusted half-width in the record is 0.00220 nats and the medians run 0.0069–0.0136. This already
> holds unadjusted at one-sided 95% (smallest 0.00141), so multiplicity is not the cause.

That is all it establishes. It is a statement about **those comparisons** and about the
**near-equality regime**. It is not a statement about the reachability of the operating point, and
not a statement about any mechanism's achievable precision.

## 3. Status of registered prediction P8

P8 predicted: *"The paired-household precision check shows `z × SE > .001` on at least one sensitive
endpoint, i.e. the `.001` margin is not demonstrable at this sample size even if the point estimate
is zero."*

**P8 stands as CONFIRMED in its literal form.** Its first clause is confirmed on every endpoint, and
its gloss is correctly conditioned on *"even if the point estimate is zero"*. The failure was in
`REPORT.md`, which dropped that condition and generalized to "for any mechanism". The literal
prediction is retained as confirmed; the generalization is withdrawn here.

## 4. Consequences for the remaining stages

1. **Candidate-specific intervals are computed and reported wherever prescribed.** They are **not**
   withheld on the grounds that historical precision was poor. Each nominated candidate gets its own
   paired-household bootstrap against the same-host reference, with its own `SE`.
2. **Bootstrap units are households.** More release draws from `Q`, more optimizer seeds, or more
   fitted channels do **not** add independent units, and a Monte Carlo replication count is reported
   as such, never as sample size.
3. **The budget ladder is not relaxed.** `.001` remains the margin. G0 informs what a *near-equality*
   result could and could not support; it does not license a looser threshold.
4. **A negative estimate is the route to a pass**, and this is now stated prospectively rather than
   discovered afterwards: for this study to pass G5 with a confirmable bound, a candidate must
   *reduce* measured additional sensitive recovery relative to same-host J, not merely fail to
   increase it. That is a harder and more honest target than the one §5 implied was impossible.

## 5. Files touched by this amendment

* `gates/G0.json` — an `amendment` block added; the superseded statements are retained verbatim under
  `superseded_conclusions` and **every measurement is left exactly as recorded**.
* `REPORT.md` §5 — the overreaching conclusion replaced by §2 above, with a pointer here.
* `HANDOFF.json` — the headline for Terminal 2 replaced by §2 above.

No other file is altered, and no historical study's record is touched.
