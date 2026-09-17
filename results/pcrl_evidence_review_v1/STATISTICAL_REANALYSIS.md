# Statistical reanalysis of the locked 2017 transport evaluation

Terminal B, independent of `scripts/report_acs_spectral_transport.py`. Every number below is
produced by `experiments/pcrl_evidence_review_v1/independent_scorer.py` and its two drivers,
which share no computation function with the study's report code.

## 0. What was reimplemented, and how independence was kept

| Component | Study | This reanalysis |
|---|---|---|
| Per-person log loss | `person_loss` in the report | own clip → renormalise → gather, on a copy |
| Candidate selection | read from `selected_scopes` | re-derived from stored 2017 validation losses via the origin/space predicate transcribed from `COMPARISONS.json` |
| Bootstrap | materialise a (2000 × 15924) replicate-by-row matrix | household aggregation: `(C @ Gv) / (C @ Gw)`, peak memory O(replicates × households) |
| Household codes | `np.unique(serial, return_inverse=True)` | `sorted(set(...))` + explicit index map |
| Simultaneous intervals | `simultaneous()` | coded from the protocol text |
| Interpreter | CPython 3.13.7 (project venv) | also run under CPython 3.14 with a different NumPy build |

Independence has a limit worth stating: to **replay the exact published intervals** the reanalysis
must use the declared randomness, `numpy.random.default_rng(20260917)` with 2,000 replicates. The
RNG stream is part of the protocol, not of the estimator, so reusing it is replay, not circularity.
The estimator arithmetic, the selection rule and the household indexing are all independent.

### Agreement

All 90 endpoints of families F1–F4 reproduce:

| Quantity | Maximum absolute difference |
|---|---|
| point estimate | 1.11e-15 |
| bootstrap standard error | 4.42e-17 |
| unadjusted interval endpoints | 2.40e-15 |
| adjusted interval endpoints | 1.73e-15 |

Critical values match exactly: F1 2.930105743069395, F2 3.023780, F3 2.728486, F4 3.205669.
Selection re-derived from validation losses for 2 scopes × 3 seeds × 14 interfaces × 11 forbidden
roles: **0 mismatches** (924 cells). Source: `FAMILY_AGREEMENT.json`, `SELECTION_RECHECK.json`.

### Conventions checked, not assumed

* **Studentisation.** `se_k` is the bootstrap SD of the replicate estimate; `T_b = max_k |θ*_bk − θ̂_k| / se_k`;
  `c` is the 95th percentile with `method='higher'`. Degenerate endpoints (`se ≤ 1e-15`) are excluded
  from the maximum and flagged. No F1–F4 endpoint is degenerate.
* **Weight denominators inside draws.** Weighted means are ratio estimators recomputed per replicate:
  the denominator is the replicate's own resampled weight total, not the fixed sample total.
* **Shared draws.** One set of household resample counts is used for every method, seed, task,
  weighting, mode and view, so paired differences are genuinely paired.
* **Household integrity.** Both rows of a two-person household always receive the same resample count.
* **Equal seed averaging.** The estimand averages the three fixed fitted systems with weight 1/3 after
  each seed's loss is computed; the bootstrap is applied to that average, not to a pooled sample.
* **Three seeds, one person.** The three seed predictions for a person are one person in the resample.
* **Recovery cancellation.** `endpoint_vectors` returns `prior − attack` (absolute recovery). Inside a
  paired contrast the shared prior cancels exactly, so **the paired difference of absolute recovery
  equals the paired difference of additional recovery**. This is why a single difference column serves
  both levels — and why the *levels* must still be reported separately (§3).

## 1. The registered F1 rule, and what it does and does not assert

The rule as implemented is

```python
res = all(d[f'{left} vs {r} / {w}']['residence_difference'] <= .001 + 1e-12 for r in rights for w in WEIGHTS)
```

with `residence_difference` the **seed-mean point estimate** of (C1 − comparator) residence log loss.
It is **one-sided** — C1 may be arbitrarily *better* — and it is a **point** criterion: no interval
enters it. The four values are 0.00029, 0.00010, 0.00059, 0.00017, so the rule passes, and it would
also pass under a two-sided reading. **The registered outcome is reproduced and stands.**

## 2. Retrospective noninferiority and equivalence at the 0.001 margin

These were not registered. They are computed here because the phrase "inside the prespecified 0.001
band" invites an interval reading that the evidence does not support.

Endpoint: `utility/same_residence`, Mode B, `transport_all`, 360. Lower is better, so the relevant
one-sided question is whether the **upper** bound is below +0.001.

| Contrast | Weight | Estimate | SE | Pointwise one-sided 95% upper | Noninferior? | Simultaneous one-sided 95% upper | Noninferior? | Two-sided simultaneous interval | Equivalent? |
|---|---|---:|---:|---:|---|---:|---|---|---|
| C1 − L1 | unweighted | 0.00029 | 0.00050 | **0.00112** | no | 0.00166 | no | [−0.00118, 0.00176] | no |
| C1 − L1 | PWGTP | 0.00010 | 0.00056 | **0.00103** | no | 0.00164 | no | [−0.00155, 0.00175] | no |
| C1 − L2 | unweighted | 0.00059 | 0.00045 | **0.00135** | no | 0.00182 | no | [−0.00073, 0.00191] | no |
| C1 − L2 | PWGTP | 0.00017 | 0.00053 | **0.00102** | no | 0.00162 | no | [−0.00138, 0.00172] | no |

**0 of 4 comparisons support noninferiority at 0.001, and 0 of 4 support two-sided equivalence** —
not under the family's simultaneous critical value, and not under the most permissive procedure we
computed, a pointwise one-sided bootstrap percentile with no multiplicity adjustment at all. The
margin is missed by 2% to 35%.

The correct reading is therefore: *C1's residence loss is within 0.001 of both local controls at the
point estimate, as the registered rule requires, and the data are compatible with a cost somewhat
larger than 0.001.* The pass is a prespecified-rule outcome on a sealed evaluation; it is not an
equivalence result and must not be reported as one. (Source: `EQUIVALENCE_F1.csv`.)

For contrast, in Mode A (F4) the C1 − L2 residence comparison *does* clear the pointwise one-sided
bound under both weightings (0.00084, 0.00049), and J − L025 clears it even simultaneously
(0.00077). Frozen attacks are a different and weaker inferential setting, and this is reported as
context, not as support for the Mode B claim.

## 3. "No significant worsening" is not "no harm"

The advantage rule requires that **no** sensitive endpoint has an adjusted lower bound above zero.
That is an absence-of-evidence condition. Two F1 endpoints illustrate the gap:

| Endpoint | Weight | Estimate | Adjusted 95% | Reading |
|---|---|---:|---|---|
| C1 − L1, A/SEX | unweighted | −0.00282 | [−0.00585, **0.00020**] | not significantly better; harmful values up to +0.0002/nat remain inside the interval |
| C1 − L2, A/SEX | unweighted | −0.00258 | [−0.00568, **0.00052**] | same; up to +0.0005 of *extra* A/SEX recovery is not excluded |

Under the PWGTP weighting both become significant. The rule is satisfied either way, because it asks
only that nothing be significantly worse. It does not certify that A/SEX is unaffected, and the paper
must not say it does.

## 4. Absolute versus additional recovery

The H baseline is itself substantial: an attacker on the bare service wire already recovers
0.0101 nats of A/SEX, 0.0165 of A/race, 0.0141 of AB/SEX and 0.0309 of AB/race. Mode B,
`transport_all`, 360, unweighted, seed-mean:

| Interface | residence gain vs H | A/SEX abs → add | A/race abs → add | AB/SEX abs → add | AB/race abs → add |
|---|---:|---|---|---|---|
| H | 0.0000 | 0.0101 → 0.0000 | 0.0165 → 0.0000 | 0.0141 → 0.0000 | 0.0309 → 0.0000 |
| J | 0.0189 | 0.0092 → **−0.0009** | 0.0241 → 0.0076 | 0.0159 → 0.0017 | 0.0384 → 0.0075 |
| spectral_L1 | 0.0285 | 0.0250 → 0.0149 | 0.0754 → 0.0590 | 0.0288 → 0.0146 | 0.0790 → 0.0481 |
| spectral_L2 | 0.0288 | 0.0248 → 0.0146 | 0.0692 → 0.0527 | 0.0275 → 0.0134 | 0.0741 → 0.0432 |
| spectral_C1 | 0.0282 | 0.0222 → 0.0121 | 0.0617 → 0.0453 | 0.0228 → 0.0087 | 0.0710 → 0.0401 |
| spectral_S0 | 0.0304 | 0.0398 → 0.0296 | 0.0905 → 0.0740 | 0.0397 → 0.0256 | 0.0928 → 0.0619 |

Two consequences:

1. **Ratios are scale-dependent; differences are not.** C1 versus J on A/race is 6.0× on increments
   and 2.6× on absolute values. The *paired difference* is 0.0377 [0.0294, 0.0459] on both scales,
   because the shared prior and H selection cancel. Quote the difference.
2. **J's A/SEX increment is negative** (−0.0009 unweighted, −0.0033 PWGTP with an interval excluding
   zero): the validation-selected fresh attack on J's wire does *worse than H's own attack*. Its
   absolute recovery, 0.0092, is below H's 0.0101. That is a property of selection, not a guarantee —
   and H's own attack, routed onto J's augmented wire, remains executable. A negative increment
   removes nothing that an H-only attacker already had.

## 5. Attack scope: common fresh versus expanded and frozen

Adding frozen 2018 candidates and catch-up trajectories lowers *additional* recovery for every
interface because it strengthens the subtracted H baseline, not because the augmented wire discloses
less. In `transport_all` the selected attack generalises worse than H's routed attack in 8 of 312
sensitive cells (15 of 312 in `common_fresh`). Counting negative selected increments: 22 of 429
unweighted role-cells in `transport_all`, 26 of 429 in `common_fresh`. None are clipped.
(Source: `SCOPE_COMPARISON.csv`, and the study's `NEGATIVE_INCREMENTS.csv`, `SELECTED_VS_ROUTED_H.csv`.)

## 6. Per-seed direction

| Family | endpoints | all 3 seeds agree with the seed-mean sign | all 3 seeds agree with the development sign |
|---|---:|---:|---:|
| F1 | 20 | 13 | 13 |
| all (F1–F4) | 90 | 48 | — |

The seven F1 endpoints where the seeds disagree are listed in `PER_SEED_DIRECTION.csv`; they include
both residence endpoints of each contrast and C1 − L1 A/SEX, where seed 2 has the opposite sign
(+0.0017 unweighted against a seed mean of −0.0028). The "20/20" statement is about seed means and
must be labelled as such.

## 7. Development versus transport: magnitude, not only resolution

F1 magnitude ratios |transport| / |development| range from **0.054× to 14.8×**
(`DEV_VS_TRANSPORT.csv`). Examples:

| Contrast / endpoint | Weight | Development seed-mean | Transport seed-mean | Ratio |
|---|---|---:|---:|---:|
| C1 − L2, AB/SEX | unweighted | −0.00032 | −0.00472 | 14.8× |
| C1 − L2, AB/race | unweighted | −0.00057 | −0.00302 | 5.3× |
| C1 − L1, AB/race | unweighted | −0.01427 | −0.00792 | 0.55× |
| C1 − L1, residence | unweighted | 0.00231 | 0.00029 | 0.13× |

A larger evaluation pool reduces standard errors; it does not move point estimates by a factor of 15.
The years differ in sample size, in calendar year, in evaluation pool and in which attacker families
were available. The study's "what changed is resolution, not direction" is therefore incomplete, and
attributing the newly significant C1-vs-L2 result to sample size alone is not supported.

## 8. Service parity versus probe quality — three different things

| Object | Status | Evidence |
|---|---|---|
| The released income/employment/coverage probability vectors | **identical**, bitwise, in every condition | structural; 210/210 released views verified |
| Service *accuracy* after shift | changes | employment +0.0138 nats, income −0.0145, coverage −0.0067 (2017 − 2018) |
| Independently fitted **source probes**, Mode B (fresh) | legacy allowance passes 3/3 seeds for all 14 interfaces | `CRITERIA_SUMMARY.csv` |
| Independently fitted **source probes**, Mode A (frozen 2018 probes) | every spectral arm **passes in only 0–1 of 3 seeds** | `CRITERIA_SUMMARY.csv` |
| Half-headroom against PCA32 / rich bank | passes in 0–2 of 3 seeds for every interface, both modes | `CRITERIA_SUMMARY.csv` |

A worse probe does not change a delivered service prediction: the vectors are the same bits. What the
Mode A row shows is that a *frozen 2018 probe* does not transfer onto a spectral wire in a new year;
refitting the probe repairs it. (`TRANSPORT_RESULTS.md` §8 states this pass count as a fail count;
see `ERRATA.md` E1.)

## 9. The C1 / J / control vector, with no exchange rate

Mode B, `transport_all`, 360, unweighted, seed-mean. Lower is better for recovery; higher is better
for residence gain. No scalar combines these columns.

| | residence gain | A/SEX add | AB/SEX add | A/race add | AB/race add | income probe | employment probe |
|---|---:|---:|---:|---:|---:|---|---|
| spectral_C1 | 0.0282 | 0.0121 | 0.0087 | 0.0453 | 0.0401 | +0.0081 vs J | +0.0135 vs J |
| J | 0.0189 | −0.0009 | 0.0017 | 0.0076 | 0.0075 | — | — |
| spectral_L1 | 0.0285 | 0.0149 | 0.0146 | 0.0590 | 0.0481 | — | — |
| spectral_L2 | 0.0288 | 0.0146 | 0.0134 | 0.0527 | 0.0432 | — | — |

C1 is better than J on exactly one component (residence, −0.0093 [−0.0150, −0.0036]) and worse on
six (four sensitive endpoints plus both source probes, all intervals excluding zero). "Same width"
is a **capacity** control — both channels are 16-dimensional — and says nothing about whether their
utility is comparable; §10 shows it is not.

## 10. Post-hoc average-utility matching under the declared withholding mechanism

For the declared family — `H` always released, an independent visible Bernoulli(p) branch fixed per
person deciding whether the channel is also released, and the routed predictor using `f_H` on
H-branch people — expected per-person loss is exactly `(1−p)L_H + p·L_aug` (verified to 2.2e-16 by
the study). Therefore the expected residence gain over H is exactly `p · gain_aug`, and the expected
additional recovery is exactly `p · additional_aug`. The probability that matches a target's **mean**
residence gain is then identified in closed form as `p* = gain_target / gain_source`.

This is **post-hoc average-utility matching**: `p*` is estimated from the same data as the outcome,
so the fixed-p intervals do not transfer to it, and it matches an *average* over people who receive
different information. The original fixed p-grid analysis is untouched.

Mode B, `transport_all`, 360, unweighted:

| Direction | p* | Identifiable in [0,1]? | Result at p* |
|---|---:|---|---|
| withhold **E** down to C1's residence gain | 1.072 | **no** | not reachable |
| withhold **A0** down to C1's | 0.976 | yes | C1 better on all four sensitive endpoints (e.g. A/race 0.0453 vs 0.0638) |
| withhold **L025** down to C1's | 1.615 | **no** | not reachable |
| withhold **L20** down to C1's | 1.555 | **no** | not reachable |
| withhold **J** down to C1's | 1.493 | **no** | **not reachable** |
| withhold **spectral_S0** down to C1's | 0.929 | yes | C1 better on all four (A/race 0.0453 vs 0.0688) |
| withhold **C1** down to J's residence gain | 0.670 | yes | **J better on all four** (A/race 0.0076 vs 0.0303) |

Two findings the original study did not extract:

1. **J cannot be made as useful as C1 by any withholding schedule.** It would need p* = 1.49. The
   C1-versus-J comparison is therefore not two points on one traversable frontier: J simply does not
   carry enough residence capability, and the same holds for E, L025 and L20.
2. **The ordering survives matching in both directions.** Where matching is identifiable, C1 beats
   withheld A0 and withheld S0 on every sensitive endpoint, and J beats withheld C1 on every
   sensitive endpoint. The negative verdict on the spectral design is robust to average-utility
   matching, and so is the positive verdict on coalition conditioning against its local controls.

Where `p*` falls outside [0,1] we report it as not identifiable and extrapolate nothing.

**Robust dominance is still absent, in both directions.** On the original fixed p-grid, withholding
dominates a given spectral arm in 1–3 of 180 comparisons and a spectral arm dominates withholding in
0–3 of 180. Neither family is robustly better on the full vector. A finite list of controls failing
to dominate C1 is not evidence that C1 is efficient; it is evidence about that list.

## 11. What could not be recovered from stored evidence, and was not invented

* **Retraining variability.** Three seeds are three fixed fitted systems. Nothing here estimates the
  variance of refitting the mechanism, and no interval is offered for it.
* **Census design variance.** PWGTP replicate weights were not used and are not in the stored
  evidence. The weighted numbers are ratio estimators under household resampling, which is not a
  design-based variance. Kish effective n under PWGTP on the final partition is 10,341.
* **A one-sided percentile bound for the *published* intervals.** The study stored 2.5/97.5
  percentiles only, so the one-sided bounds in §2 required recomputing the replicates. That is the
  smallest recomputation that answers the question, it re-fits nothing, and it reproduces the
  published two-sided endpoints to 2.4e-15 as a control.
* **Attack strength.** Every recovery number is a floor under the tested families. No procedure here
  turns it into a bound on I(S;Z|H).
