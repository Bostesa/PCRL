# EXPLORATORY_2017 — cross-year development, not a confirmation

**Every number in this file is EXPLORATORY CROSS-YEAR DEVELOPMENT evidence.**

The 2017 locked evaluation is finished and its `final_evaluation` partition is spent.
So this file:

* is **not** a second confirmation of anything;
* does **not** restate, reinterpret or overwrite the original frozen 2017 transport
  result, which keeps its historical status;
* uses the 2017 `final_evaluation` rows only as **now-exposed development
  evaluation**;
* reports **no simultaneous intervals**. No simultaneous intervals are computed on this partition. Its seal is spent, and an interval here would invite a confirmatory reading the evidence cannot support.

Probes were fitted on the 2017 attacker/task fitting partitions and selected on the
2017 validation partitions by minimum unweighted validation log loss, so selection
still never saw the evaluation rows. `ev.load_final` is deliberately not called:
reusing a spent seal would misrepresent the evidence. Years are reported separately
and rows are never pooled across years.

## Scope `common_fresh`, budget 360, seed means, unweighted

| condition | residence gain over H | add. A/SEX | add. A/RAC1P | add. AB/SEX | add. AB/RAC1P |
|---|---|---|---|---|---|
| `H` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `J` | 0.0189 | 0.0013 | 0.0039 | 0.0069 | 0.0015 |
| `E` | 0.0263 | 0.0184 | 0.0473 | 0.0161 | 0.0371 |
| `A0` | 0.0289 | 0.0325 | 0.0632 | 0.0312 | 0.0564 |
| `spectral_S0` | 0.0304 | 0.0346 | 0.0758 | 0.0317 | 0.0654 |
| `spectral_C1` | 0.0282 | 0.0170 | 0.0471 | 0.0148 | 0.0436 |
| `spectral_L1` | 0.0285 | 0.0198 | 0.0608 | 0.0207 | 0.0515 |
| `spectral_L2` | 0.0288 | 0.0196 | 0.0545 | 0.0195 | 0.0466 |
| `spectral_lin16_L1` | 0.0285 | 0.0198 | 0.0608 | 0.0207 | 0.0515 |
| `spectral_lin16_L2` | 0.0288 | 0.0196 | 0.0545 | 0.0195 | 0.0466 |
| `spectral_lin16_C1` | 0.0282 | 0.0170 | 0.0471 | 0.0148 | 0.0436 |
| `spectral_nlr16_L1` | 0.0285 | 0.0174 | 0.0528 | 0.0193 | 0.0498 |
| `spectral_nlr16_L2` | 0.0277 | 0.0108 | 0.0362 | 0.0158 | 0.0343 |
| `spectral_nlr16_C1` | 0.0276 | 0.0120 | 0.0347 | 0.0116 | 0.0300 |
| `spectral_lin8_L1` | 0.0272 | 0.0177 | 0.0347 | 0.0198 | 0.0330 |
| `spectral_lin8_L2` | 0.0257 | 0.0170 | 0.0171 | 0.0177 | 0.0185 |
| `spectral_lin8_C1` | 0.0254 | 0.0164 | 0.0162 | 0.0143 | 0.0168 |
| `spectral_nlr8_L1` | 0.0283 | 0.0145 | 0.0401 | 0.0180 | 0.0407 |
| `spectral_nlr8_L2` | 0.0255 | 0.0122 | 0.0145 | 0.0106 | 0.0161 |
| `spectral_nlr8_C1` | 0.0267 | 0.0105 | 0.0155 | 0.0127 | 0.0154 |

## Scope `transport_all`, budget 360, seed means, unweighted

| condition | residence gain over H | add. A/SEX | add. A/RAC1P | add. AB/SEX | add. AB/RAC1P |
|---|---|---|---|---|---|
| `H` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `J` | 0.0189 | -0.0009 | 0.0076 | 0.0017 | 0.0075 |
| `E` | 0.0263 | 0.0164 | 0.0629 | 0.0158 | 0.0537 |
| `A0` | 0.0289 | 0.0276 | 0.0653 | 0.0251 | 0.0536 |
| `spectral_S0` | 0.0304 | 0.0296 | 0.0740 | 0.0256 | 0.0619 |
| `spectral_C1` | 0.0282 | 0.0121 | 0.0453 | 0.0087 | 0.0401 |
| `spectral_L1` | 0.0285 | 0.0149 | 0.0590 | 0.0146 | 0.0481 |
| `spectral_L2` | 0.0288 | 0.0146 | 0.0527 | 0.0134 | 0.0432 |
| `spectral_lin16_L1` | 0.0285 | 0.0149 | 0.0590 | 0.0146 | 0.0481 |
| `spectral_lin16_L2` | 0.0288 | 0.0146 | 0.0527 | 0.0134 | 0.0432 |
| `spectral_lin16_C1` | 0.0282 | 0.0121 | 0.0453 | 0.0087 | 0.0401 |
| `spectral_nlr16_L1` | 0.0285 | 0.0125 | 0.0510 | 0.0132 | 0.0464 |
| `spectral_nlr16_L2` | 0.0277 | 0.0059 | 0.0343 | 0.0097 | 0.0309 |
| `spectral_nlr16_C1` | 0.0276 | 0.0070 | 0.0329 | 0.0055 | 0.0265 |
| `spectral_lin8_L1` | 0.0272 | 0.0127 | 0.0329 | 0.0137 | 0.0295 |
| `spectral_lin8_L2` | 0.0257 | 0.0120 | 0.0153 | 0.0116 | 0.0151 |
| `spectral_lin8_C1` | 0.0254 | 0.0114 | 0.0144 | 0.0082 | 0.0133 |
| `spectral_nlr8_L1` | 0.0283 | 0.0096 | 0.0383 | 0.0119 | 0.0373 |
| `spectral_nlr8_L2` | 0.0255 | 0.0072 | 0.0127 | 0.0045 | 0.0126 |
| `spectral_nlr8_C1` | 0.0267 | 0.0056 | 0.0137 | 0.0066 | 0.0120 |

## Does the 2018 attribution reproduce in direction?

For each nonlinear-versus-original contrast at matched rank and policy, the sign
of the change in additional recovery is compared against the 2018 development
result, per seed: **188 of 288 seed-level comparisons agree in sign**.

This is a direction check on an exposed partition. It is not a replication,
it carries no interval, and a cross-year change in effect magnitude is not
attributed to sample size here — the predecessor study saw magnitudes move by
large factors between years, and its own per-seed sign agreement was weaker than
its seed-mean statement suggested.

| scope | contrast | weight | endpoint | seeds | sign agrees |
|---|---|---|---|---|---|
| common_fresh | `spectral_nlr16_L1 vs spectral_lin16_L1` | unweighted | A/RAC1P | 3 | 1 |
| common_fresh | `spectral_nlr16_L1 vs spectral_lin16_L1` | unweighted | AB/SEX | 3 | 1 |
| common_fresh | `spectral_nlr16_L1 vs spectral_lin16_L1` | person_weighted | A/SEX | 3 | 1 |
| common_fresh | `spectral_nlr16_L1 vs spectral_lin16_L1` | person_weighted | A/RAC1P | 3 | 1 |
| common_fresh | `spectral_nlr16_L2 vs spectral_lin16_L2` | unweighted | AB/SEX | 3 | 1 |
| common_fresh | `spectral_nlr16_L2 vs spectral_lin16_L2` | unweighted | AB/RAC1P | 3 | 2 |
| common_fresh | `spectral_nlr16_L2 vs spectral_lin16_L2` | person_weighted | A/SEX | 3 | 2 |
| common_fresh | `spectral_nlr16_L2 vs spectral_lin16_L2` | person_weighted | AB/SEX | 3 | 1 |
| common_fresh | `spectral_nlr16_L2 vs spectral_lin16_L2` | person_weighted | AB/RAC1P | 3 | 2 |
| common_fresh | `spectral_nlr16_C1 vs spectral_lin16_C1` | unweighted | A/SEX | 3 | 2 |
| common_fresh | `spectral_nlr16_C1 vs spectral_lin16_C1` | unweighted | A/RAC1P | 3 | 2 |
| common_fresh | `spectral_nlr16_C1 vs spectral_lin16_C1` | unweighted | AB/SEX | 3 | 2 |
| common_fresh | `spectral_nlr16_C1 vs spectral_lin16_C1` | unweighted | AB/RAC1P | 3 | 2 |
| common_fresh | `spectral_nlr16_C1 vs spectral_lin16_C1` | person_weighted | A/SEX | 3 | 2 |
| common_fresh | `spectral_nlr16_C1 vs spectral_lin16_C1` | person_weighted | A/RAC1P | 3 | 2 |
| common_fresh | `spectral_nlr16_C1 vs spectral_lin16_C1` | person_weighted | AB/SEX | 3 | 2 |
| common_fresh | `spectral_nlr16_C1 vs spectral_lin16_C1` | person_weighted | AB/RAC1P | 3 | 2 |
| common_fresh | `spectral_nlr8_L1 vs spectral_lin8_L1` | unweighted | A/SEX | 3 | 2 |
| common_fresh | `spectral_nlr8_L1 vs spectral_lin8_L1` | unweighted | AB/SEX | 3 | 1 |
| common_fresh | `spectral_nlr8_L1 vs spectral_lin8_L1` | unweighted | AB/RAC1P | 3 | 2 |
| common_fresh | `spectral_nlr8_L1 vs spectral_lin8_L1` | person_weighted | A/SEX | 3 | 2 |
| common_fresh | `spectral_nlr8_L1 vs spectral_lin8_L1` | person_weighted | AB/SEX | 3 | 1 |
| common_fresh | `spectral_nlr8_L1 vs spectral_lin8_L1` | person_weighted | AB/RAC1P | 3 | 2 |
| common_fresh | `spectral_nlr8_L2 vs spectral_lin8_L2` | unweighted | A/SEX | 3 | 2 |
| common_fresh | `spectral_nlr8_L2 vs spectral_lin8_L2` | unweighted | A/RAC1P | 3 | 1 |
| common_fresh | `spectral_nlr8_L2 vs spectral_lin8_L2` | unweighted | AB/SEX | 3 | 1 |
| common_fresh | `spectral_nlr8_L2 vs spectral_lin8_L2` | unweighted | AB/RAC1P | 3 | 2 |
| common_fresh | `spectral_nlr8_L2 vs spectral_lin8_L2` | person_weighted | A/RAC1P | 3 | 1 |
| common_fresh | `spectral_nlr8_L2 vs spectral_lin8_L2` | person_weighted | AB/SEX | 3 | 1 |
| common_fresh | `spectral_nlr8_L2 vs spectral_lin8_L2` | person_weighted | AB/RAC1P | 3 | 0 |
| common_fresh | `spectral_nlr8_C1 vs spectral_lin8_C1` | unweighted | A/RAC1P | 3 | 2 |
| common_fresh | `spectral_nlr8_C1 vs spectral_lin8_C1` | unweighted | AB/SEX | 3 | 2 |
| common_fresh | `spectral_nlr8_C1 vs spectral_lin8_C1` | unweighted | AB/RAC1P | 3 | 2 |
| common_fresh | `spectral_nlr8_C1 vs spectral_lin8_C1` | person_weighted | A/RAC1P | 3 | 2 |
| common_fresh | `spectral_nlr8_C1 vs spectral_lin8_C1` | person_weighted | AB/SEX | 3 | 2 |
| common_fresh | `spectral_nlr8_C1 vs spectral_lin8_C1` | person_weighted | AB/RAC1P | 3 | 2 |
| transport_all | `spectral_nlr16_L1 vs spectral_lin16_L1` | unweighted | A/RAC1P | 3 | 1 |
| transport_all | `spectral_nlr16_L1 vs spectral_lin16_L1` | unweighted | AB/SEX | 3 | 1 |
| transport_all | `spectral_nlr16_L1 vs spectral_lin16_L1` | person_weighted | A/SEX | 3 | 1 |
| transport_all | `spectral_nlr16_L1 vs spectral_lin16_L1` | person_weighted | A/RAC1P | 3 | 1 |
| transport_all | `spectral_nlr16_L2 vs spectral_lin16_L2` | unweighted | AB/SEX | 3 | 1 |
| transport_all | `spectral_nlr16_L2 vs spectral_lin16_L2` | unweighted | AB/RAC1P | 3 | 2 |
| transport_all | `spectral_nlr16_L2 vs spectral_lin16_L2` | person_weighted | A/SEX | 3 | 2 |
| transport_all | `spectral_nlr16_L2 vs spectral_lin16_L2` | person_weighted | AB/SEX | 3 | 1 |
| transport_all | `spectral_nlr16_L2 vs spectral_lin16_L2` | person_weighted | AB/RAC1P | 3 | 2 |
| transport_all | `spectral_nlr16_C1 vs spectral_lin16_C1` | unweighted | A/SEX | 3 | 2 |
| transport_all | `spectral_nlr16_C1 vs spectral_lin16_C1` | unweighted | A/RAC1P | 3 | 2 |
| transport_all | `spectral_nlr16_C1 vs spectral_lin16_C1` | unweighted | AB/SEX | 3 | 2 |
| transport_all | `spectral_nlr16_C1 vs spectral_lin16_C1` | unweighted | AB/RAC1P | 3 | 2 |
| transport_all | `spectral_nlr16_C1 vs spectral_lin16_C1` | person_weighted | A/SEX | 3 | 2 |
| transport_all | `spectral_nlr16_C1 vs spectral_lin16_C1` | person_weighted | A/RAC1P | 3 | 2 |
| transport_all | `spectral_nlr16_C1 vs spectral_lin16_C1` | person_weighted | AB/SEX | 3 | 2 |
| transport_all | `spectral_nlr16_C1 vs spectral_lin16_C1` | person_weighted | AB/RAC1P | 3 | 2 |
| transport_all | `spectral_nlr8_L1 vs spectral_lin8_L1` | unweighted | A/SEX | 3 | 2 |
| transport_all | `spectral_nlr8_L1 vs spectral_lin8_L1` | unweighted | AB/SEX | 3 | 1 |
| transport_all | `spectral_nlr8_L1 vs spectral_lin8_L1` | unweighted | AB/RAC1P | 3 | 2 |
| transport_all | `spectral_nlr8_L1 vs spectral_lin8_L1` | person_weighted | A/SEX | 3 | 2 |
| transport_all | `spectral_nlr8_L1 vs spectral_lin8_L1` | person_weighted | AB/SEX | 3 | 1 |
| transport_all | `spectral_nlr8_L1 vs spectral_lin8_L1` | person_weighted | AB/RAC1P | 3 | 2 |
| transport_all | `spectral_nlr8_L2 vs spectral_lin8_L2` | unweighted | A/SEX | 3 | 2 |
| transport_all | `spectral_nlr8_L2 vs spectral_lin8_L2` | unweighted | A/RAC1P | 3 | 1 |
| transport_all | `spectral_nlr8_L2 vs spectral_lin8_L2` | unweighted | AB/SEX | 3 | 1 |
| transport_all | `spectral_nlr8_L2 vs spectral_lin8_L2` | unweighted | AB/RAC1P | 3 | 2 |
| transport_all | `spectral_nlr8_L2 vs spectral_lin8_L2` | person_weighted | A/RAC1P | 3 | 1 |
| transport_all | `spectral_nlr8_L2 vs spectral_lin8_L2` | person_weighted | AB/SEX | 3 | 1 |
| transport_all | `spectral_nlr8_L2 vs spectral_lin8_L2` | person_weighted | AB/RAC1P | 3 | 0 |
| transport_all | `spectral_nlr8_C1 vs spectral_lin8_C1` | unweighted | A/RAC1P | 3 | 2 |
| transport_all | `spectral_nlr8_C1 vs spectral_lin8_C1` | unweighted | AB/SEX | 3 | 2 |
| transport_all | `spectral_nlr8_C1 vs spectral_lin8_C1` | unweighted | AB/RAC1P | 3 | 2 |
| transport_all | `spectral_nlr8_C1 vs spectral_lin8_C1` | person_weighted | A/RAC1P | 3 | 2 |
| transport_all | `spectral_nlr8_C1 vs spectral_lin8_C1` | person_weighted | AB/SEX | 3 | 2 |
| transport_all | `spectral_nlr8_C1 vs spectral_lin8_C1` | person_weighted | AB/RAC1P | 3 | 2 |

(only rows where agreement is not unanimous are listed; the full set is in
`EXPLORATORY_2017.json`)

## Reading

**Alias check.** `spectral_lin16_L1/L2/C1` reproduce the historical
`spectral_L1/L2/C1` rows digit for digit. Those nine conditions are the historical
objects, reused and not refitted, and this table confirms the aliasing end to end
on a second year.

**The two 2018 effects reproduce in direction at the seed mean.** The nonlinear
penalty again lowers recovery at matched rank and policy (`nlr16_C1` sits below
`lin16_C1` on all four family sensitive endpoints), and rank-8 compression again
lowers race recovery sharply (`lin8_C1` additional `A/RAC1P` 0.0162 against
`lin16_C1` 0.0471).

**But the comparison against J is worse here than in 2018, not better.** On this
partition J has lower additional recovery than the best new candidate on **all**
four family sensitive endpoints (J 0.0013/0.0039/0.0069/0.0015 against `nlr8_C1`
0.0105/0.0155/0.0127/0.0154), while `nlr8_C1` keeps the residence advantage
(0.0267 against 0.0189). In 2018 the same candidate was statistically
indistinguishable from J on both race endpoints. So the one place the 2018 result
came closest to J does **not** carry over. This strengthens the no-go verdict
rather than weakening it, which is the useful thing an exploratory cross-year look
can do.

**Per-seed direction agreement is only moderate**, not the near-unanimity a
seed-mean statement would suggest. That is consistent with the independent review
finding that the predecessor's "same signs on all 20 endpoints" claim was a
seed-mean statement whose per-seed agreement was weaker. Three seeds on an exposed
partition cannot resolve this, and no interval is offered that would imply
otherwise.

**What this section is not.** It is not a confirmation, not a replication, and not
evidence that any candidate should proceed. The decision in `RESEARCH_DECISION.md`
rests on the 2018 development result and is unchanged by this table.

