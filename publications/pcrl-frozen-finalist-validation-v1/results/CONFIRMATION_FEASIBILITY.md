# Confirmation feasibility: frozen finalist `B3CTL_UNCOUPLED_MIX50_C6A`

**Data.** Public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida, in the development roles. The service predicts income class (2) and occupation group (5) from AGEP, WKHP, COW, SCHL, MAR and POBP. Readers infer the recorded SEX column from the released predictions.

**Scope.** Planning arithmetic from existing evidence. No new label, population or download was opened.

**Numbers.** They come from `CONFIRMATION_FEASIBILITY_RESULTS_V2.json` (unit FFV_FEASIBILITY_V2, `ffv/feasibility_v2.py`).
- **V2 corrects V1 after the statistical reviewer's feasibility review** (`roles/statistical_reviewer/FEASIBILITY_REVIEW.md`; V1, `CONFIRMATION_FEASIBILITY_RESULTS.json`, is kept unchanged):
  - limits computed with correlated boundary clauses;
  - assurance limits computed;
  - "rough" rather than "lower bound" for the measured refit term;
  - full curves saved.
- **No claim label, n80 or n90 value, or conclusion changed.**
  - V2 did change feasibility status labels: the 6 assurance cells moved from NOT_REACHED_BY_1E8 (V1, NaN limit) to LIMIT_BELOW_TARGET.
  - The governing-scorecard assignment changed in 8 scenario × definition cells, under the new tie rule.
  - Some governing assignments in near-tie cells depend on Monte Carlo noise. No decision depends on them (`INDEPENDENT_VERIFICATION.json`, V-D11).
- **The specification was frozen in PROTOCOL.md §10 before the DEV_REPORT re-read.** The reviewer's independent recomputation agrees: its n80 values fall inside the adjacent-grid brackets. The verifier also checks the calculation (`INDEPENDENT_VERIFICATION.json`).

## Question

How many independent households would a prospective study of this frozen release need to show all registered conditions jointly with 80% or 90% probability? The conditions are:
- the 24 utility clauses;
- the 4 within-SEX guards;
- the recipient non-worsening clauses;
- the 4 pair-recovery superiority clauses against NEURAL_RAW and H_LJ;
- separately, the 0.020 material target.

## Method and scope

**Estimand.** The frozen releases, with the frozen fitted readers and winners, applied to new independent households from the same Florida 2019 population, in an unweighted analysis.

The curves do not apply to designs that:
- refit readers or re-select winners (that endpoint is a maximum over candidates);
- transport to another state or year;
- use design weights.

The two seeds share people and are not replications.

**Approximation.**
- **Endpoint distribution.** Each endpoint is a household-clustered ratio mean or an AUC. Estimates for n households are approximately multivariate normal. The covariance is taken from the 2,048 shared household-bootstrap replicates of the logged evaluation, scaled by 3,955 / n, and eigendecomposed.
- **Monte Carlo.** Joint power is a Monte Carlo probability: 40,000 draws, seed 2026101803, common random numbers. The grid runs from 3,955 to 10^8 households. The n → ∞ limits are Monte Carlo values at n = 10^14.
- **Check.** The reviewer's nonparametric rescaling of the replicate deviations agrees with the normal Monte Carlo, so normality does not drive the result.

**Definitions** (claim B's CV and availability assumed):

| Definition | Requires |
|---|---|
| D1, the source scorecard | utility and guard point gates; claim B's recipient clause bounds ≤ 0.010; the 4 benefit clause bounds > 0 (c1 = 2.395) |
| D2, confirmatory | D1, but every utility and guard condition needs an upper bound ≤ cap |
| D3 | D2 plus every benefit bound > 0.020 |

**Scorecards.** Both are run. The less favourable governs: the larger n80, or, when neither reaches 80%, the lower power at 10^8.

## What binds: income utility slack, not recovery

| Clause (reader-free; the same in both scorecards) | Estimate | Cap | Slack | SE on DEV | Slack / SE | Posterior P(slack > 0) |
|---|---|---|---|---|---|---|
| income Brier excess vs Ucal, seed 1 | 0.004944 | 0.005 | 5.6e-5 | 0.00205 | 0.027 | 0.51 |
| income Brier excess vs Ucal, seed 0 | 0.004919 | 0.005 | 8.1e-5 | 0.00193 | 0.042 | 0.52 |
| income accuracy loss, seed 1 (U0 = Ucal) | 0.009591 | 0.010 | 4.1e-4 | 0.00372 | 0.110 | 0.54 |

**The data cannot tell this slack from zero or from a negative value.** The required sample size is therefore not identified by these data.

**At the plug-in utility slack, income utility binds.** The ENHANCED plug-in pair benefits are:

| Comparator | Seed 0 | Seed 1 |
|---|---|---|
| NEURAL_RAW | +0.0270 | +0.0217 |
| H_LJ | +0.0284 | +0.0260 |

Each already passes its clause at DEV's 3,955 households. Two other clauses have their own n80 (from the reviewer's analytic check):
- the ENHANCED seed-1 income recipient clause against H_LJ (point +0.0076, SE 0.0066) is UNRESOLVED at DEV's size and alone needs about 318,000 households for its 0.010 bound. It sets the ENHANCED requirement whenever the true utility slack is 0.002 or more, so recovery is not free of constraints;
- the ENHANCED weakest benefit, against NEURAL_RAW on seed 1, needs about 522,000 for a bound above 0.020.

## Results (ENHANCED shown; ORIGINAL in brackets where it differs)

The less favourable scorecard governs (`governing_scorecard_by_scenario`). That is ORIGINAL in the D3 column, in the D2 boundary cells and in the combined-term D1 cell, and ENHANCED elsewhere.

**Required households, by assumed true slack of the binding clauses** (80% joint power; other endpoints at plug-in):

| Assumed true slack | D1 | D2 | D3 |
|---|---|---|---|
| 0.002 (largest on the registered grid) | 323,000 [42,000] | 397,000 [323,000] | 597,000 [not reached: limit 0 under the ORIGINAL plug-in benefit] |
| 0.001 | 323,000 [70,000] | 662,000 [597,000] | 733,000 [limit 0] |
| 0.0005 | 397,000 [215,000] | 2.5 million | 2.5 million [limit 0] |
| plug-in (5.6e-5 to 4.1e-4) | 4.6 million | 59.9 million | 59.9 million [limit 0] |
| 0 | does not reach 80% at any n: limit 0.266 [0.268] | limit 0.0004 [0.0003] | limit 0.0004 [0] |
| −0.001 | limit 0 | limit 0 | limit 0 |

- At an assumed slack of 0.002, the ENHANCED D1 requirement (323,000) is set by its recipient clause against H_LJ. Requirements above the registered grid were not computed. The ORIGINAL requirement would fall further; the ENHANCED requirement cannot fall below about 318,000.
- **At the plug-in slack, D2 needs about 6×10^7 households, which exceeds the number of households in Florida (order 10^7).**

**Other scenarios** (plug-in utility):

| Scenario | D1, 80% / 90% | D2, 80% / 90% | D3 |
|---|---|---|---|
| Benefits × 0.75 (presented first: the plug-in benefits come from the favourably selected result) | 4.6 M / 9.5 M | 59.9 M / 73.5 M | not reliably establishable at 0.020 as n grows under this scenario |
| Benefits × 0.5 | 4.6 M / 9.5 M | 59.9 M / 73.5 M | as above |
| Binding clauses halfway to the cap | 17.5 M / 35.9 M | not reached by 10^8 | not reached by 10^8 [limit 0] |
| + rough measured reader-refit term (τ_r 0.0036–0.0134) | 5.1 M / 11.6 M (limit 0.975) | 59.9 M (limit 0.975) | limit 0.317 [0.216] |
| + rough reader-refit + HYPOTHETICAL release-retraining term (τ_h 0.0017 / 0.0037 ENHANCED; 0.0024 / 0.0050 ORIGINAL) | 5.1 M [5.7 M] (limit 0.970 [0.964]) | 59.9 M | limit 0.287 [0.209] |

**Assurance** (a flat-prior predictive heuristic centred on favourably selected development estimates; neither a probability that the release complies nor a frequentist statement). Averaging over development uncertainty in the true effects, the joint success probability stays at or below about 0.10 (ENHANCED) or 0.16 (ORIGINAL) at every n:

| Definition | ENHANCED limit | ORIGINAL limit |
|---|---|---|
| D1 | 0.102 | 0.161 |
| D2 | 0.102 | 0.161 |
| D3 | 0.053 | 0.064 |

**Same-size replication** (descriptive plug-in probability, not a test). A replication with 3,955 households would pass D1 with probability about 0.003 (ENHANCED) or 0.115 (ORIGINAL), and D2 with about 0. The source pass should not be described as "expected to replicate".

**Study sizes that exist.** The only Florida 2019 people not used by any model or reader are the sealed 20% (9,853 households) and the capped-out households (7,657). Both carry the replication's disclosure D2, an in-memory extraction (`POPULATION_INVENTORY.md`).

| Definition | Joint power at 10,000 households, ENHANCED [ORIGINAL] | At about 17,500 households, ENHANCED [ORIGINAL] |
|---|---|---|
| D1 | 0.007 [0.215] | 0.013 [0.252] |
| D2 | about 0 | about 0 |

## Refit variation (not covered by the conditional intervals)

**The refit terms enter only in scenario S4**, as a study-level random effect on the 4 pair-benefit endpoints, independent across benefits; a shared finalist effect would raise joint power, so this is conservative. The recipient AUC clauses get no term, because their readers were not refitted.

**Rough measured spread.**
- **Source.** Model-seed refits of seed-dependent pair winners, plus household-bootstrap refits of ATTACK_FIT. Each has at most 4 degrees of freedom.
- **Size.** Per comparison, the benefit SD across matched refits was 0.003–0.011, about the size of the conditional benefit SEs.
- **Bias.** All 24 bootstrap refits read lower on DEV_REPORT than the full-sample fits, so the bootstrap term mixes a resampling shift with variance. The direction of the net bias is unknown. It is not a lower bound, contrary to the wording of PROTOCOL.md §7.6.
- **Descriptive.** With bootstrap-trained readers, the benefit shrank in 3 of the 4 seed × reference cells: further evidence that the benefit is reader-relative.

**HYPOTHETICAL release-retraining term.**
- It is |benefit(s0) − benefit(s1)| / √2: 0.0017 (H_LJ) and 0.0037 (NEURAL_RAW).
- It has 1 degree of freedom, and is confounded with the reader seed and same-people noise.
- It is not a measured component. It enters only combined with the rough term, under the HYPOTHETICAL label.

## Conclusions (each conditional on its scenario's assumed true values)

1. **No realistic confirmation scenario was found at the frozen release's utility margins.**
   - The required sample size is not identified by these data, because it depends almost entirely on the unknown true income utility slack. It ranges:
     - from about 3.2×10^5 households under the governing ENHANCED scorecard (set by its recipient clause; ORIGINAL about 4×10^4) at an assumed slack of 0.002, the largest on the registered grid;
     - through about 4.6×10^6 (D1) and 6×10^7 (D2) at the plug-in;
     - to never reaching 80% if the slack is zero or negative.
   - At the size of the only Florida 2019 people who could test this release (about 17,500 households), D1 joint power is about 0.01 under the governing ENHANCED scorecard (about 0.25 under ORIGINAL), and D2 is about 0.
2. **A fresh population does not change this arithmetic.** The binding quantity is a property of the release's utility margins. Another state-year would be a transport, not a test of this frozen release.
3. **The 0.020 target** is not reliably establishable at 0.020 under the ORIGINAL plug-in, or under 75% or 50% shrinkage. Under the ENHANCED plug-in it would need about 6×10^7 households. Its assurance is about 0.05–0.06. This does not show that it is false under every possible true effect.
4. **The four pair-benefit clauses are not the obstacle, but recovery is not free of constraints.** The ENHANCED recipient clause against H_LJ alone needs about 318,000 households. Reader-refit spread of the same order as the conditional SEs means a prospective claim would also need refit variation in its design.
5. **These are planning numbers under stated assumptions.** Households are iid from the same population with the same SEX mix and household sizes; the analysis is unweighted; readers are frozen; there is no transport; the normal approximation holds (checked). None of these numbers is "impossible", "proved" or a statement that the release fails a cap: at the plug-in, every binding slack is positive.
