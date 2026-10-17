# Statistical review of the confirmation-feasibility stage (PROTOCOL.md §10)

Statistical/readiness reviewer, 2026-10-17, after the coordinator's logged evaluation (event 22:27:16Z).

**Inputs I used:**
- `report/DEV_REPORT_RESULTS.json` (published points, SEs, refit spreads);
- `CONFIRMATION_FEASIBILITY_RESULTS.json` and `ffv/feasibility.py`;
- the private `ENDPOINT_DRAWS.npz`: 2,048 × 140 replicate draws, SHA-256 `6f4c25bf…`, which matches the hash recorded in the report.

**Not done:** no label was read, DEV_REPORT truth was not re-read, and `HOUSEHOLD_CONTRIBUTIONS.npz` was not opened.

**My own check** is `code/sr_feas_check.py`, run as queue unit `SR_FEAS_CHECK_V1` (8.5 CPU-s), with output in `SR_FEAS_CHECK_V1.json`. It uses methods different from the implementation:
- its own enumeration of the conditions;
- analytic sample sizes for single clauses;
- **nonparametric rescaling** of the 2,048 centred replicate deviations (no normal assumption);
- an independent MVN built from an SVD of the centred draws (seed 20261017, 40,000 draws, a finer 400-point grid);
- orthant limits computed from the correlation matrix;
- the assurance limit from posterior draws of the true means.

## 1. Does the implementation match the frozen specification?

| Item | Verdict |
|---|---|
| D1 includes claim B's 8 recipient clause bounds | **Matches.** 38 conditions: 28 utility/guard, 6 recipient, 4 benefit. My own enumeration gives the same set. The 2 deterministic passes are `occupation_worsening_vs_H_LJ\|s0/s1` (SE = 0). |
| D2 / D3 rules | Match: c1 bound for utility in D2; benefit bound > 0.020 in D3. |
| Binding keys | Match: Ucal income Brier s0 and s1, plus income accuracy_loss s1 under both U0 and Ucal (perfectly collinear, they move together). |
| Grid | Matches: 100 log-spaced points from 3,955 to 10^8, a step of about 10.8%. |
| Monte Carlo | Matches: 40,000 draws, seed 2026101803, common random numbers across n, MC SE ≤ 0.0025. |
| Σ factorization | Matches: eigendecomposition with negative eigenvalues clipped. The published SEs equal the SDs of the draws exactly (max difference 0.0). |
| Slack grid / assurance / B-only sensitivity | Present and as specified. |
| Governing scorecard | The less favourable n80 governs, as specified. |
| **Limits** (`limit_n_to_infinity`) | **Wrong at boundary scenarios** (E1). |
| **τ wording and scope** | **Not as specified or justified** (E2, E3). |
| "Full power curve reported" | **Not met.** Only 6 points are saved, and they are linearly interpolated (E4). |

### Findings

**E1. The boundary limits are wrong.**
- `limit()` multiplies per-clause limits as if the clauses were independent. For `S3_binding_at_cap|D1` and `S5_binding_true_slack_0.0|D1` it reports **0.0625** (= 0.5⁴).
- The binding clauses are positively correlated: Brier s0–s1 ρ = 0.77, Brier with accuracy ρ = 0.36–0.48, and the two accuracy keys ρ = 1.
- The true limit is the orthant probability, **0.265**. The implementation's own Monte Carlo agrees: 0.266–0.268 at n = 10^6 to 10^8, on both scorecards.
- The D2 at-cap limit is about 3×10^−4, not 0.
- The `LIMIT_BELOW_TARGET` status is still correct, but the numbers are not.
- Fix: replace them with the orthant probability, or report the Monte Carlo value at 10^8 labelled as such. Otherwise label the product "independence lower bound (Slepian's inequality; all correlations here are positive)".

**E2. "Measured reader refit, a lower bound" is not supported.**
- All **24 of 24** household-bootstrap refits read lower on DEV_REPORT than the full-sample fit. Mean shifts are −0.006 to −0.033 AUC.
- So the auxiliary-sample SD, computed with the full-sample fit included, mixes a systematic shift with variance. The shift comes from duplicated households (about 63% unique) and/or from winner selection.
- At benefit level, τ_r (quadrature of the seed and auxiliary-sample components) against the auxiliary SD from resamples only:

  | Cell | τ_r | Auxiliary SD (resamples only) |
  |---|---|---|
  | S0 NEURAL_RAW | 0.0134 | 0.0103 |
  | S0 H_LJ | 0.0067 | 0.0051 |
  | S1 NEURAL_RAW | 0.0036 | 0.0017 |
  | S1 H_LJ | 0.0095 | 0.0062 |

- Omitted sources (re-selection, a new population) push the other way. The net direction of bias is unknown, so neither "lower bound" nor "upper bound" is justified.
- The seed component (refits on the full sample, seed changes only) is a clean measurement.
- **Descriptive by-product:** with bootstrap-trained readers, the benefit shrank in 3 of 4 seed × reference cells (mean shifts −0.015, −0.006 and −0.016, versus +0.007). This is further evidence that the benefit is reader-relative. It is not an endpoint.

**E3. Scope and naming of τ.**
- τ is added to the **4 pair-benefit endpoints only**. The docstring says "AUC-type endpoints", but the 6 recipient AUC clauses get no added component, because their single-view readers were not refitted.
- The random effects ξ are independent across the 4 benefits. A shared FINALIST effect across references would add positive dependence and raise joint power, so this choice is conservative.
- τ_r comes from refits of the ENHANCED winners but is also applied to the ORIGINAL scorecard.
- The scenario `S4_…release_retraining_HYPOTHETICAL` uses √(τ_r² + τ_h²): measured reader refit plus hypothetical release retraining. Its name says release retraining only.
- I verified the arithmetic: τ_h = |b(s0) − b(s1)|/√2, and the S4 limits, for example ENHANCED D3 = 0.3153.

**E4 (minor).**
- Save the full 100-point power curves; the specification requires them.
- Label `power_at` as linearly interpolated.
- Break ties for the governing scorecard by the lower power at 10^8 when both n80 values are missing. Today ties default to ENHANCED: harmless here, but arbitrary.

**E5 (minor).** The assurance limit is reported as NaN. It is computable:

| | D1 / D2 | D3 |
|---|---|---|
| ENHANCED | 0.103 | 0.053 |
| ORIGINAL | 0.164 | 0.066 |

The implementation's Monte Carlo at 10^8 (0.101 / 0.161 and 0.050 / 0.061) converges to these values.

## 2. My recomputed numbers

### Single clauses, analytic

Here z₀.₈ = 0.8416 and c1 = 2.3949991:
- D1 point gate: n = 3955·(z₀.₈·SE/slack)²;
- D2 bound: n = 3955·((z₀.₈ + c1)·SE/slack)².

The utility clauses do not depend on the readers, so these values hold for both scorecards.

| Clause | Slack | SE | Slack/SE | D1 n80 | D2 n80 | Posterior P(true slack > 0) |
|---|---|---|---|---|---|---|
| `UTILITY\|income\|Ucal\|brier_excess\|s1` | 5.62e−5 | 0.002047 | 0.027 | 3,711,082 | 54,884,621 | 0.511 |
| `UTILITY\|income\|Ucal\|brier_excess\|s0` | 8.11e−5 | 0.001925 | 0.042 | 1,579,407 | 23,358,453 | 0.517 |
| `UTILITY\|income\|{U0,Ucal}\|accuracy_loss\|s1` | 4.09e−4 | 0.003719 | 0.110 | 231,814 | 3,428,392 | 0.544 |
| ENHANCED `COMPETITIVE\|income_worsening_vs_H_LJ\|s1` (point 0.0076; cap 0.010) | 0.0024 | 0.006648 | 0.36 | bound n80 317,958 (D1 and D2) | — | 0.641 |
| ENHANCED `pair_benefit_vs_NEURAL_RAW\|s1` at 0.020 (D3) | 0.00175 | 0.006194 | 0.28 | D3 bound n80 522,255 | — | — |
| ORIGINAL `pair_benefit_vs_NEURAL_RAW\|s1` at 0.020 | −0.00011 | 0.005833 | — | unreachable at plug-in | — | — |

### Joint check: S1 plug-in

| Quantity | Implementation (MVN, 100-point grid) | Reviewer: nonparametric rescaling | Reviewer: own MVN (SVD) |
|---|---|---|---|
| ENHANCED D1 n80 / n90 | 4,632,356 / 9,486,693 | 4,392,678 / 8,947,497 | 4,392,678 / 9,177,751 |
| ENHANCED D2 n80 / n90 | 59,928,535 / 73,549,580 | 55,744,423 / 71,870,163 | 55,744,423 / 71,870,163 |
| ENHANCED D1 power at 10^6 / 10^7 / 10^8 | 0.606 / 0.914 / 1.000 | 0.602 / 0.916 / 1.000 | 0.602 / 0.912 / 1.000 |
| ENHANCED D2 power at 10^7 / 10^8 | 0.142 / 0.977 | 0.143 / 0.977 | 0.141 / 0.976 |
| D1 power at n = 3,955, ENHANCED / ORIGINAL | 0.003 / 0.115 | 0.003 / 0.104 | 0.003 / 0.112 |
| At-cap D1 / D2 limit | 0.0625 / 0.0000 (independence product) | — | **0.265 / 0.0003** (orthant) |

**Agreement.**
- My n80 values lie inside the implementation's adjacent-grid brackets: (4.18M, 4.63M] for D1 and (54.1M, 59.9M] for D2.
- Power values agree within Monte Carlo error. Nonparametric rescaling has an SE of about 0.009 with 2,048 draws, so normality does not drive the result.
- The joint D1 n80 exceeds the worst single clause (4.4M against 3.7M) because both correlated Brier clauses must pass. D2 is about the worst single clause.

## 3. Statements that would overreach, and the wording to use

1. **The plug-in n80 is not "the required sample size".**
   - The D1 value (≈4.4–4.6M households) and the D2 value (≈56–60M) hold only if the true utility slack equals its development estimate, which is 0.03–0.11 SE from the cap. The data cannot tell that slack from zero or from negative.
   - Required wording: *"Required n is not identified by these data. Over the registered slack grid, D1 needs about 4×10^4 (ORIGINAL) or 3.2×10^5 (ENHANCED, limited by its recipient clause) households when the true slack is ≥ 0.002. It needs about 4.6×10^6 at the plug-in slack. If the true slack is ≤ 0, it does not reach 80% at any n."*
   - Do not call D2 at plug-in "achievable". It exceeds the number of households in Florida, which is of order 10^7.
2. **Limits and `LIMIT_BELOW_TARGET` are conditional on the scenario.** Write *"not reliably establishable as n grows under this scenario's assumed true values"*. Never write "impossible", "proved infeasible" or "the release fails the cap". At plug-in every binding slack is positive.
3. **The 0.020 target.** The ORIGINAL D3 limit is 0 because the seed-1 benefit against NEURAL_RAW is 0.0199. Under the ENHANCED plug-in, all four benefits exceed 0.020 (minimum 0.0217) and D3 reaches 80% at about 56–60M. D3 assurance is 0.05–0.07.
   - Required wording: *"not reliably establishable at 0.020 under the ORIGINAL plug-in; under the ENHANCED plug-in it would need about 6×10^7 households; under the posterior heuristic it has about 5–7% probability."* Never write "the 0.020 target is impossible".
4. **Assurance is not a probability that the release complies.** It is a flat-prior normal predictive heuristic, centred on development estimates obtained after favourable selection. It is not proof, and not a frequentist statement.
   - Required wording: *"Averaging over development uncertainty in the true effects, the joint success probability stays at or below about 0.10 (ENHANCED) or 0.16 (ORIGINAL) at every n. Several utility clauses (posterior probability of feasibility 0.51–0.79) carry most of that."*
5. **The HYPOTHETICAL τ must stay labelled hypothetical.** It is never "release-training variance". The S4 scenario combines the measured reader-refit term with it, so any number using τ_h carries the HYPOTHETICAL label.
6. **The measured τ_r is not a lower bound (E2).**
   - Required wording: *"rough measured refit spread: 4 seed refits and 4 household-bootstrap refits per winner, each with at most 4 degrees of freedom. The bootstrap refits read systematically lower (24 of 24), so the auxiliary-sample term mixes a resampling shift with variance. The direction of the net bias is unknown."*
   - The same applies to the protocol's §7.6 phrase "partial, lower-bound".
7. **Same-size replication.** Under plug-in, a replication with 3,955 households would pass D1 with probability of only about 0.10–0.11 (ORIGINAL) or about 0.003 (ENHANCED). Do not describe the source pass as "expected to replicate". This is a descriptive plug-in probability, not a test.
8. **Scope.** Every curve is "against these frozen readers and frozen winners, on households drawn iid from the Florida ACS 2019 development population, analysed unweighted".
   - They do not apply if a prospective study refits readers or re-selects winners (that endpoint is a maximum over candidates), transports to another state or year, or uses design weights.
   - The two seeds share people and are not replications.
9. **Governing scorecard.** For S6 and S3, "governing = ENHANCED" comes from a tie in missing n80 values (E4). It is not a substantive comparison.

## 4. Must-fix before the closeout text

Version the feasibility output (for example `..._V2.json`, keeping V1 and disclosing the change). Recomputing from the existing draws needs no label access.

1. Replace the boundary limits with orthant probabilities: 0.265 for D1 and ≈3×10^−4 for D2 at the cap (E1).
2. Rename `tau_reader_refit_measured_lower_bound` to a "rough" name and drop "lower bound" from the docstring and §7.6 (E2).
3. Fix the τ scope and naming: benefits only, ξ independent and conservative, the combined S4 scenario, τ_r from ENHANCED winners applied to both scorecards (E3).
4. Save the full curves and mark `power_at` as interpolated (E4).
5. Report the assurance limits (E5).
6. Use the wording in §3 wherever these numbers are summarized. That includes PROTOCOL_DEVIATIONS if the V1 file is kept.

**Conclusion.** None of these defects changes any status label or the qualitative conclusion: D2 at plug-in sits at about 6×10^7 households, D1 at about 4.5×10^6, the required n depends almost entirely on the unknown true utility slack, and assurance stays at or below about 0.16 at every n.
