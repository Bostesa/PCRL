# Interpretation correction for the CELL-A pilot

**Date:** 2026-10-02.
**Applies to:** `results/combined_stored_model_pilot_v1/`: the ADVISOR_BRIEF.md and RESEARCH_DECISION.md wording on
the three `education_assessment` pairs and on "generalisation".
**Status:** a correction of interpretation only. Adopted by the coordinator 2026-10-02 after rerunning `notes/method/pilot_numerics.py`, which reproduced `pilot_numerics.json` exactly (timestamps excluded).
- The pilot's numbers, intervals and primary decisions are preserved unchanged.
- The new numbers below are a **versioned diagnostic** (`notes/method/pilot_numerics.json`, produced by
  `notes/method/pilot_numerics.py`, schema `pilot_numerics_v1`). They do not replace anything.
- Every input is hash-pinned in that JSON. No attacker or probe was refit. The only computations are closed-form
  re-solves of the registered G1 least-squares problem, done for numerical evidence.

## What changes in the wording

| Pilot wording | Replace with |
|---|---|
| "Where the linear certificate passes, it holds on unseen rows." (ADVISOR_BRIEF) | "No failure of the registered held-out check was established in this pilot, conditional on these fitted probes and rows." |
| "On this encoder, where the linear check passes it also holds on held-out rows." (RESEARCH_DECISION) | Same replacement. |
| "No case where the check passes but fails to generalise." | "No C2 was established: no pair's held-out G1 had a lower bound above τ." |
| "G1 is strongly negative: the fixed 1e-6 penalty is ill-conditioned on these representations." | "G1 is strongly negative because the fixed-penalty predictor transfers poorly. The float64 solve itself is accurate (see (ii))." |

The pilot statement "the stored 'pass' came from a numerically broken float32 calculation" is **kept**. Its
numerical evidence is now given in (i).

## Four objects that must not be conflated

Pairs: education_assessment × {sex, race, income}. Representation: `rep_p2` (64-d, stored as float32).

### (i) The historical float32 in-sample computation (N0) is numerically invalid on these pairs

On all 15,060 test rows:
- The centred Gram has smallest eigenvalue 0.0022 and condition number 2.6e7.
- The float32 Gram error has spectral norm 1.16, which is 527× that smallest eigenvalue. Float32 therefore carries
  no correct digits in the weak directions.

The decisive evidence: the mixed-precision weights are **not a ridge minimiser**.

| Quantity | sex | race | income |
|---|---|---|---|
| Ridge objective at the mixed-precision W | 8,237 | 5,287 | 6,061 |
| J(0) = SS_tot | 6,620 | 3,744 | 5,582 |
| Weight norm vs the stable solution | 16× | 17× | 3.2× |

An exact ridge solution satisfies J(W\*) ≤ J(0). Its in-sample R² is therefore ≥ 0. A negative raw value
(−0.244, −0.412, −0.086) can only come from an inaccurate solve. It is explained by catastrophic loss of precision in
the float32 Gram.

**Recomputed:**
- float64 normal equations: N0 = 0.00623, 0.00968, 0.00642;
- stable SVD solution: identical to within 4e-15.

These are the pilot's float64 values (0.006–0.010).

### (ii) The fixed-ridge predictor's held-out performance (G1, primary)

Fit rows: attacker_fit (7,571 rows). Score rows: assessment (5,250 rows).

**The solve is accurate.**
- cond(G + λI) = 3.9e9. The float64 forward-error bound is cond·eps ≈ 9e-7.
- Relative normal-equation residuals:
  - float64 `solve`: 1e-13 to 1e-12;
  - stable SVD: ≤ 3e-10;
  - eigh: ≤ 3e-12.
- The saved G1 predictions equal a fresh float64 solve exactly (max |Δ| = 0).
- The SVD solve gives a held-out G1 that differs by < 1e-5:

| Solve | sex | race | income |
|---|---|---|---|
| Stored (primary) | −0.2905 | −0.6044 | −4.4185 |
| Stable SVD | −0.2905 | −0.6044 | −4.4185 |

- An all-float32 solve (residual 1.5e-3 to 5e-3, weights 100 % different) is **not** a valid alternative.

**The negativity is a property of the predictor, not of the arithmetic.**
- λ = 1e-6 is 2.3e-9 of the mean Gram eigenvalue. The predictor is effectively unregularised least squares.
- 21 of the 64 fit-Gram directions lie below 1e-6 × the largest eigenvalue. Those directions carry 99.9 % of ‖W‖².
- In those directions:
  - the top 1 % of rows carry 99 % of the sum of squares, in both roles;
  - assessment rows have a median of 108× (max 9,106×) the fit-row variance.
- The top 1 % of assessment rows account for 24 % / 40 % / 82 % of SS_res.

Descriptively, the same fixed-λ solution restricted to the 43 well-supported directions scores 0.0019 / 0.0064 /
−0.0175. This is not a registered statistic and is not a replacement.

**Primary decisions unchanged:** ESTABLISHED_BELOW for all three pairs. The upper bounds are −0.077, −0.116 and
−0.507 against τ = 0.05. This is a valid decision about the registered statistic. Because the predictor is worse than
the constant prior on held-out rows, it says little about how much linear signal exists.

### (iii) The separately evaluated scale-invariant G2 (exploratory, not a substitute)

G2 is relative-ridge least squares. It uses a 1e-6 relative eigenvalue floor and ρ selected on attacker_val.
Recomputed from the saved predictions:

| G2 | sex | race | income |
|---|---|---|---|
| Point | 0.0035 | 0.0075 | 0.0038 |
| Individual 90 % interval | 0.0007–0.0063 | 0.0050–0.0097 | 0.0021–0.0054 |

These are consistent with a small held-out pooled linear R². They are reported **beside** G1, never in its place.

### (iv) The broader class of linear predictors is not bounded by any of this

Neither G1 nor G2 is a supremum over linear predictors. The pilot's own exploratory outputs show linear recovery on
another scale:

| Linear statistic | sex | race | income |
|---|---|---|---|
| Logistic attacker L, macro AUC | 0.523 | 0.594 | 0.618 |
| G1 predictor scored as AUC | 0.520 | 0.586 | 0.589 |
| Held-out ρ₁² | 0.0018 | 0.0062 | 0.00006 |

A pooled R² below τ can coexist with linear AUC above 0.55.

## Corrected statement for the three pairs

> On the education_assessment pairs, the historical float32 in-sample check was numerically invalid. Its float64 /
> stable recomputation gives 0.006–0.010, below τ. The registered held-out fixed-ridge G1 is strongly negative
> (−0.29, −0.60, −4.42). That value is a correct computation of a near-unregularised predictor that transfers poorly
> through rare-row, near-null directions; it is not a solver failure. The scale-invariant G2 (exploratory) is
> 0.004–0.008. No failure of the registered held-out check was established in this pilot, conditional on these fitted
> probes and rows. Nothing here bounds what other linear predictors recover; a logistic probe reaches macro AUC
> 0.52–0.62.
