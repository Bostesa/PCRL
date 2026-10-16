# Designer 1 cross-critique of designer 2's RCCS

- **Written:** 2026-10-16, after the coordinator's reveal.
- **Check:** unit `D1_XC_DRIVER_V1` (script `d1_xc_driver_cancellation.py`, output `D1_XC_DRIVER_RESULTS.json`, both in this directory). Data-free, exact numerical integration.
- `PROPOSAL.md` is unchanged.

## 1. Strongest objection: the protection is cancellation, and cancellation is exact only for location-family laws

**How RCCS protects.** RCCS does not remove SEX signal from the occupation output. It *adds* a driver built from the residual SEX propensity, A·ψ(r(x)), so that the linear and quadratic partial covariance with SEX, given income, sums to zero.

**When that is genuine, and when it is not.**
- For every reader, this is genuine only if the class-conditional laws of the driver and of the task coordinate differ by SEX in location alone. For a calibrated score, that means the binormal, equal-variance case.
- Otherwise the deployed release, which is injective and an explicit function of π̂, keeps signal that the ridge criterion cannot see but the MLP and HGB readers in the slate can.
- The exact prototype does not test the deployed map. Its COND arm *pools* cells using c as a coin. The real release softmax(z2 + Aψ) never pools, and designer 2's own INJ arm attains AUC*(S;X).

**My check.** One coordinate h has SEX signal, with h-alone AUC 0.650 (designer 2's occupation-given-income level). The shift h + k·u is set so that the class-conditional means **and variances** are equal, so a linear and a quadratic criterion both read exactly 0. Exact Bayes AUC after the shift:

| Driver / task law | Driver AUC alone | k | Bayes AUC after |
|---|---|---|---|
| Gaussian driver (control) | 0.64 / 0.57 | −1.1 / −2.2 | **0.500** / **0.500** |
| Binary category, P(u = 1 \| S) = 0.5 ± 0.05 | 0.55 | −5.5 | **0.654** |
| Binary category, ± 0.10 | 0.60 | −2.7 | **0.596** |
| Binary category, ± 0.20 | 0.70 | −1.4 | 0.531 |
| Three-level category | 0.60 | −1.8 | 0.553 |
| Gaussian driver, class-mixture task coordinate (h alone 0.584) | 0.64 | −0.8 | 0.526 |

- **What it shows.** Lumpy drivers retain from 21% to over 100% of the cancelled above-chance signal. That is the natural case for a tree propensity over COW, MAR, POBP and SCHL, and for a 5-class occupation simplex.
- **Weak drivers retain the most.** Designer 2's typical V/Var(π) ≈ 0.1 implies a weak driver.
- **The gate cannot protect the claim.** Its tolerance (OUT_HGB − OUT_LR ≤ 0.03) is as large as the predicted benefit (≈ 0.03).
- **Secondary point.** The "5–7.6× amplification" comes from ratios of transformed selected-reader AUCs. Income d′² has about 22% relative SE per seed, and the four protected services share models and people. Designer 2's own Stage-0 R² precondition mitigates this.

## 2. Distinct or overlapping?

**Shared ingredient:** an explicit SEX-propensity notion fitted on MODEL_FIT, acting along π-aligned directions. Both proposals make fitted-reader claims only in the general case.

**Distinct operation and regime.** Each premise is the other's null case.
- In all of designer 2's laws, S, Y1 and Y2 are conditionally independent given X (the table is ps·p1·p2). So q^F = q^M, Δ ≡ 0, and **MMP is the identity**.
- In my laws F and R, η_i is injective in π given the task field. So the outputs already determine π, **RCCS's residual driver vanishes**, and RCCS reduces to post-processing (designer 2's failure mode 2).

| | MMP | RCCS |
|---|---|---|
| Operation | Subtracts the *existing* π-term from both outputs | *Adds* a residual-propensity counter-term to occupation only |
| Propensity in the release | None | Explicit function of π̂ |
| Where removal is reader-independent | Along merged directions | Only in location-family laws |

A combination (MMP, then a conditional counter-shift on the remaining within-SEX channel) is conceivable but would inherit RCCS's shape fragility. I do not propose it.

## 3. Does it change my assessment of MMP?

**Yes, in two ways. `PROPOSAL.md` is not edited.**

1. **The amplification identity sharpens my failure mode 1.** Designer 2's Schur-complement identity is exact, and I accept it. After MMP, any SEX covariance of income that the occupation output does not explain is amplified by 1/(1 − R²) in the pair, which is about 5.6× if R² ≈ 0.82.
   - MMP removes only income's direct-gap part Δ_1.
   - To first order, the partial covariance shrinks only if income's direct within-profile gap exceeds the income difference mediated by occupation (Var(π)-weighted |g_1| > |b·g_2|). Otherwise MMP can enlarge it, which is the LOCAL_TREE pattern.
   - Falsifier F2 is therefore more likely than I stated.
2. **Designer 2's laws make my premise explicit.** MMP needs direct SEX effects on the tasks given the six fields. Two cheap pre-gates would run before my F1:
   - the held-out NLL gain from adding SEX to the teacher on MODEL_FIT, which bounds whether Δ exists and what it costs;
   - designer 2's out-of-fold decomposition W_pair = W_occ + partial term, for MMP against its twin.

**Revised subjective forecasts:**
- plain point superiority: 0.30 → **0.25**;
- material margin: stays **≤ 0.07**;
- utility-feasibility forecast: unchanged.
