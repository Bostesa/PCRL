# Research decision: matched attribute-removal benchmark

**Status:** development evidence, 2026-10-03.
- Six frozen PCRL Round-4 encoders: Adult and HMDA × seeds 0, 1, 2.
- Rows: already-used development rows.
- Defenses: official LEACE (concept-erasure 0.2.4) and Gaussian noise. Both were fitted after the lock was pushed
  (commit 3274fe1).
- Independent replay: see `INDEPENDENT_VERIFICATION.json` and `VALIDATION.md`.

Nothing here is fresh confirmation, a new algorithm, or a population privacy guarantee. A nonlinear attack on LEACE
does not refute LEACE's linear theorem; it shows what that theorem does not cover.

## What ran

| | Planned | Completed | Missing | Aliased |
|---|---|---|---|---|
| Tier 1 (2 cells × 3 seeds × A, B, C, 6σ × 3 release seeds) | 126 | 126 | 0 | 0 |
| Tier 2 E1 (untreated + LEACE-B, 12 more disallowed pairs) | 72 | 72 | 0 | 0 |
| Tier 2 E2 (LEACE-C on those pairs) | 36 | 36 | 0 | 0 |
| Tier 2 E3 (noise on the E1 pairs) | 648 | 201 | 447 (budget stop rule) | 0 |
| **Total** | **882** | **435** | **447** | **0** |

**What was fitted:**
- 60 official LEACE maps: 42 target maps (B) and 18 policy-set maps (C).
- 44,824 attacker, probe, reference and closed-form fits. That count covers every hyper-parameter grid point and
  attacker-seed refit.
- Science CPU: 5.01 CPU-h of the 8 CPU-h ceiling, on the laptop only.

**E3 stopped by the frozen rule.** It stopped before `adult__s2__employment_analysis__marital_status__D_sigma0.5_rs0`,
in the fixed order. At that point the run had spent 15,743 CPU-s, the next unit projected 27 CPU-s, and the inference
reserve was 13,080 CPU-s, against a budget of 28,800 CPU-s.
- E3 is complete for Adult income/race, employment/race and employment/age_group.
- It is partial for employment/marital_status: seeds 0–1 at every σ, seed 2 only at σ = 0.25.
- It was not run for the other 8 pairs.
- The calibration predicted this cut before any fit (E3 projected at 8.07 CPU-h).

**σ\*** was chosen on attacker_val only: Adult 2.0, HMDA 4.0. Neither fell back to the default.

## The five questions, Tier 1 primary family

The primary family has 24 endpoints, with Bonferroni simultaneous one-sided bounds at α = 0.05/24 and B = 20,000.
All values below are means over encoder × release × attacker seeds.

### 1. Does each method pass its own check on its fitting rows?

**Untreated PCRL (historical in-sample one-hot ridge R², τ = 0.05)** reproduces the stored value on all 6 encoders
(|Δ| ≤ 3.4e-6):

| Cell | s0 | s1 | s2 | Result |
|---|---|---|---|---|
| Adult income/sex | 0.035 | 0.034 | 0.010 | Passes on all 3 |
| HMDA underwriting/race | 0.044 | 0.090 | 0.081 | **Fails on s1 and s2: C1 before any defense** |

**All 14 pairs.** Across the 42 untreated pair × seed checks, every value reproduces the stored one (|Δ| ≤ 4.3e-6).
**12 of 42 fail PCRL's own stored check:**
- Adult income/race on all 3 seeds (0.058 / 0.200 / 0.362);
- Adult education/sex on s1;
- HMDA fair_lending/sex on s0 and s2;
- HMDA pricing/race on s1 and s2;
- HMDA underwriting/ethnicity on s1;
- HMDA underwriting/race on s1 and s2.

Adult s0 education × 3 again carries the numerically invalid float32 "pass": the in-sample R² is negative, and the
float64 values are 0.006–0.010. See INTERPRETATION_CORRECTION.md.

**Official LEACE (implementation bound: whitened residual ≤ svd_tol = 0.01 on defense_fit)** holds for all 60
maps. Two descriptive points:
- 6 of the 12 Tier-1 maps truncate 1–2 singular values within tolerance. Their cross-covariance is small but not
  exactly zero (for example HMDA s0 B: residual 0.0088). This is reported descriptively and is not C1.
- B and C never aliased.

### 2. Does the matched linear quantity generalise to held-out rows?

The matched quantity is G1: the fixed-penalty ridge fitted on attacker_fit and scored on assessment.

| Cell | G1 | Bounds | Decision |
|---|---|---|---|
| Adult income/sex | −0.18 | [−0.96, 0.029] | **Established below τ**. The negative mean comes from the rank-deficient seed 1 (G1 = −0.57); the scale-invariant G2 is 0.033. |
| HMDA underwriting/race | 0.063 | [0.042, 0.077] | **Unresolved** |

**No C2 was established.**

Under LEACE the held-out top canonical correlation ρ₁² drops:
- Adult: from 0.025 to 0.0006;
- HMDA: from 0.21 to 0.0001.

The linear guarantee transfers to unseen rows.

For HMDA, the untreated pooled R² of 0.06 hides a class contrast with ρ₁² = 0.21. This is the same pattern the pilot
found for Adult income/race.

### 3. What do attackers outside the guarantee's scope recover?

The statistic is the macro AUC of the nonlinear attacker on the representation, against a bar of 0.55.

| | A untreated | B LEACE target | C LEACE policy set | D at σ\* |
|---|---|---|---|---|
| Adult income/sex | 0.815 ↑ | 0.817 ↑ | 0.807 ↑ | 0.528 ↓ |
| HMDA underwriting/race | 0.870 ↑ | 0.863 ↑ | 0.863 ↑ | 0.513 ↓ |

↑ = established above 0.55; ↓ = established below.

LEACE passes its own check on every seed, yet nonlinear recovery barely moves (90 % intervals):

| Comparison | Adult income/sex | HMDA underwriting/race |
|---|---|---|
| B − A | +0.002 [0.000, 0.004] | −0.007 [−0.008, −0.006] |
| C − A | −0.009 | −0.007 |

So B and C are **C3 on both cells**: the check passes, and recovery happens outside its scope.
- A logistic-regression attacker loses −0.03 to −0.10 AUC under LEACE. Nonlinear attackers lose essentially nothing.
- The same holds across all 12 Tier-2 pairs. B−A nonlinear AUC ranges from −0.05 (HMDA pricing/race) to +0.001. Every
  B arm stays between 0.67 and 0.89 (nonlinear AUC).
- HMDA fair_lending representations take only about 4.7K–6.7K distinct values on seeds 0–1. An affine eraser cannot
  change a lookup on those values, so for fair_lending/sex the nonlinear AUC is identical across A, B and C on seed 0, and within 0.0002 on seed 1.

### 4. Do outputs preserve recovery when the representation is protected?

**Yes.** The noise arm at σ\* pushes representation-only recovery below 0.55. With clean task outputs also released,
recovery stays well above it:
- Adult 0.778 [0.764, 0.793];
- HMDA 0.758 [0.749, 0.766].

Both are established above 0.55. Once the noise has destroyed the representation, the selected attacker is simply the
outputs-only attacker.

**Outputs leak beyond the true label.** Outputs-only nonlinear AUC minus label-only AUC:
- Adult +0.177 [0.166, 0.188];
- HMDA +0.236 [0.227, 0.245].

The same comparison is +0.02 to +0.29 on every Tier-2 pair.

### 5. How much task capability is kept?

The measure is the U2 refitted probe accuracy difference against A, with a non-inferiority margin of −1 pp.

| Arm | Adult income/sex | HMDA underwriting/race |
|---|---|---|
| B | −0.04 pp, **non-inferior** | −0.13 pp, **non-inferior** |
| C | −2.1 pp, **inferior** | +0.05 pp, **non-inferior** |
| D at σ\* | −4.9 pp, **inferior** | −1.7 pp, **inferior** |

**What the D losses mean against the constant predictor:**
- Adult keeps about 40 % of the lift: U2 drops from 0.831 to 0.782, against a constant of 0.749.
- HMDA keeps 0 %: U2 at σ\* = 4 is 0.886, which equals the constant predictor.

The frozen head (U1) has lost all of its lift over the constant predictor by σ = 0.5 for Adult (0.751 against 0.749) and by σ = 2 for HMDA (0.863 against 0.886). U1 is outside LEACE's scope and is
reported as a compatibility diagnostic only.

## Did the pilot pattern replicate?

**Yes, on 6 encoders, 2 datasets and 14 pairs**, and now with a matched official-LEACE comparison:
- no C2;
- nonlinear recovery outside the linear check's scope on every pair;
- outputs leak beyond the label;
- noise protects the representation surface only, at a large utility cost.

**What is new:**
1. **Matched LEACE.** LEACE satisfies its own guarantee and transfers it to held-out rows (ρ₁² ≈ 0). It costs almost
   no probe utility for the target concept. It leaves nonlinear and output-based recovery essentially unchanged.
2. **Encoder seed matters a lot.** Adult income/sex untreated nonlinear AUC is 0.67 / 0.91 / 0.87 across seeds 0–2.
   - The pilot's single encoder (seed 0) was the least leaky.
   - Seed 0 here gives 0.669, against 0.684 in the pilot: different attacker seeds are averaged here.
3. **PCRL's own stored check fails on 12 of 42 untreated pair × seed combinations.** This includes HMDA
   underwriting/race on 2 of 3 seeds (C1). The pilot's single encoder showed only 1 such failure out of 8.

**Strongest favourable comparison.** LEACE-B is non-inferior on probe utility in both cells, and its linear guarantee
generalises (held-out ρ₁² 0.0006 / 0.0001).

**Strongest adverse comparison.** Neither LEACE arm reduces nonlinear recovery by more than 0.01 AUC on the Tier-1
cells. Noise at σ\* stops representation recovery, but outputs keep recovery at 0.76–0.78 and HMDA loses all
task lift.

## Limits

- **Data and scope.** Development data, already used. Two datasets with frozen encoders; no retraining.
- **Exposure.** defense_fit = the encoders' own training rows. 17 Adult and 42 HMDA assessment rows duplicate training
  records; this exposure is unresolved and disclosed.
- **Rank-deficient defense_fit covariance** (Adult s1 29/64; HMDA s0 and s2 16/64). Official LEACE is the identity
  outside the fit support. Out-of-support norms are reported as a transfer limitation.
- **Unsupported classes.** HMDA race classes 3–4 and Adult race classes 0 and 3 are not estimable; worst-class
  statistics cover the supported classes only.
- **E3 coverage.** Partial: 3 Adult pairs complete, marital_status partial, 8 pairs not run. Marital_status group
  values at σ ≥ 0.5 average seeds 0–1 only. Its paired "plus minus outputs" difference at those σ (+0.014) is not
  comparable to the 3-seed references.
- **Inference.** Intervals are conditional on the fitted predictors (cluster bootstrap over assessment units). The
  percentile bounds are asymptotic.

## Recommendation

**A larger benchmark is worth running only for a different question.** More pairs, seeds or noise levels on these
encoders would not change the decision: all 14 pairs already show the same pattern.

The useful next steps are:
1. **Erasure that acts on outputs or nonlinear statistics**, with the same matched protocol: for example, LEACE on
   the concatenated rep + logits, or a kernel / iterative eraser.
2. **Fresh, unopened confirmation rows**, but only after the protocol is frozen.

The NeurIPS Round-5/7 headline checkpoints remain inventoried and unrun.
