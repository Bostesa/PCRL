# Guarantee cards: matched removal benchmark v1

**Date:** 2026-10-02. Frozen before any fit and unchanged after the run; results are in RESEARCH_DECISION.md.
**Purpose:** one card per arm (A, B/C, D) and one per reference. Every card is a scope statement, not a verdict.

**Native checks:** each arm's native check is the check named on its own card. A method's criterion is never
applied to another method.

**What each card states:**
- the bounded quantity and its function class;
- the law or rows it holds on, and its assumptions;
- the release surface and the numerical tolerance;
- what falls outside its scope.

**Pins:**
- Official LEACE: PyPI `concept-erasure==0.2.4` = upstream `EleutherAI/concept-erasure` tag v0.2.4, commit
  `9b18b3d5c73f552798212c51d6533d649fa434cd`.
  - Installed `.py` tree sha256 `fffac29d…68597` is recomputed and asserted by `stored_model_eval/defenses.py`.
  - The installed tree is byte-identical to the tag (`diff -r`).
  - The upstream tests at the tag (`tests/test_leace.py` sha256 `b9d126f4…0dc1`, `tests/test_shrinkage.py`) pass
    against the install, 27/27 (`notes/method/upstream_v0.2.4_test_run.txt`).
- Paper: Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff, Biderman, "LEACE: Perfect linear concept erasure in
  closed form", NeurIPS 2023, arXiv:2306.03819. Theorem numbers below are those of the arXiv PDF (v4).
- PCRL native check: PCRL b96c412 `pcrl/purposes/verification.py:89-103` (LinearComplianceCertificate), via
  `eval_round4_dominant_axis.py:158-172`.

**Outcome labels:** these follow the pilot.

| Label | Meaning |
|---|---|
| C1 | fails its own native check on its own rows |
| C2 | the native quantity does not generalise under the registered held-out protocol |
| C3 | recovery outside scope |
| C4 | a population guarantee is contradicted under its verified assumptions |
| C5 | not estimable |

---

## Card A: untreated release with PCRL's linear R² check (Round-4 encoders)

**Release:** h_p, the frozen purpose representation. No transformation.

**Native quantity:** in-sample pooled one-hot ridge R² of the attribute on h_p.
- R² = 1 − Σ‖Z_c − H_cW‖² / Σ‖Z_c‖², with W = (H_cᵀH_c + 10⁻⁶·I)⁻¹H_cᵀZ_c.
- The penalty is absolute, on the unnormalised centred Gram.
- Centring is per variable; the value is clamped at 0.
- Pass is R² ≤ τ = 0.05.

**What it is:** a measurement on given rows. It is not a theorem, and it bounds nothing outside those rows.

**Function class:** the single ridge-fitted affine predictor, under squared loss on the one-hot indicator. It is not
a supremum over linear predictors.
- The pooled R² can sit below τ while a class contrast is linearly recoverable: in the pilot, income/race had
  ρ₁² = 0.14 against a pooled 0.05.
- It can also sit below τ while a logistic probe's macro AUC is above 0.55.

**Rows:** fit = score = the PCRL test split (historical).

**Native check in this benchmark:**
- Reproduce N0 per seed against `dominant_axis_audit.json`.
- Report both the historical mixed-precision value and the float64 value.

**Numerical tolerance:**
- The historical arithmetic computes the Gram in float32.
- Where the Gram's smallest eigenvalue is below the float32 error, the historical value is invalid. The pilot
  education pairs are the example: the float32 Gram error is 527× λ_min, and the mixed-precision weights are not even a
  ridge minimiser (`notes/method/pilot_numerics.json`).
- float64 (validated by a stable SVD solve) is the valid computation. The historical value is preserved and labelled.
- λ = 10⁻⁶ is not scale-normalised. On near-singular representations it is effectively unregularised OLS.

**Held-out measurements** (measured quantities, not guarantees):
- G1 fixed-ridge: fit on attacker_fit, scored on assessment, unclamped.
- G2 scale-invariant.
- ρ₁².

G1 can be strongly negative for a correct solve. A negative G1 says that this predictor transfers poorly; it does not
bound linear recoverability (see INTERPRETATION_CORRECTION).

**Outside scope:**
- held-out rows;
- other linear predictors and losses (0-1, log-loss, AUC);
- class contrasts;
- nonlinear predictors;
- task outputs;
- the concatenation of the representation with outputs.

**Categories:** C1 (N0 > τ); C2 (G1 lower bound > τ; not assigned when C1 already holds); C3; C5.

---

## Card B / C: official LEACE (concept-erasure 0.2.4, default settings)

**Release:** r(h) = h − (h − μ̂)(I − P̂)ᵀ.
- (P̂, μ̂) are fitted once on defense_fit rows by `LeaceFitter` with the package defaults: `method="leace"`,
  `affine=True`, `shrinkage=True`, `svd_tol=0.01`, `constrain_cov_trace=True`. Fitting is in float64.
- The map is then applied unchanged to every other row (`LeaceEraser.__call__`):
  - it uses the original fitting mean;
  - it never refits or recentres;
  - it uses no concept labels at inference.

**Concept Z:**

| Arm | Z |
|---|---|
| **B (target scope)** | the one-hot of the target attribute: Adult `sex` (2), HMDA `race` (5) |
| **C (policy scope)** | the concatenated **marginal** one-hots of the purpose's disallowed set: Adult [race(5), sex(2)]; HMDA [race(5), ethnicity(2)] |

B and C are aliases only if their fitted maps agree, max |ΔP| and max |Δμ| ≤ 10⁻¹⁰ (`alias_test`).

**What the paper proves.** The setting is X with finite first moment, Z one-hot with every P(Z = j) > 0, and a
predictor η(x) = b + Wx.

*Thm 3.1, Thm 3.3, Thm 3.4, Thm C.2, Def. 2.3:*
1. X linearly guards Z. That is, for every loss that is nonnegative and convex in the prediction, no affine predictor
   achieves lower expected loss than the best constant.
2. Every class-conditional mean E[X | Z = j] equals E[X].
3. Σ_XZ = 0.
4. Every affine predictor b + WX has equal conditional means across classes ("statistical parity" in the sense of
   Def. C.1: equal means, not equal acceptance rates).

The paper shows 1, 2 and 3 are equivalent. The direction 1 → 2 is proved through cross-entropy (Lemma 3.2, Thm 3.3).
Statement 4 is equivalent to 2 (Thm C.2).

*Thm 4.1:* the affine r(x) = Px + b yields a linearly guarded r(X) iff colsp(Σ_XZ) ⊆ ker P.

*Thms 4.2 and 4.3:* P\* = I − W⁺P_{WΣ_XZ}W with W = (Σ_XX^{1/2})⁺, and b\* = E[X] − P\*E[X].
- This pair minimises E‖PX + b − X‖²_M for **every** psd M, subject to Cov(PX + b, Z) = 0.
- It assumes finite second moments.

**Law the statement applies to:** the law whose moments are plugged in. Here that is the **empirical distribution of
the defense_fit rows**.

**How the default implementation departs from the exact theorem** (read from the installed source):
- **Shrinkage.** `shrinkage=True` replaces Σ_XX by α·S_n + β·tr(S_n)/d·I. It is full rank whenever tr(S_n) > 0.
  - Σ_XZ is the Bessel-corrected sample cross-covariance.
  - P̂Σ_XZ = 0 still holds exactly when nothing is truncated, so the guardedness of the empirical law holds.
  - The minimal-change optimality of Thm 4.2 then holds for the shrunk covariance, not the sample one. The upstream
    test asserts exactly this.
- **`svd_tol=0.01`.** Singular values of the whitened cross-covariance W Σ_XZ at or below 0.01 are **not erased**.
  - The package documents that this "may leave trace correlations intact".
  - The guarantee then weakens to an implementation bound: ‖W·Cov(r(X), Z)‖₂ ≤ 0.01 on the fit rows.
  - Structural zeros, where the marginal one-hot columns are collinear, are not a loss.
- **`constrain_cov_trace`.** It fires only if tr(P S Pᵀ) > tr(S). With the same S used for whitening and for the trace,
  this cannot happen in exact arithmetic. Whether it fired is recorded; if it fires, exact erasure is lost.
- **Covariance support.** Σ_XZ lies in range(S), so P̂ is the identity on range(S)^⊥. A scored row's component outside
  the defense_fit covariance support passes through unchanged, together with any attribute signal it carries.

**Native check (fit rows only):** `LeaceMap.native_check`. It has two levels, both reported.

| Level | Status | Condition |
|---|---|---|
| (a) exact LEACE condition | `WITHIN_TOLERANCE` | max_ij \|Cov(r(H)_j, Z_i)\| / (sd(H_j) sd(Z_i)) ≤ 10⁻⁶, and each block's fit-row affine OLS R² ≤ 10⁻⁶ |
| (b) implementation bound | `implementation_bound_holds` | whitened residual spectral norm ≤ svd_tol |

- When (a) fails, the status names the cause, for example `OUTSIDE_TOLERANCE_SVD_TOL_TRUNCATION`. That is a property
  of the official default, not a wrapper error.
- The per-block results for arm C are reported for each marginal.

**Held-out measurements** (all measured, none guaranteed):
- the held-out cross-covariance norm (`crosscov_stats` on scored rows);
- G1, G2 and ρ₁² on erased attacker_fit → assessment.

These measure finite-sample transfer. On fresh rows from the same law, the residual cross-covariance is nonzero at
O(n_fit^(−1/2)) (test_20).

**Exposure disclosure:**
- defense_fit consists of the historical encoder-training rows. The encoder already saw them.
- Duplicates that link to any test-role record are excluded.
- The theorem is unaffected, since it is about the fitting sample. Held-out transfer is what the benchmark measures.

**Outside scope, stated explicitly:**
- **Nonlinear attackers.** GBT, MLP and the NL-selected attacker; quadratic predictors (that would need QLEACE).
  Recovery by them is C3. A successful nonlinear attack does not refute the linear theorem; the test_20 XOR example
  shows this.
- **Intersections of the concatenated marginals (arm C).** The guarantee is per marginal attribute. A cell indicator
  such as 1[race = k ∧ ethnicity = j] is not linear in the marginal one-hots and can stay linearly recoverable (test_20).
  - Arm C says nothing about intersectional subgroups.
  - Arm B says nothing about the non-target policy attributes.
- **Classes with P(Z = j) = 0 on defense_fit.** No statement is made for them. Class counts are recorded in
  `concept_spec`; the HMDA race classes are all ≥ 504 in defense_fit.
- **Outputs of a nonlinear head after erasure.** The PCRL head is Linear∘ReLU∘Linear, so U1's logits are a nonlinear
  function of r(h). They can re-create a linearly detectable attribute, and they are audited as their own surface.
  - Affine post-processing of r(h) stays guarded by linearity.
- **C_rep_plus_clean_out.** The clean logits are computed from the untreated h, so they are entirely outside LEACE's
  scope. The combined release is not guarded.
- **Losses that are not convex in an affine prediction, and ranking metrics.** Equal means do not imply that every
  linear score has AUC = 0.5 on finite or held-out rows. AUC of a linear attacker is a measured quantity.
- **Held-out rows, distribution shift, and directions outside the fit support.**

**Categories:**
- C1: native check (a) fails. Report the cause; (b) is reported beside it.
- C2: the held-out affine predictor beats the constant beyond sampling error. This is exploratory; the Tier-1 primary
  family has no LEACE C2 endpoint.
- C3.
- C5.
- C4 is not expected: the result is a theorem about the fitting law, and it would need an exact-arithmetic violation.

---

## Card D: unclipped Gaussian noise on h_p (untreated encoder)

**Release:** h′ = h + σ_abs·ε, with ε ~ N(0, I). The noise is one persistent draw per person:
`default_rng(release_seed).normal` over the full matrix (`releases.gaussian_release`).
- σ_abs ∈ {0.25, 0.5, 1, 2, 4, 8}; release seeds {0, 1, 2}.
- Release seeds are replicate draws on the same people. They are not added independent units.

**Native certificate:** **none (NA).**
- The representation is not clipped, so its sensitivity is unbounded.
- AAAI Proposition 3 (μ-GDP with μ = 2C/σ, AUC ≤ Φ(μ/√2)) requires all of the following, and does **not** apply:
  - projection onto an ℓ2 ball of radius C before the noise;
  - fresh independent noise;
  - a declared release count.
- No μ, ε or AUC ceiling may be computed for these releases, including with C set to an observed maximum norm.

**Scale reporting** (`scale_report` on the training-role rows):
- the per-dimension sd and σ_train-scale = √(tr Σ/d);
- quantiles of ‖h‖ and of ‖h − μ‖;
- σ/σ_train-scale and the expected noise-to-centred-norm ratio.

Equal σ_abs is not a matched operating point across encoders or datasets.

**Release surface:** h′ only.
- The clean logits in C_rep_plus_clean_out are not noised and are outside any noise argument.
- U1 (the head on h′) is a function of h′.

**Attacker access** (see RELEASE_ACCESS_TABLE):
- the ordinary slate on released attacker_fit rows (A1);
- LRT-A2: σ-informed, released rows only;
- LRT-A4: the clean-population stress test. It is kept separate and is not a contract-level attack.

Repeated-query averaging is not part of this contract (persistent draw).

**Operating point σ\*:** the smallest σ with attacker_val NL macro AUC ≤ 0.55, chosen on validation only. This is a
diagnostic, not a certificate.

**Categories:** C3; C5. C1, C2 and C4 are not applicable.

---

## References (controls, not methods)

None of these carries a bound.

| Reference | What it is | Guarantee |
|---|---|---|
| **label-only** | P(s \| y_task) as a frequency table on attacker_fit, Laplace α = 1, scored with the target's true task label | none. An empirical comparator, not a lower bound on any recovery increment. |
| **constant** | the attacker_fit class prior | none. The floor for skill scores. |
| **clean outputs (outputs_only)** | the original head's full logit vector, identical across arms within (dataset, seed) | none |
| **untreated (arm A)** | the release itself | as card A |

---

## Summary: what each arm's own check certifies, and on which rows

| Arm | Native check | Rows | Function class | Covers outputs? |
|---|---|---|---|---|
| A | N0 one-hot ridge R² ≤ 0.05 (float64 valid; float32 historical) | test split, fit = score | one ridge-fitted affine predictor, squared loss | no |
| B | Cov(r(H), Z_target) = 0 (exact) / whitened residual ≤ svd_tol (default implementation) | defense_fit | all affine predictors, all convex losses, target attribute only | affine post-processing only |
| C | the same, per marginal of the policy set | defense_fit | the same, each marginal; not intersections | affine post-processing only |
| D | none (NA) | n/a | n/a | n/a |
