# Mathematical and implementation assessment (Phase 1)

**Who and when.** Written by the independent mathematical and implementation verifier on 2026-10-16.

**Inputs.**
- `PROBLEM_SPECIFICATION.md`, `EVIDENCE_PACK.md`, `CLOSED_LANES.md`, `ROLES_AND_OWNERSHIP.md` and `VERIFIER_BRIEF.md`.
- Both proposals, with hashes checked: designer 1 `2b7273ae…ff368`, designer 2 `83653f2d…09f21`.
- Both designers' scripts and result files.
- Published aggregates only: `STAGE_A_HEADROOM.json`, and the reader-slate matrices in `DEV_REPORT_RESULTS.json` and the frozen-hybrid `READER_ACCOUNTING.csv`.

**Method.** All recomputation used separate code in `mechanism_design_independent/` (see `README.md`), run through the queue as four units:
- `V_MMP_EXACT_V1`
- `V_RCCS_GAUSS_V1`
- `V_RCCS_FINITE_V1`
- `V_RCCS_MAP_V1`

No person-level data was read.

**Conventions.**
- S = 1 is female.
- π(x) = P(S = 1 | x) and π̄ = P(S = 1).
- q_i^s = P(Y_i | x, S = s), and η_i is the SEX-blind posterior.
- "Bayes" means the population-optimal reader of the view, AUC*(S; ·).
- The brief's identities are used throughout. For any release f that is a function of X:
  - Brier(f) − Brier(η) = E‖f − η‖²;
  - NLL(f) − NLL(η) = E KL(η‖f).
- Both identities hold for the population Bayes posterior η given X. The binding clauses compare against U0 and Ucal, which are not η, so each identity describes the excess over η, and the clause excess over U0 is E‖f − η‖² − E‖U0 − η‖².

## Verdicts at a glance

| | Designer 1: MMP | Designer 2: RCCS |
|---|---|---|
| Stated identities and bounds | All verified exactly (independently reproduced to ≤1e-16) | Linear-information identities verified exactly. The supporting Gaussian model has an internal accounting inconsistency that drives its "calibration replay" |
| What the claimed reduction is about | Bayes reduction only along exactly merged directions (pure proxies). Otherwise fitted-reader learnability, and the proposal says so | Fitted readers only; the protection is defined for linear and quadratic readers of the wires, and the proposal says so. The Bayes effect of the literal map is ≥ 0: +0.029 in the designer's own feasible law. Synthetic fitted HGB and MLP readers were nevertheless reduced |
| Prototype "feasible case" under the binding three clauses | Feasible: law R at the cap, law F with headroom | **No arm reduces the pair in any law** at cap or cap + headroom. The first COND reduction costs a 3.75 pp accuracy loss. My continuous synthetic law does show feasible fitted reductions |
| Verifier's synthetic test of the mechanism | n/a: the finite laws are exact | The literal map lowers every fitted reader by 0.019–0.045, nonlinear readers included. It beats MIN only against nonlinear readers (by 0.008–0.022); the linear readers tie |
| FATAL objections | none | none |
| SERIOUS objections | 3 | 4 |
| **Verdict** | **SOUND_WITH_FIXES** | **SOUND_WITH_FIXES** (the stated evidence is weaker than claimed; the fixes are substantial) |

---

## 1. Designer 1: SEX-mixture marginalisation (MMP)

### 1.1 Well-posedness

- **The release.** Release m̂_i = (1 − π̄) q̂_i^0 + π̄ q̂_i^1, where q̂ comes from SEX-aware CatBoost teachers fitted on MODEL_FIT.
- **Contract and access.** It is a deterministic map of the six fields: SEX is evaluated at the constants 0 and 1, and no person's SEX is read. That is the primary contract. SEX is used only on MODEL_FIT. Selection uses the frozen MODEL_SELECT rule, with no grid: the nominee is fixed by registration.
- **Population target.** m_i is the Bayes posterior under P̃ = P(x)P(s)P(y | x, s). Since the loss is proper, the minimiser in §4 of the proposal is m_i. Correct.
- **Utility definitions.** Point feasibility is judged on the 24 clauses against U0 and Ucal. The subgroup guard is reported separately, as specification §6 claim 6 allows, although specification §4 also lists it as a requirement (see 1.5).

Verdict on well-posedness: well-posed.

### 1.2 Mathematical claims

| # | Claim | Status | Proof sketch or counterexample |
|---|---|---|---|
| M1 | η = m + Δ with Δ = (π − π̄)(q¹ − q⁰) | **TRUE** (identity) | Subtract the two mixtures |
| M2 | Brier(m) − Brier(η) = E[(π − π̄)²‖q¹ − q⁰‖²] | **TRUE** (exact) | Use the brief's identity with f = m; then E‖m − η‖² = E‖Δ‖². Recomputed: error ≤ 1e-16 in laws N, F and R (`V_MMP_EXACT_V1`) |
| M3 | NLL(m) − NLL(η) = E KL(η‖m) ≤ I(S;X) | **TRUE** | The equality is the brief's identity. For the bound: η and m mix the same components (q⁰, q¹) with weights (1 − π, π) and (1 − π̄, π̄). By data processing for KL through the kernel s ↦ q^s, KL(η‖m) ≤ KL(Bern π ‖ Bern π̄). Averaging over X gives I(S;X). No floor issue: m_y = 0 implies η_y = 0 when 0 < π̄ < 1. Recomputed: NLL excess equals E KL to machine precision, and the bound holds in all three laws |
| M4 | NLL price ≈ ½E[(π − π̄)² χ²(q¹, q⁰)] | **MINOR notation error** | The second-order term is ½(π − π̄)² Σ_y (q¹_y − q⁰_y)²/m_y. The denominator is the mixture m, not q⁰. The order of magnitude is unaffected |
| M5 | Constant gap g: men's within-SEX increase = ‖g‖²[v(3π̄ − 1) + κ₃]/(1 − π̄); women's = ‖g‖²[v(2 − 3π̄) − κ₃]/π̄ | **TRUE** | Within S = s, the increase is E[‖m − q^s‖² − ‖η − q^s‖² \| S = s]. With constant g this is ‖g‖² E[π̄² − π² \| S = 0], and similarly for S = 1. Reweight by (1 − π)/(1 − π̄) or π/π̄ and expand in u = π − π̄. Verified on 300 random laws (maximum error 3.9e-16). The weighted sum is ‖g‖² v, consistent with M2 |
| M6 | "Removing Δ raises **both** groups' within-SEX Brier" (used as a signature) | **Condition-dependent** | It requires a constant gap plus the moment condition. Counterexample with a non-constant gap at π̄ = 0.458: women −0.0178, men +0.0335. **In designer 1's own law R**, the income women's Brier *falls* (−0.0007) under exact Δ removal. So the "signature" argument against NEURAL_RAW (§3.3.4) is weaker than presented. The proposal already calls it "supporting, not decisive" |
| M7 | MMP's partition is the exact frontier in law R (both budgets) and in law F with headroom; at the cap in law F, partial merges reach 0.585 | **TRUE** | Independent exhaustive enumeration (Bell 6 = 203 partitions per task, all pairs): R 0.6068 / 0.6068; F 0.5853 at the cap and 0.5000 with headroom; N 0.79875 at the cap and 0.765 with headroom. Local, S12 and S21 values match the proposal exactly |
| M8 | Law R residual = AUC*(S;H) | **TRUE** | AUC*(S;H) = 0.60680 = MMP pair. For reference, AUC*(S;Z) = 0.6229 |
| M9 | "No π-term remains in either output for the other to decode" (§3.6) | **Rhetorical** | Only the product term Δ is removed. m still depends on SEX-predictive, task-relevant fields: law R keeps 0.607 through H. The proposal concedes this in failure mode 1 |
| M10 | (implicit) MMP lowers recovery relative to the SEX-blind posterior | **Not true in general** (and not claimed) | Counterexample in `V_MMP_EXACT_V1`. η merges two cells because the π-term cancels a within-SEX difference (η_A = η_B = 9/20), while m separates them (3/8 vs 3/5). The pair Bayes AUC then *rises* from 0.5 to AUC*(S;X) = 0.8. Real data will not show exact cancellation, but the same mechanism can make MMP *more* learnable than its twin. So F1 is a necessary gate, not a formality |
| M11 | π̄ is the natural mixing weight | **True for P̃; not Brier-optimal** | Among constant weights w, the Brier price E[(π − w)²‖g‖²] is minimised at w* = E[π‖g‖²]/E‖g‖². In law R income, w* = 0.388 vs π̄ = 0.438, cutting the price from 0.00070 to 0.00062. Any constant w removes the π-term equally, so this is a free utility improvement (MINOR) |

### 1.3 Bayes recovery or fitted-reader learnability? (brief item b)

- **Exact components.** With exact q^s, m is constant along every field that matters only through SEX. Those cells merge, and AUC*(S;V) falls to AUC*(S; X_task-relevant). Laws F and R show exactly this.
- **Fitted components.** A fitted CatBoost m̂ is constant only along columns no tree splits on. AGEP and WKHP are continuous, and with up to 800 trees most columns will be used. So m̂ is essentially injective on the populated support, and AUC*(S;V_MMP) ≈ AUC*(S;X).
- **What the pilot measures.** The real-data effect MMP can deliver is therefore a fitted-reader learnability effect.
- **Honesty.** The proposal states this correctly and honestly in §3.5 and §9 ("not claimed: any AUC*(S;V) reduction beyond merged directions").
- **What is missing.** The finite laws show only the exact-merge (Bayes) regime. They contain no estimation-noise regime and no law without pure proxies. Read without §3.5, the §11 table could be over-read as Bayes evidence for the real pilot. MINOR, because it is disclosed.

### 1.4 Inference from published AUCs (brief item c)

- **The reader-type pattern.** On DEV, the pair gives DEC_LR 0.568, CONF_HGB 0.574 and LOG_MLP 0.640. On ATTACK_VALIDATION, the occupation recipient gives DEC_LR 0.513, CONF_HGB 0.523 and LOG_MLP 0.624; I checked these against `READER_ACCOUNTING.csv`.
  - What it shows: most fitted signal needs the full probability vector, not the decision or the top confidence.
  - What it does not show: that the signal is Δ. Any SEX information in the vector's shape produces the same pattern, including the m-part that MMP keeps (within-SEX effects of hours, schooling and class of worker).
  - So the pattern is **non-identifying**: consistent with the premise, but not evidence for its size.
- **The subgroup "signature".** It is confounded, as the proposal says, and per M6 it depends on conditions.
- **What remains.** NEURAL_RAW E2 is a small move from U0: it keeps 91% / 93% of U0's above-chance pair AUC (verified), and U0's target η contains Δ in full. That supports "NEURAL_RAW retains most of whatever Δ U0 had". It says nothing about Δ's share of the signal.
- **Conclusion.** The proposal labels all of this "indirect" and makes Δ's share the subject of gate F1. That is the correct reading.

### 1.5 Feasibility under the 24 clauses and the subgroup guard (brief item d)

- **Budget.**
  - The binding headroom is the minimum over U0 and Ucal, from `STAGE_A_HEADROOM.json` on MODEL_SELECT, seed 0 / seed 1:
    - occupation: Brier 0.0048 / 0.0036, NLL 0.0112 / 0.0082, accuracy 0.0064 / 0.0061;
    - income: Brier 0.0037 / 0.0045, NLL 0.0055 / 0.0065.
  - So cap plus headroom is:
    - occupation: Brier ≈ 0.0098 / 0.0086, NLL ≈ 0.021 / 0.018, accuracy ≈ 0.016;
    - income: Brier ≈ 0.0087 / 0.0095.
  - The proposal quotes the U0 headroom, which is slightly larger on seed 0. MINOR.
- **The heuristic "price ≤ cap + headroom".**
  - It needs the SEX-aware components' estimation error E‖m̂ − m‖² to be no larger than T_TREE's E‖T − η‖².
  - It also needs the cross term E⟨m̂ − m, Δ⟩ to be small.
  - Both are plausible but unproven. The rarer-SEX component extrapolates in sex-atypical cells (failure mode 5). F3 tests it directly, which is adequate.
- **Price.** The Brier price is exactly Var(π)-weighted ‖g‖². A binormal reading of AUC*(S;X) ≈ 0.68–0.72 gives Var π ≈ 0.025–0.035.
  - For occupation, ‖g‖² ≈ 0.07–0.2 (sex segregation across five broad groups) gives a price of about 0.002–0.007. That is within the budget at the low end and close to it at the high end.
  - The NLL price, about ½Var π · Σ g²/m, can reach about 0.01–0.02 if the gaps in the rare classes (natural resources/construction, production) are large. That is close to the budget.
  - Accuracy flips have no bound (failure mode 2).
  - **Feasibility is genuinely open. It is not contradicted by any published number.**
- **Subgroup guard: SERIOUS for claim 6.** By M5, at π̄ = 0.47 and κ₃ ≈ 0, women bear ≈ 1.26× and men ≈ 0.77× the population occupation Brier price.
  - In designer 1's own law R, MMP is feasible at the cap (population occupation price 0.00448). There, **women's within-SEX occupation Brier rises 0.00626 > 0.005**, measured against η.
  - The proposal's failure mode 4 points to *men's* thin seed-1 headroom. The structural risk is larger for women.
  - The subgroup guard is a separately reported claim, not a selection clause. A prediction should still be registered.

### 1.6 Critic gaming

- MMP has no critic, penalty or optimisation against privacy, so the "fools its own critic" objection does not apply.
- The release is fixed by an identity, and protection is judged by the fresh slate.
- Its rationale does not rest on a fitted-AUC claim presented as a population claim (see 1.3).

### 1.7 Frontier and falsifiers (brief item e)

- **Frontier.** The finite laws show MMP can sit *on* the exact frontier. That happens when the SEX signal enters only through pure proxies (law F with headroom) or through one task-relevant proxy (law R).
  - "Moving the frontier" relative to the critic-trained methods (§3.6: critic perturbations mask Δ instead of deleting it) is **not tested**. The masking law is "specified, not run".
  - So the frontier claim currently rests on argument plus the published LOCAL and H_LJ patterns. Those patterns are also consistent with designer 2's linear-contrast account.
- **F1 (premise).** MODEL_SELECT OUT_LR proxy, twin − MMP < 0.02 on either seed.
  - Measurable and cheap.
  - Free of criterion-class bias, because MMP does not optimise against OUT_LR.
  - Decisive for the premise as far as the frozen linear proxy goes.
  - Caveat: under the frozen rule, the OUT_LR proxy is fitted on MODEL_FIT, where the teachers' outputs are in-sample. This affects all methods alike. MINOR.
- **F2 and F3.** Measurable and decisive.
- **F4.** A code-correctness check on the designer's own laws: MMP's infeasibility in law N holds by construction. It is not a falsifier of the explanation. MINOR.

### 1.8 Prototype plan (brief item f)

What is adequate:
- **Strong baseline analogue: yes.** The exact global frontier over all deterministic releases dominates every method, NEURAL_RAW included.
- **Local, S12 and S21 controls are computed,** although no joint claim is made.
- **Negative, feasible and residual laws exist,** and every number reproduces exactly.
- **Expected outcomes are stated.** The post-hoc strengthening of law N after V1 is disclosed.

What is missing:
1. A law in which a critic-style local perturbation lowers the single-recipient readers but not the pair, while MMP lowers the pair. This is §11 item 2, and it is the test of §3.6.
2. A law **without** pure proxies, where every SEX-predictive field also has within-SEX task effects. This is the realistic ACS case, and it shows how much of the twin's signal Δ actually carries.
3. A registered within-SEX prediction.

### 1.9 Objections and verdict

| Severity | Objection |
|---|---|
| SERIOUS | **S1-MMP: unestablished premise.** The share of the coalition's learnable signal carried by Δ has no support in any published number. The reader-type and subgroup evidence is non-identifying (1.4, M6). Mitigated by the cheap, registered, unbiased gate F1 |
| SERIOUS | **S2-MMP: prototype gap.** The run laws show only exact-merge Bayes effects in laws built around pure proxies. The §3.6 frontier-moving claim (masking vs deletion) and the no-pure-proxy regime are not tested (1.8) |
| SERIOUS | **S3-MMP: subgroup feasibility.** Women bear about 1.26× the population occupation price. In the proposal's own feasible law R, women's within-SEX occupation Brier rises 0.0063 > 0.005 (1.5). The proposal points at the wrong group |
| MINOR | M4 notation (χ² denominator); M9 rhetoric; M11 (w* beats π̄); the headroom heuristic and the use of U0 headroom instead of min(U0, Ucal); F4 is not a falsifier; the over-readability of the §11 table as Bayes evidence; M10 (MMP can raise recovery, so F1 is essential) |

**Verdict: `SOUND_WITH_FIXES`.**
- Every stated identity and bound is correct, and the claims are honestly scoped.
- Fixes required before a pilot is prepared:
  - (i) run the masking law and a no-pure-proxy law, with MMP, the global frontier, local, S12 and S21;
  - (ii) register a within-SEX prediction;
  - (iii) correct the NLL approximation;
  - (iv) optionally, use w* instead of π̄ (a utility-only change).

---

## 2. Designer 2: release-conditional counter-shift (RCCS)

### 2.1 Well-posedness

- **The release.**
  - f1 = T1, the frozen T_TREE income output.
  - f2 = softmax(z2 + Aψ(x)), where ψ = (r, r·g), r = logit π̂ − ℓ̂(base wires) is the OLS residual of a cross-fitted SEX propensity, and A has centred columns.
- **Contract and access.** It is deterministic in the six fields, so it uses the primary contract. SEX is used only on MODEL_FIT.
- **The objective.** Q(A) = R²_cv(S ~ φ1, φ2(f2^A), q) − R²_cv(S ~ φ1) + γ‖A‖², minimised subject to occupation budgets.
  - It is well-posed and smooth once decisions are replaced by a softmax surrogate.
  - **Minor implementation gap:** the accuracy constraint is piecewise constant in A. SLSQP cannot follow it, so it needs a surrogate or a feasibility line search.
  - **Minor estimation issue:** the budgets b are set from T_TREE's headroom on **MODEL_FIT**, where T_TREE's outputs are in-sample, so they may overstate the held-out budget. The κ grid and MODEL_SELECT feasibility mitigate this.
- **Population statement.** "Q = 0 iff occupation's wire and quadratic features add no linear predictive information about S beyond income's" is correct at the population level with γ = 0, as a statement about L² projections.
- **Deployment mismatch.** Deployment averages the five fold models' logits, while A is fitted on out-of-fold π̂. The two have different distributions. MINOR.

### 2.2 Mathematical claims

| # | Claim | Status | Proof sketch or counterexample |
|---|---|---|---|
| R1 | W_pair = W_occ + (c1 − b′c2)²/Var(f1 \| f2) (amplification law) | **TRUE** (exact, distribution-free for linear information) | Partitioned inverse (Schur complement) of c′Σ⁻¹c. Verified on 500 random instances, maximum relative error 1.1e-14 |
| R2 | With c2 = 0, W_pair = W_inc/(1 − R²); the pair-optimal occupation target with income fixed is zero partial covariance, c2 = Σ21Σ11⁻¹c1, not c2 = 0 | **TRUE** for the linear-information objective | Minimise the partial term of R1 over occupation releases with f1 fixed. It vanishes iff the partial covariance is zero |
| R3 | Full linear removal with a driver costs ‖c‖²/Var(π) (calibrated); projection costs ‖c‖²/W; invertible maps leave W, or the Bayes information, unchanged | **TRUE in the linear model**, as stated | Sherman–Morrison. Verified (relative error ≤ 1e-13), with the projection optimal among linear post-processings |
| R3′ | (implicit) RCCS pays the calibrated cost | **FALSE for the specified map** (SERIOUS) | The calibrated cost assumes the shifted release is recalibrated (E[T \| view]). RCCS releases softmax(z2 + Aψ) as is, and A acts only on r-terms, so nothing can shrink the z2-part. The literal map pays the raw cost V‖k‖², which is **‖c‖²/V** for full removal: (V + W)/V times the calibrated cost. At the designer's favoured V/Var(π) ≈ 0.1 that is about 10×, and about 9× the projection cost. "The driver halves the cost" therefore holds for full removal only with a recalibration step. In the designer's own partial-removal Gaussian model the penalty is smaller: see 2.5 |
| R4 | Published amplification ≈ 5–7.6 in protected services vs ≈ 1 in task-only ones, "implies R²(f1 \| f2) ≈ 0.82" | **Arithmetic TRUE for seed 0; the inference is model-dependent** | See 2.4 |
| R5 | 90% of NEURAL_RAW's pair signal is occupation given income, "exactly RCCS's target" | **Arithmetic TRUE; the inference is non-specific** | See 2.4 |
| R6 | The own + pair objective needs 27% more budget than RCCS in the finite law | **TRUE as computed** (0.0142 vs 0.0113 for pair 0.596) | It depends on the equal weighting of the two terms. Also, both costs lie **outside** the binding clauses (2.6) |
| R7 | The release is injective, so AUC*(S;V) = AUC*(S;X), "equally true of NEURAL_RAW and the trees" | **TRUE but incomplete** (MINOR disclosure; see 2.3 and 2.7) | For a piecewise-constant tree base, RCCS adds the propensity residual r, which by premise is information *not* in the base release. If (f1, f2′) still determines the base pair (small shifts), data processing gives AUC*(S;V_RCCS) ≥ AUC*(S;V_base); generically it equals AUC*(S;X). In the designer's own feasible law the injective shift raises pair Bayes AUC **from 0.615 to 0.644** (`V_RCCS_FINITE_V1`). The Bayes effect of the mechanism has the opposite sign to its fitted-reader goal |
| R8 | Shift cost is second order while the S-covariance change is first order | **TRUE only if T2 ≈ Bayes along ψ** | The excess is 2E⟨T2 − η2, δ⟩ + E‖δ‖². The first term vanishes only if the base is Bayes-optimal along ψ. MINOR |

### 2.3 Bayes recovery or fitted-reader learnability? Critic gaming (brief items b and 4)

- **What the proposal says.** It states correctly that every claim is a fitted-reader claim (§3.5 and §9).
- **What it does not say.** The literal map *injects* the SEX-propensity residual into the occupation output, with a sign chosen to cancel linear (and quadratic) covariance. The criterion Q is exactly the best ridge critic on those features, so the method best-responds to a restricted reader class.
- **How it replies.** "No critic lag" answers critic *staleness*. It does not answer critic *class*. Nonlinear readers (OUT_HGB, OUT_MLP and LOG_MLP are in the slate) are outside the criterion. In the Bayes sense RCCS cannot lower and generically raises recovery (R7).
- **Synthetic test (`V_RCCS_MAP_V1`).** I tested whether this risk materialises in a smooth law, using the literal map with an oracle propensity and a large criterion sample. That is RCCS's best case. Full results are in 2.7.
  - The Bayes AUC of the RCCS release did rise (by 0.021–0.053).
  - **But every fitted reader fell** (by 0.019–0.045), and HGB and MLP fell at least as much as the linear and quadratic readers, at both 8,000 and 100,000 training rows.
- **Classification.** **MINOR (residual).** Two requirements remain:
  - The proposal should state that its Bayes effect is ≥ 0, as in R7.
  - The prototype should add a law with *discrete* S-only fields, the ACS case (POBP, MAR and COW levels). Exact cell structure could let HGB decode the injected residual there; this is untested.
- **On S6 in general.** S6 is met to the extent that the two smooth synthetic laws are representative.

### 2.4 Inference from published AUCs (brief item c)

I recomputed every number from EVIDENCE_PACK table A (`V_RCCS_GAUSS_V1`).

**Arithmetic.**
- The seed-0 amplifications reproduce: LOCAL 5.67, JOINT 5.88, H_LJ 5.06, H_JL 7.60, NEURAL 2.39, TASK_REFIT 0.94, T_TREE 1.02, U0 1.16.
- LOCAL's linear readers give 5.57 (OUT_LR).

The inferences are **model-dependent extrapolations**, for six reasons:

1. **Seed 1 is not reported, and it differs.** LOCAL 6.69, **JOINT 3.29**, H_LJ 5.96, H_JL 6.55, NEURAL 1.61, TASK_REFIT 1.60, T_TREE 1.12, U0 0.76.
2. **Sampling spread is large.** These are 95% ranges from Hanley–McNeil errors at DEV size, treated as independent:
   - LOCAL s0: [3.2, 10.8];
   - JOINT s1: [0.3, 11.0];
   - NEURAL s0: [−0.2, 7.3];
   - T_TREE s0: [−0.4, 2.8].

   The qualitative contrast between protected services (about 5–7) and task-only ones (about 1) survives. The point value "R² ≈ 0.82" does not: under c2 = 0, LOCAL s0 implies R² ∈ [0.69, 0.91].
3. **The c2 = 0 premise is false for these services.** Their occupation AUC is 0.56–0.58, i.e. d′² 0.04–0.09, not zero. With b′c2/c1 = ±0.3, the point estimate implies R² from 0.70 to 0.91.
4. **Different views use different reader classes.** The published "selected reader" differs by view and is usually nonlinear for the pair:
   - LOCAL s0: income OUT_LR or LOG_LR (0.558), occupation OUT_HGB (0.582), pair LOG_MLP (0.657);
   - NEURAL s0: income OUT_MLP, occupation LOG_MLP, pair LOG_MLP.

   The d′² values are therefore not on one linear-information scale. The AUC → d′² map also assumes binormal, equal-variance scores.
5. **The readings are reader-class sensitive.** On ATTACK_VALIDATION, NEURAL_RAW's amplification is 1.80 (OUT_LR), **0.35** (LOG_LR) and 0.88 (LOG_MLP), against 2.39 for the DEV selected readers. LOCAL's linear-only values (4.9–5.6) are stable.
6. **"90% … exactly RCCS's target" is non-specific.**
   - The share of pair d′² not in income alone is **0.81–0.92 for every service**: T_TREE 0.83 / 0.84, U0 0.86 / 0.81, TASK_REFIT 0.82 / 0.85, NEURAL 0.90 / 0.87. That follows simply because every income recipient is weak.
   - **75–79% of NEURAL_RAW's pair d′² is occupation's own marginal signal.** That is equally the target of MIN, and of NEURAL_RAW's own occupation critic.
   - The decomposition therefore does not show that NEURAL_RAW specifically retains the *partial*-covariance component.

**What would break the account.**
- A precondition measurement on T_TREE wires with R²(φ1 | φ2) < 0.6; the proposal registers this as falsifier 5, which is good.
- A pair signal that is mainly nonlinear.
- Amplification patterns caused by reader selection rather than linear structure.

**What stands.** The S2 evidence that NEURAL_RAW leaves the occupation budget unspent is sound and quantitative: occupation NLL +0.0004 / +0.0013 against a 0.010 cap on MODEL_SELECT, with only E2 feasible.

### 2.5 The Gaussian feasibility model (§10)

- **Reproduction.** Under the designer's own convention, I reproduce its numbers exactly, with the same 15 valid settings:
  - RCCS reaches pair 0.606 at 0.89–1.25 "units" in 14 settings, and at 1.78 in the weakest;
  - at one unit, RCCS gives 0.601–0.612 (0.630 in the weakest setting) and MIN gives 0.616–0.714.
- **Defect 1: accounting inconsistency.**
  - The MIN arm's cost is that of an *income-aware recalibrated* occupation release, E_lin[T | f1, g′, h′].
  - Its occupation-only leakage, however, is computed for the *un-recalibrated* shifted coordinates (g′, h′), which is not what is released.
  - With consistent accounting (convention B: the leakage of the release actually emitted), MIN at one unit gives 0.598–0.632 against RCCS's 0.592–0.607. **The MIN-versus-RCCS gap shrinks from 0.009–0.110 to 0.002–0.031.**
  - Under strictly local conventions the gap is 0.004–0.062. That includes C (occupation-only recalibration) and D (the literal raw map: MIN 0.611–0.667 against RCCS 0.601–0.612).
- **Defect 2: the "calibration check" is an artefact of the convention.**
  - Under the designer's convention, MIN reproduces LOCAL's published pair (0.654 vs 0.657) only at (R² = 0.82, d′²_X = 0.40). Across the 9 replay settings it gives 0.604–0.687.
  - Under consistent accounting (B), the model predicts LOCAL-type protection would cut the pair to **0.591–0.609**, which contradicts the published 0.657.
  - Under D it gives 0.604–0.660.
  - So the model fits LOCAL only with one convention and one parameter setting. It is a fit, not a validation.
- **Defect 3: raw cost (R3′).**
  - Price the literal map's raw cost in the designer's LOCAL-anchored units. RCCS then needs **1.05–2.15 units to reach 0.606 in 14 settings, and 6.0 in the weakest**. Its pair at one unit is 0.607–0.629 (0.644 in the weakest).
  - The available occupation budget is about 2.4–3.2 units, so the model still allows the target except in the weakest-driver setting. The headline "≈ 1 unit" is optimistic by up to about 2×.
- **Status.** None of this falsifies RCCS. It removes the model's support for the claim that conditional targeting beats local targeting by a large margin. Under consistent accounting the margin is small, and the replay that was offered as validation fails.

### 2.6 Finite laws, feasibility under the 24 clauses, falsifiers (brief items d, e, f)

**Finite laws** (`V_RCCS_FINITE_V1`, exhaustive over 4,140 partitions).
- Every number the designer reports reproduces exactly:
  - feasible law: AUC*X 0.644, income-only 0.577, TASK pair 0.615;
  - COND 0.596 at Brier 0.0113 and 0.577 at 0.0225;
  - COND_PP 0.596 at 0.0225 and 0.577 at 0.045;
  - JOINTSUM 0.596 at 0.0142 and 0.601 at B = 0.020;
  - MIN raises the pair to 0.644;
  - negative, coin and redundant laws as stated.
- **But the arms are constrained by summed Brier only.** Under the binding three clauses (accuracy ≤ 0.010 / 0.016, NLL ≤ 0.010 / 0.020, Brier ≤ 0.005 / 0.009 at cap / cap + headroom), **COND equals TASK, with no reduction, in all four laws at both budgets.**
- COND's first reduction (0.615 → 0.596) costs Brier 0.0113, NLL 0.0114 and **accuracy loss 0.0375**. That is 2.3× the cap-plus-headroom accuracy budget, above the Brier budget, and above the NLL cap.
- So the brief's "feasible case" is **not feasible at the required utility**. That is partly the coarse granularity of 8 cells of mass 1/8, but the proposal presents it as feasible. **SERIOUS** for the prototype plan.
- **The finite laws do not test RCCS itself.** COND is a *pooling* oracle that merges cells and so lowers Bayes AUC. RCCS is an injective linear counter-shift, and in the same law its Bayes AUC is 0.644 (INJ). The laws therefore test the conditional *target*, not the *map* or its reader-class robustness.

**Strong baseline analogue.**
- JOINTSUM (the RAW-J structure with income frozen), COND_PP and the coin control are good controls.
- The NR_OCC and NR_OCC_U0 pilot arms are the right existing-machinery controls.
- No joint claim is made, and RCCS is sequential with full look-ahead, consistent with specification §9.1.

**Utility feasibility.**
- Income is untouched (T_TREE passes all 24 clauses on MODEL_SELECT), and the occupation budget exists.
- Feasibility depends on the occupation accuracy flips and the subgroup guard. The shift is SEX-correlated, and LOCAL is already UNRESOLVED for men. Plausible, unproven.

**Falsifiers.**
- All are measurable, but **they are not decisive in either direction.**
- **False falsification.** Falsifiers 1 and 3, and the frozen selection rule, use the OUT_LR proxy, which is linear in the wires and inside RCCS's criterion class. In the synthetic RCCS-vs-MIN test (2.7), RCCS and MIN are **indistinguishable to linear readers** (|Δ| ≤ 0.0012) while RCCS is 0.008–0.022 lower against HGB and MLP pair readers. So falsifier 1 can reject a mechanism that helps against the slate's strongest readers.
- **False pass.** Falsifier 4's tolerance (OUT_HGB − OUT_LR gap ≤ 0.03) is looser than the margins at stake. Plain superiority needs 0.012; RCCS's own predicted benefit is ≈ 0.03; NEURAL_RAW's MODEL_SELECT proxy is 0.625. An HGB lift of 0.029 would pass the gate and could erase the whole benefit.
- Falsifier 2 (placebo and post-processing drivers) and falsifier 5 (preconditions) are well designed.
- **SERIOUS** until fixed. Evaluate falsifiers 1 and 3 and the stopping rule on a MODEL_SELECT slate-max proxy (OUT_LR, OUT_HGB and a log-probability MLP, all fitted on MODEL_FIT), and tighten falsifier 4 to a gap of ≤ 0.01.

### 2.7 Synthetic test of the literal map against nonlinear readers (`V_RCCS_MAP_V1`)

**The law.**
- w1, w2, c ~ N(0,1), where c is SEX-only.
- The SEX logit is linear in (w, c).
- Income is logistic in w. Occupation is a 3-class softmax in w.
- The base pair (f1, f2) is a function of w only, so its Bayes AUC is AUC*(S;W).

**RCCS setup.** The oracle propensity, the criterion fitted on 30,000 rows, A with 10 free parameters fitted by Powell, and budget B on the exact occupation Brier excess.

**Readers.** Linear logistic; quadratic logistic (the criterion class); HGB; and MLP (64-64). Each was trained on 8,000 and on 100,000 rows (MLP on 8,000 only) and tested on 100,000.

**Results, base → RCCS.** Test AUC for S; n_tr is the number of training rows. "Bayes" is computed from the known law: the base pair has AUC*(S;W), and the RCCS release is injective, so it has AUC*(S;X).

| Law (V/Var logit π) | B | Linear (8k) | Quadratic (8k) | HGB (8k / 100k) | MLP (8k) | Bayes, base → RCCS |
|---|---|---|---|---|---|---|
| g = 0.35 (0.20) | 0.005 | 0.678 → 0.659 | 0.678 → 0.659 | 0.655 → 0.636 / 0.677 → 0.659 | 0.678 → 0.657 | 0.678 → **0.699** |
| g = 0.35 | 0.009 | 0.678 → 0.653 | 0.678 → 0.652 | 0.655 → 0.625 / 0.677 → 0.652 | 0.678 → 0.651 | 0.678 → **0.699** |
| g = 0.6 (0.41) | 0.005 | 0.673 → 0.645 | 0.673 → 0.645 | 0.644 → 0.614 / 0.672 → 0.645 | 0.671 → 0.643 | 0.673 → **0.726** |
| g = 0.6 | 0.009 | 0.673 → 0.635 | 0.673 → 0.635 | 0.644 → 0.600 / 0.672 → 0.635 | 0.671 → 0.630 | 0.673 → **0.726** |

**Test-set costs.**
- At B = 0.005: Brier 0.0050–0.0051, NLL 0.0085, accuracy loss 0.0050. Inside the cap.
- At B = 0.009: Brier 0.0089–0.0091, NLL 0.0151–0.0153, accuracy loss 0.0082–0.0087. Inside cap + headroom on all three clauses.
- The budget binds: the final partial R² is 0.04–0.07, not 0.

**Reading.**
- **The Bayes statement R7 is confirmed.** The RCCS release raises Bayes recovery by 0.021–0.053.
- **Every fitted reader fell nonetheless,** by 0.019–0.045. HGB and MLP fell at least as much as the criterion-class readers.
- **In a smooth law, the critic-class risk did not materialise for fitted readers at n ≤ 100,000.** The Bayes-vs-fitted gap is real but went unexploited.
- **This continuous law also shows the literal map can lower fitted recovery within all three binding clauses,** which the designer's 8-cell laws could not show.

**RCCS vs MIN** (unit `V_RCCS_MAP2_V1`): same map family, same budget B = 0.009. MIN minimises occupation-only R²(S ~ φ2′, p2′p2′ᵀ). This is the analogue of falsifier 1.

| Law | Reader (training rows) | BASE | RCCS | MIN | RCCS − MIN |
|---|---|---|---|---|---|
| g = 0.35 | pair, linear (8k) | 0.681 | 0.655 | 0.653 | +0.001 |
| | pair, HGB (8k / 100k) | 0.653 / 0.680 | 0.625 / 0.655 | 0.642 / 0.667 | **−0.017 / −0.012** |
| | pair, MLP (8k) | 0.680 | 0.654 | 0.665 | **−0.011** |
| | occupation only, linear / HGB 100k / MLP | 0.681 / 0.679 / 0.680 | 0.655 / 0.654 / 0.654 | 0.654 / 0.653 / 0.653 | ≈ 0 |
| g = 0.6 | pair, linear (8k) | 0.674 | 0.636 | 0.635 | +0.001 |
| | pair, HGB (8k / 100k) | 0.647 / 0.673 | 0.610 / 0.638 | 0.632 / 0.660 | **−0.022 / −0.022** |
| | pair, MLP (8k) | 0.674 | 0.635 | 0.643 | **−0.008** |
| | occupation only, linear / HGB 100k / MLP | 0.674 / 0.673 / 0.673 | 0.636 / 0.635 / 0.636 | 0.635 / 0.634 / 0.635 | ≈ 0 |

**Reading.**
- **No advantage against linear readers.** Against linear readers, the conditional target gives no advantage over the local target (|Δ| ≤ 0.0012).
- **The predicted signature is absent.** RCCS's occupation-only AUC is not above MIN's.
- **Against the nonlinear pair readers, RCCS is 0.008–0.022 below MIN.** That is a real advantage against the slate's strongest readers. It is not the *linear* amplification story the proposal gives, and the registered falsifier 1, measured with the OUT_LR proxy, would **reject** the mechanism in this law.
- **Caveat.** This is one synthetic law, not calibrated to ACS. Here the occupation output nearly determines the income output, since both are functions of the same two-dimensional w, which is unlike ACS.

### 2.8 Objections and verdict

| Severity | Objection |
|---|---|
| SERIOUS | **S1-RCCS: model–mechanism mismatch and accounting inconsistency** (R3′, 2.5). The calibrated cost is credited to a map with no recalibration step. The MIN arm's leakage is measured on a release that is not emitted. With consistent accounting the RCCS-vs-MIN margin is 0.002–0.031, not up to 0.11, and the LOCAL replay fails (0.59–0.61 vs 0.657) |
| SERIOUS | **S2-RCCS: the published-AUC reading is a model-dependent extrapolation, presented as quantitative support.** Seed 1 is unreported (JOINT 3.29); the 95% spreads are wide; the c2 = 0 premise is false; the views mix reader classes; NEURAL's amplification ranges 0.35–2.39 by reader. The 90% share is non-specific: it is 0.81–0.92 for every service (2.4) |
| SERIOUS | **S3-RCCS: the prototype's feasible case is infeasible under the binding clauses** (no reduction at cap or cap + headroom; the first reduction costs a 3.75 pp accuracy loss). The finite laws test a pooling oracle, not the injective map (2.6) |
| SERIOUS | **S4-RCCS: the falsifiers are not decisive in either direction** (2.6). Selection and falsifiers 1 and 3 use OUT_LR. In the synthetic test, RCCS and MIN tie for linear readers while RCCS is 0.008–0.022 better against HGB and MLP, so false falsification is possible. The 0.03 nonlinear-gap tolerance exceeds the target margins, so a false pass is possible too |
| MINOR | **Critic-class dependence (S6), residual.** The map injects the propensity residual, so its Bayes effect is ≥ 0 (+0.029 in the designer's own law; +0.021 to +0.053 in my synthetic laws), which the proposal should state. In two smooth synthetic laws, however, HGB and MLP fell at least as much as linear readers (2.7). Discrete S-only fields are untested |
| MINOR | The non-smooth accuracy constraint in SLSQP; MODEL_FIT (in-sample) headroom used for budgets; the out-of-fold π̂ vs fold-average deployment mismatch; R8 (first-order term when T2 is not Bayes); R6 depends on the weighting; the subgroup risk (acknowledged) |

**Verdict: `SOUND_WITH_FIXES`.**
- No stated identity is false. The mechanism and its objective are well-posed. The claims are scoped as fitted-reader claims.
- But each quantitative leg of its feasibility case needs repair:
  - (i) recompute the Gaussian model with consistent accounting and with the literal map's raw cost (or add an explicit recalibration step to the mechanism, and count its parameters), and report the LOCAL replay under that accounting;
  - (ii) restate R² ≈ 0.82 and the 90% share as model-dependent readings, with both seeds and their spreads;
  - (iii) replace the 8-cell "feasible case" with a law in which the *literal map* lowers the pair **within all three binding clauses**, evaluated against Bayes, linear, quadratic and nonlinear readers, including a law with discrete S-only fields. My continuous synthetic law (2.7) shows such a law exists for fitted readers;
  - (iv) restate the mechanism's signature. In 2.7 the conditional-target advantage over MIN appears only for nonlinear readers, not as "lower linear pair proxy plus higher occupation-only AUC". Then evaluate falsifiers 1 and 3 on a slate-max MODEL_SELECT proxy, and tighten falsifier 4.
- **Evidence status.** Until (i) and (iii) are done, the proposal's S5 evidence is "plausible, partly shown in synthetic laws", not the calibrated 0.60–0.61 prediction it states.

---

## 3. Comparison

| Question | MMP (designer 1) | RCCS (designer 2) |
|---|---|---|
| Are the core mathematics correct? | Yes, all exact (M1–M3, M5, M7, M8) | The linear-information identities are correct (R1–R3). The supporting model is inconsistent (2.5) |
| Does the rationale depend on a reader class? | No. It removes a defined population term with no critic. Bayes reduction is limited to exact merges, and that is disclosed | Removal is defined for linear and quadratic readers, and the Bayes effect is ≥ 0. Yet in two smooth synthetic laws the fitted protection carried over to HGB and MLP. Discrete-field laws are untested |
| Does the stated mechanism explain the effect? | Exact in the finite laws. The real-data share of Δ is unknown (F1) | Only partly. In the synthetic test the advantage over local targeting appears only for nonlinear readers, not as the linear "amplification" signature the proposal registers |
| S2 evidence (what NEURAL_RAW retains) | Indirect but logically coherent ("a small move from U0, whose target contains Δ"). Δ's share is unknown and goes to F1 | Unspent occupation budget is shown. The "90%" decomposition is non-specific |
| S5 evidence (feasibility and size) | Closed-form prices. Feasibility is open; F3 settles it cheaply. No size forecast beyond subjective probabilities (material superiority 0.07) | A Gaussian prediction of 0.60–0.61 that rests on a convention-dependent model and a model-dependent AUC reading. With consistent accounting the advantage over local targeting is small |
| Prototype | Run, exact, with a strong frontier comparator. Missing the masking law and a no-pure-proxy law | Run, exact, with good controls. Its feasible case fails the binding clauses, and it tests a pooling oracle rather than the map. The verifier's continuous law partly fills the gap |
| Cheapest decisive test | F1 + F3 on MODEL_SELECT, about 0.3 CPU-h, with no criterion-class bias | Falsifier 5 preconditions on MODEL_FIT, then the RCCS-vs-MIN and NR_OCC proxies. These need a tightened nonlinear gate to be decisive |

**Mathematical view.**
- Both proposals are `SOUND_WITH_FIXES`. Neither has a FATAL objection.
- **MMP.** Its mathematics is exact and its claims are scoped to what the mathematics supports. Its risks are empirical (Δ's share, accuracy flips, women's subgroup Brier) and are resolved cheaply by registered gates that its own optimisation does not bias.
- **RCCS: the identities are right, but the feasibility case does not hold up.** It rests on three supporting computations that fail as stated:
  - the inconsistent Gaussian accounting and the replay that depends on it;
  - the non-specific and model-dependent readings of the published AUCs;
  - a "feasible" finite law that fails the binding clauses.
- **RCCS: the synthetic test cuts both ways.**
  - In its favour: the literal map does lower fitted recovery against all reader classes within the binding clauses, in a smooth synthetic law.
  - Against its stated explanation and registered falsifier: its edge over local targeting appeared only for nonlinear readers.
- **Under selection rule 2** ("stronger S2/S5 evidence, fewer unresolved SERIOUS objections"), the mathematics leans to **MMP: 3 SERIOUS vs RCCS's 4**, and MMP's are of the cheap-to-resolve kind.
- **What would change this.** RCCS's synthetic behaviour is the more promising empirical signal of the two. Once its model and falsifiers are repaired, the comparison could change.
- Nothing here bears on novelty, which is the prior-work reviewer's call.

---

## 4. Reproducibility

| Unit | CPU s (ledger) | Output |
|---|---|---|
| `V_MMP_EXACT_V1` | 0.25 | `V_MMP_EXACT_V1.json` |
| `V_RCCS_GAUSS_V1` | 7.6 | `V_RCCS_GAUSS_V1.json` |
| `V_RCCS_FINITE_V1` | 6.2 | `V_RCCS_FINITE_V1.json` |
| `V_RCCS_MAP_V1` | 167.0 | `V_RCCS_MAP_V1.json` |
| `V_RCCS_MAP2_V1` | 206.1 | `V_RCCS_MAP2_V1.json` |

Every finite-law number in the two proposals reproduced exactly with separate code. The only numbers that change are those under my stated alternative accounting conventions and binding-clause budgets.
