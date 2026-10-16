# Proposal comparison against closest prior work (Phase 2)

Role: closest-prior-work reviewer. Written 2026-10-16, 05:35Z to 05:58Z.

**Inputs (read only).**
- `roles/designer_1/PROPOSAL.md`, sha256 `2b7273ae01d6449036f25c7a6f5e74efb8d2843f1251cbceb8e4c7024c9ff368` (MMP).
- `roles/designer_2/PROPOSAL.md`, sha256 `83653f2db759bb8339edd96984ffa89683dc2f84478d37304b489894f4009f21` (RCCS).
- `CLOSED_LANES.md` and `roles/coordinator/CLOSED_LANES_HELPER_REPORT.md`.
- My `LITERATURE_BASE.md` (§ references below point there).
- The primary sources listed in §4, newly opened today.

Markers [P]/[M]/[A]/[NV] mean what they mean in `LITERATURE_BASE.md`. No priority claim is made anywhere. "Not found" means not found in the sources listed.

---

## 0. Bottom line

1. **MMP's central device is established prior art.**
   - Device: fit an attribute-aware model, then release its average over the attribute with fixed population weights.
   - Žliobaitė (2017, appendix) advocates exactly this for regression. She fits with the sensitive variable, then replaces its component by c = s̄β, "a weighted average of both, where the weight reflects the balance between both population groups".
   - In probability space it is the standard adjustment formula pr(y | x̂) = Σ_z pr(y | x, z) pr(z) (Pearl 1995, Thm 1), with z = SEX.
   - Pope & Sydnor (2011) propose "a method that can eliminate proxy effects while maintaining predictive accuracy" (abstract). Their full text was paywalled, so designer 1's description of their method is unverified.
   - MMP is an **established technique applied to a new endpoint** (coalition SEX inference). No new algorithmic ingredient was found, and designer 1 says as much.
2. **Most of RCCS's ingredients are established; one combination was not found.**
   - The established ingredients:
     - sequential design with the first release fixed (Taylor 2026; Wang & Fung 2006; this repository's S12 arms and income-frozen adapters);
     - a target equal to the incremental information of release 2 given release 1 (Erdogdu & Fawaz 2015; Taylor 2026; Sankar et al. 2013 Thm 3; this repository's RCSR conditional budgets and OWN-09);
     - a linear coalition-R² criterion (PCRL's own cross-purpose constraint, OWN-03; LEACE-type guardedness);
     - conditional parity as a notion (Ritov, Sun, Zhao 2017);
     - cross-fitted partialling-out (Chernozhukov et al., following Robinson 1988).
   - The combination not found in the sources read: a deterministic, attribute-blind output shift *driven by an explicit cross-fitted SEX propensity* to meet a second-order constraint conditional on another recipient's release. This is a bounded-search observation, not a priority claim.
3. **Repository overlap.**
   - MMP's device appears in no closed lane I could find.
   - RCCS overlaps with the closed **small-adapters** lane. The minimal adapter was already an income-frozen occupation correction, linear in 64 frozen trunk features (not pure post-processing).
   - It also overlaps with the conditional/coalition-constraint lanes (OWN-03, OWN-09, RCSR, conditional budgeted release). Its distinct elements are the propensity driver and the pair-only output-wire ridge objective.
4. **Main citation and scope issues.**
   - **MMP:** prior-art attribution should be Žliobaitė 2017 (and Žliobaitė & Custers 2016, cited there) plus the adjustment formula. "Δ is the only route" is overstated.
   - **RCCS:** the small-adapter row mislabels the adapters as post-processing.
   - **RCCS:** its "partial" objective equals the coalition ridge R² up to a constant, because income is frozen. That needs the OWN-03 precedent.

---

## 1. Designer 1: SEX-mixture marginalisation (MMP)

**Mechanism.**
- Fit SEX-aware teachers q̂ᵢˢ(x) ≈ P(Yᵢ | x, S=s) on MODEL_FIT.
- Release m̂ᵢ(x) = (1−π̄) q̂ᵢ⁰(x) + π̄ q̂ᵢ¹(x) and its argmax, deterministically. No SEX is read at deployment.
- The twin replaces π̄ by π̂(x), which recovers the SEX-blind posterior.

### 1.1 Closest methods

| Work | Objective | Training access | Release | Protects | Coalition | Optimisation | Guarantee / scope | Relation to MMP |
|---|---|---|---|---|---|---|---|---|
| **Žliobaitė 2017**, "Fairness-aware machine learning: a perspective", arXiv:1708.00754, appendix [P]. It attributes the bias derivation to Žliobaitė & Custers, AI & Law 24(2):183–201 (2016) [NV: paywalled] | remove the omitted-variable bias that makes legitimate variables proxy for s | s in training | predictions with the s-component replaced by c = s̄β | discrimination via proxies (regression) | none | OLS on full data, then substitute the constant | algebra of omitted-variable bias (linear) | **Same device.** For a linear model, b₀ + b′x + s̄β = Σ_s P(s)(b₀ + b′x + βs), i.e. MMP's fixed-weight mixture. MMP extends it to a probability-space mixture of nonlinear (CatBoost) components. |
| **Pope & Sydnor**, *Implementing Anti-discrimination Policies in Statistical Profiling Models*, AEJ: Economic Policy 3(3):206–231 (2011); AEA page [M] | "eliminate proxy effects while maintaining predictive accuracy" vs dropping contentious variables (abstract) | protected traits in estimation (abstract implies) | profiling scores | proxy effects of prohibited traits | none | [NV] | [NV] | Consistent with designer 1's guess. **The method details are not verified** (full text behind login). |
| **Pearl 1995**, Biometrika 82(4):669–710, Thm 1 (UCLA R-218-B) [P] | identify a causal effect | — | pr(y \| x̂) = Σ_z pr(y \| x,z) pr(z) | — | — | — | identifiability under the back-door criterion | **MMP is this formula with z = SEX.** Its "counterfactual law P(x)P(s)P(y \| x,s)" is the post-intervention law that cuts S → X. |
| **Kilbertus et al.**, NeurIPS 2017, pp. 656–666 (arXiv journal-ref) [P][A] | no proxy discrimination: P(R \| do(P=p)) = P(R \| do(P=p′)) (Def. 3) | needs a causal graph | predictor | proxy routes of A | none | constrained fit in a hypothesis class (linear example) | conditional on the causal model | Same goal (strip the proxy route); different construct. Needs a graph; intervenes on proxies, not on the mixture weights. |
| **Kusner et al.**, *Counterfactual Fairness*, NIPS 2017 (PDF footer) [P][A] | counterfactual fairness | causal model | predictor | A's causal influence | none | Levels 1–3 | Lemma 1: a function of the non-descendants of A is counterfactually fair | MMP satisfies only fairness through unawareness (Def. 1) at deployment. **It is not counterfactually fair**: it keeps within-SEX effects of SEX's descendants. |
| **Lipton, Chouldechova, McAuley**, NeurIPS 2018 (PDF footer) [P][A] | analysis of "disparate learning processes" (DLPs): "the sensitive feature is used during training but a group-blind classifier is produced" | s in training only | group-blind classifier | impact parity | none | — | DLPs induce indirect treatment disparity and within-class discrimination when features reveal group | MMP is a DLP by this definition. Its aim (remove the proxy mixture term, keep within-group effects) differs from parity-seeking DLPs. |
| Calders & Verwer, DMKD 21 (2010) | — | — | — | — | — | — | — | **[NV]**: Springer PDF blocked today, so not characterised. |

**Privacy use.** In the sources searched, I found no attribute-inference *privacy defense* that releases an attribute-aware model marginalised with the attribute's marginal. The precedents are fairness and causal ones.

Information-theoretic mappings that may read S at deployment (Calmon & Fawaz Remark 3; GAP's g(X,Y)) have a larger achievable region (Wang–Basciftci–Ishwar; `LITERATURE_BASE.md` §3). MMP stays in the "useful data only" class at deployment.

### 1.2 Ingredient classification

| Ingredient | Class | Evidence |
|---|---|---|
| SEX used at fit time only (attribute-aware teacher, blind release) | **established technique** | DLP class (Lipton et al.); Žliobaitė's full-data fit |
| Release = fixed-weight marginalisation over SEX | **established technique** | Žliobaitė c = s̄β; Pearl Thm 1 (standardisation) |
| Probability-space mixture of CatBoost components (vs linear coefficient replacement) | **engineering** | the adjustment formula is model-agnostic |
| Brier identity and NLL ≤ I(S;X) price | **established technique** (standard algebra; joint convexity of KL) | designer's derivation; the verifier should recompute |
| Twin α = 1 (posterior-weighted mixture) | **established** (law of total probability) as a control | — |
| Use as a coalition SEX-inference defense under per-task utility caps | **engineering combination** (new endpoint, same algorithm) | not found as a privacy defense in the sources searched |
| Potentially new algorithmic ingredient | **none found** | — |

### 1.3 Closed lanes

- MMP's device appears in no row of `CLOSED_LANES.md` or the helper report.
- A grep of repository `results/*.md` for marginalisation over SEX or SEX-aware teachers found no lane.
- **Nearest lane:** feature masks (removing fields). Designer 1's distinction is correct: masks delete within-SEX task value too.
- **"Invariant baselines" (ACS 2018), which designer 1 asked me to check:** its `RESEARCH_DECISION.md` (branch `research/pcrl-invariant-baselines-v1`) concerns an exactly **rotation-invariant** repaired penalty for a learned linear/kernel channel. It has nothing to do with SEX invariance or marginalisation. No overlap.

### 1.4 Citation and scope errors in MMP

1. **Prior art (§8 item 6).**
   - Cite Žliobaitė (2017) and, through it, Žliobaitė & Custers (2016), plus Pearl's adjustment formula, as the verified antecedents.
   - Pope & Sydnor is consistent at abstract level only.
   - "At best an engineering combination" is, if anything, generous to novelty: the algorithm is established and only the endpoint is new.
2. **"Δᵢ is the only route by which a SEX-blind posterior converts the fields' SEX-predictiveness into task predictions" (§3.1)** is overstated.
   - The components qᵢˢ(x) themselves depend on SEX-predictive fields (designer 1's own failure mode 1).
   - Correct scope: Δᵢ is the only route *through the mixture weights*.
3. **"No π-term remains in either output for the other to decode" (§3.6)** holds for the additive decomposition, not for information.
   - (m₁, m₂) are functions of x, and π(x) may remain decodable from them.
   - §3.5's honest limit, AUC\*(S;V) ≈ AUC\*(S;X) for injective releases, is the operative statement. §3.6 should defer to it.
4. **Fairness labels.** MMP is a DLP (Lipton et al.) and not counterfactually fair (Kusner et al., Lemma 1). Designer 1 claims neither, which is correct; any paper text must not call it "fair".

---

## 2. Designer 2: release-conditional counter-shift (RCCS)

**Mechanism.**
- Income is frozen at T₁.
- Occupation is f₂ = softmax(z₂ + Aψ(x)), with A ∈ ℝ^{5×7}.
- ψ = (r, r·g), where r = logit π̂(x) − ℓ̂(φ₁, φ₂) is a cross-fitted SEX propensity residualised on the base wires.
- A minimises Q(A) = R²_cv(S ~ φ₁, φ₂(f₂), q) − R²_cv(S ~ φ₁) + γ‖A‖², subject to cross-fitted occupation budgets.

### 2.1 Closest methods

| Work | Objective | Training access | Release | Protects | Coalition / conditioning | Optimisation | Guarantee / scope | Relation to RCCS |
|---|---|---|---|---|---|---|---|---|
| **LEACE** (NeurIPS 2023) [P] | zero cross-covariance with the concept, least-squares-optimal | concept labels | affine-erased representation | linear predictors (convex losses) | **unconditional** | closed form | linear guardedness | RCCS imposes a *conditional* (given income) second-order version on output wires, through a driver rather than a projection. LEACE's footnote, and Ravfogel et al.'s log-linear guardedness (ACL 2023) [P], say linear guardedness does not stop multiclass or nonlinear heads |
| **INLP** (Ravfogel et al., ACL 2020, pp. 7237–7256) [P] | iterative nullspace projection | concept labels | projected representation | linear | unconditional | iterative | none (nonlinear recovery persists) | same limit |
| **This repository's LEACE-on-outputs evidence** [P, repo] | — | — | — | — | — | — | matched-removal benchmark: "LEACE passes its own check … yet nonlinear recovery barely moves". Invariant-baselines Q6: linear cross-covariance 4e-16 but positive additional recovery | Predicts RCCS's failure mode 1. The OUT_HGB gate is the right stop rule |
| **Ritov, Sun, Zhao**, *On conditional parity…*, arXiv:1706.08519 [P] | conditional parity x ⊥ a \| z (Def. 2.1) | a, z | — | a given a legitimate factor z | conditions on z | randomisation mechanism; kernel test | CP via randomisation | **RCCS's population target is a second-moment conditional parity**, with z = the income release, achieved deterministically |
| **Johndrow & Lum**, AoAS 13(1):189–220 (2019) (Crossref) [P][M] | transform covariates to (mutual) independence from Z, preserving information | **uses Z** in the transformation | adjusted features | Z | **chain**: each variable adjusted conditional on Z and previously adjusted variables | conditional quantile matching | independence (in distribution) of the adjusted set | The nearest "adjust one release conditional on the earlier ones". Differences: it uses Z itself, adjusts features, and targets full independence rather than "keep income's redundancy" |
| **Erdogdu & Fawaz 2015; Taylor et al. 2026; Sankar et al. 2013 Thm 3** (`LITERATURE_BASE.md` §3, §7) | min distortion s.t. incremental (collusion) leakage given earlier releases | known law; earlier releases | randomised channels | S or all of X | **release conditioned on earlier releases** | convex / BA programs | exact for a known finite law | **Same target functional**, I(S; R₂ \| R₁) = I(S; R₁, R₂) − I(S; R₁), in its information-theoretic form. RCCS uses a linear-Gaussian surrogate |
| **Li et al.**, TKDE 2012 (`LITERATURE_BASE.md` §7) | colluders' LLSE no better than the best single copy | covariance | correlated-noise copies | X (linear reconstruction) | colluding recipients | noise design | linear-estimation guarantee | **Same "pair ≤ best single" goal under a linear criterion**. Li uses shared noise; RCCS uses a deterministic x-driven shift |
| **Post-processing for parity**: Hardt et al., arXiv:1610.02413 [P]; Chzhen et al., arXiv:2006.07286 [P]; Xian et al., ICML 2023 (arXiv comment) [P][A] | equalised odds or demographic parity of outputs | **attribute at decision time** (Hardt "derived … from them and the protected attribute A"; Chzhen "sensitive attribute is available for prediction"; Xian "attribute-aware") | randomised or attribute-aware outputs | unconditional parity | none | LP / barycentre | optimality given a Bayes score | Our contract forbids SEX at deployment and randomisation. RCCS is attribute-blind, deterministic and conditional. Not the same problem |
| **Cross-fitting / partialling-out**: Chernozhukov et al., arXiv:1608.00060 [P] (journal [NV]) | orthogonalised nuisance estimation | — | — | — | — | K-fold cross-fitting; Robinson (1988) partialling-out | √n inference for the target parameter | RCCS's r = logit π̂ − ℓ̂(φ₁, φ₂) with out-of-fold π̂ is this construction, used as a *driver* rather than for inference |
| **Shamir**, *How to Share a Secret* (MIT course copy of the CACM 1979 article) [P][A] | k pieces reconstruct D, k−1 reveal nothing | — | shares | D | threshold access | — | information-theoretic | The canonical "parts silent, pair informative" structure (the XOR example in `LITERATURE_BASE.md` §8.3) that RCCS's contrast argument targets |
| **PCRL OWN-03**: erase layer + cross-purpose constraint (`results/combined_empirical_preparation_v1/notes/methods/CAPABILITY_TABLE.md` l.69) [repo] | dual on linear R² of h_concat ≤ 0.10 | SEX labels | representations | SEX, linear | **full coalition** | dual ascent | linear, in-sample | **Same coalition linear-R² criterion type** (see §2.4 item 2) |
| **Minimal task-wire adapter** (`results/pcrl_minimal_task_wire_adapter_v1/RECIPE_AND_CONTROLS.md` l.4–8) [repo] | occupation residual q = q₀ + (h64/s)Rᵀ + b_res, income frozen, critics | SEX via critics | occupation probabilities | SEX | pair critic in J arms | Adam, trust region | none | **The same structure** (income frozen, low-rank shift of occupation logits) with a different driver (64 frozen trunk features, not π̂) and objective (critics, not closed-form R²) |

### 2.2 Ingredient classification

| Ingredient | Class | Evidence |
|---|---|---|
| Income frozen; occupation designed given the actual income release (sequential, look-ahead) | **established technique** | Taylor; Wang & Fung; repository S12 arms; minimal adapter |
| Target = incremental SEX information of occupation given income | **established** (IT form) | Erdogdu & Fawaz; Taylor; Sankar Thm 3; repository RCSR/OWN-09 |
| Linear + quadratic ridge-R² criterion, recomputed in closed form | **established technique** | LEACE-style guardedness; PCRL OWN-03 coalition R² |
| Zero *partial* (not marginal) covariance target, c₂ = Σ₂₁Σ₁₁⁻¹c₁ | **established algebra**; as a notion, conditional parity (Ritov et al.) | partial-covariance identities. "Amplification law" = Schur complement, not new; designer 2 does not claim it |
| Cross-fitted propensity residual (partialling-out) | **established technique** | Chernozhukov et al.; Robinson |
| SLSQP over 30 parameters under cross-fitted utility budgets | **engineering** | — |
| **Explicit SEX-propensity driver feeding a deterministic, attribute-blind output shift to enforce a conditional second-order constraint** | **potentially new combination, unproven** | Not found in the sources read. Nearest: the minimal adapter (non-output driver, critic-trained); Johndrow–Lum (Z-driven conditional transforms); attribute-aware post-processing (Hardt/Chzhen/Xian). No priority claim |

### 2.3 Closed lanes

- **Small adapters.** Overlap is substantial.
  - The minimal adapter froze income and learned a 325-parameter (5×64+5) occupation shift that is linear in the frozen trunk's 64 features. The two-recipient adapter's counts (65 = 64+1 income, 325 occupation) fit the same parameterisation (from the task-headroom table; not separately verified).
  - Both reported pair shortfalls (0.0049 and 0.0036 vs 0.020).
  - RCCS differs in its driver (π̂ residual, a SEX-supervised nonlinear function of x) and its objective (closed-form pair R² vs critics).
- **Conditional / coalition constraints.**
  - OWN-03: a full-coalition linear-R² constraint on representations.
  - OWN-09: an AB coalition CMI on a fitted finite model.
  - The RCSR stochastic channel's view-conditional budgets: "the mechanism never constrained anything".
  - Conditional budgeted release (LP kernels; S12 lower than W).
  - Conditioning on another release is therefore not new to the repository. Conditioning on *released outputs*, rather than task labels, is correctly distinguished from the conditional-kernel lane.
- **Linear erasure on outputs.** The LEACE benchmark campaigns (excluded from the helper ledger) and invariant-baselines Q6 show linear guardedness leaves nonlinear recovery.
- **Not found in the closed-lanes table:** a pair-only objective with income frozen. Designer 2's MIN / NR_OCC / COND_PP / COND_SHUF controls are the right isolation design.

### 2.4 Citation and scope errors in RCCS

1. **§3.7, small adapters: "had no SEX-propensity driver. In the exact law that makes them post-processing (COND_PP)".** This is inaccurate for the minimal adapter.
   - Its residual is (h64/s)Rᵀ, a function of frozen *trunk features* that carry input information beyond the released wires.
   - So it was **not** post-processing of the released outputs: it already had a non-output driver, though critic-trained and without an explicit propensity.
   - The correct distinction is "explicit, SEX-supervised propensity driver plus closed-form objective".
2. **§4 objective.** Because f₁ = T₁ is frozen, R²_cv(S ~ φ₁) is a constant. Minimising Q is minimising the **pair (coalition) ridge R²** plus the ridge term.
   - The "partial information" framing is a correct reading, but the objective is the coalition linear/quadratic R² with income frozen.
   - Cite PCRL's own coalition linear-R² constraint (OWN-03) as the in-repository precedent.
   - The genuine target difference from NR_OCC is that the own-view occupation term is dropped.
3. **§3.5 "exact best ridge critic … no critic lag".** This is true within the linear/quadratic class.
   - Cite LEACE's footnote, Ravfogel et al. 2023 (log-linear guardedness) and the repository's matched-removal and invariant-baselines results for the nonlinear limit.
   - Designer 2's failure mode 1 and OUT_HGB gate acknowledge it.
4. **Novelty of conditioning.** "Conditions on the other recipient's released output" is correct as a distinction from the label-conditional kernel lane. Conditioning a release on earlier releases is itself established (Erdogdu & Fawaz, Taylor, Sankar Thm 3, Wang & Fung; repository RCSR/OWN-09).
5. **Fairness scope.**
   - The deployed occupation map uses an explicit SEX propensity. That is Lipton et al.'s indirect treatment disparity in its plainest form.
   - It is allowed by the release contract (a function of the six fields), and designer 2 disclaims individual fairness.
   - Any paper text must state it. It also makes "purpose limitation" language inappropriate for RCCS.
6. **§3.4 and §10.** The Gaussian-equivalent reading d′² = 2Φ⁻¹(AUC)² of selected-reader AUCs is a modelling assumption (equal-variance Gaussian), as designer 2 notes. It is not a measurement of covariance.

---

## 3. Side-by-side

| | MMP (designer 1) | RCCS (designer 2) |
|---|---|---|
| Closest prior method | Žliobaitė 2017 "fit with s, replace by s̄β"; Pearl's adjustment formula; Pope & Sydnor (abstract) | incremental-leakage release design (Erdogdu & Fawaz; Taylor); conditional parity (Ritov et al.); LEACE-type linear criterion; repository minimal adapter and OWN-03 |
| SEX at training / deployment | training only / never | training only (π̂ and criterion) / never, but via an explicit SEX proxy π̂ |
| Release contract | primary, deterministic | primary, deterministic |
| Protected information | SEX, via the mixture-weight route only | SEX, linear + quadratic, conditional on income |
| Coalition handling | implicit (removes a term common to both outputs) | explicit (pair criterion with income fixed) |
| Optimisation | none (closed form) | SLSQP, 30 parameters, cross-fitted |
| Guarantee | utility-price identity / bound only. No privacy guarantee | none. Linear-Gaussian model predictions only |
| Novelty class | established technique, new endpoint | engineering combination; one possibly new combination (propensity driver), unproven |
| Overlap with closed lanes | none found | substantial (small adapters; coalition/conditional constraints; linear erasure) |

---

## 4. Sources newly opened for Phase 2

All URLs opened today. They are in addition to those in `README.md`.

**Anti-discrimination, causal and fairness sources:**
- https://www.aeaweb.org/articles?id=10.1257/pol.3.3.206 (Pope & Sydnor landing page, abstract)
- https://ideas.repec.org/a/aea/aejpol/v3y2011i3p206-31.html (RePEc record)
- https://www.aeaweb.org/articles/pdf/doi/10.1257/pol.3.3.206 (full text: login page, not obtained)
- https://arxiv.org/pdf/1708.00754 (Žliobaitė 2017)
- https://link.springer.com/article/10.1007/s10506-016-9182-5 (challenge page, not obtained)
- https://ftp.cs.ucla.edu/pub/stat_ser/R218-B.pdf (Pearl 1995)
- https://arxiv.org/pdf/1706.02744 (Kilbertus et al.)
- https://arxiv.org/pdf/1703.06856 (Kusner et al.)
- https://arxiv.org/pdf/1711.07076 (Lipton et al.)
- https://link.springer.com/content/pdf/10.1007/s10618-010-0190-x.pdf (Calders & Verwer: challenge page, not obtained)
- https://arxiv.org/pdf/2108.04134 (Kern et al., checked only for its Pope & Sydnor citation)

**Conditional-adjustment and post-processing sources:**
- https://arxiv.org/pdf/1703.04957 (Johndrow & Lum)
- https://arxiv.org/pdf/1706.08519 (Ritov, Sun, Zhao)
- https://arxiv.org/pdf/2211.01528 (Xian, Yin, Zhao)
- https://arxiv.org/pdf/2006.07286 (Chzhen et al.)
- https://arxiv.org/pdf/1610.02413 (Hardt, Price, Srebro)
- https://arxiv.org/pdf/1608.00060 (Chernozhukov et al.)

**Linear erasure and secret sharing:**
- https://aclanthology.org/2020.acl-main.647.pdf (INLP)
- https://aclanthology.org/2023.acl-long.523.pdf (log-linear guardedness)
- https://web.mit.edu/6.857/OldStuff/Fall03/ref/Shamir-HowToShareASecret.pdf (Shamir)

**Metadata pages:**
- arXiv abstract pages for 1703.04957, 1706.08519, 2211.01528, 1610.02413, 2006.07286, 1608.00060, 1711.07076, 1706.02744, 1703.06856, 1708.00754.
- Crossref title queries: Johndrow & Lum (10.1214/18-AOAS1201); the DML journal version was not found (NBER WP 10.3386/w23564 listed).

**Repository files read:**
- `results/pcrl_minimal_task_wire_adapter_v1/{RECIPE_AND_CONTROLS.md, RESEARCH_DECISION.md}`
- `results/combined_empirical_preparation_v1/notes/methods/CAPABILITY_TABLE.md`
- `origin/research/combined-matched-removal-benchmark-v1:results/combined_matched_removal_benchmark_v1/RESEARCH_DECISION.md`
- `origin/research/pcrl-invariant-baselines-v1:results/pcrl_invariant_baselines_v1/RESEARCH_DECISION.md`
