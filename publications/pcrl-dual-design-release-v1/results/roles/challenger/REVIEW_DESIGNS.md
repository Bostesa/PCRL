<!-- Saved verbatim by the coordinator at 2026-10-21 ~17:58Z from the challenger's hand-back message (the harness refuses report-file
writes by subagents). Author: the challenger. The hand-back's summary lines preceding the report are kept as its first section. -->

**Verdicts (challenger's hand-back summary)**
- **DA (STD): PROCEED_WITH_REQUIRED_CHANGES** (4, 6, 7, 8).
- **DB (PTC): PROCEED_WITH_REQUIRED_CHANGES** (2, 3, 14–18). If those cannot fit in 16 configurations, REDIRECT Designer B (finding 23).
- The two proposals are **not the same operation**, so neither designer is redirected for equivalence.

**Most important problems**
- The DA card draft no longer describes `ddr/da/std.py` (finding 7).
- PTC is the cell-free form of the closed PRC routing tilt. The control that closed PRC (HQ, matched here by ZTILT) must therefore be required, not optional (finding 2).
- PTC's gain is, by construction, first-moment obfuscation. It writes the person's own propensity into the vector, and its record shows the gain eroding as readers get more data (finding 3).
- Both SBC fallbacks need fixing. B's (SCHL, COW) cell label is trivially weak. A's label-derived attribute has no bound on its association with SEX (findings 6, 8, 15).

---

# REVIEW_DESIGNS.md: challenger review of the DA (STD) and DB (PTC) designs, campaign `pcrl_dual_design_release_v1`

Written 2026-10-21, 17:38–17:58Z (`date -u` 17:38:38Z at the start).

**What I read:**
- `CHALLENGER_BRIEF.md`, `DESIGN_BRIEF.md`, `PROTOCOL.md` (revised), `FAILURE_MAP.md`, statistical reviewer `RULES.md`.
- Designer A's `PROPOSAL.md` and `mech_sketch.py`; Designer B's `PROPOSAL.md`, `mech_sketch.py`, `LEARNING_CURVE*.json`, `REVISION_PROBE.json`.
- The DA card draft (`cards/DA_STD_v1.md`, file time 17:24Z), `ddr/da/std.py` (file time 17:38:53Z), `ddr/da/__init__.py`, `ddr/famkit.py`.
- The coordinator's label-free `MIXTURE_PROXY` and `PROPENSITY_STRUCTURE` (JSON and scripts), and `nrm/fa/prc.py` (`route`, `fit_teachers`).

**What I did not open:** no label file, nothing of MODEL_SELECT, no reader cache, no `.private` pickle. I ran nothing on real data.

**CPU:** about 16 CPU-s, one thread, all on invented data: A's selftest 3.5 s (PASS), B's selftest 12.1 s (PASS; "SBC on the permuted column: degenerate -> fallback"), `python3 -m ddr.da.std selftest` 0.1 s (`all_ok`).

## Cross-family findings

**1. Equivalence (ADVISORY; no redirect).** STD and PTC belong to one idea family: subtract the part of the posterior that is proportional to the propensity. They are not the same operation.
- **STD** (when the twin equals Σ_s π_s q_s): P = η − λ(π(x) − w)Δ(x), in probability space. Δ = q₁ − q₀ is person-specific and comes from the SEX-aware task models P(y | x, s).
  - Equivalently, the SEX mixture evaluated at the shrunk propensity (1 − λ)π + λw; that is exactly the coordinator's `MIXTURE_PROXY`.
  - No propensity is deployed. The released vector is a valid standardized posterior.
  - It keeps the within-SEX proxy effects by design (linear π-dependence 0.195 against the twin's 0.463).
- **PTC**: log r = log p − α(q(x) − q̄_k)W[k], in log space, with the propensity deployed at runtime.
  - By the tower identity, W[k] is the within-stratum regression slope of the centred log-probabilities on π.
  - It therefore cancels the whole first-moment association, mixture term and proxy (within-SEX) dependence alike.
- To first order, STD is the same shape with the structural direction Δ(x)/η(x) in place of the pooled statistical slope W[k]. The two have different estimands and use SEX differently: STD uses P(y | x, s); PTC uses P(s | x) and the covariances.
- Keep both: together they contrast implicit standardization with an explicit runtime propensity. The campaign summary must say that both families test one idea, so a joint negative closes one idea, not two.

**2. PTC against the closed recipes (REQUIRED, DB).**
- **Not FB2.** FB2's amplitude was N(0,1) noise. FB2's SEX-blind channel is, in effect, PTC's SHUF control (the SEX direction with a SEX-blind amplitude).
- **But PTC is the cell-free, continuous form of PRC's routing rule.**
  - `nrm.fa.prc.route` adds λ·w·(π·a + b) to each cell's task cross-entropy: the person's runtime propensity, linear, moving them against their SEX.
  - PTC's exponential tilt r ∝ p·exp(−α(π − c)W) is exactly argmin_r KL(r‖p) + α(π − c)⟨r, W⟩.
  - So PTC uses SEX information the same way as PRC. It differs only in the release form (continuous, closed form, no cells), which does remove measured failure 4 (finite releases favour table readers).
- **Consequence.** PRC's decisive failure was against HQ, the propensity replaced by its release-level cell mean (PRC_S2 vs HQ_S2 +0.0003). PTC's analogue of HQ is ZTILT (amplitude = P(SEX | the twin's release)), which B lists only as optional.
- **Required:** ZTILT becomes a required saved comparison at CAND's MODEL_FIT spend, in both versions (2 configurations), with a reading declared in advance:
  - PTC not resolved below ZTILT means PRC's failure is repeated.
  - PTC resolved below ZTILT means the gain needs propensity information beyond the release (see 3).
- STD is not a re-labelling. MMP was never fitted, and STD has no cells, noise, decoder, controller or leaf correction.

**3. Information removal or obfuscation (REQUIRED for DB, ADVISORY for DA).**
- Neither release is a function of the twin's release, so neither has a data-processing guarantee.
- PTC explicitly writes the deployed propensity into the vector, including its part beyond the release, with the sign that cancels first moments. The vector is injective on the support, which B concedes. The benefit therefore sits with first-moment (linear or smooth) readers and should erode as readers gain capacity and data. B's own record shows this (`LEARNING_CURVE.json`, V0, α 0.5):

| Reader-FIT people | Advantage over SHUF | Advantage over the twin |
|---|---|---|
| 2k | 0.0248 | 0.0317 |
| 4k | 0.0246 | 0.0316 |
| 8k | 0.0224 | 0.0296 |
| 16k | 0.0181 | 0.0253 |
| Denoised, 16k | 0.0101 | 0.0181 |

- **The plug-in world flatters PTC by construction.** Pseudo-SEX is drawn from the same q that PTC deploys. B's estimated-propensity probe (correlation 0.84) already halves the advantage over SHUF (0.024 to 0.012). The real advantage over a fair SBC is plausibly about 0.005, below the 2·se_agg resolution of about 0.010.
- **What separates the two:**
  - The release-side control is ZTILT/HQ (finding 2). It is a function of the twin's release, so it cannot add SEX information for any reader. The part of PTC's benefit beyond it is the part that has no removal guarantee.
  - The reader-side check is a data-rich reader, declared in advance and descriptive: readers (LOG_LR, HGB15, MLP64) trained on the 20,000 MODEL_FIT out-of-fold releases with MODEL_FIT SEX and scored on the DDR1 SCORE parts. This needs an `ACCESS_POLICY` entry before registration. Report it beside D2.
  - Labelling rule: a D2 pass whose data-rich benefit over ZTILT, or over the bracketing control, is ≤ 0 in either version mean is labelled READER_BUDGET_DEPENDENT and never called information removal.
- For STD the same check is advisory. STD deploys no propensity. Designer A's 16k reader kept most of the effect (pair −0.011 against −0.013 at 4k), and the coordinator's HGB-regression proxy shows −0.0097 pair against the twin for occupation-only STD.

**4. Shared twin and disclosure of the SPEND path (REQUIRED disclosure; ADVISORY sharing).**
- DA_TWIN and TW_B are the same cached `TASK.pkl` teachers. Save them once as one shared CTL twin; this saves 2 configurations and evaluation work.
- One CRS ladder may serve both families' SPEND paths (PROTOCOL 4: a shared CTL path built on that family's TWIN). The bracket points are still chosen per family; aliases are counted once.
- **Required:** check whether the CRS rungs (24, 128) and (16, 64) are bitwise the published FRC TWB / B2TWIN wires. If they are, their MODEL_SELECT outcomes are already known (the 16 × 64 twin had pair 0.6555). Disclose this in PROTOCOL section 0 and on the cards, and count them as inherited.

**5. SPEND path against continuous candidates (ADVISORY).**
- Finite coarsening leaks more to table readers (FRC: the 16 × 64 twin 0.6555 against the continuous U0 0.6497). The CRS bracket points will probably leak at least as much as the continuous TWIN, so SPEND_FRONTIER is structurally easy for both continuous candidates. Only RULES 3(d), the non-table winner, guards against this.
- Say so in both cards' predictions.
- If budget allows, add one continuous, non-invertible SEX-blind rung, such as SEX-blind rank truncation of the twin's centred log-probabilities (principal components on MODEL_FIT out-of-fold, dropping the lowest-variance directions).

**6. SBC fallback attribute, principle (REQUIRED for both; specifics in 8 and 15).** A declared fallback must be:
- **(a) approximately SEX-independent**, by a rule declared in advance: among a fixed list of seeded candidates, take the one minimizing the size-weighted |Cov(attribute, SEX)| on MODEL_FIT; accept only if |corr(A, S)| ≤ 0.03; record the cross-fitted AUC of A's fitted propensity for SEX.
- **(b) of the same type as SEX:**
  - predictability from x comparable to SEX's (cross-fitted propensity AUC about 0.68);
  - entanglement with the twin's release comparable to SEX's (R² of the propensity's logit on the pair log-probabilities about 0.71);
  - for STD, an association with y given x, so that Δ_A ≠ 0.
- **A concrete shared construction (PSEUDO2):** A_i = 1[u_i < σ(γ(β·z(x_i) + g(y_i)) + c)], where
  - g is the logit of the label-cell SEX shares, permuted under rule (a);
  - z(x) is a keyed random linear score of the standardized compact fields, residualized by OLS on logit π̂_S and standardized;
  - β sets the x-part's spread to SEX's, c matches the SEX share, and γ is chosen by the spend-reaching rule;
  - the two numbers in (b) are recorded beside SEX's.

## DA (STD): card draft and `ddr/da/std.py`

**7. The card does not describe the code (REQUIRED).**
- The card's SBC path is UND (early stopping). Its SPEND path is the inherited FRC TWB_C128, TWB_C64, ORD_S2 and ORD_S3, with 7 configurations per version.
- The code's SBC path is permuted SEX with the PSEUDO_γ fallback; its SPEND path is a new CRS ladder (24, 128), (16, 64), (8, 32) with two bracket points; UND is descriptive only; and it saves 8 configurations per version (16 in all).
- The card's inherited SPEND points also break PROTOCOL 4 (iv)–(v): they were fitted and read on MODEL_SELECT earlier and are not chosen by the bracket rule.
- Before the commit, rewrite card sections 5 and 7 from the code: the controls, the 16-configuration count, the CPU estimate, and the predictions against SBC/SPEND instead of UND.

**8. PSEUDO_γ fallback (REQUIRED, part a; ADVISORY, part b).**
- **(a)** A random permutation of the ten cell shares gives A a random correlation with SEX through the labels; this can be large. Apply finding 6(a) and record the correlation.
- **(b)** In the population, A's propensity is Σ_y P(y | x)ρ(y), a function of the joint task posterior, and Δ_A is a function of it too.
  - So the SBC is close to a re-labelling of the twin plus estimation noise. The predicted SBC pair AUC is about TWIN's, and SBC_FRONTIER adds little beyond the TWIN comparison.
  - Adding the x-part of PSEUDO2 makes it a real same-form control. If this is not adopted, state the limitation and the prediction on the card.
- The permuted-SEX first try will almost surely not reach CAND's spend (a noise column gives q₀ ≈ q₁). That is fine, but predict it.

**9. Tuning effort (ADVISORY).** The SBC teachers use CAND's iteration counts instead of their own inner-CV T* on [Z, A]. Either run the same inner-CV rule (about 10 more fits per attribute tried) or record the inner-CV curve at the fixed T.

**10. Revisions.**
- **R1 trigger (a) (REQUIRED, minor):** it refers to "twice the plug-in spend", but there is no income plug-in value because no income cache exists. Restrict (a) to occupation, or define the income reference.
- **R1 trigger (b) and R2:** R1(b) is triggered by a MODEL_SELECT outcome, so R1 is adaptive and must be labelled so. R2 is MODEL_FIT-observable and bounded.
- **R2 pooling (ADVISORY):** it creates exact top ties (within 1e-9) on exactly the rows whose argmax STD moved, which are the people with extreme π − w. Tree and table readers can detect this signature. Address it in the R2 card, for example with a margin-preserving variant.

**11. Selection (ADVISORY).**
- w (31 grid values) and the λ guards are estimated on the same out-of-fold people, so the men's guard is optimistic at λ*. Cross-fit w or report the optimism.
- Record CAND_STRENGTH_DIFFERS_BY_VERSION when λ* differs between versions.
- The selection is MODEL_FIT-only and declared in advance.

**12. Utility (ADVISORY).**
- **Most likely binding:** the MODEL_FIT S3 occupation-accuracy clause at λ = 1 (0.0039 + 2 × 0.0016 = 0.0071 against 0.008, before household clustering), then the MODEL_SELECT men's occupation guard in version 1.
  - That guard: twin −0.0023 against U0; STD's plug-in spend is +0.0014 at w 0.45 and +0.0052 at the SEX-share weight.
- Income is untested because the SEX-aware income teacher is new.
- Expected λ* is 0.75–1.0. G1 is credible at p ≈ 0.75.

**13. Recovery and other points.**
- What remains after STD: within-SEX proxy effects (42% of the linear π-dependence) that nonlinear readers can reach through the x-structure, and pair re-assembly of proxies across recipients.
- Table (LOOKUP) readers are weak against a continuous release.
- STD_OCC measures leakage transfer between recipients.
- Novelty is NOT_CLAIMED, correctly (Pope and Sydnor 2011).
- Optional (ADVISORY): one UND point read on MODEL_SELECT (2 configurations) for the "weaker model leaks less" explanation.

## DB (PTC) proposal

**14. One CAND (REQUIRED).** PTC_S2 and PTC_S3 cannot both be candidates. Choose one budget before the fit, and choose the bracket on that CAND. PTC_S2 may be saved as another strength (2 configurations) or dropped. The added task-Brier clauses are stricter than the chosen S2/S3 budget, which is fine, but declare them as a deviation.

**15. SBC (REQUIRED).** The (SCHL, COW) Bernoulli(0.5) cell fallback is not the same type of attribute:
- q_A ≈ A, a two-valued step in two fields with AUC about 1, versus SEX's 0.68 continuous six-field propensity. The tilt becomes a binary, cell-constant ±α/2 shift along a direction unrelated to SEX (mostly the schooling direction).
- Readers undo such a shift easily, and "α ≤ 50" reaches the spend only through very large shifts on part of the support. The control is trivially weak.
- **The fair control:** PSEUDO2 (finding 6), with
  - q_A an HGB with the SEX-side settings;
  - W_A and q̄_A from the same rule;
  - α on a 6-point grid (the same size as CAND's) scaled by a declared MODEL_FIT factor so that its top reaches CAND's spend + 2 SE;
  - two bracket points.

**16. Sketch inconsistency (REQUIRED).** The proposal says fall back when Cov(q, s) ≤ 1e-4, but `fit_from_stats` raises only when it is ≤ 0. With permuted SEX, a tiny positive covariance makes W a ratio of noise terms, and the result is a meaningless control. Implement the declared threshold, also per stratum. Shrink W[k] toward the global vector for strata of 620–700 people (ADVISORY).

**17. ZTILT required; SHUF advisory.** ZTILT is REQUIRED (finding 2). SHUF is ADVISORY but high priority: it carries FB2's lesson, the same SEX direction with a SEX-blind amplitude (2 configurations).

**18. Data-rich reader check and labelling (REQUIRED).** As in finding 3.

**19. Pair re-assembly (ADVISORY).** The global income tilt does not cancel Cov(c(r_I), S | the released occupation decision), and a linked reader sees that decision (Simpson's paradox). The joint-decision strata could therefore be the revision-0 operation. As written, R2 is triggered by a MODEL_SELECT outcome, so it is adaptive. R1 is MODEL_FIT-triggered and bounded, but its probe already shows the men's guard at 0.0062 (above S3's 0.0045) and 7.6% occupation decision flips, so expect it to be infeasible.

**20. Utility (ADVISORY).** The binding clauses are income Brier and accuracy (plug-in at the S3 point 0.0037 / 0.0033; with the estimated propensity 0.0052 / 0.0045), then the men's guard (0.0023–0.0031). Realistic α* ≈ 0.35. The selection is MODEL_FIT-only and declared in advance.

**21. Predictions and novelty (ADVISORY).** P4 is optimistic: the plug-in world is circular, the estimated propensity halves the effect, and readers erode it. My estimate against a fair SBC is +0.000 to +0.012, resolved with p ≈ 0.25. Novelty NOT_CLAIMED is correct. The closest published work is AttriGuard (Jia and Gong 2018), an obfuscation defence, and LEACE (first-moment erasure); the closest in this repository is PRC, not RCCS.

## Budget (finding 22)

**22. Configurations and CPU (ADVISORY).**

| Item | Configurations |
|---|---|
| Shared CTL twin | 2 |
| DA: STD, SBC × 2, SPEND × 2, COMP, STD_OCC (7 × 2 versions) | 14 |
| DB: CAND, SBC × 2, SPEND × 2 (aliases with DA counted once), ZTILT, SHUF (7 × 2 versions) | 14 |
| **Total** | **about 30** |
| Left for revisions and the replacement family | about 18 |

- Each family's complete reading is therefore 14–16 configurations (the same twin counted in both).
- CPU: DA fits 0.3–0.7 h (up to 4 SBC attribute tries × 12–22 HGB fits per version); DB fits about 0.3 h; reader grid about 30 × 8 × 16 s ≈ 1.1 h; checks about 0.3 h. Roughly 2–3 CPU-hours of the 12.

**23. Fallback redirect for Designer B.** If findings 2, 3 and 14–18 cannot fit in 16 configurations, redirect Designer B to an untested operation that does not deploy a runtime propensity: coalition-aware correlated randomness across the two recipients, or randomized response on the decisions.

## Verdicts

- **DA (STD): PROCEED_WITH_REQUIRED_CHANGES.** Required: findings 7, 8(a), 10 (R1 income reference), and 4 (disclosure).
- **DB (PTC): PROCEED_WITH_REQUIRED_CHANGES.** Required: findings 2 (ZTILT), 3 and 18 (data-rich reader check and labelling rule), 14 (one CAND), 15 (fair SBC), and 16 (threshold). Otherwise REDIRECT as in finding 23.
