# Method specification: SEX-mixture marginalisation (MMP)

**Status.** MMP was selected for a data-free prototype, conditionally (`SELECTION_DECISION.md`). **Its pre-registered prototype partly contradicted its rationale (§9), so it is not described as ready for a real-data pilot.** The pilot is prepared and rehearsed on synthetic stand-ins, but not run and not recommended as a competitive test (`PILOT_PROTOCOL_DRAFT.md`).

**Origin.** Designer 1, `roles/designer_1/PROPOSAL.md` (sha256 `2b7273ae…ff368`). Corrections from the mathematical verifier and the prior-work reviewer are applied here.

**Classification.** MMP is an **established technique applied to a new endpoint**:
- fit with the sensitive attribute, then release the prediction averaged over that attribute with fixed population weights (Žliobaitė & Custers 2016; Žliobaitė 2017; Pearl's adjustment formula, 1995);
- the new endpoint is linked-coalition SEX inference from released task outputs under per-task utility caps.

No new algorithmic ingredient is claimed, and no novelty or priority is claimed. MMP is also not "fair" in any technical sense. It is a disparate learning process in the sense of Lipton et al., and it is not counterfactually fair.

## 1. Notation and contract

The notation follows `PROBLEM_SPECIFICATION.md`:
- S = 1 is female;
- π(x) = P(S = 1 | x) and π̄ = P(S = 1);
- q_i^s(x) = P(Y_i | x, S = s);
- the SEX-blind posterior is η_i = (1 − π) q_i^0 + π q_i^1.

The release contract is the primary one. Each recipient i receives R_i = (m̂_i(x), argmax m̂_i(x)), a deterministic function of the six permitted fields. No person's SEX is read, and there is no shared state, randomness or interaction.

## 2. Release

m_i(x) = (1 − w) q_i^0(x) + w q_i^1(x), with the fixed weight w = π̄ (registered).

- **Population view.** m_i is the Bayes posterior under the counterfactual law P̃(x, s, y) = P(x) P(s) P(y | x, s), in which the fields carry no information about SEX. Equivalently, it minimises any proper task loss under P̃.
- **Fitted version.** q̂_i^s are SEX-aware CatBoost teachers fitted on MODEL_FIT with SEX as an extra input. At release they are evaluated with the SEX slot set to 0 and to 1, then mixed in probability space.
- **Utility-only variant (reported, not nominated).** The Brier-optimal constant weight is w\* = E[π‖g_i‖²] / E[‖g_i‖²], where g_i = q_i^1 − q_i^0. Any constant w removes the inferred-SEX term equally; w\* only lowers the Brier price.

## 3. What is removed, and what is not

The decomposition η_i = m_i + Δ_i is exact, with **Δ_i = (π(x) − π̄)(q_i^1(x) − q_i^0(x))**. MMP deletes Δ_i, the component by which a SEX-blind posterior converts the fields' SEX-predictiveness into task predictions **through the mixture weights**.

**What MMP keeps:**
- the within-SEX task effects q_i^s(x) of every field. Fields such as hours, schooling and class of worker predict SEX and also predict the tasks within each SEX, so their SEX information remains in m. Linked recipients can still triangulate it.

**Where the removal is a population (Bayes) reduction:**
- only along directions where q_i^s(x) is constant, i.e. pure SEX proxies. There, cells merge and AUC*(S; V) falls to the SEX information of the task-relevant fields.

**Where it is only a learnability reduction:**
- everywhere else. With fitted continuous teachers, m̂ is essentially injective on the populated inputs, so its Bayes recovery equals AUC*(S; X), exactly as for the neural control and the trees. Any real-data benefit is therefore a reduction in what readers fitted on finite auxiliary data can learn, not a population guarantee.

**It can backfire (verifier M10).** When the SEX-blind posterior pools two cells because the π-term cancels a within-SEX difference, m separates those cells, and recovery *rises*. So the premise gate F1 is essential.

## 4. Utility price: exact forms

For any release f that is a function of X:
- Brier(f) − Brier(η) = E‖f − η‖²;
- NLL(f) − NLL(η) = E KL(η ‖ f).

For MMP these give:
- **summed Brier price:** E[(π − π̄)² ‖q^1 − q^0‖²], an exact identity;
- **NLL price:** E KL(η ‖ m) ≤ E KL(Bern π ‖ Bern π̄) = I(S; X), by data processing through the kernel s ↦ q^s;
- **second-order approximation (corrected):** ½ (π − π̄)² Σ_y (q^1_y − q^0_y)² / m_y. The denominator is the mixture m. The proposal's χ²(q^1, q^0) form was a notation error.
- **Accuracy:** no bound. Removing Δ can move argmaxes away from gendered classes (failure mode 2).

**Clause excess against the binding references.** For example, the Brier excess vs U0 is

price + (E‖m̂ − η‖² − E‖m − η‖²) − E‖U0 − η‖².

The middle term is estimation error; it shrinks when the SEX-aware components are fitted well. The last term is U0's own excess, i.e. the headroom. Whether MMP is utility-feasible on real data is open, and gate F3 decides it.

**Subgroup price (verifier M5, constant gap g).**
- Men's within-SEX summed-Brier increase: ‖g‖²[v(3π̄ − 1) + κ₃]/(1 − π̄).
- Women's: ‖g‖²[v(2 − 3π̄) − κ₃]/π̄.
- Here v = Var π and κ₃ is the third central moment of π.
- At π̄ ≈ 0.47 and κ₃ ≈ 0, women bear about 1.26× the population price and men about 0.77×. **This holds only for a constant gap.**
- For any gap, the exact decomposition is: men E[(π̄² − π²)‖g‖² | S=0], women E[((1 − π̄)² − (1 − π)²)‖g‖² | S=1]. Which group bears the price depends on corr(π, ‖g‖²).
- In the prototype's continuous laws that correlation was negative, and men bore 2.2–2.6× the 0.005 guard. **The subgroup guard is a structural risk for whichever group the gap concentrates on.** It must be predicted from the exact decomposition.

## 5. Training (access by role)

| Role | Use |
|---|---|
| MODEL_FIT | Features, both task labels and SEX. Per seed and task: one SEX-aware CatBoost teacher on [X, SEX], using the T_TREE registry (depth 4, learning rate 0.05, at most 800 trees, l2 3), with 15% household internal early stopping and a refit on all rows. π̄ is the MODEL_FIT female share. For the twin only: a 5-fold household cross-fitted CatBoost SEX classifier π̂(x). For gate G0 only: a SEX-blind teacher on the same split |
| CALIBRATION | Unused (Ucal only) |
| MODEL_SELECT | Stage-0 gates and the frozen selection rule (24-clause point feasibility, then the coalition OUT_LR proxy) |
| ATTACK_FIT / ATTACK_VALIDATION | Readers only |
| DEV_REPORT | Exploratory, already opened |
| SEALED_ASSESSMENT | Closed |

There is no grid, strength or checkpoint: MMP is the single registered nominee.

## 6. Inference

1. Encode the six raw fields with MODEL_FIT moments and levels.
2. For each task, evaluate the SEX-aware teacher at [x, 0] and [x, 1].
3. Mix with π̄.
4. Release the probability vector and its argmax, taking the first class on ties.

The deployment package refuses SEX and any other prohibited column. It contains SEX-conditional task models but no SEX classifier; the twin's π̂ is used only in evaluation.

## 7. Complexity

- **Training:** about the cost of T_TREE per task and seed, plus a SEX-blind twin teacher and five propensity folds for the controls. There is no adversarial loop.
- **Inference:** two oblivious-tree ensemble passes per task, O(2·T·depth) with T ≤ 800 and depth 4.

## 8. Claims, gates and falsifiers

**Claims, kept separate.**
- **Competitive:** coalition benefit vs NEURAL_RAW ρ1 E2, with lower bound > 0 on both seeds; nested material margin 0.020.
  - Designer's subjective forecasts after cross-critique: plain point superiority 0.25; material margin established ≤ 0.07.
- **Recipe:** coalition benefit vs the twin, lower bound > 0.020.
- **Recipient worsening:** upper bound ≤ 0.010.
- **Utility:** point feasibility and confidence, as separate fields.
- **Subgroup:** separate field.
- **Novelty:** none.

**Not claimed:**
- a population or Bayes reduction beyond merged directions;
- any fitted AUC as a floor;
- protection of attributes other than SEX;
- that true-label leakage is a floor;
- a joint component.

**Gates.** All run on MODEL_FIT and MODEL_SELECT before any reader is fitted (`PILOT_PROTOCOL_DRAFT.md`):

| Gate | Requirement |
|---|---|
| G0 | Positive held-out NLL gain from adding SEX for at least one task on both seeds. Without it, Δ ≈ 0 |
| G1 = F3 | All 24 MODEL_SELECT point clauses pass |
| G2 = F1 | Twin − MMP ≥ 0.02 on the frozen OUT_LR coalition proxy, both seeds |
| G3 | Twin − MMP ≥ 0.01 on an OUT_HGB proxy, both seeds (symmetric nonlinear gate) |

**Falsifiers measured in the pilot.**
- **F2:** MMP's pair AUC ≥ NEURAL_RAW's, or MMP's pair-minus-best-single exceeds NEURAL_RAW's by more than 0.02.
- **Learning curve:** a markedly steeper rise with auxiliary sample size than the comparators' would indicate obfuscation rather than removal.

## 9. Data-free prototype results

Unit `COORD_PROTOTYPE_MMP_V1` was pre-registered in `prototype/PROTOTYPE_PREREGISTRATION.json` (commit `998726d62`) before the run. Full tables are in `prototype/PROTOTYPE_REPORT.md`; the independent recomputation is in `roles/math_verifier/PROTOTYPE_VERIFICATION.md`.

**24 of 27 pre-registered checks held.**

**Confirmed: the merging leg.**
- Exact removal of pure-proxy SEX information at exactly the predicted price.
- L_PROXY: pair 0.767 → 0.500, feasible at the cap, equal to the global frontier.
- L_MIXED: pair 0.717 → 0.650 = AUC*(S; task-relevant field).
- L_CONT_PROXY: fitted pair 0.779 → 0.723. This is the Bayes merge; the price exceeds the strict caps but sits within cap + headroom.
- The expected limits all appeared: L_NEG (MMP infeasible), L_XOR (singles 1/2, pair unchanged at 0.85) and L_NOPROXY (no Bayes effect; a partial merge reaches 0.703 where MMP stays at 0.731).

**Contradicted: the learnability leg.** In the continuous law without pure proxies, MMP's fitted pair AUC equalled the twin's (0.7207 vs 0.7211, against the pre-registered ≥ 0.005 reduction), while MMP paid an occupation Brier price of 0.005. With readers near Bayes and an injective release, deleting Δ changes nothing a reader learns. Fitted SEX-aware trees are injective in practice, giving about 1.0 release pair per distinct input in the synthetic rehearsal. A real-data benefit would therefore have needed exactly this leg.

**Also observed.**
- The within-SEX burden fell on **men** in both continuous laws, against the constant-gap prediction for women; it fell on women in L_MIXED. The direction depends on the law.
- The matched-price local linear eraser had no effect, so the "deletion beats masking" comparison is inconclusive.

**Independent verification** (`roles/math_verifier/PROTOTYPE_VERIFICATION.md`): every number reproduces and there is no bug. The verifier also found:
- **The masking comparison was mis-specified (coordinator error).** The matched-price eraser was invertible, so deletion versus masking at matched price is untested.
- **The subgroup expectation was mis-specified.** Men's within-SEX occupation increase was 2.2–2.6× the guard in the continuous laws.
- **A registered post-hoc probe** (10 task-relevant fields, 3-dimensional release) showed **no definite sign**: MMP changed pair recovery by −0.033 in one draw and **+0.018 / +0.019** in two. The sign is set by whether field-mediated and direct SEX effects align.

**Verdict.** The rationale is contradicted where it matters for real data. Without merging structure, MMP's effect on coalition recovery is indeterminate in sign, and its subgroup price can exceed the guard. **MMP is not ready for a real-data pilot.**

## 10. Why MMP might beat the neural benchmark, and why it probably will not

**Why it might.** The neural control starts from U0 and moves only slightly: its only feasible configuration is 80 updates at the smallest strength. U0 estimates the SEX-blind posterior η, which contains the inferred-SEX term Δ in full. Δ is the most direct route by which the six fields' SEX information enters both outputs, and the same latent π(x) appears in each. If Δ carries a large share of what fitted readers learn from the linked pair, MMP deletes it at an exactly known price, with no critic to fool.

**Why it probably will not.** Hours, schooling and class of worker predict SEX *and* the two tasks within each SEX, and MMP keeps those within-SEX effects. If most of the learnable coalition signal lives there, MMP cannot reach the roughly 0.606 pair AUC needed for the material margin, or even beat NEURAL_RAW. Its utility price, decision flips and women's subgroup cost all grow with the same gap that would make it effective. No published number gives Δ's share, which is why the pilot is gated on it before any attacker is fitted.
