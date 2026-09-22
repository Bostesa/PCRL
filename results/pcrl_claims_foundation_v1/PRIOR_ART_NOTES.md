# Prior-art verification for the PCRL claims (a)–(e)

Checked 2026-09-22. Read-only review. No paid APIs were used and no one was contacted. Nothing was committed.

**Method.** Each primary source was downloaded as the arXiv or proceedings PDF (the version is given below), converted with `pdftotext`, and read at the cited section, theorem or equation. Venue and pagination were confirmed from the arXiv abs page, proceedings page, dblp or ML Anthology listings where noted. A source is marked **UNVERIFIED** when only its abstract or a search snippet was seen.

**Claims under audit.**
- (a) Purpose-specific sharing: each recipient has permitted tasks and forbidden sensitive attributes.
- (b) A new channel appended beside frozen service outputs H, audited for additional disclosure relative to the H-only view.
- (c) Recipient-specific and coalition (combined-access) disclosure accounting.
- (d) The aggregate one-hot R² identity (a prevalence-weighted combination of per-class OvR R²), rare-class hiding, and exact-zero cross-covariance composition.
- (e) A finite stochastic release: a kernel Q from a finite code T to tokens Z, minimising linear expected distortion subject to I(S;Z|C_r) ≤ δ_r, convex for a fixed joint table.

Documents reviewed:
- The current manuscript `PCRL-t2-finish/papers/pcrl_satml_final_v1/main.tex` (line numbers are cited below) and its `references.bib`.
- `PCRL-t2-finish/results/pcrl_submission_review_v1/NOVELTY_AND_SCOPE.md`.
- `PCRL-terminal-3-claims/results/pcrl_task_directed_release_v1/{LITERATURE_SOURCES.json, NOVELTY_AND_ASSUMPTIONS.md}`.
- The NeurIPS 2026 submission PDF (author-local copy of the submitted PDF), §2 (related work) and §3.

---

## Part 1. Verified citation blocks

### 1. LEACE
- **Citation:** Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff, Biderman. *LEACE: Perfect linear concept erasure in closed form.* NeurIPS 36 (2023), Main Conference Track.
- **URLs:** https://proceedings.neurips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract-Conference.html (proceedings page confirmed); arXiv:2306.03819.
- **Opened:** arXiv PDF, §§2–4, App. B–C.
- **Load-bearing statements:**
  - Setting (§2): the labels Z are one-hot with every class probability P(Z=j) > 0, and X has a finite first moment.
  - Def. 2.3: *linear guardedness* means that no affine predictor η(x) = b + Wx beats the best constant predictor, for **every nonnegative loss that is convex in the prediction**.
  - Thm 3.1: equal class-conditional means imply that the trivially attainable loss cannot be improved, for any loss convex in the linear prediction. The proof uses Jensen's inequality.
  - Thm 3.3: the converse direction for cross-entropy.
  - Thm 3.4: equal class-conditional means ⇔ Σ_XZ = 0 (zero cross-covariance).
  - Thm 4.1: an affine map r(x) = Px + b guards Z iff colsp(Σ_XZ) ⊆ ker P.
  - Thms 4.2–4.3: the least-squares eraser P* = I − W⁺P_{WΣ_XZ}W, with b* = E[X] − P*E[X]. Finite second moments are assumed, stated for the population covariance.
  - App. C: linear guardedness ⇔ statistical parity of every linear predictor's *real-valued output*.
  - The guarantee **does not cover 0–1 accuracy of thresholded classifiers**, because the 0–1 loss is not convex.
- **Relation to our claims:**
  - (d): exact-zero composition follows immediately. Cov([h₁;h₂], S) is the stacked block matrix, so zero blocks give zero, and Thm 3.4 turns that into guardedness.
  - The manuscript's 20-row counterexample (thresholding at h > 0 reaches 18/20 correct with equal conditional means) is consistent with LEACE, since that is a 0–1 statement.
  - The manuscript's narrowing to "squared loss only" is *too* narrow; see M6.

### 2. SPLINCE
- **Citation:** Holstege, Ravfogel, Wouters. *Preserving Task-Relevant Information Under Linear Concept Removal.* NeurIPS 38 (2025), Main Conference Track.
- **URLs:** https://proceedings.neurips.cc/paper_files/paper/2025/hash/26dbc8565974cffe8a44731aa09b2aa8-Abstract-Conference.html (confirmed); arXiv:2506.10703.
- **Opened:** arXiv PDF, §3.1–3.3.
- **Load-bearing statements:**
  - Thm 1 minimises E‖Px − x‖²_M subject to the kernel constraint PΣ_{x,z} = 0 (erasure of concept z) and the **range constraint PΣ_{x,y} = Σ_{x,y}** (exact preservation of covariance with the task label y).
  - Solution: P* = W⁺V(UᵀV)⁻¹UᵀW.
  - Assumptions: E[x] = 0, finite second moments, nonzero covariances, and U^⊥ ∩ colsp(WΣ_{x,y}) = {0} (the concept and task covariance directions do not perfectly overlap).
  - **It uses both concept labels z and task labels y** (covariances). It preserves only the task *covariance*, not task sufficiency.
  - Thm 2 (§3.2): projections with the same kernel give identical predictions after unregularised last-layer retraining. §3.3: the range matters for regularised or frozen heads.
- **Relation to our claims:** an erasure baseline that the releaser can apply to a representation it owns. It is not an antecedent for (b), (c) or (e).

### 3. The Sadeghi–Boddeti closed-form line
**SARL.**
- **Citation:** Sadeghi, Yu, Boddeti. *On the Global Optima of Kernelized Adversarial Representation Learning.* ICCV 2019, pp. 7971–7979 (pages from secondary listing).
- **URL:** arXiv:1910.07423.
- **Opened:** arXiv PDF, §3.2 and App. A.4.
- **Statements:**
  - Linear or kernel encoder; the target and adversary are MSE (kernel) regressors.
  - Thm 2: the Lagrangian objective (9) is neither convex nor differentiable in G_E.
  - Thm 3 (Thm 7 in the appendix): the minimum is β₁+…+β_γ over the **negative eigenvalues** of B = LₓᵀQₓ⁻ᵀ(λC_sxᵀC_sx − (1−λ)C_yxᵀC_yx)Qₓ⁻¹Lₓ, with γ = min{r, #negative}.
  - The optimal G_E is spanned by the eigenvectors of the negative eigenvalues. The text notes that dropping zero-eigenvalue directions gives the least rank. **This is eigenvalue-sign rank selection.**

**OptNet-ARL.**
- **Citation:** Sadeghi, Wang, Boddeti. *Adversarial Representation Learning With Closed-Form Solvers.* ECML-PKDD 2021, LNCS pp. 731–748, DOI 10.1007/978-3-030-86520-7_45 (dblp/Springer listing).
- **URL:** arXiv:2109.05535.
- **Opened:** arXiv PDF, §4.
- **Statement:** Thm 4 applies when z is a *free vector* (not tied to the encoder) and both predictors are linear regressors. The optimal embedding dimension is then the **number of negative eigenvalues** of B = λS̃ᵀS̃ − (1−λ)ỸᵀỸ. With a deep encoder this serves as an upper bound on dimensionality. The adversary and target are kernel ridge regressors.

**K-TOpt.**
- **Citation:** Sadeghi, Dehdashtian, Boddeti. *On Characterizing the Trade-off in Invariant Representation Learning.* TMLR 2022.
- **Venue check:** confirmed via ML Anthology and the authors' page. The arXiv PDF header still reads "Under review as submission to TMLR". OpenReview was blocked by a browser check.
- **URL:** arXiv:2109.03386 (v4, 2022-12-22).
- **Statements:**
  - Eq. (7): Dep(Z,S) = Σ_j Σ_{β∈U_S} Cov²(Z_j, β(S)). This is an HSIC-like, asymmetric measure, **unconditional**.
  - Thm 4: a closed-form RKHS encoder from the top-r eigenvectors of the generalised eigenproblem (14).
  - **Cor. 4.1:** r_Opt(λ) = **the number of non-negative eigenvalues** of (14). The sign convention is the reverse of SARL's (maximisation versus minimisation).
- **Relation to our claims:** all three are owned-representation methods. The eigenvalue-sign rank selection attribution in NOVELTY_AND_SCOPE is **accurate**, with OptNet-ARL only under the Thm 4 assumptions. None is an antecedent for (b), (c) or (e).

### 4. U-FaTE
- **Citation:** Dehdashtian, Sadeghi, Boddeti. *Utility-Fairness Trade-Offs and How to Find Them.* CVPR 2024. The arXiv comment reads "IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024".
- **URL:** arXiv:2404.09454 (v2, 2024-04-24).
- **Opened:** main text §§3–4 and supplementary implementation details.
- **Dependence measure:**
  - Defs 1–2 define the Data-Space (DST) and Label-Space (LST) trade-offs using Dep(f(X), S | Y = y).
  - The conditioning is on the **target label Y = y**, which encodes *separation*-based fairness (equal opportunity, equalized odds). Dep(·,·|∅) reduces to the unconditional measure for demographic parity.
  - Eqs. (3)–(4): the K-TOpt measure evaluated on X_c ∼ P(X|Y=y), S_c ∼ P(S|Y=y), estimated as (1/n²)‖ΘK_{X_c}HL_{S_c}‖²_F.
  - The paper itself says: "K-TOpt optimizes an unconditional dependence measure… our formulation also requires conditional independence".
- **Solver:**
  - Thm 1: the global optimiser of (5) over the fair-encoder layer is the eigenvectors of the **r largest eigenvalues** of the generalised eigenproblem (6).
  - **There is no eigenvalue-sign rank selection.** Supplementary implementation details fix r = c − 1, where c is the number of target classes.
  - The full model is not purely closed-form. A feature extractor is trained by minibatch SGD, alternating with the closed-form encoder (§4.2, Fig. 5).
- **Relation to our claims:**
  - Calling it "the conditional extension" is defensible only in the narrow sense of *label-conditional* dependence for fairness.
  - It is **not** conditional on an adversary's side information or on existing public outputs, so it is not an antecedent for (b). See M3.

### 5. Rassouli & Gündüz, *On Perfect Privacy*
- **Citation:** B. Rassouli, D. Gündüz. IEEE JSAIT 2(1):177–191, 2021, DOI 10.1109/JSAIT.2021.3053432 (volume, pages and DOI from dblp/ITSoc listing).
- **URL:** arXiv:1712.08500 (v8, 2021-01-21; "the longer version of the journal paper").
- **Opened:** §§II–III, Prop. 1, Thm 1, Lemma 1, eq. (9), Remarks 1–2, Cor. 1.1–1.2.
- **Notation (reversed relative to ours):** **X = private, Y = useful**, U = released, W = observed.
- **Load-bearing statements:**
  - **Prop. 1:** with finite alphabets and full-support marginals, perfect privacy (X ⊥ U, Y not ⊥ U) is feasible **iff** dim(Null(P_{X|W}) \ Null(P_{Y|W})) ≠ 0. This is the nullspace criterion, for **unconditional** independence.
  - **Thm 1 (output perturbation, W = Y):**
    - The optimal perfect-privacy mechanism for **mutual-information utility** g₀ = sup I(Y;U) solves a standard LP, and |U| ≤ nul(P_{X|Y}) + 1 suffices.
    - The LP arises from minimising the *concave* H(Y|U) over the perfect-privacy polytope S_{X,Y}. The optimum is at extreme points (Lemma 1), which leaves an LP over extreme-point weights (eq. 9).
  - **Remark 2:** in the general W model, with MI, **MMSE E[(Y−U)²] or error probability** as utility, the perfect-privacy problem "also simplifies to an LP". This is asserted with "It can be verified" and no proof.
  - There is **no conditioning on side information** anywhere in these results.
- **Relation to our claims:**
  - Prior art for the unconditional zero-leakage feasibility geometry and the MI-utility LP.
  - A zero-budget, linear-cost version appears only in Remark 2, and only unconditionally.
  - Our conditional zero-budget problem (S ⊥ Z | C, with Z–T–(S,C)) is an LP by elementary linearity: p(s,c,z) and p(s|c)p(c,z) are both linear in Q. That is **not** stated in Rassouli–Gündüz. See M1 and M2.

### 6. Privacy funnel and "conditional privacy funnel" variants
**Privacy funnel.**
- **Citation:** Makhdoumi, Salamatian, Fawaz, Médard. *From the Information Bottleneck to the Privacy Funnel.* IEEE ITW 2014, pp. 501–505.
- **URL:** arXiv:1402.1774 (v5).
- **Opened:** §§II–III.
- **Statements:**
  - §II.C–D: when the distortion d(x,y) does not depend on the statistics, E[d] is **linear in P_{Y|X}**. So min ΔC s.t. E[d] ≤ D is convex whenever ΔC is convex. This is the Calmon–Fawaz framework they cite as [1].
  - §III.B: log-loss distortion equals H(X|Y), which is not linear in P_{Y|X}, and gives the constraint I(X;Y) ≥ R.
  - §III.C eq. (3): min I(S;Y) s.t. I(X;Y) ≥ R. The objective I(S;Y) is convex in P_{Y|X}. "**because of the constraint I(X;Y) ≥ R**, the Privacy Funnel (3) is not a convex optimization".
- **Verdict:** the manuscript's claim that the nonconvexity comes from the utility constraint is **verified accurate**.

**Conditional privacy funnel.**
- **CPF:** Rodríguez-Gálvez et al. (2021) is cited inside de Freitas & Geiger with L_CPF = I(s;z) − γI(z;x|s). The primary source was **not opened (UNVERIFIED)**.
- **CPFSI:**
  - **Citation:** de Freitas & Geiger, *Trustworthy Representation Learning via Information Funnels and Bottlenecks.* Machine Learning 114, 267 (2025).
  - **URL:** arXiv:2211.01446 (v2, 2025-11-05).
  - **Opened:** §§2–3, Def. 1.
  - Def. 1: L = I(s;z) − γI(z;x|s) − βI(y;z|s).
  - **Here "side-information" means the downstream *task label y* used as supervision (from IBSI), and the conditioning is on the sensitive attribute s.** It is not side information held by an adversary.
- **Side information at the adversary** (the relevant literature for (b)) is covered in sources 8, 9, 14 and 15.

### 7. Calmon et al., *Optimized Pre-Processing* (not cited in the manuscript)
- **Citation:** Calmon, Wei, Vinzamuri, Natesan Ramamurthy, Varshney. *Optimized Pre-Processing for Discrimination Prevention.* NIPS 30 (2017).
- **URLs:** https://proceedings.neurips.cc/paper/2017/hash/9a49a25d845a483fae4be7e341368e36-Abstract.html (confirmed); arXiv:1704.03354 (the arXiv title adds "Data").
- **Opened:** NeurIPS PDF, §§2–3.
- **Formulation:**
  - A randomized mapping p_{X̂,Ŷ|X,Y,D} over **finite** D, X and binary Y.
  - The **mapping sees the protected variable D and the label Y**. D is retained in the output.
  - Problem (6): minimise the distributional utility loss Δ(p_{X̂,Ŷ}, p_{X,Y}) (for example KL).
  - Constraint (i): discrimination control J(p_{Ŷ|D}(y|d), p_{Y_T}(y)) ≤ ε, with (3) J(p,q) = |p/q − 1| (the 80%-rule ratio). Alternative (2) is pairwise over d₁ and d₂.
  - Constraint (ii): pointwise individual distortion E[δ((x,y),(X̂,Ŷ)) | d,x,y] ≤ c_{d,x,y}, stated in (4).
  - §2 extends the constraint to conditioning on context variables B ⊂ X ("conditional discrimination").
- **Convexity:** Prop. 1 says (6) is (quasi)convex if Δ is (quasi)convex and J is quasiconvex in its first argument; for (2), J must be jointly quasiconvex.
- **Other results:** Prop. 2 gives mismatched-prior convergence rates for the empirical p_{D,X,Y}, assuming p_{Y,D} > 0. At apply time the mapping is marginalised over Y (eq. 9).
- **Relation to our claims:**
  - Close to (e) as a *convex program over a finite randomized preprocessing map fitted on an empirical joint law*.
  - It is **not the closest**. Its constraints are probability-ratio outcome parity, not mutual information; there is no adversary side information or coalition; the objective is distributional fidelity, not a frozen-decoder task loss; and the map uses D.
  - Calmon & Fawaz 2012 (source 12) and Taylor et al. 2026 (source 16) are closer to (e). It should still be cited.

### 8. Sankar, Rajagopalan & Poor (2013) and Liao et al. (2019)
**Sankar et al.**
- **Citation:** Sankar, Rajagopalan, Poor. *Utility-Privacy Tradeoffs in Databases: An Information-Theoretic Approach.* IEEE TIFS 8(6):838–852, 2013, DOI 10.1109/TIFS.2013.2253320.
- **URL:** arXiv:1102.3751 (v4). The arXiv title uses "Tradeoff".
- **Opened:** §§III–V.
- **Statements:**
  - One sanitised database is released to a user, and "every user is (potentially) also an adversary".
  - Utility is distortion; privacy is equivocation H(X_h | J, Zⁿ), with leakage bounded by I(X_h; Z) ≤ L.
  - The results are asymptotic rate–distortion–equivocation regions (Thm 1).
  - **§V.A models side information Zⁿ at the user.**
    - Thm 2: a statistically informed encoder, achieved by quantize-and-bin.
    - **Thm 3: the encoder *knows* the side information** ("sharing a related but differently sanitized view in the current interaction").
  - "Multiple legitimate consumers" appears in the abstract, but the formal model is **one** release against a user with side information. There are no per-consumer permissions and no coalition.
- **Relation to our claims:** a genuine antecedent for (b), meaning a release designed knowing what the recipient already holds. It is not multi-recipient.

**Liao et al.**
- **Citation:** Liao, Sankar, Kosut, Calmon. *Robustness of Maximal α-Leakage to Side Information.* ISIT 2019, pp. 642–646 (pages from secondary listing).
- **URL:** arXiv:1901.07105 (v2).
- **Opened:** §§I–III.
- **Statements:** Defs 3–4 define conditional (maximal) α-leakage given adversary side information Z. Thm 3: if Z ⊥ Y | X, then maximal α-leakage upper-bounds the conditional maximal α-leakage.
- **It designs no release and has no multiple consumers.** It is an adversary-side-information *leakage measure* paper.
- **Relation to our claims:** a measure-level antecedent for (b). The manuscript's grouping of it as "design a sanitised release for many consumers" is wrong (M4).

### 9. Side-information and coalition antecedents for (b) and (c)
**SRLIP.**
- **What it refers to:** **Side-channel Resistant Local Information Privacy**, from Lopuhaä-Zwakenberg, *The Privacy Funnel from the Viewpoint of Local Differential Privacy*, ICDS 2020 (PPODS track), per the arXiv comment.
- **URL:** arXiv:2002.01501 (v2).
- **Opened:** §§II–V.
- **Statements:**
  - Def. 3: for every output y, sensitive value s, **attribute subset J** and value x_J, e^{−ε} ≤ P(Y=y | S=s, X_J=x_J) / P(Y=y | X_J=x_J) ≤ e^{ε}.
  - That is pointwise privacy conditional on **every side-channel view an attacker might hold**.
  - Thm 3 is a composition theorem: per-attribute LIP with respect to all conditional priors gives Σε_j-SRLIP.
  - Thms 1–2 find optimal LDP/LIP protocols by vertex enumeration plus an LP.
- **Relation to our claims:** a stronger, worst-case, all-subsets version of the "conditioning view" idea in (b) and (c). Our C_r are two selected views under an *average* (CMI) criterion, which is strictly weaker.

**Issa, Wagner, Kamath.**
- **Citation:** *An Operational Approach to Information Leakage.* IEEE TIT 66(3):1625–1657, 2020 (pages from secondary listing).
- **URL:** arXiv:1807.07878.
- **Opened:** §IV.E.
- **Statements:**
  - Def. 6 and Thm 6: conditional maximal leakage L(X→Y|Z) given adversary side information Z.
  - Cor. 2: L(X→Y|Z) ≥ I(X;Y|Z); and L(X→(Y,Z)) ≤ L(X→Z) + L(X→Y|Z).
  - Lemma 6 is a composition lemma.
- **Relation to our claims:** a measure-level antecedent for (b) and (c).

**Zamani, Oechtering, Skoglund.**
- *Multi-User Privacy Mechanism Design with Non-zero Leakage*, IEEE ITW 2023, pp. 401–405 (secondary listing); arXiv:2211.15525. Abstract and §II opened.
  - K users; user i demands a sub-vector C_i of Y.
  - **One** disclosed U maximises a weighted sum of I(C_i;U) subject to I(X;U) ≤ ε.
  - There is a single common privacy constraint and no user side information.
- Their other papers (non-zero leakage, per-letter, direct-access, empirical-distribution variants) concern single-user, unconditional ℓ₁/χ²/MI leakage. They were seen in search results only (**UNVERIFIED** beyond titles and abstracts). I found **no** Zamani–Oechtering–Skoglund paper that designs a release conditioned on an adversary's side information in our sense. Their cache-aided variable-length coding papers involve user-side cache side information; these were not opened (**UNVERIFIED**).

**Li, Chen, Li, Zhang.**
- **Citation:** *Enabling Multilevel Trust in Privacy Preserving Data Mining.* IEEE TKDE 24(9):1598–1612, 2012 (pages from secondary listing).
- **URL:** arXiv:1104.0459.
- **Opened:** abstract and §§1–4 headings and text.
- **Statements:** differently perturbed copies go to miners at different trust levels. Jointly Gaussian noise is correlated across copies so that **any colluding collection of copies cannot reconstruct the data better than the best single copy in it** ("diversity attack").
- **Relation to our claims:** an antecedent for **(c)** coalition accounting. The protected object there is the original data under linear least-squares reconstruction, not a separate sensitive attribute.

### 10. Brief verifications
- **Madras, Creager, Pitassi, Zemel.** *Learning Adversarially Fair and Transferable Representations.* ICML 2018, PMLR 80:3384–3393; arXiv:1802.06309 v3.
  - §5.1: an (unnumbered) theorem says Δ_DP(g) of any binary classifier g is upper-bounded by the optimal group-normalised adversary objective.
  - §5.2 gives the analogous result for equalized odds and equal opportunity.
  - An owned-representation, adversarially trained baseline. The description in the manuscript is accurate.
- **Elazar & Goldberg.** *Adversarial Removal of Demographic Attributes from Text Data.* EMNLP 2018, pp. 11–21; arXiv:1808.06640.
  - Abstract, §5 and Table 3: the adversary reaches chance accuracy during training, while a post-hoc classifier still recovers the protected attribute substantially above chance.
  - The manuscript's "leaves recoverable information" is accurate.
- **Stadler, Kulynych, Gastpar, Papernot, Troncoso.** *The Fundamental Limits of Least-Privilege Learning.* ICML 2024, PMLR 235; arXiv:2402.12235 v2.
  - Def. 1 is unconditional least-privilege (LPP). Thm 1 shows it is incompatible with utility when P_{Y|X} is strictly positive.
  - **Def. 2 (LPP) bounds the attribute-inference gain from observing (Z,Y) *over the fundamental leakage of the task label Y*.** That is leakage *relative to a baseline view*. The paper ties this to **purpose limitation**.
  - Thm 2 is the corresponding trade-off.
  - Relevant to (a) and to the "relative to baseline" framing of (b).
- **Jovanović, Balunović, Dimitrov, Vechev.** *FARE: Provably Fair Representation Learning with Practical Certificates.* ICML 2023, PMLR 202:15401–15420; arXiv:2210.07213.
  - §5: *restricted encoders* map x to one of k cells (a finite code). Lemmas 5.1–5.3 give Clopper–Pearson-based high-confidence upper bounds on the demographic-parity distance of **any** downstream classifier.
  - §6 instantiates this with fair decision trees.
  - Relevant to (e)'s finite code T. FARE is deterministic and has no channel optimisation.

### 11. Purpose-specific and flexible fair representations
**FFVAE.**
- **Citation:** Creager, Madras, Jacobsen, Weis, Swersky, Pitassi, Zemel. *Flexibly Fair Representation Learning by Disentanglement.* ICML 2019, PMLR 97:1436–1445; arXiv:1906.02589.
- **Opened:** §§1, 4 and 5.1.
- **Statements:**
  - "each task may have a different task label y and sensitive attributes a".
  - The latent is [z, b], with one sensitive dimension b_i per attribute a_i.
  - §4: at test time, demographic parity with respect to a_i is obtained **by removing b_i** (or replacing it with noise). Conjunctions are handled compositionally by removing {b_i, b_j, b_k}.
  - It needs no sensitive attributes at inference.
- **Relation to our claims:** **the closest antecedent for (a)**, since different consumers get different sensitive-attribute subsets from one shared encoder.
  - There is no formal guarantee; the paper evaluates with post-hoc auditors.
  - There is **no coalition analysis**: two consumers given [z,b]\b_i and [z,b]\b_j jointly hold all dimensions.
  - There are no frozen prior outputs.

**Lechner, Ben-David, Agarwal, Ananthakrishnan.**
- **Citation:** *Impossibility results for fair representations.* arXiv:2107.03483 (2021). The NeurIPS submission cites it as 2022; that venue was not checked.
- **Opened:** abstract.
- **Statement:** no representation guarantees fairness of classifiers for different tasks. Label-independent demographic parity fails under marginal shift, and "except for trivial cases, no representation can guarantee Odds Equality fairness for any two different tasks, while allowing accurate label predictions for both".
- **Relation to our claims:** motivation for (a).

**Related work seen in search results only (UNVERIFIED).**
- Rodríguez-Gálvez et al. 2021 (CPF).
- "Robust Privatization with Multiple Tasks and the Optimal Privacy-Utility Tradeoff" (arXiv:2010.10081; one release, a set of possible tasks).
- Song et al. 2019, *Learning Controllable Fair Representations*.
- DProvDB (SIGMOD 2024; multi-analyst DP with collusion-aware provenance).

**Search result.** I found no primary paper matching the full combination: per-recipient permitted and forbidden sets, frozen pre-existing ML outputs as conditioning views, and a named coalition. That is not proof of priority.

### 12. Calmon & Fawaz (2012) — the closest antecedent for the convex program in (e)
- **Citation:** F. du Pin Calmon, N. Fawaz. *Privacy Against Statistical Inference.* 50th Allerton Conference, 2012, pp. 1401–1408, DOI 10.1109/Allerton.2012.6483382 (secondary listing).
- **URL:** arXiv:1210.2123 (v1).
- **Opened:** §§II–IV.
- **Statements:**
  - A privacy mapping p_{U|Y} is designed with the prior p_{S,Y} known to the adversary.
  - **Thm 1:** min I(S;U) s.t. E[d(Y,U)] ≤ Δ is a **convex program** in (p_{U|Y}, p_{U|S}), with the two coupled by a linear equality.
  - **Thm 2:** min expected distortion s.t. a max-information-leakage constraint is convex.
  - **Remark 1:** multiple distortion constraints are simply extra linear constraints.
  - Remark 3: the mapping may take S as input.
- **Relation to our claims:** the finite-alphabet "linear distortion versus MI leakage is convex" fact in (e) is theirs, in unconditional form.

### 13. Salamatian et al. (2015) — a finite code plus a convex channel
- **Citation:** Salamatian, Zhang, Calmon, Bhamidipati, Fawaz, Kveton, Oliveira, Taft. *Managing Your Private and Public Data: Bringing Down Inference Attacks Against Your Privacy.* IEEE JSTSP 9(7):1240–1255, 2015 (secondary listing).
- **URL:** arXiv:1408.3698.
- **Opened:** abstract, §§III–V.
- **Statements:**
  - The convex leakage-versus-distortion mapping of [Calmon–Fawaz].
  - **A quantisation (clustering) step maps the data to a finite alphabet C**, and the convex program is solved over p_{Ĉ|C} (Fig. 1, Alg. 2).
  - Thm 2 bounds how quantisation affects leakage and distortion.
  - Thm 1 bounds the effect of a mismatched prior.
- **Relation to our claims:** a direct antecedent for (e)'s structure: a fixed finite code T, a convex stochastic channel Q, and an empirical prior.

### 14. Erdogdu & Fawaz (2015) — a new release beside fixed prior releases
- **Citation:** M. A. Erdogdu, N. Fawaz. *Privacy-Utility Trade-off under Continual Observation.* IEEE ISIT 2015, pp. 1801–1805 (secondary listing).
- **URL:** https://www.cs.toronto.edu/~erdogdu/papers/priv_isit.pdf (author PDF).
- **Opened:** §§II–III.
- **Statements:**
  - Online scheme: at step t, choose p(x̂_t | x_t, s_t, x̂^{t−1}) to minimise expected distortion s.t. I(X̂^t; S^t) ≤ ε_t.
  - The previous releases X̂^{t−1} are fixed.
  - The paper notes the constraint "accounts for the leakage up to time t−1 and bounds the **incremental leakage** due to step t".
  - **Thm III.3:** with finite alphabets and MI or directed-information leakage, the problems are **convex**.
- **Relation to our claims:**
  - Structurally the single-recipient part of (b) plus (e): I(S; Z, H) ≤ ε with H fixed is exactly I(S;Z|H) ≤ ε − I(S;H).
  - Differences: the new release there may condition on the past release, while ours depends only on T; and there is a single recipient.

### 15. Taylor, Vippathalla, Coon (2026) — per-party and collusion MI constraints
- **Citation:** S. Taylor, P. K. Vippathalla, J. P. Coon. *Adaptive Privacy of Sequential Data Releases Under Collusion.*
- **URL:** arXiv:2601.21859 (v1 2026-01-29, v2 2026-07-10). No journal reference is listed.
- **Opened:** §§I–IV, VIII.
- **Statements:**
  - Parties 1…m make sequential requests R_k on a database X.
  - For party k the handler chooses p(r̂_k | r̂^{k−1}, x) to maximise utility subject to an **individual** constraint I(R̂_k; X) ≤ ε_k and a **collusion** constraint I(R̂_k, R̂^{k−1}; X) ≤ δ_k.
  - Previous releases are fixed. The collusion model is worst-case: all releases so far are pooled.
  - Utility is either expected distortion ("a convex problem"), solved optimally by a Blahut–Arimoto-style algorithm (Thm 1 and §IV), or MI (nonconvex, locally optimal).
  - §I.A says multi-party, collusion-aware sequential release "has not yet been considered".
- **Relation to our claims:** **the closest antecedent for the combination (b)+(c)+(e).** Differences:
  1. The protected object is the whole database X, not a separate sensitive S with a permitted task.
  2. Their per-party constraint is unconditional, not conditional on that party's own frozen view.
  3. Their collusion is all-prefix pooling rather than a named coalition that includes outputs the mechanism never produced (H_B).
  4. Their release may condition on previous outputs.

  It is a 2026 preprint that precedes our submission, so it must be cited and distinguished.

---

## Part 2. Mischaracterisations and corrected wording

**M1.** main.tex L176–177 says Rassouli and Gündüz "give … the linear-programme form of the zero-budget linear-cost problem". The same claim is repeated at L203–204 ("the linear-programme form at zero budget are prior work") and in the NOVELTY_AND_SCOPE table.
- **Problem:** their LP (Thm 1, eq. 9) is for **mutual-information utility**, via extreme points of the perfect-privacy polytope. Linear-cost utilities (MMSE, error probability) appear only in Remark 2, which is asserted without proof. Everything is for **unconditional** perfect privacy.
- **Corrected wording:** "Rassouli and Gündüz characterise when perfect (zero-leakage) privacy is feasible through a nullspace condition on P_{X|W} (Prop. 1). For mutual-information utility they reduce the optimal zero-leakage mechanism to a linear programme over extreme points of the perfect-privacy polytope (Thm 1), and remark that the same holds for MMSE and error-probability utilities (Remark 2). Our zero-budget case is conditional (S ⊥ Z | C) and is an LP by the same linearity; that conditional statement is ours to state, not theirs."

**M2.** main.tex L181 says "the nullspace criterion and the convexity facts are theirs".
- **Problem:** the convexity facts are textbook (convexity of I in the channel; joint convexity of relative entropy). The convex leakage–distortion program is due to Calmon & Fawaz 2012 (Thms 1–2), with Makhdoumi et al. §II.D and Salamatian et al. 2015. The nullspace criterion is Rassouli–Gündüz's, but it is unconditional.
- **Corrected wording:** "the unconditional nullspace feasibility criterion is Rassouli and Gündüz's; convexity of the finite-alphabet leakage–distortion programme is due to Calmon and Fawaz [calmon2012privacy] (see also [makhdoumi2014funnel, §II.D], [salamatian2015managing]); convexity of conditional mutual information in the channel is standard."

**M3.** main.tex L170–173 says "the closed-form line … and the conditional extension U-FaTE … solve the same shape of problem".
- **Corrected wording:** "…and U-FaTE [dehdashtian2024ufate], which extends K-TOpt's kernel dependence measure to label-conditional dependence Dep(Z, S | Y = y), so that it covers separation-based fairness criteria, pairs a closed-form top-r eigen-solver (Thm 1) with an SGD-trained feature extractor, and has no eigenvalue-sign rank selection."
- Also: U-FaTE's conditioning is on the target label, not on an adversary's existing view.

**M4.** main.tex L179–180 says "Multi-user and side-information formulations … [sankar2013utility, liao2019sideinfo] design a sanitised release for many consumers". NOVELTY_AND_SCOPE calls these the "nearest multi-consumer prior art".
- **Problem:** Liao et al. design no release. Sankar et al. release one database to a user with side information and do not model multiple permission sets.
- **Corrected wording:** "Sankar et al. model side information at the recipient, including an encoder that knows it (Thms 2–3), and Liao et al. and Issa et al. define leakage conditional on an adversary's side information. The nearest multi-recipient prior art is Taylor et al. [taylor2026collusion], who impose per-party and collusion mutual-information constraints on sequential releases. See also multi-user mechanism design [zamani2023multiuser] and collusion-robust multi-level perturbation [li2012multilevel]."

**M5.** main.tex L175–181 says "The convex programme of §4 is established", but the attribution cites only Rassouli and Makhdoumi. **Add** calmon2012privacy, salamatian2015managing, erdogdu2015continual and taylor2026collusion, and cite calmon2017optimized as the fairness-preprocessing analogue. Only the conclusion ("established, no new optimisation result") is correct.

**M6.** main.tex L149–151 says "The valid statement is narrower and about squared loss only".
- **Problem:** this understates LEACE Thm 3.1.
- **Corrected wording:** "The valid statement is about losses convex in an affine prediction: zero cross-covariance (equal class-conditional means) means that no affine predictor improves on the best constant predictor under squared, cross-entropy, hinge or any other loss convex in the prediction (Belrose et al., Thms 3.1, 3.4). It says nothing about the 0–1 accuracy of thresholded linear classifiers, as the counterexample shows."

**M7.** main.tex L73–75 ("An auditing result that survives checking … the exact convex-combination identity") and NOVELTY_AND_SCOPE §"What is actually ours" item 3 ("an original contribution").
- **Problem:** pooled one-hot R² = 1 − ΣSSE_k/ΣSST_k is by definition the variance-weighted average of per-output R², with SST_k ∝ π_k(1−π_k). This is scikit-learn's documented `multioutput='variance_weighted'` definition. The project's own `linear_r2_onehot` computes exactly this pooled quantity.
- **Corrected wording:** "We point out that the pooled one-hot R² used in compliance reporting is the variance-weighted average of per-class OvR scores (weights π_k(1−π_k)), so a rare class barely moves it; auditing per-class or worst-direction scores exposes this." Call it an audit observation with empirical evidence, not an original identity.
- Optional: the worst linear direction over all class contrasts is the top squared canonical correlation between h and one-hot(S) (Hotelling 1936). It upper-bounds every OvR score (citation details not opened).
- Also note that with scikit-learn's *default* `uniform_average`, the weights are 1/K, not prevalence-based. The identity is specific to the pooled aggregate.

**M8.** Terminal-3 notes (`LITERATURE_SOURCES.json` and `NOVELTY_AND_ASSUMPTIONS.md`) call de Freitas & Geiger (CPFSI) the "closest conceptual antecedent".
- **Correction:** in CPFSI, "side-information" means the task label y used as supervision, and the conditioning is on the sensitive s. It is an antecedent for task-aware private representations, **not** for adversary side information, frozen views or coalitions. The closest antecedents for (b), (c) and (e) are sources 12–15.

**M9.** NeurIPS submission §2 (historical, only relevant if the text is reused) says Lechner & Ben-David "prove no single representation simultaneously satisfies multiple fairness criteria across tasks".
- **Corrected wording:** "prove that no representation can guarantee Odds Equality for two different tasks while allowing accurate prediction for both (except trivial cases), and that label-independent demographic parity fails under marginal shift."
- The paper has four authors: Lechner, Ben-David, Agarwal and Ananthakrishnan.

**Checked and accurate:**
- The privacy-funnel nonconvexity statement (L178–179, L204–205).
- The SARL/OptNet-ARL/K-TOpt eigenvalue-sign rank selection row. OptNet-ARL holds only under its Thm 4 assumptions (free z, linear regressors).
- The LAFTR, Elazar–Goldberg and SPLINCE descriptions.
- The CMI convexity argument in main.tex L194–197.

**Bib hygiene:**
- `sankar2013utility`: add volume 8, number 6, pages 838–852.
- `liao2019sideinfo`: add pages 642–646.
- `sadeghi2019sarl`: pages 7971–7979.
- These three sets of pages come from secondary listings.

---

## Part 3. Missing closest prior art (BibTeX)

```bibtex
@inproceedings{calmon2012privacy,
  title     = {Privacy Against Statistical Inference},
  author    = {Calmon, Fl{\'a}vio du Pin and Fawaz, Nadia},
  booktitle = {50th Annual Allerton Conference on Communication, Control, and Computing},
  pages     = {1401--1408},
  year      = {2012},
  doi       = {10.1109/Allerton.2012.6483382},
  note      = {arXiv:1210.2123},
  url       = {https://arxiv.org/abs/1210.2123}
}

@article{salamatian2015managing,
  title   = {Managing Your Private and Public Data: Bringing Down Inference Attacks Against Your Privacy},
  author  = {Salamatian, Salman and Zhang, Amy and Calmon, Fl{\'a}vio du Pin and Bhamidipati, Sandilya and Fawaz, Nadia and Kveton, Branislav and Oliveira, Pedro and Taft, Nina},
  journal = {IEEE Journal of Selected Topics in Signal Processing},
  volume  = {9},
  number  = {7},
  pages   = {1240--1255},
  year    = {2015},
  note    = {arXiv:1408.3698},
  url     = {https://arxiv.org/abs/1408.3698}
}

@inproceedings{erdogdu2015continual,
  title     = {Privacy-Utility Trade-off under Continual Observation},
  author    = {Erdogdu, Murat A. and Fawaz, Nadia},
  booktitle = {IEEE International Symposium on Information Theory (ISIT)},
  pages     = {1801--1805},
  year      = {2015},
  url       = {https://www.cs.toronto.edu/~erdogdu/papers/priv_isit.pdf}
}

@misc{taylor2026collusion,
  title         = {Adaptive Privacy of Sequential Data Releases Under Collusion},
  author        = {Taylor, Sophie and Vippathalla, Praneeth Kumar and Coon, Justin P.},
  year          = {2026},
  eprint        = {2601.21859},
  archivePrefix = {arXiv},
  primaryClass  = {cs.IT},
  url           = {https://arxiv.org/abs/2601.21859}
}

@inproceedings{calmon2017optimized,
  title     = {Optimized Pre-Processing for Discrimination Prevention},
  author    = {Calmon, Flavio P. and Wei, Dennis and Vinzamuri, Bhanukiran and Natesan Ramamurthy, Karthikeyan and Varshney, Kush R.},
  booktitle = {Advances in Neural Information Processing Systems 30 (NIPS 2017)},
  year      = {2017},
  note      = {arXiv:1704.03354},
  url       = {https://proceedings.neurips.cc/paper/2017/hash/9a49a25d845a483fae4be7e341368e36-Abstract.html}
}

@inproceedings{creager2019ffvae,
  title     = {Flexibly Fair Representation Learning by Disentanglement},
  author    = {Creager, Elliot and Madras, David and Jacobsen, J{\"o}rn-Henrik and Weis, Marissa A. and Swersky, Kevin and Pitassi, Toniann and Zemel, Richard},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {97},
  pages     = {1436--1445},
  year      = {2019},
  url       = {https://proceedings.mlr.press/v97/creager19a.html}
}

@article{issa2020operational,
  title   = {An Operational Approach to Information Leakage},
  author  = {Issa, Ibrahim and Wagner, Aaron B. and Kamath, Sudeep},
  journal = {IEEE Transactions on Information Theory},
  volume  = {66},
  number  = {3},
  pages   = {1625--1657},
  year    = {2020},
  note    = {arXiv:1807.07878},
  url     = {https://arxiv.org/abs/1807.07878}
}

@inproceedings{lopuhaa2020funnel,
  title     = {The Privacy Funnel from the Viewpoint of Local Differential Privacy},
  author    = {Lopuha{\"a}-Zwakenberg, Milan},
  booktitle = {Fourteenth International Conference on Digital Society (ICDS 2020), PPODS track},
  year      = {2020},
  note      = {arXiv:2002.01501; venue from arXiv comment only},
  url       = {https://arxiv.org/abs/2002.01501}
}

@article{li2012multilevel,
  title   = {Enabling Multilevel Trust in Privacy Preserving Data Mining},
  author  = {Li, Yaping and Chen, Minghua and Li, Qiwei and Zhang, Wei},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  volume  = {24},
  number  = {9},
  pages   = {1598--1612},
  year    = {2012},
  note    = {arXiv:1104.0459},
  url     = {https://arxiv.org/abs/1104.0459}
}

@inproceedings{zamani2023multiuser,
  title     = {Multi-User Privacy Mechanism Design with Non-zero Leakage},
  author    = {Zamani, Amirreza and Oechtering, Tobias J. and Skoglund, Mikael},
  booktitle = {IEEE Information Theory Workshop (ITW)},
  pages     = {401--405},
  year      = {2023},
  note      = {arXiv:2211.15525},
  url       = {https://arxiv.org/abs/2211.15525}
}

@inproceedings{jovanovic2023fare,
  title     = {{FARE}: Provably Fair Representation Learning with Practical Certificates},
  author    = {Jovanovi{\'c}, Nikola and Balunovi{\'c}, Mislav and Dimitrov, Dimitar I. and Vechev, Martin},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {202},
  pages     = {15401--15420},
  year      = {2023},
  url       = {https://proceedings.mlr.press/v202/jovanovic23a.html}
}

@article{defreitas2025trustworthy,
  title   = {Trustworthy Representation Learning via Information Funnels and Bottlenecks},
  author  = {Machado de Freitas, Jo{\~a}o and Geiger, Bernhard C.},
  journal = {Machine Learning},
  volume  = {114},
  pages   = {267},
  year    = {2025},
  note    = {arXiv:2211.01446},
  url     = {https://arxiv.org/abs/2211.01446}
}
```

**Priority of these citations:**
- Essential for honest attribution of (b), (c) and (e): calmon2012privacy, erdogdu2015continual, taylor2026collusion and salamatian2015managing.
- Strongly recommended: calmon2017optimized, creager2019ffvae, lopuhaa2020funnel and issa2020operational.
- Optional: li2012multilevel, zamani2023multiuser, jovanovic2023fare and defreitas2025trustworthy.
- `stadler2024leastprivilege` is already in the bib. Its Def. 2 (leakage relative to the task-label baseline) should also be cited for the "relative to baseline" framing of (b).

---

## Part 4. Verdicts per contribution

**(a) Purpose-specific sharing: permitted tasks and forbidden attributes per recipient.**
- **Closest antecedents:**
  - Creager et al. 2019 (FFVAE §4): per-task sensitive subsets selected at test time from one shared encoder.
  - Stadler et al. 2024: least privilege formalised and tied to purpose limitation.
  - Zamani et al. 2023 and Taylor et al. 2026: per-user or per-party requests.
- **Remaining distinction:** recipient-specific *forbidden* sets that conflict across recipients, a separate coalition-level prohibition, and a frozen pre-existing service.
- **Type:** problem formulation, a specialisation and combination. **Not algorithmic novelty.** Do not claim the policy framing itself as new.

**(b) Appending a channel beside frozen outputs H and measuring increments over the H-only view.**
- **Closest antecedents:**
  - Erdogdu & Fawaz 2015: a new release designed beside fixed prior releases, with the constraint bounding incremental leakage (convex).
  - Taylor et al. 2026: adaptive release given previous releases.
  - Sankar et al. 2013 Thm 3: an encoder that knows the recipient's side information.
  - Stadler et al. 2024 Def. 2: leakage over a baseline view.
  - Conditional leakage measures: Issa et al. Thm 6; Liao et al. Def. 4; SRLIP Def. 3.
- **Remaining distinction:**
  - The side information is a third party's frozen ML prediction vectors, not the releaser's own earlier outputs or raw attributes.
  - The channel input is a label-free code of the permitted inputs (Z–T–(S,C)).
  - The empirical audit (independently fitted, validation-selected attackers measuring log-loss reduction against an H-only family, negative increments retained) is evaluation methodology rather than a formal contribution.
- **Type:** application or formulation. The *concept* of incremental disclosure relative to a fixed view is **known**.

**(c) Recipient-specific plus coalition accounting.**
- **Closest antecedents:**
  - Taylor et al. 2026: individual constraints ε_k plus collusion constraints δ_k, both MI, with previous releases fixed.
  - Li et al. 2012: collusion-robust multi-level copies.
  - Lopuhaä-Zwakenberg 2020: SRLIP over all side-channel subsets, worst-case.
  - Issa et al. 2020: composition lemma.
- **Remaining distinction:** the coalition view includes an output H_B that the mechanism never produced or controlled, and constraints are conditional on each view rather than on pooled releases. The paper itself reports that coalition conditioning "did not earn its registered attribution".
- **Type:** combination or specialisation of a known idea. **Not novel as a concept.**

**(d) One-hot R² identity, rare-class hiding, exact-zero composition.**
- **Closest antecedents:**
  - The identity is the definition of variance-weighted multi-output R² (for example scikit-learn's `variance_weighted`). It is elementary algebra.
  - Rare-class hiding follows directly from the weights π_k(1−π_k).
  - Exact-zero composition follows from blockwise cross-covariance together with LEACE Thm 3.4.
  - The worst-direction audit is related to top canonical correlation (Hotelling 1936; not opened).
- **Remaining distinction:** the empirical demonstration that a compliance-style aggregate understated a rare-class direction on the HMDA attribute (0.027 versus 0.288). This is a useful measurement caution.
- **Type:** **known** mathematics. The contribution is an audit observation and a cautionary example. Drop the "original contribution / identity" framing (M7).

**(e) Finite stochastic release with conditional-MI constraints, convex for a fixed table.**
- **Closest antecedents:**
  - Calmon & Fawaz 2012 Thms 1–2: finite-alphabet distortion versus MI or max-leakage is convex, and multiple constraints are allowed.
  - Salamatian et al. 2015: quantise to a finite code, then a convex channel.
  - Erdogdu & Fawaz 2015 Thm III.3: convex with MI constraints that include fixed prior releases.
  - Taylor et al. 2026: expected-distortion problem with individual and collusion MI constraints, convex and solved exactly by a Blahut–Arimoto-style method.
  - Calmon et al. 2017 Prop. 1: convex randomized finite preprocessing with contextual conditioning and a fairness constraint of a different type.
  - Rassouli & Gündüz: zero-leakage LP (unconditional).
- **Remaining distinction:**
  - Conditioning views are recipients' frozen service outputs (coarsened in the fitted model).
  - The utility is a frozen decoder's expected cross-entropy, which is linear in Q.
  - Cross-recipient constraints are expressed as CMI given each view. Because I(S;Z|C) = I(S;Z,C) − I(S;C) and I(S;C) is fixed, this is exactly a Taylor- or Erdogdu-style total-leakage constraint with a shifted budget.
- **Type:** **known** (an established convex programme). The instance is a new application only. The manuscript's "no new optimisation result" is correct, but the attribution must move to sources 12–15 (M1, M2, M5). Calmon et al. 2017 should be cited but is *not* the single closest prior art.

---

## Part 5. What could not be verified
- **Not opened (UNVERIFIED):** Rodríguez-Gálvez et al. 2021 (CPF); Song et al. 2019; DProvDB and multi-analyst DP; arXiv:2010.10081; Zamani cache-aided and side-information papers (titles only); Hotelling 1936.
- **Pages and volumes from secondary listings only** (dblp, ML Anthology, search listings; primary proceedings pages not opened): Sankar TIFS, Liao ISIT, SARL ICCV, Rassouli JSAIT, Issa TIT, Salamatian JSTSP, Erdogdu ISIT, Li TKDE, Zamani ITW 2023 and Calmon–Fawaz Allerton.
- **Venues checked only from the arXiv comment:** Lopuhaä-Zwakenberg ICDS 2020 and U-FaTE CVPR 2024.
- **K-TOpt:** TMLR confirmed from ML Anthology and the project page. OpenReview did not load.
- **Calmon et al. 2017:** the supplementary material was not read. The convexity proof and the Prop. 2 constants were not checked.
- **Scope of the "no prior work" statements:** these reflect targeted searches, not an exhaustive survey. A missing hit does not establish priority.
