# Literature base (Phase 1): closest-prior-work reviewer

Role: closest-prior-work reviewer, campaign `research/pcrl-mechanism-design-v1`. Written 2026-10-16, 04:47Z to 05:32Z.
Inputs: primary sources (PDF text, official landing pages, Crossref records, official code repositories) and the repository's
earlier literature assessments. §2–§7 give short prose entries. §12 gives every work's objective, training access, release
contract, protected information, coalition/side-information access, optimisation and guarantee in one row each. No data, model or per-person file was opened. Nothing was run except text extraction
(`pdftotext`) of public PDFs and metadata queries. Every URL opened is listed in `README.md`.

**Markers.**
- **[P]**: the text was extracted from the PDF today and read at the cited definition, theorem or section.
- **[M]**: venue/pages checked today on an official page (PMLR, NeurIPS, ACL Anthology, CVF, ICLR, Zenodo/DOI) or on the publisher's Crossref record.
- **[A]**: venue taken from the arXiv comment / journal-ref field, or from the paper's own header or footer, and not checked on the publisher's page.
- **[NV]**: not verified today. The reason is given in each case.

**Setting used for relevance.**
- Two deterministic releases are made about the same person. Recipient 1 gets binary income probabilities plus the argmax; recipient 2 gets five-class occupation probabilities plus the argmax.
- The inputs are six census fields. SEX is the declared protected attribute and is **not** an input.
- A coalition can link both releases. Attackers read outputs only and are trained on auxiliary SEX-labelled releases.
- Utility caps apply per task.

---

## 0. Main conclusions

1. **Stadler et al. (ICML 2024).**
   - Their impossibility protects *every* attribute, i.e. every randomised function of (X, Y), relative to what the *true* label reveals.
   - The binding attribute in the proof is a constructed "maximally revealing" attribute S\*, not a natural one.
   - The theorem implies nothing about one declared attribute (SEX), and the authors themselves say single-attribute restriction is sometimes possible (their §4, "Takeaways"; see §2 here).
   - The correct single-attribute trade-offs are data-dependent bounds: Zhao et al. 2020, Thms 3.2–3.3; Zhao & Gordon 2022.
2. **The manuscript sentence "a release useful for a task reveals what the task's label reveals [stadler2024lpl]" over-applies Stadler** (§1).
   - Releases computed from X are not bounded below by true-label leakage.
   - A hard decision with error rate ε can differ from the label-only attack success by up to ε.
3. **Neither adversary is new.**
   - *Auxiliary-data adversary* (a reader trained on labelled auxiliary data that sees only the release): Song & Shmatikov 2020 (explicit `D_aux` attacker), Jia & Gong 2018, Elazar & Goldberg 2018, Wang et al. 2019, Jayaraman & Evans 2022 and Tian et al. 2025.
   - *Linked / colluding multi-release adversary*: Ganta et al. 2008, Wang & Fung 2006 (joins across purpose-specific sequential releases), Li et al. 2012 (colluding miners), Taylor et al. 2026 (colluding parties) and DProvDB 2024 (colluding analysts). See §9.
4. **"Taylor 2026"** is S. Taylor, P. K. Vippathalla, J. P. Coon, *Adaptive Privacy of Sequential Data Releases Under Collusion*, arXiv:2601.21859 v2 (10 Jul 2026), a preprint.
   - It protects the whole record X (mutual information) under a known pmf, with randomised channels.
   - It designs releases sequentially *because future requests are unknown*. It does not claim that sequential design beats joint design.
   - It is **not** the only collusion-aware release design. Li et al. 2012, Wang & Fung 2006 and DProvDB are others. None protects a declared attribute through learned predictions.
5. **Joint versus sequential (§8).** Under the same information and randomness, a sequential designer with full look-ahead realises every jointly designed pair, since min over (f1, f2) equals min over f1 of min over f2.
   - For deterministic releases, "conditioning on the earlier release" adds nothing.
   - A joint advantage can therefore come only from greedy or online restrictions, computation, or optimisation quality.
   - No cited work claims otherwise. Erdogdu & Fawaz 2015 even formalise the online-vs-batch "regret" and prove it is zero under a hidden-Markov model.
   - Shared randomness enlarges the set of achievable coalition behaviours. That is how Li et al. (correlated noise), DProvDB (additive Gaussian synopses) and Taylor's reuse principle limit what colluders gain.
6. **Declared-attribute coalition leakage does not compose.**
   - Only "all-attribute" measures such as maximal leakage are subadditive over releases (Issa et al. 2020, Lemma 6).
   - For one attribute, a coalition can learn SEX when neither release alone reveals anything (an XOR construction, §8.3).
   - So per-recipient guarantees on SEX alone cannot certify the pair. Zhao et al. 2020's Theorem 3.1 explicitly assumes the adversary has no side information.
7. **Tree methods.**
   - FAGTB (adversary on σ(F(x)), new trees fitted to combined pseudo-residuals) and FairGBM (proxy-Lagrangian, randomised classifier) are the in-processing GBDT precedents.
   - Post-hoc leaf editing exists: EiFFFeL (discrete leaf flips) and Ren et al. 2015 (joint relearning of leaf values with fixed structure).
   - No source read today fine-tunes continuous leaf scores of a fitted boosted ensemble against online critics on released outputs, for one recipient or two. This is a bounded-search observation, not a priority claim.

---

## 1. Manuscript citation fidelity (coordinator addendum)

**Source.** `origin/research/combined-analysis-paper-v1:papers/combined_empirical_v1/main.tex`, Related work, lines 135–162; the bibliography is `papers/combined_empirical_v1/references.bib` on the same branch.

### 1.1 Identities behind the three keys (all verified today)

| Key | Work (verified) | Primary evidence | Bib status | Is the manuscript sentence faithful? |
|---|---|---|---|---|
| `taylor2026collusion` | S. Taylor, P. K. Vippathalla, J. P. Coon (Oxford), *Adaptive Privacy of Sequential Data Releases Under Collusion*, arXiv:2601.21859 [cs.IT]; v1 29 Jan 2026, v2 10 Jul 2026. No journal-ref. [P][A] | PDF v2 p.1 author line "Justin P. Coon"; arXiv abs history | Correct as a preprint. Author string "Coon, Justin" (arXiv metadata) vs "Justin P. Coon" (PDF): either is acceptable. | **Yes, with a suggested qualifier.** "Collusion-aware mechanisms exist for sequential finite-alphabet releases" is accurate. Taylor's releases are randomised channels designed with a known pmf, protecting all of X by mutual information. Suggest adding "protecting the whole record under a known distribution". |
| `tian2025fairnessprivacy` | H. Tian, G. Zhang, B. Liu, T. Zhu, M. Ding, W. Zhou, *Do Fairness Interventions Come at the Cost of Privacy: Evaluations for Binary Classifiers*, **IEEE TDSC 22(5):4654–4670, 2025**, DOI 10.1109/TDSC.2025.3551157. [P][M] | Crossref record; arXiv:2503.06150 v2 | **Outdated.** The bib lists only "arXiv preprint". Update to the TDSC journal version. | **Yes.** FD-AIA (§6.3, Eq. 12) gives an attack model the predictions of the biased model T_b(x) and the fair model T_f(x) for the same person. On CelebA (black box), FD-AIA reaches 87.6 % on the fair model, against 82.9 % for a standard AIA on the fair model and 83.4 % on the biased model (§7.2). Data are face images (CelebA, UTKFace, FairFace). |
| `wang2019balanced` | T. Wang, J. Zhao, M. Yatskar, K.-W. Chang, V. Ordonez, *Balanced Datasets Are Not Enough: Estimating and Mitigating Gender Bias in Deep Image Representations*, ICCV 2019, pp. 5309–5318, DOI 10.1109/ICCV.2019.00541. [P][M] | Crossref gives pp. 5309–5318; the CVF open-access page prints pp. 5310–5319 | Correct (IEEE pagination). | **Yes.** Their "dataset leakage" is an attacker on randomly perturbed *true labels* at matched task performance (F1). Their "model leakage" is an attacker on model outputs. Amplification means model leakage > dataset leakage (§3, Fig. 1). |

### 1.2 Sentence (a): "a release useful for a task reveals what the task's label reveals [stadler2024lpl]"

**Verdict: not a faithful statement of the theorem; an over-application in this setting.**

- **What the paper contains.** Stadler et al. *define* their least-privilege principle (Def. 2) as a bound on inference **over** the fundamental leakage Pr[S = Ŝ(Y)], with the adversary given the true label Y. They motivate this informally in related work: "useful representations must at least leak any sensitive information already revealed by the task label itself".
- **What the theorem proves.** Under Assumption A, a map with utility I_α(Y;Z) > γ violates γ-LPP. Some attribute (the constructed S\*) is inferred better than from Y by more than γ.
- **What it never proves.** There is no theorem that a useful release reveals what Y reveals about a *given* attribute.
- **Releases computed from X have no true-label floor.**
  - Suppose SEX ⊥ X while SEX and Y are dependent, for example because SEX affects only the label noise. Then every release f(X) reveals nothing about SEX, although Y does.
  - Conversely, a release can carry SEX information present in X but absent from Y.
  - The only general link is a simple coupling (my derivation). For a hard decision D = f(X) with error ε = P(D ≠ Y), every attack rule a satisfies |P[a(D) = SEX] − P[a(Y) = SEX]| ≤ ε.
  - The label-only reference is therefore approached only by near-perfect decisions. For census income and occupation tasks, whose error rates are far from zero, it is a descriptive comparator, not a disclosure floor.
- **Endpoint mismatch.** Stadler's leakage is the Bayes-optimal multiplicative accuracy gain I∞ on finite alphabets under a known law. The manuscript measures AUC of fitted readers.

**Suggested replacement sentence:**

> Stadler et al.~\citep{stadler2024lpl} prove that, when the label posterior is strictly positive, no useful representation can bound inference beyond the task label for \emph{every} attribute; the violating attribute is constructed, and the authors note that leakage of a single attribute can sometimes be restricted. Their label-only adversary (``fundamental leakage'') is the reference we report, as a descriptive comparison rather than a floor: our releases are computed from the inputs alone, so they can reveal less, or more, than the true label does.

**Minor, same paragraph.**
- "When the label depends on the protected attribute, fair representations trade accuracy for protection [zhao2019inherent, zhao2022inherent]" is accurate only for **statistical parity**. Zhao & Gordon's JMLR abstract: "if the base rates differ among groups, then any fair classifier satisfying statistical parity has to incur a large error on at least one of the groups" [P].
- For protection as attribute *inference*, the matching result is Zhao, Chi, Tian & Gordon (NeurIPS 2020), Thms 3.2–3.3: a trade-off between group-conditional accuracy and the inference advantage, governed by D_JS(D₀^Y, D₁^Y). Consider citing it there.

---

## 2. Stadler, Kulynych, Gastpar, Papernot, Troncoso (ICML 2024): what the impossibility protects

**Citation.** *The Fundamental Limits of Least-Privilege Learning.* ICML 2024, PMLR 235:46393–46411.
- Landing page: https://proceedings.mlr.press/v235/stadler24a.html [M]
- PDF: https://raw.githubusercontent.com/mlresearch/v235/main/assets/stadler24a/stadler24a.pdf [P]
- arXiv:2402.12235 v2 (also extracted)
- No code link on the PMLR page.

**Setting (p.2–3).**
- X, Y, S are discrete and finite, with full support. Z = f_E(X) may be randomised.
- **Assumption A:** P_{Y|X}(y|x) > 0 for all x, y.
- Utility is Arimoto information I_α(Y;Z), α ∈ {1, ∞}. α = ∞ is the multiplicative accuracy gain.

**Definitions.**
- **Def. 1 (unconditional LPP):** for *any attribute S ≠ Y* with S − X − Z, I∞(S;Z) ≤ γ.
- **Def. 2 (LPP):** for *any attribute S* with S − (X,Y) − Z, I∞(S;Z | Y) ≤ γ. The adversary sees (Z, Y), and the gain is measured over the "fundamental leakage" Pr[S = Ŝ(Y)].
- The supremum is the conditional maximal leakage L(X → Z | Y) (Issa et al.).

**Theorem 2.**
- Statement: under A, "Z satisfies LPP with γ" and "I_α(Y;Z) > γ" cannot both hold.
- Proof (App. A): A implies L(X → Z | Y) = L(X → Z) ≥ L(Y → Z) ≥ I_α(Y;Z).
- The supremum is attained by a **maximally revealing attribute S\*** defined over X × Z (Def. 4, Prop. 3), a constructed "shattering" variable.
- Prop. 1: perfect LPP holds iff X − Y − Z. Cor. 2: under A, γ = 0 only for constant or random maps. Remark 1: A can be weakened to "Y ≠ S\*".

**Protected information.** All attributes: every (randomised) function of (X, Y), relative to the true label. **Not** a declared attribute.

**Authors' scope statements (read today).**
- §4, "Takeaways" (p. 9): "although it is sometimes possible to restrict information leakage for a single attribute, the learned representations do not fulfil the LPP".
- The related-work paragraph (p. 2) motivates the baseline informally: "useful representations must at least leak any sensitive information already revealed by the task label itself".
- §5: restricting the attacked class (e.g. "efficiently identifiable" attributes) might be more favourable but "deviate[s] from the promise of LPL to leak 'nothing else'".
- The leakage measure is an average over the population, weaker than LDP.

**Empirics.**
- §4.1, Fig. 3: a *label-only adversary* (frequency counts of S | Y on auxiliary data D_A) measures fundamental leakage.
- The features adversary is a random forest with black-box access to f_E.
- Censoring one attribute raises another ("whack-a-mole").

**Why it must not be used as an impossibility proof for protecting SEX.**
1. **Quantifier.** Thm 2 bounds a supremum over all attributes, and the attribute achieving it is S\*, not SEX.
   - A useful map violates LPP but may leak nothing extra about SEX.
   - Trivially, if SEX ⊥ X, every release f(X) leaks nothing about SEX at full utility while violating LPP through S\*.
2. **Baseline.** Def. 2 conditions on the *true* label held by the adversary. Our recipients see only f(X), and true-label leakage is no floor (§1.2).
3. **Measure and assumptions.** The theorem needs finite discrete alphabets, Assumption A, a known law and Bayes-optimal I∞. Our endpoint is fitted-reader AUC under utility caps.
4. **The right single-attribute statements are data-dependent.**
   - Zhao et al. 2020 Thm 3.2: ACC₀ + ACC₁ ≤ 2 − D_JS(D₀^Y, D₁^Y)/3 + ADV(H_A ∘ f).
   - Zhao & Gordon 2022: statistical-parity versus group-wise error when base rates differ.
   - These bound accuracy in terms of task–attribute dependence; they are not universal impossibility.
5. **What does transfer.** Stadler's label-only adversary is the same construct as this project's label-only reference (as `results/combined_empirical_preparation_v1/notes/literature_A/stadler2024_lpl.md` already says).
   - For the coalition, the matching reference is SEX predicted from the *label pair* (income, occupation). It can exceed either single-label reference.

---

## 3. Information-theoretic privacy–utility formulations

**Tishby, Pereira, Bialek, *The information bottleneck method* (1999).**
- Links: arXiv:physics/0004057 (PDF dated 30 Sep 1999) [P]. The Allerton 1999 proceedings (pp. 368–377) appear only in secondary listings [NV].
- **Objective:** min over p(x̃|x) of L = I(X̃;X) − βI(X̃;Y) (eq. 15).
- **Training access:** the joint p(x, y) is known; no sensitive attribute.
- **Release:** a stochastic code X̃.
- **Optimisation:** self-consistent equations solved by a convergent Blahut–Arimoto-type iteration (to a local fixed point).
- **Guarantee:** none for privacy.
- **Relevance:** the parent formulation of the privacy funnel.

**Makhdoumi, Salamatian, Fawaz, Médard, *From the Information Bottleneck to the Privacy Funnel*, IEEE ITW 2014.**
- Links: arXiv:1402.1774 v5 [P]. DOI 10.1109/ITW.2014.6970882, read on Zenodo record 1274716 [M]. No code.
- **Objective:** privacy funnel, min I(S;Y) s.t. I(X;Y) ≥ R (eq. 3), over P_{Y|X}, with the Markov chain S → X → Y. The mapping never reads S.
- **Training access:** P_{S,X} is known ("Input: R, P_{S,X}"); discrete alphabets.
- **Release:** the disclosed variable Y, a deterministic merge map in Algorithm 1.
- **Protects:** a declared private S. Single analyst, no side information.
- **Optimisation:** the problem is non-convex "because of the constraint I(X;Y) ≥ R".
  - **Greedy merge (Alg. 1):** start from the identity map; repeatedly merge the pair {y_i, y_j} that most reduces I(S;Y) while I(X;Y) ≥ R holds.
  - Prop. 1 gives the merge costs.
  - "No guarantee that such a greedy algorithm induces a global optimal privacy mapping".
  - Alg. 2 (the IB variant) gives the local *maximum*, which brackets the range.
- **Guarantee:** Thm 1 bounds the inference-cost gain under any bounded cost by 2√2·L·√I(S;Y) under the known law. It is not an AUC bound.

**Calmon & Fawaz, *Privacy Against Statistical Inference*, Allerton 2012.**
- Link: arXiv:1210.2123 [P][A].
- **Objective:** minimise average (Thm 1) or maximum (Thm 2) information leakage of S through U s.t. E d(Y,U) ≤ Δ.
- **Training access:** the prior p_{S,Y} is known. The adversary knows the prior *and* the mapping, which the paper calls "the worst-case statistical side information".
- **Release:** a randomised U from p_{U|Y}. Remark 3: the mapping may read S.
- **Protects:** a declared S. Single recipient.
- **Optimisation:** a **convex program** in (p_{U|Y}, p_{U|S}) coupled by a linear equality. Remark 1: extra distortion constraints are extra linear constraints.
- **Guarantee:** exact optimum for the known finite law.

**Side information at the adversary; conditional variants (primary sources).**

- **Sankar, Rajagopalan, Poor, IEEE TIFS 8(6):838–852, 2013.** arXiv:1102.3751 [P]; Crossref [M].
  - Setup: rate–distortion–equivocation for a database release, with side information Zⁿ at the user, who is also the potential adversary.
  - Thm 2: a *statistically informed* encoder.
  - Thm 3: an *informed* encoder that knows the realised side information ("sharing a related but differently sanitized view"), achieved by quantize-and-bin.
  - Asymptotic, single release, protects declared private attributes.
- **Erdogdu & Fawaz, *Privacy–utility trade-off under continual observation*, ISIT 2015, pp. 1801–1805.** Author PDF [P]; Crossref [M].
  - Batch versus online privacy mappings for time series.
  - The online interactive scheme uses (X̂^{t−1}, X^t, S^t), and the constraint "bounds the incremental leakage due to step t".
  - Thm III.3: convex for finite alphabets under MI or directed information.
  - **Def. 3: online-versus-batch "regret". Thm III.4: no regret under a hidden-Markov model.**
- **Liao, Sankar, Kosut, Calmon, *Robustness of Maximal α-Leakage to Side Information*, ISIT 2019.** arXiv:1901.07105 [P][A].
  - Defines conditional (maximal) α-leakage given adversary side information Z.
  - If Z ⊥ released data | private data, side information cannot increase leakage.
  - A measure only; no release design.
  - **Effect in our coalition depends on what counts as private.**
    - If the private data are all inputs X, the condition holds (both releases are functions of X). The result then bounds only the all-attribute leakage about X.
    - If the private data are SEX, the other release is *not* conditionally independent of the release given SEX, so the result does not apply.
- **Issa, Wagner, Kamath, *An Operational Approach to Information Leakage*, IEEE TIT 66(3):1625–1657, 2020.** arXiv:1807.07878 [P]; Crossref [M].
  - Def. 6 and Thm 6: conditional maximal leakage given adversary side information.
  - Cor. 2(8): L(X → (Y,Z)) ≤ L(X → Z) + L(X → Y | Z).
  - **Lemma 6: if Z − X − Y, then L(X → (Y,Z)) ≤ L(X → Z) + L(X → Y).** This subadditivity holds for leakage about *all* functions of X, not for one attribute.
- **Lopuhaä-Zwakenberg, *The Privacy Funnel from the Viewpoint of Local Differential Privacy*, ICDS 2020.** arXiv:2002.01501 [P][A].
  - Def. 3, SRLIP: pointwise privacy conditional on every side-channel attribute subset.
- **Wang, Basciftci, Ishwar, *Privacy-Utility Tradeoffs under Constrained Data Release Mechanisms*.** arXiv:1710.09295 preprint [P]; venue [NV].
  - The trade-off region given only the useful data is no larger than with full data (S included).
  - Relevant because our deployed maps cannot read SEX, whereas GAP (g(X,Y)), Xie et al. (E(x,s)) and Calmon & Fawaz Remark 3 may.
- **Conditional privacy funnel (CPF).** Rodríguez-Gálvez, Thobaben, Skoglund, *A Variational Approach to Privacy and Fairness*, ITW 2021 per arXiv. arXiv:2006.06332 v3 [P][A]; code `burklight/VariationalPrivacyFairness` (opened).
  - inf I(S;Y) s.t. I(X;Y|S) ≥ r, with S ↔ X ↔ Y.
  - "Conditional" means the utility is measured *given S*, not adversary side information.
- **CPFSI.** de Freitas & Geiger, *Trustworthy Representation Learning via Information Funnels and Bottlenecks*, Machine Learning 114:267 (2025). arXiv:2211.01446 v2 [P][A]; code `jmachadofreitas/funck-ml2024` (opened).
  - L = I(s;z) − γI(z;x|s) − βI(y;z|s) (Def. 1).
  - "Side-information" is the downstream **task label**, not adversary knowledge.

**Takeaway.** Formulations with adversary side information exist: Sankar Thm 3 (informed encoder), Erdogdu & Fawaz, Taylor, and the measures of Liao and Issa. All assume a known finite law, randomised mappings and MI or maximal-leakage criteria. None uses learned predictors or fitted readers.

---

## 4. Adversarial obfuscation and fair or private representations

**Edwards & Storkey, *Censoring Representations with an Adversary*, ICLR 2016.** arXiv:1511.05897 [P][A].
- **Objective:** an encoder minimises an adversarial critic's ability to predict S from R, alongside task and reconstruction terms. Minimax optimised by alternating SGD.
- **Training access:** S labels. **Release:** a representation.
- **Guarantee:** none. The critic serves as an adaptive dependence measure.

**Xie, Dai, Du, Hovy, Neubig, *Controllable Invariance through Adversarial Feature Learning*, NIPS 2017.** Official page [M]; arXiv:1705.11122 [P].
- **Objective:** min over (E, M), max over D of γ log q_D(s | E(x,s)) − log q_M(y | E(x,s)). **The encoder reads s.**
- **Guarantee:** a non-parametric equilibrium analysis (infinite capacity) giving "win-win" versus "competing" equilibria.
- The code link printed in the paper (`qizhex/Controllable-Invariance`) returned HTTP 404 today.

**Madras, Creager, Pitassi, Zemel, LAFTR, ICML 2018, PMLR 80:3384–3393.** Landing page and PDF [M][P].
- **Objective:** encoder f(X) → Z (no A at input), with classifier and adversary; the adversary objective is group-normalised.
- **Training access:** A and Y labels. **Release:** a representation, reused for transfer.
- **Guarantee:** an unnumbered §5 theorem says the demographic-parity distance of any binary g on Z is at most the **optimal** adversary's objective. It is conditional on optimality and covers DP/EO of classifiers, not attribute recovery.
- A repository `VectorInstitute/laftr` titled with the paper exists (opened). It is not linked from the PMLR page, so whether it is official was not confirmed.

**Zhang, Lemoine, Mitchell, *Mitigating Unwanted Biases with Adversarial Learning*, AIES 2018, pp. 335–340.** Crossref [M]; arXiv:1801.07593 [P].
- **Objective:** a predictor updated with ∇L_P − proj_{∇L_A}∇L_P − α∇L_A. The adversary reads Ŷ (DP) or (Ŷ, Y) (EO).
- **Release:** predictions. **Applies to:** any gradient-trained model.
- **Guarantee:** Prop. 2. If the predictor and adversary converge, under stated conditions (constant-output weights exist, convex L_A), the adversary gains nothing from Ŷ. It is conditional on convergence.

**Bertran et al., *Adversarially Learned Representations for Information Obfuscation and Inference*, ICML 2019, PMLR 97:614–623.** Landing page [M]; PDF [P]; code `MartinBertran/AIOI` (linked by PMLR, opened).
- **Objective:** min over p(y|x) of I(U; X | Y) s.t. I(S;Y) ≤ k, equivalently max I(U;Y). Implemented as an unconstrained adversarial game with stochastic transformations.
- **Training access:** utility U and sensitive S labels. **Release:** a domain-preserving stochastic transformation of X.
- **Guarantee:** Lemmas 2.1–2.3 bound the achievable trade-off through latent-variable mappings. There is no guarantee against post-hoc readers.

**Zhao, Chi, Tian, Gordon, *Trade-offs and Guarantees of Adversarial Representation Learning for Information Obfuscation*, NeurIPS 2020 (Adv. NeurIPS 33, from p. 9485).** Official page [M]; arXiv:1906.07902 v3 [P].
- **Objective:** min over f of H(Y|f(X)) − λH(A|f(X)) (eq. 3), deterministic f, binary attribute A.
- **Guarantees.**
  - Thm 3.1: any adversary errs with probability ≥ H\*/(2 lg(6/H\*)), **assuming I(Â; A | Z) = 0, i.e. no external information about A.** That assumption fails for a coalition holding the other release.
  - Thms 3.2–3.3: trade-off between conditional accuracies and the inference advantage via D_JS(D₀^Y, D₁^Y), for all binary adversaries on Z.
- No code link found in the paper.

**Huang, Kairouz, Chen, Sankar, Rajagopal, *Context-Aware Generative Adversarial Privacy*, Entropy 19(12):656, 2017.** Crossref [M]; arXiv:1710.09549 [P].
- **Objective:** a privatizer g(X, Y) versus an adversary inferring the private Y, under a distortion constraint. **The privatizer may read the private variable.**
- Under log-loss the game reduces to MI privacy; game-theoretically optimal mechanisms are derived for binary and Gaussian-mixture models.

**Hamm, *Minimax Filter*, JMLR 18 (2017) 1–31 (arXiv journal-ref).** arXiv:1610.03577 [P][A]; code `jihunhamm/MinimaxFilter` (cited in paper; opened, now redirects to `hammlab/MinimaxFilter`).
- **Objective:** min-diff-max filter optimisation against inference attacks, plus an ERM-style generalisation analysis.
- **Release:** a filtered continuous representation. A noisy variant combines it with DP.

**Tripathy, Wang, Ishwar, *Privacy-Preserving Adversarial Networks*.** arXiv:1712.07008 v3 [P]; venue [NV].
- Adversarially trained randomised release mechanisms with a variational MI-privacy approximation. They match analytic optima on synthetic data.

**Elazar & Goldberg, *Adversarial Removal of Demographic Attributes from Text Data*, EMNLP 2018, pp. 11–21.** ACL Anthology [M]; PDF [P]; code `yanaiela/demog-text-removal` (cited in paper, opened).
- The adversary "achieves chance-level development-set accuracy during training", but **a post-hoc attacker** trained on the encoded sentences recovers the attribute well above chance (Table 3, Δ = attacker − adversary).

**Song & Shmatikov, *Overlearning Reveals Sensitive Attributes*, ICLR 2020.** ICLR virtual page [M]; arXiv:1905.11742 v3 [P].
- **Adversary:** "an auxiliary set D_aux of labeled (x, s) pairs and black-box oracle E"; it trains M_attack on (E(x), s).
- Censored representations (adversarial or information-theoretic) still leak. A de-censoring transform (Alg. 1) raises recovery.
- They state that "overlearning is intrinsic for some tasks".

**Belrose et al., LEACE, NeurIPS 2023 (Adv. NeurIPS 36, from p. 66044).** Official page [M]; NeurIPS PDF [P]; code `EleutherAI/concept-erasure` (cited in paper, opened).
- **Release:** an affine eraser of the representation. **Training access:** concept labels Z.
- **Guarantee:** Def. 2.3 and Thms 4.1–4.3. No affine predictor beats a constant under any loss convex in its prediction (linear guardedness), with a least-squares-optimal eraser.
- **Scope (text read):** the softmax probabilities of a multiclass logistic head "can leak the removed information if another classifier is stacked on top". The limitations section conjectures that general nonlinear erasure is intractable.

**Jovanović, Balunović, Dimitrov, Vechev, FARE, ICML 2023, PMLR 202:15401–15420.** Landing page [M]; PDF [P]; code `eth-sri/fare` (opened; the paper cites it).
- **Encoder:** a *restricted encoder* mapping x to one of k cells, instantiated with fair decision trees. **Training access:** S is used to build the trees.
- **Release:** the cell representation.
- **Guarantee:** a finite-sample, high-confidence (Clopper–Pearson, Lemmas 5.1–5.3) upper bound on the demographic-parity distance of **any** downstream classifier that reads only the cell. It does not cover attribute-recovery AUC or a separately released output.
- **Coalition corollary (repository observation, confirmed as logic):** two restricted encoders together form one with ≤ k₁k₂ cells, so the same procedure certifies the pair, more loosely.

**Creager et al., FFVAE, *Flexibly Fair Representation Learning by Disentanglement*, ICML 2019, PMLR 97:1436–1445.** Landing page [M]; PDF [P].
- "Each task may have a different task label y and sensitive attributes a". Fairness for a subset is obtained at test time by removing latent dimensions.
- It is the closest learned-representation precedent for per-consumer sensitive sets. No coalition analysis was found in the sections read.

**Zhao & Gordon, *Inherent Tradeoffs in Learning Fair Representations*, JMLR 23 (2022) 1–26; NeurIPS 2019 conference version.** arXiv:1906.08386 v6 (JMLR header) [P][A].
- Lower bound on the sum of group-wise errors for statistical-parity-fair classifiers when base rates differ.

---

## 5. Tree-based fair or private learning

**Grari, Ruf, Lamprier, Detyniecki, FAGTB, ICDM 2019, pp. 1060–1065.** Crossref [M]; arXiv:1911.05369 [P]; code `vincent-grari/FAGTB` (README opened; it states the ICDM 2019 acceptance).
- **Objective:** boosting loss versus a neural adversary predicting S from σ(F(x)) for DP, or from (σ(F(x)), y) for equalised odds (§III-A).
- **Optimisation:** at each round, the pseudo-residuals u = r − λt combine task and adversary gradients; a **new tree** is fitted to u, followed by a line search; the adversary is then updated.
- Experiments start the adversary from a "biased GTB" and run several adversary steps per round.
- **Release:** a score or classifier.
- **Guarantee:** a remark only. If a perfectly fair and perfectly accurate pair exists, it is a global optimum of the min-max.
- Claim: "the first adversarial learning method for generic classifiers, including decision trees".

**Cruz et al., FairGBM, ICLR 2023.** ICLR poster page [M]; arXiv:2209.07850 v5 [P]; code `feedzai/fairgbm` (opened).
- **Objective:** a proxy-Lagrangian (Cotter et al.) with cross-entropy-based proxies of group FPR/FNR/parity constraints, solved by dual ascent over boosting rounds.
- **Training access:** S at training only. **Release:** a classifier.
- **Guarantee:** inherited approximate feasibility and optimality for the **randomised** classifier (a coarse-correlated equilibrium). The deployed last iterate is deterministic with no guarantee.
- **Authorship discrepancy:** the ICLR poster page lists 5 authors (Cruz, Belém, Bravo, Saleiro, Bizarro); arXiv v5 also lists Sérgio Jesus. Cite from the official ICLR/OpenReview record. OpenReview could not be opened today (challenge page).

**Abebe, Lucchese, Orlando, EiFFFeL, ACM SAC 2022, pp. 429–436.** Crossref [M]; arXiv:2112.14435 [P].
- Post-hoc **relabelling (flipping) of leaves** of selected trees in a random forest, toward a group-fairness target under an accuracy-drop limit. Discrete, binary.
- Extends Kamiran et al. (ICDM 2010, leaf relabelling) [NV: not opened today].

**Ren, Cao, Wei, Sun, *Global Refinement of Random Forest*, CVPR 2015.** CVF PDF [P].
- Keeps the forest structure and "jointly relearns the leaf nodes of all trees under a global objective". The fixed-routing leaf-value re-optimisation parameterisation predates this project.

**Not re-opened today:** LightGBM `refit` / XGBoost `refresh` documentation and the ZestFinance patent US 11,720,962 B2 (cited in `results/pcrl_frozen_tree_control_completion_v1/TARGETED_METHOD_NOVELTY.md`) [NV].

**Takeaway.** Adversarial or constrained boosting (FAGTB, FairGBM) and leaf-level post-hoc edits (EiFFFeL; Ren-style refits) are established.
- In the sources read, no method fine-tunes continuous leaf scores of a fitted boosted ensemble under frozen routing against online critics reading released probabilities plus decision.
- None has a pair or coalition critic.

---

## 6. Attribute inference from released outputs with auxiliary data

**Fredrikson et al., *Privacy in Pharmacogenetics*, USENIX Security 2014.** PDF front matter (ISBN 978-1-931971-15-7) [P][A]; pages [NV] (USENIX landing page behind a challenge).
- **Model inversion:** given a linear model (black box), the target's other attributes, the stable dose (true label) and marginals, infer genotype.
- **The sensitive attribute is a model input.** The baseline is guessing from marginals.

**Fredrikson, Jha, Ristenpart, *Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures*, CCS 2015, pp. 1322–1333.** Crossref [M]; author PDF [P].
- Black- and white-box attacks that exploit confidences: decision-tree lifestyle surveys (a sensitive *input* feature) and face recognition.
- Countermeasures: a privacy-aware CART variant and rounded confidences.

**Yeom, Giacomelli, Fredrikson, Jha, *Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting*, IEEE CSF 2018, pp. 268–282.** Crossref [M]; arXiv:1709.01604 [P].
- Experiment 2 / Def. 5: attribute advantage from partial information φ(z) about *training* versus population points.
- Def. 6: comparison against the Bayes-optimal simulator without the model, i.e. the imputation baseline.

**Mehnaz et al., *Are Your Sensitive Attributes Private? Novel Model Inversion Attribute Inference Attacks on Classification Models*, USENIX Security 2022.** PDF front matter [P][A]; pages [NV].
- The adversary knows non-sensitive attributes of training records and has black-box access.
- A label-only attack "matches" the confidence-score attack.
- The attribute is a model input.

**Jayaraman & Evans, *Are Attribute Inference Attacks Just Imputation?*, CCS 2022, pp. 1569–1582.** Crossref [M]; arXiv:2209.01292 [P]; code `bargavj/EvaluatingDPML` (cited in paper, opened).
- An imputation adversary with distributional knowledge versus attackers with white-box, black-box or no model access, and an auxiliary set D_aux of varying size and distribution (Table 3).
- Many published AIA results do not exceed imputation.
- They introduce "sensitive value inference".

**Jia & Gong, AttriGuard, USENIX Security 2018 (arXiv comment).** arXiv:1805.04810 [P][A].
- The **attacker trains a classifier on users who publicly disclose the attribute** and applies it to non-disclosing users.
- The defense adds minimum evasion noise to the user's *public data*, plus a randomised mechanism under a utility budget.
- No formal privacy guarantee.

**Wang et al. 2019 (ICCV)** and **Tian et al. 2025 (TDSC):** see §1.1. Wang compares the model-output attacker with a true-label attacker at matched performance. Tian's FD-AIA trains an attack model on two models' predictions for the same person.

---

## 7. Composition, linkage, collusion and multiple recipients

**Ganta, Kasiviswanathan, Smith, *Composition Attacks and Auxiliary Information in Data Privacy*, KDD 2008, pp. 265–273.** Crossref [M]; arXiv:0803.0032 [P].
- An adversary combines **independent anonymised releases about overlapping populations** (an intersection attack) to breach k-anonymity-type schemes.
- Randomised notions such as DP "provably resist composition attacks".
- Deterministic releases enjoy no such property.

**Wang & Fung, *Anonymizing Sequential Releases*, KDD 2006, pp. 414–423.** Crossref [M]; author PDF [P].
- An organisation "releases a separate view for each data sharing purpose (such as classifying a different target variable)". A **global quasi-identifier across releases** links them.
- The current release T1 is generalised **given a fixed previous release T2**, so that the join becomes "lossy". The requirement, (X,Y)-privacy, covers k-anonymity and confidence bounding of sensitive inference.
- Information requirement: classification utility. Optimisation: greedy top-down specialisation (optimal is NP-hard).
- **The closest classical precedent for purpose-specific, sequentially designed releases defended against linkage.**

**Li, Chen, Li, Zhang, *Enabling Multilevel Trust in Privacy Preserving Data Mining*, IEEE TKDE 24(9):1598–1612, 2012.** Crossref [M]; arXiv:1104.0459 [P].
- Differently perturbed copies go to miners at different trust levels; colluding miners combine copies (a "diversity attack").
- The noise is **correlated across copies** so that colluders reconstruct no better than with the best single copy in their collection, under linear least-squares estimation. Copies can be generated on demand.

**Taylor, Vippathalla, Coon 2026.** See §1.1. [P]
- **Objective:** per request k, min over p(r̂_k | r̂^{k−1}, x) of −U s.t. I(R̂_k; X) ≤ ε_k and the collusion constraint I(R̂_k, R̂^{k−1}; X) ≤ δ_k.
- **Collusion model:** worst case, "one malicious actor may gain access to the data supplied to all parties up to the most recent release". Subset collusion is only proposed as an extension (§IX).
- **Optimisation:** Blahut–Arimoto style, globally optimal per step for distortion utility, local for MI utility. Experiment on UCI Adult with |X| = 32.
- **Explicitly online:** "As she does not have knowledge of future requests at the time of a data release, the data handler cannot jointly optimise all releases."
- No code.

**Zhang & He, *DProvDB: Differentially Private Query Processing with Multi-Analyst Provenance*, SIGMOD 2024 (PACMMOD, per arXiv PDF and repository).** arXiv:2309.10240 [P][A]; code `DProvDB/DProvDB` (cited in paper, opened).
- Multi-analyst DP. When all analysts collude, the loss lies between max and sum of per-analyst budgets (Thm 3.2, "trivial" bounds).
- The additive Gaussian approach derives each analyst's synopsis from one hidden global synopsis (**shared, correlated noise**). Collusion loss is then bounded by the global synopsis's budget.

**Zamani, Oechtering, Skoglund, *Multi-User Privacy Mechanism Design with Non-zero Leakage*.** arXiv:2211.15525 [P].
- One disclosed U serves K users, each demanding a part of the data, under a single common leakage constraint. No collusion or side-information model.
- The ITW 2023 venue appears only in a secondary listing in the repository notes [NV].

---

## 8. Analytic note 1: the joint-versus-sequential premise

### 8.1 Statement

**Setup.** Releases are R₁ = f₁(X) and R₂ = f₂(X), with f_r in function classes F_r. Let J(f₁, f₂) be any criterion over the individual and pair views (risk on R₁, R₂ and (R₁, R₂)). Utility caps are per task, U_r(f_r) ≤ c_r.

**A joint design is realisable sequentially.**
- A joint designer solves min over (f₁, f₂) ∈ F₁ × F₂ of J.
- A sequential designer with full look-ahead knows purpose 2's task, cap and the pair criterion when choosing f₁. It chooses f₁ to minimise V(f₁) = min over f₂ of J(f₁, f₂), then chooses f₂ as a best response.
- Since min over (f₁, f₂) of J = min over f₁ of min over f₂ of J, every jointly designed deterministic pair (f₁\*, f₂\*) is realisable sequentially: pick f₁ = f₁\*, then f₂ = f₂\*. The reverse holds trivially.

**Conditioning on the earlier release adds nothing here.** For deterministic releases computed from the same X, letting the second release depend on R₁ is vacuous: R₁ = f₁(X) is already computable from X (also noted in the JCV review).

**Where a "joint benefit" can come from.** Only from:
- (i) a **restricted sequential procedure**: a myopic first stage chosen without the coalition term, an online first stage where purpose 2 is unknown (Taylor), or a first release fixed by a third party;
- (ii) **computation**;
- (iii) **optimisation quality**: non-convex adversarial training, critic quality, sample efficiency, local optima.

Simultaneous gradient training and alternating (block-coordinate) look-ahead training are both local methods; neither carries an optimality guarantee here.

### 8.2 Randomness

**Independent randomness.** With randomised releases R_r ~ p_r(·|X) and independent private randomness, the pair channel is p₁(r₁|x)·p₂(r₂|x): conditionally independent given X.

**Shared randomness.** With shared randomness, or a second mechanism that sees the *realised* R₁ (Taylor's p(r̂_k | r̂^{k−1}, x); the "interactive" scheme of Erdogdu & Fawaz), every joint channel p(r₁, r₂ | x) = p₁(r₁|x)·p(r₂ | r₁, x) is achievable.

**Individual versus coalition leakage.** Individual leakages depend only on the marginal channels; coalition leakage depends on the coupling. Shared randomness therefore enlarges the achievable set of (individual, coalition) behaviours, in both directions:
- **It can lower coalition leakage.** Re-using or correlating noise limits what colluders gain. Independent noise instead lets colluders average.
  - Li et al. 2012: colluders' linear reconstruction is no better than with the best single copy.
  - DProvDB: collusion loss is bounded by the hidden global synopsis's budget.
  - Taylor §III: a repeated request reuses the earlier release, adding no collusion leakage.
- **It can raise it.** With shared U, R₁ = B ⊕ U and R₂ = U are each independent of B, yet together they reveal B.

**Order of design versus information.** "Joint versus sequential" must be compared under the *same* randomness model. A sequential design that sees realised earlier releases is as expressive as a joint design with shared randomness. A non-interactive one with independent noise is, in general, strictly less expressive.

**In this campaign's setting.** Releases are deterministic, so randomness is moot: the achievable set is exactly F₁ × F₂. Randomising over deterministic pairs (time-sharing, or FairGBM's randomised classifier) would convexify that set, but this is outside the contract.

### 8.3 Coalitions and declared attributes

Per-recipient protection of a declared attribute does not imply pair protection, even for deterministic functions of X.
- **Example.** Let X = (B₁, B₂) be uniform bits, SEX = B₁ ⊕ B₂, R₁ = B₁ and R₂ = B₂. Then I(SEX; R₁) = I(SEX; R₂) = 0, but I(SEX; R₁, R₂) = 1 bit.
- **Contrast with all-attribute measures.** Only these compose subadditively (Issa et al., Lemma 6; Z − X − Y holds for functions of X).
- **Consequence.** A per-recipient (local) criterion on the declared attribute cannot, by itself, certify the coalition. A pair term, or a certificate on the pair partition (FARE-style, k₁k₂ cells), is needed for any coalition claim.

### 8.4 Does any cited work claim otherwise?

**No.** Each sequential or online design in the literature is justified by an information or legacy constraint, never by superiority at equal information.

| Work | What it compares | Consistent with 8.1? |
|---|---|---|
| Taylor et al. 2026 | adaptive (history-conditioned) vs non-adaptive (product channels); sequential *because* "the data handler cannot jointly optimise" with unknown future requests | Yes. The gain is from coupling (8.2), not from order. |
| Erdogdu & Fawaz 2015 | online vs batch; Def. 3 "regret" = distortion gap; Thm III.4: no regret under HMM (with suitable budgets) | Yes. They formalise the gap as a cost of online information. |
| Wang & Fung 2006 | current release designed given a fixed previous release, "since it is not an option to anonymize previously released data" | Yes. A legacy constraint. |
| Li et al. 2012 | on-demand copies with correlated noise | Yes. Shared randomness is the lever. |
| JCV review in this repository (`results/pcrl_joint_complete_view_method_v1/review/PRIOR_WORK_DELTA_REVIEW.md` §5) | "any sequential policy is a particular joint policy" | Yes. One clarification: where every arm was told both purposes, the J–S12 gap reflects S12's frozen, myopic first stage (an algorithmic restriction). It is an information effect only if the first stage was denied purpose 2. |

---

## 9. Analytic note 2: who already defines this campaign's adversaries

**The auxiliary-data recipient adversary is not new.** "An output-only reader trained on auxiliary SEX-labelled releases" is already defined by:
- **Song & Shmatikov 2020:** D_aux of labelled (x, s) plus black-box access to the release map; an attack model trained on (E(x), s).
- **Jia & Gong 2018 (AttriGuard):** a classifier trained on users who disclose the attribute.
- **Elazar & Goldberg 2018:** a post-hoc attacker trained on released encodings.
- **Wang et al. 2019:** an attacker on model outputs, with a true-label "dataset leakage" reference.
- **Jayaraman & Evans 2022:** D_aux size and distribution, black-box/white-box/no-model access, and the imputation baseline.
- **Tian et al. 2025:** black-box AIA on predictions, trained via shadow models.
- **Stadler et al. 2024:** label-only and features adversaries on auxiliary D_A.
- **Yeom et al. 2018:** formal attribute experiment, with the Def. 6 simulator baseline.

Fredrikson 2014/2015 and Mehnaz 2022 define attribute inference where the attribute is a *model input*. That differs from SEX, which is not an input.

**The linked, colluding multi-release adversary is not new.**
- **Ganta et al. 2008:** composition across independent releases.
- **Wang & Fung 2006:** joins across purpose-specific sequential releases through a global quasi-identifier.
- **Li et al. 2012:** colluding miners pooling copies.
- **Taylor et al. 2026:** a malicious actor pooling all releases to date; subsets in §IX.
- **DProvDB 2024:** colluding analysts.
- **Tian et al. 2025:** combining two models' predictions for the same person.
- **Formalised as side information at the adversary:** Sankar et al. 2013 (an encoder that knows the user's other view), Liao et al. 2019 and Issa et al. 2020.

**What may be specific to this campaign, as a setting rather than an adversary:**
- two recipients with *different* permitted prediction tasks;
- deterministic probability-plus-decision releases;
- a declared attribute that is not an input;
- perfect linkage assumed;
- fitted readers chosen on validation with per-task utility caps.

Even this combination is a bounded-search observation, not evidence of priority.

---

## 10. Reuse of the repository's earlier literature assessments

| Earlier statement (file) | Verdict today |
|---|---|
| Stadler note: definitions, Thm 2, S\*, Assumption A, the single-attribute quote, the label-only adversary as fundamental leakage (`results/combined_empirical_preparation_v1/notes/literature_A/stadler2024_lpl.md`) | **Confirmed** on the PMLR PDF. |
| "A useful release leaks what its label leaks (`stadler2024lpl`)" (`results/combined_analysis_paper_v1/CLOSEST_PRIOR_ANALYSES.md` §3 item 4; manuscript l.~148) | **Corrected.** An over-application (§1.2). |
| "Taylor et al. 2026: the only located collusion-aware release mechanism" (`CLOSEST_PRIOR_ANALYSES.md` §1) and reconciliation item 11 ("no learned-representation method with a declared sensitive set and a coalition guarantee was located") | **Narrowed.** Li et al. 2012, Wang & Fung 2006 and DProvDB 2024 are also collusion- or linkage-aware release designs. Item 11's corrected statement stands as a bounded-search observation. |
| Taylor note: v2 date, constraints, all-prefix collusion, §IX subsets, Adult \|X\| = 32 (`.../literature_A/taylor2026_collusion.md`) | **Confirmed**, including the online quote. |
| `tian2025fairnessprivacy` "journal details UNVERIFIED" (`CLOSEST_PRIOR_ANALYSES.md`) | **Resolved:** IEEE TDSC 22(5):4654–4670 (Crossref). Update the bib. |
| `wang2019balanced` [V-prep] description (dataset vs model leakage at matched F1) | **Confirmed** [P]. |
| Elazar & Goldberg Table 3 Δ; Song & Shmatikov D_aux attacker and de-censoring | **Confirmed.** Song's "PIPA pairs" detail was not re-checked [NV]. |
| LEACE footnote on stacked softmax leakage and the §7 conjecture | **Confirmed** (NeurIPS PDF). |
| FARE: certificate on the DP distance of any classifier reading only cells; k-means restricted-encoder comparison | **Certificate confirmed.** The App. C detail was not re-read [NV]. |
| Fredrikson 2014 pp. 17–32 and Mehnaz 2022 pp. 4579–4596 | **Cannot verify pages** (USENIX pages behind a challenge). Venues confirmed from PDF front matter. |
| Claims-foundation notes (`origin/research/pcrl-claims-foundation-v1:results/pcrl_claims_foundation_v1/PRIOR_ART_NOTES.md`): Calmon & Fawaz Thms 1–2 and Remarks 1 and 3; privacy-funnel non-convexity from the utility constraint; Sankar Thms 2–3; Liao (measure only); Erdogdu & Fawaz (incremental leakage, Thm III.3); CPFSI "side-information = task label"; Li et al.; Issa Def. 6, Thm 6, Lemma 6 | **Confirmed.** Add from today: Erdogdu & Fawaz's online-vs-batch regret (Def. 3, Thm III.4). |
| Claims-foundation: CPF (Rodríguez-Gálvez et al.) "UNVERIFIED" | **Resolved:** CPF = inf I(S;Y) s.t. I(X;Y\|S) ≥ r (arXiv:2006.06332 v3). The conditioning is on S, not adversary side information. |
| Frozen-tree novelty check (`results/pcrl_frozen_tree_control_completion_v1/TARGETED_METHOD_NOVELTY.md`): FAGTB adversary input, biased-GTB start, multiple adversary steps, "first adversarial … decision trees" claim; FairGBM randomised-classifier guarantee; EiFFFeL leaf flips; Ren 2015; Zhang Prop. 2 | **Confirmed.** Add the FairGBM authorship discrepancy. Patent and library docs not re-opened [NV]. |
| JCV review: "any sequential policy is a particular joint policy"; "J vs S12 conflates foreknowledge…" | **Confirmed, with the clarification in §8.4.** |
| Combined-preparation priority list: "No published learned-representation method designs releases to resist collusion" | **Not contradicted** by today's bounded search. FFVAE has per-consumer sensitive sets but no coalition analysis in the sections read. |

---

## 11. Gaps in this review

- **Pages not opened.**
  - Kamiran et al. 2010 (ICDM leaf relabelling).
  - The Allerton 1999 proceedings page for the information bottleneck (venue from secondary listings).
  - USENIX landing pages (challenge; PDFs were read instead).
  - OpenReview pages for ICLR papers (challenge; ICLR virtual pages and PDF headers were used instead).
- **Searches.** Privacy-attack venues and 2025–2026 preprints were sampled, not exhausted.
- **Not covered as separate entries.** Sweeney-style linkage, Narayanan–Shmatikov de-anonymisation, and the wider DP-composition literature are covered only through Ganta et al. and DProvDB.
- **Scope of every "not found" statement.** It means not found in the sources listed in `README.md`.

---

## 12. Field matrix (one row per work; Phase 2 comparisons will cite these rows)

**Columns:** Obj = objective; Train = what the method or attacker uses in training; Release = release contract; Protects = protected information; Coal/SI = coalition, multi-recipient or side-information access; Opt = optimisation; Guar = guarantee and its scope. "—" means not applicable.

| # | Work | Obj | Train | Release | Protects | Coal/SI | Opt | Guar |
|---|---|---|---|---|---|---|---|---|
| 1 | Stadler 2024 | show LPP vs utility incompatibility | theory: known law; experiments: Y, plus S for censoring | any feature map Z = f(X), including predictions | **every** attribute beyond the true label | none | theorem; ERM/GRAD/MAX-ENT experiments | impossibility for all-attribute LPP (finite alphabets, Assumption A) |
| 2 | IB 1999 | min I(X̃;X) − βI(X̃;Y) | known p(x,y) | stochastic code | — | none | Blahut–Arimoto-type | none |
| 3 | Privacy funnel 2014 | min I(S;Y) s.t. I(X;Y) ≥ R | known P_{S,X}; map reads X only | deterministic merged code | declared S | none | greedy merging (local) | bounded-cost gain ≤ 2√2·L·√I(S;Y) (known law) |
| 4 | Calmon–Fawaz 2012 | min average/maximum leakage s.t. distortion | known p_{S,Y}; map may read S | randomised U | declared S | adversary knows prior and map | convex program | exact optimum (finite, known law) |
| 5 | Sankar 2013 | rate–distortion–equivocation region | statistics, or the realised side information | sanitised database | private attributes | side information at user/adversary; informed encoder | quantize-and-bin coding | asymptotic regions |
| 6 | Erdogdu–Fawaz 2015 | min distortion s.t. I(X̂^t; S^t) ≤ ε_t | known joint; online scheme sees past releases | randomised time series | private series | earlier releases (incremental leakage) | convex | optimum per scheme; zero online regret under HMM |
| 7 | Liao 2019 | conditional maximal α-leakage (measure) | — | — | any function of private data | adversary side information | — | side information cannot raise leakage if conditionally independent of the release given the private data |
| 8 | Issa 2020 | maximal leakage and its conditional form (measure) | — | — | any function of X | side information; composition | — | L(X→(Y,Z)) ≤ L(X→Z) + L(X→Y) when Z − X − Y |
| 9 | SRLIP 2020 | local information privacy robust to side channels | known law | randomised | declared S | every attribute subset as side channel | vertex enumeration + LP | ε-SRLIP by construction |
| 10 | Wang–Basciftci–Ishwar 2017 | trade-off regions under input constraints | known law | randomised | declared sensitive | none | IT analysis | S-only ⊆ useful-only ⊆ full-data regions |
| 11 | CPF 2021 | inf I(S;Y) s.t. I(X;Y\|S) ≥ r | S labels | stochastic representation | declared S | none | variational | none |
| 12 | CPFSI 2025 | I(s;z) − γI(z;x\|s) − βI(y;z\|s) | s and y labels | representation | declared s | none ("side information" = task label) | amortised variational | none |
| 13 | Edwards–Storkey 2016 | encoder vs adversarial critic | S labels | representation | declared S | none | alternating minimax SGD | none |
| 14 | Xie 2017 | encoder E(x,s) vs discriminator | s, y | representation | declared s | none | adversarial SGD | non-parametric equilibrium analysis |
| 15 | LAFTR 2018 | encoder, classifier, adversary | A, Y | representation (transfer) | declared A (DP/EO) | none | adversarial SGD | DP/EO gap ≤ optimal adversary objective |
| 16 | Zhang 2018 | predictor with projected adversary gradient; adversary reads Ŷ | attribute, Y | predictions | declared attribute (DP/EO) | none | modified SGD | at convergence, under conditions, the adversary gains nothing |
| 17 | Bertran 2019 | max I(U;Y) s.t. I(S;Y) ≤ k | U and S labels | stochastic transform of X | declared S | none | adversarial DNN game | trade-off bounds only |
| 18 | Zhao 2020 | min H(Y\|Z) − λH(A\|Z) | A, Y | deterministic representation | declared binary A | **none; Thm 3.1 assumes no side information about A** | adversarial training | adversary-error lower bound; JS-divergence trade-off |
| 19 | GAP 2017 | privatizer g(X,Y) vs adversary, distortion-constrained | private labels | sanitised data | declared private variable | none | constrained minimax | game-theoretic optima (binary/Gaussian); log-loss ⇒ MI |
| 20 | Hamm 2017 | min-diff-max filter | target and sensitive labels | filtered representation | declared attributes | none | minimax | ERM-style generalisation analysis |
| 21 | PPAN 2019 | randomised release, variational MI | sensitive labels | randomised release | declared sensitive | none | adversarial training | empirical match to analytic optima |
| 22 | Elazar–Goldberg 2018 | adversarial removal in text encoders | protected labels | representation | declared attribute | none | adversarial training | none (shows post-hoc recovery) |
| 23 | Song–Shmatikov 2020 | overlearning and de-censoring attacks | attacker: D_aux (x,s) + black-box encoder | representation (attacked) | — | — | — | threat-model anchor |
| 24 | LEACE 2023 | least-squares affine eraser, zero cross-covariance | concept labels | erased representation | declared concept, linear readers | none | closed form | linear guardedness for convex losses; not nonlinear, not 0–1 |
| 25 | FARE 2023 | restricted encoder (fair trees) + certificate | S, Y | cell index | declared S (DP) | none (pair certifiable via k₁k₂ cells: repository logic) | tree induction | high-confidence DP-distance bound for any classifier on cells |
| 26 | FFVAE 2019 | disentangled VAE, per-attribute latents | sensitive labels | representation minus b_i per consumer | consumer-specific sensitive sets | multiple consumers; no coalition analysis found | VAE with FactorVAE-style terms (not re-read) | none |
| 27 | Zhao–Gordon 2022 | statistical parity vs accuracy (theory) | — | classifier | declared A (parity) | none | LP (TV barycentre) | lower bound on group errors |
| 28 | FAGTB 2019 | boosting vs adversary on σ(F(x)) | S, Y | score | declared S (DP/EO) | none | new tree per round on combined pseudo-residuals | global-optimum remark only |
| 29 | FairGBM 2023 | proxy-Lagrangian group-rate constraints | S at training | classifier | declared S | none | dual ascent within boosting | approximate, for the randomised classifier only |
| 30 | EiFFFeL 2022 | post-hoc leaf relabelling toward parity | S | forest decisions | declared S | none | tree- or leaf-based relabelling | none |
| 31 | Ren 2015 | jointly relearn leaves, fixed structure | Y | forest predictions | — | none | global objective (not re-read) | — |
| 32 | Fredrikson 2014 | model inversion (attack) | attacker: model, target's other attributes + label, marginals | (attack) | genotype, a model input | — | MAP-type | — |
| 33 | Fredrikson 2015 | inversion with confidences; countermeasures | attacker: API (black/white box) | (attack) | input features, faces | — | — | — |
| 34 | Yeom 2018 | formal attribute/membership advantage | attacker: partial record, model, distribution | (analysis) | training-record attribute | — | — | advantage bounds (overfitting/influence) |
| 35 | Mehnaz 2022 | confidence and label-only AIA | attacker: non-sensitive attributes, black box | (attack) | attribute that is a model input | — | — | — |
| 36 | Jayaraman–Evans 2022 | AIA vs imputation; sensitive value inference | attacker: D_aux, varying model access | (attack) | record attribute | — | — | — |
| 37 | AttriGuard 2018 | evasion noise on public data | attacker trains on disclosing users; defender trains its own classifier | noised public data | declared attribute | none | evasion + randomised mechanism under a utility budget | none formal |
| 38 | Wang 2019 | leakage metrics + adversarial mitigation | gender labels | predictions | gender | none | adversarial removal at an intermediate layer | none |
| 39 | Tian 2025 | FD-AIA / FD-MIA attacks | attacker: shadow models; biased + fair predictions | (attack) | sensitive subgroup | **two model versions, same person** | — | — |
| 40 | Ganta 2008 | composition attacks; DP composes | — | anonymised tables | sensitive values | **independent releases, overlapping populations** | — | DP-type notions resist composition |
| 41 | Wang–Fung 2006 | anonymise the current release given the previous one | QIDs, sensitive attributes, class labels | generalised purpose-specific view | identity linkage + sensitive inference | **join with the earlier release** | greedy top-down specialisation | (X,Y)-privacy by construction |
| 42 | Li 2012 | multi-trust perturbed copies | data covariance | perturbed copies | original data (LLSE reconstruction) | **colluding miners** | correlated Gaussian noise | colluders no better than the best single copy |
| 43 | Taylor 2026 | max utility s.t. individual + collusion MI | known pmf; X and past releases | randomised channels | whole record X | **all-prefix collusion** (subsets proposed) | per-step Blahut–Arimoto style | per-step optimum (distortion); local (MI) |
| 44 | DProvDB 2024 | multi-analyst DP query answering | — | noisy answers | DP | **colluding analysts** | additive Gaussian synopses (shared noise) | collusion loss ≤ global-synopsis budget |
| 45 | Zamani 2023 | one U for K users, one MI leakage constraint | agent sees useful and private data | randomised U | private X | none | Functional Representation Lemma bounds | upper and lower bounds |
