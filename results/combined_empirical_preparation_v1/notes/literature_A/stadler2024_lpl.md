# Stadler, Kulynych, Gastpar, Papernot, Troncoso (2024). The Fundamental Limits of Least-Privilege Learning

## Citation and version read
- **Citation:** T. Stadler\*, B. Kulynych\*, M. C. Gastpar, N. Papernot, C. Troncoso. *The Fundamental Limits of Least-Privilege Learning.* ICML 2024, PMLR 235:46393–46411. Landing page: https://proceedings.mlr.press/v235/stadler24a.html (PMLR date 2024-07-08).
- **Read in full, including Appendix A (proofs) and Appendix B (experiments, Figs 5–10).** Source: the PMLR PDF via the landing page's link, https://raw.githubusercontent.com/mlresearch/v235/main/assets/stadler24a/stadler24a.pdf (19 pp.), fetched 2026-10-01.
- **Also fetched:** arXiv:2402.12235v2 (26 Jun 2024). It has the same structure. I did not diff it line by line.

## Actual question (authors' scope)
- Formalise the **least-privilege principle (LPP)**: a representation should be useful for task Y but should prevent inference of **any** other attribute (p.1–2).
- Determine whether high-utility representations can satisfy it.
- The authors tie this explicitly to GDPR **purpose limitation** (§3.4 "Interpretation", p.5).

## Formal setting (exact)
- **Spaces:** X, Y, S are **discrete and finite**, the domain is non-trivial with full support (p.2), and Z = f_E(X) may be randomised.
- **Assumption A (strictly positive posterior):** P_{Y|X}(y|x) > 0 for all x, y (p.2).
- **Utility:** Arimoto information I_α(Y;Z) with α ∈ {1, ∞}. α = ∞ gives log of normalised accuracy (eq. 1, p.3).
- **Leakage:** a Bayes-optimal adversary's multiplicative gain I_∞(S;W) (eqs 2–4, p.4). It is measured in **expectation over the population**, which the authors note is weaker than LDP (§5, p.9).
- **Def. 1, unconditional LPP:** sup over **all attributes S ≠ Y** with Markov chain S−X−Z of I_∞(S;Z) ≤ γ (eq. 6, p.4).
- **Def. 2, LPP:**
  - The condition is sup over **all attributes S** with S−(X,Y)−Z of I_∞(S;Z | Y) ≤ γ (eq. 8, p.5).
  - In words, the adversary sees **(Z, Y)**, Y being the *true task label*. The gain is measured over the **fundamental leakage** Pr[S = Ŝ(Y)].
  - The supremum equals conditional **maximal leakage** L(X→Z|Y) (eq. 9; Issa et al.).
- **Thm 1:** under Assumption A, unconditional LPP with parameter γ and I_α(Y;Z) > γ cannot both hold (p.4).
- **Thm 2, the main result:** under Assumption A, "Z = f_E(X) satisfies the LPP with parameter γ" and "I_α(Y;Z) > γ" cannot both hold (p.5).
  - **Proof (App. A, p.13–14):** strict positivity makes supp(X|Y=y) = supp(X). Hence L(X→Z|Y) = L(X→Z), and I_α(Y;Z) ≤ L(Y→Z) ≤ L(X→Z) ≤ γ.
  - The supremum is attained by a **constructed "maximally revealing" (shattering) attribute S\*** over X × Z (Def. 4, Prop. 3, p.12). S\* is not a natural or declared attribute.
  - Remark 1 (p.13): Assumption A can be weakened.
- **Prop. 1 / Cor. 1–2:** perfect LPP (γ = 0) is possible iff X−Y−Z. This holds, for example, if Y = g(X) is deterministic. Under A, it is possible only for constant or random maps (p.5).
- **Prop. 2:** ε-LDP implies ε-LPP (p.6).

## What it quantifies over, and what follows for a FINITE declared set S
- **The quantifier ranges over every attribute** satisfying the Markov chain, i.e., every (randomised) function of (X, Y), including the constructed S\*.
- **For a finite declared set {S₁…S_k} (my inference from the proof structure; the authors say this in words):**
  - The theorem does **not** say that any particular declared attribute must leak.
  - A representation could have zero gain over Y on S₁…S_k and still high utility, e.g., if each S_i ⊥ X | Y. Such a representation violates LPP only through other attributes (such as S\*).
  - Authors' own statements:
    - §2 and §4.2 "Takeaways" (p.9): "although it is sometimes possible to restrict information leakage for a single attribute, the learned representations do not fulfil the LPP".
    - §5 Limitations (p.9): restricting the class of targeted attributes, e.g., to "efficiently identifiable" ones, "might" give a more favourable trade-off but "deviate[s] from the promise of LPL to leak 'nothing else'".
- **Therefore:**
  - Stadler does **not** prohibit restricted policies on a finite declared S.
  - Our methods do **not** "evade" the theorem. They simply do not claim LPP.
- **Do not overstate:** the theorem also needs finite discrete Z for its closed forms (Lemma 1). Continuous learned representations are outside its literal scope (inference).

## Does it cover task outputs?
- **Theorem:**
  - It applies to *any* feature map Z = f_E(X), so a released prediction map Ŷ = f(X) is itself a "Z" to which Thm 2 applies (my inference: a useful prediction map with I_α(Y;Ŷ) > γ violates γ-LPP under A).
  - The **baseline** is the *true* label Y given to the adversary, not the model's prediction Ŷ.
- **Empirics (§4.1, Fig. 3, p.7):**
  - The authors **measure fundamental leakage empirically** with a *label-only adversary* (frequency counts of S|Y over D_A) across 12 × 12 LFWA+ task/attribute pairs, and relate it to Pearson correlation (e.g., Attractive vs Male, r = −0.3094).
  - They do **not** attack a trained model's output predictions as a separate surface.

## Recoverability, causal use, fairness or privacy?
- **Recoverability** (attribute inference), framed as privacy / purpose limitation.
- Fairness is not the target.
- No causal use.

## Empirical protocol
- **Data and model:** LFWA+, 13,143 images. 20% go to the adversary as D_A; the rest are split 80/20 into train and eval. The model is CNN256 (Melis et al.), using the last-layer representation (p.6).
- **Adversaries:**
  - Label-only: Ŝ(Y).
  - "Features adversary": a Random Forest with 50 trees on (z, y, s), with **black-box access to f_E** for the auxiliary examples (p.7).
- **Utility:** Ĩ_∞(Y;Z) on the eval set.
- **Repetitions:** 5 per pair, averaged. No CIs are shown.
- **Learning techniques:**
  - ERM;
  - gradient-reversal censoring (GRAD, parameter 100), censoring the attribute that leaks most;
  - App. B.3: MAX-ENT adversarial representation learning on Adult (income task; age, education, race, sex).
- **Other settings:**
  - App. B.4: gradients in collaborative learning;
  - App. B.5: hidden layers;
  - App. B.6: ResNet-18;
  - App. B.7: Texas Hospital (TabNet).
- **Recipients:** a single recipient. No multiple releases, coalitions or defense-aware tier beyond black-box f_E.

## Strongest results
- **Theorem 2.**
- **Fig. 4 (p.8):** "for every task, there is at least one sensitive attribute with ΔAdv > 0", where ΔAdv = Ĩ_∞(S;Z|Y) − Ĩ_∞(Y;Z).
  - Censoring creates a "whack-a-mole" effect: "as we censor one attribute, leakage of another attribute increases".
  - Note that empirically the violating attribute **is** a natural attribute among the 12 tested.
- **App. B.3 (MAX-ENT):** the inference gain exceeds utility for 2 of the 4 Adult attributes.

## Limitations relevant to our analysis
- The theory assumes finite discrete spaces and Assumption A.
- The leakage measure is an average-case gain with a Bayes-optimal adversary.
- The empirical adversary is a single RF family.
- Utility is measured in-distribution.
- There is no notion of a declared set with permitted versus forbidden recipients.

## What our work repeats, extends or contradicts
**Durable-guarantees (AAAI)**
- **paper.tex:171–175** says Stadler "prove that a representation cannot combine high task utility with preventing inference beyond the task label, so a leakage floor exists. Their theorem is stated slightly more generally than our output-side phrasing. We contribute its measurement."
  - **Imprecise:** the theorem is about *all* attributes, including constructed ones, under A. It is not about a given S.
  - **Overclaims:** Stadler already **measured** fundamental leakage with a label-only adversary (§4.1, Fig. 3). AAAI's "label-coupling predictor" (GBM recovering s from y alone) is the same construct.
  - **What AAAI genuinely adds:**
    - relating this label-only quantity to **trained-model output leakage** (attack on ŷ) across 27 cells;
    - relating it to **removal cost**;
    - external pre-registered cells.
  - **Missing from AAAI:** a defense-aware tier and certificate-approved configurations.
- **paper.tex:474:** "These approach the leakage floor established by \citet{stadler2024}." Stadler establishes no numeric floor for these cells. The comparator is AAAI's own label-only baseline. Suggested rewording: "approach the label-only (fundamental-leakage, Stadler et al.) baseline".
- **Contradicts:** nothing.
  - AAAI's high-noise survivors leave output surfaces at the label-coupling floor. This is consistent with Stadler's statement that LPP-respecting maps still leak fundamental leakage.

**PCRL**
- **Not cited.** The NeurIPS bibliography.bib has no Stadler entry. Yet PCRL's motivation *is* purpose limitation under GDPR (S1-introduction.tex:4), which Stadler formalise.
- PCRL bounds linear R² on a **finite declared** set of disallowed attributes per purpose. Stadler does not prohibit this.
- PCRL must not describe h_p as revealing only purpose-relevant information. Under Assumption A, any useful h_p leaks some attribute beyond Y (Thm 2).
- **Extends:** multiple purposes with conflicting permitted/forbidden sets, and a concatenation (coalition) audit. Stadler has a single task and a single recipient.
- The PCRL claims-foundation notes (PRIOR_ART_NOTES Part 4, row (a)) already list Stadler as a closest antecedent for (a) and (b) ("Def. 2: leakage over a baseline view"). That is consistent with this reading.
