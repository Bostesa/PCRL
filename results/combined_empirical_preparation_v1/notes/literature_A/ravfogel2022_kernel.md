# Ravfogel, Vargas, Goldberg, Cotterell (2022). Kernelized concept erasure (EMNLP 2022)

## Citation and version read
- **Venue and title:** EMNLP 2022 main conference, pp. 6034–6055, doi 10.18653/v1/2022.emnlp-main.405. The ACL Anthology landing page (https://aclanthology.org/2022.emnlp-main.405/) gives the title **"Adversarial Concept Erasure in Kernel Space"**. The **PDF itself is titled "Kernelized Concept Erasure"** (p.1, 6034). Both titles refer to the same paper; cite the Anthology title and note the alias.
- **Read in full, including Appendices A–A.7 (Tables 4–8):** https://aclanthology.org/2022.emnlp-main.405.pdf (22 pp.), fetched 2026-10-01.
- **Code:** https://github.com/shauli-ravfogel/adv-kernel-removal (p.1). Not inspected.

## Actual question (authors' scope)
- Can the linear minimax erasure game of R-LACE be **kernelized** so that **non-linearly** encoded concepts are erased (p.1 abstract; §3)?
- Is there "a unique kernel such that, for any choice of non-linear predictor, the adversary cannot recover gender" (the "exhaustive RKHS hypothesis", §5, p.6)?

## Formal content and its assumptions
- **Lemma 1** (a minimax representer lemma, local optima only, with no regularizer), **Lemma 2** and **Theorem 1:** the RKHS game is equivalent to the Euclidean game in eq. 7 (p.3–4; App. A.1–A.2).
- **Proposition 1:** the cost is O(N⁴).
- **Nyström approximation** (L = 1024 landmarks) then gives a game that has the same form as the linear one (eq. 10).
- **A pre-image MLP** maps the neutralized features back to input space (eq. 11; §3.4).
- **There is no guarantee against any adversary.** The theory concerns tractability, not protection.

## Recoverability, causal use, fairness or privacy?
- **Recoverability** by kernel SVMs and an MLP. Bias association (WEAT) is reported too (§5.3).
- Fairness motivation only.
- No privacy and no causal-use analysis.

## What is exposed
- The pre-image representation r(x) in input space is exposed.
- No task outputs are attacked.

## Datasets and task–attribute relationships
- **GloVe word vectors:** binary gender labels from Ravfogel et al. (2020); splits 7,350/3,150/4,500 (App. A.3, p.16). Gender is **linearly separable** in the original space, above 99% with every kernel (§5, p.6).
- **Bias-in-Bios:** BERT [CLS] representations for an extrinsic profession test. Only the RBF kernel is used (§6.2).

## Removal methods and attacker families
- **Neutralizing kernels:** Poly, RBF, Laplace, Linear, Sigmoid, EasyMKL and a uniform multiple-kernel combination, each over hyperparameter grids (App. A.3).
- **Attackers (sklearn):**
  - SVMs with the **exact** kernel;
  - transfer kernels with fixed hyperparameters (App. A.7);
  - an MLP with one hidden layer of 128 units.

## Adaptive knowledge and query access
- In the "same adversary" test the attacker shares the neutralizing kernel and its hyperparameters. This is a form of defense-matched attacker.
- In the "transfer" test the attacker uses a different kernel or different hyperparameters.
- No attacker knows the pre-image network.
- Each input is released once.

## Utility protocol and model selection
- **Utility:**
  - SimLex-999 correlation: 0.400 originally, 0.387–0.396 after the intervention (§6.1);
  - profession accuracy with a linear classifier on the pre-images: 74.19 ± 0.056 vs 76.93 original (§6.2).
- **Model selection:** "choose the projection matrix P which leads to the biggest decrease in the linear classification accuracy on the evaluation set" (App. A.3). That linear check reads 50.59 ± 0.04 against a majority baseline of 50.58 (App. A.3.1).
- **Runs:** four runs with random restarts, reported as mean ± sd (§4).

## Individual vs multiple recipients
Individual only.

## Strongest results
- **Same-kernel protection is partial** (Table 1, p.7). Mean gender accuracy is Poly 0.59 ± 0.15, RBF 0.69, Laplace 0.75, Linear 0.54, Sigmoid 0.49, EasyMKL 0.69 and UniformMK 0.49. In the authors' words, protection holds "for most—but not all—kernels".
- **Transfer fails** (Table 2, p.8). For every neutralizing kernel, the MLP recovers gender at 0.97. Poly adversaries with other hyperparameters reach 0.98. The authors call this "a complete lack of generalization of our concept erasure intervention to other types of non-linear predictors" (§5.2).
- **Conclusion** (§8, p.9): "Exhaustive concept erasure and protection against a diverse set of non-linear adversaries remains an open problem."

## Limitations relevant to our analysis
- Only one binary concept is studied, on word vectors where it is already linearly separable.
- The checkpoint is selected against a linear probe.
- Attacker hyperparameters are fixed and there is no tuning sweep.
- There is no confidence analysis beyond 4 seeds.
- There is no defense-aware attacker, no output surface and no combination of releases.

## What our work repeats, extends or contradicts
**Durable-guarantees (AAAI)**
- **Already cited.** paper.tex:152 reads "showed that erasing a concept against one class of nonlinear attackers does not protect against a different class". This matches §5.2 and Table 2 accurately.
- **Repeats:**
  - The AAAI MMD/HSIC kernel-statistic projections that fail against XGBoost, MLP and LoRA attackers are a tabular-data instance of this non-transfer.
  - Obliviator's failure is another.
- **Extends:** defense-aware tiers, the output surface, consistent utility, and many configurations.
- **Contradicts:** nothing.

**PCRL**
- PCRL does not cite this paper.
- It shows that a non-linear erasure (including kernel-dependence guided erasure) does not certify against other non-linear families. PCRL therefore cannot upgrade its linear R² bound to a non-linear claim by adding one kernel or HSIC term. PCRL's own cross-purpose MLP/XGBoost audits are consistent with that caution.

---

## Which "Ravfogel et al. 2022" did a reviewer most plausibly mean? (ambiguity record)
The reviewer citation is **known only from the meeting summary**. No official review text is available (AGENT_CONTEXT.md).

| Candidate | Statement that matches "nonlinear attackers recover what linear erasure removes / empirical analyses of erasure" | Cited by our manuscripts? |
|---|---|---|
| ICML 2022 R-LACE (arXiv 2201.12091) | **Direct:** after rank-1 linear erasure, "RBF-SVM and a ReLU MLP … predict gender in above 90% accuracy" (§5.1, p.6). App. B.1: "not expected to be robust to nonlinear adversaries". Also: adversarially trained BERT passes its own adversary but fresh adversaries recover gender at 99%+ (Table 2, fn 9). | **PCRL NeurIPS:** yes, under two keys (`ravfogel2022linear`, correct; `ravfogel2022rlace`, wrong author list). **AAAI:** no. |
| EMNLP 2022 kernel erasure | **Extension:** even **non-linear** (kernel) erasure fails against other non-linear attackers; the MLP recovers at 0.97 (Table 2). This is an empirical analysis of erasure. | **AAAI:** yes (`ravfogel2022kernelized`, Anthology title, pp. 6034–6055). **PCRL NeurIPS:** no. |

**Assessment (inference):**
- If the critique is that *linear* erasure leaves the attribute recoverable by non-linear attackers, the **ICML R-LACE** paper is the more literal source.
- If the critique is that the paper's systematic re-testing against stronger attacker classes was already shown, the **EMNLP** paper fits better. It is about non-transfer *across non-linear classes* and is the only one of the two the AAAI manuscript cites.
- Neither can be settled without the review text.

**Recommendation:** cite both, and state precisely what each shows.
- **R-LACE:** a linear certificate does not stop non-linear recovery, and adversarial training does not generalise to fresh adversaries.
- **Kernel paper:** kernel erasure does not transfer across kernels or to an MLP.
- **Fixes to the PCRL bib:** remove the duplicate R-LACE key with the wrong authors, and correct the "subsequent … oblique projections" wording.
