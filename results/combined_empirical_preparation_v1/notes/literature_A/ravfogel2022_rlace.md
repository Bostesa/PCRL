# Ravfogel, Twiton, Goldberg, Cotterell (2022). Linear Adversarial Concept Erasure (R-LACE)

## Citation and version read
- S. Ravfogel, M. Twiton, Y. Goldberg, R. Cotterell. *Linear Adversarial Concept Erasure.* ICML 2022, PMLR 162:18400–18421.
  Landing page: https://proceedings.mlr.press/v162/ravfogel22a.html (PMLR date 2022-06-28).
- **Read in full, including Appendix A–B.11:** PMLR PDF https://proceedings.mlr.press/v162/ravfogel22a/ravfogel22a.pdf (22 pp. incl. appendix), fetched 2026-10-01.
- **Also checked:** arXiv:2201.12091 **v8** (17 Dec 2024; abstract page lists v1 28 Jan 2022 to v8 17 Dec 2024, "Accepted in ICML 2022; a revised version"). I read the passages that differ: §2.3 with footnote 4, §5.1, and §7. I did not compare v8 line by line in full.
- Code: https://github.com/shauli-ravfogel/rlace-icml (linked on p.1). Not inspected.

## Actual question (authors' scope)
Can a **linear** concept subspace be found and removed by an orthogonal projection, so that **linear** predictors can no longer recover the concept while the representation is otherwise preserved (p.1 §1; p.2 §2.3, eq. 3)? It is a post-hoc method on fixed, pretrained vectors (p.1).

## Formal guarantee and its assumptions
- The minimax game is min over θ, max over P ∈ P_k of Σ ℓ(y_n, g⁻¹(θᵀP x_n)). P_k is the set of rank-k-neutralising **orthogonal** projections P = I − WᵀW (p.3 eq. 3).
- **Prop. 3.1 (linear regression):** the equilibrium P removes the single direction Xᵀy (rank 1), after which the objective equals Var(y) (p.3; proof App. B.3, p.14).
- **Lemma 3.2 / Prop. 3.3:** Rayleigh-quotient losses (PLS) have a closed form: remove the top-k eigenvectors (p.4; App. B.2).
- **Prop. 4.1:** INLP is not minimal-rank for regression. **Prop. 4.2:** INLP is optimal for Rayleigh-quotient losses (p.5).
- **Classification has no guarantee.** It uses a Fantope convex relaxation (eq. 17) solved by alternating projected gradient. App. B.4 says alternating optimisation "is not guaranteed to find that equilibrium". The authors therefore pick the P that maximises a linear classifier's dev-set loss (p.15 B.4, p.16 B.6).
- **Correction in the arXiv version (my reading of v8 §2.3, footnote 4):** "An earlier version of this paper erroneously claimed that Eq. (2) is a convex–concave game… The game is actually convex–convex." v8 reorders the game to max–min. The ICML text (§3.4, App. B.4) still calls the relaxed game concave–convex.
- Every guarantee is against **linear** predictors only. App. B.1 (p.13): "The method is not expected to be robust to nonlinear adversaries."

## Recoverability, causal use, fairness or privacy?
- **Recoverability by linear probes** (post-projection accuracy) plus fairness (TPR-GAP, §5.2) and bias association (WEAT, §5.1).
- No privacy framing.
- No causal-use analysis. The related work points to amnesic probing for causal use (p.8 §6).

## What is exposed
- **Exposed:** the projected representation XP. Downstream deep classifiers are re-fitted on it (§5.2).
- **Not attacked:** task outputs.

## Datasets and task–attribute relationships
1. **GloVe 300-d, uncased.** Binary gender is induced from the he–she projection: the 7,500 most male- and female-biased words. Splits are 7,350/3,150/4,500 (App. B.6, p.16).
2. **Bias-in-Bios** with BERT [CLS], reduced to 300-d by PCA. Gender is the protected attribute and profession is the task. Splits are 255,710/39,369/98,344; R-LACE is fit on the first 100k only (App. B.7). Profession is correlated with gender (σ(GAP, %women) is reported).
3. **CelebA raw pixels** at 50×50 grey (2,500-d). Concepts are glasses, smile, mustache, beard, bald and hat. These are qualitative only (§5.3).

## Removal methods and attacker families
- **Removal methods:** R-LACE rank k = 1…20 (GloVe) and k = 1…100 (BERT). Baselines are INLP, PCA/Bolukbasi, and BERT finetuned with gradient-reversal adversaries (linear, and an MLP with one hidden layer of 300) (§5.2).
- **Attackers:**
  - sklearn logistic regression and linear SVM at test time (App. B.6);
  - RBF-SVM and a 1-hidden-layer ReLU MLP (128) on GloVe (§5.1, p.6);
  - main-task deep classifiers.

## Adaptive knowledge and query access
- Attackers are trained on labelled projected representations: train-split labels, fresh test evaluation.
- No attacker is told the projection. With an orthogonal projection the removed coordinates are gone, so knowing P adds little for linear recovery (my inference).
- Single release per input.

## Utility protocol and model selection
- **Utility measures:** profession accuracy after re-fitting a linear head on projected representations, SimLex-999 correlation, and nearest neighbours.
- **Model selection:** learning rate 0.005 was "chosen by experimenting with the development set". The adversary checkpoint "which yielded the highest classification loss" is selected using a freshly trained linear classifier (App. B.6–B.7).
- **Runs:** 5 runs with random initialisations, mean ± sd.

## Individual vs multiple recipients
Individual only. Each released representation faces one attacker at a time.

## Strongest results
- **GloVe:** a rank-1 projection drops linear gender accuracy to about 50%, while INLP does not reach majority even at rank 20 (Fig. 2, p.6). SimLex moves from 0.399 to 0.392.
- **BERT-frozen:** rank-1 R-LACE gives gender accuracy 52.48 and profession 78.86 (vanilla 79.14). INLP at rank 100 gives profession 71.94 (Table 2, p.7).
- **Nonlinear recovery after linear erasure (p.6 §5.1):** "as expected with a linear information removal method, non-linear classifiers are still able to recover gender: both RBF-SVM and a ReLU MLP with 1 hidden layer of size 128 predict gender in above 90% accuracy." The paper repeats INLP's recommendation to feed the output only to linear classifiers.
- **Adversarial training passes its own adversary but not fresh ones (§5.2, Table 2, footnote 9, p.7):** BERT-adv adversaries "converged to close-to-random gender prediction accuracy; but this did not generalize to new adversaries in test time". Test-time gender accuracy is 99.57 (MLP adversary) and 99.23 (linear adversary).

## Limitations relevant to our analysis
- Only a binary concept is studied.
- The textual data and attacker sweep are small.
- The nonlinear-recovery check is one sentence on GloVe. It is not run on BERT and has no table.
- Dev-set selection of the adversary is done with a linear classifier, so selection is matched to the certificate class.
- There is no defense-aware attacker, no output-surface attack, and no multiple-release evaluation.

## What our work repeats, extends or contradicts
**Durable-guarantees (AAAI "Outputs Leak What They Use")**
- **Repeats:** the core observation that a linear certificate passes while nonlinear attackers recover the attribute is already stated for GloVe (§5.1). Gradient-reversal adversaries pass their training adversary and fail fresh ones (Table 2, footnote 9).
- **Extends:**
  - 67 certificate-approved configurations on tabular data across six datasets;
  - several attacker families (XGBoost, MLP, LoRA reader) and a defense-aware tier;
  - the output surface;
  - a utility-cost analysis.
- **Not run:** the AAAI paper does not run R-LACE or INLP; its table runs LEACE.
- **Contradicts:** nothing.
- **Citation:** AAAI references.bib does **not** contain the ICML R-LACE entry. It cites only the EMNLP kernel paper (see ravfogel2022_kernel.md).

**PCRL (NeurIPS "One Encoder, Many Purposes")**
- PCRL's per-purpose linear-R² constraint is a linear-guardedness target in the R-LACE/LEACE family. R-LACE's own scope warning (App. B.1) applies to every PCRL "bound".
- PCRL Prop. (rank-r LoRA erasure threshold, S5-experiments.tex:65) says "The framework is RLACE's". That attribution is reasonable, since R-LACE's rank-k projection game has the same structure.
- **Citation errors in the PCRL source** (`Formatting_Instructions_For_NeurIPS_2026 (21)/bibliography.bib`):
  - `ravfogel2022rlace` lists authors "Ravfogel, **Belrose**, **Gonen**, Twiton, Goldberg, Cotterell". This is wrong: the PDF lists Ravfogel, Twiton, Goldberg, Cotterell.
  - That key is a duplicate of the correct `ravfogel2022linear`.
- **Mischaracterisation (S2-related-work.tex:4):** the text calls R-LACE among "subsequent refinements … to oblique projections". This is wrong on two counts: R-LACE (2022) predates LEACE (2023), and it uses orthogonal rank-k projections (§2.3).
- **Extends:** PCRL handles multiple purposes and conflicting permissions, and adds cross-purpose MLP/XGBoost auditors.
- **Contradicts:** nothing.

## Stated scope vs my inference
- **Stated in the paper:** everything above with a page or section reference.
- **My inference:**
  - that knowing P adds little for linear recovery;
  - that dev-set selection is matched to the certificate class.
