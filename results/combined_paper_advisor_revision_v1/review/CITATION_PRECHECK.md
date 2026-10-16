# Citation precheck: claims the owner is likely to change

**Role.** Claims and provenance reviewer, 2026-10-16. This file suggests; it edits nothing. No priority or novelty claim
is suggested.

**Manuscript lines** refer to `papers/combined_empirical_v2/main.tex` at `133d2c740` (SHA-256 prefix `cf0b25f2`).

**Repository keys** are as in `CLAIM_SOURCE_INDEX.md` §1:
- MRB = `results/combined_matched_removal_benchmark_v1/` @`70f978ffc`;
- OAR = `results/combined_output_aware_removal_v1/` @`f7425b15c`;
- CAP = `results/combined_analysis_paper_v1/`;
- COR = `publications/pcrl-mechanism-design-v1/results/` @`c9709a298`.

**Method.** Every item was checked against primary sources only: publisher or proceedings pages and PDFs, arXiv PDFs,
the Crossref record of the DOI, and official code repositories. Two read-only helpers transcribed passages in
parallel. Markers:
- **[R]**: the reviewer re-read the passage in the primary PDF (text extracted with `pdftotext`) or in the Crossref
  record.
- **[V]**: transcribed by a helper from the primary source; not re-read by the reviewer.

Quotes are verbatim except for flattened mathematical symbols. "p." is the PDF page.

## Summary

| Key | Cited at | Verdict on the manuscript's claim | Bibliography |
|---|---|---|---|
| `stadler2024lpl` | l.189 | **Not supported as stated.** The theorem is about every attribute, with the label granted to the adversary. The sentence is also contradicted by the paper's own hard-decision data. The proposed replacement (COR item 2) is supported clause by clause with three qualifications. | correct |
| `tian2025fairnessprivacy` | l.202, 209 | Supported with qualification (a specific two-model attack, image benchmarks) | **wrong venue**: published in IEEE TDSC 22(5):4654–4670 (2025), DOI 10.1109/TDSC.2025.3551157 |
| `wang2019balanced` | l.178, 208 | Supported (labels randomly perturbed to the model's F1; attacker on logits) | correct (pp. 5309–5318) |
| `li2012multilevel` | l.201 | Supported with qualification: linear least-squares reconstruction, correlated Gaussian noise | correct |
| `jia2018attriguard` | l.132, 198 | l.132 supported. **l.198 not supported**: AttriGuard's attacker reads users' public data, not what a model exposes. | **pages missing** (513–529) |
| `song2020overlearning` | l.132, 172, 198 | Auxiliary-data attacker on representations: supported. "recover what the in-training adversary could not": not measured by Song (supported by Elazar & Goldberg). | correct |
| `belrose2023leace` | l.156, 167, 170 | Supported with qualification (linear guardedness for convex losses, population or fitted moments; least squares among affine erasers). "Restricts its claim to linear adversaries": supported. | correct; DOI optional |
| `jovanovic2023fare` | l.157, 191 | Supported with qualification (holds with probability ≥ 1−ε; held-out splits; binary attribute in the main text) | correct |
| `jayaraman2022imputation` | l.182 | Supported with qualification (training-record sensitive-value inference; the adversary knows the distribution) | correct |
| `ganta2008composition` | l.199 | Supported | correct |

---

## a. `stadler2024lpl`: Stadler, Kulynych, Gastpar, Papernot, Troncoso, "The Fundamental Limits of Least-Privilege Learning", ICML 2024

**Sources opened:**
- https://proceedings.mlr.press/v235/stadler24a.html;
- the PMLR PDF (https://raw.githubusercontent.com/mlresearch/v235/main/assets/stadler24a/stadler24a.pdf) [R];
- arXiv 2402.12235 v2 [V].

**Bibliography.** Proceedings of the 41st ICML, PMLR 235:46393–46411. The PDF footer reads "Proceedings of the 41st
International Conference on Machine Learning". The bib entry is correct.

**Standing assumptions** (§2, p.2) [R]:
> "we assume that the spaces X, Y, S are discrete and finite; and that the data domain is non-trivial (|X| > 1 and
> |Y| > 1) and has full support."
>
> "Assumption A (Strictly positive posterior). We say that the posterior distribution, P_{Y|X}, is strictly positive if
> for any x ∈ X, y ∈ Y we have P_{Y|X}(y | x) > 0."

**Fundamental leakage**, the label-only reference (§3.4, p.4) [R]:
> "users must be willing to reveal to the service provider at least the intended result of the computation, i.e., the
> task label Y. As a consequence, they cannot conceal from the service provider any information that can be inferred
> from Y itself. … We call this information the fundamental leakage of the task. For a given attribute S, the
> fundamental leakage equals Pr[S = Ŝ(Y)]."

The experiments (§4, p.6–7) [R]: "The label-only adversary Ŝ(Y) computes the relative frequency counts over D_A".

**Definition 2 (LPP)** (§3.4, p.4) [R]:
> "a feature map f_E(X) = Z satisfies the LPP with parameter γ if for any attribute S which follows the Markov chain
> S − (X, Y) − Z, the attribute inference gain from observing (Z, Y) over the fundamental leakage is bounded:
> Pr[S = Ŝ(Z, Y)] / Pr[S = Ŝ(Y)] ≤ 2^γ".
>
> Equivalently, sup over S: S−(X,Y)−Z of I∞(S; Z | Y) ≤ γ.

**Theorem 2** (p.5) [R]:
> "Theorem 2 (LPP and Utility Trade-Off). Suppose that P_{Y|X} is strictly positive (Assumption A). Then, for
> α ∈ {1, ∞}, the following two properties cannot hold at the same time: I) Z = f_E(X) satisfies the LPP with parameter
> γ II) I_α(Y; Z) > γ".

**Every attribute, not one declared attribute.** The LPP quantifies over *all* attributes (the supremum above), and the
violating attribute is constructed:
- p.5 [R]: "strict positivity of the posterior distribution (Assumption A) prevents the task label Y from being a
  maximally revealing attribute".
- App. A, p.12 [R]: "We demonstrate that there exists an attribute that is maximally revealing, i.e., achieves the
  supremum … and show how to construct this attribute." Definition 4 builds it from a "shattering distribution".
- The authors contrast this with single-attribute work (§1 Related Work, p.2) [R]: "They focus on a single attribute,
  whereas the LPP demands that the representations shared with the service provider should prevent inference of any
  information other than the intended task".
- The single-attribute remark (§4.2 "Takeaways", p.9) [R]: "although it is sometimes possible to restrict information
  leakage for a single attribute, the learned representations do not fulfil the LPP." It is empirical, made under
  attribute censoring.

**The manuscript's current sentence** (l.188–189), "a release useful for a task reveals what the task's label reveals
\citep{stadler2024lpl}": **not supported as stated.**
- The nearest wording in the paper is informal (§1, p.2) [R]: "useful representations must at least leak any sensitive
  information already revealed by the task label itself".
- That premise rests on the provider receiving Y (§3.4). Definition 2 builds it in by giving the adversary (Z, Y).
- Nothing in the paper shows that a release computed from X alone reveals what Y reveals. Under Assumption A, Y is not
  a function of X.
- **This paper's stored data contradict the sentence for decisions.** Hard decisions reveal *less* than the label-only
  reference: OAR `EXPLORATORY.csv` l.40, Adult R(O_hard) − R(LO) = −0.0885 [−0.0975, −0.0798], and l.92, HMDA −0.0162
  [−0.0246, −0.0092] (exploratory 90 %). Full outputs reveal *more*: l.41 and l.93, +0.174 and +0.246; MRB
  `MATCHED_COMPARISONS.csv` l.468 and l.660, +0.177 and +0.236.

**Replacement sentence**, COR `CLOSEST_PRIOR_WORK.md` §1 item 2 at `c9709a298`, clause by clause:

| Clause | Verdict | Note |
|---|---|---|
| "prove that, when the label posterior is strictly positive" | Supported with qualification | The condition is Assumption A, P_{Y\|X}(y\|x) > 0 for all x, y. It holds together with finite discrete spaces and full support (§2). Suggest adding "for finite discrete data with full support". |
| "no useful representation can bound inference beyond the task label for *every* attribute" | Supported with qualification | The result is quantitative. LPP at level γ and utility I_α(Y; Z) > γ cannot both hold. "Useful" therefore means utility above γ (the yes/no reading is exact only at γ = 0, Corollary 2). "Beyond the task label" is the gain of an adversary who sees the representation **and the true label** over one who sees the label alone (Definition 2). |
| "the violating attribute is constructed" | Supported | App. A, p.12; Definition 4. |
| "the authors note that leakage of a single attribute can sometimes be restricted" | Supported | §4.2 "Takeaways", p.9 (COR says "§4"). Empirical, via censoring. |
| "Their label-only adversary ('fundamental leakage') is the reference we report" | Supported with qualification | Both terms are theirs (§3.4; §4). Their quantity is the Bayes label-only **accuracy** Pr[S = Ŝ(Y)], reported as a gain over the majority guess. This paper's reference is a label-only **macro AUC** (R(LO), e.g. OAR `EXPLORATORY.csv` l.6: 0.601). "An analogue of their label-only adversary" would be exact. |
| "as a descriptive comparison rather than a floor: our releases are computed from the inputs alone, so they can reveal less, or more, than the true label does" | The manuscript's own claim | Supported by this paper's data (above). Stadler et al. treat the fundamental leakage as a floor because their provider receives Y. Saying "in our setting" avoids implying they are wrong. |

---

## b. `tian2025fairnessprivacy`: Tian, Zhang, Liu, Zhu, Ding, Zhou

**Sources opened:**
- https://api.crossref.org/works/10.1109/TDSC.2025.3551157 [R];
- https://arxiv.org/abs/2503.06150 and the v2 PDF [R].

**Bibliography (Crossref)** [R]: "Do Fairness Interventions Come at the Cost of Privacy: Evaluations for Binary
Classifiers", *IEEE Transactions on Dependable and Secure Computing*, vol. 22, no. 5, pp. 4654–4670, issued 2025-09,
DOI 10.1109/TDSC.2025.3551157. Authors in order: Huan Tian, Guangsheng Zhang, Bo Liu, Tianqing Zhu, Ming Ding, Wanlei
Zhou.
- The claimed correction (COR `CLOSEST_PRIOR_WORK.md` §1 item 4) is **right in every field**.
- **The bib entry is wrong**: it gives `journal = {arXiv preprint arXiv:2503.06150}`. It should be the TDSC article
  above. The title and authors already match.

**Passages:**
- Abstract [R]: "we uncover a potential threat mechanism that exploits prediction discrepancies between fair and biased
  models, leading to advanced attack results for both MIAs and AIAs."
- §6.1 threat model [R]: "requiring only access to the predictions from both a biased model and a fair model."
- §6.3, black-box FD-AIA [R]: "With results from both biased and fair models, the subgroup information can be
  predicted as: A(x) = f_aa(T_b(x), T_f(x)). (12)"
- Table 5 [V] (accuracy: attack on the biased model / on the fair model / FD-AIA):
  - black-box: CelebA 56.5 / 47.5 / 75.2; UTKFace 66.0 / 57.0 / 77.3; FairFace 51.9 / 49.6 / 53.4;
  - white-box: 83.4 / 82.9 / 87.6; 84.5 / 84.1 / 85.9; 68.3 / 67.8 / 73.6.
- A helper noted that the §7.2 prose mislabels one block as black-box. Cite the table, not the prose.

**Claim** (l.201–202): "combining a biased and a debiased version of one model raises attribute inference".
**Supported with qualification.**
- The increase is relative to standard attribute-inference attacks on either model alone, in every reported cell.
- It is one attack (FD-AIA) that trains an attack model on both models' outputs for the same input.
- The setting is binary classifiers on image benchmarks (CelebA, UTKFace, FairFace), with fair mixup as the main
  intervention.

The Position paragraph (l.208–209) names Tian et al. as closest "on combined releases". That is the authors' judgement
from a bounded search, consistent with CAP `CLOSEST_PRIOR_ANALYSES.md` §4.

---

## c. `wang2019balanced`: Wang, Zhao, Yatskar, Chang, Ordonez, ICCV 2019

**Sources opened:**
- the CVF open-access PDF [R];
- https://api.crossref.org/works/10.1109/ICCV.2019.00541 [R].

**Bibliography (Crossref)** [R]: 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 5309–5318, DOI
10.1109/ICCV.2019.00541. The bib entry is correct.
- The CVF open-access PDF numbers its pages one higher: the §3 page carries footer "5312". Keep the DOI pages.

**Passages** (§3, "Leakage and Amplification") [R]:
> "We define dataset leakage at a performance a by perturbing ground truth labels, with some function r(Y_i, a), such
> that the overall accuracy of the changed labels with respect to the ground truth achieves an accuracy a … This allows
> us to measure the leakage of a model whose performance is a and whose mistakes cannot be attributed to systematic
> bias. Across all experiments, we use F1 as the performance measure".
>
> "Model Leakage: Similar to dataset leakage, we would like to measure the degree a model, M produces predictions,
> Ŷ_i = M(X_i), that leak information about the protected variable g_i."
>
> "Computing Leakage: Model leakage is predicted from pre-activation logits while dataset leakage is predicted from
> binary labels."

**Claim** (l.178–179): "compare leakage from predictions with that implied by the true labels at matched task
performance". **Supported.**
- Nuance: the label reference uses randomly perturbed true labels matched on F1.
- The attackers are supervised: one reads binary labels, the other logits.

The Position-paragraph description (l.208, "utility-qualified output leakage") is fair.

---

## d. `li2012multilevel` and `jia2018attriguard`

### `li2012multilevel`: Li, Chen, Li, Zhang, "Enabling Multilevel Trust in Privacy Preserving Data Mining"

**Sources opened:**
- https://api.crossref.org/works/10.1109/TKDE.2011.124 [R];
- arXiv 1104.0459 v2 PDF [R].

**Bibliography (Crossref)** [R]: IEEE TKDE 24(9):1598–1612, 2012-09, DOI 10.1109/TKDE.2011.124; authors Yaping Li,
Minghua Chen, Qiwei Li, Wei Zhang. **Correct.**

**Passages:**
- Abstract [R]: "a malicious data miner may have access to differently perturbed copies of the same data through
  various means, and may combine these diverse copies to jointly infer additional information about the original data
  … We address this challenge by properly correlating perturbation across copies at different trust levels. … for data
  miners who have access to an arbitrary collection of the perturbed copies, our solution prevent them from jointly
  reconstructing the original data more accurately than the best effort using any individual copy in the collection."
- §3.2 threat model [R]: "we assume adversaries only perform linear estimation attacks, where estimates can only be
  linear functions of the perturbed data Y. It is known that if X follows a jointly Gaussian distribution, then LLSE
  estimation achieves the minimum estimation error among both linear and nonlinear estimation methods. For X with
  general distribution, LLSE estimation has the minimum estimation error among all linear estimation methods."

**Claim** (l.199–201): "recipients of differently perturbed copies of the same data can pool them, and mechanisms
exist under which the pool reconstructs the data no more accurately than the best single copy". **Supported with
qualification.**
- The guarantee is for linear least-squares (LLSE) reconstruction.
- It assumes additive Gaussian noise correlated across copies, against an adversary who knows the means and
  covariances.
- It covers all estimators only when the data are jointly Gaussian.
- Suggest adding "under linear least-squares reconstruction".

### `jia2018attriguard`: Jia and Gong, "AttriGuard: A Practical Defense Against Attribute Inference Attacks via Adversarial Machine Learning"

**Sources opened:** the official USENIX PDF, `usenix.org/system/files/conference/usenixsecurity18/sec18-jia.pdf`
[R]; arXiv 1805.04810 [V].

**Bibliography** [R]:
- Cover page: "This paper is included in the Proceedings of the 27th USENIX Security Symposium. August 15–17, 2018 •
  Baltimore, MD, USA. ISBN 978-1-939133-04-5".
- Page footers run from 513 (first page) to 529 (last page).
- **The bib entry is incomplete:** add `pages = {513--529}` and `publisher = {USENIX Association}`. The USENIX
  presentation page, https://www.usenix.org/conference/usenixsecurity18/presentation/jia-jinyuan, is a better `url`
  than arXiv.

**Passages:**
- Abstract, p.513 [R]: "an attacker leverages a machine learning classifier to infer a target user's private
  attributes (e.g., location, sexual orientation, political view) from its public data (e.g., rating scores, page
  likes)."
- §2.1, p.515 [R]: "The classifier can be learnt on a training dataset consisting of both public data and attribute
  values of users who also make their attributes public."
- §3.2, p.516 [R]: "some users do not treat the attribute s as private, so they publicly disclose it. Via collecting
  data from such users, the attacker can learn the machine learning classifier."

**Claims:**
- l.130–132, "attackers train on labelled examples or on people who disclosed the attribute": **supported** (Song for
  the first part, AttriGuard for the second).
- l.196–198, "the auxiliary-data attacker of attribute inference, which learns from labelled examples to read a
  sensitive attribute from what a model exposes \citep{song2020overlearning,jia2018attriguard}": **not supported for
  AttriGuard.**
  - Its attacker reads a user's *public data* (ratings, page likes); no model output or representation is involved.
  - Options: cite only Song for "what a model exposes", or write "from what a model or a person's public data
    exposes".

---

## e. `song2020overlearning`: Song and Shmatikov, "Overlearning Reveals Sensitive Attributes", ICLR 2020

**Sources opened:** arXiv 1905.11742 v3 PDF, whose header reads "Published as a conference paper at ICLR 2020" [R].
The OpenReview forum (SJeNz04tDS) could not be opened (browser challenge).

**Bibliography:** ICLR 2020, no pages or DOI. The bib entry is correct.

**Passages:**
- §3.1 [R]: "Suppose an adversary can observe the representation z⋆ of a trained model M on input x⋆ at inference
  time but cannot observe x⋆ directly. … We assume that the adversary has an auxiliary set D_aux of labeled (x, s)
  pairs and black-box oracle E to compute the corresponding E(x). … it need not be drawn from the same dataset as x⋆.
  The adversary uses supervised learning on the (E(x), s) pairs to train an attack model M_attack."
- §4.2 [R]: "When representations are censored with adversarial training, accuracy drops for both the main and
  inference tasks. Accuracy of inference is much higher than in (Xie et al., 2017). The latter uses logistic
  regression, which is weaker than the training-time censoring-adversary network, whereas we use the same
  architecture for both the training-time and post-hoc adversaries."
- Table 2 [R], sensitive-attribute accuracy, RAND / BASE / ADV: UTKFace 42.52 / 62.18 / 53.28; FaceScrub 1.42 / 33.65
  / 30.23; Places365 1.37 / 31.03 / 12.56.

**Does an adversary with labelled auxiliary data read the sensitive attribute from a representation?** **Yes** (§3.1).
- Note: Song's auxiliary data "need not be drawn from the same dataset". This paper's same-population grant is a
  stronger assumption.

**Claims:**
- l.132 and l.198: **supported** for Song. The exposure is the model's internal representation.
- l.170–172, "post-hoc attackers recover what the in-training adversary could not
  \citep{elazar2018adversarial,song2020overlearning}": **supported with qualification.**
  - Song show post-hoc inference well above chance from adversarially censored representations (Table 2), raised
    further by de-censoring (Table 3).
  - They do not report the in-training adversary's own accuracy, so "could not" is shown by Elazar & Goldberg (Table 3
    per CAP `CLOSEST_PRIOR_ANALYSES.md` §2), not by Song.
  - Safer: "post-hoc attackers still recover the attribute after adversarial training".

---

## f. `belrose2023leace`: Belrose et al., "LEACE: Perfect linear concept erasure in closed form", NeurIPS 2023

**Sources opened:**
- the NeurIPS abstract page and Paper-Conference PDF [R];
- arXiv 2306.03819 [V];
- the official repository https://github.com/EleutherAI/concept-erasure (README and `concept_erasure/leace.py`) [V].

**Bibliography.** Advances in NeurIPS 36 (2023). The NeurIPS page shows DOI 10.52202/075280-2884 and no page range
[R]. The bib entry is correct; the DOI could be added.

**Passages:**
- Definition 2.3, p.2 [R]: "If X (V, L)-guards Z, where L is the class of nonnegative loss functions which are convex
  in their first argument, and V is the class of linear predictors η(x) = b + Wx, we say that X linearly guards Z."
- §3, p.2 [R], the equivalent conditions: "1. The data X linearly guards the labels Z. … 2. For all convex losses L, the
  trivially attainable loss is optimal on (X, Z). … 3. The class-conditional mean vectors E[X | Z = i] are equal to the
  unconditional mean E[X]. 4. Every component of X has zero covariance with every component of Z. 5. Every linear
  classifier evaluated on X exhibits statistical parity w.r.t. Z."
- Theorem 4.2, p.5 [R]: "Let X and Z be centered random vectors … each of finite second moment. Let M ∈ R^{d×d} be a
  p.s.d. matrix defining a (possibly degenerate) inner product … Then the objective argmin_P E‖PX − X‖²_M subject to
  Cov(PX, Z) = 0 has the following solution: P∗ = I − W⁺ P_{WΣXZ} W". Theorem 4.3 extends it to affine erasers [V].
- Footnote 5, p.8 [R]: "The softmax probabilities of a multiclass logistic regression classifier can leak the removed
  information if another classifier is stacked on top of it [31], though this setup is not linear."
- §7, p.10 [R]: "In this work, we focused exclusively on linear concept erasure due to its simplicity and
  tractability. … We conjecture that it is intractable to nondestructively edit X so as to prevent a general nonlinear
  adversary from recovering Z, unless the data generating process for X is known in detail."
- Implementation: `LeaceFitter(..., affine=True, constrain_cov_trace=True, shrinkage=True, svd_tol=0.01)`. Singular
  values at or below `svd_tol` are truncated [V]. This repository records the same defaults for v0.2.4 (MRB
  `PROTOCOL.md` l.110–113) [R].

**Claims:**
- l.167–168, LEACE "guarantees it in closed form with minimal change to the representation": **supported with
  qualification.**
  - The guarantee is linear guardedness: no linear or affine predictor beats the best constant predictor, for any
    convex loss; equivalently, zero cross-covariance.
  - It is stated for the moments of the distribution the eraser is fitted on. The paper gives no finite-sample or
    held-out guarantee.
  - "Minimal change" means the least expected squared change in any p.s.d. inner-product norm, among linear or affine
    erasers satisfying the constraint.
  - The official implementation truncates at `svd_tol`. That is why this paper's native check is an "implementation
    bound" (l.238). The matched benchmark found 6 of 12 Tier-1 maps truncated within tolerance (MRB
    `RESEARCH_DECISION.md` l.66–69).
- l.170, "LEACE restricts its claim to linear adversaries": **supported** (§7). Footnote 5 also anticipates the
  score-level leakage this paper measures (l.355–360).

---

## g. `jovanovic2023fare`: Jovanović, Balunović, Dimitrov, Vechev, "FARE: Provably Fair Representation Learning with Practical Certificates", ICML 2023

**Sources opened:**
- https://proceedings.mlr.press/v202/jovanovic23a.html and its PDF [R];
- arXiv 2210.07213 and https://github.com/eth-sri/fare [V].

**Bibliography.** Proceedings of the 40th ICML, PMLR 202:15401–15420. The bib entry is correct.

**Passages:**
- Definition 4.1, p.4 [R]: "a practical DP distance certificate is a value T*(n, D) ∈ R such that sup_{g∈G}
  ∆DP_{Z0,Z1}(g) ≤ T*(n, D) holds with probability 1 − ε (over the data sampling), where G is the set of all
  downstream classifiers."
- §5, p.6 [R]: "We assume a standard setting, where a set D of datapoints … is split into a training set D_train, used
  to train f, validation set D_val, held-out for the upper-bounding procedure (and not used in training of f in any
  capacity), and a test set D_test"; the bound "holds with confidence at least 1 − ε, where ε is the hyperparameter of
  the procedure (we use ε = 0.05)".
- The lemmas [V]: Lemma 5.1 is a Clopper–Pearson bound on base rates; Lemma 5.2 gives per-cell Clopper–Pearson bounds
  under a union bound; Lemma 5.3 is a Hoeffding bound on D_test.
- Restricted encoders map each x "to one of k possible values (cells)" (§5) [V].
- §7, p.7 [R]: "a tradeoff of restricted encoders—FARE computes a practical certificate, but loses some information,
  limiting the predictive power of classifiers."
- App. C, p.14 [R]: "as k-means encoders are fairness-unaware, both empirical results and upper bounds are unfavorable
  and greatly outperformed by FARE."
- The main text treats a binary sensitive attribute; App. D generalises it [V].

**Claim** (l.191–193): FARE "restricts the encoder to a finite partition and certifies the demographic-parity
distance of any classifier that reads only the cell; its authors note that restricted encoders lose information and
compare fairness-unaware k-means encoders". **Supported with qualification.**
- The certificate is a finite-sample upper bound holding with probability at least 1 − ε (ε = 0.05 in the paper and
  in this study, OAR `PROTOCOL.md` l.177).
- It is computed on held-out splits.
- "Any classifier" means any downstream classifier of the representation, i.e. of the cell.

This paper's own use (D_val/D_test 50/50 split of the certification rows; all-pairs primary and race {0,1,2}
secondary) is consistent with the method (OAR `PROTOCOL.md` l.176–182).

---

## h. `jayaraman2022imputation`: Jayaraman and Evans, "Are Attribute Inference Attacks Just Imputation?", CCS 2022

**Sources opened:**
- https://api.crossref.org/works/10.1145/3548606.3560663 [R];
- arXiv 2209.01292 v1 PDF [R].

**Bibliography (Crossref)** [R]: Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security,
pp. 1569–1582, DOI 10.1145/3548606.3560663. The bib entry is correct.

**Passages:**
- Abstract [R]: "Our main conclusions are: (1) previous attribute inference methods do not reveal more about the
  training data from the model than can be inferred by an adversary without access to the trained model, but with the
  same knowledge of the underlying distribution as needed to train the attribute inference attack; (2) black-box
  attribute inference attacks rarely learn anything that cannot be learned without the model; but (3) white-box
  attacks, which we introduce and evaluate in the paper, can reliably identify some records with the sensitive value
  attribute that would not be predicted without having access to the model."
- §2.3 [V]: "this silly baseline already outperforms the three of the attribute inference attacks (Yeom, CAI, and
  CSMIA) in three of four settings". The baseline is "Predict Most Common".
- Table 1 [V]: with little prior data (500 or 50 records), the best black-box attack did beat imputation.

**Claim** (l.182): "Many black-box attribute-inference results do not exceed imputation, so we report constant and
label-only references". **Supported with qualification.**
- The scope is inference of sensitive values of training records.
- It holds when the adversary has distributional knowledge comparable to what the attack needs.
- The paper's own wording ("rarely") is stronger than "many".
- The constant reference is directly motivated by the "Predict Most Common" baseline.

---

## i. `ganta2008composition`: Ganta, Kasiviswanathan, Smith, "Composition Attacks and Auxiliary Information in Data Privacy", KDD 2008

**Sources opened:**
- https://api.crossref.org/works/10.1145/1401890.1401926 [R];
- arXiv 0803.0032 v2 PDF [R].

**Bibliography (Crossref)** [R]: Proceedings of the 14th ACM SIGKDD, pp. 265–273, DOI 10.1145/1401890.1401926. The bib
entry is correct (Crossref gives the title in sentence case).

**Passage** (abstract) [R]:
> "we investigate the effectiveness of current anonymization schemes in preserving privacy when multiple organizations
> independently release anonymized data about overlapping populations. 1. We investigate composition attacks, in which
> an adversary uses independent anonymized releases to breach privacy. … even a simple instance of a composition attack
> can breach privacy in practice, for a large class of currently proposed techniques. The class includes k-anonymity
> and several …"

**Claim** (l.199): "Combining independent releases is a classical route to breaches". **Supported.**
- "Independent" is the authors' term: uncoordinated releases by different organizations about overlapping
  populations.
- The breached schemes are partition-based (k-anonymity and its variants). The paper also shows that differential
  privacy resists such attacks [V].

---

## Other bibliography notes for these keys

- `tian2025fairnessprivacy`: replace the arXiv `journal` with the TDSC article (item b).
- `jia2018attriguard`: add pages 513–529 and the publisher (item d).
- `belrose2023leace`: optionally add DOI 10.52202/075280-2884.
- `wang2019balanced`: keep pp. 5309–5318 (IEEE/DOI). The CVF PDF's pagination differs by one.
- The other keys checked here (`stadler2024lpl`, `li2012multilevel`, `song2020overlearning`, `jovanovic2023fare`,
  `jayaraman2022imputation`, `ganta2008composition`) are correct.
