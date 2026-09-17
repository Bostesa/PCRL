# Prior work and novelty audit: residual spectral auxiliary channel

## Scope

- **Date:** 2026-09-17.
- **Mechanism audited:** the residual spectral auxiliary channel described in
  [MATHEMATICS.md](../redesign_20260910_acs_residual_spectral_v1/MATHEMATICS.md).
  It uses fixed service predictions H_A and H_B, a 16-dimensional channel Z that only recipient A receives, a PCA32 teacher and random Fourier features. Residualization uses a degree-2 polynomial basis of H_A. The protection moments use cross-fitted conditional distributions. The arms are S0, M025, M1, L025, L1, C025, C1 and L2.
- **Sources checked:**
  - Abstract or proceedings pages for every required reference.
  - Extracted full text for the two Sadeghi–Boddeti papers, KCI, RCIT/RCoT, K-TOpt and U-FaTE.
  - Eight targeted searches for related work on side information, conditional fair representations, multiple recipients or collusion, and task-agnostic private representations. Four further bibliographic lookups confirmed venues.
  - Repository context: `docs/PCRL_PRIOR_WORK.md`, `docs/ACCURACY_CERTIFICATE_RETIREMENT.md` and the MATHEMATICS.md linked above. The LEACE and SPLINCE comparison is already in `docs/PCRL_PRIOR_WORK.md` and is not repeated here.
- **Limits:**
  - A bounded search cannot show that something is new. Nothing below claims this mechanism is "first".
  - "Not found" means only that these searches did not surface it.
  - Supplementary material was not read unless stated.
  - A paper's claims are reported only where its abstract or extracted text was read. Anything else is marked *unverified*.
  - No code was run and no result was changed.

## Headline finding

**Close prior match (verified):** the closed-form kernel line of Sadeghi, Dehdashtian and Boddeti.

- **SARL (ICCV 2019):** a closed-form spectral solution for adversarial representation learning with linear or kernel players.
- **K-TOpt (TMLR 2022):** a closed-form kernel solution with random Fourier features.
- **U-FaTE (CVPR 2024):** extends these to a *conditional* dependence penalty.

All three solve for the top-r eigenvectors of a matrix of the form (utility dependence) − λ·(sensitive dependence). U-FaTE's penalty is conditioned on the label Y, and it is evaluated on FolkTables, which is ACS data.

Our construction differs from these papers in three ways:

1. **What we condition on.** We condition on *continuous released predictions* H, using cross-fitted logistic nuisance models and basis interactions. U-FaTE instead stratifies on a discrete label.
2. **Utility.** Our utility is least-squares reconstruction of a residualized teacher, not dependence on a label.
3. **Recipients.** Penalties are specific to each recipient and to the coalition, and only one recipient receives a channel.

These are adaptations of known components. None of them is a new optimization principle. The following are not our contributions:

- **The spectral trace step** is Ky Fan's maximum principle (Ky Fan 1949), applied exactly as SARL and K-TOpt apply it.
- **Our marginal arms (M025, M1)** are, up to parametrization, SARL's Linear-ARL. It is applied to whitened random-Fourier features with a continuous vector target.
- **Our conditional moments** are a finite subset of the partial-association characterization of conditional independence that KCI uses (attributed there to Daudin, 1980). RCoT approximates the same characterization with random features.

## Reference-by-reference relation

### Spectral and closed-form adversarial representation learning

**Sadeghi, Yu & Boddeti (2019), *On the Global Optima of Kernelized Adversarial Representation Learning*, ICCV 2019 (arXiv:1910.07423)**

- **What it establishes (verified in the full text).** The target and adversary are linear least-squares regressors, and attributes are vectors, so one-hot classes and regression are both covered. The Lagrangian objective reduces to tr(G'BG) over orthonormal G, where B is the whitened form of λ·C_sx'C_sx − (1−λ)·C_yx'C_yx (Eq. 12). Theorem 3 gives the global optimum: the eigenvectors of the *negative* eigenvalues of B, capped at r. The paper extends this to kernels and describes "imparting" invariance to a frozen pre-trained representation.
- **Assumptions.** Linear or kernel encoder, squared loss for both players, and a full-rank covariance. The guarantee holds only against linear or kernel least-squares adversaries.
- **Relation to our method.**
  - With a constant basis and a constant prior, our G_j is C_VS and P_M is C_VS·C_SV. So M025 and M1 are this construction on whitened V with target R, apart from trace normalization and our λ convention.
  - One real difference is dimension. We always keep r = 16 eigenvectors, including ones with negative eigenvalues. SARL's Theorem 3 keeps only the negative-eigenvalue directions of its minimization form, which are the directions where utility beats penalty. Our fixed r can therefore include directions whose penalty exceeds their utility. This is a design choice made to match baseline dimension, not an optimum.

**Sadeghi, Wang & Boddeti (2021), *Adversarial Representation Learning with Closed-Form Solvers* (OptNet-ARL), ECML PKDD 2021 (arXiv:2109.05535)**

- **What it establishes (verified).** The target and adversary are kernel ridge regressors solved in closed form, while the encoder is trained by gradient descent on the resulting one-shot objective. The paper states that the method generalizes to multiple targets and sensitive attributes with multiple λ values. Theorem 4 says that, for a free embedding with linear players, the optimal dimension is the number of negative eigenvalues of λS̃ᵀS̃ − (1−λ)ỸᵀỸ.
- **Assumptions.** Ridge best responses, with dependence measured through the adversary's regression loss.
- **Relation to our method.** It shows that multiple sensitive attributes combined through weighted penalties are already standard. It supports our use of closed-form best responses and warns that a fixed r = 16 may be above the optimal dimension. It does not model conditioning or recipients.

**Sadeghi, Dehdashtian & Boddeti (2022), *On Characterizing the Trade-off in Invariant Representation Learning* (K-TOpt), TMLR (arXiv:2109.03386)**

- **What it establishes (verified).**
  - Dependence is measured through covariance operators (HSIC-like).
  - Theorem 4 gives the encoder as the top-r eigenvectors of a generalized eigenproblem.
  - Corollary 4.1 ties the optimal dimension to the number of non-negative eigenvalues.
  - The implementation uses random Fourier features and representations with uncorrelated coordinates.
- **Assumptions.** Encoders in an RKHS. The abstract states an approximate-normality assumption on low-dimensional projections. The dependence is *unconditional*; U-FaTE says so explicitly.
- **Relation to our method.** The combination of random Fourier features, whitening or decorrelation, and a spectral utility-minus-dependence solution is already published here.

**Dehdashtian, Sadeghi & Boddeti (2024), *Utility-Fairness Trade-Offs and How to Find Them* (U-FaTE), CVPR 2024 (arXiv:2404.09454)**

- **What it establishes (verified in the full text).**
  - For separation criteria (equal opportunity and equalized odds), it defines a conditional dependence Dep(f(X), S | Y = y). This is computed from cross-covariances of encoder coordinates with a basis of the sensitive attribute's RKHS, *within strata* X_c ~ P(X | Y = y) (Eqs. 3–4).
  - Theorem 1 gives the closed-form top-r generalized eigenvectors of (1−λ)·utility − λ·conditional dependence.
  - Experiments include FolkTables for Washington State, with employment as the target and age as the sensitive attribute.
- **Assumptions.** The main text defines this conditional measure "when Y is not a continuous label". The supplementary material, which the paper says holds the per-criterion solutions, was not read.
- **Relation to our method.** This is **the closest match found**: a spectral solution with a conditional kernel-style dependence penalty, evaluated on ACS. Our differences are:
  - We condition on continuous fixed predictions, not on a label.
  - We estimate the conditional mean with cross-fitted logistic models and interact it with a polynomial basis in H. U-FaTE stratifies instead.
  - Our utility is teacher reconstruction.
  - Our penalties are specific to recipients and to the coalition.

  Whether any of these changes helps is an empirical question.

### Conditional-independence moments

**Zhang, Peters, Janzing & Schölkopf (2011), *Kernel-based Conditional Independence Test and Application in Causal Discovery* (KCI), UAI 2011, pp. 804–813 (arXiv:1202.3775)**

- **What it establishes (verified).** Lemma 2 characterizes X ⊥ Y | Z as uncorrelatedness of "residual" functions of the form f(X, Z) − E[f | Z] with functions of the other variable. It builds a kernel test statistic from this, with an asymptotic null distribution, and estimates the regressions with kernel ridge or Gaussian-process regression.
- **Assumptions.** Characteristic kernels, consistent regression estimates, and asymptotic calibration.
- **Relation to our method.** For categorical S, the residual function (1{S = c} − m_c(H)) paired with the test function V·b_k(H) is one case of Lemma 2(iii)/(v). Our G_jk is a *finite, fitted* collection of these moments. We use them as an optimization penalty, not a test, and we inherit none of KCI's calibration.

**Strobl, Zhang & Visweswaran (2019), *Approximate Kernel-Based Conditional Independence Tests for Fast Non-Parametric Causal Discovery* (RCIT/RCoT), Journal of Causal Inference 7(1) (arXiv:1702.03877)**

- **What it establishes (verified).** RCIT and RCoT approximate KCIT with random Fourier features. The RCoT statistic is n·‖Σ̂_AB·C‖²_F, the squared Frobenius norm of the cross-covariance of residuals after linear ridge regression on random features of C (Eq. 26).
- **Assumptions.** Enough random features and an approximated null distribution.
- **Relation to our method.** P_j is structurally an RCoT-like partial cross-covariance, projected through W. The differences are:
  - We use logistic rather than ridge nuisance models for S.
  - Basis interactions replace residualization of the random-feature side.
  - In the coalition role, V is residualized on q_A only, while the nuisance models use H_AB. The two residualizations are therefore not symmetric.

**Cross-fitting (Chernozhukov et al., *Double/Debiased Machine Learning for Treatment and Causal Parameters*, arXiv:1608.00060)**

- **What it establishes (verified).** The abstract names K-fold sample splitting "cross-fitting", used to avoid overfitting bias in the nuisance models.
- **Relation to our method.** Our household-grouped 3-fold nuisance models follow this practice. We do not claim DML's orthogonality or rate guarantees for our moments. The journal version was not checked in this audit.

**Ky Fan (1949), *On a Theorem of Weyl Concerning Eigenvalues of Linear Transformations I*, PNAS 35(11):652–655**

- **Bibliographic record only (PNAS/PubMed).** The maximum principle is derived directly in MATHEMATICS.md and cited as standard. SARL and K-TOpt apply the same trace-optimization result, citing Kokiopoulou et al.

### Information-theoretic privacy

**Makhdoumi, Salamatian, Fawaz & Médard (2014), *From the Information Bottleneck to the Privacy Funnel*, IEEE ITW 2014 (arXiv:1402.1774)**

- **What it establishes (verified abstract).** Under log-loss, privacy leakage and utility become mutual information. Inference threat under any bounded cost is upper-bounded by a function of I(S; released). The resulting non-convex privacy funnel is attacked with a greedy, locally optimal algorithm, evaluated on US census data.
- **Assumptions.** Known discrete joint distribution and a randomized mapping.
- **Relation to our method.** It justifies reading log-loss attackers as information-related. Our mechanism is deterministic and finite-moment, and it does not optimize I(S; Z | H).

**Rassouli & Gündüz (2021), *On Perfect Privacy*, IEEE JSAIT 2(1):177–191 (arXiv:1712.08500)**

- **What it establishes (verified abstract).** It characterizes the largest possible I(Y; U) under exact independence X ⊥ U, for the output-perturbation and full-data-observation models. For jointly Gaussian (X, Y), perfect privacy is impossible under output perturbation but possible under full observation.
- **Relation to our method.** It gives the kind of population feasibility statement our finite moments cannot give. Which of its two models our channel matches depends on whether the channel input determines S. We did not analyze this.

**Sankar, Rajagopalan & Poor (2013), *Utility-Privacy Tradeoff in Databases: An Information-theoretic Approach*, IEEE TIFS (arXiv:1102.3751; DOI 10.1109/TIFS.2013.2253320)**

- **What it establishes (verified abstract).** Utility–privacy tradeoff regions with *multiple legitimate information consumers* and modelled prior knowledge (side information) at the user or source.
- **Relation to our method.** Side information held by recipients, and more than one consumer, already appear in information-theoretic privacy. The abstract does not describe coalition-specific policies; the body was not read.

**Liao, Sankar, Kosut & Calmon (2019), *Robustness of Maximal α-Leakage to Side Information*, ISIT 2019 (arXiv:1901.07105)**

- **What it establishes (verified abstract).** It defines conditional maximal α-leakage. Side information that is conditionally independent of the public data given the private data cannot increase leakage.
- **Relation to our method.** Our H and Z are both functions of the same covariates, so that condition generally does not hold. Robustness to H cannot be assumed from this result.

**Stadler, Kulynych, Gastpar, Papernot & Troncoso (2024), *The Fundamental Limits of Least-Privilege Learning*, ICML 2024, PMLR 235:46393–46411**

- **What it establishes (verified abstract; details in `docs/PCRL_PRIOR_WORK.md`).** A representation cannot have high utility for its intended task while also preventing inference of unrelated attributes. The limit holds regardless of how the representation is learned.
- **Relation to our method.** Under its assumptions (finite spaces, full support), any information in Z beyond H_A is itself an inferable attribute. Our method can therefore suppress only *declared* attributes and cannot claim least privilege. Its formalization conditions on the task label, not on released predictions, so the mapping to our setting is not exact.

### Empirical and measurement cautions

**Elazar & Goldberg (2018), *Adversarial Removal of Demographic Attributes from Text Data*, EMNLP 2018, pp. 11–21 (arXiv:1808.06640)**

- **What it establishes (verified).** A training adversary at chance level does not stop a post-hoc classifier from recovering demographics.
- **Relation to our method.** This justifies our independent post-hoc attackers. Our development finding that fresh attackers recover more SEX/race from spectral channels than from neural J is exactly the kind of result such audits exist to expose.

**McAllester & Stratos (2020), *Formal Limitations on the Measurement of Mutual Information*, AISTATS 2020, PMLR 108:875–884 (arXiv:1811.04251)**

- **What it establishes (verified).** Any distribution-free, high-confidence *lower* bound on mutual information from N samples is O(ln N).
- **Relation to our method.** Attacker log-loss differences should not be reported as certified information quantities. The theorem concerns lower bounds. A "low leakage" claim is an upper-bound-type claim, which a finite attack set does not provide at all.

**Belrose et al. (2023), *LEACE: Perfect Linear Concept Erasure in Closed Form*, NeurIPS 2023 (arXiv:2306.03819)**

- **What it establishes (verified abstract).** A closed-form eraser that prevents all linear classifiers from detecting a concept, with minimal change to the representation.
- **Relation to our method.** This is the unconditional linear baseline. Exact zero cross-covariance is achievable in closed form, so our M arms are a soft-penalty relative of it. The full comparison is in `docs/PCRL_PRIOR_WORK.md`.

**Holstege, Ravfogel & Wouters (2025), *Preserving Task-Relevant Information Under Linear Concept Removal* (SPLINCE), NeurIPS 2025 (arXiv:2506.10703)**

- **What it establishes (verified abstract).** An oblique projection that removes linear predictability of the concept while exactly preserving covariance with a target, with minimal distortion.
- **Relation to our method.** Task-preserving erasure is prior work. Our utility term, the covariance with the teacher residual, plays the role of SPLINCE's preserved target, but we trade it off softly instead of preserving it exactly.

### Found by targeted search

**Zhao, Coston, Adel & Gordon, *Conditional Learning of Fair Representations* (arXiv:1910.07162)**

- **What it establishes (verified abstract).** Conditional alignment of representations given Y, plus balanced error rate, to target accuracy parity and equalized odds.
- **Venue.** ICLR 2020 is *unverified*; the OpenReview page could not be loaded.
- **Relation to our method.** Conditioning representation constraints on Y is established. Here the conditioning is on the label, not on released predictions.

**Hwa, Zhao, Lahiri, Masood, Salimi & Adeli (2024), *Enforcing Conditional Independence for Fair Representation Learning and Causal Image Generation*, CVPR 2024 Workshops (arXiv:2404.13798)**

- **What it establishes (verified abstract).** Enforces conditional independence of high-dimensional latents with respect to a protected attribute under equalized odds, using Jensen–Shannon divergence terms.
- **Relation to our method.** More prior work on learned conditional independence, again conditioned on labels.

**Arevalo, Noorbakhsh, Dong, Hong & Wang (2024), *Task-Agnostic Privacy-Preserving Representation Learning for Federated Learning Against Attribute Inference Attacks* (TAPPFL), AAAI 2024 (arXiv:2312.06989)**

- **What it establishes (verified abstract).** Two mutual-information goals: minimal information about the private attribute and maximal information about the data. It includes worst-case privacy guarantees.
- **Relation to our method.** Task-agnostic, reconstruction-style utility for private representations is prior work. The repository matrix's TIPRDC entry, not re-verified here, is similar.

**Recipient-specific policies with a coalition constraint in representation learning were not found.** The multi-recipient search surfaced:

- collusion in cryptographic aggregation, federated learning and individual DP;
- colluding eavesdroppers in wireless secrecy;
- the multiple-consumer model of Sankar et al.

None of these were read in enough depth to rule out a match. The repository's existing note on maximal-leakage composition (Issa, Wagner & Kamath) was not re-verified in this audit.

## Contribution table

| Existing ingredient | What our implementation changes | Theorem or assumptions actually available | Experiment that could distinguish it | Current evidence |
|---|---|---|---|---|
| Spectral closed-form ARL (SARL, K-TOpt, U-FaTE) | Continuous teacher-residual target; fixed r = 16, including negative-eigenvalue directions; trace-normalized U and P | Ky Fan: exact optimum of the fixed empirical trace objective only | Compare fixed r = 16 with r chosen by eigenvalue sign (SARL Thm 3 / K-TOpt Cor 4.1) at matched utility | development: see DEVELOPMENT_RESULTS.md; transport: pending |
| Marginal linear dependence penalty (SARL Linear-ARL, LEACE) | M arms: soft penalty on whitened random-feature coordinates | Zero C_ZS removes affine least-squares prediction only (retired-certificate note) | M1 versus LEACE or SPLINCE applied to V at matched dimension and independent attackers | development: see DEVELOPMENT_RESULTS.md; transport: pending |
| Conditional dependence penalty (U-FaTE; KCI/RCoT characterization) | Conditioning on continuous released H via a polynomial basis and cross-fitted logistic nuisance models, instead of stratifying on a label | Vanishing of a finite, fitted moment set only; no CI theorem; nuisance misspecification can mimic or hide leakage | L arms versus a U-FaTE-style penalty stratified on binned H; nuisance-misspecification sensitivity | development: see DEVELOPMENT_RESULTS.md; transport: pending |
| Residualizing features on side information (KCI/RCoT residual functions) | V residualized on q_A(H_A) only; utility measured beyond H_A | Least-squares orthogonality to span(q_A) on training rows | Channel utility beyond H versus a no-residualization control; basis-degree sensitivity | development: see DEVELOPMENT_RESULTS.md; transport: pending |
| Reconstruction or information-retention utility (TAPPFL, K-TOpt supervised kernel PCA) | Utility is explained reconstruction of the frozen PCA32 teacher residual | Exact identity: trace(W'U_raw W) = reduction in least-squares error | Held-out *task* utility (residence) versus reconstruction ranking across arms | development: see DEVELOPMENT_RESULTS.md; transport: pending |
| Multiple sensitive attributes and weighted penalties (OptNet-ARL) | Separate local (A) and coalition (AB) penalty families; only A gets Z | None beyond the fixed objective; coalition moments use H_AB nuisance models while V is residualized on H_A only | C1 versus L1 and the equal-total-penalty L2 on coalition SEX/race attacks | development: C1 did not robustly beat both L1 and L2 (see DEVELOPMENT_RESULTS.md); transport: pending |
| Adversarially trained neural channels (ARL; Elazar–Goldberg audit practice) | Replaces iterative training with one eigen-solve | None against nonlinear attackers | Fresh attacker suite on spectral arms versus neural J at matched utility | development: attackers recover more SEX/race from spectral arms than from J (see DEVELOPMENT_RESULTS.md); transport: pending |
| Randomized withholding (privacy-funnel log-loss view) | Visible Bernoulli branch releases Z with probability p | I(S; B, Z_B \| H) = p·I(S; Z \| H) for an independent, persistent branch (MATHEMATICS.md) | Spectral arm versus J withheld at matched utility | development: see DEVELOPMENT_RESULTS.md; transport: pending |

## Does prior work already contain the combination?

**Partially, and closely.**

- **Already published:**
  - U-FaTE has a closed-form spectral solution with a *conditional* kernel-style dependence penalty and a utility term, evaluated on ACS data.
  - K-TOpt adds random Fourier features and decorrelated coordinates.
  - OptNet-ARL has multiple sensitive attributes with weighted penalties.
  - SARL frames this as adding invariance to a frozen representation.
  - KCI and RCoT supply the residual-moment characterization.
- **Not found in this bounded search:**
  - Conditioning on *continuous, fixed released predictions* using cross-fitted nuisance models.
  - A *teacher-reconstruction* utility measured beyond those predictions.
  - *Recipient-specific plus coalition* penalty families where only one recipient receives a channel.

  This is an absence of evidence, not a novelty finding.

**Assessment.** The method contribution is best described as an engineering adaptation of known spectral and conditional-moment tools to a fixed-service, multi-recipient interface. It is a thin methodological delta.

The application and evaluation could still carry a paper: the fixed exact-output interface, coalition-aware audits with independent attackers, matched neural and withholding baselines, and locked temporal transport. That requires the evidence to be framed honestly. Current development results do not support "the coalition penalty helps": C1 did not robustly beat L1 and L2, and spectral arms leak more SEX/race than J to fresh attackers.

A defensible framing is an evaluation study, possibly a negative one, of whether closed-form conditional penalties transfer to this interface. The framing should not present a new method. U-FaTE, K-TOpt and SARL must be cited as direct antecedents, and ideally a U-FaTE-style label- or bin-stratified penalty should be included as a baseline.

## Required distinctions

1. **Fixed H keeps outputs exact, not accurate.** H_A and H_B are released unchanged, but their accuracy under temporal or distribution shift is not preserved or guaranteed.
2. **The eigensolution optimizes a finite surrogate.** It is globally optimal for a fixed empirical matrix objective, given the chosen features, nuisance models, basis, bandwidth and rank. It does not minimize I(S; Z | H) or any conditional mutual information.
3. **Residualization only zeroes finite fitted moments.** Residual moments vanish on the training sample for a finite basis. That does not imply conditional or marginal independence (see the S·H and |Z| counterexamples in MATHEMATICS.md), and it does not imply out-of-sample vanishing.
4. **Reconstruction is not general usefulness.** Explained least-squares reconstruction of the PCA32 teacher residual does not guarantee usefulness for any particular downstream task, including held-out ones.
5. **Coalition protection and individual protection are different properties.** Suppressing SEX/race for AB given H_AB is different from suppressing them for A alone given H_A. Neither implies the other, and some attributes are authorized for one party but not the other.
6. **A development mean is not generalization.** Averages over three development seeds on 2018 data do not establish performance on 2017 transport data or elsewhere.
7. **Only A gets a channel.** B receives H_B only. Any B-side or coalition risk from Z arises only through A's participation in a coalition.
8. **Subtracting a fixed anchor loss does not change gradients.** Reporting attack loss relative to an H-only anchor changes how results are presented, not what is optimized. It has zero derivative with respect to mapper parameters.
9. **Prior guarantees are stated for their own objectives.** SARL, K-TOpt and U-FaTE optimality statements, KCI and RCoT calibration, and privacy-funnel or perfect-privacy results each hold under their own assumptions. None transfers to our finite ACS attacks.

## Unverified or unchecked items

- The ICLR 2020 venue for Zhao et al. (the OpenReview page could not be loaded).
- The journal version of the double-ML paper.
- The contents of the U-FaTE supplementary material, including whether it handles continuous conditioning.
- The bodies of Sankar et al., Liao et al. and TAPPFL; only their abstracts were read.
- Daudin (1980), which was seen only as KCI's citation.
- Random Fourier features (Rahimi and Recht), cited as a standard ingredient but not checked.
- The repository's Issa–Wagner–Kamath and TIPRDC entries.
- The collusion papers surfaced by search.

## Bibliography

```bibtex
@inproceedings{sadeghi2019sarl,
  title     = {On the Global Optima of Kernelized Adversarial Representation Learning},
  author    = {Sadeghi, Bashir and Yu, Runyi and Boddeti, Vishnu Naresh},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2019},
  note      = {arXiv:1910.07423},
  url       = {https://arxiv.org/abs/1910.07423}
}

@inproceedings{sadeghi2021optnetarl,
  title     = {Adversarial Representation Learning with Closed-Form Solvers},
  author    = {Sadeghi, Bashir and Wang, Lan and Boddeti, Vishnu Naresh},
  booktitle = {Machine Learning and Knowledge Discovery in Databases (ECML PKDD 2021)},
  pages     = {731--748},
  year      = {2021},
  doi       = {10.1007/978-3-030-86520-7_45},
  note      = {arXiv:2109.05535; pages and DOI from dblp/search listing (Springer page not loadable)},
  url       = {https://arxiv.org/abs/2109.05535}
}

@article{sadeghi2022ktopt,
  title   = {On Characterizing the Trade-off in Invariant Representation Learning},
  author  = {Sadeghi, Bashir and Dehdashtian, Sepehr and Boddeti, Vishnu Naresh},
  journal = {Transactions on Machine Learning Research},
  year    = {2022},
  note    = {arXiv:2109.03386; OpenReview id 3gfpBR1ncr},
  url     = {https://arxiv.org/abs/2109.03386}
}

@inproceedings{dehdashtian2024ufate,
  title     = {Utility-Fairness Trade-Offs and How to Find Them},
  author    = {Dehdashtian, Sepehr and Sadeghi, Bashir and Boddeti, Vishnu Naresh},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024},
  note      = {arXiv:2404.09454},
  url       = {https://arxiv.org/abs/2404.09454}
}

@inproceedings{zhang2011kci,
  title     = {Kernel-based Conditional Independence Test and Application in Causal Discovery},
  author    = {Zhang, Kun and Peters, Jonas and Janzing, Dominik and Sch{\"o}lkopf, Bernhard},
  booktitle = {Proceedings of the 27th Conference on Uncertainty in Artificial Intelligence (UAI)},
  pages     = {804--813},
  year      = {2011},
  note      = {arXiv:1202.3775},
  url       = {https://arxiv.org/abs/1202.3775}
}

@article{strobl2019rcit,
  title   = {Approximate Kernel-Based Conditional Independence Tests for Fast Non-Parametric Causal Discovery},
  author  = {Strobl, Eric V. and Zhang, Kun and Visweswaran, Shyam},
  journal = {Journal of Causal Inference},
  volume  = {7},
  number  = {1},
  year    = {2019},
  doi     = {10.1515/jci-2018-0017},
  note    = {arXiv:1702.03877},
  url     = {https://arxiv.org/abs/1702.03877}
}

@misc{chernozhukov2016dml,
  title         = {Double/Debiased Machine Learning for Treatment and Causal Parameters},
  author        = {Chernozhukov, Victor and Chetverikov, Denis and Demirer, Mert and Duflo, Esther and Hansen, Christian and Newey, Whitney and Robins, James},
  year          = {2016},
  eprint        = {1608.00060},
  archivePrefix = {arXiv},
  note          = {journal version not verified in this audit},
  url           = {https://arxiv.org/abs/1608.00060}
}

@article{fan1949weyl,
  title   = {On a Theorem of {W}eyl Concerning Eigenvalues of Linear Transformations {I}},
  author  = {Fan, Ky},
  journal = {Proceedings of the National Academy of Sciences of the United States of America},
  volume  = {35},
  number  = {11},
  pages   = {652--655},
  year    = {1949},
  doi     = {10.1073/pnas.35.11.652}
}

@inproceedings{makhdoumi2014funnel,
  title     = {From the Information Bottleneck to the Privacy Funnel},
  author    = {Makhdoumi, Ali and Salamatian, Salman and Fawaz, Nadia and M{\'e}dard, Muriel},
  booktitle = {2014 IEEE Information Theory Workshop (ITW)},
  pages     = {501--505},
  year      = {2014},
  note      = {arXiv:1402.1774},
  url       = {https://arxiv.org/abs/1402.1774}
}

@article{rassouli2021perfect,
  title   = {On Perfect Privacy},
  author  = {Rassouli, Borzoo and G{\"u}nd{\"u}z, Deniz},
  journal = {IEEE Journal on Selected Areas in Information Theory},
  volume  = {2},
  number  = {1},
  pages   = {177--191},
  year    = {2021},
  doi     = {10.1109/JSAIT.2021.3053432},
  note    = {arXiv:1712.08500},
  url     = {https://arxiv.org/abs/1712.08500}
}

@article{sankar2013utility,
  title   = {Utility-Privacy Tradeoffs in Databases: An Information-Theoretic Approach},
  author  = {Sankar, Lalitha and Rajagopalan, S. Raj and Poor, H. Vincent},
  journal = {IEEE Transactions on Information Forensics and Security},
  year    = {2013},
  doi     = {10.1109/TIFS.2013.2253320},
  note    = {arXiv:1102.3751},
  url     = {https://arxiv.org/abs/1102.3751}
}

@inproceedings{liao2019sideinfo,
  title     = {Robustness of Maximal $\alpha$-Leakage to Side Information},
  author    = {Liao, Jiachun and Sankar, Lalitha and Kosut, Oliver and Calmon, Flavio P.},
  booktitle = {IEEE International Symposium on Information Theory (ISIT)},
  year      = {2019},
  note      = {arXiv:1901.07105},
  url       = {https://arxiv.org/abs/1901.07105}
}

@inproceedings{stadler2024leastprivilege,
  title     = {The Fundamental Limits of Least-Privilege Learning},
  author    = {Stadler, Theresa and Kulynych, Bogdan and Gastpar, Michael and Papernot, Nicolas and Troncoso, Carmela},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {235},
  pages     = {46393--46411},
  year      = {2024},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v235/stadler24a.html}
}

@inproceedings{elazar2018adversarial,
  title     = {Adversarial Removal of Demographic Attributes from Text Data},
  author    = {Elazar, Yanai and Goldberg, Yoav},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  pages     = {11--21},
  year      = {2018},
  address   = {Brussels, Belgium},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/D18-1002},
  url       = {https://aclanthology.org/D18-1002/}
}

@inproceedings{mcallester2020formal,
  title     = {Formal Limitations on the Measurement of Mutual Information},
  author    = {McAllester, David and Stratos, Karl},
  booktitle = {Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics},
  series    = {Proceedings of Machine Learning Research},
  volume    = {108},
  pages     = {875--884},
  year      = {2020},
  publisher = {PMLR},
  note      = {arXiv:1811.04251},
  url       = {https://proceedings.mlr.press/v108/mcallester20a.html}
}

@inproceedings{belrose2023leace,
  title     = {{LEACE}: Perfect Linear Concept Erasure in Closed Form},
  author    = {Belrose, Nora and Schneider-Joseph, David and Ravfogel, Shauli and Cotterell, Ryan and Raff, Edward and Biderman, Stella},
  booktitle = {Advances in Neural Information Processing Systems 36 (NeurIPS 2023)},
  year      = {2023},
  note      = {arXiv:2306.03819},
  url       = {https://proceedings.neurips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract-Conference.html}
}

@inproceedings{holstege2025splince,
  title     = {Preserving Task-Relevant Information Under Linear Concept Removal},
  author    = {Holstege, Floris and Ravfogel, Shauli and Wouters, Bram},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS 2025)},
  year      = {2025},
  note      = {arXiv:2506.10703},
  url       = {https://arxiv.org/abs/2506.10703}
}

@misc{zhao2019conditional,
  title         = {Conditional Learning of Fair Representations},
  author        = {Zhao, Han and Coston, Amanda and Adel, Tameem and Gordon, Geoffrey J.},
  year          = {2019},
  eprint        = {1910.07162},
  archivePrefix = {arXiv},
  note          = {ICLR 2020 venue not verified in this audit},
  url           = {https://arxiv.org/abs/1910.07162}
}

@inproceedings{hwa2024enforcing,
  title     = {Enforcing Conditional Independence for Fair Representation Learning and Causal Image Generation},
  author    = {Hwa, Jensen and Zhao, Qingyu and Lahiri, Aditya and Masood, Adnan and Salimi, Babak and Adeli, Ehsan},
  booktitle = {IEEE/CVF CVPR Workshop on Fair, Data-Efficient, and Trusted Computer Vision},
  year      = {2024},
  note      = {arXiv:2404.13798},
  url       = {https://arxiv.org/abs/2404.13798}
}

@inproceedings{arevalo2024tappfl,
  title     = {Task-Agnostic Privacy-Preserving Representation Learning for Federated Learning Against Attribute Inference Attacks},
  author    = {Arevalo, Caridad Arroyo and Noorbakhsh, Sayedeh Leila and Dong, Yun and Hong, Yuan and Wang, Binghui},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2024},
  note      = {arXiv:2312.06989},
  url       = {https://arxiv.org/abs/2312.06989}
}
```
