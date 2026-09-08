# PCRL redesign: verified prior-work matrix

Reviewed 2026-09-07 against accessible primary papers and author repositories.
Repository starting commit: `15dbcc3c3338f6707e7a0b3901d738d99651a77d`.
This is a focused comparison, not an exhaustive novelty search. No methods were
run, dependencies installed, or historical results modified for this review.

The reviewed literature already contains task-aware linear erasure, adversarial
representation learning, transfer to previously unseen task labels, and
inference-time selection of protected-attribute subsets. PCRL needs evidence for
a more specific contribution than combining these descriptions. The final column
below is our assessment of what remains unresolved for this repository; it is
not a claim that the question is absent from all prior work.

Here, **release** means the actual representation or predictions received by an
external party. **Coalition** means a party observing several releases for the
same individual. Multiple task labels, multiple protected attributes, subgroup
fairness, and multiple recipients are different properties. A paper's
“composition” terminology does not by itself establish coalition protection.

| Work and primary sources | Interface and assumptions | Attackers, guarantees, and empirical scope | Multiple purposes and coalitions | Consequence and unresolved question for PCRL |
|---|---|---|---|---|
| **LEACE**, Belrose et al., NeurIPS 2023. [Paper, Definitions 2.3 and Theorems 3.1/4.2; §4.3](https://arxiv.org/html/2306.03819v3); [official code](https://github.com/EleutherAI/concept-erasure). | Fit an affine eraser to representations and prohibited labels; apply using features alone. Finite moments; categorical guardedness assumes supported classes. Continuous vector targets are explicitly supported for ordinary least squares. | Exact cross-covariance removal and minimum mean-square representation distortion in the stated affine class. Categorical guardedness concerns affine scores and losses convex in those scores. It does **not** establish universal threshold-classifier accuracy or nonlinear secrecy. Finite-sample fitting is not a population certificate. | A target matrix can contain several prohibited signals. Fitting separately for each declared policy is a direct baseline. The reviewed formulation does not evaluate a recipient combining policy-specific releases. | Separate LEACE is essential. Linear leakage elimination or per-purpose fitting alone cannot establish an additional PCRL contribution. Does adaptation improve held-out utility under a stronger, independently fitted attack? |
| **SPLINCE**, *Preserving Task-Relevant Information Under Linear Concept Removal*, Holstege, Ravfogel and Wouters, NeurIPS 2025. [Paper, Theorems 1–2 and §3.3](https://arxiv.org/html/2506.10703v2); [official code](https://github.com/fholstege/SPLINCE). | Affine transformation fitted with features, prohibited labels, and task labels; categorical or continuous labels. Theorem 1 requires finite second moments and zero intersection between the whitened task and prohibited covariance spans. | Removes prohibited covariance while exactly retaining task covariance and minimizing distortion under both constraints. Covariance retention is not an accuracy guarantee. Theorem 2 gives identical predictions for projections with the same kernel under its unregularized refitting and unique-minimizer assumptions; regularized or frozen heads can differ. Experiments examine these differences. | Vector task/prohibited labels are permitted. Separate purpose transforms are a natural comparison; a coalition threat model is not established in the examined sections. | Task-preserving erasure is already explicit prior work. Comparisons must state head refitting and regularization. Could a claimed improvement merely come from projection coordinates or the head protocol? |
| **The Fundamental Limits of Least-Privilege Learning**, Stadler et al., ICML 2024. [Paper, §2, Definition 2 and Theorem 2](https://arxiv.org/pdf/2402.12235); [proceedings](https://proceedings.mlr.press/v235/stadler24a.html). | Possibly randomized feature release. Formal analysis uses finite discrete spaces, full support, and strictly positive task posterior `P(Y=y\|X=x)` for every pair. The adversary may also know the authorized task label. | Bounds worst-case conditional inference gain over **all** attribute distributions satisfying the stated Markov relation, using conditional maximal leakage. Under its assumptions, an LPP budget `γ` cannot coexist with task information `Iα(Y;Z)>γ`, for `α=1,∞`. Empirical censoring results illustrate the tradeoff. | One intended task; leakage is measured beyond what its label already reveals. This is stronger than suppressing a fixed declared list. It does not directly settle the continuous deterministic Gaussian pilot or a recipient-specific coalition policy. | Declare protected targets and unavoidable task leakage. Success on U/V/S cannot imply universal least privilege. Is the real policy absolute suppression or protection beyond authorized outputs? |
| **Fundamental Limits of Perfect Concept Erasure**, Chowdhury et al., AISTATS 2025. [Paper, assumptions A1–A5, Lemma 3.3, Definition 4.1 and Method 4.2](https://arxiv.org/html/2503.20098v1); [official PEF code](https://github.com/brcsomnath/PEF). | Release `Z=f(X)` without a concept label at inference. Theory assumes finite input/output/concept supports and disjoint input supports across concept groups. Equal group distributions up to permutation permit a piecewise bijection to a common distribution. | Perfect erasure is `I(Z;A)=0`, against unrestricted attribute inference; retained information obeys `I(Z;X)≤H(X\|A)`. The exact construction reaches this ceiling under the additional permutation conditions. Approximate learned mappings and GPT-4 representation experiments do not certify arbitrary continuous distributions. | Task-agnostic retention of original information rather than utility for specified task heads. One erased concept/release is formalized; joint observation of independently constructed releases requires additional analysis. | A stronger privacy objective and informative feasibility assumptions already exist. Which useful data distribution and implementable interface support a stronger guarantee than empirical probe failure? |
| **Robust Privatization with Multiple Tasks and the Optimal Privacy-Utility Tradeoff**, Liu and Wang. [Full paper, §§II–IV](https://arxiv.org/html/2010.10081); [version record: 2020 preprint, 2024 v3](https://arxiv.org/abs/2010.10081). | A common sanitized release supports a family of possibly unknown tasks. Discrete memoryless source model, known joint distribution, task distortion constraints. Special tractable case: independent input components, componentwise deterministic private feature, and tasks reconstructing component subsets. | Information-theoretic rate/leakage/distortion characterization; leakage is mutual information. Under log loss and the special structure, parallel privacy funnels plus an LP characterize optimal allocation. General optimization remains nonconvex. Numerical examples examine robustness to unknown tasks. | Explicit multiple-task utility constraints for one release and one private feature vector. Distinct recipients with conflicting authorized sets and pooling of their releases are not the model examined. | Preserving utility for many possible tasks is established prior work. How does a proposed recipient-specific policy differ from a common-release task family, and is its benefit measurable under comparable interfaces? |
| **LAFTR**, *Learning Adversarially Fair and Transferable Representations*, Madras et al., ICML 2018. [Paper, §§5–6.2](https://www.cs.toronto.edu/~zemel/documents/laftr-icml.pdf); [official code](https://github.com/VectorInstitute/laftr). | Train an encoder using task/reconstruction and adversarial fairness losses; release its features to another learner. Fairness objectives include demographic parity and label-conditioned equalized odds/equal opportunity. | Theoretical fairness bounds require an optimal adversary for the relevant function class/objective. A finite trained MLP does not establish that optimum. Independent downstream models and a separate attribute MLP provide empirical evaluation. Health experiments hold out ten medical-condition task labels from representation training. | Explicit reuse by downstream learners with new tasks. Equalized-odds arguments depend on which task label conditions the adversary; transfer results are empirical. No coalition certificate is identified in the examined formulation. | A strong transferable adversarial baseline, with objective and task conditioning matched. Can PCRL improve on it when tested on genuinely held-out tasks rather than its own training heads? |
| **FFVAE**, *Flexibly Fair Representation Learning by Disentanglement*, Creager et al., ICML 2019. [Paper, §§4–5](https://proceedings.mlr.press/v97/creager19a/creager19a.pdf); [proceedings](https://proceedings.mlr.press/v97/creager19a.html). | Learn a nonsensitive latent block and attribute-aligned latent coordinates with reconstruction, sensitive-label prediction, and total-correlation penalties. Sensitive labels are needed for fitting, not inference. Remove selected coordinates at test time. | Desired factorization and attribute predictiveness motivate the construction; finite neural density-ratio estimation and optimization do not certify exact independence. Evaluation freezes the encoder and fits MLP audits, including new task labels and subgroup definitions. | Explicit adaptation to multiple sensitive attributes and their conjunctions after training. This is policy flexibility, not a demonstrated guarantee when a recipient pools complementary releases. | Configurable removal and held-out task transfer already coexist in prior work. A PCRL contribution needs more than renamed purpose labels. Can a justified coalition policy be satisfied while retaining useful task-family flexibility? |
| **Compositional Fairness Constraints for Graph Embeddings**, Bose and Hamilton, ICML 2019. [Paper, §4, Theorem 1 and §5](https://proceedings.mlr.press/v97/bose19a/bose19a.pdf); [official code](https://github.com/joeybose/Flexible-Fairness-Constraints). | Average selected learned attribute-filter outputs on a shared graph embedding. Random attribute masks during training; graph relation utility and adversarial attribute losses. Users may request different subsets at inference. | Nonlinear MLP discriminators and independent fitted auditors. Theorem 1 assumes sufficient capacity, optimal discriminator updates, and adversarial weight tending to infinity; trivial representations remain possible. Finite-budget recommendation/graph experiments are empirical. | Tests previously unseen combinations of protected attributes. The paper explicitly notes that separate attribute independence need not imply joint subgroup fairness. Its composition is filter combination, not pooled recipient views. | A directly relevant policy-conditioned adversarial baseline. Multi-policy sharing alone is not a novelty argument. Do separate-release risks require a different constraint, and does it survive fresh attacks? |
| **TIPRDC**, Li et al., KDD 2020. [Paper, §§3–5](https://par.nsf.gov/servlets/purl/10281575); [official conference record](https://kdd.org/kdd2020/accepted-papers/view/tiprdc-task-independent-privacy-respecting-data-crowdsourcing-framework-for.html). | Distribute a feature extractor to data contributors and collect representations for unknown or changing downstream tasks. A chosen private attribute guides fitting; utility seeks retained input information conditional on that attribute. | Nonlinear privacy adversary plus a neural mutual-information objective. Variational adversary substitutions and finite optimization are not computable universal leakage upper bounds. Image/text experiments evaluate the privacy–utility tradeoff. | Task-independent representation reuse and user-selected protected attributes are explicit motivations. A common collected representation is evaluated; the examined model does not provide a bound for pooling policy-specific releases. | A prior application-oriented formulation of reusable private features. The missing PCRL evidence is a concrete requirement that beats an adequate prediction bank or trusted service, not simply an unknown-task motivation. |
| **Replacement AutoEncoder / sensor-data transformations**, Malekzadeh et al., IoTDI 2018 / PMC 2020. [RAE paper, §§II–IV](https://haddadi.github.io/papers/RAE2018IoTDI.pdf); [expanded paper, §§3–5](https://arxiv.org/pdf/1911.05996); official [RAE code](https://github.com/mmalekzadeh/replacement-autoencoder) and [MotionSense code/data](https://github.com/mmalekzadeh/motion-sense). | Release transformed sensor windows of the original shape through an on-device mediator. Separate required, sensitive, and neutral activities; replace sensitive windows with neutral patterns. An additional anonymizing autoencoder targets identity. Declared application requirements and representative labeled windows guide fitting. | Empirical utility, sensitive-activity classification, identity prediction and replacement-detection tests. The expanded paper's §4.2 identity attack trains on **raw** data and evaluates transformed data; it does not establish resistance to an attacker refitted on the release. Discussion flags correlations across activities and consecutive windows. | Explicit application-specific permissions and a cascade of transformations. Cascade evaluation is different from adversarial pooling of independently released app views. | A concrete wearable release interface and compatibility motivation already exist. A new HAR study needs fresh transformed-data attackers and a valid inference policy; publishing a derived activity task while prohibiting all activity information remains contradictory. |
| **R-LACE**, *Linear Adversarial Concept Erasure*, Ravfogel et al., ICML 2022. [Paper, §§2–4](https://proceedings.mlr.press/v162/ravfogel22a/ravfogel22a.pdf); [official code](https://github.com/shauli-ravfogel/rlace-icml). | Postprocess frozen features with a projection of specified rank, playing a minimax game with a generalized linear predictor. Classification and regression objectives are distinguished. | Closed-form results for squared regression and Rayleigh objectives; convex relaxation for classification. Nonlinear downstream bias mitigation is an empirical result, not unrestricted erasure. Rank and distortion restrictions differ from LEACE. | One concept-removal intervention; separate per-policy fits are possible baselines. No coalition result is identified in the examined formulation. | Adversarially optimized erasure itself is established. Any comparison must equalize retained dimensions, predictor loss and fitting effort, or explicitly report the differences. |
| **Adversarial Removal of Demographic Attributes from Text Data**, Elazar and Goldberg, EMNLP 2018. [Paper, §§2, 5.1–5.2](https://aclanthology.org/D18-1002.pdf); [official code](https://github.com/yanaiela/demog-text-removal). | Task-trained text encoder with a gradient-reversal demographic adversary. After freezing the encoder, fit a separate attacker on its representations. | Demonstrates demographic recovery despite a training adversary near chance, including an attacker with the same architecture. Larger adversaries, changed weights, and ensembles are investigated. This is empirical failure evidence, not an impossibility theorem for all adversarial learning. | Several task/attribute pairs are tested; no recipient coalition construction. | Fresh attackers and frozen-release continuation diagnostics are necessary evaluation practice, not a new method contribution. Does an apparent privacy gain survive retraining, initialization changes, and correct final-release preprocessing? |

For the least-privilege, robust-privatization, FFVAE, and TIPRDC papers, an author
implementation was not identified in the checked paper/proceedings pages and
targeted repository searches. This does not establish that no code exists.
The other repository links were reached from the paper or verified against
their author README. Repository contents were inspected read-only; their code
was not installed, executed, or certified to reproduce the publications.
The SPLINCE repository also uses the shorter spelling “SPLICE” in parts of its
README; the matrix uses the paper's SPLINCE name.

For precision, let `G(A;B)=max_g Pr[A=g(B)]` and
`G(A)=max_a Pr[A=a]`. For a fixed declared attribute A, absolute guessing gain
is `log₂(G(A;Z)/G(A))`; gain beyond an authorized output Y is
`log₂(G(A;(Z,Y))/G(A;Y))`. Maximal leakage instead ranges over arbitrary finite
randomized inference targets:

```text
L(X→Z)   = sup[A−X−Z]       log₂(G(A;Z) / G(A))
L(X→Z|Y) = sup[A−(X,Y)−Z]   log₂(G(A;(Z,Y)) / G(A;Y))
```

These are optimal population guessing probabilities, not fitted probe scores.
For finite alphabets, existing composition results give
`L(X→(Z1,Z2)) ≤ L(X→Z1)+L(X→Z2|Z1)`; conditional independence
`Z1−X−Z2` further permits the sum of unconditional leakages. This is already
prior theory, with explicit conditioning requirements. [Issa, Wagner and Kamath,
Definitions 1/6, Corollary 2 and Lemma 6](https://arxiv.org/pdf/1807.07878).
The least-privilege paper constrains the conditional quantity with the task
label as Y. A declared-label audit is a narrower evaluation. Predictive R²
is neither of these guessing-gain measures. Applying the sum bound to a
particular protected S requires conditional independence given S; releases
being deterministic functions of X does not establish that assumption.

The comparison has four immediate implications for the redesign:

1. **Define the release and the allowed inference first.** A finite family of
   known labels can often be served by a prediction bank. Representation access
   needs an independently justified requirement, such as fitting new models for
   a specified held-out task family. Published transfer experiments demonstrate
   such an evaluation interface; they do not establish that PCRL's prospective
   data owner must export features instead of operating a trusted service.
2. **Separate empirical attack results from claims about all attackers.**
   Least-squares fitting statistics, held-out affine prediction, nonlinear probe
   performance, optimal-adversary theorems, and mutual information answer
   different questions. The repository's [retired accuracy certificate](ACCURACY_CERTIFICATE_RETIREMENT.md)
   remains invalid. Failed finite attacks provide evidence under an audit
   budget; they do not prove that the protected signal is absent.
3. **Write a coalition policy separately.** In PCRL's synthetic conflict,
   U and V become authorized when both recipients combine their releases; S
   remains prohibited. Requiring the coalition to forget U and V would contradict
   the individual tasks. Exact zero covariance with the same prohibited S
   composes under concatenation by blockwise linearity of covariance. Approximate
   per-view R² does not give the same conclusion, as the repository's
   `N ± δS` example demonstrates. This is a mathematical scope distinction, not
   a newly discovered failure of exact linear erasure.
4. **Require a gain beyond appropriate simple baselines.** Independent LEACE,
   task-aware erasure where assumptions apply, transferable adversarial learning,
   and configurable adversarial filters cover different portions of the design.
   Match release dimensions, actual task-head access, preprocessing, attacker
   labels/capacity, optimization budgets, and validation-only selection. A
   difference against one mismatched baseline cannot establish a general gain.

The narrow unresolved research question is whether a **justified reusable
release interface**, with a declared family of future tasks and recipient
coalitions, admits a useful improvement over these existing methods at matched
utility and independent attack strength. Neither this literature review nor
the repository's synthetic pilots establishes that improvement. Application
selection and a precise policy should precede another method experiment. The
separate [application screen](PCRL_APPLICATION_SELECTION.md) examines HAR,
Diabetes, and ACS; the additional literature here does not add application
candidates or establish that those three satisfy the release-necessity gate.

## Amendment — 2026-09-07: scope of the ACS transfer benchmark

The authorized [ACS transfer experiment](../results/redesign_20260907_acs_transfer_v1/PROTOCOL.md)
uses a defined one-time release and equal downstream fitting access. A trusted
service is a deployment alternative, not an empirical admission gate. This
controlled comparison asks whether source-only features support two reserved task
identities beyond rich neural and tree prediction banks; it does not claim a
new transfer-learning formulation or a protection advantage. The earlier focused
literature matrix and its limits remain unchanged.
