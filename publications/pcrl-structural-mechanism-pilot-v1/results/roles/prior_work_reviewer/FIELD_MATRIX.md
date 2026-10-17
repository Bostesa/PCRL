# Field matrix: structural (many-to-one) release mechanisms and their closest prior work

Role: closest-work and baseline reviewer, PCRL structural-mechanism campaign (phase 1). Written 2026-10-16.

**What was done.** I read primary papers (PDF text extracted with `pdftotext`, read at the cited sections) and official code (raw files at pinned commits), plus this repository's FARE and FAGTB code and the ACS replication result files. I opened no real-person data (`.private/`, `.npz`) and ran nothing except downloads and text extraction. I did not open `roles/designer_1/` or `roles/designer_2/`. No priority claim is made anywhere. "Not found" means not found in the sources listed in §6.

**Markers.**
- **[P]** paper text read today at the cited definition, algorithm or section.
- **[C]** official code read today at the stated commit or branch.
- **[R]** this repository's file read today.
- **[W]** a web record or abstract page read today.
- **[prev]** taken from the earlier reviewer's `results/pcrl_mechanism_design_v1/roles/prior_work_reviewer/LITERATURE_BASE.md` and not re-opened today.
- **[D]** the reviewer's own elementary derivation; it is not a claim found in the cited source.

---

## 1. Matrix

Rows are keyed by ID, and the two tables share those IDs. "Many-to-one" means the deployed map sends distinct inputs to one identical released object by construction.

### 1A. Objective, partition or representation, optimisation, task heads

| ID | Work | Objective | Input partition or representation (how cells or codes form; many-to-one?) | Optimisation | Task heads (how probabilities come from cells or codes; calibration) |
|---|---|---|---|---|---|
| F1 | **FARE**, Jovanović, Balunović, Dimitrov, Vechev, ICML 2023, PMLR 202:15401–15420 [P][W]; official code `eth-sri/fare` @ `89cb1b6` [C] | A *restricted encoder* f: X → {z_1…z_k} for which a practical certificate on downstream unfairness can be computed (§5). Instantiated as a fair tree with FairGini(D) = (1 − γ)Gini_y(D) + γ(0.5 − Gini_s(D)) (§6). The second term "aims to maximize Gini_s(D), i.e., make the distribution of s in each leaf … as close to uniform" | **Many-to-one: yes.** A CART tree on (x, y) is grown best-first to at most k̄ leaves (`max_leaf_nodes`), with at least n_i training rows per leaf (`min_samples_leaf`). Every row in leaf i is encoded as z_i: "the median for continuous, and the most common value for categorical features" (§6; `main.py` L110–134). Categoricals are ordinal, with a generalised Breiman ordering under several orderings q. Paper ranges: γ ∈ [0, 1], k̄ ∈ [2, 200], n_i ∈ [50, 1000] (App. E) | Greedy top-down split selection minimising the weighted FairGini (`sktree/_criterion.pyx` `FairGiniDP` L831–851: `(1.0 - alpha) * g_label + alpha * (0.5 - g_sens)`). There is no global optimisation and no refinement after growth | **None inside the method.** The released object is z, and data consumers train their own downstream classifiers on it: a "1-hidden-layer neural network (1-NN)" in the main results, with others in App. H.1 (decision tree with ≤ 2,500 leaves, logistic regression, …). The tree's own `predict_proba` (leaf class frequencies) is used only in its evaluation (`main.py` L85–91). Calibration is not discussed. The official script uses one label: with several labels "getting 0th label!" (`main.py` L26–28) |
| F2 | **FARE as run in this repository** (ACS FL 2019) [R]: `acs_replication/fare_worker.py`, `oar/fare_official.py` | Same official tree, one tree **per task**. Settings F1 (γ 0.3, k̄ 100), F2 (γ 0.7, k̄ 50), F3 (γ 0.85, k̄ 20), all with n_i 100. Twins Z1–Z3 have γ = 0 and the same k̄ and n_i, which is exactly the plain Gini tree | Many-to-one: yes. F1 and Z1 realised 100 cells for each task on both seeds (`MODEL_MANIFEST.json`) | The official build. The single compiled change is the sensitive-count buffer size (`FARE_ENV_RECIPE.md`) | `jcv.finalize.fit_head`: StandardScaler plus LogisticRegression on **one-hot cell ids**, with C ∈ {0.01…100} chosen by internal-fold log loss on a household-disjoint split of MODEL_FIT. The result is one regularised probability vector per cell. No separate calibration |
| K1 | **Discrimination-aware decision trees**, Kamiran, Calders, Pechenizkiy, ICDM 2010 (author PDF [P]; pages 869–874 appear in an institutional record that surfaced in search but was not opened) | The most accurate tree whose discrimination disc_B(C, D) is ≤ ε | Many-to-one: yes (a tree). Three split criteria combine IGC, the information gain on the class, with IGS, the gain on the sensitive B: **IGC − IGS**, IGC/IGS and IGC + IGS. The paper notes that IGC + IGS "favors splits that result in a homogenous" partition w.r.t. B, and that it works in combination with relabelling | Greedy splitting, then **leaf relabelling**. RELAB is shown equivalent to KNAPSACK (Thm 1) and approximated greedily by the ratio Δdisc/Δacc (Alg. 1) | **Hard labels only.** Each leaf gets its majority class, flipped for the selected leaves. No probabilities |
| K2 | **Optimal and fair decision trees (MIP)**, Aghaei, Azizi, Vayanos, AAAI 2019 (arXiv 1903.10598) [P] | min ℓ(T, ŷ) + λ·ℓ_d(T, ŷ), with ℓ_d = DIDI (disparate impact) or DTDI (disparate treatment) (eq. 6–7a) | Many-to-one: yes (a depth-bounded tree; variants with linear branching or linear leafing) | Mixed-integer program with a convex piecewise-linear objective | Leaf predictions: classes, or linear leaf scores for regression. No probabilities |
| B1 | **FAGTB**, Grari, Ruf, Lamprier, Detyniecki, ICDM 2019 (arXiv 1911.05369) [P]; official code `vincent-grari/FAGTB` `functions.py` [C]; repository port `acs_replication/fagtb.py` [R] | arg min_F max_θA Σ L_F − λ Σ L_A, where adversary A predicts S from σ(F(x)) for demographic parity, or from (σ(F(x)), y) for equalised odds (eq. 8, §III-A) | **Not many-to-one by design.** An additive ensemble of regression trees (depth 9 hard-coded in the code) gives a continuous score | Each round fits a **new** tree to u = r − λt (task plus adversary pseudo-residuals), then a line search (paper) or a fixed learning rate (code). The adversary is updated once per round. The code's adversary is a single logistic unit in TensorFlow, binary task only | p = σ(F(x)) and the argmax. No calibration. The repository's multiclass version is a documented softmax adaptation, not official (`FAGTB_AUDIT.md` §10) |
| B2 | **FairGBM**, Cruz et al., ICLR 2023 (arXiv 2209.07850) [P]; code `feedzai/fairgbm` [prev] | min loss s.t. group-rate constraints (FPR, FNR, parity), via a "proxy-Lagrangian" with cross-entropy-based proxies | Not many-to-one by design (GBDT score) | Gradient descent on the model and ascent on the multipliers, within boosting | σ(score). The guarantee is for a **randomised** classifier over iterates; "using solely the last iterate … will result in a deterministic classifier" without that guarantee (§3) |
| L1 | **Learning Fair Representations (LFR)**, Zemel, Wu, Swersky, Pitassi, Dwork, ICML 2013, PMLR 28 [P] | L = A_z·L_z + A_x·L_x + A_y·L_y (eq. 4). L_z = Σ_k \|M_k⁺ − M_k⁻\| is statistical parity of mean prototype memberships; L_x is reconstruction; L_y is log loss | **Soft, not many-to-one.** K prototypes v_k, with P(Z = k \| x) = softmax(−d(x, v_k)) (eq. 2) and feature weights α. The extended model uses **group-specific α⁺ and α⁻** | L-BFGS | **Per-prototype probabilities:** ŷ_n = Σ_k M_{n,k} w_k with w_k ∈ [0, 1] (eq. 11). This is the closest precedent found for "probability heads on cells", in soft form. No calibration step |
| C1 | **Optimized pre-processing**, Calmon, Wei, Vinzamuri, Ramamurthy, Varshney, NeurIPS 2017 (arXiv 1704.03354) [P] | min Δ(p_{X̂,Ŷ}, p_{X,Y}) s.t. discrimination control J(p_{Ŷ\|D}(y\|d), p_{Y_T}(y)) ≤ ε and distortion control E[δ] ≤ c_{d,x,y} for every (d, x, y) (eq. 7) | **A randomised mapping** p_{X̂,Ŷ\|X,Y,D} on discrete (D, X, Y), with a reduced test-time map p_{X̂\|X,D} (§2). It can move or merge mass; it is not a deterministic partition | A (quasi)convex program on a known, in practice empirical, law (Prop. 1) | None inside the method: any model is trained on the transformed data (Ŷ \| X̂, D) |
| P1 | **Privacy funnel**, Makhdoumi, Salamatian, Fawaz, Médard, ITW 2014 (arXiv 1402.1774) [P] | min I(S; Y) s.t. I(X; Y) ≥ R (eq. 3). **Utility is information about the raw input X, not about a task** | **Many-to-one: yes.** Alg. 1 starts from the identity map on the finite alphabet X and repeatedly merges two output symbols, so the final P_{Y\|X} is a 0/1 matrix | Greedy: merge the pair that most reduces I(S; Y) while I(X; Y) ≥ R holds, with closed-form merge costs (Prop. 1). "No guarantee that such a greedy algorithm induces a global optimal privacy mapping". Alg. 2 gives the local maximum | None. The release is the merged symbol |
| P2 | **Agglomerative information bottleneck (AIB)**, Slonim and Tishby, NIPS 12 (1999), pp. 617–623 [P] | Compress X into m clusters while keeping I(Z; Y) about a relevant variable Y | **Hard partition** of finite X, built bottom-up from singletons ("fully deterministic, yielding … hard clusters") | Greedy: merge the pair whose loss in I(Z; Y) is smallest. That loss is a prior-weighted Jensen–Shannon divergence of p(y \| z_i) and p(y \| z_j) | The merged conditionals p(y \| z) |
| P3 | **IB with side information (IBSI)**, Chechik and Tishby, NIPS 15 (2002) [P] | ℒ = I(X; T) − β[I(T; Y⁺) − γ·I(T; Y⁻)] (eq. 1). Keep information about a relevant Y⁺ and **remove information about an irrelevance variable Y⁻** | A stochastic map p(t \| x) over categorical variables "whose co-occurrence distributions are known". It collapses to hard clustering in the limit. The paper assumes Y⁺ ⊥ Y⁻ \| X; since T depends on X only, the functional itself needs only p(x, y⁺) and p(x, y⁻) [D] | Self-consistent iterations. The paper also lists deterministic annealing, "greedy agglomerative hard clustering [13]" (that is, AIB) and a sequential K-means-like algorithm, which is the one it used (§5) | p(y⁺ \| t) by construction; it is not used as a predictor in the paper |
| P4 | **Watchdog with subset merging**, Zarrabian, Ding, Sadeghi, Rakotoarivelo, arXiv 2110.04724 (2021) [P][W]; the ICASSP 2022 venue comes from a search listing and was not verified. Base watchdog: Hsu, Asoodeh, Calmon, ISIT 2019 (author PDF returned HTTP 403, not read) | Lift-based privacy: every released symbol's lift (the ratio of P(s \| ·) to P(s)) lies within threshold ε, while utility I(X; Y) is maximised | **Many-to-one on a subset.** Symbols whose lift exceeds ε are "high-risk". The base watchdog merges all of them into one; this paper merges them greedily, starting "with a singleton high-risk symbol, which is iteratively fused with others until the resulting subsets are private" (abstract, Alg. 1). Low-risk symbols are released as-is | Greedy agglomerative search for the finest admissible partition of the high-risk set | None |
| P5 | **Fair clustering through fairlets**, Chierichetti, Kumar, Lattanzi, Vassilvitskii, NIPS 2017 (arXiv 1802.05733) [P] | k-center or k-median clustering in which every cluster has balance ≥ t between two colours (protected classes) (Def. 1) | **Hard clusters of a fixed point set.** The clustering is transductive; no rule for new points | Fairlet decomposition (approximation algorithms) followed by ordinary clustering | None (unsupervised) |
| S1 | **Multilevel-trust PPDM**, Li, Chen, Li, Zhang, IEEE TKDE (the PDF header reads "accepted … 2011" [P]; 24(9), 2012 [prev]) (arXiv 1104.0459) | Copies at different trust levels such that colluders reconstruct no better than with the best single copy they hold | Additive Gaussian noise, **correlated across copies**; not many-to-one | Closed-form construction of the noise covariance; copies are generated on demand | — |
| S2 | **Adaptive privacy of sequential releases under collusion**, Taylor, Vippathalla, Coon, arXiv 2601.21859 v2 (2026) [P] | Per request: maximise utility s.t. I(R̂_k; X) ≤ ε_k and the collusion constraint on (R̂_k, R̂^{k−1}) [prev for the exact form] | Randomised channels p(r̂_k \| r̂^{k−1}, x) on finite alphabets | Blahut–Arimoto-style: global optimum for distortion utility, local for MI utility (abstract) | — |
| S3 | **Anonymizing sequential releases**, Wang and Fung, KDD 2006, pp. 414–423 [P] | Generalise the current release T₁ so that its join with a **fixed previous release** T₂ satisfies (X, Y)-privacy, while T₁ stays useful for its own classification purpose | **Many-to-one: yes.** Generalisation coarsens quasi-identifier values along taxonomies; the "lossy join" is the device | Greedy top-down specialisation with monotonicity pruning; an optimal solution is NP-hard | The recipient trains its own classifier on the generalised table |
| S4 | **Composition attacks**, Ganta, Kasiviswanathan, Smith, KDD 2008, pp. 265–273 (arXiv 0803.0032) [P] | An attack paper. Independent anonymised releases about overlapping populations are combined (intersection) to breach partition-based schemes | The attacked objects are **partition-based** releases: k-anonymity and its variants | — | — |
| R1 | **This repository's closed "latent bottlenecks / codes" lane** (Adult) [R]: `results/pcrl_mechanism_design_v1/CLOSED_LANES.md` L27–31 | Discrete codes with task heads: decision-preserving codes, a capacity ladder to the i8o64 code, privacy-trained codes, 10-bit codes with MMD repair | **Many-to-one: yes** (finite codes) | Neural training, with privacy terms on some arms | Heads on the codes |

### 1B. Sensitive access, utility constraint, release, guarantee, multiple recipients

| ID | Sensitive-label access | Utility constraint (penalty or hard; metric) | Release contract | Guarantee: object and reader class | Multiple recipients or collusion |
|---|---|---|---|---|---|
| F1 | **Training:** S enters Gini_s when growing the tree. **Certificate:** S is needed on held-out D_val and D_test, with base rates from D_train. **Deployment:** encode reads x only (`T.apply`) | **Scalarised inside the split criterion** (γ), plus capacity limits k̄ and n_i. The utility metric is Gini_y, an accuracy proxy. No hard utility constraint | **The cell representation z** (median vector) | With probability ≥ 1 − ε (ε = 0.05) over the sampling of D_train, D_val and D_test, and **for the fixed encoder**: sup over g of the DP distance ≤ T\*, for **any downstream classifier that reads z alone** (Def. 4.1; Lemmas 5.1–5.3, Clopper–Pearson plus Hoeffding). Equivalently, 2·BA(h\*) − 1 ≤ T\* and d_TV(Z₀, Z₁) ≤ T\*. **Not** an AUC bound (TV 0.5 is compatible with AUC 0.875 [D, repo note]) and not an MI bound. The code asserts that every cell appears in every split | **Transfer:** one representation reused for other tasks (§7, Health). The certificate covers any consumer of z, so it covers **any coalition whose views are functions of the same z** [D]. No collusion model is stated |
| F2 | S on MODEL_FIT for tree growth only. Heads use task labels only. Deployment reads the six fields | None inside the method. Selection applied the 24-clause caps afterwards | **Each recipient's head probability vector plus argmax**, computed from its own task's cells. Cells are not released | No certificate was reported in the ACS replication files read | Per-task trees, so the coalition reads a function of the **product partition** (≤ 100 × 100 cells). A FARE certificate for the pair would have to be computed on the product partition, which is looser [D] |
| K1 | Training: IGS and relabelling counts use B. Whether B is also a tree input was not checked | **Hard budget on discrimination** (ε) with accuracy maximised (RELAB). The split criteria are scalarised | Class labels | None on disclosure. A knapsack equivalence, plus a greedy bound in the cited technical report (not opened) | None |
| K2 | Protected features enter DTDI and DIDI. Disparate treatment concerns their use as inputs | **Penalty** λ on DIDI or DTDI. Utility metric: misclassification or regression loss | Decisions or scores | MIP optimality for the regularised training objective. Nothing on disclosure | None |
| B1 | Training only (adversary labels) | **Penalty** λ; log loss | Score or probability | A remark only: if a perfectly fair and perfectly accurate pair exists, it is a global optimum of the min-max | None |
| B2 | Training only | **Hard constraints** on group rates, with slack ε. The objective is the loss | Classifier score | Approximate feasibility and optimality for the randomised classifier, inherited from Cotter et al. Nothing for the deployed last iterate | None |
| L1 | Training (L_z). **Deployment too, if group-specific α⁺ and α⁻ are used** | **Penalty** weights A_x, A_y, A_z | The representation (prototype memberships) and its predictions | None formal. The paper argues that small L_z implies small MI | One representation for a "vendor"; a transfer experiment. No collusion |
| C1 | **D is retained, and the test-time map reads D** (p_{X̂\|X,D}). A suppressed-D variant is discussed (§3.2) | **Hard** discrimination and distortion constraints. Utility (distribution closeness Δ) is the objective | Transformed dataset or features | Optimality for the given law, plus degradation bounds when the law is estimated (§3.2) | None |
| P1 | **Known P_{S,X}** at design time. The map reads X only | **Hard constraint** I(X; Y) ≥ R | Merged symbol | Thm 1: for any bounded cost, the inference gain is at most 2√2·L·√(D(P_{S\|Y=y} ‖ P_S)). MI-based, for a Bayes adversary who knows the law | None |
| P2 | None | Cluster count, or a target I(Z; Y) | Cluster index | None | None |
| P3 | Known joint of (X, Y⁻). **With Y⁻ := S, this is a task-aware, Lagrangian privacy funnel** [D] | **Lagrangian** (β, γ) | Cluster | None | None |
| P4 | Lifts require P_{S\|X}, i.e. a known or given law | **Hard privacy** bound ε. Utility I(X; Y) is the objective | Merged symbols, plus low-risk symbols as-is | Bounded lift implies bounded MI, Sibson MI, α-leakage and LDP (the paper's statement, citing [2]). Pointwise, under the law | None |
| P5 | Colours are used to form the clusters | **Hard balance** constraint. The objective is clustering cost | Cluster assignment of the given points | Approximation ratios | None |
| S1 | — (protects the data values) | Noise level per trust level | Perturbed data copies | Under linear least-squares estimation, colluders gain nothing over the best single copy | **Designed for collusion** ("diversity attack") |
| S2 | Protects the whole record X under a **known pmf** | Utility is the objective; MI privacy constraints are hard | Randomised responses | MI constraints under the known law (all-attribute, Bayes) | **All-prefix collusion.** Sequential "as future requests are not known" |
| S3 | Sensitive values enter the privacy check | Information/anonymity trade-off scores for classification | Generalised purpose-specific table | (X, Y)-privacy by construction (syntactic) | **Designed against the join with an earlier release**; a legacy constraint |
| S4 | — | — | Anonymised tables | Randomised notions such as DP "provably resist composition attacks" | **The intersection threat itself.** Deterministic partitions released separately can be intersected |
| R1 | S at training | Caps checked at selection | Head outputs | None | The pair was trained and attacked. Reported costs: the finest decision-preserving code +0.020 nats occupation log loss; privacy-trained P\* −0.034 pair AUC with an occupation log-loss upper bound of 0.0121; 10-bit codes −0.034 with occupation NLL +0.032 |

---

## 2. FARE in this repository: why it was utility-infeasible on ACS

**What was run** [R].
- The tree is the official code: `fair_gini_dp`, commit `89cb1b6`, hash-pinned in `oar/fare_official.py` L60–74.
- Fit data: MODEL_FIT with task labels and SEX, one tree per task.
- Head and release are as in row F2.
- Selection used the frozen MODEL_SELECT rule (`SCIENCE_LOCK.json` L614–645).

**Result.** No setting was feasible.
- Every FARE configuration (F1–F3) failed **all 24** MODEL_SELECT clauses (`SELECTION_DETAIL.json` `family_tables/FARE`).
- F1 failed all 12 DEV_REPORT clauses per seed (`SERVICE_TABLE.csv`; `RESEARCH_DECISION.md` L116).

**MODEL_SELECT clause values against U0, seed 0 / seed 1** (`SELECTION_DETAIL.json` `per_seed`). Caps: accuracy loss 0.010, NLL excess 0.010, summed Brier excess 0.005.

| Setting | Income acc. loss | Income NLL exc. | Income Brier exc. | Occ. acc. loss | Occ. NLL exc. | Occ. Brier exc. | Proxy AUC |
|---|---|---|---|---|---|---|---|
| F1 (γ 0.3, k̄ 100) | 0.017 / 0.019 | 0.023 / 0.023 | 0.020 / 0.019 | 0.018 / 0.014 | 0.027 / 0.030 | 0.011 / 0.012 | 0.618 |
| **Z1 (γ 0, k̄ 100)** | 0.016 / 0.019 | 0.025 / 0.025 | 0.021 / 0.021 | 0.0096 / 0.0096 (within cap) | 0.019 / 0.023 | 0.0085 / 0.0095 | 0.631 |
| F2 (γ 0.7, k̄ 50) | 0.034 / 0.038 | 0.056 / 0.057 | 0.046 / 0.046 | 0.028 / 0.028 | 0.046 / 0.049 | 0.019 / 0.020 | 0.571 |
| Z2 (γ 0, k̄ 50) | 0.020 / 0.024 | 0.025 / 0.026 | 0.021 / 0.021 | 0.005 / 0.008 | 0.017 / 0.021 | 0.0068 / 0.0079 | 0.635 |
| F3 (γ 0.85, k̄ 20) | 0.043 / 0.047 | 0.075 / 0.076 | 0.062 / 0.062 | 0.034 / 0.034 | 0.060 / 0.064 | 0.026 / 0.027 | 0.542 |

**Reading.**
1. **Compression binds before fairness.** The no-fairness twins fail too.
   - Z1 is the plain Gini tree with 100 cells and a per-cell logistic head. It fails all 12 DEV_REPORT clauses per seed (`SERVICE_TABLE.csv`).
   - On MODEL_SELECT it fails every income clause and the occupation NLL and Brier clauses; only its occupation accuracy loss (0.0096) is within the cap.
   - Its income NLL excess is about 2.5× the cap, its income Brier excess about 4× the cap, and its occupation NLL excess about 2× the cap.
   - The repository's own review says the same: "most of the cost is shared with compression" (`CLAIMS_REVIEW.md` item 7).
   - F1's extra cost over Z1 is small on income and moderate on occupation.
2. **More fairness weight, fewer cells.** Higher γ with smaller k̄ (F2, F3) lowers the MODEL_SELECT proxy AUC (0.618 → 0.571 → 0.542), at several times the cap.
3. **DEV_REPORT, seed 0 / seed 1** (`SERVICE_TABLE.csv`; `RESEARCH_DECISION.md` L114–124):
   - F1 coalition 0.640 / 0.633, against Z1 0.667 / 0.666 and NEURAL_RAW 0.640 / 0.638.
   - F1 occupation recipient 0.596 / 0.599, against Z1 0.658 / 0.660.
   - F1 occupation accuracy 0.469 / 0.469, against U0 0.484 / 0.485.
4. **A structural observation for this campaign** (descriptive).
   - FARE's **income** recipient recovered SEX at 0.643 / 0.639, and Z1's at 0.646 / 0.646. U0's income output gave 0.560 / 0.567.
   - So a 100-cell income partition with per-cell probabilities revealed **more** SEX than U0's continuous income score.
   - One reading, an interpretation that was not tested: each cell has its own probability, so the released value identifies the cell, and the readers see the whole partition (§4.2(b)).
5. **Not tested here.** The paper's k̄ range extends to 200, and n_i down to 50 (App. E); the repository used k̄ ≤ 100 and n_i = 100. Whether finer trees reach feasibility is unknown. More cells also loosen the certificate (paper App. E; repo note §2).

**FAGTB on ACS (for completeness)** [R].
- No λ was feasible.
- At λ = 0.03 on MODEL_SELECT: income NLL excess 0.084 / 0.095 and occupation accuracy loss 0.059 / 0.060.
- On DEV_REPORT, all 12 clauses failed per seed; the accuracy loss was 1.1 / 2.8 pp (income) and 5.5 / 5.7 pp (occupation); coalition 0.648 / 0.656.
- "Underfit" (the 300 × 0.01 shrinkage cap) is an interpretation only, because all runs used λ > 0 (`CLAIMS_REVIEW.md` item 6; `FAGTB_AUDIT.md` §0, §8).
- FAGTB is not a structural method: it modifies a continuous boosted score.

---

## 3. Relation of FARE to the campaign's release contract

- **The certified object differs from what is released here.** FARE certifies the cells, and the official code releases cells (medians). This campaign releases each recipient's probability vector and argmax. A per-cell head makes each release a function of the cells, so FARE's certificate would cover a recipient's release *if* computed on that recipient's cells [D]. The pair is covered only through the product partition, or through a shared partition (§4.2(c)).
- **The certified quantity differs from the endpoint.** The certificate bounds the DP distance, TV distance and balanced accuracy of hard rules on z. It does not bound reader AUC, which is this campaign's endpoint. A finite-cell AUC can exceed (1 + T\*)/2.
- **Sample-size premise.** The certificate requires every cell to appear in the held-out splits, which is a binding sample-size premise at k̄ = 100 on ACS role sizes [R: `oar/fare_official.py` L550–560, from the official assertions].

---

## 4. What a distinct ingredient would have to look like

### 4.1 Combinations already covered

| A proposal that is essentially… | is already… | Closest rows |
|---|---|---|
| a supervised many-to-one **tree** partition grown with a task-impurity term minus or plus a **sensitive-composition** term, with per-cell outputs | FARE (Gini_y + γ(0.5 − Gini_s)); Kamiran et al. 2010 (IGC − IGS, IGC/IGS, IGC + IGS). FARE cites Kamiran for the criterion | F1, K1 |
| a FARE tree followed by **per-cell probability heads** fitted on task labels | FARE as run here (logistic head on one-hot cells); the tree's own leaf frequencies; LFR's per-prototype w_k | F2, L1 |
| **leaf-level edits** of labels under a discrimination budget | Kamiran relabelling (knapsack, greedy); EiFFFeL [prev] | K1 |
| **greedy agglomerative merging** of input symbols under a known or empirical law, trading I(Z; S) against I(X; Z) | the privacy funnel, Alg. 1 | P1 |
| the same with **task** information I(Z; Y) as utility and an S penalty | IBSI, with Y⁺ = task and Y⁻ = S, solved by AIB-style or sequential clustering. AIB alone if S is absent | P2, P3 |
| merging **only high-risk symbols** until each merged group meets a pointwise posterior-to-prior (lift) bound | the watchdog and watchdog subset merging | P4 |
| partitions with **balanced sensitive composition** per cell as a hard constraint | fairlets (unsupervised, transductive); FARE's Gini_s as the soft supervised analogue | P5, F1 |
| **soft prototypes** with per-prototype probabilities and a parity penalty | LFR | L1 |
| a **discrete code** of fixed capacity with heads, trained with or without a privacy term | this repository's closed latent-code lane | R1 |
| adversarial or constrained **boosting** that grows new trees | FAGTB, FairGBM | B1, B2 |
| a randomised discrete **pre-processing map** with discrimination and distortion constraints | Calmon et al. 2017 (reads D at test time, so it is outside this contract) | C1 |
| purpose-specific releases **coarsened** so that a join with another release is lossy | Wang and Fung 2006 (generalisation; previous release fixed) | S3 |
| quantising or merging the **outputs of a frozen reference** (U0 or teacher posteriors) | the closed "frozen-code rescue" lane (decoders, calibrators, mixtures on frozen outputs). Rounding confidences as a defence: Fredrikson et al. 2015 [prev] | — |

### 4.2 Elementary facts any per-cell probability release obeys [D]

These are textbook identities stated for the population law. U0 is not the Bayes posterior, so they guide reasoning but are not cap computations.

**(a) Merging has a utility price that is computable in advance.** Let C be the cell and the release be q(C).
- **NLL.** The population NLL of the per-cell conditional frequency P(Y | C) is H(Y | C). Its excess over the Bayes posterior p(X) is I(X; Y) − I(C; Y) = E KL(p(X) ‖ P(Y | C)). Merging two cells raises it by exactly the AIB merge cost, a prior-weighted Jensen–Shannon divergence (P2).
- **Brier.** For summed Brier, E‖e_Y − q(C)‖² = E‖e_Y − p(X)‖² + E‖p(X) − q(C)‖², because the cross term vanishes given X. The excess is therefore the within-cell variance of the posterior.
- **Consequence for the caps.** The 0.010-nat NLL and 0.005 Brier caps act as merge budgets. On ACS, a 100-cell tree with per-cell heads already exceeded them (Z1, §2). A structural proposal must explain which merges are cheap under these budgets, and why.

**(b) A per-cell head with distinct outputs releases the cell.**
- If distinct cells receive distinct probability vectors, any reader can recover the cell from the release. SEX recoverability is then that of the partition, whatever the head.
- "Many-to-one relative to the raw inputs" is therefore not "less SEX information than the reference". The FARE income partition leaked more than U0's score (§2 item 4).
- A mechanism that lowers SEX information *below* what its partition carries must also merge in **output space**: different cells must receive identical released vectors.

**(c) Two recipients.**
- **Same partition.** If both releases are functions of one shared partition C, then AUC\*(S; V) ≤ AUC\*(S; C). If f₁ is injective on C, then f₂ is a function of R₁, and linkage adds nothing for Bayes readers.
- **Different partitions.** If the partitions differ, the coalition reads their intersection (the product partition). That is the classical composition and intersection weakness of deterministic partition-based releases (Ganta et al., S4).
- **Precedents for "pair no worse than one copy".** Li et al. (correlated noise, S1), Taylor's reuse of an earlier release, and DProvDB's single hidden synopsis [prev] all achieve it through shared randomness or reuse. A deterministic shared-partition design that achieves it by construction was **not found in the sources read**. That is a bounded-search observation; and the bound is only as good as the shared partition.

### 4.3 What could count as distinct, and what would not

**Would not count as distinct (renaming or engineering of covered rows):**
- FARE with another k̄, n_i or γ, another split ordering, or a better head or calibrator on its cells.
- Kamiran-style criteria swapped into FARE.
- A privacy funnel or IBSI run on empirical cells with a different trade-off parameter.
- A discrete code with a different capacity (closed lane R1).
- Quantising frozen U0 or teacher outputs (closed frozen-code lane).
- Any partition whose merges are driven by an explicit SEX-propensity score at deployment. That is close to the declined or unvalidated lanes (MMP, RCCS) in `CORRECTIONS.md`, and runs against the purpose-limitation framing.

**Not found in the sources read.** These could be argued as new *combinations*. Each still needs the evidence in `PROBLEM_SPECIFICATION.md` §6, claim 7, and none is a priority claim.
1. **Merge decisions under per-task hard caps.**
   - The merge decisions are taken under the benchmark's **per-task hard constraints**: NLL, Brier and accuracy against U0 and Ucal, for two tasks.
   - The objective is a pair-level SEX criterion.
   - Prior merge algorithms use one hard constraint on I(X; Y) (P1, P4), a single Lagrangian (P3), or a knapsack on decisions (K1).
   - This is an algorithmic variant of constrained greedy merging, i.e. engineering unless the constraint handling is itself new.
2. **A shared two-task partition with per-task heads and a coalition criterion**, using (c).
   - FARE's official tree uses one label (the 0th) and reuses its cells for other tasks only by transfer.
   - Multi-output trees in general were not reviewed today.
3. **Output-space ties under the caps.**
   - Some cells are assigned identical released vectors (merging in output space, (b)), chosen under the caps.
   - This differs from rounding a frozen model's outputs only if the partition and heads are learned jointly rather than post hoc. Otherwise it falls in the closed frozen-code lane.
4. **Generalisation to unseen inputs.**
   - P1–P4 define maps only on a finite known alphabet; FARE and LFR use parametric partitions.
   - On these six fields, most field combinations are unseen, so a proposal must say how its merges apply to unseen inputs. If the answer is "a tree", it is FARE-like.

### 4.4 Contract note for the coordinator

`PROBLEM_SPECIFICATION.md` §7 lists "coarsened outputs" as an example of a contract change. A deterministic many-to-one release whose vectors stay in the simplex, and whose argmax is computed from them, arguably remains in-contract: it transmits only f_i(x) and d_i(x). The coordinator should rule explicitly whether a release taking few distinct probability values counts as "coarsened outputs". The answer decides which comparators must be re-evaluated (§7).

---

## 5. Gaps

**Not opened today:**
- the Hsu–Asoodeh–Calmon ISIT 2019 author PDF (HTTP 403);
- Kamiran's cited technical report with the greedy bound;
- EiFFFeL;
- Fredrikson 2015;
- the DProvDB and FairGBM code (taken from the earlier reviewer, [prev]).

**Not searched exhaustively:** fair quantisation or binning of scores; multi-output fair trees; vector-quantised (discrete-bottleneck) fairness methods; the k-anonymity generalisation literature beyond Wang and Fung.

---

## 6. Sources opened today

**FARE**
- PMLR landing page: https://proceedings.mlr.press/v202/jovanovic23a.html (no code link on that page).
- Paper PDF: https://proceedings.mlr.press/v202/jovanovic23a/jovanovic23a.pdf
- Official code at commit `89cb1b66ed268c16659cbf7428c43e60da2df641`:
  - `code/src/tree/main.py`: https://raw.githubusercontent.com/eth-sri/fare/89cb1b66ed268c16659cbf7428c43e60da2df641/code/src/tree/main.py
  - `code/src/tree/alphabeta_adversary.py`: https://raw.githubusercontent.com/eth-sri/fare/89cb1b66ed268c16659cbf7428c43e60da2df641/code/src/tree/alphabeta_adversary.py
  - `sktree/_criterion.pyx` (FairGiniDP L831–851): https://raw.githubusercontent.com/eth-sri/fare/89cb1b66ed268c16659cbf7428c43e60da2df641/sktree/_criterion.pyx
  - `README.md`: https://raw.githubusercontent.com/eth-sri/fare/89cb1b66ed268c16659cbf7428c43e60da2df641/README.md

**FAGTB**
- Paper: https://arxiv.org/pdf/1911.05369
- Official code: https://api.github.com/repos/vincent-grari/FAGTB/contents (repository listing), https://raw.githubusercontent.com/vincent-grari/FAGTB/master/functions.py and https://raw.githubusercontent.com/vincent-grari/FAGTB/master/README.md

**Privacy funnel, information bottleneck and watchdog**
- Privacy funnel: https://arxiv.org/pdf/1402.1774
- Agglomerative IB: https://proceedings.neurips.cc/paper_files/paper/1999/file/be3e9d3f7d70537357c67bb3f4086846-Paper.pdf
- IB with side information: https://papers.neurips.cc/paper_files/paper/2002/file/04048aeca2c0f5d84639358008ed2ae7-Paper.pdf (equation (1) read on the rendered page)
- Watchdog subset merging: https://arxiv.org/abs/2110.04724 and https://arxiv.org/pdf/2110.04724
- Base watchdog (attempted, HTTP 403): https://scholar.harvard.edu/sites/scholar.harvard.edu/files/shahab/files/isit19.pdf

**Fair representations, trees and boosting**
- LFR: https://proceedings.mlr.press/v28/zemel13.pdf
- Optimized pre-processing: https://arxiv.org/pdf/1704.03354
- Kamiran et al. 2010: https://mpechen.win.tue.nl/publications/pubs/KamiranICDM2010.pdf (reached via win.tue.nl/~mpechen/publications/pubs/KamiranICDM2010.pdf); record page https://research.tue.nl/en/publications/discrimination-aware-decision-tree-learning/
- Aghaei et al. 2019: https://arxiv.org/pdf/1903.10598
- FairGBM: https://arxiv.org/pdf/2209.07850
- Fairlets: https://arxiv.org/pdf/1802.05733

**Collusion and sequential releases**
- Li et al. 2012: https://arxiv.org/pdf/1104.0459
- Taylor et al. 2026: https://arxiv.org/pdf/2601.21859
- Wang and Fung 2006: https://www.dmas.lab.mcgill.ca/fung/pub/WF06kdd.pdf
- Ganta et al. 2008: https://arxiv.org/pdf/0803.0032

**Repository files read**
- Code: `acs_replication/fare_worker.py`, `acs_replication/fagtb.py` (header), `oar/fare_official.py`
- FARE method notes: `results/combined_output_aware_removal_v1/notes/fare/FARE_METHOD_NOTES.md`
- ACS replication results in `results/pcrl_acs_protected_tree_replication_v1/`: `SERVICE_TABLE.csv`, `SELECTION_DETAIL.json`, `SCIENCE_LOCK.json` (FARE and FAGTB blocks), `MODEL_MANIFEST.json`, `RESEARCH_DECISION.md`, `CLAIMS_REVIEW.md`, `baselines/FAGTB_AUDIT.md` (§0, §10)
- Campaign documents: `results/pcrl_mechanism_design_v1/CLOSED_LANES.md`, `PROBLEM_SPECIFICATION.md`, `CLOSEST_PRIOR_WORK.md`, and the earlier reviewer's `LITERATURE_BASE.md`
