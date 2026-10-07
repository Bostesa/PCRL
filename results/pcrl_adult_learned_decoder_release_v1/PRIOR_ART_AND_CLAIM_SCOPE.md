# Prior art and claim scope (Adult learned decoder and constrained release, lra)

**Owner and date.**
- Role F (claims and custody reviewer), 2026-10-07, written between 04:45Z and 05:30Z.
- This was after SOURCE_ADMISSION_LOCK and before CORRECTNESS_LOCK and SCIENCE_LOCK.
- No fixture algorithm, Adult fit or assessment value had been run or read for this file. Nothing was installed or
  cloned.

**What this file replaces.**
- It replaces the port's verbatim copy of the lcr scope file (`results/pcrl_learned_decoder_constrained_release_v1/
  PRIOR_ART_AND_CLAIM_SCOPE.md`, lcr tip 091afc2). That file in turn extends the cbp, qpc and dpc reviews, which remain
  the detailed readings of AIB, multivariate IB, PURIFIER and Stadler et al.
- Three things are new here:
  - the two primary sources named in prompt §16 were re-read today, and the sections relied on are cited below;
  - the label wording now follows this study's registration: the correctness-only gate, the status suffix and CLASS|D1
    in T\*;
  - the decision-floor consequence for claim A is added (§5.8).

## 1. Plain statement

**These five ideas are established. None is ours, and combining or tuning them is not automatically algorithmic
novelty.**

| Idea | Where it is already established (§2) | Where it appears in lra |
|---|---|---|
| **Hard utility constraints** in privacy-mapping design | Privacy Funnel §II.C–§II.D: a distortion constraint E[d(X,Y)] ≤ D, rewritten in §III.B as I(X;Y) ≥ R. Taylor et al. §II: hard per-party and collusion constraints. du Pin Calmon and Fawaz (2012), the framework the Privacy Funnel builds on | The fitting budgets L_i ≤ L_i(U)+0.005 and B_i ≤ B_i(U)+0.003, and the local caps I_i ≤ I_i(C-TASK) |
| **Privacy-funnel objectives**: minimise I(S; release) for an attribute the deployed map does not read | Privacy Funnel eq. (3) and Algorithm 1 | Φ = I12 + 0.5(I1+I2), and each I_i in constrained LOCAL |
| **Proper-loss calibration**: fit a released probability from labels, shrunk toward a prior | Proper scoring rules (Brier 1950; Gneiting and Raftery 2007). Histogram-binning calibration (Zadrozny and Elkan 2001). Additive and Dirichlet pseudo-count smoothing | The per-token D1 decoder: log loss + 0.5·Brier + κ·KL(p̄_t ‖ q), κ = 32 |
| **Greedy partition search** | AIB merging (Slonim and Tishby 1999). Privacy Funnel Algorithm 1. Sequential-IB draw-and-reassign moves (Slonim, Friedman and Tishby 2002). Multivariate-IB joint greedy steps (2001) | Whole-fine-cell moves to same-class tokens; the best strictly improving feasible move; at most five sweeps |
| **Sequential collusion accounting** | Taylor, Vippathalla and Coon (2026): each release constrained alone and jointly with all earlier releases | SEQ-12 and SEQ-21: stage one optimises Φ against the partner's CLASS-ONLY view; the final release enforces both recipients' constraints |

**Other ingredients that are also not ours:**
- class-restricted clustering of probability vectors;
- decision containment and the data-processing inequality, including the decision floor of §5.8;
- penalty (Lagrangian) sweeps of a privacy weight, as in the D weighted controls;
- local search with compound (paired) moves;
- caching by sufficient statistics;
- argmax-preserving recalibration.

**What remains to evaluate.**
- The study tests one registered combination: a learned class-preserving decoder plus budget-constrained reassignment of
  label-blind fine cells, for two fixed recipients, on the reused Adult development benchmark.
- Its value is decided by claims A, B and C against the matched controls.
- Any statement of an algorithmic contribution needs both:
  - claim B passing (and, for the paired component, claim C); and
  - a separate novelty review against the literature in §2.
- The correctness gate, the fitting objectives and any fitting-row dominance cannot establish it.

**No novelty claim is made by this file.** A favourable result is an exploratory development result about this contract,
this reused Adult benchmark and this attacker slate.

## 2. Sources

### 2.1 Re-read today (primary sources named in prompt §16)

**Makhdoumi, Salamatian, Fawaz and Médard, *From the Information Bottleneck to the Privacy Funnel*, arXiv:1402.1774
v5 (30 Sep 2014; cs.IT; IEEE ITW 2014, venue verified in the lcr review).**
- Read today: the full five-page PDF from https://arxiv.org/abs/1402.1774.

Sections that bear on lra:
- **§II.A, setup.**
  - The user releases Y through a privacy mapping P_{Y|X}: "Throughout the paper, we assume S → X → Y form a Markov
    chain." The joint P_{S,X} is treated as given.
  - All variables are discrete ("All random variables are assumed to be discrete, unless mentioned otherwise").
  - **lra:** SEX is not a deployment input of any map, so the deployed release is a function of the permitted inputs.
    But SEX does shape the fitted partition of every privacy arm, and the joint is a plug-in fitting-row estimate, not a
    known law.
- **§II.B, privacy metric.** The inference-cost gain ΔC = c₀* − E[c_Y*] of an adversary who picks a belief to minimise
  an expected cost.
- **§II.C–§II.D, accuracy and the trade-off.**
  - The average distortion E[d(X,Y)] ≤ D "is linear in P_{Y|X}".
  - "If ΔC is convex in P_{Y|X}, then optimization (1) is a convex optimization."
  - This convexity is over **randomised** mappings, and only for a distortion that does not depend on P_{Y|X}.
- **§III.A, Lemma 1 and Theorem 1.**
  - Under log-loss cost, ΔC = I(S;Y).
  - For any cost bounded by L, ΔC ≤ 2√2·L·√I(S;Y).
  - This bounds a Bayes adversary's expected cost gain under the known law. It is **not** an AUC bound, and it says
    nothing about fitted plug-in MI on a sample or about a finite attacker slate.
- **§III.B, utility under log-loss.**
  - With d(x,y) = −log P(X=x|Y=y), the constraint becomes I(X;Y) ≥ R.
  - "It should be noted that the average distortion under the log-loss is not linear in P_{Y|X}."
- **§III.C, eq. (3), the Privacy Funnel: min I(S;Y) subject to I(X;Y) ≥ R.**
  - I(S;Y) is convex in P_{Y|X}.
  - "However, because of the constraint I(X;Y) ≥ R, the Privacy Funnel (3) is not a convex optimization."
  - The non-convexity is already present over randomised maps, before any restriction to deterministic partitions.
- **§IV.A, Algorithm 1 and Proposition 1.**
  - Greedy merging of output symbols from the identity map, while I(X;Y) ≥ R holds.
  - Proposition 1 gives each merge's cost as a weighted entropy difference.
  - "The greedy algorithm is locally optimal at every step … However, there is no guarantee that such a greedy algorithm
    induces a global optimal privacy mapping."
  - Algorithm 2, a greedy maximiser, brackets the achievable range (Note 1, Fig. 1).
- **§IV.B, data.**
  - The US 1994 census (Adult), with S = (age, income level) and X = (age, gender, education level).
  - Gender is therefore **released** there, not protected. That is the opposite role to lra's SEX.

**What it establishes for lra:**
- hard-constrained minimisation of leakage about an attribute the deployed map does not read;
- greedy deterministic partition search for it, with no global-optimality guarantee;
- the IB/Lagrangian connection;
- non-convexity of the constrained problem.

**Taylor, Vippathalla and Coon, *Adaptive Privacy of Sequential Data Releases Under Collusion*, arXiv:2601.21859 v2
(v1 29 Jan 2026 15:29:38 UTC; v2 10 Jul 2026 09:56:05 UTC; cs.IT).**
- Read today: https://arxiv.org/html/2601.21859v2 (sections I–IX) and the abstract page.
- The HTML equations render imperfectly. Numeric details are quoted only where the text states them.

Sections that bear on lra:
- **§II, problem setup.**
  - Party k sends request R_k and receives R̂_k, produced by a **randomised** conditional distribution
    p(r̂_k | r̂^{k−1}, x).
  - The private variable is the database X itself.
  - Per-party constraint I(R̂_k; X) ≤ ε_k; collusion constraint I(R̂_k, R̂^{k−1}; X) ≤ δ_k.
  - Utility is either −E[d(R̂_k, R_k)] or I(R̂_k; R_k).
  - "She does not have knowledge of future requests at the time of a data release, the data handler cannot jointly
    optimise all releases."
- **§III-A, general optimisation problem.** The Lagrangian dual: maximise over μ₁, μ₂ ≥ 0 the minimum over the channel
  of −U + μ₁(I(R̂;X) − ε) + μ₂(I(R̂,Z;X) − δ), with Z the earlier releases.
- **§III-B, solution overview; §IV, Blahut–Arimoto-style algorithm.**
  - The expected-distortion problem is convex, and its algorithm converges to the global minimum.
  - For mutual-information utility, "the primal problem is not convex". The algorithm (§IV-E) reaches only a local
    minimum and needs multiple random initialisations.
- **§V-B.** Time-sharing: a random convex combination of conditional distributions between solutions.
- **§VI, adaptive privacy scheme.**
  - Each release depends on X, its request and all earlier releases.
  - A targeted bisection maps the (ε, δ) targets to Lagrange multipliers.
- **§VII, experiments.**
  - UCI Adult, with education, income and age discretised into 4, 2 and 4 bins.
  - Hamming distortion.
  - Baselines: a non-adaptive mechanism (no conditioning on history) and a symmetric channel.
- **§VIII, connections to machine learning.** Progressive neural networks with an IB-style objective.
- **What the paper does not contain.** No deterministic quantisation maps, no released model outputs (probabilities or
  decisions), and no attacker inferring an attribute S distinct from the database X.

**What it establishes for lra:**
- sequential releases with explicit per-party plus collusion accounting;
- constraint-form (hard) leakage budgets;
- penalised/dual solutions;
- a non-adaptive baseline, which is the analogue of our LOCAL.

### 2.2 Carried over from the lcr, cbp, qpc and dpc reviews (not re-read today)

| Source | What it covers here |
|---|---|
| Slonim and Tishby, *Agglomerative Information Bottleneck*, NIPS 12 (1999) | Greedy merging with closed-form merge costs |
| Slonim, Friedman and Tishby, *Multivariate Information Bottleneck*, NIPS 14 (2001) | Joint greedy steps over several compression variables (the JOINT neighbourhood structure) |
| Slonim, Friedman and Tishby, *Unsupervised document classification using sequential information maximization*, SIGIR 2002 | Draw-and-reassign moves: the nearest published form of lra's single whole-fine-cell moves |
| du Pin Calmon and Fawaz, *Privacy against statistical inference*, Allerton 2012 (Privacy Funnel ref. [1]) | Distortion-constrained privacy mappings; the inference-cost-gain metric |
| Yamamoto, IEEE Trans. IT 29(6), 1983 (Privacy Funnel ref. [4]) | Foundational finite-alphabet utility–equivocation trade-off |
| Brier (1950); Gneiting and Raftery, *JASA* 102 (2007) | Strictly proper scoring rules; the calibrated-teacher null |
| Zadrozny and Elkan, ICML 2001 | Histogram-binning calibration (D1 is a regularised, class-dominance-constrained per-token version) |
| Guo et al., ICML 2017 | Argmax-preserving recalibration (temperature scaling) |
| Miller–Madow; Paninski, *Neural Computation* 15 (2003) | Plug-in MI bias |
| Yang et al., *PURIFIER*, AAAI-23 | Transformation of released confidence scores; a membership target, not class-preserving |
| Stadler, Kulynych, Gastpar, Papernot and Troncoso, ICML 2024 | No least-privilege claim; decision containment is the floor |
| Shokri et al., IEEE S&P 2017 | Coarsening the confidence vector as a defence |
| Taylor, Vippathalla and Coon, arXiv:2604.08630 | DP stopping filters; not a baseline |
| arXiv:2403.04778 (DC PF solver), arXiv:2405.00616 (EM-relaxed PF), arXiv:2404.02696 (deep PF) | Titles only. PF optimisation is an active line; all of them optimise randomised or continuous mappings; none is a baseline |

## 3. Code availability

| Method | Official code | Status |
|---|---|---|
| Privacy Funnel (Makhdoumi et al. 2014) | none | lcr found none on 2026-10-06 23:58Z to 2026-10-07 00:03Z (arXiv page, Zenodo record 1274716, GitHub and web searches). Today's PDF re-read shows no code reference. Not re-searched on GitHub today |
| Taylor, Vippathalla and Coon (2601.21859) | none | Today: the abstract page lists no code, data, comments or journal-ref field, and the v2 HTML mentions no repository. The lcr GitHub/web searches (0 hits) are carried over |

**Consequence.**
- Neither method can be run as an official baseline.
- lra's constrained LOCAL, K-SEQ-12, K-SEQ-21 and the weighted W-SEQ arms are **matched adaptations**.
- "The Privacy Funnel algorithm" or "the Taylor solver" must never be written for them.
- Claims A, B and C do not depend on either official method.

## 4. How the lra arms map onto prior work

| Arm or ingredient | Nearest prior procedure | What differs here | Status |
|---|---|---|---|
| D1 decoder (every new arm, every fixed-map D1 control, CLASS\|D1) | Histogram-binning calibration with pseudo-count shrinkage; proper-loss fitting | Per token: log loss + 0.5·Brier + κ·KL(p̄_t‖q), κ = 32; class-dominance constraint; affine ε-smoothing; certified released vector | Standard calibration under a constraint. **Changes no token, so removes no information** (§5.1) |
| B controls: D1 on the exact admitted D0 maps (DIRECT-TASK, FINE-TASK, 24 privacy maps) | Recalibration of an existing code | Assignments unchanged; privacy-trained maps stay **privacy-trained** | Calibration-only controls |
| CLASS\|D1 (registered addition, R-1) | Recalibrating a decision-only release | The tokens are the decisions | Privacy-untrained, in T\*; sits at the decision floor (§5.8) |
| d0s__ same-map D0 diagnostics | Mean-teacher (D0) decoding of a new map's unchanged tokens | Tokens bitwise equal to the D1 release | Decoder-only diagnostics; never candidates |
| C-TASK | Supervised sequential-IB-style reassignment under task loss | Class-restricted whole-fine-cell moves; D1 re-solved per move; FINE-TASK start; no SEX | Standard local search, supervised. **Privacy-untrained** |
| Constrained LOCAL (K-LOCAL) | Privacy Funnel: a hard utility constraint, min I(S;Y) | Reassignment moves, not merges; true-label log-loss and Brier budgets of a learned decoder instead of I(X;Y) ≥ R; local cap I_i ≤ I_i(C-TASK); class preservation; plug-in law | Matched adaptation, **not the Privacy Funnel algorithm** |
| K-SEQ-12 / K-SEQ-21 | Taylor et al.'s sequential collusion accounting; their non-adaptive baseline corresponds to LOCAL | Listed below | Matched adaptation, **not the Taylor solver** |
| K-JOINT-SINGLE | Multivariate-IB-style joint objective with single-variable moves | Φ is leakage to reduce, under both recipients' budgets and caps | Matched adaptation |
| K-JOINT-PAIR | Local search with compound (2-exchange-type) moves | Atomic paired moves from a fixed 8×8 bank per recipient; an equal TOTAL proposal-evaluation ceiling with JOINT-SINGLE and both sequential orders | The component tested by claim C. Larger neighbourhoods are a standard local-search device |
| D weighted controls (W-LOCAL, W-SEQ-12, W-SEQ-21, W-JOINT × 6 λ) | Lagrangian / penalty sweep | True-label task objective T with D1; λ grid fixed in advance | Design, not method |
| Fitting budgets 0.005 nats / 0.003 Brier | Hard utility constraints (Privacy Funnel; Taylor et al.) | A registered design choice applied to accepted moves on fitting rows | Design, not a confidence certificate |

**What K-SEQ-12 and K-SEQ-21 (and W-SEQ) change relative to Taylor et al.:**
- deterministic tokens, where they use randomised channels;
- leakage about S, not about the database X;
- greedy single-move search, not Blahut–Arimoto dual ascent;
- **the roles of objective and constraint are swapped.** We minimise the leakage Φ under utility budgets and local
  caps; they maximise utility under leakage constraints ε_k, δ_k. The two correspond only through a Lagrangian, and no
  equivalence is claimed;
- stage one is fitted against the other recipient's CLASS-ONLY view, which the contract always discloses. The temporary
  partner is not required to be feasible;
- the order and both requests are fixed in advance, whereas in Taylor et al. the later request is unknown;
- the arms inherit none of their convergence or optimality results.

**They must never be called** "the Taylor solver", "a Taylor baseline" or "an official sequential baseline". "Beats
Taylor" must not be written.

## 5. Mandatory caveats (carried into every decision document)

### 5.1 The decoder caveat

**A public deterministic decoder of an unchanged token removes no information.**
- For a class-preserving code, recipient i's complete interface is R_i = (t_i, q_i, d_i), with q_i = h_i(t_i) and
  d_i = class(t_i).
- R_i is a deterministic function of t_i, and t_i is part of R_i. Hence I(S; R_i) = I(S; t_i) and
  I(S; R_1, R_2) = I(S; t_1, t_2) for every choice of h_1 and h_2.
- Replacing D0 by D1 on the same tokens leaves the Bayes-optimal complete-interface ROC unchanged.

Consequences:
- A finite attacker's AUC may move after re-decoding. That movement is reader behaviour, not information removal, and it
  is never credited as privacy.
- Privacy credit requires a **changed partition**.
- The identical-token diagnostics (DECODER_ONLY_ABLATION.csv, DECODER_UTILITY_ABLATION.csv, the d0s__ pairs) must show
  bitwise-identical tokens and identical plug-in MI, with full-token and probability-only recovery side by side.
- A D1 gain on a fixed map is a **utility (calibration) gain**. It can make an existing private partition usable.
- If claim A passes with a calibrated construction, the protection came from the existing cbp partition, and the decoder
  made it usable.

### 5.2 Supervised label use on OSF_DEFENSE_FIT (disclose plainly)

**The material change.**
- The qpc and cbp codebooks never read a **true task label**:
  - their task-only maps read no label at all;
  - their privacy maps read SEX on OSF_DEFENSE_FIT (cbp METHOD_CARD: "True task labels never enter a fit").
- This study reads the **true task labels** of OSF_DEFENSE_FIT (15,434 rows) for:
  - every D1 decoder, including the fixed-map calibration controls and CLASS|D1;
  - C-TASK;
  - the weighted controls' task objective;
  - the fitting budgets.
- It reads **SEX** there for:
  - the privacy objectives;
  - the weighted controls;
  - the local caps (I_i(C-TASK) is measured with SEX after a SEX-free search);
  - the fitted and permutation-null MI diagnostics.
- "Label-blind" therefore describes only the fixed fine partitions. It must not describe the source codebooks without
  that qualification.

**What stays label-blind or unchanged:**
- the fine partitions are fixed and label-blind;
- the D1 per-token objective uses no SEX;
- SEX is never a deployment input;
- no encoder, task head or feature extractor is fitted.

**Exposure of the fitting rows.**
- The released vectors q_t depend on fitting-row label counts y_t, and privacy partitions depend on fitting-row SEX.
- No membership or attribute guarantee is given for the fitting rows.
- Maps, decoders and attackers stay private.

**The teachers' training rows.**
- The U encoders and deployed heads were fitted on OSF_DEFENSE_FIT under the osf protocol.
- So D1 calibrates against in-sample teacher outputs, and L_i(U), B_i(U) in the budgets are in-sample losses.
- Any held-out benefit is an empirical inner/assessment question.

### 5.3 Not convex overall

- **For a fixed token, the D1 objective is convex** in u over a convex polytope:
  - q is affine in u;
  - −log q is convex;
  - the Brier term is quadratic;
  - KL(p̄_t‖q) = const − Σ p̄ log q;
  - the simplex intersected with {u[d] ≥ u[k]} is convex.
- The fixed-token certificate (KKT and stationarity residuals, projection magnitude) covers only that token's continuous
  problem.
- **The discrete search is not convex.** Search over partitions, with decoders re-solved after each move, is
  combinatorial. The Privacy Funnel is already non-convex over randomised maps (§III.C), and Taylor et al.'s
  MI-utility problem is non-convex too (§III-B).
- Every mapper result is a **heuristic local optimum** of its registered search. No global optimum, optimality gap or
  "mathematically infeasible" verdict follows.
- A map that violates a budget under D1 is "infeasible under this registered decoder". D1 is regularised (κ = 32) and
  is not the unregularised Bayes rule.

### 5.4 Fitted MI is not a population guarantee

- **I_i, I12 and Φ are plug-in quantities** on OSF_DEFENSE_FIT, the rows used to fit the partition.
  - They are biased upward by alphabet size and downward by selection.
  - A fitted decrease, a value below a permutation null, or a satisfied local cap is a **fitting-law constraint**, not
    protection.
- **The fitting budgets are design choices.** 0.005 nats and 0.003 Brier are applied to accepted moves on fitting rows.
  - They are not population confidence certificates.
  - D1 is fitted on the same rows, so they are optimistic.
- Only the unchanged inner rules and the assessment bounds decide confidence preservation. A fitting-budget certificate
  does not imply a passing assessment clause.
- **Privacy Funnel Theorem 1** bounds a Bayes adversary's cost gain under a known law. It gives no AUC bound for any
  fitted attacker here.

### 5.5 Sequential arms are matched adaptations, not the official Taylor solver

See §4. This holds for the constrained arms (K-SEQ-12, K-SEQ-21) and the weighted controls (W-SEQ-12, W-SEQ-21).

### 5.6 Fixed-task output-release scope

- Per recipient, the release is a fixed-task **output**: a categorical token identity, its decoded probability vector
  and the unchanged teacher decision.
- It is not a reusable feature representation, and not privacy-preserving representation learning.
- It does not combine the encoder/LoRA line and the stochastic-release line into a pipeline.

### 5.7 Other limits

- **Exposure.** The registered statement is in EXPOSURE_LEDGER.md.
  - All roles have been used historically.
  - Nominal intervals ignore the adaptive history.
  - A masked assessment does not make the rows fresh.
  - Repeated nominal success on these rows is not confirmation.
- **Attacks.** The slate is a declared matched slate. Weak recovery by it is not a population or all-attacker
  certificate.
- **References.** RAW-J, FARE, F0 and LEACE operate under different contracts. Comparisons with them are descriptive.
- **Fitting-objective dominance** (for example, a JOINT arm against its witnesses) holds by construction on the fitting
  rows. It does not imply a held-out gain.
- **Adaptive selection.** The selection rule (original eligibility without cbp's headroom buffer) was chosen after cbp's
  headroom rule failed on these rows. That is adaptive, and it must be stated wherever an existing (D0) or calibrated
  map becomes P\*.
- **Equal work.** JOINT-PAIR is matched by a TOTAL proposal-evaluation ceiling, not by CPU.
  - A claim-C result must report each arm's actual evaluations, stop reasons and CPU.
  - A claim-C failure in which JOINT-PAIR stopped at its ceiling is a budget-limited result.

### 5.8 The decision floor (new in lra; PREDECESSOR_GATE_DIAGNOSIS.md)

- Every class-preserving complete-token release determines both decisions. So I(S; R) ≥ I(S; CLASS), for the
  population law and exactly for the fitting plug-in. The Bayes-optimal recovery ROC on R dominates CLASS's.
- CLASS (D0 and D1) is in the T\* pool (R-1).
- **If a CLASS variant is inner-eligible and is T\*,** claim A's clause 1 (pair-AUC lower bound > 0.02 below CLASS)
  can be met only through finite-reader behaviour on the registered slate. A pass must then be worded as a measured
  finite-slate difference, with the added sentence: "Every class-preserving release carries at least the decision-only
  release's information, so this difference reflects the registered attackers' behaviour, not information removed
  below the decisions."
- **If CLASS is ineligible,** the floor still bounds every release. Report CLASS's measured recovery and I12(CLASS)
  beside every privacy result (DECISION_FLOOR_AND_FEASIBILITY.csv), never as an MI guarantee derived from AUC.
- **Claim C.** The C_pair\* pool contains the privacy-untrained releases, so the same sentence applies to claim C
  whenever C_pair\* is a CLASS variant. If C_pair\* is a continuous reference, the floor orders nothing; say so.

## 6. Wording per label (prompt §12; LABEL_TRUTH_TABLE.json v2)

**Label format.**
- Every overall label carries the status suffix " [A=<status>; B=<status>; C=<status>; Q=<status>]"
  (`lra.family.status_suffix`).
- A headline quoted without the suffix (`label_headline`) must be followed in the same paragraph by all four statuses.
- No sentence may present the headline as hiding an incomplete claim.

### 6.1 Naming the construction of P\*

`LABEL_TRUTH_TABLE.winning_family_naming` names ONE real representative of P\*'s exact alias set. Within P\*'s own pool,
it takes the lowest construction rank (existing < calibrated < task-only < weighted < constrained), then the lowest
family rank, then the config ID. Every alias is listed separately.

| Construction | Config IDs | Required description | Required attribution sentence |
|---|---|---|---|
| **existing** | `U\|<FAM>\|i8o64\|l<λ>` (D0) | "the admitted cbp <FAM> λ <x> map with its original mean-teacher decoder (D0); nothing was refitted" | "The learned decoder and the new search added no demonstrated benefit. The map was nominated by this study's selection rule (original eligibility without cbp's headroom buffer), which was chosen after cbp; this is adaptive." |
| **calibrated** | `<D0 id>\|D1` | "the cbp <FAM> λ <x> partition, unchanged, with the learned decoder D1" | "The protection comes from the existing cbp partition; the learned decoder made it usable. No new assignment search contributed (a DECODER-ENABLED result)." |
| **weighted** | `U\|W-<FAM>\|i8o64\|l<λ>\|D1` | "a true-label weighted control (<FAM>, λ <x>) with the learned decoder" | "A weighted baseline produced the best usable protection; the hard-budget constrained search added no demonstrated benefit." |
| **constrained** | `U\|K-<ARM>\|i8o64\|D1` | "the registered constrained <ARM> search with the learned decoder" | "Claim A alone does not show that the constraints, rather than decoding or ordinary supervised search, caused the gain; that is claim B." |

**Further naming rules.**
- **identical to privacy-untrained.** A privacy-trained release may be byte-identical to a privacy-untrained one, for
  example a K-LOCAL or W-arm that accepted no move from C-TASK, or a release equal to CLASS. The label then carries
  "; identical to privacy-untrained <ids>". The prose says: "This release is identical to the privacy-untrained <ids>;
  no privacy training changed it, and no privacy credit is given to training."
- **Fixed-map D1 privacy releases are privacy-trained controls.** Never call them task-only or privacy-untrained
  because the decoder itself uses no SEX.
- **D1 is never an alias of D0.** A D1 release never aliases its D0 map, because the released vectors differ.
  Same-token, different-decoder pairs are reported as decoder-only pairs, not aliases.
- **Construction words.** P\* is never C-TASK or CLASS: both are privacy-untrained. "Changing the assignments" may be
  said only for a constrained or weighted construction.

### 6.2 Allowed and forbidden wording for every label

| Label | Allowed sentence (fill the brackets) | Must not be written |
|---|---|---|
| **PRIVACY_RELEASE_DEVELOPMENT_CRITERION_MET (<family>; <construction>[; identical to privacy-untrained …]) [A=PASS; …]** | "On the reused Adult development benchmark, <construction description of P\*> reduced combined-view SEX recovery by more than 0.02 pair AUC (registered lower bound) relative to the strongest eligible privacy-untrained release T\* (<T\* config>). Neither recipient's own AUC rose by 0.01 or more. Both tasks stayed within the original accuracy, log-loss and Brier allowances, and every decision was unchanged (exploratory, locked, single opening of reused rows; not confirmation)." Then the §6.1 attribution sentence; the §5.8 sentence when T\* is a CLASS variant; and B, C and Q statuses | "new method/algorithm", "our constrained search works" (unless B passes), "joint advantage" (unless C passes), "private", "certified", "chance-level", "information removed by the decoder", "the decoder protects", "task-only" for a calibrated or existing privacy map, "removes information below the decisions", "beats Taylor / the Privacy Funnel / PURIFIER", "privacy-preserving representation", any population or MI guarantee |
| **+ CONSTRAINED_SEARCH_INCREMENT_ESTABLISHED** | "The registered constrained search (N\* = <K-arm>, D1) beat the strongest eligible calibration and true-label weighted incumbent (C\* = <config>) by the registered margins on this reused benchmark and attacker slate. The incumbents used the same decoder, starts, caps and single-move neighbourhood. Novelty is reviewed separately." If A did not pass: "…but no release met the useful-release criterion against strong task-only compression (claim A <status>)." | "hard constraints beat penalties in general", "the constraints caused the gain" beyond this matched bank, "globally optimal", "a new algorithm" (pending the novelty review), "joint advantage" unless N\* is a joint arm AND C passes, "beats the Privacy Funnel / Taylor" |
| **+ PAIRED_JOINT_INCREMENT_ESTABLISHED** | "The paired-move joint search (J\* = K-JOINT-PAIR, D1) beat the strongest eligible non-pair release (C_pair\* = <config>) by the registered margins. C_pair\* ranged over constrained LOCAL, both sequential orders, JOINT-SINGLE, every incumbent control and the untrained and reference releases. JOINT-SINGLE and both sequential orders had the same total proposal-evaluation ceiling; actual evaluations, stop reasons and CPU were <numbers>. This is evidence for the paired component on this benchmark and slate." | "joint design is better than sequential (in general)", "coordination theorem", "universal superiority", "beats Taylor", "the joint objective causes the gain" (JOINT-SINGLE has the same objective; C isolates the paired neighbourhood), any claim that hides unequal actual work |
| **CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION [A=…; B=…; C=…; Q=PASS]** | "The fixed original D0 DIRECT-TASK i8o64 code (Q) kept both tasks' log loss and Brier within the original allowances on all four registered bounds. No privacy-method claim (A, B, C) passed: A <status>, B <status>, C <status>. Q repeats the qpc and cbp outcome on the same reused rows; it is not evidence for learned decoders." For an incomplete method claim: "…; claim <X> is incomplete because <root cause> (shown)." | "learned decoders improve confidence" (Q is D0), "privacy release works", "capacity impossibility", "confirmation", omitting or softening an incomplete method claim |
| **EXPERIMENTAL_NO_ADVANTAGE [all statuses]** | "The registered work was complete and valid, but no method claim (A, B, C) and no confidence bound set (Q) passed on this locked assessment. This closes this exact recipe (D1 with κ = 32, these budgets, neighbourhoods and λ grid) on these rows." Then each claim's failed clauses with their class (MEASURED_VIOLATION vs NOT_ESTABLISHED_PRECISION vs NOT_ESTABLISHED_POINT) | "learned decoders cannot help", "constrained search cannot work", "privacy release is impossible", "a universal privacy floor", "a competitive negative against Taylor / the Privacy Funnel", "a relaxed λ, budget or κ would pass", any recommendation of another small adjustment on the same assessment |
| **ENGINEERING_BLOCKED_NOT_RUN [A=NOT_RUN; B=NOT_RUN; C=NOT_RUN; Q=NOT_RUN]** | "The new correctness gate was ENGINEERING_BLOCKED at <check and cause> after bounded code-only repairs (<amendments>). No Adult decoder or mapping was fitted, so there is no Adult result in either direction. The engineering evidence, the oracle replay and the runnable implementation are delivered." | "the method fails on Adult", "negative result", "constrained search is useless", any Adult number presented as a test of the method, reusing MECHANISM_GATE_NOT_MET for this study |
| **INCOMPLETE_NOT_RUN (<concrete reason>) [all NOT_RUN]** | "The study stopped before any Adult fit because of <the pre-fit budget or admission blocker>. No Adult claim was run." | any Adult result, "too expensive to work", "inconclusive method" |
| **INCOMPLETE_OR_INVALID (overall) [statuses shown]** | "Adult science began but an unresolved global technical failure remains: <root cause>. <Claim statuses>. A separately valid Q result is reported beside it and does not replace this label." | any completed head-to-head positive or negative, "Q shows the method works", letting Q headline the study |
| **MECHANISM_GATE_NOT_MET (historical, lcr only)** | "The predecessor (lcr) closed with its fixture gate GATE_NOT_MET; that is a property of its fixture bank and trigger (PREDECESSOR_GATE_DIAGNOSIS.md), not an Adult result." | as this study's label or launch verdict; as evidence about Adult |

### 6.3 Claim statuses that appear beside a label

| Status | Allowed | Must not be written |
|---|---|---|
| PASS (claim) | Only with a valid nominee and comparator and every clause passing; use the §6.2 sentence for that claim | "passes" for a DESCRIPTIVE_ONLY fallback row, even if its numbers clear the targets |
| NOT_ESTABLISHED | "Claim <X> was not established: clause(s) <ids> failed (<class>)." A point inside the limit with the bound outside is "preservation not established", **not** a violation. A violation is stated only for MEASURED_VIOLATION | "the method is worse", "violates confidence" (unless MEASURED_VIOLATION) |
| NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE | "No <role> candidate passed the registered inner nomination (<ORDINARY_UTILITY_FAILURE / LOCAL_GUARD_FAILURE / CONSTRAINED_FIT_INFEASIBLE>). This refutes this nomination procedure for this role, not the idea in general. The descriptive fallback <config> is shown for information only." For CONSTRAINED_FIT_INFEASIBLE, add: "The constrained search found no release that met its fitting budgets under this registered decoder and search." | "a relaxed rule would pass", any claim from the fallback, "the constraints are mathematically infeasible" |
| INCOMPLETE_OR_INVALID (claim) | "Claim <X> is incomplete because <exact root cause: missing comparator / failed required control / nonfinite statistic / technical fit-record failure / unresolved verification failure>." | any completed positive or negative head-to-head statement for that claim; CONSTRAINED_FIT_INFEASIBLE wording for a technical fit-record failure |
| NOT_RUN | "Not run: <stage> did not run because <reason>." | any result |

### 6.4 The three opening answers (prompt §17)

1. **"Did learning the probabilities improve Adult confidence on the SAME codes?"**
   - Answer "Learning the probabilities [did / did not clearly] improve confidence on the same codes."
   - The rule is `LABEL_TRUTH_TABLE.report_rules.decoder_sentence`: the best inner-selected D1 fixed-map privacy control
     against its paired D0 release, with the paired per-task D1 − D0 log-loss and Brier contrasts on the same 1,999
     draws and the supplementary nominal 95% z.
   - Write "did" iff all four upper bounds are < 0.
   - Add: "The tokens were identical, so this is a calibration change, not information removal."
   - Q is D0 and cannot answer this question.
2. **"Did any useful private release meet the original full criterion?"**
   - Claim A's status, exactly, with its root cause or failed clauses.
   - If A passed, name the construction (§6.1). Fill "[Changing the assignments / The calibrated existing method / The
     existing cbp method, unchanged / A weighted baseline] produced the best usable protection."
   - If there is no P\*, write: "No release produced usable protection under the registered nomination."
3. **"Did new constrained or paired joint search add anything beyond the strongest controls?"**
   - Claims B and C, exactly.
   - "Yes" only for a PASS.
   - When P\* is a control (calibrated, existing or weighted), write that the new constrained search added no demonstrated
     benefit.

### 6.5 Scope sentences that must appear in the final answer (prompt §17)

- "A decoder-only utility change is not information removal."
- "A fixture result is not an Adult result."
- "An Adult development result is not confirmation."
- "A useful calibrated existing code is not a newly invented algorithm."
- "Fewer leaking finite attackers is not a population privacy guarantee."

### 6.6 Interpretation lines (prompt §17)

| Outcome | Allowed interpretation |
|---|---|
| A only | A useful measured privacy release on this reused development benchmark |
| B | Added value of this constrained search over the registered, equally decoded controls |
| C | Added value of this paired search over the registered alternatives |
| Q only | Confidence feasibility, no method criterion |
| Only a same-map D1 utility improvement | Calibration evidence, no privacy claim |
| A valid full loss | Close this exact decoder/constrained recipe. Do not prescribe another small λ, budget or κ adjustment on the same assessment |
| An unresolved technical or resource blocker | Incomplete work, not a scientific negative |

**Confirmation.**
- If a full development claim passes, produce a separately reviewable confirmation plan: exposure reconciliation, frozen
  candidate, power, and household/group conventions.
- Do not open that cohort.

## 7. Gaps that remain

| Gap | What remains open |
|---|---|
| G1. Official baselines | No runnable official Privacy Funnel or Taylor et al. code (§3). Results speak only to the registered arms and admitted references |
| G2. Randomised mechanisms | None in the bank. Randomised channels and time-sharing (Taylor et al. §V-B) reach trade-off points that deterministic maps cannot |
| G3. Plug-in quantities | Every I and Φ value is a fitting-row plug-in. The i8o64 pair alphabet makes the I12 null level comparable to the differences being optimised |
| G4. Decision floor | I(S; R) ≥ I(S; CLASS) for every class-preserving release (§5.8). Measured differences against CLASS are finite-reader differences |
| G5. In-sample calibration | D1 corrects the teacher's in-sample miscalibration on OSF_DEFENSE_FIT (§5.2) |
| G6. Compute matching | JOINT-PAIR is matched by a TOTAL evaluation ceiling, not by CPU. Actual work is reported (§5.7) |
| G7. Training-data exposure | Decoders hold fitting-row label statistics, and partitions hold SEX-dependent groupings. Both stay private; no fitting-row privacy guarantee is given |
| G8. Novelty | No genuinely new algorithmic step has been identified. A favourable B or C needs a separate novelty review before any contribution wording |

## 8. Summary

**Allowed.** "A class-preserving output code at fixed capacity (i8o64), whose released probabilities are fitted per token
from true task labels with a fixed teacher prior, and whose assignments of label-blind fine cells are searched under
explicit fitting budgets with standard greedy moves. It was evaluated once on a locked, reused Adult development
benchmark against calibration-only, task-only (including the decision-only release with both decoders), true-label
weighted, local, sequential and single-move joint controls and admitted continuous references."

**Not allowed:**
- novelty of hard utility constraints, privacy-funnel objectives, proper-loss calibration, pseudo-count shrinkage, greedy
  or paired-move partition search, penalty sweeps or sequential collusion accounting, or of their combination, before a
  passing B/C and a separate novelty review;
- crediting a decoder change on unchanged tokens as information removal;
- crediting a difference against CLASS as information removed below the decisions;
- calling the discrete program convex, or any mapper result optimal;
- any population, attribute-specific or training-data privacy guarantee from fitted MI, fitting budgets, held-out AUC or
  structural properties;
- "the Taylor solver", "a Taylor baseline", "the Privacy Funnel algorithm", "beats Taylor", "beats the Privacy Funnel"
  or "beats PURIFIER";
- reusable or general-purpose representation learning;
- describing the source codebooks as "label-blind" without saying that their privacy maps read SEX.
