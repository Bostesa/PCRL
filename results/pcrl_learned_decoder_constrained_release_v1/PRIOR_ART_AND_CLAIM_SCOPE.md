# Prior art and claim scope (learned decoders and confidence-constrained releases, lcr)

**Owner and date.** Role F (claims and custody reviewer). Written between 2026-10-06T23:56Z and 2026-10-07T00:15Z, and
revised at 00:27Z to match the resolved fairness-review items R-1, R-3 and R-4. All of this was before FIXTURE_LOCK and
SCIENCE_LOCK. No fixture algorithm, real-data fit or assessment value had been run or read.
Nothing was installed or cloned, and no Adult data was touched.

**What this file updates.** It extends the cbp scope file
(`results/pcrl_confidence_budgeted_privacy_v1/PRIOR_ART_AND_CLAIM_SCOPE.md`, cbp tip 7f3ec67). That file builds on the
qpc and dpc reviews (`results/pcrl_confidence_capacity_v1/PRIOR_ART_AND_BASELINE_GAPS.md` and
`results/pcrl_decision_preserving_compression_v1/PRIOR_ART_AND_BASELINE_GAPS.md`). Those files remain the detailed
readings of AIB, multivariate IB, PURIFIER and Stadler et al. Today the two primary sources named in prompt §16 were
re-read, and both code-availability facts were re-checked (section 3). Everything new in lcr is mapped onto prior work
in section 4.

## 1. Plain statement

**These five ideas are established. None is ours, and combining or tuning them is not automatically algorithmic
novelty:**

| Idea | Where it is already established (section 2) | Where it appears in lcr |
|---|---|---|
| **Hard utility constraints** in privacy-mapping design | Privacy Funnel §II.C–III.C: a distortion constraint E[d(X,Y)] ≤ D, rewritten as I(X;Y) ≥ R. Taylor et al. §II: hard per-party and collusion constraints. du Pin Calmon and Fawaz (2012): the constrained privacy-against-inference framework that the Privacy Funnel builds on | The fitting budgets L_i ≤ L_i(U)+0.005 and B_i ≤ B_i(U)+0.003, and the local caps I_i ≤ I_i(C-TASK) |
| **Privacy-funnel objectives**: minimise I(S; release) for an attribute the map never reads | Privacy Funnel eq. (3) and Algorithm 1 | Φ = I12 + 0.5(I1+I2), and each I_i in constrained LOCAL |
| **Proper-loss calibration**: fitting a released probability from labels, shrunk toward a prior | Proper scoring rules (Brier 1950; Gneiting and Raftery 2007). Histogram-binning calibration (Zadrozny and Elkan 2001). Additive and Dirichlet pseudo-count smoothing | The per-token D1 decoder: log loss + 0.5·Brier + κ·KL(p̄_t ‖ q), κ = 32 teacher pseudo-observations |
| **Greedy partition search** | AIB merging (Slonim and Tishby 1999). Privacy Funnel Algorithm 1. Sequential IB draw-and-reassign moves (Slonim, Friedman and Tishby 2002). Multivariate-IB joint greedy steps (2001) | Whole-fine-cell moves to same-class tokens; best strictly improving feasible move; at most five sweeps |
| **Sequential collusion accounting** | Taylor, Vippathalla and Coon (2026): each release is constrained on its own and jointly with every earlier release | SEQ-12 and SEQ-21: stage one optimises Φ against the partner's CLASS-ONLY view; the final release enforces both recipients' constraints |

**Other ingredients that are also not ours:**
- class-restricted clustering of probability vectors;
- decision containment and the data-processing inequality;
- penalty (Lagrangian) sweeps of a privacy weight, as in the D weighted controls;
- local search with compound (paired) moves;
- caching by sufficient statistics.

**What remains to evaluate.** The study tests one registered combination: a learned class-preserving decoder plus
budget-constrained reassignment of label-blind fine cells, for two fixed recipients. Its value is decided by claims A,
B and C and the matched controls. Any statement of an algorithmic contribution needs both:
- claim B (and, for the paired component, claim C) passing; and
- a separate novelty review against the literature in section 2.

The fixture gate and the fitting objectives cannot establish it.

**No novelty claim is made by this file.** A favourable result is an exploratory development result about this
contract, this reused Adult benchmark and this attacker slate.

## 2. Sources

### 2.1 Read today (primary sources named in prompt §16)

**Makhdoumi, Salamatian, Fawaz and Médard, *From the Information Bottleneck to the Privacy Funnel*, arXiv:1402.1774
v5 (30 Sep 2014).**
- Read: the full five-page PDF from https://arxiv.org/abs/1402.1774.
- Venue verified today: IEEE ITW 2014, DOI 10.1109/itw.2014.6970882, through the Zenodo/OpenAIRE record 1274716. The
  dpc review had recorded the venue as "believed but not verified".

Sections that bear on lcr:
- **§II.A, setup.** "S → X → Y form a Markov chain"; "All random variables are assumed to be discrete"; the joint
  P_{S,X} is known. The map reads X only.
  - lcr: SEX is not a deployment input of any map. It does influence the fitted partition in privacy arms, and the
    joint is a plug-in fitting-row estimate, not a known law.
- **§II.C, accuracy metric.** Utility is a hard constraint on average distortion, E[d(X,Y)] ≤ D, which is linear in
  P_{Y|X}. If the privacy metric is convex, problem (1) is a convex optimisation **over randomised mappings**.
- **§III.A, Lemma 1 and Theorem 1.**
  - Under log-loss inference cost, the adversary's gain equals I(S;Y).
  - For any cost bounded by L, the gain is at most 2√2·L·√I(S;Y).
  - This bounds a Bayes adversary's expected cost gain **under the known law**. It is not an AUC bound, and it says
    nothing about fitted plug-in MI on a sample.
- **§III.B, utility under log-loss.** With d(x,y) = −log P(X=x|Y=y), the distortion constraint becomes I(X;Y) ≥ R.
- **§III.C, eq. (3), the Privacy Funnel: min I(S;Y) subject to I(X;Y) ≥ R.**
  - The paper says I(S;Y) is convex in P_{Y|X}.
  - "However, because of the constraint I(X;Y) ≥ R, the Privacy Funnel (3) is not a convex optimization."
  - The non-convexity is already present over randomised maps, before any discrete restriction.
- **§IV.A, Algorithm 1 and Proposition 1.**
  - Greedy merging from the identity map, while the utility constraint holds.
  - Proposition 1 gives the merge cost as a weighted Jensen–Shannon-type entropy difference.
  - "There is no guarantee that such a greedy algorithm induces a global optimal privacy mapping."
  - Algorithm 2 (greedy maximisation) is used to bracket the achievable range.
- **§IV.B, data.** The US 1994 census (Adult) with S = (age, income level) and X = (age, gender, education level). Gender
  is therefore released there, not protected.

**What it establishes for lcr.**
- Hard-constrained minimisation of leakage about an attribute the deployed map does not read.
- Greedy deterministic partition search for it.
- A Lagrangian/IB connection.
- Non-convexity of the constrained problem.
- No global-optimality guarantee for greedy search.

**Taylor, Vippathalla and Coon, *Adaptive Privacy of Sequential Data Releases Under Collusion*, arXiv:2601.21859 v2
(10 Jul 2026; v1 29 Jan 2026), cs.IT, University of Oxford.**
- Read through https://arxiv.org/html/2601.21859v2 (sections I–IX) and the abstract page.
- The equations rendered imperfectly in the HTML. Quote numeric details only after checking them against the PDF.

Sections that bear on lcr:
- **§II, problem setup.**
  - Party k receives R̂_k, a **randomised** channel p(r̂_k | r̂^{k−1}, x).
  - The release minimises −U(R̂_k, R_k) subject to I(R̂_k; X) ≤ ε_k and I(R̂_k, R̂^{k−1}; X) ≤ δ_k.
  - The private variable is the database X itself.
  - The utility is either a distortion −E[d(R̂, R)] or I(R̂; R).
  - "As she does not have knowledge of future requests at the time of a data release, the data handler cannot jointly
    optimise all releases."
- **§III.A, dual problem.** Maximise over μ1, μ2 ≥ 0 of the minimum over the channel of
  (−U + μ1(I(R̂;X) − ε) + μ2(I(R̂,Z;X) − δ)), where Z is all previous releases.
- **§IV, Blahut–Arimoto-style algorithms.**
  - Algorithm 1 (distortion): the problem is convex, and the algorithm converges to the global minimum. The convergence
    argument needs strictly positive iterates.
  - Algorithm 2 (mutual-information utility): the problem "is not convex". The algorithm reaches only a local minimum,
    with multiple initialisations.
- **§VI, Algorithm 3, sequential strategy.**
  - Bisection over (μ1, μ2) on a grid to hit the (ε, δ) targets.
  - Time-sharing, a random convex combination of neighbouring solutions.
- **§VII, experiments.**
  - UCI Adult, with education, income and age discretised into 4, 2 and 4 bins, so |X| = 32.
  - Hamming distortion.
  - Baselines: a non-adaptive mechanism with only the per-party constraint (no collusion term), and a symmetric channel.
- **§VIII.** A connection to progressive neural networks through an IB-style objective. It does not discuss releasing
  model outputs, deterministic maps or attribute-inference attackers.

**What it establishes for lcr.**
- Sequential releases designed with explicit per-party plus collusion accounting.
- Constraint-form (hard) leakage budgets.
- Penalised/dual solutions.
- The non-adaptive baseline, which is the analogue of our LOCAL.

### 2.2 Carried over from the cbp, qpc and dpc reviews (not re-read today)

| Source | What it covers here |
|---|---|
| Slonim and Tishby, *Agglomerative Information Bottleneck*, NIPS 12 (1999) | Greedy merging with closed-form merge costs |
| Slonim, Friedman and Tishby, *Multivariate Information Bottleneck*, NIPS 14 (2001) | Joint greedy steps over several compression variables: the JOINT objective/neighbourhood structure |
| du Pin Calmon and Fawaz, *Privacy against statistical inference*, Allerton 2012 (Privacy Funnel ref. [1]) | The distortion-constrained privacy-mapping framework and the inference-cost-gain metric |
| Yamamoto, *A source coding problem for sources with additional outputs to keep secret from the receiver or wiretappers*, IEEE Trans. IT 29(6), 1983 (Privacy Funnel ref. [4]) | Foundational finite-alphabet utility–equivocation trade-off |
| Yang et al., *PURIFIER*, AAAI-23 | Transformation of released confidence scores. Membership target; not class-preserving |
| Stadler, Kulynych, Gastpar, Papernot and Troncoso, ICML 2024 | No least-privilege claim. Decision containment is the floor |
| Banerjee et al. 2005; Nock et al. 2008; Arthur and Vassilvitskii 2007 | KL/Bregman clustering and seeding (the admitted DIRECT-TASK and fine partitions) |
| Shokri et al., IEEE S&P 2017 | Coarsening the confidence vector as a defence |
| Taylor, Vippathalla and Coon, *Realisation-Level Privacy Filtering*, arXiv:2604.08630 | DP stopping filters; not a baseline (see the cbp file, section 2.3) |

### 2.3 Standard results cited for the new ingredients (from the standard references; not re-read today)

- **Proper scoring rules.**
  - Log loss and the Brier score are strictly proper. For a fixed information set, the conditional label distribution
    uniquely minimises expected loss (Brier, *Mon. Weather Rev.* 1950; Gneiting and Raftery, *JASA* 102, 2007).
  - This is the basis of the calibrated-teacher null in prompt §2. If p̄ is the true conditional law given the
    permitted input, then within a fixed token the conditional mean is Bayes for both losses. No population gain is
    available from re-decoding.
- **Histogram-binning calibration.** Zadrozny and Elkan, ICML 2001: replace a score by the empirical label frequency of
  its bin. D1 is a regularised, class-dominance-constrained version of this, applied per token.
- **Pseudo-count shrinkage.** κ·KL(p̄_t ‖ q) adds κ teacher pseudo-observations. This is a Dirichlet-prior / additive
  smoothing device, standard in categorical estimation.
- **Argmax-preserving recalibration** (for example temperature scaling, Guo et al., ICML 2017) is a known way to change
  confidence without changing decisions. The class-dominance constraint u[d_t] ≥ u[k] serves the same purpose here.
- **Sequential IB.** Slonim, Friedman and Tishby, SIGIR 2002: draw one element out of its cluster and reassign it to the
  cluster that most improves the objective, sweeping until no change. This is the nearest published form of lcr's
  single moves of a whole fine cell.
- **Plug-in MI bias.** Miller–Madow; Paninski, *Neural Computation* 15 (2003). Carried over from cbp.

### 2.4 Later privacy-funnel solvers found today (titles and repositories only; not read)

- A difference-of-convex PF solver (arXiv:2403.04778, third-party PyTorch prototype `hui811116/dcaPF-torch`).
- An EM-relaxed PF method (arXiv:2405.00616).
- Deep (variational) privacy-funnel models (Razeghi et al.; arXiv:2404.02696).

They confirm that optimising the Privacy Funnel is an active, established line. None is a baseline here; all optimise
randomised or continuous mappings.

## 3. Code availability (re-verified 2026-10-06 23:58Z to 2026-10-07 00:03Z)

| Method | Official code | Can it be pinned? | Status today |
|---|---|---|---|
| Privacy Funnel (Makhdoumi et al. 2014) | none | **No** | **Not found.** The arXiv abstract page lists no code or data link. The PDF references no code. The Zenodo record 1274716 holds `article.pdf` only. GitHub repository searches return 0 for "1402.1774" and "privacy funnel greedy". "privacy funnel" returns 51, mostly web-analytics products. The research hits are third-party: `AGhafaryy/Privacy-Funnel`, a 2021 student project on PF under maximal leakage, not by the authors; `BehroozRazeghi/*`, deep PF models; and `hui811116/dcaPF-torch`, the DC solver. The GitHub accounts found under the authors' surnames hold no PF code. A web search found no official implementation |
| Taylor, Vippathalla and Coon solver (2601.21859) | none | **No** | **Not found** (unchanged from cbp). The abstract page (v1 29 Jan 2026 15:29:38 UTC; v2 10 Jul 2026 09:56:05 UTC) has no comments or journal-ref field, and its "Code, Data, Media" toggles show no linked repository. The v2 HTML holds only arXiv/LaTeXML URLs and author e-mail addresses. GitHub repository searches for "Vippathalla", "2601.21859", "adaptive privacy sequential data releases" and "sequential data releases collusion" return 0. The pith.science review page links only its own site owner's repositories. A web search found no repository |
| PURIFIER (context only) | https://github.com/wljLlla/Purifier_Code | **No** | Still empty at 00:02Z: `size` 0, `language`/`license` null, `pushed_at` 2022-11-29T13:57:42Z; the `/commits` endpoint returns HTTP 409 |

**Consequence.**
- Neither Privacy Funnel nor Taylor et al. can be run as an official baseline.
- Our constrained LOCAL, SEQ-12 and SEQ-21 are **matched adaptations**, and "the Privacy Funnel algorithm" or "the
  Taylor solver" must not be written for them.
- The registered claims A, B and C do not depend on either official method.

## 4. How the lcr arms map onto prior work

| Arm or ingredient | Nearest prior procedure | What differs here | Status |
|---|---|---|---|
| D1 decoder (every new arm and the B controls) | Histogram-binning calibration with pseudo-count shrinkage; proper-loss fitting | Per token: log loss + 0.5·Brier + κ·KL(p̄_t‖q), κ = 32; class-dominance constraint; affine ε-smoothing; certified released vector | Standard calibration under a constraint. **Changes no token, so removes no information** (section 5.1) |
| B controls (D1 on the exact fixed D0 maps) | Recalibration of an existing code | Assignments unchanged; privacy-trained maps stay **privacy-trained** | Calibration-only control |
| C-TASK | Supervised sequential-IB-style reassignment under task loss | Class-restricted whole-fine-cell moves; D1 re-solved per move; FINE-TASK start; no SEX | Standard local search, supervised. **Privacy-untrained** |
| Constrained LOCAL | Privacy Funnel (hard utility constraint, min I(S;Y)) | Reassignment moves, not merges; true-label log-loss and Brier budgets of a learned decoder instead of I(X;Y) ≥ R; local cap I_i ≤ I_i(C-TASK); class preservation; plug-in law | Matched adaptation, **not the Privacy Funnel algorithm** |
| Constrained SEQ-12 / SEQ-21 | Taylor et al. sequential collusion accounting; their non-adaptive baseline corresponds to LOCAL | See the list below | Matched adaptation, **not the Taylor solver** |
| JOINT-SINGLE | Multivariate-IB-style joint objective with single-variable moves | Φ is leakage to reduce, under both recipients' budgets and caps | Matched adaptation |
| JOINT-PAIR | Local search with compound (2-exchange-type) moves | Atomic paired moves from a fixed 8×8 bank per recipient; equal TOTAL proposal-evaluation ceiling with JOINT-SINGLE and both sequential orders | The component tested by claim C. Larger neighbourhoods are a standard local-search device |
| D weighted controls | Lagrangian / penalty sweep (as in cbp) | True-label task objective T with D1; λ grid fixed in advance | Design, not method |
| Fitting budgets 0.005 nats / 0.003 Brier | Hard utility constraints (Privacy Funnel; Taylor et al.) | A registered design choice applied to accepted moves on fitting rows | Design, not a confidence certificate |

**What SEQ-12 and SEQ-21 change relative to Taylor et al.:**
- deterministic tokens, where they use randomised channels;
- leakage about S, not about the database X;
- greedy single-move search, not Blahut–Arimoto dual ascent;
- **the roles of objective and constraint are swapped.** We minimise the leakage Φ subject to utility budgets and local
  caps. They maximise utility subject to the leakage constraints ε_k and δ_k. The two correspond only through a
  Lagrangian, and no equivalence is claimed;
- stage one is fitted against the other recipient's CLASS-ONLY view, which the contract always discloses. The temporary
  partner is not required to be feasible;
- the order and both requests are fixed in advance. In Taylor et al. the later request is unknown;
- the arms inherit none of their convergence or optimality results.

**They must never be called** "the Taylor solver", "a Taylor baseline" or "an official sequential baseline". "Beats
Taylor" must not be written.

## 5. Mandatory caveats (to be carried into every decision document)

### 5.1 The decoder caveat

**A public deterministic decoder of an unchanged token removes no information.**
- For a class-preserving code, recipient i's complete interface is R_i = (t_i, q_i, d_i), with q_i = h_i(t_i) and
  d_i = class(t_i).
- So R_i is a deterministic function of t_i, and t_i is part of R_i. Hence I(S; R_i) = I(S; t_i) and
  I(S; R_1, R_2) = I(S; t_1, t_2) for every choice of decoders h_1 and h_2.
- Replacing D0 by D1 on the same tokens leaves the Bayes-optimal complete-interface ROC unchanged.

Consequences:
- A finite attacker's AUC may move after re-decoding. That movement is reader behaviour, not information removal. It must
  not be credited as privacy.
- Privacy credit requires a **changed partition**.
- The D0-versus-D1 identical-token diagnostic must show exactly unchanged token counts and plug-in MI, with full-token and
  probability-only recovery reported side by side.
- A D1 gain on a fixed map is a **utility (calibration) gain**. It can make an existing private partition usable.
- If claim A passes with a calibrated construction, the privacy came from the cbp partition, and the decoder made it
  usable.

### 5.2 Supervised training-label use on OSF_DEFENSE_FIT (disclose plainly)

**This is a material change from qpc and cbp, whose codebooks were fitted without labels.** On OSF_DEFENSE_FIT (15,434
rows), this study reads:
- **the true task labels**, for the D1 decoders, C-TASK, the weighted controls' task objective and the fitting budgets;
- **SEX**, for the privacy objectives, the weighted controls and the local caps.

What stays label-blind or unchanged:
- The fine partitions remain fixed and label-blind.
- The D1 per-token objective uses no SEX.
- SEX is never a deployment input.
- SEX does shape the fitted partition in every privacy arm and weighted control.

**Exposure of fitting rows.** Released vectors q_t depend on fitting-row label counts y_t, and privacy partitions depend
on fitting-row SEX. No membership or attribute guarantee for the fitting rows is given, and the maps and decoders stay
private.

**The teachers' training rows.** The U encoders and deployed heads were fitted on OSF_DEFENSE_FIT (osf protocol). The
consequences:
- D1 calibrates against the teacher's in-sample outputs.
- L_i(U) and B_i(U) in the fitting budgets are U's in-sample losses.
- Any held-out confidence benefit of D1 is an empirical inner/assessment question, not a consequence of the fitting
  objective.

### 5.3 Not convex overall

- For a **fixed token**, the D1 objective is convex in u over a convex polytope:
  - q is affine in u;
  - −log q is convex;
  - the Brier term is quadratic;
  - KL(p̄_t‖q) = const − Σ p̄ log q;
  - the simplex intersected with {u[d] ≥ u[k]} is convex.
- The fixed-token certificate (KKT and stationarity residuals, projection magnitude) applies only to that token's
  continuous problem.
- The discrete search over partitions, with decoders re-solved after each move, is **combinatorial and not convex**. The
  Privacy Funnel is already non-convex over randomised maps (§III.C).
- Every mapper result is a **heuristic local optimum** of its registered search. No global optimum, no optimality gap
  and no "mathematically infeasible" verdict follows.
- A map that violates a budget under D1 is "infeasible under this registered decoder", not infeasible in general,
  because D1 is regularised (κ = 32) and is not the unregularised Bayes rule.

### 5.4 Fitted MI is not a population guarantee

- I_i, I12 and Φ are plug-in quantities on OSF_DEFENSE_FIT, the same rows used to fit the partition.
  - They are biased upward by alphabet size and downward by selection.
  - A fitted decrease, a value below a permutation null, or a satisfied local cap I_i ≤ I_i(C-TASK) is a
    **fitting-law constraint**, not protection.
- The fitting budgets (0.005 nats, 0.003 Brier) are **design choices** applied to accepted moves on fitting rows.
  - They are not population confidence certificates.
  - The D1 decoder is fitted on the same rows, so they are optimistic.
- Only the unchanged inner rules and the assessment bounds decide confidence preservation.
- A fitting-budget certificate does not imply a passing assessment clause.
- Privacy Funnel Theorem 1 bounds a Bayes adversary's cost gain under a known law. It gives no AUC bound for any fitted
  attacker here.

### 5.5 Sequential arms are matched adaptations, not the official Taylor solver

See section 4. This holds for both the constrained arms (K-SEQ-12, K-SEQ-21) and the weighted controls (W-SEQ-12,
W-SEQ-21).

### 5.6 Fixed-task output-release scope

- Per recipient, the release is a fixed-task **output**: a categorical token identity, its decoded probability vector
  and the unchanged teacher decision.
- It is not a reusable feature representation, and not privacy-preserving representation learning.
- It does not combine the encoder/LoRA line and the stochastic-release line into a pipeline.
- Decisions are preserved exactly, so I(S; C1, C2) ≥ I(S; d1, d2). Decision-only recovery is a floor for every arm. The
  source measured a decision-only pair AUC of about 0.74.

### 5.7 Other limits carried from cbp

- Exposure: the registered statement in `EXPOSURE_LEDGER.md`.
  - All roles have been used.
  - Nominal intervals ignore the adaptive history.
  - Repeated nominal success on these rows is not confirmation.
- Attacks: a declared matched slate. Weak recovery by it is not a population or all-attacker certificate.
- References (RAW-J, FARE, F0, LEACE) operate under different contracts. Comparisons with them are descriptive.
- Fitting-objective dominance (for example a JOINT arm against its witnesses) holds by construction on the fitting rows.
  It does not imply a held-out gain.
- The new selection rule (original eligibility without cbp's headroom buffer) was chosen after cbp's headroom rule
  failed on these rows. That is adaptive and must be stated wherever an existing (D0) map becomes P\*.

## 6. Wording per possible label (prompt §12)

### 6.1 Naming the construction of P\*

`lcr.select.winning_name` returns `<simplest family>; <construction>` over P\*'s exact alias set (identical deployed
tokens and released vectors on every seed). Write it out as follows:

| Construction | Config IDs | Required description | Required attribution sentence |
|---|---|---|---|
| **existing** | `U\|<FAM>\|i8o64\|l<λ>` (D0) | "the admitted cbp <FAM> λ <x> map with its original mean-teacher decoder (D0); nothing was refitted" | "The learned decoder and the new search added no demonstrated benefit. The map was nominated by this study's new selection rule (original eligibility without cbp's headroom buffer), which was chosen after cbp; this is adaptive." |
| **calibrated** | `<D0 id>\|D1` | "the cbp <FAM> λ <x> partition, unchanged, with the learned decoder D1" | "The protection comes from the existing cbp partition; the learned decoder made it usable. No new assignment search contributed (a DECODER-ENABLED result)." |
| **weighted** | `U\|W-<FAM>\|i8o64\|l<λ>\|D1` | "a true-label weighted control (<FAM>, λ <x>) with the learned decoder" | "A weighted baseline produced the best usable protection; the hard-budget constrained search added no demonstrated benefit." |
| **constrained** | `U\|K-<ARM>\|i8o64\|D1` | "the registered constrained <ARM> search with the learned decoder" | "Claim A alone does not show that the constraints, rather than decoding or ordinary supervised search, caused the gain; that is claim B." |

- When aliases span constructions, the simplest one is named (existing < calibrated < weighted < constrained). Families
  are named LOCAL < SEQ-12 = SEQ-21 < JOINT(-SINGLE) < JOINT-PAIR.
- A D1 release never aliases its D0 map, because the released vectors differ.
- **Fixed-map D1 privacy releases are privacy-trained controls.** They must never be described as task-only or
  privacy-untrained because the decoder itself uses no SEX.

### 6.2 Allowed and forbidden wording

| Label | Allowed sentence (fill the brackets) | Must not be written |
|---|---|---|
| **PRIVACY_RELEASE_DEVELOPMENT_CRITERION_MET (<family>; <construction>)** | "On the reused Adult development benchmark, <construction description of P\*> reduced combined-view SEX recovery by more than 0.02 pair AUC (registered lower bound) relative to the strongest eligible privacy-untrained release T\* (<T\* config>). Neither recipient's own AUC rose by 0.01 or more. Both tasks stayed within the original accuracy, log-loss and Brier allowances, and every decision was unchanged (exploratory, locked, single opening of reused rows; not confirmation)." Then add the attribution sentence of section 6.1. | "new method/algorithm", "our constrained search works" (unless B passes), "joint advantage" (unless C passes), "private", "certified", "chance-level", "information removed by the decoder", "the decoder protects", "task-only" for a calibrated or existing privacy map, "beats Taylor / the Privacy Funnel / PURIFIER", "privacy-preserving representation", any population or MI guarantee |
| **CONSTRAINED_SEARCH_INCREMENT_ESTABLISHED** | "The registered constrained search (N\* = <K-arm>, D1) beat the strongest eligible calibration and true-label weighted incumbent (C\* = <config>) by the registered margins on this reused benchmark and attacker slate. The incumbents used the same decoder, starts, caps and single-move neighbourhood. Novelty is reviewed separately." If A did not pass: "...but no release met the useful-release criterion against strong task-only compression (claim A <status>)." | "hard constraints beat penalties in general", "the constraints caused the gain" beyond this matched bank, "globally optimal", "a new algorithm" (pending the novelty review), "joint advantage" unless N\* is a joint arm AND C passes, "beats the Privacy Funnel / Taylor" |
| **PAIRED_JOINT_INCREMENT_ESTABLISHED** | "The paired-move joint search (J\* = K-JOINT-PAIR, D1) beat the strongest eligible non-pair release (C_pair\* = <config>) by the registered margins. C_pair\* ranged over constrained LOCAL, both sequential orders, JOINT-SINGLE and every incumbent control. JOINT-SINGLE and both sequential orders had the same total proposal-evaluation ceiling; actual proposals and CPU were <numbers>. This is evidence for this paired component on this benchmark and slate." | "joint design is better than sequential (in general)", "coordination theorem", "universal superiority", "beats Taylor", "the joint objective causes the gain" (JOINT-SINGLE has the same objective; the C claim isolates the paired neighbourhood), any claim that hides unequal actual work |
| **CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION** | "The fixed original D0 DIRECT-TASK i8o64 code (Q) kept both tasks' log loss and Brier within the original allowances on all four registered bounds. No privacy-method claim (A, B, C) passed. Q's result repeats the qpc and cbp outcome on the same reused rows; it is not new evidence for learned decoders." | "learned decoders improve confidence" (Q is D0), "privacy release works", "capacity impossibility", "confirmation" |
| **EXPERIMENTAL_NO_ADVANTAGE** | "The registered work was complete and valid, but no method claim (A, B, C) and no confidence bound set (Q) passed on this locked assessment. This closes this exact recipe (D1 with κ = 32, these budgets and neighbourhoods) on these rows." Then list each claim's exact failed clauses with their classification (MEASURED_VIOLATION vs NOT_ESTABLISHED_PRECISION vs NOT_ESTABLISHED_POINT). | "learned decoders cannot help", "constrained search cannot work", "privacy release is impossible", "a universal privacy floor", "a competitive negative against Taylor / the Privacy Funnel", "a relaxed λ, budget or κ would pass", any recommendation of another small adjustment on the same assessment |
| **MECHANISM_GATE_NOT_MET** | "The registered fixture gate failed at <exact criterion and fixture>. Adult was not fitted. There is no Adult result in either direction. The fixture decision, independent oracle replay and runnable implementation are delivered." | "competitive method negative on Adult", "the method fails on Adult", "constrained search is useless", "learned decoders do not work", any Adult number presented as a test of the method |

### 6.3 Statuses and states that can appear beside a label

| Status / state | Allowed | Must not be written |
|---|---|---|
| NOT_ESTABLISHED (a claim) | "Claim <X> was not established: clause(s) <ids> failed (<classification>)." A point inside the limit with the bound outside is "preservation not established", **not** a violation. A violation is stated only for MEASURED_VIOLATION | "the method is worse", "violates confidence" (unless MEASURED_VIOLATION) |
| NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE | "No <role> candidate passed the registered inner nomination (<ORDINARY_UTILITY_FAILURE / LOCAL_GUARD_FAILURE / CONSTRAINED_FIT_INFEASIBLE>). This refutes this nomination procedure for this role, not the idea in general. The descriptive fallback <config> is shown for information only." For CONSTRAINED_FIT_INFEASIBLE, add: "The constrained search found no release that met its fitting budgets under this registered decoder." | "a relaxed rule would pass", any claim from the descriptive fallback, "the constraints are mathematically infeasible" |
| INCOMPLETE_OR_INVALID (a claim or overall) | "Claim <X> is incomplete because <exact root cause: missing comparator / failed required control / nonfinite statistic / unresolved verification failure>." | Any completed positive or negative head-to-head statement for that claim |
| CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION beside an incomplete method claim | The registered label follows the prompt-literal §12 order, as changed in review R-1: when Q passes and no method claim passes, the label is CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION even if a method claim is INCOMPLETE_OR_INVALID. Then write: "...; claim <X> is incomplete because <exact root cause> (shown)." | Calling the incomplete claim a completed negative, or omitting it |
| DECODER_ENABLED_ROUTE / ASSIGNMENT_SEARCH_ROUTE (fixture gate) | "On the registered fixtures, the gate passed through <route>." These are fixture classifications that decide which Adult controls are emphasised | Any Adult claim from the route |

### 6.4 The opening sentences of the final answer (prompt §17)

"Learning the probabilities [did/did not] improve confidence."
- Base this on the identical-token D0-versus-D1 comparison (`DECODER_ONLY_ABLATION.csv`): the scored best D1 fixed-map
  privacy control against its paired D0 release, with intervals, and the inner bank as context.
- Q is D0 and cannot answer it.
- Write "did" only if that comparison shows lower log loss and Brier for D1 on both tasks with the interval excluding
  zero. Otherwise write "did not clearly".
- This is descriptive: no registered primary claim tests it. The rule is registered before fits as
  `LABEL_TRUTH_TABLE.report_rules.decoder_sentence` (review R-3). It uses the paired per-task D1 − D0 contrasts with the
  same bootstrap draws and the primary z; "did" iff all four upper bounds are < 0, otherwise "did not clearly".

"[Changing the assignments / The calibrated existing method / A weighted baseline] produced the best usable
protection."
- Fill it from P\*'s construction (section 6.1):
  - constrained means "Changing the assignments";
  - calibrated means "The calibrated existing method";
  - existing (D0) means "The existing cbp method, unchanged";
  - weighted means "A weighted baseline".
- P\* is never C-TASK, because C-TASK is privacy-untrained.
- If there is no P\*, write: "No release produced usable protection under the registered nomination."

"The full development criterion [passed/did not pass]." Claim A's status, exactly.

## 7. Gaps that remain

| Gap | What remains open |
|---|---|
| G1. Official baselines | No runnable official Privacy Funnel or Taylor et al. code (section 3). Results speak only to the registered arms and admitted references |
| G2. Randomised mechanisms | None in the bank. Randomised channels and time-sharing (Taylor et al. §VI) reach trade-off points that deterministic maps cannot |
| G3. Plug-in quantities | Every I and Φ value is a fitting-row plug-in. The i8o64 pair alphabet makes the I12 null level comparable to the differences being optimised (cbp MATH_REVIEW §3.3) |
| G4. Decision floor | I(S; C1, C2) ≥ I(S; d1, d2) for every class-preserving release |
| G5. Decoder fitted on the teacher's training rows | D1's correction targets the teacher's in-sample miscalibration on OSF_DEFENSE_FIT (section 5.2) |
| G6. Compute matching | JOINT-PAIR has a different neighbourhood. Matching is by a TOTAL proposal-evaluation ceiling, not by CPU; actual CPU and proposals must be reported |
| G7. Training-data exposure | Decoders hold fitting-row label statistics and partitions hold SEX-dependent groupings. Both are kept private, and no fitting-row privacy guarantee is given |
| G8. Novelty | No genuinely new algorithmic step has been identified. A favourable B or C needs a separate novelty review before any contribution wording |

## 8. Summary

**Allowed:** "a class-preserving output code at fixed capacity (i8o64), whose released probabilities are fitted per
token from true task labels with a fixed teacher prior, and whose assignments of label-blind fine cells are searched
under explicit fitting budgets with standard greedy moves. It was evaluated once on a locked, reused Adult development
benchmark against calibration-only, task-only, true-label weighted, local, sequential and single-move joint controls
and admitted continuous references."

**Not allowed:**
- novelty of hard utility constraints, privacy-funnel objectives, proper-loss calibration, pseudo-count shrinkage,
  greedy or paired-move partition search, penalty sweeps or sequential collusion accounting, or of their combination
  before a passing B/C and a separate novelty review;
- crediting a decoder change on unchanged tokens as information removal;
- calling the discrete program convex, or any mapper result optimal;
- any population, attribute-specific or training-data privacy guarantee from fitted MI, fitting budgets, held-out AUC or
  structural properties;
- "the Taylor solver", "a Taylor baseline", "the Privacy Funnel algorithm", "beats Taylor", "beats the Privacy Funnel"
  or "beats PURIFIER";
- reusable or general-purpose representation learning.
