# Prior work and novelty — ccm (role D, Stage B)

Bounded review, 2026-10-08. Primary sources were read as full text (pdftotext of the arXiv/PMLR PDF) in the sections
listed. No data was loaded. "Not found" means not found in this bounded reading; it is not evidence of priority.

## 0. Sources

| Key | Source (version read) | Read |
|---|---|---|
| TVC | Taylor, Vippathalla, Coon, Adaptive Privacy of Sequential Data Releases Under Collusion, 2601.21859v2 | full |
| LKSC | Liao, Kosut, Sankar, Calmon, Privacy Under Hard Distortion Constraints, arXiv:1806.00063 (ITW 2018) | full |
| KKS | Kalantari, Kosut, Sankar, IT Privacy with General Distortion Constraints, arXiv:1708.05468v3 | §I–III |
| CZK | Cai, Zhang, Khalili, Privacy-Aware Randomized Quantization via LP, PMLR 244:499–516 (UAI 2024) | §3–4 |
| PF | Makhdoumi, Salamatian, Fawaz, Médard, From the IB to the Privacy Funnel, arXiv:1402.1774v5 | full |
| IB | Tishby, Pereira, Bialek, The information bottleneck method, arXiv:physics/0004057 | §2.2, §3 |
| RG | Rassouli, Gündüz, On Perfect Privacy, arXiv:1712.08500v8 (JSAIT 2021) | §I–III |
| ML | Issa, Wagner, Kamath, An Operational Approach to Information Leakage, arXiv:1807.07878 | §I–III |
| DWCS | Diaz, Wang, Calmon, Sankar, Robustness of IT Privacy Measures and Mechanisms, arXiv:1811.06057v3 | Thm 1, 6, 8 |
| GSOS | Grosse, Saeidian, Oechtering, Skoglund, Privacy Mechanism Design … Empirical Distributions, 2509.22428 | §I |
| ZOS | Zamani, Oechtering, Skoglund, Multi-User Privacy Mechanism Design, Non-zero Leakage, 2211.15525 | §I–II |
| MG | Jia, Salem, Backes, Zhang, Gong, MemGuard (membership-inference defence), arXiv:1909.10594 | §3–4 |
| MAD | Orekondy, Schiele, Fritz, Prediction Poisoning (model-stealing defence), arXiv:1906.10908 | §3 |
| AG | Jia, Gong, AttriGuard (attribute-inference defence), arXiv:1805.04810 | §3–4 |

Seen only in reference lists (not read): Slonim and Tishby, agglomerative IB (PF ref. [3]); Calmon and Fawaz 2012 (PF
ref. [1]); Salamatian et al., QPM quantization (AG ref. [19]); Zamani et al., ITW 2021 (TVC ref. [13], known here only
through TVC's description). No knowledge-distillation paper was read, so this review cannot assess distillation prior art.

## 1. Objectives and utility/distortion constraints that already exist

- **Hard per-input distortion.** LKSC §III, Eq. (9) requires every output to lie in B_D(x) = {y : d(x,y) ≤ D} w.p.1,
  for any d. G is such a ball with x = p, y = q: (NLL) is D_∞(p‖q) = log max_k p_k/q_k ≤ d (the worst-label log-loss
  excess); (Brier) and (Class) intersect further balls. PROTOCOL §3's "every supported output" is LKSC's w.p.1.
- **Hard constraints on released confidence vectors.** MG Eq. (2) and MAD Eq. (8) impose hard argmax preservation,
  with a distance budget (MG Eq. (3): expected L1; MAD Eq. (7)). CZK Eq. (1) makes E[M(x)] = x for every x (mean only).
- **Expected distortion** (averaged over inputs): PF §II-C/D, Eq. (1); KKS Def. 3, Eq. (10), L(D) = min I(private;
  release) s.t. E d ≤ D; TVC §II, U = −E d (Hamming in §VII); AG Eq. (1), KL objective s.t. E d ≤ β. KKS Thm 1–4 add
  cost and tail (CDF) constraints on dataset-averaged distortion; KKS §I-B separates these from LKSC's per-input balls.
- **Log-loss.** PF Lemma 1: under log-loss inference cost, leakage = I(S;Y); PF §III-B: log-loss distortion of the
  input gives I(X;Y) ≥ R. IB Thm 4 (Eq. (16)) and Thm 5 (Eq. (30)): assigning x to a cell costs KL[p(y|x) ‖ p(y|cell)].
  **G_exp's KL(p‖q) ≤ d is thus a per-input hard IB distortion, and G's NLL its D_∞ (worst-label) strengthening.** In
  IB the cell vector is the Bayes centroid (Eq. (17)); in ccm it is any certified q.
- **Leakage measures.** MI (PF, KKS, TVC); maximal leakage (ML Def. 1/Thm 1, L(X→Y) = log Σ_y max_x P(y|x), equal to
  log|supp Y| for any deterministic release, Example 6); maximal α-leakage (LKSC Def. 1, Thm 1); guessing probability
  P_c(U|V) (DWCS Def. 3, i.e. Bayes accuracy); PML (GSOS). ccm's plug-in MI and Bayes accuracy of SEX are standard.
  DP (CZK Eq. (6): p(x,i)/p(x',i) ≤ e^ε over all inputs) bounds a ratio of the mechanism's output laws. It is not
  attribute-recovery privacy, and not G's ratio, which compares q with p.
- **Roles.** TVC maximises utility s.t. individual (ε_k) and cumulative (δ_k) MI budgets (Eq. (1)–(2)); PF, KKS, LKSC
  and ccm minimise leakage s.t. utility. In TVC's convex case both trace one frontier (dual, Eq. (3)); for
  deterministic codes under hard guards they need not.

## 2. Standard solvers and assignment updates

- **LP for randomised mechanisms.** RG Thm 1: perfect privacy is a standard LP (≤ nul(P)+1 outputs suffice, each
  supported on ≤ rank(P) inputs). CZK Lemma 1, Thm 5–6, Alg. 2 OPTM: linearised MAE and DP constraints, LP plus grid
  search. AG §4.3: convex program via KKT. ML Cor. 1(6): exp L is convex in P_{Y|X}, so minimising maximal leakage
  under a convex guard is convex. ccm's stochastic-local LP belongs to this class.
- **Blahut–Arimoto / IB alternating minimisation.** IB §2.2, Thm 5; TVC Prop. 1, Thm 1, Alg. 1–2 (BA with a
  conditioning variable z), Remarks 1–3 (convergence), Alg. 3 (multiplier grid, then bisection and time-sharing to
  reach (ε, δ)). BA yields soft assignments, not certified bins.
- **Greedy/agglomerative deterministic merging.** PF Alg. 1 starts at the identity and merges the pair that most
  reduces I(S;Y) while I(X;Y) ≥ R (merge costs: Prop. 1); locally optimal, no global guarantee (§IV-A). **ccm's
  "local" arm is this algorithm with G-admissibility plus the capacity cap replacing I(X;Y) ≥ R.**
- **Exact hard-distortion optimum.** LKSC Thm 3/4, Eq. (12): q* = sup_Q inf_x Q(B_D(x)), achieved by
  P*(y|x) ∝ Q*(y)·1{y ∈ B_D(x)}; Thm 5 is a worked case with a deterministic optimum.
- **Certification** (float64; closed-form necessary condition for infeasibility) is a safeguard, not a new solver.

## 3. Two colluding recipients versus Taylor et al.

**TVC's setting** (§I–II, Figs. 1–2):
- m parties request R_k online; the handler "cannot jointly optimise all releases" (§II).
- R̂_k ~ p(r̂_k | r̂^{k−1}, x): the handler sees the whole database X and conditions on the realised past releases.
- Privacy covers all of X: I(R̂_k;X) ≤ ε_k and I(R̂_k, R̂^{k−1};X) ≤ δ_k, against one actor holding all releases to
  date; §IX extends this to collusion by subsets.
- §VII: Adult with X = (education, income, age), |X| = 32, requests (E, J, A, E), Hamming distortion; SEX is not in
  X. §III: a repeated request reuses R̂_1 at zero cumulative cost.

**Consequence.** Two recipients with their own constraints plus a coalition constraint is TVC with m = 2; multi-user
disclosure with one shared release is also prior (ZOS; TVC ref. [13]). ccm must not claim this structure as new.

**What differs in ccm** (each ingredient is known; §1–2):
- (a) The target is an attribute (SEX) outside the map's input: RG's output-perturbation model, not X itself.
- (b) Utility is a hard pointwise all-label guard, and privacy is the objective.
- (c) Releases are deterministic, capacity-bounded codes with a disclosed fallback.
- (d) The joint arm uses both purposes up front: the offline relaxation TVC excludes by assumption, an information
  advantage rather than an algorithmic one. Exact joint coalition MI ≤ exact sequential coalition MI only by
  construction (same feasible set, plug-in objective, fitted law). Nothing orders local leakage, finite-attacker
  accuracy or estimated laws.
- (e) **Open point for role A.** TVC's release conditions on the realised t1. If ccm's t2 is a function of p2 alone,
  chosen knowing code 1, it is non-adaptive and must be labelled so. S12/S21 inherit no TVC optimality (TVC's is
  global only for convex expected distortion).

## 4. Does a finite-law optimum transfer to estimated Adult laws?

- **Every optimum read here assumes a known pmf.** TVC Alg. 3 takes {p(r,z,x)} as input (§VII: an empirical pmf on 32
  cells); PF, RG, KKS and ZOS assume the joint law is known. TVC's "optimal" means optimal for that pmf.
- **The robustness bound is vacuous here.** DWCS Thm 1: |L(P̂_n,W) − L(P,W)| = O(√((|S|·|X| − log β)/n)), uniformly
  in W; with X the continuous reference vector, |X| ≈ n. DWCS Thm 6 (consistency) and Thm 8 (uniform mechanisms
  designed at ε − C_L·r) are finite-alphabet and asymptotic. GSOS needs an explicit uncertainty set B_β(P̂).
- **The support changes at deployment.** The maximal-leakage optimum depends on P_X only through its support (LKSC
  conclusion; ML after Thm 1), but deployment brings new p vectors, and the fallback releases a continuous q = p, for
  which ML Lemma 1 gives no finite bound. Utility under G is distribution-free per input; privacy is not.
- **Conclusion.** The oracle's exact plug-in MI/Bayes optimum describes the fitted (toy or OSF_DEFENSE_FIT) law only.
  It is **not a population certificate**. Only INNER_SELECTION finite-attacker scores face new rows, and they are
  exploratory.

## 5. What kind of improvement is plausible

- **Application, plus at most a small algorithmic adaptation.** Existing pieces applied to releasing calibrated
  classifier scores to two recipients: hard-distortion privacy design (LKSC) with an IB-type KL/D_∞ distortion,
  constrained privacy-funnel merging (PF Alg. 1), and LP for stochastic arms (RG, CZK). The adaptation is certified
  admissible bins under a capacity cap.
- **Not architectural.** No encoder is learned; a code is a partition of fixed teacher outputs.
- **An algorithmic claim** needs ccm's selector to beat constrained PF merging and the LP arm at the same guard,
  capacity and inputs (§7). Otherwise this is an application or feasibility result.

## 6. Key fact: a hard D_∞-type guard and achievable privacy

- **LKSC Thm 3/4 apply to G** (their proofs use only the balls). For maximal leakage (every α > 1) about p itself, the
  optimum is −log q*, achieved by Eq. (12), dependent on P_X only through its support (Remark 1). Thm 5's optimum is
  log ⌈(n+1)/(2m+1)⌉, the number of balls in an exact partition of the inputs, with a deterministic map.
- **Our LP-duality step (not stated in LKSC).** 1/q* is the fractional cover number of the inputs by the sets
  {p : q ∈ B(p)}. Summed over predicted classes on the fitted support, **log(F2 packing) ≤ min L(p→release) ≤
  log(F1 cover)** for any mechanism satisfying G w.p.1. F2's NLL-only test is valid because G lies inside the NLL ball,
  and cross-class pairs violate (Class). This is leakage about p, not SEX; ML Lemma 1 (DPI): L(SEX→·) ≤ L(p→·).
- **Geometry** (UTILITY_CONTRACT §3): every admissible bin has TV diameter ≤ e^d − 1 ≈ 0.005. The release pins p to
  a 0.5%-TV cell, so SEX signal in p at coarser scales is disclosed by every arm. Privacy-aware choice acts only on SEX
  variation within cells and on which cover is chosen, which caps the headroom before any data are used.
- **Perfect privacy.** RG Eq. (5), Prop. 1: S ⊥ release iff every output's posterior lies in the polytope S_{X,Y}; for
  deterministic tokens, each cell must be SEX-balanced at the prior, under G inside 0.5%-TV cells. RG Thm 2, §V:
  output perturbation (ccm's model: the map sees p, not SEX) can rule out perfect privacy where full-data observation
  allows it. d = 0 forces q = p (identity), as in KKS Thm 3's D_g = D_min case.
- **No paper read proves a G-specific impossibility for SEX.** Whether G forces near-identity is the empirical
  geometry question F1–F4 answer. A NEAR_IDENTITY verdict would agree with LKSC; it would not be novel.

## 7. Comparison required to justify a contribution

**Shared by all arms:** same Ucal inputs (3 seeds), G (d = 0.005, b = 0.0025), i8o64 capacity, fallback rule and
certification. Fallback flags and vectors are part of every view.
1. **Correctness on TOY_LAWS.** Recover the exhaustive optimum, with an independent LP re-check, before any real law.
2. **SEX-blind floors and controls:** decision-only (confidence-ineligible); CLASS, if admissible; task-only minimal
   cover; the LKSC Eq. (12) maximal-leakage-optimal mechanism.
3. **Literature-faithful baselines:** PF Alg. 1 restricted to G-admissible merges; the LP stochastic-local arm; a
   TVC-faithful sequential arm in both orders, where t2 may condition on t1 (or a stated non-adaptive design).
4. **Incumbents.** Re-certify the hcal/lra i8o64 codes under G; report ineligible codes as ineligible.
5. **Measures:** exact plug-in MI and Bayes accuracy on the fitted law (labelled known-law); the band [log F2, log F1];
   the registered finite attackers on new rows (linear, nonlinear, token/cell, coalition, defence-aware,
   fallback-metadata).
6. **Claim rule.** A contribution needs a pre-registered advantage (pair benefit ≥ 0.02 with the 0.01 local guard)
   over items 2–3, not just task-only, on all 3 seeds at matched capacity and fallback rate, and not explained by
   fallback. Joint vs sequential is reported as foreknowledge, never as "sequential is inferior" or "optimal".

## 8. Guardrails

- **Not new; never claim it:** collusion-aware multi-recipient release (TVC, ZOS); hard-distortion privacy (LKSC);
  KL/D_∞-bounded score quantisation (IB); argmax-preserving confidence release (MG, MAD); greedy privacy-funnel
  merging (PF).
- **Keep measures separate:** DP (CZK), maximal leakage and PML (ML, LKSC, GSOS), MI, and Bayes-accuracy attribute
  recovery. None implies another without a stated inequality.
- **Known-law optima are not population certificates** (§4). Adult results are exploratory development evidence.
