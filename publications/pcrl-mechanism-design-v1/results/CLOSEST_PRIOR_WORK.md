# Closest prior work

Curated from the closest-prior-work reviewer's two files, both in `roles/prior_work_reviewer/`:
- `LITERATURE_BASE.md`: 47 PDFs read in full text, plus a 45-row field matrix;
- `PROPOSAL_COMPARISON.md`: 22 further sources.

Every link below was opened by the reviewer on 2026-10-16. No priority claim is made anywhere. "Not found" means not found in the sources read.

## 1. Scope corrections the program needs

1. **Stadler et al., "The Fundamental Limits of Least-Privilege Learning" (ICML 2024)**, https://proceedings.mlr.press/v235/stadler24a.html.
   - Their Theorem 2 shows that no useful representation can bound inference beyond the true label for **every** attribute. The violating attribute S\* is a constructed, "maximally revealing" one.
   - The authors note that leakage of a single attribute can sometimes be restricted (§4).
   - The theorem therefore says nothing about protecting one declared attribute such as SEX, and must not be cited as an impossibility result for this problem.
   - The data-dependent single-attribute trade-offs that do apply are Zhao, Chi, Tian and Gordon (NeurIPS 2020, Thms 3.2–3.3) and Zhao and Gordon (JMLR 2022, statistical parity).
2. **True labels are not a disclosure floor.** The combined manuscript currently says "a release useful for a task reveals what the task's label reveals \citep{stadler2024lpl}". That over-applies Stadler et al.
   - Releases are computed from the inputs, so they can reveal less than the label or more than it. `prototype/DEMONSTRATIONS.json` D2 shows both directions exactly.
   - For a hard decision with error rate ε, any attack's accuracy from the decision differs from the label-only attack's by at most ε.
   - Replacement sentence (reviewer):
     > Stadler et al.~\citep{stadler2024lpl} prove that, when the label posterior is strictly positive, no useful representation can bound inference beyond the task label for \emph{every} attribute; the violating attribute is constructed, and the authors note that leakage of a single attribute can sometimes be restricted. Their label-only adversary (``fundamental leakage'') is the reference we report, as a descriptive comparison rather than a floor: our releases are computed from the inputs alone, so they can reveal less, or more, than the true label does.
3. **Sequential collusion.** `taylor2026collusion` is S. Taylor, P. K. Vippathalla and J. P. Coon, "Adaptive Privacy of Sequential Data Releases Under Collusion", https://arxiv.org/abs/2601.21859 (preprint, v2 July 2026).
   - It protects the whole record under a known distribution, with randomised channels. It is sequential only because future requests are unknown, and it does not claim that sequential design beats joint design.
   - It is not the only collusion-aware release design. Others are Li et al. (TKDE 2012), Wang and Fung (KDD 2006) and DProvDB.
4. **Bibliography fixes.**
   - `tian2025fairnessprivacy` is now IEEE TDSC 22(5):4654–4670 (DOI 10.1109/TDSC.2025.3551157).
   - `wang2019balanced` is ICCV 2019, pp. 5309–5318.

## 2. The adversary is not new

**The recipient** is the auxiliary-data attribute-inference adversary, which reads a sensitive attribute from what a model exposes using labelled auxiliary data:
- Song and Shmatikov (ICLR 2020), https://arxiv.org/abs/1905.11742;
- Jia and Gong, AttriGuard (USENIX Security 2018), https://www.usenix.org/conference/usenixsecurity18/presentation/jia-jinyuan;
- Fredrikson et al. (USENIX Security 2014);
- Yeom et al., https://arxiv.org/abs/1709.01604;
- Mehnaz et al. (USENIX Security 2022);
- Jayaraman and Evans (CCS 2022), https://arxiv.org/abs/2209.01292;
- Wang et al. (ICCV 2019);
- Tian et al., https://arxiv.org/abs/2503.06150.

**The coalition** is the linked-releases and collusion setting:
- Ganta, Kasiviswanathan and Smith (KDD 2008), https://arxiv.org/abs/0803.0032;
- Wang and Fung (KDD 2006), https://www.dmas.lab.mcgill.ca/fung/pub/WF06kdd.pdf;
- Li, Chen, Li and Zhang (TKDE 2012), https://arxiv.org/abs/1104.0459, whose criterion is that colluders do no better than the best single copy;
- Taylor et al. (2026);
- DProvDB (colluding analysts), https://github.com/DProvDB/DProvDB.

**Coalition leakage of one declared attribute does not compose.** Only all-attribute measures such as maximal leakage are subadditive over releases (Issa et al., https://arxiv.org/abs/1807.07878). A coalition can learn SEX when neither release reveals anything alone; Shamir's secret sharing is the canonical structure. Per-recipient protection therefore never certifies the pair.

## 3. Information-theoretic formulations of the declared-attribute problem

| Work | Formulation | What it needs | Relevance |
|---|---|---|---|
| Information bottleneck (Tishby, Pereira, Bialek 1999), https://arxiv.org/abs/physics/0004057 | min I(X̃; X) − β I(X̃; Y) | known law | compression/utility template |
| Privacy funnel (Makhdoumi et al. 2014), https://arxiv.org/abs/1402.1774 | min I(S; Y) s.t. I(X; Y) ≥ R; greedy merging algorithm | known P(S, X) | the declared-attribute funnel. Merging is the only way a deterministic release lowers *Bayes* leakage |
| Calmon and Fawaz 2012, https://arxiv.org/abs/1210.2123 | min leakage s.t. distortion; randomised maps that may read S | known law | a larger achievable region when the map may read S. Our contract forbids that |
| Sankar et al. 2013, https://arxiv.org/abs/1102.3751 | rate–distortion–equivocation with side information | statistics | side information at the adversary |
| Erdogdu and Fawaz 2015, https://www.cs.toronto.edu/~erdogdu/papers/priv_isit.pdf | incremental leakage given earlier releases; zero online regret under an HMM | known law | the same functional as designer 2's target |
| Wang, Basciftci and Ishwar 2017 | S-only ⊆ useful-only ⊆ full-data regions | known law | our deployment is "useful data only" |
| Bertran et al. (ICML 2019), https://proceedings.mlr.press/v97/bertran19a.html; Zhao et al. (NeurIPS 2020) | adversarial obfuscation with trade-off bounds | S and task labels | Zhao's Thm 3.1 assumes no adversary side information |

## 4. Adversarial, fair-representation and tree methods

- **Adversarial or fair representations:**
  - Edwards and Storkey (2016), https://arxiv.org/abs/1511.05897;
  - Xie et al. (NeurIPS 2017);
  - LAFTR (Madras et al., ICML 2018), https://proceedings.mlr.press/v80/madras18a.html;
  - Zhang, Lemoine and Mitchell (2018), https://arxiv.org/abs/1801.07593.
- **Post-hoc recovery after adversarial training:** Elazar and Goldberg (EMNLP 2018), https://aclanthology.org/D18-1002/.
- **Linear erasure:**
  - LEACE (NeurIPS 2023), https://github.com/EleutherAI/concept-erasure;
  - INLP (ACL 2020);
  - log-linear guardedness (ACL 2023).
  - All three state that linear guardedness does not stop nonlinear or multiclass recovery.
- **FARE** (Jovanović et al., ICML 2023), https://proceedings.mlr.press/v202/jovanovic23a.html, code https://github.com/eth-sri/fare.
- **Boosted trees:**
  - FAGTB (Grari et al., ICDM 2019), https://github.com/vincent-grari/FAGTB;
  - FairGBM (ICLR 2023), https://github.com/feedzai/fairgbm;
  - post-hoc leaf editing: EiFFFeL (discrete leaf relabelling); Ren et al. (CVPR 2015), which relearns leaves with the structure fixed.
  - In the sources read, none fine-tunes the continuous leaf scores of a fitted boosted ensemble against online critics on released outputs. That is a bounded-search observation, not a priority claim.

## 5. The two proposals against their closest methods

### Designer 1, MMP: fit with SEX, release the fixed-weight average over SEX

**Closest methods, all established:**
- Žliobaitė (2017, appendix), https://arxiv.org/abs/1708.00754, citing Žliobaitė and Custers (AI & Law 2016): fit with the sensitive attribute, then replace its component by the population-weighted average.
- Pearl's adjustment formula, pr(y | x̂) = Σ_z pr(y | x, z) pr(z) (Biometrika 1995, Thm 1), https://ftp.cs.ucla.edu/pub/stat_ser/R218-B.pdf.
- Pope and Sydnor (AEJ: Policy 2011), https://www.aeaweb.org/articles?id=10.1257/pol.3.3.206. This one is consistent at the level of the abstract; its method details were not verified because the full text is paywalled.

**Related constructs:**
- Kilbertus et al. (NeurIPS 2017), https://arxiv.org/abs/1706.02744: proxy discrimination, which needs a causal graph.
- Kusner et al. (NIPS 2017), https://arxiv.org/abs/1703.06856: MMP is **not** counterfactually fair.
- Lipton, Chouldechova and McAuley (NeurIPS 2018), https://arxiv.org/abs/1711.07076: MMP is a "disparate learning process" in their sense, using SEX in training only.

**Classification:**
- The device and its utility identities are established technique.
- The probability-space mixture of CatBoost components is engineering.
- Using it as a coalition SEX-inference defense under per-task caps is an engineering combination: the endpoint is new, the algorithm is not.
- **No new algorithmic ingredient was found.**

**Closed lanes:** none found. The nearest lane is feature masks, which delete a field's within-SEX task value as well; MMP does not.

**Scope corrections:**
- "Δ is the only route" holds only through the mixture weights.
- "No π-term remains" is a statement about the decomposition, not about information.
- MMP must not be called "fair".

### Designer 2, RCCS: income frozen, propensity-driven occupation counter-shift

**Closest methods:**
- The incremental-leakage release design of Erdogdu and Fawaz, Taylor et al., and Sankar et al. (Thm 3).
- Conditional parity (Ritov, Sun and Zhao), https://arxiv.org/abs/1706.08519.
- Johndrow and Lum (AoAS 2019), https://arxiv.org/abs/1703.04957: adjust each variable conditional on the earlier ones.
- LEACE-type linear criteria.
- Cross-fitted partialling-out (Chernozhukov et al.), https://arxiv.org/abs/1608.00060.
- Li et al.'s "pair no better than the best single copy".
- Within this repository: the minimal task-wire adapter (income frozen, a low-rank occupation shift driven by 64 frozen trunk features) and PCRL's own coalition linear-R² constraint (OWN-03).

**Classification:**
- Mostly established ingredients.
- One combination was not found in the sources read: an explicit cross-fitted SEX-propensity driver feeding a deterministic, attribute-blind output shift to meet a conditional second-order constraint. That observation is unproven and carries no priority claim.

**Closed lanes:** substantial overlap with the small-adapters lane and with the coalition- and conditional-constraint lanes.

**Errors found:**
- The minimal adapter was not pure post-processing.
- With income frozen, the "partial" objective equals the coalition ridge R² plus a constant.
- Deployment computes an explicit SEX proxy, so purpose-limitation language does not fit.

## 6. Joint versus sequential, and randomness

The reviewer confirms the premise in `PROBLEM_SPECIFICATION.md` §9:
- **Deterministic releases.** Under the same information, a sequential designer with full look-ahead realises every jointly designed pair, because min over (f1, f2) equals min over f1 of min over f2. A joint advantage can therefore come only from greedy or online restrictions, computation, or optimisation quality.
- **No cited work claims otherwise.** Erdogdu and Fawaz even show zero online regret under a hidden Markov model.
- **Randomness.** Shared randomness enlarges the achievable set of joint release laws. That is how Li et al. (correlated noise), DProvDB (shared Gaussian synopses) and Taylor's reuse principle limit what colluders gain. It is a contract change here and was not proposed.
