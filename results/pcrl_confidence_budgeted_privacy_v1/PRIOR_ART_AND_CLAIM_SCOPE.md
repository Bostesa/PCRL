# Prior art and claim scope (confidence-budgeted privacy compression, cbp)

**Owner and date.** Role E (math and claims reviewer). Written 2026-10-06, before FIT_LOCK.

**Time box.** About 15 of the allowed 45 minutes went on new literature work, between 16:50Z and 17:05Z. Nothing was
installed or cloned, and no Adult data was touched.

**What this file updates.** It starts from the source study's review,
`results/pcrl_confidence_capacity_v1/PRIOR_ART_AND_BASELINE_GAPS.md` (qpc @ 9dd06da). That file, and the dpc review it
cites, remain the detailed readings of each primary source. Only what is new or re-checked is written out in full here.

## 1. Plain statement

**Every ingredient of this study is prior work.** None of the following may be claimed as ours:
- clustering of probability vectors under KL divergence, and its seeding;
- compression, coarsening or transformation of released confidence scores;
- privacy-funnel objectives, which minimise I(S; release) for an attribute the map never reads;
- penalty (Lagrangian) sweeps of a privacy weight, including the intermediate λ grid of this study;
- the data-processing inequality and decision containment;
- sequential releases under a collusion constraint;
- selection with a confidence margin ("headroom"). This is a design choice of this protocol, not a method.

**This study changes no algorithm.** It interpolates the privacy weight at a fixed, already-registered code capacity
(income 8 and occupation 64 states per predicted class). It then selects releases that keep a confidence margin. The
optimiser, objectives, fine partitions and attacker slate are the source study's, unchanged (`MATH_REVIEW.md` section 6).

**No novelty claim is made.** A favourable result would be an exploratory development result about this contract,
this benchmark and this attack protocol.

## 2. Sources

### 2.1 Carried over from the source review (not re-read today)

| Source | What it covers here |
|---|---|
| Slonim and Tishby, *Agglomerative Information Bottleneck*, NIPS 12 (1999) | Greedy merging with closed-form merge costs: the FINE-TASK merge step, and the greedy step of every family |
| Slonim, Friedman and Tishby, *Multivariate Information Bottleneck*, NIPS 14 (2001) | Joint greedy merging over several compression variables: the JOINT-GREEDY start |
| Makhdoumi, Salamatian, Fawaz and Médard, *From the Information Bottleneck to the Privacy Funnel*, arXiv:1402.1774 (2014) | Greedy merging that minimises I(S; Y) for an attribute the map does not read: the LOCAL family. Its Theorem 1 bounds a bounded-cost inference gain, not AUC |
| Taylor, Vippathalla and Coon, *Adaptive Privacy of Sequential Data Releases Under Collusion*, arXiv:2601.21859 (v1 29 Jan 2026; v2 10 Jul 2026) | Sequential releases with per-party and collusion MI constraints, randomised channels and Blahut-Arimoto-style updates. The nearest prior procedure to SEQ-12 and SEQ-21 |
| Yang et al., *PURIFIER: Defending Data Inference Attacks via Transforming Confidence Scores*, AAAI-23 | Transforming released confidence scores as a defence. Its target is membership inference, and it is not class-preserving (it has a label swapper) |
| Stadler, Kulynych, Gastpar, Papernot and Troncoso, *The Fundamental Limits of Least-Privilege Learning*, ICML 2024 | No least-privilege claim is made. Decision containment is the structural floor |
| Banerjee, Merugu, Dhillon and Ghosh, *Clustering with Bregman Divergences*, JMLR 6 (2005) | KL k-means: the DIRECT-TASK code and the fine partitions |
| Nock, Luosto and Kivinen, ECML PKDD 2008; Arthur and Vassilvitskii, SODA 2007 | Bregman / D² seeding of the fine partitions. No approximation factor is claimed |
| Shokri, Stronati, Song and Shmatikov, IEEE S&P 2017 | Coarsening the prediction vector as a membership mitigation. Output compression of confidence is therefore not new |

### 2.2 Standard results used in `MATH_REVIEW.md` (cited, not new)

- **Data-processing inequality and the chain rule for mutual information.** Cover and Thomas, *Elements of
  Information Theory*, 2nd ed., chapter 2. Used in MATH_REVIEW sections 2 and 3.2.
- **Blackwell ordering of experiments.** Blackwell, *Equivalent comparisons of experiments*, Ann. Math. Statist. 24
  (1953). The ROC of a garbling is dominated. This is a population, Bayes-optimal statement only (MATH_REVIEW section 2,
  DP5).
- **Plug-in MI bias.**
  - To first order, the bias is about (cells − 1)/(2N) (Miller-Madow; reviewed in Paninski, *Estimation of entropy
    and mutual information*, Neural Computation 15, 2003).
  - First order underestimates it when cells are sparse. At the i8o64 pair alphabet the synthetic null is 0.198 nats
    against a first-order 0.158 (MATH_REVIEW section 3.3).

### 2.3 Checked for this study

- **Taylor, Vippathalla and Coon, *Realisation-Level Privacy Filtering*, arXiv:2604.08630 (v1 9 Apr 2026; v2 12 Jul
  2026).** Found by today's search; this is the same group as 2601.21859.
  - It is about differentially private stopping filters for adaptive query sequences, with (ε, δ)-DP guarantees.
  - It concerns neither deterministic class-preserving output codes nor attribute leakage measured by attacks.
  - It is not a baseline here. Its arXiv page lists no code.
  - It is recorded so that "sequential release" wording does not appear to ignore it.

## 3. Code availability (re-verified 2026-10-06, 16:53-17:05Z)

| Method | Official code | Can it be pinned? | Status |
|---|---|---|---|
| PURIFIER (Yang et al. 2023) | https://github.com/wljLlla/Purifier_Code, the URL named in the paper | **No** | **Still empty**. GitHub API at 16:53Z: `size` 0, `language` null, `license` null, `created_at` 2022-11-29T13:57:41Z, `pushed_at` 2022-11-29T13:57:42Z, `updated_at` 2024-10-03T23:38:53Z, all unchanged since the source check. The `/commits` endpoint returns HTTP 409 "Git Repository is empty". The owner's six public repositories hold no other copy (`PEI_Code` is also size 0; the others are unrelated). GitHub repository search for "purifier membership inference" returns 0. "purifier confidence score" returns one unrelated repository (`neocode-spec/Legacy-data`, an African-market data-curation project created 2026-08-21). A web search returns only the paper pages, which cite the empty URL |
| Taylor, Vippathalla and Coon solver (2601.21859) | none | **No** | **Not found**. Neither the arXiv abstract page (v2) nor the v2 HTML (493 kB, grepped) holds any code or data repository link; the only GitHub URLs are arXiv/LaTeXML boilerplate. The pith.science review page holds only its own site-metadata links. GitHub repository searches for "Vippathalla", "2601.21859", "sequential data releases collusion" and "adaptive privacy sequential releases" return 0. A web search finds no repository |

**Consequence.**
- Neither method can be run as an official baseline. Any comparison would be a re-implementation with disclosed
  changes, and none is registered.
- The registered comparisons do not depend on either method:
  - claims A, B and C;
  - the admitted references: official FARE, no-fairness FARE (F0), official LEACE, and the RAW-J continuous source.
- The admitted references carry the source study's provenance. They were not re-verified here.

## 4. How the arms map onto prior work

| Arm | Nearest prior procedure | What differs here | Status |
|---|---|---|---|
| DIRECT-TASK i8o64 | Bregman hard clustering per predicted class (Banerjee et al.) with Bregman k-means++ starts | Class restriction, ε-smoothed prototypes, best of three starts | Standard algorithm under a constraint; reused from qpc |
| FINE-TASK | AIB greedy merging plus exchange refinement | Class-restricted; KL to the teacher's soft vector | Standard; reused |
| LOCAL | Privacy-funnel greedy merging (Makhdoumi et al., Algorithm 1), penalised | Class-restricted; penalty instead of constraint; S never read by the deployed map | Matched adaptation |
| SEQ-12 / SEQ-21 | The sequential collusion setting of Taylor et al. (2026) | See the list below | Matched adaptation, **not the Taylor solver** |
| JOINT | Multivariate-AIB-style joint greedy step | The pair term is leakage to be reduced; five fixed starts with witness dominance on the fitting objective | Matched adaptation |
| The λ grid {0.01, 0.025, 0.04, 0.06, 0.08, 0.1} | An ordinary penalty-weight sweep | Fixed before fitting; no per-seed λ; no outcome-dependent pruning | Design, not method |
| Headroom selection (log-loss excess ≤ 0.006, Brier excess ≤ 0.0035 per task and seed) | Margin-based model selection | Registered inner-selection rule motivated by the previous uncertainty; not proved optimal | Design, not method |

What SEQ-12 and SEQ-21 change relative to Taylor et al.:
- the tokens are deterministic, where Taylor et al. use randomised channels;
- the leakage measured is about S, not about the database;
- a penalty replaces their constraints;
- greedy merging plus exchange refinement replaces Blahut-Arimoto dual ascent;
- stage one is fitted against the other recipient's CLASS-ONLY release under F_joint. This is the source's correction,
  and in this design the order of releases is known in advance.

**The sequential arms specifically.**
- SEQ-12 and SEQ-21 are deterministic, class-preserving, greedy-plus-exchange adaptations, fitted under this study's
  F_joint against the other recipient's CLASS-ONLY release.
- They inherit none of the Taylor et al. convergence or optimality results.
- In Taylor et al., a later party's request is unknown when the first release is designed. Here both recipients and
  their order are fixed in advance.
- **They must never be called "the Taylor solver", "a Taylor baseline", or "an official sequential baseline".** "Beats
  Taylor" must not be written.

## 5. Claim scope (stated exactly as the prompt does)

### 5.1 What a claim C pass means, and does not mean

**A C win means useful privacy training beyond compression on this reused benchmark.** Precisely:
- at the fixed capacity i8o64, the headroom-selected privacy nominee P* keeps exact decisions;
- it keeps both tasks' confidence within the ORIGINAL allowances, at the registered uncertainty bounds;
- it lowers combined-view SEX recovery by more than 0.02 pair AUC (lower bound) relative to T*. T* is the strongest
  ordinarily eligible privacy-UNTRAINED release from the closed list DIRECT-TASK, FINE-TASK, U, CLASS-ONLY and F0;
- the upper bound of each recipient's own AUC increase over T* is below 0.01;
- all of this holds under the declared matched attacker slate, on the reused Adult development assessment rows, as an
  exploratory locked comparison.

**It does NOT mean:**
- a new algorithm in the literature;
- joint superiority;
- dominance over every published defence;
- chance-level privacy;
- a population certificate;
- general-purpose representation learning.

**Be equally clear if a simpler family wins.**
- If the winning nominee is LOCAL, SEQ-12 or SEQ-21, the result is reported as that family's win.
- No joint contribution is earned unless claims A and B both pass.
- If sequential matches joint, the simpler empirical story is preferred.

### 5.2 Further limits

**Output-release scope differs from reusable features.**
- The release is a fixed-task OUTPUT: per recipient, a categorical token identity, its decoded codebook probability
  vector and the unchanged teacher decision.
- It provides no reusable features for arbitrary future learning tasks.
- It must not be described as privacy-preserving representation learning, the original PCRL ambition.

**The structural guarantees** (`MATH_REVIEW.md` sections 1 and 2) are:
- exact decision preservation, Theorem 1, under its stated assumptions;
- the standard data-processing facts.

They do NOT imply:
- a SEX AUC bound for any fitted attacker;
- a population MI bound;
- training-data privacy.

They cannot repair a teacher error. For example, occupation class 5 has recall 0 under the teacher and under every
code.

**Fitted MI** is a training criterion on OSF_DEFENSE_FIT:
- It is biased upward by alphabet size and downward by selection.
- A fitted decrease, or a value below its permutation null, is not protection (`MATH_REVIEW.md` section 3.4).

**JOINT dominance**:
- It holds only on the fitting F_joint, over its unchanged witnesses at the same λ.
- It holds by construction, because the witnesses are candidates.
- It does not extend to components, held-out rows, recovery or DIRECT-TASK. On synthetic data, DIRECT-TASK has a LOWER
  fitting F_joint than JOINT.
- JOINT uses more search than the sequential arms (five starts, four consumed witness fits). The comparison is not
  compute-matched.

**Claims A and B, if they pass (JOINT_DEVELOPMENT_CRITERION_MET).**
- They mean that J* beats the strongest eligible nonjoint controls at the fixed rate (C_rate) and over the declared
  bank (C_global), by the registered margins, on this benchmark and slate.
- This holds under a computational asymmetry that favours JOINT.
- It is not a general joint-design contribution, and not superiority over sequential design in general.

**Exposure and confirmation.**
- All roles, the assessment included, have been used historically. The design was motivated by opened results.
- Nominal intervals condition on fitted artifacts and ignore the adaptive history.
- Repeated nominal success on these rows is not confirmation.
- A C pass recommends an independently planned confirmation study. It does not open its data.

**Attacks.**
- Recovery is measured by a declared matched slate.
- Strong recovery is evidence. Weak recovery by this slate is not proof that every attack fails.

**References.**
- FARE, F0, LEACE and RAW-J operate under different contracts: continuous outputs, and in several cases changed
  accuracy.
- Comparisons with them are descriptive (and RAW-J and F0 enter C_global as controls). They are not claims of
  dominance over those methods in general.

### 5.3 Wording per possible label

| Label | Allowed sentence | Must not be written |
|---|---|---|
| PRIVACY_COMPRESSION_DEVELOPMENT_CRITERION_MET | "On the reused Adult development benchmark, the headroom-selected <FAMILY> λ <x> code reduced combined SEX recovery beyond strong task-only compression while keeping both tasks' confidence within the original allowances and every decision unchanged (exploratory, locked; not confirmation)." | "new method", "joint advantage" (unless A and B pass), "private", "certified", "chance-level", "beats PURIFIER/Taylor", "privacy-preserving representation" |
| JOINT_DEVELOPMENT_CRITERION_MET | "J* beat the strongest matched nonjoint controls by the registered margins on this benchmark and slate, under a search that used more starts than the sequential arms." | "joint design is better", "the joint objective causes the gain" (no compute-matched control) |
| CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION | "A decision-preserving code at i8o64 keeps confidence within the allowances; no privacy-training criterion passed." | "privacy compression works", "capacity impossibility" |
| EXPERIMENTAL_NO_ADVANTAGE | "No registered method or feasibility criterion passed on this assessment." | "privacy compression cannot work", "a universal privacy floor" |
| NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE | "No privacy code passed the registered headroom rule; this refutes this nomination procedure, not useful privacy compression in general." | "the λ curve shows no benefit" (unless the inner curve shows it), "a relaxed rule would pass" |
| INCOMPLETE_OR_INVALID | "The <claim> is incomplete because <exact root cause>." | any completed head-to-head negative or positive for that claim |

## 6. Gaps that remain

| Gap | What remains open |
|---|---|
| G1. Official baselines | No runnable official PURIFIER or Taylor et al. solver exists (section 3). Results speak only to the registered arms and the admitted references |
| G2. Randomised mechanisms | No randomised release is in the bank, by design. Randomised channels and time-sharing can reach trade-off points no deterministic map reaches, so nothing here bounds them |
| G3. Plug-in quantities | Every D and I value is a fitted plug-in quantity on OSF_DEFENSE_FIT. At the i8o64 pair alphabet the λ-weighted null level of I12 is comparable to the headroom over the upper half of the grid (`MATH_REVIEW.md` section 3.3) |
| G4. Decision floor | I(S; C1, C2) ≥ I(S; d1, d2) for every class-preserving release. No arm can remove the decisions' SEX signal; the source measured a decision-only pair AUC of about 0.74 |
| G5. Interpolation is not confirmation | Choosing an intermediate λ on rows already used for development is adaptive. If headroom removes the benefit, the prompt directs closing this line, not trying another λ on these rows |
| G6. Compute asymmetry | JOINT is not compute-matched to the sequential arms (`MATH_REVIEW.md` section 5.3) |
| G7. Training-data exposure of the maps | Maps contain fitting-row statistics and SEX-dependent groupings. They are kept private, and no membership or attribute guarantee for the fitting rows is given |

## 7. What may and may not be claimed (summary)

**Allowed:** "a class-preserving public score code of fixed capacity (i8o64), fitted by standard KL clustering and
matched penalised merging at a registered grid of privacy weights, selected with a registered confidence margin, and
evaluated once on a locked, reused Adult development benchmark against matched task-only, local, sequential and joint
controls and admitted continuous references."

**Not allowed:**
- novelty of KL clustering, seeding, agglomeration, privacy-funnel objectives, penalty sweeps, margin selection, output
  compression, data processing or sequential collusion constraints;
- "beats PURIFIER", "beats Taylor", "a Taylor baseline" or "the Taylor solver";
- any population, attribute-specific or training-data privacy guarantee from fitted MI, held-out AUC or the structural
  properties;
- "JOINT dominates DIRECT-TASK", "JOINT is optimal", or any dominance beyond the fitting objective over the unchanged
  witnesses;
- joint superiority without claims A and B, or attributing a joint gain to the joint objective rather than to extra
  search;
- reusable or general-purpose representation learning.
