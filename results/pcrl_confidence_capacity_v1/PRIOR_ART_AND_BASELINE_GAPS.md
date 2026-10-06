# Prior art and baseline gaps (confidence capacity and privacy, qpc)

Owner: role C (mathematics, invariants and protocol review). Written 2026-10-06 (UTC), before STAGE_A_LOCK.

**Time box.** About 10 minutes of the allowed 45 were spent on new literature work. Nothing was installed or cloned,
and no Adult data was touched. This file updates the completed source review,
`results/pcrl_decision_preserving_compression_v1/PRIOR_ART_AND_BASELINE_GAPS.md` (dpc @0a7b05a), which remains the
detailed reading of each primary source. Only what is new or re-checked is written out in full here.

## 1. Plain statement

**Every ingredient of this study is prior work.** None of the following may be claimed as ours:

- clustering of probability vectors under KL divergence, and its seeding;
- compression, coarsening or transformation of released confidence scores;
- privacy-funnel objectives (minimise I(S; release) for an attribute the map never reads);
- the data-processing inequality and decision containment;
- sequential releases under a collusion constraint.

**Our sequential arms are matched adaptations, not the Taylor solver.** SEQ-12 and SEQ-21 are deterministic,
class-preserving, greedy-plus-exchange adaptations fitted against the other recipient's class-only release under our
F_joint. They inherit none of the Taylor, Vippathalla and Coon convergence or optimality results.

**No novelty claim is made.** A larger codebook is a rate increase, not an algorithmic contribution. Any measured gain
is an exploratory development result under this contract and attack protocol.

## 2. Sources and what each covers

### 2.1 Carried over from the source review (re-read there, not re-read here)

| Source | What it covers in this study | Source-review section |
|---|---|---|
| Slonim and Tishby, *Agglomerative Information Bottleneck*, NIPS 12 (1999). https://papers.nips.cc/paper_files/paper/1999/file/be3e9d3f7d70537357c67bb3f4086846-Paper.pdf | Greedy merging with closed-form JS merge costs; the FINE-TASK merge step (with smoothing, the identity holds to O(K eps)) | 2.1 |
| Slonim, Friedman and Tishby, *Multivariate Information Bottleneck*, NIPS 14 (2001). https://proceedings.neurips.cc/paper/2001/hash/1113d7a76ffceca1bb350bfe145467c6-Abstract.html | Greedy merging over several compression variables evaluated jointly; the JOINT greedy step | 2.2 |
| Makhdoumi, Salamatian, Fawaz and Medard, *From the Information Bottleneck to the Privacy Funnel*, arXiv:1402.1774 (2014). https://arxiv.org/abs/1402.1774 | Greedy merging that minimises I(S; Y) for an attribute the map does not read; the LOCAL family. Its Theorem 1 bounds a bounded-cost inference gain, not AUC | 2.3 |
| Taylor, Vippathalla and Coon, *Adaptive Privacy of Sequential Data Releases Under Collusion*, arXiv:2601.21859 v2 (2026). https://arxiv.org/html/2601.21859v2 | Sequential releases with per-party and collusion MI constraints; randomised channels; Blahut-Arimoto-style dual ascent | 2.4 |
| Yang et al., *PURIFIER: Defending Data Inference Attacks via Transforming Confidence Scores*, AAAI-23. https://ojs.aaai.org/index.php/AAAI/article/view/26289 | Transforming released confidence scores as a defence; membership-inference target; not class-preserving (label swapper) | 2.5 |
| Stadler, Kulynych, Gastpar, Papernot and Troncoso, *The Fundamental Limits of Least-Privilege Learning*, ICML 2024 (arXiv:2402.12235) | No least-privilege claim; decision containment is the structural floor | 2.6 |

### 2.2 Added for this study (Stage A codebooks and the larger alphabets)

Only standard results are used, and only the stated parts. Bibliographic details were re-checked by search on
2026-10-06.

- **Banerjee, Merugu, Dhillon and Ghosh, *Clustering with Bregman Divergences*, JMLR 6 (2005) 1705-1749.**
  https://www.jmlr.org/papers/volume6/banerjee05b/banerjee05b.pdf
  - KL(p || q) is the Bregman divergence of negative entropy with the centroid q as second argument.
  - For any Bregman divergence the arithmetic mean is the unique minimiser of the summed divergence to a point.
  - So Lloyd-type hard Bregman clustering (assign to nearest centroid, then replace by the member mean) does not
    increase the objective, and stops in finitely many steps.
  - Stage A's DIRECT-TASK k-means is this algorithm, restricted to each teacher-predicted class.
  - Our prototypes are smoothed means, smooth(mean) = (mean + eps 1 + eps e_c)/(1 + (K+1) eps). That is not the exact
    minimiser, so monotonicity holds only up to O(eps). See `MATH_REVIEW.md`, section on convergence.
- **Dhillon, Mallela and Kumar, *A divisive information-theoretic feature clustering algorithm for text
  classification*, JMLR 3 (2003).** KL k-means of conditional distributions with weighted means: the same assignment
  and update rules, in a different application. (Cited from memory of the standard reference; not re-read today.)
- **Arthur and Vassilvitskii, *k-means++: The Advantages of Careful Seeding*, SODA 2007.** D^2 seeding.
- **Nock, Luosto and Kivinen, *Mixed Bregman clustering with approximation guarantees*, ECML PKDD 2008 (LNCS 5212,
  154-169).** k-means++-style seeding with Bregman divergences: sample the next centre with probability proportional
  to the divergence to the nearest chosen centre. The study's "KL/Bregman k-means++" starts are this construction.
  Approximation guarantees for Bregman seeding need conditions (for example mu-similarity) that the probability
  simplex near its boundary does not satisfy uniformly, so **no approximation factor is claimed for our starts.**
- **Plug-in mutual-information bias.** The finite-sample bias of plug-in entropy and MI grows with the occupied
  alphabet, roughly (cells - 1)/(2N) to first order (Miller-Madow; reviewed in Paninski, *Estimation of entropy and
  mutual information*, Neural Computation 15 (2003)). This is why the larger Stage B alphabets make fitted I1, I2 and
  especially I12 less interpretable (prompt section 9). It is standard, not a finding.
- **Confidence-vector coarsening as a membership defence.** Restricting the prediction vector to its top classes or
  coarsening its precision was already discussed as a mitigation by Shokri, Stronati, Song and Shmatikov, *Membership
  Inference Attacks Against Machine Learning Models*, IEEE S&P 2017. Output compression of confidence is therefore
  not new either, even before PURIFIER. (Cited from the standard reference; not re-read today.)
- **Quantisation and rate-distortion.** Lower distortion at more cells is the ordinary rate-distortion behaviour of
  vector quantisation. The capacity curve in `CAPACITY_CURVE.csv` is an empirical description of that trade-off on
  these teachers, not a new result.

## 3. Code availability (re-checked 2026-10-06, 03:56-03:58Z)

| Method | Official code | Pin possible | Status |
|---|---|---|---|
| PURIFIER (Yang et al. 2023) | https://github.com/wljLlla/Purifier_Code (named in the paper) | **No** | **Still EMPTY.** GitHub API, 2026-10-06T03:56Z: `size` 0, `language` null, `license` null, `created_at` 2022-11-29T13:57:41Z, `pushed_at` 2022-11-29T13:57:42Z (unchanged since the source review). The commits endpoint returns HTTP 409 "Git Repository is empty". `updated_at` reads 2024-10-03T23:38:53Z, but `pushed_at` is unchanged, so this is a repository-metadata update (for example a star), not code. The owner's six public repositories include no other copy: `PEI_Code` is also size 0. GitHub repository searches for "purifier confidence score transformation" and "purifier membership inference" return 0 results. |
| Taylor, Vippathalla and Coon solver (2026) | none | **No** | **NOT FOUND.** The arXiv abstract page for 2601.21859 (v1 29 Jan 2026, v2 10 Jul 2026) has no code link; its "Code, Data, Media" tab shows only the generic third-party finders. The v2 HTML contains no repository URL other than arXiv and LaTeXML boilerplate. GitHub repository searches ("Vippathalla", "2601.21859", "adaptive privacy sequential releases", "sequential data releases collusion", "blahut arimoto collusion privacy") return 0. A web search found no repository; the one hit, `richardcui18/sequential-data-attack`, belongs to a different paper (arXiv:2510.24807, an attack on sequential releases). A third-party review page (pith.science, dated 2026-08-03) lists no code either. |

**Consequence.** Neither official method can be pinned, so neither can be run as an official baseline. Any comparison
would be a re-implementation with disclosed changes, as the source review sets out (its section 4). The narrow
registered comparisons (claims A, B and C against our own DIRECT-TASK, FINE-TASK, LOCAL, SEQ-12, SEQ-21, class-only,
continuous sources and admitted FARE, F0 and LEACE outputs) do not depend on either.

## 4. How this study's arms map onto prior work

| Arm | Nearest prior procedure | What changes here | Status |
|---|---|---|---|
| DIRECT-TASK (Stage A) | Bregman hard clustering (Banerjee et al. 2005) per predicted class, with deterministic and Bregman k-means++ starts (Nock et al. 2008) | Class restriction; eps-smoothed prototypes; best-of-three starts by fitting KL; a 200-round cap with a declared stopping rule | Standard algorithm under a constraint |
| FINE-TASK | AIB greedy merging plus exchange refinement | Class-restricted; KL to the teacher's soft vector | Standard |
| LOCAL | Privacy-funnel greedy merging (Makhdoumi et al. 2014, Algorithm 1), penalised | Class-restricted; penalty instead of constraint; S never read by the deployed map | Matched adaptation |
| SEQ-12 / SEQ-21 | The sequential collusion setting of Taylor et al. (2026) | Deterministic tokens; leakage about S rather than the database; penalty instead of constraints; greedy plus exchange instead of BA dual ascent; **stage one fitted against the other recipient's class-only release under F_joint (the sequential correction of this study)** | Matched adaptation, **not the Taylor solver** |
| JOINT | Multivariate-AIB-style joint greedy step | The joint term is leakage to be reduced; five starts with witness dominance | Matched adaptation |

**On the sequential correction specifically.**
- The correction makes stage one condition on information the contract always discloses (the other recipient's
  predicted class). That is closer in spirit to Taylor et al.'s use of previously released information.
- It remains our own deterministic construction. It is not their algorithm, and not their order of releases (in their
  setting a later party's request is unknown when the first release is designed).
- "Beats Taylor" or "a Taylor baseline" must not be written.

## 5. Gaps that remain (updated from the source review, section 4.3)

- **G1 (official baselines).** There is still no runnable official PURIFIER or Taylor solver. Results speak only to
  the registered arms and admitted references.
- **G2 (randomised mechanisms).** No randomised release is in the bank, by design (prompt section 8). Time-sharing or
  randomised channels can reach trade-off points no deterministic map reaches, so nothing here bounds them.
- **G3 (plug-in quantities).** All D and I values are fitted plug-in quantities on OSF_DEFENSE_FIT. With up to 64
  occupation states per class in Stage A, and fine partitions of up to 128 per class in Stage B, the joint
  alphabet can reach hundreds or thousands of occupied pair cells on 15,434 rows. The plug-in I12 bias then becomes
  comparable to the lambda-weighted differences being optimised. A fitted MI decrease is not protection, not a
  population bound and not a reliable ranking by itself.
- **G4 (decision floor).** The chain rule I(S; C1, C2) = I(S; d1, d2) + I(S; C1, C2 | d1, d2) still holds for every
  class-preserving release, so no arm can remove the decisions' SEX signal.
- **G5 (capacity is not an algorithm).** Moving from m2 = 8 to 64 is a rate change. If it passes the confidence gate,
  that is a fact about these teachers' confidence geometry, not evidence for any privacy method.

## 6. What may and may not be claimed

- **Allowed:** "a class-preserving public score code of declared capacity, fitted by standard KL k-means or by
  matched penalised merging, evaluated on a locked, reused Adult development benchmark against matched task-only,
  local and sequential controls."
- **Not allowed:**
  - novelty of KL k-means, Bregman seeding, agglomeration, privacy-funnel objectives, output compression, data
    processing or sequential collusion constraints;
  - "beats PURIFIER", "beats Taylor" or "a Taylor baseline";
  - any population, attribute-specific or training-data privacy guarantee from fitted MI or held-out AUC;
  - an algorithmic contribution from a larger codebook.
