# Prior art and baseline gaps

Owner: math/design reviewer (role 3). Written 2026-10-05, before the selection lock (spec section 16). Time box:
about 35 minutes elapsed and well under 0.1 CPU-hour, of the allowed 45 minutes and 1 CPU-hour. Nothing was installed
or cloned, and nothing was run on Adult data.

How the sources were read: a delegated reading pass opened the AIB, AMIB, Privacy Funnel and PURIFIER PDFs
(PURIFIER through arXiv v1, including its appendix) and pages 1-9 of Stadler et al. Taylor et al. v2 was read only
through quotes extracted from the arXiv HTML, whose equations rendered poorly; check numeric details against the PDF
before quoting them. The reviewer re-checked the two code-availability facts directly on 2026-10-05: the GitHub API
record of the PURIFIER repository and the arXiv record of Taylor et al. v2.

## 1. What our study is, in the vocabulary of this literature

For each of two fixed prediction purposes we release a deterministic finite token C_i = g_i(p_i). It is computed
from a frozen teacher's probability vector, and tokens never cross the teacher's predicted classes. The fitting
criteria are

- D_i: mean KL(p_i || decoded prototype);
- the plug-in mutual informations I(S;C_1), I(S;C_2) and I(S;C_1,C_2), with S = SEX, measured on fitting rows.

Development claims rest on held-out attacker AUC, not on mutual information.

Ideas that are established and must not be claimed as ours:

- agglomerative merging with closed-form Jensen-Shannon merge costs (Slonim and Tishby 1999);
- greedy merging over several compression variables at once (Slonim, Friedman and Tishby 2001);
- greedy merging that minimises I(S;Y) for an attribute that the map does not read (Makhdoumi et al. 2014);
- sequential releases under a collusion constraint (Taylor et al. 2026);
- transforming released confidence scores as a defence (Yang et al. 2023);
- the impossibility of least privilege across all attributes (Stadler et al. 2024).

The only possible contribution is empirical: whether a coordinated release that preserves every decision helps under
this particular score-quality contract and attack protocol. That is an application or contract, not a new algorithm.

## 2. Sources

### 2.1 Slonim and Tishby, Agglomerative Information Bottleneck

**Source.** NIPS 12 (1999), pp. 617-623. Full text read at
https://papers.nips.cc/paper_files/paper/1999/file/be3e9d3f7d70537357c67bb3f4086846-Paper.pdf

**Assumptions.**
- Two finite discrete variables X and Y. The input is the empirical joint p(x,y): Fig. 1, "Input: Empirical
  probability matrix p(x,y)".
- The output is a hard partition, which is the beta -> infinity limit of IB (section 1.2).
- There is no adversary and no sensitive variable. Utility is I(Z;Y).

**Algorithm.**
- Bottom-up greedy pairwise merging. The merge cost is Proposition 1,
  delta I = (p(z_i)+p(z_j)) JS_Pi(p(Y|z_i), p(Y|z_j)).
- Ties: "choose arbitrarily".
- The paper gives no optimality guarantee and states that hard partitions are suboptimal (p. 623).

**How we differ.**
- We have two relevance variables, a penalised sensitive term, merges restricted to the same predicted class, a KL
  distortion to the teacher's scores, refinement sweeps after merging, and held-out evaluation.

**Directly applicable.**
- With unsmoothed barycentre prototypes, the increase in our D_i from merging cells a and b is exactly
  (n_a+n_b)/N * JS_Pi(m_a, m_b). This is AIB Proposition 1 with Y set to the teacher's soft label.
- `test_kl_zero_terms_and_merge_cost_sufficient_statistics` checks this numerically.
- The implemented prototypes are smoothed (eps = 1e-12), so the identity holds only to O(K eps).

### 2.2 Slonim, Friedman and Tishby, Agglomerative Multivariate Information Bottleneck

**Source.** NIPS 14 (2001), pp. 929-936. Abstract at
https://proceedings.neurips.cc/paper/2001/hash/1113d7a76ffceca1bb350bfe145467c6-Abstract.html and full text at
https://proceedings.neurips.cc/paper_files/paper/2001/file/1113d7a76ffceca1bb350bfe145467c6-Paper.pdf

**Assumptions.**
- A known joint P(X_1..X_n) and several compression variables T_j.
- The Lagrangian L = I^{G_out} - beta^{-1} I^{G_in}.
- Hard clustering in the experiments. No adversary.

**Algorithm.**
- "The greedy procedure evaluates all the potential mergers (for all T_j) and then applies the best one."
- Theorem 4.1 gives the merge cost in closed form; its proof is deferred in the paper.
- Proposition 4.2 says which other variables' cached merge costs change after a merge.
- Section 7 notes "the only local-optimality of the resulting solutions".

**How we differ.**
- Our joint term I(S;C_1,C_2) is leakage to be minimised, not information to preserve.
- Merges are class-restricted, and the two recipients are fixed.

**Directly applicable.**
- Our JOINT greedy step is a multivariate-AIB-style step. "Evaluate all eligible same-class merges on either
  recipient, including their effect on the complete pair table" (spec section 9) is the same rule as theirs.
- The reduction in I(S;C_1,C_2) from a C_1 merge is the expected conditional JS term of their eq. (6), with B = S and
  T_2 = C_2.

### 2.3 Makhdoumi, Salamatian, Fawaz and Medard, From the Information Bottleneck to the Privacy Funnel

**Source.** arXiv:1402.1774, v5 (30 Sep 2014). Full text read at https://arxiv.org/abs/1402.1774. The arXiv record
gives no venue; an IEEE ITW 2014 appearance is believed but was not verified.

**Assumptions.**
- The Markov chain S -> X -> Y with a privacy mapping P_{Y|X}, so S is not an input to the map.
- "All random variables are assumed to be discrete." Algorithm 1 takes P_{S,X} as known ("Input: R, P_{S,X}").
- A single analyst who may also act as the adversary.
- Leakage is I(S;Y). For log-loss, the gain in inference cost equals I(S;Y) (Lemma 1). For any bounded cost the gain
  is at most 2 sqrt(2) L sqrt(I(S;Y)) (Theorem 1).
- Utility is the constraint I(X;Y) >= R.

**Algorithm.**
- The Privacy Funnel min I(S;Y) subject to I(X;Y) >= R, which is not convex.
- Algorithm 1 merges greedily, starting from the identity map, and returns a deterministic map. Proposition 1 gives
  the merge cost as p(y_ij) times a JS term.
- "There is no guarantee that such a greedy algorithm induces a global optimal privacy mapping" (section IV-A).
- The paper does not claim NP-hardness.

**How we differ.**
- They also use UCI Adult, but with S = (age, income) and X = (age, gender, education), so gender is released rather
  than protected.
- A single recipient with no coalition, MI utility rather than KL to a teacher, no decision preservation, a
  constraint form rather than a penalty, and in-sample MI evaluation.

**Directly applicable.**
- Our LOCAL family is a penalised, class-restricted adaptation of Privacy-Funnel Algorithm 1.
- Our I_i merge costs are their Proposition 1.
- Theorem 1 bounds the gain in a bounded cost over the prior, not AUC. It gives no AUC guarantee for our codes.

### 2.4 Taylor, Vippathalla and Coon, Adaptive Privacy of Sequential Data Releases Under Collusion

**Source.** arXiv:2601.21859 v2 (10 Jul 2026; v1 29 Jan 2026), cs.IT, read through
https://arxiv.org/html/2601.21859v2. The reviewer confirmed on 2026-10-05 that the abstract page has no comments field
and no code link.

**Assumptions.**
- m parties request data in sequence.
- The private variable is the database itself: "We define the private variable as X=(E,J,A)" (section VII-A).
- Releases come from RANDOMISED channels p(r_k | r^{k-1}, x).
- Finite alphabets, with the joint p(r,z,x) as an input. In the experiments it is an empirical distribution over
  discretised records.
- Constraints: I(R_k;X) <= eps_k for each party, and I(R_k, R^{k-1}; X) <= delta_k for collusion.
- "The data handler cannot jointly optimise all releases," because future requests are unknown.

**Algorithm.**
- Dual ascent over (mu_1, mu_2), with Blahut-Arimoto-style alternating minimisation.
- For distortion utility "its BA-style algorithm will converge to the global minimum", with a proof that needs strictly
  positive iterates. For MI utility the result is only a local minimum, with random restarts.
- A grid with bisection and time-sharing hits the targets.
- Experiments use UCI Adult with education, income and age discretised to |X| = 32 cells.

**How we differ.**
- Our leakage concerns a separate attribute S that the map never reads.
- Our releases are deterministic finite tokens. We use a penalty, not constraints, and evaluate with held-out
  attackers.
- We impose class preservation.
- We have a JOINT arm, which their setting rules out by design.
- At fixed multipliers, their stage-2 Lagrangian corresponds to our SEQ stage-2 objective D_2 + lam (I_2/2 + I_12)
  with mu_1 = lam/2 and mu_2 = lam, after swapping the leakage variable from X to S.

**Consequence.** Our SEQ-12 and SEQ-21 arms are deterministic, class-preserving, matched adaptations. They are not
Taylor's algorithm and inherit none of its convergence guarantees. "Beats Taylor" must not be written.

### 2.5 Yang et al., PURIFIER: Defending Data Inference Attacks via Transforming Confidence Scores

**Source.** AAAI-23, vol. 37 no. 9, pp. 10871-10879, DOI 10.1609/aaai.v37i9.26289. The OJS page
https://ojs.aaai.org/index.php/AAAI/article/view/26289 was read for metadata only. Full text read: arXiv:2212.00612
v1, including its appendix.

**Assumptions.**
- A black-box attacker with an auxiliary set.
- The defender trains on a reference set of non-members, and the label swapper also needs the training set.
- The main target is membership inference: NSH, ML-Leaks, Adaptive, BlindMI and label-only attacks.
- Secondary evaluations cover model inversion and one attribute-inference experiment: UTKFace with task gender and
  race (5 values) as the sensitive attribute, where attacker accuracy falls from 31.06% to 20.94% (Table 3).
- No sensitive attribute enters the objective, and there is one recipient.

**Mechanism.**
- A confidence reformer: a CVAE conditioned on the predicted label, trained with L2 reconstruction plus
  lambda * cross-entropy to the predicted label (eq. 1).
- Gaussian latent noise at inference, so outputs are stochastic.
- A label swapper that deliberately replaces some training members' predicted labels with their second-largest class.
- The appendix gives layer sizes, Adam with lr 1e-4 and 300 epochs for UTKFace. It gives no lambda or batch size.

**How we differ.**
- Their output is a continuous vector, not a finite token. The argmax is only encouraged by the cross-entropy term and
  is deliberately flipped by the swapper, so PURIFIER is not class-preserving.
- Their objective has no S or coalition term, and their threat is membership.
- PURIFIER must not be presented as an attribute-inference guarantee.

### 2.6 Stadler, Kulynych, Gastpar, Papernot and Troncoso, The Fundamental Limits of Least-Privilege Learning

**Source.** arXiv:2402.12235 v2 (26 Jun 2024), ICML 2024 (PMLR 235). Pages 1-9 read; the appendix and proofs were not.

**Assumptions.**
- X, Y and S are discrete and finite.
- Assumption A: P_{Y|X} > 0 everywhere.
- The map may be randomised. The adversary is Bayes-optimal, and leakage is I_inf(S;Z|Y).

**Main result (Theorem 2).**
- Under Assumption A, Z cannot both satisfy the least-privilege property with parameter gamma for EVERY attribute and
  have I_alpha(Y;Z) > gamma.
- The paper notes that restricting leakage for a single fixed attribute is sometimes possible, but such
  representations "do not fulfil the LPP" (section 4.2).

**Consequences for us.**
- We may not claim least privilege, or that a release carries "nothing but the task".
- The structural counterpart in our design is that the tokens determine the decision pair pointwise, so
  I(S;C_1,C_2) >= I(S;d_1,d_2). Any attacker that uses only the decisions can be simulated from the tokens.
- Decision-only recovery is therefore a floor for every class-preserving release. `test_chain_rule_identities_on_fitting_law`
  checks this inequality on the fitting law.

## 3. Code availability (checked 2026-10-05)

| Method | Official code | Pin | Status |
|---|---|---|---|
| PURIFIER (Yang et al. 2023) | The paper's ethics statement points to https://github.com/wljLlla/Purifier_Code | none possible | **The repository is EMPTY.** The GitHub API, re-checked by the reviewer, reports size 0, no licence, no language, created 2022-11-29T13:57:41Z and last pushed 2022-11-29T13:57:42Z. The reading pass found the commits endpoint returning "Git Repository is empty", with no tags or releases. Repository and author searches found no other copy. |
| Taylor, Vippathalla and Coon solver (2026) | none | none possible | **NOT FOUND.** The arXiv abstract and the v2 HTML have no code link, and searches of the web and GitHub for title, author, Blahut-Arimoto and collusion terms returned nothing. |

## 4. Could either be added as a correctly scoped secondary comparison?

Short answer: **neither can be added as an official-method comparison.** There is no official code to pin, and every
faithful mapping onto our setting changes the method. The lead's narrow registered comparisons are against our own
explicitly defined LOCAL and SEQ controls. The absence of these two baselines does not invalidate them (spec section
16), but it does rule out any statement that we are better than either of them.

### 4.1 PURIFIER

**What a re-implementation would need.** Re-implementing it from the paper is computationally feasible within 1 CPU-hour.

- Two independent CVAEs, one per purpose.
- Trained on the teacher vectors of OSF_DEFENSE_FIT (2-d and 6-d), conditioned on the one-hot predicted class.
- Architecture: the UTKFace layout [4,32,64,128,2,128,64,32,2] for income. The paper has no 6-class layout, so one
  such as [12,32,64,128,2,128,64,32,6] would have to be assumed.
- Adam with lr 1e-4 and 300 epochs.
- lambda and batch size would have to be chosen on utility only and disclosed.

**Changes that would make it NOT the official method.** Each would have to be stated.

1. The label swapper turned off, because it deliberately flips decisions.
2. A post-hoc projection to force argmax = teacher class. Report the raw violation rate before projection.
3. Deterministic decoding, or a fixed noise seed.
4. Optionally, quantisation to a finite alphabet.

**Further conditions.**
- The paper's assumption that the reference set contains only non-members must be checked: OSF_DEFENSE_FIT must not
  have been used to train the teachers.
- It would be a LOCAL-type comparison with no S term. Any SEX AUC reduction would be a side effect of the 2-d
  bottleneck and the noise.

**Cost estimate (not measured).** 2 purposes x 5 lambda x 3 seeds, at about 0.5-2 CPU-minutes per model, is about
10-40 CPU-minutes, plus FINAL-slate attackers on every output.

**Recommendation.** Do not add it before the selection lock. At best it would be "a PURIFIER-inspired re-implementation
with four disclosed changes", a different method from the one the paper evaluates, and it adds no matched evidence for
claims A, B or C.

### 4.2 Taylor et al.

**What a mapping would require.** The solver itself would take seconds to minutes in numpy, but no faithful mapping
exists.

1. Their leakage concerns the mechanism's own input X. Either S becomes the input, which breaks "the map never reads
   S", or the leakage terms are re-derived as I(R;S) and I(R,Z;S) over fine cells. The second option is still convex in
   p(r|x) for the distortion form, but it means new update equations, so it is no longer their algorithm.
2. Their releases are randomised. They would have to be sampled for held-out AUC, or made deterministic, which is a
   further change.
3. It corresponds only to SEQ-12 and SEQ-21. Their non-adaptive baseline is our LOCAL, and they have no JOINT
   counterpart.
4. Class preservation needs infinite cross-class distortion, a support mask. That breaks the strictly-positive-iterate
   requirement of their convergence proof.

**Recommendation.** Do not add it. The honest label for any such run would be "Taylor-style BA adapted to S-leakage".
That would be a new, unverified method, outside the registered bank.

### 4.3 Gaps that remain

- **G1.** There is no runnable official baseline for (a) confidence-score transformation defences (PURIFIER) or (b)
  sequential-release collusion solvers (Taylor). Results speak only to our registered task-only, LOCAL, SEQ-12, SEQ-21
  and JOINT families, the class-only control, the continuous source scores and the admitted official FARE, F0 and
  LEACE score releases.
- **G2.** No randomised release mechanism is in the bank, by design (spec section 9). The literature treats randomised
  channels as the general case: Taylor et al. use randomised channels and time-sharing, and Stadler et al. allow
  randomised encoders. Time-sharing between deterministic maps already yields tradeoff points that no single
  deterministic map attains. Our deterministic results say nothing about randomised mechanisms.
- **G3.** All information quantities are plug-in quantities on the fitting rows. The literature's guarantees assume a
  known joint distribution. Ours have no population interpretation, and the claims rest on held-out attacker AUC.
- **G4.** Every class-preserving release is bounded below by the decision pair. The chain rule
  I(S;C_1,C_2) = I(S;d_1,d_2) + I(S;C_1,C_2|d_1,d_2) is verified exactly on the fitting law. It means that no family
  in the bank can remove the decisions' SEX signal (spec section 8; compare Stadler et al.).

## 5. What may be claimed

- **Allowed:** "a coordinated, decision-preserving score-compression release, evaluated on a locked Adult development
  benchmark against matched task-only, local and sequential controls."
- **Not allowed:**
  - novelty of agglomeration, multivariate merging, privacy-funnel objectives, output transformation, data processing
    or multi-recipient collusion;
  - "beats Taylor" or "beats PURIFIER";
  - an attribute-inference guarantee derived from PURIFIER;
  - least privilege;
  - any population or nonlinear privacy guarantee from the fitted MI.
- **If the literature is compared:** the comparison must separate (a) our application and contract, (b) our
  adaptations (LOCAL from Privacy-Funnel Algorithm 1; SEQ from the Taylor sequential setting; JOINT from the
  multivariate-AIB greedy step) and (c) any genuinely new algorithmic step. None is identified at present.
  Class-restricted merging is a constraint on known procedures, not a new algorithm.
