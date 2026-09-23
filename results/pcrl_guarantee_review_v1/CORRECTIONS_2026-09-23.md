# Dated corrections and reconciliation (2026-09-23, claims role)

These supersede specific sentences in earlier reports. **The earlier files are not edited.** Every item
names the exact source text, its owner, the corrected statement and the evidence for it. Historical
decisions (the registered 2016 decisions, and the method study's stop decision) are unchanged.

IDs CR-1 to CR-6 are new. They do not duplicate the claims-audit IDs (T3-1 … T3-17) or the manuscript
owner's C-numbers.

## CR-1 — A measured finite-probe gain is not a Bayes gain and does not set a necessary radius

- **Superseded text (claims role, own files):**
  - MATH_REVIEW.md M3: "Any release whose Bayes gain over H is ≥ g must have κ ≥ g. So a κ-constrained
    release that matches Q's scale carries a sensitive guarantee of about 0.026 nats…"
  - FULL_VIEW_GUARANTEE_REVIEW.md Corollary 1, "Usefulness at the proposed utility level".
  - REVIEW_INDEX.md: "A release that adds Q's measured 2016 task gain (~0.026 nats over H) needs κ ≥ 0.026."
  - The same sentence in my final report of the guarantee review.
- **Correct statement.** κ ≥ I(Y;Z|W) is exact: the Bayes-optimal incremental log-loss gain cannot exceed
  the radius. But the 2016 value ≈ 0.026 nats is CE(H-only probe) − CE(H+Z probe) for two *fitted,
  validation-selected finite* predictors. It can overstate I(Y;Z|H), when the H-only probe is weak
  (claims fixture F9: a gain of 0.6365 nats with zero conditional information). It can also understate
  it, when the augmented probe is weak.
  - It therefore establishes neither the Bayes gain nor a necessary κ threshold.
  - Use it only as a scale: "a radius-κ release cannot provide a *Bayes* gain above κ. Whether κ ≈ 0.026
    would permit Q-like measured gains is untested."
- **Also affects:** the method owner's FULL_VIEW_THEORY §2 cites that number "as a scale comparison" via
  M3. The wording is acceptable, but add "not a Bayes-gain estimate or a threshold".

## CR-2 — "Violated by construction" becomes "not established"

- **Superseded text:**
  - FULL_VIEW_GUARANTEE_REVIEW.md Proposition 3: "The actual code violates the assumption by
    construction."
  - REVIEW_INDEX.md: "The actual code T0 violates homogeneity by construction."
  - REVIEWED_CLAIMS_MATRIX.csv M02: "violated by T0 by construction".
- **Why it was wrong.** T0 = g(PCA32, H_A) depends functionally on H_A. But T ⟂ H | (S,B) is a
  distributional property. Functional dependence does not by itself refute it. For example, if the
  within-bin variation of H_A were independent of everything that changes T given (S,B), the assumption
  could hold.
- **Attempted evidence.** `analysis/pcrl_guarantee_review_v1/homogeneity_test.py` (output
  HOMOGENEITY_TEST.json) tried a likelihood-ratio test of T ⟂ C_fine | (S,B) on Terminal 3's 2018
  aggregate counts. The finer partition is **not nested** in the coarse bins (0/18 role × code × anchor
  tables nest), so the aggregates cannot test the hypothesis. The test needs the joint (S, B, C_fine, T)
  table, which only row-level data provides.
- **Correct statement.** "Within-bin homogeneity T ⟂ H | (S,B) is **not established** for the residual
  code. Its implementation depends on H_A inside each bin, which makes the assumption a strong one, but
  no test has been run. A finite-partition test on held-out rows could refute it but never establish it
  at continuous H."

## CR-3 — The precise impossibility (not "no useful kernel can be certified")

- **Superseded text:**
  - MATH_REVIEW.md M5 item 5: "every kernel with two reachable distinct rows fails".
  - FULL_VIEW_GUARANTEE_REVIEW.md Proposition 4 table: "Impossible for any kernel with two reachable
    distinct rows".
  - fullview_fixtures `g6` "consequence" string.
  - REVIEWED_CLAIMS_MATRIX.csv M11.
  - My final report: "With no restriction on the data distribution, no useful kernel can be certified at
    all."
- **What was actually proved** (G6, G9). Fix P(S|H=h)=π and let P(T|S,h) be unrestricted. The rows of
  the S→Z channel range over conv{Q_t}, and I_π(S;Z) is convex in those rows. So the worst case is exactly
  max over assignments s ↦ t_s of reachable rows of I_π(S;Z). Consequently:
  1. It equals **0 if and only if** all reachable rows coincide, i.e. the channel is constant on
     reachable codes.
  2. It is always ≤ min(R(Q), H(S|H)).
  3. Nonconstant channels can have **small positive** certified worst cases. G9: BSC(1/2−η) with uniform
     S gives log 2 − h(1/2−η) ≈ 2η², for example 2.0e-4 nats at η = 0.01. That is certifiable
     distribution-free, but such a channel carries at most R(Q) (here the same ≈ 2η²) nats of task
     information.
- **Correct statement.** "Under an unrestricted law envelope, *zero* worst-case leakage forces a
  constant channel. A positive budget δ is certifiable exactly when the worst-case assignment
  information is ≤ δ, which small-radius nonconstant channels satisfy, at a correspondingly small ceiling
  on task information."

## CR-4 — Terminal 3's "information-radius bound" is a distribution-dependent mutual information

- **Superseded text (owner: evaluation/diagnosis role, `research/pcrl-objective-diagnosis-v1`
  @ 3e67c52470c16e47afeee42398c64f884a044867):**
  - OBJECTIVE_DIAGNOSIS.md line 31: "The distribution-free, support-aware information-radius bound
    I(S;Z|H) ≤ I(T;Z|H) ≤ Σ_t p_t KL(Q_t ‖ Σ_u p_u Q_u) is 1.384, 1.470, and 1.650 nats for Q, versus
    2.586, 2.535, and 2.584 for D17 on the 2018 mechanism state masses."
  - The field name `information_radius_upper_bound_nats` in RANDOMIZATION_PROFILE.json.
- **What is right.** The calculation itself is valid. Σ_t p_t KL(Q_t ‖ p·Q) = I_p(T;Z) under input law
  p. Under the Markov chain Z–T–(S,H), I(S;Z|H) ≤ I(T;Z|H) = I(T;Z) − I(H;Z) ≤ I(T;Z). So it **is** an
  upper bound on I(S;Z|H) *when p is the true distribution of T in the population in question*.
- **What is wrong.** It is **not** distribution-free and **not** the information radius.
  - At the 2018 empirical masses it is a plug-in estimate for that population, and it changes with p.
    For example, it will differ in 2016.
  - The worst-case, distribution-free radius is the capacity: interval-certified at 1.762 / 1.937 /
    1.968 nats (Q) and log 16 / log 14 / log 15 (D17), per T4_EVIDENCE_REVIEW §2.
- **Requested dated correction**, keeping the calculation. Rename the field to
  `mutual_information_at_2018_state_masses_nats`. Replace the sentence with: "At the 2018 mechanism state
  masses p, I_p(T;Z) = Σ_t p_t KL(Q_t ‖ pQ) is 1.384, 1.470 and 1.650 nats for Q and 2.586, 2.535 and
  2.584 for D17. This upper-bounds I(S;Z|H) for a population whose code distribution is p. It is
  distribution-dependent (the worst-case radius, i.e. capacity, is 1.762–1.968 for Q and log 14–16 for
  D17), and both are far above 0.01."

## CR-5 — Effect-size shrinkage and uncertainty are both true and must be stated together

- **Source (manuscript owner, `research/pcrl-submission-finish-v1` @ 55c0c5a3):** main.tex lines 382–383,
  "The conjunction failed because the effect shrank on a fresh year, not because the test was
  underpowered for the effect it expected."
- **Reconciliation** (from committed 2016 aggregates, PROSPECTIVE_RECOMPUTATION.json):
  - The development estimate −0.0163 did shrink, to −0.0033 / −0.0040 (Q) and −0.0047 / −0.0051 (D17).
  - **Every one of these task point estimates is still beyond the −0.003 margin.**
  - The clauses failed because, at that shrunken size, the one-sided 97.5% bounds (SE 0.0009–0.0011) do
    not exclude values above −0.003: Q's bounds are −0.00151 / −0.00187, and D17's are −0.00291 /
    −0.00295, missing by less than 1e-4.
  - The test was powered for the development effect it planned for, not for an effect at the margin.
- **Replacement for lines 382–383:** "The effect shrank on the fresh year—from $-0.0163$ to $-0.0033$
  and $-0.0040$ nats for $Q$—but both point estimates still lie beyond the margin. What failed is
  resolution: at that size the registered one-sided bounds ($-0.00151$, $-0.00187$; $D_{17}$ within
  $10^{-4}$) cannot exclude a gain smaller than $0.003$. The test was well powered for the development
  effect it planned for, not for an effect at the margin itself."
- **Unchanged:** the registered decision (fail, 8/10), and the margin.

## CR-6 — D17's radius uses its verified reachable actions, not log 17

- **Superseded text:**
  - MATH_REVIEW.md M3 ("up to log 17 = 2.83 nats").
  - FULL_VIEW_GUARANTEE_REVIEW.md Corollary 1 ("up to log 17 = 2.83").
  - REVIEW_INDEX.md ("The evaluated D17 has a vacuous κ (log 17)").
  - REVIEWED_CLAIMS_MATRIX.csv M06.
  - My final report ("its bound is log 17 = 2.83 nats").
- **Correct statement.** D17 uses 16, 14 and 15 distinct actions across the 32 allowed codes (anchors 0,
  1, 2), verified from the archived kernels by SHA-256. Its radius is therefore exactly log 16 = 2.773,
  log 14 = 2.639 and log 15 = 2.708 nats (interval-certified, INTERVAL_RADIUS.json), matching the method
  owner's EXISTING_RADIUS.json. The conclusion "vacuous for both attributes" is unchanged.
- The 17×17 identity kernel in fixture G4 and the kernel test is a **synthetic** deterministic kernel and
  is labelled as such. It is not D17.

## Paper impact

Only CR-5 and the still-open P-1 (main.tex line 365, "capping added recovery") touch the manuscript. The
radius, homogeneity and impossibility statements (CR-1 to CR-4, CR-6) appear only in review and research
reports. The manuscript at 55c0c5a3 makes no full-view guarantee claim.
