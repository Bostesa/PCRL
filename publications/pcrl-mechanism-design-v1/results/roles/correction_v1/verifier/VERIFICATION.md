# Independent verification, correction pass (phase 1)

Independent verifier, 2026-10-16. Data-free.

**What I read.** `corrections/CHECKS_PREREGISTRATION.md`, `PROBLEM_SPECIFICATION.md` §1–4 (the metric definitions), the guard module (`mechanism_design/no_real_data.py`) and the queue (`mechanism_design/correction/cqueue.py`, `mechanism_design/jobqueue.py`).

**What I did not read or import while doing the work.** `mechanism_design/correction/exact_checks.py`, `corrections/EXACT_CHECKS.json`, `CORRECTIONS.md`, the math reviewer's or adversary reviewer's correction_v1 folders, `finite_law.py`, or any other campaign metric module. I also opened no array or model file and no `.private/` path.

**How the code was built.** I typed every law in again by hand from the pre-registration. Every metric is my own implementation from its definition, in `fractions.Fraction`. `math.log` is used only for NLL, mutual information, KL and the C2 approximation. The code is `mechanism_design_independent/correction_v1/v_exact_checks.py`. It reads no file and writes only `V_EXACT_CHECKS.json`.

**Not pre-registered.** Items S1–S3 below are supplementary checks that the pre-registration does not contain. I wrote their expected outcomes in the module docstring before the single run. That docstring is not committed or independently timestamped, so treat S1–S3 as disclosed supplementary checks, not registered ones.

## 1. Pre-registered checks: expected outcome, my exact value, pass/fail, and agreement with the hand values

All 16 registered checks pass. All 15 checks that carry a hand value agree with it exactly (C2a has none). Every decimal approximation in the pre-registration is the correct 5-decimal rounding of the exact value.

| Check | Expected (pre-registered) | My exact value | Result | Hand value(s) | Agrees |
|---|---|---|---|---|---|
| A1 | AUC\*(S;R) = AUC\*(S;X); R injective | AUC\*(S;X) = AUC\*(S;R) = **3299/4606** ≈ 0.7162397; 4 distinct release values; P(S=1) = 49/96 | PASS | 3299/4606 ≈ 0.71624 | yes |
| A2 | The η-score reader and its reversal are both strictly below AUC\* | AUC(η) = **1763/4606** ≈ 0.3827616; AUC(reversal) = **2843/4606** ≈ 0.6172384; both < 3299/4606 | PASS | 1763/4606 ≈ 0.38276; 2843/4606 ≈ 0.61724 | yes, yes |
| B1 | R_i(x) = P(Y_i \| X = x) exactly in every cell; metrics equal the Bayes values | Exact equality in 6/6 cells for both tasks. All release vectors lie in the simplex. Income: accuracy **29/40**, summed Brier **73/200**, NLL 0.5428771827 nats. Occupation: accuracy **5/8**, summed Brier **53/100**, NLL 0.9000763390 nats. Release and Bayes values are identical. The smallest probability on a positive-mass label is 1/10, so no floor is needed. | PASS | 29/40, 73/200, 5/8, 53/100 (no hand NLL) | yes ×4 |
| B2 | AUC\*(S;X) = 1 | **1** | PASS | 1 | yes |
| B3 | AUC\*(S;R_1) = AUC\*(S;R_2) = AUC\*(S;(R_1,R_2)) = 1/2 | **1/2**, **1/2**, **1/2** | PASS | 1/2 | yes |
| B4 | 3 distinct pairs for 6 inputs; I(S;X) = h(2/5); I(S;R) = 0 | 3 pairs for 6 inputs. I(S;X) = 0.9709505944546685 bits, against h(2/5) = 0.9709505944546686. I(S;(R_1,R_2)) = 0 exactly: P(r,s) = P(r)P(s) in every cell. | PASS | 3; ≈ 0.97095; 0 | yes |
| C1a | On O = {0,2,4}, f and the identity both pass sample uniqueness | f: 3 distinct outputs for 3 inputs; identity: 3 for 3 | PASS | 3 for 3 | yes |
| C1b | f is 2-to-1 on the support | 3 outputs for 6 cells; preimage sizes 2, 2, 2 | PASS | 3 for 6 | yes |
| C1c | AUC\*(S;f(X)) < AUC\*(S;X) | **1091/1782** ≈ 0.6122334 < **487/594** ≈ 0.8198653 | PASS | 1091/1782 ≈ 0.61223; 1461/1782 ≈ 0.81987 | yes. 1461/1782 is 487/594 written over the common denominator, not an error. |
| C2a | The closed form equals brute-force enumeration at (3,4) and (2,5) | P(3,4) = **77/216** (462 of 1296 samples) by both methods; P(2,5) = **31/256** (124 of 1024) by both | PASS | none registered (equality only) | — |
| C2b | P(1000,20) ∈ [0.88, 0.94]; P(1000,60) ∈ [0.35, 0.47]; P(1000,200) < 10⁻³ | Exact values: P(1000,20) = 0.9093531248; P(1000,60) = 0.4127016496; P(1000,200) = 4.88621751 × 10⁻⁵ | PASS | Intervals from exp(−n(n−1)/4M) = 0.9093729, 0.4127142, 4.7728 × 10⁻⁵ | yes. The approximations also fall inside the intervals; their error relative to the exact value is 2.3% at n = 200. |
| D1 | The release is 2-to-1 | Blocks {0,2} and {1,3}; preimage sizes 2, 2 | PASS | 2 outputs for 4 cells | yes |
| D2 | AUC\*(S;R) = AUC\*(S;X) = 4/5 | **4/5** and **4/5** | PASS | 4/5 | yes |
| E1 | AUC\*(S;block) ≤ AUC\*(S;X) for all 52 partitions | 52 enumerated as restricted-growth strings (52 distinct; Bell(5) = 52 by an independent recurrence). All are ≤ AUC\*(S;X) = **569/833** ≈ 0.6830732. The largest non-equal value is 241/357 ≈ 0.6750700, for {0,1} merged. | PASS | 52 | yes |
| E2 | Equality exactly for the finest partition and {{0},{1,2},{3},{4}} | Exactly these 2; the first is injective, the second is not. Equality ⇔ every block has constant π holds for all 52. | PASS | the 2 partitions | yes |
| E3 | The other 50 are strictly below | **50** strictly below, 2 equal, 52 in total | PASS | 50 | yes |

AUC\*(S;X) = 569/833 for law E is not a registered hand value. My own pencil calculation before the run gave the same number. The smallest partition value is 1/2, for the single block.

**Remarks on the pre-registration text (no failures).**
- **B, "stated, not computed".** "Clause excess ≤ 0 against any reference" holds in expectation under the law: Brier(f) − Brier(η) = E‖f − η‖² ≥ 0, the NLL gap is E KL ≥ 0, and Bayes argmax accuracy is maximal. On a finite evaluation sample, an empirical clause can still come out positive by chance.
- **B, structure.** The counterexample uses a special structure: S is independent of T, and Y is independent of S given T. It refutes the general impossibility claim, as the pre-registration's scope paragraph says, and nothing more.

## 2. C2 cross-method agreement (S1, supplementary)

I computed P(M, n) exactly with four methods:
1. the registered closed form Σ_k C(M,k)(−1)^{M−k} 2^k k^n / (2M)^n;
2. a sequential DP over the n draws, whose state is the number of pairs with one member drawn;
3. a Stirling-number sum, Σ_j C(M,j) 2^j j! S₂(n,j) / (2M)^n;
4. a DP over pairs, A_i(r) = A_{i−1}(r) + 2 Σ_{t<r} C(r,t) A_{i−1}(t).

Brute-force enumeration was added wherever M ≤ 5. Methods 2 and 4 do not use the alternating binomial sum.

| (M, n) | Closed form | DP over draws | Stirling sum | DP over pairs | Brute force |
|---|---|---|---|---|---|
| (3, 4) | 77/216 | equal | equal | equal | equal (462/1296) |
| (2, 5) | 31/256 | equal | equal | equal | equal (124/1024) |
| (1000, 20) | 63-digit / 63-digit fraction ≈ 0.909353124800768 | equal | equal | equal | — |
| (1000, 60) | 195-digit / 195-digit fraction ≈ 0.412701649590048 | equal | equal | equal | — |
| (1000, 200) | 653-digit / 657-digit fraction ≈ 4.88621751392301 × 10⁻⁵ | equal | equal | equal | — |

At further small cases all four methods equal brute force: (1,1) = 1, (1,3) = 1/4, (2,2) = 3/4, (2,6) = 63/1024, (3,3) = 7/12, (3,5) = 29/144, (4,4) = 235/512 and (5,3) = 73/100. The case (3,0) = 1 checks the 0⁰ = 1 convention. Agreement is exact as Fractions everywhere, which confirms the EGF (2eˣ − 1)^M. The full exact fractions are in `V_EXACT_CHECKS.json` under `supplementary.C2_cross_method`.

## 3. Mathematical statements

Setting for (a)–(e): S ∈ {0,1} with 0 < p = P(S=1) < 1.
- For a view V, a test is a measurable φ(V) ∈ [0,1]. Its operating point is (α, β) = (E[φ | S=0], E[φ | S=1]).
- ROC\*_V(α) = sup{β : α' ≤ α}, taken over randomized tests.
- AUC\*(S;V) = ∫₀¹ ROC\*_V.
- For a discrete V, the Neyman–Pearson lemma makes ROC\*_V the piecewise-linear curve through the likelihood-ratio threshold points. Its area is the cross-multiplication formula with ties counted ½, which is what the code implements.

**(a) Data processing — VERIFIED (with conditions).**
- **Proof.** Suppose S → X → R is Markov: P(R ∈ · | X, S) = K(X, ·) with a kernel K that does not depend on S. For any test φ on R, set ψ(x) = ∫φ(r) K(x, dr) ∈ [0,1]. Then E[ψ(X) | S=s] = E[E[φ(R) | X, S] | S=s] = E[φ(R) | S=s]. So every operating point achievable from R is achievable from X, which gives ROC\*_R ≤ ROC\*_X pointwise and AUC\*(S;R) ≤ AUC\*(S;X).
- **Equality is not implied.** C1c (1091/1782 < 1461/1782), B (1/2 < 1) and the 50 strict partitions in E3 all have strict inequality.
- **Conditions.**
  - Binary S with a non-degenerate prior.
  - Any randomness in R is independent of S given X. This is automatic for a deterministic R = f(X).
  - AUC\* means the randomized optimal ROC, with ties counted ½.
- **Scope.** The bound concerns the unrestricted optimal attacker only. A restricted reader can sit anywhere below it, and even below ½ (A2: 0.383).

**(b) Reconstruction ⇒ equality — VERIFIED (with conditions).**
- **Proof.** Suppose g is measurable and g(R) = X a.s. under the population law. That also holds a.s. under each P(· | S=s), since P(S=s) > 0. For any test ψ on X, the test ψ∘g on R has the same operating point, so the X-achievable set is contained in the R-achievable set. Together with (a), the two ROC curves are equal, and so are the AUC\* values.
- **Conditions.**
  - The Markov condition from the setup is needed for the "≤" half. Without it, R = (X, S) has g(R) = X but AUC\*(S;R) = 1.
  - "a.s." must hold under the population law, not on a sample. C1 and C2 show that a sample cannot certify it.
  - For a deterministic Borel f on a standard Borel space, injectivity on a Borel set of full X-measure is enough: by Lusin–Souslin, f(B) is Borel and f⁻¹ is Borel there.
- **Finite instance.** A1.

**(c) Posterior is a function of R ⇒ equality, without reconstruction — VERIFIED (with conditions).**
- **Proof.**
  - Suppose π(X) = h(R) a.s. Under the Markov chain, P(S=1 | X, R) = π(X), so π_R(R) = P(S=1 | R) = E[π(X) | R] = h(R) = π(X) a.s.
  - For any view V, the likelihood ratio dQ₁/dQ₀ is an increasing function of the posterior π_V: L = [π_V/(1−π_V)]·[(1−p)/p], with L = ∞ exactly when π_V = 1.
  - The optimal ROC is the ROC of the randomized threshold test on L, so it depends only on the joint law of (π_V, S).
  - That joint law is the same for V = R and V = X, so the ROC curves and the AUC\* values are equal.
- **Conditions.** The Markov condition is needed; otherwise R could carry S directly. (b) is the special case h = π∘g.
- **Finite instances.** D2: the release is 2-to-1 and both AUC\* values are 4/5. E2: the partition {{0},{1,2},{3},{4}}.

**(d) The converse — TRUE under S → X → R (VERIFIED WITH CONDITIONS); FALSE without the Markov condition.**
- **Claim.** If S → X → R and AUC\*(S;R) = AUC\*(S;X), then P(S=1 | X) = P(S=1 | R) a.s. In particular, P(S=1 | X) is a.s. a measurable function of R.
- **Proof.**
  1. By (a), ROC\*_R ≤ ROC\*_X. Both curves are concave and non-decreasing, hence continuous on (0,1). Their difference is a continuous non-negative function that integrates to 0, so the curves are equal on (0,1).
  2. Together with p, the optimal ROC determines the joint law of (π_V, S):
     - for α ∈ (0,1), its right derivative is the NP threshold, an upper (1−α)-quantile of L under Q₀, so it fixes the Q₀-law of L;
     - ROC\*(0+) = Q₁(L = ∞);
     - Q₁(L ∈ B) = E_{Q₀}[L·1{L ∈ B}] + Q₁(L = ∞)·1{∞ ∈ B};
     - and π_V = pL/(pL + 1 − p).
     Hence π_R(R) and π_X(X) have the same law.
  3. By the Markov property, π_R(R) = E[π_X(X) | R].
  4. Then E[(π_X − π_R)²] = E[π_X²] − E[π_R²] = 0, using orthogonality and equal laws. So π_X = π_R a.s. ∎
- **Finite instance.** E2/E3: for all 52 deterministic releases of law E, equality holds if and only if every block has constant π.
- **Counterexample without the Markov condition.** Take S ~ Bern(½), X = S ⊕ N₁ and R = S ⊕ N₂, with N₁, N₂ i.i.d. Bern(1/5) and independent of S. Both AUC\* values are 4/5, but π(X) is not a function of R, since P(X ≠ R) = 0.32.
- **Caveats.**
  - The converse uses AUC\*, which integrates the whole ROC. It holds for any functional that is strictly monotone under pointwise ROC dominance. It fails for single-operating-point summaries: in S2, 8 of law E's partitions keep the Bayes accuracy for S at 13/20 while AUC\* drops strictly.
  - It is an a.s. statement about the population. It gives no reconstruction of X (D, E2).
  - Equal finite-sample AUC estimates, or equal AUCs for a restricted reader class, do not license it.

**(e) Determinism, continuity and real-valued outputs do not imply injectivity — VERIFIED.**
- **Simplest example.** A constant map R ≡ c is deterministic, continuous and real-valued, and it is not injective on any support with two or more points. Concrete instances: C1's piecewise-constant f, and B's release, which merges z.
- **Continuity can rule injectivity out.**
  - A continuous real-valued map on any set in ℝ^d with non-empty interior, d ≥ 2, is never injective. Restrict it to an open ball: removing one point leaves the ball connected but would disconnect the interval image.
  - More generally, by invariance of domain, no continuous map from an open set in ℝ^d to ℝ^m with m < d is injective.
- **Discrete inputs.** Injectivity is a property of the map on the population support. C1 and C2 show that sample uniqueness cannot certify it.
- **A non-injective map can keep all S-information.**
  - D2: 2-to-1, AUC\* 4/5 = 4/5.
  - E2's second partition.
  - In general, the posterior release π(X): it is many-to-one whenever two cells share a posterior, yet it keeps the entire optimal ROC, by (c).
- **Consequence.** Merging inputs is not protection, and injectivity is not needed for full leakage.

**MMP and proper-scoring identities.**

Notation: (X, S, Y) ~ P with K classes, and e_Y is the one-hot vector. q^s(x) = P(Y | X=x, S=s), defined where P(S=s | x) > 0. π = P(S=1 | X). η = P(Y | X) = (1−π)q⁰ + πq¹ by total probability; this holds for any version of q^s on null sets. m = (1−w)q⁰ + wq¹.

- **Brier(f) − Brier(r) = E‖f−η‖² − E‖r−η‖² — VERIFIED (with conditions).**
  - **Proof.** For σ(X)-measurable f, E[‖f − e_Y‖² | X] = ‖f − η‖² + E[‖η − e_Y‖² | X], because E[e_Y | X] = η and so the cross term vanishes. Hence Brier(f) = E‖f − η‖² + Brier(η).
  - **Conditions.**
    - The expectation is under the evaluation law whose Y | X conditional is η. Under a shifted evaluation law, η must be that law's conditional.
    - f and r depend on X only. They may be fitted, but on data independent of the evaluation data, and are then treated as fixed.
    - Summed Brier over the K columns. For the one-column binary Brier, both sides halve.
    - The identity is about population expectations. Empirical differences add sampling noise, and in-sample fitting breaks the identity.
  - **When S is used.** For predictors that use S the identity fails. S3(v): Brier(q^S) − Brier(η) = −E‖q^S − η‖² = −1747/45000 < 0.
- **Brier(m) − Brier(η) = E‖η−m‖² = E[(π−w)²‖q¹−q⁰‖²] — VERIFIED (with conditions).**
  - **Proof.** The first equality is the lemma above. The second holds pointwise, because η − m = (π − w)(q¹ − q⁰).
  - **Conditions.**
    - The same law as above, with q^s and π the true conditionals of that law.
    - w is fixed, or a function of X.
    - No positivity is needed. Where π ∈ {0,1}, any version of q^s satisfies the identity, but m then depends on the arbitrary version.
- **NLL(m) − NLL(η) = E KL(η‖m) — VERIFIED (with conditions).**
  - **Proof.** E[−log f_Y | X] = H(η) + KL(η‖f) for a probability vector f(X).
  - **Conditions.**
    - The same law as above.
    - Positivity: m_k > 0 wherever η_k > 0. Otherwise both sides are +∞. For w ∈ (0,1) this is automatic, since m_k ≥ min(w, 1−w)·q^s_k. For w ∈ {0,1} it can fail where 0 < π < 1.
    - Natural logs, and no clipping. The specification's floor max(p, 10⁻¹²) leaves the identity intact only if it never binds on a positive-mass label.
- **What changes when q^s are fitted (q̂^s).**
  - m̂ = (1−w)q̂⁰ + wq̂¹ is still an X-only predictor, so Brier(m̂) − Brier(η) = E‖m̂ − η‖² and NLL(m̂) − NLL(η) = E KL(η‖m̂) still hold, for a fit on independent data. Positivity now depends on q̂.
  - The closed form no longer gives the excess. E‖m̂ − η‖² = E[(π−w)²‖q¹−q⁰‖²] + E‖m̂ − m‖² + 2E⟨m − η, m̂ − m⟩, and the cross term can take either sign.
  - Plugging q̂ into the closed form gives the wrong number. In S3(iv) the closed form gives 412261/19440000 ≈ 0.02121, while the true excess is 30871/1215000 ≈ 0.02541.
  - π is still the population posterior, so estimating it adds further error.
  - The clauses compare against a fitted reference r (U0/Ucal), not against η: Brier(m̂) − Brier(r) = E‖m̂−η‖² − E‖r−η‖². This needs the unknown E‖r−η‖², and it can be negative. The NLL clause behaves the same way, with KL in place of the squared distance.
  - Finite-sample estimates add sampling error.
- **Exact numerical confirmation (S3, supplementary).** I checked all three identities on a hand-made 3-cell, 3-class law with π = (1/4, 1/2, 4/5) and w = 1/3:
  - (i) 26671/1215000 = 26671/1215000 = 26671/1215000, exact;
  - (ii) NLL gap 0.0292241156499402 against E KL 0.0292241156499403 (|Δ| = 5.6 × 10⁻¹⁷);
  - (iii) −71/6000 = −71/6000;
  - (iv) the general identity and the decomposition hold exactly, but the closed form with q̂ differs;
  - (v) Brier(q^S) − Brier(η) = −1747/45000 = −E‖q^S − η‖².

**S2 (supplementary).** Exactly 8 partitions of law E keep the Bayes accuracy for S at 13/20 while AUC\* drops strictly; 8 was the expected count. They are the 10 partitions that never merge across π = ½, minus the 2 with constant-π blocks.

## 4. Queue record

| Unit | Stage | Return code | Charged CPU (s) | Wall (s) | Peak RSS (sampled) | Output sha256 |
|---|---|---|---|---|---|---|
| V_CORR_EXACT_CHECKS_V1 | verify | 0 (done) | 2.342 | 2.581 | 17.5 MB | a79c0e5c39f1a03c11152efc5e244f8695cf0a3edc7b2647cfa695350734a9ca |

That was the only run: no failed attempt and no rerun. The process measured 2.313 s of CPU itself. The unit finished at 2026-10-16T12:58:23Z, and the JSON reports 0 errors.

## 5. Comparison with the coordinator's `corrections/EXACT_CHECKS.json`

I opened the file only after sections 1–4 were written. It records unit CORR_EXACT_CHECKS_V1 at 2026-10-16T12:47:21Z, prepared from the pre-registration at commit b2958c76d, with 28/28 sub-checks passing. I did not open the coordinator's code.

**Exact quantities: identical.**
- I compared 20 exact fractions and found no mismatch: A (4 values), B (4 metrics and 4 AUC\* values), C1 (2), C2 small cases (2), D (2), and E's AUC\*(S;X) and largest non-equal value.
- All 52 law-E partitions have the same AUC\* fraction and the same π-constant flag in both files. Both files give the same 2 equality partitions.
- The integer counts are identical: B 6→3; C1 3/3/3 and 3 for 6; D 2.

**Floating-point quantities: equal up to the last bit.**
- C2: the exact large-M values and the exp(−n(n−1)/4M) approximations are bit-identical.
- Logarithmic quantities differ by at most 2.2 × 10⁻¹⁶, which is log-summation rounding: income NLL 0.5428771827370372 against mine 0.5428771827370371; I(S;X) 0.9709505944546687 against 0.9709505944546685. Occupation NLL and I(S;R) = 0 are identical.

**Coverage difference (for the coordinator).** The coordinator's JSON stores the large-M C2 values only as floats from the closed form. It contains no second method for the large cases. My run supplies that: three further exact methods agree with the closed form as Fractions at M = 1000 and n ∈ {20, 60, 200} (section 2).

**Conclusion.** No discrepancy in any registered quantity or outcome.

---

# Phase 2: review of the corrected record (commit `717b566d8` against `12ffa8558`)

Independent verifier, 2026-10-16, about 13:15–13:45Z. Read-only. No numerical job was run in phase 2: unit V_CORR_PHASE2_V1 was not needed.

**Method.**
- I read every change with `git diff 12ffa8558 717b566d8 -- results/pcrl_mechanism_design_v1/`, plus MATH_REVIEW.md, the adversary reviewer's files and the cited sources.
- Three read-only helper forks of this verifier did the mechanical parts, under the same data rules and with no writes in the worktree: one for the patch hashes and line citations, one for the numerical-quotation sweep, and one for the phrase and consistency scan. I checked their key claims myself before including them: the stale hash, the uncovered historical lines, the 0.0183 drop, and five line citations.
- I did the mathematics and the status derivation myself.

## P2.1 Findings, ranked

The counts are 2 MUST-FIX, 15 SHOULD-FIX and 21 MINOR. Line numbers are at `717b566d8`; the working tree is identical. The consistency fork found M1 independently of me.

**MUST-FIX**

| # | Where | Finding | Suggested fix |
|---|---|---|---|
| M1 | `START_HERE.md:11`; `ADVISOR_BRIEF.md:12`; historical `SELECTION_DECISION.md:70`, which no note covers | The headline reason for the status still rests on the withdrawn premise: "…contradicted the part of its rationale that a real-data benefit needs / would need". The original justified this with "fitted CatBoost releases are essentially injective, so they do not merge cells" (`PROTOTYPE_REPORT.md:95`). It contradicts CORRECTIONS row 19 and §7 ("what changes is the stated reason"), the PROTOTYPE_REPORT note (`:137`, which calls the phrase "conditional"), and START_HERE's own banner ("its rationale is corrected"). | "Its pre-registered prototype contradicted the learnability leg of its rationale in a synthetic law, and the verifier's probe showed its effect can go either way. A real-data benefit would need either merging structure in the Florida fields or a learnability effect; which applies, if either, is unknown." Add a bullet on line 70 to the SELECTION_DECISION note. |
| M2 | `CORRECTIONS.md:41`, §1(b) | "Injectivity on a finite sample, or on a declared domain that the population does not fill, is neither necessary nor sufficient for this." The declared-domain half is false. The population support lies inside the declared domain, so injectivity there is **sufficient**, for equality with the AUC\* of the variable on that domain. MATH_REVIEW §1(b) says this correctly. No conclusion changes, but §1 is the authoritative statement. | "Injectivity on a finite sample is neither necessary nor sufficient for equality. Injectivity on a declared domain is sufficient but not necessary. On the encoded domain it gives AUC\*(S; E(X)), not AUC\*(S; X_raw). Declared-domain non-injectivity, as for the pooled encoder, does not imply non-injectivity on the population support." |

**SHOULD-FIX**

| # | Where | Finding | Suggested fix |
|---|---|---|---|
| S1 | `CORRECTIONS.md:49`, §1(d) | "'nearly injective' would give near-equality **only if** the merged fibres have small population mass." The corollary proves sufficiency, not necessity, and necessity is false: merging cells whose posteriors are close costs little AUC\*, however large their mass. MATH_REVIEW's own wording is correct. | "Near-equality is guaranteed when the inputs in posterior-mixing fibres have small population mass; a sample collision count does not measure that mass." |
| S2 | `METHOD_SPECIFICATION.md:43`, §3 | "…the residual AUC\* equals the SEX information of the task-relevant fields **only if** the release is one-to-one in them." Injectivity is sufficient, not necessary; the exact condition is that the release keeps P(S \| task-relevant fields), by §1(c)–(d). The sentence reintroduces injectivity as necessary, against CORRECTIONS §1(c). MATH_REVIEW C5 and E1 carry the same imprecision; they are role text, so this is only noted. | "…equals AUC\* of the task-relevant fields if the release is one-to-one in them, and in general exactly when it keeps P(S \| task-relevant fields)." |
| S3 | `START_HERE.md:12`; `METHOD_SPECIFICATION.md:171`; `PILOT_PROTOCOL_DRAFT.md:11`; historical `SELECTION_DECISION.md:72` and `PROTOTYPE_REPORT.md:125`, endorsed by the note's "It stands" (`:139`) | "Without merging structure, MMP's effect … had no definite sign" misdescribes P7. P7's 3-dimensional releases of 10 fields are not injective: PROTOTYPE_VERIFICATION §3 says "merging through dimension reduction is real". CORRECTIONS §2 itself counts P7 as merging, and METHOD_SPECIFICATION:171 uses "merging structure" in both senses in one paragraph. The verified condition is that no pure proxy is present, which is the wording CORRECTIONS:213 already uses. | "Without pure-proxy fields (all ten task-relevant), MMP's net effect relative to its SEX-blind twin had no definite sign." Qualify "It stands" the same way. |
| S4 | `CORRECTIONS.md:195–200`, §7 | Not every item listed as a rule-4 ground is one. Rule 4 is triggered by the prototype contradicting the rationale. "No novelty" informs the label and does not decide eligibility (`SELECTION_CRITERIA.md:18–20`). The subgroup burden and the strict-cap failure are utility findings. MATH_REVIEW §6 and `HANDOFF.json` `final_status` give the precise derivation: check (a), reinforced by P7. CORRECTIONS also omits that check (a) is a low-power test (`PROTOTYPE_VERIFICATION.md` §4(a): injective release, readers within about 0.001 of Bayes). | "Under rule 4 it follows from the pre-registered learnability check (a) failing: 0.0004 against ≥ 0.005, a low-power test in a law with an injective release. P7 reinforces it. The utility findings and the absence of novelty inform the label but are not rule-4 grounds." |
| S5 | `CORRECTIONS.md:156` (§5) and row 14 (`:98`); compare `ADVISOR_BRIEF.md:41` and `HANDOFF.json:73` (S5 only) | Rule 2 applies only if both proposals are eligible. RCCS fails S5, so rule 1 applies. The SELECTION_DECISION note gets this right ("Even if both had been eligible…"). The RCCS rejection grounds are also stated differently across documents. | "S5 fails as presented, so under rule 1 MMP was the only eligible proposal. Even if both had been eligible, rule 2 would favour MMP (3 SERIOUS objections against 4)." Use the same grounds everywhere. |
| S6 | `START_HERE.md:60` | "All of these statements are pre-registered and exactly checked in `corrections/`" overstates what the replay covers. Checks A–E are finite-law instances. The general statements rest on proofs, and the ACS statements rest on code reading (see P2.6). | "The finite-law instances behind these statements (checks A–E) are pre-registered, exactly checked and independently recomputed. The general statements rest on the proofs in the mathematical review and the verification, and the ACS-specific ones on code reading." |
| S7 | `START_HERE.md:48` | "Where MMP merges inputs, it removes SEX information exactly, at the predicted price" is unscoped. Merging inputs with equal posteriors removes nothing (check D; CORRECTIONS §1). Relative to the twin, the effect can go either way (P7, as line 53 says). | "In the prototype laws, where MMP merged inputs with different SEX posteriors, it removed SEX information exactly, at the predicted price." |
| S8 | `ADVISOR_BRIEF.md:85`; `AGENT_ROSTER.json:106` | Both cite `RESOURCE_INVENTORY.json`, the second its section `correction_pass_2026_10_16`, for correction-pass resources. That file is unchanged since `12ffa8558` and contains no such data. "About 2 hours" is the budget, not a measurement. | Add the section at closeout with measured figures, or cite the correction queue summary. |
| S9 | `CORRECTIONS.md:65`, §2 | The kept-level counts, SCHL 02/03/05 → OTHER, and the 145 columns have no source. All are verified in `results/pcrl_acs_protected_tree_replication_v1/EXTRACTION_SUMMARY.json` at `12ffa8558`: `category_levels_kept`, `feature_names` (145 entries; kept SCHL levels 01, 04, 06–24) and `input_width`. | Cite that file and those keys. |
| S10 | `CORRECTIONS.md:77`, and the plural in row 3 (`:87`) | "Both releases in the verifier's 10-field probe P7" are said to have optimal AUC below 0.6996. Only MMP's exact Bayes values exist (0.5900–0.6346). The twin's optimal AUC was never computed, and its fitted values are lower bounds, so they cannot show it lies below. | "MMP's release in P7 (exact Bayes 0.590–0.635 against 0.6996; the twin's Bayes value was not computed)". In row 3, make clear that "releases" means MMP's three draws. |
| S11 | `CORRECTIONS.md:143`; `ADVISOR_BRIEF.md:38`; `START_HERE.md:82`; `SELECTION_DECISION.md:83` | "Fell by 0.019–0.045" is a rounding slip inherited from `MATH_ASSESSMENT.md` 2.7 (also `DECISION_LOG.md:152` and MATH_REVIEW:384). The source data `roles/math_verifier/V_RCCS_MAP_V1.json` gives a smallest drop of 0.01827 (HGB, 100k rows, g = 0.35, B = 0.005: 0.6770 → 0.6587; also 0.677 → 0.659 in MATH_ASSESSMENT's own table) and a largest of 0.04465. I checked this by direct lookup of the 28 `rccs_minus_base` values. | "0.018–0.045", noting that the source text says 0.019. |
| S12 | `SELECTION_DECISION.md:24` against its note (`:88`); `CORRECTIONS.md:156`; Bottom line 5 | The same file gives RCCS 5 SERIOUS objections (the original S4 row) and 3 against 4 (the note). 4 is the verifier's final count (`DECISION_LOG.md:154`; `MATH_ASSESSMENT.md:38, 388`). | Add that pointer to the SELECTION_DECISION note. |
| S13 | `HANDOFF.json` (draft v2) | `:39` lists AGENT_ROSTER.json as `d44d3d03…`, but the file at `717b566d8` is `d04c0153…`. `head_at_write` is `3d6cb3b7f`, and the correction-commit list omits `717b566d8`. The VERIFICATION.md hash at `:60` (`bbff28b3…`) and the CORRECTIONS.md hash change with phase 2 and §9. The other 34 artifact hashes match. | Refresh at closeout. |
| S14 | `CORRECTIONS.md:242` | It says the verifier's phase-2 review "is §9"; no §9 exists. | Add §9, or point to this section. |
| S15 | Historical files not covered by a CORRECTIONS row or a dated note | Uncovered lines: `roles/adversary_reviewer/PROPOSAL_THREAT_ASSESSMENT.md:22` ("Everywhere else … learnability"; ADVERSARY_REVIEW does not cover it); `DECISION_LOG.md:100` ("genuine where cells merge, learnability elsewhere"); `MATH_ASSESSMENT.md:74` ("The real-data effect MMP can deliver is therefore a fitted-reader learnability effect", the withdrawn conclusion; row 15 cites only `:73`) and `:34`; `roles/designer_1/CROSS_CRITIQUE.md:13` ("the deployed release, which is injective"); `PROTOTYPE_REPORT.md:48` ("i.e. along pure-proxy directions"). | Add these locations to rows 15 and 10, or to the dated notes. |

**MINOR**

| # | Where | Finding and fix |
|---|---|---|
| m1 | `CORRECTIONS.md:43` | "The converse holds, and only under S → X → R" → "holds under S → X → R; without that condition it can fail". |
| m2 | `CORRECTIONS.md:85`, row 1 | Say which equality: AUC\*(S;R) = AUC\*(S;X) exactly when P(S \| X) is a.s. a function of R. The same holds with E(X) in place of X for the first inequality. |
| m3 | `CORRECTIONS.md:86`, row 2 class | The attribution holds if and only if the compared releases have the same optimal AUC. Keeping P(S \| X) is sufficient, not necessary. |
| m4 | `CORRECTIONS.md:94` (row 10); `METHOD_SPECIFICATION.md:44` | "Exist whenever the inputs have more dimensions than the task-relevant part" holds on a continuum (P7). On the discrete ACS support, dimension counting proves nothing (MATH_REVIEW §1(e)2). Scope it to continuous inputs, so it cannot be read as implying that ACS releases merge. "Outside the span" means the orthogonal complement, the directions on which q⁰ and q¹ are constant. |
| m5 | `ADVISOR_BRIEF.md:19, 20, 30, 34` | Line 20: "exactly as much" should be "essentially as much"; fitted readers differ by 0.0004, and only the Bayes values are equal. Line 34: "tested exactly", but the results include fitted readers. Line 19 omits the mixture-coincidence case. Line 30: "seven fixed readers", but the pair bank also contains readers that ignore one recipient. |
| m6 | `ADVISOR_BRIEF.md:29` | "on their declared domain" → "on the declared domain". |
| m7 | Banners: `METHOD_SPECIFICATION.md:3`; `ADVISOR_BRIEF.md:5`; `CORRECTIONS.md:32` | The METHOD_SPECIFICATION banner cites rows 10–12 and §4 and says "four passages". It also edits the §1 classification and §9 (rows 8, 19). The ADVISOR_BRIEF banner omits other §1 bullets and §5. CORRECTIONS:32 says "three overstatements", while §6 lists seven corrections besides the exclusion rationale. |
| m8 | `CORRECTIONS.md:197`; `METHOD_SPECIFICATION.md:169`; SELECTION_DECISION note | Write "registered before it ran; post-hoc relative to the prototype pre-registration" instead of "registered post-hoc". |
| m9 | `CORRECTIONS.md:65, 66, 70` | Prefix the code paths with `acs_replication/`. Use `trees.py:57–59, 83–114`. For "every evaluated model", cite the workers' `load_role` lines (stage_a.py:40–42, stage_b.py:43–48, fare_worker.py:42–43, fagtb_worker.py:43–44, neural_raw_worker.py:45–53, acs_hybrids/common.py:112). Cite neural.py:21–24 for the ReLU networks. |
| m10 | `CORRECTIONS.md:92`, row 8 | The JSON keys are `stage0.per_seed.{0,1}.merge_diagnostic_mmp` and `merge_diagnostic_twin`. |
| m11 | `CORRECTIONS.md:68` | "1–2 MMP collisions per seed" → "2 and 1 fewer distinct pairs than inputs (at least one merge per seed)". |
| m12 | `CORRECTIONS.md:45` | "8 of law E's partitions keep the Bayes accuracy for S" → "8 keep it while AUC\* falls (10 keep it in all)". This matches V_EXACT_CHECKS `keeps_bayes_accuracy_total` = 10. |
| m13 | `CORRECTIONS.md:190` | "references.bib b7043344… (unchanged)" means identical to v1. The patch does change the bib relative to `3b5065805`. |
| m14 | `CORRECTIONS.md:97`, row 13 | The coordinator's closing message is not in the repository. Mark it as not archived. |
| m15 | `CORRECTIONS.md:94, 134` against `:197` | P7 sign conventions are mixed (MMP − twin against twin − MMP). Pick one. |
| m16 | `START_HERE.md:49`; `CORRECTIONS.md:133`; `METHOD_SPECIFICATION.md:160` | The L_CONT_NOPROXY pair values are unlabelled and their order flips. Label them everywhere: twin 0.7211, MMP 0.7207 (`DECISION_LOG.md:137`). |
| m17 | `START_HERE.md:82`; the DECISION_LOG 13:10Z entry; MATH_REVIEW:442 | "That smooth Gaussian law" (singular) against "two smooth Gaussian laws" elsewhere. Use "two settings (g = 0.35, 0.6) of one smooth Gaussian family". |
| m18 | `SELECTION_DECISION.md:83` | "Within the clauses" holds for the strict caps only at B = 0.005. At B = 0.009 it is cap plus headroom. |
| m19 | `ADVISOR_BRIEF.md:84` | 0.42 CPU-h became 0.43 (0.4323 h; 0.43 is the correct rounding) without a note, while `START_HERE.md:90` says no number changed. Note the change. |
| m20 | DECISION_LOG 13:10Z entry | '"if male / if female" implies counterfactuals' reads as an assertion. Write "wrongly suggests counterfactuals". |
| m21 | `ADVERSARY_AND_MOTIVATION.md` §1 and §5 headings; `AGENT_ROSTER.json` | The headings name the advisor. DECISION_LOG says the public build replaces the name; confirm that it does. In the roster, "About four roles" → "Four roles". Also record that the verifier's phase 2 used three read-only helper forks with no numerical job. |

## P2.2 Mathematics (item 1)

**Agreement.** Apart from M2, S1, S2 and m1–m4, CORRECTIONS §1–§2 and §4, and METHOD_SPECIFICATION §2–§4, agree with my phase-1 assessment (§3 above) and with MATH_REVIEW §1, §2 and §4:
- **(a) Data processing:** a bound, which also covers randomness independent of S given X.
- **(b) Reconstruction:** gives equality, and "almost surely" is relative to the population law.
- **(c) Sufficiency:** gives equality without reconstruction (checks D and E).
- **(d) The converse:** holds under S → X → R and fails without it. The noisy-copies counterexample and the Bayes-accuracy caveat are quoted correctly.
- **(e) Injectivity:** the reasons why determinism, continuity and real-valued outputs do not imply it are stated correctly, including the scoping that dimension counting applies on a continuum, not on the discrete ACS inputs.

**MATH_REVIEW's quantitative corollary (CORRECTIONS §1(d)).** I re-derived it. On {X₁ ∉ M, X₀ ∉ M} the two posteriors coincide, so the AUC integrands agree. Each integrand lies in [0, 1]. Hence 0 ≤ AUC\*(S;X) − AUC\*(S;R) ≤ P(M | S=1) + P(M | S=0). It is correct; only the paraphrase in S1 is wrong.

**MMP identities.**
- CORRECTIONS §4 and METHOD_SPECIFICATION §4 state the right assumptions:
  - the same known law;
  - f and r functions of X;
  - positivity, automatic for 0 < w < 1;
  - the 1e−12 clipping;
  - fitted references U0 and Ucal rather than η.
- The quoted illustration (0.02121 against 0.02541) is my S3 value, correctly labelled post-hoc.
- I also checked MATH_REVIEW's additional bound, E KL(η‖m) ≤ I(S;X) + KL(Bern π̄ ‖ Bern w): data processing for KL through s ↦ q^s, and the algebra checks out.

**METHOD_SPECIFICATION §2.** "Hypothetical product law P(x)P(s)P(y \| x, s)" is correct: m with w = π̄ is its Bayes predictor. The classification paragraph's adjustment-formula sentence (back-door condition plus positivity; no causal reading claimed) is correct and agrees with MATH_REVIEW §4(c).

**Population non-injectivity: not claimed anywhere.** I grepped the changed files for injective, one-to-one and pooled wording:
- START_HERE.md:57, ADVISOR_BRIEF.md:29, CORRECTIONS.md:21, 65, and HANDOFF.json:68, 99 claim non-injectivity only "on the declared domain".
- They mark the population question as unknown. HANDOFF.json:99 adds "plausible for POBP; not verified".
- No changed file claims that the ACS models are proved non-injective on the population.
- The declared-domain claim is correct: SCHL codes 02, 03 and 05 are valid codes that all map to OTHER (EXTRACTION_SUMMARY.json). The one place that could be read in the opposite direction is the dimension-counting sentence in row 10 (m4).

## P2.3 Status derivation (item 3)

**The status follows, without the withdrawn claim.**
- **Selection.** RCCS fails S5 as presented: its own feasible case violates the binding clauses (3.75 pp accuracy), and its accounting is inconsistent. So MMP is the only eligible proposal, under rule 1. The S6 re-classification to "unresolved" does not change eligibility.
- **After selection.** Rule 4 is triggered by pre-registered check (a): L_CONT_NOPROXY, 0.0004 against ≥ 0.005, registered in advance as a contradiction of the rationale. So MMP "is not described as ready for a real-data pilot".
- None of this uses injectivity of any ACS release.
- **The label.** `NO_DEFENSIBLE_NEW_MECHANISM_IDENTIFIED` is rule 3's label, applied here with a qualifier to a rule-4 outcome. CORRECTIONS §7 and HANDOFF `final_status` disclose this and state the literal outcome. That is acceptable. Rule 3's own consequences, such as "prototype restoration and pilot preparation … not applicable", did not apply.
- **Precision of the grounds.** The listing in §7 needs S4: the precise derivation is check (a), reinforced by P7, and the low-power caveat should be stated.

**The four distinctions are drawn correctly.**
- No competitive method has been established: correct.
- Candidates were declined under the campaign's criteria: correct. MMP under rule 4; RCCS under S5 as presented.
- Specific hypotheses were contradicted under specified synthetic laws: correct. The learnability prediction failed in L_CONT_NOPROXY, and P7's registered prediction failed in 2 of 3 draws. MATH_REVIEW also lists the subgroup expectation and RCCS's feasible case.
- No universal impossibility holds: correct. No theorem exists in the record, check B refutes the general form in a constructed law, and PROBLEM_SPECIFICATION.md:142, 151 deny such a theorem.

**Wording to fix.** The sentences in M1 still state the old reason in the two most visible summaries.

## P2.4 Manuscript patch (item 5): PASS

- **Patch identity.** `MANUSCRIPT_ADVERSARY_PATCH.diff` and the reviewer's `MANUSCRIPT_ADVERSARY_PATCH_v2.diff` have the same sha256, both at `717b566d8` and in the working tree: `3d296d2be026c381d8c5b3fa7bd6b3c6dc1d7ecd51d0f3efc32526f2b9d4e068`.
- **Clean application.** On a fresh extraction of `3b5065805:papers/combined_empirical_v1/{main.tex, references.bib}` into an empty scratch directory, `patch -p1 --dry-run`, `git apply --check -v`, `git apply -v` (outside any repository) and `patch -p1` on a second copy all apply every hunk with no fuzz, no offsets and no rejects.
- **Resulting files.** `main.tex` = `cf0b25f2323b8b78c23af915505d405473a0b3df330f443726e7d44e28918c86` and `references.bib` = `b7043344021b75388abf43e5292305eb6ae13c2c9b309ee793830c616881fea1`, as expected.
- **Note.** v1 gives the same `references.bib` and a different `main.tex` (`fedd71ed…`). So "unchanged" at CORRECTIONS.md:190 means "the same as v1" (m13).

## P2.5 Numerical quotations and line citations (item 2)

**Numbers.** Apart from S10, S11 and m11, m12 and m19, every number checked in the changed text matches its source.
- **Exact checks** (CORRECTIONS §8, START_HERE, the DECISION_LOG entries): they match both exact-check JSONs. §8's roundings (0.7162, 0.3828, 0.6172, 0.612, 0.820, 0.971 bits, 0.909, 0.413, 4.9 × 10⁻⁵) are correct.
- **Prototype laws:** L_PROXY 0.767 → 0.500 and L_MIXED 0.717 → 0.650 come from PROTOTYPE_RESULTS. L_CONT_PROXY Bayes 0.7878 → 0.7216 (rounded 0.788 → 0.722) uses the verifier's exact quadrature values, not the coordinator's MC values, at `PROTOTYPE_VERIFICATION.md:92–93`. L_CONT_NOPROXY 0.7211 / 0.7207 has a gap of 0.0004.
- **P7:** MMP Bayes 0.6026 / 0.6346 / 0.5900, written as 0.590–0.635; AUC\*(S;X) = 0.6996; twin − MMP +0.033 / −0.018 / −0.019 at 1M rows.
- **Subgroup:** 2.2–2.6× the guard (0.0111 and 0.0130 against 0.005).
- **RCCS:**
  - 0.615 → 0.644; +0.021 to +0.053 (0.678 → 0.699, 0.673 → 0.726);
  - |Δ| ≤ 0.0012 and 0.008–0.022 against MIN;
  - "up to 0.11 → 0.002–0.031"; 3.75 accuracy points;
  - "21% to over 100%" (designer 1 `CROSS_CRITIQUE.md:27`).
- **Population and encoder:** 74,091 people; 8,001 / 3,000; 145 columns; kept levels COW 8 / MAR 5 / POBP 105 / SCHL 21 (EXTRACTION_SUMMARY.json).
- **Rehearsal:** 800 rows; 799 inputs; 797 / 798 and 799 / 799 pairs (PILOT_REHEARSAL_RESULTS.json).
- **Caps and the patch:** the 24 caps and the 1e−12 floor; the patch hashes, 11 pages, 0 warnings, and the 42-row adversary table.
- **Campaign resources:** 0.43 CPU-h, about 2 h elapsed, and the 0.25 h allowance (RESOURCE_INVENTORY.json: 0.4323 h, 1.863 h, 900 s).
- **HANDOFF v2:** 34 of 35 artifact hashes, and every commit hash and timestamp.
- **Unverifiable without `.private`** (not errors): the coordinator unit's "0.06 CPU-seconds", the correction pass's CPU, and the roster's durations.

**Line citations.** All 47 original-claim line citations in CORRECTIONS.md point to the quoted text at `12ffa8558`, far more than the 10 spot checks requested. I re-checked DECISION_LOG:93, ADVISOR_BRIEF:56, METHOD_SPECIFICATION:154, PROBLEM_SPECIFICATION:125, SELECTION_DECISION:51 and main.tex:29 myself.
- **Decision log and summaries:** DECISION_LOG :93, :112, :143, :145, :161; START_HERE :46–49; ADVISOR_BRIEF :10, :24–29, :43, :45–50, :56, :61; METHOD_SPECIFICATION :27, :38–42, :143, :154, :158; PILOT_PROTOCOL_DRAFT :5, :79; PROTOTYPE_REPORT :95; SELECTION_DECISION :26, :35, :51; ADVERSARY_AND_MOTIVATION :16, :19, :27, :28, :32, :88.
- **Patch, specification and manuscript:** v1 patch :55; PROBLEM_SPECIFICATION :125 (in the §7 list from :116), §3 (:41), §9.4 (:150); main.tex :29 at `3b5065805`.
- **Role files:** MATH_ASSESSMENT :73, :189; designer_2 PROPOSAL :52; designer_1 PROPOSAL :75; PROPOSAL_THREAT_ASSESSMENT :69–70.
- **Code citations** (all in `acs_replication/` at `12ffa8558`) support the stated facts, with partial support only where m9 says so:
  - schema.py :37, :39, :43, :55, :59;
  - extract.py :109–133, :133, :202, :210;
  - deploy.py :29–41;
  - trees.py :85–99;
  - stage_b.py :1–9;
  - fare_worker.py :63–69;
  - neural.py :72, :86.
- **Unverifiable:** the "closing message" in row 13 (m14).

## P2.6 Replay versus interpretation (item 6)

**Validated by numerical replay (exact recomputation or mechanical re-execution):**
- **Checks A–E.** All 16 registered expectations and 15 hand values, every exact fraction, and the AUC\* of all 52 partitions. My own exact code, unit V_CORR_EXACT_CHECKS_V1, in phase 1. They agree with the coordinator's unit.
- **C2.** Four exact methods plus brute force.
- **Supplementary S2 and S3.** The Bayes-accuracy count of 8, and the identity values, including 0.02121 against 0.02541.
- **The manuscript patch.** Re-applied to a fresh extraction, with all three sha256 values recomputed.
- **Line and code citations.** Mechanical lookup at `12ffa8558`. I spot-checked five lines and main.tex:29 myself.
- **Rehearsal counts (800 rows; 799 inputs; 797/798 and 799 pairs) and the encoder facts (145 columns; kept levels).** Read from the JSON records, not recomputed.

**Validated only by reading and argument:**
- **Statements (a)–(e) as general theorems.** These rest on proofs: VERIFICATION §3 and MATH_REVIEW §1. The finite checks only instantiate them.
- **The MMP identities, in general and with their assumptions.** S3 checks one law only.
- **Everything ACS-specific.** The shared encoder, the pooling and declared-domain non-injectivity, the piecewise-constant tree arms and the precision path all come from reading code and provenance.
- **The status derivation.** From reading SELECTION_CRITERIA and the decision records.
- **The RCCS reconciliation.** Numbers quoted from the earlier verifier's records, checked against those records but not recomputed in this pass.
- **Prototype and P7 values.** Checked against PROTOTYPE_RESULTS, PROTOTYPE_REPORT and PROTOTYPE_VERIFICATION, not recomputed here.
- **Classifications and replacement wording.** Interpretation throughout.

**Not validated at all, and correctly marked UNKNOWN in the record:**
- injectivity or posterior preservation of any actual ACS release on the population;
- AUC\*(S;X) for ACS;
- the merging structure of the Florida data and the sign of MMP's effect there;
- feasibility of any many-to-one release at the 24 caps.

## P2.7 Consistency (item 4)

**Clean.** I scanned START_HERE, ADVISOR_BRIEF, CORRECTIONS, METHOD_SPECIFICATION, PILOT_PROTOCOL_DRAFT, ADVERSARY_AND_MOTIVATION, HANDOFF, AGENT_ROSTER and the appended notes.
- **Pilot:** no document implies it is authorised. All say "not run", "not recommended as a competitive test" and "not authorised by the correction".
- **Competitive claim:** none is made or implied (`HANDOFF.json` `competitive_method = NONE_ESTABLISHED`).
- **Population non-injectivity:** not claimed anywhere (P2.2).
- **Listed phrases that do not survive unmarked:**
  - "no deterministic release can reduce leakage";
  - "a contract change is needed" for coarsened probability outputs;
  - a universal privacy–utility limit;
  - "if male / if female" as a counterfactual (one log wording, m20);
  - "removal only via pure proxies" (except the historical `PROTOTYPE_REPORT.md:48`, S15);
  - RCCS "only linear/quadratic" (ADVISOR_BRIEF:37 correctly limits it to the objective);
  - "other access = provider breach";
  - "no attacker can be calibrated";
  - "pooling cannot be forbidden";
  - "none knows the training objective".

**Survivors.**
- **"Essentially injective" reasoning:** only through M1 ("the part … a real-data benefit needs") and the uncovered historical lines in S15.
- **"All differences are learnability":** only in the uncovered historical lines (S15).
- **MMP's exact price:** stated with scope everywhere I read.
- **Internal contradictions:**
  - M1 against row 19, §7 and the PROTOTYPE_REPORT note;
  - S3 ("merging structure" in two senses);
  - S12 (5 against 4 SERIOUS objections);
  - m15–m17 (sign conventions, value order, one law or two);
  - S5 (RCCS rejection grounds).
- **Historical coverage:** apart from S15, the earlier decision-log entries and role statements are covered.
  - DECISION_LOG :93, :112, :143, :145, :161 by rows 1/2, 14, 9, 8, 13.
  - SELECTION_DECISION :26, :35, :51 by rows 14 and 11 and the dated note.
  - PROTOTYPE_REPORT :78, :87, :95 by its note.
  - MATH_ASSESSMENT :73, :189, designer 2 :52 and designer 1 :75 by row 15.
  - The adversary reviewer's drafts by ADVERSARY_REVIEW rows 39–42.

## P2.8 Overall

- **Mathematics:** correct in substance, but M2, S1 and S2 state a sufficient condition as a necessary one, or the reverse. Fix them before publication.
- **Status:** `NO_DEFENSIBLE_NEW_MECHANISM_IDENTIFIED` for a real-data pilot follows from rules 1 and 4 without the withdrawn claim. The four distinctions are drawn correctly.
- **Headline summaries:** the most visible remaining defect is M1, where START_HERE and ADVISOR_BRIEF still give the old, injectivity-based reason.
- **Numbers and citations:** correct apart from S10, S11 and the MINOR items.
- **Manuscript patch:** passes.
- **Process:** no numerical queue unit was run in phase 2. V_CORR_PHASE2_V1 is unused; the only computation was a lookup of 28 stored values in `V_RCCS_MAP_V1.json`.
