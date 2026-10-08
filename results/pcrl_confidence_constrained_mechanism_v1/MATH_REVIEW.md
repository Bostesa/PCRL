# Math review: ccm utility contract G(d = 0.005, b = 0.0025) (role B, independent)

| Item | Value |
|---|---|
| Reviewed | PROTOCOL.md §2–§6 (the version that includes F3u) and UTILITY_CONTRACT.md §1–§5, as on disk on 2026-10-08 |
| Scripts | `math_review/_mr.py` (shared predicates) and `r1_…py` to `r9_…py`. They use numpy, scipy and the standard library (`fractions`, `decimal`). Data is SYNTHETIC only, and nothing imports `ccm.*` or `hcal.*`. |
| Run | `OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python -P results/pcrl_confidence_constrained_mechanism_v1/math_review/rN_….py`. Every script finishes in 1–20 s, so none was run under sema. |
| Exactness | "Exact" means rational arithmetic on the exact binary values of the floats. exp(±d) is bracketed to 1e-50, and logs use 60-digit decimals. |
| Notation | D := e^d − 1 = 0.0050125. M_k := max over bin members of p_k. δ := q − p. S := ‖q‖² − ‖p‖². Brier(q, y) := Σ_k (q_k − 1[k = y])². |

Tags: PROVED · CHECKED-NUMERICALLY · COUNTEREXAMPLE · BLOCKING-ISSUE · NOTE.

## Verdict

There are **two BLOCKING-ISSUES**. Both are text or specification issues and each has a one-paragraph fix (§10). Everything else in
§3–§6 is mathematically correct as stated, apart from the scope notes below.

- **B1 (R3, R8):** "release Ucal(p) itself, which satisfies G trivially (q = p)" is false when Ucal(p) has an exactly
  tied top, because strict Class then fails. Such ties come from exact U0 ties. Tempering can also create them from
  U0 tops that are 1–3 ulps apart. In synthetic probes, 1,180 of 21,883 such near-tie vectors became tied while still
  passing hcal's numpy-argmax check.
- **B2 (R1.5, R2.6, R4.4):** the contract says to certify "in float64 with no tolerance" but never fixes the float
  predicate. Algebraically equivalent forms disagree at the boundary, which is exactly where NLL-tight representatives
  sit. In 84% of NLL-tight K=6 bins, the multiplicative and log forms give different verdicts. The Brier forms (dot,
  per-row difference, fsum) disagree on 46% of Brier-tight vectors. A float-certified q fails the exact check. The
  simplex tolerance for q and the margin for the closed-form infeasibility test are also unspecified. Roles C and E
  can therefore disagree on certification.

**Exact diameter result (R5).** Under NLL + Brier + Class jointly, the supremum of an admissible bin's TV-diameter is
exactly D = e^d − 1 = 0.0050125. This is the same as under NLL alone: the Brier guard does not lower the worst case. The
supremum is attained in K = 2 by representatives c ∈ [0.854, 0.995] and is certified exactly at c = 0.9.

The diameter does depend on location. For K = 2 the bin width is W(c) ≈ min{D, b / (4c(1 − c))}:

| c | W(c) |
|---|---|
| 0.51–0.55 | 0.0025 |
| 0.70 | 0.0030 |
| 0.80 | 0.0039 |
| ≥ 0.854 | 0.0050 |

Any admissible output q, whether deterministic or randomised, pins every coordinate of p to the interval
[e^d·q_k − D, e^d·q_k]. That interval has width 0.0050. The release is therefore a quantization of p at resolution
≈ 0.005 (≈ 0.0025 in the Brier-binding binary range).

Covering one whole income class needs at least **147** admissible bins, against a capacity of 8 (**100** under NLL
alone). Covering a uniform occupation class region needs at least **5.3·10¹⁰** bins, against 64.

**Headroom result (R6).** Two assumptions are needed: (i) the deployed release reads only p and independent randomness,
and (ii) P(SEX | p) is L-Lipschitz in TV. Under them, every admissible release has SEX Bayes accuracy within L·D of
releasing p itself, and so within L·D of every other admissible release. A 0.02 pair benefit therefore needs L ≥ 4.
Without (ii), a pathological law loses all information in one admissible bin. Without (i), admissibility does not bound
leakage from above.

---

## R1. NLL guard (r1_nll_guard.py)

**R1.1 PROVED.** Suppose q_k ≥ e^{−d}·p_k for all k. Then for every y with p_y > 0, log(p_y/q_y) ≤ d.
- If p_y = 0 and q_y > 0, the excess is −∞.
- If p_y = q_y = 0, it is 0/0. The contract's "positive probabilities" premise excludes that case, and the clipped form
  gives 0.

**R1.2 PROVED.** The clipped version also holds. Let ℓ_c(x) = −log clip(x, c, 1) with c = 1e-12. Then
ℓ_c(x) = −φ(log x), where φ(u) = max(log c, min(0, u)) is non-decreasing and 1-Lipschitz. So
ℓ_c(q_y) − ℓ_c(p_y) ≤ max(0, log p_y − log q_y) ≤ d. Each regime follows:
- p_y ≤ c: excess ≤ 0.
- q_y < c ≤ p_y (the contract's case): excess = log(p_y/c) < log(p_y/q_y) ≤ d.
- Both ≥ c: excess = log(p_y/q_y).
- Zeros: excess 0.
- q_y > 1 by rounding: clipped to 1.

CHECKED-NUMERICALLY: 14,750 label pairs in 60-digit arithmetic, covering all four regimes (3,000 targeted at
q_y < c ≤ p_y), with 0 violations.

**R1.3 NOTE.** The converse fails. Take p = (1 − 1e-13, 1e-13) and q = (1, 0). The clipped excess is ≤ d for every label,
yet NLL fails. G's NLL is strictly stronger than "clipped excess ≤ d at every label", which is the intended direction.

**R1.4 CHECKED-NUMERICALLY.** There is no underflow.
- Under the frozen log-input rule with α ∈ [0.25, 4], the smallest Ucal entry is ≈ (1e-12)^α ≥ 1e-48, a normal float64.
- e^{−d}·p never underflows.
- Zeros or subnormals can occur only on the α = 1 path, where p = U0 exactly.

NOTE: on α = 1 rows, the premise "positive probabilities" rests on U0. A label-free count of zero entries is advisable.

**R1.5 BLOCKING-ISSUE (part of B2).** Take boundary constructions q_k = fl(fl(e^{−d})·p_k) on 200,000 coordinates.

| Predicate | Pass rate |
|---|---|
| Multiplicative form `q >= exp(-d)*p` | 100% |
| `p/q <= exp(d)` | 58.9% |
| `log p − log q <= d` | 63.1% |
| Clipped form | 65.2% |
| Real arithmetic (q ≥ e^{−d}p exactly) | 54% |

The real-arithmetic failures are by at most 9.3e-17 in log terms. The float-evaluated log excess exceeds d by up to
2.6e-15. For realistic NLL-tight representatives (q = e^{−d}M + slack·e_d, K = 6), mult-form and log-form certification
disagree on 16,782 of 20,000 bins. The implication "log(p_y/q_y) ≤ d" holds in real arithmetic. In float it holds only up
to ≈ 3e-15.

## R2. All-label Brier guard (r2_brier.py)

**R2.1 PROVED.** Brier(q, y) = ‖q‖² − 2q_y + 1. So Brier(q, y) − Brier(p, y) = S − 2(q_y − p_y): the contract's formula,
full Brier with no ½ factor, matching `dpc.utility.per_row`. CHECKED-NUMERICALLY: max difference 7.8e-16.

**R2.2 PROVED.** The averages and the maximum over labels are:
- E_{y∼p}[excess] = ‖q − p‖²
- E_{y∼q}[excess] = −‖q − p‖²
- max_y excess = ‖δ‖² + 2(p·δ − min_y δ_y) = ‖δ‖² + 2·Σ_k p_k(δ_k − δ_min)

So all-label implies expected. The converse is a COUNTEREXAMPLE: p = (0.99, 0.01), q = (0.995, 0.005) has
‖q − p‖² = 5e-5, but the label-1 excess is 0.0199 ≈ 8b.

**R2.3 PROVED.** The label-y guard is equivalent to ‖q − e_y‖ ≤ (Brier(p, y) + b)^{1/2}, a ball around a vertex. Each
member's feasible set is:

F(p) = Δ ∩ {q ≥ e^{−d}p} ∩ ⋂_y Ball_y ∩ {q_d > q_k}

This set is convex. CHECKED: 0 midpoint failures.

**R2.4 PROVED (deficit identity).** The label-y guard holds if and only if p_y − q_y ≤ (b − S)/2. In words, a deficit is
at most b/2 plus half the decrease in Σq². With NLL, the joint per-member bounds are:

−min{(1 − e^{−d})p_k, (b − S)/2} ≤ q_k − p_k ≤ min{(1 − e^{−d})(1 − p_k), b/(2p_k) − m}

where m = max deficit. Consequently TV(p, q) ≤ 1 − e^{−d}.

**R2.5 PROVED + CHECKED (closed-form Brier-only bin value).**

V_B(bin) := min_q max_{i,y} [Brier(q, y) − Brier(p_i, y)] = max_{c∈Δ} [1 − ‖c‖² − c·β]

Here β_y = min_i Brier(p_i, y). The maximiser is water-filling, c_y = max(0, (τ − β_y)/2), and q = c* attains V_B. The
proof is Sion minimax: Σ_y c_y‖q − e_y‖² = 1 − ‖c‖² + ‖q − c‖², and c ∈ Δ. Every c ∈ Δ gives a certified lower bound,
which is a second closed-form infeasibility certificate. For a single member, V_B = 0. CHECKED: agrees with SLSQP within
6e-15 on 300 random bins.

**R2.6 BLOCKING-ISSUE (part of B2).** Take Brier-tight q (bisected to max excess = b, K = 6). Three float forms disagree
on 9,047 of 19,775 vectors:
- `q@q − p@p − 2(q_y − p_y)`
- the source per-row difference
- the same with `math.fsum`

## R3. Strict class, tie rule, exact ties (r3_ties.py)

**R3.1 PROVED.** The source decision is numpy argmax, first index. An exact U0 top tie is preserved by Ucal: the α = 1
path copies, and identical inputs give identical outputs. q = Ucal(p) then has no strict argmax.

**R3.2 COUNTEREXAMPLE.** Tempering creates exact top ties from strictly ordered U0. The probes were normalised K ∈ {3, 6}
vectors whose top two entries are 1–3 ulps apart, with α ~ U[0.25, 4]:

| Outcome | Count (of 21,883) |
|---|---|
| Tied, with argmax = d (passes hcal's check, fails strict Class) | 1,180 |
| Tied, with argmax ≠ d (hcal would raise) | 1,035 |

Example (K = 6, α = 1.46577): U0 top entries are 0.3280549986372077 and 0.32805499863720766. Both Ucal entries are
0.37805258757426274. For binary inputs the top two entries are p₀ and 1 − p₀, so a tie needs p₀ = ½ exactly.

**R3.3 PROVED (nudge).** For every p, ties included, q = (1 − η)p + η·e_d satisfies G whenever
0 < η ≤ η₀ = (√(1 + 8b) − 1)/4 = 0.0024876.
- NLL: η ≤ 1 − e^{−d}.
- Class: q_d − q_k = (1 − η)(p_d − p_k) + η > 0.
- Brier: excess_d = −η(2 − η)‖e_d − p‖² ≤ 0. For y ≠ d, excess_y = η²‖e_d − p‖² + 2η(p_d + p_y − ‖p‖²) ≤ 2η² + η ≤ b.

CHECKED: 600,000 cases, 0 failures, worst excess 0.995·b.

**R3.4 PROVED.** In a same-decision bin, M_d ≥ M_k for every k: the member attaining M_k has p_d ≥ p_k. CHECKED.

## R4. Bin feasibility (r4_bins.py)

**R4.1 PROVED (NLL-only: necessary AND sufficient).** There is a q ∈ Δ with q ≥ e^{−d}p for every member if and only if
ΣM ≤ e^d.
- Necessity: sum the inequalities.
- Sufficiency: q = e^{−d}M + (1 − e^{−d}ΣM)·r works for any r ∈ Δ, and this family is exactly the NLL-feasible set.

**R4.2 PROVED (NLL + strict Class).** The bin is admissible if and only if one of these holds:
- ΣM < e^d (take r = e_d and use R3.4), or
- ΣM = e^d and M_d > max_{k≠d} M_k.

**R4.3 COUNTEREXAMPLE (NLL alone, K = 3; pairwise overlap is not sufficient).** Take p^(i) = (⅓ − s)·1 + 3s·e_i with
s = 0.0012. Each pair has ΣM = 1.0036 ≤ e^d, but the triple has ΣM = 1.0072 > e^d. Verified exactly.

**R4.4 COUNTEREXAMPLE (full G, K = 6, one decision).** Take p^(i) = (½, ⅒, ⅒, ⅒, ⅒ − s, ⅒ − s) + 2s·e_i for i = 1, 2, 3,
with s = 1/500.
- Every pair is G-admissible under the EXACT rational check. The pair raising coordinates 1 and 2 uses
  q = (0.498513710, 0.103481298, 0.103481298, 0.099501248, 0.097511223, 0.097511223); its max Brier excess is 0.0018.
- The triple has ΣM = 1.008 > e^d.
- The solver's NLL-tight q passed float G but failed the exact check until it was moved 1e-9 inside the box. This is B2.

**R4.5 COUNTEREXAMPLE (the NLL closed form is not sufficient for G).** Take the binary pair a = 0.52 and a = 0.5245.
- ΣM = 1.0045 ≤ e^d, so NLL + Class admits the pair.
- Their exact feasible-c intervals, [0.51870, 0.52120] and [0.52319, 0.52569], are disjoint.
- A rational dual c = (4177199/8000000, 3822801/8000000) certifies min–max Brier excess ≥ 0.004481 > b.

**R4.6 COUNTEREXAMPLE (Brier-only, K = 3).** Three members whose pairwise V_B values are 0.00217, 0.00217 and 0.00241
(all ≤ b), while the triple has V_B = 0.002805 > b. Certified by a rational dual.

**R4.7 PROVED (Helly).** F(p) is convex in the (K − 1)-dimensional plane Σq = 1. So a bin is admissible if and only if
every K of its members are.
- For K = 2 (income), pairwise feasibility IS sufficient. A bin is admissible if and only if its two extreme members are.
  Each member's feasible c = q₀ interval is

  I(a) = [max(e^{−d}a, 1 − ((1 − a)² + b/2)^{1/2}), min(1 − e^{−d}(1 − a), (a² + b/2)^{1/2})] ∩ (½, 1]

  CHECKED: 0 of 50,000 random binary bins violate this.
- NOTE: the contract sentence "Pairwise feasibility is NOT sufficient" is correct for occupation (6-subsets needed) but
  not for income.

**R4.8 PROVED + CHECKED (exact joint value).** Let P = {q ∈ Δ, q ≥ e^{−d}M, q_d ≥ q_k}. Then

min_{q∈P} max_y [Brier(q, y) − β_y] = max_{c∈Δ} [dist²(c, P) + 1 − ‖c‖² − c·β]

This is the closure of strict Class. Any c gives a certified lower bound; dropping dist² recovers R2.5. CHECKED: on 6
bins, the primal agrees with a 1/60-grid dual to within the grid resolution.

## R5. Geometry and the diameter bound (r5_geometry.py)

**R5.1 PROVED.** Σ_k max(p_k, p′_k) = 1 + TV(p, p'). So an NLL-admissible bin has TV-diameter ≤ D, and also
ℓ∞-diameter ≤ D. The contract §3 claim is confirmed (CHECKED, error 5e-16).

**R5.2 PROVED + CHECKED (exact joint diameter).** Take decision 0, a = p₀ and representative q = (c, 1 − c). The exact
member set is [L(c), R(c)], where:
- L(c) = max(1 − e^d(1 − c), (c² − b/2)^{1/2}, ½)
- R(c) = min(e^d·c, 1 − ((1 − c)² − b/2)^{1/2}, 1)

sup_c [R(c) − L(c)] = D exactly, attained for c ∈ [0.854, 0.995]. Exact certificate: members a = 0.899498748 and
0.904511269 share q = (0.9, 0.1) under rational G, with TV = 0.005012521. So the joint supremum is D in general K, and it
is attained.

Below c = 0.854 the Brier guard binds and W(c) ≈ b/(4c(1 − c)):

| c | 0.51 | 0.55 | 0.60 | 0.70 | 0.80 | 0.85 | ≥ 0.854 |
|---|---|---|---|---|---|---|---|
| W(c) | 0.00250 | 0.00253 | 0.00261 | 0.00298 | 0.00393 | 0.00496 | 0.0050125 |

Exact single-member shifts for K = 2:
- Toward class 0: s⁺(a) = min{(1 − e^{−d})(1 − a), (a² + b/2)^{1/2} − a}.
- Toward class 1: s⁻(a) = min{(1 − e^{−d})a, ((1 − a)² + b/2)^{1/2} − (1 − a), a − ½}.

| a | 0.5 | 0.7 | 0.9 | 0.99 |
|---|---|---|---|---|
| s⁺ | 0.00125 | 0.00089 | 0.00050 | 0.00005 |
| s⁻ | 0 | 0.00208 | 0.00449 | 0.00494 |

**R5.3 PROVED (disclosure resolution).** Let q be any admissible output, from any mechanism, deterministic or randomised.
Then the compatible inputs satisfy

A(q) ⊂ {p ∈ Δ : p ≤ e^d q} = e^d q − D·Δ

Three consequences:
- Each coordinate satisfies p_k ∈ [e^d q_k − D, e^d q_k], an interval of width 0.0050125.
- The TV-diameter is ≤ D.
- The (K − 1)-volume is ≤ D^{K−1}·vol(Δ).

So covering a class region entirely takes at least 1/(K·D^{K−1}) bins: about 100 for K = 2 and 5.3·10¹⁰ for K = 6. For
K = 2 the exact minimum is 147 bins per class under G and 100 under NLL alone. Greedy chaining is optimal here because
L and R are monotone. CHECKED: 0 interval violations, and the largest compatible TV found was 0.0039.

Implication for the go rule: F3 ≥ 0.95 requires at least 95% of each class's rows to sit in at most 8 (income) or 64
(occupation) cells of TV-diameter ≤ 0.005.

**R5.4 COUNTEREXAMPLE.** A(q) is NOT convex in p for K ≥ 3, because the Brier part ‖p − e_y‖² ≥ ‖q − e_y‖² − b is
reverse-convex. Example (K = 3, TV(p₁, p₂) = 0.0044):
- q = (0.117224, 0.760577, 0.122198) is admissible for p₁ = (0.117715, 0.764066, 0.118219) and for
  p₂ = (0.113288, 0.764127, 0.122585).
- It is not admissible for their midpoint: Brier excess 0.002510 > b.

So membership must be checked per input, as F4 does. Never infer it from hull or extreme members. For K = 2, A(q) is an
interval.

**R5.5 PROVED + CHECKED (F3u).** F3u is a valid upper bound. Each bin lies inside the N-set of any of its members, and
disjoint bins give distinct members. But it can be loose, for two reasons: neighbourhoods have TV radius D while bins have
diameter D, and overlapping neighbourhoods are double-counted. On synthetic binary samples with C = 8:

| Sample | Exact max coverage | F3u |
|---|---|---|
| Uniform | 0.111 | 0.275 |
| 70% of mass in [0.97, 1] | 0.713 | 1.000 |

In the second case the failure is intrinsic, but F3u would label it INCOMPLETE. NOTE / recommendation: for income,
compute the EXACT maximum coverage with an O(nC) dynamic program over maximal windows [a_i, R(c*(a_i))]. This is
implemented in r5 R5.g.

## R6. SEX information in any admissible release (r6_release_info.py)

**R6.1 PROVED.** Randomised releases add nothing: each supported output q satisfies G for every input that can emit it.
The posterior support of p given q therefore lies in A(q), with diameter ≤ D.

**R6.2 PROVED (assumptions stated).** Assume:
- (i) Markov S − p − T: the deployed release reads only p plus independent randomness.
- (ii) η(p) = P(S = 1 | p) is L-Lipschitz in TV.

Let κ = min_t η̄_t(1 − η̄_t) and let h be binary entropy. Then:

Bacc(S | p) − L·D ≤ Bacc(S | T) ≤ Bacc(S | p)
I(S; p) − min{(L·D)²/κ, h(L·D)} ≤ I(S; T) ≤ I(S; p)

Proof:
1. |η(p) − η̄(T)| ≤ L·D on the support.
2. Use the chi-square bound on the Bernoulli KL, or concavity and subadditivity of h.
3. Data processing gives the upper side.

Consequences:
- The same bound relative to any fine quantizer Q_h with cells of TV-diameter h is
  |I(S; T) − I(S; Q_h(p))| ≤ max{ε(D), ε(h)}.
- For the coalition, use D per recipient.
- Any two admissible Markov releases differ in Bayes accuracy by at most L·D ≈ 0.005·L. This covers task-only versus
  local, joint or sequential. A 0.02 benefit needs P(SEX | p) to move by ≥ 0.02 within TV 0.005, i.e. L ≥ 4.

CHECKED-NUMERICALLY (binary smooth laws):

| L | Loss of the min-MI admissible partition (nats) | Bound |
|---|---|---|
| 1 | 2.7e-6 | 1.3e-4 |
| 5 | 3.7e-5 | 9.3e-2 |
| 20 | 1.6e-4 | 0.33 |

The LP-optimal randomised channel (L = 5) lowers Bayes accuracy by only 8.0e-5 against the bound 0.025, with posterior
width 0.0050 ≤ D.

**R6.3 COUNTEREXAMPLE (no smoothness).** Let S alternate with period 0.0005 in a over [0.9, 0.904), so that
I(S; p) = log 2. One admissible token, q = (0.9, 0.1), certified for all 4,000 inputs, gives I(S; T) = 0 and
Bayes accuracy ½. Tiny-ball coarsening CAN remove all information. Plug-in η on finite real rows has exactly this form,
since each distinct p has a 0/1 SEX. Plug-in "headroom" on rows is therefore not evidence of attackable headroom.

**R6.4 COUNTEREXAMPLE (no Markov).** Let p be constant and S independent of p. If the release chooses between two
admissible q's by SEX, then I(S; T) = log 2 while I(S; p) = 0. A non-Markov-robust bound still holds:
Bacc(S | T) ≥ Bacc(S | p) − P(|η(p) − ½| ≤ L·D), via an attacker who thresholds η at any point of A(T).

NOTE: SEX-aware assignment fitting (pilot) must deploy as a p-only map for R6.2 to apply.

## R7. Pointwise G vs teacher-expected G_exp (r7_expected_guard.py)

**R7.1 PROVED.** G ⊂ G_exp: the KL and ‖q − p‖² terms are p-averages of the per-label excesses. CHECKED on 91,140
admissible pairs, 0 failures.

**R7.2 CHECKED-NUMERICALLY.**
- For K = 2, the G_exp bin width is up to 0.0707 = 2(b/2)^{1/2} (Brier-limited), 4–28 times G's.
- For K = 6, the widest bin found has TV-diameter 0.0999, against the Pinsker + triangle bound (2d)^{1/2} = 0.1. The
  contract's "about 0.1" is right for occupation (0.0707 for income).

**R7.3 COUNTEREXAMPLE (calibration).** p = (0.99, 0.01) and q = (0.995, 0.005) is G_exp-admissible
(KL = 0.0019) but not G-admissible. The expected log-loss excess under the true label law:

| True label law | Excess | In units of d |
|---|---|---|
| (0.99, 0.01) | 0.0019 | 0.4 |
| (0.95, 0.05) | 0.0299 | 6 |
| (0.90, 0.10) | 0.0648 | 13 |

A G-admissible q has excess ≤ d for every label, so it stays ≤ d under ANY label law.

**R7.4 NOTE.** F3u's NLL neighbourhood is valid for G only. Under G_exp, a valid necessary pairwise condition is
TV(p, p') ≤ (2d)^{1/2} together with ‖p − p'‖₂ ≤ 2b^{1/2}. Reusing the G neighbourhood for the secondary diagnostic would
not give an upper bound.

**R7.5 NOTE.** Averages over rows are not pointwise. Rows with excesses (2d, 0) average d. Pointwise implies average,
but not conversely.

## R8. Fallback (r8_fallback.py)

**R8.1 PROVED.** q = p passes NLL and Brier exactly in float64: fl(e^{−d}p) ≤ p, and the excess is exactly 0. Strict
Class holds if and only if the top of p is strict. Hence **B1**.

**R8.2 COUNTEREXAMPLE.** The flag alone leaks. Take 4 equiprobable inputs with η = (0.3, 0.3, 0.7, 0.7), where x₁ and x₂
share one token and x₃ and x₄ fall back. Then I(S; flag) = 0.082 nats and Bayes accuracy is 0.70, even though the token
carries nothing. The fallback vectors also disclose p exactly.

**R8.3 PROVED.** In deterministic deployments the flag is a function of the output: output ∈ registered set if and only
if there was no fallback. If p equals a registered q, that q is admissible for p. The flag adds nothing beyond the
output, and both must be attacked, as the contract says.

**R8.4 COUNTEREXAMPLE / NOTE.** The rule for choosing among several admissible registered representatives changes
leakage. In a toy case, "first registered" gives I = 0.044 and Bayes accuracy 0.60, while "last registered" gives 0.173
and 0.80. Contract §4 does not fix this rule. It should be registered, deterministic and p-only (R6.4).

## R9. Oracle formulations (r9_oracles.py)

**R9.1 PROVED.** Decision-only is confidence-ineligible. e_d violates NLL whenever any p_k > 0 with k ≠ d, and a release
with no vector lies outside G's domain.

**R9.2 PROVED (floor).** Strict Class makes d(p) = argmax(T) a function of every admissible output. So
I(S; T) ≥ I(S; d) and Bacc(S | T) ≥ Bacc(S | d). For a coalition, (d₁, d₂) is a function of (T₁, T₂). Decision-only is
therefore a leakage floor for every admissible arm. CHECKED: 0 violations.

**R9.3 PROVED.** CLASS is eligible if and only if the whole class is one admissible bin.
- For K = 2 this requires a_max − a_min ≤ W(c*) ≤ D.
- The NLL closed form is necessary. Use R4.8 or R2.5 to settle Brier.

**R9.4 PROVED.** Merging outputs never increases MI or Bayes accuracy. Min-leakage deterministic partitions can
therefore be taken among the coarsest admissible partitions, which prunes the enumeration.

**R9.5 PROVED + CHECKED (stochastic arm).**
- Outputs may WLOG be the maximal admissible sets: relabel each output with a maximal set containing its support, then
  merge equal labels. The finite LP is then exact.
- Randomisation can strictly beat every deterministic partition. In a 3-input toy, Bayes accuracy is 0.513 for the
  stochastic channel against 0.689 for the best partition. This needs non-smooth η (R6).
- A cap of C tokens per class is a cardinality constraint. The LP alone is exact only if each class has at most C
  maximal sets. Otherwise, enumerate output subsets or use a MILP.

NOTE: §6 should state whether the stochastic arm is capacity-bound. The LP was independently re-checked against plug-in
Bayes accuracy (difference ≤ 1e-9).

## 10. Required fixes (blocking) and recommendations

**B1 (fix before FEASIBILITY_LOCK).** Amend PROTOCOL §3 "Unseen inputs" and UTILITY_CONTRACT §1 and §4 as follows:
- "Ucal(p) satisfies G iff its top is strict."
- Report a label-free count of permitted rows whose Ucal top is tied (expected 0).
- For tied rows, define the fallback as q = (1 − η)Ucal(p) + η·e_d with a fixed η, e.g. 1e-6. This is admissible by
  R3.3.
- Reword "Ucal preserves it" as "preserves the first-index numpy argmax; strictness is not preserved at ties".

**B2 (fix before FEASIBILITY_LOCK).** Add a float64-semantics paragraph covering:
1. **Canonical predicates.** NLL is `q[k] >= E*p[k]` with one named constant E = exp(−d). Brier is one named expression
   with a fixed summation order, e.g. `math.fsum`. Simplex membership is q ≥ 0 and |Σq − 1| ≤ τ, with τ stated.
2. **Construction margin.** Build and certify representatives against d − 1e-9 and b − 1e-9. Then every algebraically
   equivalent float form (multiplicative, ratio, log, clipped; dot, per-row, fsum) also passes at the true d and b, and
   C and E agree. Checking stays at d and b.
3. **Conservative margins in closed-form tests.** For the infeasibility test, F2 and F3u, allow for Ucal row sums being
   within 1e-12 of 1:
   - declare infeasible only if Σmax > e^d(1 + 1e-12);
   - count a neighbour if Σmax ≤ e^d(1 + 1e-12).
4. **Log-form statements.** State that the log-form and clipped implications hold in real arithmetic, and in float to
   ≈ 3e-15.

**Recommendations (not blocking).**
- Scope "pairwise is not sufficient" to K ≥ 3. For K = 2, pairwise or extreme-pair suffices (R4.7).
- Optionally add the Brier-only closed form (R2.5) as a second infeasibility certificate.
- For income, replace or supplement F3u with the exact dynamic-program coverage, which removes INCOMPLETE for K = 2
  (R5.5).
- Do not reuse the G neighbourhood for G_exp (R7.4).
- Register the representative-choice rule (R8.4) and the stochastic arm's capacity semantics (R9.5).
- F3's "largest fraction" is the top-C of the disjoint greedy F1 family, i.e. a constructive lower bound. Word it that
  way.
- Record R6.2 as the theoretical expectation for Stage E: under smooth laws, all admissible arms lie within L·D of each
  other.
