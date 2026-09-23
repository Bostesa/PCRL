# Independent review of the full-view method study

**Pinned evidence.** `research/pcrl-full-view-protection-v1` @ `0e90d0b3da97bec28cf200abe2213803c2fb3b71`,
`results/pcrl_full_view_protection_v1/` (protocol 63f288382). Reviewed 2026-09-23 by the claims role
(Terminal 2 in the new numbering).

**Inputs read, and nothing else:**
- the committed reports and synthetic checkpoints;
- the six archived 2018 kernels (Q.npz), whose SHA-256 values match EXISTING_RADIUS.json;
- Terminal 3's private 2018 aggregate count tables (fineC_anchor_*.joblib), for one attempted test.

No 2016 row, no person-level row and no fit.

**Rigour vocabulary:**
- **interval-certified**: evaluated in mpmath interval arithmetic on exact rationals, so the result is a
  mathematical bound for the stated finite object;
- **numerical**: floating point, with a guard;
- **proved**: a written proof checked here;
- **not established**.

## Verdict

The study's own decision is correct and well scoped:
- stop before any ACS grid;
- 0/27 main fits;
- no algorithmic contribution;
- the finite-list robust programme is the same programme as the matched adaptation of published robust
  information design.

Every checked number reproduces. Several of its numerical brackets are now interval-certified. No
genuinely distinct, supported construction remains (see the final section).

## 1. Continuity constants (FULL_VIEW_THEORY §4) — proved, reproduced, and improvable

**The premise chain is correct.** A TV bound on (S,T)|h passes through the fixed kernel to (S,Z)|h and to
both marginals (data processing for TV). The prior-error and conditional-error triangle bound
ε ≤ ε_p + ε_t is valid (coupling), with TV = ½‖·‖₁.

**The constant is valid.** f_d(ε) = ε log(d−1) + h₂(ε) for ε ≤ 1 − 1/d is the tight entropy continuity
bound. Summing it over H(S), H(Z) and H(S,Z) bounds |ΔI|.

**The thresholds reproduce exactly.** For |Z| = 17 and δ₀ = 0:
- ε ≤ 2.9735e-4 (SEX) and 2.6617e-4 (RAC1P) for a 0.01-nat term;
- ε ≤ 2.4305e-5 and 2.2212e-5 for 0.001.

**A valid, tighter bridge exists, independent of |Z|.** Write I = H(S) − H(S|Z) and use Winter's classical
conditional-entropy continuity. Winter, *Commun. Math. Phys.* 2016, Lemma 2, read at arXiv:1507.07775v6:
for classical conditioning, |ΔH(S|Z)| ≤ ε log|S| + (1+ε) h(ε/(1+ε)). Then

  |ΔI| ≤ f_|S|(ε) + ε log|S| + (1+ε) h(ε/(1+ε)).

The required ε becomes 5.67e-4 (SEX) and 4.62e-4 (RAC1P) at 0.01 nats, and 4.39e-5 / 3.75e-5 at 0.001.
That is roughly 1.7–1.9× larger, but **the conclusion is unchanged**: uniform per-h conditional-law
accuracy of about 5e-4 TV at every continuous H value is not available from the reused 2018 rows. The
study correctly makes no coverage claim.

## 2. Capacity brackets (EXISTING_RADIUS.json) — orientation correct, now interval-certified

**Orientation is right.** The mutual information at the Blahut–Arimoto input prior is a *lower* bound on
capacity = radius. max_t KL(Q_t‖r) at the reference r is an *upper* bound on R(Q). Both are taken over all
32 allowed rows. The iteration-cap flag on Q does not affect validity.

Interval certification (`analysis/pcrl_guarantee_review_v1/interval_radius.py`, output
INTERVAL_RADIUS.json) used Terminal 4's own prior and reference, re-evaluated on the exact-rational,
row-normalised kernels. Row-sum error before normalisation was at most 2.2e-16.

| Anchor | Kernel | Certified lower | Certified upper | Inside T4's numerical bracket |
|---:|---|---:|---:|---|
| 0 | Q | 1.762167334 | 1.762167652 | yes |
| 1 | Q | 1.936615652 | 1.936615660 | yes |
| 2 | Q | 1.967602333 | 1.967602369 | yes |
| 0 | D17 | log 16 = 2.772588722 | same | yes; 16 actions used |
| 1 | D17 | log 14 = 2.639057330 | same | yes; 14 actions used |
| 2 | D17 | log 15 = 2.708050201 | same | yes; 15 actions used |

**How to read them.** The Q radii exceed log 2. They lie below log 9, but also above ~1.24 nats. That is
the 2016 H-only attacker's held-out cross-entropy for A/RAC1P, which estimates an upper bound on
H(RAC1P|H_A) (up to sampling). Since I(S;Z|H) ≤ H(S|H) always, the radius is therefore **practically
uninformative for race as well**, not only for sex. This is an estimate-based remark, not a certified
one. D17's radii are exactly log(actions used).

## 3. Finite-law guarantees — hashes verified; feasibility within guard, not exact

`review_t4_synthetic.py` (output T4_SYNTHETIC_REVIEW.json) did the following:
- rebuilt all nine fixtures' laws exactly in rationals from the committed generator;
- confirmed that every float law's SHA-256 matches and that the exact rebuild rounds to the committed
  arrays;
- re-evaluated all 80 reported optimised channels (36 robust, 36 radius-control, 8 nominal) in interval
  arithmetic.

Results:
- **Costs** agree with the exact rational costs to ≤ 5.6e-17.
- **Robust channels (36 cells).** 13 are **certified feasible** (interval upper bound ≤ δ). 23 are
  **within guard**: they exceed δ by at most 1.3e-9 (for example partly_coupled δ=.01 reaches
  0.0100000002), inside the declared 1e-7 guard.
  - So the guarantee card's wording ("≤ δ within 1e-7 numerical tolerance") is exactly right.
  - Without the tolerance the claim would be false by ≤ 1.3e-9.
  - A certified-feasible channel is obtained by mixing θ ≤ 2⁻²⁰ toward a constant row (see §4).
- **Radius-control channels (36).** Their full-view CMI is ≤ δ: 27 certified and 9 within guard.
- **Nominal channels on list-uncertainty fixtures (6 of 8 cells)** violate the robust list target, as the study
  itself reports. Confirmed; they are not feasible competitors.
- **The rational fixture at δ=0 is solved exactly as a linear programme.** The optimum cost is 3/10 at
  P(Z=1|T) = (1/2, 0, 1), and the best zero-leakage deterministic map costs 2/5. This exact-rational
  certificate matches T4's 0.30000 / 0.40000 and the committed task-directed fixture.

## 4. Solver objective bounds — now rigorously bracketed

For each CMI-constrained cell, I derived an **independent** lower bound. For any λ ≥ 0,
OPT ≥ Σ_k λ_k (f_k(q*) − ∇f_k(q*)·q* − δ) + Σ_t min_z [c_tz + Σ_k λ_k ∇f_k(q*)_tz]. This is valid because
each f_k is convex in q and every row lies in a simplex. λ was chosen by solving the resulting concave
piecewise-linear maximisation as a linear programme, and the bound was then evaluated in interval
arithmetic. The rigorous upper bound is the exact cost of a certified-feasible mixture of the reported
channel.

| | Result |
|---|---|
| Robust cells bracketed | 27 of 27 (positive δ) |
| Largest rigorous gap (upper − lower) | 2.04e-6 (xor/safe_correlated at δ=.05) |
| Median rigorous gap | 8.6e-9 |
| partly_coupled δ=.01 (headline) | OPT ∈ [0.3785448, 0.3785456] |
| rational_separation δ=.01 | OPT ∈ [0.26218675209, 0.26218675210] |
| Any T4 numerical lower bound above a rigorous upper bound | none |

T4's supporting-hyperplane lower bounds are correctly labelled numerical. They are consistent with the
certified ones, and with LP-chosen multipliers the certified lower bound is at least as high as T4's in
all 27 cells, so every T4 bracket is confirmed and tightened. A lower bound can exceed the cost of a *slightly infeasible* channel, which I observe
at the 1e-9 level. So a two-sided statement needs a certified-feasible upper point, which is provided
here.

## 5. Guarantee card and theory — accurate, with three wording notes for the method owner

1. §2 of FULL_VIEW_THEORY cites "descriptive Q gain over H ≈ .026 nats (Terminal 2 review M3)" as a scale
   comparison. M3 overstated what that number establishes (CORRECTIONS_2026-09-23 item 1). Keep it as
   scale only, and add: "a finite-probe loss difference, not an estimate of I(Y;Z|H) or a necessary
   radius".
2. "The unchanged Q radius … is nonvacuous for nine-class race": add that it exceeds the estimated
   H(RAC1P|H_A) of about 1.24 nats (§2 above).
3. §5, "Nontrivial attribute-specific guarantees require genuine conditional-law restrictions or a nearly
   constant channel": this is correct and matches the precise impossibility (CORRECTIONS item 3).

## Does any useful, genuinely distinct construction remain supported?

**No.** Each candidate has been examined against the evidence:
- **Finite-law robust channel.** It is exact and useful only for listed laws, and it is identical to the
  matched published robust-information programme. Its task advantage (for example 0.3785 versus 0.4576
  for the radius control at .01) is a known privacy-funnel property of known finite laws. Interval
  certification now makes those synthetic numbers rigorous. It does not make them novel or transferable.
- **Distribution-free radius.** It is established mathematics. For the actual archived kernels it is
  certified at 1.76–1.97 (Q) and log 14–16 (D17): useless at any meaningful δ. A radius-constrained
  kernel with a small κ is possible in principle, but no evidence shows it would carry useful residence
  information.
- **Continuous-H bridge.** It is valid, and it improves by about 2× with Winter's lemma. It needs an
  unavailable uniform envelope near 5e-4 TV.
- **Within-bin homogeneity route.** Not established for the actual code (item 2 of the corrections). The
  available aggregates cannot test it.

What would make a construction distinct and supported is a verifiable conditional-law envelope for
continuous H on genuinely untouched data, plus a win over a matched-certificate radius kernel. Neither
exists.
