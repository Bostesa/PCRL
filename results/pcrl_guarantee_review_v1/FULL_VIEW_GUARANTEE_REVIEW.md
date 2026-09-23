# Full-view guarantee review

This is the formal companion to [MATH_REVIEW.md](MATH_REVIEW.md), which was the early handoff to the
method owner. Each statement below is labelled **proved** (a proof is written here or cited at source),
**refuted** (an exact counterexample), **conditional** (true under a named assumption), or **numerical**
(finite-instance evidence only). Proof coverage and numerical coverage are reported separately in the
final table.

Setting: the code g and the public kernel Q are fixed. T = g(inputs). Z ~ Q(·|T), or Q_b(·|T) with
b = b(H_A) a public state, using fresh randomness independent of everything else. A recipient's view W is
H_A for role A and H_AB = (H_A, H_B) for the coalition. Nothing here concerns privacy of the training
data.

## Theorem 1 (distribution-free release guarantee) — proved

Let κ_b = min_r max_{t : P(T=t, b)>0} KL(Q_b(·|t) ‖ r) and κ = max_b κ_b. Then for **every** joint law of
(S, Y, H_A, H_B, T) and every view W that determines b (so both A and AB qualify):

- I(S;Z | W) ≤ κ and I(Y;Z | W) ≤ κ;
- I(S;Z | W) ≤ η_TV(Q_b)·H(S | W) ≤ η_TV·log|S|, where η_TV = max_b max_{t,t'} TV(Q_b(·|t), Q_b(·|t')).

*Proof.* Given W=w, the state b is fixed and Z depends on (S, Y, W) only through T. By conditional data
processing, I(S;Z|W=w) ≤ I(T;Z|W=w) = Σ_t p(t|w) KL(Q_t‖q_w). This is ≤ Σ_t p(t|w) KL(Q_t‖r) ≤ κ_b,
because the output mixture q_w minimises the weighted divergence. The same argument applies to Y. The
second line is the SDPI: η_KL(Q) = sup_{U→T→Z} I(U;Z)/I(U;T) (Polyanskiy–Wu, arXiv:1508.06025, eq. 17),
and η_KL ≤ η_TV (their Thm 1, citing Cohen–Kemperman–Zbăganu 1998), applied at each w with U = S. ∎

*Conditions.*
- Each supp Q_b(·|t) ⊆ supp r for the rows in use; otherwise KL is infinite and the bound is vacuous.
- Zero-probability rows do not count.
- A kernel may depend on b(H_A). If it depends on H_B, the local encoder would need an input it does not
  have, which is not allowed.
- A view that does not determine b is not covered: G2b gives I(S;Z|H)=0 but I(S;Z)=log 2.

*Tightness.* κ_b is the capacity of Q_b. This is the Gallager–Ryabko minimax-redundancy theorem, with
Csiszár (1995) for finite input sets and van Erven & Harremoës, IEEE Trans. Inf. Theory 2014, Thm 34
(read at arXiv:1206.2459v2, §"Channel Capacity and Minimax Redundancy"). The bound is attained when S
copies T under the capacity-achieving input (exact example: BSC(1/4), I = ¾ log 3 − log 2; G2). This is
**established mathematics, not a new PCRL theorem**.

*What it does not cover.*
- It says nothing about the registered +0.001 clause, which compares fitted-attacker losses between two
  releases.
- It is not a bound on any finite attacker's measured increment. A restricted attacker can gain more than
  the true CMI (claims-branch fixture F9).
- It is not a bound for a finite, imperfect J probe.
- It says nothing about training-data privacy.

## Corollary 1 (utility ceiling) — proved

I(Y;Z|W) is exactly the Bayes-optimal incremental log-loss gain from adding Z to W. So a release with
radius κ cannot improve any authorised task by more than κ nats of Bayes log loss over the published
view (exact example: G3).

**Usefulness at the proposed utility level (quantitative).**
- Q's 2016 descriptive task gain over H is about 0.026 nats (J's is about 0.023).
- Any release with a Bayes gain of g over H needs κ ≥ g.
- At κ = 0.026 the guarantee is I(S;Z|W) ≤ 0.026 nats for SEX, for RAC1P, and for every other attribute
  and every side-information view. That is informative against log 2 = 0.693 and log 9 = 2.197.
- Whether a κ = 0.026 kernel can actually deliver 0.026 nats of residence gain is an **empirical
  question**. The ceiling is attained only if all of the channel's information is about Y beyond W.
  It must not be assumed.
- A strong all-input constraint (small κ) caps every capability equally. That is its cost.

**The evaluated releases have no useful κ.** D17 is deterministic: η_TV = 1 and κ = log(#actions used),
up to log 17 = 2.83, which is vacuous. Q's κ needs its kernel, which is only in the private archive.
Entry point: `python analysis/pcrl_guarantee_review_v1/kernel_guarantees.py kernel.json` (validated on
BSC and on the identity kernel). Terminal 3's archive holder can run it without touching any
individual-level data.

## Proposition 2 (chain rule) — proved; the example is exact

For B = b(H): I(S;Z|H) = I(S;Z|B) + I(H;Z|S,B) − I(H;Z|B). Standard chain rule (Cover & Thomas Thm 2.5.2).

The correction can be positive (XOR: log 2 hidden) or negative (S=H, Z=S: log 2 overstated) (G1). It
yields no estimate and no bound by itself. A learned discriminator supplies neither an MI upper bound nor
a uniform guarantee over H.

## Proposition 3 (when binned CMI is conservative) — conditional, proved

If T ⊥ H | (S, B), then Z ⊥ H | (S, B), so I(H;Z|S,B) = 0 and I(S;Z|H) = I(S;Z|B) − I(H;Z|B) ≤ I(S;Z|B).
Under this **within-bin homogeneity** assumption, a *true-law* binned budget certifies the full-view CMI
(G8: 0.0143 ≤ 0.0150).

**The actual code violates the assumption by construction.** T0 cuts logit p(PCA32, H_A) −
logit b(H_A), so it depends on H_A inside each of the two H_A cells. When the assumption fails, the
binned value can understate by a large factor (G8: 0.0006 versus 0.068). This proposition is the weakest
clean assumption I found under which the study's binned constraints would mean something at the level of
H. It is **not** satisfied by the evaluated mechanism.

A code built as a function of (X_A, b(H_A)) alone, with X_A conditionally independent of H_A given
(S, b), would satisfy it. The second condition is untestable at continuous H without further assumptions.

## Proposition 4 (robust envelope) — mixed

Model: for each h, the unknown P(T|S, H=h) lies in a set U(h). P(S|H) is either known or also uncertain.
Q and the decoder or cost are fixed while evaluating a constraint and optimised while designing.

| Question | Answer | Status |
|---|---|---|
| Is sup_{P∈𝒫} I_P(S;Z\|H) convex in Q? | Yes: a pointwise supremum of convex functions (G5 midpoint check) | proved; numerical check |
| Is the worst-case oracle convex? | No. For fixed Q it maximises a convex function of P(T\|S,h), so the maximum is at a vertex of U(h) | proved |
| Exact, bounded or sampled? | Exact only by vertex enumeration (single-atom moves for TV or ℓ₁ balls); upper-bounded by Theorem 1; sampling gives only a lower bound (G5: 0.0142 sampled versus 0.0502 exact) | proved + numerical |
| Does a finite scenario set cover the claimed laws? | Only the listed scenarios | — |
| Does a bin-level calibrated radius cover continuous H? | **No** (G5c: bin-level radius 0, yet I(S;T\|H) = log 2). It needs a Lipschitz or smoothness bound on h ↦ P(T\|S,h), or Proposition 3's homogeneity | refuted as stated; conditional repair |
| Does a finite-sample confidence set cover the law? | Only if it accounts for household dependence, survey weights, support gaps, the learned code fitted on the same data, and selection. Calibration, ensemble spread, or a bootstrap of the selected model is not a confidence set for P(T\|S,H) | — |
| Is an unrestricted envelope possible? | Impossible for any kernel with two reachable distinct rows (G6). This is not a claim that practical mechanisms fail | proved |
| How large can the radius be? | For a deterministic 17-action kernel near a uniform nominal law, an H-level TV radius of 0.05 allows a worst case of 0.050 nats; certifying 0.01 nats needs a radius ≤ 0.0239 at every h (G5) | numerical, exact enumeration |

**Weakest assumptions that give a useful attribute-specific statement.** Either:
- (i) Proposition 3's homogeneity plus a true-law binned budget; or
- (ii) U(h) of TV radius ε at every h, justified by a stated smoothness assumption, with the worst case
  computed exactly by vertex enumeration.

A trivial ceiling check always applies: a certified value ≥ log|S| is valid but uninformative. A constant
channel (G7) is a correctness fixture, not a protected method.

## Coverage summary

| Statement | Proof coverage | Numerical coverage |
|---|---|---|
| Theorem 1, radius part | full (written above; the minimax equality cited at source) | 400 random laws, 0 violations; exact tightness |
| Theorem 1, SDPI part | full (cited at source: Polyanskiy–Wu eq. 17 and Thm 1) | 300 random laws, 0 violations |
| Corollary 1 | full | exact BSC example |
| Proposition 2 | full (textbook) | exact XOR and over-statement; 300 random tables (max error 1.4e-16) |
| Proposition 3 | full | exact-rational table checked in float (G8) |
| Proposition 4 convexity / oracle | full | midpoint check; exact vertex enumeration on one kernel |
| Proposition 4 radius and bin coverage | counterexample exact | — |
| Kernel guarantee for the evaluated Q | none (the kernel was not accessed) | entry point provided |
