# MATH_REVIEW — for the owner of the full-view method research

**From:** the claims role (new numbering: assignment 2; older records call it Terminal 3). Branch
`research/pcrl-guarantee-review-v1`.

**Status:** no method branch or handoff from you existed when this was written (2026-09-23). This review
is therefore **independent**: it states what is true about the constructions named in the assignment, so
you can build on it or cite a counterexample. None of it is an instruction to stop.

**Fixtures:** `analysis/pcrl_guarantee_review_v1/fullview_fixtures.py` (G1–G7) and
`composition_fixtures.py`. The exact fixtures work in rational arithmetic in the {log 2, log 3} basis.
The random checks are labelled numerical and are not proofs.

**Notation:** S sensitive; Y task; H the recipient's actual view (H_A alone, H_AB for the coalition);
B=b(H) a finite bin; T the code; Z the released token; Q(z|t) the public kernel. The code and model are
treated as fixed (training-data privacy is a separate problem).

## M1. The chain-rule identity is correct, and it is only an identity

For B = b(H) and finite information quantities:

  I(S;Z|H) = I(S;Z|B) + I(H;Z|S,B) − I(H;Z|B).

*Proof.* Because B is a function of H, I(S;Z|H) = I(S;Z|H,B). Expand I(S,H;Z|B) by the chain rule in the
two orders. ∎ (Cover & Thomas, Thm 2.5.2.)

The correction I(H;Z|S,B) − I(H;Z|B) can have **either sign** (G1):
- XOR case (Z = S⊕H, B constant): the binned CMI is 0 while the true CMI is log 2 (under-statement).
- S = H, Z = S, B constant: the binned CMI is log 2 while the true CMI is 0 (over-statement).

The identity estimates neither correction term and yields no empirical upper bound. A learned
discriminator gives neither an MI upper bound nor a guarantee that holds uniformly over H.

## M2. The row-radius bound is correct and distribution-free, and that is its cost

*Claim.* Let Z be drawn from a fixed Q(·|T) with independent randomness, so Z ⊥ (S, W) | T for **any**
side information W: H_A, H_AB or anything else. If max_{t ∈ supp T} KL(Q(·|t) ‖ r) ≤ κ for some r, then
I(S;Z|W) ≤ κ.

*Proof.* Conditional data processing gives I(S;Z|W) ≤ I(T;Z|W). Then
I(T;Z|W=w) = Σ_t p(t|w) KL(Q_t ‖ q_w) ≤ Σ_t p(t|w) KL(Q_t ‖ r) ≤ κ, because the output mixture q_w
minimises the weighted divergence. Average over w. ∎

**Conditions.**
- supp Q_t ⊆ supp r for every t with positive probability; otherwise KL = ∞ and the bound is vacuous.
- Rows with p(t)=0 are irrelevant.
- If Q depends on a public state b(H_A), the bound must hold within every state, with r_b allowed to
  depend on the state. It then covers every view that contains H_A (both A and AB).
- A local encoder may not use H_B.
- G2b: a state-dependent kernel gives I(S;Z|H)=0 while I(S;Z)=log 2, so the guarantee applies only to
  views that contain the state.

**Tightness.** The best radius min_r max_t KL(Q_t‖r) equals the capacity of Q. This is the
Gallager–Ryabko minimax-redundancy theorem, with Csiszár (1995) for finite input sets and
van Erven–Harremoës (IEEE TIT 2014) Thm 34 for general inputs and all Rényi orders. The theorem was read
at arXiv:1206.2459v2 §B. It is **established mathematics, not a PCRL theorem**. G2: 400 random
laws, 0 violations, maximum I/radius = 0.9987. The exact tightness example is BSC(1/4) with S = T
uniform, where I = radius = ¾ log 3 − log 2.

## M3. The same κ caps the utility

The same proof gives I(Y;Z|H) ≤ κ. With log loss, I(Y;Z|H) is exactly the **Bayes-optimal** incremental
log-loss gain of adding Z to H (exact on BSC(1/4), G3).

It is **not** a bound on the gain over a finite, imperfect J probe, or over any fitted predictor. It is
also **not** the empirical +0.001 cap, which bounds CE_J − CE_M for fitted attackers: a difference of
measured losses between two releases, not an information quantity of one release.

**Scale.** The 2016 descriptive task gain of Q over H is about 0.026 nats, and J's is about 0.023. Any
release whose Bayes gain over H is ≥ g must have κ ≥ g. So a κ-constrained release that matches Q's scale
carries a sensitive guarantee of about 0.026 nats. That is informative against log 2 = 0.693 (sex) and
log 9 = 2.197 (race), but it is 26× the +0.001 scale of the registered cap, and it is an upper bound
rather than a measured value.

**The evaluated releases do not satisfy any useful κ.** D17 is deterministic with up to 17 distinct
actions, so its radius is log(#actions used), up to log 17 = 2.83 nats: vacuous for both attributes (G4).
Q's radius needs its kernel, which is in the private archive. The entry point is
`analysis/pcrl_guarantee_review_v1/kernel_guarantees.py`.

## M4. An attribute-specific alternative that is already distribution-free (SDPI)

  I(S;Z|H) ≤ η_KL(Q)·I(S;T|H) ≤ η_TV(Q)·H(S|H) ≤ η_TV(Q)·log|S|,

where η_TV(Q) = max_{t,t'} TV(Q_t, Q_t') is the Dobrushin coefficient.

*Sources.* Polyanskiy & Wu, arXiv:1508.06025: eq. (17) (the mutual-information characterisation of
η_KL) and Thm 1 (η_f ≤ η_TV, from Cohen–Kemperman–Zbăganu 1998). Apply them conditionally on H=h, with
U=S and X=T.

*Assessment.* Valid, but for random kernels it is looser than the radius bound in 299 of 300 cases (G4).
For deterministic kernels η_TV = 1, so it is vacuous too. Neither bound rescues a near-deterministic
release.

## M5. The robust envelope for P(T|S,H): what holds and what does not

1. **Convex in Q.** For a fixed law P and a fixed decoder or cost, I_P(S;Z|H) is convex in Q. The
   supremum over any set of laws is convex (a pointwise supremum). A 17×17 midpoint check found 0
   violations (G5).
2. **The oracle is not convex.** For fixed Q, the worst case over P(T|S,h) maximises a convex function,
   so the maximum sits at a vertex of the envelope. An exact oracle needs vertex enumeration (for a TV or
   ℓ₁ ball the vertices are single-atom moves) or a certified upper bound.
   - **Sampled scenarios are only a lower bound on the worst case.** In G5, 2,000 interior samples found
     0.0142 nats against an exact vertex worst case of 0.0502 nats.
   - A finite scenario set covers only those scenarios.
3. **What a calibrated radius covers.** A radius calibrated at the bin level covers bin-level laws, not
   laws at the level of H.
   - G5c: P(T|S,B) is estimated perfectly (it is uniform in S, so the bin-level radius is 0), yet
     I(S;T|H) = log 2.
   - Covering continuous H needs an explicit assumption, for example Lipschitz continuity of
     h ↦ P(T|S,H=h) with a known constant plus a covering argument, or within-bin homogeneity. Neither is
     testable from the data without that assumption.
4. **How large a radius can be tolerated.** For a deterministic 17-action kernel around a uniform
   nominal law, an H-level TV radius of 0.05 allows a worst case of 0.050 nats. Certifying 0.01 nats
   requires a radius ≤ 0.0239 (G5), at every H.
5. **The unrestricted envelope is impossible.** If any law of (S,T) given H is allowed, every kernel with
   two reachable distinct rows fails. Map S=0 to t and S=1 to t'; the worst case is then the binary-input
   channel information, which is > 0 (G6).
   - With a fixed prior for S, the worst case over all laws is controlled only by constraints that bound
     row separation, which is essentially the radius and SDPI regime.
   - This is an impossibility for *arbitrary laws*. It does not say that practical mechanisms fail.
6. **The finite-sample confidence set must account for:**
   - the learned code, fitted on the same data;
   - household dependence (bootstrap over households, not persons);
   - survey weights;
   - support gaps (unobserved (S, cell) combinations);
   - selection among candidate mechanisms.
   Calibration of a fitted probability, ensemble spread, or a bootstrap of the already-selected model is
   **not** a confidence set for the unknown conditional law.
7. **Prior art.** Robust design over an uncertainty set of laws is:
   - Diaz–Wang–Calmon–Sankar (IEEE TIT 2020): uniform privacy mechanisms over an ℓ₁ ball; finite
     alphabets; the adversary has *no* side information; Shannon MI is flagged as outside its O(1/√n)
     technique.
   - Lopuhaä-Zwakenberg–Goseling, *Entropy* 26(3):233, 2024: RLDP over a Rényi ball; finite alphabets;
     the mechanism sees X=(S,U).
   Neither covers conditioning on a continuous published view H. That is the smallest real distinction
   available, and it needs one of the assumptions in item 3 to become a theorem.

## M6. What I will check when your method lands

For the entry point, see `results/pcrl_guarantee_review_v1/REVIEW_INDEX.md` § "Incremental review of the
method".
- The exact probability model, stating who observes each variable.
- Which of the objects {law, code, bins, decoder, radius} are fixed and which are optimised.
- Whether the oracle is exact, bounded or sampled.
- How the radius is calibrated, and at which level (bin or H).
- A quantitative comparison of the certified bound with log|S| at the utility level being claimed.
- Whether any use of H_B enters the A-local encoder.
