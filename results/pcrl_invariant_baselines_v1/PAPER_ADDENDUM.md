# PAPER_ADDENDUM — text Terminal 2 can integrate

Drop-in material for the manuscript. Every number is traceable to a committed artifact
in `results/pcrl_invariant_baselines_v1/`. Claims are written at the strength the
evidence supports and no stronger.

**Status of the evidence:** 2018 is development; 2017 is development for these methods
and is labelled exploratory everywhere; **2016 is UNSCORED and was not touched.**

---

## 1. One-paragraph summary

> A specification error identified but not repaired by the predecessor study — a
> conditional-moment penalty that was not invariant under rotations of the released
> channel, when the disclosure it measures is — was repaired exactly and re-run. The
> repaired objective is invariant by construction rather than by convergence: it is a
> function on the Grassmannian, and the predecessor's own outcome-free acceptance gate
> passes with a measured rotation-only share of `1.4e-15` against a `0.10` threshold,
> where the defective objective measured a mean of `0.628`. **The repair did not
> improve measured disclosure; it worsened it.** At matched rank and policy the repaired
> arm leaks significantly more than the defective one on coalition race recovery, with
> no endpoint significantly better, and all four registered coordination cells fail. In
> the first method-level comparison this line of work has had, a closed-form linear
> erasure method (LEACE, 2023) applied to a frozen auxiliary channel the project already
> possessed **dominates** the mechanism under development — significantly lower recovery
> on all four sensitive endpoints with no statistically detectable utility cost — and is
> statistically indistinguishable from the frozen neural comparator on every endpoint,
> while the developed mechanism is significantly worse than that comparator on three of
> four.

## 2. The mechanism result

Use this in place of any text that treats the 63% rotation share as a bound or as a
causal decomposition.

> The predecessor measured that a mean 63% (range 37–91%) of its optimiser's
> training-objective progress was reachable by a rotation of the released channel — a
> transformation under which the utility term, the linear penalty and the information
> content of the release are all invariant. That share is an **attained** value, not a
> supremum: every rotation search terminated on budget exhaustion or line-search
> failure. It was therefore evidence of surrogate movement, not a decomposition of cause.
>
> Repairing the construction removes the slack exactly. Weighting the off-diagonal
> degree-two monomials by `sqrt(2)` makes the quadratic block the true squared Frobenius
> norm of the symmetric conditional moment, which rotation conjugates as `M -> Q'MQ`;
> replacing the finite Fourier bank with an exact radial kernel on a frozen bounded
> subset makes the second block depend on the data only through pairwise distances;
> dropping per-feature standardisation in favour of one scalar per block removes the
> remaining coordinate dependence. Measured on real data at rank 16,
> `|L(WQ) − L(W)| = 1.1e-16`.

## 3. The finding that matters

> The predecessor argued that because roughly 37% of the surrogate movement bought
> statistically significant reductions in measured recovery, removing the provably inert
> remainder should let the same compute buy more. **This is refuted by direct
> experiment.** Under the repaired objective the optimiser reaches a strictly better
> point in all 18 conditions, and the resulting release leaks *more*: at rank 16 the
> coalition race endpoint worsens by +0.0078 (adjusted [+0.0026, +0.0131]) and at rank 8
> local race worsens by +0.0079 ([+0.0027, +0.0132]), with no endpoint significantly
> improved.
>
> The natural reading is that the coordinate dependence had been acting as an implicit
> regulariser on an under-determined surrogate, holding the optimiser near the
> linear-moment solution. Removing it let the optimiser travel further into a region
> where the finite empirical moment family and the actual attacker slate disagree.
> **Surrogate–attacker mismatch, not rotation, is the binding constraint on this
> interface** — and the repair is what made that visible, by removing the confound that
> had been masking it.

This is a **negative result with a mechanism**, which is worth more than the qualified
negative the predecessor could report.

## 4. The external-baseline comparison

Previously the manuscript could state only that a competitive-method claim would require
baselines "which have not been run" (`CONTRIBUTION_ASSESSMENT.md` §6). They have now
been run, under the fairness conditions that section specified — with two of its own
prescriptions corrected.

> Three published methods were adapted to this interface under conditions that keep the
> comparison fair: every release preserves the published service output `H` bitwise, no
> representation or baseline fit sees the held-out residence or commute labels, each is
> evaluated at a declared width against the same attacker families with the same
> selection rule, and every departure from a published algorithm is named as a departure.
>
> Against the developed mechanism at rank 16, LEACE achieves significantly lower recovery
> on all four family sensitive endpoints (−0.008 to −0.016) with a residence cost that is
> **not** statistically detectable, and it clears the legacy source allowance in 3 of 3
> seeds where the developed mechanism clears it in 1–2. SPLINCE and the OptNet-ARL
> adaptation also beat it on every sensitive endpoint but pay a significant residence
> cost, so they are trades rather than dominations. Against the frozen neural channel,
> all three adaptations are statistically indistinguishable on all five family endpoints,
> while the developed mechanism is significantly worse on three of four.

**Two specification corrections, which the manuscript should carry:**

> A linear-erasure baseline must act on the auxiliary channel **alone** and be appended
> to an unchanged `H`. Applying it to the concatenated wire would edit `H`, because the
> erasure operator is a single oblique projection over all coordinates with nothing
> constraining it to act as the identity on the service output — which would destroy the
> exact-preservation guarantee that is the study's structural contribution. And a
> covariance-preserving erasure baseline must preserve covariance with the **authorised
> training tasks**, never with the held-out residence label, which would defeat the
> held-out-task test.

## 5. Prior-art attribution, now checked numerically

> The study's marginal spectral arms are SARL-style residual-teacher adaptations. This is
> established constructively rather than asserted: rebuilding each arm from the
> identification — top eigenvectors of `U − λ̄P` in whitened coordinates, with the
> intercept-only basis and class-prior nuisance making the per-class moment exactly the
> whitened sensitive cross-covariance — reproduces the stored maps **bitwise**. SARL's
> eigenvalue-sign rank rule and the fixed width agree here (32 positive eigenvalues
> against a width of 16), and the truncation boundary is well separated. One genuine
> mismatch remains: the arms trace-normalise each protected attribute's moment Gram
> separately before averaging, which is a per-block reweighting rather than a global
> scalar and is therefore not absorbable into the trade-off parameter; it moves the
> selected subspace by a projector distance of 0.10–0.20. The arms are credited and
> reused as adaptations; **no duplicate model was fitted to attach a published name.**

## 6. Rank behaviour

> The positive-eigenvalue rank rule returns 16 in all three seeds, unchanged, and the
> rank-8 arm remains a predeclared compression alternative rather than the rule's output.
> The new rank finding is on the erasure side and is support-driven: linear erasure
> deletes exactly `rank(Σ_XZ)` directions, so the 16-coordinate channel drops to rank 6,
> 6 and 7 across seeds. The third seed differs because one race category has zero
> population support there, making the centred joint one-hot rank 9 rather than 10.
> **Protected-class support therefore determines the realised width of a released
> channel**, which is a practical consequence worth stating for anyone deploying
> closed-form erasure on survey data with rare categories.

## 7. Sentences to retire or correct

| Existing text | Replacement |
|---|---|
| "roughly 37% of the surrogate movement bought the real reductions, so removing the provably inert 63% should let the same compute buy more" | Refuted by direct experiment. Removing it reaches a better surrogate point and a **worse** release. |
| Any reading of the 63% share as a bound or a causal decomposition | It is an **attained** value; every rotation search stopped on budget exhaustion or line-search failure. |
| "no endpoint significantly worse" (nonlinear-penalty attribution row) | **False.** `spectral_nlr8_L1 − spectral_lin8_L1` on `recovery/A/RAC1P` unweighted is +0.00862, adjusted [+0.00168, +0.01556]. |
| "+0.0098" for the `nlr8_C1` vs `J` person-weighted `A/SEX` deficit | **+0.01210.** The generated tables were always right; the prose was not. |
| "the baselines in §6 … have not been run" | They have. See §4 above and `BASELINE_ADAPTATIONS.md`. |
| Any framing in which fixing the rotation defect is the open question | The defect is fixed exactly and was **not** what limited the method. |

## 8. Framing

The defensible framing is **unchanged in kind but stronger in evidence**: an empirical
study of the value and limits of coalition-conditioned incremental releases. What this
addendum adds is a third deliverable alongside the structural guarantee and the locked
sealed-year demonstration:

> A negative method result whose cause is now **isolated rather than merely narrowed**.
> A specification error was identified, repaired exactly, and shown by direct experiment
> not to have been the limiting factor — and in the same run, method-level baselines
> established that a closed-form linear erasure applied to an existing channel
> outperforms the mechanism under development.

**This remains a study, not a competitive-method paper.** The method does not win.
Reporting that a simpler published alternative beats it is the contribution.

## 9. Do not write

* That rotation invariance is harmful in principle. This is one objective on one
  interface; the finding is about surrogate–attacker mismatch.
* That LEACE "solves" the problem. Its guarantee covers **only the transformed channel**
  against **affine** predictors; `H_A` still discloses; its own authors conjecture it
  does not extend to general nonlinear adversaries; and this study's own fixture shows a
  quadratic probe recovering from an erased channel. Its advantage here is measured on
  repeatedly-used development pools.
* That any arm is ready for confirmation. **None is**, and 2016 remains unscored.
* That the comparison is a benchmark of the cited methods. It is a bounded adaptation
  study with every departure declared.
