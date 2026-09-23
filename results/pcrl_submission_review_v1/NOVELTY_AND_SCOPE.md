# NOVELTY_AND_SCOPE — what is new here, and what is not

## Not new, and cited as such

| component | prior art | our use |
|---|---|---|
| Convexity of finite-alphabet leakage–distortion design for a fixed joint law | Calmon and Fawaz 2012; after quantising to a finite code, Salamatian et al. 2015 | used directly; **no new optimisation result is claimed** |
| Nullspace feasibility criterion for perfect privacy; LP reduction of the zero-leakage problem **with mutual-information utility** | Rassouli and Gündüz, *On Perfect Privacy* | cited for feasibility; our zero-budget case is conditional and linear by the same argument, not by their LP |
| A new release designed **beside fixed earlier releases** under an incremental-leakage budget | Erdogdu and Fawaz 2015 | closest antecedent for the "release beside what is already out" shape |
| **Per-party and collusion mutual-information constraints** on sequential releases | Taylor et al. 2026 | the closest antecedent for our coalition constraint structure; our novelty narrows accordingly |
| Randomised finite pre-processing under fairness constraints | Calmon et al. 2017 | related optimised pre-processing |
| Privacy funnel and its nonconvexity | Makhdoumi et al. | cited to explain why we do **not** impose a utility constraint: the nonconvexity comes from that constraint |
| Multi-user / side-information utility–privacy tradeoffs | Sankar et al.; Liao et al. | nearest multi-consumer prior art; designs one sanitised release for many users |
| Linear concept erasure and task-preserving variants | LEACE; SPLINCE | executed as baselines with disclosed label access |
| Adversarial fair/transferable representations | Madras et al. | prior art for the adversarially trained comparator channel |
| Closed-form eigenvalue-sign rank selection | SARL / OptNet-ARL / K-TOpt | prior art; historical comparators |
| Convexity of conditional mutual information in the channel for a fixed distribution; linearity of expected distortion for a fixed cost table | standard | stated as standard, not derived |

**The convex programme in this paper is an established one applied to a new release contract.** Calling it
a new algorithm would be wrong, and a renamed established programme is not a contribution.

## What is actually ours

1. **The setting**: the conditioning view is a *third party's immutable published prediction service*
   that a coalition partner also holds, with conflicting permitted and protected uses per recipient.
   Per-party and collusion constraints as such are not new (Taylor et al. 2026); the third-party
   immutable-service conditioning is what remains ours.
2. **The measurement contract**: per-recipient and coalition audit roles, independently fitted attackers,
   additional disclosure always reported against the fixed service baseline, absolute recovery beside every
   increment, negative increments retained unclipped, per-endpoint results never averaged.
3. **The audit caution**: a pooled one-hot score is a prevalence-weighted average of per-class scores, so
   it can conceal a rare-class direction that a per-class audit exposes. **The identity itself is not
   original** — it is the familiar variance-weighted multi-output $R^2$ (Terminal 3, M7) — and the
   dominant axis is only a lower bound on the best linear sensitive direction. What survives is the
   auditing caution, demonstrated on real checkpoints, not a new identity.
4. **The evaluated operating point**: a task-directed release that improves the permitted task over the
   strongest prior channel with bounds excluding zero, reported together with the seven unresolved
   sensitive endpoints and the matched deterministic baseline that performs comparably.
5. **The corrections** in `CORRECTIONS_FINAL.md`, including the refuted regression-to-classification
   guarantee.

## The honest summary sentence

A local task-directed release beating the prior channel on the permitted task is useful without
demonstrating that randomisation is necessary, that coalition conditioning helps, or that the mechanism is
competitive under the registered conjunction. **The method contribution is an adaptation plus an evaluated
operating point**, and the paper says so in those words.
