# NOVELTY_AND_SCOPE — what is new here, and what is not

## Not new, and cited as such

| component | prior art | our use |
|---|---|---|
| Finite-alphabet privacy–utility optimisation; nullspace feasibility criterion; LP form of the zero-budget linear-cost problem | Rassouli and Gündüz, *On Perfect Privacy* | used directly; **no new optimisation result is claimed** |
| Privacy funnel and its nonconvexity | Makhdoumi et al. | cited to explain why we do **not** impose a utility constraint: the nonconvexity comes from that constraint |
| Multi-user / side-information utility–privacy tradeoffs | Sankar et al.; Liao et al. | nearest multi-consumer prior art; designs one sanitised release for many users |
| Linear concept erasure and task-preserving variants | LEACE; SPLINCE | executed as baselines with disclosed label access |
| Adversarial fair/transferable representations | Madras et al. | prior art for the adversarially trained comparator channel |
| Closed-form eigenvalue-sign rank selection | SARL / OptNet-ARL / K-TOpt | prior art; historical comparators |
| Convexity of conditional mutual information in the channel for a fixed distribution; linearity of expected distortion for a fixed cost table | standard | stated as standard, not derived |

**The convex programme in this paper is an established one applied to a new release contract.** Calling it
a new algorithm would be wrong, and a renamed established programme is not a contribution.

## What is actually ours

1. **The release problem**: an immutable published service output, a channel added for one recipient only,
   and explicitly conflicting permitted and protected uses per recipient and for the coalition.
2. **The measurement contract**: per-recipient and coalition audit roles, independently fitted attackers,
   additional disclosure always reported against the fixed service baseline, absolute recovery beside every
   increment, negative increments retained unclipped, per-endpoint results never averaged.
3. **The dominant-axis auditing result** and its exact convex-combination identity: an aggregate one-hot
   score is a prevalence-weighted combination of per-class scores, so it can conceal a rare-class
   direction. This is an original contribution that survives checking.
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
