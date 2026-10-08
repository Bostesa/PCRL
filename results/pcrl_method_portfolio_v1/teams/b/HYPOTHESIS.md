# Team B: retrained task-sufficient bottlenecks

**Status: design draft, before any Team B real-data fit.** This is an exploratory development search on repeatedly used Adult outcomes. It is not a fresh confirmation study or a privacy guarantee.

## Question

Can two purpose-specific encoders learned together from the permitted Adult predictors preserve task accuracy and useful probability information while disclosing less SEX information through each recipient's **complete feature-plus-head view** and their aligned coalition than matched task-only, local, sequential, and ordinary raw adversarial models?

The proposed information-changing operation is a stochastic representation bottleneck **before** each task head: an input is mapped to a distribution over a compact latent vector, and only one draw of that vector and the output computed from that same draw are released. An explicit KL-to-fixed-prior rate cost discourages carrying unnecessary input distinctions. Task labels and frozen-teacher probabilities guide each representation toward its declared task. Local and coalition critics supply a separate empirical SEX-protection pressure. None of these terms by itself certifies SEX privacy.

The common development targets are each task accuracy within 0.010, mean NLL excess at most 0.010 nats, and mean multiclass Brier excess at most 0.005 against **both** U0 and Ucal; coalition SEX AUC benefit at least 0.020 beyond the strongest applicable eligible control; and no recipient AUC worsening over 0.010. S1/S2 own the exact simultaneous-interval and boundary rules. Useful minority/class recall must be reported even if a mean gate passes.

## Pre-fit predictions and possible disproof

1. A task-only stochastic bottleneck may compress some SEX proxies with less utility cost than the old mandatory final LEACE map. It may also retain task-correlated SEX signal that cannot be removed at the common utility limits.
2. The joint critic should matter only where the paired vectors expose a complementary clue beyond the local views. A no-complementarity synthetic law should not generate a claimed joint advantage.
3. If deterministic mean release at matched architecture, KL term, training budget and attack slate performs as well, attribute any benefit to retrained bottleneck learning, **not** to sampling.
4. If separately private or either sequential order matches the joint arm, do not claim a joint-design contribution.
5. If the full feature view or a fresh adaptive attacker recovers the apparent gain, stop the lane. A score-only improvement would not answer this feature-sharing question.

## Evidence boundary

Use only S2-admitted fitting/calibration/attacker roles. Never access HEAD_VALIDATION labels, old assessment person arrays or labels, or another confirmation population. Report every seed, recipient and coalition. Any promising result is a selected development lead. The repository's prior JCV experiment found that mandatory final LEACE harmed both Adult tasks; this proposal removes that exact eraser rather than relabeling the old raw adversarial penalty. The historical result is a warning, not a proof that a task-sufficient stochastic bottleneck will work.
