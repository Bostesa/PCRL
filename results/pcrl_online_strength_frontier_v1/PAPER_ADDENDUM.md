# Paper addendum: online strength frontier (exploratory development; not for the abstract as a positive result)

**Setting.**
- Two recipients (income; six-class occupation group) receive 16-dimensional learned features plus their deployed task head's centred logits.
- Recovery of SEX is measured with a fixed, strong attacker slate (fitted on AUDIT_FIT, selected on INNER_SELECTION), for each recipient and for the coalition of both views.
- Training uses ONLINE adversarial critics and a recovery surrogate. Adversarial representation learning follows Madras et al. (ICML 2018).
- Two update rules are compared:
  - raw penalties, θ ← θ − η(t + β∇P);
  - per-encoder normalised updates, q_i = ρ s_i(a) ‖t_i‖/‖p_i‖ · p_i, at fixed strengths and fixed income/occupation allocations.
- All arms start from the same warm state, run 40 epochs on a common minibatch order, and are selected on inner roles only.
- The assessment is a locked, pushed, single scoring of a fixed union of four previously used pools (13,936 rows). This is a reused benchmark, not confirmation.

**Results (means over 3 seeds).**

| Model | Pair AUC | Income-recipient AUC | Occupation-recipient AUC | Income acc. | Occupation acc. |
|---|---:|---:|---:|---:|---:|
| U (task only) | 0.883 | 0.859 | 0.878 | 0.844 | 0.475 |
| Raw joint, β 0.3 (incumbent; best development model) | 0.802 | 0.756 | 0.787 | 0.845 | 0.467 |
| Normalised joint, ρ 3 (descriptive; no feasible nominee) | 0.786 | 0.746 | 0.758 | 0.844 | 0.462 |
| Raw joint, β 0.6 (descriptive; no feasible nominee) | 0.763 | 0.704 | 0.742 | 0.840 | 0.456 |
| Best local (normalised local, ρ 3) | 0.841 | 0.751 | 0.777 | 0.844 | 0.466 |
| Official FARE (task-infeasible) | 0.705 | 0.685 | 0.634 | 0.844 | 0.451 |
| Official LEACE (task-infeasible) | 0.871 | 0.846 | 0.864 | 0.794 | 0.436 |

**Findings for the paper (keep the LEACE and FARE rows).**
- No method advantage. Both registered nominees (normalised joint; ordinary raw joint) had no feasible configuration on inner selection, and all three nine-clause conjunctions are NOT_ESTABLISHED.
- Matching the incumbent's measured average strength (realized per-encoder ratio about 3, varying 1.7–4.5 within a run) with a constant normalised ratio of 3 reproduced the raw trade-off:
  - 0.016 lower pair AUC [0.011, 0.020];
  - 0.5 occupation points lower [0.02, 0.95];
  - it sits on the raw β 0.3 → 0.6 line (post hoc interpolation).
- Lower recovery cost occupation accuracy in every family. The descriptive nominees lose 1.4 and 2.0 points against U, beyond a one-point guard.
- Fixed recipient allocation shifted leakage between recipients or traded it for occupation accuracy.
- Joint training lowered coalition recovery relative to local training at identical update-norm budgets (−0.055 pair AUC at ρ 3).
- The best development model still permits substantial recovery (pair AUC 0.80). It is not a privacy guarantee.

**Limits.**
- All rows are historically exposed, and earlier results shaped this study.
- Intervals condition on fitted models and the declared attacker slate.
- No population, differential-privacy, mutual-information or optimality claim is made.
- Norm rescaling and coefficient banks are not algorithmic novelty.
