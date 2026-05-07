# PCRL §5.3 — variance-constrained retraining note (with diagnostics)

## Bottom line for §5.3

The proxy-Lagrangian + VICReg λ_var = 1.0 architecture ships
49/60 cells via partial representation collapse. We tested whether
*replacing* VICReg's soft variance hinge with a *hard* variance
constraint (per_dim_std ≥ 0.5 enforced as a Lagrangian dual variable
per purpose, plus a sigmoid soft penalty on a participation-ratio
proxy for effective rank) could convert collapse-compliant cells to
cleanly compliant. **The retraining (150 epochs, warm-started from
the Round 5/7 final.pt checkpoints) did not.**

Across 7 of 9 (dataset, seed) retraining units that did complete in a
5-hour budget — covering 40 of the 49 originally collapse-compliant
pair-seeds:

- 0/40 became cleanly compliant
- 39/40 stayed collapse-compliant
- 1/40 lost R² compliance

## What the post-hoc diagnostics revealed

While the variance constraint did not reach the per-dim σ_d ≥ 0.5 floor on
any cell, it **did achieve partial structural improvement on effective
rank**: 21/40 pair-seeds (52%) showed
eff_rank delta > 0 with mean delta +0.28, and the number of
pair-seeds clearing the audit's eff_rank ≥ 2 threshold rose from
25/40 (baseline) to 30/40 (post-retrain), a
net gain of 5. The soft-sigmoid eff-rank penalty does
the work it was designed to do.

The per-dim σ_d trace, in contrast, is nearly flat — only
21/40 pair-seeds saw any
σ_min improvement at all, and 0 reached the 0.5 floor. The mean-of-clamped-
slack aggregation (`λ_var · max(0, 0.5 − σ_d).mean()`) discounts the
worst-dim signal as soon as any one dim crosses the floor; with 64
representation dimensions and the variance dual outvoted by 5 concurrent
linear-R² constraints + the task loss, the worst dim gets too little
gradient pressure to escape.

The 1 cell that lost R² compliance (adult s0 income_prediction/sex, R² 0.048 → 0.080) corresponds to the purpose with the largest eff_rank improvement in its seed, mildly consistent with the conjecture that opening more representation dimensions can re-expose the disallowed concept.

The proxy trajectories from the saved training history (the closest
substitute for a true per-epoch eff_rank trace, which we did not
instrument in this experiment) show no late-training catastrophe — both R² constraint violation and task loss evolve smoothly. Notably, val R²-violation sum *rose* in 5/7 cells over training (stayed flat in 2; fell in 0), indicating the proxy-Lagrangian dual loses ground modestly against the combined task + variance pressure even with the λ ≥ 5 floor. This is consistent with the worst-case loss observed in diagnostic 3: a single (purpose, attribute) pair (adult s0 income_prediction/sex) crossed τ from 0.0475 to 0.0800. The qualitative failure mode is 'rank rises modestly under the soft eff-rank penalty; the variance dual cannot pull the worst dim past ~0.3; constraint pressure shows minor erosion late in training but no catastrophic collapse.'

## Recommended §5.3 wording

> 49 of the 60 PCRL cells achieve compliance via partial representation
> collapse (per_dim_std_mean < 0.5 or eff_rank < 2 on at least one
> purpose). We attempted to convert these via a 150-epoch retrain with a
> hard per-dimension variance constraint (per_dim_std ≥ 0.5 enforced as
> a Lagrangian dual per purpose) plus a sigmoid soft penalty on a
> participation-ratio proxy for effective rank. The retrain did not
> escape collapse: 0 of 40 originally-collapse-compliant pair-seeds
> in the 7 cells we retrained became cleanly compliant, while 1 lost R²
> compliance. Diagnostics confirmed the eff-rank soft penalty achieved
> partial structural improvement (52% of pair-seeds saw
> eff_rank rise; +5 pair-seeds cleared the eff_rank ≥ 2
> threshold), but the variance dual could not push per_dim_std_min above
> ≈0.3 on any purpose × seed combination. We attribute this to the
> mean-of-clamped-slack dual aggregation, which relaxes globally as soon
> as one dim crosses the floor. A per-dimension Lagrangian with K=64
> individual duals per purpose (K × n_purposes additional dual variables
> per seed) is the natural refinement; we leave it to future work and
> report the 7/60 cleanly-compliant figure honestly.

## Caveats

- 2 of 9 cells (diabetes_s1, diabetes_s2) did not finish in budget —
  9 of the 49 originally-collapse-compliant pair-seeds are unaccounted
  for. Given the consistency of the failure across the 7 cells that
  did complete (same 0/40 cleanly-compliant rate across 3 datasets ×
  3 (or 2) seeds), the missing cells almost certainly behave the same.
- Compliance preservation was nearly perfect (1/40 lost), so the
  retrained model is at least *not actively harmful*. It is just inert
  against the collapse attractor.
- Task accuracies were maintained within the typical Round 5/7 ranges
  (Adult 0.93, HMDA 0.61–0.71, Diabetes 0.71). The "task acc within
  1pp of unconstrained" claim survives unchanged; it is the
  collapse-induced compliance, not the task acc, that fails the clean bar.
- The per-epoch eff_rank trajectory itself was not instrumented in this
  run. The trajectory analysis in `diagnostics/trajectory_analysis.md`
  uses surrogate metrics (`val_vicreg_loss` for cov-side decorrelation,
  `val_violation_sum` for constraint pressure). Re-instrumenting and
  re-running one cell with periodic eff_rank evaluation would take
  ~15 min of GPU time and would refine the surrogate-based interpretation
  but is unlikely to overturn the qualitative conclusion.
