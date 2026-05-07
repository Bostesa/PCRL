# PCRL §5.3 — variance-constrained retraining note

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

The hard variance constraint pulled `per_dim_std_min` from a baseline
range of 0.04–0.38 to a post-train range of 0.06–0.29 — never crossing
the 0.5 floor on any purpose × seed combination. The mean-of-clamped-slack
formulation averages across the 64 representation dimensions, so once
*any* dim crosses the floor, dual ascent relaxes for the whole purpose,
permitting other dims to drift back down. With five concurrent linear-R²
constraints and the task loss competing for the same 64-dim representation
budget, the variance dual is consistently outvoted.

## Recommended §5.3 wording

> 49 of the 60 PCRL cells achieve compliance via partial representation
> collapse (per_dim_std_mean < 0.5 or eff_rank < 2 on at least one
> purpose). We attempted to convert these via a 150-epoch retrain with a
> hard per-dimension variance constraint (per_dim_std ≥ 0.5 enforced as
> a Lagrangian dual per purpose) plus a sigmoid soft penalty on a
> participation-ratio proxy for effective rank. The retrain did not
> escape collapse: 0 of 40 originally-collapse-compliant pair-seeds in
> the 7 cells we retrained became cleanly compliant, while 1 lost R²
> compliance. The dual ascent could not push per_dim_std_min above
> ≈0.3 on any purpose × seed combination — the mean-of-clamped-slack
> formulation relaxes globally as soon as one dim crosses the floor,
> while five concurrent R² constraints and the task loss outvote the
> variance dual on the 64-dim representation. We therefore report the
> 7/60 cleanly-compliant figure honestly and note that hard variance
> regularization is not a sufficient fix; full elimination of the
> collapse mode would require either a per-dim Lagrangian (one dual
> per dimension, 64 × per-purpose duals total) or a structural change
> that decouples concept erasure from representation geometry.

## Caveats

- 2 of 9 cells (diabetes_s1, diabetes_s2) did not finish — their
  results are unknown. Total unfinished pair-seeds = 9 (of the original
  49 collapse-compliant). The 0/40 result is on the *completed* subset;
  remaining 9 could in principle behave differently, but the consistency
  of the failure across 3 datasets × 3 (or 2) seeds suggests they would
  not.
- Compliance preservation was nearly perfect (only 1 lost), so the
  retraining is at least *not actively harmful*. It is just inert against
  the collapse attractor.
- Task accuracies were maintained within typical Round 5/7 ranges
  (Adult 0.93, HMDA 0.61–0.71, Diabetes 0.71). The "task acc within
  1pp of unconstrained" claim is supported by the data; it is the
  collapse-induced compliance, not the task acc, that fails the clean
  bar.
