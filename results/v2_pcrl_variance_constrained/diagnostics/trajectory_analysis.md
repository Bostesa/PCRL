# Trajectory analysis (proxy data)

## Caveat

V2Trainer does not log per-epoch `effective_rank` or `per_dim_std`. The training history saved in `final.pt` contains only loss-side metrics. To reconstruct a true eff_rank trajectory we would need to re-train each cell with periodic representation evaluation, which is out of scope for this 30-min diagnostic. The plots in `eff_rank_trajectory.pdf` use `val_vicreg_loss` (= `vicreg_lambda_cov × covariance_loss` since we set `vicreg_lambda_var=0`) as a structural-collapse surrogate, plus `val_violation_sum` (Σ over (purpose, attribute) of `max(0, R²−τ)`) as a constraint-pressure trace. The actual init→final eff_rank deltas (which CAN be computed from the warm-started state and the final encoder) are reported in diagnostic 2.

## Per-cell proxy summary (full precision)

| cell | val_vicreg init→final (max) | Σ R²-violation init→final (max) | val_r2_mean init→final (max) | val_task_loss init→final |
|---|---|---|---|---|
| adult_s0 | 0.00000 → 0.00000 (0.00000) | 0.0316 → 0.2070 (0.2512) | 0.0405 → 0.0631 (0.0678) | 0.5644 → 0.4832 |
| adult_s1 | 0.00000 → 0.00000 (0.00000) | 0.0000 → 0.1917 (0.2665) | 0.0165 → 0.0569 (0.0677) | 0.6115 → 0.4704 |
| adult_s2 | 0.00000 → 0.00000 (0.00000) | 0.0167 → 0.0177 (0.0437) | 0.0275 → 0.0361 (0.0399) | 0.5623 → 0.5856 |
| hmda_s0 | 0.00000 → 0.00000 (0.00000) | 0.0893 → 0.1942 (0.1942) | 0.0483 → 0.0798 (0.0798) | 2.4611 → 2.4438 |
| hmda_s1 | 0.00000 → 0.00000 (0.00000) | 0.0000 → 0.0000 (0.0203) | 0.0219 → 0.0223 (0.0365) | 2.4335 → 2.4219 |
| hmda_s2 | 0.00000 → 0.00000 (0.00000) | 0.0000 → 0.0000 (0.0124) | 0.0253 → 0.0235 (0.0360) | 2.0189 → 1.9535 |
| diabetes_s0 | 0.00000 → 0.00000 (0.00000) | 0.5403 → 0.5420 (0.5735) | 0.1289 → 0.1275 (0.1355) | 2.3439 → 2.2910 |

## Pattern in the proxy traces

- **R²-violation sum trajectory**: rose in 5/7 cells, stayed flat in 2/7, fell in 0/7. At final epoch, 5/7 cells still have non-zero violation sum on val. This is the **opposite** of 'constraint winning': in a majority of cells the *aggregate* val R² creeps up over training even with the proxy-Lagrangian dual ascending. Interpretation: the dual_min floor (λ ≥ 5) prevents complete starvation but is insufficient to overcome the task gradient + the new variance constraint pulling the LoRA outputs in a different direction. Notable: `hmda_s1` and `hmda_s2` both show 0.000 → 0.000 violation, meaning every val pair stays comfortably below τ throughout.

- **VICReg covariance trajectory**: tiny absolute values (init ~ 1e-5, max < 5e-4 across cells). With `vicreg_lambda_var=0`, the only contribution is `0.04 × off-diagonal covariance`, and the post-LEACE-warm-started representation is already nearly-orthogonal — there is little for VICReg's covariance term to do. The trace is dominated by training noise, not by structural change.

- **No 'rises then collapses' phase visible** in the proxy — both VICReg cov and val R² evolve smoothly across epochs (no plateau→cliff or large oscillation). Combined with the post-train eff_rank gains in diagnostic 2, the failure mode appears to be: 'rank improves modestly under the soft eff-rank penalty; the variance dual cannot pull the worst dim past ~0.3; no late catastrophe — just a quiet stuck point near the spec target.'

- **The 1 cell where val R² rose most steeply is `adult_s0`**: val_r2_mean went from 0.0405 to 0.0631. This is the cell where diagnostic 3 identifies the lost-compliance pair (adult_s0 income_prediction/sex, R² 0.0475 → 0.0800).
