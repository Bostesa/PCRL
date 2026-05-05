# HMDA PCRL — collapse diagnosis (2026-04-25)

**STATUS: PROVISIONAL.** PCRL results in this directory should not be treated as final until rerun with fixed early-stopping criterion.

## Summary

PCRL seed-0 encoder on HMDA is **collapsed** (matches Folktables fingerprint): low representation variance, every task head predicts pure majority class.

| | per_dim_std_mean | l2_norm_std | eff_rank /64 | loan_decision acc | loan_amount_band acc | tract_denial_high acc |
|---|---|---|---|---|---|---|
| Standard s0 | 1.22 | 7.01 | 35.0 | **0.911** (maj 0.893, +0.018) | **0.601** (maj 0.211, +0.390) | **0.653** (maj 0.634, +0.019) |
| PCRL s0 (per purpose) | 0.009–0.042 | 0.03–0.11 | 18–22 | **0.893** (maj 0.893, +0.000) | **0.211** (maj 0.211, -0.000) | **0.634** (maj 0.634, +0.000) |

PCRL per-dim std is 30× smaller than Standard's (0.04 vs 1.22 max). Effective rank cut from 35 → 18-22. Task accuracies pinned to per-task majority baseline to 4 decimals across all three tasks. Same functional outcome as Folktables/Diabetes.

The reported "PCRL 5.33/6 pairs pass, Δ-values ≤ 0.020 across attrs" is degenerate: with task heads emitting majority predictions, the audit's delta and R² metrics are trivially satisfied. The fairness audit measured nothing.

## Source data
- Extracted from `checkpoints/hmda_pcrl_0/best.pt` and `checkpoints/hmda_std_0/best.pt` on instance [redacted-instance-id] (2026-04-25)
- Test set: 13,661 samples, 78 features, 3 purposes
