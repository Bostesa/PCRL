# Diabetes PCRL — collapse diagnosis (2026-04-25)

**STATUS: PROVISIONAL.** PCRL results in this directory should not be treated as final until rerun with fixed early-stopping criterion.

## Summary

PCRL seed-0 encoder on Diabetes is **functionally collapsed**: every task head predicts pure majority class regardless of input.

| | per_dim_std_mean | l2_norm_std | eff_rank /64 | primary_diag acc | readmission acc | medchange acc |
|---|---|---|---|---|---|---|
| Standard s0 | 0.67 | 2.12 | 51.9 | **0.419** (maj 0.314, +0.105) | 0.912 (maj 0.912, +0.000) | **0.999** (maj 0.550, +0.450) |
| PCRL s0 (per purpose) | 1.07–2.02 | 9.8–18.7 | 28–32 | **0.314** (maj 0.314, +0.000) | 0.912 (maj 0.912, +0.000) | **0.550** (maj 0.550, +0.000) |

PCRL representations have non-trivial variance (per-dim std 1–2, effective rank 28–32) — so this isn't the "constant output" collapse fingerprint Folktables showed. But the variance is in directions **orthogonal to task signal**: task heads converge to predicting class priors regardless of z. Outcome is identical to Folktables: PCRL cannot beat majority on any task.

The reported "PCRL 5/6 pairs pass with std=0 across seeds" is degenerate: a head that emits constant majority predictions trivially has R² = 0 with any attribute and acc-baseline ≤ 0 (auditors can't beat majority on a constant-output target). The fairness audit measured nothing because the model encoded nothing useful.

## Source data
- Per-purpose representation stats and task accuracies: extracted from `checkpoints/diab_pcrl_s0/best.pt` and `checkpoints/diab_std_s0/best.pt` on instance i-0202e8ea199fdad3a (2026-04-25)
- Test set: 10,728 samples, 170 features, 3 purposes
