# Closing audit analysis

Prediction-only passes 3/3 validation and 3/3 fresh-test audits under the frozen per-purpose utility and three-family leakage criteria.
The unchanged native scalar predictions, longer fresh MLP fits, and complementary trees close the declared audit. This is empirical sufficiency for the two fixed tasks under tested attacks, not universal privacy or sufficiency for unknown future tasks.

## Per-target scalar versus oracle differences

Paired prediction-only minus oracle on the common fresh test; mean ± sample SD, no significance claim.

| Metric | Paired difference |
|---|---:|
| utility_p1_U | -0.001960 ± 0.000097 |
| utility_p2_V | -0.003145 ± 0.000831 |
| individual_linear | 0.000030 ± 0.000189 |
| combined_linear | -0.000048 ± 0.000124 |
| linear_p1_V | 0.000009 ± 0.000138 |
| linear_p1_S | -0.000037 ± 0.000075 |
| linear_p2_U | 0.000077 ± 0.000197 |
| linear_p2_S | -0.000017 ± 0.000070 |
| linear_combined_S | -0.000048 ± 0.000124 |
| individual_mlp | 0.000272 ± 0.000384 |
| combined_mlp | 0.000129 ± 0.000341 |
| mlp_p1_V | 0.000607 ± 0.001552 |
| mlp_p1_S | 0.000316 ± 0.000315 |
| mlp_p2_U | -0.000276 ± 0.000557 |
| mlp_p2_S | -0.000019 ± 0.000211 |
| mlp_combined_S | 0.000129 ± 0.000341 |
| individual_histgb | 0.000036 ± 0.000739 |
| combined_histgb | -0.004556 ± 0.009157 |
| histgb_p1_V | -0.001302 ± 0.013907 |
| histgb_p1_S | -0.004734 ± 0.011274 |
| histgb_p2_U | -0.001869 ± 0.011597 |
| histgb_p2_S | -0.000250 ± 0.016301 |
| histgb_combined_S | -0.004556 ± 0.009157 |
| direct_p1_U | -0.002147 ± 0.000101 |
| direct_p2_V | -0.003255 ± 0.000874 |

## Frozen full-representation positive control

| Target | Saved training validation/test | Saved continued validation/test | Fresh MLP validation/test | Trees validation/test |
|---|---|---|---|---|
| p1_V | 0.013991 / 0.025645 | 0.358366 / 0.380132 | 0.362621 / 0.349013 | 0.130771 / 0.155012 |
| p1_S | 0.027886 / 0.042964 | 0.191509 / 0.170615 | 0.173521 / 0.149281 | 0.052776 / 0.031187 |
| p2_U | 0.047024 / 0.057699 | 0.287370 / 0.256471 | 0.328057 / 0.296702 | 0.241212 / 0.195354 |
| p2_S | 0.014323 / 0.013812 | 0.191671 / 0.187390 | 0.200067 / 0.185202 | 0.051726 / 0.045486 |
| combined_S | 0.012826 / -0.038064 | 0.355174 / 0.346936 | 0.376137 / 0.352950 | 0.126506 / 0.125327 |

The full control retains its historical infeasibility; the audit does not improve protection. No final map was refitted. The saved continued weights retain prior exposure and are compatible only with the full release.

## Budget and limits

Fresh MLPs use two1800-step trajectories per view, each460800 row presentations, totaling921600. This exceeds900+640 updates per successful continued trajectory but uses2048 distinct attacker examples; original training weights also saw3072 other examples. Input dimensions and output sharing differ, so equal nominal updates cannot establish equal attack strength. HistGB uses200 full fitting-data passes per target; these are separately counted, not converted into Adam updates.
Task probes fit4096 representation-training examples, attackers fit2048 attacker examples, validation2048 selects checkpoints/restarts, and fresh test4096 is generated after saved selection. Observation preprocessing and all release tensors remain unchanged. Covariance certificates play no role.
Reused validation makes this an exploratory diagnosis. A failed attacker does not bound all attacks. Exposed-target controls check competence including S, and oracle controls check finite-sample spurious predictive success.

The original toy benchmark stops here. The research decision must depend on a defensible real application and strong simple controls, not another protection-strength sweep.

## Measured internal runtime

- Seed0: 14.801s.
- Seed1: 14.350s.
- Seed2: 19.626s.
- Sum: 48.777s; process wall timings are in execution logs.
