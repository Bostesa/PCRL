# Fixed nonlinear upstream-adaptation pilot

Final-test predictive R²; mean ± sample SD. Negative scores are retained and do not denote negative information.

| Method | Linear U | Linear V | MLP U | MLP V | Worst linear leak | Worst MLP leak | Combined S linear | Combined S MLP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_frozen_no_erasure | 0.9976 ± 0.0002 | 0.9962 ± 0.0010 | 0.9975 ± 0.0002 | 0.9962 ± 0.0010 | 0.9976 ± 0.0001 | 0.9963 ± 0.0004 | 0.3307 ± 0.1231 | 0.5062 ± 0.1324 |
| B_frozen_leace | 0.9972 ± 0.0004 | 0.9955 ± 0.0006 | 0.9974 ± 0.0003 | 0.9958 ± 0.0006 | 0.0040 ± 0.0075 | 0.3264 ± 0.0424 | -0.0042 ± 0.0028 | 0.1563 ± 0.0406 |
| C_task_only | 0.9976 ± 0.0004 | 0.9959 ± 0.0006 | 0.9978 ± 0.0002 | 0.9962 ± 0.0006 | 0.0037 ± 0.0073 | 0.3280 ± 0.0674 | -0.0055 ± 0.0081 | 0.1622 ± 0.0464 |
| D_protection_0.1 | 0.8607 ± 0.1832 | 0.9537 ± 0.0285 | 0.9114 ± 0.1104 | 0.9723 ± 0.0132 | 0.0003 ± 0.0014 | 0.2711 ± 0.1199 | -0.0073 ± 0.0052 | 0.1085 ± 0.0122 |
| D_protection_1 | 0.8535 ± 0.1764 | 0.8188 ± 0.1287 | 0.9156 ± 0.1001 | 0.8985 ± 0.0574 | -0.0002 ± 0.0024 | 0.2453 ± 0.2025 | -0.0055 ± 0.0076 | 0.0909 ± 0.0052 |

Worst leak = maximum over P1→V,S and P2→U,S within each seed. Combined recipient may access U,V; only S is prohibited.

## Per-seed test results

| Seed | Method | Linear U | Linear V | MLP U | MLP V | Worst linear leak | Worst MLP leak | Combined S linear | Combined S MLP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | A_frozen_no_erasure | 0.99773 | 0.99665 | 0.99766 | 0.99664 | 0.99769 | 0.99670 | 0.40074 | 0.64770 |
| 0 | B_frozen_leace | 0.99754 | 0.99543 | 0.99765 | 0.99575 | -0.00038 | 0.35766 | -0.00133 | 0.12742 |
| 0 | C_task_only | 0.99777 | 0.99566 | 0.99790 | 0.99600 | 0.00088 | 0.37770 | -0.01456 | 0.12108 |
| 0 | D_protection_0.1 | 0.99140 | 0.92102 | 0.99294 | 0.95743 | -0.00117 | 0.40678 | -0.00602 | 0.09503 |
| 0 | D_protection_1 | 0.99419 | 0.84096 | 0.99479 | 0.89859 | -0.00270 | 0.47346 | -0.00358 | 0.08543 |
| 1 | A_frozen_no_erasure | 0.99737 | 0.99697 | 0.99733 | 0.99692 | 0.99740 | 0.99629 | 0.40280 | 0.48564 |
| 1 | B_frozen_leace | 0.99672 | 0.99613 | 0.99715 | 0.99650 | 0.01258 | 0.34335 | -0.00416 | 0.20265 |
| 1 | C_task_only | 0.99709 | 0.99654 | 0.99755 | 0.99684 | 0.01194 | 0.35487 | 0.00123 | 0.21258 |
| 1 | D_protection_0.1 | 0.93936 | 0.96717 | 0.95559 | 0.97713 | 0.00064 | 0.22708 | -0.00291 | 0.11159 |
| 1 | D_protection_1 | 0.91080 | 0.93502 | 0.94885 | 0.95586 | -0.00008 | 0.17553 | 0.00096 | 0.09589 |
| 2 | A_frozen_no_erasure | 0.99761 | 0.99512 | 0.99759 | 0.99511 | 0.99760 | 0.99582 | 0.18850 | 0.38522 |
| 2 | B_frozen_leace | 0.99742 | 0.99494 | 0.99753 | 0.99524 | -0.00029 | 0.27813 | -0.00698 | 0.13879 |
| 2 | C_task_only | 0.99783 | 0.99541 | 0.99793 | 0.99578 | -0.00186 | 0.25132 | -0.00321 | 0.15297 |
| 2 | D_protection_0.1 | 0.65122 | 0.97306 | 0.78577 | 0.98248 | 0.00153 | 0.17933 | -0.01298 | 0.11888 |
| 2 | D_protection_1 | 0.65559 | 0.68054 | 0.80305 | 0.84098 | 0.00218 | 0.08698 | -0.01390 | 0.09132 |

## Every forbidden target

| Seed | Method | Attacker | P1→V | P1→S | P2→U | P2→S | Combined→S |
|---|---|---|---:|---:|---:|---:|---:|
| 0 | A_frozen_no_erasure | linear | 0.99662 | 0.40074 | 0.99769 | 0.40074 | 0.40074 |
| 0 | A_frozen_no_erasure | mlp | 0.99556 | 0.65031 | 0.99670 | 0.63735 | 0.64770 |
| 0 | B_frozen_leace | linear | -0.00544 | -0.00038 | -0.00934 | -0.00134 | -0.00133 |
| 0 | B_frozen_leace | mlp | 0.34777 | 0.04026 | 0.35766 | 0.07204 | 0.12742 |
| 0 | C_task_only | linear | -0.00620 | 0.00088 | -0.00825 | -0.00373 | -0.01456 |
| 0 | C_task_only | mlp | 0.37770 | 0.03779 | 0.34375 | 0.07133 | 0.12108 |
| 0 | D_protection_0.1 | linear | -0.00584 | -0.00406 | -0.00503 | -0.00117 | -0.00602 |
| 0 | D_protection_0.1 | mlp | 0.40678 | 0.04594 | 0.13377 | 0.05443 | 0.09503 |
| 0 | D_protection_1 | linear | -0.00569 | -0.00326 | -0.00695 | -0.00270 | -0.00358 |
| 0 | D_protection_1 | mlp | 0.47346 | 0.03029 | 0.04995 | 0.07248 | 0.08543 |
| 1 | A_frozen_no_erasure | linear | 0.99698 | 0.40280 | 0.99740 | 0.40280 | 0.40280 |
| 1 | A_frozen_no_erasure | mlp | 0.99535 | 0.47567 | 0.99629 | 0.46119 | 0.48564 |
| 1 | B_frozen_leace | linear | 0.01258 | -0.00503 | -0.00118 | -0.00401 | -0.00416 |
| 1 | B_frozen_leace | mlp | 0.29011 | 0.08550 | 0.34335 | 0.15970 | 0.20265 |
| 1 | C_task_only | linear | 0.01194 | -0.00476 | -0.00089 | -0.00445 | 0.00123 |
| 1 | C_task_only | mlp | 0.29058 | 0.07743 | 0.35487 | 0.17264 | 0.21258 |
| 1 | D_protection_0.1 | linear | -0.00387 | -0.00188 | 0.00064 | -0.00379 | -0.00291 |
| 1 | D_protection_0.1 | mlp | 0.12223 | 0.11650 | 0.22708 | 0.10265 | 0.11159 |
| 1 | D_protection_1 | linear | -0.00373 | -0.00008 | -0.00462 | -0.00096 | 0.00096 |
| 1 | D_protection_1 | mlp | 0.11558 | 0.07775 | 0.17553 | 0.09271 | 0.09589 |
| 2 | A_frozen_no_erasure | linear | 0.99493 | 0.18850 | 0.99760 | 0.18850 | 0.18850 |
| 2 | A_frozen_no_erasure | mlp | 0.99441 | 0.37677 | 0.99582 | 0.37028 | 0.38522 |
| 2 | B_frozen_leace | linear | -0.00029 | -0.00703 | -0.01175 | -0.00749 | -0.00698 |
| 2 | B_frozen_leace | mlp | 0.27813 | 0.16193 | 0.21849 | 0.08375 | 0.13879 |
| 2 | C_task_only | linear | -0.00186 | -0.00649 | -0.01261 | -0.00738 | -0.00321 |
| 2 | C_task_only | mlp | 0.25132 | 0.17222 | 0.23061 | 0.08245 | 0.15297 |
| 2 | D_protection_0.1 | linear | 0.00153 | -0.00520 | -0.00821 | -0.01030 | -0.01298 |
| 2 | D_protection_0.1 | mlp | 0.02091 | 0.13450 | 0.05308 | 0.17933 | 0.11888 |
| 2 | D_protection_1 | linear | 0.00218 | -0.00428 | -0.00908 | -0.01391 | -0.01390 |
| 2 | D_protection_1 | mlp | 0.03130 | 0.08682 | 0.02147 | 0.08698 | 0.09132 |

## Validation selection

- Seed 0: full grid **D_protection_0.1**, feasible=False; positive strengths **D_protection_0.1**, feasible=False. Selection recorded before test generation.
- Seed 1: full grid **D_protection_1**, feasible=False; positive strengths **D_protection_1**, feasible=False. Selection recorded before test generation.
- Seed 2: full grid **D_protection_1**, feasible=False; positive strengths **D_protection_1**, feasible=False. Selection recorded before test generation.

## Pretraining

Direct task-head test R² before → after task pretraining (selection uses validation only):

- Seed 0: U: -0.07368 → 0.99756; V: -0.08586 → 0.99659
- Seed 1: U: -0.07253 → 0.99737; V: -0.02774 → 0.99693
- Seed 2: U: -0.03334 → 0.99749; V: -0.08921 → 0.99502

Total measured complete-seed computation: 30.56 seconds.

This small pilot compares the recorded attacks and utility only. Near-zero linear R² is not nonlinear privacy, and a failed attack is not a universal guarantee. See PROTOCOL.md for objective, splits, budgets, thresholds, and method limitations.
