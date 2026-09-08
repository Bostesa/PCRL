# Nonlinear conflict pilot: paired analysis

Completed seeds: 0, 1, 2. Predictive R² is out of sample; negative scores are retained. Cells show mean ± sample SD across seeds. SD is undefined with only one seed.

A = frozen pretrained encoder; B = A plus per-purpose LEACE; C = task-only upstream adaptation plus recalibrated LEACE; D0.1/D1 = matched adaptation with protection strength 0.1/1 plus recalibrated LEACE. The positive selected arm reuses the previously saved validation choice between D0.1 and D1.

## Utility and remaining headroom

Task utility averages P1→U and P2→V within each seed. Headroom is the distance from B to predictive R²=1; LEACE loss is the paired A−B difference.

| Probe | A utility | B utility | B headroom (1−B) | LEACE utility loss (A−B) |
|---|---:|---:|---:|---:|
| linear | 0.99691 ± 0.00047 | 0.99636 ± 0.00016 | 0.00364 ± 0.00016 | 0.00055 ± 0.00031 |
| mlp | 0.99687 ± 0.00046 | 0.99664 ± 0.00023 | 0.00336 ± 0.00023 | 0.00024 ± 0.00025 |

## Every forbidden target

The combined recipient intentionally receives both task signals; only combined→S is prohibited. These are results for the fitted probe families and budgets.

| Arm | Probe | P1→V | P1→S | P2→U | P2→S | Combined→S |
|---|---|---:|---:|---:|---:|---:|
| A | linear | 0.99617 ± 0.00109 | 0.33068 ± 0.12314 | 0.99756 ± 0.00015 | 0.33068 ± 0.12314 | 0.33068 ± 0.12314 |
| A | mlp | 0.99510 ± 0.00061 | 0.50091 ± 0.13851 | 0.99627 ± 0.00044 | 0.48961 ± 0.13578 | 0.50619 ± 0.13244 |
| B | linear | 0.00228 ± 0.00928 | -0.00415 ± 0.00341 | -0.00742 ± 0.00554 | -0.00428 ± 0.00309 | -0.00416 ± 0.00282 |
| B | mlp | 0.30534 ± 0.03723 | 0.09590 ± 0.06150 | 0.30650 ± 0.07656 | 0.10516 ± 0.04759 | 0.15629 ± 0.04055 |
| C | linear | 0.00129 ± 0.00947 | -0.00346 ± 0.00386 | -0.00725 ± 0.00592 | -0.00518 ± 0.00193 | -0.00551 ± 0.00814 |
| C | mlp | 0.30653 ± 0.06468 | 0.09581 ± 0.06907 | 0.30975 ± 0.06876 | 0.10881 ± 0.05556 | 0.16221 ± 0.04645 |
| D0.1 | linear | -0.00272 ± 0.00382 | -0.00371 ± 0.00169 | -0.00420 ± 0.00448 | -0.00509 ± 0.00470 | -0.00731 ± 0.00515 |
| D0.1 | mlp | 0.18331 ± 0.20005 | 0.09898 ± 0.04681 | 0.13798 ± 0.08707 | 0.11214 ± 0.06299 | 0.10850 ± 0.01222 |
| D1 | linear | -0.00242 ± 0.00410 | -0.00254 ± 0.00219 | -0.00688 ± 0.00223 | -0.00586 ± 0.00703 | -0.00551 ± 0.00761 |
| D1 | mlp | 0.20678 ± 0.23477 | 0.06495 ± 0.03036 | 0.08231 ± 0.08197 | 0.08406 ± 0.01043 | 0.09088 ± 0.00524 |
| positive_validation_selected | linear | -0.00246 ± 0.00416 | -0.00281 ± 0.00236 | -0.00624 ± 0.00247 | -0.00535 ± 0.00742 | -0.00632 ± 0.00743 |
| positive_validation_selected | mlp | 0.18455 ± 0.19701 | 0.07017 ± 0.02147 | 0.11026 ± 0.07968 | 0.07804 ± 0.02064 | 0.09408 ± 0.00243 |

## Paired changes

All differences are left minus right within the same seed. Positive utility change means improvement. Negative leakage change means lower predictive success for that probe. B−A records the signed erasure utility change; A−B above expresses its loss. SD below is computed from paired differences, not from separate arm SDs.

| Comparison | Probe | Mean task R² change | Worst individual leakage change | Combined S leakage change |
|---|---|---:|---:|---:|
| B_minus_A | linear | -0.00055 ± 0.00031 | -0.99359 ± 0.00760 | -0.33484 ± 0.12071 |
| B_minus_A | mlp | -0.00024 ± 0.00025 | -0.66989 ± 0.04198 | -0.34990 ± 0.14868 |
| C_minus_B | linear | +0.00035 ± 0.00011 | -0.00032 ± 0.00144 | -0.00136 ± 0.01031 |
| C_minus_B | mlp | +0.00036 ± 0.00011 | +0.00158 ± 0.02496 | +0.00592 ± 0.01083 |
| D0.1_minus_C | linear | -0.08951 ± 0.08226 | -0.00331 ± 0.00743 | -0.00179 ± 0.00937 |
| D0.1_minus_C | mlp | -0.05511 ± 0.05011 | -0.05690 ± 0.07952 | -0.05371 ± 0.04114 |
| D1_minus_C | linear | -0.16053 ± 0.14554 | -0.00385 ± 0.00803 | +0.00001 ± 0.01083 |
| D1_minus_C | mlp | -0.08998 ± 0.07354 | -0.08264 ± 0.15468 | -0.07133 ± 0.04138 |
| positive_validation_selected_minus_C | linear | -0.14766 ± 0.15755 | -0.00334 ± 0.00811 | -0.00081 ± 0.00962 |
| positive_validation_selected_minus_C | mlp | -0.08048 ± 0.08253 | -0.10487 ± 0.11624 | -0.06813 ± 0.04566 |

## Validation feasibility

An arm is feasible only when every prohibited target meets its saved threshold for both linear and MLP attackers. This is an empirical validation criterion. Undefined observations fail. Thresholds: p1_V≤0.05, p1_S≤0.05, p2_U≤0.05, p2_S≤0.05, combined_S≤0.1.

| Arm | Linear feasible seeds | MLP feasible seeds | Both feasible seeds | Both, by seed |
|---|---:|---:|---:|---|
| A | 0/3 | 0/3 | 0/3 | 0: fail, 1: fail, 2: fail |
| B | 3/3 | 0/3 | 0/3 | 0: fail, 1: fail, 2: fail |
| C | 3/3 | 0/3 | 0/3 | 0: fail, 1: fail, 2: fail |
| D0.1 | 3/3 | 0/3 | 0/3 | 0: fail, 1: fail, 2: fail |
| D1 | 3/3 | 0/3 | 0/3 | 0: fail, 1: fail, 2: fail |
| positive_validation_selected | 3/3 | 0/3 | 0/3 | 0: fail, 1: fail, 2: fail |

Saved positive-strength selections:

- Seed 0: D_protection_0.1.
- Seed 1: D_protection_1.
- Seed 2: D_protection_1.

## Per-seed paired changes

| Seed | Comparison | Probe | Mean task R² change | Worst individual leakage change | Combined S leakage change |
|---|---|---|---:|---:|---:|
| 0 | B_minus_A | linear | -0.00070 | -0.99807 | -0.40207 |
| 0 | B_minus_A | mlp | -0.00045 | -0.63905 | -0.52028 |
| 1 | B_minus_A | linear | -0.00075 | -0.98482 | -0.40697 |
| 1 | B_minus_A | mlp | -0.00030 | -0.65294 | -0.28299 |
| 2 | B_minus_A | linear | -0.00018 | -0.99789 | -0.19548 |
| 2 | B_minus_A | mlp | +0.00004 | -0.71769 | -0.24643 |
| 0 | C_minus_B | linear | +0.00023 | +0.00126 | -0.01323 |
| 0 | C_minus_B | mlp | +0.00025 | +0.02004 | -0.00634 |
| 1 | C_minus_B | linear | +0.00039 | -0.00064 | +0.00539 |
| 1 | C_minus_B | mlp | +0.00036 | +0.01152 | +0.00993 |
| 2 | C_minus_B | linear | +0.00044 | -0.00157 | +0.00377 |
| 2 | C_minus_B | mlp | +0.00047 | -0.02681 | +0.01418 |
| 0 | D0.1_minus_C | linear | -0.04050 | -0.00205 | +0.00853 |
| 0 | D0.1_minus_C | mlp | -0.02176 | +0.02908 | -0.02605 |
| 1 | D0.1_minus_C | linear | -0.04355 | -0.01129 | -0.00414 |
| 1 | D0.1_minus_C | mlp | -0.03083 | -0.12780 | -0.10099 |
| 2 | D0.1_minus_C | linear | -0.18448 | +0.00340 | -0.00977 |
| 2 | D0.1_minus_C | mlp | -0.11273 | -0.07198 | -0.03409 |
| 0 | D1_minus_C | linear | -0.07914 | -0.00358 | +0.01098 |
| 0 | D1_minus_C | mlp | -0.05026 | +0.09577 | -0.03565 |
| 1 | D1_minus_C | linear | -0.07391 | -0.01202 | -0.00027 |
| 1 | D1_minus_C | mlp | -0.04483 | -0.17934 | -0.11669 |
| 2 | D1_minus_C | linear | -0.32856 | +0.00404 | -0.01068 |
| 2 | D1_minus_C | mlp | -0.17484 | -0.16433 | -0.06165 |
| 0 | positive_validation_selected_minus_C | linear | -0.04050 | -0.00205 | +0.00853 |
| 0 | positive_validation_selected_minus_C | mlp | -0.02176 | +0.02908 | -0.02605 |
| 1 | positive_validation_selected_minus_C | linear | -0.07391 | -0.01202 | -0.00027 |
| 1 | positive_validation_selected_minus_C | mlp | -0.04483 | -0.17934 | -0.11669 |
| 2 | positive_validation_selected_minus_C | linear | -0.32856 | +0.00404 | -0.01068 |
| 2 | positive_validation_selected_minus_C | mlp | -0.17484 | -0.16433 | -0.06165 |

This report performs no fitting or model selection. All selections come from saved validation records. Small or negative measured leakage is not a universal privacy guarantee. Per-target coverage, validation violations, input hashes, and all paired values are in paired_comparisons.json. TABLE.md contains the runner's raw per-seed arm results.
