# Stronger frozen prediction audit

Utility is independently fitted MLP task R²; direct native predictions appear separately. 
Individual leakage is the largest of the four prohibited relationships within the named family.
All attackers/checkpoints/family choices were frozen using validation before the fresh test.

| Seed | Release | U | V | Individual linear | MLP | Trees | Combined S linear | MLP | Trees | Validation feasible | Test feasible |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | oracle | 0.999994 | 0.999997 | -0.000213 | 0.000325 | -0.029974 | -0.000418 | -0.002299 | -0.051459 | True | True |
| 0 | prediction_only | 0.998005 | 0.996804 | -0.000312 | 0.000228 | -0.029192 | -0.000593 | -0.002245 | -0.065374 | True | True |
| 0 | exposed_target | 0.999892 | 0.999868 | 1.000000 | 0.999900 | 0.997599 | 1.000000 | 0.999940 | 0.997599 | False | False |
| 1 | oracle | 0.999998 | 0.999998 | -0.000884 | -0.000595 | -0.026723 | -0.002602 | -0.003354 | -0.050047 | True | True |
| 1 | prediction_only | 0.998146 | 0.997708 | -0.000944 | 0.000075 | -0.026701 | -0.002529 | -0.002852 | -0.045662 | True | True |
| 1 | exposed_target | 0.999899 | 0.999897 | 1.000000 | 0.999845 | 0.998124 | 1.000000 | 0.999948 | 0.996922 | False | False |
| 2 | oracle | 0.999993 | 0.999997 | -0.000719 | -0.001155 | -0.040075 | -0.003574 | -0.005492 | -0.075486 | True | True |
| 2 | prediction_only | 0.997953 | 0.996046 | -0.000473 | -0.000912 | -0.040771 | -0.003616 | -0.005660 | -0.079625 | True | True |
| 2 | exposed_target | 0.999917 | 0.999927 | 1.000000 | 0.999865 | 0.997983 | 1.000000 | 0.999934 | 0.996635 | False | False |
| 2 | full_E_dual_0.01 | 0.990444 | 0.992069 | -0.000817 | 0.349013 | 0.195354 | -0.001221 | 0.352950 | 0.125327 | False | False |

Oracle and exposed-target rows are diagnostics. Full E is the unchanged seed2 E_dual_0.01 checkpoint; it was protection-infeasible historically.

## Native scalar prediction utility

| Seed | Validation U | Validation V | Test U | Test V |
|---|---:|---:|---:|---:|
| 0 | 0.997703 | 0.996001 | 0.997879 | 0.996539 |
| 1 | 0.997646 | 0.997496 | 0.997939 | 0.997704 |
| 2 | 0.998080 | 0.996573 | 0.997743 | 0.995993 |

## Across-seed mean ± sample SD

| Release | U | V | Worst individual linear / MLP / trees | Combined S linear / MLP / trees |
|---|---:|---:|---|---|
| oracle | 0.999995 ± 0.000003 | 0.999997 ± 0.000001 | -0.000606 ± 0.000350 / -0.000475 ± 0.000747 / -0.032257 ± 0.006963 | -0.002198 ± 0.001616 / -0.003715 ± 0.001627 / -0.058997 ± 0.014297 |
| prediction_only | 0.998035 ± 0.000100 | 0.996853 ± 0.000832 | -0.000576 ± 0.000328 / -0.000203 ± 0.000619 / -0.032221 ± 0.007508 | -0.002246 ± 0.001531 / -0.003586 ± 0.001822 / -0.063554 ± 0.017054 |
| exposed_target | 0.999903 ± 0.000013 | 0.999897 ± 0.000030 | 1.000000 ± 0.000000 / 0.999870 ± 0.000028 / 0.997902 ± 0.000272 | 1.000000 ± 0.000000 / 0.999941 ± 0.000007 / 0.997052 ± 0.000495 |
| full_E_dual_0.01 | 0.990444 (n=1) | 0.992069 (n=1) | -0.000817 (n=1) / 0.349013 (n=1) / 0.195354 (n=1) | -0.001221 (n=1) / 0.352950 (n=1) / 0.125327 (n=1) |

Three seeds are preliminary; no significance claim. Mean/SD never replaces per-seed feasibility. Negative R² is not negative information.
Every target, family and validation-selected family test score is in PER_TARGET.md/csv and per-seed metrics. All fixed-family test results are shown; test does not select attackers.

## Compute and interface

| Seed | Release | Dimensions P1/P2/combined | Cached bytes per row | Fit + task probe seconds | Release extraction 4096 rows, seconds | Audit + task inference 4096 rows, seconds |
|---|---|---|---|---:|---:|---:|
| 0 | oracle | [1, 1, 2] | [8, 8, 16] | 5.042 | 0.000013 | 0.113855 |
| 0 | prediction_only | [1, 1, 2] | [8, 8, 16] | 4.645 | 0.003114 | 0.114932 |
| 0 | exposed_target | [3, 3, 6] | [24, 24, 48] | 4.718 | 0.000029 | 0.084857 |
| 1 | oracle | [1, 1, 2] | [8, 8, 16] | 4.629 | 0.000016 | 0.118547 |
| 1 | prediction_only | [1, 1, 2] | [8, 8, 16] | 4.598 | 0.003026 | 0.110697 |
| 1 | exposed_target | [3, 3, 6] | [24, 24, 48] | 4.744 | 0.000029 | 0.081090 |
| 2 | oracle | [1, 1, 2] | [8, 8, 16] | 5.082 | 0.000014 | 0.108322 |
| 2 | prediction_only | [1, 1, 2] | [8, 8, 16] | 4.545 | 0.002569 | 0.104575 |
| 2 | exposed_target | [3, 3, 6] | [24, 24, 48] | 4.719 | 0.000029 | 0.080259 |
| 2 | full_E_dual_0.01 | [8, 8, 16] | [64, 64, 128] | 4.737 | 0.003005 | 0.111253 |

Cache arrays are float64. Native prediction values originate in float32, so scalar releases can be serialized in4 bytes/purpose (8 combined); oracle/full-reference precision is not changed for evaluation.
Short extraction timings are local single-batch observations including Python overhead, not deployment throughput guarantees. No representation training or eraser fitting was performed.
