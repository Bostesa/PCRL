# Frozen-release diagnostic analysis

The frozen protocol and INSPECTION.md describe the original discrepancy, training exposure and coordinate conventions. Tables show each target on identical examples.

## What the evidence supports

Continuing the saved weights improves fresh-test R² on 30/30 identical-target/checkpoint pairs, with gains from 0.041983 to 0.418441. B exceeds fresh C on 28/30 pairs, and exceeds independent audit D on 30/30. Fresh C exceeds D on 29/30.

Thus saved weights can recover the missed leakage with additional fitting on the fixed final release. Fresh initialization does not help substantially more; inherited weights usually help under the matched added budget. This is not a comparison of equal lifetime exposure.

The original hidden widths, prohibited-only output heads, loss and saved preprocessing are sufficient to match/exceed D after fitting. Architecture or coordinate differences are therefore not necessary to explain this miss. Their separate effects remain unmeasured; no capacity/loss factorial experiment was run. The predeclared seed2 preprocessing bridge did not trigger because no residual validation gap exceeded .05.

The final eraser does create a transfer mismatch: on seed2 E.01, P2_U validation R² falls from .192596 (map375/stats375) to .069501 (map400/stats400) to .047024 (finalmap/stats400). Merely using375stats on the finalmap gives .052381. The exact original audit .252487 is also P2_U on the same2048examples. Map changes can change released information, so these sensitivity results do not establish a unique causal decomposition.

The bounded conclusion is that an under-fitted/deployment-mismatched adversary missed recoverable leakage; it was not incapable of representing a stronger attack. Evidence cannot identify moving representations as the sole cause or separate the added fitting pool, optimizer reset and validation selection from extra updates.

One next action: add a mandatory frozen-final-release catch-up audit before future protection decisions, using this saved-coordinate replay and per-target validation selection. Validate that evaluation change on existing checkpoints before considering encoder retraining. Keep prediction-only release as the simpler fixed-task solution.

Learning curves are saved as learning_curves.png/pdf. Lines show the strongest candidate at each common count, not final-test-selected curves. B/C counts are added updates (B already had900 prior updates); D counts are original audit updates. Each method has two trajectories.

## Last-training versus final eraser (validation only)

| Seed | Release | Target | Map375/stats375 | Map400/stats400 | Final/stats375 | Final/stats400 (A) |
|---|---|---|---:|---:|---:|---:|
| 0 | E_dual_0.01 | p1_V | 0.065306 | 0.077329 | 0.044982 | 0.068258 |
| 0 | E_dual_0.01 | p1_S | 0.046802 | -0.003346 | -0.008861 | -0.009873 |
| 0 | E_dual_0.01 | p2_U | 0.182479 | 0.193568 | 0.161830 | 0.167983 |
| 0 | E_dual_0.01 | p2_S | 0.044003 | 0.028641 | 0.013432 | 0.016835 |
| 0 | E_dual_0.01 | combined_S | 0.097287 | 0.033085 | 0.032908 | 0.034476 |
| 0 | D_fixed_0.01 | p1_V | 0.163801 | 0.158849 | 0.141273 | 0.152934 |
| 0 | D_fixed_0.01 | p1_S | 0.036881 | -0.000258 | -0.001894 | 0.001767 |
| 0 | D_fixed_0.01 | p2_U | 0.199017 | 0.200721 | 0.163408 | 0.171556 |
| 0 | D_fixed_0.01 | p2_S | 0.040342 | 0.023587 | 0.010580 | 0.014259 |
| 0 | D_fixed_0.01 | combined_S | 0.094461 | 0.035618 | 0.058537 | 0.069442 |
| 1 | E_dual_0.1 | p1_V | 0.073033 | 0.081468 | 0.075543 | 0.076048 |
| 1 | E_dual_0.1 | p1_S | 0.027338 | 0.010851 | 0.010237 | 0.010381 |
| 1 | E_dual_0.1 | p2_U | 0.009738 | 0.007739 | -0.001419 | -0.002949 |
| 1 | E_dual_0.1 | p2_S | 0.020605 | 0.011810 | 0.011329 | 0.007221 |
| 1 | E_dual_0.1 | combined_S | 0.063104 | 0.004082 | 0.027590 | 0.003636 |
| 1 | D_fixed_0.1 | p1_V | 0.240703 | -0.053545 | -0.020758 | -0.031182 |
| 1 | D_fixed_0.1 | p1_S | 0.012041 | 0.011116 | 0.011157 | 0.010410 |
| 1 | D_fixed_0.1 | p2_U | 0.000834 | 0.015182 | 0.006404 | 0.005579 |
| 1 | D_fixed_0.1 | p2_S | 0.013383 | 0.014719 | 0.008046 | 0.007242 |
| 1 | D_fixed_0.1 | combined_S | 0.047484 | 0.005120 | 0.025652 | 0.014854 |
| 2 | E_dual_0.01 | p1_V | 0.256545 | 0.078848 | -0.013755 | 0.013991 |
| 2 | E_dual_0.01 | p1_S | 0.116025 | 0.053523 | 0.019625 | 0.027886 |
| 2 | E_dual_0.01 | p2_U | 0.192596 | 0.069501 | 0.052381 | 0.047024 |
| 2 | E_dual_0.01 | p2_S | 0.078359 | 0.071804 | 0.015283 | 0.014323 |
| 2 | E_dual_0.01 | combined_S | 0.171980 | 0.117951 | -0.015863 | 0.012826 |
| 2 | D_fixed_0.01 | p1_V | 0.369883 | 0.169967 | 0.084663 | 0.143285 |
| 2 | D_fixed_0.01 | p1_S | 0.133738 | 0.042648 | 0.062630 | 0.076832 |
| 2 | D_fixed_0.01 | p2_U | 0.217875 | 0.075972 | 0.045479 | 0.032352 |
| 2 | D_fixed_0.01 | p2_S | 0.109111 | 0.099765 | 0.023362 | 0.011324 |
| 2 | D_fixed_0.01 | combined_S | 0.232095 | 0.116824 | -0.058520 | -0.044676 |

All map comparisons hold the final encoder and saved adversary fixed. Map375 was fitted to an earlier encoder; these are deployment-interface sensitivities, not reconstruction of unsaved last joint-training examples. The final50 adversary updates used map375, then encoder400/refresh400 occurred with no further adversary update. Changing maps may change information as well as coordinates.

## Oracle and exposed-target controls: fresh test

| Seed | Control | Family | Target | R² |
|---|---|---|---|---:|
| 0 | A_oracle | linear | p1_V | -0.001127 |
| 0 | A_oracle | linear | p1_S | -0.001567 |
| 0 | A_oracle | linear | p2_U | -0.001534 |
| 0 | A_oracle | linear | p2_S | -0.001554 |
| 0 | A_oracle | linear | combined_S | -0.002105 |
| 0 | A_oracle | mlp | p1_V | 0.000996 |
| 0 | A_oracle | mlp | p1_S | -0.001287 |
| 0 | A_oracle | mlp | p2_U | -0.001766 |
| 0 | A_oracle | mlp | p2_S | -0.002340 |
| 0 | A_oracle | mlp | combined_S | -0.003076 |
| 0 | exposed_target | linear | p1_V | 1.000000 |
| 0 | exposed_target | linear | p1_S | 1.000000 |
| 0 | exposed_target | linear | p2_U | 1.000000 |
| 0 | exposed_target | linear | p2_S | 1.000000 |
| 0 | exposed_target | linear | combined_S | 1.000000 |
| 0 | exposed_target | mlp | p1_V | 0.999389 |
| 0 | exposed_target | mlp | p1_S | 0.998823 |
| 0 | exposed_target | mlp | p2_U | 0.999029 |
| 0 | exposed_target | mlp | p2_S | 0.999300 |
| 0 | exposed_target | mlp | combined_S | 0.999471 |
| 1 | A_oracle | linear | p1_V | -0.001743 |
| 1 | A_oracle | linear | p1_S | -0.002388 |
| 1 | A_oracle | linear | p2_U | -0.000804 |
| 1 | A_oracle | linear | p2_S | -0.002957 |
| 1 | A_oracle | linear | combined_S | -0.002961 |
| 1 | A_oracle | mlp | p1_V | -0.000513 |
| 1 | A_oracle | mlp | p1_S | -0.001661 |
| 1 | A_oracle | mlp | p2_U | -0.001770 |
| 1 | A_oracle | mlp | p2_S | -0.003220 |
| 1 | A_oracle | mlp | combined_S | -0.000550 |
| 1 | exposed_target | linear | p1_V | 1.000000 |
| 1 | exposed_target | linear | p1_S | 1.000000 |
| 1 | exposed_target | linear | p2_U | 1.000000 |
| 1 | exposed_target | linear | p2_S | 1.000000 |
| 1 | exposed_target | linear | combined_S | 1.000000 |
| 1 | exposed_target | mlp | p1_V | 0.999405 |
| 1 | exposed_target | mlp | p1_S | 0.999373 |
| 1 | exposed_target | mlp | p2_U | 0.999109 |
| 1 | exposed_target | mlp | p2_S | 0.999318 |
| 1 | exposed_target | mlp | combined_S | 0.999241 |
| 2 | A_oracle | linear | p1_V | 0.000282 |
| 2 | A_oracle | linear | p1_S | -0.003934 |
| 2 | A_oracle | linear | p2_U | 0.000194 |
| 2 | A_oracle | linear | p2_S | -0.004124 |
| 2 | A_oracle | linear | combined_S | -0.003925 |
| 2 | A_oracle | mlp | p1_V | -0.001588 |
| 2 | A_oracle | mlp | p1_S | -0.001787 |
| 2 | A_oracle | mlp | p2_U | -0.000445 |
| 2 | A_oracle | mlp | p2_S | -0.000791 |
| 2 | A_oracle | mlp | combined_S | -0.004473 |
| 2 | exposed_target | linear | p1_V | 1.000000 |
| 2 | exposed_target | linear | p1_S | 1.000000 |
| 2 | exposed_target | linear | p2_U | 1.000000 |
| 2 | exposed_target | linear | p2_S | 1.000000 |
| 2 | exposed_target | linear | combined_S | 1.000000 |
| 2 | exposed_target | mlp | p1_V | 0.999081 |
| 2 | exposed_target | mlp | p1_S | 0.999302 |
| 2 | exposed_target | mlp | p2_U | 0.999154 |
| 2 | exposed_target | mlp | p2_S | 0.999026 |
| 2 | exposed_target | mlp | combined_S | 0.999535 |

## Budget, bridge and integrity

- Seed0: 3.162s internal runtime.
  E_dual_0.01: frozen state/output checks True/True; bridge run=False.
  D_fixed_0.01: frozen state/output checks True/True; bridge run=False.
- Seed1: 2.812s internal runtime.
  E_dual_0.1: frozen state/output checks True/True; bridge run=False.
  D_fixed_0.1: frozen state/output checks True/True; bridge run=False.
- Seed2: 3.206s internal runtime.
  E_dual_0.01: frozen state/output checks True/True; bridge run=False.
  D_fixed_0.01: frozen state/output checks True/True; bridge run=False.

## Interpretation boundaries

B−A measures the result of added fitting on a frozen final release, a new attacker fitting pool, reset optimizer and validation checkpoint selection together. B retains prior900-update exposure; C does not. B/C isolate inherited weights under matched added fitting, not total lifetime training.

D differs from B/C in output target sharing (all U,V,S), preprocessing, initialization and minibatch streams. No architecture or loss sweep was run. Catch-up cannot alone prove that moving representations caused the original miss, because stopping movement and adding training co-occur.

Validation has been reused across exploratory pilots. The fresh test assesses these selected attacks, not an independent confirmation of the entire research hypothesis. Negative predictive R² is not negative information. Successful attacks improve measurement, not protection: encoders, heads and final releases never change.

The broader result remains: prediction-only release passed the fixed-task pilot and no full representation did. This diagnostic establishes neither PCRL novelty nor a need for reusable representations.
