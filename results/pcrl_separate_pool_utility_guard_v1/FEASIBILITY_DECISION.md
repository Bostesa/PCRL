# Feasibility decision

**COMPLETE_NEGATIVE_FEASIBILITY.** The privacy comparison is **NOT_TESTED**. Stage B (fresh readers, INNER assessment) did not run.

This is a trainability result, not an attacker-based privacy shortfall. No AUC was computed for any new model.

## The frozen gate, applied once to VALIDATION (inherited FIT fold 2, 5,211 rows)

| SEPARATE_GUARD criterion | Result |
| --- | --- |
| Accepted nonzero privacy-bearing update in both seeds | **Yes.** All 8/8 scanned accepted updates per seed changed the probe occupation wire. The privacy-minus-counterfactual displacement was 0.0087–0.0116 (L2), against tolerances of 3.6e-9 to 1.2e-8 |
| Released occupation wire changed | **Yes**, at every saved epoch >0, in both seeds (probe and VALIDATION) |
| All incremental VALIDATION proxies pass in both seeds at some epoch >0 | **No.** The occupation NLL cap and the summed-Brier cap fail at epochs 1, 3 and 5 in both seeds |
| Chosen common epoch >0 | **No.** The frozen rule selects epoch 0, the unchanged start |
| Independent feasibility replay | **Complete, PASS.** Same FAIL classification; no discrepancy (INDEPENDENT_VERIFICATION.json) |

SEPARATE_GUARD occupation on VALIDATION, compared with the unchanged start. Caps are accuracy loss ≤ 0.010, NLL ≤ +0.010 and Brier ≤ +0.005.

| Seed | Epoch | Accuracy loss | NLL increase | Brier increase | Pass |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 0.0025 | 0.0120 | 0.0058 | no |
| 0 | 3 | 0.0063 | 0.0150 | 0.0073 | no |
| 0 | 5 | 0.0065 | 0.0156 | 0.0077 | no |
| 1 | 1 | 0.0058 | 0.0135 | 0.0059 | no |
| 1 | 3 | 0.0084 | 0.0147 | 0.0067 | no |
| 1 | 5 | 0.0088 | 0.0159 | 0.0075 | no |

Income is bitwise identical to the start in every checkpoint, so all income proxies pass trivially.

## Did the whole-pool check generalize beyond its fitting pool?

Only partly. The absolute GUARD budget held exactly where it was enforced, on the GUARD pool itself. At epoch 5:
- GUARD Brier was +0.0050, at the budget, in both seeds.
- GUARD NLL was +0.0079 (seed 0) and +0.0084 (seed 1).
- GUARD accuracy changed by −0.0005 and −0.0078.

On the separate VALIDATION slice, the same models lost about 1.5–2 times as much:
- NLL rose +0.0156 and +0.0159.
- Brier rose +0.0077 and +0.0075.

GUARD was also inside the shared task objective, so the continuation fitted the check pool's own labels. A 2,043-row fitting pool did not certify held-out usefulness at these tolerances.

## Matched controls (descriptive; selection is the same frozen rule)

| Arm | Chosen epoch | Accepted / 160 (s0, s1) | VALIDATION occupation NLL increase at epoch 5 (s0, s1) | Brier increase at epoch 5 (s0, s1) |
| --- | --- | --- | --- | --- |
| SEPARATE_GUARD | 0 | 80, 102 | 0.0156, 0.0159 | 0.0077, 0.0075 |
| GLOBAL_UPDATE_GUARD | 0 | 70, 61 | 0.0086, 0.0106 | 0.0044, 0.0050 |
| OLD_BATCH_GUARD | 0 | 114, 119 | 0.0312, 0.0325 | 0.0165, 0.0174 |
| ORDINARY | 0 | 160, 160 | 0.0426, 0.0432 | 0.0230, 0.0226 |
| TASK_ONLY | 5 | 160, 160 | 0.0000, 0.0015 | 0.0002, 0.0009 |

Notes on the controls:
- **GLOBAL_UPDATE_GUARD** checked the larger 8,180-row UPDATE pool. Seed 0 passed every cap at epochs 1, 3 and 5. In seed 1, NLL exceeded its cap at every epoch by 0.0006–0.0009 and Brier reached the cap. So its common epoch is 0 as well. This single-run contrast is descriptive and is not a registered pool-size effect.
- The guards did reduce harm relative to the unguarded controls. ORDINARY and OLD_BATCH_GUARD degraded about 2–3 times more by epoch 5.
- TASK_ONLY applies no privacy gradient, and its "success" is the absence of a privacy update.

## What follows

- Report COMPLETE_NEGATIVE_FEASIBILITY for this fixed recipe. Training changed the occupation wire, and the privacy term really drove the accepted steps, but no learned checkpoint kept the incremental VALIDATION usefulness proxies in both seeds.
- No privacy AUC exists for these models. Any statement about recovery is NOT_TESTED, not zero and not failed.
- The frozen-income utility uncertainty from the corrected predecessor trial is unchanged: 12 income interval clauses remain unresolved there, and occupation training cannot affect them.
- No new defense epochs, strengths, budgets or pools were tried after seeing VALIDATION.
