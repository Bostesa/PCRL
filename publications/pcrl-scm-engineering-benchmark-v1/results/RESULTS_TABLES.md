# Results tables (exploratory development evidence; all values from frozen records)

## 1. Utility bank on MODEL_SELECT (24 held-out point clauses per arm; worst normalised clause = max value/cap)

| Arm | Cells s0 / s1 | Point-feasible (both seeds) | Worst normalised s0 / s1 | Cross-fitted pair proxy s0 / s1 | Men occ. Brier excess vs U0 s0 / s1 |
|---|---|---|---|---|---|
| C0 | 512 / 512 | True | -0.262 / 0.012 | 0.6480 / 0.6478 | 0.0006 / 0.0016 |
| SCM_b0.5 | 348 / 333 | True | -0.062 / 0.276 | 0.6257 / 0.6250 | 0.0017 / 0.0040 |
| SCM_b1.0 | 261 / 266 | True | 0.434 / 0.669 | 0.6105 / 0.6085 | 0.0044 / 0.0059 |
| SCM_b1.5 | 216 / 218 | False | 1.179 / 1.060 | 0.5931 / 0.5950 | 0.0083 / 0.0073 |
| TWIN_b0.5 | 348 / 333 | True | -0.287 / 0.012 | 0.6478 / 0.6464 | 0.0010 / 0.0020 |
| TWIN_b1.0 | 261 / 266 | True | -0.244 / 0.012 | 0.6461 / 0.6457 | 0.0018 / 0.0025 |
| TWIN_b1.5 | 216 / 218 | True | -0.190 / 0.012 | 0.6459 / 0.6454 | 0.0024 / 0.0027 |
| IBSI_b1.0_g0.5 | 261 / 266 | True | -0.213 / 0.125 | 0.6403 / 0.6368 | 0.0018 / 0.0026 |
| IBSI_b1.0_g1.0 | 261 / 266 | True | 0.122 / 0.437 | 0.6250 / 0.6196 | 0.0027 / 0.0041 |
| IBSI_b1.0_g2.0 | 261 / 266 | False | 1.411 / 1.642 | 0.5785 / 0.5779 | 0.0075 / 0.0100 |
| SHUF_b1.0_r0 | 261 / 266 | True | -0.176 / 0.087 | 0.6462 / 0.6435 | 0.0026 / 0.0027 |
| SHUF_b1.0_r1 | 261 / 266 | True | -0.150 / 0.175 | 0.6437 / 0.6472 | 0.0015 / 0.0020 |
| SHUF_b1.0_r2 | 261 / 266 | True | -0.123 / 0.045 | 0.6443 / 0.6436 | 0.0018 / 0.0034 |
| SHUF_b1.0_r3 | 261 / 266 | True | -0.087 / 0.037 | 0.6464 / 0.6457 | 0.0022 / 0.0029 |
| SHUF_b1.0_r4 | 261 / 266 | True | -0.225 / 0.012 | 0.6447 / 0.6446 | 0.0021 / 0.0029 |
| FROZEN_NEURAL_RAW (frozen) | n/a | True | 0.340 / 0.570 | 0.6236 / 0.6269 | not recorded |
| FROZEN_H_LJ (frozen) | n/a | True | 0.341 / 0.295 | 0.6224 / 0.6233 | not recorded |

Nominee SCM_b1.0; twin TWIN_b1.0 (matched capacity: True); IBSI IBSI_b1.0_g1.0 (eligible gammas ['IBSI_b1.0_g0.5', 'IBSI_b1.0_g1.0']); SHUF SHUF_b1.0_r2. Registered F1 check on the proxy (TWIN - SCM): 0.0356 / 0.0372.

## 2. Final bank on DEV_REPORT: recovery (selected reader AUC, 8-reader bank incl. LOOKUP) and point utility

| Seed | Release | Pair AUC (se) | Income AUC | Occupation AUC | Pair AUC, 7-reader slate | Pair winner | DEV point-feasible, this seed (12 clauses) |
|---|---|---|---|---|---|---|---|
| S0 | C0 | 0.6620 (0.0065) | 0.5751 | 0.6512 | 0.6620 | OUT_MLP | True |
| S0 | H_LJ | 0.6414 (0.0067) | 0.5582 | 0.5604 | 0.6414 | OUT_MLP | True |
| S0 | IBSI | 0.6317 (0.0065) | 0.5733 | 0.6271 | 0.6317 | OUT_LR | True |
| S0 | NEURAL_RAW | 0.6400 (0.0065) | 0.5462 | 0.6217 | 0.6400 | LOG_MLP | True |
| S0 | SCM | 0.6175 (0.0066) | 0.5643 | 0.6010 | 0.6175 | OUT_LR | False |
| S0 | SHUF | 0.6560 (0.0065) | 0.5827 | 0.6444 | 0.6560 | OUT_MLP | True |
| S0 | TWIN | 0.6584 (0.0065) | 0.5808 | 0.6472 | 0.6584 | LOG_MLP | True |
| S1 | C0 | 0.6626 (0.0066) | 0.5699 | 0.6507 | 0.6626 | OUT_MLP | True |
| S1 | H_LJ | 0.6428 (0.0068) | 0.5489 | 0.5833 | 0.6428 | OUT_MLP | True |
| S1 | IBSI | 0.6358 (0.0065) | 0.5707 | 0.6236 | 0.6358 | OUT_MLP | True |
| S1 | NEURAL_RAW | 0.6377 (0.0065) | 0.5509 | 0.6227 | 0.6377 | LOG_MLP | True |
| S1 | SCM | 0.6219 (0.0066) | 0.5655 | 0.6131 | 0.6219 | OUT_LR | True |
| S1 | SHUF | 0.6577 (0.0066) | 0.5875 | 0.6435 | 0.6577 | LOG_MLP | True |
| S1 | TWIN | 0.6595 (0.0066) | 0.5763 | 0.6528 | 0.6595 | OUT_MLP | True |

## 3. Primary endpoints (benefit = comparator AUC - SCM AUC; worsening = SCM - TWIN; clause one-sided alpha/3; family K = 42)

Status is each endpoint's own family-interval status (K = 42) against its margin. It is not a claim status. The nominee fails the DEV utility gate, so every RECIPE, ATTRIBUTION and COMPETITIVE endpoint is descriptive; the claim labels are in section 4. Clause bound is the one-sided alpha/3 lower bound (L1) for benefit endpoints, and the upper bound (U1) for worsening, utility and subgroup endpoints.

| Endpoint | Point | Clause bound | Family interval | Status |
|---|---|---|---|---|
| ATTRIBUTION|coalition_benefit_vs_IBSI|s0 | 0.0141 | 0.0061 | [0.0033, 0.0249] | ESTABLISHED |
| ATTRIBUTION|coalition_benefit_vs_IBSI|s1 | 0.0139 | 0.0044 | [0.0010, 0.0269] | ESTABLISHED |
| ATTRIBUTION|coalition_benefit_vs_SHUF|s0 | 0.0385 | 0.0264 | [0.0222, 0.0547] | ESTABLISHED |
| ATTRIBUTION|coalition_benefit_vs_SHUF|s1 | 0.0359 | 0.0245 | [0.0205, 0.0512] | ESTABLISHED |
| COMPETITIVE|coalition_benefit_vs_H_LJ|s0 | 0.0238 | 0.0102 | [0.0054, 0.0423] | ESTABLISHED |
| COMPETITIVE|coalition_benefit_vs_H_LJ|s1 | 0.0210 | 0.0074 | [0.0026, 0.0393] | ESTABLISHED |
| COMPETITIVE|coalition_benefit_vs_NEURAL_RAW|s0 | 0.0225 | 0.0084 | [0.0034, 0.0415] | ESTABLISHED |
| COMPETITIVE|coalition_benefit_vs_NEURAL_RAW|s1 | 0.0159 | 0.0024 | [-0.0024, 0.0341] | UNRESOLVED |
| RECIPE|coalition_benefit_vs_TWIN|s0 | 0.0409 | 0.0287 | [0.0244, 0.0573] | ESTABLISHED |
| RECIPE|coalition_benefit_vs_TWIN|s1 | 0.0377 | 0.0253 | [0.0209, 0.0544] | ESTABLISHED |
| RECIPE|income_worsening_vs_TWIN|s0 | -0.0165 | 0.0010 | [-0.0403, 0.0072] | PASS |
| RECIPE|income_worsening_vs_TWIN|s1 | -0.0108 | 0.0051 | [-0.0324, 0.0107] | UNRESOLVED |
| RECIPE|occupation_worsening_vs_TWIN|s0 | -0.0462 | -0.0345 | [-0.0620, -0.0304] | PASS |
| RECIPE|occupation_worsening_vs_TWIN|s1 | -0.0397 | -0.0278 | [-0.0559, -0.0236] | PASS |
| SUBGROUP|occupation_brier_excess_vs_U0|sex0|s0 | 0.0062 | 0.0111 | [-0.0005, 0.0129] | UNRESOLVED |
| SUBGROUP|occupation_brier_excess_vs_U0|sex0|s1 | 0.0078 | 0.0127 | [0.0012, 0.0145] | UNRESOLVED |
| SUBGROUP|occupation_brier_excess_vs_U0|sex1|s0 | -0.0015 | 0.0036 | [-0.0084, 0.0054] | UNRESOLVED |
| SUBGROUP|occupation_brier_excess_vs_U0|sex1|s1 | -0.0048 | 0.0001 | [-0.0113, 0.0018] | PASS |
| UTILITY|income|U0|accuracy_loss|s0 | 0.0017 | 0.0105 | [-0.0103, 0.0136] | UNRESOLVED |
| UTILITY|income|U0|accuracy_loss|s1 | 0.0049 | 0.0136 | [-0.0070, 0.0167] | UNRESOLVED |
| UTILITY|income|U0|brier_excess|s0 | 0.0018 | 0.0066 | [-0.0047, 0.0083] | UNRESOLVED |
| UTILITY|income|U0|brier_excess|s1 | 0.0019 | 0.0066 | [-0.0046, 0.0083] | UNRESOLVED |
| UTILITY|income|U0|nll_excess|s0 | 0.0015 | 0.0081 | [-0.0073, 0.0104] | UNRESOLVED |
| UTILITY|income|U0|nll_excess|s1 | 0.0022 | 0.0087 | [-0.0065, 0.0110] | UNRESOLVED |
| UTILITY|income|Ucal|accuracy_loss|s0 | 0.0017 | 0.0105 | [-0.0103, 0.0136] | UNRESOLVED |
| UTILITY|income|Ucal|accuracy_loss|s1 | 0.0049 | 0.0136 | [-0.0070, 0.0167] | UNRESOLVED |
| UTILITY|income|Ucal|brier_excess|s0 | 0.0034 | 0.0079 | [-0.0026, 0.0095] | UNRESOLVED |
| UTILITY|income|Ucal|brier_excess|s1 | 0.0028 | 0.0074 | [-0.0034, 0.0090] | UNRESOLVED |
| UTILITY|income|Ucal|nll_excess|s0 | 0.0048 | 0.0105 | [-0.0028, 0.0125] | UNRESOLVED |
| UTILITY|income|Ucal|nll_excess|s1 | 0.0042 | 0.0101 | [-0.0037, 0.0121] | UNRESOLVED |
| UTILITY|occupation|U0|accuracy_loss|s0 | -0.0008 | 0.0085 | [-0.0135, 0.0118] | UNRESOLVED |
| UTILITY|occupation|U0|accuracy_loss|s1 | -0.0002 | 0.0093 | [-0.0130, 0.0127] | UNRESOLVED |
| UTILITY|occupation|U0|brier_excess|s0 | 0.0024 | 0.0061 | [-0.0025, 0.0073] | UNRESOLVED |
| UTILITY|occupation|U0|brier_excess|s1 | 0.0017 | 0.0052 | [-0.0030, 0.0064] | UNRESOLVED |
| UTILITY|occupation|U0|nll_excess|s0 | 0.0089 | 0.0173 | [-0.0025, 0.0203] | UNRESOLVED |
| UTILITY|occupation|U0|nll_excess|s1 | 0.0080 | 0.0160 | [-0.0030, 0.0189] | UNRESOLVED |
| UTILITY|occupation|Ucal|accuracy_loss|s0 | -0.0008 | 0.0085 | [-0.0135, 0.0118] | UNRESOLVED |
| UTILITY|occupation|Ucal|accuracy_loss|s1 | -0.0002 | 0.0093 | [-0.0130, 0.0127] | UNRESOLVED |
| UTILITY|occupation|Ucal|brier_excess|s0 | 0.0033 | 0.0067 | [-0.0014, 0.0079] | UNRESOLVED |
| UTILITY|occupation|Ucal|brier_excess|s1 | 0.0018 | 0.0053 | [-0.0029, 0.0065] | UNRESOLVED |
| UTILITY|occupation|Ucal|nll_excess|s0 | 0.0106 | 0.0185 | [-0.0000, 0.0212] | UNRESOLVED |
| UTILITY|occupation|Ucal|nll_excess|s1 | 0.0082 | 0.0162 | [-0.0026, 0.0190] | UNRESOLVED |

Secondary endpoints (SECONDARY_ENDPOINTS.csv) are descriptive differences: use their intervals and difference status only.

## 4. Claims

- **ATTRIBUTION_VS_PUB_IBSI_AND_SHUF:** NOT_TESTED_RECIPE_NOT_ESTABLISHED
- **A_RECIPE_VS_TWIN:** RECIPE_EFFECT_NOT_ESTABLISHED
- **B_COMPETITIVE:** NOMINEE_FAILS_DEV_UTILITY_GATE
- **C_SUBGROUP:** SUBGROUP_UNRESOLVED
- **D_JOINT_COMPONENT:** NOT_CLAIMED
- **E_NOVELTY:** NOT_CLAIMED (existing-method engineering benchmark)

Separate status fields: {"confidence_status": {"counts": {"ESTABLISHED_VIOLATION": 0, "PASS": 0, "UNRESOLVED": 24}, "status": "CONFIDENCE_UNRESOLVED"}, "point_feasibility": {"DEV_REPORT": "POINT_INFEASIBLE", "DEV_REPORT_failed_keys": ["UTILITY|occupation|Ucal|nll_excess|s0"], "MODEL_SELECT": "POINT_FEASIBLE_24"}, "subgroup_status": "SUBGROUP_UNRESOLVED"}

Controls: {"CONTROLS_VALID": true, "NULL_PASS": true, "PLANTED_PASS": true}

## 5. Coverage on the attack roles (pair view; diagnostic)

| Seed | Release | Distinct pair vectors ATTACK_FIT | Median people per vector | Share of ATTACK_VALIDATION people whose pair vector occurs in ATTACK_FIT |
|---|---|---|---|---|
| S0 | C0 | 510 | 13.0 | 1.000 |
| S0 | H_LJ | 7531 | 1.0 | 0.103 |
| S0 | IBSI | 261 | 24.0 | 1.000 |
| S0 | NEURAL_RAW | 7531 | 1.0 | 0.103 |
| S0 | SCM | 260 | 18.0 | 1.000 |
| S0 | SHUF | 260 | 23.0 | 1.000 |
| S0 | TWIN | 261 | 24.0 | 1.000 |
| S1 | C0 | 508 | 13.5 | 0.999 |
| S1 | H_LJ | 7527 | 1.0 | 0.104 |
| S1 | IBSI | 265 | 23.0 | 0.999 |
| S1 | NEURAL_RAW | 7531 | 1.0 | 0.103 |
| S1 | SCM | 265 | 18.0 | 0.999 |
| S1 | SHUF | 264 | 21.5 | 0.999 |
| S1 | TWIN | 266 | 24.0 | 1.000 |
