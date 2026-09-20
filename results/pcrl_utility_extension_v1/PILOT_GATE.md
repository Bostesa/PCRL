# PILOT_GATE — `pcrl_utility_extension_v1` Tier 2

Gate decided on VALIDATION evidence only (PROTOCOL section 2). No residence, commute or test-split
number entered it. **Status: FAIL — 0 of 12 extension configurations passed.** No larger grid was
launched (PROTOCOL section 2, T2).

## Which leg failed

| leg | result |
|---|---|
| reconstruction >= 10% in >= 2 of 3 anchors | **PASS for all 14 units** (mean reduction 0.111-0.451) |
| historical source-probe allowance (income, employment, public coverage) | **PASS for all 14 units**, every anchor, both weightings |
| mean validation sensitive increment over J <= .001 on each of 4 endpoints | **FAIL for all 14 units** |
| no anchor increment > .003 | **FAIL for all 14 units** |
| H/J byte parity, inference-input restriction, probability integrity | PASS (asserted per unit; 0 quarantines) |

The capability proxy worked and the protection screen did not. The extension R reliably carries
information the frozen J channel does not (10-45% of the residual teacher variance), and it carries
sensitive information with it at every beta tried.

## The measured frontier (validation, seed means)

| unit | recon reduction (per anchor) | A/SEX | AB/SEX | A/RAC1P | AB/RAC1P |
|---|---|---|---|---|---|
| `X_r2_C1_b001` | 0.406 (0.39, 0.42, 0.41) | +0.0216 | +0.0193 | +0.0163 | +0.0161 |
| `X_r2_C1_b003` | 0.361 (0.34, 0.41, 0.33) | +0.0139 | +0.0133 | +0.0088 | +0.0059 |
| `X_r2_C1_b010` | 0.279 (0.26, 0.28, 0.29) | +0.0073 | +0.0057 | +0.0026 | +0.0041 |
| `X_r2_C1_b030` | 0.123 (0.00, 0.15, 0.22) | +0.0003 | +0.0049 | -0.0009 | +0.0001 |
| `X_r2_L1_b001` | 0.437 (0.44, 0.45, 0.42) | +0.0227 | +0.0204 | +0.0261 | +0.0225 |
| `X_r2_L1_b003` | 0.415 (0.40, 0.44, 0.41) | +0.0228 | +0.0206 | +0.0118 | +0.0114 |
| `X_r2_L1_b010` | 0.339 (0.30, 0.38, 0.34) | +0.0127 | +0.0127 | +0.0048 | +0.0052 |
| `X_r2_L1_b030` | 0.212 (0.15, 0.23, 0.26) | +0.0026 | +0.0063 | -0.0004 | +0.0022 |
| `X_r2_L2_b001` | 0.424 (0.42, 0.44, 0.41) | +0.0213 | +0.0193 | +0.0174 | +0.0178 |
| `X_r2_L2_b003` | 0.375 (0.35, 0.40, 0.38) | +0.0171 | +0.0153 | +0.0047 | +0.0053 |
| `X_r2_L2_b010` | 0.266 (0.25, 0.27, 0.28) | +0.0079 | +0.0091 | +0.0020 | +0.0026 |
| `X_r2_L2_b030` | 0.111 (0.00, 0.18, 0.15) | +0.0016 | +0.0016 | -0.0000 | +0.0007 |
| `U_r2` | 0.451 (0.44, 0.47, 0.44) | +0.0247 | +0.0225 | +0.0358 | +0.0301 |
| `P_r2` | 0.436 (0.42, 0.47, 0.41) | +0.0186 | +0.0167 | +0.0131 | +0.0159 |

Increments are over the cloud re-audit of untouched J under the identical slate, unweighted;
person-weighted values are in `PILOT_SCREEN.json` and lead to the same decision.

## Reading

* **beta buys protection and costs capability, monotonically.** Coalition arm: reconstruction
  0.406 -> 0.123 as beta goes 1 -> 30, worst sensitive increment +0.0216 -> +0.0049.
* **The closest unit is `X_r2_L2_b030`** (worst increment +0.0016, reconstruction 0.111). It still
  misses the .001 screen, on A/SEX and AB/SEX, under both weightings.
* **No coalition advantage.** At matched beta the coalition policy C1 is not better than the
  strength-matched local control L2 on the sensitive endpoints (at beta = 30, C1 worst +0.0049 vs
  L2 worst +0.0016), reproducing the predecessor study's pattern on this new mechanism.
* **The unprotected control `U_r2` is the most informative single row**: the best reconstruction
  (0.451) and the largest disclosure (+0.0358 on A/RAC1P). The adversarial term does move the
  channel; it does not move it far enough at any beta tried.
* At beta = 30 the equal-budget selection returned the zero extension on anchor 0 for two units
  (reconstruction 0.002), i.e. the rule preferred releasing untouched J there. One release
  (`X_r2_C1_b030`, seed 0) was an exact duplicate of another and was audited once.

## What this does not establish

These are point-estimate screening thresholds, not confidence statements. The pilot shows that this
finite family, at r = 2 with these budgets, did not reach the declared operating point. It does not
show that no extension of J can, and a J-relative increment near zero would not have shown that no
information remains. 2016 was not opened.
