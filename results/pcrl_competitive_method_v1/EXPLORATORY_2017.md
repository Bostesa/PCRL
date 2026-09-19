# EXPLORATORY_2017 — fixed panel transported to the already-used 2017 pools

Backing: `EXPLORATORY_2017.json` (`e68280f43b40b34b33e37e3758080b507f47562295bd34eab9e6c8d09f0da9f1`). **EXPLORATORY CROSS-YEAR DEVELOPMENT.** 2017 has been
used repeatedly; the first locked 2017 experiment keeps its historical status and nothing
here is a confirmation. No 2017 encoder or eraser was fitted: the 2018 maps were applied to
the frozen 2017 channels and only fresh attackers/probes were fitted (predecessor Mode B,
`common_fresh` scope, budget 360). 18/18 transported units fitted and scored, 0 scorer faults.
The neural nominees and their controls are bitwise J and inherit J's rows. Seed means,
increments over H; **no interval is claimed** (3 anchors).

## unweighted

| condition | A/SEX | AB/SEX | A/RAC1P | AB/RAC1P | residence gain |
|---|---|---|---|---|---|
| `A0` | +0.0325 | +0.0312 | +0.0632 | +0.0564 | +0.0289 |
| `J` | +0.0013 | +0.0069 | +0.0039 | +0.0015 | +0.0189 |
| `leace_A0` | +0.0071 | +0.0072 | +0.0145 | +0.0101 | +0.0219 |
| `splince_A0` | +0.0021 | +0.0009 | +0.0053 | +0.0049 | +0.0148 |
| `optnet16_C1` | +0.0093 | +0.0088 | +0.0211 | +0.0185 | +0.0139 |
| `E_J_C_k8` | -0.0010 | +0.0006 | -0.0009 | +0.0008 | +0.0116 |
| `E_J_C_k6` | -0.0011 | -0.0014 | -0.0003 | +0.0018 | +0.0187 |
| `E_J_L_k8` | -0.0001 | +0.0009 | -0.0012 | -0.0002 | +0.0131 |
| `E_J_LX_k8` | -0.0015 | +0.0002 | -0.0007 | -0.0002 | +0.0132 |
| `E_J_L_k6` | -0.0005 | +0.0019 | -0.0002 | +0.0002 | +0.0176 |
| `E_J_LX_k6` | -0.0010 | -0.0002 | -0.0006 | +0.0011 | +0.0172 |

## person_weighted

| condition | A/SEX | AB/SEX | A/RAC1P | AB/RAC1P | residence gain |
|---|---|---|---|---|---|
| `A0` | +0.0284 | +0.0284 | +0.0606 | +0.0527 | +0.0266 |
| `J` | -0.0014 | +0.0038 | +0.0019 | -0.0001 | +0.0171 |
| `leace_A0` | +0.0048 | +0.0065 | +0.0127 | +0.0084 | +0.0197 |
| `splince_A0` | +0.0006 | +0.0010 | +0.0027 | +0.0024 | +0.0146 |
| `optnet16_C1` | +0.0077 | +0.0080 | +0.0205 | +0.0179 | +0.0135 |
| `E_J_C_k8` | -0.0026 | -0.0011 | -0.0019 | +0.0006 | +0.0105 |
| `E_J_C_k6` | -0.0028 | -0.0031 | -0.0009 | +0.0009 | +0.0159 |
| `E_J_L_k8` | -0.0012 | -0.0003 | -0.0013 | -0.0007 | +0.0117 |
| `E_J_LX_k8` | -0.0032 | -0.0022 | -0.0009 | -0.0007 | +0.0117 |
| `E_J_L_k6` | -0.0028 | -0.0004 | -0.0006 | +0.0001 | +0.0149 |
| `E_J_LX_k6` | -0.0042 | -0.0035 | -0.0007 | +0.0006 | +0.0149 |

## Per-anchor differences from J (unweighted; negative recovery = leaks less)

| release | endpoint | anchor 0 | anchor 1 | anchor 2 |
|---|---|---|---|---|
| `E_J_C_k8` | increment/AB/SEX | -0.0041 | -0.0097 | -0.0050 |
| `E_J_C_k8` | increment/A/RAC1P | -0.0025 | -0.0024 | -0.0095 |
| `E_J_C_k8` | residence_gain_vs_H | +0.0001 | -0.0194 | -0.0026 |
| `E_J_C_k6` | increment/AB/SEX | -0.0117 | -0.0097 | -0.0035 |
| `E_J_C_k6` | increment/A/RAC1P | -0.0025 | -0.0037 | -0.0063 |
| `E_J_C_k6` | residence_gain_vs_H | +0.0060 | -0.0082 | +0.0017 |
| `E_J_L_k6` | increment/AB/SEX | -0.0054 | -0.0060 | -0.0038 |
| `E_J_L_k6` | increment/A/RAC1P | -0.0025 | -0.0023 | -0.0074 |
| `E_J_L_k6` | residence_gain_vs_H | +0.0059 | -0.0086 | -0.0013 |
| `E_J_LX_k6` | increment/AB/SEX | -0.0089 | -0.0097 | -0.0029 |
| `E_J_LX_k6` | increment/A/RAC1P | -0.0025 | -0.0023 | -0.0086 |
| `E_J_LX_k6` | residence_gain_vs_H | +0.0054 | -0.0089 | -0.0015 |

Reading: both J projections leak less than J on AB/SEX and A/RAC1P in all three anchors;
the residence difference changes sign across anchors (k6: +0.0060, −0.0082, +0.0017; k8 costs
residence in two of three). The local controls `E_J_L_k6` and `E_J_LX_k6` show the same pattern,
so 2017 gives **no coalition-specific signal**, consistent with 2018.
