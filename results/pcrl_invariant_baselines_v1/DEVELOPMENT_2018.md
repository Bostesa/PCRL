# DEVELOPMENT_2018 — repaired objective and external baselines

**These are DEVELOPMENT numbers on pools that have been used repeatedly.** The
paired household-cluster bootstrap quantifies sampling variability for fixed fitted
systems. It cannot undo repeated use, and it is neither Census replicate-weight
variance nor retraining variability. Every seed and both weightings are reported.

## Reading the tables

Each cell is `estimate [adjusted simultaneous 95% interval]` for **left minus right**.

**Sign convention, stated explicitly because it differs between the two kinds of
endpoint.** `utility/same_residence` is a **log-loss** difference, so a **positive**
estimate means the left arm has a *higher* loss, i.e. **less** residence capability.
`recovery/*` is a recovery difference (the prior cancels), so a **negative** estimate
means the left arm leaks **less**. In both cases **negative is better for the left
arm**. `**-**` marks an adjusted interval entirely below zero (left significantly
better), `**+**` one entirely above (left significantly worse).

The adjustment is single-step studentized max-|t| across the five family endpoints
within each contrast and weighting, exactly the completed study's procedure.

Bootstrap: 2000 replicates, 20147 cohort-household clusters, seed 20260918.

## Mechanism gate, recorded before any score below was read

* Rotation-only share of the training gain: **max abs 1.40e-15** across 18 cells (gate `< 0.10`, registered forecast `< 0.01`). **Passes: True.**
* Direct invariance identity: the penalty moves by at most **5.55e-17** under random rotations.
* Predecessor, same diagnostic, same cells: mean **0.628**, range 0.367-0.907 — and
  that share is an **attained** value, not a supremum, because every rotation search
  stopped on budget exhaustion or line-search failure.

## Q3 — does removing the provably inert slack change MEASURED recovery?

Repaired arm minus the defective predecessor, at matched rank and policy.

| contrast | residence | A/SEX | AB/SEX | A/RAC1P | AB/RAC1P |
|---|---|---|---|---|---|
| `spectral_riv16_L1` vs `spectral_nlr16_L1` (unweighted) | -0.0007 [-0.0043, +0.0028] | -0.0021 [-0.0059, +0.0017] | -0.0005 [-0.0040, +0.0031] | -0.0000 [-0.0054, +0.0053] | +0.0013 [-0.0035, +0.0061] |
| `spectral_riv16_L1` vs `spectral_nlr16_L1` (person_weighted) | -0.0010 [-0.0051, +0.0031] | -0.0034 [-0.0078, +0.0010] | -0.0007 [-0.0046, +0.0033] | -0.0011 [-0.0071, +0.0050] | +0.0014 [-0.0041, +0.0068] |
| `spectral_riv16_L2` vs `spectral_nlr16_L2` (unweighted) | +0.0005 [-0.0019, +0.0029] | -0.0021 [-0.0053, +0.0011] | +0.0027 [-0.0009, +0.0063] | +0.0099 [+0.0039, +0.0159] **+** | +0.0045 [-0.0002, +0.0093] |
| `spectral_riv16_L2` vs `spectral_nlr16_L2` (person_weighted) | +0.0005 [-0.0023, +0.0033] | -0.0030 [-0.0069, +0.0009] | +0.0021 [-0.0024, +0.0066] | +0.0107 [+0.0034, +0.0180] **+** | +0.0042 [-0.0013, +0.0097] |
| `spectral_riv16_C1` vs `spectral_nlr16_C1` (unweighted) | +0.0021 [-0.0010, +0.0053] | +0.0017 [-0.0016, +0.0050] | -0.0023 [-0.0057, +0.0012] | +0.0019 [-0.0038, +0.0076] | +0.0078 [+0.0026, +0.0131] **+** |
| `spectral_riv16_C1` vs `spectral_nlr16_C1` (person_weighted) | +0.0010 [-0.0027, +0.0046] | +0.0017 [-0.0021, +0.0055] | -0.0010 [-0.0050, +0.0030] | +0.0057 [-0.0012, +0.0126] | +0.0066 [+0.0003, +0.0129] **+** |
| `spectral_riv8_L1` vs `spectral_nlr8_L1` (unweighted) | -0.0021 [-0.0049, +0.0008] | +0.0003 [-0.0023, +0.0029] | -0.0000 [-0.0033, +0.0032] | +0.0030 [-0.0004, +0.0063] | +0.0035 [-0.0040, +0.0109] |
| `spectral_riv8_L1` vs `spectral_nlr8_L1` (person_weighted) | -0.0018 [-0.0053, +0.0017] | +0.0010 [-0.0023, +0.0042] | +0.0008 [-0.0029, +0.0045] | +0.0037 [-0.0004, +0.0078] | +0.0043 [-0.0041, +0.0127] |
| `spectral_riv8_L2` vs `spectral_nlr8_L2` (unweighted) | -0.0022 [-0.0050, +0.0007] | +0.0025 [-0.0000, +0.0050] | -0.0001 [-0.0032, +0.0031] | +0.0023 [-0.0010, +0.0056] | -0.0015 [-0.0047, +0.0018] |
| `spectral_riv8_L2` vs `spectral_nlr8_L2` (person_weighted) | -0.0022 [-0.0055, +0.0012] | +0.0020 [-0.0009, +0.0049] | -0.0011 [-0.0049, +0.0028] | +0.0032 [-0.0006, +0.0070] | -0.0020 [-0.0057, +0.0017] |
| `spectral_riv8_C1` vs `spectral_nlr8_C1` (unweighted) | -0.0005 [-0.0033, +0.0022] | +0.0015 [-0.0009, +0.0038] | +0.0033 [+0.0003, +0.0062] **+** | +0.0079 [+0.0026, +0.0132] **+** | +0.0066 [+0.0006, +0.0126] **+** |
| `spectral_riv8_C1` vs `spectral_nlr8_C1` (person_weighted) | -0.0009 [-0.0040, +0.0022] | +0.0026 [-0.0000, +0.0053] | +0.0028 [-0.0007, +0.0063] | +0.0085 [+0.0018, +0.0153] **+** | +0.0073 [-0.0000, +0.0146] |

## The repaired arm against its own linear-moment control

| contrast | residence | A/SEX | AB/SEX | A/RAC1P | AB/RAC1P |
|---|---|---|---|---|---|
| `spectral_riv16_L1` vs `spectral_L1` (unweighted) | -0.0007 [-0.0039, +0.0025] | -0.0037 [-0.0073, +0.0000] | -0.0019 [-0.0057, +0.0019] | +0.0007 [-0.0053, +0.0067] | -0.0021 [-0.0075, +0.0034] |
| `spectral_riv16_L1` vs `spectral_L1` (person_weighted) | -0.0006 [-0.0042, +0.0031] | -0.0017 [-0.0060, +0.0026] | -0.0011 [-0.0054, +0.0033] | +0.0013 [-0.0056, +0.0082] | -0.0019 [-0.0079, +0.0042] |
| `spectral_riv16_L2` vs `spectral_L2` (unweighted) | +0.0040 [-0.0003, +0.0083] | -0.0080 [-0.0123, -0.0038] **-** | -0.0015 [-0.0056, +0.0027] | -0.0072 [-0.0147, +0.0003] | -0.0050 [-0.0146, +0.0046] |
| `spectral_riv16_L2` vs `spectral_L2` (person_weighted) | +0.0022 [-0.0029, +0.0073] | -0.0059 [-0.0107, -0.0012] **-** | +0.0001 [-0.0046, +0.0049] | -0.0024 [-0.0109, +0.0060] | +0.0004 [-0.0105, +0.0113] |
| `spectral_riv16_C1` vs `spectral_C1` (unweighted) | +0.0009 [-0.0026, +0.0044] | -0.0047 [-0.0086, -0.0007] **-** | -0.0021 [-0.0067, +0.0024] | -0.0054 [-0.0119, +0.0012] | -0.0038 [-0.0128, +0.0052] |
| `spectral_riv16_C1` vs `spectral_C1` (person_weighted) | -0.0003 [-0.0042, +0.0037] | -0.0018 [-0.0064, +0.0027] | +0.0006 [-0.0047, +0.0058] | +0.0014 [-0.0063, +0.0092] | +0.0014 [-0.0090, +0.0118] |
| `spectral_riv8_L1` vs `spectral_lin8_L1` (unweighted) | -0.0010 [-0.0037, +0.0017] | -0.0014 [-0.0049, +0.0020] | +0.0005 [-0.0030, +0.0041] | +0.0116 [+0.0048, +0.0184] **+** | +0.0069 [-0.0013, +0.0151] |
| `spectral_riv8_L1` vs `spectral_lin8_L1` (person_weighted) | -0.0004 [-0.0036, +0.0027] | +0.0005 [-0.0033, +0.0043] | +0.0024 [-0.0015, +0.0064] | +0.0105 [+0.0024, +0.0187] **+** | +0.0074 [-0.0016, +0.0164] |
| `spectral_riv8_L2` vs `spectral_lin8_L2` (unweighted) | -0.0020 [-0.0052, +0.0013] | -0.0060 [-0.0103, -0.0017] **-** | +0.0063 [-0.0054, +0.0180] | +0.0043 [-0.0000, +0.0086] | +0.0011 [-0.0030, +0.0052] |
| `spectral_riv8_L2` vs `spectral_lin8_L2` (person_weighted) | +0.0008 [-0.0028, +0.0043] | -0.0043 [-0.0089, +0.0002] | +0.0014 [-0.0035, +0.0063] | +0.0056 [+0.0005, +0.0106] **+** | +0.0013 [-0.0035, +0.0061] |
| `spectral_riv8_C1` vs `spectral_lin8_C1` (unweighted) | -0.0003 [-0.0030, +0.0024] | -0.0051 [-0.0093, -0.0008] **-** | -0.0022 [-0.0056, +0.0012] | +0.0077 [+0.0017, +0.0138] **+** | +0.0024 [-0.0014, +0.0061] |
| `spectral_riv8_C1` vs `spectral_lin8_C1` (person_weighted) | +0.0009 [-0.0021, +0.0038] | -0.0020 [-0.0067, +0.0027] | -0.0024 [-0.0061, +0.0014] | +0.0091 [+0.0020, +0.0162] **+** | +0.0025 [-0.0016, +0.0067] |

## Q5 — the repaired arm against the frozen neural channel J

| contrast | residence | A/SEX | AB/SEX | A/RAC1P | AB/RAC1P |
|---|---|---|---|---|---|
| `spectral_riv16_L1` vs `J` (unweighted) | -0.0080 [-0.0157, -0.0003] **-** | +0.0161 [+0.0079, +0.0243] **+** | +0.0119 [+0.0046, +0.0192] **+** | +0.0317 [+0.0179, +0.0455] **+** | +0.0296 [+0.0161, +0.0431] **+** |
| `spectral_riv16_L1` vs `J` (person_weighted) | -0.0071 [-0.0154, +0.0012] | +0.0175 [+0.0089, +0.0260] **+** | +0.0121 [+0.0040, +0.0201] **+** | +0.0307 [+0.0154, +0.0460] **+** | +0.0256 [+0.0098, +0.0414] **+** |
| `spectral_riv16_L2` vs `J` (unweighted) | -0.0030 [-0.0106, +0.0046] | +0.0108 [+0.0031, +0.0185] **+** | +0.0075 [+0.0006, +0.0144] **+** | +0.0185 [+0.0052, +0.0317] **+** | +0.0130 [+0.0027, +0.0232] **+** |
| `spectral_riv16_L2` vs `J` (person_weighted) | -0.0043 [-0.0126, +0.0040] | +0.0126 [+0.0044, +0.0207] **+** | +0.0082 [+0.0006, +0.0157] **+** | +0.0205 [+0.0056, +0.0354] **+** | +0.0133 [+0.0009, +0.0257] **+** |
| `spectral_riv16_C1` vs `J` (unweighted) | -0.0041 [-0.0116, +0.0035] | +0.0135 [+0.0059, +0.0212] **+** | +0.0066 [-0.0005, +0.0136] | +0.0149 [+0.0022, +0.0277] **+** | +0.0136 [+0.0038, +0.0234] **+** |
| `spectral_riv16_C1` vs `J` (person_weighted) | -0.0050 [-0.0133, +0.0034] | +0.0152 [+0.0071, +0.0234] **+** | +0.0077 [-0.0003, +0.0157] | +0.0188 [+0.0048, +0.0328] **+** | +0.0139 [+0.0022, +0.0256] **+** |
| `spectral_riv8_L1` vs `J` (unweighted) | -0.0065 [-0.0144, +0.0014] | +0.0164 [+0.0082, +0.0246] **+** | +0.0094 [+0.0022, +0.0167] **+** | +0.0253 [+0.0128, +0.0379] **+** | +0.0205 [+0.0082, +0.0328] **+** |
| `spectral_riv8_L1` vs `J` (person_weighted) | -0.0057 [-0.0147, +0.0033] | +0.0190 [+0.0104, +0.0277] **+** | +0.0114 [+0.0035, +0.0193] **+** | +0.0266 [+0.0128, +0.0405] **+** | +0.0225 [+0.0083, +0.0368] **+** |
| `spectral_riv8_L2` vs `J` (unweighted) | -0.0059 [-0.0138, +0.0020] | +0.0109 [+0.0035, +0.0183] **+** | +0.0067 [+0.0000, +0.0133] **+** | +0.0061 [-0.0042, +0.0164] | +0.0055 [-0.0033, +0.0144] |
| `spectral_riv8_L2` vs `J` (person_weighted) | -0.0054 [-0.0148, +0.0040] | +0.0129 [+0.0050, +0.0209] **+** | +0.0070 [-0.0007, +0.0148] | +0.0104 [-0.0009, +0.0217] | +0.0087 [-0.0020, +0.0195] |
| `spectral_riv8_C1` vs `J` (unweighted) | -0.0049 [-0.0127, +0.0029] | +0.0121 [+0.0046, +0.0196] **+** | +0.0058 [-0.0004, +0.0119] | +0.0063 [-0.0040, +0.0167] | +0.0049 [-0.0033, +0.0131] |
| `spectral_riv8_C1` vs `J` (person_weighted) | -0.0052 [-0.0142, +0.0038] | +0.0147 [+0.0066, +0.0228] **+** | +0.0070 [+0.0000, +0.0140] **+** | +0.0099 [-0.0016, +0.0213] | +0.0078 [-0.0022, +0.0178] |

## External adaptations against J

| contrast | residence | A/SEX | AB/SEX | A/RAC1P | AB/RAC1P |
|---|---|---|---|---|---|
| `leace_A0` vs `J` (unweighted) | -0.0005 [-0.0084, +0.0075] | +0.0055 [-0.0006, +0.0116] | -0.0020 [-0.0060, +0.0020] | +0.0019 [-0.0081, +0.0118] | -0.0022 [-0.0099, +0.0054] |
| `leace_A0` vs `J` (person_weighted) | -0.0010 [-0.0102, +0.0082] | +0.0056 [-0.0010, +0.0122] | -0.0016 [-0.0060, +0.0029] | +0.0034 [-0.0083, +0.0151] | -0.0027 [-0.0129, +0.0074] |
| `splince_A0` vs `J` (unweighted) | +0.0048 [-0.0021, +0.0117] | +0.0047 [-0.0015, +0.0110] | -0.0011 [-0.0048, +0.0026] | -0.0034 [-0.0118, +0.0050] | -0.0028 [-0.0086, +0.0030] |
| `splince_A0` vs `J` (person_weighted) | +0.0041 [-0.0037, +0.0119] | +0.0052 [-0.0016, +0.0121] | -0.0007 [-0.0052, +0.0037] | -0.0013 [-0.0106, +0.0080] | -0.0025 [-0.0093, +0.0044] |
| `optnet16_L1` vs `J` (unweighted) | +0.0017 [-0.0062, +0.0095] | +0.0104 [+0.0025, +0.0183] **+** | +0.0017 [-0.0055, +0.0090] | +0.0093 [-0.0045, +0.0232] | +0.0077 [-0.0031, +0.0185] |
| `optnet16_L1` vs `J` (person_weighted) | -0.0004 [-0.0086, +0.0079] | +0.0094 [+0.0008, +0.0179] **+** | +0.0008 [-0.0077, +0.0093] | +0.0073 [-0.0086, +0.0231] | +0.0064 [-0.0069, +0.0197] |
| `optnet16_L2` vs `J` (unweighted) | +0.0056 [-0.0018, +0.0129] | +0.0031 [-0.0039, +0.0101] | -0.0026 [-0.0089, +0.0036] | -0.0035 [-0.0139, +0.0069] | -0.0062 [-0.0132, +0.0008] |
| `optnet16_L2` vs `J` (person_weighted) | +0.0032 [-0.0047, +0.0111] | +0.0028 [-0.0050, +0.0105] | -0.0035 [-0.0113, +0.0044] | -0.0022 [-0.0137, +0.0093] | -0.0073 [-0.0150, +0.0003] |
| `optnet16_C1` vs `J` (unweighted) | +0.0045 [-0.0032, +0.0122] | +0.0061 [-0.0015, +0.0137] | -0.0044 [-0.0118, +0.0029] | -0.0012 [-0.0115, +0.0092] | -0.0009 [-0.0088, +0.0070] |
| `optnet16_C1` vs `J` (person_weighted) | +0.0026 [-0.0056, +0.0108] | +0.0064 [-0.0019, +0.0146] | -0.0041 [-0.0128, +0.0047] | +0.0005 [-0.0108, +0.0119] | -0.0012 [-0.0105, +0.0081] |

## External adaptations against the repaired arm

| contrast | residence | A/SEX | AB/SEX | A/RAC1P | AB/RAC1P |
|---|---|---|---|---|---|
| `leace_A0` vs `spectral_riv16_C1` (unweighted) | +0.0036 [-0.0028, +0.0100] | -0.0081 [-0.0148, -0.0013] **-** | -0.0086 [-0.0156, -0.0015] **-** | -0.0131 [-0.0242, -0.0019] **-** | -0.0158 [-0.0248, -0.0069] **-** |
| `leace_A0` vs `spectral_riv16_C1` (person_weighted) | +0.0040 [-0.0031, +0.0110] | -0.0097 [-0.0170, -0.0023] **-** | -0.0093 [-0.0173, -0.0013] **-** | -0.0154 [-0.0291, -0.0017] **-** | -0.0166 [-0.0273, -0.0059] **-** |
| `splince_A0` vs `spectral_riv16_C1` (unweighted) | +0.0089 [+0.0032, +0.0146] **+** | -0.0088 [-0.0157, -0.0019] **-** | -0.0077 [-0.0150, -0.0003] **-** | -0.0183 [-0.0301, -0.0066] **-** | -0.0165 [-0.0261, -0.0069] **-** |
| `splince_A0` vs `spectral_riv16_C1` (person_weighted) | +0.0091 [+0.0024, +0.0158] **+** | -0.0100 [-0.0178, -0.0022] **-** | -0.0084 [-0.0171, +0.0002] | -0.0201 [-0.0344, -0.0059] **-** | -0.0163 [-0.0273, -0.0054] **-** |
| `optnet16_L1` vs `spectral_riv16_C1` (unweighted) | +0.0057 [-0.0002, +0.0116] | -0.0032 [-0.0094, +0.0031] | -0.0048 [-0.0113, +0.0017] | -0.0056 [-0.0155, +0.0043] | -0.0060 [-0.0141, +0.0022] |
| `optnet16_L1` vs `spectral_riv16_C1` (person_weighted) | +0.0046 [-0.0025, +0.0117] | -0.0059 [-0.0131, +0.0014] | -0.0069 [-0.0147, +0.0009] | -0.0116 [-0.0238, +0.0006] | -0.0075 [-0.0173, +0.0023] |
| `optnet16_L2` vs `spectral_riv16_C1` (unweighted) | +0.0096 [+0.0033, +0.0160] **+** | -0.0104 [-0.0171, -0.0038] **-** | -0.0092 [-0.0162, -0.0022] **-** | -0.0184 [-0.0291, -0.0078] **-** | -0.0198 [-0.0293, -0.0104] **-** |
| `optnet16_L2` vs `spectral_riv16_C1` (person_weighted) | +0.0082 [+0.0008, +0.0155] **+** | -0.0125 [-0.0198, -0.0051] **-** | -0.0112 [-0.0195, -0.0028] **-** | -0.0210 [-0.0346, -0.0074] **-** | -0.0212 [-0.0321, -0.0103] **-** |
| `optnet16_C1` vs `spectral_riv16_C1` (unweighted) | +0.0086 [+0.0023, +0.0148] **+** | -0.0074 [-0.0137, -0.0011] **-** | -0.0110 [-0.0180, -0.0040] **-** | -0.0161 [-0.0268, -0.0054] **-** | -0.0145 [-0.0232, -0.0058] **-** |
| `optnet16_C1` vs `spectral_riv16_C1` (person_weighted) | +0.0076 [+0.0004, +0.0148] **+** | -0.0089 [-0.0157, -0.0021] **-** | -0.0117 [-0.0196, -0.0039] **-** | -0.0183 [-0.0310, -0.0056] **-** | -0.0151 [-0.0253, -0.0048] **-** |

## Erasure baselines against the source channel they transformed

| contrast | residence | A/SEX | AB/SEX | A/RAC1P | AB/RAC1P |
|---|---|---|---|---|---|
| `leace_A0` vs `A0` (unweighted) | +0.0098 [+0.0036, +0.0159] **+** | -0.0253 [-0.0334, -0.0172] **-** | -0.0223 [-0.0300, -0.0146] **-** | -0.0524 [-0.0664, -0.0385] **-** | -0.0402 [-0.0536, -0.0267] **-** |
| `leace_A0` vs `A0` (person_weighted) | +0.0078 [+0.0011, +0.0145] **+** | -0.0233 [-0.0324, -0.0142] **-** | -0.0203 [-0.0292, -0.0113] **-** | -0.0439 [-0.0599, -0.0278] **-** | -0.0327 [-0.0480, -0.0173] **-** |
| `splince_A0` vs `A0` (unweighted) | +0.0151 [+0.0089, +0.0213] **+** | -0.0261 [-0.0341, -0.0181] **-** | -0.0214 [-0.0291, -0.0138] **-** | -0.0577 [-0.0717, -0.0437] **-** | -0.0408 [-0.0543, -0.0273] **-** |
| `splince_A0` vs `A0` (person_weighted) | +0.0129 [+0.0059, +0.0199] **+** | -0.0236 [-0.0330, -0.0143] **-** | -0.0194 [-0.0288, -0.0101] **-** | -0.0486 [-0.0654, -0.0318] **-** | -0.0324 [-0.0480, -0.0168] **-** |

## Q4 — the registered coordination cells

The coordination rule is a one-sided **point-estimate** rule, reused for
continuity. **A pass is not statistical equivalence and not noninferiority
within .001.** Equivalence and noninferiority are reported separately per
contrast in `DEVELOPMENT_2018.json`; absence of significance is neither.

| cell | supported | advantage on all | residence within .001 on all | local race significantly worse |
|---|---|---|---|---|
| `coordination_repaired_rank16_C1_vs_local` | **False** | False | False | False |
| `coordination_repaired_rank16_C1_vs_J` | **False** | False | True | True |
| `coordination_repaired_rank8_C1_vs_local` | **False** | False | False | False |
| `coordination_repaired_rank8_C1_vs_J` | **False** | False | True | False |

## Numerical integrity

* Independent score replay: **3264 checks**, maximum absolute difference **4.44e-16**.
* Every arm's `H_A` and `H_B` coordinates were asserted **bitwise** on all seven
  released pools, at release time and again at evaluation time against the frozen
  historical anchors.

