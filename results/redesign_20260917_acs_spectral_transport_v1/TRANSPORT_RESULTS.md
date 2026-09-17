# Transport results: California ACS 2017, locked evaluation

Sealed final partition: **15,924 people in 10,701 published households** (Kish effective n under PWGTP: 10341). Lock: `TRANSPORT_LOCK.json` (31afce09394c…, 17,639 hashed inputs), committed and pushed before the first final read. Score replay: 9660 selected losses recomputed from saved predictions, maximum difference 4.4e-16. Uncertainty: paired household-cluster bootstrap, 2000 replicates over 10701 final households, with single-step max-|t| adjustment inside each predeclared family. All differences are candidate minus comparator; lower is better for residence loss and for recovery.

## 1. Headline

Residual preservation transports: every augmented interface clears the 0.01-nat residence reference in every seed and both weightings, and the spectral arms carry about 0.025–0.030 nats against 0.017–0.019 for the frozen neural J channel. The coalition penalty also behaves as designed on a new survey year: **spectral_C1 has a simultaneous-interval advantage over both spectral_L1 and the equal-total-mass control spectral_L2, under both weightings, with the residence difference inside the prespecified 0.001 band** — the fixed coordination criterion that development had failed. It comes with a small but significant income-probe cost (about 0.0013–0.0019 nats).

That is not a method win. Under the same fresh attacks, every spectral arm still leaks far more than the neural J channel: C1's additional A-race recovery is 0.045 nats against J's 0.008, and its additional AB/SEX recovery is 0.009 against 0.002. **spectral_C1 versus J fails the same criterion on transport** (local race significantly worse), exactly as in development. The mechanism knob works; the operating point it reaches is still worse than the frozen adversarial baseline on every sensitive endpoint, and better only on residence utility.

## 2. Mode B (fresh 2017 attackers and probes), scope `transport_all`, budget 360

*unweighted* (three-seed mean, household-bootstrap 95% interval)

| Interface | residence gain vs H | A/SEX | A/race | AB/SEX | AB/race |
|---|---:|---:|---:|---:|---:|
| H | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E | 0.0263 [0.0212, 0.0315] | 0.0164 [0.0121, 0.0209] | 0.0629 [0.0556, 0.0698] | 0.0158 [0.0115, 0.0204] | 0.0537 [0.0470, 0.0601] |
| A0 | 0.0289 [0.0245, 0.0333] | 0.0276 [0.0236, 0.0313] | 0.0653 [0.0575, 0.0728] | 0.0251 [0.0215, 0.0287] | 0.0536 [0.0472, 0.0596] |
| L025 | 0.0175 [0.0147, 0.0202] | 0.0010 [-0.0011, 0.0031] | 0.0093 [0.0061, 0.0124] | 0.0060 [0.0032, 0.0086] | 0.0121 [0.0085, 0.0154] |
| L20 | 0.0181 [0.0150, 0.0212] | 0.0015 [-0.0006, 0.0037] | 0.0083 [0.0055, 0.0110] | 0.0088 [0.0065, 0.0113] | 0.0168 [0.0134, 0.0201] |
| J | 0.0189 [0.0162, 0.0216] | -0.0009 [-0.0038, 0.0016] | 0.0076 [0.0042, 0.0109] | 0.0017 [-0.0004, 0.0035] | 0.0075 [0.0048, 0.0100] |
| spectral_S0 | 0.0304 [0.0246, 0.0364] | 0.0296 [0.0256, 0.0337] | 0.0740 [0.0661, 0.0813] | 0.0256 [0.0219, 0.0294] | 0.0619 [0.0548, 0.0691] |
| spectral_M025 | 0.0296 [0.0242, 0.0351] | 0.0214 [0.0176, 0.0254] | 0.0658 [0.0583, 0.0731] | 0.0192 [0.0152, 0.0234] | 0.0522 [0.0450, 0.0590] |
| spectral_M1 | 0.0298 [0.0246, 0.0349] | 0.0230 [0.0192, 0.0269] | 0.0632 [0.0556, 0.0705] | 0.0186 [0.0146, 0.0226] | 0.0506 [0.0437, 0.0575] |
| spectral_L025 | 0.0299 [0.0242, 0.0354] | 0.0221 [0.0179, 0.0265] | 0.0664 [0.0588, 0.0735] | 0.0189 [0.0151, 0.0229] | 0.0547 [0.0477, 0.0614] |
| spectral_L1 | 0.0285 [0.0230, 0.0339] | 0.0149 [0.0108, 0.0190] | 0.0590 [0.0518, 0.0658] | 0.0146 [0.0111, 0.0183] | 0.0481 [0.0412, 0.0548] |
| spectral_C025 | 0.0295 [0.0238, 0.0350] | 0.0190 [0.0149, 0.0233] | 0.0592 [0.0517, 0.0661] | 0.0159 [0.0123, 0.0195] | 0.0501 [0.0429, 0.0571] |
| spectral_C1 | 0.0282 [0.0228, 0.0337] | 0.0121 [0.0078, 0.0163] | 0.0453 [0.0383, 0.0518] | 0.0087 [0.0047, 0.0127] | 0.0401 [0.0340, 0.0463] |
| spectral_L2 | 0.0288 [0.0235, 0.0341] | 0.0146 [0.0111, 0.0184] | 0.0527 [0.0459, 0.0595] | 0.0134 [0.0099, 0.0171] | 0.0432 [0.0369, 0.0495] |

*PWGTP* (three-seed mean, household-bootstrap 95% interval)

| Interface | residence gain vs H | A/SEX | A/race | AB/SEX | AB/race |
|---|---:|---:|---:|---:|---:|
| H | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E | 0.0219 [0.0160, 0.0281] | 0.0168 [0.0118, 0.0225] | 0.0584 [0.0498, 0.0664] | 0.0160 [0.0108, 0.0216] | 0.0489 [0.0412, 0.0561] |
| A0 | 0.0266 [0.0214, 0.0318] | 0.0250 [0.0206, 0.0294] | 0.0607 [0.0517, 0.0695] | 0.0241 [0.0198, 0.0285] | 0.0503 [0.0429, 0.0572] |
| L025 | 0.0160 [0.0130, 0.0189] | -0.0005 [-0.0028, 0.0018] | 0.0084 [0.0047, 0.0116] | 0.0033 [0.0002, 0.0065] | 0.0123 [0.0084, 0.0163] |
| L20 | 0.0158 [0.0123, 0.0196] | -0.0008 [-0.0034, 0.0017] | 0.0086 [0.0055, 0.0116] | 0.0064 [0.0035, 0.0094] | 0.0135 [0.0096, 0.0171] |
| J | 0.0171 [0.0140, 0.0202] | -0.0033 [-0.0064, -0.0004] | 0.0055 [0.0014, 0.0094] | 0.0003 [-0.0022, 0.0025] | 0.0064 [0.0033, 0.0092] |
| spectral_S0 | 0.0263 [0.0192, 0.0334] | 0.0293 [0.0246, 0.0341] | 0.0690 [0.0595, 0.0778] | 0.0263 [0.0218, 0.0310] | 0.0566 [0.0482, 0.0650] |
| spectral_M025 | 0.0266 [0.0199, 0.0332] | 0.0213 [0.0167, 0.0261] | 0.0603 [0.0512, 0.0690] | 0.0187 [0.0136, 0.0239] | 0.0469 [0.0387, 0.0550] |
| spectral_M1 | 0.0267 [0.0204, 0.0331] | 0.0207 [0.0162, 0.0255] | 0.0569 [0.0475, 0.0657] | 0.0171 [0.0123, 0.0220] | 0.0459 [0.0376, 0.0537] |
| spectral_L025 | 0.0262 [0.0189, 0.0330] | 0.0224 [0.0174, 0.0275] | 0.0618 [0.0530, 0.0704] | 0.0203 [0.0158, 0.0252] | 0.0510 [0.0425, 0.0589] |
| spectral_L1 | 0.0254 [0.0185, 0.0320] | 0.0161 [0.0114, 0.0212] | 0.0556 [0.0470, 0.0638] | 0.0161 [0.0119, 0.0204] | 0.0430 [0.0352, 0.0509] |
| spectral_C025 | 0.0264 [0.0194, 0.0333] | 0.0196 [0.0144, 0.0249] | 0.0540 [0.0453, 0.0624] | 0.0178 [0.0137, 0.0221] | 0.0450 [0.0367, 0.0533] |
| spectral_C1 | 0.0253 [0.0184, 0.0319] | 0.0111 [0.0063, 0.0164] | 0.0397 [0.0317, 0.0477] | 0.0093 [0.0044, 0.0145] | 0.0354 [0.0281, 0.0426] |
| spectral_L2 | 0.0254 [0.0187, 0.0321] | 0.0151 [0.0109, 0.0195] | 0.0487 [0.0403, 0.0568] | 0.0140 [0.0099, 0.0183] | 0.0390 [0.0318, 0.0462] |

## 3. Mode A (frozen 2018 attackers and probes), scope `kernel_expanded_catchup`, budget 360

*unweighted* (three-seed mean, household-bootstrap 95% interval)

| Interface | residence gain vs H | A/SEX | A/race | AB/SEX | AB/race |
|---|---:|---:|---:|---:|---:|
| H | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E | 0.0279 [0.0233, 0.0325] | 0.0119 [0.0076, 0.0162] | 0.0578 [0.0500, 0.0649] | 0.0089 [0.0046, 0.0132] | 0.0493 [0.0425, 0.0559] |
| A0 | 0.0306 [0.0261, 0.0351] | 0.0253 [0.0216, 0.0289] | 0.0606 [0.0534, 0.0679] | 0.0224 [0.0189, 0.0259] | 0.0481 [0.0414, 0.0550] |
| L025 | 0.0189 [0.0161, 0.0217] | 0.0006 [-0.0001, 0.0013] | 0.0091 [0.0067, 0.0116] | 0.0059 [0.0040, 0.0076] | 0.0090 [0.0068, 0.0114] |
| L20 | 0.0191 [0.0161, 0.0222] | -0.0075 [-0.0119, -0.0035] | 0.0034 [0.0019, 0.0050] | 0.0075 [0.0049, 0.0101] | 0.0124 [0.0091, 0.0155] |
| J | 0.0204 [0.0175, 0.0235] | -0.0029 [-0.0050, -0.0009] | 0.0077 [0.0049, 0.0107] | 0.0012 [0.0000, 0.0023] | 0.0059 [0.0040, 0.0078] |
| spectral_S0 | 0.0328 [0.0278, 0.0378] | 0.0270 [0.0232, 0.0312] | 0.0665 [0.0591, 0.0736] | 0.0256 [0.0218, 0.0294] | 0.0516 [0.0445, 0.0583] |
| spectral_M025 | 0.0288 [0.0247, 0.0328] | 0.0192 [0.0152, 0.0232] | 0.0555 [0.0485, 0.0621] | 0.0182 [0.0149, 0.0216] | 0.0420 [0.0353, 0.0486] |
| spectral_M1 | 0.0332 [0.0282, 0.0384] | 0.0188 [0.0150, 0.0225] | 0.0514 [0.0444, 0.0580] | 0.0176 [0.0141, 0.0210] | 0.0386 [0.0321, 0.0450] |
| spectral_L025 | 0.0316 [0.0264, 0.0368] | 0.0189 [0.0154, 0.0225] | 0.0557 [0.0488, 0.0623] | 0.0179 [0.0140, 0.0217] | 0.0460 [0.0394, 0.0524] |
| spectral_L1 | 0.0291 [0.0240, 0.0341] | 0.0144 [0.0111, 0.0177] | 0.0431 [0.0366, 0.0493] | 0.0140 [0.0107, 0.0174] | 0.0368 [0.0306, 0.0432] |
| spectral_C025 | 0.0311 [0.0259, 0.0364] | 0.0171 [0.0134, 0.0209] | 0.0486 [0.0420, 0.0551] | 0.0153 [0.0119, 0.0189] | 0.0408 [0.0344, 0.0472] |
| spectral_C1 | 0.0282 [0.0237, 0.0327] | 0.0112 [0.0080, 0.0146] | 0.0311 [0.0253, 0.0366] | 0.0098 [0.0067, 0.0130] | 0.0240 [0.0185, 0.0298] |
| spectral_L2 | 0.0283 [0.0236, 0.0328] | 0.0117 [0.0085, 0.0149] | 0.0388 [0.0328, 0.0449] | 0.0109 [0.0078, 0.0141] | 0.0301 [0.0242, 0.0361] |

*PWGTP* (three-seed mean, household-bootstrap 95% interval)

| Interface | residence gain vs H | A/SEX | A/race | AB/SEX | AB/race |
|---|---:|---:|---:|---:|---:|
| H | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E | 0.0240 [0.0181, 0.0298] | 0.0118 [0.0070, 0.0169] | 0.0540 [0.0450, 0.0623] | 0.0105 [0.0054, 0.0156] | 0.0441 [0.0362, 0.0517] |
| A0 | 0.0281 [0.0226, 0.0333] | 0.0227 [0.0185, 0.0270] | 0.0584 [0.0494, 0.0667] | 0.0222 [0.0181, 0.0261] | 0.0454 [0.0378, 0.0526] |
| L025 | 0.0172 [0.0140, 0.0204] | 0.0000 [-0.0009, 0.0009] | 0.0095 [0.0065, 0.0124] | 0.0041 [0.0020, 0.0062] | 0.0088 [0.0062, 0.0114] |
| L20 | 0.0167 [0.0129, 0.0206] | -0.0065 [-0.0109, -0.0029] | 0.0028 [0.0010, 0.0046] | 0.0060 [0.0027, 0.0090] | 0.0106 [0.0071, 0.0141] |
| J | 0.0177 [0.0142, 0.0213] | -0.0049 [-0.0074, -0.0024] | 0.0095 [0.0062, 0.0131] | 0.0003 [-0.0011, 0.0017] | 0.0047 [0.0023, 0.0068] |
| spectral_S0 | 0.0296 [0.0231, 0.0359] | 0.0260 [0.0214, 0.0309] | 0.0638 [0.0550, 0.0723] | 0.0265 [0.0219, 0.0312] | 0.0488 [0.0408, 0.0567] |
| spectral_M025 | 0.0273 [0.0220, 0.0325] | 0.0187 [0.0140, 0.0238] | 0.0541 [0.0457, 0.0621] | 0.0204 [0.0162, 0.0248] | 0.0395 [0.0318, 0.0474] |
| spectral_M1 | 0.0300 [0.0235, 0.0362] | 0.0172 [0.0130, 0.0214] | 0.0500 [0.0416, 0.0583] | 0.0181 [0.0140, 0.0224] | 0.0360 [0.0283, 0.0435] |
| spectral_L025 | 0.0287 [0.0217, 0.0353] | 0.0199 [0.0157, 0.0244] | 0.0549 [0.0466, 0.0630] | 0.0205 [0.0160, 0.0252] | 0.0436 [0.0362, 0.0512] |
| spectral_L1 | 0.0264 [0.0196, 0.0327] | 0.0154 [0.0115, 0.0198] | 0.0425 [0.0347, 0.0501] | 0.0163 [0.0125, 0.0205] | 0.0340 [0.0267, 0.0414] |
| spectral_C025 | 0.0283 [0.0213, 0.0349] | 0.0184 [0.0140, 0.0230] | 0.0479 [0.0399, 0.0556] | 0.0183 [0.0142, 0.0226] | 0.0384 [0.0309, 0.0459] |
| spectral_C1 | 0.0262 [0.0203, 0.0317] | 0.0119 [0.0080, 0.0161] | 0.0297 [0.0227, 0.0364] | 0.0123 [0.0082, 0.0165] | 0.0193 [0.0125, 0.0264] |
| spectral_L2 | 0.0258 [0.0198, 0.0316] | 0.0126 [0.0089, 0.0167] | 0.0372 [0.0297, 0.0445] | 0.0130 [0.0092, 0.0170] | 0.0272 [0.0199, 0.0346] |

## 4. Primary comparison F1: spectral_C1 against both local controls (Mode B)

Decision: **supported** — advantage over both comparators under both weightings: True; residence difference within 0.001 everywhere: True; local race significantly worse anywhere: False. Critical value c = 2.93 over 20 endpoints.

| Contrast | Endpoint | Weight | Estimate | Seeds 0/1/2 | Adjusted 95% | Significant |
|---|---|---|---:|---|---|---|
| spectral_C1 − spectral_L1 | utility/same_residence | unweighted | 0.00029 | 0.0005 / 0.0004 / -0.0001 | [-0.00118, 0.00176] |  |
| spectral_C1 − spectral_L1 | utility/same_residence | PWGTP | 0.00010 | -0.0001 / -0.0000 / 0.0004 | [-0.00155, 0.00175] |  |
| spectral_C1 − spectral_L1 | recovery/A/SEX | unweighted | -0.00282 | -0.0046 / -0.0055 / 0.0017 | [-0.00585, 0.00020] |  |
| spectral_C1 − spectral_L1 | recovery/A/SEX | PWGTP | -0.00498 | -0.0050 / -0.0108 / 0.0008 | [-0.00854, -0.00142] | better |
| spectral_C1 − spectral_L1 | recovery/AB/SEX | unweighted | -0.00593 | -0.0046 / -0.0094 / -0.0038 | [-0.00915, -0.00271] | better |
| spectral_C1 − spectral_L1 | recovery/AB/SEX | PWGTP | -0.00688 | -0.0050 / -0.0143 / -0.0013 | [-0.01052, -0.00323] | better |
| spectral_C1 − spectral_L1 | recovery/A/RAC1P | unweighted | -0.01369 | -0.0145 / -0.0129 / -0.0137 | [-0.01737, -0.01002] | better |
| spectral_C1 − spectral_L1 | recovery/A/RAC1P | PWGTP | -0.01586 | -0.0135 / -0.0168 / -0.0173 | [-0.02036, -0.01135] | better |
| spectral_C1 − spectral_L1 | recovery/AB/RAC1P | unweighted | -0.00792 | -0.0054 / -0.0109 / -0.0075 | [-0.01105, -0.00478] | better |
| spectral_C1 − spectral_L1 | recovery/AB/RAC1P | PWGTP | -0.00766 | -0.0056 / -0.0081 / -0.0093 | [-0.01124, -0.00408] | better |
| spectral_C1 − spectral_L2 | utility/same_residence | unweighted | 0.00059 | 0.0012 / 0.0009 / -0.0004 | [-0.00073, 0.00191] |  |
| spectral_C1 − spectral_L2 | utility/same_residence | PWGTP | 0.00017 | 0.0006 / 0.0002 / -0.0003 | [-0.00138, 0.00172] |  |
| spectral_C1 − spectral_L2 | recovery/A/SEX | unweighted | -0.00258 | -0.0020 / -0.0051 / -0.0007 | [-0.00568, 0.00052] |  |
| spectral_C1 − spectral_L2 | recovery/A/SEX | PWGTP | -0.00398 | -0.0011 / -0.0104 / -0.0004 | [-0.00768, -0.00028] | better |
| spectral_C1 − spectral_L2 | recovery/AB/SEX | unweighted | -0.00472 | -0.0058 / -0.0060 / -0.0024 | [-0.00804, -0.00140] | better |
| spectral_C1 − spectral_L2 | recovery/AB/SEX | PWGTP | -0.00475 | -0.0056 / -0.0080 / -0.0007 | [-0.00871, -0.00079] | better |
| spectral_C1 − spectral_L2 | recovery/A/RAC1P | unweighted | -0.00741 | -0.0054 / -0.0074 / -0.0095 | [-0.01088, -0.00394] | better |
| spectral_C1 − spectral_L2 | recovery/A/RAC1P | PWGTP | -0.00900 | -0.0051 / -0.0092 / -0.0127 | [-0.01312, -0.00487] | better |
| spectral_C1 − spectral_L2 | recovery/AB/RAC1P | unweighted | -0.00302 | 0.0004 / -0.0039 / -0.0056 | [-0.00524, -0.00081] | better |
| spectral_C1 − spectral_L2 | recovery/AB/RAC1P | PWGTP | -0.00366 | -0.0004 / -0.0034 / -0.0071 | [-0.00629, -0.00104] | better |

## 5. Fixed neural replication F2 and secondary comparison F3

F2 (J against its own local controls): **supported** (c = 3.02). The replication is carried by coalition endpoints; J's A/SEX and A/race differences are not significant.

| Contrast | Endpoint | Weight | Estimate | Seeds 0/1/2 | Adjusted 95% | Significant |
|---|---|---|---:|---|---|---|
| J − L025 | utility/same_residence | unweighted | -0.00142 | 0.0022 / -0.0115 / 0.0050 | [-0.00365, 0.00081] |  |
| J − L025 | utility/same_residence | PWGTP | -0.00115 | 0.0030 / -0.0114 / 0.0050 | [-0.00392, 0.00162] |  |
| J − L025 | recovery/A/SEX | unweighted | -0.00190 | -0.0057 / 0.0000 / -0.0000 | [-0.00531, 0.00150] |  |
| J − L025 | recovery/A/SEX | PWGTP | -0.00275 | -0.0063 / 0.0000 / -0.0020 | [-0.00677, 0.00126] |  |
| J − L025 | recovery/AB/SEX | unweighted | -0.00429 | -0.0048 / -0.0000 / -0.0081 | [-0.00860, 0.00002] |  |
| J − L025 | recovery/AB/SEX | PWGTP | -0.00296 | -0.0028 / 0.0020 / -0.0081 | [-0.00791, 0.00199] |  |
| J − L025 | recovery/A/RAC1P | unweighted | -0.00174 | -0.0047 / -0.0046 / 0.0041 | [-0.00596, 0.00248] |  |
| J − L025 | recovery/A/RAC1P | PWGTP | -0.00287 | -0.0056 / -0.0072 / 0.0042 | [-0.00767, 0.00193] |  |
| J − L025 | recovery/AB/RAC1P | unweighted | -0.00462 | -0.0044 / -0.0057 / -0.0037 | [-0.00837, -0.00087] | better |
| J − L025 | recovery/AB/RAC1P | PWGTP | -0.00600 | -0.0056 / -0.0099 / -0.0024 | [-0.01045, -0.00155] | better |
| J − L20 | utility/same_residence | unweighted | -0.00075 | 0.0073 / -0.0102 / 0.0006 | [-0.00361, 0.00210] |  |
| J − L20 | utility/same_residence | PWGTP | -0.00129 | 0.0073 / -0.0114 / 0.0002 | [-0.00470, 0.00213] |  |
| J − L20 | recovery/A/SEX | unweighted | -0.00245 | -0.0044 / 0.0000 / -0.0030 | [-0.00649, 0.00160] |  |
| J − L20 | recovery/A/SEX | PWGTP | -0.00246 | -0.0064 / 0.0000 / -0.0009 | [-0.00709, 0.00216] |  |
| J − L20 | recovery/AB/SEX | unweighted | -0.00712 | -0.0042 / -0.0056 / -0.0116 | [-0.01096, -0.00328] | better |
| J − L20 | recovery/AB/SEX | PWGTP | -0.00611 | -0.0025 / -0.0027 / -0.0131 | [-0.01067, -0.00155] | better |
| J − L20 | recovery/A/RAC1P | unweighted | -0.00066 | -0.0092 / 0.0022 / 0.0050 | [-0.00473, 0.00341] |  |
| J − L20 | recovery/A/RAC1P | PWGTP | -0.00313 | -0.0132 / -0.0017 / 0.0055 | [-0.00823, 0.00196] |  |
| J − L20 | recovery/AB/RAC1P | unweighted | -0.00936 | -0.0133 / -0.0059 / -0.0090 | [-0.01336, -0.00536] | better |
| J − L20 | recovery/AB/RAC1P | PWGTP | -0.00716 | -0.0108 / -0.0064 / -0.0043 | [-0.01180, -0.00253] | better |

F3 (spectral_C1 against frozen neural J): **not supported** (c = 2.73). C1 buys about 0.009 nats of residence at 0.013–0.038 nats of extra sensitive recovery on every endpoint.

| Contrast | Endpoint | Weight | Estimate | Seeds 0/1/2 | Adjusted 95% | Significant |
|---|---|---|---:|---|---|---|
| spectral_C1 − J | utility/same_residence | unweighted | -0.00932 | -0.0141 / -0.0020 / -0.0118 | [-0.01500, -0.00363] | better |
| spectral_C1 − J | utility/same_residence | PWGTP | -0.00815 | -0.0141 / 0.0002 / -0.0105 | [-0.01515, -0.00114] | better |
| spectral_C1 − J | recovery/A/SEX | unweighted | 0.01301 | 0.0201 / 0.0034 / 0.0155 | [0.00753, 0.01848] | worse |
| spectral_C1 − J | recovery/A/SEX | PWGTP | 0.01439 | 0.0247 / 0.0010 / 0.0175 | [0.00775, 0.02104] | worse |
| spectral_C1 − J | recovery/AB/SEX | unweighted | 0.00697 | 0.0088 / 0.0014 / 0.0107 | [0.00151, 0.01242] | worse |
| spectral_C1 − J | recovery/AB/SEX | PWGTP | 0.00893 | 0.0114 / 0.0012 / 0.0142 | [0.00232, 0.01553] | worse |
| spectral_C1 − J | recovery/A/RAC1P | unweighted | 0.03767 | 0.0428 / 0.0383 / 0.0320 | [0.02941, 0.04594] | worse |
| spectral_C1 − J | recovery/A/RAC1P | PWGTP | 0.03422 | 0.0423 / 0.0337 / 0.0267 | [0.02418, 0.04426] | worse |
| spectral_C1 − J | recovery/AB/RAC1P | unweighted | 0.03268 | 0.0330 / 0.0318 / 0.0332 | [0.02504, 0.04032] | worse |
| spectral_C1 − J | recovery/AB/RAC1P | PWGTP | 0.02903 | 0.0296 / 0.0297 / 0.0278 | [0.01981, 0.03824] | worse |

## 6. Frozen operational transfer F4 (Mode A)

C1 versus its local controls: **supported**; J versus its local controls: **not supported**. Frozen 2018 attacks reproduce the Mode B ordering, which means the C1-over-L1/L2 ordering is not an artifact of refitting attackers on the new year.

| Contrast | Endpoint | Weight | Estimate | Seeds 0/1/2 | Adjusted 95% | Significant |
|---|---|---|---:|---|---|---|
| spectral_C1 − spectral_L1 | utility/same_residence | unweighted | 0.00094 | 0.0030 / 0.0007 / -0.0008 | [-0.00086, 0.00273] |  |
| spectral_C1 − spectral_L1 | utility/same_residence | PWGTP | 0.00015 | 0.0001 / 0.0013 / -0.0009 | [-0.00222, 0.00252] |  |
| spectral_C1 − spectral_L1 | recovery/A/SEX | unweighted | -0.00318 | -0.0031 / -0.0046 / -0.0019 | [-0.00515, -0.00121] | better |
| spectral_C1 − spectral_L1 | recovery/A/SEX | PWGTP | -0.00357 | -0.0026 / -0.0047 / -0.0034 | [-0.00593, -0.00121] | better |
| spectral_C1 − spectral_L1 | recovery/AB/SEX | unweighted | -0.00416 | -0.0025 / -0.0054 / -0.0045 | [-0.00624, -0.00207] | better |
| spectral_C1 − spectral_L1 | recovery/AB/SEX | PWGTP | -0.00400 | -0.0034 / -0.0052 / -0.0034 | [-0.00654, -0.00146] | better |
| spectral_C1 − spectral_L1 | recovery/A/RAC1P | unweighted | -0.01198 | -0.0101 / -0.0153 / -0.0105 | [-0.01556, -0.00840] | better |
| spectral_C1 − spectral_L1 | recovery/A/RAC1P | PWGTP | -0.01282 | -0.0121 / -0.0162 / -0.0102 | [-0.01696, -0.00867] | better |
| spectral_C1 − spectral_L1 | recovery/AB/RAC1P | unweighted | -0.01284 | -0.0054 / -0.0176 / -0.0155 | [-0.01624, -0.00944] | better |
| spectral_C1 − spectral_L1 | recovery/AB/RAC1P | PWGTP | -0.01468 | -0.0086 / -0.0176 / -0.0178 | [-0.01881, -0.01055] | better |
| spectral_C1 − spectral_L2 | utility/same_residence | unweighted | 0.00014 | -0.0007 / 0.0026 / -0.0014 | [-0.00123, 0.00151] |  |
| spectral_C1 − spectral_L2 | utility/same_residence | PWGTP | -0.00039 | -0.0005 / 0.0009 / -0.0015 | [-0.00210, 0.00132] |  |
| spectral_C1 − spectral_L2 | recovery/A/SEX | unweighted | -0.00043 | -0.0011 / -0.0010 / 0.0008 | [-0.00227, 0.00140] |  |
| spectral_C1 − spectral_L2 | recovery/A/SEX | PWGTP | -0.00076 | -0.0010 / -0.0012 / -0.0000 | [-0.00301, 0.00149] |  |
| spectral_C1 − spectral_L2 | recovery/AB/SEX | unweighted | -0.00106 | 0.0002 / -0.0003 / -0.0031 | [-0.00328, 0.00116] |  |
| spectral_C1 − spectral_L2 | recovery/AB/SEX | PWGTP | -0.00063 | -0.0007 / 0.0000 / -0.0012 | [-0.00339, 0.00212] |  |
| spectral_C1 − spectral_L2 | recovery/A/RAC1P | unweighted | -0.00772 | -0.0060 / -0.0092 / -0.0079 | [-0.01076, -0.00468] | better |
| spectral_C1 − spectral_L2 | recovery/A/RAC1P | PWGTP | -0.00752 | -0.0069 / -0.0095 / -0.0062 | [-0.01111, -0.00393] | better |
| spectral_C1 − spectral_L2 | recovery/AB/RAC1P | unweighted | -0.00613 | -0.0013 / -0.0082 / -0.0089 | [-0.00896, -0.00329] | better |
| spectral_C1 − spectral_L2 | recovery/AB/RAC1P | PWGTP | -0.00787 | -0.0053 / -0.0069 / -0.0114 | [-0.01134, -0.00440] | better |
| J − L025 | utility/same_residence | unweighted | -0.00148 | 0.0021 / -0.0066 / 0.0001 | [-0.00387, 0.00092] |  |
| J − L025 | utility/same_residence | PWGTP | -0.00053 | 0.0037 / -0.0058 / 0.0006 | [-0.00361, 0.00255] |  |
| J − L025 | recovery/A/SEX | unweighted | -0.00351 | -0.0075 / 0.0000 / -0.0030 | [-0.00662, -0.00039] | better |
| J − L025 | recovery/A/SEX | PWGTP | -0.00491 | -0.0094 / 0.0000 / -0.0053 | [-0.00859, -0.00123] | better |
| J − L025 | recovery/AB/SEX | unweighted | -0.00470 | -0.0109 / 0.0000 / -0.0032 | [-0.00724, -0.00217] | better |
| J − L025 | recovery/AB/SEX | PWGTP | -0.00379 | -0.0075 / 0.0000 / -0.0038 | [-0.00682, -0.00076] | better |
| J − L025 | recovery/A/RAC1P | unweighted | -0.00140 | -0.0044 / -0.0016 / 0.0019 | [-0.00496, 0.00217] |  |
| J − L025 | recovery/A/RAC1P | PWGTP | -0.00005 | -0.0020 / -0.0012 / 0.0031 | [-0.00411, 0.00401] |  |
| J − L025 | recovery/AB/RAC1P | unweighted | -0.00306 | 0.0000 / -0.0036 / -0.0056 | [-0.00632, 0.00021] |  |
| J − L025 | recovery/AB/RAC1P | PWGTP | -0.00410 | 0.0000 / -0.0072 / -0.0051 | [-0.00792, -0.00029] | better |
| J − L20 | utility/same_residence | unweighted | -0.00127 | 0.0042 / -0.0092 / 0.0012 | [-0.00401, 0.00148] |  |
| J − L20 | utility/same_residence | PWGTP | -0.00104 | 0.0041 / -0.0090 / 0.0018 | [-0.00450, 0.00241] |  |
| J − L20 | recovery/A/SEX | unweighted | 0.00457 | -0.0020 / 0.0185 / -0.0027 | [-0.00233, 0.01147] |  |
| J − L20 | recovery/A/SEX | PWGTP | 0.00160 | -0.0028 / 0.0141 / -0.0065 | [-0.00525, 0.00846] |  |
| J − L20 | recovery/AB/SEX | unweighted | -0.00639 | -0.0022 / -0.0058 / -0.0112 | [-0.01025, -0.00253] | better |
| J − L20 | recovery/AB/SEX | PWGTP | -0.00562 | -0.0005 / -0.0052 / -0.0112 | [-0.01034, -0.00091] | better |
| J − L20 | recovery/A/RAC1P | unweighted | 0.00427 | 0.0026 / 0.0077 / 0.0025 | [0.00045, 0.00810] | worse |
| J − L20 | recovery/A/RAC1P | PWGTP | 0.00666 | 0.0080 / 0.0071 / 0.0048 | [0.00204, 0.01127] | worse |
| J − L20 | recovery/AB/RAC1P | unweighted | -0.00646 | -0.0111 / -0.0020 / -0.0063 | [-0.01082, -0.00210] | better |
| J − L20 | recovery/AB/RAC1P | PWGTP | -0.00600 | -0.0100 / -0.0025 / -0.0055 | [-0.01102, -0.00097] | better |

## 7. What the advantage costs

Endpoints where a contrast's unadjusted 95% interval excludes zero on the *worse* side (Mode B, `transport_all`, 360):

| Contrast | Weight | Endpoint | Estimate | Unadjusted 95% |
|---|---|---|---:|---|
| spectral_C1 − J | unweighted | utility/income_binary | 0.00811 | [0.00586, 0.01047] |
| spectral_C1 − J | unweighted | utility/civilian_at_work | 0.01353 | [0.01102, 0.01605] |
| spectral_C1 − J | unweighted | recovery/A/SEX | 0.01301 | [0.00913, 0.01704] |
| spectral_C1 − J | unweighted | recovery/A/RAC1P | 0.03767 | [0.03179, 0.04360] |
| spectral_C1 − J | unweighted | recovery/AB/SEX | 0.00697 | [0.00308, 0.01089] |
| spectral_C1 − J | unweighted | recovery/AB/RAC1P | 0.03268 | [0.02733, 0.03810] |
| spectral_C1 − J | PWGTP | utility/income_binary | 0.00724 | [0.00431, 0.01027] |
| spectral_C1 − J | PWGTP | utility/civilian_at_work | 0.01144 | [0.00865, 0.01430] |
| spectral_C1 − J | PWGTP | recovery/A/public_coverage | 0.00529 | [0.00142, 0.00907] |
| spectral_C1 − J | PWGTP | recovery/A/SEX | 0.01439 | [0.00981, 0.01944] |
| spectral_C1 − J | PWGTP | recovery/A/RAC1P | 0.03422 | [0.02714, 0.04165] |
| spectral_C1 − J | PWGTP | recovery/AB/SEX | 0.00893 | [0.00441, 0.01389] |
| spectral_C1 − J | PWGTP | recovery/AB/RAC1P | 0.02903 | [0.02261, 0.03586] |
| J − L025 | unweighted | utility/civilian_at_work | 0.00149 | [0.00073, 0.00220] |
| J − L025 | PWGTP | utility/civilian_at_work | 0.00119 | [0.00034, 0.00211] |
| J − L20 | unweighted | recovery/A/public_coverage | 0.00315 | [0.00129, 0.00502] |
| spectral_C1 − spectral_L1 | unweighted | utility/income_binary | 0.00192 | [0.00102, 0.00293] |
| spectral_C1 − spectral_L1 | PWGTP | utility/income_binary | 0.00156 | [0.00054, 0.00262] |
| spectral_C1 − spectral_L2 | unweighted | utility/income_binary | 0.00125 | [0.00043, 0.00204] |
| spectral_C1 − spectral_L2 | PWGTP | utility/income_binary | 0.00140 | [0.00054, 0.00237] |

C1's coalition advantage is paid for in income-probe utility against both local controls; against J it is paid for in every sensitive endpoint and in both source probes. No cross-target exchange rate is applied: these are separate components of one vector.

## 8. Criteria: residence capability, half-headroom, source allowance

| Interface | Mode | Weight | mean residence gain | min seed | .01 in all seeds | half-headroom seeds | source allowance seeds |
|---|---|---|---:|---:|---|---:|---:|
| H | B | unweighted | 0.0000 | 0.0000 | no | 0/3 | 3/3 |
| H | B | PWGTP | 0.0000 | 0.0000 | no | 0/3 | 3/3 |
| E | B | unweighted | 0.0263 | 0.0209 | yes | 0/3 | 3/3 |
| E | B | PWGTP | 0.0219 | 0.0171 | yes | 0/3 | 3/3 |
| A0 | B | unweighted | 0.0289 | 0.0223 | yes | 1/3 | 3/3 |
| A0 | B | PWGTP | 0.0266 | 0.0200 | yes | 2/3 | 3/3 |
| L025 | B | unweighted | 0.0175 | 0.0143 | yes | 0/3 | 3/3 |
| L025 | B | PWGTP | 0.0160 | 0.0124 | yes | 0/3 | 3/3 |
| L20 | B | unweighted | 0.0181 | 0.0156 | yes | 0/3 | 3/3 |
| L20 | B | PWGTP | 0.0158 | 0.0125 | yes | 0/3 | 3/3 |
| J | B | unweighted | 0.0189 | 0.0134 | yes | 0/3 | 3/3 |
| J | B | PWGTP | 0.0171 | 0.0111 | yes | 1/3 | 3/3 |
| spectral_S0 | B | unweighted | 0.0304 | 0.0270 | yes | 1/3 | 3/3 |
| spectral_S0 | B | PWGTP | 0.0263 | 0.0215 | yes | 1/3 | 3/3 |
| spectral_M025 | B | unweighted | 0.0296 | 0.0278 | yes | 1/3 | 3/3 |
| spectral_M025 | B | PWGTP | 0.0266 | 0.0240 | yes | 1/3 | 3/3 |
| spectral_M1 | B | unweighted | 0.0298 | 0.0281 | yes | 1/3 | 3/3 |
| spectral_M1 | B | PWGTP | 0.0267 | 0.0236 | yes | 2/3 | 3/3 |
| spectral_L025 | B | unweighted | 0.0299 | 0.0279 | yes | 1/3 | 3/3 |
| spectral_L025 | B | PWGTP | 0.0262 | 0.0232 | yes | 1/3 | 3/3 |
| spectral_L1 | B | unweighted | 0.0285 | 0.0281 | yes | 1/3 | 3/3 |
| spectral_L1 | B | PWGTP | 0.0254 | 0.0236 | yes | 1/3 | 3/3 |
| spectral_C025 | B | unweighted | 0.0295 | 0.0282 | yes | 1/3 | 3/3 |
| spectral_C025 | B | PWGTP | 0.0264 | 0.0243 | yes | 2/3 | 3/3 |
| spectral_C1 | B | unweighted | 0.0282 | 0.0275 | yes | 1/3 | 3/3 |
| spectral_C1 | B | PWGTP | 0.0253 | 0.0236 | yes | 1/3 | 3/3 |
| spectral_L2 | B | unweighted | 0.0288 | 0.0287 | yes | 1/3 | 3/3 |
| spectral_L2 | B | PWGTP | 0.0254 | 0.0239 | yes | 1/3 | 3/3 |
| H | A | unweighted | 0.0000 | 0.0000 | no | 0/3 | 3/3 |
| H | A | PWGTP | 0.0000 | 0.0000 | no | 0/3 | 3/3 |
| E | A | unweighted | 0.0279 | 0.0250 | yes | 0/3 | 3/3 |
| E | A | PWGTP | 0.0240 | 0.0208 | yes | 0/3 | 3/3 |
| A0 | A | unweighted | 0.0306 | 0.0230 | yes | 0/3 | 3/3 |
| A0 | A | PWGTP | 0.0281 | 0.0192 | yes | 1/3 | 3/3 |
| L025 | A | unweighted | 0.0189 | 0.0187 | yes | 0/3 | 3/3 |
| L025 | A | PWGTP | 0.0172 | 0.0166 | yes | 0/3 | 3/3 |
| L20 | A | unweighted | 0.0191 | 0.0166 | yes | 0/3 | 3/3 |
| L20 | A | PWGTP | 0.0167 | 0.0137 | yes | 0/3 | 3/3 |
| J | A | unweighted | 0.0204 | 0.0168 | yes | 0/3 | 3/3 |
| J | A | PWGTP | 0.0177 | 0.0143 | yes | 0/3 | 3/3 |
| spectral_S0 | A | unweighted | 0.0328 | 0.0290 | yes | 2/3 | 0/3 |
| spectral_S0 | A | PWGTP | 0.0296 | 0.0246 | yes | 2/3 | 0/3 |
| spectral_M025 | A | unweighted | 0.0288 | 0.0242 | yes | 1/3 | 1/3 |
| spectral_M025 | A | PWGTP | 0.0273 | 0.0211 | yes | 1/3 | 1/3 |
| spectral_M1 | A | unweighted | 0.0332 | 0.0289 | yes | 1/3 | 0/3 |
| spectral_M1 | A | PWGTP | 0.0300 | 0.0238 | yes | 1/3 | 0/3 |
| spectral_L025 | A | unweighted | 0.0316 | 0.0288 | yes | 1/3 | 0/3 |
| spectral_L025 | A | PWGTP | 0.0287 | 0.0246 | yes | 2/3 | 0/3 |
| spectral_L1 | A | unweighted | 0.0291 | 0.0267 | yes | 0/3 | 0/3 |
| spectral_L1 | A | PWGTP | 0.0264 | 0.0231 | yes | 0/3 | 0/3 |
| spectral_C025 | A | unweighted | 0.0311 | 0.0285 | yes | 0/3 | 1/3 |
| spectral_C025 | A | PWGTP | 0.0283 | 0.0241 | yes | 1/3 | 1/3 |
| spectral_C1 | A | unweighted | 0.0282 | 0.0242 | yes | 0/3 | 0/3 |
| spectral_C1 | A | PWGTP | 0.0262 | 0.0218 | yes | 0/3 | 0/3 |
| spectral_L2 | A | unweighted | 0.0283 | 0.0235 | yes | 0/3 | 0/3 |
| spectral_L2 | A | PWGTP | 0.0258 | 0.0227 | yes | 0/3 | 0/3 |

Source-output parity is structural and verified: on all 210 released 2017 views (3 seeds × 5 partitions × 14 interfaces), the A wire's first four coordinates equal the income/employment service vectors bitwise, the B wire equals the coverage vector, and AB is their concatenation (`INDEPENDENT_VERIFICATION.json`). The **legacy source-probe allowance** separates the modes: under fresh adaptation (Mode B) every interface passes in all three seeds, while under frozen transfer (Mode A) every spectral arm fails in 0–1 of 3 seeds. Development saw the Mode A pattern. The frozen 2018 probes do not carry to 2017 on spectral wires; refitting the probe on the new year repairs it. The half-headroom criterion, which compares against the PCA32 and richer banks rather than H, still fails for almost every interface in both modes.

## 9. Service quality after shift

| Service task | 2018 log loss | 2017 log loss | 2018 PWGTP | 2017 PWGTP | 2018 accuracy | 2017 accuracy |
|---|---:|---:|---:|---:|---:|---:|
| civilian_at_work | 0.2884 | 0.3022 | 0.2773 | 0.3027 | 0.8997 | 0.8931 |
| income_binary | 0.2986 | 0.2840 | 0.3106 | 0.2807 | 0.8670 | 0.8732 |
| public_coverage | 0.5046 | 0.4979 | 0.5141 | 0.5067 | 0.7611 | 0.7665 |

The exact released probability vectors are unchanged, but their accuracy moves with the year: income log loss improves by about 0.015 nats, employment degrades by about 0.014. Income is nominal and unadjusted across years (ADJINC differs), which limits interpretation of the income column.

## 10. Attack scope, and attacks that generalize worse than H

| Interface | Endpoint | Mode B common_fresh | Mode B transport_all | Mode A kernel_expanded_catchup |
|---|---|---:|---:|---:|
| J | AB/SEX | 0.0069 | 0.0017 | 0.0012 |
| J | A/RAC1P | 0.0039 | 0.0076 | 0.0077 |
| J | AB/RAC1P | 0.0015 | 0.0075 | 0.0059 |
| spectral_C1 | AB/SEX | 0.0148 | 0.0087 | 0.0098 |
| spectral_C1 | A/RAC1P | 0.0471 | 0.0453 | 0.0311 |
| spectral_C1 | AB/RAC1P | 0.0436 | 0.0401 | 0.0240 |
| spectral_L1 | AB/SEX | 0.0207 | 0.0146 | 0.0140 |
| spectral_L1 | A/RAC1P | 0.0608 | 0.0590 | 0.0431 |
| spectral_L1 | AB/RAC1P | 0.0515 | 0.0481 | 0.0368 |
| spectral_L2 | AB/SEX | 0.0195 | 0.0134 | 0.0109 |
| spectral_L2 | A/RAC1P | 0.0545 | 0.0527 | 0.0388 |
| spectral_L2 | AB/RAC1P | 0.0466 | 0.0432 | 0.0301 |
| spectral_S0 | AB/SEX | 0.0317 | 0.0256 | 0.0256 |
| spectral_S0 | A/RAC1P | 0.0758 | 0.0740 | 0.0665 |
| spectral_S0 | AB/RAC1P | 0.0654 | 0.0619 | 0.0516 |

Adding frozen 2018 attackers and catch-up trajectories (`transport_all`) lowers *additional* recovery relative to the fresh-only scope because it strengthens the H baseline that is subtracted. Selected attacks sometimes generalize worse than H's own attack routed onto the augmented wire: 8 of 312 sensitive cells in `transport_all` and 15 of 312 in `common_fresh`. Negative selected increments remain unclipped: in Mode B `transport_all` at budget 360, 22 of 429 unweighted role-cells are negative. J's own A/SEX increment is negative under PWGTP (−0.0033, interval excluding zero), so part of J's apparent protection is a property of validation-selected attacks, not a demonstrated guarantee.

## 11. Randomized withholding controls

Expected losses are computed from aligned per-person loss arrays for p in {0, .25, .5, .75, 1}, with a visible independent Bernoulli branch fixed per person; the per-person mixture equals the linear combination to 2.2e-16, and one realized routing per source and p is reported beside it. **No withholding mechanism dominates spectral_C1 across all seeds and both weightings** (0 such mechanisms). In Mode B `transport_all`, withholding dominates a given spectral arm in 1–3 of 180 fixed comparisons and a spectral arm dominates withholding in 0–3 of 180. The frontier therefore does not reduce to withholding a simpler channel, and neither family is robustly better on the full vector.

## 12. Development versus transport

| Family | Status | Endpoints |
|---|---|---:|
| F1_primary | consistent | 20 |
| F2_neural_replication | consistent | 17 |
| F2_neural_replication | unresolved | 3 |
| F3_secondary | consistent | 10 |
| F4_frozen_transfer | consistent | 36 |
| F4_frozen_transfer | contradicted | 1 |
| F4_frozen_transfer | unresolved | 3 |

Every F1 endpoint keeps its development sign (20/20). What changed is resolution, not direction: on 2018 the C1-versus-L2 differences were near zero (AB/SEX −0.0003, AB/race −0.0006) and failed the criterion; on the 2017 final partition, with 15,924 people instead of 2,982, the same-signed differences are larger (−0.0047, −0.0030) and their simultaneous intervals exclude zero. One F4 endpoint is contradicted and three are unresolved.

## 13. Support limitations

RAC1P class 3 (Alaska Native alone) has 1 person in the final partition and is absent from 2018 attacker fitting, so class-specific recall and AUROC are undefined for it and it is flagged insufficient for class-level claims. Full nine-class log loss is always scored with the 1e-12 floor. The comparable-category diagnostic, restricted to classes supported in every fitting pool of both modes, reproduces the F1 race differences almost exactly (for example C1 − L1 A/race −0.0137 unweighted against −0.0137 on all rows), so the conclusions are not an artifact of unsupported classes.

## 14. Verification

`INDEPENDENT_VERIFICATION.json`: 15 checks, all passing — household disjointness and person-key uniqueness across all five partitions, lock integrity (with two code-only amendments listed), final access strictly after the lock, source-output identity, B-only identity across interfaces, projection and coalition-inheritance rules, Mode B selection identity recomputed from stored validation scores, independent recomputation of every selected score under both weightings, incremental baselines, paired differences, withholding arithmetic, an independently coded bootstrap replay of one primary endpoint (matching estimate and standard error), adjusted-interval arithmetic, and service quality. In addition, all 59,874 stored final predictions were reproduced exactly by an independent cross-process replay (`PREDICTION_REPLAY.json`), and 37,122 validation predictions were recomputed with **0 selection changes** (`VALIDATION_SCORE_AUDIT.json`).

Two lock amendments are recorded, both code-only: a release-file write race, and rare load-dependent numerical corruption observed during the first scoring pass under extreme machine memory pressure. Every final unit was rescored afterwards with confirmed (twice-computed) predictions; the original outputs are preserved under `superseded_nondeterminism_20260917/`.
