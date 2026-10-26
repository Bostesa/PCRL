# RFA1 tables (aggregate development records; exploratory; reused MODEL_SELECT people)

## Reading RFA_v1 (registered; card RF_RFIT_v1.md)

Outcome class **BOUNDED_NEGATIVE_EXPERIMENT**. D1 TASK_FEASIBLE; D2 PARTIAL_START_ONLY; D3 NOT_SUPERIOR; D4 NOT_MET; LC LEARNABILITY_LIMITATION; D2R ROBUSTNESS_NOT_CONFIRMED; D5 NOT_APPLICABLE_NOT_JOINT; D6 NOT_CLAIMED.

| D2 comparison | at | pair benefit mean | se_agg | lower | upper |
|---|---|---|---|---|---|
| START | NEURAL_RAW | +0.0127 | 0.0059 | +0.0008 | +0.0246 |
| CONT | RF_TASK | +0.0086 | 0.0092 | -0.0097 | +0.0270 |
| STOPFIT | RF_STOP_HI | +0.0028 | 0.0045 | -0.0061 | +0.0117 |
| ALTCRIT | RF_ALTC_HI | +0.0005 | 0.0010 | -0.0015 | +0.0025 |
| RAWJ | RF_RAWJ_HI | -0.0016 | 0.0084 | -0.0185 | +0.0153 |

| reference (D3) | pair benefit mean | se_agg | recipient gates | screen |
|---|---|---|---|---|
| B2TWIN | +0.0109 | 0.0125 | False | AGG_POSITIVE_POINT_ONLY |
| H_LJ | +0.0168 | 0.0115 | False | AGG_POSITIVE_POINT_ONLY |
| NEURAL_RAW | +0.0127 | 0.0059 | True | AGG_POSITIVE |
| RETIRED_B3CTL | -0.0038 | 0.0138 | False | AGG_NOT_POSITIVE |
| SUCC | +0.0037 | 0.0120 | False | AGG_POSITIVE_POINT_ONLY |
| TR100 | +0.0343 | 0.0098 | True | AGG_POSITIVE |
| TT100 | +0.0346 | 0.0088 | True | AGG_POSITIVE |
| TWJ | +0.0266 | 0.0113 | False | AGG_POSITIVE |
| W | +0.0351 | 0.0095 | True | AGG_POSITIVE |

| member | pair AUC v0 / v1 | occupation view | income view |
|---|---|---|---|
| RETIRED_B3CTL | [0.6061, 0.5986] | [0.5665, 0.5682] | [0.5281, 0.5329] |
| RF_RAWJ_HI | [0.6068, 0.6023] | [0.6039, 0.5977] | [0.5306, 0.5059] |
| SUCC | [0.6076, 0.6121] | [0.5665, 0.5682] | [0.5373, 0.5381] |
| RF_FULL | [0.6099, 0.6024] | [0.5971, 0.5969] | [0.5358, 0.5319] |
| RF_ALTC_HI | [0.6104, 0.6031] | [0.6026, 0.5986] | [0.5358, 0.5319] |
| RF_STOP_LO | [0.6122, 0.6039] | [0.6012, 0.6021] | [0.5358, 0.5319] |
| RF_STOP_HI | [0.6132, 0.6048] | [0.5986, 0.5989] | [0.5358, 0.5319] |
| RF_TASK | [0.6154, 0.6142] | [0.6058, 0.6138] | [0.5358, 0.5319] |
| B2TWIN | [0.6183, 0.6158] | [0.5665, 0.5682] | [0.5403, 0.5401] |
| NEURAL_RAW | [0.62, 0.6178] | [0.6089, 0.6132] | [0.536, 0.5319] |
| H_LJ | [0.6259, 0.62] | [0.5665, 0.5682] | [0.5447, 0.5425] |
| Ucal | [0.631, 0.6363] | [0.6187, 0.6272] | [0.5506, 0.5483] |
| TWJ | [0.6321, 0.6335] | [0.5665, 0.5682] | [0.5536, 0.5628] |
| U0 | [0.6336, 0.6325] | [0.6188, 0.6277] | [0.551, 0.5483] |
| TR100 | [0.6387, 0.6422] | [0.6329, 0.6372] | [0.5373, 0.5381] |
| TT100 | [0.6408, 0.6408] | [0.6375, 0.6312] | [0.5373, 0.5381] |
| W | [0.6415, 0.6412] | [0.6311, 0.6354] | [0.5373, 0.5381] |

| member | G1 at the point v0 / v1 | max value / cap v0 / v1 | men's guard v0 / v1 |
|---|---|---|---|
| RF_ALTC_HI | True / True | 0.452 / 0.784 | 0.00226 / 0.00375 |
| RF_FULL | True / True | 0.502 / 0.885 | 0.00251 / 0.00442 |
| RF_RAWJ_HI | True / False | 0.543 / 1.059 | 0.00271 / 0.00019 |
| RF_STOP_HI | True / True | 0.482 / 0.824 | 0.00241 / 0.00412 |
| RF_STOP_LO | True / True | 0.406 / 0.767 | 0.00203 / 0.00384 |
| RF_TASK | True / True | 0.36 / 0.694 | 0.0018 / 0.00347 |
| NEURAL_RAW | True / True | 0.34 / 0.57 | 0.00147 / 0.00198 |

| reference | G1 at the point v0 / v1 | max value / cap v0 / v1 | men's guard v0 / v1 |
|---|---|---|---|
| B2TWIN | True / True | 0.883 / 0.788 | 0.00442 / 0.00394 |
| H_LJ | True / True | 0.883 / 0.788 | 0.00442 / 0.00394 |
| RETIRED_B3CTL | True / True | 0.883 / 0.883 | 0.00442 / 0.00394 |
| SUCC | True / True | 0.883 / 0.788 | 0.00442 / 0.00394 |
| TR100 | None / None | nan / nan | None / None |
| TT100 | None / None | nan / nan | None / None |
| TWJ | True / True | 0.883 / 0.788 | 0.00442 / 0.00394 |
| W | True / True | 0.012 / 0.175 | -0.00385 / -0.00229 |

| second split (D2R) comparator | label | registered mean | robustness mean | robustness se | sign change |
|---|---|---|---|---|---|
| B2TWIN | NOT_RUN (not in the robustness bank) | | | | |
| H_LJ | ROBUSTNESS_NOT_CONFIRMED | +0.0168 | +0.0090 | 0.0127 | False |
| NEURAL_RAW | ROBUSTNESS_NOT_CONFIRMED | +0.0127 | +0.0120 | 0.0075 | False |
| RETIRED_B3CTL | ROBUSTNESS_NOT_CONFIRMED | -0.0038 | -0.0161 | 0.0191 | False |
| RF_ALTC_HI | ROBUSTNESS_NOT_CONFIRMED | +0.0005 | -0.0023 | 0.0049 | True |
| RF_RAWJ_HI | ROBUSTNESS_NOT_CONFIRMED | -0.0016 | -0.0044 | 0.0090 | False |
| RF_STOP_HI | ROBUSTNESS_NOT_CONFIRMED | +0.0028 | -0.0009 | 0.0036 | True |
| RF_TASK | ROBUSTNESS_NOT_CONFIRMED | +0.0086 | +0.0102 | 0.0074 | False |
| SUCC | ROBUSTNESS_NOT_CONFIRMED | +0.0037 | -0.0036 | 0.0130 | True |
| TR100 | NOT_RUN (not in the robustness bank) | | | | |
| TT100 | NOT_RUN (not in the robustness bank) | | | | |
| TWJ | NOT_RUN (not in the robustness bank) | | | | |
| W | NOT_RUN (not in the robustness bank) | | | | |

Falsifiers: {'F0': 'NOT_TRIGGERED', 'F1': 'NOT_TRIGGERED', 'F2': 'TRIGGERED', 'F3': 'TRIGGERED', 'F4': 'TRIGGERED', 'F5': 'NOT_TRIGGERED', 'F6': 'NOT_TRIGGERED'}. Learning-curve limitations against: ['RF_TASK', 'RF_ALTC_HI', 'H_LJ'].
D2R caveat: the pair benefit changes sign on the robustness cells against RF_STOP_HI, RF_ALTC_HI, SUCC; a single-view recipient condition fails on the robustness cells against RF_RAWJ_HI, H_LJ, RETIRED_B3CTL, SUCC: carried into the headline.

- fit v0: k* S20, F0 MECHANISM_ACTIVE (orthogonal share 0.102, cos 0.9951), lambda 3.972, brackets {'ALT': {'lo': 100, 'hi': 120}, 'ALTCRIT': {'lo': 10, 'hi': 20}, 'STOPFIT': {'lo': 10, 'hi': 20}}, ALTCRIT lag median 0.10948118632843748, CAND MODEL_FIT out-of-fold total-NLL spend over the start 0.0008 (se 0.0002), MODEL_FIT out-of-fold total NLL over the fold start at S20 {'FULL': 0.002918, 'STOPFIT': 0.003136, 'ALTCRIT': 0.003102, 'TASK': 0.001478}, fit CPU 232.2 s
- fit v1: k* S70, F0 MECHANISM_ACTIVE (orthogonal share 0.112, cos 0.9940), lambda 3.604, brackets {'ALT': {'lo': 170, 'hi': 200}, 'ALTCRIT': {'lo': 60, 'hi': 70}, 'STOPFIT': {'lo': 60, 'hi': 70}}, ALTCRIT lag median 0.21044219601019093, CAND MODEL_FIT out-of-fold total-NLL spend over the start 0.0041 (se 0.0005), MODEL_FIT out-of-fold total NLL over the fold start at S70 {'FULL': 0.00441, 'STOPFIT': 0.004981, 'ALTCRIT': 0.004877, 'TASK': -0.000367}, fit CPU 233.3 s

## Reading RFA_R1 (ADAPTIVE_R1; card RF_RFIT_R1.md)

Outcome class **ADAPTIVE_R1__BOUNDED_NEGATIVE_EXPERIMENT**. D1 ADAPTIVE_R1__TASK_FEASIBLE; D2 ADAPTIVE_R1__NOT_SEPARATED; D3 ADAPTIVE_R1__NOT_SUPERIOR; D4 ADAPTIVE_R1__NOT_MET; LC ADAPTIVE_R1__LEARNABILITY_LIMITATION; D2R ROBUSTNESS_NOT_CONFIRMED; D5 ADAPTIVE_R1__NOT_APPLICABLE_NOT_JOINT; D6 ADAPTIVE_R1__NOT_CLAIMED.

| D2 comparison | at | pair benefit mean | se_agg | lower | upper |
|---|---|---|---|---|---|
| START | NEURAL_RAW | +0.0098 | 0.0054 | -0.0010 | +0.0206 |
| CONT | RF1_TASK | +0.0090 | 0.0051 | -0.0013 | +0.0192 |
| STOPFIT | RF1_STOP_HI | -0.0014 | 0.0041 | -0.0097 | +0.0068 |
| ALTCRIT | RF1_ALTC_HI | -0.0030 | 0.0045 | -0.0121 | +0.0060 |
| RAWJ | RF1_RAWJ_HI | -0.0046 | 0.0052 | -0.0149 | +0.0057 |

| reference (D3) | pair benefit mean | se_agg | recipient gates | screen |
|---|---|---|---|---|
| B2TWIN | +0.0080 | 0.0116 | False | AGG_POSITIVE_POINT_ONLY |
| H_LJ | +0.0139 | 0.0108 | False | AGG_POSITIVE_POINT_ONLY |
| NEURAL_RAW | +0.0098 | 0.0054 | True | AGG_POSITIVE_POINT_ONLY |
| RETIRED_B3CTL | -0.0067 | 0.0136 | False | AGG_NOT_POSITIVE |
| SUCC | +0.0008 | 0.0106 | False | AGG_POSITIVE_POINT_ONLY |
| TR100 | +0.0314 | 0.0095 | True | AGG_POSITIVE |
| TT100 | +0.0317 | 0.0089 | True | AGG_POSITIVE |
| TWJ | +0.0237 | 0.0106 | False | AGG_POSITIVE |
| W | +0.0322 | 0.0089 | True | AGG_POSITIVE |

| member | pair AUC v0 / v1 | occupation view | income view |
|---|---|---|---|
| RETIRED_B3CTL | [0.6061, 0.5986] | [0.5665, 0.5682] | [0.5281, 0.5329] |
| SUCC | [0.6076, 0.6121] | [0.5665, 0.5682] | [0.5373, 0.5381] |
| RF1_RAWJ_HI | [0.6081, 0.6008] | [0.5944, 0.5986] | [0.5165, 0.514] |
| RF1_ALTC_HI | [0.6083, 0.6038] | [0.6, 0.6028] | [0.5358, 0.5319] |
| RF1_STOP_HI | [0.6084, 0.6068] | [0.6002, 0.6011] | [0.5358, 0.5319] |
| RF_FULL | [0.6099, 0.6024] | [0.5971, 0.5969] | [0.5358, 0.5319] |
| RF1_FULL | [0.6115, 0.6066] | [0.5979, 0.5983] | [0.5358, 0.5319] |
| RF1_STOP_LO | [0.6121, 0.6077] | [0.6036, 0.6003] | [0.5358, 0.5319] |
| B2TWIN | [0.6183, 0.6158] | [0.5665, 0.5682] | [0.5403, 0.5401] |
| NEURAL_RAW | [0.62, 0.6178] | [0.6089, 0.6132] | [0.536, 0.5319] |
| RF1_TASK | [0.6206, 0.6155] | [0.6135, 0.6131] | [0.5358, 0.5319] |
| H_LJ | [0.6259, 0.62] | [0.5665, 0.5682] | [0.5447, 0.5425] |
| Ucal | [0.631, 0.6363] | [0.6187, 0.6272] | [0.5506, 0.5483] |
| TWJ | [0.6321, 0.6335] | [0.5665, 0.5682] | [0.5536, 0.5628] |
| U0 | [0.6336, 0.6325] | [0.6188, 0.6277] | [0.551, 0.5483] |
| TR100 | [0.6387, 0.6422] | [0.6329, 0.6372] | [0.5373, 0.5381] |
| TT100 | [0.6408, 0.6408] | [0.6375, 0.6312] | [0.5373, 0.5381] |
| W | [0.6415, 0.6412] | [0.6311, 0.6354] | [0.5373, 0.5381] |

| member | G1 at the point v0 / v1 | max value / cap v0 / v1 | men's guard v0 / v1 |
|---|---|---|---|
| RF1_ALTC_HI | True / True | 0.551 / 0.839 | 0.00276 / 0.0042 |
| RF1_FULL | True / True | 0.634 / 0.915 | 0.00317 / 0.00457 |
| RF1_RAWJ_HI | True / False | 0.571 / 1.137 | 0.00261 / 0.00201 |
| RF1_STOP_HI | True / True | 0.593 / 0.958 | 0.00297 / 0.00479 |
| RF1_STOP_LO | True / True | 0.56 / 0.87 | 0.0028 / 0.00435 |
| RF1_TASK | True / True | 0.349 / 0.683 | 0.00174 / 0.00342 |
| NEURAL_RAW | True / True | 0.34 / 0.57 | 0.00147 / 0.00198 |

| reference | G1 at the point v0 / v1 | max value / cap v0 / v1 | men's guard v0 / v1 |
|---|---|---|---|
| B2TWIN | True / True | 0.883 / 0.788 | 0.00442 / 0.00394 |
| H_LJ | True / True | 0.883 / 0.788 | 0.00442 / 0.00394 |
| RETIRED_B3CTL | True / True | 0.883 / 0.883 | 0.00442 / 0.00394 |
| SUCC | True / True | 0.883 / 0.788 | 0.00442 / 0.00394 |
| TR100 | None / None | nan / nan | None / None |
| TT100 | None / None | nan / nan | None / None |
| TWJ | True / True | 0.883 / 0.788 | 0.00442 / 0.00394 |
| W | True / True | 0.012 / 0.175 | -0.00385 / -0.00229 |

Family-change reading FC_R1 (beside; never gating): ADAPTIVE_R1__FAMILY_CHANGE_NOT_POSITIVE, pair benefit -0.0029 (se 0.0046; version means {'0': -0.0015989184493598874, '1': -0.004213900075758031}), spend state R1_SPENDS_MORE_UNRESOLVED, secondary metrics resolved higher: ['occupation|men_brier'].

| second split (D2R) comparator | label | registered mean | robustness mean | robustness se | sign change |
|---|---|---|---|---|---|
| B2TWIN | NOT_RUN (not in the robustness bank) | | | | |
| H_LJ | ROBUSTNESS_NOT_CONFIRMED | +0.0139 | +0.0095 | 0.0133 | False |
| NEURAL_RAW | ROBUSTNESS_NOT_CONFIRMED | +0.0098 | +0.0124 | 0.0070 | False |
| RETIRED_B3CTL | ROBUSTNESS_NOT_CONFIRMED | -0.0067 | -0.0156 | 0.0189 | False |
| RF1_ALTC_HI | ROBUSTNESS_NOT_CONFIRMED | -0.0030 | +0.0016 | 0.0049 | True |
| RF1_RAWJ_HI | ROBUSTNESS_NOT_CONFIRMED | -0.0046 | -0.0011 | 0.0077 | False |
| RF1_STOP_HI | ROBUSTNESS_NOT_CONFIRMED | -0.0014 | -0.0007 | 0.0048 | False |
| RF1_TASK | ROBUSTNESS_NOT_CONFIRMED | +0.0090 | +0.0118 | 0.0064 | False |
| SUCC | ROBUSTNESS_NOT_CONFIRMED | +0.0008 | -0.0031 | 0.0130 | True |
| TR100 | NOT_RUN (not in the robustness bank) | | | | |
| TT100 | NOT_RUN (not in the robustness bank) | | | | |
| TWJ | NOT_RUN (not in the robustness bank) | | | | |
| W | NOT_RUN (not in the robustness bank) | | | | |

Falsifiers: {'F0': 'NOT_TRIGGERED', 'F1': 'NOT_TRIGGERED', 'F2': 'TRIGGERED', 'F3': 'TRIGGERED', 'F4': 'TRIGGERED', 'F5': 'NOT_TRIGGERED', 'F6': 'NOT_TRIGGERED'}. Learning-curve limitations against: ['H_LJ'].
D2R caveat: the pair benefit changes sign on the robustness cells against RF1_ALTC_HI, SUCC; a single-view recipient condition fails on the robustness cells against RF1_RAWJ_HI, H_LJ, RETIRED_B3CTL, SUCC: carried into the headline.

- fit v0: k* S30, F0 MECHANISM_ACTIVE (orthogonal share 0.098, cos 0.9956), lambda 4.100, brackets {'ALT': {'lo': 110, 'hi': 140}, 'ALTCRIT': {'lo': 20, 'hi': 30}, 'STOPFIT': {'lo': 20, 'hi': 30}}, ALTCRIT lag median 0.24739345219833608, CAND MODEL_FIT out-of-fold total-NLL spend over the start 0.0016 (se 0.0003), MODEL_FIT out-of-fold total NLL over the fold start at S30 {'FULL': 0.003699, 'STOPFIT': 0.003958, 'ALTCRIT': 0.004027, 'TASK': 0.001349}, fit CPU 302.6 s
- fit v1: k* S60, F0 MECHANISM_ACTIVE (orthogonal share 0.101, cos 0.9953), lambda 3.729, brackets {'ALT': {'lo': 150, 'hi': 180}, 'ALTCRIT': {'lo': 50, 'hi': 60}, 'STOPFIT': {'lo': 50, 'hi': 60}}, ALTCRIT lag median 0.2565273416551006, CAND MODEL_FIT out-of-fold total-NLL spend over the start 0.0035 (se 0.0004), MODEL_FIT out-of-fold total NLL over the fold start at S60 {'FULL': 0.003756, 'STOPFIT': 0.004191, 'ALTCRIT': 0.004274, 'TASK': -0.000438}, fit CPU 311.6 s

