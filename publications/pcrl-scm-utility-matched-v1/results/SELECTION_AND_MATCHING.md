# Selection and matching (MODEL_SELECT only; frozen in SELECTION_LOCK.json)

"MATCHED" below is the MODEL_SELECT role status. None of these matches carried over to DEV_REPORT (`RESEARCH_DECISION.md` §6).

**Status:** `NOMINEE_SELECTED`; common threshold t* = 0.008; cells K* = 298 / 300; expenditure e* = 0.00825, 0.00843 nats (seed 0, seed 1).

## SCM nomination grid (both seeds; eligible = 24 clauses + occupation NLL excess ≤ 0.008 + both within-SEX guards)

| t | cells s0 / s1 | eligible | worst normalised | mean proxy | expenditure s0 / s1 | failed conditions |
|---|---|---|---|---|---|---|
| 0.000 | 512 / 512 | True | 0.322 | 0.6479 | 0.00000 / 0.00000 | — |
| 0.001 | 462 / 466 | True | 0.365 | 0.6447 | 0.00098 / 0.00166 | — |
| 0.002 | 433 / 424 | True | 0.414 | 0.6414 | 0.00263 / 0.00290 | — |
| 0.003 | 402 / 400 | True | 0.546 | 0.6363 | 0.00318 / 0.00436 | — |
| 0.004 | 374 / 370 | True | 0.492 | 0.6333 | 0.00378 / 0.00484 | — |
| 0.005 | 355 / 352 | True | 0.575 | 0.6297 | 0.00517 / 0.00599 | — |
| 0.006 | 341 / 333 | True | 0.743 | 0.6253 | 0.00643 / 0.00720 | — |
| 0.007 | 319 / 320 | True | 0.824 | 0.6192 | 0.00748 / 0.00803 | — |
| 0.008 | 298 / 300 | True | 0.849 | 0.6166 | 0.00825 / 0.00843 | — |
| 0.009 | 285 / 289 | False | 1.065 | 0.6147 | 0.00856 / 0.00999 | s1: subgroup|sex0|occupation_brier_excess_vs_U0 |
| 0.010 | 261 / 266 | False | 1.189 | 0.6095 | 0.01055 / 0.01113 | s1: subgroup|sex0|occupation_brier_excess_vs_U0 |

## Roles (per seed)

| seed | role | status | trajectory | labels | cells | release | Δ expenditure (comparator − SCM) | proxy | 12 clauses |
|---|---|---|---|---|---|---|---|---|---|
| 0 | SCM | NOMINEE_SELECTED | SCM | N0.008 | 298 | e6f5b8d1f0d7c8c9 | +0.00000 | 0.6170 | True |
| 0 | ORD_NLLM | MATCHED | ORD | N0.006 | 129 | 4b522bb361be87ef | +0.00003 | 0.6448 | True |
| 0 | ORD_CNT | SAME_COUNT | ORD | S0.008 | 298 | 7c6a63ae8f2dbdb2 | -0.00569 | 0.6461 | True |
| 0 | IBSI_NLLM | MATCHED | IBSI_g2.0 | N0.007 | 473 | f00521b665dbc140 | -0.00065 | 0.6201 | True |
| 0 | IBSI_CNT | SAME_COUNT | IBSI_g1.0 | S0.008 | 298 | 050ece55d6410ec5 | -0.00187 | 0.6252 | True |
| 0 | SHUF_NLLM | MATCHED | SHUF_r2 | N0.007 | 158 | 92dbffd3ca006f6e | -0.00056 | 0.6422 | True |
| 0 | NEURAL_RAW | FROZEN_REFERENCE | — |  | — | FROZEN_NEURAL_RAW | -0.00074 | 0.6236 | True |
| 0 | H_LJ | FROZEN_REFERENCE | — |  | — | FROZEN_H_LJ | +0.00138 | 0.6224 | True |
| 0 | C0 | REFERENCE | SCM | C512 | 512 | a1f1af7283062a1c | -0.00825 | 0.6480 | True |
| 0 | HIST_SCM | HISTORICALLY_SELECTED_DIAGNOSTIC | SCM | TERM,HIST | 261 | ac661f81d4b158a3 | +0.00231 | 0.6105 | True |
| 1 | SCM | NOMINEE_SELECTED | SCM | N0.008 | 300 | d82ea5d4b8d1fc50 | +0.00000 | 0.6162 | True |
| 1 | ORD_NLLM | MATCHED | ORD | N0.007 | 105 | 56f840c63ffd8a02 | -0.00015 | 0.6430 | True |
| 1 | ORD_CNT | SAME_COUNT | ORD | S0.008 | 300 | 0f0374a57196cb68 | -0.00571 | 0.6464 | True |
| 1 | IBSI_NLLM | MATCHED | IBSI_g2.0 | N0.007 | 475 | b046cee06ff09eb3 | +0.00016 | 0.6179 | True |
| 1 | IBSI_CNT | SAME_COUNT | IBSI_g1.0 | S0.008 | 300 | b30fa55c94f79d2d | -0.00096 | 0.6211 | True |
| 1 | SHUF_NLLM | MATCHED | SHUF_r2 | N0.007 | 180 | 4351213d567db348 | +0.00027 | 0.6384 | True |
| 1 | NEURAL_RAW | FROZEN_REFERENCE | — |  | — | FROZEN_NEURAL_RAW | -0.00263 | 0.6269 | True |
| 1 | H_LJ | FROZEN_REFERENCE | — |  | — | FROZEN_H_LJ | -0.00105 | 0.6233 | True |
| 1 | C0 | REFERENCE | SCM | C512 | 512 | 4d990c314543e156 | -0.00843 | 0.6478 | True |
| 1 | HIST_SCM | HISTORICALLY_SELECTED_DIAGNOSTIC | SCM | TERM,HIST | 266 | 4d340fb4b092857a | +0.00270 | 0.6085 | True |

## Matching outcome

- ORD_NLLM available (MODEL_SELECT match within 0.001 and 12 clauses): seed 0 True, seed 1 True
- IBSI_NLLM: trajectory IBSI_g2.0, matched in both seeds: True, admissible: IBSI_g0.5, IBSI_g1.0, IBSI_g2.0
- SHUF_NLLM: trajectory SHUF_r2, matched in both seeds: True, admissible: SHUF_r0, SHUF_r2, SHUF_r3
- ORD_CNT available at exactly K* cells in both seeds: True
- IBSI_CNT: IBSI_g1.0 (24-clause eligible: True)

## Selection effort (distinct candidate releases examined)

```
{
 "IBSI_CNT": 3,
 "IBSI_NLLM": {
  "IBSI_g0.5": {
   "0": 29,
   "1": 30
  },
  "IBSI_g1.0": {
   "0": 32,
   "1": 30
  },
  "IBSI_g2.0": {
   "0": 32,
   "1": 31
  }
 },
 "ORD_NLLM": {
  "0": 31,
  "1": 30
 },
 "SHUF_NLLM": {
  "SHUF_r0": {
   "0": 28,
   "1": 27
  },
  "SHUF_r1": {
   "0": 29,
   "1": 26
  },
  "SHUF_r2": {
   "0": 28,
   "1": 28
  },
  "SHUF_r3": {
   "0": 29,
   "1": 27
  },
  "SHUF_r4": {
   "0": 29,
   "1": 28
  }
 }
}
```

## Candidates nearest the SCM expenditure (per comparator trajectory and seed)

| seed | trajectory | labels | cells | Δ expenditure (candidate − SCM) | match | 12 clauses | proxy |
|---|---|---|---|---|---|---|---|
| 0 | IBSI_g0.5 | C96,N0.009 | 96 | +0.00019 | True | True | 0.6356 |
| 0 | IBSI_g0.5 | N0.008 | 109 | -0.00022 | True | True | 0.6359 |
| 0 | IBSI_g0.5 | C128,N0.007 | 128 | -0.00034 | True | True | 0.6388 |
| 1 | IBSI_g0.5 | C96 | 96 | -0.00023 | True | True | 0.6333 |
| 1 | IBSI_g0.5 | N0.008 | 106 | -0.00058 | True | True | 0.6320 |
| 1 | IBSI_g0.5 | N0.009 | 90 | +0.00100 | True | True | 0.6333 |
| 0 | IBSI_g1.0 | N0.007 | 225 | +0.00039 | True | True | 0.6256 |
| 0 | IBSI_g1.0 | C224 | 224 | +0.00043 | True | True | 0.6257 |
| 0 | IBSI_g1.0 | C192 | 192 | +0.00047 | True | True | 0.6243 |
| 1 | IBSI_g1.0 | HIST,S0.010 | 266 | +0.00038 | True | True | 0.6196 |
| 1 | IBSI_g1.0 | C256 | 256 | +0.00054 | True | True | 0.6195 |
| 1 | IBSI_g1.0 | S0.009 | 289 | -0.00069 | True | True | 0.6204 |
| 0 | IBSI_g2.0 | N0.007 | 473 | -0.00065 | True | True | 0.6201 |
| 0 | IBSI_g2.0 | N0.008 | 464 | +0.00118 | False | True | 0.6106 |
| 0 | IBSI_g2.0 | S0.001 | 462 | +0.00139 | False | True | 0.6104 |
| 1 | IBSI_g2.0 | N0.007 | 475 | +0.00016 | True | True | 0.6179 |
| 1 | IBSI_g2.0 | N0.006 | 479 | -0.00030 | True | True | 0.6190 |
| 1 | IBSI_g2.0 | N0.005 | 486 | -0.00139 | False | True | 0.6240 |
| 0 | ORD | C128 | 128 | +0.00000 | True | True | 0.6450 |
| 0 | ORD | N0.006 | 129 | +0.00003 | True | True | 0.6448 |
| 0 | ORD | N0.005 | 157 | -0.00126 | False | True | 0.6455 |
| 1 | ORD | N0.007 | 105 | -0.00015 | True | True | 0.6430 |
| 1 | ORD | C96 | 96 | +0.00066 | True | True | 0.6433 |
| 1 | ORD | N0.008 | 85 | +0.00126 | False | True | 0.6434 |
| 0 | SHUF_r0 | N0.008 | 149 | +0.00031 | True | True | 0.6422 |
| 0 | SHUF_r0 | C160 | 160 | -0.00105 | False | True | 0.6419 |
| 0 | SHUF_r0 | N0.007 | 174 | -0.00124 | False | True | 0.6430 |
| 1 | SHUF_r0 | C160 | 160 | +0.00050 | True | True | 0.6420 |
| 1 | SHUF_r0 | N0.008 | 159 | +0.00056 | True | True | 0.6418 |
| 1 | SHUF_r0 | N0.007 | 177 | -0.00082 | True | True | 0.6425 |
| 0 | SHUF_r1 | C192 | 192 | -0.00025 | True | True | 0.6417 |
| 0 | SHUF_r1 | N0.007 | 168 | +0.00057 | True | True | 0.6415 |
| 0 | SHUF_r1 | N0.006 | 196 | -0.00060 | True | True | 0.6422 |
| 1 | SHUF_r1 | N0.009 | 144 | -0.00105 | False | True | 0.6423 |
| 1 | SHUF_r1 | N0.008 | 159 | -0.00144 | False | True | 0.6434 |
| 1 | SHUF_r1 | C128,N0.010,TERM | 128 | +0.00156 | False | True | 0.6405 |
| 0 | SHUF_r2 | N0.008 | 142 | +0.00039 | True | True | 0.6424 |
| 0 | SHUF_r2 | N0.007 | 158 | -0.00056 | True | True | 0.6422 |
| 0 | SHUF_r2 | C160 | 160 | -0.00064 | True | True | 0.6423 |
| 1 | SHUF_r2 | N0.007 | 180 | +0.00027 | True | True | 0.6384 |
| 1 | SHUF_r2 | N0.006 | 195 | -0.00087 | True | True | 0.6396 |
| 1 | SHUF_r2 | C192 | 192 | -0.00093 | True | True | 0.6396 |
| 0 | SHUF_r3 | C192 | 192 | +0.00030 | True | True | 0.6403 |
| 0 | SHUF_r3 | N0.006 | 199 | -0.00041 | True | True | 0.6400 |
| 0 | SHUF_r3 | C224 | 224 | -0.00137 | False | True | 0.6413 |
| 1 | SHUF_r3 | N0.008 | 144 | +0.00004 | True | True | 0.6475 |
| 1 | SHUF_r3 | N0.009 | 142 | +0.00007 | True | True | 0.6474 |
| 1 | SHUF_r3 | C160 | 160 | -0.00069 | True | True | 0.6476 |
| 0 | SHUF_r4 | N0.010,TERM | 124 | -0.00117 | False | True | 0.6394 |
| 0 | SHUF_r4 | C128 | 128 | -0.00118 | False | True | 0.6394 |
| 0 | SHUF_r4 | N0.009 | 142 | -0.00198 | False | True | 0.6406 |
| 1 | SHUF_r4 | C160 | 160 | +0.00015 | True | True | 0.6417 |
| 1 | SHUF_r4 | N0.008 | 156 | +0.00030 | True | True | 0.6419 |
| 1 | SHUF_r4 | N0.007 | 170 | -0.00031 | True | True | 0.6425 |
