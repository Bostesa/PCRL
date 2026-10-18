# Screen operating-characteristic study (question C)

Reader investigator, PCRL correction-transfer campaign. Public-use ACS 2019 1-Year PUMS Florida person records, MODEL_SELECT development role (adaptively used). Development readers estimate the recorded SEX column from released probability vectors only. Frozen and run before any new method candidate was scored with readers.

**What this is.** The source's per-cell recipient gate (OLD: every one of the 8 cells of candidate-minus-comparator single-output SCORE AUC <= +0.010, income and occupation views, each of the 6 required comparators) beside the statistical reviewer's predeclared aggregate paired rule (AGG1: U = dbar + z sqrt(phi v_samp) <= 0.010 per comparison, phi = max(1, phi_prior, phi_emp), gross phi_prior income 5.6365 / occupation 0.7782 / pair 1.0867, exact aliases pass). Both come from `cts.rules`. All rates are pass rates under the stated laws only, never universal error rates. An AGG1 pass is permission for further development, not evidence of protection. The source label NO_QUALIFYING_CONFIGURATION is never rewritten.

## Pre-outcome record: H3's frozen income view (every H3-anchored release)

| Comparator | Old gate | Old max cell | AGG1 | U | dbar | phi_emp |
|---|---|---|---|---|---|---|
| B2TWIN | pass | -0.0045 | AGG_PASS | 0.0018 | -0.0273 | 1.51 |
| H_LJ | FAIL | 0.0309 | AGG_FAIL | 0.0227 | -0.0108 | 2.58 |
| NEURAL_RAW | pass | -0.0042 | AGG_PASS | 0.0044 | -0.0248 | 0.69 |
| RETIRED_B3CTL | pass | 0.0087 | AGG_FAIL | 0.0159 | -0.0139 | 0.69 |
| SUCC | pass | 0.0000 | EXACT_ALIAS | 0.0000 | 0.0000 | - |
| TWJ | FAIL | 0.0147 | AGG_PASS | -0.0007 | -0.0339 | 0.35 |

Every H3-anchored release fails both configuration-level screens through its income view alone, whatever its occupation output (old gate: against H_LJ and TWJ; AGG1: against H_LJ and RETIRED_B3CTL).

## Part A: frozen source releases (label-free)

- Old gate reproduces the source's recipient failures for all 18 two-seed configurations: True.
- Descriptive what-if (recipient gate replaced by AGG1, everything else as recorded): **NO_QUALIFYING_CONFIGURATION (unchanged under AGG1)**.
- Required-comparator wire groups: occupation [['B2TWIN', 'H_LJ', 'RETIRED_B3CTL', 'SUCC', 'TWJ'], ['NEURAL_RAW']] (2 distinct wires); income [['B2TWIN'], ['H_LJ'], ['NEURAL_RAW'], ['RETIRED_B3CTL'], ['SUCC'], ['TWJ']].

| Configuration | Source label | Old fails | AGG1 fails | What-if label (AGG1) |
|---|---|---|---|---|
| A/A1_PO_16@H3 | RECIPIENT_POINT_FAILED | 8 | 8 | RECIPIENT_SCREEN_FAILED_AGG1 |
| A/A1_PO_8@H3 | RECIPIENT_POINT_FAILED | 8 | 8 | RECIPIENT_SCREEN_FAILED_AGG1 |
| B/B2_SINGLE@H3 | RECIPIENT_POINT_FAILED | 8 | 8 | RECIPIENT_SCREEN_FAILED_AGG1 |
| B/B2_STACK@H3 | RECIPIENT_POINT_FAILED | 8 | 8 | RECIPIENT_SCREEN_FAILED_AGG1 |
| B/B3_CALTR@H3 | RECIPIENT_POINT_FAILED | 8 | 8 | RECIPIENT_SCREEN_FAILED_AGG1 |
| INC/INC_H3_PO@COMPLETE | RECIPIENT_POINT_FAILED | 9 | 9 | RECIPIENT_SCREEN_FAILED_AGG1 |
| INC/INC_TW_PO@COMPLETE | RECIPIENT_POINT_FAILED | 12 | 12 | RECIPIENT_SCREEN_FAILED_AGG1 |
| PROT/P2A_COMP_B80@H3 | RECIPIENT_POINT_FAILED | 7 | 7 | RECIPIENT_SCREEN_FAILED_AGG1 |
| PROT/P2A_CONT@H3 | RECIPIENT_POINT_FAILED | 8 | 8 | RECIPIENT_SCREEN_FAILED_AGG1 |
| PROT/P2A_LOC_B80@H3 | RECIPIENT_POINT_FAILED | 7 | 7 | RECIPIENT_SCREEN_FAILED_AGG1 |
| PROT/P2A_PAIR_B50@H3 | RECIPIENT_POINT_FAILED | 7 | 7 | RECIPIENT_SCREEN_FAILED_AGG1 |
| PROT/P2A_PAIR_B80@H3 | RECIPIENT_POINT_FAILED | 7 | 7 | RECIPIENT_SCREEN_FAILED_AGG1 |
| PROT/P2_COMP_B80@H3 | RECIPIENT_POINT_FAILED | 7 | 7 | RECIPIENT_SCREEN_FAILED_AGG1 |
| PROT/P2_CONT@H3 | RECIPIENT_POINT_FAILED | 8 | 8 | RECIPIENT_SCREEN_FAILED_AGG1 |
| PROT/P2_LOC_B80@H3 | RECIPIENT_POINT_FAILED | 7 | 7 | RECIPIENT_SCREEN_FAILED_AGG1 |
| PROT/P2_PAIR_B50@H3 | RECIPIENT_POINT_FAILED | 7 | 7 | RECIPIENT_SCREEN_FAILED_AGG1 |
| PROT/P2_PAIR_B80@H3 | RECIPIENT_POINT_FAILED | 7 | 7 | RECIPIENT_SCREEN_FAILED_AGG1 |
| PROT/P2_PAIR_M90@H3 | RECIPIENT_POINT_FAILED | 7 | 7 | RECIPIENT_SCREEN_FAILED_AGG1 |

**L0a (exact aliases, shared cached readers).**

| View | Ordered alias pairs | All cells exactly 0 | Old pass | AGG1 pass |
|---|---|---|---|---|
| income | 380 | True | 1.000 | 1.000 |
| occupation | 28 | True | 1.000 | 1.000 |
| pair | 0 | True | - | - |

**Every pair of distinct frozen wires (descriptive; the truth is unknown).** Pass rates by the observed 8-cell mean:

- income: 9 wires, 72 oriented pairs; old 0.333, AGG1 0.278; old pass / AGG1 fail 8, old fail / AGG1 pass 4.

  | Observed mean band | n | Old pass | AGG1 pass |
  |---|---|---|---|
  | [-1.000, -0.020) | 11 | 0.818 | 1.000 |
  | [-0.020, -0.010) | 12 | 0.750 | 0.500 |
  | [-0.010, -0.005) | 7 | 0.429 | 0.429 |
  | [-0.005, +0.005) | 12 | 0.250 | 0.000 |
  | [+0.005, +0.010) | 7 | 0.000 | 0.000 |
  | [+0.010, +0.020) | 12 | 0.000 | 0.000 |
  | [+0.020, +1.000) | 11 | 0.000 | 0.000 |

- occupation: 21 wires, 420 oriented pairs; old 0.521, AGG1 0.540; old pass / AGG1 fail 2, old fail / AGG1 pass 10.

  | Observed mean band | n | Old pass | AGG1 pass |
  |---|---|---|---|
  | [-1.000, -0.020) | 130 | 1.000 | 1.000 |
  | [-0.020, -0.010) | 26 | 0.962 | 1.000 |
  | [-0.010, -0.005) | 23 | 0.957 | 1.000 |
  | [-0.005, +0.005) | 62 | 0.677 | 0.774 |
  | [+0.005, +0.010) | 23 | 0.000 | 0.000 |
  | [+0.010, +0.020) | 26 | 0.000 | 0.000 |
  | [+0.020, +1.000) | 130 | 0.000 | 0.000 |

## L0b: identical releases under separately fitted readers (true zero shift)

Members ['SUCC', 'H_LJ', 'NEURAL_RAW', 'A/A1_PO_16@H3', 'PROT/P2A_PAIR_B80@H3']; candidate readers refitted with new reader seeds, comparator = the same release with the registered (cached) or other new seeds, same FIT / VALIDATION / SCORE splits. Split noise cancels by construction, so this isolates reader fitting and winner selection. Entries share releases and seeds (counts, not independent rates).

| View | Design | Entries | Old pass | AGG1 pass | Max abs cell | Median rms cell SE | Share of cells exactly 0 |
|---|---|---|---|---|---|---|---|
| income | all_ordered_slot_pairs | 18 | 1.000 | 1.000 | 0.0074 | 0.0015 | 0.597 |
| income | primary | 3 | 1.000 | 1.000 | 0.0053 | 0.0010 | 0.583 |
| occupation | all_ordered_slot_pairs | 24 | 0.875 | 1.000 | 0.0134 | 0.0033 | 0.198 |
| occupation | primary | 4 | 1.000 | 1.000 | 0.0134 | 0.0036 | 0.156 |
| pair | all_ordered_slot_pairs | 30 | 0.700 | 0.967 | 0.0231 | 0.0038 | 0.167 |
| pair | primary | 5 | 0.600 | 1.000 | 0.0217 | 0.0038 | 0.175 |

## L1: cell-level synthetic law (exact truth; 20,000 simulations per setting)

d = mu + A_{s,R} + B_{s,R,k}; A ~ N(0, sd_A^2) correlated rho_seed across seeds; B ~ N(0, sd_k^2); reported SE = se_cell. Components (difference level, frozen grid): income: sd_A 0.021237, sd_k 0.001849, se 0.008962; occupation: sd_A 0.007565, sd_k 0.003682, se 0.009070; pair: sd_A 0.007505, sd_k 0.005924, se 0.008245.

| View | rho_seed | n_R | mu | Old pass (MC SE) | AGG1 pass (MC SE) |
|---|---|---|---|---|---|
| income | 0.0 | 2 | -0.030 | 0.8704 (0.0024) | 0.8913 (0.0022) |
| income | 0.0 | 2 | -0.020 | 0.6921 (0.0033) | 0.6418 (0.0034) |
| income | 0.0 | 2 | -0.010 | 0.4369 (0.0035) | 0.2967 (0.0032) |
| income | 0.0 | 2 | +0.000 | 0.1903 (0.0028) | 0.0762 (0.0019) |
| income | 0.0 | 2 | +0.005 | 0.1058 (0.0022) | 0.0282 (0.0012) |
| income | 0.0 | 2 | +0.010 | 0.0514 (0.0016) | 0.0082 (0.0006) |
| income | 0.0 | 2 | +0.020 | 0.0091 (0.0007) | 0.0008 (0.0002) |
| income | 0.0 | 3 | -0.030 | 0.8170 (0.0027) | 0.9822 (0.0009) |
| income | 0.0 | 3 | -0.020 | 0.5810 (0.0035) | 0.8495 (0.0025) |
| income | 0.0 | 3 | -0.010 | 0.2934 (0.0032) | 0.4709 (0.0035) |
| income | 0.0 | 3 | +0.000 | 0.0817 (0.0019) | 0.1135 (0.0022) |
| income | 0.0 | 3 | +0.005 | 0.0352 (0.0013) | 0.0348 (0.0013) |
| income | 0.0 | 3 | +0.010 | 0.0112 (0.0007) | 0.0094 (0.0007) |
| income | 0.0 | 3 | +0.020 | 0.0006 (0.0002) | 0.0003 (0.0001) |
| income | 0.76 | 2 | -0.030 | 0.8944 (0.0022) | 0.7861 (0.0029) |
| income | 0.76 | 2 | -0.020 | 0.7520 (0.0031) | 0.5605 (0.0035) |
| income | 0.76 | 2 | -0.010 | 0.5496 (0.0035) | 0.3100 (0.0033) |
| income | 0.76 | 2 | +0.000 | 0.3120 (0.0033) | 0.1162 (0.0023) |
| income | 0.76 | 2 | +0.005 | 0.2092 (0.0029) | 0.0659 (0.0018) |
| income | 0.76 | 2 | +0.010 | 0.1366 (0.0024) | 0.0315 (0.0012) |
| income | 0.76 | 2 | +0.020 | 0.0411 (0.0014) | 0.0056 (0.0005) |
| income | 0.76 | 3 | -0.030 | 0.8479 (0.0025) | 0.9210 (0.0019) |
| income | 0.76 | 3 | -0.020 | 0.6537 (0.0034) | 0.7396 (0.0031) |
| income | 0.76 | 3 | -0.010 | 0.4015 (0.0035) | 0.4297 (0.0035) |
| income | 0.76 | 3 | +0.000 | 0.1756 (0.0027) | 0.1595 (0.0026) |
| income | 0.76 | 3 | +0.005 | 0.0983 (0.0021) | 0.0731 (0.0018) |
| income | 0.76 | 3 | +0.010 | 0.0491 (0.0015) | 0.0331 (0.0013) |
| income | 0.76 | 3 | +0.020 | 0.0078 (0.0006) | 0.0033 (0.0004) |
| occupation | 0.0 | 2 | -0.030 | 1.0000 (0.0000) | 1.0000 (0.0000) |
| occupation | 0.0 | 2 | -0.020 | 0.9990 (0.0002) | 0.9999 (0.0001) |
| occupation | 0.0 | 2 | -0.010 | 0.9448 (0.0016) | 0.9830 (0.0009) |
| occupation | 0.0 | 2 | +0.000 | 0.4870 (0.0035) | 0.4234 (0.0035) |
| occupation | 0.0 | 2 | +0.005 | 0.1681 (0.0026) | 0.0752 (0.0019) |
| occupation | 0.0 | 2 | +0.010 | 0.0255 (0.0011) | 0.0039 (0.0004) |
| occupation | 0.0 | 2 | +0.020 | 0.0001 (0.0000) | 0.0000 (0.0000) |
| occupation | 0.0 | 3 | -0.030 | 1.0000 (0.0000) | 1.0000 (0.0000) |
| occupation | 0.0 | 3 | -0.020 | 0.9977 (0.0003) | 1.0000 (0.0000) |
| occupation | 0.0 | 3 | -0.010 | 0.9174 (0.0019) | 0.9996 (0.0001) |
| occupation | 0.0 | 3 | +0.000 | 0.3382 (0.0033) | 0.6557 (0.0034) |
| occupation | 0.0 | 3 | +0.005 | 0.0673 (0.0018) | 0.1258 (0.0023) |
| occupation | 0.0 | 3 | +0.010 | 0.0034 (0.0004) | 0.0040 (0.0004) |
| occupation | 0.0 | 3 | +0.020 | 0.0000 (0.0000) | 0.0000 (0.0000) |
| occupation | 0.76 | 2 | -0.030 | 1.0000 (0.0000) | 1.0000 (0.0000) |
| occupation | 0.76 | 2 | -0.020 | 0.9983 (0.0003) | 0.9970 (0.0004) |
| occupation | 0.76 | 2 | -0.010 | 0.9513 (0.0015) | 0.9338 (0.0018) |
| occupation | 0.76 | 2 | +0.000 | 0.5599 (0.0035) | 0.4034 (0.0035) |
| occupation | 0.76 | 2 | +0.005 | 0.2608 (0.0031) | 0.1237 (0.0023) |
| occupation | 0.76 | 2 | +0.010 | 0.0701 (0.0018) | 0.0171 (0.0009) |
| occupation | 0.76 | 2 | +0.020 | 0.0008 (0.0002) | 0.0001 (0.0000) |
| occupation | 0.76 | 3 | -0.030 | 1.0000 (0.0000) | 1.0000 (0.0000) |
| occupation | 0.76 | 3 | -0.020 | 0.9989 (0.0002) | 1.0000 (0.0000) |
| occupation | 0.76 | 3 | -0.010 | 0.9284 (0.0018) | 0.9905 (0.0007) |
| occupation | 0.76 | 3 | +0.000 | 0.4209 (0.0035) | 0.5825 (0.0035) |
| occupation | 0.76 | 3 | +0.005 | 0.1354 (0.0024) | 0.1757 (0.0027) |
| occupation | 0.76 | 3 | +0.010 | 0.0175 (0.0009) | 0.0167 (0.0009) |
| occupation | 0.76 | 3 | +0.020 | 0.0001 (0.0001) | 0.0000 (0.0000) |
| pair | 0.0 | 2 | -0.030 | 0.9998 (0.0001) | 1.0000 (0.0000) |
| pair | 0.0 | 2 | -0.020 | 0.9945 (0.0005) | 0.9998 (0.0001) |
| pair | 0.0 | 2 | -0.010 | 0.8746 (0.0023) | 0.9767 (0.0011) |
| pair | 0.0 | 2 | +0.000 | 0.3532 (0.0034) | 0.4593 (0.0035) |
| pair | 0.0 | 2 | +0.005 | 0.1106 (0.0022) | 0.1130 (0.0022) |
| pair | 0.0 | 2 | +0.010 | 0.0146 (0.0008) | 0.0076 (0.0006) |
| pair | 0.0 | 2 | +0.020 | 0.0000 (0.0000) | 0.0000 (0.0000) |
| pair | 0.0 | 3 | -0.030 | 0.9999 (0.0001) | 1.0000 (0.0000) |
| pair | 0.0 | 3 | -0.020 | 0.9907 (0.0007) | 1.0000 (0.0000) |
| pair | 0.0 | 3 | -0.010 | 0.8271 (0.0027) | 0.9993 (0.0002) |
| pair | 0.0 | 3 | +0.000 | 0.2144 (0.0029) | 0.6770 (0.0033) |
| pair | 0.0 | 3 | +0.005 | 0.0355 (0.0013) | 0.1723 (0.0027) |
| pair | 0.0 | 3 | +0.010 | 0.0021 (0.0003) | 0.0082 (0.0006) |
| pair | 0.0 | 3 | +0.020 | 0.0000 (0.0000) | 0.0000 (0.0000) |
| pair | 0.76 | 2 | -0.030 | 0.9999 (0.0001) | 0.9998 (0.0001) |
| pair | 0.76 | 2 | -0.020 | 0.9928 (0.0006) | 0.9958 (0.0005) |
| pair | 0.76 | 2 | -0.010 | 0.8899 (0.0022) | 0.9261 (0.0018) |
| pair | 0.76 | 2 | +0.000 | 0.4184 (0.0035) | 0.4289 (0.0035) |
| pair | 0.76 | 2 | +0.005 | 0.1735 (0.0027) | 0.1474 (0.0025) |
| pair | 0.76 | 2 | +0.010 | 0.0423 (0.0014) | 0.0273 (0.0012) |
| pair | 0.76 | 2 | +0.020 | 0.0003 (0.0001) | 0.0000 (0.0000) |
| pair | 0.76 | 3 | -0.030 | 0.9997 (0.0001) | 1.0000 (0.0000) |
| pair | 0.76 | 3 | -0.020 | 0.9900 (0.0007) | 1.0000 (0.0000) |
| pair | 0.76 | 3 | -0.010 | 0.8360 (0.0026) | 0.9869 (0.0008) |
| pair | 0.76 | 3 | +0.000 | 0.2792 (0.0032) | 0.6048 (0.0035) |
| pair | 0.76 | 3 | +0.005 | 0.0707 (0.0018) | 0.2015 (0.0028) |
| pair | 0.76 | 3 | +0.010 | 0.0088 (0.0007) | 0.0245 (0.0011) |
| pair | 0.76 | 3 | +0.020 | 0.0000 (0.0000) | 0.0000 (0.0000) |

**Multiplicity (configuration level: all 12 comparisons must pass; release-level effects, candidate shared).**

| Structure | Pattern | rho_seed | n_R | Old pass-all (MC SE) | AGG1 pass-all (MC SE) |
|---|---|---|---|---|---|
| independent | ALL_0 | 0.76 | 2 | 0.0043 (0.0005) | 0.0002 (0.0001) |
| independent | ALL_-0.010 | 0.76 | 2 | 0.1083 (0.0022) | 0.0319 (0.0012) |
| independent | IUT_income_+0.010_others_0 | 0.76 | 2 | 0.0034 (0.0004) | 0.0003 (0.0001) |
| independent | IUT_income_+0.010_others_-0.010 | 0.76 | 2 | 0.0493 (0.0015) | 0.0078 (0.0006) |
| independent | IUT_income_+0.020_others_0 | 0.76 | 2 | 0.0018 (0.0003) | 0.0001 (0.0001) |
| independent | IUT_income_+0.020_others_-0.010 | 0.76 | 2 | 0.0192 (0.0010) | 0.0024 (0.0003) |
| independent | IUT_occupation_+0.010_others_0 | 0.76 | 2 | 0.0015 (0.0003) | 0.0000 (0.0000) |
| independent | IUT_occupation_+0.010_others_-0.010 | 0.76 | 2 | 0.0104 (0.0007) | 0.0009 (0.0002) |
| independent | IUT_occupation_+0.020_others_0 | 0.76 | 2 | 0.0000 (0.0000) | 0.0000 (0.0000) |
| independent | IUT_occupation_+0.020_others_-0.010 | 0.76 | 2 | 0.0001 (0.0001) | 0.0000 (0.0000) |
| independent | ALL_0 | 0.0 | 2 | 0.0006 (0.0002) | 0.0003 (0.0001) |
| independent | ALL_-0.010 | 0.0 | 2 | 0.0478 (0.0015) | 0.0355 (0.0013) |
| independent | IUT_income_+0.010_others_0 | 0.0 | 2 | 0.0002 (0.0001) | 0.0000 (0.0000) |
| independent | IUT_income_+0.010_others_-0.010 | 0.0 | 2 | 0.0138 (0.0008) | 0.0047 (0.0005) |
| independent | IUT_income_+0.020_others_0 | 0.0 | 2 | 0.0001 (0.0001) | 0.0000 (0.0000) |
| independent | IUT_income_+0.020_others_-0.010 | 0.0 | 2 | 0.0029 (0.0004) | 0.0005 (0.0002) |
| independent | IUT_occupation_+0.010_others_0 | 0.0 | 2 | 0.0001 (0.0000) | 0.0000 (0.0000) |
| independent | IUT_occupation_+0.010_others_-0.010 | 0.0 | 2 | 0.0015 (0.0003) | 0.0001 (0.0000) |
| independent | IUT_occupation_+0.020_others_0 | 0.0 | 2 | 0.0000 (0.0000) | 0.0000 (0.0000) |
| independent | IUT_occupation_+0.020_others_-0.010 | 0.0 | 2 | 0.0000 (0.0000) | 0.0000 (0.0000) |
| independent | ALL_0 | 0.76 | 3 | 0.0002 (0.0001) | 0.0016 (0.0003) |
| independent | ALL_-0.010 | 0.76 | 3 | 0.0383 (0.0014) | 0.0875 (0.0020) |
| independent | IUT_income_+0.010_others_0 | 0.76 | 3 | 0.0001 (0.0001) | 0.0006 (0.0002) |
| independent | IUT_income_+0.010_others_-0.010 | 0.76 | 3 | 0.0108 (0.0007) | 0.0159 (0.0009) |
| independent | IUT_income_+0.020_others_0 | 0.76 | 3 | 0.0001 (0.0000) | 0.0003 (0.0001) |
| independent | IUT_income_+0.020_others_-0.010 | 0.76 | 3 | 0.0025 (0.0004) | 0.0021 (0.0003) |
| independent | IUT_occupation_+0.010_others_0 | 0.76 | 3 | 0.0000 (0.0000) | 0.0001 (0.0001) |
| independent | IUT_occupation_+0.010_others_-0.010 | 0.76 | 3 | 0.0008 (0.0002) | 0.0022 (0.0003) |
| independent | IUT_occupation_+0.020_others_0 | 0.76 | 3 | 0.0000 (0.0000) | 0.0000 (0.0000) |
| independent | IUT_occupation_+0.020_others_-0.010 | 0.76 | 3 | 0.0000 (0.0000) | 0.0000 (0.0000) |
| h3_actual | ALL_0 | 0.76 | 2 | 0.0183 (0.0009) | 0.0016 (0.0003) |
| h3_actual | ALL_-0.010 | 0.76 | 2 | 0.1497 (0.0025) | 0.0450 (0.0015) |
| h3_actual | IUT_income_+0.010_others_0 | 0.76 | 2 | 0.0102 (0.0007) | 0.0011 (0.0002) |
| h3_actual | IUT_occupation_+0.010_others_0 | 0.76 | 2 | 0.0024 (0.0003) | 0.0001 (0.0001) |

## L2: semi-synthetic SEX-dependent shift of frozen views (one grid realization per row; truth approximate)

| Base view | Target LOG_LR gain | Realized slate mean shift | Fixed-recipe shift (10 extra splits) | Old | AGG1 | U |
|---|---|---|---|---|---|---|
| JOINT_OCC | +0.005 | +0.0201 | +0.0029 | FAIL | FAIL | +0.0274 |
| JOINT_OCC | +0.010 | +0.0376 | +0.0068 | FAIL | FAIL | +0.0476 |
| JOINT_OCC | +0.020 | +0.0824 | +0.0165 | FAIL | FAIL | +0.0934 |
| NEURAL_RAW_OCC | +0.005 | +0.0239 | +0.0049 | FAIL | FAIL | +0.0318 |
| NEURAL_RAW_OCC | +0.010 | +0.0347 | +0.0100 | FAIL | FAIL | +0.0435 |
| NEURAL_RAW_OCC | +0.020 | +0.0620 | +0.0197 | FAIL | FAIL | +0.0725 |
| H_LJ_INC | +0.005 | +0.0149 | +0.0062 | FAIL | FAIL | +0.0483 |
| H_LJ_INC | +0.010 | +0.0171 | +0.0115 | FAIL | FAIL | +0.0424 |
| H_LJ_INC | +0.020 | +0.0349 | +0.0211 | FAIL | FAIL | +0.0617 |
| A1_PO_16_OCC (addition) | +0.005 | +0.1525 | +0.0067 | FAIL | FAIL | +0.1638 |
| A1_PO_16_OCC (addition) | +0.010 | +0.1701 | +0.0117 | FAIL | FAIL | +0.1818 |
| A1_PO_16_OCC (addition) | +0.020 | +0.1961 | +0.0220 | FAIL | FAIL | +0.2082 |

## Limits

- Rates hold under the stated laws only. L0b and L2 are single realizations of the registered cells; L0b isolates reader fitting (split noise cancels for identical releases), L2 truth is approximate.
- The frozen-source rows are adaptive development evidence on MODEL_SELECT; readers are the registered slate only.
- The what-if is descriptive: it changes one component of the source nomination and rewrites nothing.
