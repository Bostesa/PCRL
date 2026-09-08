# Support and exposed-control competence

This stage reuses the original household pools and fixed two-class SEX/nine-class RAC1P schema. All original test outcomes are DEVELOPMENT EVALUATION. Every candidate’s category support, recall, precision, F1 and AUROC appears in [PER_CLASS.csv](PER_CLASS.csv). Class codes below are the original Census codes, not zero-based indices.

## Original pools

| Seed | Pool | People | Households | Target | Valid | Missing/inapplicable | Class support |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | representation_fit | 10513 | 7051 | same_residence | 10513 | 0 | [2466, 8047] |
| 0 | representation_fit | 10513 | 7051 | commute_over20 | 6985 | 3528 | [3447, 3538] |
| 0 | representation_fit | 10513 | 7051 | income_binary | 10513 | 0 | [8531, 1982] |
| 0 | representation_fit | 10513 | 7051 | civilian_at_work | 10513 | 0 | [3392, 7121] |
| 0 | representation_fit | 10513 | 7051 | public_coverage | 10513 | 0 | [7976, 2537] |
| 0 | representation_fit | 10513 | 7051 | SEX | 10513 | 0 | [5391, 5122] |
| 0 | representation_fit | 10513 | 7051 | RAC1P | 10513 | 0 | [5989, 555, 90, 1, 32, 1674, 49, 1527, 596] |
| 0 | source_validation | 3004 | 2015 | same_residence | 3004 | 0 | [694, 2310] |
| 0 | source_validation | 3004 | 2015 | commute_over20 | 1996 | 1008 | [1017, 979] |
| 0 | source_validation | 3004 | 2015 | income_binary | 3004 | 0 | [2459, 545] |
| 0 | source_validation | 3004 | 2015 | civilian_at_work | 3004 | 0 | [969, 2035] |
| 0 | source_validation | 3004 | 2015 | public_coverage | 3004 | 0 | [2263, 741] |
| 0 | source_validation | 3004 | 2015 | SEX | 3004 | 0 | [1577, 1427] |
| 0 | source_validation | 3004 | 2015 | RAC1P | 3004 | 0 | [1639, 179, 15, 0, 6, 515, 7, 472, 171] |
| 0 | downstream_fit | 4539 | 3022 | same_residence | 4539 | 0 | [1049, 3490] |
| 0 | downstream_fit | 4539 | 3022 | commute_over20 | 3026 | 1513 | [1466, 1560] |
| 0 | downstream_fit | 4539 | 3022 | income_binary | 4539 | 0 | [3688, 851] |
| 0 | downstream_fit | 4539 | 3022 | civilian_at_work | 4539 | 0 | [1462, 3077] |
| 0 | downstream_fit | 4539 | 3022 | public_coverage | 4539 | 0 | [3427, 1112] |
| 0 | downstream_fit | 4539 | 3022 | SEX | 4539 | 0 | [2355, 2184] |
| 0 | downstream_fit | 4539 | 3022 | RAC1P | 4539 | 0 | [2578, 257, 39, 0, 13, 678, 20, 725, 229] |
| 0 | downstream_validation | 2984 | 2015 | same_residence | 2984 | 0 | [700, 2284] |
| 0 | downstream_validation | 2984 | 2015 | commute_over20 | 1984 | 1000 | [1009, 975] |
| 0 | downstream_validation | 2984 | 2015 | income_binary | 2984 | 0 | [2434, 550] |
| 0 | downstream_validation | 2984 | 2015 | civilian_at_work | 2984 | 0 | [981, 2003] |
| 0 | downstream_validation | 2984 | 2015 | public_coverage | 2984 | 0 | [2275, 709] |
| 0 | downstream_validation | 2984 | 2015 | SEX | 2984 | 0 | [1551, 1433] |
| 0 | downstream_validation | 2984 | 2015 | RAC1P | 2984 | 0 | [1723, 135, 16, 0, 4, 426, 10, 490, 180] |
| 0 | attacker_fit | 2993 | 2015 | same_residence | 2993 | 0 | [702, 2291] |
| 0 | attacker_fit | 2993 | 2015 | commute_over20 | 1973 | 1020 | [1019, 954] |
| 0 | attacker_fit | 2993 | 2015 | income_binary | 2993 | 0 | [2413, 580] |
| 0 | attacker_fit | 2993 | 2015 | civilian_at_work | 2993 | 0 | [956, 2037] |
| 0 | attacker_fit | 2993 | 2015 | public_coverage | 2993 | 0 | [2285, 708] |
| 0 | attacker_fit | 2993 | 2015 | SEX | 2993 | 0 | [1512, 1481] |
| 0 | attacker_fit | 2993 | 2015 | RAC1P | 2993 | 0 | [1704, 173, 17, 0, 4, 461, 22, 424, 188] |
| 0 | attacker_validation | 2985 | 2014 | same_residence | 2985 | 0 | [717, 2268] |
| 0 | attacker_validation | 2985 | 2014 | commute_over20 | 1987 | 998 | [996, 991] |
| 0 | attacker_validation | 2985 | 2014 | income_binary | 2985 | 0 | [2442, 543] |
| 0 | attacker_validation | 2985 | 2014 | civilian_at_work | 2985 | 0 | [960, 2025] |
| 0 | attacker_validation | 2985 | 2014 | public_coverage | 2985 | 0 | [2258, 727] |
| 0 | attacker_validation | 2985 | 2014 | SEX | 2985 | 0 | [1541, 1444] |
| 0 | attacker_validation | 2985 | 2014 | RAC1P | 2985 | 0 | [1737, 144, 18, 0, 11, 468, 16, 409, 182] |
| 0 | development_evaluation | 2982 | 2015 | same_residence | 2982 | 0 | [660, 2322] |
| 0 | development_evaluation | 2982 | 2015 | commute_over20 | 1971 | 1011 | [959, 1012] |
| 0 | development_evaluation | 2982 | 2015 | income_binary | 2982 | 0 | [2439, 543] |
| 0 | development_evaluation | 2982 | 2015 | civilian_at_work | 2982 | 0 | [972, 2010] |
| 0 | development_evaluation | 2982 | 2015 | public_coverage | 2982 | 0 | [2199, 783] |
| 0 | development_evaluation | 2982 | 2015 | SEX | 2982 | 0 | [1534, 1448] |
| 0 | development_evaluation | 2982 | 2015 | RAC1P | 2982 | 0 | [1685, 130, 22, 0, 9, 486, 7, 465, 178] |
| 1 | representation_fit | 10428 | 7051 | same_residence | 10428 | 0 | [2371, 8057] |
| 1 | representation_fit | 10428 | 7051 | commute_over20 | 6840 | 3588 | [3474, 3366] |
| 1 | representation_fit | 10428 | 7051 | income_binary | 10428 | 0 | [8509, 1919] |
| 1 | representation_fit | 10428 | 7051 | civilian_at_work | 10428 | 0 | [3454, 6974] |
| 1 | representation_fit | 10428 | 7051 | public_coverage | 10428 | 0 | [7809, 2619] |
| 1 | representation_fit | 10428 | 7051 | SEX | 10428 | 0 | [5351, 5077] |
| 1 | representation_fit | 10428 | 7051 | RAC1P | 10428 | 0 | [5993, 529, 64, 1, 25, 1626, 35, 1536, 619] |
| 1 | source_validation | 3010 | 2015 | same_residence | 3010 | 0 | [744, 2266] |
| 1 | source_validation | 3010 | 2015 | commute_over20 | 2024 | 986 | [1039, 985] |
| 1 | source_validation | 3010 | 2015 | income_binary | 3010 | 0 | [2444, 566] |
| 1 | source_validation | 3010 | 2015 | civilian_at_work | 3010 | 0 | [957, 2053] |
| 1 | source_validation | 3010 | 2015 | public_coverage | 3010 | 0 | [2283, 727] |
| 1 | source_validation | 3010 | 2015 | SEX | 3010 | 0 | [1554, 1456] |
| 1 | source_validation | 3010 | 2015 | RAC1P | 3010 | 0 | [1684, 170, 32, 0, 7, 479, 18, 456, 164] |
| 1 | downstream_fit | 4521 | 3022 | same_residence | 4521 | 0 | [1086, 3435] |
| 1 | downstream_fit | 4521 | 3022 | commute_over20 | 3023 | 1498 | [1459, 1564] |
| 1 | downstream_fit | 4521 | 3022 | income_binary | 4521 | 0 | [3656, 865] |
| 1 | downstream_fit | 4521 | 3022 | civilian_at_work | 4521 | 0 | [1426, 3095] |
| 1 | downstream_fit | 4521 | 3022 | public_coverage | 4521 | 0 | [3458, 1063] |
| 1 | downstream_fit | 4521 | 3022 | SEX | 4521 | 0 | [2298, 2223] |
| 1 | downstream_fit | 4521 | 3022 | RAC1P | 4521 | 0 | [2536, 227, 29, 0, 9, 788, 18, 652, 262] |
| 1 | downstream_validation | 3041 | 2015 | same_residence | 3041 | 0 | [670, 2371] |
| 1 | downstream_validation | 3041 | 2015 | commute_over20 | 2018 | 1023 | [995, 1023] |
| 1 | downstream_validation | 3041 | 2015 | income_binary | 3041 | 0 | [2488, 553] |
| 1 | downstream_validation | 3041 | 2015 | civilian_at_work | 3041 | 0 | [993, 2048] |
| 1 | downstream_validation | 3041 | 2015 | public_coverage | 3041 | 0 | [2274, 767] |
| 1 | downstream_validation | 3041 | 2015 | SEX | 3041 | 0 | [1557, 1484] |
| 1 | downstream_validation | 3041 | 2015 | RAC1P | 3041 | 0 | [1755, 145, 23, 0, 14, 477, 25, 453, 149] |
| 1 | attacker_fit | 3021 | 2015 | same_residence | 3021 | 0 | [729, 2292] |
| 1 | attacker_fit | 3021 | 2015 | commute_over20 | 2031 | 990 | [968, 1063] |
| 1 | attacker_fit | 3021 | 2015 | income_binary | 3021 | 0 | [2455, 566] |
| 1 | attacker_fit | 3021 | 2015 | civilian_at_work | 3021 | 0 | [942, 2079] |
| 1 | attacker_fit | 3021 | 2015 | public_coverage | 3021 | 0 | [2289, 732] |
| 1 | attacker_fit | 3021 | 2015 | SEX | 3021 | 0 | [1574, 1447] |
| 1 | attacker_fit | 3021 | 2015 | RAC1P | 3021 | 0 | [1694, 181, 30, 0, 2, 476, 11, 469, 158] |
| 1 | attacker_validation | 2964 | 2014 | same_residence | 2964 | 0 | [694, 2270] |
| 1 | attacker_validation | 2964 | 2014 | commute_over20 | 1964 | 1000 | [981, 983] |
| 1 | attacker_validation | 2964 | 2014 | income_binary | 2964 | 0 | [2405, 559] |
| 1 | attacker_validation | 2964 | 2014 | civilian_at_work | 2964 | 0 | [958, 2006] |
| 1 | attacker_validation | 2964 | 2014 | public_coverage | 2964 | 0 | [2295, 669] |
| 1 | attacker_validation | 2964 | 2014 | SEX | 2964 | 0 | [1548, 1416] |
| 1 | attacker_validation | 2964 | 2014 | RAC1P | 2964 | 0 | [1713, 163, 25, 0, 9, 407, 9, 455, 183] |
| 1 | development_evaluation | 3015 | 2015 | same_residence | 3015 | 0 | [694, 2321] |
| 1 | development_evaluation | 3015 | 2015 | commute_over20 | 2022 | 993 | [997, 1025] |
| 1 | development_evaluation | 3015 | 2015 | income_binary | 3015 | 0 | [2449, 566] |
| 1 | development_evaluation | 3015 | 2015 | civilian_at_work | 3015 | 0 | [962, 2053] |
| 1 | development_evaluation | 3015 | 2015 | public_coverage | 3015 | 0 | [2275, 740] |
| 1 | development_evaluation | 3015 | 2015 | SEX | 3015 | 0 | [1579, 1436] |
| 1 | development_evaluation | 3015 | 2015 | RAC1P | 3015 | 0 | [1680, 158, 14, 0, 13, 455, 15, 491, 189] |
| 2 | representation_fit | 10551 | 7051 | same_residence | 10551 | 0 | [2343, 8208] |
| 2 | representation_fit | 10551 | 7051 | commute_over20 | 6946 | 3605 | [3426, 3520] |
| 2 | representation_fit | 10551 | 7051 | income_binary | 10551 | 0 | [8579, 1972] |
| 2 | representation_fit | 10551 | 7051 | civilian_at_work | 10551 | 0 | [3452, 7099] |
| 2 | representation_fit | 10551 | 7051 | public_coverage | 10551 | 0 | [7951, 2600] |
| 2 | representation_fit | 10551 | 7051 | SEX | 10551 | 0 | [5458, 5093] |
| 2 | representation_fit | 10551 | 7051 | RAC1P | 10551 | 0 | [5990, 581, 78, 0, 28, 1672, 58, 1562, 582] |
| 2 | source_validation | 2979 | 2015 | same_residence | 2979 | 0 | [740, 2239] |
| 2 | source_validation | 2979 | 2015 | commute_over20 | 1981 | 998 | [990, 991] |
| 2 | source_validation | 2979 | 2015 | income_binary | 2979 | 0 | [2393, 586] |
| 2 | source_validation | 2979 | 2015 | civilian_at_work | 2979 | 0 | [951, 2028] |
| 2 | source_validation | 2979 | 2015 | public_coverage | 2979 | 0 | [2279, 700] |
| 2 | source_validation | 2979 | 2015 | SEX | 2979 | 0 | [1493, 1486] |
| 2 | source_validation | 2979 | 2015 | RAC1P | 2979 | 0 | [1722, 147, 15, 0, 10, 431, 12, 465, 177] |
| 2 | downstream_fit | 4464 | 3022 | same_residence | 4464 | 0 | [1059, 3405] |
| 2 | downstream_fit | 4464 | 3022 | commute_over20 | 2953 | 1511 | [1505, 1448] |
| 2 | downstream_fit | 4464 | 3022 | income_binary | 4464 | 0 | [3666, 798] |
| 2 | downstream_fit | 4464 | 3022 | civilian_at_work | 4464 | 0 | [1461, 3003] |
| 2 | downstream_fit | 4464 | 3022 | public_coverage | 4464 | 0 | [3362, 1102] |
| 2 | downstream_fit | 4464 | 3022 | SEX | 4464 | 0 | [2278, 2186] |
| 2 | downstream_fit | 4464 | 3022 | RAC1P | 4464 | 0 | [2472, 254, 30, 0, 11, 712, 15, 687, 283] |
| 2 | downstream_validation | 3024 | 2015 | same_residence | 3024 | 0 | [737, 2287] |
| 2 | downstream_validation | 3024 | 2015 | commute_over20 | 2072 | 952 | [1030, 1042] |
| 2 | downstream_validation | 3024 | 2015 | income_binary | 3024 | 0 | [2447, 577] |
| 2 | downstream_validation | 3024 | 2015 | civilian_at_work | 3024 | 0 | [915, 2109] |
| 2 | downstream_validation | 3024 | 2015 | public_coverage | 3024 | 0 | [2313, 711] |
| 2 | downstream_validation | 3024 | 2015 | SEX | 3024 | 0 | [1603, 1421] |
| 2 | downstream_validation | 3024 | 2015 | RAC1P | 3024 | 0 | [1731, 129, 27, 0, 5, 468, 22, 464, 178] |
| 2 | attacker_fit | 2941 | 2015 | same_residence | 2941 | 0 | [735, 2206] |
| 2 | attacker_fit | 2941 | 2015 | commute_over20 | 1998 | 943 | [1011, 987] |
| 2 | attacker_fit | 2941 | 2015 | income_binary | 2941 | 0 | [2371, 570] |
| 2 | attacker_fit | 2941 | 2015 | civilian_at_work | 2941 | 0 | [909, 2032] |
| 2 | attacker_fit | 2941 | 2015 | public_coverage | 2941 | 0 | [2222, 719] |
| 2 | attacker_fit | 2941 | 2015 | SEX | 2941 | 0 | [1508, 1433] |
| 2 | attacker_fit | 2941 | 2015 | RAC1P | 2941 | 0 | [1670, 165, 22, 0, 5, 450, 9, 450, 170] |
| 2 | attacker_validation | 3062 | 2014 | same_residence | 3062 | 0 | [688, 2374] |
| 2 | attacker_validation | 3062 | 2014 | commute_over20 | 2017 | 1045 | [1013, 1004] |
| 2 | attacker_validation | 3062 | 2014 | income_binary | 3062 | 0 | [2539, 523] |
| 2 | attacker_validation | 3062 | 2014 | civilian_at_work | 3062 | 0 | [1008, 2054] |
| 2 | attacker_validation | 3062 | 2014 | public_coverage | 3062 | 0 | [2325, 737] |
| 2 | attacker_validation | 3062 | 2014 | SEX | 3062 | 0 | [1611, 1451] |
| 2 | attacker_validation | 3062 | 2014 | RAC1P | 3062 | 0 | [1747, 158, 25, 0, 11, 496, 9, 454, 162] |
| 2 | development_evaluation | 2979 | 2015 | same_residence | 2979 | 0 | [686, 2293] |
| 2 | development_evaluation | 2979 | 2015 | commute_over20 | 1955 | 1024 | [938, 1017] |
| 2 | development_evaluation | 2979 | 2015 | income_binary | 2979 | 0 | [2411, 568] |
| 2 | development_evaluation | 2979 | 2015 | civilian_at_work | 2979 | 0 | [996, 1983] |
| 2 | development_evaluation | 2979 | 2015 | public_coverage | 2979 | 0 | [2231, 748] |
| 2 | development_evaluation | 2979 | 2015 | SEX | 2979 | 0 | [1510, 1469] |
| 2 | development_evaluation | 2979 | 2015 | RAC1P | 2979 | 0 | [1723, 139, 20, 1, 9, 479, 6, 430, 172] |

## Reused exposed controls: every family and restart

The primary exposed control is selected by saved validation log loss. Every failed configuration remains visible. Positive recall for every class is necessary for a complete attribute assessment; unsupported classes cannot pass.

| Seed | Split | Target | Candidate | Primary | LL | Accuracy | Balanced accuracy | AUROC / macro | Fit support | Eval support | Failing codes | Recall by code |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | validation | SEX | logistic | False | 0.001185 | 1.000000 | 1.000000 | 1.000000 | [1512, 1481] | [1541, 1444] | [] | 1.000000, 1.000000 |
| 0 | development evaluation | SEX | logistic | False | 0.001185 | 1.000000 | 1.000000 | 1.000000 | [1512, 1481] | [1534, 1448] | [] | 1.000000, 1.000000 |
| 0 | validation | SEX | mlp_0 | True | 0.000026 | 1.000000 | 1.000000 | 1.000000 | [1512, 1481] | [1541, 1444] | [] | 1.000000, 1.000000 |
| 0 | development evaluation | SEX | mlp_0 | True | 0.000026 | 1.000000 | 1.000000 | 1.000000 | [1512, 1481] | [1534, 1448] | [] | 1.000000, 1.000000 |
| 0 | validation | SEX | mlp_1 | False | 0.000062 | 1.000000 | 1.000000 | 1.000000 | [1512, 1481] | [1541, 1444] | [] | 1.000000, 1.000000 |
| 0 | development evaluation | SEX | mlp_1 | False | 0.000062 | 1.000000 | 1.000000 | 1.000000 | [1512, 1481] | [1534, 1448] | [] | 1.000000, 1.000000 |
| 0 | validation | SEX | hist_gb_20 | False | 0.000101 | 1.000000 | 1.000000 | 1.000000 | [1512, 1481] | [1541, 1444] | [] | 1.000000, 1.000000 |
| 0 | development evaluation | SEX | hist_gb_20 | False | 0.000101 | 1.000000 | 1.000000 | 1.000000 | [1512, 1481] | [1534, 1448] | [] | 1.000000, 1.000000 |
| 0 | validation | SEX | hist_gb_5 | False | 0.000101 | 1.000000 | 1.000000 | 1.000000 | [1512, 1481] | [1541, 1444] | [] | 1.000000, 1.000000 |
| 0 | development evaluation | SEX | hist_gb_5 | False | 0.000101 | 1.000000 | 1.000000 | 1.000000 | [1512, 1481] | [1534, 1448] | [] | 1.000000, 1.000000 |
| 0 | validation | RAC1P | logistic | False | 0.001294 | 1.000000 | undefined | undefined | [1704, 173, 17, 0, 4, 461, 22, 424, 188] | [1737, 144, 18, 0, 11, 468, 16, 409, 182] | [4] | 1.000000, 1.000000, 1.000000, undefined, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 0 | development evaluation | RAC1P | logistic | False | 0.001302 | 1.000000 | undefined | undefined | [1704, 173, 17, 0, 4, 461, 22, 424, 188] | [1685, 130, 22, 0, 9, 486, 7, 465, 178] | [4] | 1.000000, 1.000000, 1.000000, undefined, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 0 | validation | RAC1P | mlp_0 | True | 0.000143 | 1.000000 | undefined | undefined | [1704, 173, 17, 0, 4, 461, 22, 424, 188] | [1737, 144, 18, 0, 11, 468, 16, 409, 182] | [4] | 1.000000, 1.000000, 1.000000, undefined, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 0 | development evaluation | RAC1P | mlp_0 | True | 0.000143 | 1.000000 | undefined | undefined | [1704, 173, 17, 0, 4, 461, 22, 424, 188] | [1685, 130, 22, 0, 9, 486, 7, 465, 178] | [4] | 1.000000, 1.000000, 1.000000, undefined, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 0 | validation | RAC1P | mlp_1 | False | 0.000187 | 1.000000 | undefined | undefined | [1704, 173, 17, 0, 4, 461, 22, 424, 188] | [1737, 144, 18, 0, 11, 468, 16, 409, 182] | [4] | 1.000000, 1.000000, 1.000000, undefined, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 0 | development evaluation | RAC1P | mlp_1 | False | 0.000185 | 1.000000 | undefined | undefined | [1704, 173, 17, 0, 4, 461, 22, 424, 188] | [1685, 130, 22, 0, 9, 486, 7, 465, 178] | [4] | 1.000000, 1.000000, 1.000000, undefined, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 0 | validation | RAC1P | hist_gb_20 | False | 0.007544 | 0.996315 | undefined | undefined | [1704, 173, 17, 0, 4, 461, 22, 424, 188] | [1737, 144, 18, 0, 11, 468, 16, 409, 182] | [4, 5] | 1.000000, 1.000000, 1.000000, undefined, 0.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 0 | development evaluation | RAC1P | hist_gb_20 | False | 0.006714 | 0.996982 | undefined | undefined | [1704, 173, 17, 0, 4, 461, 22, 424, 188] | [1685, 130, 22, 0, 9, 486, 7, 465, 178] | [4, 5] | 1.000000, 1.000000, 1.000000, undefined, 0.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 0 | validation | RAC1P | hist_gb_5 | False | 0.000145 | 1.000000 | undefined | undefined | [1704, 173, 17, 0, 4, 461, 22, 424, 188] | [1737, 144, 18, 0, 11, 468, 16, 409, 182] | [4] | 1.000000, 1.000000, 1.000000, undefined, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 0 | development evaluation | RAC1P | hist_gb_5 | False | 0.000136 | 1.000000 | undefined | undefined | [1704, 173, 17, 0, 4, 461, 22, 424, 188] | [1685, 130, 22, 0, 9, 486, 7, 465, 178] | [4] | 1.000000, 1.000000, 1.000000, undefined, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 1 | validation | SEX | logistic | False | 0.001175 | 1.000000 | 1.000000 | 1.000000 | [1574, 1447] | [1548, 1416] | [] | 1.000000, 1.000000 |
| 1 | development evaluation | SEX | logistic | False | 0.001175 | 1.000000 | 1.000000 | 1.000000 | [1574, 1447] | [1579, 1436] | [] | 1.000000, 1.000000 |
| 1 | validation | SEX | mlp_0 | True | 0.000006 | 1.000000 | 1.000000 | 1.000000 | [1574, 1447] | [1548, 1416] | [] | 1.000000, 1.000000 |
| 1 | development evaluation | SEX | mlp_0 | True | 0.000006 | 1.000000 | 1.000000 | 1.000000 | [1574, 1447] | [1579, 1436] | [] | 1.000000, 1.000000 |
| 1 | validation | SEX | mlp_1 | False | 0.000037 | 1.000000 | 1.000000 | 1.000000 | [1574, 1447] | [1548, 1416] | [] | 1.000000, 1.000000 |
| 1 | development evaluation | SEX | mlp_1 | False | 0.000037 | 1.000000 | 1.000000 | 1.000000 | [1574, 1447] | [1579, 1436] | [] | 1.000000, 1.000000 |
| 1 | validation | SEX | hist_gb_20 | False | 0.000100 | 1.000000 | 1.000000 | 1.000000 | [1574, 1447] | [1548, 1416] | [] | 1.000000, 1.000000 |
| 1 | development evaluation | SEX | hist_gb_20 | False | 0.000100 | 1.000000 | 1.000000 | 1.000000 | [1574, 1447] | [1579, 1436] | [] | 1.000000, 1.000000 |
| 1 | validation | SEX | hist_gb_5 | False | 0.000100 | 1.000000 | 1.000000 | 1.000000 | [1574, 1447] | [1548, 1416] | [] | 1.000000, 1.000000 |
| 1 | development evaluation | SEX | hist_gb_5 | False | 0.000100 | 1.000000 | 1.000000 | 1.000000 | [1574, 1447] | [1579, 1436] | [] | 1.000000, 1.000000 |
| 1 | validation | RAC1P | logistic | False | 0.001244 | 1.000000 | undefined | undefined | [1694, 181, 30, 0, 2, 476, 11, 469, 158] | [1713, 163, 25, 0, 9, 407, 9, 455, 183] | [4] | 1.000000, 1.000000, 1.000000, undefined, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 1 | development evaluation | RAC1P | logistic | False | 0.001265 | 1.000000 | undefined | undefined | [1694, 181, 30, 0, 2, 476, 11, 469, 158] | [1680, 158, 14, 0, 13, 455, 15, 491, 189] | [4] | 1.000000, 1.000000, 1.000000, undefined, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 1 | validation | RAC1P | mlp_0 | True | 0.000109 | 1.000000 | undefined | undefined | [1694, 181, 30, 0, 2, 476, 11, 469, 158] | [1713, 163, 25, 0, 9, 407, 9, 455, 183] | [4] | 1.000000, 1.000000, 1.000000, undefined, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 1 | development evaluation | RAC1P | mlp_0 | True | 0.000113 | 1.000000 | undefined | undefined | [1694, 181, 30, 0, 2, 476, 11, 469, 158] | [1680, 158, 14, 0, 13, 455, 15, 491, 189] | [4] | 1.000000, 1.000000, 1.000000, undefined, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 1 | validation | RAC1P | mlp_1 | False | 0.000159 | 1.000000 | undefined | undefined | [1694, 181, 30, 0, 2, 476, 11, 469, 158] | [1713, 163, 25, 0, 9, 407, 9, 455, 183] | [4] | 1.000000, 1.000000, 1.000000, undefined, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 1 | development evaluation | RAC1P | mlp_1 | False | 0.000166 | 1.000000 | undefined | undefined | [1694, 181, 30, 0, 2, 476, 11, 469, 158] | [1680, 158, 14, 0, 13, 455, 15, 491, 189] | [4] | 1.000000, 1.000000, 1.000000, undefined, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 1 | validation | RAC1P | hist_gb_20 | False | 0.006398 | 0.996964 | undefined | undefined | [1694, 181, 30, 0, 2, 476, 11, 469, 158] | [1713, 163, 25, 0, 9, 407, 9, 455, 183] | [4, 5] | 1.000000, 1.000000, 1.000000, undefined, 0.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 1 | development evaluation | RAC1P | hist_gb_20 | False | 0.009154 | 0.995688 | undefined | undefined | [1694, 181, 30, 0, 2, 476, 11, 469, 158] | [1680, 158, 14, 0, 13, 455, 15, 491, 189] | [4, 5] | 1.000000, 1.000000, 1.000000, undefined, 0.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 1 | validation | RAC1P | hist_gb_5 | False | 0.000169 | 1.000000 | undefined | undefined | [1694, 181, 30, 0, 2, 476, 11, 469, 158] | [1713, 163, 25, 0, 9, 407, 9, 455, 183] | [4] | 1.000000, 1.000000, 1.000000, undefined, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 1 | development evaluation | RAC1P | hist_gb_5 | False | 0.000202 | 1.000000 | undefined | undefined | [1694, 181, 30, 0, 2, 476, 11, 469, 158] | [1680, 158, 14, 0, 13, 455, 15, 491, 189] | [4] | 1.000000, 1.000000, 1.000000, undefined, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 2 | validation | SEX | logistic | False | 0.001200 | 1.000000 | 1.000000 | 1.000000 | [1508, 1433] | [1611, 1451] | [] | 1.000000, 1.000000 |
| 2 | development evaluation | SEX | logistic | False | 0.001201 | 1.000000 | 1.000000 | 1.000000 | [1508, 1433] | [1510, 1469] | [] | 1.000000, 1.000000 |
| 2 | validation | SEX | mlp_0 | True | 0.000023 | 1.000000 | 1.000000 | 1.000000 | [1508, 1433] | [1611, 1451] | [] | 1.000000, 1.000000 |
| 2 | development evaluation | SEX | mlp_0 | True | 0.000022 | 1.000000 | 1.000000 | 1.000000 | [1508, 1433] | [1510, 1469] | [] | 1.000000, 1.000000 |
| 2 | validation | SEX | mlp_1 | False | 0.000055 | 1.000000 | 1.000000 | 1.000000 | [1508, 1433] | [1611, 1451] | [] | 1.000000, 1.000000 |
| 2 | development evaluation | SEX | mlp_1 | False | 0.000055 | 1.000000 | 1.000000 | 1.000000 | [1508, 1433] | [1510, 1469] | [] | 1.000000, 1.000000 |
| 2 | validation | SEX | hist_gb_20 | False | 0.000102 | 1.000000 | 1.000000 | 1.000000 | [1508, 1433] | [1611, 1451] | [] | 1.000000, 1.000000 |
| 2 | development evaluation | SEX | hist_gb_20 | False | 0.000102 | 1.000000 | 1.000000 | 1.000000 | [1508, 1433] | [1510, 1469] | [] | 1.000000, 1.000000 |
| 2 | validation | SEX | hist_gb_5 | False | 0.000102 | 1.000000 | 1.000000 | 1.000000 | [1508, 1433] | [1611, 1451] | [] | 1.000000, 1.000000 |
| 2 | development evaluation | SEX | hist_gb_5 | False | 0.000102 | 1.000000 | 1.000000 | 1.000000 | [1508, 1433] | [1510, 1469] | [] | 1.000000, 1.000000 |
| 2 | validation | RAC1P | logistic | False | 0.001328 | 1.000000 | undefined | undefined | [1670, 165, 22, 0, 5, 450, 9, 450, 170] | [1747, 158, 25, 0, 11, 496, 9, 454, 162] | [4] | 1.000000, 1.000000, 1.000000, undefined, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 2 | development evaluation | RAC1P | logistic | False | 0.010592 | 0.999664 | 0.888889 | 0.944444 | [1670, 165, 22, 0, 5, 450, 9, 450, 170] | [1723, 139, 20, 1, 9, 479, 6, 430, 172] | [4] | 1.000000, 1.000000, 1.000000, 0.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 2 | validation | RAC1P | mlp_0 | True | 0.000093 | 1.000000 | undefined | undefined | [1670, 165, 22, 0, 5, 450, 9, 450, 170] | [1747, 158, 25, 0, 11, 496, 9, 454, 162] | [4] | 1.000000, 1.000000, 1.000000, undefined, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 2 | development evaluation | RAC1P | mlp_0 | True | 0.001747 | 0.999664 | 0.888889 | 1.000000 | [1670, 165, 22, 0, 5, 450, 9, 450, 170] | [1723, 139, 20, 1, 9, 479, 6, 430, 172] | [4] | 1.000000, 1.000000, 1.000000, 0.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 2 | validation | RAC1P | mlp_1 | False | 0.000136 | 1.000000 | undefined | undefined | [1670, 165, 22, 0, 5, 450, 9, 450, 170] | [1747, 158, 25, 0, 11, 496, 9, 454, 162] | [4] | 1.000000, 1.000000, 1.000000, undefined, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 2 | development evaluation | RAC1P | mlp_1 | False | 0.002071 | 0.999664 | 0.888889 | 1.000000 | [1670, 165, 22, 0, 5, 450, 9, 450, 170] | [1723, 139, 20, 1, 9, 479, 6, 430, 172] | [4] | 1.000000, 1.000000, 1.000000, 0.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 2 | validation | RAC1P | hist_gb_20 | False | 0.005172 | 0.996408 | undefined | undefined | [1670, 165, 22, 0, 5, 450, 9, 450, 170] | [1747, 158, 25, 0, 11, 496, 9, 454, 162] | [4, 5] | 1.000000, 1.000000, 1.000000, undefined, 0.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 2 | development evaluation | RAC1P | hist_gb_20 | False | 0.013433 | 0.996643 | 0.777778 | 0.944127 | [1670, 165, 22, 0, 5, 450, 9, 450, 170] | [1723, 139, 20, 1, 9, 479, 6, 430, 172] | [4, 5] | 1.000000, 1.000000, 1.000000, 0.000000, 0.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 2 | validation | RAC1P | hist_gb_5 | False | 0.000115 | 1.000000 | undefined | undefined | [1670, 165, 22, 0, 5, 450, 9, 450, 170] | [1747, 158, 25, 0, 11, 496, 9, 454, 162] | [4] | 1.000000, 1.000000, 1.000000, undefined, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| 2 | development evaluation | RAC1P | hist_gb_5 | False | 0.009382 | 0.999664 | 0.888889 | 0.944444 | [1670, 165, 22, 0, 5, 450, 9, 450, 170] | [1723, 139, 20, 1, 9, 479, 6, 430, 172] | [4] | 1.000000, 1.000000, 1.000000, 0.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 |

## Primary and independent-only coverage checks

Fit support is from the selected candidate’s actual fitting rows. Coverage also requires every category in attacker validation and evaluated rows plus positive primary exposed-control recall. Coverage limits apply even when the scalar log-loss inequality is satisfied.

| Seed | Split | Release | Selector | Target | Candidate | Complete | Limitations (Census codes) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | validation | Original PCA | selected | SEX | mlp_0 | True | {} |
| 0 | validation | Original PCA | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | validation | PCA + LEACE | selected | SEX | mlp_1 | True | {} |
| 0 | validation | PCA + LEACE | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | validation | Rich neural bank | selected | SEX | mlp_1 | True | {} |
| 0 | validation | Rich neural bank | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | validation | Neural bank + LEACE | selected | SEX | mlp_1 | True | {} |
| 0 | validation | Neural bank + LEACE | selected | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | validation | Rich tree bank | selected | SEX | mlp_0 | True | {} |
| 0 | validation | Rich tree bank | selected | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | validation | Tree bank + LEACE | selected | SEX | mlp_0 | True | {} |
| 0 | validation | Tree bank + LEACE | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | validation | C task-only bottleneck | selected | SEX | mlp_1 | True | {} |
| 0 | validation | C task-only bottleneck | selected | RAC1P | catchup | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | validation | D protected bottleneck | selected | SEX | mlp_0 | True | {} |
| 0 | validation | D protected bottleneck | selected | RAC1P | catchup | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | validation | Fitting prior | selected | SEX | prior | True | {} |
| 0 | validation | Fitting prior | selected | RAC1P | prior | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | validation | Original PCA | independent_selected | SEX | mlp_0 | True | {} |
| 0 | validation | Original PCA | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | validation | PCA + LEACE | independent_selected | SEX | mlp_1 | True | {} |
| 0 | validation | PCA + LEACE | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | validation | Rich neural bank | independent_selected | SEX | mlp_1 | True | {} |
| 0 | validation | Rich neural bank | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | validation | Neural bank + LEACE | independent_selected | SEX | mlp_1 | True | {} |
| 0 | validation | Neural bank + LEACE | independent_selected | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | validation | Rich tree bank | independent_selected | SEX | mlp_0 | True | {} |
| 0 | validation | Rich tree bank | independent_selected | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | validation | Tree bank + LEACE | independent_selected | SEX | mlp_0 | True | {} |
| 0 | validation | Tree bank + LEACE | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | validation | C task-only bottleneck | independent_selected | SEX | mlp_1 | True | {} |
| 0 | validation | C task-only bottleneck | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | validation | D protected bottleneck | independent_selected | SEX | mlp_0 | True | {} |
| 0 | validation | D protected bottleneck | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | validation | Fitting prior | independent_selected | SEX | prior | True | {} |
| 0 | validation | Fitting prior | independent_selected | RAC1P | prior | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | development evaluation | Original PCA | selected | SEX | mlp_0 | True | {} |
| 0 | development evaluation | Original PCA | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | development evaluation | PCA + LEACE | selected | SEX | mlp_1 | True | {} |
| 0 | development evaluation | PCA + LEACE | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | development evaluation | Rich neural bank | selected | SEX | mlp_1 | True | {} |
| 0 | development evaluation | Rich neural bank | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | development evaluation | Neural bank + LEACE | selected | SEX | mlp_1 | True | {} |
| 0 | development evaluation | Neural bank + LEACE | selected | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | development evaluation | Rich tree bank | selected | SEX | mlp_0 | True | {} |
| 0 | development evaluation | Rich tree bank | selected | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | development evaluation | Tree bank + LEACE | selected | SEX | mlp_0 | True | {} |
| 0 | development evaluation | Tree bank + LEACE | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | development evaluation | C task-only bottleneck | selected | SEX | mlp_1 | True | {} |
| 0 | development evaluation | C task-only bottleneck | selected | RAC1P | catchup | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | development evaluation | D protected bottleneck | selected | SEX | mlp_0 | True | {} |
| 0 | development evaluation | D protected bottleneck | selected | RAC1P | catchup | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | development evaluation | Fitting prior | selected | SEX | prior | True | {} |
| 0 | development evaluation | Fitting prior | selected | RAC1P | prior | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | development evaluation | Original PCA | independent_selected | SEX | mlp_0 | True | {} |
| 0 | development evaluation | Original PCA | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | development evaluation | PCA + LEACE | independent_selected | SEX | mlp_1 | True | {} |
| 0 | development evaluation | PCA + LEACE | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | development evaluation | Rich neural bank | independent_selected | SEX | mlp_1 | True | {} |
| 0 | development evaluation | Rich neural bank | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | development evaluation | Neural bank + LEACE | independent_selected | SEX | mlp_1 | True | {} |
| 0 | development evaluation | Neural bank + LEACE | independent_selected | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | development evaluation | Rich tree bank | independent_selected | SEX | mlp_0 | True | {} |
| 0 | development evaluation | Rich tree bank | independent_selected | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | development evaluation | Tree bank + LEACE | independent_selected | SEX | mlp_0 | True | {} |
| 0 | development evaluation | Tree bank + LEACE | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | development evaluation | C task-only bottleneck | independent_selected | SEX | mlp_1 | True | {} |
| 0 | development evaluation | C task-only bottleneck | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | development evaluation | D protected bottleneck | independent_selected | SEX | mlp_0 | True | {} |
| 0 | development evaluation | D protected bottleneck | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 0 | development evaluation | Fitting prior | independent_selected | SEX | prior | True | {} |
| 0 | development evaluation | Fitting prior | independent_selected | RAC1P | prior | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | validation | Original PCA | selected | SEX | mlp_0 | True | {} |
| 1 | validation | Original PCA | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | validation | PCA + LEACE | selected | SEX | mlp_1 | True | {} |
| 1 | validation | PCA + LEACE | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | validation | Rich neural bank | selected | SEX | mlp_0 | True | {} |
| 1 | validation | Rich neural bank | selected | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | validation | Neural bank + LEACE | selected | SEX | mlp_1 | True | {} |
| 1 | validation | Neural bank + LEACE | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | validation | Rich tree bank | selected | SEX | mlp_1 | True | {} |
| 1 | validation | Rich tree bank | selected | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | validation | Tree bank + LEACE | selected | SEX | mlp_1 | True | {} |
| 1 | validation | Tree bank + LEACE | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | validation | C task-only bottleneck | selected | SEX | mlp_0 | True | {} |
| 1 | validation | C task-only bottleneck | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | validation | D protected bottleneck | selected | SEX | catchup | True | {} |
| 1 | validation | D protected bottleneck | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | validation | Fitting prior | selected | SEX | prior | True | {} |
| 1 | validation | Fitting prior | selected | RAC1P | prior | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | validation | Original PCA | independent_selected | SEX | mlp_0 | True | {} |
| 1 | validation | Original PCA | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | validation | PCA + LEACE | independent_selected | SEX | mlp_1 | True | {} |
| 1 | validation | PCA + LEACE | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | validation | Rich neural bank | independent_selected | SEX | mlp_0 | True | {} |
| 1 | validation | Rich neural bank | independent_selected | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | validation | Neural bank + LEACE | independent_selected | SEX | mlp_1 | True | {} |
| 1 | validation | Neural bank + LEACE | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | validation | Rich tree bank | independent_selected | SEX | mlp_1 | True | {} |
| 1 | validation | Rich tree bank | independent_selected | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | validation | Tree bank + LEACE | independent_selected | SEX | mlp_1 | True | {} |
| 1 | validation | Tree bank + LEACE | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | validation | C task-only bottleneck | independent_selected | SEX | mlp_0 | True | {} |
| 1 | validation | C task-only bottleneck | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | validation | D protected bottleneck | independent_selected | SEX | mlp_0 | True | {} |
| 1 | validation | D protected bottleneck | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | validation | Fitting prior | independent_selected | SEX | prior | True | {} |
| 1 | validation | Fitting prior | independent_selected | RAC1P | prior | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | development evaluation | Original PCA | selected | SEX | mlp_0 | True | {} |
| 1 | development evaluation | Original PCA | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | development evaluation | PCA + LEACE | selected | SEX | mlp_1 | True | {} |
| 1 | development evaluation | PCA + LEACE | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | development evaluation | Rich neural bank | selected | SEX | mlp_0 | True | {} |
| 1 | development evaluation | Rich neural bank | selected | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | development evaluation | Neural bank + LEACE | selected | SEX | mlp_1 | True | {} |
| 1 | development evaluation | Neural bank + LEACE | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | development evaluation | Rich tree bank | selected | SEX | mlp_1 | True | {} |
| 1 | development evaluation | Rich tree bank | selected | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | development evaluation | Tree bank + LEACE | selected | SEX | mlp_1 | True | {} |
| 1 | development evaluation | Tree bank + LEACE | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | development evaluation | C task-only bottleneck | selected | SEX | mlp_0 | True | {} |
| 1 | development evaluation | C task-only bottleneck | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | development evaluation | D protected bottleneck | selected | SEX | catchup | True | {} |
| 1 | development evaluation | D protected bottleneck | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | development evaluation | Fitting prior | selected | SEX | prior | True | {} |
| 1 | development evaluation | Fitting prior | selected | RAC1P | prior | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | development evaluation | Original PCA | independent_selected | SEX | mlp_0 | True | {} |
| 1 | development evaluation | Original PCA | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | development evaluation | PCA + LEACE | independent_selected | SEX | mlp_1 | True | {} |
| 1 | development evaluation | PCA + LEACE | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | development evaluation | Rich neural bank | independent_selected | SEX | mlp_0 | True | {} |
| 1 | development evaluation | Rich neural bank | independent_selected | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | development evaluation | Neural bank + LEACE | independent_selected | SEX | mlp_1 | True | {} |
| 1 | development evaluation | Neural bank + LEACE | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | development evaluation | Rich tree bank | independent_selected | SEX | mlp_1 | True | {} |
| 1 | development evaluation | Rich tree bank | independent_selected | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | development evaluation | Tree bank + LEACE | independent_selected | SEX | mlp_1 | True | {} |
| 1 | development evaluation | Tree bank + LEACE | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | development evaluation | C task-only bottleneck | independent_selected | SEX | mlp_0 | True | {} |
| 1 | development evaluation | C task-only bottleneck | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | development evaluation | D protected bottleneck | independent_selected | SEX | mlp_0 | True | {} |
| 1 | development evaluation | D protected bottleneck | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 1 | development evaluation | Fitting prior | independent_selected | SEX | prior | True | {} |
| 1 | development evaluation | Fitting prior | independent_selected | RAC1P | prior | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 2 | validation | Original PCA | selected | SEX | mlp_1 | True | {} |
| 2 | validation | Original PCA | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 2 | validation | PCA + LEACE | selected | SEX | mlp_1 | True | {} |
| 2 | validation | PCA + LEACE | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 2 | validation | Rich neural bank | selected | SEX | mlp_1 | True | {} |
| 2 | validation | Rich neural bank | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 2 | validation | Neural bank + LEACE | selected | SEX | mlp_0 | True | {} |
| 2 | validation | Neural bank + LEACE | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 2 | validation | Rich tree bank | selected | SEX | mlp_1 | True | {} |
| 2 | validation | Rich tree bank | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 2 | validation | Tree bank + LEACE | selected | SEX | mlp_0 | True | {} |
| 2 | validation | Tree bank + LEACE | selected | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 2 | validation | C task-only bottleneck | selected | SEX | mlp_1 | True | {} |
| 2 | validation | C task-only bottleneck | selected | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 2 | validation | D protected bottleneck | selected | SEX | mlp_1 | True | {} |
| 2 | validation | D protected bottleneck | selected | RAC1P | catchup | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 2 | validation | Fitting prior | selected | SEX | prior | True | {} |
| 2 | validation | Fitting prior | selected | RAC1P | prior | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 2 | validation | Original PCA | independent_selected | SEX | mlp_1 | True | {} |
| 2 | validation | Original PCA | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 2 | validation | PCA + LEACE | independent_selected | SEX | mlp_1 | True | {} |
| 2 | validation | PCA + LEACE | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 2 | validation | Rich neural bank | independent_selected | SEX | mlp_1 | True | {} |
| 2 | validation | Rich neural bank | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 2 | validation | Neural bank + LEACE | independent_selected | SEX | mlp_0 | True | {} |
| 2 | validation | Neural bank + LEACE | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 2 | validation | Rich tree bank | independent_selected | SEX | mlp_1 | True | {} |
| 2 | validation | Rich tree bank | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 2 | validation | Tree bank + LEACE | independent_selected | SEX | mlp_0 | True | {} |
| 2 | validation | Tree bank + LEACE | independent_selected | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 2 | validation | C task-only bottleneck | independent_selected | SEX | mlp_1 | True | {} |
| 2 | validation | C task-only bottleneck | independent_selected | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 2 | validation | D protected bottleneck | independent_selected | SEX | mlp_1 | True | {} |
| 2 | validation | D protected bottleneck | independent_selected | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 2 | validation | Fitting prior | independent_selected | SEX | prior | True | {} |
| 2 | validation | Fitting prior | independent_selected | RAC1P | prior | False | {"fit":[4],"validation":[4],"evaluation":[4],"exposed_recall":[4]} |
| 2 | development evaluation | Original PCA | selected | SEX | mlp_1 | True | {} |
| 2 | development evaluation | Original PCA | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"exposed_recall":[4]} |
| 2 | development evaluation | PCA + LEACE | selected | SEX | mlp_1 | True | {} |
| 2 | development evaluation | PCA + LEACE | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"exposed_recall":[4]} |
| 2 | development evaluation | Rich neural bank | selected | SEX | mlp_1 | True | {} |
| 2 | development evaluation | Rich neural bank | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"exposed_recall":[4]} |
| 2 | development evaluation | Neural bank + LEACE | selected | SEX | mlp_0 | True | {} |
| 2 | development evaluation | Neural bank + LEACE | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"exposed_recall":[4]} |
| 2 | development evaluation | Rich tree bank | selected | SEX | mlp_1 | True | {} |
| 2 | development evaluation | Rich tree bank | selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"exposed_recall":[4]} |
| 2 | development evaluation | Tree bank + LEACE | selected | SEX | mlp_0 | True | {} |
| 2 | development evaluation | Tree bank + LEACE | selected | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"exposed_recall":[4]} |
| 2 | development evaluation | C task-only bottleneck | selected | SEX | mlp_1 | True | {} |
| 2 | development evaluation | C task-only bottleneck | selected | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"exposed_recall":[4]} |
| 2 | development evaluation | D protected bottleneck | selected | SEX | mlp_1 | True | {} |
| 2 | development evaluation | D protected bottleneck | selected | RAC1P | catchup | False | {"fit":[4],"validation":[4],"exposed_recall":[4]} |
| 2 | development evaluation | Fitting prior | selected | SEX | prior | True | {} |
| 2 | development evaluation | Fitting prior | selected | RAC1P | prior | False | {"fit":[4],"validation":[4],"exposed_recall":[4]} |
| 2 | development evaluation | Original PCA | independent_selected | SEX | mlp_1 | True | {} |
| 2 | development evaluation | Original PCA | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"exposed_recall":[4]} |
| 2 | development evaluation | PCA + LEACE | independent_selected | SEX | mlp_1 | True | {} |
| 2 | development evaluation | PCA + LEACE | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"exposed_recall":[4]} |
| 2 | development evaluation | Rich neural bank | independent_selected | SEX | mlp_1 | True | {} |
| 2 | development evaluation | Rich neural bank | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"exposed_recall":[4]} |
| 2 | development evaluation | Neural bank + LEACE | independent_selected | SEX | mlp_0 | True | {} |
| 2 | development evaluation | Neural bank + LEACE | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"exposed_recall":[4]} |
| 2 | development evaluation | Rich tree bank | independent_selected | SEX | mlp_1 | True | {} |
| 2 | development evaluation | Rich tree bank | independent_selected | RAC1P | mlp_1 | False | {"fit":[4],"validation":[4],"exposed_recall":[4]} |
| 2 | development evaluation | Tree bank + LEACE | independent_selected | SEX | mlp_0 | True | {} |
| 2 | development evaluation | Tree bank + LEACE | independent_selected | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"exposed_recall":[4]} |
| 2 | development evaluation | C task-only bottleneck | independent_selected | SEX | mlp_1 | True | {} |
| 2 | development evaluation | C task-only bottleneck | independent_selected | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"exposed_recall":[4]} |
| 2 | development evaluation | D protected bottleneck | independent_selected | SEX | mlp_1 | True | {} |
| 2 | development evaluation | D protected bottleneck | independent_selected | RAC1P | mlp_0 | False | {"fit":[4],"validation":[4],"exposed_recall":[4]} |
| 2 | development evaluation | Fitting prior | independent_selected | SEX | prior | True | {} |
| 2 | development evaluation | Fitting prior | independent_selected | RAC1P | prior | False | {"fit":[4],"validation":[4],"exposed_recall":[4]} |

Training and catch-up support are recorded in the associated fitting metadata and training evidence; their history is distinct from the fresh attacker-fit pool. Missing race categories are never merged, resampled into a new split, or represented as a privacy pass.

## Representation-fitting attribute support

| Seed | Target | Support | Missing | Schema complete |
| --- | --- | --- | --- | --- |
| 0 | SEX | [5391, 5122] | 0 | True |
| 0 | RAC1P | [5989, 555, 90, 1, 32, 1674, 49, 1527, 596] | 0 | True |
| 1 | SEX | [5351, 5077] | 0 | True |
| 1 | RAC1P | [5993, 529, 64, 1, 25, 1626, 35, 1536, 619] | 0 | True |
| 2 | SEX | [5458, 5093] | 0 | True |
| 2 | RAC1P | [5990, 581, 78, 0, 28, 1672, 58, 1562, 582] | 0 | False |

No training-category support gap is repaired by changing the cohort or schema. All policy criteria still require adequate fixed audit-pool and exposed-control coverage.
