# Planning-relevant uncertainty from the published 2018 development inference

Sources:
- `results/pcrl_adaptive_release_v1/INFERENCE.json`: schema `pcrl-adaptive-outer-inference-v1`, lock SHA `4f894f98…`.
- `ENDPOINT_TABLE.json`: 74 rows, `outer_aggregate_crosscheck = PASS_ALL_THREE_ANCHORS`.

Only aggregate results are used. These are development numbers on repeatedly used 2018 data. They are conditional on the fitted, selected objects.

## How the intervals are built

- **Primary family**: 70 endpoints, two-sided Bonferroni at α = .05, giving z = norm.isf(.05/140) = **3.384**.
  - U slot: 30 endpoints = 3 comparator groups (D17, TaskOnly, historical_Q) × 5 roles × 2 weightings.
  - P slot: 40 endpoints = 4 groups (plus GRADIENT_001).
- **H-capability family**: 4 endpoints, z = **2.498**, kept separate.
- **Interval**: estimate ± z·SE_boot, where SE_boot comes from 10,000 common household multinomial draws (seed 20260923, 0 rejected draws). The half-width (HW) equals z·SE.
- **Outcome**: U passed 2/30 and P passed 5/40. Of those 7 passes, 6 are the exact-zero "identical" TaskOnly contrasts (est = SE = 0).
  The only non-trivial pass is P vs historical_Q on A/SEX, U weighting.

## Key endpoints vs D17 (est; HW at z = 3.384; SE)

| Slot / clause | Weighting | Estimate | HW | SE | Threshold |
|---|---|---|---|---|---|
| U task (CE_cand − CE_D17) | U | +0.001450 | 0.004513 | 0.001334 | ≤ −0.003 |
| U task | PWGTP | +0.000914 | 0.005269 | 0.001557 | ≤ −0.003 |
| U AB/SEX guard | U | +0.000453 | 0.006820 | 0.002015 | ≤ +0.001 |
| U AB/SEX guard | PWGTP | −0.003082 | 0.007743 | 0.002288 | ≤ +0.001 |
| U A/SEX guard | U / PWGTP | +0.001102 / +0.001326 | 0.003682 / 0.003931 | 0.00109 / 0.00116 | ≤ +0.001 |
| U A/RAC1P guard | U / PWGTP | +0.000462 / +0.000518 | 0.003109 / 0.003993 | 0.00092 / 0.00118 | ≤ +0.001 |
| U AB/RAC1P guard | U / PWGTP | +0.001474 / +0.002083 | 0.004013 / 0.004861 | 0.00119 / 0.00144 | ≤ +0.001 |
| P task | U | +0.009539 | 0.005999 | 0.001773 | ≤ +0.001 |
| P task | PWGTP | +0.009835 | 0.006077 | 0.001796 | ≤ +0.001 |
| P AB/SEX target (recovery difference) | U | −0.003672 | 0.005127 | 0.001515 | ≤ −0.002 |
| P AB/SEX target | PWGTP | −0.003325 | 0.005914 | 0.001748 | ≤ −0.002 |
| P A/SEX guard | U / PWGTP | −0.001277 / +0.000313 | 0.005063 / 0.005704 | 0.00150 / 0.00169 | ≤ +0.001 |
| P A/RAC1P guard | U / PWGTP | +0.000462 / +0.000518 | 0.003109 / 0.003993 | 0.00092 / 0.00118 | ≤ +0.001 |
| P AB/RAC1P guard | U / PWGTP | +0.000781 / +0.000774 | 0.003255 / 0.004126 | 0.00096 / 0.00122 | ≤ +0.001 |
| H capability U cand (CE_cand − CE_H), z = 2.498 | U / PWGTP | −0.03615 / −0.03883 | 0.01351 / 0.01729 | 0.00541 / 0.00692 | ≤ −0.01 |
| H capability P cand, z = 2.498 | U / PWGTP | −0.02806 / −0.02991 | 0.01153 / 0.01485 | 0.00462 / 0.00595 | ≤ −0.01 |

SE range across all non-identical primary endpoints and comparators (min / mean / max, n):

| Role | Min | Mean | Max | n |
|---|---|---|---|---|
| task | 0.00133 | 0.00182 | 0.00224 | 14 |
| A/SEX | 0.00108 | 0.00145 | 0.00189 | 14 |
| A/RAC1P | 0.00070 | 0.00095 | 0.00126 | 10 |
| AB/SEX | 0.00071 | 0.00175 | 0.00280 | 14 |
| AB/RAC1P | 0.00064 | 0.00097 | 0.00144 | 12 |

The PWGTP weighting is consistently about 10-20% noisier than U.
Comparators other than D17 have larger SEs: TaskOnly task SE is about 0.0022 and TaskOnly AB/SEX SE is about 0.0026-0.0028.

## Power arithmetic (normal approximation)

To have 80% power to pass `upper ≤ threshold`, the true difference must be ≤ threshold − (z_m + 0.842)·SE.

| Bonferroni m (z) | U task, t = −.003, SE .00145 | P AB/SEX target, t = −.002, SE .00163 | Guard, t = +.001, SE .00095 (RAC1P) | Guard, SE .0018 (task/SEX) | Guard, SE .0025 (AB/SEX) |
|---|---|---|---|---|---|
| 70 (3.384) | ≤ −0.0091 | ≤ −0.0089 | ≤ −0.0030 | ≤ −0.0066 | ≤ −0.0096 |
| 40 (3.227) | −0.0089 | −0.0086 | −0.0029 | −0.0063 | −0.0092 |
| 20 (3.023) | −0.0086 | −0.0083 | −0.0027 | −0.0060 | −0.0087 |
| 10 (2.807) | −0.0083 | −0.0079 | −0.0025 | −0.0056 | −0.0081 |
| 4 (2.498) | −0.0078 | −0.0074 | −0.0022 | −0.0050 | −0.0073 |
| 1 (1.960) | −0.0071 | −0.0066 | −0.0017 | −0.0040 | −0.0060 |

Planning implications:
1. Cutting the family from 70 to 10 shrinks the required effect by only about 10%. Precision is dominated by the SE, about 0.001-0.0028 nats with roughly 2.4k outer households per anchor. The anchors share households, so averaging three anchors adds little independent information.
2. Under the old rule, every guard is a *non-inferiority upper bound at +0.001*. With the observed SEs, a candidate that is truly *equal* to D17 on a sensitive role passes that guard with probability Φ(0.001/SE − z), which is only about 0.1% to 3% (z·SE ≫ 0.001). A full conjunction of 8 guards is therefore essentially unpassable unless the candidate strictly improves every sensitive role, or the contrast is exactly identical because both sides select the same H-only route. A new study should pre-register a guard margin that is commensurate with the SE (on the order of 3-4×SE), or a separate guard family and correction. It should not reuse +0.001 under the upper-bound rule.
3. SE scales roughly as 1/√(households). Halving the HW needs about 4× the outer households, which 2018 cannot supply.
