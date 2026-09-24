# Encoder capacity: was the richer class actually instantiated?

**Yes, on all three anchors. It was trained and available at every round, and the fixed-bank LP used it only in the privacy-first form.**

- **Source:** `ENCODER_CAPACITY.json`, generated on the study host by `reports capacity --index`.
- **Rows used:** coefficient_split and inner_selection rows only. No inner_check or outer row, and no label value.
- **Tolerance:** 1e-12 total variation.
- **Definitions:**
  - *trained*: a policy differs from D17 on coefficient rows.
  - *used*: a fitted round's law varies within some T32 state.
  - *selected*: the final release differs from D17.
  - *V*: the weighted mean total-variation distance to the person's T32-state mean law.

## 1. Trained: every richer policy differs from D17 inside every T32 state

Fraction of coefficient-split people whose token differs from D17 (unweighted / PWGTP-weighted), and unique households affected:

| Policy | Anchor 0 | Anchor 1 | Anchor 2 |
|---|---|---|---|
| task_only | 0.706 / 0.726, 2,670 hh | 0.535 / 0.524, 2,208 hh | 0.592 / 0.594, 2,454 hh |
| local_priced | 0.862 / 0.873, 3,159 hh | 0.687 / 0.694, 2,714 hh | 0.839 / 0.849, 3,150 hh |
| coalition_priced | 0.840 / 0.849, 3,107 hh | 0.756 / 0.760, 2,913 hh | 0.593 / 0.594, 2,490 hh |
| all_priced_x2 | 0.930 / 0.931, 3,364 hh | 0.739 / 0.750, 2,865 hh | 0.584 / 0.588, 2,452 hh |

- **Within-state variation:** every policy varies within all 32 T32 states on every anchor.
- **Aliases:** none; no policy was an exact alias of another.
- **Price fallback:** all-zero round-0 duals forced the registered fallback price in these groups:
  - anchor 0: all three priced groups;
  - anchor 1: the coalition group;
  - anchor 2: the local group.
- **Oracle limitation:** the full-data paired oracle deviated from D17 far more often than on the 30% smoke, where task_only differed for about 10%. The fixed .002-nat margin does not scale with estimation error. This limitation was registered at M2.

## 2. Used and selected, per unit

"Final status" is the unit's registered selection outcome (M3/M4/M5). The eta column lists the fitted η for rounds 0–6; η is not identified, so V is the operative measure.

| anchor | unit | final status | within-T32 V (W) | TV to D17 (W mean) | stochastic weight | eta by round (0..6) |
|---|---|---|---|---|---|---|
| 0 | NM1_U | SELECTED_ROUND r1 | 0.000 | 0.718 | 0.000 | 0.00,0.00,0.00,0.00,0.00,0.00,0.00 |
| 0 | NM4_U | SELECTED_ROUND r1 | 0.000 | 0.718 | 0.000 | 0.00,0.00,0.00,0.00,0.00,0.00,0.00 |
| 0 | T32_U | SELECTED_ROUND r1 | 0.000 | 0.718 | 0.000 | 0.00,0.00,0.00,0.00,0.00,0.00,0.00 |
| 0 | NM1_P | SELECTED_ROUND r5 | 0.222 | 0.520 | 0.964 | 0.51,0.61,0.65,0.52,0.52,0.70,0.53 |
| 0 | NM4_P | SELECTED_ROUND r5 | 0.520 | 0.673 | 0.869 | 0.49,0.00,0.39,0.29,0.44,0.78,0.00 |
| 0 | T32_P | SELECTED_ROUND r3 | 0.000 | 0.802 | 0.067 | 0.00,0.00,0.00,0.00,0.00,0.00,0.00 |
| 0 | DET_SEL1 | ALL_D17_ASSIGNMENT | 0.000 | 0.000 | 0.000 |  |
| 0 | DET_SEL4 | DETERMINISTIC_ASSIGNMENT | 0.298 | 0.426 | 0.000 |  |
| 0 | RD_TASK | SELECTED_ROUND r0 | 0.000 | 0.000 | 0.000 | -,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,- |
| 0 | RD_PRIV | WITNESS_SELECTED | 0.000 | 0.000 | 0.000 | -,-,-,- |
| 0 | ADV_B1 | WITNESS_SELECTED | 0.000 | 0.000 | 0.000 |  |
| 0 | ADV_B2 | WITNESS_SELECTED | 0.000 | 0.000 | 0.000 |  |
| 0 | ADV_B1_P | WITNESS_FALLBACK | 0.000 | 0.000 | 0.000 |  |
| 0 | ADV_B2_P | WITNESS_FALLBACK | 0.000 | 0.000 | 0.000 |  |
| 1 | NM1_U | WITNESS_FALLBACK | 0.000 | 0.000 | 0.000 | 0.00,0.00,0.00,0.00,0.00,0.00,0.00 |
| 1 | NM4_U | WITNESS_FALLBACK | 0.000 | 0.000 | 0.000 | 0.00,0.00,0.00,0.00,0.00,0.00,0.00 |
| 1 | T32_U | WITNESS_FALLBACK | 0.000 | 0.000 | 0.000 | 0.00,0.00,0.00,0.00,0.00,0.00,0.00 |
| 1 | NM1_P | WITNESS_SELECTED | 0.000 | 0.000 | 0.000 | 1.00,1.00,1.00,1.00,1.00,1.00,1.00 |
| 1 | NM4_P | SELECTED_ROUND r3 | 0.000 | 0.867 | 0.055 | 0.26,0.00,0.00,0.00,0.72,0.62,0.00 |
| 1 | T32_P | WITNESS_SELECTED | 0.000 | 0.000 | 0.000 | 0.00,0.00,0.00,0.00,0.00,0.00,0.00 |
| 1 | DET_SEL1 | ALL_D17_ASSIGNMENT | 0.000 | 0.000 | 0.000 |  |
| 1 | DET_SEL4 | ALL_D17_ASSIGNMENT | 0.000 | 0.000 | 0.000 |  |
| 1 | RD_TASK | SELECTED_ROUND r0 | 0.000 | 0.000 | 0.000 | -,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,- |
| 1 | RD_PRIV | WITNESS_SELECTED | 0.000 | 0.000 | 0.000 | -,-,-,- |
| 1 | ADV_B1 | WITNESS_SELECTED | 0.000 | 0.000 | 0.000 |  |
| 1 | ADV_B2 | WITNESS_SELECTED | 0.000 | 0.000 | 0.000 |  |
| 1 | ADV_B1_P | WITNESS_FALLBACK | 0.000 | 0.000 | 0.000 |  |
| 1 | ADV_B2_P | WITNESS_SELECTED | 0.000 | 0.000 | 0.000 |  |
| 2 | NM1_U | WITNESS_FALLBACK | 0.000 | 0.000 | 0.000 | 0.00,0.00,0.00,0.00,0.00,0.00,0.00 |
| 2 | NM4_U | WITNESS_FALLBACK | 0.000 | 0.000 | 0.000 | 0.00,0.00,0.00,0.00,0.00,0.00,0.00 |
| 2 | T32_U | WITNESS_FALLBACK | 0.000 | 0.000 | 0.000 | 0.00,0.00,0.00,0.00,0.00,0.00,0.00 |
| 2 | NM1_P | SELECTED_ROUND r3 | 0.028 | 0.049 | 0.956 | 0.99,1.00,0.99,0.97,0.94,0.99,0.98 |
| 2 | NM4_P | SELECTED_ROUND r0 | 0.000 | 0.718 | 0.069 | 0.00,0.00,0.00,0.00,0.00,0.00,0.00 |
| 2 | T32_P | SELECTED_ROUND r4 | 0.000 | 0.883 | 0.030 | 0.00,0.00,0.00,0.00,0.00,0.00,0.00 |
| 2 | DET_SEL1 | ALL_D17_ASSIGNMENT | 0.000 | 0.000 | 0.000 |  |
| 2 | DET_SEL4 | DETERMINISTIC_ASSIGNMENT | 0.231 | 0.143 | 0.000 |  |
| 2 | RD_TASK | SELECTED_ROUND r3 | 0.005 | 0.002 | 0.000 | -,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,- |
| 2 | RD_PRIV | WITNESS_SELECTED | 0.000 | 0.000 | 0.000 | -,-,-,- |
| 2 | ADV_B1 | WITNESS_SELECTED | 0.000 | 0.000 | 0.000 |  |
| 2 | ADV_B2 | WITNESS_SELECTED | 0.000 | 0.000 | 0.000 |  |
| 2 | ADV_B1_P | WITNESS_FALLBACK | 0.000 | 0.000 | 0.000 |  |
| 2 | ADV_B2_P | WITNESS_FALLBACK | 0.000 | 0.000 | 0.000 |  |

## 3. Three-level summary

| Family | Trained | Used by the LP (any round, V>0) | Final release has within-T32 variation |
|---|---|---|---|
| NM U-form (NM1_U, NM4_U) | yes, 3/3 anchors | **no, 0/6.** η = 0 at every round. The utility LP never bought a richer column. | 0/6. Anchor 0 keeps a deterministic T32 kernel identical to T32_U; anchors 1–2 fall back to D17. |
| NM P-form (NM1_P, NM4_P) | yes | yes, 5/6 unit-anchors | 3/6: NM4_P a0 (V=.520), NM1_P a0 (V=.222), NM1_P a2 (V=.028) |
| DET_SEL4 | yes | yes | 2/3: a0 (V=.298), a2 (V=.231). It is deterministic. |
| RD_TASK | yes | — | 1/3: a2 (V=.005). Anchors 0–1 are D17 after inner selection. |
| RD_PRIV, ADV (all four) | yes | — | 0/3. All select D17 by rule or witness fallback. |

## 4. What the selected nominees actually are

- **U nominee NM1_U** is an exact alias of NM4_U and T32_U on all three anchors. On anchor 0 it is a deterministic T32 kernel far from D17 (mean TV .718). On anchors 1 and 2 it is exactly D17. It contains **no** decision beyond the old code.
- **P nominee NM4_P**:
  - carries a genuinely richer stochastic law only on **anchor 0** (V=.520, 87% of weight on stochastic rows);
  - on anchors 1 and 2 its selected round is a T32 kernel with η=0.

  Its three-anchor outer estimates therefore mix one richer release with two old-code kernels.

## 5. Reading

- **The richer-input hypothesis received a real test in this study.** The richer columns were trained, differ from D17 inside every state, and were in the LP's feasible set at every round on every anchor.
- **The utility form declined them everywhere.** The task-only policy's predicted within-state task gains did not survive the out-of-sample coefficient rows or inner selection. This agrees with the RD_TASK preflight, which showed no outer task gain over D17: +.00017 U / +.00019 W.
- **The privacy form used them**, but the closing-refit and witness rules retained a richer law on only three unit-anchors.

Whether the retained richer laws help is answered in `RESEARCH_DECISION.md`, not here.
