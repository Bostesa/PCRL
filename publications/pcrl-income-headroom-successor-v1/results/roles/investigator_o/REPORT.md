# Family O report: occupation-side margin repair (investigator O)

*Saved by the coordinator, verbatim from investigator O's final hand-back: the harness refused the investigator's own write of
this file with the message "Subagents should return findings as text, not write report files. Include this content in your
final response instead." (PROTOCOL_DEVIATIONS.md item 4).*

PCRL income-headroom successor development campaign. Exploratory development on previously used records, not confirmation.

Data and service:
- Public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida.
- The service predicts an income class (2 classes) and an occupation group (5 classes) from six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP; admitted 145-column encoding `X`).
- Each recipient receives only its own float64 probability vector and argmax decision.
- SEX is the dataset's recorded demographic column (sex0 = SEX code 1, men; sex1 = SEX code 2, women).
- Readers are separate evaluation classifiers that estimate SEX from the released probability vectors.

Measurement conventions:
- All values are MODEL_SELECT development values (8,001 people, 5,306 households) from `ihs.devrun score`.
- SEs are household-clustered paired SEs. A margin is cap minus value, also given in SE units.
- MODEL_SELECT is a development role that has already been used adaptively.
- Full-precision aggregates are in `RESULTS_SUMMARY.json`, the card and revision log in `CARD.md`, the code in `ihs/fam_o.py`.

## 1. Answer

**The best occupation output is `O1_SEXBAL` on both seeds.** It recalibrates the inherited JOINT_TREE E20 occupation output (the occupation output of H_LJ) with one temperature and a 5-class log-bias: p' = softmax(clr(p) / T + b).
- T is fitted by occupation NLL on CALIBRATION, without SEX.
- b is fitted on MODEL_FIT so that the two SEX groups' occupation summed-Brier excess against U0 is equal (minimax). SEX is used only as a training-time label.
- Income is H_LJ's own LOCAL_TREE E20 income output, bitwise identical, so the only difference from H_LJ is the occupation output.

| Quantity | H_LJ (s0 / s1) | O1_SEXBAL (s0 / s1) |
|---|---|---|
| Men's guard | 0.004417 / 0.003941 (0.32 / 0.62 SE of margin) | **0.001659 / 0.002299** (1.83 / 1.62 SE) |
| Women's guard | −0.004938 / −0.003039 | −0.002185 / −0.002759 (4.13 / 4.68 SE) |
| Occupation NLL vs Ucal | 0.003406 / 0.002947 | 0.002684 / 0.001642 |
| Occupation-view proxy | 0.5694 / 0.5800 | 0.5682 / 0.5786 (no rise) |
| Pair proxy | 0.6241 / 0.6239 | 0.6252 / 0.6235 |

- All 6 occupation clauses pass on both seeds with at least 2.57 SE of margin.
- Reserve screen: RESERVE_MET on both seeds, with occupation side CHANGED_NON_WORSENING.
- **The 2-SE target is not met.** The men's guard is 0.00031 / 0.00069 short of the 2-SE reserve and passes the screen only through the non-worsening branch. The card's target (men's margin of at least 2 SE on both seeds) is therefore not met: the inherited constraint is relieved, not removed.

**Attribution** (sections 4 and 5):
- **Calibration alone worsens the guard.** The ordinary temperature control (OC_TCAL) moves the men's guard to 0.005126 / 0.005085, failing on both seeds. An ordinary per-class bias fitted on CALIBRATION is also worse (OC_TBCAL, seed 0: 0.005729).
- **The improvement has two parts of similar size**, both measured against OC_TCAL, which shares O1's temperature:
  - Fitting the bias on MODEL_FIT without SEX (OC_BFIT): −0.00186 / −0.00114. This part also lowers occupation NLL for everyone.
  - The SEX-balanced objective (O1 minus OC_BFIT): −0.00160 / −0.00165 for men and +0.00181 / +0.00185 for women, an almost one-for-one transfer of guard slack from women to men.
- **The SEX-balanced part of the bias is the same on both seeds:** about (−0.014, −0.013, −0.023, +0.033, +0.016) in log-odds for (management/business/science/arts, service, sales/office, natural resources/construction/maintenance, production/transportation/material moving).

## 2. What was run

| Config | Role | Operation | Seeds | Model fits per seed |
|---|---|---|---|---|
| OC_TCAL | control named by the brief | softmax(clr(p_J) / T); T by NLL on CALIBRATION | 0, 1 | 1 |
| OC_TBCAL | control: ordinary bias calibration | T and b jointly by NLL on CALIBRATION (no SEX) | 0 | 1 |
| OC_BFIT | attribution control (added 02:41Z, disclosed) | T as OC_TCAL; b by unweighted Brier on MODEL_FIT (no SEX) | 0, 1 | 2 |
| O1_SEXBAL | proposed component | T as OC_TCAL; b by SEX-balanced minimax on MODEL_FIT | 0, 1 | 47 (1 T + 46 bias fits in the lambda bisection) |

- **Queue units:** each configuration and seed has a fit unit `IHS_O_<config>_S<s>_V1` and a scoring unit `IHS_O_<config>_S<s>_SCORE_V1`.
- **Configurations:** 7 of 8 used, each counted toward the 36. Ledger entries 27, 29, 32, 37, 40, 47 and 49; decision entry 56.
- **Model fits:** 101 in total.
- **Revisions:** none. The card's revision triggers were not met (section 6).
- **Invented-data check:** unit IHS_O_SELFTEST_V1, 8 of 8 checks pass (`INVENTED_DATA_CHECK.json`).
- **Matched controls of the best configuration:** OC_TCAL (same temperature, no bias; the brief's control) and OC_BFIT (same temperature, bias form and loss, no SEX weighting). H_LJ is the fixed-occupation control.

Fitted parameters (`PARAMS.json` in each run directory):

| Config | Seed | T | b (centred; classes 0–4) | lambda* |
|---|---|---|---|---|
| OC_TCAL | 0 / 1 | 0.9774694444 / 0.9582767076 | 0 | – |
| OC_TBCAL | 0 | 0.9938345565 | (0.03852, 0.07788, −0.03139, 0.02697, −0.11198) | – |
| OC_BFIT | 0 | 0.9774694444 | (−0.06820, 0.02904, −0.01167, 0.07969, −0.02887) | unweighted (men's share 0.5103) |
| OC_BFIT | 1 | 0.9582767076 | (−0.02386, 0.08088, −0.05980, −0.06293, 0.06571) | unweighted |
| O1_SEXBAL | 0 | 0.9774694444 | (−0.08196, 0.01625, −0.03427, 0.11267, −0.01269) | 0.5325586 (men's relative weight 1.093) |
| O1_SEXBAL | 1 | 0.9582767076 | (−0.03826, 0.06833, −0.08344, −0.02890, 0.08228) | 0.5337525 (men's relative weight 1.099) |

Classes: 0 management, business, science and arts; 1 service; 2 sales and office; 3 natural resources, construction and maintenance; 4 production, transportation and material moving.

## 3. Results on MODEL_SELECT

### 3.1 Within-SEX occupation guards (summed-Brier excess vs U0, cap 0.005)

| Config | Seed | Men: value (SE) | Men: margin (SE units) | Women: value (SE) | Women: margin (SE units) | Reserve |
|---|---|---|---|---|---|---|
| H_LJ | 0 | 0.004417 (0.001814) | 0.000583 (0.32) | −0.004938 (0.001698) | 0.009938 (5.85) | inherited |
| OC_TCAL | 0 | 0.005126 (0.001824) | −0.000126 (−0.07) | −0.005629 (0.001690) | 0.010629 (6.29) | NOT_MET (guard sex0) |
| OC_TBCAL | 0 | 0.005729 (0.001847) | −0.000729 (−0.39) | −0.006563 (0.001675) | 0.011563 (6.90) | NOT_MET (guard sex0) |
| OC_BFIT | 0 | 0.003264 (0.001838) | 0.001736 (0.94) | −0.003992 (0.001738) | 0.008992 (5.17) | MET |
| **O1_SEXBAL** | 0 | **0.001659 (0.001826)** | **0.003341 (1.83)** | −0.002185 (0.001739) | 0.007185 (4.13) | **MET** |
| H_LJ | 1 | 0.003941 (0.001697) | 0.001059 (0.62) | −0.003039 (0.001710) | 0.008039 (4.70) | inherited |
| OC_TCAL | 1 | 0.005085 (0.001723) | −0.000085 (−0.05) | −0.004520 (0.001712) | 0.009520 (5.56) | NOT_MET (guard sex0) |
| OC_BFIT | 1 | 0.003950 (0.001683) | 0.001050 (0.62) | −0.004604 (0.001650) | 0.009604 (5.82) | NOT_MET (guard sex0; 0.000009 below the inherited margin) |
| **O1_SEXBAL** | 1 | **0.002299 (0.001672)** | **0.002701 (1.62)** | −0.002759 (0.001657) | 0.007759 (4.68) | **MET** |

The reserve rule asks for a men's margin of 2 × max(own SE, H_LJ SE): 0.003652 on seed 0 and 0.003394 on seed 1. O1_SEXBAL is short by 0.000311 and 0.000693, so it passes only through non-worsening. Every other occupation condition of O1_SEXBAL meets the 2-SE reserve.

### 3.2 The six occupation clauses of O1_SEXBAL, with H_LJ for comparison

Each cell shows value (SE); margin (SE units).

| Clause (cap) | O1 seed 0 | H_LJ seed 0 | O1 seed 1 | H_LJ seed 1 |
|---|---|---|---|---|
| accuracy loss vs U0 (0.010) | −0.003000 (0.003535); 0.013000 (3.68) | −0.002000 (0.003509); 0.012000 (3.42) | −0.002125 (0.003484); 0.012125 (3.48) | −0.004000 (0.003543); 0.014000 (3.95) |
| NLL excess vs U0 (0.010) | 0.001804 (0.003008); 0.008196 (2.72) | 0.002526 (0.003008); 0.007474 (2.48) | 0.001594 (0.002725); 0.008406 (3.08) | 0.002899 (0.002765); 0.007101 (2.57) |
| Brier excess vs U0 (0.005) | −0.000241 (0.001261); 0.005241 (4.16) | −0.000208 (0.001245); 0.005208 (4.19) | −0.000202 (0.001174); 0.005202 (4.43) | 0.000490 (0.001201); 0.004510 (3.76) |
| accuracy loss vs Ucal (0.010) | −0.003000 (0.003535); 0.013000 (3.68) | −0.002000 (0.003509); 0.012000 (3.42) | −0.002125 (0.003484); 0.012125 (3.48) | −0.004000 (0.003543); 0.014000 (3.95) |
| NLL excess vs Ucal (0.010) | 0.002684 (0.002849); 0.007316 (2.57) | 0.003406 (0.002871); 0.006594 (2.30) | 0.001642 (0.002701); 0.008358 (3.09) | 0.002947 (0.002735); 0.007053 (2.58) |
| Brier excess vs Ucal (0.005) | 0.000090 (0.001202); 0.004910 (4.08) | 0.000123 (0.001196); 0.004877 (4.08) | −0.000197 (0.001166); 0.005197 (4.46) | 0.000494 (0.001189); 0.004506 (3.79) |

- **NLL:** both occupation NLL clauses improve on both seeds. The NLL vs Ucal clause was the one that came close to its cap on DEV_REPORT for H_LJ.
- **Accuracy:** slightly lower than H_LJ's on seed 1 (by 0.0019), still better than U0's.
- **Decisions:** the bias changes 302 (seed 0) and 421 (seed 1) of the 8,001 MODEL_SELECT occupation decisions relative to H_LJ. The temperature alone changes none.
- **Income clauses:** all 12 are H_LJ's (identical income output).
- **Controls' occupation NLL vs Ucal:** OC_TCAL 0.003458 / 0.002679, OC_BFIT 0.002719 / 0.001655, OC_TBCAL 0.003571 (seed 0).

### 3.3 Reader proxies (2 × 2 development grid)

Mean [min, max] over the four cells; cell values in the order A0, A1, B0, B1.

| Config | Seed | Occupation view | Occupation cells | Pair (max incl. single views) | Pair cells |
|---|---|---|---|---|---|
| H_LJ | 0 | 0.5694 [0.5639, 0.5749] | 0.5749, 0.5749, 0.5639, 0.5639 | 0.6241 [0.6233, 0.6249] | 0.6233, 0.6233, 0.6249, 0.6249 |
| OC_TCAL | 0 | 0.5695 [0.5641, 0.5749] | 0.5749, 0.5749, 0.5641, 0.5641 | 0.6241 [0.6232, 0.6250] | 0.6232, 0.6232, 0.6250, 0.6250 |
| OC_TBCAL | 0 | 0.5663 [0.5585, 0.5733] | – | 0.6246 [0.6231, 0.6258] | – |
| OC_BFIT | 0 | 0.5678 [0.5602, 0.5744] | 0.5736, 0.5744, 0.5602, 0.5632 | 0.6246 [0.6237, 0.6253] | 0.6240, 0.6237, 0.6253, 0.6253 |
| **O1_SEXBAL** | 0 | **0.5682 [0.5619, 0.5750]** | 0.5750, 0.5742, 0.5619, 0.5619 | 0.6252 [0.6242, 0.6268] | 0.6242, 0.6268, 0.6249, 0.6249 |
| H_LJ | 1 | 0.5800 [0.5762, 0.5827] | 0.5812, 0.5827, 0.5800, 0.5762 | 0.6239 [0.6235, 0.6245] | 0.6245, 0.6239, 0.6235, 0.6235 |
| OC_TCAL | 1 | 0.5800 [0.5761, 0.5816] | 0.5816, 0.5812, 0.5812, 0.5761 | 0.6241 [0.6237, 0.6250] | 0.6250, 0.6240, 0.6237, 0.6237 |
| OC_BFIT | 1 | 0.5780 [0.5748, 0.5811] | 0.5811, 0.5811, 0.5748, 0.5748 | 0.6234 [0.6230, 0.6239] | 0.6239, 0.6238, 0.6230, 0.6230 |
| **O1_SEXBAL** | 1 | **0.5786 [0.5748, 0.5829]** | 0.5829, 0.5821, 0.5748, 0.5748 | 0.6235 [0.6230, 0.6240] | 0.6240, 0.6239, 0.6230, 0.6230 |

O1_SEXBAL comparisons with the eligible references (per cell: candidate minus reference for recipients, reference minus candidate for the pair benefit):

| Comparison | Seed 0 | Seed 1 |
|---|---|---|
| Occupation recipient vs H_LJ | mean −0.0012, max +0.00003 | mean −0.0014, max +0.0017 |
| Income recipient vs H_LJ | 0 (identical view) | 0 (identical view) |
| Pair benefit vs H_LJ | mean −0.0011, min −0.0035 | mean +0.0004, min −0.00002 |
| Pair benefit vs NEURAL_RAW | mean +0.0043, min +0.0009 | mean +0.0122, min +0.0100 |

The occupation recipient changes are far inside the 0.010 bar, and the mean does not rise.

Reader detail:
- **Seed-0 pair cell A1** (0.6268 vs H_LJ's 0.6233) comes from one seed-dependent reader, LOG_MLP (0.6268 vs 0.6043). H_LJ's winning pair reader, CATI1 (0.6233), reads 0.6232 under O1. LOG_MLP itself varies by 0.007–0.015 across H_LJ's own cells.
- **Occupation view:** the winning reader, CATI1, is nearly unchanged (0.5742 vs 0.5749; 0.5619 vs 0.5639). This is expected, because its design is spanned by log-contrasts and their splines and products, which an affine map of the centred log-ratio leaves invariant up to regularization.
- **The coordinator's combinations:** the coordinator reported a pair cost of about +0.0007 to +0.0013 when O1_SEXBAL replaces the JOINT_TREE occupation output (CMB_P2JR_O1SEXBAL 0.6146 / 0.6144; CMB_P3JR_O1SEXBAL seed 0 0.6079 vs P3's 0.6072). That is the same size as the reader variability above.

## 4. Attribution: bias fitted on MODEL_FIT vs the SEX-balanced objective

Decomposition of the MODEL_SELECT guard change (metric units; negative = better for that group):

| Step | Men s0 | Men s1 | Women s0 | Women s1 | Occupation NLL vs Ucal s0 / s1 |
|---|---|---|---|---|---|
| H_LJ → OC_TCAL (CALIBRATION temperature, T < 1) | +0.000709 | +0.001144 | −0.000691 | −0.001481 | +0.000052 / −0.000268 |
| OC_TCAL → OC_BFIT (bias on MODEL_FIT, no SEX) | −0.001862 | −0.001135 | +0.001637 | −0.000084 | −0.000739 / −0.001024 |
| OC_BFIT → O1_SEXBAL (SEX-balanced weighting) | −0.001605 | −0.001651 | +0.001807 | +0.001845 | −0.000035 / −0.000013 |
| **Total H_LJ → O1_SEXBAL** | −0.002758 | −0.001642 | +0.002753 | +0.000280 | −0.000722 / −0.001305 |
| For reference: H_LJ → OC_TBCAL (T and bias on CALIBRATION, NLL) | +0.001312 | – | −0.001625 | – | +0.000165 / – |

1. **Ordinary calibration alone makes the men's guard worse.**
   - The held-out temperature sharpens the output (T = 0.977 / 0.958) and moves Brier from women to men. The men's guard then fails on both seeds (0.005126 / 0.005085), with the occupation-view proxy unchanged (0.5695 / 0.5800).
   - The NLL-optimal bias on CALIBRATION (2,982 people, no SEX) points in a different direction from every MODEL_FIT bias and worsens the men's guard further (seed 0: 0.005729).
2. **Fitting the bias on MODEL_FIT without SEX helps, but not consistently.**
   - It corrects each seed's own marginal distortions of the JOINT_TREE output, in a direction that differs by seed. Seed 0 raises construction/maintenance and lowers management. Seed 1 raises service and production/transportation and lowers sales/office and construction.
   - It lowers occupation NLL vs Ucal by 0.0007 / 0.0010.
   - It helps the men's guard by −0.0019 on seed 0, but on seed 1 (−0.0011) it only cancels the temperature's harm, leaving the guard at H_LJ's level (0.003950).
3. **The SEX-balanced objective is a clean transfer that is stable across seeds.**
   - Relative to OC_BFIT, it moves 0.00160 / 0.00165 of the men's guard excess to women (+0.00181 / +0.00185), with overall NLL essentially unchanged.
   - Its bias component is the same on both seeds: (−0.0138, −0.0128, −0.0226, +0.0330, +0.0162) on seed 0 and (−0.0144, −0.0126, −0.0236, +0.0340, +0.0166) on seed 1. It raises the two male-dominated groups and lowers sales/office, management and service.
   - The equalizing weight is modest: the men's weight is about 1.09–1.10 relative to the unweighted fit. For comparison, the source's A3 used a 2 : 1 weight inside the full correction fit and moved the guards past each other (men −0.0082, women +0.0020).
4. **No task learning or compression is involved.** No task model was refitted, the routing and corrections are frozen, and the map is a bijection, so everything here is calibration.
   - The proposed component accounts for about 58% (seed 0) and essentially all (seed 1) of O1's net improvement in the men's guard over H_LJ.
   - The rest on seed 0 is the MODEL_FIT bias, net of the temperature's harm.

## 5. Why the men's guard stops at 1.6–1.8 SE

MODEL_FIT people are in-sample for the teacher routing, the JOINT_TREE corrections and U0. The table compares the in-sample values on MODEL_FIT (aggregates from `FIT_DIAG.json`) with MODEL_SELECT:

| Output | Seed | Men in-sample | Men MODEL_SELECT | Women in-sample | Women MODEL_SELECT | Gap (men − women) in-sample / MODEL_SELECT |
|---|---|---|---|---|---|---|
| H_LJ | 0 | 0.00448 | 0.00442 | −0.00122 | −0.00494 | 0.0057 / 0.0094 |
| O1_SEXBAL | 0 | 0.00113 | 0.00166 | 0.00113 | −0.00219 | 0.0000 / 0.0039 |
| H_LJ | 1 | −0.00014 | 0.00394 | −0.00345 | −0.00304 | 0.0033 / 0.0070 |
| O1_SEXBAL | 1 | −0.00288 | 0.00230 | −0.00288 | −0.00276 | 0.0000 / 0.0051 |

- **Changes carry over reasonably well; levels do not.** O1's change in the men's guard is −0.00335 in-sample vs −0.00276 on MODEL_SELECT (seed 0) and −0.00274 vs −0.00164 (seed 1).
- **The levels carry an offset.** On both seeds the in-sample men-minus-women gap is about 0.0037 smaller than on MODEL_SELECT. Where the offset sits depends on the seed: on seed 0 the women's excess is 0.0037 lower out of sample; on seed 1 the men's is 0.0041 higher.
- **The result is under-correction.** The minimax equalizes the groups in-sample but leaves a MODEL_SELECT gap of 0.0039 / 0.0051. Equalizing on MODEL_SELECT would need a further transfer of about 0.002 / 0.0025, which the women's slack could afford.
- **Correcting the offset is not allowed.** Only MODEL_SELECT, ATTACK_* and DEV_REPORT have out-of-sample SEX labels, and none may be used for fitting. Taking the offset from H_LJ's recorded MODEL_SELECT guards would mean fitting a release parameter on MODEL_SELECT, so it was not done.
- **The other listed revisions were not run.** A richer information-preserving calibrator (per-class temperatures or a full affine map, fitted by the same in-sample minimax) would face the same offset. A blend toward the LOCAL_TREE corrections would raise occupation-view recovery (H_LL reads 0.5874 / 0.5931). No revision was run (decision entry 56); one configuration of the allotment is unused.
- **Planning arithmetic only.** This assumes the same per-household variance and is not a confidence statement. DEV_REPORT SEs are about 1.158 times MODEL_SELECT's, so if the MODEL_SELECT values held, O1's men's margin would be about 1.58 / 1.40 DEV-scale SE.

## 6. Card criteria, checked

| Criterion fixed in the card | Outcome |
|---|---|
| Revise if the seed-0 men's guard improves by less than 1 SE (0.0018) | not triggered: it improved by 0.002758 (1.52 SE) |
| …or only by bringing the women's guard or an occupation clause within 1 SE of its cap | not triggered: all ≥ 2.57 SE |
| …or if the occupation-view proxy mean rises by more than 0.005 | not triggered: −0.0012 / −0.0014 |
| Target: men's margin ≥ 2 SE on both seeds | **not met** (1.83 / 1.62 SE) |
| Target: women's guard and the 6 occupation clauses ≥ 2 SE | met (minimum 2.57 SE) |
| Target: occupation proxy ≤ H_LJ + 0.010 in every cell, preferably no rise in the mean | met (cell max +0.0017; mean lower) |
| Target: pair mean ≤ H_LJ + 0.003 | met (+0.0011 / −0.0004) |
| If a matched control explains the result, report the control as the finding | OC_TCAL and OC_TBCAL worsen the guard. OC_BFIT explains the non-SEX part (section 4), and that part is reported as such. |

## 7. Deployment and records

**Code paths:**
- `ihs.fam_o.predict(seed, config_id, X, names)` recomputes both outputs from the admitted encoding of the six raw fields.
- `predict_occupation(...)` returns the occupation output alone, for combinations.
- Both read only `PARAMS.json` (T, b, arms, label) plus the frozen T_TREE routing and the stored stage-B theta.

**Checks:**
- Every fit ran `verify`: predict() reproduces the saved release bitwise on all six roles (MODEL_FIT, CALIBRATION, MODEL_SELECT, ATTACK_FIT, ATTACK_VALIDATION, DEV_REPORT).
- Income is bitwise equal to the frozen-bank H_LJ income on MODEL_SELECT (max abs difference 0.0).
- The recomputed base occupation output equals H_LJ's bitwise.
- Distinct occupation vectors are preserved (7,500 / 7,498 on MODEL_SELECT).

**Files:**
- Releases are under `.private/pcrl_income_headroom_successor_v1/runs/O/<config>/S<seed>/`: RELEASE.npz, RELEASE.json, PARAMS.json, FIT_DIAG.json, EVAL_MS.json.
- The matched control is recorded in RELEASE.json for runs fitted after 02:39Z, and in this report for all runs.

**Roles read:**
- CALIBRATION features and occupation: the temperature, and the OC_TBCAL bias.
- MODEL_FIT features, occupation and SEX: the O1 bias (SEX as a training-time label), the OC_BFIT bias (no SEX), and in-sample subgroup diagnostics.
- Features of every role: label-free release computation.
- MODEL_SELECT labels: only through `ihs.devrun score` (7 logged evaluations).
- No ATTACK or DEV_REPORT labels were read.

**REPORT.md block:** the harness refused to write this file ("Subagents should return findings as text, not write report files. Include this content in your final response instead."). The text was handed to the coordinator.

## 8. Recommendation

1. **Use O1_SEXBAL as the occupation output wherever the inherited JOINT_TREE output binds.**
   - It improves the men's guard by 1.52 / 0.97 SE and improves the occupation NLL clauses on both seeds.
   - It keeps occupation-view recovery at or below H_LJ's, and it passes the reserve screen.
   - Keep H_LJ's JOINT_TREE output as the fixed-occupation control, and OC_TCAL and OC_BFIT as its matched calibration controls.
2. **Describe it accurately.**
   - It is a SEX-free recalibration (a bijection of the occupation vector) whose bias was chosen with SEX as a training-time label.
   - It moves Brier from women to men. It does not reduce the privacy cost of the JOINT_TREE output.
   - About half of its seed-0 effect, and none of its seed-1 effect, is ordinary in-sample bias recalibration.
   - Information is unchanged, but fitted readers can move: one seed-0 pair cell rose by 0.0035 through LOG_MLP, and the coordinator's combinations show +0.0007 to +0.0013.
3. **Do not claim a 2-SE reserve on the men's guard.** It is 1.83 / 1.62 SE on MODEL_SELECT, from an in-sample criterion, on an adaptively used role.
4. **Not recommended:**
   - tuning the SEX weight to MODEL_SELECT's guard gap, which would fit a release parameter on MODEL_SELECT;
   - blending toward the LOCAL_TREE or TASK_REFIT corrections, which trades occupation recovery for margin.

## 9. Resources

- **Queue units:** 15, all done, none failed (1 self-test, 7 fits, 7 scorings).
- **CPU:** 298.4 CPU-s in the queue (0.083 CPU-h), plus a separate upper-estimate charge of 60 CPU-s for light work outside the queue (label IHS_O_AGENT_LIGHT: JSON summaries and compile checks; no real-data computation). Total about 0.10 CPU-h of the 0.8 allotted.
- **Configurations:** 7 of 8.
- **Model fits:** 101 (OC_TCAL 1 + 1; OC_TBCAL 1; OC_BFIT 2 + 2; O1_SEXBAL 47 + 47).
- **Reader fits:** the evaluator requested 168 per scoring row, partly served from its cache because the income view equals H_LJ's.
- **Compute:** no paid or cloud compute. Nothing was committed or pushed, and no branch was switched.
