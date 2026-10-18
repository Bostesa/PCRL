# Investigator P report: pair-aware learned income output with the occupation output frozen (PAIO)

*Saved by the coordinator, verbatim from investigator P's final hand-back: the harness refused the investigator's own write of
this file with the message "Subagents should return findings as text, not write report files. Include this content in your
final response instead." (PROTOCOL_DEVIATIONS.md item 4).*

PCRL income-headroom successor development campaign. This is exploratory development on previously used Florida ACS 2019 records. MODEL_SELECT has been used adaptively here and by earlier campaigns, so nothing below is confirmation.

Written 2026-10-18, about 02:57Z. Related files: `CARD.md` (card and revision log), `ihs/fam_p.py` (code), `REFERENCE_DISTORTIONS.json` (label-free pre-fit check).

**Context, in exact terms.** Public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida. The service predicts income class (2 classes) and occupation group (5 classes) from six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP), using the admitted 145-column encoding `X`. Each recipient receives only its own float64 probability vector and argmax decision.

SEX is the dataset's recorded demographic column. Readers are separate evaluation classifiers that estimate SEX from the released probability vectors: each output alone and the linked pair. All proxies below come from the frozen 2 × 2 development grid (`IHS1-dev-proxy-v1`; the pair value is the maximum including single-view readers). They are not final readers.

## 1. Bottom line

**Best configuration: `P3_JR_B60` (revision 2).**
- Income: the T_TREE income routing plus learned income leaf-score corrections. These are trained on MODEL_FIT against online SEX critics on the income wire and on the linked pair wire (the frozen accepted JOINT_TREE E20 occupation wire), with rho = 1 for 30 epochs.
- After training, the corrections get one projection toward the teacher, to an explicit label-free distortion budget: BRIER 0.006 / ACC 0.0045 / KL 0.010. ACC binds; the end scale is 0.6812 / 0.7385.
- Occupation: the accepted JOINT_TREE E20 output, unchanged.
- Matched task-only twin: `PC_TWIN_B40` (identical loop, lambda = 0).

**MODEL_SELECT results, seed 0 / seed 1:**
- Point-eligible on all 12 clauses and both guards.
- RESERVE_MET on both seeds. Min income ratio is 3.7246 / 2.6768; the binding condition is seed-1 income accuracy loss, 0.001750 (SE 0.003082).
- Pair proxy 0.607159 / 0.611296 (cells 0.6069–0.6075 / 0.6091–0.6134); two-seed mean 0.609228.
- Below H_LJ in every cell (min +0.015771 / +0.011026) and below NEURAL_RAW (min +0.020389 / +0.020616).
- Below B2TWIN in every cell (min +0.009912 / +0.005716).
- Income recipient: below both references in every cell (max difference −0.002644). The occupation output is identical to H_LJ's.

**My dry run of the frozen nomination rule:**
- (a) yes, (b) RESERVE_MET, c1 yes, c2 yes, c3 yes.
- Flags: no PAIR_NEAR_TIE and no RECIPIENT_EDGE. READER_UNSTABLE on seed 0: the winning pair reader is ET25 in repetition A and OUT_LR in repetition B.
- Utility confidence: UNRESOLVED for 8 of 28 conditions. Six are inherited occupation-side; the other two are seed-1 income accuracy loss vs U0 and vs Ucal (upper bound 0.011378).

**Attribution, honestly stated.**
- Against its matched twin, P3 reads 0.028367 / 0.024980 lower on the pair.
- Matched-cost temperature shrinkage of the twin does nothing (0.636261 on seed 0), so the result is not "less confident".
- But the no-training comparator at the same budget, `PC_JSCALE_B60`, gets most of the benefit. It is the stored JOINT_TREE E20 income corrections scaled toward the teacher by the same label-free budget. It reads 0.608279 / 0.611959 and also qualifies by my reading (min income ratio 3.4025 / 2.2122).
- P3 is lower by 0.001120 / 0.000663 (6 of 8 cells), with lower MODEL_SELECT income Brier (by 0.000595 / 0.001288 vs Ucal), but its accuracy loss is not lower on seed 0.
- So two components do the work: (i) a pair-trained income correction direction and (ii) the explicit label-free distortion budget that makes it eligible with reserve. Training that direction against the frozen accepted occupation output, this family's specific step, adds a small, mostly consistent gain within the grid's reader variability.

**Resources.** 14 seed-specific configurations (allotment raised from 12 to 14 by the coordinator), initial version plus 2 revisions. 86 individual model fits. 34 queue units, 0 failed, 741.3 CPU-s (0.206 CPU-h), plus a 120 CPU-s agent upper estimate.

## 2. What I tried, in order

### Mechanism (all trained configurations)

- **Parametrization.** Income raw score = T_TREE income teacher raw + theta[leaves(x)].sum / T (the `acs_replication.leafcorr` parametrization). theta = 0 is the teacher exactly. Probabilities come from `trees.probabilities('income', raw)`; the decision is the argmax.
- **Training on MODEL_FIT only (20,000 people).**
  - Task loss: income cross-entropy with the leafcorr recipe (AdamW lr 0.01, weight decay 0.01, batch 256, 40 steps per epoch, 30 epochs).
  - Privacy term: lambda × [risk(income wire) + risk(pair wire)], from the leafcorr online SEX critic bank (linear / MLP32 / MLP64 per view; 80 warm rounds, 5 rounds per update).
  - The pair wire is this income wire plus the frozen JOINT_TREE E20 occupation wire. The fit asserts that wire equals `ihs.bases.joint_occupation` within 1e-10, and it is never updated.
  - lambda = rho × median‖grad CE‖ / median‖grad P‖ over 16 fixed draws at the start. At rho = 1 this gives lambda = 0.9956 / 0.8380.
  - SEX is a training-time label for the critics only.
- **Explicit income-utility budget (label-free).** Three statistics, measured against the frozen teacher on the MODEL_FIT people:
  - BRIER = 2·mean(dp²), the expected summed-Brier increase if the teacher were calibrated;
  - ACC = mean(flip × |2 p_T − 1|);
  - KL = mean KL(p_T ‖ p).

  theta is scaled toward the teacher (the largest s in [0, 1], found by bisection) until all three hold. No label enters the budget, so the in-sample position of MODEL_FIT people in the teacher routing does not bias it, and no internal MODEL_FIT fold is used.
- **Pre-fit check** (`REFERENCE_DISTORTIONS.json`, features only, before registration). On the stored releases, the label-free BRIER matched the published MODEL_SELECT Brier increments: JOINT_TREE E20 0.00945 / 0.00953 vs +0.00938 / +0.00945; KL 0.01144 / 0.01161 vs NLL +0.01182 / +0.01191. It never under-stated an increment by more than 0.0007.
- **Runtime and rebuild.**
  - Runtime input is `X` only; the income output does not read the occupation output at runtime.
  - `ihs.fam_p.predict(seed, config_id, X, names)` rebuilds both outputs from features. Income: T_TREE routing plus `THETA.npz`. Occupation: T_TREE occupation routing plus `stage_b_theta(seed, 'JOINT_TREE', 'E20')[1]`.
  - Bitwise parity with the saved release was asserted on MODEL_SELECT and DEV_REPORT features inside every fit.

### Versions

1. **P1 (initial): `P1_PAIR_B40`, `P1_PAIR_B50`, seed 0.** rho = 3, projection after every epoch. Two budget levels were fixed before any fit: BRIER 0.004 / ACC 0.003 / KL 0.008, and 0.005 / 0.0035 / 0.009.
2. **P2 (revision 1, 02:42Z, ledger 33): `P2_JR_B40`.** P1 was dominated by the scaled stored JOINT_TREE income at the same budget.
   - Hypothesis: the training dynamics (rho = 3 against a projection every epoch) cause the inefficiency.
   - Change: rho = 1 (JOINT_TREE's strength), no per-epoch projection, one end projection to the B40 budget. Same loop, people, batches and epochs as P1.
3. **P3 (revision 2, 02:46Z, ledger 46): `P3_JR_B60`.** P2 qualified with wide utility margins but a thin seed-1 c3 margin.
   - Change: only the end-projection budget, now BRIER 0.006 / ACC 0.0045 / KL 0.010.
   - The training is bitwise identical to P2 (verified: identical pre-projection trajectory).
   - The ledger decision text says "about 02:57Z"; that is a typo, and the row time 02:46:42Z is correct.

### Controls

- **`PC_TWIN_B40` (both seeds):** the matched task-only twin of P1, P2 and P3. Identical loop with lambda = 0; the critics are updated but unused. Its projection never binds, so it is the exact twin of every P configuration.
- **`PC_JSCALE_B40` and `PC_JSCALE_B60` (both seeds each):** the stored JOINT_TREE E20 income corrections scaled toward the teacher by the same budget rule, with no training. This is the matched component control for "training against the frozen accepted occupation" versus "JOINT_TREE's training against a co-evolving occupation".
- **`PC_TEMP_P2`, `PC_TEMP_P3` (seed 0):** the twin's income log-odds divided by tau. tau is chosen label-free on MODEL_FIT so the BRIER distortion equals P2's (tau 1.2984) or P3's (tau 1.3756). This is ordinary matched-cost shrinkage.
- **Not run: `PC_TSCALE_B40`** (TASK_REFIT corrections under the scaling rule, the matched task-only control of a JSCALE configuration), because no JSCALE configuration is recommended.

## 3. Numbers (MODEL_SELECT, full precision from EVAL_MS.json; ledger entry per row)

How to read the main table:
- Pair = mean over the 4 cells of `max_incl_single`.
- c1 and c3 minima are per-cell reference minus configuration; positive means the configuration reads lower.
- Income recipient max = the largest per-cell (configuration minus reference) income-view maximum over H_LJ and NEURAL_RAW; it must be ≤ 0.010.
- The occupation output is the inherited JOINT_TREE E20 output everywhere: occupation view 0.569415 / 0.580003; occupation recipient vs H_LJ 0.0, vs NEURAL_RAW −0.0498 / −0.0465.

| Config | seed | kind | ledger | pair mean | pair cells A0 / A1 / B0 / B1 | income view | c1 min vs H_LJ | c1 min vs NEURAL_RAW | c3 min vs B2TWIN | income recipient max | reserve (min ratio) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P1_PAIR_B40 | 0 | fit | 26 | 0.617984 | 0.6183 / 0.6183 / 0.6177 / 0.6177 | 0.542312 | +0.004942 | +0.009407 | -0.000833 | +0.002105 | RESERVE_MET (2.8001) |
| P1_PAIR_B50 | 0 | fit | 30 | 0.615827 | 0.6162 / 0.6171 / 0.6150 / 0.6150 | 0.539879 | +0.006228 | +0.010647 | +0.001816 | +0.000176 | RESERVE_MET (3.0125) |
| PC_TWIN_B40 | 0 | control | 28 | 0.635526 | 0.6345 / 0.6345 / 0.6366 / 0.6366 | 0.558142 | -0.011622 | -0.007694 | -0.019732 | +0.018416 | RESERVE_MET (3.7778) |
| PC_TWIN_B40 | 1 | control | 41 | 0.636276 | 0.6360 / 0.6360 / 0.6365 / 0.6365 | 0.558710 | -0.013014 | -0.002003 | -0.020958 | +0.021420 | RESERVE_MET (3.1100) |
| PC_JSCALE_B40 | 0 | control | 31 | 0.614964 | 0.6146 / 0.6155 / 0.6149 / 0.6149 | 0.538454 | +0.007797 | +0.012216 | +0.001947 | -0.000399 | RESERVE_MET (4.1777) |
| PC_JSCALE_B40 | 1 | control | 38 | 0.616910 | 0.6187 / 0.6196 / 0.6147 / 0.6147 | 0.538755 | +0.004327 | +0.014416 | -0.000787 | +0.001267 | RESERVE_MET (2.7991) |
| P2_JR_B40 | 0 | fit | 39 | 0.613303 | 0.6131 / 0.6124 / 0.6139 / 0.6139 | 0.537849 | +0.010199 | +0.014969 | +0.002932 | -0.002227 | RESERVE_MET (4.1481) |
| P2_JR_B40 | 1 | fit | 42 | 0.615664 | 0.6168 / 0.6170 / 0.6145 / 0.6145 | 0.539231 | +0.006916 | +0.017005 | +0.001136 | +0.001920 | RESERVE_MET (3.8152) |
| PC_TEMP_P2 | 0 | control | 52 | 0.636172 | 0.6353 / 0.6353 / 0.6370 / 0.6370 | 0.557859 | -0.012076 | -0.008148 | -0.020186 | +0.018411 | RESERVE_MET (3.7774) |
| **P3_JR_B60** | 0 | fit | 50 | **0.607159** | 0.6075 / 0.6073 / 0.6069 / 0.6069 | 0.532973 | +0.015771 | +0.020389 | +0.009912 | -0.007223 | RESERVE_MET (3.7246) |
| **P3_JR_B60** | 1 | fit | 51 | **0.611296** | 0.6134 / 0.6129 / 0.6098 / 0.6091 | 0.534940 | +0.011026 | +0.020616 | +0.005716 | -0.002644 | RESERVE_MET (2.6768) |
| PC_TEMP_P3 | 0 | control | 60 | 0.636261 | 0.6354 / 0.6354 / 0.6371 / 0.6371 | 0.557820 | -0.012168 | -0.008240 | -0.020277 | +0.018421 | RESERVE_MET (2.5607) |
| PC_JSCALE_B60 | 0 | control | 59 | 0.608279 | 0.6081 / 0.6080 / 0.6085 / 0.6085 | 0.533353 | +0.015209 | +0.019699 | +0.008297 | -0.005255 | RESERVE_MET (3.4025) |
| PC_JSCALE_B60 | 1 | control | 55 | 0.611959 | 0.6154 / 0.6156 / 0.6084 / 0.6084 | 0.533723 | +0.008312 | +0.018401 | +0.003199 | -0.003599 | RESERVE_MET (2.2122) |

References on the same grid (seed 0 / seed 1):
- H_LJ 0.6241 / 0.6239
- NEURAL_RAW 0.6295 / 0.6357
- B2TWIN 0.6185 / 0.6173 (cells 0.6205 / 0.6198 / 0.6168 / 0.6168 and 0.6191 / 0.6188 / 0.6156 / 0.6156)
- RETIRED_B3CTL 0.6051 / 0.6056 (reserve not met)
- H_JJ 0.5952 / 0.5968 (point-ineligible)

Income clauses (value, with household-clustered SE in brackets) and the label-free budget:

| Config | seed | acc loss vs U0 (SE) | NLL excess vs U0 (SE) | NLL excess vs Ucal (SE) | Brier excess vs U0 (SE) | Brier excess vs Ucal (SE) | label-free BRIER / ACC / KL | MS Brier increment over teacher / ratio |
|---|---|---|---|---|---|---|---|---|
| P1_PAIR_B40 | 0 | 0.002125 (0.002736) | -0.001261 (0.001866) | -0.000161 (0.001581) | -0.000016 (0.001417) | 0.000432 (0.001271) | 0.00299 / 0.00299 / 0.00374 | 0.00410 / 1.37 |
| P1_PAIR_B50 | 0 | 0.000125 (0.002831) | -0.000412 (0.001900) | 0.000688 (0.001627) | 0.000603 (0.001454) | 0.001052 (0.001311) | 0.00354 / 0.00349 / 0.00443 | 0.00472 / 1.33 |
| PC_TWIN_B40 | 0 | -0.000625 (0.002724) | -0.007106 (0.001751) | -0.006006 (0.001590) | -0.004525 (0.001349) | -0.004077 (0.001255) | 0.00033 / 0.00022 / 0.00049 | -0.00041 / n/a |
| PC_TWIN_B40 | 1 | 0.000875 (0.002934) | -0.007616 (0.001819) | -0.007341 (0.001719) | -0.005246 (0.001417) | -0.005196 (0.001352) | 0.00045 / 0.00022 / 0.00066 | -0.00073 / n/a |
| PC_JSCALE_B40 | 0 | -0.001750 (0.002811) | -0.003280 (0.001854) | -0.002180 (0.001541) | -0.001531 (0.001402) | -0.001083 (0.001247) | 0.00255 / 0.00300 / 0.00308 | 0.00259 / 1.02 |
| PC_JSCALE_B40 | 1 | 0.001625 (0.002992) | -0.003077 (0.001915) | -0.002802 (0.001683) | -0.001624 (0.001473) | -0.001575 (0.001354) | 0.00288 / 0.00299 / 0.00350 | 0.00289 / 1.00 |
| P2_JR_B40 | 0 | -0.002125 (0.002923) | -0.003651 (0.001865) | -0.002551 (0.001560) | -0.001916 (0.001417) | -0.001468 (0.001265) | 0.00268 / 0.00300 / 0.00328 | 0.00220 / 0.82 |
| P2_JR_B40 | 1 | -0.001375 (0.002981) | -0.004377 (0.001919) | -0.004102 (0.001700) | -0.002667 (0.001477) | -0.002617 (0.001365) | 0.00265 / 0.00298 / 0.00318 | 0.00185 / 0.70 |
| PC_TEMP_P2 | 0 | -0.000625 (0.002724) | -0.001356 (0.002275) | -0.000256 (0.001645) | -0.001013 (0.001592) | -0.000565 (0.001283) | 0.00268 / 0.00022 / 0.00450 | 0.00311 / 1.16 |
| **P3_JR_B60** | 0 | -0.001250 (0.003020) | -0.001959 (0.001937) | -0.000859 (0.001636) | -0.000590 (0.001481) | -0.000142 (0.001331) | 0.00414 / 0.00450 / 0.00507 | 0.00353 / 0.85 |
| **P3_JR_B60** | 1 | 0.001750 (0.003082) | -0.002816 (0.001989) | -0.002541 (0.001774) | -0.001405 (0.001539) | -0.001356 (0.001428) | 0.00413 / 0.00450 / 0.00495 | 0.00311 / 0.75 |
| PC_TEMP_P3 | 0 | -0.000625 (0.002724) | 0.001329 (0.002433) | 0.002429 (0.001756) | 0.000697 (0.001681) | 0.001145 (0.001343) | 0.00414 / 0.00022 / 0.00685 | 0.00482 / 1.16 |
| PC_JSCALE_B60 | 0 | -0.001750 (0.002894) | -0.001358 (0.001927) | -0.000258 (0.001617) | 0.000005 (0.001468) | 0.000453 (0.001314) | 0.00410 / 0.00450 / 0.00495 | 0.00413 / 1.01 |
| PC_JSCALE_B60 | 1 | 0.003250 (0.003051) | -0.001195 (0.001987) | -0.000920 (0.001756) | -0.000118 (0.001535) | -0.000068 (0.001417) | 0.00440 / 0.00449 / 0.00536 | 0.00439 / 1.00 |

Notes on this table:
- Accuracy loss vs Ucal equals accuracy loss vs U0 in every row.
- Increment = Brier excess vs Ucal minus the teacher's −0.0036718 / −0.0044628.
- Every configuration passes all 12 clauses and both guards at the point on every seed it was run.
- The guards and occupation clauses are the inherited JOINT_TREE ones in every row (men's guard 0.004417, SE 0.001814 / 0.003941, SE 0.001697).

P3 per-cell detail. Pair columns are reference minus P3; income columns are P3 minus reference.

| seed | cell | pair | vs H_LJ | vs NEURAL_RAW | vs B2TWIN | income vs H_LJ | income vs NEURAL_RAW | winner |
|---|---|---|---|---|---|---|---|---|
| 0 | A0 | 0.6075 | +0.0158 | +0.0228 | +0.0130 | -0.0096 | -0.0072 | pair ET25 |
| 0 | A1 | 0.6073 | +0.0160 | +0.0204 | +0.0125 | -0.0096 | -0.0072 | pair ET25 |
| 0 | B0 | 0.6069 | +0.0180 | +0.0242 | +0.0099 | -0.0099 | -0.0076 | pair OUT_LR |
| 0 | B1 | 0.6069 | +0.0180 | +0.0219 | +0.0099 | -0.0099 | -0.0076 | pair OUT_LR |
| 1 | A0 | 0.6134 | +0.0111 | +0.0206 | +0.0057 | -0.0088 | -0.0026 | pair ET25 |
| 1 | A1 | 0.6129 | +0.0110 | +0.0211 | +0.0059 | -0.0088 | -0.0026 | pair ET25 |
| 1 | B0 | 0.6098 | +0.0137 | +0.0276 | +0.0058 | -0.0056 | -0.0027 | pair ET25 |
| 1 | B1 | 0.6091 | +0.0144 | +0.0283 | +0.0065 | -0.0056 | -0.0027 | pair ET25 |

**Simultaneous utility bounds** (value + 3.1237346 × own SE; reported, not a gate):
- P3: 8 of 28 unresolved. Occupation NLL vs U0 and vs Ucal on both seeds, and the men's guard on both seeds, are inherited. The other two are seed-1 income accuracy loss vs U0 and vs Ucal (0.011378).
- P2: 6 of 28 unresolved, all inherited.
- PC_JSCALE_B60: 8 of 28, the same pattern as P3.

## 4. What each step taught; which effects come from what

- **Task learning alone gives no privacy.** The twin (task-only corrections on the same people) slightly improves income utility over the teacher: Brier vs Ucal −0.004077 / −0.005196 against the teacher's −0.003672 / −0.004463. But it reads 0.635526 / 0.636276 on the pair, above H_LJ, and fails the income recipient screen (+0.018416 / +0.021420).
- **Ordinary shrinkage is not the explanation.** At P2's and P3's label-free cost, temperature on the twin reads 0.636172 and 0.636261 on seed 0, i.e. no change from the twin. It also costs more MODEL_SELECT Brier than P2 / P3 at the same label-free distortion (ratio 1.16, against 0.82 / 0.85). P3 minus PC_TEMP_P3 on seed 0 is −0.029102 on the pair.
- **Ordinary compression is beaten.** B2TWIN (the 6-level LOCAL code with CALIBRATION per-level probabilities) reads 0.6185 / 0.6173 at Brier vs Ucal +0.00095 / −0.00019. P3 reads 0.0057–0.0130 lower in every cell at Brier vs Ucal −0.000142 / −0.001356.
- **P1 was inefficient because of its training dynamics.**
  - P1 (rho = 3, projection every epoch) did real critic work: pair −0.0175 against its twin.
  - But its MODEL_SELECT cost was 1.37× (B40) and 1.33× (B50) its label-free distortion, and it only reached B2TWIN's level.
  - Revision 1 showed the dynamics are responsible. The same objective at rho = 1, trained freely and projected once at the end (P2), costs 0.82 / 0.70× its label-free distortion. On seed 0 it reads 0.004681 lower than P1_PAIR_B40 with a 0.0019 smaller Brier increment.
  - Interpretation (not tested further): strong critic pressure against a per-epoch projection keeps injecting in-sample, critic-chasing corrections that do not generalize.
- **The label-free budget is the engineering component that matters.**
  - It predicted MODEL_SELECT Brier increments at 0.70–1.02× for every end-projected or scaled configuration, and over-predicted for P2 / P3.
  - The ACC budget bound in every configuration: expected accuracy loss tracks decision flips near the boundary.
  - MODEL_SELECT accuracy was noisy around it: P2 / P3 improved accuracy on seed 0, while P3 lost 0.0024 against the teacher on seed 1.
  - No label of MODEL_FIT, CALIBRATION or MODEL_SELECT was used to set or enforce it. The P1 budget levels were fixed before any fit; P3's level was chosen after logged MODEL_SELECT feedback (an adaptive choice).
- **The family's specific step adds little beyond the scaled stored JOINT_TREE income.** That step is training against the frozen accepted occupation output. Matched-budget comparisons, configuration minus PC_JSCALE at the same budget:

  | Budget | seed | pair difference (cells lower) | income Brier vs Ucal difference | accuracy loss difference |
  |---|---|---|---|---|
  | B40 (P2) | 0 | -0.001661 (4 of 4) | -0.000385 | -0.000375 |
  | B40 (P2) | 1 | -0.001246 (4 of 4) | -0.001042 | -0.003000 |
  | B60 (P3) | 0 | -0.001120 (4 of 4) | -0.000595 | +0.000500 |
  | B60 (P3) | 1 | -0.000663 (2 of 4) | -0.001288 | -0.001500 |

  - The pair mean and income Brier differences are consistent in sign but small, and within the grid's reader variability.
  - At a matched budget, the end projection is what turns either pair-trained direction into an eligible release with reserve.
  - PC_JSCALE at B40 fails c3 on seed 1 (cell A1: 0.6196 vs 0.6188). At B60 it qualifies by my reading.
- **No recovery moved to a recipient.** P3's income view (0.532973 / 0.534940) is below H_LJ's (0.5427 / 0.5421) and NEURAL_RAW's (0.5404 / 0.5376). The occupation recipient is unchanged.

## 5. Recommendation

**Recommend `P3_JR_B60` as family P's candidate.**
- Its controls:
  - matched task-only twin `PC_TWIN_B40`, recorded as `matched_twin` in P3's RELEASE.json on both seeds;
  - matched component control `PC_JSCALE_B60`, both seeds;
  - shrinkage control `PC_TEMP_P3`, seed 0.
- By my dry run of the frozen nomination rule it qualifies, and it has the best score in this family (two-seed mean 0.609228). The rule's order also picks it:
  - P2_JR_B40 (0.614483) is outside P3's 0.005 near-tie window, by 0.005255.
  - PC_JSCALE_B60 (0.610119) is inside the window but has the smaller min income ratio (2.2122 vs 2.6768).
- **P2_JR_B40 is the conservative alternative.** Its min income ratio is 3.8152 and its only unresolved utility bounds are inherited, but its seed-1 c3 margin is just +0.001136.

**Claims I would not make.**
- No mechanism attribution to "training against the frozen accepted occupation" beyond the small matched-budget differences above. The attribution that survives is "pair-trained income correction plus explicit label-free distortion budget", against task-only learning, temperature shrinkage and quantization.
- P3's budget was chosen after MODEL_SELECT feedback on P2 (adaptive).
- The binding reserve condition (seed-1 income accuracy, ratio 2.6768) is the noisiest clause.
- The inherited occupation constraints still bind (men's guard 0.32 / 0.62 SE from the cap), and no income-side mechanism changes them.
- The grid's readers are trained on less data than the final ATTACK_FIT readers.

**Packaging.**
- Rebuild from features with `ihs.fam_p.predict(seed, 'P3_JR_B60', X, names)`. All six roles are saved.
- Parameters: `.private/pcrl_income_headroom_successor_v1/runs/P/P3_JR_B60/S<s>/THETA.npz` (theta_income, already scaled; sha256 prefixes 3a43ba61f99e39de / 818479a2a9f27453).
- Release sha256 prefixes: f40c8a74f95093e9 / a818db786461ac28.
- The P1 records were fitted before the `matched_twin` key existed and lack it; their matched twin is PC_TWIN_B40.

## 6. Resources and counts

- **Configurations: 14 of 14**, each with `counts_toward_36` = 1:
  - P1_PAIR_B40 S0, P1_PAIR_B50 S0
  - PC_TWIN_B40 S0 + S1, PC_JSCALE_B40 S0 + S1
  - P2_JR_B40 S0 + S1, PC_TEMP_P2 S0
  - P3_JR_B60 S0 + S1, PC_TEMP_P3 S0, PC_JSCALE_B60 S0 + S1
- **Versions:** initial (P1) plus 2 revisions (P2, P3). Ledger decisions: entries 33 (P1 → P2), 46 (P2 → P3) and 65 (stop).
- **Individual model fits: 86.**
  - 8 trained configurations × 10 fits (income corrections 1 + 9 online critics; the twin's critics are updated but unused) = 80. The trained configurations are P1_PAIR_B40, P1_PAIR_B50, PC_TWIN_B40 ×2, P2_JR_B40 ×2 and P3_JR_B60 ×2.
  - 6 no-training controls × 1 (4 scale choices, 2 temperatures) = 6.
  - Reader fits are counted by the evaluator (168 requested per evaluation, cached).
- **CPU:** 34 queue units, 0 failed. Fits 403.1 s, scoring 304.4 s, self-tests 32.2 s, diagnostic 1.6 s; total 741.3 CPU-s (0.206 CPU-h). An agent upper estimate of 120 CPU-s was charged as IHS_P_AGENT_ANALYSIS. The allotment was about 2 CPU-h.
- **Invented-data checks:** `python -m ihs.fam_p selftest`, units IHS_P_SELFTEST_V1–V5; final run 22 of 22 PASS.
- **Pre-registration diagnostic:** unit IHS_P_DIAG_REFDIST_V1 (02:26Z) ran a label-free reference-distortion check on features and stored outputs only. It ran before the registration commit at 02:35Z; it read no labels and fitted nothing.
