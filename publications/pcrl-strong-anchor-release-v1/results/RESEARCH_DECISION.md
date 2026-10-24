# Research decision: strong-anchor release campaign (`pcrl_strong_anchor_release_v1`, SAR1)

First written 2026-10-24 from 06:55Z (`date -u`). Revised from 07:30Z after the statistical reviewer's claims review
(`roles/statistical_reviewer/CLAIMS_REVIEW.md`: 12 required corrections, all applied, and 11 recommended changes, all adopted) and
verifier wave 3.

All results are development evidence on the Florida ACS 2019 development cohort. Every recovery number is read on adaptively reused
MODEL_SELECT people for the declared reader bank. Intervals do not correct the adaptive search, and nothing here is a confirmation.

Conventions and sources:
- Benefit = comparator pair AUC − candidate pair AUC; positive favours the candidate. `se` is the rules' se_agg.
- Records: `ENDPOINTS.json` (machine-readable), `TABLES.md`, `claims/*/DECISIONS.json` and `claims/CAMPAIGN_SUMMARY.json`.
- The D2R robustness labels come from `claims/*_D2R/D2R.json`, written after the write-once decision records. Those decision records
  still read "D2R NOT_RUN".

## Decision

**No competitive release method is established.**

- **No reading met D2, D3 or D4.**
  - D2 is a reliable gain over the release's own start and the matched controls.
  - D3 is superiority to the strong references.
  - D4 is the material target.
- **Both D2 looks were non-registered (adaptive) revisions, FB_R1 and FA_R2.** The registered readings FA_REG and FB_REG found no
  feasible strength, so no registered reading reached D2. D3 and D4 were met in none of the six readings. No multiplicity correction
  is applied (`claims/CAMPAIGN_SUMMARY.json`).
- **Two releases are packaged, but neither is a method claim.** FA_TFRB_R2 and FB_PFO met all 28 usefulness conditions at the point.
  That is point eligibility only: 27 of 28 conditions pass the confidence conjunction, and neither is reserve-qualified.
- **FB's small edge is not stable.** FB_PFO's edge over NEURAL_RAW's own recipe continued to matched spend (ALT) changes sign on the
  robustness split: +0.0012 registered, −0.0012 robustness, se 0.0070 (D2R `ROBUSTNESS_NOT_CONFIRMED`).

## The three outcomes, separately

1. **Reliable component gain: none.**
   - **FA_TFRB_R2** (FA, revision 2; start SUCC): `ADAPTIVE_R2__PARTIAL_CONT_ONLY`.
     - Against SUCC: +0.0066 (se 0.0141), point only, with the occupation recipient gate failing.
     - Against the untargeted control at its bracket (FA_R2_CTRL_HI): +0.0066 (0.0140), `NOT_SEPARATED_FROM_TASK_SPENDING`. In version
       1 the bracketing control point is SUCC itself, so there the control comparison repeats the START comparison. Only version 0 is
       compared with a distinct untargeted control.
     - Only the equally exposed zero-protection continuation FA_TFRO_R1_CONT (CONT) is beaten: +0.0432 (0.0183). That is an
       attribution against the weak zero-protection twin, not a component gain.
   - **FB_PFO** (FB, revision 1; start NEURAL_RAW): `ADAPTIVE_R1__PARTIAL_START+CONT_ONLY`.
     - Against NEURAL_RAW: +0.0125 (0.0056), resolved.
     - Against the equally exposed zero-protection continuation FB_PFO_TASK (CONT): +0.0143 (0.0055). That is an attribution against
       the weak zero-protection twin, not a component gain.
     - It is not separated from NEURAL_RAW's own recipe continued to matched spend (ALT): +0.0012 (0.0074) at the bracket. The bracket
       spends about 0.0008 nats more total NLL, and FB_PFO's income view fails the recipient gate there.
     - The SEX-blind quantization control had no bracketing point: its saved upper point spends less than the candidate. FB_R1 saved
       only the upper CTRL and ALT points (deviation 7), so the frontier's cheaper-point check covers only the origin.
2. **Strong references under the original conditions: not superior** (D3 `NOT_SUPERIOR` in both point-eligible readings).
   - **FA_TFRB_R2 passes only W, TR100 and TT100.**
     - NEURAL_RAW +0.0241 (0.0191), H_LJ +0.0208 (0.0145), B2TWIN +0.0119 (0.0143) and SUCC +0.0066 (0.0141) are point only. Against
       H_LJ, B2TWIN and SUCC the occupation recipient gate also fails; against NEURAL_RAW it passes.
     - RETIRED_B3CTL −0.0014 (0.0158).
     - TWJ +0.0313 (0.0144) is resolved, but the occupation recipient gate fails.
   - **FB_PFO passes only NEURAL_RAW, its own start.**
     - H_LJ +0.0091 (0.0096), SUCC −0.0050 (0.0093), RETIRED_B3CTL −0.0130 (0.0106).
     - It is resolved against TR100, TT100, TWJ and W but fails their recipient gates.
3. **Material target: not established.** The target is a resolved 0.020 against NEURAL_RAW and H_LJ.
   - FA_TFRB_R2 `ADAPTIVE_R2__MATERIAL_0020_POINT_ONLY`. Both pooled point estimates exceed 0.020, but the version-0 means are +0.0078
     and +0.0090, and mean − 2 se is far below 0.020.
   - FB_PFO `ADAPTIVE_R1__NOT_MET`.

The usefulness-margin reading (UM) is preparatory evidence only:
- **FA_TFRB_R2:** `ADAPTIVE_R2__MARGIN_GAIN_PROFILE_NOT_RETAINED`.
  - SUCC's binding men's occupation guard improves, just resolved on reused people: pooled −0.0029, se 0.0013, upper bound −0.0002.
    Version 0 is −0.0053 and version 1 only −0.0004.
  - The occupation view is +0.0252 above SUCC's in version 0, and −0.0141 in version 1.
- **FB_PFO:** `ADAPTIVE_R1__MARGIN_NOT_RESOLVED`.

## Robustness on a different household reader split (D2R; descriptive, no retuning)

Repetitions R3 / R4 with reader seeds k2 / k3. The refit was genuinely effective in both banks.
- **FA_R2:** `ROBUSTNESS_NOT_CONFIRMED`, no sign change.
  - SUCC +0.0034 (0.0139) and CTRL_HI +0.0040 (0.0134).
  - CONT FA_TFRO_R1_CONT is confirmed at +0.0374 (0.0156). That is an attribution against the weak zero-protection twin, not a
    component gain.
  - Beside: NEURAL_RAW +0.0183 (0.0177) and H_LJ +0.0158 (0.0135). Both are below 0.020 at the point on these cells.
- **FB_R1:** `ROBUSTNESS_NOT_CONFIRMED`.
  - NEURAL_RAW +0.0119 (0.0067), point only.
  - CONT FB_PFO_TASK is confirmed at +0.0140 (0.0064). That is an attribution against the weak zero-protection twin.
  - **The ALT comparison changes sign** (+0.0012 registered, −0.0012 robustness, se 0.0070), carried into the headline above.

## Readings

| Reading | Card | Start | Candidate (versions 0 / 1) | What decided it | Class |
|---|---|---|---|---|---|
| FA_REG | FA_TFR_v1 | SUCC | none | The first strength (rho 0.3) fails income Brier (version 1 also income NLL and occupation accuracy) and both guards on the selection data | `TASK_INFEASIBLE` (NO_FEASIBLE_STRENGTH) |
| FB_REG | FB_PFD_v1 | NEURAL_RAW | none (version 0 had k* = E1) | Version 1's first checkpoint fails income Brier on the MODEL_FIT out-of-fold selection data (bound 0.00383 > 0.0025) | `TASK_INFEASIBLE` (NO_FEASIBLE_STRENGTH) |
| FB_R1 | FB_PFO_R1 (adaptive) | NEURAL_RAW | FB_PFO, S50 / S50 | Point-feasible (G1); D2 START + CONT only; ALT not separated | `ADAPTIVE_R1__FEASIBLE_NO_RELIABLE_GAIN` |
| FA_R1 | FA_TFRO_R1 (adaptive) | SUCC | FA_TFRO_R1, rho 1.0 / 0.6 | Version 1's men's guard on MODEL_SELECT +0.0099 (se 0.0020) > 0.005 | `ADAPTIVE_R1__TASK_INFEASIBLE` |
| FS_CPI | FS_CPI_v1 (replacement family, adaptive) | SUCC | none (version 1 had lambda 0.05) | Version 0's first strength fails CALIBRATION income accuracy (bound 0.0052 > 0.0040) | `ADAPTIVE_FS__TASK_INFEASIBLE` |
| FA_R2 | FA_TFRB_R2 (adaptive) | SUCC | FA_TFRB_R2, rho 0.6 / 1.0 | Point-feasible (G1); D2 CONT only; D4 point only | `ADAPTIVE_R2__FEASIBLE_NO_RELIABLE_GAIN` |

Recovery at the registered cells (MODEL_SELECT SCORE AUC of the registered winner; version 0 / version 1):

| Release | Pair | Income view | Occupation view |
|---|---|---|---|
| FA_TFRB_R2 | 0.6225 / 0.5975 | 0.5259 / 0.5323 | 0.5996 / 0.5651 |
| SUCC (start A) | 0.6178 / 0.6155 | 0.5259 / 0.5323 | 0.5743 / 0.5792 |
| FB_PFO | 0.6192 / 0.6241 | 0.5387 / 0.5363 | 0.6071 / 0.6120 |
| NEURAL_RAW (start B) | 0.6303 / 0.6380 | 0.5387 / 0.5363 | 0.6266 / 0.6324 |
| H_LJ | 0.6315 / 0.6301 | 0.5486 / 0.5388 | 0.5743 / 0.5792 |
| RETIRED_B3CTL | 0.6083 / 0.6089 | 0.5310 / 0.5307 | 0.5743 / 0.5792 |

Usefulness on MODEL_SELECT, under the unchanged caps:
- **FA_TFRB_R2** meets all 28 conditions at the point (G1).
  - Worst value over cap 0.012 / 0.698.
  - Men's guard −0.0008 / +0.0035, against SUCC's +0.0044 / +0.0039 and the 0.005 cap.
  - 27 of 28 conditions pass the confidence conjunction (release label CONFIDENCE_UNRESOLVED; version 1's men's guard is unresolved).
    SUCC passes 26 of 28.
- **FB_PFO** meets all 28 conditions at the point.
  - Worst value over cap 0.601 / 0.785.
  - 27 of 28 conditions pass the confidence conjunction (CONFIDENCE_UNRESOLVED; version 1's men's guard).
- Point feasibility is not confidence preservation, and neither release is reserve-qualified.

## What was tested, and the limits

**FA: three cards on SUCC.**
- **FA_TFR** (revision 0) is a SUCC-anchored joint teacher-feature residual. It moved both outputs, distilled toward the admitted HGB
  teachers' out-of-fold probabilities, with online SEX critics (`acs_replication.leafcorr.Critics`: linear and MLP critics on the
  income, occupation and pair outputs).
- **FA_TFRO_R1** (revision 1) moves the occupation output only, and SUCC's income output stays bitwise. Its guard estimate is
  L_cal + (1 − p)·C_mf, with budgets equal to SUCC's own margins.
- **FA_TFRB_R2** (revision 2) also keeps SUCC's income bitwise. It deploys the five-fold cross-fit bag, so the guard contrast is
  measured on the deployed object and its SE includes the between-fold spread.
  - At the selected strengths the between-fold SE was the binding one: 0.0034 / 0.0025, against a household SE of 0.0014 / 0.0013
    (verifier wave 2).
  - In version 0 the lower strength (rho 0.6) kept the distillation's usefulness gain but not SUCC's occupation protection.

**FB: two cards on NEURAL_RAW.**
- **FB_PFD** is pair Fisher-discriminant training of the task networks: a penalty on the SEX separation of whitened degree-2 features
  of both outputs' logit differences.
- **FB_PFO** applies it to the occupation network only. Its income output is frozen at NEURAL_RAW's, equal within 7e-8 but not bitwise
  (deviation 10).
- FB was stopped after revision 1 by the coordinator's decision, with one revision unused, because its gain was not separated from
  ALT. No PROTOCOL 9 stop rule required it.

**FS: the replacement family, one card on SUCC.**
- **FS_CPI** is a conditional pair-increment residual on SUCC's income. Occupation is frozen, and the residual is trained against FB_PFD's
  closed-form whitened quadratic best response.
- CALIBRATION accuracy noise from decision flips bound version 0 at its first strength, so no candidate exists.
- A decision-preserving revision was declined on investigator B's stop memo (`roles/investigator_b/HANDBACK_FS_STOP.md`).
  - The memo predicted a pair benefit over SUCC of about 0.005 (80% interval 0.000–0.011).
  - It predicted the part not reproduced by SUCC's own operation (ALT) at about 0.001–0.004 on the fitting data, so D2 could not be
    met.

**Not tested.**
- A joint design: no local twin or sequential orders were built, and D5 is not applicable.
- The previously declined cross-recipient swap.
- Any repeated-query or model-access setting.

These results do not show that other mechanisms cannot work.

## What prevents the next claim

1. **FA_TFRB_R2's version heterogeneity.**
   - The pooled pair benefit over SUCC is +0.0066 with se_agg 0.014. Version 0 is −0.0047 and version 1 +0.0179.
   - D2 needs mean − 2 se > 0 against SUCC and the bracketing control, with the recipient gates.
   - Version 0's strength was bounded by the guard estimate's between-fold uncertainty, and at that strength its occupation view rose
     by +0.025.
2. **The material target needs much more than the best release delivers.**
   - It requires mean − 2 se_agg > 0.020 and both version means > 0.020 against NEURAL_RAW and H_LJ.
   - With se_agg near 0.015–0.019, that means a pair AUC of about 0.58 or lower in both versions.
   - The best version here is 0.5975, and the other is 0.6225.
3. **FB's targeting is not separated from its start's own recipe.** That recipe, continued to matched spend, is not separated from
   FB_PFO, and the sign of the difference flips across reader splits.

## Plain answers

1. **What operation was genuinely different?**
   - FA_TFRB_R2 is an occupation residual on SUCC, trained by distillation toward the admitted HGB teachers against online SEX critics.
     It is deployed as its five-fold cross-fit bag, so the subgroup-guard contrast that sets its strength is measured on the deployed
     object, including run-to-run variance.
   - FB_PFO is occupation-only pair Fisher-discriminant training of NEURAL_RAW's network.
   - FS_CPI is a conditional pair-increment residual on SUCC's income, with occupation frozen.
   - FB_PFO turned out not to be separated from NEURAL_RAW's own recipe run further.
2. **Which strong starting release and binding condition did it address?**
   - FA addressed SUCC and its men's occupation guard (+0.0044 / +0.0039 of 0.005).
   - FB addressed NEURAL_RAW, whose pair recovery sits in its occupation view and whose usefulness margins are wide.
   - FS addressed SUCC's income increment: the pair AUC sits 0.034 above SUCC's occupation view on the input-and-pair bank.
3. **Did both tasks and subgroup conditions survive?**
   - FA_TFRB_R2 and FB_PFO met all 28 conditions at the point. For each, 27 of 28 pass the confidence conjunction (version 1's men's
     guard is unresolved), and neither is reserve-qualified.
   - FA_TFRO_R1 failed version 1's men's guard on MODEL_SELECT.
   - FA_TFR, FB_PFD and FS_CPI had no feasible strength in at least one version (FB_PFD version 0 had E1, FS_CPI version 1 had lambda
     0.05). A reading needs both versions.
4. **Did recovery improve beyond its own start and matched controls?**
   - No. FA_TFRB_R2 is +0.0066 (0.0141) over SUCC and +0.0066 (0.0140) over the untargeted control, both unresolved. In version 1 that
     control point is SUCC itself.
   - FB_PFO is +0.0125 (0.0056) over NEURAL_RAW. That gain is not separated from NEURAL_RAW's own continued recipe, and the
     quantization control had no bracket.
5. **Did it beat the strong references under the original criterion?** No. D3 is NOT_SUPERIOR for both, and D4 is not met.
   FA_TFRB_R2's +0.0241 / +0.0208 over NEURAL_RAW / H_LJ are point only.
6. **Did a component, joint-design or novelty claim earn evidence?** No component gain: D2 was not met in either look, and both looks
   were adaptive. Joint design was not claimed, and no twin or sequential orders were built (D5). Novelty was not claimed (D6).
7. **What survived independent reader refitting?**
   - The labels and numbers the verifier recomputed with its own code reproduced.
     - Waves 1-2 ran 256 checks, 253 PASS. The 3 FAILs concern two missing plan hashes and one card's bitwise claim.
     - The CALIBRATION spends were re-applied, not recomputed.
     - Its reader refits were bitwise.
   - On the different household reader split, only the comparisons against the zero-protection continuations were confirmed: weak-twin
     attributions, not component gains.
   - FB's edge over its own-recipe control changed sign.
   - FA_TFRB_R2's point gains over NEURAL_RAW and H_LJ fell below 0.020.
8. **What exact uncertainty or failure prevents the next claim?**
   - **FA_TFRB_R2:** version 0 is weaker than SUCC (pair −0.0047; occupation view +0.025). Its strength was bounded by the guard
     estimate's between-fold SE (0.0034, about 2.5 times the household SE). With se_agg about 0.014, D2 against SUCC and the bracketing
     control needs a pooled benefit above about 0.028 with both version means positive and the recipient gates passed.
   - **The material target:** it needs both version means above 0.020 and mean − 2 se above 0.020. FA_TFRB_R2's version-0 benefits are
     +0.0078 / +0.0090.
   - **FB:** its targeting is not separated from its start's own recipe.

## Integrity notes

**Ordering.**
- The registered cards FA_TFR_v1 and FB_PFD_v1 (05:10:48Z), the limits, the comparison definitions and the access boundary were
  committed before the first label computation. The public registration receipt was verified at 05:11:39Z; the first label read was at
  05:11:51Z.
- Each later card was committed before its own first fit, after earlier MODEL_SELECT outcomes, hence ADAPTIVE: FB_PFO_R1 05:32:12Z,
  FA_TFRO_R1 05:37:57Z, FA_TFRB_R2 06:08:14Z, FS_CPI_v1 06:12:17Z.
- Each reading's plan was committed before that reading's first MODEL_SELECT computation.

**Deviations.** Fourteen are recorded, append-only, in `PROTOCOL_DEVIATIONS.md`, and none changed a label.
- Items 11-12 are verifier-file drift recorded in coordinator units.
- Item 13: the public registration receipt carries the owner's name and host name. The final publication redacts them, but the receipt
  commit is unchanged; rewriting public history is the owner's decision.
- Item 14: deployment equivalence is stated as a 6e-16 tolerance with identical decisions. Outputs are bitwise for the full evaluated
  batch.

**Verification.** Three waves ran 365 checks: 355 PASS and 10 FAIL. The failures are kept and adjudicated:
- 3 in wave 1: two plan hashes missing from write-once records (sidecars now hold them) and one card statement of bitwise equality that
  holds only to 7e-8 (deviations 9-10).
- 6 in wave 3: FB packages not bitwise on 50-row requests, differences at most 5.6e-16 with identical decisions (deviation 14).
- 1 in wave 3: the owner's name in the receipt files (deviation 13).

Reader refits reproduced bitwise: 6 of 6, 6 of 6, and 2 of 2 robustness-cell refits.
