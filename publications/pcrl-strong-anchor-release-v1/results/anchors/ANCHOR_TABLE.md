# Anchor table (campaign `pcrl_strong_anchor_release_v1`; label-free, from saved records)

One reader bank: IPR1_MAIN (the input-and-pair campaign's registered grid; 8 registered cells, admitted 16-recipe bank, validation-selected winner SCORE AUC on the 8,001 MODEL_SELECT people). Utility: that campaign's MODEL_SELECT records, 12 task clauses + 2 within-SEX occupation guards per version (24 + 4 = 28), original caps (accuracy loss 0.010, NLL excess 0.010, summed Brier excess 0.005, guards 0.005 vs U0). Development evidence on adaptively reused people; lower recovery for these readers is not protection against every reader.

## 1. Recovery (AUC, mean (se_agg) over the 8 cells; per-version means in the JSON)

| Release | Pair | Income view | Occupation view | Pair - max single view |
|---|---|---|---|---|
| RETIRED_B3CTL | 0.5964 (0.0126) | 0.5255 (0.0103) | 0.5731 (0.0118) | 0.0232 |
| B2TWIN | 0.6061 (0.0124) | 0.5368 (0.0110) | 0.5731 (0.0118) | 0.0330 |
| SUCC | 0.6074 (0.0113) | 0.5312 (0.0126) | 0.5731 (0.0118) | 0.0343 |
| H_LJ | 0.6186 (0.0123) | 0.5373 (0.0116) | 0.5731 (0.0118) | 0.0455 |
| DDR_PTC_R1 | 0.6239 (0.0115) | 0.5359 (0.0149) | 0.6111 (0.0119) | 0.0128 |
| NEURAL_RAW | 0.6253 (0.0110) | 0.5336 (0.0124) | 0.6253 (0.0114) | -0.0000 |
| TWJ | 0.6359 (0.0118) | 0.5607 (0.0136) | 0.5731 (0.0118) | 0.0627 |
| W | 0.6459 (0.0112) | 0.5312 (0.0126) | 0.6443 (0.0118) | 0.0015 |
| TT100 | 0.6454 (0.0113) | 0.5312 (0.0126) | 0.6472 (0.0109) | -0.0018 |
| TR100 | 0.6479 (0.0115) | 0.5312 (0.0126) | 0.6483 (0.0113) | -0.0004 |
| U0 | 0.6383 (0.0108) | 0.5501 (0.0139) | 0.6352 (0.0116) | 0.0031 |
| Ucal | 0.6382 (0.0108) | 0.5501 (0.0139) | 0.6358 (0.0113) | 0.0023 |
| CTL_TWIN | 0.6480 (0.0108) | 0.5498 (0.0175) | 0.6443 (0.0118) | 0.0037 |

## 2. Paired pair benefit against the material-target references (benefit = comparator AUC - release AUC; positive favours the release; se_agg; label of the inherited screen; 0.020 target screen)

| Release | vs NEURAL_RAW | vs H_LJ | vs U0 |
|---|---|---|---|
| RETIRED_B3CTL | 0.0289 (0.0139) AGG_POSITIVE; target TARGET_AGG_POINT_ONLY | 0.0222 (0.0111) AGG_POSITIVE; target TARGET_AGG_POINT_ONLY | 0.0420 (0.0130) AGG_POSITIVE; target TARGET_AGG_POINT_ONLY |
| B2TWIN | 0.0192 (0.0135) AGG_POSITIVE_POINT_ONLY; target TARGET_AGG_NOT_MET | 0.0125 (0.0069) AGG_POSITIVE_POINT_ONLY; target TARGET_AGG_NOT_MET | 0.0322 (0.0128) AGG_POSITIVE; target TARGET_AGG_POINT_ONLY |
| SUCC | 0.0179 (0.0105) AGG_POSITIVE_POINT_ONLY; target TARGET_AGG_NOT_MET | 0.0112 (0.0074) AGG_POSITIVE_POINT_ONLY; target TARGET_AGG_NOT_MET | 0.0309 (0.0106) AGG_POSITIVE; target TARGET_AGG_POINT_ONLY |
| H_LJ | 0.0067 (0.0124) AGG_POSITIVE_POINT_ONLY; target TARGET_AGG_NOT_MET | - | 0.0197 (0.0121) AGG_POSITIVE_POINT_ONLY; target TARGET_AGG_NOT_MET |
| DDR_PTC_R1 | 0.0014 (0.0109) AGG_POSITIVE_POINT_ONLY; target TARGET_AGG_NOT_MET | -0.0053 (0.0104) AGG_NOT_POSITIVE; target TARGET_AGG_NOT_MET | 0.0144 (0.0099) AGG_POSITIVE_POINT_ONLY; target TARGET_AGG_NOT_MET |
| NEURAL_RAW | - | -0.0067 (0.0124) AGG_NOT_POSITIVE; target TARGET_AGG_NOT_MET | 0.0130 (0.0047) AGG_POSITIVE; target TARGET_AGG_NOT_MET |
| TWJ | -0.0106 (0.0103) AGG_NOT_POSITIVE; target TARGET_AGG_NOT_MET | -0.0173 (0.0090) AGG_NOT_POSITIVE; target TARGET_AGG_NOT_MET | 0.0025 (0.0099) AGG_POSITIVE_POINT_ONLY; target TARGET_AGG_NOT_MET |
| W | -0.0206 (0.0095) AGG_NOT_POSITIVE; target TARGET_AGG_NOT_MET | -0.0273 (0.0100) AGG_NOT_POSITIVE; target TARGET_AGG_NOT_MET | -0.0076 (0.0085) AGG_NOT_POSITIVE; target TARGET_AGG_NOT_MET |
| TT100 | -0.0201 (0.0090) AGG_NOT_POSITIVE; target TARGET_AGG_NOT_MET | -0.0268 (0.0104) AGG_NOT_POSITIVE; target TARGET_AGG_NOT_MET | -0.0071 (0.0084) AGG_NOT_POSITIVE; target TARGET_AGG_NOT_MET |
| TR100 | -0.0226 (0.0098) AGG_NOT_POSITIVE; target TARGET_AGG_NOT_MET | -0.0294 (0.0112) AGG_NOT_POSITIVE; target TARGET_AGG_NOT_MET | -0.0096 (0.0085) AGG_NOT_POSITIVE; target TARGET_AGG_NOT_MET |
| U0 | -0.0130 (0.0047) AGG_NOT_POSITIVE; target TARGET_AGG_NOT_MET | -0.0197 (0.0121) AGG_NOT_POSITIVE; target TARGET_AGG_NOT_MET | - |
| Ucal | -0.0129 (0.0043) AGG_NOT_POSITIVE; target TARGET_AGG_NOT_MET | -0.0196 (0.0121) AGG_NOT_POSITIVE; target TARGET_AGG_NOT_MET | 0.0002 (0.0019) AGG_POSITIVE_POINT_ONLY; target TARGET_AGG_NOT_MET |
| CTL_TWIN | -0.0227 (0.0072) AGG_NOT_POSITIVE; target TARGET_AGG_NOT_MET | -0.0294 (0.0109) AGG_NOT_POSITIVE; target TARGET_AGG_NOT_MET | -0.0097 (0.0072) AGG_NOT_POSITIVE; target TARGET_AGG_NOT_MET |

## 3. Utility (per version v0 / v1: max value / cap over the 28; binding condition; whole and occupation-side two-SE reserve ratio; men's and women's occupation Brier excess vs U0 (cap 0.005))

| Release | max/cap v0 / v1 | whole binding v0 / v1 | reserve whole v0 / v1 | men's guard v0 / v1 (se) | women's guard v0 / v1 | income NLL vs U0 / Ucal v0 | occ NLL vs U0 / Ucal v0 | distinct income / occ vectors v0 |
|---|---|---|---|---|---|---|---|---|
| RETIRED_B3CTL | 0.883 / 0.883 | s0|guard|sex0 / s1|income|Ucal|brier_excess | 0.32 / 0.35 | +0.0044 (0.0018) / +0.0039 (0.0017) | -0.0049 (0.0017) / -0.0030 (0.0017) | +0.0055 / +0.0066 | +0.0025 / +0.0034 | 6 / 7500 |
| B2TWIN | 0.883 / 0.788 | s0|guard|sex0 / s1|guard|sex0 | 0.32 / 0.62 | +0.0044 (0.0018) / +0.0039 (0.0017) | -0.0049 (0.0017) / -0.0030 (0.0017) | +0.0013 / +0.0024 | +0.0025 / +0.0034 | 6 / 7500 |
| SUCC | 0.883 / 0.788 | s0|guard|sex0 / s1|guard|sex0 | 0.32 / 0.62 | +0.0044 (0.0018) / +0.0039 (0.0017) | -0.0049 (0.0017) / -0.0030 (0.0017) | -0.0047 / -0.0036 | +0.0025 / +0.0034 | 7484 / 7500 |
| H_LJ | 0.883 / 0.788 | s0|guard|sex0 / s1|guard|sex0 | 0.32 / 0.62 | +0.0044 (0.0018) / +0.0039 (0.0017) | -0.0049 (0.0017) / -0.0030 (0.0017) | -0.0058 / -0.0047 | +0.0025 / +0.0034 | 7453 / 7500 |
| DDR_PTC_R1 | 0.011 / 0.325 | None / None | nan / nan | +0.0001 (0.0019) / +0.0016 (0.0018) | -0.0084 (0.0018) / -0.0074 (0.0017) | -0.0061 / -0.0050 | -0.0087 / -0.0078 | 7277 / 7449 |
| NEURAL_RAW | 0.340 / 0.570 | s0|guard|sex0 / s1|income|U0|brier_excess | 1.95 / 1.56 | +0.0015 (0.0005) / +0.0020 (0.0004) | -0.0010 (0.0005) / -0.0013 (0.0004) | +0.0013 / +0.0024 | +0.0004 / +0.0013 | 7500 / 7500 |
| TWJ | 0.883 / 0.788 | s0|guard|sex0 / s1|guard|sex0 | 0.32 / 0.62 | +0.0044 (0.0018) / +0.0039 (0.0017) | -0.0049 (0.0017) / -0.0030 (0.0017) | -0.0084 / -0.0073 | +0.0025 / +0.0034 | 7484 / 7500 |
| W | 0.012 / 0.175 | s0|income|U0|accuracy_loss / s1|income|U0|accuracy_loss | 3.51 / 2.86 | -0.0039 (0.0017) / -0.0023 (0.0017) | -0.0097 (0.0017) / -0.0087 (0.0017) | -0.0047 / -0.0036 | -0.0161 / -0.0153 | 7484 / 7427 |
| TT100 | 0.012 / 0.175 | s0|guard|sex0 / s1|guard|sex0 | 3.33 / 2.60 | -0.0010 (0.0015) / +0.0006 (0.0014) | -0.0093 (0.0015) / -0.0079 (0.0015) | -0.0047 / -0.0036 | -0.0121 / -0.0112 | 7484 / 7500 |
| TR100 | 0.012 / 0.175 | s0|guard|sex0 / s1|income|U0|accuracy_loss | 3.41 / 2.86 | -0.0012 (0.0015) / -0.0004 (0.0015) | -0.0102 (0.0016) / -0.0082 (0.0015) | -0.0047 / -0.0036 | -0.0138 / -0.0129 | 7484 / 7500 |
| U0 | 0.110 / 0.028 | s0|guard|sex0 / s1|occupation|U0|accuracy_loss | 2.76 / 2.82 | +0.0000 (0.0000) / +0.0000 (0.0000) | +0.0000 (0.0000) / +0.0000 (0.0000) | +0.0000 / +0.0011 | +0.0000 / +0.0009 | 7500 / 7500 |
| Ucal | 0.220 / 0.059 | s0|guard|sex1 / s1|guard|sex1 | 2.30 / 2.75 | -0.0017 (0.0003) / -0.0003 (0.0001) | +0.0011 (0.0003) / +0.0003 (0.0001) | -0.0011 / +0.0000 | -0.0009 / +0.0000 | 7500 / 7500 |
| CTL_TWIN | -0.425 / -0.100 | None / None | nan / nan | -0.0039 (0.0017) / -0.0023 (0.0017) | -0.0097 (0.0017) / -0.0087 (0.0017) | -0.0086 / -0.0075 | -0.0161 / -0.0153 | 6956 / 7427 |

## 4. Contract, exposure, selection history, reader sensitivity, package, limiting condition

### RETIRED_B3CTL

- **contract:** income: 6 decision-aligned levels of 0.5 LOCAL_TREE + 0.5 JOINT_TREE income log-odds (3 quantile bins per side of 0, MF thresholds), per-level probabilities from an L2 logistic map h fitted on all CALIBRATION; occupation: JOINT_TREE E20 (bitwise the H_LJ / SUCC occupation output). 6 distinct income vectors.
- **exposure:** JOINT / LOCAL: MF income, occupation, SEX of all 20,000 (in-sample); h: CAL income (all 2,982); the 0.5 mix chosen on a CAL screen; uncoupled twin and finalist promotion were adaptive steps on MS development proxies.
- **selection:** autonomous-method-development finalist B3CTL_UNCOUPLED_MIX50_C6A (adaptive, MS proxies); frozen-finalist validation reproduced it exactly and RETIRED it as a competitive candidate (2026-10-17).
- **reader_sensitivity:** READER_SENSITIVE (frozen-finalist validation, enhanced bank: claim B fails on the seed-1 income clause vs H_LJ through H_LJ's unstable income winner); DEV benefits +0.027/+0.022 vs NEURAL_RAW, +0.028/+0.026 vs H_LJ, none above 0.020 at its clause bounds.
- **package:** amd/ffv deployment packages (B3CTL; admitted copy under the ihs mirror admitted/amd_deploy); MS wire admitted by hash.
- **limiting:** utility slack ~0: the binding income Brier clauses sit 0.03-0.04 SE below the cap on DEV and the men's occupation guard 0.32 SE (v0); confirmation infeasible (frozen-finalist validation). Coarse 6-level income.

### B2TWIN

- **contract:** income: 6-level aligned code of LOCAL_TREE income log-odds with CAL-fitted per-level probabilities; occupation: JOINT_TREE E20. 6 distinct income vectors.
- **exposure:** as RETIRED_B3CTL (LOCAL only in the code); h on all CAL; the uncoupled twin was an adaptive step on MS proxies.
- **selection:** task-only-code twin of the B3CTL finalist in the autonomous-method-development campaign (a control, adaptive).
- **reader_sensitivity:** not separately established; a 6-level code view (few distinct inputs: reader seeds partly effective).
- **package:** admitted MS / MF wires (ihs mirror admitted/amd_runs); raw-input package via the amd code path.
- **limiting:** the shared JOINT occupation output's men's occupation Brier guard (reserve 0.32 / 0.62 SE).

### SUCC

- **contract:** income: H3_STRONG_LJDIR_PB = Platt(a (s_B + D_L + beta D_J) + b), s_B the HGB + T_TREE stacked task-only score, D_L / D_J the LOCAL / JOINT leaf-correction directions, beta 0.25 (v0) / 0.20 (v1) chosen on CAL; occupation: JOINT_TREE E20 (leaf-score corrections on the frozen T_TREE CatBoost routing, trained with income, occupation and linked-pair SEX critics). Continuous probabilities on both outputs.
- **exposure:** MF income, occupation and SEX of all 20,000 (JOINT, D_L, D_J, in-sample); HGB / stack / beta / Platt: MF and CAL income; H3 nominated on MS by its source rule and evaluated once on DEV_REPORT.
- **selection:** the income-headroom successor (H3) composed with JOINT occupation by the occupation-headroom campaign ("the source successor"); fixed composition, not tuned here.
- **reader_sensitivity:** income-headroom successor: P (pair) claim fails and reader-sensitive; occupation binds (ihs decision).
- **package:** H3 income (ihs mirror runs/H/H3_STRONG_LJDIR_PB) + JOINT occupation (acs_replication STAGE_A routing + STAGE_B JOINT_TREE__E20.npz); both stored locally.
- **limiting:** men's within-SEX occupation summed-Brier excess vs U0 (JOINT occupation output) at 0.883 (v0) / 0.788 (v1) of the 0.005 cap, reserve 0.32 / 0.62 SE, while the women's guard has wide slack; the pair (0.607) sits 0.034 above its occupation view (0.573): the H3 income vector was never jointly protected with JOINT occupation.

### H_LJ

- **contract:** income: LOCAL_TREE E20 (leaf-score corrections on the T_TREE routing with own-wire SEX critics); occupation: JOINT_TREE E20.
- **exposure:** MF income, occupation, SEX of all 20,000 (in-sample); fixed composition chosen after its components' DEV results.
- **selection:** frozen-hybrids preselected primary (H_LJ_FIXED_COMPOSITION_NOT_SUPPORTED); a material-target reference.
- **reader_sensitivity:** its income winner is unstable (frozen-finalist validation: chosen on an ATTACK_VALIDATION near tie).
- **package:** acs_replication STAGE_A/STAGE_B (LOCAL_TREE / JOINT_TREE E20) and deploy/LOCAL_TREE_S<v>.
- **limiting:** the shared JOINT occupation men's guard (0.32 / 0.62 SE); pair 0.619 vs occupation view 0.573.

### DDR_PTC_R1

- **contract:** dual-design adaptive revision: HGB task teachers with a linear SEX-propensity tilt (closed idea family).
- **exposure:** MF income, occupation, SEX (all 20,000); adaptive revision chosen on the same MS cells (winner's curse).
- **selection:** DB_PTC_R1, the dual-design campaign's strongest release; never separated from its release-only tilt control.
- **reader_sensitivity:** not separated from DB_ZTILT_R1 (release-only).
- **package:** ddr deployment packages (local).
- **limiting:** idea family closed (propensity tilts, release-only tilts); not separated from its release-only control.

### NEURAL_RAW

- **contract:** both outputs from the U0 network (d-64-64-16-K per task) trained from U0 with the joint objective (same online critics, row draws and calibration as the tree arms; Adam 1e-4, clip 5), rho 1, checkpoint E2. Continuous.
- **exposure:** MF task labels and SEX of all 20,000 (in-sample); rho / checkpoint chosen on MS by the replication's registered rule (post-selection MS values).
- **selection:** the replication's eligible neural control; a material-target reference (frozen-hybrids claim B).
- **reader_sensitivity:** none recorded beyond the general reader-bank caveat.
- **package:** checkpoints NEURAL_RAW__rho{1,3,10}__E{0,2,5,10,20}.pt for both seeds (acs_replication runs/NEURAL_RAW_<v>); a raw-input deploy package exists for S0 only.
- **limiting:** protection is weak on the occupation view (0.625 = its pair; the pair recovery is the occupation view); its stronger grid points (rho 3, 10 / later epochs) were not chosen by the replication rule; reserve 1.95 / 1.56 (men's guard v0, income Brier vs U0 v1) but point margins 0.34 / 0.57 of cap.

### TWJ

- **contract:** HC_STRONG (task-only) income + JOINT_TREE E20 occupation.
- **exposure:** as HC_STRONG and JOINT.
- **selection:** occupation-headroom combined reference.
- **reader_sensitivity:** -
- **package:** local (ihs mirror + Stage B).
- **limiting:** task-only income re-opens the pair (0.636).

### W

- **contract:** A1_PO_16 occupation witness + H3 income.
- **exposure:** MF occupation (witness), as H3.
- **selection:** occupation-headroom witness reference.
- **reader_sensitivity:** -
- **package:** local (ohn mirror).
- **limiting:** task-only occupation (0.644 occupation view).

### TT100

- **contract:** H3 income + T_TREE occupation.
- **exposure:** MF; as H3.
- **selection:** training-size reference.
- **reader_sensitivity:** -
- **package:** local.
- **limiting:** task-only occupation.

### TR100

- **contract:** H3 income + TASK_REFIT occupation.
- **exposure:** MF; as H3.
- **selection:** training-size reference.
- **reader_sensitivity:** -
- **package:** local.
- **limiting:** task-only occupation.

### U0

- **contract:** utility reference U0 (neural, task-only).
- **exposure:** MF task labels.
- **selection:** -
- **reader_sensitivity:** -
- **package:** U0_MODEL.pt
- **limiting:** utility reference, not a protection.

### Ucal

- **contract:** U0 with one CAL temperature per task.
- **exposure:** MF, CAL task labels.
- **selection:** -
- **reader_sensitivity:** -
- **package:** U0 + temperatures
- **limiting:** utility reference.

### CTL_TWIN

- **contract:** input-and-pair shared twin: cached HGB task teachers (task-only).
- **exposure:** MF task labels.
- **selection:** the recent campaigns' weak task-only start.
- **reader_sensitivity:** -
- **package:** ipr deploy.
- **limiting:** weak start (pair 0.648).

Point eligibility (all 28 at the point) is not confidence preservation (the reserve ratios), and neither is a competitive claim. A retired release can be a starting point or comparator without recovering its retired claim.

