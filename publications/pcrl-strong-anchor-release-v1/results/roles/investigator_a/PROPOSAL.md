# Investigator A proposal: SUCC-anchored joint teacher-feature residual (FA_TFR)

> **Correction, 2026-10-24 05:05Z (coordinator adjudication: PROCEED with required changes).** Two points of the text below are
> superseded; `CARD_DRAFT.md` and `sar/fa/tfr.py` govern.
> 1. **CALIBRATION is not out of sample for all of SUCC.**
>    - SUCC's income output H3 is in-sample on CALIBRATION: its STRONG stack, its β budget and its Platt map were fitted on all 2,982
>      CALIBRATION people. Ucal's temperatures are in-sample there too.
>    - Out of sample on CALIBRATION are only the occupation side (JOINT_TREE), the HGB teachers and the residual.
>    - So the income-side CALIBRATION spends over SUCC are biased in SUCC's favour. That is conservative for the candidate's selection,
>      and they are not honest estimates.
> 2. **The single closest-point control FA_TFR_TASK is replaced by a two-point bracket**, FA_TFR_TASK_LO / _HI, on CALIBRATION paired
>    total-NLL spend over SUCC.
>    - The path's origin is START: s = 0 is SUCC exactly.
>    - CONT is declared gating.
>    - The guard part of the strength rule uses the LO point as its reference.
>    - The specification count becomes at most 8.
>
> The registered predictions are unchanged; new predictions for the bracket points are in the card's dated note.

Written 2026-10-24 04:51Z (`date -u`). Everything here was produced label-free: committed records, code, admitted features and frozen
release arrays, plus invented data. No label file was opened and nothing was computed from MODEL_SELECT labels. Development evidence on
previously used records; lower recovery for these readers is not protection against every reader.

## 1. Starting release and the binding condition addressed

**Start:** SUCC_S<v> = H3_STRONG_LJDIR_PB income + JOINT_TREE E20 occupation. My loader (`sar.fa.tfr._succ`, i.e. `ihs.fam_h.predict`)
reproduces the pinned MODEL_SELECT wires bitwise for both outputs in both versions (`check_teachers_labelfree.json`, label-free).

**Conditions addressed (both are one mechanism):**
- **The men's within-SEX occupation summed-Brier guard:** +0.0044 / +0.0039 against the 0.005 cap. Its cause is the base, not the
  correction: the JOINT correction costs men and women about the same (+0.0056 / +0.0043 and +0.0053 / +0.0052 against TASK_REFIT),
  but the T_TREE routing leaves men only −0.0012 / −0.0004 of slack against U0.
- **The admitted HGB task teachers** (nrm `fa_cache/V<v>/TASK.pkl`, A1_PO_16 settings, POBP ordinal code; the occupation output of
  CTL_TWIN and W):
  - their men's guard is −0.0039 / −0.0023 and their women's −0.0097 / −0.0087;
  - their occupation NLL is 0.0186 / 0.0187 nats better than SUCC's against U0 (anchor table);
  - their gain over the tree base falls mostly on men (occupation-headroom decision, section 1).
- **The linked pair:** 0.6074, which is 0.034 above SUCC's occupation view (0.5731). H3's income was never protected jointly with this
  occupation output.
- **The room to move (label-free, MODEL_FIT features):**
  - SUCC sits 0.026-0.028 nats (occupation KL) and 0.011-0.012 nats (income KL) from the teachers' out-of-fold outputs;
  - 15-17% of occupation decisions and 6-7% of income decisions differ.

## 2. The operation (one; both recipients at once)

```
r_theta(phi(x)) = (r_I, r_O),   phi = [compact six fields (AGEP, WKHP, SCHL, POBP code standardized; COW, MAR one-hot),
                                       SUCC logits (d_S, z_S), HGB teacher logits (d_H, z_H)]          (29 inputs)
income      = softmax(-(d_S + r_I)/2, (d_S + r_I)/2)        occupation = softmax(centre(z_S + r_O))      decisions = argmax
```

**The residual network.**
- A 64-64 ReLU network whose output layer starts at zero, so θ = 0 is SUCC exactly.
- Teacher logits in φ are the household out-of-fold values on MODEL_FIT training rows and the deployed models elsewhere.

**The training objective** (MODEL_FIT):

```
L(theta) = CE(p_HGB_oof(x) -> income(x)) + CE(p_HGB_oof(x) -> occupation(x))
         + lambda * [risk(income) + risk(occupation) + risk(pair)]
```

- **The critic term** uses the lineage's online SEX critics unchanged: `acs_replication.leafcorr.Critics`, linear / MLP32 / MLP64 per
  wire, 80 warm rounds, 5 rounds per update and references refreshed every epoch.
- **λ** = ρ · median‖∇task‖ / median‖∇joint risk‖ at θ = 0 over 16 draws (the lineage's calibration rule).
- **The optimizer:** AdamW, lr 1e-3, weight decay 0.01 (it shrinks θ toward SUCC), clip 5; 20 epochs × 40 steps × 256 rows, kept at
  E20.

**The task term is label-free.** It is distillation toward the honest out-of-fold strong teachers, and that is deliberate.
- Every SUCC component (T_TREE routing, JOINT θ, H3's HGB) is in-sample on every MODEL_FIT person. T_TREE's in-sample optimism is
  about 0.034 nats; family C of the occupation-headroom campaign failed on exactly this.
- So a label-trained residual on SUCC would see SUCC as better than it is, and would not move.

**Strength.** ρ\* is the largest ρ on {0.3, 1, 3, 10} such that it and every smaller ρ pass both parts below (section 6 gives the
budgets).
- **Part (a):** the CALIBRATION paired spend over SUCC passes BUDGET_CAL. SUCC, the teachers and the residual are all out of sample
  on CALIBRATION.
- **Part (b):** the MODEL_FIT out-of-fold men's and women's occupation Brier spend over the λ = 0 point matched on CALIBRATION total
  NLL passes GUARD_DID. SUCC's in-sample fit enters both terms of this difference, so it cancels to first order.

**Code:**
- `sar/fa/tfr.py`: `train`, `select_rho`, `stage_fit`, `predict`, `model_paths`, `frozen_inputs`, `selftest`;
- `sar/fa/__init__.py`: the deploy interface.

**What it is meant to do.**
- At ρ = 0 the residual simply moves SUCC toward the strong teachers. This recovers the task margin, the men's margin above all, and
  gives back SUCC's protection; the end point is about CTL_TWIN (pair 0.648).
- At ρ > 0 the joint critics, including the pair critic that was never applied to H3 with this occupation output, accept only the
  parts of that movement that do not raise readability.
- They can also push both outputs below SUCC's pair readability, spending the recovered margin and the women's and income slack.

## 3. Closest existing implementations, and the operational difference

| Closest | What it did | What TFR changes |
|---|---|---|
| Occupation-headroom P2 / P2A (`ohn/protect.py`) | leaf-score correction on the FROZEN T_TREE occupation routing, base = task-only witness (cross-fitted), pair critic with H3's income frozen, end projection | not a leaf-score correction: a network over the teachers' logits and compact fields, so it can separate people who share every T_TREE leaf (POBP ranges, where the teachers' men-favourable gain lives); starts AT a protected release and moves toward the task teacher, instead of protecting a task-only base; corrects BOTH outputs jointly; task signal = distillation toward an honest out-of-fold teacher, not labels on an in-sample base |
| Income-headroom family P | income-only leaf-score corrections with the pair critic, occupation frozen; +0.001 beyond scaling the stored JOINT direction | both outputs move; not on any routing; the occupation output (the binding side) is in play |
| NEURAL_RAW | U0 network, all weights trained with the same joint objective from U0 (task-only start), label CE | start = SUCC (protected), not U0; trainable part = a residual on teacher features (the strong HGB teachers are inputs); task term = teacher distillation |
| H3 transplant / an occupation transplant z_HGB + (z_J − z_R) | adds a stored correction direction to a better base, no SEX at fit | the λ = 0 control path (SUCC → teacher in logit space, by a learned residual) plays this role; TFR's difference is that the critics decide which part of the teacher's gain is taken |
| O1_SEXBAL | bijective recalibration (temperature + SEX-balanced bias) of the JOINT occupation output | a learned function of the six inputs that changes which people get which vectors |

**Known methods.** Adversarial debiasing (Zhang, Lemoine & Mitchell 2018), knowledge distillation (Hinton et al. 2015) and residual
stacking. No novelty is claimed, and no literature search was run.

## 4. Evidence that would contradict the rationale

1. **The critics add nothing beyond moving toward the teacher.** FA_TFR is not less recoverable on the pair than FA_TFR_TASK, the
   λ = 0 point at the same CALIBRATION cost. D2 is then NOT_SEPARATED, and the protection is just "move less".
2. **No SEX-neutral part of the teacher's gain.** The candidate does not recover task quality: its CALIBRATION occupation NLL spend
   over SUCC is ≥ 0 at ρ\*. The joint critics then block the movement toward the teacher.
3. **No men's recovery.** The men's guard on MODEL_SELECT does not improve on SUCC's +0.0044 / +0.0039.
4. **The profile is lost.** The candidate's pair is not below SUCC's (benefit ≤ 0), or its occupation view exceeds SUCC's by more than
   0.010 (recipient gate).
5. **ρ\* = None.** The critics' cost exceeds the recovered margin even at ρ = 0.3, so recovered margin cannot be turned into
   protection inside SUCC's slack.
6. **Blunted critics** (a diagnostic risk, not a refutation). JOINT and H3 were trained against MODEL_FIT SEX, so SUCC's wires look
   less readable on MODEL_FIT than on MODEL_SELECT. The descriptive MODEL_FIT readers will show that; if SUCC's MODEL_FIT pair AUC is
   far below 0.607, the critics see a blunted target.

## 5. Matched control and continuation

**FA_TFR_TASK** (the simplest informative matched control).
- **What it is:** the identical network, features, rows, folds, optimizer and schedule with λ = 0, released at scale s_c on
  S_GRID = {0.05, 0.1, …, 1}. It is plain learned interpolation from SUCC toward the same teachers.
- **Cost matching:** prospectively, s_c = the grid scale whose CALIBRATION paired total NLL over SUCC is closest to the candidate's.
  The bracketing neighbours are recorded, with the flag CANDIDATE_SPEND_OUTSIDE_CONTROL_PATH if the candidate spends more than SUCC.
- **Why CALIBRATION and not MODEL_FIT:** MODEL_FIT costs of anything relative to SUCC are biased by SUCC's in-sample fit.
- **Effort:** equal by construction. Same folds (SAR_xf_FA_S<v>), same rows, one fit per fold.

**FA_TFR_CONT** (the zero-protection continuation). λ = 0 at s = 1, about the HGB teachers. Attribution needs it:
- it shows how much of SUCC's protection a task-only re-fit of the same function class gives back;
- it shows where the candidate sits between SUCC and the task teacher.

**The other PROTOCOL-4 controls are not built here.**
- The SEX-blind spend path (DQ of SUCC) and the release-only tilt of SUCC are available as shared controls if the coordinator wants
  them. The spec budget (≤ 6) does not allow them inside this family.
- The SEX-independent-attribute path (critics on a permuted SEX) would be the natural SBC. I propose it as the first revision's
  control, not as an initial spec.

## 6. Bounded choices, strength rule and budgets

**Fixed before any fit:**
- ρ grid {0.3, 1, 3, 10}: the lineage's {1, 3, 10} plus 0.3, the source TCR1 grid's lowest point, so that a weak strength exists;
- network 64-64, zero-initialized output layer, recipe constants as in section 2, S_GRID 13 points;
- household 5-fold cross-fitting on MODEL_FIT (famrun.crossfit, tag SAR_xf_FA_S<v>), the same folds for the candidate and the controls.

**Budgets, from SUCC's actual MODEL_SELECT point margins** (cap − value, v0 / v1; anchor table):

| Component | SUCC margin | Limit (estimate + 2 SE) | Share of margin |
|---|---|---|---|
| occupation NLL (CALIBRATION paired over SUCC) | 0.0075 / 0.0066 (vs Ucal v0 0.0066) | 0.0040 | 0.53-0.61 |
| occupation summed Brier (CAL) | about 0.005 | 0.0025 | about 0.5 |
| occupation accuracy loss (CAL) | about 0.010 | 0.0050 | about 0.5 |
| income NLL (CAL) | ≥ 0.0136 | 0.0060 | about 0.44 |
| income summed Brier (CAL) | about 0.0073 / 0.0083 vs Ucal | 0.0030 | about 0.4 |
| income accuracy loss (CAL) | 0.0082 v1 (SE 0.0029; the binding income clause) | 0.0035 | about 0.43 |
| men's occupation Brier: MF out-of-fold, over the matched λ = 0 point | 0.0006 / 0.0011 | 0.0005 | about SUCC's own point margin |
| women's occupation Brier: MF out-of-fold, over the matched λ = 0 point | 0.0099 / 0.0080 | 0.0050 | 0.50-0.63 |

**Why these budgets do not force near-identity updates.**
- The λ = 0 direction itself recovers task quality: on MODEL_SELECT the teachers are 0.0186 nats and 0.0083 / 0.0062 men's Brier
  better than SUCC.
- So the CALIBRATION task components are met with room. The rule then picks the largest ρ that keeps them, plus the guard difference.
- The men's limit says only that the critics may cost men at most 0.0005 more than plain interpolation at the same honest cost.

**What the rule does not cover.**
- The 2-SE rule on CALIBRATION (2,982 people) is wide, roughly 2.6 times MODEL_FIT's SEs.
- The guard difference cancels SUCC's in-sample term only to first order.
- The lineage's MF-to-MS guard transfer error (±0.002) is not covered by any rule here. Point feasibility on MODEL_SELECT is predicted,
  not guaranteed.

## 7. Measurements

**Registered (the coordinator's grid, unchanged):**
- pair, income-view and occupation-view SEX AUC on the IPR1_MAIN-type registered cells;
- paired pair benefit against SUCC, FA_TFR_TASK, FA_TFR_CONT, NEURAL_RAW and H_LJ (D2 / D3 / D4 as the rules define them);
- the 28 utility conditions, the reserve, and the men's and women's guards per version.

**Descriptive, from the fit record (`results/<campaign>/fa/FIT_V<v>.json`):**
- the selection table: for each ρ, its CALIBRATION spends, matched scale and guard difference;
- λ;
- MODEL_FIT out-of-fold readers (biased: SUCC is in-sample);
- the biased MODEL_FIT spends over SUCC, beside the CALIBRATION ones.

**Recommended beside them (label-free on MODEL_SELECT outputs):**
- KL from SUCC and toward the teachers;
- decision changes against SUCC per output;
- distinct vectors.

## 8. How a failure would be read

| Observation | Reading |
|---|---|
| ρ\* = None in a version | nothing saved for that version; the critics cost more than the honest recovered margin even at 0.3. A negative answer to "turn recovered margin into protection" at this function class |
| FA_TFR ≈ FA_TFR_TASK on the pair (not separated) | the protection is "move less toward the teacher"; the critics do not shape the movement (contradiction 1) |
| FA_TFR below SUCC on the pair, but not separated from FA_TFR_TASK | a recovery gain that the matched interpolation also has: margin recovery without a protection mechanism |
| men's guard better, pair ≥ SUCC | margin recovered, profile not kept: the teacher's men-favourable gain is SEX-readable |
| men's guard worse on MS while it passed on MF / CAL | the known guard transfer error (the rule does not cover it) or a failure of the first-order cancellation; check the λ = 0 point's MF-against-CAL difference |
| occupation view up by more than 0.010 while the pair falls | leakage moved between recipients (FAILURE_MAP note 3) |

## 9. Registered predictions (per version unless stated; registered-cell means for AUCs)

| Quantity | Prediction |
|---|---|
| ρ\* | P(0.3) 0.25, P(1) 0.35, P(3) 0.20, P(10) 0.05, P(None) 0.15 |
| candidate's CALIBRATION occupation NLL spend over SUCC < 0 (recovers task quality), both versions | 0.70 |
| MS men's guard of FA_TFR below SUCC's, both versions | 0.60; men's reserve ratio ≥ 1: 0.45; ≥ 2: 0.15 |
| all 28 conditions at the point, both versions | 0.60 |
| FA_TFR pair AUC | 0.600 (80% interval 0.585-0.620); occupation view 0.580 (0.565-0.600); income view 0.530 (0.520-0.545) |
| pair benefit over SUCC | +0.005 (80%: −0.012 to +0.020); P(> 0 at the point) 0.55; P(resolved) 0.10 |
| pair benefit over FA_TFR_TASK | +0.012 (80%: −0.005 to +0.030); P(resolved, both version means > 0) 0.25 |
| FA_TFR_CONT pair AUC | 0.645 (0.635-0.652); FA_TFR resolved below it: 0.90 |
| control spend match flagged outside the path (candidate spends more than SUCC on CAL) | 0.25 |
| D3 (resolved below SUCC and every strong reference) | 0.04 |
| D4 (resolved 0.020 vs NEURAL_RAW and H_LJ) | 0.05 |

## 10. Specification count and CPU

**Specifications: 6** (FA_TFR, FA_TFR_TASK, FA_TFR_CONT × 2 versions). If ρ\* = None in a version, that version saves nothing.

**CPU per version** (measured label-free: `check_cpu_labelfree.json`, about 9.5 CPU-s per 20-epoch critic fit on 16,000 rows, about
0.5 CPU-s task-only):
- 24 critic fits (4 ρ × (5 folds + 1 full)): about 230 s;
- 6 task-only fits: about 3 s;
- SUCC and teacher arrays: about 5 s;
- releases of 3 configurations on 6 roles: about 30 s;
- MF readers: about 60-120 s.

So about 5-7 CPU-min per version and 10-14 CPU-min for both, on one thread in the torch venv.
