# FA_TFR_v1: SUCC-anchored joint teacher-feature residual (card draft; to be committed as `cards/FA_TFR_v1.md` before the first fit)

Investigator A. First draft 2026-10-24 04:56Z; revised 05:04Z (`date -u`) under the coordinator's adjudication (PROCEED with required
changes 1-7). Family FA, starting release SUCC.

- **Code:** `sar/fa/tfr.py`. Deployment interface: `sar/fa/__init__.py` (`predict`, `model_paths`, `frozen_inputs`).
- **Interpreters:** fitting runs in the torch venv `<repository>/.venv/bin/python` on one thread; deployment needs numpy and
  sklearn only.
- **Status:** development evidence on previously used records.

## 1. Operation

**Runtime inputs:** the six permitted fields only (the admitted 145-column X). SEX is never an input or output.

**Fitted statistics:**

| Statistic | Learner and settings | Role and labels |
|---|---|---|
| Residual network r_θ | 29 inputs → 64 → 64 (ReLU) → 6, zero-initialized output layer | MODEL_FIT. Task term: cross-entropy of both released vectors to the admitted HGB teachers' household out-of-fold probabilities; the fit reads no task label. Critic term: MODEL_FIT SEX |
| Joint critics | `acs_replication.leafcorr.Critics` unchanged: linear / MLP32 / MLP64 on the income(3), occupation(6) and pair(9) wires, Adam 0.003, 80 warm rounds, 5 rounds per update, DiagonalReference refreshed every epoch on 256 reference rows | MODEL_FIT SEX |
| λ | ρ · median‖∇task‖ / median‖∇(risk_I + risk_O + risk_pair)‖ at θ = 0, over 16 draws | MODEL_FIT SEX |
| Feature map | means and SDs of the numeric fields and of the 12 teacher logits; the COW and MAR levels | label-free, training rows |

**Optimizer:** AdamW, lr 1e-3, weight decay 0.01, clip 5; 20 epochs × 40 steps × 256 rows; E20 kept.

**Frozen inputs:**
- **START = SUCC_S<v>,** computed as `ihs.fam_h.predict(v, 'H3_STRONG_LJDIR_PB', X, names)`. It is checked bitwise against the pinned
  MODEL_SELECT wires (`roles/investigator_a/check_teachers_labelfree.json`).
- **SUCC's 7 frozen files** per version (`frozen_inputs(v)`):
  - H3: STATE.json, STEP1.json, HGB.pkl;
  - T_TREE: the income and occupation teacher JSONs;
  - LOCAL_TREE and JOINT_TREE: the E20 θ files.
  - These must match pins in this campaign's `SOURCE_ADMISSION.json`. `preflight` refuses before any label read if a pin is missing or
    differs, and the fit records the hashes in every model file. `predict` refuses if the files change.
- **The HGB task teachers:** nrm `fa_cache/V<v>/TASK.pkl`, with their stored MODEL_FIT out-of-fold outputs. The pin is read from this
  campaign's `SOURCE_ADMISSION.json`, where `source_private_artefacts` already holds it.

**Released vectors:**
- φ(x) = [AGEP, WKHP, SCHL, POBP code, standardized; COW, MAR one-hot; d_S, z_S (SUCC's logits); d_H, z_H (the teachers' logits)].
- income = softmax(−(d_S + s·r_I)/2, (d_S + s·r_I)/2); occupation = softmax(centre(z_S + s·r_O)); decisions = argmax.
- s = 1 for CAND and CONT. s = 0 releases START (SUCC's own vectors) exactly.
- Deterministic; one vector per person.

**Entry points:**
- `python -m sar.fa.tfr preflight --version V` (label-free);
- `python -m sar.fa.tfr fit --version V --card <this card>`;
- `python -m sar.fa.tfr selftest`.

## 2. Closest previous implementation and the substantive difference

**The closest:**
- **Occupation-headroom P2 / P2A** (`ohn/protect.py`): a leaf-score correction on the frozen T_TREE occupation routing over a task-only
  witness base, with a pair critic and H3 frozen. It moved occupation only.
- **Also close:** income-headroom family P (income-only leaf corrections, pair critic) and NEURAL_RAW (the U0 network trained with the
  joint objective from U0).

**Differences:**
1. **Not a leaf-score correction on any routing.** It is a network over teacher logits and compact fields, so it can separate people
   who share every T_TREE leaf.
2. **The start is the protected release SUCC itself** (θ = 0), and the movement goes toward a strong out-of-fold teacher.
3. **Both outputs move under one joint objective.** H3's income is protected jointly with this occupation output for the first time.
4. **The task signal is teacher distillation, not labels.** SUCC is in-sample on every MODEL_FIT person, which blinded family C.
5. **Task components are selected on CALIBRATION,** plus a MODEL_FIT guard difference in which SUCC's in-sample term enters both terms.

## 3. Measured failure addressed

**The binding condition:** SUCC's men's within-SEX occupation summed-Brier guard, +0.0044 (0.0018) / +0.0039 (0.0017) against 0.005,
with a two-SE reserve of 0.32 / 0.62 (`anchors/ANCHOR_TABLE.md`, section 3).
- The JOINT correction costs men and women alike against TASK_REFIT.
- The guard binds because the T_TREE base leaves men only −0.0012 / −0.0004 against U0.

**The pair:** 0.6074, which is 0.034 above the occupation view (0.5731). H3's income was never protected jointly with the JOINT
occupation output.

## 4. Why usefulness may be preserved

**The teachers' margins.** The admitted HGB teachers (the occupation output of CTL_TWIN and W) have:
- a men's guard of −0.0039 / −0.0023;
- occupation NLL 0.0186 / 0.0187 nats better than SUCC's, against U0.

**Label-free numbers on MODEL_FIT features:**
- SUCC sits 0.026-0.028 nats (occupation) and 0.011-0.012 (income) KL from the teachers' out-of-fold outputs
  (`check_teachers_labelfree.json`).
- The λ = 0 residual closes about 73% of the occupation gap in 2 of 20 epochs (`check_cpu_labelfree.json`).
- These are label-free numbers, not utility results.

**The usefulness result reported is the MODEL_SELECT one,** with the original, unchanged caps: the 24 clauses and 4 guards against U0 and
Ucal. The CALIBRATION and MODEL_FIT quantities below serve only selection and matching.

## 5. Twin

**FA_TFR_CONT:** the equally trained release with protection off. It uses the same network, features, rows, folds, optimizer and
schedule with λ = 0 (no critics, no SEX), at s = 1. It is about the HGB teachers.

**Inherited references:** START (SUCC) and CTL_TWIN.

## 6. Strength grid and selection rule (MODEL_FIT, plus CALIBRATION task labels as stated)

**Grid:** ρ ∈ {0.3, 1, 3, 10}.

**The rule, per version:** ρ\* = the largest ρ such that it and every smaller ρ meet both parts, each at estimate + 2 household-clustered
SE:
- **(a)** The CALIBRATION paired spend of the full-MODEL_FIT fit over SUCC stays within:

  | Component | Limit |
  |---|---|
  | occupation NLL | 0.0040 |
  | occupation Brier | 0.0025 |
  | occupation accuracy | 0.0050 |
  | income NLL | 0.0060 |
  | income Brier | 0.0030 |
  | income accuracy | 0.0035 |

- **(b)** The MODEL_FIT household out-of-fold occupation summed-Brier spend over the λ = 0 path's **LO point for that ρ** (section 7's
  rule; s = 1 if no LO point exists) is at most 0.0005 for men and 0.0050 for women. Folds: 5, SAR_xf_FA_S<v>.

**The limits** are 0.4-0.6 of SUCC's MODEL_SELECT point margins; the men's limit is about SUCC's own point margin. The derivation is in
PROPOSAL.md, section 6.

**If ρ = 0.3 fails,** that version releases nothing (status NO_FEASIBLE_STRENGTH).

**CALIBRATION exposure (correction 1 of the adjudication).**
- **Out of sample on CALIBRATION:** SUCC's occupation side (JOINT_TREE on the T_TREE routing), the HGB teachers and the residual.
- **In-sample on CALIBRATION:** SUCC's income output H3 (its STRONG stack, its β budget, which was CAL cross-fitted, and its Platt map
  were fitted on all 2,982 CALIBRATION people) and Ucal's temperatures.
- **Consequence:** the income-side CALIBRATION spends over SUCC are biased in SUCC's favour. That is conservative for CAND's selection.
  They are disclosed as such, and no honesty is claimed for them.
- **The same bias reaches the CTRL bracket.** The bracket is matched on total NLL, which includes the income part.

**Label use.**
- CALIBRATION task labels (income and occupation; never SEX) are read only for selection, part (a), and for the CTRL bracket. Nothing is
  fitted on CALIBRATION, and the controls' exposure is identical.
- **The first draft's guard reference was "the closest λ = 0 point".** It is replaced by the LO point. The coordinator replaced the
  closest-point match with the bracket rule, and the LO point is the conservative reference: it is at least as useful as CAND.

## 7. Matched controls (adjudication 2-4)

**CTRL** is the λ = 0 interpolation path FA_TFR_TASK, with origin START (s = 0 is SUCC exactly), on
S_PATH = {0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, …, 1.0}.
- **Selection data:** CALIBRATION paired total-NLL spend over SUCC. Two points per version are fixed before any MODEL_SELECT read:
  - **FA_TFR_TASK_LO:** the S_PATH point with the largest spend that is still ≤ CAND's spend;
  - **FA_TFR_TASK_HI:** the point with the smallest spend ≥ CAND's spend + 2 SE. The SE is the household-clustered CALIBRATION SE of
    that point's paired difference against CAND.
- **Flags:**
  - CONTROL_PATH_MAY_NOT_BRACKET if no point reaches HI;
  - CONTROL_PATH_NO_LO_POINT if no point is ≤ CAND's spend.
- **Comparisons are on spend values, never on s.** This also holds when CAND's spend over SUCC is ≤ 0. Ties go to the smaller s.
- **Endpoints are aliases.** A CTRL point at s = 0 is START itself, and a point at s = 1 is CONT. These are recorded as aliases
  (START, FA_TFR_CONT) and not saved twice (`ALIAS_ENDPOINTS = True`, coordinator-adjustable).

**CONT** is FA_TFR_CONT (λ = 0, s = 1): the zero-protection continuation, **declared gating**.

**Not built in the initial comparison:**
- the SEX-blind DQ spend path of SUCC;
- the release-only tilt of SUCC;
- an SBC (critics on a permuted SEX).

They are available as shared or revision controls.

**Ablation:** CTRL is the critic ablation of CAND.

**D5:** not declared jointly designed against separately designed recipients. No sequential orders are built, so D5 is not claimed.

## 8. Revisions (at most two)

| Revision | Trigger | Change | Stop rule |
|---|---|---|---|
| **R1** | ρ\* = None in a version because (a) fails, while the λ = 0 path meets (a) at s ≥ 0.5 | the critic term restricted to the pair view | the same rule; if ρ = 0.3 still fails, stop |
| **R2** | CAND is not separated from CTRL on the registered reading, while CAND's CALIBRATION total-NLL spend over SUCC is negative | the SBC control: the identical fit with SEX permuted within household-clustered strata, at CAND's ρ\* | no further revision |

## 9. Registered predictions, specification count, CPU

**Registered predictions** (first draft, 04:56Z; unchanged; per version unless stated; registered-cell means):

| Quantity | Prediction |
|---|---|
| ρ\* | P(0.3) 0.25, P(1) 0.35, P(3) 0.20, P(10) 0.05, P(None) 0.15 |
| CALIBRATION occupation NLL spend over SUCC < 0 in both versions | 0.70 |
| MODEL_SELECT men's guard below SUCC's in both versions | 0.60 (reserve ratio ≥ 1: 0.45) |
| all 28 conditions met at the point | 0.60 |
| pair AUC | 0.600 (80%: 0.585-0.620); occupation view 0.580 (0.565-0.600); income view 0.530 (0.520-0.545) |
| pair benefit over SUCC | +0.005 (80%: −0.012 to +0.020); P(> 0) 0.55; P(resolved) 0.10 |
| pair benefit over FA_TFR_TASK | +0.012 (80%: −0.005 to +0.030); P(resolved) 0.25 |
| FA_TFR_CONT pair AUC | 0.645 (0.635-0.652) |
| spend match flagged outside the path | 0.25 |
| D3 | 0.04 |
| D4 | 0.05 |

**Dated note, 2026-10-24 05:04Z.** The two-point bracket replaces the single closest match. The registered numbers above are not edited.
For the new objects:

| Quantity | Prediction |
|---|---|
| Pair benefit over FA_TFR_TASK_LO (the control at least as useful as CAND) | +0.015 (80%: −0.003 to +0.033) |
| Pair benefit over FA_TFR_TASK_HI (the control less useful by ≥ 2 SE) | +0.008 (80%: −0.008 to +0.025) |
| P(resolved over both, both version means > 0) | 0.20 |
| P(CONTROL_PATH_MAY_NOT_BRACKET in at least one version) | 0.35 (it needs CAND to improve on START by ≥ 2 SE on CALIBRATION, since s = 0 has spend 0) |
| P(CONTROL_PATH_NO_LO_POINT in at least one version) | 0.10 |
| P(at least one CTRL point is an alias of START or CONT) | 0.45 |

The registered "spend match flagged outside the path" (0.25) referred to the superseded closest-match rule; its successor is the HI flag
above. The guard reference change (section 6) may lower ρ\*: P(ρ\* ≤ 1) moves from 0.60 to about 0.65, as a note only.

**Specifications (adjudication 4):** at most 8 = CAND 2 (FA_TFR × 2 versions) + CTRL 4 (FA_TFR_TASK_LO and _HI × 2) + CONT 2
(FA_TFR_CONT × 2). There are fewer when a CTRL point is an alias or missing, or a version has no feasible ρ.

**Fit record** (`results/pcrl_strong_anchor_release_v1/fa/FIT_V<v>.json`, adjudication 5), per version:
- CAND: its config (`FA_TFR`), ρ\*, status (FITTED / NO_FEASIBLE_STRENGTH) and its CALIBRATION total-NLL spend over SUCC (value, SE,
  plus all components);
- CTRL: the origin (START, s = 0), the selection data (CALIBRATION), every S_PATH point's spend over SUCC and the SE of its paired
  difference against CAND, the saved LO / HI config ids or aliases with their spends, and the flags;
- CONT: its config (`FA_TFR_CONT`, gating);
- the per-ρ selection table (parts (a) and (b), the guard reference s, the bracket points and flags per ρ);
- the descriptive MODEL_FIT readers and the biased MODEL_FIT spends;
- the TASK.pkl and frozen-file hashes.

**CPU:** about 5-7 CPU-min per version on one thread: 24 critic fits at about 9.5 CPU-s each (measured label-free), plus releases and the
MODEL_FIT readers at about 2 min.

## 10. Novelty status

Not claimed. The closest published methods are adversarial debiasing (Zhang, Lemoine & Mitchell 2018), knowledge distillation (Hinton,
Vinyals & Dean 2015), and stacking and residual learning on frozen teachers. No literature search was run.
