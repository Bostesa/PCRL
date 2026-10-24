# Investigator A hand-back (saved verbatim by the coordinator, 2026-10-24 about 04:57Z)

## Investigator A hand-back: FA_TFR, a joint residual on SUCC with teacher features

Written 2026-10-24, from 04:22Z to 04:54Z. Everything was label-free: no label file was opened, nothing was computed from MODEL_SELECT labels, no git, no shared module edited. The selftest passes all 16 checks.

### 1. A finding that shapes the design (it also applies to Investigator B)
Every SUCC component is fitted on all 20,000 MODEL_FIT people: the T_TREE routing, JOINT θ and H3's HGB. T_TREE's in-sample optimism there is about 0.034 nats, and family C of the occupation-headroom campaign failed on exactly this.
- **Consequence:** any MODEL_FIT task-cost estimate relative to SUCC is badly biased. A residual trained on labels over SUCC would see SUCC as better than it is and would not move.
- **My design avoids this:**
  - the task term is distillation toward the honest out-of-fold HGB teachers, so no label is read by the fit itself;
  - the strength rule uses CALIBRATION for the task components, since SUCC and the teachers are out of sample there;
  - the guard part is a MODEL_FIT difference against the matched λ = 0 point, so SUCC's in-sample term cancels to first order.
- NEURAL_RAW is also in-sample on MODEL_FIT, so B may face the same issue.

### 2. The proposed operation
- **Release:** income = softmax(−(d_S + r_I)/2, (d_S + r_I)/2); occupation = softmax(centre(z_S + r_O)).
- **The residual r_θ:** a 64-64 ReLU network with a zero-initialized output layer, so θ = 0 is SUCC exactly. Its 29 inputs are the compact six fields (POBP as its code), SUCC's own logits and the admitted HGB teachers' logits (nrm `TASK.pkl`; out-of-fold values on training rows).
- **Training:** cross-entropy of both outputs to the teachers' out-of-fold probabilities, plus λ times the lineage's joint critic risk (income + occupation + pair; `leafcorr.Critics` unchanged). λ comes from the lineage's 16-draw calibration rule.
- **Optimizer:** AdamW, lr 1e-3, weight decay 0.01, clip 5; 20 epochs × 40 steps × 256 rows, E20 kept.

**Why it addresses both conditions:**
- The men's guard binds because the tree base is weak for men. The teachers are strong for men (men's guard −0.0039 / −0.0023) and 0.0186 nats better on occupation NLL.
- The residual moves SUCC toward the teachers only where the joint critics accept it. The same objective protects H3's income jointly with this occupation output for the first time (the 0.034 pair gap).

**Why it is not a renamed closed recipe:**
- It is not a leaf-score correction on any routing. A network over teacher logits and fields can separate people who share every T_TREE leaf.
- It starts at a protected release and moves toward a task teacher. P2/P2A and NEURAL_RAW started task-only and added protection.
- Both outputs move under one objective. Family P moved income only; P2 moved occupation only.
- Its task signal is teacher distillation, not labels.
- The transplant-like path (SUCC toward the teacher in logit space) is the λ = 0 control, not the candidate.

### 3. The card's key choices
- **Grid:** ρ ∈ {0.3, 1, 3, 10}.
- **Rule:** ρ\* = the largest ρ such that it and every smaller ρ pass, each at estimate + 2 household-clustered SE.
  - CALIBRATION paired spend over SUCC: occupation NLL ≤ 0.0040, occupation Brier ≤ 0.0025, occupation accuracy ≤ 0.0050, income NLL ≤ 0.0060, income Brier ≤ 0.0030, income accuracy ≤ 0.0035. These are 0.4-0.6 of SUCC's MODEL_SELECT margins.
  - MODEL_FIT out-of-fold occupation Brier over the λ = 0 point matched on CALIBRATION total NLL: men ≤ 0.0005 (about SUCC's own point margin), women ≤ 0.0050.
  - If ρ = 0.3 fails, nothing is saved for that version.
  - These limits do not force near-identity updates: the λ = 0 direction itself recovers task quality.
- **Controls:**
  - FA_TFR_TASK: λ = 0, released at the scale whose CALIBRATION total NLL over SUCC is closest to the candidate's. Neighbours and an out-of-range flag are recorded.
  - FA_TFR_CONT: λ = 0 at s = 1, the zero-protection continuation (about the HGB teachers).
- **Specifications:** 6 (three configurations × two versions).
- **Revisions:** R1 is a pair-only critic, if (a) fails while the λ = 0 path meets it. R2 is a permuted-SEX critic control, if the candidate is not separated from FA_TFR_TASK while it recovered task quality.

### 4. Predictions
| Quantity | Prediction |
|---|---|
| ρ\* | 0.3: 0.25; 1: 0.35; 3: 0.20; 10: 0.05; none: 0.15 |
| CALIBRATION occupation NLL below SUCC's, both versions | 0.70 |
| MODEL_SELECT men's guard below SUCC's, both versions | 0.60 (reserve ratio ≥ 1: 0.45) |
| All 28 conditions met at the point | 0.60 |
| Pair AUC | 0.600 (80% interval 0.585-0.620) |
| Benefit over SUCC | +0.005 (80%: −0.012 to +0.020); P(resolved) 0.10 |
| Benefit over FA_TFR_TASK | +0.012; P(resolved) 0.25 |
| FA_TFR_CONT pair AUC | about 0.645 |
| D3 / D4 | 0.04 / 0.05 |

### 5. What I built and tested
- `sar/fa/tfr.py`: train, select_rho, stage_fit, predict, model_paths, frozen_inputs and selftest.
- `sar/fa/__init__.py`: lazy `predict`, `model_paths` and `frozen_inputs` for `sar.deploy`, as you asked. It imports under plain `python3`; torch is needed only for fitting.
- Selftest (`.venv/bin/python -m sar.fa.tfr selftest`): all 16 checks true, 19.1 CPU-s. The checks cover:
  - θ = 0 equals the start;
  - the numpy forward equals the torch forward;
  - determinism and λ > 0;
  - valid vectors and the task term moving toward the teacher;
  - the spend match and its out-of-range flag;
  - the rule stopping at the first failure;
  - bitwise deployment through `predict` and refusal of keys;
  - all three configurations saved with the right kinds.
- Label-free checks on real data, in my role directory:
  - My SUCC loader reproduces the pinned MODEL_SELECT SUCC wires bitwise, both outputs and versions (`check_teachers_labelfree.json`).
  - SUCC sits 0.026-0.028 (occupation) and 0.011-0.012 (income) nats KL from the teachers. Decisions differ for 15-17% (occupation) and 6-7% (income).
  - At real size a critic fit costs about 9.5 CPU-s and a task-only fit about 0.5 (`check_cpu_labelfree.json`). Two λ = 0 epochs close 73% of the occupation KL gap.
- **CPU:** about 92 CPU-s of my 600 used. Estimated cost of the real fits: about 5-7 CPU-min per version.

### 6. Changes needed in shared code (not made by me)
1. `sar/common.py` has no FA family. CARD_PREFIX needs `'FA': 'FA_'` (and FB) and FIT_ENTRY needs `'sar.fa'` (and `sar.fb`). Without them, `C.load` and `release.save` refuse sar.fa.
2. My rule reads CALIBRATION income and occupation for the selection and the control's match; nothing is fitted on CALIBRATION. POLICY's `calibrate` text says "calibration parameters only; not planned by the registered card", so the registration needs to allow this use.
3. Admission pins are needed for:
   - `fa_cache/V<v>/TASK.pkl`: my loader checks the input-and-pair campaign's pin; repoint it to this campaign's SOURCE_ADMISSION once that exists.
   - SUCC's frozen files, read through `ihs.fam_h.predict` (listed by `frozen_inputs(v)`, all under `.private/pcrl_income_headroom_successor_v1/`):
     - `runs/H/H3_STRONG_LJDIR_PB/S<v>/STATE.json`
     - `runs/H/STEP1/S<v>/STEP1.json` and `HGB.pkl`
     - `admitted/stage_a/S<v>/TREE_TEACHER_income.json` and `TREE_TEACHER_occupation.json`
     - `admitted/stage_b/S<v>/LOCAL_TREE__E20.npz` and `JOINT_TREE__E20.npz`
   - `predict` also reads the configuration's own MODEL.pkl / MODEL.json, which hold the residual weights, the feature map and the pickled HGB teachers.
4. Fits run in the torch venv (numpy 2.4.2), so run `sar.deploy store` with the same interpreter. The residual's numpy matrix products may differ in the last bits under `python3`'s numpy 2.4.4.
5. `stage_fit` writes `results/<campaign>/fa/FIT_V<v>.json`. Change the path if you prefer another.

### 7. Open questions
- The guard rule assumes SUCC's in-sample term cancels to first order; this is an assumption, not a result. The lineage's ±0.002 guard transfer error is not covered, so MODEL_SELECT point feasibility is predicted, not guaranteed.
- The critics train on SUCC's MODEL_FIT wires. JOINT and H3 were trained against MODEL_FIT SEX, so the critics' target may be blunted.
- Methodological overlap with B (neural network, lineage critics). The difference is the protected start, the teacher-distillation task term and the stronger teacher features.
- The DQ spend path, the release-only tilt and a permuted-SEX control were not budgeted inside the 6 specifications. Add them as shared controls if you want PROTOCOL-4 completeness.

Files are in `<repository>/.claude/worktrees/pcrl-strong-anchor-release-v1/`:
- `results/pcrl_strong_anchor_release_v1/roles/investigator_a/PROPOSAL.md`
- `results/pcrl_strong_anchor_release_v1/roles/investigator_a/CARD_DRAFT.md`
- `results/pcrl_strong_anchor_release_v1/roles/investigator_a/selftest_output.json`
- `results/pcrl_strong_anchor_release_v1/roles/investigator_a/check_teachers_labelfree.py` (and `.json`)
- `results/pcrl_strong_anchor_release_v1/roles/investigator_a/check_cpu_labelfree.py` (and `.json`)
- `sar/fa/tfr.py`
- `sar/fa/__init__.py`
