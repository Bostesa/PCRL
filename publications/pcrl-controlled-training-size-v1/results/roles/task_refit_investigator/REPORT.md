<!-- Attribution: written by the task-refit investigator (an agent assignment of this campaign) and returned as text in its final
hand-back, because the harness refused report files written by that agent. Saved verbatim by the coordinator at 00:27Z
(2026-10-19, date -u); not coordinator analysis. Coordinator notes on its open issues: (1) the cyclic S50 epoch order was already
registered before any fit (PROTOCOL.md section 5, PROTOCOL_DEVIATIONS item 3); (2) a support failure stopping its lane is accepted
(a terminal failure is reported missing and the lane is resumed by the coordinator past it). -->

# Task-refit investigator: size fits, equally exposed control, lane plan

Campaign `pcrl_controlled_training_size_v1` (EXPLORATORY DEVELOPMENT on previously used records). Written 2026-10-19 between 00:01Z and 00:24Z (from `date -u` output).

**What ran on real data**
- Before the receipt, I ran only label-free steps on MODEL_FIT / MODEL_SELECT features: subset sizes and dictionary collapse counts (section 2).
- Everything else I ran used invented data, apart from the two records below.
- **Incident (deviation 4).** At 00:20:13Z one selftest run made a real label read (MODEL_FIT occupation, S100, 20,000 people). That was 6 s after the registration receipt was verified at 00:20:07Z, so the old "refusal before registration" probe was no longer refused. The data was discarded unused.
  - Fix: the probe is removed.
  - Both modules now set `_NO_REAL_ACCESS` in selftest and timing, which makes the real label hook raise.
- **S100 size fits.** These were run by the coordinator through the queue (units TSZ_A1_S100_V{0,1}_FIT/VERIFY/REPRO_V1).

## 1. What is implemented

| File | Interpreter | Commands |
|---|---|---|
| tsz/sizefit.py (sha de5d9cb4…, commit 77a388930) | python3 3.14.3, scikit-learn 1.8.0 (the source's) | selftest, timing, `fit --subset SID --version V [--dictionary fit_only\|admitted]`, `verify --fit-id ID`, `repro --version V`; `predict(fit_id, X, names)` |
| tsz/control.py (sha fd778afb…) | teachers: CatBoost venv; stage_b: torch venv; verify, repro: python3 | selftest, timing, `teachers --subset SID --version V`, `stage_b --subset SID --version V`, `verify --subset SID --version V`, `repro --version V`; `predict(sid, v, X, names[, arm])` |
| LANES_SIZE.json, LANE_SIZE_1.json, LANE_SIZE_2.json | - | two lanes (section 7) |

**Source code reused unchanged by import**
- `ohn.fam_a`: CONFIGS['A1_PO_16'], rep_matrix, cv_select, fit_model, release_prob, raw_full, fit_temperature, inner_tag, _curve_summary.
- `ohn.common`: household_folds, teacher, frozen_bank, stage_a_outputs, stage_b_outputs, stage_b_theta.
- `ohn.devrun.load_occupation`.
- `cts.trackf`: fit_dictionary, admitted_dictionary, recode.
- `cts.track_r`: fit_teachers_arrays (the replication Stage A T_TREE loop), _recipe, mock_teacher.
- `acs_replication`: data.internal_split, data.order_rows, trees, leafcorr.Trainer / run_arm, stage_b.ARMS.

The only transcribed code is `tsz.control.run_arm_cyclic`: leafcorr.run_arm copied line by line, with one marked line changed (section 3, item 1).

### sizefit.fit

- **Output.** Fit id `A1__<SID>__V<v>`, with suffix `__ADM` for the encoding sensitivity. Directory `.private/<campaign>/runs/SIZE/<fit id>/`, write-once.
- **Files.** MODEL.pkl, RELEASE.npz (key = role), DICT.json, CURVE.npz (full pooled inner curve), FIT.json (schema TSZ1-sizefit-v1).
- **Label reads.**
  - `C.load('MODEL_FIT', ('features','occupation'), 'fit', …, subset=SID)`.
  - `C.load('CALIBRATION', ('features','occupation'), 'calibrate', …)`.
  - Never SEX, never income.
- **Dictionary.** FIT-only dictionary on the subset's X. `__ADM` uses the admitted dictionary and is allowed only for S50(o), v = 0.
- **Coding.** `cmp_po` coding of the recoded X.
- **Inner folds.** `ohn.common.household_folds(ALL MODEL_FIT units, 5, 'OHN_A_cv_S<v>')`, restricted to the subset rows.
- **Fitting.** `FA.cv_select` gives T*; then `FA.fit_model(cfg, Xr, y, T*, v)`.
- **Releases.** On all six roles, from features alone. The MODEL_FIT release covers all 20,000 people.
- **Temperature.** `FA.fit_temperature` on the deployed raw CALIBRATION scores, recorded with `applied: false`.

FIT.json records:
- subset, version, dictionary kind;
- T*, the T_max flag, the curve summary and the values at T*;
- inner fold sizes, people and households, the row-set SHA-256;
- class counts in the subset and in every inner training set, with their minimums;
- collapse counts per block;
- MODEL_SELECT shares with a collapsed level and with a level absent from the subset, per block and for any block;
- release, file and row-id SHA-256s, and the feature names;
- cpu_s / wall_s, utc, unit id, interpreter and code SHA-256s;
- n_model_fits = 6 and n_calibrator_fits = 1, also passed to `C.unit_stats(n_model_fits=6, calibrator_fits=1)`;
- for S100, `aliases` O0..O3 (one fit per version).

Class support is fixed:
- It is checked before any fit, on the subset and on every inner training set.
- On failure, SUPPORT_FAILURE.json is written. It is terminal: a later `fit` of the same cell is refused, with no redraw. The unit exits with code 3.

### sizefit.verify and sizefit.repro

- **verify** (label-free): `predict()` from each role's features is compared bitwise with RELEASE.npz on all six roles.
  - Five refusals are also checked: reordered names, incomplete one-hot block, non-finite value, wrong width, integer matrix.
  - Output VERIFY.json; the command exits 1 on any failure.
- **repro** (label-free): see section 5.

### control.teachers

- **Output.** `.private/<campaign>/runs/CONTROL/<SID>__V<v>/TEACHERS/`.
- **Label reads.** MODEL_FIT income and occupation of the subset only; no SEX.
- **Dictionary.** `tsz.sizefit.subset_dictionary`, the same function and input as the size fit. `control.verify` checks that the two DICT.json files are byte-identical.
- **Recipe.** `cts.track_r.fit_teachers_arrays`, which is the replication Stage A T_TREE loop:
  - CatBoost 1.2.8, depth 4, lr 0.05, l2 3, at most 800 trees;
  - seed `seed_for('ACSR1catboost', v, task, 4)`, unreduced;
  - 15% household internal split of the subset, with the two-task class-support repair of seed v;
  - tree count = first argmin + 1, then refit on the subset.
- **Parity.** `trees.verify_against_catboost` on all six roles.
- **Outputs.** T_TREE outputs for both tasks on all six roles; `unit_stats(n_model_fits=4, n_teacher_fits=4)`.

### control.stage_b (torch venv)

- **Identity check.** Zero-correction identity against the saved teacher outputs, on all six roles.
- **Label reads.** MODEL_FIT income, occupation and SEX of the subset. SEX is used only by the source loop's critics, which never reach the released parameters at lambda 0.
- **Sequence.** The replication's Stage B sequence: a fresh Trainer with warm critics and the 16-draw calibration, then the arm in its own fresh Trainer.
- **Arm.** TASK_REFIT only (T_REFIT, lambda 0):
  - 20 epochs x 40 updates x 256 rows, AdamW lr 0.01 / wd 0.01;
  - checkpoints E0, E2, E5, E10, E20;
  - probe and reference rows chosen by the replication's hash order within the subset.
- **Outputs.** TASK_REFIT outputs for both tasks on all six roles; `unit_stats(n_model_fits=1, n_correction_fits=1, n_critic_fits=18)`.

### control.verify (label-free)

- TASK_REFIT and T_TREE recomputed from features alone, compared bitwise for six roles x two tasks.
- Four refusals are checked.
- The dictionary cross-check against the size fit.

## 2. Label-free facts (features only)

| Subset | People | Households | Collapsed COW/SCHL/MAR/POBP | Source Stage B slice (40 x 256) |
|---|---|---|---|---|
| S50_O0 | 9,934 | 6,669 | 0/3/0/21 | batch 38 has 206 rows, batch 39 is empty |
| S50_O1 | 9,957 | 6,669 | 0/1/0/21 | batch 38 has 229 rows, batch 39 is empty |
| S50_O2 | 10,046 | 6,669 | 0/3/0/23 | batch 39 has 62 rows |
| S50_O3 | 10,014 | 6,669 | 0/1/0/19 | batch 39 has 30 rows |
| S70_O0..O3 | 13,954–14,040 | 9,336 | 0/0–1/0/7–11 | full (about 73% of the subset per epoch) |
| S100 | 20,000 | 13,337 | 0/0/0/0 | full (51%, as in the replication) |

`check_subsets` reports no problem. At S100 the FIT-only dictionary equals the admitted one, and `recode(X)` equals X as float64 bitwise.

## 3. Departures from the source recipe

1. **S50 Stage B epoch order (needs sign-off).** With n below 10,240, the source loop slices one permutation per epoch with `order[b*256:(b+1)*256]`.
   - S50_O2 and S50_O3 would get one short batch per epoch.
   - S50_O0 and S50_O1 would get one short batch and one empty batch per epoch. The empty batch has a NaN loss and a zero gradient, so it becomes a momentum-and-weight-decay-only AdamW step; parameters stay finite (checked).
   - `cts.track_r` refuses such pools.
   - **Implemented, for n < 10,240 only:** the same permutation (same random stream, one draw per epoch) is extended cyclically to 10,240 positions. Every update has 256 rows, and every person is used once before 194–306 of them are reused within the epoch.
   - For n ≥ 10,240 the unchanged leafcorr.run_arm runs. The transcription equals it bitwise there (selftest, arms T_REFIT and L_TREE).
   - Alternative: the unchanged source slice. STAGE_B.json records `epoch_order` and `source_slice_batches_not_full` either way.
2. **TASK_REFIT alone.** The replication ran JOINT_TREE, LOCAL_TREE, TASK_REFIT, each in a fresh Trainer whose random streams are all seeded from the version alone.
   - With lambda 0, the critics never reach theta.
   - Selftest, bitwise: TASK_REFIT alone equals TASK_REFIT from the three-arm run at every checkpoint, and this module's path equals `cts.track_r.run_stage_b_arrays`.
3. **Failed calibration.** If the 16-draw calibration raises in a TASK_REFIT-only run, the failure is recorded and TASK_REFIT continues unchanged (selftest: identical theta). The source loop would have stopped. Multi-arm runs still raise.
4. **Subset formation.**
   - The dictionary is re-collapsed on the subset.
   - The T_TREE internal split (with its repair) and the probe / reference rows are formed within the subset.
   - A1_PO_16 inner folds are the source folds restricted to the subset, so they are only approximately balanced.
   - Stage B per-epoch coverage is min(10,240, n)/n; the update count is fixed.
5. **CatBoost input dtype.** CatBoost receives the recoded X as float32; the values are exact float32 values (asserted). Selftest: the S100 teachers equal a transcription of the Stage A loop on the unrecoded X, with identical payload SHA-256.
6. **No cross-fitting in sizefit.** The source's 5 cross-fitting pipelines and two cross-fitted temperature diagnostics are not run here; none of them enters the release. The conditional stage owns W100's cross-fitting base.
7. **Added deployment guards.** These refuse bad inputs; outputs on valid inputs are unchanged.

## 4. Timing (invented data at real size, one thread)

| Unit | Measured CPU | Source record |
|---|---|---|
| A1_PO_16 fit S100 / S70 / S50 | 22.6 / 17.5 / 14.7 s (real S100 units: 24.6 / 24.5 s) | source deployed + cross-fitting: 125–128 s |
| sizefit verify | 0.6 s | - |
| teachers S100 / S50 (invented labels) | 8.9 / 4.1 s; expect 15–25 s at real tree counts | replication Stage A teachers: 15.5 s |
| Stage B TASK_REFIT S100 / S50 (cyclic) | 37.3 / 35.5 s | replication, 3 arms: 135 s |

The plan has 102 units: 22 A1 fits, 18 teacher fits, 18 TASK_REFIT runs, 40 verifies and 4 repros. That is about 0.5 CPU-h, or about 15–20 minutes of wall time per lane. With a safety factor of 3, budget 1.5 CPU-h.

## 5. Reproduction checks (label-free; reported, never tuned)

### sizefit repro --version v

- **Source.** `ohn.devrun.load_occupation('A','A1_PO_16',v)`, with OCC.npz checked against OCC.json. The SHA-256s of OCC.npz, OCC.json, STATE.json and MODEL.pkl are recorded.
- **Per role (six).** Bitwise equality, max absolute difference, decision agreement, and both array SHA-256s.
- **Also compared.** T* against the source, the every-25-iteration inner-curve summary, the CALIBRATION diagnostic temperature, MODEL.pkl SHA-256, collapse counts, and `source_admission_check` against SOURCE_ADMISSION.json (OCC.npz SHA and T*).
- **Verdict rule.** `reproduced` = all six roles bitwise and T* equal.
- **Result (queue units):** REPRODUCED for v0 and v1. All roles have difference 0.0; T* is 168 / 154; tau and curve are equal; MODEL.pkl is identical; the admission check is equal.

### control repro --version v

- **Teachers, per task.** S100 teacher payload SHA-256 against `ohn.common.teacher(v, task)`, tree count, routing identity on all six roles, and the admission record `T_TREE/<task>/S<v>`.
- **Stage A outputs.** T_TREE outputs against the admitted STAGE_A_OUTPUTS, six roles x two tasks.
- **Theta.** TASK_REFIT E20 theta against the admitted TASK_REFIT__E20.npz, bitwise.
- **Stage B outputs.** TASK_REFIT outputs against the admitted STAGE_B_OUTPUTS_E20.
- **Frozen bank.** MODEL_SELECT TASK_REFIT and T_TREE outputs (income and occupation) against `frozen_bank(v)`. The primary comparison is TASK_REFIT occupation.
- **Verdict rule.** `reproduced` = identical teacher payloads, bitwise theta and bitwise bank outputs.
- **Pre-checks already done.** A label-free smoke test showed that the admitted teachers and theta reproduce the bank outputs bitwise for v0 and v1. Separately, the correction-transfer campaign's ROUTING_REPRO_S0 reproduced the v0 occupation teacher exactly with this CatBoost venv.

## 6. Selftests (invented data; final files)

### sizefit (python3): PASS, 20 checks
- S100 equals `ohn.fam_a.run_pipeline`'s deployed release bitwise on six roles, with T* equal.
- repro detects a 1e-15 change.
- S50 T* equals a brute-force fold-restricted curve.
- The deployed model equals a fresh refit.
- verify passes bitwise with 5 refusals.
- Write-once and ADM guards hold.
- Label-read policy: MODEL_FIT `fit` (subset) and CALIBRATION `calibrate` only, occupation only.
- A support failure is recorded and terminal.
- Overrides are refused on the real root.

### control, CatBoost venv: PASS, 9 checks
- Parity on all roles.
- S100 equals the Stage A loop.
- verify passes.
- The dictionary equals sizefit's.
- SEX is read only by stage_b.
- repro logic works.
- Write-once holds.

### control, torch venv: PASS, 13 checks
- TASK_REFIT alone equals the three-arm run.
- The control path equals the track_r path.
- The cyclic copy equals leafcorr with full batches.
- On a small pool, the cyclic path gives full batches and finite parameters.
- A calibration failure is recorded and TASK_REFIT is unchanged.

## 7. Lane plan

- **Structure.** Two lanes of 51 units: lane 1 is version 0, lane 2 is version 1.
- **Order within each lane.**
  - First the S100 cell: A1 fit, verify, repro, then teachers, stage_b, control verify, control repro.
  - Then the eight S50/S70 cells: A1 fit, verify, teachers, stage_b, control verify.
  - Then two encoding fits with their verifies: S50_O0/O1 in lane 1, S50_O2/O3 in lane 2, all v0.
- **Unit ids.** `TSZ_<A1|CTL>_<SID>_V<v>_<STEP>_V1`. The `V<v>` after the subset is the source version; the final `_V1` is the attempt.
- **Stages.** `size_fit` and `control_fit`. Memory reservation is 1.5 GiB, and 2.0 GiB for Stage B.
- **Already run.** The coordinator already ran the S100 A1 units under these ids; tsz.lane skips them.

## 8. Open issues

1. The S50 epoch order needs sign-off (section 3, item 1).
2. A support failure exits with code 3 and stops its lane.
3. No change to tsz/common.py is needed.
4. The deviation-4 label read is recorded in PROTOCOL_DEVIATIONS.
