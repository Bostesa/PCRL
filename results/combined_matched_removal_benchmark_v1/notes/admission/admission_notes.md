# Admission notes (ROLE 1: artifacts and admission), 2026-10-02

This step fitted nothing: no eraser, attacker, probe or ridge. It ran only frozen forward passes (eval mode, no grad,
CPU, `OMP_NUM_THREADS=1`). Row-level arrays are kept outside git, under `~/PCRL_eval_cache_private/bench_v1/inputs/`.
This directory holds only counts, hashes and match flags.

- Scripts: `scripts/admission/a1…a6`. They are idempotent, and a full rerun reproduced every npz, CSV and the index
  byte for byte.
- `INPUTS_INDEX.json` sha256: `2af25b1327457a6f038739d856fc3e562ad33d93fae1fe25bbb28272987baee2`.

## EARLY FLAGS

1. **HMDA Tier-1 target `race` is only partly supported.** Classes 3 and 4 are NE (rows per role below). Macro AUC
   and the worst-class/pair statistics therefore cover only classes {0, 1, 2}: 3 classes, 3 pairs. Nothing is pooled.

   | race class | defense_fit | attacker_fit | attacker_val | assessment | status |
   |---|---|---|---|---|---|
   | 0 | 41,109 | 4,411 | 1,358 | 3,089 | supported |
   | 1 | 3,792 | 415 | 124 | 303 | supported |
   | 2 | 16,955 | 1,780 | 540 | 1,247 | supported |
   | 3 | 1,344 | 121 | 55 | **93** | NE (assessment < 100) |
   | 4 | 504 | **67** | **12** | **46** | NE (attacker_fit, attacker_val, assessment) |

   - All 5 classes are supported in defense_fit, so the B and C erasers can use the full 5-class one-hot. The method
     owner decides this; NE applies only to scoring.
   - Class codes come from the b96c412 `prepare_hmda.py` map: 0 White, 1 Black, 2 Asian, 3 AIAN/NHPI/2+ minority,
     4 Joint.
2. **Some Adult Tier-2 cells have unsupported classes.**
   - `race` classes 0 and 3 are NE in all three attacker roles (attacker_fit 67 / 63, attacker_val 21 / 19,
     assessment 61 / 40), the same gap the pilot had. That leaves 3 supported classes.
   - Task `occupation_group` class 5 is NE in every role, which affects U2 for employment_analysis.
   - Tier 1 (`sex`, `income`) is fully supported: the smallest count is 722 in attacker_val.
3. **defense_fit uses the PREFERRED route for both datasets, which means encoder exposure.** defense_fit is the
   historical encoder-training split, so the LEACE fit rows already trained the encoder. Disclose this.
   - defense_fit: Adult 24,127 rows; HMDA 63,704 rows.
   - Excluded as `excluded_dup` (train rows whose record equals a test record): Adult 18 rows (17 units), HMDA 43 rows
     (42 units).
   - Unresolved exposure: some test-role rows have an identical record in the encoder-training split.
     - Adult: 17 rows (attacker_fit 6, attacker_val 4, assessment 7).
     - HMDA: 42 rows (21 / 7 / 14).
   - The route was decided from provenance alone (`defense_route.json`), before any benchmark fit, so the fallback
     was not triggered.
4. **The Adult record key needs canonicalisation for cross-split grouping.**
   - The pilot key hashes the raw row, and `adult.test` writes the income label as `>50K.` / `<=50K.`. With the
     pilot key, no train row could ever match a test row.
   - Fix: units and overlap use `canon_key`, the same hash with that trailing `.` removed.
   - Within the test split the two keys give the same partition, so pilot roles and units are unchanged.
   - Both arrays are stored: `record_key` (pilot definition, verbatim) and `canon_key`.
5. **HMDA has no record identifier.**
   - The HMDA key is the pilot hash over the 17 kept columns of `load_filtered(hmda_2023_ca.csv)`. The public LAR
     carries no applicant identifier in those columns.
   - So key-equal rows could be different applications. Grouping them into one unit is conservative.
   - Duplicate rows: 119 within train and 6 within test. Adult has 15 within train and 5 within test.
6. **Train-row identity is inferred indirectly** (Adult and HMDA).
   - What is directly verified: the regenerated test rows, their order, and their train-derived normalisation.
     Every seed's `best.pt` fingerprint matches `per_seed_results.json`: health statistics within relative 6e-7 for
     3 seeds × 3 purposes, and task accuracies exactly.
   - What is inferred: the train rows. The evidence is that train and test share one seeded permutation; test
     features depend on train-only statistics (HMDA also on the train county list and the loan-band cut-offs); and
     `global_step` = 205·⌈n_train/256⌉ for every seed.
   - Why it matters for HMDA: the Round-4 AWS run re-downloaded the CSV on 2026-04-29. The fingerprint is the
     evidence that the re-downloaded data equals the local snapshot.
7. **`per_seed_results.json` describes `best.pt`, not `final.pt`.** Its health, task accuracy and adjusted-pass
   numbers come from the Cotter-selected `best.pt`, at epochs 11–12 (Adult) and 30–37 (HMDA).
   - Only `lambdas_final` and `last_epoch` describe the final iterate.
   - The benchmark encoder is `final.pt`, epoch 204. `dominant_axis_audit.json` is a `final.pt` audit.
   - Do not quote `per_seed_results` health as `final.pt` numbers.
8. **The six `best.pt` files are provenance evidence only.** They were extracted to compute the row fingerprint and
   are never benchmark encoders. Location: `inputs/provenance_ckpt/`.
9. **Not done here (a fit):** the in-sample one-hot ridge R² of `dominant_axis_audit.json`. It is left to the
   arm-A native check owner. The lineage checks here use only non-fitted quantities: epoch, λ, history, config, step
   count, class priors and the frozen fingerprint.
10. **Equal absolute σ is not equal relative noise.** See `scale.csv`, computed on defense_fit rows.
    - The rms per-dim std of the Tier-1 purpose is 0.378 / 0.366 / 0.390 for Adult s0–s2 and 0.528 / 0.370 / 0.439
      for HMDA s0–s2.
    - ‖h‖ carries a large mean offset. For HMDA underwriting the median ‖h‖ is 6.4–10.1 while the median centred norm
      is 0.9–3.9. Ratios against the centred norm are reported separately.

## 1. Encoders: Round-4 `final.pt` (all 6 admitted)

- **Extraction:** only the needed members, via bsdtar `-x -q`, from `archives/fl-PCRL-main-checkpoints.tar`; nothing
  was written to the drive.
  - Every member's sha256 and size equal the drive inventory. Seed-0 finals are the preparation's existing private
    copies, re-hashed.
  - Admitted finals are now in `~/PCRL_eval_cache_private/checkpoints/v2_{adult,hmda}_s{0,1,2}_final.pt`, with
    `SHA256SUMS` updated.
- **Lineage checks** (`lineage.json`, reference origin/main@55e4cb1d1 `results/v2_*_ROUND4/`). All pass for all 6:
  - `state.epoch` = 204 = `last_epoch` = the DA-audit epoch;
  - saved `lambdas` are exactly equal to `lambdas_final`: same keys, same order, and float `==`;
  - 205 history entries; `checkpoint_dir` = `v2_<ds>_s<k>`; epochs 200 + warmup 5;
  - `global_step` is consistent with the regenerated n_train (Adult 19,475 = 205 × 95; HMDA 51,250 = 205 × 250);
  - input_dim = 105 / 78; no erase-layer buffers;
  - test priors equal the DA-audit priors for every pair (max abs diff < 1e-12);
  - `best.pt` `best_epoch` equals the per_seed `best_epoch`.

| dataset | seed | final.pt sha256 | λ (Tier-1 pairs) |
|---|---|---|---|
| adult | 0 | 1cfc2fef… | income_prediction__sex 8.981, __race 0.112 |
| adult | 1 | aca07164… | 4.686, 0.514 |
| adult | 2 | e701b5bd… | 3.620, 0.300 |
| hmda | 0 | e29d0367… | underwriting__race 3.073, __ethnicity 0.430 |
| hmda | 1 | 1c971565… | 50.466, 44.878 |
| hmda | 2 | b6db78c3… | 2.985, 0.605 |

- **Extension:** `extension_inventory.csv` lists 18 Round-5 Adult/HMDA and Round-7 Diabetes headline members
  (final.pt and best.pt), with hash and size only. Nothing was extracted, and none appears in any Round-4 row.

## 2. Purposes (`purposes.json`)

The purpose order agrees across four sources for every seed: b96c412 `get_{adult,hmda}_purposes()`, the checkpoint
`task_heads`, the checkpoint `lambdas` keys, and the LoRA adapter index.

| Dataset | Tier-1 purpose (index) | Task | Target | Policy set (C) |
|---|---|---|---|---|
| Adult | `income_prediction` (0) | `income` (2) | `sex` (2) | {race (5), sex (2)} |
| HMDA | `underwriting` (0) | `loan_decision` (2) | `race` (5) | {race (5), ethnicity (2)} |

Other indices:
- Adult: employment_analysis = 1, education_assessment = 2.
- HMDA: pricing_analysis = 1, fair_lending_audit = 2.

## 3. Rows and roles (`rows_provenance.json`, `pilot_roles_check.json`, `defense_route.json`)

- **Adult.** b96c412 `AdultDataset` gives train 24,145 and test 15,060 rows.
  - The raw-row indices equal the preparation's, and the input file hashes equal the preparation's.
  - Pilot roles were recomputed (`pilot-roles-v1`) and equal the pilot `labels.npz` row by row: 7,571 / 2,239 /
    5,250. Record keys, unit partition, sensitive attributes and test features are also identical.
- **HMDA.** b96c412 `prepare_hmda.py` was rerun on the local raw CSV: 91,068 filtered rows.
  - The regenerated features, all task labels, all attributes, normalisation statistics and label statistics are
    equal to `data/hmda_processed` for train, val and test.
  - Test roles use `bench-roles-hmda-v1`: attacker_fit 6,794 / attacker_val 2,089 / assessment 4,778 rows
    (6,790 / 2,089 / 4,776 units).
- **Row ids.** Test rows have row_id = test position (the pilot's ids for Adult). Train rows have n_test + train
  position.
- **Roles per row:** defense_fit, attacker_fit, attacker_val, assessment, excluded_dup. No unit spans two scored
  roles; this is asserted.

## 4. Forward caches (`forward.json`)

- Output: `inputs/<ds>_s<k>_forward.npz`, holding row_id, split, `rep_p0..2` and `logits_<purpose>`.
  - The arrays are float64 exact upcasts of the float32 forward from `stored_model_eval/forward.py`.
  - Each split is run separately at batch size 512.
- Checks:
  - two independent runs are bitwise identical, and a rerun in a fresh process is also bitwise identical;
  - there are no non-finite values;
  - Adult s0 test reps and logits are bitwise equal to the pilot cache `adult_s0_test.npz` for all 15,060 matching
    row ids.

## 5. Labels and index

- `inputs/<ds>_labels.npz` holds: row_id, split, unit, record_key, canon_key, role, the sensitive attributes by name,
  and the tasks as `task_<name>`. Adult's sensitive `income` equals `task_income`.
- `inputs/<ds>_roles.npz` holds `<role>__row_id` and `<role>__unit`.
- `INPUTS_INDEX.json` follows BENCH_DESIGN "Interface contracts". It records:
  - every file's path and sha256, and per-role row and unit hashes;
  - the purpose → index, task, rep and logits key maps;
  - per seed: the checkpoint path, sha256 and lineage status, and the forward path and sha256.
- Hash convention for string arrays: sha256 of `"\n".join(values)`.

## 6–8. Support, scale, panel

- Support files: `support_counts.csv` (per dataset × variable × role × class, rows and units) and `support_pairs.csv`
  (per Tier-1 and Tier-2 pair, plus C-fit support in defense_fit).
- Scale: `scale.csv`, per dataset × seed × purpose × σ.
- `PANEL_draft.csv`:
  - **Tier 1:** 126 units, all admitted. That is 2 datasets × 3 seeds × (A, B, C + 6 σ × 3 release seeds).
  - **Tier 2 (registered; inputs admitted):** E1 = 72 (12 pairs × 3 seeds × A, B), E2 = 36 C units, E3 = 648.
  - E2 units for income_prediction/race and underwriting/ethnicity reuse the Tier-1 C map of that seed.
- **Runtime (admission only, not science budget):** under 30 s CPU in total. The private inputs take 668 MB, including the provenance `best.pt` copies.
