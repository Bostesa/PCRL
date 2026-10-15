# Source and exposure ledger

## What this study used
- **Source models:** the admitted RAW_JOINT strength-0.3 epoch-5 checkpoints, seeds 0 and 1, with their current folded own heads. Hashes are in `SCIENCE_LOCK.json` (`starting_checkpoints`) and match the geometry bindings. No older `model.pt` head was used.
- **Geometry:** the inherited per-seed head row-space bases, hash-bound (`geometry_bindings`). Admission verified A Bᵀ B = A (max error ≤ 1.1e-15), orthonormal bases, and measured ranks 1 (income) and 5 (occupation).
- **O_PREVIOUS:** the previous occupation-only J_TRUST epoch-10 checkpoint, admitted from the public `MODEL_MANIFEST.json` of publication `604871a`. Checkpoint, R and b hashes matched, and so did the scale (1.0). It reproduces the previous stored INNER release bitwise in both seeds.
- **Roles (inherited):**
  - UPDATE: 8,180 rows (8,176 groups).
  - GUARD: 2,043 rows.
  - FIT2: inherited FIT fold 2, 5,211 rows.
  - ATT: 4,064 rows (4,061 groups).
  - INNER: 2,235 rows (2,234 groups).
  - Duplicate groups have identical features in every role, and role groups are disjoint (`ADMISSION.json`).
- **New authorized export:** income labels for UPDATE and GUARD (`INCOME_EXPORT.json`). The worker re-derived both roles from the private split, confirmed that the features, units and occupation labels are byte-identical to the inherited exports, and wrote income-only files. The class counts are UPDATE 6,187/1,993 and GUARD 1,550/493.
- **Who sees which labels:** SEX reaches the training critics from UPDATE only; GUARD has no SEX export. FIT2 labels were used once, for selection, after `TRAINING_COMPLETE_LOCK`. ATT SEX trains and orients fresh readers only. INNER labels were opened once, after the committed `REPORTING_LOCK`.

## Exposure
- **Source training:** the source models were trained on all of FIT, which includes UPDATE, GUARD and FIT2.
- **Class-5 limitation:** occupation class 5 has 0 GUARD rows, 2 UPDATE rows and 2 FIT2 rows. Class-specific statistics for class 5 are not estimable, and no protection claim is made for unsupported classes.
- **Design disclosure:** this method and protocol were designed after observing the previous occupation-only adapter results, including that study's FIT2 and INNER outcomes.
- **Status of FIT2 and INNER:** both are exposed development roles, not fresh confirmation. All people are treated as repeatedly used development data, and the two model seeds reuse the same people.
- **What the intervals cover:** they are conditional on the fitted models and readers and do not account for the adaptive research history.
- **What stayed closed:** the historical assessment, HEAD_VALIDATION, every reserved confirmation population, and any new population acquisition. A new split would not undo prior exposure.

## Lock chronology (research branch, local commits)
1. `27625c5` protocol seal (14:26Z).
2. `6ee0864` admission, income export, engineering tests, prefit math review, and `SCIENCE_LOCK` with forecasts. The scrubbed prefit receipt was published at `3f094fa` on `publication/pcrl-two-recipient-task-wire-adapter-v1` before any bank fit.
3. `405aa12` `TRAINING_COMPLETE_LOCK`, before FIT2 opened.
4. `dae14cc` FIT2 scored once, plus `SELECTION_LOCK`, before any reader fit.
5. `f00c6b4` `REPORTING_LOCK` (all 1,296 reader fits frozen), before INNER labels opened.
6. `9c49b84` INNER report, opened once.

There were no network failures. Code added after `SCIENCE_LOCK` (`deployment_worker.py`, `custody_worker.py`, `figures.py`) is engineering for packaging, backup and plotting. It does not touch training, selection or the INNER report; its hashes are in `UNIT_MANIFEST.json`.
