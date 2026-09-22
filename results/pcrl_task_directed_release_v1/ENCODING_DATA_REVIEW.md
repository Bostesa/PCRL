# Encoding/data review — 2026-09-21

Read-only review of `experiments/pcrl_task_directed_release_v1/{encoding,data,config}.py`, the referenced teacher API in `audits.py`, registration commit `53ce8b897`, PROTOCOL/DATA_USE, and AMENDMENT_1. No ACS rows, feature/prediction NPZ members, or outcomes were opened. No real model was fitted/refitted. No repository edits by this reviewer for this task.

## Finding corrected during review

The initial `load_anchor` checked only that a test evaluation permit was non-None before accessing test split/PCA/H/release members. Its existence and selection_frozen flag were checked only later in `read_rows`. A missing path or explicit false permit therefore allowed feature access before rejection. This violated the registered feature seal. I reported it immediately; the parent introduced `validate_evaluation_permit` at the beginning of `load_anchor` as well as `read_rows`.

The corrected path passed synthetic guards: with `np.load` replaced by a sentinel, None, missing-path and `selection_frozen:false` permits all raised PermissionError before any NPZ access. No actual evaluation feature was accessed by this review. AMENDMENT_1 records the correction.

No further material teacher-freezing, split, ownership, coding or dictionary defect was found. This is source/synthetic verification, not real-data execution certification.

## Verified contracts

- `data.household_roles`: same fixed household SHA-256 draw, thresholds .48/.60, yields 48% teacher fit,12% internal validation,40% mechanism. Roles partition people and household membership never splits. A synthetic 10,000-household/two-person population produced 4780/1242/3978 households, respectively, with no cross-role overlap. These are synthetic counts, not study counts.
- `fit_encoder`: codebook indices are the union of unfiltered fit/internal-validation roles, before residence-complete masks restrict supervised teachers. Risk fitting uses the original teacher-fit role and each protected target's available labels. Construction60% and mechanism40% are disjoint.
- `fit_encoder` takes the actual selected TokenCandidate object and never refits it. `_fit_mlp` deep-copies the selected checkpoint, loads its stored best weights and disables gradients. Selection uses balanced validation CE, lexical model-ID tie breaks, earliest checkpoint ties. `fit_predictor_slate` uses the registered logistic/200-epoch MLP slate.
- Encoder runtime is only `RuntimeInputs(PCA32,H_A4)`. Task teacher gets their concatenation; baseline gets H_A. No label, ID, PWGTP or H_B enters runtime encoding.
- `Codebook`: duplicate/exterior quantile cuts merged, searchsorted-right convention; `Ttask=2*T0+child` and `Trisk=2*T0+child`, checked exactly at assignment. Task median ties use only the fixed cosine PCA32 projection; a second exact tie remains child0. Degenerate support is reported rather than split by identifiers.
- Risk output is SEX2+RAC1P9. Missing training classes receive exact probability1e-8 via `(1-k*1e-8)*p+1e-8`. Synthetic single-class predictors retained correct widths, normalization and floor. Training classes and mechanism support counts are logged; absent classes must remain identified as unassessed.
- Risk KMeans uses fixed projection extrema, one start, farthest-vector repair only if extrema coincide, lexical center ordering; parents under40 construction rows use global centers. The config's abbreviated 'farthest pair' wording is not an all-pairs diameter search; code diagnostics are precise.
- Local C_A uses only H_A income/employment columns[1,3]. C_AB crosses that with H_B coverage median. Existing test changes H_B while holding H_A fixed and confirms unchanged A, changed AB and AB//2=A. Coverage ownership is correct.

## Dictionary distinction that must remain disclosed

Clipped p,b define residual groups. Each offset solves `sum w*(sigmoid(logit(b)+a)-p)=0` on [-12,12], selecting a boundary from derivative signs when necessary. Balanced weights are formed on the full construction set then restricted to each group. The equation is the derivative of convex **unclipped sigmoid** soft-label CE. Brent tolerance is1e-13. Zero remains action0; offsets within1e-10 merge;16 groups plus zero give at most17 actions.

Deployment action probabilities are clipped again to[1e-5,1-1e-5]. A root therefore need not minimize the clipped deployment CE. Synthetic b=[1e-5,.99],p=[.2,.9],unit weights gives root9.3157009885 with clipped CE0.8480529838, whereas offset10.1266211038 gives0.8258519850. This is not a defect in the registered stationary-root construction, now clarified in AMENDMENT_1 and fit_loss metadata. Do not describe it as exact clipped-loss minimization. Q must use its frozen clipped action losses. Both bounded-root directions and the existing known-offset test passed.

## Data lineage and boundary

The loader uses a fixed2018 raw relative path under the caller's owned-input root and has no global resolver. INPUT_RESTORE distinguishes per-file verified copies from a fresh original-stream verification. Loader calls assume that verified bundle and do not rehash all files per call. Default pools omit test; NPZ members are selected lazily only for requested pools.

PCA32 is saved historical PCA, standardized float64 then castfloat32 with the frozen checkpoint mean/scale. A0/J standardizers must match. Stored releases remain authoritative after1e-5 portability replay; H_A dtype/value parity is checked and H_A returned untouched. DATA_USE correctly states original preprocessing/PCA fitted **all** original RF features, including the new mechanism40%; the new supervised split does not undo that history. Frozen H also has broader inherited training history. All2018 evidence remains development evidence.

CSV skiprows maps zero-based saved raw rows to header-offset rows, enforces sorted unique requests and preserves order. A synthetic skipped row with nonnumeric labels/weights did not affect selected-column numeric inference; leading-zero household strings were retained. Label formulas match inherited acs_transfer_data.py. IDs remain private for alignment/splitting/bootstrap. Encoder inputs exclude them. Split hashes use numeric raw-row arrays; no object-address hash issue occurs in this path. Encoder split_rows.npz stores indices relative to RF, whereas its split_hashes hash original raw-row indices; replay must preserve RF order.

Public metadata should retain aggregate counts/hashes only. Model artifacts, RF relative indices, person IDs, labels, probabilities and losses remain in the private archive.

## Verification

Existing test_encoding.py: **4 passed** (one environment-only joblib core-detection warning). Additional pure synthetic review probes passed; see adjacent `encoding_data_review_probes.py` and `ENCODING_DATA_REVIEW_PROBES.json`, including reviewed source hashes. Live config differed from initial committed CONFIGS only in the supplementary HistGB amendment already documented in AMENDMENT_1. Freeze registration+amendment+exact config/source hashes before comparative execution.

Runner integration still owns actual inherited-pool disjointness, real split counts/hashes, action-cost reuse, Q unsupported-state embedding, complete private artifact persistence, selection/contrast freeze and publication separation. None was inferred from synthetic success.
