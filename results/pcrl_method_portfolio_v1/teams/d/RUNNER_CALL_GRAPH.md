# Team D executable call graph (prefit design)

This is the integration contract for a queued Adult screen. It is not a run
receipt. At 2026-10-08, role-only files are S4 admitted, but D has run no Adult
fit; S1's effective science seal and S3's science gate still govern every
numerical unit. Private arrays/checkpoints/calibrators/attack predictions stay
under the campaign's ignored private custody root. Public outputs contain only
unit IDs, aggregate metrics and artifact hashes.

## Admission and identities

1. S1 binds the D arm IDs, source checkpoint SHAs, 83-column schema SHA
   `7101f2fb37b623a08221c32b33eedf132cf0c4c376bb4042ab9cf16da08ad975`,
   source/attacker seeds, S2 reader bank, calibration family, exact unit table
   and effective lock. S3 verifies that lock, CPU/memory capacity and current
   occupancy before opening its shared science gate. A terminal in the D
   worktree never launches an unqueued fit.
2. Each unit constructs `SafeRoleStore(DATA_ACCESS_MANIFEST.json)` and uses
   `read(role, "features")` for the exact needed role. For task labels use
   `read("OSF_DEFENSE_FIT", "income", "defense_fit")` and occupation likewise;
   `CALIBRATION_HELDOUT` uses procedure `calibration`, `INNER_SELECTION` uses
   `screen`. Fitting SEX is read only for protected/LEACE/local/sequential
   defense fits via procedure `defense_fit`; attacker SEX uses `attack` on
   `ATTACK_FIT_NEW`; INNER SEX uses `screen` only in scoring. No role request
   targets HEAD_VALIDATION or old assessment. `references_s{seed}` is read for
   admitted frozen U0/Ucal probabilities. Every joined record passes
   `common_eval.assert_same_people` and its `(row_id, unit)` hash equals
   `DATA_ACCESS_MANIFEST.real_file_specs["ROLE:features"].row_unit_pair_sha256`.
3. Load each historical U source encoder through
   `FrozenUEncoder.load(model_path, model_sha, expected_columns, schema_sha, seed)`
   only after source checkpoint admission. Forward permitted 83-column `X`
   through `encode_pair(X,columns,role)` separately for FIT, CAL, ATTACK and INNER.
   JCV's `model.pt` is one state dictionary containing both purpose encoders
   per seed; use `(model_sha,model_sha)` for D cache source identity unless S2
   admits separate derived encoder hashes. The ledger's purpose forwards are
   not extra checkpoint files. The runner keeps role latents in memory during
   the queued unit and saves source/role/row/schema parity receipts plus the
   calibrated release wire privately; it does not persist latent arrays.
   RAW-J uses its separately
   admitted source encoder and the same role-only forwarding rule. Each encoder
   forward is a counted S3 queue unit or an explicitly enumerated subunit.

## Fit and calibration units

4. Build FIT `DTrainingData(h0,h1,y_income,y_occupation,Ucal_1,Ucal_2,sex,row_id,row_id,
   "OSF_DEFENSE_FIT")` for protected arms; build a separate task-only object
   with `sex=None`. Call `prepare_control_cache(task_only,seed,config,
   source_model_sha256=(u_model_sha,u_model_sha),row_order_sha256=admitted_pair_hash)` once:
   one physical `PurposeNormalizer.fit` call estimates two purpose transforms,
   and three KMeans jobs produce one pooled and two local initializers. A
   versioned private cache must serialize all arrays plus its receipt and hash.
   Before **every** arm fit, reload it and call `cache.validate` against the
   admitted source/seed/config/row/latent identities. Missing validation means
   the arm must refit and charge preprocessing; a similar output is no alias.
5. Run the four pooled `train_d` arms with the identical cached pooled
   `(normalizer,centroids)`: three task-only `sex=None` settings at reconstruction
   coefficients 0, .25, 1, and one protected `.25` arm with FIT SEX. Run
   `fit_local_or_sequential` for LOCAL, SEQ12 and SEQ21 with the cached
   normalizer and two purpose initializers. Run `fit_continuous` on the
   task-only FIT object for UNCHANGED, and on the protected FIT object for
   official LEACE, with the same cached normalizer. Run `fit_raw_joint_reference`
   only after RAW-J source identity and FIT feature-forward parity are checked;
   its historical encoder is an alias, but two newly fitted heads are charged.
   Save each release privately with an exact arm/seed/fit hash; register every
   fit, including failure and retry, in S3's global queue ledger.
6. For each arm/seed/task, release CAL scores through its actual protected
   feature/code, then call `fit_global_temperature` using only CAL task labels.
   `save_calibrator` writes a private, immutable scalar record and returns its
   SHA. On CAL, ATTACK and INNER apply `apply_global_temperature` to the head
   probabilities, recompute argmax, set metadata
   `calibration="H-GLOBAL-TEMP"` and `calibrator_sha256=<saved SHA>`, and
   validate with `AlignedDViews`. No scored `UNFITTED` release is allowed.
   Recipient deployment takes only its permitted raw `X` through its own U/RAW-J
   encoder and its own fitted D map/head; neither SEX nor the other recipient's
   per-person input is used.

## Matched compression and recovery

7. With the three task-only arms already fitted/calibrated, make CAL
   `MatchObservation`s from S2-calibrated task probabilities and purpose
   standardized latent reconstruction. Run `select_match` with exact S2 metric
   ID `S2-ARGMAX_FIRST-NLL_NAT_CLIP1E-12-BRIER_SUM-v1`, fit/calibrator IDs and
   admitted CAL row/unit hash. Seal the receipt before INNER recovery. On INNER
   call `recheck_inner` with the frozen choice, never reselect. Keep all three
   task-only arms in the attack/control comparison. S2 `task_metrics` and
   `utility_against_references` score every arm against U0 and Ucal; the D
   matcher is a control-matching diagnostic, not a replacement endpoint scorer.
8. On ATTACK and INNER, build `AlignedDViews` using the same row IDs for both
   recipients and verify the hash against the S2 role registry. Use
   `as_s2_attack_view(views,recipient0|recipient1|pair,AttackView)`; each public
   wire field has an exact nonoverlapping column interval. It exposes hard
   categorical IDs, decoded feature, calibrated probabilities/decision, all
   metadata, ten code bits in the main matrix and auxiliary bit view,
   public-only defense-aware transforms, and exact
   local/pair categorical keys. S4 checks these builders against deployed
   release packages and independently replays `decoded == sealed_codebook[code_id]`
   and calibrated probabilities from the sealed code-only head plus calibrator,
   checking checkpoint/codebook hashes. `fit_slate` gets ATTACK SEX only and its fixed group
   ATTACK_TRAIN/TUNE split; `score_slate` gets INNER SEX only after fitting and
   orientation are frozen. The three D direct CellReaders per code view use the
   same ATTACK_TRAIN/TUNE split and join the frozen common-reader envelope;
   their exact integration and count must be sealed with S2 before the fit.
   Pair ignore-recipient predictions are exact local-reader aliases only when
   their complete fit/view identity matches. All attack predictions are private.
9. S1 ranks only within the feature track using the sealed strongest-control
   and utility gates. Report every seed/arm/task/view, the CAL match and INNER
   recheck, full hard-ID cell recovery, local and pair AUC, uncertainty and
   missing/failed units. S4 independently replays endpoints and view/schema
   checks without importing this runner's selection or scoring code.

S3 prefers one immutable preprocessing/cache queue unit per source seed,
separate arm fit units, then calibration, forward and attack units. This requires
private cache serialization with hash checks and immutable release checkpoints.
A compound seed unit is an S3-approved fallback only if its lock lists every
subunit and it writes durable per-subunit timing/output receipts; it cannot
silently redo completed work on retry.
