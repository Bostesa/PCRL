# hcal provenance report: held-out status of AUDIT_FIT for the frozen lra teachers (role D-provenance)

- Worktree: `<WORKTREE>`, on branch `research/pcrl-heldout-calibration-v1` at HEAD `976202546a100f2b89a863cb3785c3465cf76c6a`.
- Mode: read-only.
  - No tracked file was edited.
  - No training was run.
  - No label array was loaded or indexed.
  - Nothing was unsealed: no `unseal=True` call and no `*.assess` import.
- What was done:
  - read code and committed JSON/MD manifests;
  - ran `git show`, `git log` and `git rev-parse`;
  - computed sha256 of the code files and of the admitted input `<PRIVATE_CACHE>/jcv_v1/inputs/adult_jcv.npz`. Its recomputed hash `e0d9e54a…2f12` equals the pinned `SRC_SHA`.
- File references below are `path:line` at HEAD. Git blob hashes are in the table at the end.

## Lineage of the frozen teachers (what "the U teacher" physically is)

**Admission chain.** The lra study admits `rel__s{k}__U` and `rel__s{k}__RAW-J_b0.3`, for k = 0, 1, 2:
- Source: the cbp store, pinned at tip 7f3ec67 (`lra/admit.py:4-16,54-68,146-157`).
- cbp admitted them from qpc (`cbp/admit.py:1-25`).
- qpc admitted them from dpc, falling back to the osf store (`qpc/admit.py:1-30`).
- dpc admitted them from the osf store (`dpc/admit.py:1-40`).
- osf admitted them from the smf units `tl__s{k}__e40` (U) and `raw__s{k}__RAW-J__b0.3__e40` (`osf/run.py:97-106,154-181`).

**Byte identity across the chain.** I compared committed hashes across six manifests. For all 6 teacher × seed units, `model.pt`, `head_0.joblib` and `head_1.joblib` are byte-identical at every step (smf → osf → dpc → qpc → cbp → lra):
- `results/pcrl_online_strength_frontier_v1/ADMISSION.json` (smf files, `admitted.*.files_sha256`);
- osf `MODEL_MANIFEST.json` (`units[].complete_files_sha256`);
- dpc `ADMISSION.json` (`teachers.*.complete_files_sha256`);
- qpc `SOURCE_ADMISSION.json` (`teacher_admission_plan.teachers`);
- cbp `SOURCE_ADMISSION.json` (`admitted_teacher_artifacts`);
- lra `SOURCE_ADMISSION.json` (`admitted_teacher_dirs`).

For example, U seed 0 has `model.pt` 5061a007…, `head_0.joblib` 3fd34f9b… and `head_1.joblib` d7508d98….

**What changes and what does not.**
- `release.npz` differs only between smf (29,030 rows) and osf (39,170 rows). It is equal from osf through lra.
- Parameters were fitted once, by smf. They were never refitted.
- osf's own head refit inside `save_release_unit` (`osf/run.py:117-129`) reproduced the smf head files byte for byte.

**Teacher outputs used downstream.**
- lra parity: the stored teacher probabilities p1 and p2 are the plain forward application of `model.pt` plus the heads. They match bitwise for all 6 units (`lra SOURCE_ADMISSION.json` `parity.teachers`, `lra/admit.py:146-157`).
- No temperature or other recalibration was ever applied to the teacher outputs.

**Data rows.**
- Loader chain: `lra.data.load` → `qpc.data.load` → `dpc.data.load` → `osf.data.load` (`lra/data.py:74-93`, `qpc/data.py:233-255`, `dpc/data.py:176-192`, `osf/data.py:100-129`). D passes through unchanged.
- Today's working-tree `osf/data.py` and `dpc/data.py` hash to their pinned blobs at 925e0fd and 0a7b05a.
- The role rule is that of smf, which is in turn rgj's (`osf/data.py:52-81`, `smf/data.py:53-67`, `rgj/data.py:60-73`).
- `AUDIT_FIT` is the old `attacker_fit` of `oar-roles-v1`:
  - the UCI `adult.test` rows that `role_of(record_key, "pilot-roles-v1|")` assigned to attacker_fit;
  - minus exposure exclusions;
  - minus the 20 % `oar-cert-v1` carve-out.
- `OSF_DEFENSE_FIT` is a subset of the 80 % seed-42 train split of `adult.data`. So the two roles come from **different source files**.
- Evidence: `rgj/data.py:33`, `oar/study.py:57-87`, `admission_common.py:43-44,83-91`, `a5_labels_and_index.py:7-14,91-101`, and `defense_route.json:16` (route `PREFERRED`).
- Group isolation is checked and committed:
  - the role × role exact-record shared-group matrix is zero off the diagonal (`results/pcrl_confidence_capacity_v1/ROLE_MANIFEST.json:247-292`);
  - osf asserts that no kept group spans another role or a dropped row (`osf/data.py:84-97`; osf `ROLE_MANIFEST.json` "assertions").

## (a) Teacher / encoder parameter fitting

**Answer: OSF_DEFENSE_FIT only.** That is 15,434 rows / 15,428 groups, with row-id sha 3f6bb936… and group-set sha 5e6f14fe…. It uses inputs, task labels and, for RAW-J critics only, SEX.

**U (task-only, epoch 40)** is trained in two stages:
1. Warm start: `jcv.train.warm_start` (Adam 1e-3, 20 epochs; `jcv/train.py:235-254`) on `rgj.train.TData(D)`.
2. Task line: 40 epochs, task-only, SGD 0.05 (`smf/run.py:107-119,166-176`; the engine is `osf/train.py:170-366` / `smf.train`).

`TData` holds **only** `D["idx"]["DEFENSE_FIT"]` rows: X, income, occupation_group and SEX (`rgj/train.py:154-169`). Minibatches are permutations of those rows (`osf/train.py:230-233`). The DEFENSE_FIT alias resolves to OSF_DEFENSE_FIT (`osf/data.py:43-44,126-127`).

Receipts:
- osf replayed the 3 warm starts bitwise on OSF_DEFENSE_FIT and refuses on any mismatch (`osf/run.py:157-168`; `VALIDATION.md:20`; `RUN_STATUS.json:30-32` "replayed_bitwise": 3).
- osf replayed all 40-epoch runs bitwise: 15/15, at epochs 20 and 40 (`osf/run.py:305-333`; `VALIDATION.md:24`).
- The import-guarded independent verifier retrained U (3 seeds) and RAW-J β0.3 (seed 0) for 40 epochs from the admitted warm state, bit-exact. It used "OSF_DEFENSE_FIT labels only" (`INDEPENDENT_VERIFICATION.json:11250,11411`; `verification/replay_osf.py:480-491,1429-1450`).
- The osf admission receipts state "rgj_TData_uses_DEFENSE_FIT_only" and "smf_run_warm_jcv_warm_start_on_TData" (`ADMISSION.json:44-50`). The warm-unit record names rows "NEW_DEFENSE_FIT (15,434)" (`ADMISSION.json:52-84`).
- The fit tensors (X, labels, SEX, row ids of the fitting rows) are byte-identical between osf and smf (`ROLE_MANIFEST.json:4199-4201`, `fit_tensor_sha`).

**RAW-J β0.3** follows the same path, plus online critics:
- Critic minibatches come from `data.cf`, which is CRITIC_FIT (a subset of OSF_DEFENSE_FIT).
- The floored-ZCA critic transform is refit every step on a fixed 4,096-row reference subset of CRITIC_FIT (`rgj/train.py:165-169,382,392`; `osf/train.py:238-252`).
- The critic-view head is the fixed warm-start head (`rgj/train.py:173-176`).

## (b) Deployed task-head parameter fitting

**Answer: OSF_DEFENSE_FIT only.** The pipeline is StandardScaler + LogisticRegression(max_iter 3000), fitted on the teacher's 16-d representation of OSF_DEFENSE_FIT rows with their task labels:
- `rgj/finalize.py:20-29,38-40`: `tr = D["idx"]["DEFENSE_FIT"]`;
- `jcv/finalize.py:27-37`: `m.fit(R[tr], y[tr])`.

Receipts:
- osf admission: the scaler n equals |OSF_DEFENSE_FIT|, and the scaler mean and variance equal those of the OSF_DEFENSE_FIT features bitwise. Checked for U, RAW-J and E heads on every seed (`ADMISSION.json:86-150`, e.g. lines 126 and 128).
- dpc re-checks this: `osf_fit_role_receipts` (`dpc ADMISSION.json:58-140`).
- The independent osf verifier refit all heads with its own code (OSF_DEFENSE_FIT fit, HEAD_VALIDATION C) and got 126/126 bit-exact (`INDEPENDENT_VERIFICATION.json:271`; `replay_osf.py:610-621,655`).
- The qpc, cbp and lra verifiers re-check `scaler_moments_bitwise_OSF_DEFENSE_FIT` (lra `INDEPENDENT_VERIFICATION.json:1242,1249,…`).

## (c) Selection: early stopping, checkpoint, head and hyperparameters

**Head C.** C is chosen from {0.01, 0.1, 1, 10, 100} by **HEAD_VALIDATION** log loss (1,500 rows / 1,499 groups); ties go to the smaller C (`jcv/finalize.py:24-37`).
- Receipts: `HEAD_VALIDATION_log_loss_equals_recorded` (abs diff 0.0) and `selected_C_is_table_argmin` (`ADMISSION.json:128`; dpc `ADMISSION.json:113-140`).
- `osf.data.ALLOW["heads"]` is (OSF_DEFENSE_FIT, HEAD_VALIDATION) (`osf/data.py:45-47`).

**U: no data-driven checkpoint or early stopping.**
- The warm start is fixed at 20 epochs. The task line is fixed at 40 epochs.
- Epoch 40 is the registered "final utility reference" (`smf/run.py:166-176`). osf admits `tl__s{k}__e40` by rule (`osf/run.py:97-106`).
- Hyperparameters (83-64-64-16 architecture, SGD 0.05, clip 5, batch 256, C grid) are registered constants (`jcv/train.py:40-46`, `osf/train.py:50-54`). No search over them exists in code.
- U is the task-only reference. It was not nominated by any inner audit.

**RAW-J β0.3: hyperparameter selection is attacker-mediated (disclose).**
- In osf, β = 0.3 became C* (`SELECTION.json:21-35`). The rule is the lowest mean inner pair SEX AUC, gated by utility on INNER_SELECTION (`osf/select.py:1-20`).
- The AUCs come from attackers **fitted on AUDIT_FIT** (SEX labels) and scored and selected on INNER_SELECTION (`osf/inner.py:1-5`, `osf/audit.py:8`).
- smf's C* was also RAW-J (osf `EXPOSURE_LEDGER.md:122-127`).
- dpc then fixed "U and RAW-J β 0.3" as teachers (`dpc PROTOCOL.md:29`).
- So AUDIT_FIT influenced *which* RAW-J strength was carried forward, only through attacker fitting. It never entered RAW-J's parameters or heads.

**Downstream codes (D0 bank).** The rate i8o64 was chosen by INNER_SELECTION utility, with no attackers and no SEX (`qpc/run.py:264-310`). Nominees and λ were chosen by inner audits (attackers fitted on AUDIT_FIT, selected on INNER_SELECTION).

## (d) Preprocessing fitting

**Fitted statistics.** Only OSF_DEFENSE_FIT is used for fitted statistics. Every other preprocessing step is a fixed rule.

**Numeric standardisation (5 columns).**
- The admitted file stores the jcv normalisation, which jcv fitted on old `defense_train` = `adult.data` train-split rows (`jcv/data.py:110-121`; `DATA_ADMISSION.json:66`). That set contains no `adult.test` row.
- osf inverts it to exact raw integers. It asserts an inversion error below 0.05, then rounds.
- osf then re-standardises with population mean and sd computed on `role == "OSF_DEFENSE_FIT"` only (`osf/data.py:20-22,111`; `smf/data.py:87-103`, the mask at line 100).
- Receipts: `ROLE_MANIFEST.json:567` ("…computed on OSF_DEFENSE_FIT rows only; the transform is applied to all kept rows") and lines 640-700 (recovered exactly on all roles; equal to the smf refit).

**Categorical vocabularies.**
- These are fixed constants, `pcrl.data.adult.AdultDataset.CATEGORY_VALUES` (`pcrl/data/adult.py:129-157`). The education and marital lists are keys of constant dicts.
- One-hot encoding uses `pd.Categorical(..., categories=fixed)` (`jcv/data.py:122-125`). No fitted vocabulary (`ROLE_MANIFEST.json:564`).

**Column selection (83).**
- This is a registered semantic rule:
  - remove sex, race, income, occupation, fnlwgt and ids;
  - keep 5 numeric and 5 categorical columns (`jcv/data.py:7-16,39-43`).
- It is asserted at load (`osf/data.py:121-123`). The feature-names sha is 7101f2fb….

**Head-internal scaler.** Covered under (b): OSF_DEFENSE_FIT.

**Critic ZCA (RAW-J only).** Fitted on CRITIC_FIT, a subset of OSF_DEFENSE_FIT.

**Label-free or label-touching operations over ALL rows, AUDIT_FIT included (reported explicitly).** None of these is a fitted statistic.
1. **Missing-value exclusion.** The pinned loader applies `dropna` to the raw files over every column, including the label columns and occupation (`pcrl/data/adult.py:296-311`). The train split is the first 80 % of a seed-42 permutation of `adult.data` (`:196-206`).
2. **Record keys and groups.** `record_key` and `canon_key` are sha256 prefixes of the full raw row, including labels and fnlwgt. `unit` is the inverse index of `np.unique(canon_key)` over all 39,205 rows (`a2_build_rows.py:11-17,51,101-106`).
3. **Roles by hash.**
   - Test-split roles: hash of `record_key` (`admission_common.py:83-91`).
   - Cert and head carve-outs: hash of `canon_key` (`oar/study.py:68-73`).
   - rgj and smf splits: hash of the integer `unit` (`rgj/data.py:55-73`, `smf/data.py:48-67`).
4. **Exposure exclusions.** `excluded_exposure` and `excluded_dup` come from `canon_key` equality across splits (`oar/study.py:64-66`). osf drops pool rows whose groups overlap a fitting role (`osf/data.py:69-80`).
5. **jcv build.** The race code list is `sorted(raw["race"].unique())` over all rows; this is a label vocabulary, not an input. The documentation-only "shortcut audit" statistics read SEX over all rows; the registered column rule was not changed by them (`jcv/data.py:95-97,131-139,158-162`).
6. **All-row checks and transforms.**
   - The `refit_numeric` inversion-error assertion runs over all kept rows (`smf/data.py:96-99`).
   - The one-hot exactness and raw-recovery checks run over all roles (`ROLE_MANIFEST.json:640-700`).
   - The numeric transform and one-hot encoding are applied to all rows.
7. **Inference only.** Over all rows: teacher forward passes, fine-cell assignment, LEACE maps and FARE encodes.

## (e) Fine partitions and per-token teacher means

**What lra admits.** lra admits `fine__s{k}` from qpc Stage B, through cbp. These are label-blind KL/Bregman k-means fine partitions per teacher-predicted class: income ≤ 32 cells, occupation ≤ 128 (`lra/admit.py:9`).

**Fitting rows.** They are fitted on the U teacher's p and argmax decision of **OSF_DEFENSE_FIT** rows only (`qpc/run.py:173-174,316-326` with `tr = D["idx"]["DEFENSE_FIT"]`; `qpc/partition.py:59-75`):
- `fit_fine_pair(T["p1"][tr], …)`;
- the assertion `bincount(assign[tr]) == f.n` (lines 67-70).
- kmeans: "No task label or SEX enters any step" (`qpc/kmeans.py:1-55`). The starts use fixed seeds 20261006 and 20261007.

**dpc partitions.** These are also fitted on DEFENSE_FIT (`dpc/run.py:184-197`; `dpc/partition.py:1-33`). They were used only for qpc's A1 reproduction and are not admitted by lra.

**Independent qpc verifier.** It refit every `fine__s{k}` on `D.fit_idx` (OSF_DEFENSE_FIT) and got bitwise partitions, fingerprints, assignments and `deployment_on_fit_reproduces_stats` (`replay_qpc.py:2874-2912`; qpc `INDEPENDENT_VERIFICATION.json` `stage_b_fine_partitions`).

**token_S / token_n.**
- These are sequential sums of member fine-cell statistics `fine.S` and `fine.n` (`dpc/release.py:27-52`). That makes them sums of OSF_DEFENSE_FIT teacher probabilities and counts.
- Policies:
  - DIRECT-TASK i8o64: label-free on `tr` (`qpc/run.py:246`, `qpc/stagea.py:76-88`).
  - FINE-TASK, LOCAL, SEQ and JOINT: `tr` plus SEX on OSF_DEFENSE_FIT (`qpc/run.py:343-366`, `qpc/compress.py:751-773`).
  - The cbp intermediate λ: `tr`, S_fit (`cbp/run.py:204-205,230-275`).
- lra D1 decoders re-check `token_n` and `token_S` against row-level sums on the fitting rows and add OSF_DEFENSE_FIT true task labels (`lra/decoder.py:625-650`; lra `EXPOSURE_LEDGER.md:15,28-38,64`).

## (f) RAW-J β0.3 teacher and the references E, F and F0

**RAW-J β0.3.** See (a), (b) and (c):
- parameters, critics and ZCA: OSF_DEFENSE_FIT (critics on its CRITIC_FIT subrole);
- heads: OSF_DEFENSE_FIT, with C chosen on HEAD_VALIDATION;
- β: selected via inner audits with attackers fitted on AUDIT_FIT.

**E (official LEACE, `lc__s{k}__E`).**
- Map: one map per recipient, fitted on the U features of OSF_DEFENSE_FIT with one-hot SEX of OSF_DEFENSE_FIT, using official defaults (`osf/baselines.py:19-27`, `smf/baselines.py:12-20`).
- Receipts:
  - n_fit, fit-row-id hash, H_fit hash and Z_fit hash all equal OSF_DEFENSE_FIT (`ADMISSION.json:1036-1070`);
  - the verifier confirms `map_fit_rows_are_OSF_DEFENSE_FIT` (`INDEPENDENT_VERIFICATION.json:12281`).
- Heads: OSF_DEFENSE_FIT, with C chosen on HEAD_VALIDATION.
- No configuration selection.

**F (official FARE, `fare__s{k}__p{0,1}__c1`).**
- Trees: fitted on OSF_DEFENSE_FIT X, task labels and SEX.
- Receipts: `n_fit`, `fit_rows_sha256` and `fit_labels_sha256` equal OSF_DEFENSE_FIT (`ADMISSION.json:5832-5839`; `ADMISSION.json:44-50` "FARE").
- Heads: OSF_DEFENSE_FIT, with C chosen on HEAD_VALIDATION.
- **The configuration (c1 of the 6-grid) was selected by INNER_SELECTION gates plus the inner local SEX AUC of attackers fitted on AUDIT_FIT** (`osf/baselines.py:42-50`, `smf/baselines.py:36-44`).

**F0 (zero-fairness twin, `…__Z1`).**
- Same fit rows and receipts as F (`ADMISSION.json` `fare_trees.smf__s*__p*__Z1`).
- Its configuration is inherited from F's selection.

## Was AUDIT_FIT used historically for anything other than attacker fitting?

**Teacher lineage.** In the lineage that produced these teachers (smf, osf, dpc, qpc, cbp, lra): **no fitting use.** Specifically:
- RAW-J and osf critics use CRITIC_FIT, a subset of OSF_DEFENSE_FIT (`rgj/train.py:165-169`), never AUDIT_FIT.
- The AUDIT_FIT uses in these studies are:
  - attacker slates;
  - attacker-side preprocessing: Canon centring and SVD whitening on AUDIT_FIT, and the MLP's internal 10 % early-stopping split (`smf/audit.py:31,43-56`);
  - shuffled-SEX null and planted controls (`osf/audit.py:272-273`; `qpc/data.py:86` "controls");
  - lra cell readers and the AUDIT_FIT SEX prior for unseen tokens (`lra/audit.py:106-127`);
  - final attacker refits at assessment and closeout.

**Post-lock assessment uses of AUDIT_FIT labels in osf.** These are not attackers on SEX. They are a frozen-release evaluation, not teacher fits:
- `osf.assess.probe` fits a task probe r_i → task_i on **AUDIT_FIT task labels**, with C chosen on INNER_SELECTION (`osf/assess.py:47-49,90,315-326`);
- linear R² of SEX (AUDIT_FIT → assessment) and race diagnostics (`osf/assess.py:329-334`).

**Earlier lineage (not ancestors of these teachers; jcv and pnx are listed INELIGIBLE at `ADMISSION.json:7611-7630`).**
- PCRL v2: part of the reported test set of every Adult round.
- oar, odx and cap: U2 task-utility LR probes (task labels) and constant predictors fitted on attacker_fit (`oar/study.py:325-345`).
- pnx critic-gap diagnostic: "fresh_att" critics trained on attacker_fit (`pnx/critic_gap.py:15-22`). This is the "critics" entry in osf `EXPOSURE_LEDGER.md:86`.

## Canonical group ID

**Definition.** The canonical group ID is the exact-record group `unit`:
- an `np.int64` per row: the inverse index of `np.unique(canon_key)` over all 39,205 regenerated rows (adult.test rows 0..15059 and train rows 15060+);
- values lie in [0, 39168);
- `canon_key = sha256("|".join(map(str, raw_row)))[:20]`, taken after stripping the trailing "." from adult.test income;
- defined at `a2_build_rows.py:11-17,51,57` (`unit.astype(np.int64)`) and `admission_common.py:74-76`;
- carried unchanged through `adult_labels.npz` → `oar.load_world` → `jcv.data.build` → `adult_jcv.npz["unit"]` (`oar/study.py:74`, `jcv/data.py:127`).

**In D.** `osf.data.load` sets `D["unit"] = z["unit"][keep]`, row-aligned with `D["row_id"]`, `D["role"]` and `D["X"]` (`osf/data.py:105,109`). dpc, qpc and lra return D unchanged.

**Canonical string.** The decimal integer with no padding, as in every predecessor hash rule: `f"{seed}|{salt}|{int(u)}"` (`rgj/data.py:55-57`, `smf/data.py:48-50`; `role_check.py:145-146`; osf `ROLE_MANIFEST.json:53`: "'<seed>|<salt>|<unit as decimal int>'").

**Group-set hash convention.** `sha256(np.unique(u).astype(np.int64).tobytes())` (`dpc/data.py:86-87`, `role_check.py:131-132`). For AUDIT_FIT it is `d8b65f7f8658456f6bf5ca8509f6e168e330c8f469eec565435a85714fb3a25c` (6,061 groups). The AUDIT_FIT row-id sha is `36f0393b…85f`.

**Recipe without reading any label:**

```python
from lra import data as LD
D = LD.load()                                  # sealed; verify=True runs only label-free role/hash checks
assert D["sealed"]
a = D["idx"]["AUDIT_FIT"]                      # 6,065 row positions
u = D["unit"][a]                               # canonical group ids (np.int64)
g = np.unique(u)                               # 6,061 groups; a group is never split
assert hashlib.sha256(np.ascontiguousarray(g.astype(np.int64)).tobytes()).hexdigest() == \
       "d8b65f7f8658456f6bf5ca8509f6e168e330c8f469eec565435a85714fb3a25c"
key = {int(x): hashlib.sha256(f"hcal-v1|heldout|20261011|{int(x)}".encode()).hexdigest() for x in g}
```

**Rules for that recipe:**
- Touch only `D["unit"]`, `D["idx"]`/`D["role"]` and `D["row_id"]`. Never index `D["sex"]`, `D["race"]`, `D["y_income"]`, `D["y_occupation_group"]` or `D["y"]`.
- `osf.data.load` does copy the label arrays into D. For a split stage that must not even load labels, use the label-free path of `provenance/role_check.py`: read only `role`, `unit` and `row_id` from the npz, then `keep = osf.data.assign_roles(role, unit)[0] != ""`. That path is pure in (old role, unit); its record shows "label_arrays_read_by_independent_step: []".
- Note: `unit` is pseudo-random with respect to labels, but formally it derives from hashes of full records that include labels. No label value is read to compute it.

## Verdict

**TEACHER_HELDOUT_PROVENANCE_VERIFIED**

**Scope.** This covers:
- U teacher parameter fitting (warm start, encoder and training heads);
- deployed-head fitting;
- deployed-head selection;
- preprocessing fitting;
- fine partitions and token statistics.

The same holds for RAW-J β0.3, E, F and F0 parameters and heads. The basis has two parts:
1. Every fit uses rows of OSF_DEFENSE_FIT only, and head C is selected on HEAD_VALIDATION. This is established by code, bitwise replays, fit-row hash receipts and independent refits.
2. OSF_DEFENSE_FIT, HEAD_VALIDATION and AUDIT_FIT are row- and group-disjoint, and come from different source files (`adult.data` vs `adult.test`).

**Mandatory disclosures (none is a gap in the four required properties):**
- **D1.** RAW-J's β = 0.3 (osf and smf C*), FARE's configuration c1 (F, and F0 by inheritance) and every downstream code nominee or λ were selected using inner audits whose attackers were fitted on AUDIT_FIT. For RAW-J, F and F0, AUDIT_FIT is therefore not selection-independent. The dependence runs only through attacker fitting. U and E carry no AUDIT_FIT-informed selection.
- **D2.** The deterministic all-row rules listed in (d), items 1-5, hash or compare full records that include labels. These are not fitted statistics.
- **D3.** AUDIT_FIT rows have historical non-attacker exposure outside the teachers: the PCRL v2 test set, oar/odx/cap task probes, pnx diagnostic critics, and the osf.assess task probe and linear diagnostics. The registered hyperparameter constants could have been shaped by that adaptive history. Code cannot exclude this, and the exposure ledgers concede it.
- **D4.** The warm-start bitwise replay is a lead receipt (osf.run refuses on mismatch; RUN_STATUS 3/3). The independent verifier retrained only from the admitted warm state. The warm-start fitting rows rest on code (`rgj.train.TData`) plus that receipt.

## Files cited

| File | git blob (HEAD) | Lines cited |
|---|---|---|
| osf/data.py | a82547fc84966d043ab511879d2bf96a128fd4ee | 3-25, 43-47, 52-97, 100-129, 132-139 |
| osf/train.py | 072b5a11b70102319a5ead781ee8d6277c4e41fb | 1-26, 50-54, 170-366 (230-252) |
| osf/run.py | acccc523420b76709573d3520f0b3803b414aa1c | 97-106, 117-129, 154-194, 305-333, 396 |
| osf/inner.py | 69891a6bf954f243a3c520b7384bec2e2f5c3959 | 1-5, 16-25 |
| osf/select.py | 13d6a92ca69d12aa4c0ea7ee75022b8d3ee83fc0 | 1-20 |
| osf/baselines.py | 4d68846bd341af5b59b094cc135b1463091d0063 | 1-50 |
| osf/assess.py | 1dc80e28bf2a98da10b30fb3a0dcfb475a9c612e | 47-49, 90, 315-334 |
| smf/data.py | 9c02aa590056e80ba666faa235af413b7906e33b | 3-18, 48-67, 87-103 |
| smf/run.py | 8542ae5816125ff3dd3cbee2113f6c1a9f57b0da | 86-97, 107-119, 166-200 |
| smf/baselines.py | c3e57036d8eef5860d9594cceedd38d63662b79c | 1-44 |
| rgj/data.py | 27153d977fb851926f660a6713839ff059432a3f | 6-15, 28-33, 55-73 |
| rgj/train.py | 63b09b34bbaab5ad949e8006601b70c59f81db4a | 154-176, 382, 392 |
| rgj/finalize.py | 575a6657d9da58c6556e407fd20020e4b53cbdf7 | 1-7, 20-40 |
| jcv/data.py | 9b6650f48036cdc70c92738618f31eb742a699b4 | 7-16, 39-43, 67-78, 95-139, 158-162 |
| jcv/train.py | 50d808070e45c2ffd9a6f17afa38902a47e1da1a | 40-46, 235-254 |
| jcv/finalize.py | af6b59c54614b1951c4467e9895c4b8814372642 | 5-6, 24-37 |
| pcrl/data/adult.py | ee8218d3a9720956f85b009b886bdf4f93163bb7 | 129-157, 196-206, 296-311 |
| oar/study.py | 590e8dfa9aa8542568c555ac6d66906bec4c6037 | 57-87, 325-345 |
| results/combined_matched_removal_benchmark_v1/scripts/admission/a2_build_rows.py | 48b930f9b87d5baa47aa6629bf3dc3f75c3a63b3 | 11-17, 46-57, 101-106 |
| results/combined_matched_removal_benchmark_v1/scripts/admission/admission_common.py | 8a5f80b5bf3fa0343d2b72df6c945a6750e18b0c | 43-44, 74-91 |
| results/combined_matched_removal_benchmark_v1/scripts/admission/a5_labels_and_index.py | cbc7069a1baa7d619b761e37071ad4ec15a0d91d | 7-14, 91-101 |
| results/combined_matched_removal_benchmark_v1/notes/admission/defense_route.json | 7ad6230e9f3e1f4c7496566ae793e14095be8ddf | 16 |
| results/combined_matched_removal_benchmark_v1/notes/admission/rows_provenance.json | 90730db06c9e62885ff38e4330dad1f67b125c2a | 22, 29 |
| dpc/data.py | 977a23ea9df3f81af98745e3f320b483a5b52759 | 64-69, 86-87, 176-192 |
| dpc/partition.py | ed430299baa13472c8178d551384df282c3bb151 | 1-33 |
| dpc/release.py | d5c4038fe8411d884cbc7d0a94ef7ba96272df7b | 27-52 |
| dpc/run.py | a2d05487c3429b10755faaa97e139baa6e0ad540 | 184-197 |
| dpc/admit.py | 6998e2010e31be77a1f4f320d49390fd91d38140 | 1-40 |
| qpc/data.py | a17586b4ee32976d4d7bdf2221edfc11532a09fb | 1-27, 79-86, 191-255 |
| qpc/run.py | 311b20f6e71e5a5c5e0b6cacda153815b3ab9111 | 173-174, 218-262, 264-310, 316-366 |
| qpc/partition.py | 61e088915852b3d9a5d946605ab1fac6022058c4 | 1-12, 59-75 |
| qpc/kmeans.py | 14413ab99bfc102897b1f39dd597b2272a1c7a09 | 1-55 |
| qpc/stagea.py | c18ad41fe20c4ec9d511a07c5ac6732c843bedff | 76-88 |
| qpc/compress.py | e4d8e92ece21eb75e9ed44af96fd87a462bcd5c0 | 751-773 |
| qpc/admit.py | 2c701bafaee92fe3eb538c6b11bd8ca478f3d070 | 1-30 |
| cbp/admit.py | 6a4efdc02abf0f1bfc51c55da6f04b6d1dc12203 | 1-25 |
| cbp/run.py | 668bfe08e1e0c17269257c58935e0689756c8fa1 | 204-205, 230-275 |
| cbp/fit.py | b62214dc1fa762c97dec9e4864e9edeb8fead666 | 1-25 |
| lra/data.py | 84b9bea355451563d38c6113fc4ced5aa268c271 | 1-16, 74-93 |
| lra/admit.py | 7bd65b8cf6182854793ca03ce5f1fd18d0a8b1dd | 1-24, 54-68, 146-157 |
| lra/decoder.py | dee37efb611d1a3ebf82eddcf633d593332ff297 | 625-650 |
| pnx/critic_gap.py | 6f1759f538ac2e439c3a9be9dd1c947427dd136b | 15-22 |
| results/pcrl_online_strength_frontier_v1/provenance/role_check.py | f7a8362b2ae36af995ecf845df9dd0c1352e125e | 6-14, 126-132, 145-146 |
| results/pcrl_online_strength_frontier_v1/verification/replay_osf.py | aae06d805aa501bf66f7ce023e83e6d6d2284f2f | 480-491, 610-621, 655, 1429-1450 |
| results/pcrl_confidence_capacity_v1/verification/replay_qpc.py | 1643ba6a41db546296ed0c1fee58770dcc52859b | 2874-2912 |
| results/pcrl_online_strength_frontier_v1/ROLE_MANIFEST.json | 10d80e33a09d9e4ffa619532a576f27fbb6d03e2 | 53, 59-60, 266, 305, 314, 564, 567, 640-700, 3905-3957, 4199-4201 |
| results/pcrl_online_strength_frontier_v1/ADMISSION.json | 889bc2379f966b3abdb63890ca8258bd80ffd893 | 44-50, 52-84, 86-150, 1036-1070, 5832-5839, 7611-7630 |
| results/pcrl_online_strength_frontier_v1/MODEL_MANIFEST.json | e5f7b9b2de926d4d6e88aea927f64d1e8595609b | units[] complete_files_sha256 |
| results/pcrl_online_strength_frontier_v1/SELECTION.json | 4b36f5d66fe7f80bdfb376062b7a1dc428b6a368 | 21-35 |
| results/pcrl_online_strength_frontier_v1/VALIDATION.md | 5db63e1c06a2e268b2c7486c5e19353e459d4dfb | 20, 24, 93-96 |
| results/pcrl_online_strength_frontier_v1/RUN_STATUS.json | 8d0e77c05f582ac77097c081e92428e7d7d8d153 | 30-32 |
| results/pcrl_online_strength_frontier_v1/INDEPENDENT_VERIFICATION.json | b8d959b36b75f62d0ce7b63a8e44289be041f135 | 271, 11250, 11411, 12281 |
| results/pcrl_online_strength_frontier_v1/EXPOSURE_LEDGER.md | 2ca857e34df77568520e933b348621cc4ff8bcbb | 80-89, 122-127 |
| results/pcrl_decision_preserving_compression_v1/ADMISSION.json | f208507975717b7489a7ef1333d5759ec7e7b365 | 58-140, 1160 |
| results/pcrl_decision_preserving_compression_v1/PROTOCOL.md | c954557d2edc6960bf3b9d552b3165670440da70 | 29 |
| results/pcrl_confidence_capacity_v1/ROLE_MANIFEST.json | 36a2a8ca66effaf0c7a0c16e20d37636e6c0edb9 | 247-292 |
| results/pcrl_confidence_capacity_v1/SOURCE_ADMISSION.json | 04f98d4ce93707a135057dc6000f81668ab8842d | teacher_admission_plan |
| results/pcrl_confidence_capacity_v1/INDEPENDENT_VERIFICATION.json | 866ec854db7123468ddf91650320182e60688e5f | checks.stage_b_fine_partitions |
| results/pcrl_confidence_budgeted_privacy_v1/SOURCE_ADMISSION.json | 12039ae90049eb1df678cc67042e8370557e745b | admitted_teacher_artifacts |
| results/pcrl_adult_learned_decoder_release_v1/SOURCE_ADMISSION.json | c65c940fe94af93c29de37897472d2d31834b46f | 13-98 (parity.teachers), 99+ (admitted_teacher_dirs) |
| results/pcrl_adult_learned_decoder_release_v1/SOURCE_INDEX.json | a5d5e17cc24dbb00d84a40ccc697e01566c38a2e | source pins |
| results/pcrl_adult_learned_decoder_release_v1/ROLE_MANIFEST.json | 9096dfe7521504fba54c70de1950eec4d53ca08c | 1-31 |
| results/pcrl_adult_learned_decoder_release_v1/MODEL_MANIFEST.json | 8e09ed9503cdd9a27091757dfd256d4255ca62c1 | schema/label |
| results/pcrl_adult_learned_decoder_release_v1/EXPOSURE_LEDGER.md | 885c5a89c2f3b1c7bf8f0841d543ccae3e2e0cec | 15-18, 28-38, 64-72 |
| results/pcrl_adult_learned_decoder_release_v1/INDEPENDENT_VERIFICATION.json | 6b20825b2f1f3c224b9efe4790f1dd09104ca59e | 1242, 1249 |
| results/pcrl_joint_complete_view_method_v1/DATA_ADMISSION.json | c713c54a920b489158a3c94b858db5312fb5fa28 | 66, 89, 204 |

