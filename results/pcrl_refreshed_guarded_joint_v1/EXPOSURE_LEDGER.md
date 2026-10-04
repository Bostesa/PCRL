# Exposure ledger: refreshed guarded joint study

> **Every row used by this study has been used before.** All five roles below are UCI Adult records that earlier
> PCRL studies, and the durable-guarantees repository, have already used to train encoders, fit heads and attackers,
> select models and report results. **DEVELOPMENT_ASSESSMENT is a new development partition. It is not fresh
> confirmation**, and its intervals do not undo this history of reuse.

Owner: data and provenance role. Inspected read-only on 2026-10-04 at base `b628ff5b920316517c24b473c33d0442a341a2e3`.
Counts and hashes are in `ROLE_MANIFEST.json`. The partition was recomputed independently by
`provenance/role_check.py` and matches `rgj.data.load()` exactly. This file publishes no label statistic of any role.

## 1. Where the rows come from

The admitted input is `adult_jcv.npz` (sha256 `e0d9e54a…5f12`), built by `jcv/data.py`. It holds 39,205
post-dropna Adult rows from the pinned PCRL loader (`pcrl.data.adult.AdultDataset`, file unchanged since `b96c412`).

| Origin | Row ids | Old roles (oar-roles-v1, as admitted by jcv) | New roles |
|---|---|---|---|
| PCRL **train split**: 80% seed-42 permutation of `adult.data`, 24,145 rows | 15060+ | defense_train 19,230; defense_val 4,897; excluded_dup 18 | DEFENSE_FIT; HEAD_VALIDATION + DEVELOPMENT_ASSESSMENT; dropped |
| PCRL **test split**: `adult.test`, 15,060 rows | 0–15059 | attacker_fit 6,065; attacker_val 2,235; assessment 5,243; cert 1,500; excluded_exposure 17 | AUDIT_FIT; INNER_SELECTION; dropped; dropped; dropped |

The PCRL v2 encoders' own 20% validation split of `adult.data` (used for their checkpoint choice) is not in this file.

## 2. Historical uses per new role

Study short names and pins:

| Short name | Branch @ commit | Results path |
|---|---|---|
| PCRL v2 lineage | main and the v2 round branches (2026-03 to 2026-05) | `results/adult*`, rebuttal folders |
| durable-guarantees (dg) | Bostesa/durable-guarantees @ `956f5c883f515646aa457db55ecbd74b913768b2` | `docs/`, `results/` |
| pilot | research/combined-stored-model-pilot-v1 @ `661db8dcb` | `results/combined_stored_model_pilot_v1/` |
| bench | research/combined-matched-removal-benchmark-v1 @ `70f978ffc` | `results/combined_matched_removal_benchmark_v1/` |
| oar | research/combined-output-aware-removal-v1 @ `f7425b15c` | `results/combined_output_aware_removal_v1/` |
| odx | research/combined-output-diagnosis-v1 @ `dc6a2e961` | `results/combined_output_diagnosis_v1/` |
| cap | research/combined-analysis-paper-v1 @ `3b5065805` | `results/combined_analysis_paper_v1/` |
| jcv | research/pcrl-joint-complete-view-method-v1 @ `568f97068` | `results/pcrl_joint_complete_view_method_v1/` |
| pnx | research/pcrl-penalty-no-erasure-v1 @ `b628ff5b9` | `results/pcrl_penalty_no_erasure_v1/` |

### DEFENSE_FIT (old defense_train; 19,230 rows / 19,221 groups)

- **PCRL v2 lineage.** These rows are part of the PCRL Adult training split. That split is the same for every training seed and round (`results/combined_empirical_preparation_v1/DATA_EXPOSURE_LEDGER.csv`), and it trained the Round-4 seeds 0–2 checkpoints used by pilot and bench. Encoder lineage: `results/combined_matched_removal_benchmark_v1/notes/admission/lineage.json`; its check `global_step = 205 × ceil(n_train/256)` matches the full train split. Baselines trained on the PCRL training split are exposed the same way.
- **dg.** Its Adult cells are built from the same PCRL train split (`utils/pcrl_io.py`: `build_adult_train_loader`, `build_train_loader`, `split="train"`). It uses 75/25 attacker re-splits and a 50/50 "fresh partition" of those rows. At least 16 experiment scripts load this split.
- **pilot and bench.** Part of `defense_fit` (24,127 rows): it fitted the official LEACE maps and noise defenses, and ran the native checks.
- **oar, odx and cap.** It was the `head_fit` part of `defense_fit` (release-head fitting). Official FARE trees were fitted on the whole of `defense_fit`.
- **jcv.** Warm starts, all neural arms, critics, LEACE maps and FARE trees (`jcv__s{k}__p{i}__{c,Z}*`). Deployed heads were fitted here.
- **pnx.** PN and LN encoders and critics. The critic-gap `fresh_def` critics were trained here. pnx reused jcv's warm starts, U, E, JP and FARE units through hash-checked aliases.

### HEAD_VALIDATION and DEVELOPMENT_ASSESSMENT (old defense_val = oar head holdout `oar-head-v1`; 4,897 rows)

The new study splits the old defense_val by group into HEAD_VALIDATION (1,500 rows / 1,499 groups) and
DEVELOPMENT_ASSESSMENT (3,397 rows / 3,393 groups). **Both halves have the same history:**

- **PCRL v2 lineage.** Part of the same encoder training split (same for every seed and round), so they are exposed exactly like DEFENSE_FIT.
- **dg.** Part of the train split behind every dg Adult cell (attacker fitting and scoring, cost and floor analyses).
- **pilot and bench.** Part of `defense_fit`. LEACE maps and defenses were fitted on these rows.
- **oar.** The `oar-head-v1` holdout (20% of `defense_fit` by `canon_key` hash). Its labels **selected the C of every release head** (LR grid, U2 budget, holdout log loss): `oar/study.py` head selection; `PROTOCOL.md`, view-3 head. FARE trees were also fitted on these rows, because they were fitted on all of `defense_fit`.
- **odx.** Reused the oar heads (HEAD__A, selected on this holdout).
- **cap.** Useful-head release heads were fitted on `defense_fit` minus this holdout and **selected on its log loss** (`PROTOCOL_USEFUL_HEADS.md`, "Fit role").
- **jcv.** The defense_val labels **chose head C for every released unit** (U, E, L, J, JP, S12, S21 and every FARE F/F0 head): `jcv/finalize.py:fit_head`, whose tables are recorded as `defense_val_log_loss` in each unit's `record.json`. Every inner utility gate and every nomination was computed with those heads. The warm starts did *not* use defense_val (fixed 20-epoch schedule; see `SOURCE_INDEX.json`, `reuse_audit`).
- **pnx.** The same defense_val head-C selection for PN and LN releases. The reused jcv units carry their defense_val-selected heads.
- **Design influence.** The predecessor outcomes motivated this study's method, and those outcomes were computed through heads whose C was chosen on these labels: PN's income-recipient increase, seed 0's NO_FEASIBLE_NOMINEE, and the critic gap.

### AUDIT_FIT (old attacker_fit; 6,065 rows / 6,061 groups)

- **PCRL v2 lineage.** Part of the official `adult.test` split, the fixed test set of every Adult round (R1–R7, held-out seed 3, baselines). Results were reported on it and methods were iterated against it.
- **pilot and bench.** Part of `pilot-roles-v1` attacker_fit (7,571 rows). Every attacker slate was fitted on it.
- **oar, odx and cap.** attacker_fit after the exposure and cert carve-outs. Attackers, U2 probes and constant predictors were fitted on it.
- **jcv and pnx.** Inner-selection attackers and final audit attackers were fitted on it. The pnx `fresh_att` critics were trained here.

### INNER_SELECTION (old attacker_val; 2,235 rows / 2,234 groups)

- **PCRL v2 lineage.** Part of `adult.test`, as above.
- **pilot and bench.** `pilot-roles-v1` attacker_val (2,239 rows). It was used for attacker selection (NL = GBT vs MLP by log loss) and σ\* selection (`oar/study.py` SIGMA_STAR "selected on attacker_val only").
- **oar, odx and cap.** Attacker selection by log loss, frozen-head usefulness gates (odx), and controls (cap).
- **jcv.** Inner utility gates, inner recovery and **nomination of every candidate**. It chose the FARE configuration (config 1, whose zero-fairness twin `Z1` is the only F0 tree that exists) and selected the final attackers.
- **pnx.** Inner gates and the nominee or NO_FEASIBLE_NOMINEE status per seed, final attacker selection, and the critic-gap scoring rows (same attacker_val rows for online and fresh critics).

### Dropped pools (no new use)

| Pool | Rows | Historical use | Treatment here |
|---|---|---|---|
| assessment | 5,243 | Outer scoring in pilot, bench, oar, odx, cap, jcv and pnx. It is part of `adult.test`. | Dropped at load; labels not read; no new predictions |
| cert | 1,500 | FARE certificates (oar; jcv amendment A3, `FARE_CERTIFICATES_A3.json`) | Dropped |
| excluded_exposure | 17 | Test records identical to a training record | Dropped |
| excluded_dup | 18 | Train rows duplicating those test records (share their groups) | Dropped |

The jcv FARE cell files (`fare_cache/*/cells.npy`, unit `release.npz`) hold historical encodings of all 39,205
rows, dropped pools included. Any reuse here must subset them to the five new roles.

## 3. Why DEVELOPMENT_ASSESSMENT is development, not confirmation

1. **Its labels already selected models.** Old defense_val chose head C in oar, odx, cap, jcv and pnx. DEVELOPMENT_ASSESSMENT is 70% of that selection set, so the predecessor verdicts already depended on these labels.
2. **Its rows trained the historical encoders.** It is part of the PCRL v2 encoder training split and of every dg Adult cell.
3. **The method was designed after seeing outcomes that went through these labels** (see "Design influence" above).
4. **Same population and near-duplicate inputs.** Groups are exact full-record duplicates. 1,385 of the 3,397 DEVELOPMENT_ASSESSMENT rows (41%) have an input vector (83 permitted columns) that also occurs in DEFENSE_FIT. AUDIT_FIT has 40% and INNER_SELECTION 38%, so this is a property of Adult's mostly categorical inputs, not a defect of the split. jcv amendment A3 found the same thing for cert rows ("476 certificate rows are identical to fit rows").
5. **What is new is only procedural.** The split is fixed by a label-blind group hash (seed 20261004) before any score. These rows are unused by *this* study until `EVALUATION_LOCK.json` is pushed. Its intervals describe this fitted comparison on reused rows. They are not population inference on unseen people and not confirmation. A confirmation population needs its own audit (`CONFIRMATION_PLAN.md`).

## 4. Permitted inputs and the proxies kept

**83 permitted columns** (verified by `role_check.py`; same contract as jcv `DATA_ADMISSION.json`):
- Numeric, standardised on DEFENSE_FIT only (mean 0 and sd 1 on DEFENSE_FIT verified): age, education-num, capital-gain, capital-loss, hours-per-week.
- One-hot with the loader's fixed category sets: workclass 8, education 16, marital-status 7, relationship 6, native-country 41.

**Excluded from encoder inputs:**

| Column | Reason |
|---|---|
| sex | Primary protected attribute; forbidden to both recipients |
| race | Secondary stress audit |
| income | Task-1 label |
| occupation | Source of the task-2 label |
| fnlwgt | Census record weight |
| row ids, record keys, group ids | Never inputs |

No column name in the input file matches these.

**Kept proxies (disclosed, not repaired).**
- **relationship = Husband / Wife.** The jcv shortcut audit (`jcv/data.py`; `results/pcrl_joint_complete_view_method_v1/DATA_ADMISSION.json`, `shortcut_audit`) finds that these two values together cover about 46% of all 39,205 rows. Husband is 99.99% male and Wife 99.9% female, so they determine SEX for those rows. The other four relationship values do not.
- **marital-status.** Correlated with sex and with relationship.
- **education and education-num.** Two encodings of one fact; kept.
- **workclass and education.** Ordinary correlated predictors of occupation_group. No column recodes it (largest per-workclass class share 0.54).

**Limitation.** Protection is studied with these proxies left in place, as the prompt requires. Any SEX recovery
measured here includes signal carried directly by permitted inputs: relationship alone nearly identifies SEX for
almost half of the people. A defense must remove that signal to look protected, and the useful tasks also depend on
these columns. Results here describe protection against attackers reading the two releases (and their coalition)
under this input contract. They do not describe protection when proxies are removed. They are also not a guarantee
for other attributes, other purposes, population privacy, DP or MI.
