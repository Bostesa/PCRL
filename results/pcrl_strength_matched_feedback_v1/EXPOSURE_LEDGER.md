# Exposure ledger: strength-matched feedback study

> **Every row used by this study has been used before.** All five roles are UCI Adult records that earlier PCRL
> studies, and the durable-guarantees repository, have used to train encoders, fit heads, LEACE maps, FARE trees,
> critics and attackers, select models and report results.
>
> **NEW_DEVELOPMENT_ASSESSMENT is not untouched.** Its 3,796 rows are old `defense_train` rows. They trained every
> predecessor encoder, warm start, LEACE map and FARE tree, and the PCRL v2 encoders. The partition only withholds
> them from *this* procedure until `EVALUATION_LOCK.json` is pushed. **Its results are development evidence, never
> fresh confirmation.**

Owner: data and provenance role. Inspected read-only on 2026-10-05 (UTC) at the source pin
`ccdcc5a372d538cd12332e496e08f7753f7c9922`. Counts and hashes are in `ROLE_MANIFEST.json`. The partition and the
numeric refit were recomputed independently by `provenance/role_check.py`. They match `smf.data.load()` exactly
(verdict `MATCH`). This file publishes no label statistic of any role.

## 1. Where the rows come from

The admitted input is `adult_jcv.npz` (sha256 `e0d9e54a…5f12`), built by `jcv/data.py` from the pinned PCRL loader
(`pcrl.data.adult.AdultDataset`, blob unchanged since `b96c412`). It has 39,205 post-dropna rows.

| Origin | Row ids | Old role (oar-roles-v1) | Refreshed study (rgj) | This study (smf) |
|---|---|---|---|---|
| PCRL **train split** (80% seed-42 permutation of `adult.data`) | 15060+ | defense_train 19,230 | DEFENSE_FIT | **NEW_DEFENSE_FIT 15,434** + **NEW_DEVELOPMENT_ASSESSMENT 3,796** (group hash, seed 20261005, salt `assess`, 20%) |
| same | 15060+ | defense_val 4,897 | HEAD_VALIDATION 1,500 / DEVELOPMENT_ASSESSMENT 3,397 | HEAD_VALIDATION 1,500 (kept) / dropped |
| same | 15060+ | excluded_dup 18 | dropped | dropped |
| PCRL **test split** (`adult.test`) | 0–15059 | attacker_fit 6,065 | AUDIT_FIT | AUDIT_FIT (kept) |
| same | 0–15059 | attacker_val 2,235 | INNER_SELECTION | INNER_SELECTION (kept) |
| same | 0–15059 | assessment 5,243; cert 1,500; excluded_exposure 17 | dropped | dropped |

NEW_DEFENSE_FIT is split by group (seed 20261005, salt `critic`) into CRITIC_FIT 10,764, CRITIC_VAL 2,343 and
CONTROLLER_CALIB 2,327 rows. This is a **new draw**; it is not the refreshed study's critic split. Encoders and
training heads train on all NEW_DEFENSE_FIT rows, including the CRITIC_VAL and CONTROLLER_CALIB labels.

The PCRL v2 encoders' own 20% validation split of `adult.data` is not in this file.

## 2. Historical uses per role

Study short names and pins:

| Short name | Branch @ commit | Results path |
|---|---|---|
| PCRL v2 lineage | main and the v2 round branches (2026-03 to 2026-05) | `results/adult*`, rebuttal folders |
| dg | Bostesa/durable-guarantees @ `956f5c883f515646aa457db55ecbd74b913768b2` | `docs/`, `results/` |
| pilot | research/combined-stored-model-pilot-v1 @ `661db8dcb` | `results/combined_stored_model_pilot_v1/` |
| bench | research/combined-matched-removal-benchmark-v1 @ `70f978ffc` | `results/combined_matched_removal_benchmark_v1/` |
| oar | research/combined-output-aware-removal-v1 @ `f7425b15c` | `results/combined_output_aware_removal_v1/` |
| odx | research/combined-output-diagnosis-v1 @ `dc6a2e961` | `results/combined_output_diagnosis_v1/` |
| cap | research/combined-analysis-paper-v1 @ `3b5065805` | `results/combined_analysis_paper_v1/` |
| jcv | research/pcrl-joint-complete-view-method-v1 @ `568f97068` | `results/pcrl_joint_complete_view_method_v1/` |
| pnx | research/pcrl-penalty-no-erasure-v1 @ `b628ff5b9` | `results/pcrl_penalty_no_erasure_v1/` |
| rgj | research/pcrl-refreshed-guarded-joint-v1 @ `ccdcc5a37` | `results/pcrl_refreshed_guarded_joint_v1/` |

### NEW_DEFENSE_FIT and NEW_DEVELOPMENT_ASSESSMENT (both are old defense_train = rgj DEFENSE_FIT)

**Both parts have the same history.** The 20% hash split is new; the rows are not.

- **PCRL v2 lineage.** Part of the PCRL Adult training split. That split is the same for every training seed and round (`results/combined_empirical_preparation_v1/DATA_EXPOSURE_LEDGER.csv`). It trained the Round-4 checkpoints used by pilot and bench (lineage: `results/combined_matched_removal_benchmark_v1/notes/admission/lineage.json`). Baselines trained on that split are exposed the same way.
- **dg.** Its Adult cells are built from the same PCRL train split (`utils/pcrl_io.py`: `build_adult_train_loader`, `build_train_loader`, `split="train"`; blob `262326bb…`). Re-checked at the pin: 16 experiment scripts call the train-split loaders, and no script references the Adult test or validation split. dg also used 75/25 attacker re-splits and a 50/50 "fresh partition" (seed 20260724) of these rows.
- **pilot and bench.** Part of `defense_fit` (24,127 rows). It fitted the official LEACE maps and noise defenses.
- **oar, odx and cap.** It was the `head_fit` part of `defense_fit`. Official FARE trees were fitted on all of `defense_fit`.
- **jcv.** It trained the warm starts `warm__s{0,1,2}`, every neural arm (U, E, L, J, JP, S12, S21) and their critics. It also fitted the LEACE maps and the FARE trees. Re-checked here, label-free: all **42 of 42** jcv FARE trees have `n_fit = 19,230`, and their fit-row fingerprint equals the old defense_train inputs. Deployed heads were fitted on these rows.
- **pnx.** It trained the PN and LN encoders and their critics, including the critic-gap `fresh_def` critics. pnx reused jcv's warm starts and its U, E, JP and FARE units through hash-checked aliases.
- **rgj.** It trained every encoder: the task line (U-B, U), L-O, L-R, J-G, L-G, J-R and J-O. All of them started from the jcv warm starts and trained on all 19,230 rows. It also fitted:
  - the training heads;
  - the official LEACE maps `lc__s{k}__E` on U's features;
  - the deployed heads' scalers and LR weights.

  The jcv FARE trees were reused, and their heads were refit. Critics were trained or used on these rows according to the rgj critic subrole of each NEW_DEVELOPMENT_ASSESSMENT row:

  | rgj subrole | What happened there | NEW_DEVELOPMENT_ASSESSMENT rows |
  |---|---|---|
  | CRITIC_FIT | critic training | 2,679 |
  | CRITIC_VAL | refit early stopping and attempt choice | 566 |
  | CALIB | local budgets c_i, multiplier updates, critic-tracking scores | 551 |
- **Input preprocessing.** The admitted numeric standardisation was fitted on all 19,230 rows (`DATA_ADMISSION.json` `numeric_norm`), including today's assessment rows. smf refits it on NEW_DEFENSE_FIT only. `role_check.py` regenerates the raw integers from the pinned loader and confirms the refit bitwise (§5).
- **Published aggregates.** Earlier public files hold label aggregates over sets that contain these rows:
  - the rgj `EVALUATION_LOCK.json` `sex_prior_defense_fit`, at full precision over all 19,230 rows;
  - the jcv/rgj fitting priors;
  - the rgj critic-tracking CE on CALIB.

  See §7 for the consequence.
- **Design influence.** The motivation for this study was computed with models trained on these rows. That covers the J-O/J-R gradient-ratio confound, the inactive multipliers and the critic gap.

### HEAD_VALIDATION (30% of old defense_val; 1,500 rows / 1,499 groups; unchanged from rgj)

- **PCRL v2 lineage, dg, pilot and bench.** It is part of the same PCRL training split, so it has the same exposure as above. LEACE maps were fitted on it in pilot and bench.
- **oar.** It is part of the `oar-head-v1` holdout, whose labels **selected the C of every release head**. FARE trees were also fitted on these rows.
- **odx.** odx reused the oar heads that were selected on this holdout.
- **cap.** The useful-head release heads were **selected on its log loss**.
- **jcv and pnx.** The old defense_val labels **chose head C for every released unit**, through `jcv/finalize.py:fit_head` (`defense_val_log_loss`).
- **rgj.** It is exactly rgj HEAD_VALIDATION. It **chose deployed head C** for every rgj unit, the controls included.
- **This study** uses it again for head C selection only.

### AUDIT_FIT (old attacker_fit; 6,065 rows / 6,061 groups; unchanged)

- **PCRL v2 lineage.** It is part of `adult.test`, the fixed test set of every Adult round. Results were reported on it, and methods were iterated against it.
- **pilot and bench.** It is part of `pilot-roles-v1` attacker_fit. Every attacker slate was fitted on it.
- **oar, odx and cap.** Attackers, U2 probes and constant predictors were fitted on it.
- **jcv, pnx and rgj.** **Inner-selection attackers and final audit attackers were fitted on it**, as were the pnx `fresh_att` critics.
- **dg.** No use was found at the pin.

### INNER_SELECTION (old attacker_val; 2,235 rows / 2,234 groups; unchanged)

- **PCRL v2 lineage.** It is part of `adult.test`.
- **pilot, bench, oar, odx and cap.** It was used for attacker selection and σ\* selection, for the frozen-head usefulness gates (odx) and for the controls (cap).
- **jcv.** It set the inner utility gates and inner recovery, **nominated every candidate**, chose the FARE configuration and selected the final attackers.
- **pnx.** It set the inner gates, determined the nominee or NO_FEASIBLE_NOMINEE status per seed and selected the final attackers. It also supplied the critic-gap scoring rows.
- **rgj.** It applied the Stage B and Stage C utility gates and **selected the controls and nominees**: no valid J-G nominee on any seed. It also chose the AUC and CE attackers and their orientation, and supplied the critic-tracking scores.

### Dropped pools (no new use)

| Pool | Rows | Historical use | Treatment here |
|---|---|---|---|
| rgj DEVELOPMENT_ASSESSMENT | 3,397 | Scored once by rgj after its EVALUATION_LOCK. It is 70% of old defense_val, the head-C selection set of oar to pnx. | Dropped at load. No new prediction is made, and it is not rescored. |
| old assessment | 5,243 | Outer scoring in pilot, bench, oar, odx, cap, jcv and pnx. Part of `adult.test`. | Dropped |
| old cert | 1,500 | FARE certificates (oar; jcv amendment A3) | Dropped |
| excluded_exposure | 17 | Test records identical to a training record | Dropped |
| excluded_dup | 18 | Train rows duplicating those test records | Dropped |

`role_check.py` asserts three things about these pools:
- no row of a dropped pool receives a new role;
- no new-role group occurs in any dropped pool;
- the dropped rows are exactly these five pools (10,175 rows).

The jcv FARE cell files hold historical encodings of all 39,205 rows, so they must never be read here.

## 3. Why NEW_DEVELOPMENT_ASSESSMENT is development, not confirmation

1. **Its rows trained every predecessor model.** This covers the PCRL v2 encoders, the dg Adult cells, and the warm starts, encoders, critics, LEACE maps and FARE trees of jcv, pnx and rgj (§2).
2. **The method was designed after seeing outcomes produced by models trained on these rows.**
3. **Same population and shared inputs.** 1,419 of its 3,796 rows (37%) have an input vector (83 permitted columns) that also occurs in NEW_DEFENSE_FIT. This is a property of Adult's mostly categorical inputs, not a defect of the split. The rates for the other roles are HEAD_VALIDATION 563/1,500, AUDIT_FIT 2,255/6,065 and INNER_SELECTION 800/2,235.
4. **What is new is procedural only.**
   - A label-blind group hash (seed 20261005) fixes the split before any score.
   - The labels are masked to −1 at load until `EVALUATION_LOCK.json` is pushed (verified).
   - Its intervals describe this fitted comparison on reused rows. They are conditional on the fitted models and the history of reuse.
   - A complete pass would be DEVELOPMENT_ADVANTAGE_ESTABLISHED, never confirmation (`CONFIRMATION_PLAN.md`).

## 4. Why old checkpoints cannot be matched controls

Every stored predecessor artifact was fitted with NEW_DEVELOPMENT_ASSESSMENT rows *and their labels*. This includes:
- jcv: 3 warm starts and 102 `nn` units;
- pnx: 18 PN units and 18 critic units;
- rgj: 228 checkpoints, 57 runs, 24 task-line, 3 continuation, 3 calibration and 3 LEACE units;
- every LEACE map;
- all 42 FARE trees;
- every training head.

Reusing them as matched scientific controls fails for four reasons:

1. **In-sample assessment.** For these models, the new assessment rows are training rows. Their task accuracy and SEX recovery on those rows would be training-set numbers. A newly trained candidate's numbers would be held-out numbers.
2. **Unmatched fitting data.** They were fitted on 19,230 rows. The new arms fit on 15,434 rows. Any difference mixes method with data size and with which rows were seen.
3. **Input contract shift.** The predecessors read numeric columns standardised on all 19,230 rows. smf re-standardises on NEW_DEFENSE_FIT, which changes the mean and sd of all 5 numeric columns (`ROLE_MANIFEST.json` `numeric_refit`). An old encoder fed smf inputs is not the model that was trained.
4. **Selection history.** Their heads were C-selected on HEAD_VALIDATION or old defense_val. Their nominations used INNER_SELECTION, including the decisions that motivated this study.

Old checkpoints may supply **engineering or synthetic reference behaviour only**, such as timing, numerical parity or shape tests. All scientific arms are retrained from scratch on NEW_DEFENSE_FIT:
- warm starts;
- U;
- the raw-penalty J-O and L-O arms;
- the normalised arms;
- the Phase B twins.

Official LEACE maps and FARE / zero-fairness trees are refitted on NEW_DEFENSE_FIT (prompt §5, §11).

## 5. The 83-column input contract

The contract was verified by `role_check.py` and is identical to the jcv `DATA_ADMISSION.json` (feature-name list sha256 is in `ROLE_MANIFEST.json`).

- **Numeric (5):** age, education-num, capital-gain, capital-loss, hours-per-week.
  - Population mean and sd are fitted on NEW_DEFENSE_FIT only.
  - The raw integers were regenerated from the pinned loader and tied to the admitted rows by bitwise equality of all 83 admitted columns.
  - Rounding the admitted inversion recovers them exactly. The maximum inversion error is 0.0023, for capital-gain.
  - smf's refitted columns equal the independent recomputation bitwise.
- **One-hot (78):** workclass 8, education 16, marital-status 7, relationship 6, native-country 41.
  - They use the loader's fixed category sets, with no fitted vocabulary.
  - Every block is exact one-hot on every row; no unseen category occurs.

**Excluded from encoder inputs:**

| Column | Reason |
|---|---|
| sex | Primary protected attribute; forbidden to both recipients |
| race | Secondary audit |
| income | Task-1 label |
| occupation | Source of the six-class task-2 label |
| fnlwgt | Census record weight |
| row ids, record keys, group ids | Never inputs |

No input column name matches these.

## 6. Husband/Wife proxy limitation (kept, as required)

- **relationship = Husband / Wife.** The jcv shortcut audit (`DATA_ADMISSION.json` `shortcut_audit`) finds that these two values cover about 46% of all 39,205 rows. Husband is 99.99% male and Wife 99.9% female, so they effectively determine SEX for those rows. The other four relationship values do not.
- **marital-status** is correlated with sex and with relationship. **education** and **education-num** are two encodings of one fact. **workclass** and **education** are ordinary correlated predictors of occupation_group; no column recodes it.

**Limitation.** Protection is studied with these proxies left in place (prompt §5: do not remove Husband/Wife or another hard proxy to get an easier result).
- Any SEX recovery measured here includes signal carried directly by permitted inputs. Relationship alone nearly identifies SEX for almost half of the people.
- To look protected, a defense must remove that signal from both releases and their coalition, while the useful tasks depend on the same columns.
- The results do not describe protection after proxy removal.
- They are not a guarantee for other attributes or purposes, population privacy, DP or MI.

## 7. Publication cautions for this study

- **Do not publish a NEW_DEFENSE_FIT label prior at full precision without considering the differencing risk.** The rgj `EVALUATION_LOCK.json` already published `sex_prior_defense_fit` over all 19,230 old defense_train rows at full precision. A NEW_DEFENSE_FIT SEX prior plus the public row counts would reveal the sealed NEW_DEVELOPMENT_ASSESSMENT SEX share by subtraction. The same holds for task priors if an earlier file published them over all 19,230 rows.
  - The assessment's labels were seen historically, so this is a rule issue, not new exposure.
  - The safe options are to publish only the prior's sha256 (keeping the value in the private lock), or to disclose the derivability explicitly.
- Public files use `<PRIVATE_CACHE>`, `<WORKTREE>` and `<DRIVE_ROOT>` placeholders. They never contain the drive's folder name, its volume label or a home-folder name.
  - The rgj branch still carries the drive folder name in four earlier commits.
  - That history is not rewritten here, and it is not claimed to be removed.
