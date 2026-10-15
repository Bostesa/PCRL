_Orchestrator note (2026-10-15): statuses originally drafted as UNUSED_VERIFIED are relabelled NO_USE_FOUND_IN_SEARCHED_RECORDS, because non-use outside the searched PCRL/durable-guarantees records (S3 listings, deleted EC2 disks, other machines) is UNKNOWN. No prospective label was opened in this audit._

# Population and provenance audit: frozen L_TREE confirmation feasibility

**Role:** POPULATION/PROVENANCE, campaign `pcrl_frozen_tree_control_completion_v1` (branch `research/pcrl-frozen-tree-control-completion-v1`, seal commit `3c6d4aadd`).
**Date:** 2026-10-15 (audit run about 19:15–20:00Z).
**Mode:** read-only.
- No task-label or sensitive-label array was opened for any population.
- No class frequency was computed, nothing was trained or scored, and no population was ranked by performance.
- No paid cloud access was used.
- Every count below comes from a committed manifest or receipt, an ID or feature file, or arithmetic on those. Each section says which.
- **Disclosure:** some committed receipts I read embed class-support counts for already-exposed roles: L_TREE `ADMISSION.json` facts, oar `ROLES_AND_SUPPORT.json` and `row_index_verification.json`. None is reproduced here and none was used.

Companion table: `POPULATION_COMPATIBILITY_AND_USAGE.csv`.

---

## Bottom line

**(a) Frozen-release confirmation of the Adult L_TREE packages: no compatible unexposed population exists.**
- **Adult is closed-world.** All 48,842 UCI records are accounted for below, and none is both raw83-compatible and unused.
- **The 39,170-row raw83 row space is fully exposed.** Every role in it has been used for training, selection, attacker fitting, calibration or reporting.
- **The 9,672 rows outside it:**
  - 6,017 rows are the PCRL `val` split, already spent as the v1/v2 checkpoint-selection set.
  - 3,620 rows were dropped for missing values and cannot be used under raw83.
  - 35 rows are exact duplicates that were deliberately excluded.
- **No Adult rows are formally reserved.**
- **No other dataset uses the raw83 schema.** The frozen packages therefore cannot be confirmed on new people anywhere.

**(b) Recipe replication (a transported benchmark that refits the recipe end to end under a new, registered input contract):**
- **Only ACS PUMS state-years outside the spent California years are candidates.** None has been acquired.
  - **TX 2018 1-Year:** reserved by name. It is the primary confirmation cohort of the shared-context release programme.
  - **NY 2018:** named only as that programme's conditional TX+NY fallback.
  - **Other 2018 states:** no use found.
  - **2019 and 2021–2023:** no use found, but the relationship variable changed (`RELP` became `RELSHIPP`).
- **CA 2016, 2017 and 2018 are all spent.**
- **No candidate's person or household count is documented in any receipt.** The only documented sizes are the TX and NY zip byte sizes from HTTP HEAD requests.
- **None of these runs is confirmation of the frozen Adult release.** Adult's raw83 preprocessing and its six `occupation_group` classes do not transfer to ACS automatically (section 3.5).

---

## 1. Adult usage ledger (48,842 records)

### 1.1 How the 48,842 records map onto roles

**Sources:**
- Raw files `adult.data` (32,561 rows, sha256 `5b002646…`) and `adult.test` (16,281 rows, sha256 `a2a9044b…`). Hashes are from jcv `DATA_ADMISSION.json`. The 32,561 and 16,281 counts are UCI totals; they also appear in `pcrl/data/adult.py` (fallback constants) and `row_index_verification.json`.
- The loader `pcrl/data/adult.py` (blob `ee8218d3`) has not changed since `e3ac31107` (2026-04-25) in either git store.
- The loader permutes `adult.data` with `np.random.seed(42)`. The first 26,048 rows become `train` and the remaining 6,513 become `val`.
- `dropna` runs **after** the split. `adult.test` (header skipped, then `dropna`) becomes `test`.

**Arithmetic check:** 32,561 + 16,281 = 48,842.

| Block | Rows (groups) | Derivation and count source |
|---|---|---|
| **Removed by `dropna`** (missing workclass, occupation or native-country) | **3,620** | Total 48,842 − 45,222 (2,399 in `adult.data` = 1,903 in the train portion + 496 in the val portion; 1,221 in `adult.test`). Arithmetic on the receipt counts 30,162 / 24,145 / 6,017 / 15,060; no raw file was opened. |
| **PCRL `val`** (20% of `adult.data`) | **6,017** | `results/combined_evaluation_preparation_v1/notes/artifacts/row_index_verification.json` (`n_val`, row-index sha `3a516f79…`), which is consistent with 30,162 − 24,145. Not in `adult_jcv.npz`. |
| **PCRL `test`** (`adult.test`, after `dropna`) | 15,060 | jcv `DATA_ADMISSION.json`: row ids 0..15059 by `adult.test` position. |
| ├ attacker_fit, later **AUDIT_FIT** | 6,065 (6,061) | = **ATTACK_FIT_NEW (ATT) 4,064 (4,061)** + **CALIBRATION_HELDOUT 2,001 rows (2,000 groups, 2,000 representatives)**. Sources: hcal `ROLE_MANIFEST.json` containment checks; salt `hcal-v1\|heldout\|20261011\|`. |
| ├ attacker_val, later **INNER_SELECTION** | 2,235 (2,234) | oar, jcv, osf and hcal manifests. |
| ├ assessment, later **ORIG_ASSESSMENT** | 5,243 (5,243) | Part of OSF_DEVELOPMENT_ASSESSMENT. |
| ├ **CERT** (oar-cert-v1 carve-out from attacker_fit) | 1,500 (1,500) | Part of OSF_DEVELOPMENT_ASSESSMENT. |
| └ **excluded_exposure** | 17 (17) | Test records equal to a training record, removed from every scored role (oar `PROTOCOL.md`). |
| **PCRL `train`** (80% of `adult.data`, after `dropna`) | 24,145 | `row_index_verification.json`; equals 39,205 − 15,060 (jcv). |
| ├ defense_train, later rgj DEFENSE_FIT | 19,230 (19,221) | Split by `u = sha256('20261005\|assess\|<unit>')`: |
| │ ├ **OSF_DEFENSE_FIT** (u ≥ 0.20) | 15,434 (15,428) | = **UPDATE 8,180 (8,176)** + **GUARD 2,043 (2,043)** (inherited FIT folds 0+1, salt `SPG1\|UPDATE-GUARD-split\|20261013\|`) + **FIT2 5,211 (5,209)** (inherited FIT fold 2). Source: SPG `ROLE_SPLIT.json`. |
| │ └ **SMF_DEV** (u < 0.20) | 3,796 (3,793) | Part of OSF_DEVELOPMENT_ASSESSMENT. |
| ├ defense_val (oar head holdout) | 4,897 (4,892) | Split by `u('20261004\|dev')`: |
| │ ├ **HEAD_VALIDATION** (u < 0.30) | 1,500 (1,499) | |
| │ └ **RGJ_DEV** (u ≥ 0.30) | 3,397 (3,393) | Part of OSF_DEVELOPMENT_ASSESSMENT. |
| └ **excluded_dup** | 18 (17) | Train rows that duplicate a test-role record. Never used in an October role. |

**Derived totals:**
- **OSF_DEVELOPMENT_ASSESSMENT** (the "historical" or "old" assessment) = ORIG_ASSESSMENT 5,243 + RGJ_DEV 3,397 + SMF_DEV 3,796 + CERT 1,500 = **13,936 (13,929 groups)**.
- **The 39,170 raw83 row space** (osf `ROLE_MANIFEST.json`, partition fingerprint `d6835fdb…`) = OSF_DEFENSE_FIT 15,434 + HEAD_VALIDATION 1,500 + AUDIT_FIT 6,065 + INNER_SELECTION 2,235 + OSF_DEVELOPMENT_ASSESSMENT 13,936.
  - It equals `adult_jcv.npz` (39,205 rows, sha256 `e0d9e54a…`) minus the 35 exclusions.
  - It matches `deploy_input.npz`: I inspected only the npy headers, which show `X` (39,170 × 83, float32) plus `feature_names`, with no label or id member.
- **Reconciliation:** 39,170 + 35 + 6,017 + 3,620 = **48,842**.

**Inside OSF_DEFENSE_FIT,** these sub-roles were also used:
- CALIBRATION_TRAIN_MATCHED: 2,002 rows (2,000 groups).
- Critic sub-roles: CRITIC_FIT 10,764, CRITIC_VAL 2,343 and DIAGNOSTIC_CALIB 2,327.

**Completeness scan of role manifests:**
- I scanned 21,835 role, manifest, admission, split and receipt JSON files under `~/PCRL/.worktrees/*`, including `.private`.
- Every Adult role size found is one of the sizes listed above.
- No role exists on the PCRL `val` split or on the `dropna` rows.

**Count discrepancy (resolved):**
- The 2026-10-01 `DATA_EXPOSURE_LEDGER.csv` rows 17 and 18 give train 24,129 and val 6,033. Rows 1–3 give pre-`dropna` counts (26,048 / 6,513 / 16,281).
- The regenerated index verification of 2026-10-02 (24,145 / 6,017) supersedes them, for two reasons:
  - it matches the jcv arithmetic (39,205 − 15,060);
  - it matches durable-guarantees' 6,037-row assessment vectors, since ceil(0.25 × 24,145) = 6,037.

### 1.2 Usage history by role

| Population | Use (cumulative) | Status |
|---|---|---|
| UPDATE / GUARD / FIT2 (= OSF_DEFENSE_FIT) | <ul><li>PCRL v2 encoder training (`train` split).</li><li>All durable-guarantees Adult experiments.</li><li>oar/jcv/pnx defense_train; rgj/smf DEFENSE_FIT.</li><li>U0 teacher and heads; RAW_PROJ / RAW_J; historical score codes and FARE.</li><li>L_TREE: CatBoost teachers on UPDATE+GUARD; critics on UPDATE SEX; FIT2 for selection and the FIT2 SEX proxy (SELECTION_LOCK `b8d3c4d7e`).</li></ul> | USED |
| HEAD_VALIDATION | <ul><li>PCRL `train` split.</li><li>oar head holdout (defense_val).</li><li>Deployed head C selection (rgj/smf/osf; U0 "HEAD_VALIDATION exposure").</li><li>Never read by hcal; barred or closed in every study since; closed in L_TREE and this campaign.</li></ul> | USED |
| ATT (ATTACK_FIT_NEW) | <ul><li>PCRL `test` split (v2 tuning; "SPENT").</li><li>Attacker fitting in every study since the bench/oar line (as part of attacker_fit/AUDIT_FIT).</li><li>Fresh readers from hcal onward; the L_TREE reader banks (REPORTING_LOCK `26a055eb8`).</li></ul> | USED |
| CALIBRATION_HELDOUT | <ul><li>PCRL `test` split.</li><li>Attacker fitting (SEX) as part of AUDIT_FIT.</li><li>Task-label calibration in hcal (Ucal).</li><li>Closed as fitting data in L_TREE and this campaign.</li></ul> | USED |
| INNER_SELECTION | <ul><li>PCRL `test` split.</li><li>Attacker, nominee and utility selection in every study since oar.</li><li>The repeatedly used reporting role; the L_TREE INNER report was opened once (`83a1c64ee`).</li></ul> | USED |
| OSF_DEVELOPMENT_ASSESSMENT | <ul><li>ORIG_ASSESSMENT and CERT come from the PCRL `test` split; SMF_DEV and RGJ_DEV from the `train` split.</li><li>Outer pool scored by pilot, bench, oar, odx, cap, jcv and pnx.</li><li>CERT used for FARE certificates (oar; jcv A3).</li><li>Development assessments in rgj and smf.</li><li>Consolidated set opened in osf and at least qpc, cbp, lra and hcal.</li><li>Embargoed in every study from method-portfolio (2026-10-08) on.</li></ul> | USED |
| excluded_dup (18) / excluded_exposure (17) | <ul><li>PCRL v2 `train` and `test` splits.</li><li>Excluded from every October role as duplicates.</li></ul> | USED, unusable |
| PCRL `val` (6,017) | <ul><li>PCRL v1/v2 per-epoch validation (task loss and per-pair R² with sensitive labels).</li><li>Cotter / best.pt checkpoint selection (`experiments/run_v2_dataset.py:127-131,230,319-339`; `pcrl/training/v2_trainer.py:1178-1230`).</li><li>Many v1 sweeps (`run_sweep.py`, `run_seeds_fixed.py`, `run_baselines.py` and others).</li><li>Ledger: "used for checkpoint selection (spent as a selection set)"; owner reconciliation (row 39): "SPENT overall".</li><li>Never in `adult_jcv.npz`, so never touched by the October raw83 lineage, including L_TREE.</li><li>Never read by durable-guarantees.</li></ul> | USED |
| `dropna` rows (3,620) | <ul><li>Never in any PCRL or dg role.</li><li>Values are missing in workclass, occupation and/or native-country.</li><li>Rows missing occupation have no `occupation_group` label.</li><li>Unusable under the raw83 contract without a new imputation or encoding rule (rgj `CONFIRMATION_PLAN.md` A5).</li></ul> | Unused but incompatible (CSV: NO_USE_FOUND_IN_SEARCHED_RECORDS, estimand NONE) |

### 1.3 Why the row space is 39,170 rather than 48,842

The difference is **9,672 = 6,017 + 3,620 + 35**.

**PCRL `val` split (6,017 rows).**
- The bench, oar and jcv lineage built its roles only from the PCRL `train` split (encoder-training rows) and `test` split (jcv `row_id_convention`). The v2 validation split was never imported.
- These rows are **not reserved.** They were consumed by PCRL v1/v2 checkpoint selection.

**Missing-value rows (3,620).** These were filtered by the loader's `dropna` and never entered any role.

**Duplicates (35).**
- 17 test records equal a training record (excluded_exposure).
- 18 train rows duplicate a test-role record (excluded_dup).
- The osf study dropped both sets.

**durable-guarantees did not use the remainder.** It used only the PCRL `train` split (24,145 rows), which lies inside the row space. See section 2.

### 1.4 Reservations on Adult

None found.
- The only "reserved" Adult language is generic: "every reserved confirmation population" is closed in the L_TREE and two-recipient locks and in this campaign's seal.
- Every population reservation record found concerns ACS TX 2018 (and the TX+NY fallback).
- The CERT and CALIBRATION_HELDOUT carve-outs are method roles, not confirmation reservations, and both are now exposed.

### 1.5 Could any Adult rows still give a "least-exposed" check?

Only the 6,017 PCRL `val` rows sit outside the L_TREE lineage. A check on them would still not be confirmation:

**Exposure.**
- Their task and SEX labels were used, in aggregate, to select PCRL v2 encoders.
- Those encoders shaped earlier phases of the programme.

**Duplicates.**
- The exact-record grouping (`unit`) was computed only inside `adult_jcv.npz`.
- Duplicates between `val` and the 39,205 rows are **UNKNOWN**.

**Size.** 6,017 rows is too small to hold separate training, selection, attacker and assessment roles. It would be usable only as an assessment set for frozen packages and frozen readers.

Using it would be an owner decision, disclosed as a non-fresh, out-of-lineage holdout.

---

## 2. durable-guarantees (the AAAI repository)

**Access.**
- The local clone was relocated. Its only copy is the YOTUO drive archive `tree-durable-guarantees.tar` (sha256 `55a116f1…`, per `~/storage-relocation-20260930/RESTORE.md`). The drive is not mounted (`/Volumes` lists only `Macintosh HD`).
- The GitHub remote is **public and reachable read-only**:
  - `git ls-remote https://github.com/Bostesa/durable-guarantees.git` gives `main` = `956f5c883f51…` and `archive-cleanup` = `e702c4b73bb5…`. `archive-cleanup` is a strict ancestor of `main` (behind 4, ahead 0).
  - `gh repo view` reports PUBLIC, last pushed 2026-08-19.
  - I read files through `gh api` only; nothing was cloned.
- The relocation inventory (`inventories/tree-durable-guarantees.json.gz`, metadata only) shows the local refs at relocation:
  - the sha256 of the local `refs/heads/main` file equals the sha256 of `"956f5c8…\n"`;
  - the same holds for `archive-cleanup` (`e702c4b…`).
  - So there were **no unpushed commits** on those branches.

**Splits used, from code at `956f5c8`:**

| Data | What dg used | Evidence |
|---|---|---|
| Adult | **PCRL `train` split only** (24,145 rows after `dropna`), for every experiment. The fresh-partition rebuttal used a 50/50 split of those same rows (seed 20260724). Never `val` or `test`. | `utils/pcrl_io.py:150,188` (`split="train"`); `experiments/diagnostic.py:96-120` (`load_cell` calls `build_train_loader`); `experiments/run_fresh_partition.py:67,86-95`; `docs/fresh_partition_prediction.md`; `row_split_mapping_notes.md` ("durable-guarantees never used them") |
| ACS / Folktables | **CA 2018 1-Year only**: ACSIncome, ACSEmployment and ACSPublicCoverage, with a stratified 80/20 train/holdout split (seed 42). Experiments ran on the 80% part; the 20% holdouts are unread by dg but overlap persons used by PCRL. | `utils/folktables_io.py:56-60` (`STATE="CA"`, `YEAR="2018"`); the inventory's `data_cache/folktables` holds only `{income,employment,public_coverage}_CA_2018_train.npz` |
| Other | HMDA CA 2023 (PCRL `train`); Diabetes 130-US (PCRL `train` and a fairlearn 80/20 build); LSAC and Dutch Census 2001 (80/20; holdouts apparently unread); CelebA (train partition; val/test cached but unused) | `utils/expansion_io.py`; inventory `data_cache/{expansion,celeba}`; `DATA_EXPOSURE_LEDGER.csv` rows 1–16 |

**Residual UNKNOWN.**
- Two things cannot be inspected without mounting YOTUO:
  - any uncommitted working-tree edits that were present at relocation;
  - the git-ignored `analysis/` outputs.
- They would only add uses of the same populations (all `load_cell`-based paths read the PCRL `train` split or CA 2018).

---

## 3. ACS / Folktables

### 3.1 Spent (used)

| State-year | Use | Documented size | Evidence |
|---|---|---|---|
| CA 2018 1-Year (`psam_p06.csv`) | <ul><li>PCRL v2 Folktables encoder lineage (AGEP ≥ 16; 310,901 rows split 80/10/10 into 248,720 / 31,090 / 31,091).</li><li>Legacy ACSIncome (195,665 rows).</li><li>September 2026 redesign and release studies (the AGEP 19–34 transfer cohort of 30,000 persons in 20,147 households, plus many dependent studies).</li><li>durable-guarantees cells.</li><li>The locked independent evaluation found **0 eligible persons left: EXHAUSTED.**</li></ul> | <ul><li>Raw file: 378,817 person rows.</li><li>Eligible AGEP 19–34: 80,329 persons in 53,907 households.</li></ul> | `residual-spectral-20260910@349efa454:results/redesign_20260910_acs_locked_evaluation_v1/DATA_INDEPENDENCE.md`; `~/PCRL/results/redesign_20260907_acs_transfer_v1/config.json`; ledger rows 8–11, 26–30, 37, 40 |
| CA 2017 1-Year (`csv_pca_2017.zip`) | <ul><li>Spectral transport study (AGEP 19–34; household-hash partition).</li><li>Final partition of 15,924 persons **opened 2026-09-17** (first access 16:46:33Z, lock `31afce09…`).</li><li>Reanalysed by the evidence paper.</li></ul> | 79,869 eligible persons in 53,504 households | `349efa454:results/redesign_20260917_acs_spectral_transport_v1/{FINAL_ACCESS_LOG.json,DATA_ADMISSION.md}`; ledger row 31 |
| CA 2016 1-Year (`csv_pca_2016.zip`, `ss16pca.csv`) | <ul><li>The admission read the whole file, including label columns.</li><li>Fitting pool of 39,697 and validation pool of 15,917.</li><li>Final partition of 23,684 persons in 15,928 households **scored once**: EVALUATION_LOCK created 2026-09-23T00:49:24Z, dataset acs2016, 156 units.</li></ul> | <ul><li>Raw file: 376,035 rows.</li><li>Eligible: 79,298 persons in 53,093 households.</li></ul> | `research/pcrl-final-prospective-v1@5e154e5c4:results/pcrl_final_prospective_v1/EVALUATION_LOCK.json`; `research/pcrl-evidence-paper-v1@0d8f4b67b:results/pcrl_evidence_review_v1/DATA_2016_ADMISSION.md`; ledger rows 32–33 |

The memory notes ("2018, 2017 and 2016 spent") are **verified**.

**CA 2016/2017 persons outside the AGEP 19–34 cohort.**
- No model has used them.
- But the files were opened, and these people share households with used persons.
- The programme ledgers call CA 2016 and 2017 spent at population level.
- I mark them **UNKNOWN**: an owner ruling is needed, and they are not recommended.

### 3.2 Reserved

**TX 2018 1-Year (`csv_ptx.zip` → `psam_p48.csv`)**
- **What:** named primary confirmation cohort of the **shared-context release programme**: "same-year geographic transport"; AGEP 19–34; same-residence and service tasks.
- **Where:** `CONFIRMATION_PLAN.md`, first committed in `66e8186f8` on 2026-09-24; branch tip `research/pcrl-shared-context-release-v1@537e74a44`.
- **Lock status:** the plan says it "authorises nothing". It pre-declares the partition salt `PCRL-shared-context-confirmation-v1|2018|48|` and a future `CONFIRMATION_LOCK.json` with a single opening owner. **No such lock exists.**
- **Precondition not met:** the programme closed with `EXPERIMENTAL_NO_ADVANTAGE`. The privacy-first selector's Texas power table was **not triggered** (`8fdc61e39`).
- **Access history:** only HTTP HEAD requests; never downloaded.
- **Why it still counts as reserved:** every later plan or ledger treats it as reserved, typically as "Texas 2018 carries a prior reservation". That covers the rgj, smf and osf confirmation plans and the cbp, lra, lcr and hcal exposure ledgers (2026-10-04 to 10-07).
- **Owner:** the decision to release or spend it rests with the user.

**NY 2018 1-Year (`csv_pny.zip`)**
- No use found.
- Named only as the registered **conditional fallback** (TX+NY pooled, triggered only if the TX label-free count falls short), in the same plan.
- Treat it as entangled with the TX reservation until the user releases it.

### 3.3 No use found (candidates for recipe replication only)

| Candidate | Status | Schema notes | Documented size |
|---|---|---|---|
| Other 2018 1-Year states (not CA, TX or NY) | **No use found** (bounded search, section 3.6) | Same national 2018 dictionary as CA | None in any receipt |
| Any state, 2019 and 2021–2023 1-Year | **No use found** | `RELP` (00–17) replaced by `RELSHIPP` (20–38) with no exact back-map; 2023 renames `ST` to `STATE` | None |
| 2020 1-Year | Ineligible | Experimental PUMS only | — |
| Any state, 2010–2015 1-Year | No use found (shared-context pickaxe covered `folktables/20{10-17}` and `acs_20{10-15}`) | Dictionary differences **not checked** | None |
| ACS 5-Year files that include 2016–2018 | Overlap the spent CA records | — | — |

**Documented sizes for TX and NY:**
- `csv_ptx.zip` is 47,257,733 B and `csv_pny.zip` is 34,958,982 B (HTTP HEAD, 2026-09-24, shared-context plan section 2).
- The plan's "TX ≈ 0.7× CA cohort" is an "unverified expectation, not a count".

### 3.4 Usage verification performed in this audit (2026-10-15)

**Ref-tip scan for PUMS file names.**
- I ran `git grep -P` across every ref tip in **both** stores: `~/PCRL/.git` (129 refs) and `~/PCRL/.worktrees/pcrl-method-portfolio-v1/.git` (108 refs).
- Non-CA PUMS names found: only `psam_p48`, `csv_ptx.zip`, `psam_p36` and `csv_pny.zip`.
- CA names found: only `psam_p06`, `psam_h06`, `csv_pca.zip`, `csv_pca_2017.zip`, `csv_pca_2016.zip` and `ss16pca`, plus the 2016–2018 dictionaries.

**Pickaxe history.**
- `git log --all -S` for `csv_ptx`, `psam_p48`, `csv_pny` and `psam_p36` in both stores finds only plan and ledger commits: `66e8186f8` (2026-09-24), `b67f04b5e` (10-01), `2be8ceabe` and `65a6eeb44` (10-04).
- No data commit was found.

**Other repository checks.**
- No other state name appears in any CONFIRMATION, EXPOSURE, DATA or PROTOCOL file on any ref.
- The relocation inventories (`*.json.gz`) contain only CA PUMS names.
- On the local disk, the only PUMS file is `~/PCRL/data/folktables/2018/1-Year/psam_p06.csv`.
- The PCRL Folktables loader defaults to CA 2018. A contingency line (`--states NY,TX,CA`, `results/folktables/adaptation_notes.md`, commit `e3ac31107`) has no matching data or run record.

**Other local repositories.**
- I ran a file-content scan over `removal-pricing`, `offcriterion`, `offcriterion-full-paper`, `fair-pca`, `Obliviator`, `datasets`, `data` and `uwaudit`.
- No PUMS or Folktables use was found. The two `csv_p…` hits in offcriterion were `csv_path` false positives.
- `fair-pca` is a third-party clone (Kleindessner et al.; last upstream commit 2026-01-05) with an Adult loader (`comparison_with_Agarwal/load_data.py`).
  - The loader expects `adult.data` and `adult.test`, and no Adult file exists in that clone, so there is no evidence it was ever run.
  - Any use would only add exposure to already-exposed Adult records. It could also touch the `dropna` rows, since that loader does not drop missing values.

### 3.5 Schema and task compatibility: why ACS cannot confirm the frozen release

L_TREE consists of CatBoost depth-4 teachers on raw83 with leaf-score corrections. Raw83 is the Adult loader's fixed one-hot category sets plus 5 numeric columns standardised on OSF_DEFENSE_FIT. The frozen release (per-task probabilities plus decision) cannot take ACS rows without a new input contract:

| Element | Problem |
|---|---|
| `capital-gain`, `capital-loss` | No ACS equivalent. Dropping or substituting them (for example with interest or dividend income) changes the inputs. |
| `relationship` | Adult's Husband/Wife nearly determine SEX for about 46% of rows (jcv shortcut audit). ACS `RELP` spouse codes are sex-neutral, so the main development SEX proxy disappears or changes. |
| `marital-status`, `workclass`, `education`, `native-country` | `MAR`, `COW`, `SCHL` and `POBP` use different category sets. Adult's 1994-CPS labels (for example Married-AF-spouse, "South", "Hong", "Outlying-US", Yugoslavia) need registered crosswalks. |
| `occupation_group` (6 classes) | Needs a registered `OCCP`→Adult occupation→group crosswalk. Class 5 (Armed-Forces) is extremely sparse in Adult and may be empty under civilian filters. |
| income label | A nominal >50K threshold in 1994 dollars versus 2018 dollars (and ADJINC). The class meaning shifts. |
| SEX | Available (`SEX`). Grouping must be by household (`SERIALNO`). |

**Any ACS study is therefore a transported benchmark: the recipe is refit end to end.** That was already the prior programme verdict (rgj `CONFIRMATION_PLAN.md` section 2: "No fresh rows exist in Adult's 83-column schema … only the frozen procedure can be tested … not confirmation of a released model").

A geographic transport such as TX 2018 cannot show that every person is new, because a person who moved within the survey year could appear at two addresses. A temporal transport would need its own relationship-variable mapping.

### 3.6 Search scope and limits

**Covered:**
- both PCRL git stores (all ref tips, plus history pickaxes for the TX and NY names);
- the results and `.private` trees under every worktree;
- the durable-guarantees GitHub repository @ `956f5c8`, read through the API;
- the relocation inventories and local file names;
- the memory notes.

**Not covered:**
- S3 object listings (excluded by the no-paid-cloud rule; the 2026-09-24 audit listed S3 names and found only `psam_p06.csv`);
- deleted EC2 disks, other machines and archive tar members;
- the contents of the YOTUO drive.

"No use found" is bounded evidence, not a certificate. Re-run the checklist in rgj `CONFIRMATION_PLAN.md` section 5 immediately before any acquisition.

---

## 4. Other populations

None is schema-compatible with raw83 or with the income + occupation_group + SEX task set.

| Population | Use | Compatibility |
|---|---|---|
| HMDA CA 2023 | PCRL 63,747 / 13,660 / 13,661 (test "SPENT"); dg `train` | Mortgage applications; no income-threshold or occupation task. NONE. |
| Diabetes 130-US | PCRL 50,053 / 10,725 / 10,728 (by patient); dg fairlearn build (80/20) | Clinical. NONE. |
| CelebA | Train subsample, val and test rows 0–4,999 spent; rows 5,000–19,961 possibly untouched [unverified; identities overlap] | Images. NONE. |
| Bias in Bios | Train and dev spent; test not loaded by stored code [unverified against cloud logs] | Text. NONE. |
| UCI HAR | v1 experiments | Sensors. NONE. |
| LSAC, Dutch Census 2001 (dg) | 80% train used; 20% holdouts apparently unread [unverified] | Different variables; Dutch has a binary occupation task and no income. NONE for this recipe. |
| CPS-derived sources (for example 1994–95 CPS extracts such as UCI Census-Income KDD) | **Not referenced in any receipt** (no `IPUMS`, `ASEC` or `Current Population Survey` hit in either store) | Closest to Adult, but not raw83. Adult itself was extracted from the 1994 census database, so person overlap cannot be ruled out (no ids). UNKNOWN; at most a transported replication. |

---

## 5. Overhead and sizing (no labels used)

**Frozen-release confirmation (if a fresh, compatible population existed):**
- **Fitting:** no training and no selection, since the nominee is frozen (L_TREE rho1 E20, seeds 0 and 1).
- **Readers:** either reuse the frozen REPORTING_LOCK readers (fitted on the exposed ATT set), or fit fresh readers, which needs an ATT-sized fit and validation set (about 4,064 people).
- **Assessment precision:** at the INNER size (2,235 people, 2,234 groups) the established pair benefit versus RAW_PROJ, [0.0257, 0.0754], is a simultaneous interval: K = 78, Bonferroni t = 3.418, 2,048 paired duplicate-group bootstrap draws.
  - Its half-width is about 0.025, so SE ≈ 0.0073.
  - SE scales roughly as 0.0073 × sqrt(2,235 / N). Multiply by the critical value of whatever endpoint family the confirmation registers.
  - Example: all 6,017 `val` rows give SE ≈ 0.0044.
- **Competitive comparisons cannot be established at any size.** The point estimates against the historical codes (0.014–0.018) already sit below the 0.020 margin, so a larger N cannot establish them if those points are accurate.

**Recipe replication (ACS):** the L_TREE role sizes are:

| Role | People |
|---|---|
| UPDATE + GUARD (teachers) | 10,223 |
| FIT2 (selection) | 5,211 |
| ATT (readers) | 4,064 |
| INNER (report) | 2,235 |
| **Subtotal** | **21,733** |
| CALIBRATION_HELDOUT (optional, for a Ucal-style reference) | 2,000 |
| HEAD_VALIDATION (optional, for head selection) | 1,500 |
| **Total with both optional roles** | **about 25.2k** |

- ACS roles must be allocated by household (CA 2018 AGEP 19–34 averaged about 1.49 eligible persons per household).
- All six occupation groups and both SEX classes need support in the fitting pools, which must be checked label-blind at admission.
- No candidate's count is documented, so feasibility needs a label-blind admission count (feature, key and weight columns only) after a registered protocol.

---

## 6. UNKNOWNs and what would resolve them

| # | UNKNOWN | Resolution |
|---|---|---|
| U1 | Exact-record duplicates between the PCRL `val` split (6,017) and the 39,205-row space | A label-free duplicate check on the 83 inputs plus `fnlwgt`, using the jcv canonical-key rule without labels. Only matters if the owner considers a non-fresh `val` check. |
| U2 | Per-column composition of the 3,620 `dropna` rows (for example rows missing only native-country) | Counting them requires reading the occupation column, the task-label source. Deferred to the owner; irrelevant to frozen confirmation, because the rows fall outside the raw83 contract anyway. |
| U3 | Uncommitted or git-ignored durable-guarantees local state at relocation | Mount YOTUO, verify `tree-durable-guarantees.tar` (sha256 `55a116f1…`), and inspect it read-only. |
| U4 | Person and household counts for TX 2018, NY 2018, other 2018 states and 2019+ years | A label-blind admission (feature, key and weight columns only) after a registered protocol, or official Census sample-size documentation. Not available from any receipt. |
| U5 | Non-use outside the searched scopes (S3 objects, deleted EC2 disks, other machines) for TX, NY and other states | Re-run the S3 name listing (owner-authorised; outside this role's no-paid-cloud rule) and the rgj section 5 checklist immediately before any acquisition. |
| U6 | Whether the TX 2018 reservation (and the NY fallback) is still live after the shared-context programme closed | An explicit user decision to keep, release or transfer the reservation. |
| U7 | CA 2016/2017 persons outside AGEP 19–34 | An owner ruling on whether population-level "spent" covers them (the files were opened and households are linked). |
| U8 | Schema of 2010–2015 1-Year dictionaries relative to 2018 | A dictionary diff from official documentation only. |
| U9 | Person overlap between CPS-derived alternatives and Adult | Cannot be resolved (no person ids). Treat such sources as non-independent of Adult. |

---

## Evidence index

**Worktree files.** Paths below are relative to `~/PCRL/.worktrees/pcrl-frozen-tree-control-completion-v1/` unless absolute.

Adult roles and lineage:
- `results/pcrl_joint_complete_view_method_v1/DATA_ADMISSION.json`
- `results/pcrl_online_strength_frontier_v1/ROLE_MANIFEST.json`
- `results/pcrl_heldout_calibration_v1/ROLE_MANIFEST.json`
- `results/pcrl_heldout_calibration_v1/EXPOSURE_LEDGER.md`
- `results/combined_output_aware_removal_v1/PROTOCOL.md`

L_TREE and this campaign:
- `results/pcrl_task_headroom_protected_outputs_v1/{SCIENCE_LOCK.json,ADMISSION.json,LOCK_NOTES.md}`
- `results/pcrl_frozen_tree_control_completion_v1/PROTOCOL_SEAL.json`

Split counts and exposure ledgers:
- `results/combined_evaluation_preparation_v1/notes/artifacts/{row_index_verification.json,row_split_mapping_notes.md}`
- `results/combined_empirical_preparation_v1/DATA_EXPOSURE_LEDGER.csv`

Confirmation plans and population notes:
- `results/pcrl_refreshed_guarded_joint_v1/CONFIRMATION_PLAN.md`
- `results/pcrl_online_strength_frontier_v1/CONFIRMATION_PLAN.md`
- `results/pcrl_learned_decoder_constrained_release_v1/EXPOSURE_LEDGER.md`

Code:
- `pcrl/data/adult.py`
- `experiments/run_v2_dataset.py`
- `pcrl/training/v2_trainer.py`

Private role exports:
- `~/PCRL/.worktrees/pcrl-separate-pool-utility-guard-v1/.private/separate_pool_utility_guard/roles/ROLE_SPLIT.json`
- `~/PCRL/.worktrees/pcrl-utility-reference-repair-v1/results/pcrl_utility_reference_repair_v1/.private/role_export_v1/EXPORT_RECEIPT.json`
- `~/PCRL/.worktrees/pcrl-compressed-income-safeguard-v1/.private/compressed_income_safeguard/r1/SOURCE_PATHS.json`
- `~/PCRL_eval_cache_private/hcal_v1/admitted/inputs/{deploy_input.npz (headers only),schema.json}`

**Main PCRL store (git refs):**
- `research/pcrl-shared-context-release-v1@537e74a44` (`CONFIRMATION_PLAN.md`)
- `research/pcrl-privacy-first-selector-v1@8fdc61e39`
- `research/pcrl-final-prospective-v1@5e154e5c4`
- `research/pcrl-evidence-paper-v1@0d8f4b67b`
- `residual-spectral-20260910@349efa454` (2017 transport; 2018 locked evaluation)
- `~/PCRL/results/redesign_20260907_acs_transfer_v1/config.json`

**durable-guarantees:**
- `github.com/Bostesa/durable-guarantees@956f5c8`: `utils/pcrl_io.py`, `experiments/diagnostic.py`, `utils/folktables_io.py`, `experiments/run_fresh_partition.py`, `docs/fresh_partition_prediction.md`
- `~/storage-relocation-20260930/{RESTORE.md,inventories/tree-durable-guarantees.json.gz}`
