# Exposure ledger: online-strength frontier study

> **Every row in this study has been used before.** All roles and all four assessment pools are UCI Adult records
> that earlier PCRL studies, and the durable-guarantees repository, used to train encoders, fit heads, LEACE maps,
> FARE trees, critics and attackers, select models, compute certificates and report results.
>
> **OSF_DEVELOPMENT_ASSESSMENT is a reused benchmark.** It is the fixed union of four previously used pools. It is
> **not fresh data, not confirmation and not an independent replication.** Scores on it are locked exploratory
> development checks.
>
> **Earlier outcomes on these same pools shaped this study.** The hypotheses, the strength grid and the incumbent were
> chosen after those results were seen (§5). Masking the labels now is procedural. It cannot undo that adaptation, and
> nominal intervals do not correct for it.

Owner: data/custody role. Inspected read-only on 2026-10-05 (UTC). Base: the closed strength-matched feedback study at
`a9951ed2fed9943d445a208a8a7e456a56f39114`.

- Counts and hashes are in `ROLE_MANIFEST.json`.
- `provenance/role_check.py` recomputed the roles, pools and numeric refit independently. Its `--compare-loader` step
  matches `osf.data.load()` exactly (verdict `MATCH`).
- The certification-pool determination is in `ROLE_MANIFEST.json` (`cert_eligibility`, §6 below). Checkpoint
  admission is in `ADMISSION.json`.
- This file publishes no label statistic of any role.

## 1. Rows and roles

The admitted input is `adult_jcv.npz` (sha256 `e0d9e54a…5f12`, 39,205 post-dropna rows). It was built by
`jcv/data.py` from the pinned PCRL loader (`pcrl.data.adult.AdultDataset`).

| Origin | Old role (oar-roles-v1) | rgj | smf | **This study (osf)** |
|---|---|---|---|---|
| PCRL **train split** (80% seed-42 permutation of `adult.data`) | defense_train 19,230 | DEFENSE_FIT | NEW_DEFENSE_FIT 15,434 / NEW_DEVELOPMENT_ASSESSMENT 3,796 | **OSF_DEFENSE_FIT 15,434** / **SMF_DEV 3,796** |
| same | defense_val 4,897 | HEAD_VALIDATION 1,500 / DEVELOPMENT_ASSESSMENT 3,397 | HEAD_VALIDATION / dropped | **HEAD_VALIDATION 1,500** / **RGJ_DEV 3,397** |
| same | excluded_dup 18 | dropped | dropped | dropped |
| PCRL **test split** (`adult.test`) | attacker_fit 6,065 | AUDIT_FIT | AUDIT_FIT | **AUDIT_FIT 6,065** |
| same | attacker_val 2,235 | INNER_SELECTION | INNER_SELECTION | **INNER_SELECTION 2,235** |
| same | assessment 5,243 | dropped | dropped | **ORIG_ASSESSMENT 5,243** |
| same | cert 1,500 | dropped | dropped | **CERT 1,500** (custody: eligible, §6) |
| same | excluded_exposure 17 | dropped | dropped | dropped |

- **OSF_DEVELOPMENT_ASSESSMENT** = ORIG_ASSESSMENT + RGJ_DEV + SMF_DEV + CERT = **13,936 rows / 13,929 groups**.
  - No pool group overlaps a fitting role or an exclusion, so no row was excluded for overlap.
  - No group is shared between two pools.
  - Without CERT it would be 12,436 rows / 12,429 groups (both variants are in `ROLE_MANIFEST.json`).
- **OSF_DEFENSE_FIT subroles** are smf's draw, unchanged: CRITIC_FIT 10,764, CRITIC_VAL 2,343 and DIAGNOSTIC_CALIB
  2,327 (= smf CONTROLLER_CALIB; no controller is trained here). Encoders and training heads train on every
  OSF_DEFENSE_FIT row.

## 2. Studies and pins

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
| smf | research/pcrl-strength-matched-feedback-v1 @ `a9951ed2f` | `results/pcrl_strength_matched_feedback_v1/` |

Per-study detail for pilot through smf is in the predecessors' own ledgers:
- `results/pcrl_refreshed_guarded_joint_v1/EXPOSURE_LEDGER.md`
- `results/pcrl_strength_matched_feedback_v1/EXPOSURE_LEDGER.md`

This file consolidates that detail and adds the smf uses.

## 3. Summary of use, by pool and role

Key:
- **T**: trained or fitted on (encoders, heads, critics, maps, trees, attackers)
- **S**: selected on (head C, configurations, nominees, attackers)
- **Sc**: scored or reported on
- **C**: FARE certificate statistic computed on it
- **–**: dropped or not used

| Rows | PCRL v2 | dg | pilot / bench | oar / odx / cap | jcv / pnx | rgj | smf |
|---|---|---|---|---|---|---|---|
| OSF_DEFENSE_FIT | T | T | T (LEACE, defenses) | T (heads, FARE trees) | T (all models) | T (all models; critic subroles) | T (all models) |
| SMF_DEV | T | T | T | T | T | T (critics, refit stopping, calibration by rgj subrole) | **Sc once** after its EVALUATION_LOCK |
| HEAD_VALIDATION | T | T | T | **S** (head C), T (FARE) | **S** (head C) | **S** (head C) | **S** (head C) |
| RGJ_DEV | T | T | T | **S** (head C), T (FARE) | **S** (head C) | **Sc once** after its EVALUATION_LOCK | – |
| AUDIT_FIT | Sc (test set) | – | T (attackers) | T (attackers, probes) | T (attackers, critics) | T (attackers) | T (attackers) |
| INNER_SELECTION | Sc (test set) | – | S (attackers, σ\*) | S | **S** (nomination, FARE config) | **S** (controls, nominees) | **S** (Phase A/B, C\*) |
| ORIG_ASSESSMENT | Sc (test set) | – | **Sc** (outer) | **Sc** (outer) | **Sc** (outer) | – | – |
| CERT | Sc (test set) | – | T (inside attacker_fit) | **C** (oar, odx); carve-out made in oar | **C** (jcv A3; pnx reused) | – | – |

Some boundaries were not traced:
- dg's use of the `adult.test` rows was not found at its pin (see the predecessors' ledgers).
- The PCRL v2 encoders' own 20% validation split of `adult.data` is not in the input file.

## 4. History per row set

### OSF_DEFENSE_FIT (= smf NEW_DEFENSE_FIT; old defense_train minus SMF_DEV)

- **PCRL v2 lineage.** These rows are part of the Adult training split, which is the same for every seed and round.
  They trained the Round-4 checkpoints later used by pilot and bench.
- **dg.** Every Adult cell was built from the same train split.
- **pilot and bench.** LEACE maps and noise defenses were fitted on these rows.
- **oar, odx and cap.** Release heads were fitted on them, and official FARE trees on all of `defense_fit`.
- **jcv, pnx and rgj.** Every encoder, warm start, critic, LEACE map and FARE tree was fitted on these rows, together
  with SMF_DEV.
- **smf.** smf refitted everything on exactly these 15,434 rows:
  - warm starts, U and the RAW-J/RAW-L β 0.1/0.3 arms;
  - Phase A/B normalized arms;
  - LEACE maps and FARE trees;
  - deployed-head scalers and weights.

  The critic subroles trained, validated or calibrated critics and the controller.
- **This study.** It reuses the admitted smf checkpoints that were fitted on these rows (`ADMISSION.json`). All new
  arms are fitted on the same rows.

### HEAD_VALIDATION (30% of old defense_val; unchanged since rgj)

- **Head selection.** Its labels chose deployed-head C in oar/odx/cap (`oar-head-v1` holdout), jcv/pnx (old
  defense_val), rgj and smf.
- **This study.** It is used again for head C selection only.

### AUDIT_FIT and INNER_SELECTION (old attacker_fit / attacker_val; `adult.test`)

- **PCRL v2 lineage.** Part of the reported test set of every Adult round.
- **Attacker fitting.** AUDIT_FIT fitted every attacker slate from pilot through smf.
- **Selection.** INNER_SELECTION selected attackers and σ\*, set the utility gates, and nominated candidates and
  controls in jcv, pnx, rgj and smf (including smf's C\* = RAW-J).
- **This study.** Both are used in the same roles again.

### ORIG_ASSESSMENT (old assessment; 5,243 rows; `adult.test`)

- **PCRL v2 lineage.** Part of the fixed Adult test set. Results were reported on it and methods were iterated
  against it.
- **Outer assessment.** It was the outer assessment of pilot, bench, oar, odx, cap, jcv (27 outer units) and pnx
  (39 outer units). Their published endpoints and verdicts were computed on these rows.
- **Published aggregates.** oar published the role's SEX and task class counts (`ROLES_AND_SUPPORT.json`, support
  section).
- **rgj and smf.** Dropped at load: no prediction and no score.
- **Never fitted on.** These rows never fitted a defense, head or attacker in any traced study.

### RGJ_DEV (rgj DEVELOPMENT_ASSESSMENT; 70% of old defense_val; 3,397 rows)

- **Training history.** Same as OSF_DEFENSE_FIT for the PCRL v2 lineage, dg, pilot and bench: these rows trained
  encoders and fitted LEACE maps.
- **oar to pnx.** As part of the old defense_val (`oar-head-v1` holdout), their labels **selected the C of every
  release head** in oar, odx, cap, jcv and pnx. FARE trees in oar were fitted on all of `defense_fit`, which includes
  them.
- **rgj.** Withheld from rgj's fitting and selection, then **scored once** (39 outer units) after rgj's pushed
  EVALUATION_LOCK. Its J-G / J-R / J-O outcomes are published in `results/pcrl_refreshed_guarded_joint_v1/`.
- **smf.** Dropped at load.

### SMF_DEV (smf NEW_DEVELOPMENT_ASSESSMENT; 20% of old defense_train; 3,796 rows)

- **Training history.** Same as OSF_DEFENSE_FIT up to and including rgj. These rows trained every predecessor
  encoder, warm start, LEACE map and FARE tree (all 42 jcv trees have n_fit = 19,230), and the PCRL v2 and dg models.
- **rgj critic subroles.** By rgj subrole, these rows were used for critic training (2,679 rows), refit early
  stopping (566) and multiplier calibration (551).
- **Numeric standardisation.** The admitted standardisation was fitted on all 19,230 rows, including these.
- **smf.** Withheld (labels masked) from every smf fit and selection, then **scored once** (51 outer units) after
  smf's pushed EVALUATION_LOCK. The admitted smf checkpoints encoded these rows for that scoring only (scoring, not
  fitting; `ROLE_MANIFEST.json` `assessment_exclusion_from_eligible_models`).

### CERT (old certification pool; 1,500 rows; `adult.test`)

- **PCRL v2 lineage.** Part of the fixed Adult test set.
- **pilot and bench.** Part of `pilot-roles-v1` attacker_fit (7,571 rows): **attackers were fitted on these rows**.
- **oar.** The carve-out was created here: 20% of the exposure-cleaned attacker_fit groups (salt `oar-cert-v1`)
  became the FARE certification set. oar computed FARE's demographic-parity certificates on it and published the
  role's SEX and task class counts (`ROLES_AND_SUPPORT.json`).
- **odx.** FARE certificates again: unavailable or vacuous, and the refusals counted rows identical to fit rows.
- **jcv.** FARE certificates (amendment A3, `FARE_CERTIFICATES_A3.json`): vacuous for the F income tree and
  unavailable for the others. "No registered decision uses certificates."
- **pnx.** Reused the jcv certificate status ("unavailable or vacuous").
- **rgj and smf.** Dropped at load.
- **Never fitted on, after oar.** From oar onward, no traced study fitted a defense, head or tree on these rows.
  Their uses were attacker fitting before the carve-out (pilot/bench) and certificate statistics after it.

## 5. What the consolidated assessment is, and how earlier outcomes shaped this study

1. **Reused benchmark.**
   - Every pool was scored, selected on, certified on or trained on before.
   - SMF_DEV and RGJ_DEV were each scored once by the previous two studies.
   - ORIG_ASSESSMENT was the outer assessment of seven studies.
   - Pooling them buys measurement precision on old rows. It does not create a holdout. Renaming or resplitting
     cannot restore independence.
2. **Earlier outcomes shaped this design.** Each of the following was seen before this study was designed:
   - **SMF_DEV (smf).** RAW-J β 0.3 reached pair SEX AUC 0.810 against U 0.882 at equal accuracy, and its logged
     combined penalty/task ratio was about 2.6–3.2. This study's incumbent, its higher ρ grid (1.5, 3, 5) and its
     raw-versus-normalized framing come directly from that.
   - **RGJ_DEV (rgj).** The failure of the refreshed guarded joint method was scored here.
   - **ORIG_ASSESSMENT (pilot through pnx).** Outer outcomes on these rows (for example, outputs still leaking after
     erasure in oar/odx/cap) shaped the research line that led to joint training from jcv onward.
   - **CERT (oar, odx, jcv).** Vacuous or unavailable certificates are why no certificate claim is made.
3. **Not fresh, not confirmation, not replication.**
   - Locked scoring of the frozen bank on these rows gives exploratory development evidence, conditional on the fitted
     models and on this history of reuse.
   - A pass would be a development criterion met on a reused benchmark. It would not be a population claim or an
     independent replication.
   - No genuinely unused Adult cohort exists in this schema.
4. **Shared inputs.** 5,191 of the 13,936 assessment rows have an 83-column input vector that also occurs in
   OSF_DEFENSE_FIT:

   | Pool | Rows sharing an input vector with OSF_DEFENSE_FIT |
   |---|---:|
   | ORIG_ASSESSMENT | 1,942 |
   | RGJ_DEV | 1,298 |
   | SMF_DEV | 1,419 |
   | CERT | 532 |

   These are different exact-record groups. This is a property of Adult's mostly categorical inputs, and they are
   never pooled as one person.

## 6. Certification-pool eligibility (custody determination)

**Rule.** Prompt §6: CERT enters only if custody shows it never entered the eligible current models' fitting or
selection. Fit provenance takes precedence over the pool's historical name.

**Verdict: ESTABLISHED.** `ROLE_MANIFEST.json` `cert_eligibility.verdict = ESTABLISHED`, `CERT_ELIGIBLE = true`.
The evidence is label-free unless stated otherwise.

- **Code at the evidence SHA.** `rgj/data.py` and `smf/data.py` drop the old cert, old assessment and rgj assessment
  rows at load. `rgj.train.TData` uses DEFENSE_FIT rows only, and the critics use CRITIC_FIT within them. These files
  are byte-identical to the smf lock hashes and to this worktree. Recomputed independently, no CERT, ORIG_ASSESSMENT
  or RGJ_DEV row receives an smf role.
- **Receipts of every eligible model.** This covers 3 warm starts, 3 U, 24 RAW (e20/e40), 12 RAW run receipts,
  3 LEACE, 42 FARE units and 42 FARE trees.
  - **Integrity.** Every COMPLETE.json verifies.
  - **Release rows.** Release row sets contain **0** CERT, ORIG_ASSESSMENT, RGJ_DEV or excluded rows.
  - **Deployed heads.** All 102 heads have scaler moments bitwise equal to the OSF_DEFENSE_FIT rows (n = 15,434).
  - **LEACE.** Fit-row id hashes and fit-feature hashes equal OSF_DEFENSE_FIT and U's features on it.
  - **FARE.** Tree n_fit, ordered fit-row fingerprints and fit-row hash sets equal the OSF_DEFENSE_FIT inputs. The
    trees encoded only the 29,030 smf rows.
  - **RAW.** The critic head equals the warm head.
- **Selection.** None of the eligible models' configuration banks was chosen on CERT:
  - head C used HEAD_VALIDATION only;
  - β ∈ {0.1, 0.3} was registered in smf's DATA_AND_ENGINEERING_LOCK;
  - the FARE grid equals oar's EXECUTION_LOCK grid, frozen before any oar fit, and oar's FARE selection used
    attacker_val only.

  Certificates were descriptive (jcv A3).
- **Admission** (`ADMISSION.json`) adds the label-dependent receipts:
  - LEACE concept hashes;
  - FARE fit-label and full input fingerprints;
  - HEAD_VALIDATION log losses equal to the recorded tables;
  - bitwise forward passes on the osf inputs.
- **Remaining gap.** The encoder weights of the warm starts, U and RAW are tied to OSF_DEFENSE_FIT by pinned code and
  receipts. They have not yet been re-derived bitwise from these rows: the lead's `replay` stage (and a recommended
  warm-start replay) does that.
- **Scope.** Eligibility concerns fit provenance only. CERT remains historically exposed (§4).

## 7. Reused checkpoints

**Admitted** (`ADMISSION.json`; private copies under `<PRIVATE_CACHE>/osf_v1/admitted/`):
- smf warm starts;
- U (`tl__s{k}__e40`);
- RAW-J/RAW-L β 0.1/0.3 at epochs 20/40, with their run receipts;
- official LEACE `lc__s{k}__E`;
- all 42 smf FARE units and their 42 trees.

These checkpoints are fitted on OSF_DEFENSE_FIT only, but those rows are themselves historically exposed (§4).

**Ineligible:**
- Every jcv, pnx and rgj checkpoint, map and tree. They were fitted on all 19,230 old defense_train rows, which
  include SMF_DEV, and read numerics standardised on those rows. Their heads were C-selected on old defense_val, which
  includes RGJ_DEV.

**Not admitted** (not needed):
- smf Phase A/B arms, whose trajectory and schedule differ from the common 40-epoch path;
- U at epoch 20;
- smf inner and outer records.

## 8. Input contract and proxies

- **Columns.** 83 permitted columns, unchanged:
  - 5 numeric columns, re-standardised on OSF_DEFENSE_FIT exactly as in smf. Raw integers were recovered and checked
    against the pinned raw loader for every kept row, including all four pools.
  - 78 one-hot columns with the fixed vocabulary. No unseen category occurs in any role or pool.
- **Excluded inputs.** SEX, race, income, occupation, fnlwgt and identifiers are not inputs.
- **Proxies kept.** relationship = Husband / Wife are kept as difficult proxies (prompt §6). In the jcv shortcut
  audit they nearly determine SEX for about 46% of rows. Recovery here includes signal carried directly by permitted
  inputs.

## 9. Publication cautions

- **Label aggregates.** Earlier public files already hold label aggregates over these pools:
  - oar's SEX and task class counts for the old assessment and cert roles;
  - rgj's full-precision SEX prior over all 19,230 old defense_train rows.

  This is historical exposure, not new exposure. Do not publish a new OSF_DEFENSE_FIT or pool label aggregate before
  EVALUATION_LOCK in a way that, with these, lets a sealed pool's share be derived by subtraction. smf published only
  the prior's hash.
- **Placeholders.** Public files use `<PRIVATE_CACHE>`, `<DRIVE_ROOT>`, `<WORKTREE>` and `<SOURCE_WORKTREE>`. They
  never contain a home folder, user name, volume label or drive path.
- **Historical drive-folder exposure.** It remains in four earlier commits of the rgj branch. It is unchanged and not
  claimed to be removed. No history was rewritten.
