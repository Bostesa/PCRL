# Exposure ledger — hcal

## Registered exposure statement (verbatim, prompt §3)

"This study is motivated by observed outcomes on repeatedly used Adult development data. Frozen maps, teachers and
earlier decoder outcomes are known. New calibration roles, fitting rules and selection rules are chosen after those
outcomes. All new assessment results are exploratory development evidence, not fresh confirmation. Their intervals do
not account for the full history of research selection. Assessment masking controls this execution's selection process;
it cannot undo previous exposure. No new population is opened."

## What is already known (before any hcal fit)

- **lra outcomes on the same assessment rows.**
  - The D1 learned decoder (κ = 32, fitted on the teacher's own fitting rows) raised held-out occupation log loss on
    identical tokens: +0.010 on the selected fixed map, +0.024 on C-TASK.
  - All twelve named same-token contrasts worsened (nominal 95%).
  - The lra P\* (D0 JOINT λ0.1) cut pair AUC by 0.034 versus FINE-TASK and passed 10 of 11 clauses. Its occupation
    log-loss excess was 0.0081 (upper bound 0.0121 against 0.01). This repeats qpc on the same rows.
- **Anchors.** The source assessment anchors (pair AUC, occupation LL) of U, Q, FINE-TASK, JOINT λ0.1, SEQ-21 D1 and
  CLASS are known.
- **This design is outcome-motivated.** The held-out calibration roles, the calibrator families and the
  calibration-matched gate were chosen after those outcomes.

## Roles and their use in hcal (ROLE_MANIFEST.json)

| Role | Rows / groups | Historical use | hcal use |
|---|---|---|---|
| OSF_DEFENSE_FIT | 15,434 / 15,428 | teacher encoders and heads; fine partitions; teacher means; privacy-map search (SEX); lra D1 and C-TASK (task labels) | frozen priors only (no new fit); TRAIN_MATCHED task labels (diagnostic T-TOKEN32); constant predictor; descriptive fitting-row losses (never an objective, gate or selection input) |
| CALIBRATION_TRAIN_MATCHED (inside OSF_DEFENSE_FIT) | 2,002 / 2,000 (2,000 representatives) | as OSF_DEFENSE_FIT | T-TOKEN32 fitting (4 diagnostic partitions); fitting-role contrast |
| HEAD_VALIDATION | 1,500 / 1,499 | head selection | never read |
| AUDIT_FIT | 6,065 / 6,061 | attacker fitting in every earlier study | split below |
| CALIBRATION_HELDOUT (inside AUDIT_FIT) | 2,001 / 2,000 (2,000 representatives) | attacker fitting (SEX) | task labels for H-TOKEN32, H-GLOBAL-TEMP, H-CLASS-TEMP and the U calibrations; calibration-row losses |
| ATTACK_FIT_NEW (inside AUDIT_FIT) | 4,064 / 4,061 | attacker fitting (SEX) | fresh reader fitting (SEX) |
| INNER_SELECTION | 2,235 / 2,234 | attacker and nominee selection | utility gates, Ucal\*, reader selection, nomination (task labels and SEX) |
| OSF_DEVELOPMENT_ASSESSMENT | 13,936 / 13,929 | assessed by qpc, cbp, lra and earlier studies | sealed until this study's pushed EVALUATION_LOCK; one opening |

**Legacy readers.** The admitted legacy readers (lra complete-interface banks) were fitted on all historical AUDIT_FIT
rows, including the rows now in CALIBRATION_HELDOUT. This is disclosed. They never fit on inner or assessment labels.
The calibrators fit task labels on CALIBRATION_HELDOUT, the fresh readers fit SEX on the disjoint ATTACK_FIT_NEW, and
no calibrator reads SEX.

## Material label use (stated plainly)

- **New supervised use.**
  - Task labels of CALIBRATION_HELDOUT: rows held out from the teachers' parameter fitting and head selection, but used
    historically for attacks.
  - Task labels of CALIBRATION_TRAIN_MATCHED: inside the teachers' fitting rows; diagnostic only.
- **Unchanged.** No new teacher, encoder, head, fine partition, token assignment, privacy weight or capacity is fitted.
- **Privacy-trained partitions.** These retain their disclosure: SEX entered their assignment search on
  OSF_DEFENSE_FIT historically.

## Staged exposure boundaries

- **Admission** (SOURCE_ADMISSION_LOCK): hash checks, verified copies, teacher forward passes, release re-encoding, the
  frozen-bank tables and the role split. No label is read.
- **Engineering** (ENGINEERING_LOCK): synthetic data and known laws only.
- **First Adult label use:** only after SCIENCE_LOCK is pushed.
- **Assessment:** only `hcal.assess`, after the pushed EVALUATION_LOCK. Its gate is new; the lra gate opens only for
  lra.
- **lra assessment artifacts** (`outer__*`): never admitted or opened before this study's EVALUATION_LOCK.
- **Predecessor custody** commands that unseal the assessment (osf.closeout.backup) run only after this study's opening,
  if at all, and only when the external drive is mounted.

## Cohorts

- No new state, year or population is opened.
- ACS 2016–2018 are spent, and Texas 2018 has a prior reservation. None of them is touched.

The realised use is appended at closeout.

## Realised use (closeout, 2026-10-07)

- **First Adult label read: after SCIENCE_LOCK.** SCIENCE_LOCK was pushed at 13:59Z; calibration started at 13:59:0xZ.
  Labels read under it:
  - task labels of CALIBRATION_HELDOUT and CALIBRATION_TRAIN_MATCHED representatives (calibration);
  - INNER_SELECTION task labels (utility);
  - OSF_DEFENSE_FIT task labels (constant predictor and descriptive fitting-row losses);
  - SEX of ATTACK_FIT_NEW (fresh readers), AUDIT_FIT (control pipeline and frozen-winner refits) and INNER_SELECTION.
- **Not read.** HEAD_VALIDATION labels were never read.
- **Assessment labels.** Read only by `hcal.assess` after EVALUATION_LOCK was pushed at 15:00:31Z, in one opening
  (15:00:43–15:04:30Z). Role E's phase 3 then read them from the hash-checked assessment units.
- **lra assessment artifacts** (`outer__*`): never admitted or opened.
- **No new population, cohort or year was opened.** Predecessor custody commands that unseal the assessment were not
  run (drive absent).
