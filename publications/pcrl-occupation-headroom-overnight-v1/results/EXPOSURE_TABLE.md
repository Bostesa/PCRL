# Exposure table (before any new fit; brief section 6)

Which people each frozen teacher, encoder, calibrator, anchor and reference has already seen, and what that means for this
campaign's checks. Public-use ACS 2019 1-Year PUMS person records for Florida, development roles of the protected-tree replication
(MODEL_FIT 20,000 people / 13,337 households; CALIBRATION 2,982 / 1,974; MODEL_SELECT 8,001 / 5,306; ATTACK_FIT 8,001 / 5,319;
ATTACK_VALIDATION 3,000 / 1,974; DEV_REPORT 5,943 / 3,955; the sealed assessment and capped-out records are closed). Sources: the
replication's POPULATION_AND_SCHEMA.md and module docstrings (`acs_replication/trees.py`, `neural.py`, `neural_raw.py`,
`leafcorr.py`), the source lineage's mechanism cards, and `ihs/fam_h.py` / `ihs/fam_o.py`.

| Component | Operation | Labels seen (people) | Label-free exposure | Used here as | Consequence |
|---|---|---|---|---|---|
| 145-column input encoding | z-scored AGEP, WKHP (MODEL_FIT moments); one-hot COW, SCHL, MAR, POBP with levels kept if >= 10 MODEL_FIT people, else a per-variable OTHER column | none | MODEL_FIT inputs (moments, level counts) | input of every release | dictionary learned on authorized training inputs only; identical for every role; any new re-coding must be deterministic from it or learned on MODEL_FIT |
| T_TREE (per task, per seed) | CatBoost depth 4, lr 0.05, <= 800 trees, l2 3; tree count by a 15% household-disjoint internal split, then refit; stored as an exact numpy routing | MODEL_FIT task labels (all 20,000) | MODEL_FIT inputs | equally exposed task-only reference; routing for leaf corrections; possible frozen base (family C) | in-sample on every MODEL_FIT person: a MODEL_FIT fold is never held out for it; out of sample on CALIBRATION, MODEL_SELECT and later roles. No CatBoost is installed, so it cannot be refitted per fold |
| U0 (per task, per seed) | neural 145-64-64-16-K, Adam, <= 30 epochs, epoch count by a 15% household-disjoint internal split, refit on all MODEL_FIT, no calibration | MODEL_FIT task labels (all) | MODEL_FIT inputs | utility reference (caps); possible frozen base | in-sample on MODEL_FIT (its MODEL_FIT outputs favour it in cross-fitted comparisons); out of sample elsewhere |
| Ucal | U0 with one temperature per task fitted on CALIBRATION | MODEL_FIT (U0), CALIBRATION task labels (temperature) | as U0 | utility reference (caps) | CALIBRATION is not held out for Ucal's temperature |
| TASK_REFIT | task-only leaf-score corrections on the T_TREE routing (replication Stage B, E20) | MODEL_FIT task labels | MODEL_FIT inputs | equally exposed task-only reference; existing residual learner | in-sample on MODEL_FIT |
| LOCAL_TREE / JOINT_TREE (Stage B, E20) | leaf-score corrections on the T_TREE routing trained against online SEX critics (LOCAL: each wire; JOINT: also the linked pair) | MODEL_FIT task labels and SEX | MODEL_FIT inputs | JOINT_TREE occupation = the inherited occupation output (H_LJ, the source successor); LOCAL / JOINT income corrections are transplanted inside H3 | in-sample on MODEL_FIT; SEX entered only at training time on MODEL_FIT |
| NEURAL_RAW | U0 start, critic-trained neural releases (replication; same critic schedule as the tree arms) | MODEL_FIT task labels and SEX | MODEL_FIT inputs | eligible neural reference | in-sample on MODEL_FIT |
| H_LJ, H_JJ, H_LL | frozen hybrids of LOCAL / JOINT income and occupation outputs | as LOCAL / JOINT | as above | eligible tree reference (H_LJ) and descriptive references | as above |
| B2TWIN, RETIRED_B3CTL | 6-level decision-aligned income codes of LOCAL (B2) or 0.5 LOCAL + 0.5 JOINT (B3) income log-odds with the JOINT_TREE occupation output; code choice screened on CALIBRATION | MODEL_FIT (corrections, SEX), CALIBRATION (screen) | MODEL_FIT inputs | code references (descriptive / per the frozen rule) | as above |
| H3_STRONG_LJDIR_PB income (anchor H3) | HistGradientBoosting income learner (household 5-fold CV on MODEL_FIT, refit on all MODEL_FIT), 3-coefficient stack with T_TREE income on CALIBRATION, budgeted transplant of LOCAL + beta * JOINT income corrections (beta by a CALIBRATION budget), 2-parameter Platt map on CALIBRATION | MODEL_FIT income labels; CALIBRATION income labels; MODEL_FIT SEX through the transplanted corrections | MODEL_FIT inputs | frozen income anchor | in-sample on MODEL_FIT and CALIBRATION; selected adaptively on MODEL_SELECT by the source campaign; evaluated once on DEV_REPORT there |
| HC_STRONG income (anchor TW) | the same learner and stack without the transplant | MODEL_FIT, CALIBRATION income labels | MODEL_FIT inputs | frozen task-only income twin | as above, no SEX |
| O1_SEXBAL (source family O) | JOINT_TREE occupation, temperature by NLL on CALIBRATION, per-class bias on MODEL_FIT equalizing the two SEX groups' Brier excess vs U0 | CALIBRATION occupation labels; MODEL_FIT occupation labels and SEX | MODEL_FIT inputs | control / starting component (no DEV_REPORT result) | MODEL_SELECT margins were seen by the source campaign |

Roles as data, beyond model fitting:
- MODEL_SELECT: used for adaptive selection by every earlier campaign of the lineage and by the source campaign's development
  reader grid; here it is again an adaptively used development role (utility selection and the FIT / VALIDATION / SCORE reader grid).
  No new result on it is fresh.
- ATTACK_FIT / ATTACK_VALIDATION: fitted and selected the final readers of the validation and the source campaign; here only for a
  promoted finalist's final readers.
- DEV_REPORT: read 14 times in the lineage (SOURCE_ADMISSION.json); any new evaluation is exploratory.

Cross-fitting consequences (brief section 6). A pipeline that uses T_TREE, U0, TASK_REFIT or the Stage B corrections as an input,
offset or stack feature is NOT fully out of sample on MODEL_FIT, because those layers saw every MODEL_FIT household. Families
record such layers in OCC.json, and the scorer flags the comparison of cross-fitted MODEL_FIT predictions with U0 / T_TREE (both
in-sample on MODEL_FIT). Full cross-fitting claims require every label-using layer to be refitted per household fold
(`ohn.common.crossfit_folds`, 5 folds, tag OHN1_crossfit).
