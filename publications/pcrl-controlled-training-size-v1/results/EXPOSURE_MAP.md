# Exposure map: controlled training-size test with one conditional protection experiment (PCRL)

Which people's labels reach which fitted component. Registered before any new real-label computation. "Subset" means a registered
MODEL_FIT subset (SUBSETS.json): S50_O<o>, S70_O<o> (four orderings, nested) or S100 (all 20,000 MODEL_FIT people). v is the source
version (0 or 1). MODEL_SELECT labels never reach a fitted component of this campaign.

## Size stage

| Component | Training rows | Labels | Other inputs | Selection inside the component |
|---|---|---|---|---|
| FIT-only dictionary (COW / SCHL / MAR / POBP levels with < 10 subset people -> __OTHER__) | subset | none (label-free) | admitted features | none |
| Admitted dictionary (encoding sensitivity only; S50, v = 0) | all MODEL_FIT features | none (label-free) | admitted features | none |
| A1_PO_16 inner CV (T*) | subset, household folds 'OHN_A_cv_S<v>' restricted to the subset | occupation | none | T* = first argmin of the pooled inner-CV NLL |
| A1_PO_16 deployed model | subset | occupation | T* | none |
| A1_PO_16 diagnostic temperature (recorded, never applied) | CALIBRATION (2,982 people) | occupation | the deployed model's raw scores | none |
| T_TREE income and occupation teachers (control, stage 1) | subset (85% internal fit / 15% internal validation by the source's household split of seed v, then refit on the subset) | income, occupation | the subset's FIT-only dictionary | tree count = first argmin on the internal 15% |
| TASK_REFIT (control, stage 2) | subset | income, occupation (task loss); SEX reaches the source loop's warm critics, whose term has weight 0 for TASK_REFIT | the subset's T_TREE routing | none (fixed 20 x 40 x 256 updates) |
| H3 income (frozen, not refitted) | the H3 source's own training (all MODEL_FIT + CALIBRATION, per its manifests) | - | - | - |
| U0, Ucal, references (frozen, not refitted) | the replication's own training (all MODEL_FIT; Ucal's calibrator on CALIBRATION) | - | - | - |

At S100 every subset-specific component is identical for the four orderings (one fit per version, aliased).

## Conditional stage (runs only if the frozen trigger fires)

| Component | Training rows | Labels | Notes |
|---|---|---|---|
| Witness W100_v | S100 | occupation | the size stage's S100 fit of version v (no refit) |
| Correction base: W100_v's 5-fold household cross-fitting ('OHN1_crossfit') | for fold k: S100 minus fold k (inner CV restricted) | occupation | out-of-fold logits on S100 = the correction's training base |
| Routing (T_TREE occupation teacher, seed v) | S100 | income (class-support repair of the internal split), occupation | the control pipeline's S100 teacher |
| CONT / LOC / PAIR corrections (one full pipeline each) | S100 | occupation (task loss); SEX (critics; CONT's critics are updated but unused) | PAIR's critic also reads the frozen H3 income wire S<v> on MODEL_FIT |
| Nested strength-selection estimator, outer fold k of 'TSZ1_ncv10' (10 folds) | S100 minus fold k (about 18,000 people) for EVERY upstream layer: dictionary, witness, its 5-fold base, routing, LOC / PAIR corrections | occupation, income (routing support repair), SEX (critics) | held-out costs on fold k use fold k's occupation and SEX labels only as evaluation of submodels that never saw fold k; H3 income is the frozen service anchor (fitted on all MODEL_FIT; it enters only PAIR's training critic, never the held-out costs) |
| Strength selection | none (applies the frozen rule to the pooled nested costs) | - | MODEL_SELECT never enters |

## Evaluation roles

| Use | Role | Labels | Notes |
|---|---|---|---|
| Original benchmark, reserve, size contrasts, trigger | MODEL_SELECT (8,001 people, 5,306 households) | income, occupation, SEX | adaptively used by every earlier campaign; development evidence only |
| Development readers | MODEL_SELECT, fresh household splits (READER_SPLITS.json): FIT 50% / VALIDATION 20% / SCORE 30% per repetition | SEX | readers see only released probability vectors and decisions |
| Independent verification | MODEL_FIT, CALIBRATION, MODEL_SELECT | income, occupation, SEX | separate code (operation 'verify') |
| ATTACK_FIT, ATTACK_VALIDATION, DEV_REPORT | - | closed | features only (deployment parity) |

## Earlier exposure of the same people (disclosed)

- MODEL_SELECT was used adaptively by the occupation-headroom, H3 and correction-transfer campaigns (selection, diagnosis, readers).
- The source A1_PO_16 S0 / S1 packages were fitted on all MODEL_FIT with the same recipe; the S100 fits here are refits of that
  recipe on the same people (reproduction, not replication).
- The correction-transfer campaign fitted the A1_PO_16 recipe on 42-70% of MODEL_FIT (partitions D0, D1, R0, R1, P70, P70A) and
  read their MODEL_SELECT outcomes; its training-size diagnostic is the adaptive motivation of this study.
