# Exposure map: autonomous development of a different release mechanism (PCRL)

Which people's labels reach which fitted component. Registered before any new real-label computation. Family-specific rows are
added by each mechanism card, which is committed before the fit it governs. Its label reads are logged with the card's path and
SHA-256. v is the version (0 or 1): training seeds and internal folds of version v, evaluated against U0_S<v> and Ucal_S<v>.

## New fitted components (this campaign)

| Component | Training rows | Labels | Notes |
|---|---|---|---|
| Task heads of a family and of its task-only twin (income, occupation) | all MODEL_FIT (20,000 people, 13,337 households) unless the card states otherwise; household-grouped internal folds for early stopping or cross-fitting | income, occupation | the full pool is used where possible; any internal training size is stated in the release record |
| Protection operation of a family (cells / routing, dependence objective, channel parameters, critics) | all MODEL_FIT, or household-grouped internal folds of it as the card states | SEX (as a training label of the protection objective only), income, occupation | SEX is never a runtime input or output |
| Calibration parameters (if any) | CALIBRATION (2,982 people), or correctly separated internal MODEL_FIT training folds | income, occupation (CALIBRATION has no SEX label) | never MODEL_SELECT |
| Challenger controls (SEX-blind coarsening / quantization, temperature or smoothing paths, post-hoc grouping) | as the control's card states (MODEL_FIT, CALIBRATION) | income, occupation; SEX only for a control labelled SEX-aware | a control is never called SEX-blind if its fitting used SEX |
| Out-of-fold estimates (nested or non-nested, stated) | MODEL_FIT minus the held-out households for every affected upstream component | as the component | submodel costs are never a guarantee for the full-data release |

**MODEL_SELECT labels never reach a fitted release component.** They are read only for:
- utility (`nrm.evaluate`);
- screening readers (`nrm.screen`);
- the grid (`nrm.devreaders`);
- verification.

## Frozen references (no refit)

| Release | Its own training (per its manifests) |
|---|---|
| U0_S<v>, Ucal_S<v> (utility references) | the replication's own training on all MODEL_FIT; Ucal's calibrator on CALIBRATION |
| NEURAL_RAW, H_LJ, T_TREE, TASK_REFIT (frozen bank) | MODEL_FIT (H_LJ's JOINT_TREE occupation and its income correction used MODEL_FIT SEX) |
| B2TWIN, RETIRED_B3CTL (lineage code releases) | MODEL_FIT (+ CALIBRATION per their manifests); SEX-aware |
| SUCC = H3 + JOINT_TREE occupation, TWJ = HC_STRONG + JOINT_TREE occupation | H3: all MODEL_FIT + CALIBRATION, with a JOINT_TREE income correction trained with MODEL_FIT SEX; HC_STRONG: task-only strong income twin |
| W = A1_PO_16 S<v> occupation + H3 income S<v> (previous witness) | A1_PO_16: all MODEL_FIT (occupation only), inner household CV folds 'OHN_A_cv_S<v>' |

## Evaluation roles

| Use | Role | Labels | Notes |
|---|---|---|---|
| Original benchmark, reserve, paired utility costs | MODEL_SELECT (8,001 people, 5,306 households) | income, occupation, SEX | adaptively used by every earlier campaign; development evidence only |
| Screening readers | MODEL_SELECT, NRM1_screen halves | SEX | diagnosis only; adaptive |
| Development reader grid | MODEL_SELECT, fresh splits NRM1_devgrid_R1..R4 (FIT 50% / VALIDATION 20% / SCORE 30%) | SEX | readers see only released probability vectors and decisions |
| Independent verification | MODEL_FIT, CALIBRATION, MODEL_SELECT | income, occupation, SEX | separate code (operation 'verify') |
| ATTACK_FIT, ATTACK_VALIDATION, DEV_REPORT | - | closed | features only (deployment parity) |

## Earlier exposure of the same people (disclosed)

- MODEL_SELECT was used adaptively by the occupation-headroom, H3, correction-transfer and controlled training-size campaigns
  (selection, diagnosis, readers). The grid splits here are new household splits of the same people. Their overlaps with the
  OHN1 and TSZ1 grid splits are recorded in READER_SPLITS.json.
- Every MODEL_FIT person's labels reached earlier fitted components: A1_PO_16, H3, the frozen bank and the lineage releases.
  New fits here are new uses of the same people, not replication.
