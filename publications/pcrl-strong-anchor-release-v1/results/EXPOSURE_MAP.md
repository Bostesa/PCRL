# Exposure map (campaign `pcrl_strong_anchor_release_v1`; registered before any label read)

Which people's labels reach which component. Roles: MODEL_FIT (20,000 people, 13,337 households), CALIBRATION (2,982 people;
income / occupation labels only if a card says so, never SEX), MODEL_SELECT (8,001 people, 5,306 households). ATTACK_FIT,
ATTACK_VALIDATION and DEV_REPORT labels are closed (DEV_REPORT features only, label-free, for deployment parity). The sealed
assessment and capped-out records are closed. No records are downloaded. Detailed component rows for the inherited releases are in
`results/pcrl_correction_transfer_stability_v1/EXPOSURE_MAP.md` section 1 (code-verified).

## Inherited components (read-only; no new label read to use them)

| Component | Labels | People |
|---|---|---|
| SUCC = H3_STRONG_LJDIR_PB income + JOINT_TREE E20 occupation (anchor A) | JOINT_TREE: MODEL_FIT income, occupation and SEX (all 20,000, in-sample; linked-pair critic); H3: MODEL_FIT income (HGB, T_TREE) and, through the transplanted D_L / D_J corrections, MODEL_FIT occupation and SEX; CALIBRATION income (stack, beta budget, Platt; all 2,982); MODEL_SELECT aggregates set H3's budget B1 and nominated H3 (adaptive) | frozen |
| NEURAL_RAW rho 1 E2 (anchor B) | MODEL_FIT income, occupation and SEX (all 20,000, in-sample, from U0); rho / checkpoint chosen on MODEL_SELECT by the replication's rule | frozen |
| U0, Ucal (utility references) | U0: MODEL_FIT task labels; Ucal: plus CALIBRATION task labels (temperatures) | frozen |
| T_TREE, TASK_REFIT, LOCAL_TREE, JOINT_TREE (all checkpoints) | MODEL_FIT task labels (T_TREE, TASK_REFIT), plus SEX (LOCAL, JOINT) | frozen; usable as teachers or initializations only as a card says |
| References NEURAL_RAW, H_LJ, B2TWIN, RETIRED_B3CTL, SUCC, TWJ, W, TR100, TT100; beside DDR_PTC_R1 | as their own records | frozen; never refitted |
| The input-and-pair campaign's IPR1_MAIN grid and MODEL_SELECT utility records | MODEL_SELECT SEX, income, occupation (that campaign's reads) | read as saved aggregates by the label-free anchor table only |

## This campaign

| Stage | Labels read | People | Reaches |
|---|---|---|---|
| Design (investigators, reviewer) | none (label-free features, frozen release arrays, committed records, invented data) | - | the proposals and cards |
| Family fits (cards FA_*, FB_*, CTL_*, FS_*) | MODEL_FIT income, occupation and, where the card says so, SEX (training label of a protection objective or a within-SEX utility constraint); CALIBRATION income / occupation only if a card says so | 20,000 (CALIBRATION as recorded) | task models, release parameters, strengths and control points named by each card; strength and control-point selection on MODEL_FIT out-of-fold or nested held-out losses only; calibration maps on separated MODEL_FIT household folds |
| MODEL_FIT checks named by a card | MODEL_FIT SEX (evaluation) | 20,000 | nothing deployed |
| Utility | MODEL_SELECT income, occupation, SEX (evaluation, `benchmark`) | 8,001 | nothing deployed |
| Reader grid SAR1 | MODEL_SELECT SEX (`readers`) | per repetition reader FIT about 4,000, VALIDATION about 1,600, SCORE about 2,400 | development readers only |
| Verification | MODEL_FIT, CALIBRATION, MODEL_SELECT (`verify`) | as replayed | nothing deployed |

- MODEL_SELECT households never enter the fitting of any task learner, calibration map, release parameter, strength, control point
  or selection rule. MODEL_SELECT is used for the original-benchmark utility, the development readers on their frozen subroles and the
  decisions (adaptive development evidence: the same 8,001 people were used by every earlier campaign; new household splits do not
  reverse that exposure).
- A fold used to train a teacher is not a held-out check of that teacher: cross-fitted quantities name their folds, and in-sample
  inherited components (SUCC, NEURAL_RAW, JOINT_TREE, H3) are never treated as held out on MODEL_FIT.
- Training size of every deployed release: the 20,000 MODEL_FIT people (CALIBRATION in addition only where a card says so, for task
  labels only, and then for its controls and continuation as well, so training exposure stays matched).
- Runtime prediction reads only AGEP, WKHP, COW, SCHL, MAR, POBP.
