# Exposure map (campaign `pcrl_dual_design_release_v1`; registered before any label read)

Which people's labels reach which component. Roles: MODEL_FIT (20,000 people, 13,337 households), CALIBRATION (income /
occupation labels authorized, never SEX), MODEL_SELECT (8,001 people, 5,306 households). ATTACK_FIT, ATTACK_VALIDATION and
DEV_REPORT labels are closed (DEV_REPORT features only, label-free, for deployment parity). The sealed assessment and capped-out
records are closed. No records are downloaded.

## Inherited components (read-only; no new label read to use them)

| Component | Labels | People |
|---|---|---|
| Task teachers qI, qO (HGB, A1_PO_16 settings; inner household CV T*, full refit, 5-fold household out-of-fold predictions; `TASK.pkl` of the new-release-mechanisms campaign) | MODEL_FIT income, occupation | all 20,000 MODEL_FIT people |
| SEX-side teachers pi = P(SEX = 1 \| x) and P(occupation \| x, SEX) (`SEXSIDE.pkl` of the finite-routing comparison) | MODEL_FIT SEX, occupation | all 20,000 MODEL_FIT people |
| B1 finite twin TWB_C64 and the PRC / control releases of the finite-routing comparison | as their own records | frozen; never refitted |
| References (NEURAL_RAW, H_LJ, B2TWIN, RETIRED_B3CTL, SUCC, TWJ, W, TR100, TT100, U0, Ucal) | as their own records | frozen; never refitted |

## This campaign

| Stage | Labels read | People | Reaches |
|---|---|---|---|
| Design (designers, challenger) | none (cached MODEL_FIT out-of-fold predictions only) | - | the proposals' feasibility numbers |
| Family fits (cards DA_*, DB_*, CTL_*, FS_*) | MODEL_FIT income, occupation and, where the card says so, SEX (training label); CALIBRATION income / occupation only if a card says so | 20,000 (CALIBRATION as recorded) | task models, SEX-side statistics and release parameters named by each card; strength selection on MODEL_FIT out-of-fold / nested held-out losses only |
| MODEL_FIT checks named by a card | MODEL_FIT SEX (evaluation) | 20,000 | nothing deployed |
| Utility | MODEL_SELECT income, occupation, SEX (evaluation, `benchmark`) | 8,001 | nothing deployed |
| Reader grid DDR1 | MODEL_SELECT SEX (`readers`) | per repetition reader FIT about 4,000, VALIDATION about 1,570, SCORE about 2,410 | development readers only |
| Verification | MODEL_FIT, CALIBRATION, MODEL_SELECT (`verify`) | as replayed | nothing deployed |

- MODEL_SELECT households never enter the fitting of any release parameter, strength, table or selection rule. MODEL_SELECT is used
  for the original-benchmark utility, the development readers and the decisions (adaptive development evidence).
- Training size of every deployed release: the 20,000 MODEL_FIT people (CALIBRATION in addition only where a card says so, for
  task labels only, and then for its twin and controls as well, so training exposure stays matched).
- Runtime prediction reads only AGEP, WKHP, COW, SCHL, MAR, POBP.
