# Exposure map (campaign `pcrl_input_and_pair_release_v1`; registered before any label read)

Which people's labels reach which component. Roles: MODEL_FIT (20,000 people, 13,337 households), CALIBRATION (2,982 people;
income / occupation labels authorized, never SEX), MODEL_SELECT (8,001 people, 5,306 households). ATTACK_FIT, ATTACK_VALIDATION and
DEV_REPORT labels are closed (DEV_REPORT features only, label-free, for deployment parity). The sealed assessment and capped-out
records are closed. No records are downloaded.

## Inherited components (read-only; no new label read to use them)

| Component | Labels | People |
|---|---|---|
| Task teachers qI, qO (HGB, A1_PO_16 settings; inner household CV T*, full refit, 5-fold household out-of-fold predictions; `TASK.pkl` of the new-release-mechanisms campaign) | MODEL_FIT income, occupation | all 20,000 MODEL_FIT people |
| SEX-side teachers pi = P(SEX = 1 \| x) and P(occupation \| x, SEX) (`SEXSIDE.pkl` of the finite-routing comparison) | MODEL_FIT SEX, occupation | all 20,000 MODEL_FIT people |
| The dual-design release campaign's saved releases (its shared twin DA_TWIN, its strongest release DB_PTC_R1 and its release-only control DB_ZTILT_R1, read as inherited references) | as their own records (MODEL_FIT income, occupation, SEX) | frozen; never refitted |
| References (NEURAL_RAW, H_LJ, B2TWIN, RETIRED_B3CTL, SUCC, TWJ, W, TR100, TT100, U0, Ucal) | as their own records | frozen; never refitted |

## This campaign

| Stage | Labels read | People | Reaches |
|---|---|---|---|
| Design (designers, challenger) | none (cached MODEL_FIT out-of-fold predictions and label-free features only) | - | the proposals' feasibility numbers |
| Family fits (cards IA_*, IB_*, CTL_*, FS_*) | MODEL_FIT income, occupation and, where the card says so, SEX (training label); CALIBRATION income / occupation only if a card says so | 20,000 (CALIBRATION as recorded) | task models, SEX-side statistics and release parameters named by each card; strength and control-point selection on MODEL_FIT out-of-fold / nested held-out losses only |
| MODEL_FIT checks named by a card | MODEL_FIT SEX (evaluation) | 20,000 | nothing deployed |
| Utility | MODEL_SELECT income, occupation, SEX (evaluation, `benchmark`) | 8,001 | nothing deployed |
| Reader grid IPR1 | MODEL_SELECT SEX (`readers`) | per repetition reader FIT about 4,000, VALIDATION about 1,570, SCORE about 2,410 | development readers only |
| Data-rich reader check | MODEL_FIT SEX (training of the readers, inside the family fit units) and MODEL_SELECT SEX (scoring, `readers`) | 20,000 / 8,001 | development readers only (descriptive) |
| Verification | MODEL_FIT, CALIBRATION, MODEL_SELECT (`verify`) | as replayed | nothing deployed |

- MODEL_SELECT households never enter the fitting of any release parameter, strength, table or selection rule. MODEL_SELECT is used
  for the original-benchmark utility, the development readers and the decisions (adaptive development evidence: the same 8,001
  people were used by every earlier campaign).
- Training size of every deployed release: the 20,000 MODEL_FIT people (CALIBRATION in addition only where a card says so, for
  task labels only, and then for its twin and controls as well, so training exposure stays matched).
- Runtime prediction reads only AGEP, WKHP, COW, SCHL, MAR, POBP.
