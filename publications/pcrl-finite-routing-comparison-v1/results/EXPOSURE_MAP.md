# Exposure map (campaign `pcrl_finite_routing_comparison_v1`; registered before any label read)

Which people's labels reach which component. Roles: MODEL_FIT (20,000 people, 13,337 households), CALIBRATION (authorized, not
planned), MODEL_SELECT (8,001 people, 5,306 households). ATTACK_FIT, ATTACK_VALIDATION and DEV_REPORT labels are closed (DEV_REPORT
features only, label-free, for deployment parity). The sealed assessment and capped-out records are closed.

## Inherited components (source campaign NRM1; read-only; no new label read)

| Component | Labels | People |
|---|---|---|
| Task teachers qI, qO (HGB, A1_PO_16 settings; inner household CV T*, full refit, 5-fold household out-of-fold predictions) | MODEL_FIT income, occupation | all 20,000 MODEL_FIT people (NRM1 stage 1) |
| Saved twin routing books PRC_TWIN (16 x 64), PRC_TWIN_C128 (24 x 128) | none (fitted on the teachers' out-of-fold predictions) | 20,000 |
| Saved label-refit decoders (construction A) | MODEL_FIT income, occupation | 20,000 (out-of-fold-routed) |
| References (NEURAL_RAW, H_LJ, B2TWIN, RETIRED_B3CTL, SUCC, TWJ, W, TR100, TT100, U0, Ucal) | as their own records | frozen; never refitted |

## This campaign

| Stage | Labels read | People | Reaches |
|---|---|---|---|
| T0 reproduction | none | - | - |
| T1 nested comparison | MODEL_FIT income, occupation | 20,000 (each outer-training part about 16,000; held-out about 4,000) | refitted teachers, books and A tables inside each outer fold; held-out losses only (never a deployed parameter) |
| T2 B1 twins | none | - | B1 tables = means of the inherited teachers' out-of-fold predictions |
| T5 prospective capacity | none for B1; MODEL_FIT income, occupation for A | 20,000 | the (32, 256) book (label-free) and its tables |
| P1 paths | MODEL_FIT income, occupation, SEX (training label) | 20,000 | SEX propensity pi and SEX-conditional occupation teacher (full refit and out-of-fold); the nested selection (held-out losses and held-out LOOKUP; the selection rule's subgroup rows use SEX as an evaluation of candidates) |
| P2 configurations | none for B1 (MODEL_FIT income, occupation for A) | 20,000 | protected books and cell SEX-rate tables through the P1 out-of-fold pi; ORD receives no SEX |
| P3 check | MODEL_FIT SEX (evaluation) | 20,000 | nothing deployed |
| E1 utility | MODEL_SELECT income, occupation, SEX (evaluation) | 8,001 | nothing deployed |
| E2 / E3 reader grid | MODEL_SELECT SEX | reader FIT about 3,980, VALIDATION about 1,620, SCORE about 2,400 per repetition | development readers only |
| V verification | MODEL_FIT, CALIBRATION, MODEL_SELECT (verify) | as replayed | nothing deployed |

- MODEL_SELECT households never enter the fitting of any release parameter, codebook, table or selection rule; MODEL_SELECT is used
  for the original-benchmark evaluation, the qualification rule and the development readers (adaptive development evidence).
- Training size of every deployed release: 20,000 MODEL_FIT people (full authorized exposure). The nested estimator's 16,000-person
  refits are internal diagnostics, never deployed.
- Runtime prediction reads only AGEP, WKHP, COW, SCHL, MAR, POBP.
