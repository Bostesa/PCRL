# Exposure map (campaign `pcrl_reader_fit_aware_release_v1`; registered before any label read)

Which people's labels reach which component. Roles: MODEL_FIT (20,000 people, 13,337 households), MODEL_SELECT (8,001 people, 5,306
households). CALIBRATION labels are closed for this campaign (no 'calibrate' operation). ATTACK_FIT, ATTACK_VALIDATION and DEV_REPORT
labels, the sealed assessment and the capped-out records are closed; the admitted roles' features are used label-free to compute each
release for deployment parity. No records are downloaded. Inherited rows follow `results/pcrl_strong_anchor_release_v1/EXPOSURE_MAP.md`
and `results/pcrl_direct_recovery_frontier_v1/EXPOSURE_MAP.md`.

## Inherited components (read-only; no new label read to use them)

| Component | Labels | People |
|---|---|---|
| U0_S<v> (Stage A task networks, Recipe A) | MODEL_FIT income, occupation (all 20,000) | frozen; the fold starts refit its recipe on fold training rows |
| NEURAL_RAW_S<v> (U0 continued with RAW-J online critics, rho 1, E2) | MODEL_FIT income, occupation, SEX (all 20,000); rho / checkpoint chosen on MODEL_SELECT by the replication's rule (post-selection) | frozen START of every arm; its income network is the released income of every occupation-only arm |
| The nine references and U0 / Ucal | as their own records (SAR1 exposure map) | frozen; never refitted |

## This campaign

| Stage | Labels read | People | Reaches |
|---|---|---|---|
| Design, rules, derivative checks, synthetic examples | none (invented data, committed records) | - | code, rules, cards |
| Family FAM fits (`rfa.fam.fit fit`, committed RF_ card) | MODEL_FIT income, occupation, SEX | all 20,000 | the occupation network of every arm. SEX enters only through the episode objectives (FULL, STOPFIT, ALTCRIT) and RAW-J, and the within-SEX guard rows of the strength rule; the TASK arm reads no SEX. Every auxiliary-training (A) and auxiliary-evaluation (E) episode row is MODEL_FIT training data of the release. Cross-fitting: the fold's training households fit, its held-out households give the out-of-fold spends and the descriptive MODEL_FIT reader check |
| Utility evaluation (`rfa.evaluate`) | MODEL_SELECT income, occupation, SEX | all 8,001 | aggregate utility records only |
| Reader grid (`rfa.devreaders`) | MODEL_SELECT SEX | FIT / VALIDATION / SCORE subroles | fresh reader fits, selection and SCORE AUCs only |
| Verification (`roles/verifier/code/`) | MODEL_FIT and MODEL_SELECT labels as needed | as the checks need | verification records only |

MODEL_SELECT never reaches a task learner, calibration map, release parameter, inner reader, episode, strength or operating choice.
The released vectors of every role are computed from that role's six permitted fields alone.
