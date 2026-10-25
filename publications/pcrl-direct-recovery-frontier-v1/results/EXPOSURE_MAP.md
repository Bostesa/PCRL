# Exposure map (campaign `pcrl_direct_recovery_frontier_v1`; registered before any label read)

Which people's labels reach which component. Roles: MODEL_FIT (20,000 people, 13,337 households), MODEL_SELECT (8,001 people, 5,306
households). CALIBRATION is not read by this campaign. ATTACK_FIT, ATTACK_VALIDATION and DEV_REPORT labels, the sealed assessment and the
capped-out records are closed (features of the admitted roles are used label-free to compute each release for deployment parity). No
records are downloaded. Inherited rows follow `results/pcrl_strong_anchor_release_v1/EXPOSURE_MAP.md`.

## Inherited components (read-only; no new label read to use them)

| Component | Labels | People |
|---|---|---|
| NRM1 task teachers qI, qO (A1_PO_16 HGB settings; `nrm` fa_cache TASK.pkl: full-pool models and 5-fold household out-of-fold posteriors) | MODEL_FIT income, occupation (all 20,000; inner household CV for T*) | frozen |
| FRC1 P1 propensity teacher pi (`frc` fa_cache SEXSIDE.pkl: full-pool model and out-of-fold pi) | MODEL_FIT SEX (all 20,000; inner household CV for T*) | frozen |
| FRC1 twin TWB_C64 (16 x 64 KL k-means book on the out-of-fold posteriors; B1 tables) | label-free given the teachers (the book and tables use the teachers' out-of-fold posteriors) | frozen; the family's identity |
| The nine references and U0 / Ucal | as their own records (SAR1 exposure map) | frozen; never refitted |
| FRC1 MODEL_SELECT utility ledger of TWB_C64 (published aggregates) | MODEL_SELECT income, occupation, SEX (FRC1's reads) | read as saved aggregates to set the optimizer budgets (adaptive, disclosed) |

## This campaign

| Stage | Labels read | People | Reaches |
|---|---|---|---|
| Design, rules, finite-law verification, tiny exact instances | none (invented data, committed records) | - | code, rules, card |
| Family DR fits (card DR_DRF_v1; `drf.dr fit` / `optimistic`) | MODEL_FIT income, occupation, SEX | all 20,000 | the lookup tables A_inc / A_occ (objective: estimated law pi for DR_AUC / DR_MI, true SEX for OPT_L / OPT_S; constraints: true task labels and SEX); B1 tables use the teachers' out-of-fold posteriors only; nested estimator: outer-training households fit, held-out households evaluate |
| CTRL coarsening path | MODEL_FIT income, occupation (no SEX in the path; SEX only in its spend records' guard rows) | all 20,000 | the merge order (task-only) |
| Utility evaluation (`drf.evaluate`) | MODEL_SELECT income, occupation, SEX | all 8,001 | aggregate utility records only |
| Reader grid (`drf.devreaders`) | MODEL_SELECT SEX | FIT / VALIDATION / SCORE subroles | reader fits, selection and SCORE AUCs only |
| Verification (`roles/verifier/code/`) | MODEL_FIT and MODEL_SELECT labels as needed | as the checks need | verification records only |

MODEL_SELECT never reaches a task learner, calibration map, release parameter, output support or mapping rule. The released vectors
of every role are computed from that role's features alone.
