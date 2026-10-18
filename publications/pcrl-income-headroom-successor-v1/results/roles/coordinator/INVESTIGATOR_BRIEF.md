# Brief: mechanism investigators (PCRL income-headroom successor development campaign)

Written by the coordinator (main session). Read this whole file before doing anything. Your family-specific assignment is in
the prompt that launched you.

## Research context (keep terminology exact; never rename variables)

Statistical machine-learning research on public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida, in the
existing development roles only. The service predicts an income class (2 classes) and an occupation group (5 classes) from six
raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP; admitted 145-column encoding `X`). Each recipient receives only its own float64
probability vector and argmax decision. SEX is the dataset's recorded demographic column. Readers are separate evaluation
classifiers that estimate SEX from the released probability vectors (each output alone and the linked pair). The aim: a successor
release whose measured demographic recovery is lower while its useful predictions keep meaningful margin below the original
quality caps. Success is possible but not guaranteed; adverse evidence is preserved.

Read: `results/pcrl_income_headroom_successor_v1/PROTOCOL.md`; `REFERENCE_MS_TABLE.csv`; `INHERITED_OCCUPATION_MARGINS.json`;
`results/pcrl_frozen_finalist_validation_v1/RESEARCH_DECISION.md`; `results/pcrl_autonomous_method_development_v1/MECHANISM_CARDS.md`
(what was already tried) and `COMPARISON_TABLE.csv`. Once they exist (the reviewer is writing them now):
`roles/statistical_reviewer/UTILITY_RESERVE_RULE.md` and `NOMINATION_RULE.md`.

What the record says (MODEL_SELECT development values, seed 0 / seed 1, this campaign's grid; pair proxy = maximum over the
development readers including single-view readers, mean over the 2 x 2 grid):
- H_LJ (eligible reference: LOCAL_TREE income + JOINT_TREE occupation): pair 0.6241 / 0.6239; income Brier excess vs Ucal
  -0.00306 / -0.00394. NEURAL_RAW (eligible reference): pair 0.6295 / 0.6357.
- RETIRED_B3CTL (6-level code of 0.5 LOCAL + 0.5 JOINT income log-odds, CALIBRATION per-level probabilities, JOINT_TREE
  occupation): pair 0.6051 / 0.6056 but income Brier excess vs Ucal +0.00437 (SE 0.00158) / +0.00442 (SE 0.00167) against 0.005,
  NLL excess vs Ucal +0.00663 / +0.00663. Retired: no utility reserve.
- B2TWIN (6-level code of LOCAL income only): pair 0.6185 / 0.6173; income Brier +0.00095 / -0.00019 (about 3 SE of margin).
- H_JJ (JOINT_TREE income, continuous): pair 0.5952 / 0.5968; income Brier +0.00571 / +0.00499 (fails or at the cap).
- T_TREE (task-only teacher, both outputs): income Brier vs Ucal -0.00367 / -0.00446; TASK_REFIT -0.00396 / -0.00488.
- Pure occupation-to-income prediction (source B1, income = h(occupation vector)) failed badly: income NLL +0.054.
- The inherited JOINT_TREE occupation output binds on its own: men's within-SEX occupation Brier excess vs U0 +0.00442 (SE
  0.00181) / +0.00394 (SE 0.00170) against 0.005; on DEV_REPORT, occupation NLL excess vs Ucal seed 0 was 0.40 SE from its cap.
  Changing income cannot improve a condition attached to an identical occupation output.

## Infrastructure (use it; do not re-implement it)

- `ihs.common`: `load_role(role, wanted, purpose)` (role-access rules enforced; labels of MODEL_FIT / CALIBRATION / MODEL_SELECT
  are refused until the registration commit exists), `teacher`, `stage_a_outputs`, `stage_b_outputs`, `stage_b_theta`,
  `household_folds`, `write_json`.
- `ihs.bases`: verified base outputs (`base_outputs(seed, arm, roles, label)` for T_TREE / LOCAL_TREE / JOINT_TREE / TASK_REFIT,
  epochs E0, E2, E5, E10, E20), `joint_occupation(seed, roles)` (the frozen accepted occupation output), `stored_reference`
  (U0 / Ucal / T_TREE stored outputs on any role, for internal diagnostics), coarse-code helpers and the logistic map `H`.
- `ihs.devrun.save_release(family, config_id, seed, outs_by_role, record)`: saves a release for every role you compute (compute
  all six roles from features: MODEL_FIT, CALIBRATION, MODEL_SELECT, ATTACK_FIT, ATTACK_VALIDATION, DEV_REPORT; feature arrays
  of every role are label-free and allowed). The record must include `mechanism`, `inputs_used`, `roles_read` (a string naming
  every role and field you read), `labels_used_at_training`, `n_model_fits` (individual model fits behind this configuration,
  including calibrators and critics), `fit_cpu_s`, and enough parameters to rebuild the release from the six raw fields.
- `python3 -m ihs.devrun score --family F --version V --config ID --seed s --counts 1 --kind fit|control|diagnostic --hypothesis
  ... --change ... --training ... --selection ... [--notes ...] [--code ihs/fam_x.py]`: the ONLY way to read MODEL_SELECT labels.
  It computes the utility clauses with household-clustered SEs, the reserve screen (`ihs.reserve`), the 2 x 2 development
  reader grid, comparisons with NEURAL_RAW and H_LJ, and appends the ledger row. Results: `.private/.../runs/<F>/<ID>/S<s>/EVAL_MS.json`.
- `python3 -m ihs.jobqueue run --unit <UNIT_ID> --stage <stage> --rss-gib <GiB> -- <command ...>`: EVERY numerical job (fits,
  scoring, diagnostics on real data) goes through this queue (two slots shared by all agents, one thread per job, CPU and time
  budgets enforced). Unit ids are unique and versioned (`IHS_<F>_<config>_S<s>_V1`; a retry is `_V2`); failed attempts stay in
  the record. Do not run numerical work outside the queue. Use `<repository>/.venv/bin/python` only if you need
  torch; otherwise `python3`.
- Decisions (revise, abandon, stop): `python3 -m ihs.ledger decision --family F --config <FAMILY_F_...> --decision "<text>"`.

## Rules

1. Roles: fit release parameters on MODEL_FIT only. Calibrators only on CALIBRATION (or a household-separated internal split).
   SEX on MODEL_FIT only as a training-time label (critics, subgroup diagnostics), never a runtime input or output. A
   MODEL_FIT person is in-sample for the frozen teacher routing: never describe an internal MODEL_FIT fold as unseen
   calibration or guard data. Never train on ATTACK_FIT, ATTACK_VALIDATION or DEV_REPORT labels (you cannot read them anyway).
   MODEL_SELECT may guide adaptive choices and revisions only through `ihs.devrun score`; never fit parameters on it.
2. Releases are computed from the six permitted raw fields (the admitted encoding `X`) only: no SEX, no labels, no other
   recipient's hidden quantities at runtime. An income output may depend on the released occupation output (that is a released
   quantity); say so in the card.
3. Before your first real fit: write your mechanism card (`roles/investigator_<f>/CARD.md`): the exact operation that changes the
   release; why it could improve the observed trade-off; the measured bottleneck it addresses; the simple control that could
   explain away its benefit; the observation that triggers revision or abandonment; the closest known method. No novelty or
   theorem is required. Check your code on invented data first (no real data) and record the check.
4. Staging: a cheap first-seed (seed 0) fit; evaluate (score); compare with your matched task-only control (run it: it counts
   as a configuration); diagnose the dominant failure; revise with a recorded reason or move on. If income quality is poor
   before your privacy operation, repair or replace the task learner first. If a matched control explains the result,
   investigate that; do not add complexity to keep a method label. Promising configurations then on seed 1.
5. At most an initial version and two justified revisions in your family. Your configuration allotment is in your prompt; every
   seed-specific fitted configuration (including controls) counts. Report individual model-fit counts too.
6. Do not promote a near-cap configuration because its reader AUC looks attractive. The goal is the utility reserve plus lower
   pair recovery than both eligible references, without shifting recovery to an individual recipient (recipient screens), and
   consistent across the four development reader cells.
7. Keep everything private under `.private/`; never print person-level arrays. Do not commit, push or switch branches. Do not
   modify files outside your own module(s) `ihs/fam_<f>*.py` and `results/pcrl_income_headroom_successor_v1/roles/investigator_<f>/`.
   Do not edit shared modules; if you need a change there, ask the coordinator in your report.
8. Keep shell commands simple and separate (the harness refuses compound commands and heredocs); write files with your
   file-writing tool; run everything from the worktree root.
9. Platform blockers: if any tool or classifier blocks work, stop that action, keep your state, and report the exact message.

## Deliverables

- `ihs/fam_<f>.py` (runnable; `fit --seed s --config ID` and anything else you need) with a `predict(seed, config_id, X, names)`
  or equivalent that recomputes the release from features alone, so the coordinator can package a finalist.
- `roles/investigator_<f>/CARD.md` (card first, then each revision appended with time, reason and result).
- `roles/investigator_<f>/REPORT.md`: what you tried, the numbers (full precision from EVAL_MS / the ledger), what each revision
  taught, which effects come from task learning, ordinary compression or calibration, or your proposed component, your best
  configuration(s) on both seeds with its matched control, reserve status, and your recommendation. Finish by the deadline in
  your prompt even if the family failed; a clear negative is a valid result.
