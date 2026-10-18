# Common brief for every agent of the overnight occupation-headroom campaign (coordinator, 2026-10-18)

Worktree (run every command from here): `<repository>/.claude/worktrees/pcrl-occupation-headroom-overnight-v1`,
branch `research/pcrl-occupation-headroom-overnight-v1`. Campaign start 04:36:11Z; research window ends 12:36Z; closeout to 13:06Z.

## What this is

Statistical machine-learning research on already admitted public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for
Florida (housing-unit residents aged 25-64, civilian employed), development roles only. The service predicts an income class (2:
adjusted personal income above $50,000) and an occupation group (5 official broad civilian groups: 0 management/business/
science/arts, 1 service, 2 sales/office, 3 natural resources/construction/maintenance, 4 production/transportation/material moving)
from six raw fields AGEP, WKHP, COW, SCHL, MAR, POBP (admitted 145-column encoding `X`; its category dictionary and moments were
learned label-free on MODEL_FIT). Each of two recipients receives only its own float64 probability vector and argmax decision.
SEX is the dataset's recorded demographic column (sex0 = men, SEX code 1; sex1 = women, SEX code 2). Separate evaluation
classifiers ("readers") estimate SEX from the released probability vectors: each output alone and the two outputs aligned to the
same person (the linked pair). SEX is never a runtime input or an output; it may be a training-time label on MODEL_FIT only.
Use this terminology accurately. Results are exploratory development evidence on previously used records, never confirmation.

Scientific question: can a stronger occupation predictor create enough task-quality margin (below the ORIGINAL caps) to support
a useful reduction in SEX recovery from both released outputs, while keeping the income component of the source campaign
(`H3_STRONG_LJDIR_PB`, "H3")? Better occupation utility is not evidence that recovery falls; combining two useful components is
not assumed to add their benefits. Both are tested.

Source record (read as needed): `results/pcrl_income_headroom_successor_v1/` (START_HERE.md, RESEARCH_DECISION.md,
MECHANISM_CARDS.md, roles/investigator_{h,p,o}/CARD.md and REPORT.md, REFERENCE_MS_TABLE.csv, INHERITED_OCCUPATION_MARGINS.json,
EXPERIMENT_LEDGER.csv). Source code: `ihs/` (do not edit), `acs_replication/trees.py` (teacher), `acs_replication/neural.py` (U0).

## Hard rules

- Data roles (enforced by `ohn.common.load_role`): MODEL_FIT (20,000 people / 13,337 households: features, income, occupation,
  SEX) for fitting; CALIBRATION (2,982 / 1,974: features, income, occupation; no SEX) for calibrators only (or household-separated
  internal folds of MODEL_FIT); MODEL_SELECT (8,001 / 5,306) labels and SEX ONLY through `python3 -m ohn.devrun score` and
  `ohn.devreaders` (logged development selection and diagnosis; never a gradient-fitting or calibration set); ATTACK_FIT,
  ATTACK_VALIDATION and DEV_REPORT labels: never. You may compute outputs on every role from features (label-free).
- NEVER call `ihs.common.load_role`, `ihs.evaluate.*`, `ihs.devrun.*` or any other `ihs`/`ffv`/`amd` loader for labels: they
  bypass this campaign's gates and logs. Label-free `ihs.bases` functions (teacher routing, stored corrections, `stored_reference`
  for U0 / Ucal / T_TREE outputs) are fine. All label reads go through `ohn.common.load_role(role, wanted, purpose=...)`, which
  refuses label reads until the registration is committed and its public receipt is verified (about 05:45Z). Before that, work on
  invented data and label-free real features only.
- Releases are computed from the six permitted fields only (the 145-column `X`, or a deterministic re-coding of `X` / the raw
  fields). Never feed SEX, labels, identifiers, household ids or another recipient's hidden quantities into a runtime output.
- No new population, download, sealed record or capped-out record; no paid or cloud compute; no change to any cap or benchmark
  definition. Do not edit `ihs/`, `ffv/`, `amd/`, `acs_replication/`, `scm_bench/`, locked files or anything under
  `results/pcrl_income_headroom_successor_v1/`.
- Compute: every numerical job (fits, evaluations, tests on real data, anything over about 2 CPU-seconds) runs through the ONE
  shared queue, which allows two heavy jobs at once with one thread each:
  `python3 -m ohn.jobqueue run --unit <UNIQUE_VERSIONED_ID> --stage <fit|score|selftest|diagnostic|readers|verification> --rss-gib <GiB> -- <command>`
  (unit ids are append-only: a retry is `..._V2`; failed attempts stay recorded). Use `--rss-gib 2` unless you need more (limit
  8 GiB total). Light out-of-queue work (reading JSON, py_compile) is fine; charge an honest upper estimate at the end:
  `python3 -m ohn.jobqueue charge --label <ROLE>_AGENT --cpu-s <s> --note "<what>"`. Total campaign budget 14 CPU-hours;
  be economical and never repeat an equivalent fit to fill time.
- Interpreters: `python3` = Homebrew Python 3.14.3 (numpy 2.4.4, scikit-learn 1.8.0, scipy 1.17.1; no torch / xgboost);
  `<repository>/.venv/bin/python` = Python 3.13.7 (numpy 2.4.2, scikit-learn 1.8.0, torch 2.10.0, xgboost 3.2.0).
  No CatBoost or LightGBM exists. Every release records its interpreter (save_occupation does this).
- Platform: follow tool permissions. If an action is refused, stop that action, keep the state, and report the exact refusal
  message. The platform has refused SUBAGENT REPORT FILES before ("Subagents should return findings as text, not write report
  files. Include this content in your final response instead."): write code, mechanism cards and machine records as asked, but
  return your report as the TEXT of your final response; the coordinator saves it with a disclosure.
- Never commit, push, merge, rebase or switch branches (the coordinator owns git). Never kill processes you did not start.

## Interfaces (package `ohn/`)

- `ohn.common`: `load_role`, `crossfit_folds(unit)` (household 5-fold split of MODEL_FIT, tag OHN1_crossfit), `household_folds`,
  `teacher(seed, task)`, `stage_a_outputs(seed)` (stored U0 / Ucal / T_TREE outputs on every role), `SEEDS = (0, 1)`.
- `ohn.devrun.save_occupation(family, config_id, seed, occ_by_role, record, crossfit=None)`: writes the occupation release
  (all six roles, (n, 5) float64 probability vectors in admitted row order) and its record; refuses overwrites (revisions get
  new ids). `record` must contain `kind` ('task_model' | 'privacy' | 'control' | 'income_branch' | 'stability'),
  `operation`, `roles_read`, `exposure`, `n_model_fits`, `n_calibrator_fits`, `n_crossfit_fits`, `fit_cpu_s` (and anything
  else useful). `crossfit` = {'MODEL_FIT__oof': (20000, 5), 'MODEL_SELECT__foldavg': (8001, 5), 'fold': (20000,)}.
- `ohn.devrun.source_occupation(name, seed, role)` and `anchor_income(anchor, seed, role)` (anchors H3, TW = HC_STRONG).
- Scoring (coordinator's evaluator; reads MODEL_SELECT labels, logged):
  `python3 -m ohn.jobqueue run --unit <ID> --stage score --rss-gib 2 -- python3 -m ohn.devrun score --family F --config ID --seed s --anchor H3 --hypothesis "..." --change "..." --training "..." --selection "..."`
  -> `runs/<F>/<ID>/S<s>/EVAL_MS_<anchor>.json` and one `EXPERIMENT_LEDGER.csv` row: the 12 utility clauses of the complete
  release (anchor income + your occupation) against U0 and Ucal with household-clustered paired SEs and margins, the men's and
  women's occupation Brier guards with SEs, the whole-release development reserve (statistical reviewer's rule), cross-fitting
  diagnostics (cross-fitted MODEL_FIT utility, fold-average versus deployed MODEL_SELECT differences), per-class and per-SEX
  decompositions and comparisons with the equally exposed task-only references (T_TREE, TASK_REFIT).
- Ledger: every fit, evaluation, failure and decision is a row of `results/pcrl_occupation_headroom_overnight_v1/EXPERIMENT_LEDGER.csv`
  (scoring appends one; record failed or abandoned fits with `python3 -m ohn.ledger add ...` when available, else list them in
  your report and the coordinator records them).

## Configuration accounting (ceilings, not quotas)

Seed-specific fitted release configurations: at most 36 occupation task-model configurations (12 per family A/B/C: an initial
design and at most two justified revisions, at most two predefined capacity settings per revision, both seeds); at most 24
privacy-stage; at most 8 conditional income-branch; at most 12 matched-control or stability. Cross-fitting submodels, calibrators
and readers are counted separately as individual fits. A copied or combined release is not a new fit.
