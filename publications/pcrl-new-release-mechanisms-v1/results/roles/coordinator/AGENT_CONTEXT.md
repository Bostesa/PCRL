# Shared context for every agent of the PCRL new-release-mechanisms campaign

Campaign `pcrl_new_release_mechanisms_v1` ("PCRL autonomous development: find a different competitive release mechanism").
Coordinator: the main session. This file is the common brief; your assignment message adds your role.

## What the research is (keep this terminology)

- Statistical machine-learning research on already admitted **public-use U.S. Census Bureau ACS 2019 1-Year PUMS** person records
  for Florida (development cohort, household groups).
- Income (2 classes) and occupation (5 official broad groups) are useful prediction tasks.
- The service receives six raw fields: AGEP, WKHP, COW, SCHL, MAR, POBP. The admitted encoding X has 145 columns.
- Each recipient receives only its own float64 probability vector and its argmax decision.
- Separate **evaluation classifiers ("readers")** estimate the recorded SEX column from the released probability vectors (one
  recipient's output, or linked recipients pooling both outputs). "Recovery" = the readers' AUC for SEX. Lower recovery for the
  declared readers is never "protection against every reader".
- SEX coding: sex 0 = men (SEX code 1), sex 1 = women. SEX is a training label on MODEL_FIT and an evaluation label on MODEL_SELECT;
  never a runtime input or output.
- If an operation is blocked by the platform, preserve state and report it. Never bypass a refusal.

## Paths and names

| Item | Value |
|---|---|
| Worktree (run everything from here) | `<repository>/.claude/worktrees/pcrl-correction-transfer-stability-v1` |
| Research branch (local only, never pushed) | `research/pcrl-new-release-mechanisms-v1` |
| Campaign package | `nrm/` (coordinator infrastructure: `nrm/common.py`, `nrm/jobqueue.py`, `nrm/release.py`, `nrm/evaluate.py`, `nrm/screen.py`, `nrm/devreaders.py`) |
| Family code | Designer A: `nrm/fa/`. Designer B: `nrm/fb/`. A third family, if approved: `nrm/fc/`. Challenger controls: `nrm/ctl/`. Statistical reviewer: `nrm/rules.py`, `nrm/test_rules.py`. Verifier: `results/pcrl_new_release_mechanisms_v1/roles/verifier/code/` |
| Public-safe records | `results/pcrl_new_release_mechanisms_v1/` (your role folder: `roles/<role>/`) |
| Private tree (never committed) | `.private/pcrl_new_release_mechanisms_v1/` (runs, caches, logs) |
| Latest pinned source | publication `ed7bc1ccd550f452c112ba3144a9d64b014c52fd` (`publications/pcrl-controlled-training-size-v1`); the same files are in this worktree under `results/pcrl_controlled_training_size_v1/` and code in `tsz/` |
| Interpreters | `python3` (3.14.3; numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0). Torch: `<repository>/.venv/bin/python` (3.13.7; torch 2.10.0, numpy 2.4.2, scikit-learn 1.8.0). CatBoost (fit-time only): `.private/pcrl_correction_transfer_stability_v1/venv_catboost/bin/python` |

## Clock (real UTC; never backdate)

- Campaign start: **2026-10-20T14:29:43Z**. Research window ends **22:29:43Z**; closeout ends **22:59:43Z**.
- Plan:
  - until ~15:45Z: source admission, failure map, proposals, protocol receipt;
  - 15:45-18:45Z: working prototypes, task feasibility, justified revisions;
  - 18:45-20:30Z: matched protection comparisons and robustness;
  - 20:30-22:30Z: decisive ablations, independent replay, deployment, closeout.

## Budgets (shared; all numerical work counts)

- **CPU:** 14 CPU-hours in total, including failed attempts, reader fits and agents' numerical work.
- **Heavy jobs:** at most 2 at once, one numerical thread each, all through the ONE shared queue:
  `python3 -m nrm.jobqueue run --unit <UNIQUE_ID> --stage <stage> --rss-gib <G> -- <interpreter> -m <module> <args>`.
  - Unit ids are append-only: a retry needs a new id (`..._V2`).
  - Stages: `fit`, `evaluate`, `screen`, `readers`, `verify`, `deployment_check`, `custody`, `closeout`, `publication`.
  - Small label-free scripts may run outside the queue. Charge their CPU with
    `python3 -m nrm.jobqueue charge --label <you> --cpu-s <upper estimate> --note "<what>"`.
- **Configurations:** at most 60 new seed-specific or training-repetition-specific release configurations, fitted matched controls
  included.
  - One configuration = one (family, config_id, version) release saved through `nrm.release.save`, which counts them and refuses
    the 61st.
  - Record internal cross-fitting, critic, calibration and reader fits separately in the release record.
- **Paid or cloud compute:** zero.
- Do not fill time with duplicate equivalent fits or irrelevant experiments.

## Data roles and the single logged access layer

| Role | People / households | Labels admitted | Use in this campaign |
|---|---|---|---|
| MODEL_FIT | 20,000 / 13,337 | income, occupation, SEX (+ raw_inputs) | fitting release parameters (full pool where possible; household grouping for any internal folds) |
| CALIBRATION | 2,982 | income, occupation (no SEX) | calibration parameters only |
| MODEL_SELECT | 8,001 / 5,306 | income, occupation, SEX | logged adaptive selection, diagnosis, utility benchmark and development readers ONLY. Never fit gradients, calibration parameters or continuous release parameters on its labels |
| ATTACK_FIT, ATTACK_VALIDATION, DEV_REPORT | - | **closed** (features only, for deployment parity) | no label read |

- The sealed assessment and capped-out people are refused by every loader.
- There is no new population and no microdata download.
- Label-free features: `nrm.common.features(role)` returns `X` (n, 145), `feature_names`, `row_id`, `unit` (household id).
- Labels: only through `nrm.common.load(role, fields, operation, purpose)`.
  - It refuses everything before the protocol receipt is verified on GitHub.
  - Afterwards it allows only the registered operation and entry-point module, and appends every read to the access log.
- MODEL_SELECT labels are read only by the coordinator's entry points: `nrm.evaluate` (utility), `nrm.screen` (development
  screening readers on a screening split), `nrm.devreaders` (formal grid) and the verifier's code.
  - Families get MODEL_SELECT feedback from those outputs. They never read MODEL_SELECT labels themselves.
- MODEL_SELECT was used adaptively by every earlier campaign: all evidence here is development evidence, never confirmation.

## Utility (unchanged original limits)

- For each source version v in {0, 1} there are 12 clauses: 2 tasks x 2 references (U0_S<v>, Ucal_S<v>) x 3 metrics.
  - Accuracy loss <= 0.010.
  - NLL excess <= 0.010 nats.
  - Summed Brier excess <= 0.005.
- Plus the men's and women's occupation summed-Brier excess vs U0, each <= 0.005.
- Over both versions that gives 24 clauses + 4 guards, computed by `ohn.evaluate.complete_eval`. Both tasks come from the release
  itself; nothing forces H3 income.
- The original recipient non-worsening allowances and the 0.020 material-benefit target are unchanged. The target is reported
  separately from smaller positive improvements.
- The two-SE reserve (`ohn.rules.reserve`, k = 2, H_LJ SE floor) is a separate development robustness heuristic, not a cap or a
  guarantee. Historical references often fail it.
- A "version" v means: training seeds and internal folds of version v, and evaluation against U0_S<v> / Ucal_S<v>. Fit every
  configuration in both versions.

## What has been learned (latest source; preserve every earlier verdict)

- Full permitted training exposure restores the occupation task margin on this pool. Task-only controls also improve with size.
- **A1_PO_16 at 100% of MODEL_FIT** (sklearn HGB on the POBP-ordinal coding; recipe in `tsz/sizefit.py` and `ohn/`):
  - occupation NLL excess vs U0 of -0.0161 / -0.0128 nats (v0 / v1);
  - men's Brier excess -0.0039 / -0.0023;
  - large occupation slack.
- **Leaf-score correction** at the selected s = 0.3 kept utility and lowered linked-pair reader AUC by only ~0.005, not stably. Its
  coalition term had no advantage. That line is CLOSED: no correction-strength, controller, temperature or nearby strength tuning.
- **Strong inherited releases are more protective at their measured operating points.** Pair AUC means from the last grid (8 cells;
  the fresh grid here re-measures everything):

  | Release | Pair AUC |
  |---|---|
  | W (A1 + H3) | 0.6486 |
  | TASK_REFIT + H3 | 0.6459 |
  | NEURAL_RAW | 0.6245 |
  | H_LJ | 0.6183 |
  | B2TWIN | 0.6203 |
  | RETIRED_B3CTL | 0.6038 |
  | SUCC | 0.6138 |
  | TWJ | 0.6413 |

  - All six references pass the original point limits but fail the reserve heuristic.
  - The 0.020 material target is measured against NEURAL_RAW and H_LJ.
- H3 (frozen income) is a useful benchmark with known income limitations. It is not compulsory: both task heads may be trained from
  scratch.
- No competitive new mechanism is established.
- Relevant earlier families (read their results folders under `results/pcrl_*`):
  - adversarial penalties and refreshed critics: refreshed_guarded_joint, strength_matched_feedback, online_strength_frontier,
    penalty_no_erasure, joint_complete_view_method;
  - leaf-score corrections: correction_transfer_stability, controlled_training_size;
  - frozen-bank random channels: latent_and_sampled_methods, and the stochastic-channel and replacement-channel studies;
  - post-hoc shared-cell merging, SCM and IBSI: structural_mechanism_pilot, scm_engineering_benchmark, scm_utility_matched;
  - calibration and learned decoders: learned_decoder_constrained_release, adult_learned_decoder_release, heldout_calibration;
  - latent bottlenecks, codes and grouped feature selection: decision_preserving_compression, confidence_capacity,
    confidence_budgeted_privacy, capacity_matched_feature_selection, occupation_bottleneck, autonomous_method_development,
    income_headroom_successor;
  - task-conditioned representations: task_conditioned_representations.
- Lesson from SCM: lower pair AUC came with MORE task spending than its matched ordinary (SEX-blind) coarsening controls, so the
  benefit was not separated from spending. Any new mechanism must beat a SEX-blind control on the observed utility-recovery
  frontier, not just at one point.

## Release contract and storage

- A release maps the six raw fields (through the admitted encoding) to an income vector (n, 2) and an occupation vector (n, 5).
  Decisions are the argmax. No other field is transmitted. Internal cell ids are not transmitted, but anything recoverable from
  the vectors is part of the disclosure analysis.
- Randomized releases supply ONE draw per recipient per person. Utility is computed on each actual emitted vector and its emitted
  argmax, then averaged over R predeclared independent one-draw-per-person repetitions. Never use an averaged vector unless that is
  the actual release.
- Save through `nrm.release.save(family, config_id, version, arrays, record)`.
  - It writes `.private/.../runs/<FAMILY>/<config_id>/V<v>/` and refuses overwrites: a revision gets a new config id.
  - `arrays[role] = {'income': (n,2), 'occupation': (n,5)}` for all six roles in the admitted row order, or a `(R, n, k)` stack of
    draws for randomized releases.
  - `record` fields are listed in the `nrm/release.py` docstring (mechanism card, kind, fit counts, roles read, exposure, CPU,
    training sizes).

## Process rules

- **Commit before use.** Commit code before any job it governs. Commit a family's mechanism card (and every revision card) before
  the fit it governs. Imported code is hashed at launch by the queue.
- **Commits.** Commit only your own files: `git add <your paths>` then `git commit -m "<message>" -- <your paths>`.
  - Retry if `.git/index.lock` exists.
  - Never use `git add -A`, amend, rebase, reset, stash, checkout or switch branches, or push. Never touch `main`, other branches,
    manuscripts or earlier campaigns' files.
  - Commit messages are plain text exactly as you write them. Never add Co-Authored-By, Claude-Session or any other trailer.
- **Shell.** No heredocs, no `git -C` into the main checkout, no foreground `sleep`. Put temporary scripts in your own scratch
  folder under `<session scratch>`.
- **Records.** Write code, JSON and CSV records in your folder.
  - Return reports and proposals as text in your final message. The coordinator saves them verbatim with attribution. You may also
    write them to your role folder if you are able to.
- **Results.** Preserve every failed check and original result. Never widen a rule after a failure to obtain PASS. Never
  manufacture a positive result: a finite negative result is acceptable.
- **People.** Send no messages to people outside this session.
- **No sub-agents** (added 15:02Z, after deviation 3). Do not spawn sub-agents, forks, teams or helper agents of any kind. The brief
  allows at most six live agents including the coordinator, and the coordinator manages that count. Do the work yourself.
