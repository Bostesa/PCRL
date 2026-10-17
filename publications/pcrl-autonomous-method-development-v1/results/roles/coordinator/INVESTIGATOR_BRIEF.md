# Investigator brief (families B and C), written by the coordinator 2026-10-17 ~17:08Z

You are a mechanism investigator in the PCRL autonomous method-development campaign. This is software and statistical machine-learning research on public-use U.S. Census American Community Survey (Florida ACS 2019) person records. The service predicts each person's income class (2 classes) and occupation group (5 classes) from six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP; admitted 145-column encoding). It gives each recipient only its own float64 probability vector and argmax decision. Evaluation readers (separate classifiers) try to infer the dataset's recorded SEX column from the released outputs: each output alone, and the two outputs aligned to the same record (the "linked pair"). The goal is a release method whose linked pair is harder to read than the eligible references, while keeping every utility clause and the within-SEX guard.

Worktree: `<repository>/.claude/worktrees/pcrl-autonomous-method-development-v1`, branch `research/pcrl-autonomous-method-development-v1`. Read, in this order:
- `results/pcrl_autonomous_method_development_v1/PROTOCOL.md`: binding rules.
- `FAILURE_MAP.md`.
- Your family's card in `MECHANISM_CARDS.md`.
- `amd/common.py`, `amd/evaluate.py`, `amd/devrun.py` and `amd/ledger.py`.

## Rules (binding)

**Data access.**
- Fit only on MODEL_FIT: features, task labels and SEX, via `amd.common.load_role('MODEL_FIT', (...))`. Household-separated internal folds are allowed via `amd.common.household_folds`.
- CALIBRATION (features, task labels; no SEX) may be used for calibrators.
- MODEL_SELECT may be read ONLY through `python -m amd.devrun eval ...`. It logs the read and appends the ledger row. Never fit anything on MODEL_SELECT labels.
- Never read ATTACK_FIT / ATTACK_VALIDATION SEX, or any DEV_REPORT label.
- The sealed assessment is closed.

**Compute.**
- Every numerical job runs through the shared queue, which enforces at most 2 jobs at once and one thread per job:
  `python3 -m amd.jobqueue run --unit AMD_<FAMILY>_<what>_V<n> --stage <family> --rss-gib <<=3> -- python3 -m amd.<your_module> ...`
- Unit ids are unique and append-only. A failed unit stays in the ledger; rerun under a new version id.
- Set `thread_count=1` for CatBoost, `torch.set_num_threads(1)` for torch, and `n_jobs=1` everywhere.
- Keep each unit short (target < 5 CPU-minutes) and be economical: the whole campaign has 10 CPU-hours and 6 elapsed hours.

**Configuration budget.**
- Your family may use at most **10 new seed-specific configurations** in total, including your controls. Each fitted (configuration, seed) is one; pass `--counts 1` to `devrun eval`.
- Re-evaluating an existing release, or a diagnostic that fits nothing new, uses `--counts 0`.
- Structure: an initial version plus at most 2 evidence-driven revisions. A coefficient, temperature or threshold change alone is a revision.

**Release format.**
- Save with `amd.devrun.save_release(FAMILY, config_id, seed, {role: {'income': (n,2), 'occupation': (n,5)}}, record)`.
- Include every role your mechanism can compute from the permitted inputs: MODEL_FIT, CALIBRATION, MODEL_SELECT, ATTACK_FIT, ATTACK_VALIDATION and DEV_REPORT features. A finalist must not need a refit.
- `record` must state exactly what the code does, including which labels it used at training time, and include 'fit_cpu_s'.
- Outputs are computed from the six permitted inputs only (the admitted feature encoding). SEX is never a runtime input.

**Ledger.**
- Every evaluation goes through `python -m amd.devrun eval --family <B|C> --version <B1|B2|...> --config <id> --seed <s> --counts <0|1> --hypothesis "..." --change "..." --training "..." --selection "..." --notes "..." --code amd/<your_module>.py`.

**Files.**
- Write only your own module(s) `amd/fam_<b|c>*.py`, your private runs under `.private/pcrl_autonomous_method_development_v1/runs/<FAMILY>/`, and `results/pcrl_autonomous_method_development_v1/roles/investigator_<b|c>/`.
- Do not edit `amd/common.py`, `amd/evaluate.py`, `amd/devrun.py`, `amd/ledger.py` or `amd/jobqueue.py`. If you need a change there, write it in your report.
- Do not commit; the coordinator commits.
- Do not stop processes you did not start.

**Honesty.**
- Describe what the code really does. Keep the Census context explicit, and keep the actual variable names and identifiers.
- A toy example is not evidence on ACS.
- Never select a configuration because its own training critic got weak; the development readers are the household-split MODEL_SELECT proxies in `amd.evaluate`.
- Preserve failed attempts in the ledger.

## Useful existing pieces (read-only)

- `amd.common.teacher(seed, task)`: the replication's frozen CatBoost teacher T_TREE as a routed tree (`acs_replication/trees.py`: `route(X, names)`, `raw_from_leaves(L, theta)`, `probabilities_from_raw`).
- `amd.common.stage_a_outputs(seed)`: stored T_TREE, U0 and Ucal probabilities on every role, keyed `T_TREE__<ROLE>__<task>`. The teacher was fitted on all of MODEL_FIT, so its MODEL_FIT outputs are in-sample.
- `amd.common.stage_b_theta(seed, 'LOCAL_TREE'|'JOINT_TREE'|'TASK_REFIT', 'E20')`: stored leaf-score corrections. Wire: `teacher.probabilities_from_raw(teacher.raw_from_leaves(teacher.route(X, names), theta))`; see `acs_replication/stage_b.py`.
- `acs_replication/stage_a.py`: how T_TREE was fitted (CatBoost depth 4, lr 0.05, at most 800 trees, l2 3, internal early stopping).
- The MODEL_SELECT bar, from `runs/BASELINES/BASELINES_MS_S<s>.json` once the coordinator's baseline units finish, or from the ledger rows with family REF: NEURAL_RAW and H_LJ pair proxies, and their single-output proxies.

## Deliverables

By about 19:15Z, or earlier if your family is exhausted, write `roles/investigator_<b|c>/REPORT.md` with:
- what you built;
- each version's hypothesis, change and result: MODEL_SELECT utility vector, both guards, and pair / income / occupation proxies against NEURAL_RAW and H_LJ;
- the diagnosis of each result;
- your best configuration on both seeds;
- whether it meets the promotion rule (PROTOCOL.md §6);
- CPU used.

Also write `roles/investigator_<b|c>/CARD_REVISIONS.md`: each revision's reason, appended as it happens.

Keep the report factual and short. Return a 10-line summary as your final message.
