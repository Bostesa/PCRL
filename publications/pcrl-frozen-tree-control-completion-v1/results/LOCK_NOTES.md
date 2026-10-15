# Lock chronology and disclosed deviations (completion campaign)

| step | evidence | note |
|---|---|---|
| Seal and queue | commit `3c6d4aadd`; queue opened 19:16:06Z | no AGENTS.md anywhere in scope |
| Frozen admission | unit `TC_ADMISSION_V1` PASS (46 checks) | L_TREE INNER release recomputed from its deployment package bitwise in both seeds |
| COMPLETION_LOCK | commit `6484c0abc`; **public** receipt on GitHub (`github` remote) `ba54ecc` (~19:22Z), before any new fit | The pushed public copy contained one absolute local directory string (the `origin` local-mirror path) in the remotes note; a follow-up public commit `4db5534` scrubs it. History was not rewritten. The private copy keeps the original bytes and hash. |
| Control fits | units `TC_NEURAL_S{0,1}_V1`, `TC_FARE_S{0,1}_V1` | completed first attempt |
| COMPLETION_TRAINING_LOCK | before FIT2 opened | — |
| FIT2 + selection | units `TC_FIT2_S{0,1}_V1`, `TC_SELECTION_V1`; COMPLETION_SELECTION_LOCK commit `86593ae08` | the selection unit sat queued ~10 min because the driver loop exited just before it was enqueued (no effect) |
| Readers | units `TC_READERS_S{0,1}_V1` | — |
| REPORTING_LOCK | commit `5302cdec8` | — |
| Engineering correction (report) | unit `TC_REPORT_V1` **failed** with a KeyError in label assignment (rows were indexed by list position with wrong offsets for C2-C5) | The worker had loaded INNER truth but crashed before writing any output; no completion outcome was visible. Fixed to select rows by endpoint name with a grouping assertion; rerun as `TC_REPORT_V2`. Endpoint definitions, data and inference unchanged. |
| Label strings | `COMPLETION_REPORT.json` | The report writes `MARGIN_ESTABLISHED` for the lock's `MARGIN_OVER_NEURAL_RAWJ_ESTABLISHED` (C2) and its FARE analogue (C4). C3 and C5 labels (`NEURAL_PRIVACY_EFFECT_*`, `FARE_FAIRNESS_EFFECT_*`) follow the locked endpoint rule (lower bound > 0) but were not spelled out in the lock. Wording only; no status or rule changed. |
| Power code revisions (planning only) | `tree_completion/power.py` | After the statistics review: prevalence scenarios shift utility means; joint power is computed directly at grid points (boundary clauses report alpha-level power instead of 0); flat-prior assurance added. No control-comparison result changed. |
| FARE fallback rule wording | `COMPLETION_LOCK.json` added_controls.FARE_TC.nomination; `tree_completion/selection_worker.py` (hash bound in COMPLETION_LOCK.code_sha256) | The lock prose says only "else descriptive fallback" for FARE. The locked selection code implements the same rule as for RAW_J (minimum worst-seed normalized violation), which selected F1 (6.69x vs 9.47x and 13.06x). A lowest-proxy fallback would have picked F3, which is more protective and more utility-violating. FARE is descriptive either way. Raised by the independent verifier. |
