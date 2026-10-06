# lcr team plan (lead-owned; read-only for every other role)

Study: learned decoders and confidence-constrained releases (lcr). Branch `research/pcrl-learned-decoder-constrained-release-v1`
from cbp final tip 7f3ec67. Execution prompt: `<USER_DOWNLOADS>/PCRL_Learned_Decoder_And_Constrained_Release_Execution_Prompt.txt`
(sections are cited as §n). Package `lcr/`; public results `results/pcrl_learned_decoder_constrained_release_v1/`
(= PKG); private store `<PRIVATE_CACHE>/lcr_v1/` (`run/units/` holds every unit).

## Clock and budget

- **Clock.**
  - Session start 2026-10-06T23:42:46Z; hard ceiling 2026-10-07T09:42:46Z.
  - The reserve (final 2 h, 4 CPU-h) begins at 07:42:46Z. The assessment must finish before that.
- **Compute.**
  - At most 2 heavy numerical processes for the WHOLE team, through ONE semaphore:
    `OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m lcr.sema --label <ROLE>:<what> -- env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python <cmd>`.
    The verifier calls it by path with `-P`.
  - Every numerical test, timing run, fit, replay or review computation takes a slot. Reading code takes none.
  - Each process uses one thread. Aggregate memory is at most 8 GiB.
- **Cloud.** $0. No AWS.

## Roles and file ownership

Write only your own files. Send findings and requests to the lead.

| Role | Owns |
|---|---|
| A lead | lcr/{run,lock,sema,data,admit,select,family,infer,eval_lock,report}.py, lcr/tests/test_{admit,select,late,truth_table}.py, every lock, PROTOCOL.md, EXPOSURE_LEDGER.md, SOURCE_*.json, PREDICTIONS.json, FIT_MANIFEST.json, SELECTION_RULES.json, LABEL_TRUTH_TABLE.json, PRIMARY_FAMILY.json, the decision documents, all commits |
| B decoder/math | lcr/decoder.py, lcr/fixtures.py, lcr/deploy.py, lcr/tests/test_decoder.py, lcr/tests/test_fixtures.py, lcr/tests/test_deploy.py, PKG/FIXTURE_LAWS.json, PKG/FIXTURE_GATE_RULE.json, PKG/MATH_REVIEW.md, PKG/DECODER_CERTIFICATES.json (written by the runs), the fixture part of METHOD_CARD.md (PKG/METHOD_CARD_DECODER.md; the lead merges) |
| C mapper | lcr/mapper.py, lcr/tests/test_mapper.py, PKG/SEARCH_RULES.json, PKG/METHOD_CARD_MAPPER.md, TIMING.json["fitting"] |
| D attacker/statistics | lcr/audit.py, lcr/baselines.py, lcr/assess.py, lcr/tests/test_audit.py, PKG/AUDIT_COMPUTE.json, TIMING.json["audit"] |
| E independent verifier | PKG/verification/replay_lcr.py, PKG/INDEPENDENT_VERIFICATION.json, PKG/FIXTURE_ORACLE_REPORT.md. Imports no lcr/cbp/qpc/dpc/osf/smf/rgj/jcv/oar/stored_model_eval/pcrl module. |
| F claims/custody | lcr/closeout.py, lcr/tests/test_closeout.py, PKG/PRIOR_ART_AND_CLAIM_SCOPE.md, PKG/BASELINE_FAIRNESS_REVIEW.md, BACKUP_VERIFICATION.json, RESTORE_INDEX.json, provenance/ |

Review tests go in `lcr/review_tests/`, which is NOT locked, so reviewers never edit locked files.

## Shared contracts (defined in lcr/run.py — use its helpers, never re-derive names)

- **Config IDs.**
  - D0 admitted: `U|DIRECT-TASK|i8o64`, `U|FINE-TASK|i8o64`, `U|CLASS|i1o1`, `U|{LOCAL,SEQ-12,SEQ-21,JOINT}|i8o64|l{lam}`.
  - D1 fixed-map: `<D0 id>|D1`.
  - C-TASK: `U|C-TASK|i8o64|D1`.
  - Weighted: `U|W-{FAM}|i8o64|l{lam}|D1`.
  - Constrained: `U|K-{LOCAL,SEQ-12,SEQ-21,JOINT-SINGLE,JOINT-PAIR}|i8o64|D1`.
  - Sources and references: `SRC|U`, `SRC|RAW-J_b0.3`, `REF|E|F|F0`.
  - `run.parse_id(cid)` gives arm (d0, d1_fixed, ctask, weighted, constrained), base_family, decoder, lam and
    privacy_trained.
- **Units.**
  - `pol__s{k}__<safe>`: D0, admitted from cbp.
  - `dec__s{k}__<safe>`: D1 fixed-map.
  - `new__s{k}__<safe>`: new fits.
  - Also `tea__`, `ref__`, `fine__` and `inner__<unit>`.
  - Write units with `run.save(name, files, record)`. It is atomic, hash-complete and stores finite JSON only.
- **release.npz (all code releases, D0 and D1).** Exactly `row_id, tok1, q1, hard1, alpha1, tok2, q2, hard2, alpha2`.
  - It covers all D rows.
  - `q_i` is the ACTUAL released vector, `(u + eps*1 + eps*e_d)/(1+(K+1)eps)` with eps = 1e-12.
  - `hard_i` equals the teacher decision.
  - Tokens are qpc canonical token IDs of the assignment map.
  - This keeps cbp.audit's policy views unchanged for D1 releases (same keys, different q).
- **decoder.json (D1 units).** Per recipient: K, the token classes, per-token fitting count n_t, label counts y_t,
  teacher sum S_t, base vector u_t, released q_t, fallback flags, the certificate (KKT/stationarity residuals, active
  set, projection magnitude, objective) and the sufficient-statistic hash.
- **policy.json (new fits).** A `qpc.PolicyPair` for the token MAP (cell→token). Its token_proto are the D0 means and are
  NOT released. The released vectors come from decoder.json. A D1 deployment binds BOTH files: teacher, schema, policy
  fingerprint and decoder hash.
- **Labels.** Task labels and SEX on OSF_DEFENSE_FIT are read via `qpc.data.labels_for(D, "fitting", "OSF_DEFENSE_FIT")`.
  This is the new supervised use and is disclosed.

## Milestones (UTC, 2026-10-07)

| Time | Milestone |
|---|---|
| ~00:30 | SOURCE_ADMISSION_LOCK pushed; admission run |
| ~01:15 | B: decoder.py + tests delivered; FIXTURE_LAWS.json + FIXTURE_GATE_RULE.json delivered for review. No algorithm has run on a fixture yet. |
| ~01:45 | C: mapper.py (all arms, both neighbourhoods, budgets, caches, receipts) + tests + full-bank synthetic timing. D: audit/baselines/assess adapted + views for D1 releases + composition over the full bank + timing. E: replay skeleton + independent decoder/oracle. F: prior-art scope + fairness review of SEARCH_RULES + closeout skeleton. |
| ~02:00 | FIXTURE_LOCK pushed, then the fixture stage. E replays the fixture oracle, and the gate result is registered. |
| ~02:45 | SCIENCE_LOCK pushed (if the gate passes). |
| ~02:50–03:50 | d1 + fit (2 shards) |
| ~03:50–05:30 | inner, inner_src, controls, select; E selection replay |
| ~05:45 | EVALUATION_LOCK |
| ~05:50–06:50 | assessment |
| 07:00–09:00 | inference, E phase 3, deployment, backup, documents |

## Hard rules (from the prompt)

- No real-data fit before the gate passes and SCIENCE_LOCK is pushed.
- No assessment label is read before the pushed EVALUATION_LOCK.
- No per-seed λ, no result-based pruning, and no tuning of kappa = 32, the Brier coefficient or eps.
- Public files carry no absolute or home paths and no user names. Use `<PRIVATE_CACHE>` and `<WORKTREE>`.
- New JSON is finite-or-null (`allow_nan=False`).
- Do not touch other worktrees, main, manuscripts, or the "BackgroundSyncService Setup" image.
