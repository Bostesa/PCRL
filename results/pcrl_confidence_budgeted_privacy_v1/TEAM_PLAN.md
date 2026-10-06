# Team plan: confidence-budgeted privacy compression (cbp)

| Item | Value |
|---|---|
| Branch | `research/pcrl-confidence-budgeted-privacy-v1`, from qpc tip `d0c8a45c879d01fb8b736ccc091ec3e2c3e9b351` (evidence `9dd06da6b64e558e1c079f76e43982b60b327e63`) |
| Package | `cbp/`. qpc, dpc and osf are pinned in this tree and imported read-only, or copied with a provenance note and a minimal documented diff. Never edit `qpc/`, `dpc/` or `osf/` |
| Public evidence | `results/pcrl_confidence_budgeted_privacy_v1/` |
| Private store | `<PRIVATE_CACHE>/cbp_v1/` |
| Start | 2026-10-06T16:44:45Z |
| Ceilings | 10 h (hard stop 02:44Z); 20 CPU-h in total including agents; 8 GiB; ≥ 5 GiB free; $0 cloud |
| Reserve | 00:44Z–02:44Z and 4 CPU-h, for verification, packaging, backup and closeout |
| Prompt | `<USER_DOWNLOADS>/PCRL_Confidence_Budgeted_Privacy_Execution_Prompt.txt` (local; not committed) |

## Shared semaphore and ledger: two heavy processes in total

Every numerical process runs under the semaphore. That covers fitting, attacks, replay, restores, synthetic timing, and also every numerical TEST run, including reviewers' (prompt §3: "numerical tests, fitting and replay must acquire a slot"):

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m cbp.sema --label <ROLE>:<what> -- <command>
```

When calling by file path, add `-P`. Children inherit the slot. The wrapper logs CPU to `<PRIVATE_CACHE>/cbp_v1/run/SEMA_LOG.jsonl`, the single resource ledger. Never kill another role's process. To stop your own run, send SIGTERM to its wrapper; the wrapper terminates its child and logs the release. Reading code and writing documents needs no slot.

## Who may touch real data

- **Only the lead** starts real-data stages (admit, fit, inner, inner_src, controls, select, assess, infer), through `cbp.run` or `cbp.assess`, after the governing lock is pushed.
- **Agents** develop and test on synthetic fixtures.
- **Nobody** reads assessment labels or prior per-person assessment predictions.
- **The verifier (F)** reads real data only for its replay phases, under the semaphore. It reads assessment labels only after verifying that EVALUATION_LOCK is on origin.

## Roles and exclusive file ownership

| Role | Owns |
|---|---|
| A. Lead | `cbp/{sema,lock,run,data,admit,select,family,infer,eval_lock,report}.py`, `cbp/tests/{test_pipeline,test_truth_table,test_late,test_admit}.py`; PROTOCOL, EXPOSURE_LEDGER, SOURCE_INDEX, SOURCE_ADMISSION, ROLE_MANIFEST, PREDICTIONS, FIT_MANIFEST, RUN_STATUS, HEADROOM_SELECTION_RULES, PRIMARY_FAMILY, LABEL_TRUTH_TABLE, locks, decision documents |
| B. Optimizer engineer | `cbp/fit.py`, `cbp/deploy.py`, `cbp/tests/test_fit.py`; METHOD_CARD.md, TIMING.json (key "fitting") |
| C. Statistics and selection reviewer | `results/.../SELECTION_REVIEW.md`, `cbp/review_tests/test_selection_review.py` (never locked) |
| D. Attacker engineer | `cbp/audit.py`, `cbp/baselines.py`, `cbp/assess.py`, `cbp/tests/test_audit.py`; AUDIT_COMPUTE.json, TIMING.json (key "audit") |
| E. Math and claims reviewer | MATH_REVIEW.md, PRIOR_ART_AND_CLAIM_SCOPE.md, `cbp/review_tests/test_math_review.py` (never locked) |
| F. Independent verifier and custody | `results/.../verification/replay_cbp.py`, INDEPENDENT_VERIFICATION.json, `cbp/closeout.py`, `cbp/tests/test_closeout.py`, BACKUP_VERIFICATION.json, RESTORE_INDEX.json |

**Rules:**
- **Tests:** reviewers' tests live ONLY in `cbp/review_tests/`, which no lock ever globs. Never edit a file under `cbp/` or `cbp/tests/` after it is locked; `python -m cbp.lock verify <lock>` lists what is locked.
- **Git:** only the lead commits and pushes.
- **Communication:** send findings as reports or concrete patches to the lead (SendMessage to "main").

## Fixed design (prompt §§6–11)

**Rate.** ONE rate: income 8 / occupation 64 states per teacher-predicted class (`i8o64`). The fine partitions are the admitted qpc `fine__s{k}` (income 32, occupation 128).

**λ grid.** {0.01, 0.025, 0.04, 0.06, 0.08, 0.1} × {LOCAL, SEQ-12, SEQ-21, JOINT} × seeds {0, 1, 2} = 72 privacy units.
- λ 0.01 and 0.1 (24 units) are admitted from qpc after parity checks.
- λ 0.025, 0.04, 0.06 and 0.08 (48 units) are new fits with `qpc.compress.fit_unit`: unchanged qpc definitions, the corrected sequential design, the five JOINT starts with witnesses, and the 5-sweep cap.
- No other optimiser change.

**Admitted references:**
- DIRECT-TASK i8o64, FINE-TASK i8o64 and CLASS-ONLY (`U|CLASS|i1o1`), 3 seeds each;
- continuous sources SRC|U and SRC|RAW-J_b0.3;
- references REF|E (LEACE), REF|F (FARE) and REF|F0.

**Config IDs** (qpc scheme; λ formatted with `f"{lam:g}"`): `U|{FAM}|i8o64|l{lam}`, `U|FINE-TASK|i8o64`, `U|DIRECT-TASK|i8o64`, `U|CLASS|i1o1`, `SRC|U`, `SRC|RAW-J_b0.3`, `REF|E/F/F0`.

**Unit names:** `pol__s{k}__<cid with | replaced by _>`, `tea__s{k}__U`, `ref__s{k}__F`, `fine__s{k}`, `inner__<unit>`, `outer__s{k}__<safe label>`. These sit in `<PRIVATE_CACHE>/cbp_v1/run/units/`, in the same release.npz / policy.json / teacher.npz / reference.npz formats as qpc.

**Inner bank:** per seed, 27 codes plus 2 sources plus 3 references; 96 inner units in total. SRC|U composes over ALL 27 codes of its seed.

**Selection** (`HEADROOM_SELECTION_RULES.json`; prompt §9):
- Ordinary eligibility is per seed and per task, as in qpc.
- Privacy NOMINEES additionally need log-loss excess ≤ 0.006 and Brier excess ≤ 0.0035 on every task and seed.
- T* comes from the closed list: DIRECT-TASK, FINE-TASK, U, CLASS-ONLY, F0. RAW-J is excluded.
- C_rate is DIRECT, FINE, LOCAL, SEQ-12 or SEQ-21 at any λ. C_global covers the full nonjoint bank.
- Comparators need ordinary eligibility only.
- Guards: individual inner AUC ≤ comparator + 0.005 for each recipient and seed.
- Ordering: mean pair AUC, mean summed log loss, states, ID.
- The fallback ordering uses the three registered shortfalls.

**Family:** 37 slots, z = NormalDist().inv_cdf(1 − 0.05/74), B = 1,999, exact-record-group bootstrap, **seed 20261008**.

**Labels** (`LABEL_TRUTH_TABLE.json`): PRIVACY_COMPRESSION_DEVELOPMENT_CRITERION_MET (family), JOINT_DEVELOPMENT_CRITERION_MET, CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION, EXPERIMENTAL_NO_ADVANTAGE, NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE, INCOMPLETE_OR_INVALID.

**JSON:** new JSON is finite or null (`allow_nan=False`). Historical qpc JSON with ±Infinity is read with an explicit parser note.

## Stages and target times (UTC)

| Stage | What happens | Target |
|---|---|---|
| 0 | Admission, engineering, synthetic timing, predictions | SOURCE_ADMISSION_LOCK 17:45 → admit |
| 1 | Freeze. B, D, lead code and reviews done; FIT_LOCK + AUDIT_AND_SELECTION_LOCK pushed before ANY new fit | 19:15 |
| 2 | 48 new fits | Minutes |
| 3 | Inner audits (96 units), composed sources, controls, select; F verifies selection BEFORE the assessment | 19:20–20:45 |
| 4 | EVALUATION_LOCK | |
| 5 | One assessment opening + inference | |
| 6 | Verification, deployment, reports, backup, closeout | |
