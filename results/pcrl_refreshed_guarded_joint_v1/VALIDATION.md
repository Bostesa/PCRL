# Validation

## Before any nonzero fit

| Check | Result |
|---|---|
| Inputs and roles | The admitted `adult_jcv.npz` hash matches. The new partition was recomputed independently by the data and provenance owner (`provenance/role_check.py`, no runner imports) and matches `rgj.data` on all 8 roles and subroles, by row-id sets and hashes. No group spans two roles. The old assessment, cert and excluded rows are dropped at load. There are 83 permitted columns, none forbidden; the Husband/Wife proxies are kept and disclosed (`EXPOSURE_LEDGER.md`). |
| Reuse audit | The warm starts were trained on DEFENSE_FIT only, with a fixed schedule; the environment parity reproduces the predecessor U bitwise. The 42 FARE trees were fitted on DEFENSE_FIT rows only (fit rows and labels re-hashed); their heads were refit on HEAD_VALIDATION (`SOURCE_INDEX.json`). |
| Mathematical review (`MATH_REVIEW.md`) | **R1** (critic views built on the drifting training head; a block-fixed floored transform amplified it about 3,000×, so the online critics degraded within blocks) and **R2** (a feasible task-only alias was excluded from C\*) were both fixed before any nonzero fit, and their fixtures now pass. **η = β** was judged weak but not inconsistent, so no repair was authorised; its consequence was registered as predictions 7–8. Advisories A3–A5 were adopted; A2 and A6–A10 are reported. A bounded prior-art check is included. |
| Tests | 72 pass (`rgj/tests`): 20 pipeline (lead), 34 math-review fixtures, 17 audit, 1 inference end-to-end. They cover the 13 required pre-fit checks, including a leakage-transfer case, task = sensitive, the XOR coalition, a poisoned shadow bank (the local arm's coalition gradient is exactly absent), transport in an amplified direction, the full-conjunction logic, bootstrap grouping and output closure. |
| Engineering parity | Before the lock, two rounds ran: pre-R1 and pre-A4. Their 26 receipts were kept as `*.quarantined_*`. Final round under the pushed lock: **21/21 bitwise** (task line = predecessor U; every arm at β = 0, λ = 0 equals the task-only continuation from the same initialisation in both stage orders). |
| Lock | `CODE_LOCK.json` was pushed at `d7367e0` (05:22Z) before any nonzero fit. The later files entered through dated amendments: **A1** (diagnostics, before they ran), **A2** (evaluation-lock writer, inference, deploy, before EVALUATION_LOCK), **A3** (reporting and closeout), **A4** (closeout privacy fix). All locked documents still match their lock hashes. |

## During the run

| Check | Result |
|---|---|
| Training health | 57 runs: 0 nonfinite, 0 rescues, 31 clip hits in total. Every refit, early stop, restart or continue choice, λ value, gradient norm and block-end input scale is logged (`GRADIENT_AND_WEIGHT_DIAGNOSTICS.csv`). |
| Stage C receipts | `tc__s{k}`: each Stage C arm at β = 0 started from the frozen L-R equals the task-only continuation from L-R bitwise, on 3/3 seeds. |
| Ordering | The Stage B freeze was pushed (`2be8cea`, 05:31:45Z) before Stage C. EVALUATION_LOCK was committed at 05:49:42Z and on origin by 05:49:43Z; the first outer-unit file was written at 05:50:23Z. Model development, including the optional ablation, stopped before the assessment opened. |
| Audit controls (inner roles) | J-G, L-G, J-O, U and F on seed 0 with the final slate. Frozen-permutation nulls are ≤ 0.55, and the planted leaks are detected, including one at amplitude 1e-6. `all_ok = true` (`AUDIT_CONTROLS.json`). |
| Deployment | `rgj.deploy` reproduces the stored J-G (seed 1) and LEACE releases bitwise from permitted inputs only, and refuses an 84-column input. |
| Reproducibility | Rerunning a finished stage is a no-op. Rerunning `rgj.infer` reproduces PRIMARY, SECONDARY and RAW_LEVELS byte for byte. Rebuilding the evaluation lock reproduces its seed specifications exactly. |

## Backup and restore (`BACKUP_VERIFICATION.json`, `RESTORE_INDEX.json`)

- 3,062 private files (2.25 GB) were copied to `<drive>/<relocation folder>/private_rgj_v1_20261004/rgj_v1` with `SHA256SUMS`.
- All 3,062 were re-read with F_NOCACHE and match. This was an uncached read, not a physical cold-disk read.
- From the drive copy alone:
  - U, J-G (descriptive) and L-G for seed 1 rebuild bitwise;
  - refitting the recorded AUC-selected coalition attacker of `outer__s1__J-G` from the drive-copy release reproduces its saved assessment probabilities (maximum difference 0).
- Private files that changed after the backup were synced to the same drive folder under `post_backup_sync/20261004T062830Z/`, with their own `SHA256SUMS` and an uncached read-back (2/2 match). There were two: `run/ACTIVITY_LOG.jsonl`, appended by the tested QUICKSTART commands, and `run/closeout_backup.log`. The rerun inference output was byte-identical, so it needed no sync.
- Nothing was deleted.

## Independent replay (`INDEPENDENT_VERIFICATION.json`, `verification/replay_rgj.py`)

**Result: 33 PASS, 0 FAIL, 0 WARN, 3 INFO on the study, plus 12/12 self-tests.**

**Independence.**
- The verifier is a separate agent working from PROTOCOL.md and the saved artifacts.
- An import guard refuses `rgj.select/infer/family/eval_lock/report/critic_track/whiten_diag/audit/assess`, `jcv.infer/select`, `pnx.*`, `oar.*` and the predecessor inference modules, including through unpickling. None was loaded.
- DEVELOPMENT_ASSESSMENT labels were masked until the lock and the outer units existed.

**Reproduced exactly:**
- **Selection.** Its own implementation of PROTOCOL §4 reproduces both stages, C\* = J-O on every seed, the FARE choice, `SEED_STATUS.json` and all 216 rows of `SELECTION_TABLE.csv`. Refitting the inner slate for the frozen L-R units reproduces the recorded AUCs.
- **Endpoints.** Its own paired multinomial bootstrap reproduces all 48 endpoint slots and 730 per-seed levels to 1e-9 relative. The CSVs agree to 6 decimals.
- **Releases.** 40 units rebuilt with its own forward pass match bitwise, including LEACE, whose fitting-row cross-covariance is about 1e-15.
- **Training logs.** All 21 guarded runs' λ trajectories replay exactly.
- **Critic tracking.** The critic gaps recompute from the per-kind values, and 270 critics re-evaluated on CALIB match to 1e-5.
- **Integrity.** All COMPLETE.json files verify, and all 246 evaluation-lock file hashes match.
- **Restore.** Restoring from the drive passes.
- **Claims.** A and B are both NOT_ESTABLISHED (7/9 and 6/9 clauses, with the status requirements unmet).

**Warning raised and resolved.** The first phase-2 run raised one WARN: `FIT_MANIFEST.json`, a locked document, had been edited after the lock (an `outcome` section was added at closeout). The locked version was restored byte for byte, and the outcome accounting moved to `RUN_STATUS.json`. The rerun shows check 8.4 PASS.

A further rerun I started by mistake pointed `--drive` at the parent folder, and its restore check failed with a missing file. The rerun with the correct copy root passes. Both receipts are superseded by the final JSON (06:23:42Z).

**Disclosed limit.**
- The "closest descriptive checkpoint" rule (smallest worst violation over gates and guards; ties to smaller β, then earlier checkpoint) is defined in the locked code (`rgj/select.py` docstring and `closest()`, CODE_LOCK `d7367e0`), not in PROTOCOL prose.
- The verifier therefore checked only that each descriptive unit comes from that arm's grid on that seed. This affects J-G on every seed, L-G on seed 0, J-R on seed 2, and F.
- No claim depends on it: descriptive units can never pass a claim.
