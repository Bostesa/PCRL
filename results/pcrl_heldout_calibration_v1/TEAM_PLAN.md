# Team plan — hcal (held-out, shared calibration of frozen releases)

Started 2026-10-07T12:22:22Z (`<PRIVATE_CACHE>/hcal_v1/START.txt`). The worktree branch is
`research/pcrl-heldout-calibration-v1`, from the lra tip 9762025. The package is `hcal/`, the tests are in
`tests/pcrl_heldout_calibration_v1/`, and the public results directory is `results/pcrl_heldout_calibration_v1/`. The
private store is `<PRIVATE_CACHE>/hcal_v1`, named by `PCRL_HCAL_PRIVATE_CACHE`. The closed lra code, results and store
are read only.

## Roles

Owned files are as in PROTOCOL.md §10.

| Role | Responsibility | Owns |
|---|---|---|
| A, coordinator | Admission, locks, roles, scheduling, selection, inference, integration, commits, handoffs | hcal/ids.py, data.py, admit.py, run.py, lock.py, sema.py, select.py, family.py, eval_lock.py, assess.py, infer.py, deploy.py, report code; PROTOCOL.md, SELECTION_RULES.json and every integrated document |
| B, math reviewer | Objectives, convexity and solver claims, certificates, inference arithmetic | MATH_REVIEW.md (findings only) |
| C, calibration implementer | H-TOKEN32, T-TOKEN32, H-GLOBAL-TEMP, H-CLASS-TEMP, the U calibrations | hcal/calib.py, tests/…/test_calib.py |
| D, attack/control owner | Common attack banks, fresh bank, controls, composition | hcal/bank.py, hcal/controls.py, tests/…/test_bank.py, test_controls.py |
| E, independent verifier | Separately written replay that imports no hcal calibration, runner, metrics, selection or inference code | results/…/verification/ |
| F, custody/resource/reporting | Copies, restores, accounting, status, identity and consistency scans, prior art and claim scope | BACKUP_VERIFICATION.json, RESTORE_INDEX.json, COST_AND_CLOSEOUT.md (draft), consistency and identity scans |

A's dispatch on 2026-10-07 also gave F these files, which PROTOCOL.md §10 does not yet list (for A to integrate):
`hcal/closeout.py`, `tests/pcrl_heldout_calibration_v1/test_closeout.py`, `PRIOR_ART_AND_CLAIM_SCOPE.md` and this
`TEAM_PLAN.md`.

## Shared rules

**Ownership and review.**
- Only A writes locks, commits or pushes.
- Reviewers write their findings separately (B in MATH_REVIEW.md; the others in their own files or messages to A). They
  never edit another role's file, and never a locked file.
- A integrates or disposes of every finding.

**Semaphore.** Every heavy process runs as a child of the shared semaphore. Heavy means more than about 30 s of CPU:
stages, attackers, verification, restores, timing and the test suites.
- Command: `OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m hcal.sema --label <ROLE>:<what> -- <command>`. E may instead run
  `<python> -P <WORKTREE>/hcal/sema.py`.
- At most **2 heavy processes in total**, each with **one BLAS/torch thread** (OMP_NUM_THREADS=1;
  torch.set_num_threads(1)).
- Resource limits: ≤ 8 GiB memory in aggregate, and ≥ 5 GiB free disk.
- Every acquire and release is logged in `<PRIVATE_CACHE>/hcal_v1/run/SEMA_LOG.jsonl`.

**Budget (PROTOCOL §9).** $0 cloud. At most 20 CPU-hours and 10 elapsed hours for the whole study, including agents,
verification and restores. Units are resumable and atomic.

**Labels.**
- No Adult task or SEX label is read before SCIENCE_LOCK is pushed. Admission and engineering read none.
- Assessment labels are unsealed only by `hcal.assess`, after the pushed EVALUATION_LOCK.
- No `outer__*` unit is opened before that.

**Processes and files.**
- No agent kills a process it did not start.
- No agent deletes anything outside its own private scratch.
- The unrelated installer disk image is never touched.

## Reporting cadence

F reports to A every 30–60 minutes while work runs, using the read-only `python -m hcal.closeout status`. Each report
gives:
- completed and remaining units (against the queue manifest);
- active owned processes (hcal.run / hcal.sema / hcal.assess / hcal.verify);
- CPU, wall time and memory: semaphore child CPU by role, elapsed time since START.txt against the 20 CPU-hour and
  10-hour budget, and the peak concurrent holds against the limit of 2;
- free disk against the 5 GiB floor;
- the current stage;
- unresolved findings.

Work outside the semaphore, such as the agents' own processes, is not in the ledgers. The report says so.

## Stage plan (PROTOCOL §8)

1. **SOURCE_ADMISSION_LOCK, then `admit`.** Done: 540 units and 171 frozen-bank tables, with bitwise parity.
2. **ENGINEERING_LOCK, then `engineering`.** Synthetic checks only, reviewed by B, D and E. The verdict is
   ENGINEERING_READY on correctness alone.
3. **SCIENCE_LOCK, then the science stages.** In order: `calibrate`, `utility`, `audit`, `compose`, `controls`,
   `select`, `replay`; then E's replay.
4. **EVALUATION_LOCK, then the assessment.** A single `hcal.assess` opening, then `hcal.infer`.
5. **Closeout.**
   - Reports and E's final phase.
   - F's custody: a same-device copy, plus an off-device copy when the content-identified drive is mounted (otherwise
     PENDING).
   - Restores from the copy alone: the U teacher, the frozen bank, every calibrator family, the selected and control
     releases, and the selected reader refit.
   - BACKUP_VERIFICATION.json and RESTORE_INDEX.json, then the handoffs.
