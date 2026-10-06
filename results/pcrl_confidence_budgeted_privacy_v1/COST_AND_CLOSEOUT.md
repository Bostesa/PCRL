# Cost and closeout (cbp)

## 1. Budget

| Item | Limit | Used |
|---|---|---|
| Elapsed | 10 h from 2026-10-06T16:44:45Z | about 5 h to the final semaphore hold (21:41:03Z). The reserve (from 00:44:45Z) was never entered. |
| Study CPU, measured | 20 CPU-h (4 CPU-h reserved) | **4.010 CPU-h** (14,434.8 CPU-s) over 104 semaphore holds (SEMA_LOG.jsonl, through the final hold at 21:41:03Z). 0 holds left open. |
| Study CPU, estimated and unmeasured | — | under 0.1 CPU-h: git, JSON/CSV parsing, deployment refusal checks and read-only document review, all outside the semaphore |
| Heavy numerical processes | at most 2 at any time | peak **2** concurrent holds (reconstructed per slot by role F) |
| Threads per process | 1 | OMP, OpenBLAS, MKL and vecLib set to 1; torch 1 |
| Memory | 8 GiB aggregate | largest single process 1.46 GiB peak RSS (inference / assessment); at most 2 processes |
| Disk | at least 5 GiB free | private store 590 MB; same-device copy 610 MB; 120 GiB free at close |
| Cloud | $0 | **$0**; no AWS access was used |

**Measured CPU by role (CPU-h, holds):**

| Role | CPU-h | Holds |
|---|---|---|
| A (lead) | 3.178 | 44 |
| B | 0.055 | 4 |
| C | 0.020 | 22 |
| D | 0.233 | 7 |
| E | 0.023 | 7 |
| F | 0.501 | 20 |

**Lead's stages (CPU-min):**

| Stage | CPU-min | Notes |
|---|---|---|
| Assessment | 90.3 | includes the 5 duplicated units, about 4 CPU-min |
| Inner audits | 79.9 | |
| Controls | 13.6 | |
| Fits | 1.7 | 48 new units plus 33 parity checks |
| Tests | 3.5 | |
| Inference | 0.9 | |
| Selection, with inner validation | 0.4 | |
| Admission | 0.2 | |
| Reports, deployment, the evaluation lock, closeout re-checks, the final refresh and post-lock reporting | 0.3 | |

Synthetic timing before fitting projected 0.028 CPU-h for fits (0.056 with the ×2 margin) and 2.89 CPU-h for audits
plus assessment (TIMING.json; 2.83 after the qpc calibration in AUDIT_COMPUTE.json). Actual cost was 0.03 and about
3.1 CPU-h.

**Non-zero exits.** These were all development or test runs, never a scientific stage.
- C: 13, review-fixture runs before the patches landed.
- F: 3, one self-test crash and two failing closeout-test runs, all fixed.
- E: 1.
- A: 1, a pytest run with an inverted fixture sign, fixed in the fixture, not the code.

## 2. Timeline (UTC, 2026-10-06)

| Time | Event |
|---|---|
| 16:44:45 | Session start |
| 16:54:10 | SOURCE_ADMISSION_LOCK pushed (5cc8ecb); admission ran 12 s later |
| 16:55:08 | PREDICTIONS pushed (357b300), before any fit |
| 17:20:21 | FIT_LOCK committed, then pushed and remote-verified (bc0022f) |
| 17:20:34 | AUDIT_AND_SELECTION_LOCK committed, then pushed and remote-verified (25d20f2; verified by 17:20:40) |
| 17:20:47–17:22:29 | Fit: endpoint parity 33/33, then 48 new fits |
| 17:22:44–18:01:53 | Inner audits (2 shards) |
| 18:02–18:03:56 | Composed source audits |
| 18:04–18:12:21 | Controls (all_ok) |
| 18:12:55 | Selection written |
| 18:19:40 | Independent selection verification: PASS |
| 18:23:51 | EVALUATION_LOCK pushed (7611a77) |
| 18:24:01–19:10:00 | Single assessment opening: 48 units, 2 load-balancing workers, 5 duplicates |
| 19:11:58 | Inference complete |
| 19:18:32 | Same-device backup and restore |
| 20:05:57–20:59:04 | Phase 3 independent verification: overall WARN (26 PASS, 2 WARN, 0 FAIL; no result affected) |
| 20:59 | Copy refreshed (role F) |
| 21:01:32–21:01:45 | Lead's closeout re-checks (`status`, `all --dry-run --plan-only`, `refresh --dry-run`); drive not mounted; nothing written |
| 21:16–21:22 | Post-lock reporting completion (`cbp.report_post`) |
| 21:41:03 | Final copy refresh (lead); last semaphore hold |

## 3. Custody

**Status: `LOCAL_SAME_DEVICE_COPY_VERIFIED_OFF_DEVICE_BACKUP_PENDING`** (BACKUP_VERIFICATION.json, RESTORE_INDEX.json).
- **Copy.** `<PRIVATE_CACHE>/cbp_v1_local_copy_20261006` holds 1,144 files, 635,850,300 bytes after the final refresh at 21:41:03Z.
  - It contains the 1,143-file store and the bundled pinned input `adult_jcv.npz`, which the cbp store did not hold.
  - Its SHA256SUMS has sha256 a6dfb0ddb0ba247a6da17174d9120fc0c01363c1de56dacdbf65df20a678ff97. Earlier sums are kept
    beside it, and nothing was deleted.
  - The copy's ledger includes every semaphore hold except the final refresh's own release record.
- **Checks on the copy.**
  - Every file was re-read uncached (F_NOCACHE), with 1,144/1,144 matches. This was an uncached read, not a physical
    cold read.
  - `shasum -c` exited 0 on all entries.
- **Restores from the copy alone, all bitwise.** Teacher U and RAW-J (seed 1), Q, P\*, the J\* fallback and the selected
  pair attacker (HGB, 3 attacker seeds).
- **Off-device backup: PENDING.** The external drive is absent: 0 candidate volumes. The 2 volumes skipped by name
  include the unrelated "BackgroundSyncService Setup" image, which was never read.
- **qpc / dpc / osf / smf custody: PENDING**, as before. It is recorded in `provenance/qpc_custody/STATUS.json` with the
  documented command.
- **The single command to finish everything once the drive is mounted.** The drive is found by content:

  ```sh
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. <python> -m cbp.sema --label F:closeout -- \
      env OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m cbp.closeout all --targets <PRIVATE_CACHE>/cbp_v1/run/closeout_targets.json
  ```

- **Retention.** Nothing was deleted anywhere.
  - The 5 duplicated assessment units are retained in `run/quarantine_assess_duplicates/` and are bitwise equal to the
    kept units.
  - Originals and unique outputs are untouched.
  - Other studies' stores were only read.
- **Privacy.** Per-person predictions, fitted attackers, policy files (fitting-row statistics and SEX-dependent
  groupings), checkpoints and sensitive labels stay in `<PRIVATE_CACHE>/cbp_v1`. Public files carry aggregates, code and
  sanitized hashes. Every push was preceded by a scan for home-directory paths, user names and drive names.

## 4. Process inventory at close

- **Study processes.** No study worker, waiter, watchdog or replay process is running: `ps` shows no `cbp.run`,
  `cbp.assess`, `cbp.sema`, `cbp.infer` or `replay_cbp`. The semaphore has 0 open holds.
- **Agents.** Agents B, C, D, E and F have finished. The read-only document-review workflow has finished.
- **What was not done.** No broad process kill, container restart, account-wide cleanup or instance action. Only this
  study's own background waits were used, and all have exited.
- **Exposure.** No new state or year was opened, and no confirmation data was spent. OSF_DEVELOPMENT_ASSESSMENT was
  opened once, through the pushed EVALUATION_LOCK.

## 5. Decision-document review

An adversarial, read-only review checked RESEARCH_DECISION.md, ADVISOR_BRIEF.md and PAPER_ADDENDUM.md. It ran in
four dimensions (numbers in the decision, numbers in the brief and addendum, claim scope and wording, completeness). Each
dimension had a checker and an independent skeptic. The reviewers ran no study code and held no semaphore slot.

- **Scale.** 579 numbers and claims were checked. 56 findings were raised: 49 confirmed by the skeptic and 7 refuted.
- **Corrections applied to the documents.**
  - Unsupported or wrong numbers:
    - the brief's "about 3 CPU-h in total" is now the measured 4.0 CPU-h;
    - "within 0.003" is now 0.0031;
    - LOCAL λ 0.01 is no longer described as a stronger candidate;
    - the P24–P25 clause kinds are correctly named.
  - Mixed baselines: the λ 0.1 trade-off is now stated with its excess over U and over T\* (0.008 vs 0.005).
  - Overreach:
    - the λ 0.1 reduction is no longer called "useful", because its confidence was not established;
    - "almost linearly" is removed, since the slopes are family-dependent (Figure 1b);
    - the descriptive JOINT λ 0.04 contrast is labelled post hoc and context-only;
    - headroom is no longer credited with confidence;
    - "decision floor" is restated as an information statement (MATH_REVIEW DP3/DP5), not an AUC bound.
- **Additions.**
  - Per-claim statuses beside the label in the brief and the addendum.
  - The full exposure caveat.
  - A novelty and claim-scope section (no novelty; not the Taylor solver; JOINT not compute-matched).
  - Reference accuracies plus income Brier columns in the comparison table, with a note that the references fail the
    contract.
  - A diagnostics paragraph (states vs recovery, optimiser receipts).
  - The descriptive joint-vs-C_rate result.
  - An answer to "what remains unknown and what was backed up".
  - T\*'s per-seed confidence values.
  - The runnable J\* fallback and U baseline.
- **Completeness, as separate files.**
  - RUN_STATUS.json was brought up to date.
  - The role-named shared handoffs were written.
  - A post-lock reporting completion, `cbp/report_post.py`, only ADDS files; the locked `cbp/report.py` and its outputs
    are unchanged:
    - `ENDPOINT_RECEIPTS.json`: prompt §13 diagnostics of the 9 task-only reference units, copied from the aggregate
      endpoint-parity blocks;
    - `INNER_STATES_VS_RECOVERY_v2.csv`, which adds Q, skipped by the locked reader;
    - `figures/fig1b_inner_tradeoff`, the direct privacy-vs-confidence panel;
    - `figures/fig2_annotated`, `figures/fig3_annotated` and `figures/fig4_annotated`, which add legends, recipient
      names, a zoom inset, and Q in Figure 4.
  - No endpoint, selection, label or scientific procedure changed.
- **Refuted, so no change.** Seven findings, mostly premises that became out of date once VALIDATION.md and the PHASE_3
  verification were written.
