# Validation

## Before any real fit

| Check | Result |
|---|---|
| Inputs | The admitted predecessor input `adult_jcv.npz` was reused (SHA-256 `e0d9e54a…`). It has 39,205 rows and the 83-column corrected input contract. All 8 role row-id hashes are recorded in `SELECTION_LOCK.json`. No new data, state, year or population was opened. |
| Mathematical review | Independent and read-only (`review/MATH_REVIEW.md`). All 7 REQUIRED items were applied before the lock: R1 (the critic diagnostic evaluates θ_{T−1} with the saved whitener), R2 (primary statistic registered), R3 and R4 (overclaims about PN vs JP and PN vs LN corrected), R5 (`valid_reference`), R6 (fallback restricted to nonzero β) and R7 (comparator status and unit recorded). Advisories A1, A2, A4 and A6 were also applied. |
| Fixtures | 6 tests pass (`pnx/tests`): β = 0 bitwise parity, fidelity of the copied loop to the predecessor's JP, L and U, no-erasure specifications with an equal 6-critic budget, label mapping, complete bank records, and the θ_{T−1} critic snapshot. |
| Lock | `f1d1b76` was pushed before any fit. `pnx.run` refuses to train unless the lock verifies and parity has passed. It pins 36 code hashes, 117 reused predecessor `COMPLETE.json` hashes, 3 document hashes and the budget. |

## Engineering parity (before any new fit; `PARITY_20261004T000218Z.json`, private)

| Check | Seeds | Result |
|---|---|---|
| U retrained here equals the predecessor's U (environment) | 0, 1, 2 | Model state bitwise |
| PN at β = 0 equals U | 0, 1, 2 | Model bitwise; release bitwise; hard decisions identical; identity map; 0 protection steps |
| LN at β = 0 equals U | 0, 1, 2 | Same |
| Copied loop reproduces the predecessor's JP β = 1 | 0 | Model bitwise |

After parity, β = 0 is an exact alias of U (`nn__s{k}__U`). No β = 0 candidate was selected, so the TASK_ONLY_ALIAS label was not needed.

## During the run

| Check | Result |
|---|---|
| Units | 122/122 rows of `UNIT_MANIFEST.csv` are complete and hash-verified, with 0 failed and 0 budget-unrun. They comprise 18 new fits, 18 inner audits, 39 outer scorings, 18 critic-gap units, the parity receipt, the controls unit, and the hash-checked reused or alias rows (117 alias records). |
| Training health | 18/18 fits ran 1,520 protection steps with 0 nonfinite gradients and no rescue. Task and penalty gradient norms and clip hits are in `TRAINING_DIAGNOSTICS.csv`. |
| No erasure | Every PN and LN unit has an identity map and no `leace_*` directory. This is asserted at finalisation, at deployment and by the verifier. |
| Selection lock | `b0b9256` was pushed before any outer scoring; `pnx.outer` refuses otherwise. Outer scoring started 14 s after that commit. |
| Reference reproducibility | The re-scored U, E, JP, F and F0 reproduce the predecessor's outer predictions bitwise (15/15). |
| Audit controls (attacker_fit / attacker_val only) | On the seed-0 PN, LN and U local and coalition views: label-permutation nulls 0.495–0.513, all ≤ 0.55; planted leaks 0.943–0.971, all detected. |
| Linear diagnostics | `NATIVE_VS_AUDIT.csv` records the fitting-row cross-covariance, OLS R² (with a shuffled-label null of about 0.002) and held-out correlation. They are ≈ 0 only for the erased arms, as expected. |
| Deployment | `pnx.deploy --unit pn__s1__PN__b0.1` reproduces the saved release bitwise (r, centred logits, probabilities and hard decisions for both recipients; re-run 2026-10-04 at closeout). It refuses an 84-column input with SEX appended. |

## Backup and restore

- The drive copy `<drive>/private_pnx_v1_20261003` holds all 561 private files plus `SHA256SUMS` and `BACKUP_RECORD.json`.
- All 561 files were re-read uncached (F_NOCACHE; not a cold-disk unmount) and match.
- No private file changed after the backup (checked at closeout).
- Restored from the drive copy alone, PN (`pn__s1__PN__b0.1`), LN (`pn__s1__LN__b0.1`) and the erased reference E (`nn__s1__E`) reproduce their releases, centred logits, probabilities and decisions with zero difference.
- Nothing was deleted.

## Independent replay

See `INDEPENDENT_VERIFICATION.json` and `verification/replay_pnx.py`. The verifier is a separate agent. An import guard refuses `pnx`, `jcv`, `oar`, `odx`, `cap`, `stored_model_eval`, `report` and `pcrl`, and the verifier asserts that none was loaded, including through pickles. Its only fits are refits of recorded configurations, the seed-0 task-only loop and replays of saved final steps. The whole run takes about 135 s.

**First run (00:41Z): 37 PASS, 0 FAIL, 2 WARN, 5 INFO.**

What reproduced exactly (maximum absolute difference 0 unless noted):
- **Inputs and aliases.** The input hash, all 8 role hashes and all 117 alias records match, including the source `COMPLETE.json` hashes and every listed source file.
- **Deployment and heads.** All 18 PN and LN releases rebuild from `model.pt` and the heads. Refitting the 36 heads gives the same C and the same coefficients.
- **Update rule.** The final training step of all 18 units, replayed from θ_{T−1} with the saved critics and saved whitener, reproduces `model.pt` bitwise. Without the penalty it does not, so the penalty acted.
- **β = 0 parity.** The verifier's own task-only loop reproduces the predecessor's seed-0 U bitwise.
- **Selection.** The verifier's own implementation of PROTOCOL §4 matches 247 fields of `SELECTION_LOCK.json`, and no gate margin is within 1e-12 of zero. The seed-0 PN exclusion is confirmed: its occupation-view inner recovery is 0.8267, against an allowance of 0.8148 + 0.01 = 0.8248. Inner recovery replays to 1.1e-16.
- **Endpoints.** All 18 primary and 30 secondary endpoints were rebuilt with the verifier's own bootstrap. Points, SEs and bounds equal `inference.json`; the CSVs agree to rounding (5e-7). All 48 decisions agree. All 611 level points and all 78 rows of `ACTUAL_TASK_UTILITY.csv` match.
- **Claims.** A and B are both NOT_ESTABLISHED: 8/9 clauses each (P02 and P11 fail), and the status requirement fails on seed 0.
- **Audit banks and attackers.** All 156 coalition banks carry their own table, both ignore-other-view tables and the bank, and the selected view is the argmin every time. The 10 attacker refits and one full 14-attacker table reproduce.
- **Critic gap.** The registered statistic recomputes exactly (positive in 45/45 unit-views and 18/18 units). All 108 online critics re-evaluated at θ_{T−1} reproduce.
- **Integrity.** 211/211 `COMPLETE.json` files, 36/36 lock code hashes, the locked documents and 102 selection-lock artifact hashes all verify.

**Warnings and how they were resolved.**
- **24 (`CRITIC_GAP.csv`).** The public column `gap_online_minus_fresh_def` (committed in `ec1ae45`) was the best-of-bank difference, not the registered mean of paired differences; they differed by up to 0.034.
  - *Fixed.* The CSV now carries `primary_mean_paired_online_minus_fresh_def` (registered), `sensitivity_thetaT_refit_whitener_primary` and `best_of_bank_online_minus_fresh_def`.
  - `CRITIC_GAP_DIAGNOSIS.md` labels its CE columns as best-of-bank, and one rounding error was corrected (PN 0.1 v1: +0.026).
  - The conclusion is unchanged, because both statistics are positive in 45/45 cells.
- **34 (`RUN_STATUS.json`).** It still said `LOCKED_BEFORE_FITS`. *Fixed:* it is now COMPLETE.

**Information items.**
- The FARE purpose-1 releases are identical across seeds, so the F and F0 seeds are not independent replicates. This was already disclosed in the predecessor.
- F0's status comes from its own gates; its pairing to F is recorded separately.
- Checking the JP-replication part of the parity receipt needs the study's LEACE wrapper, which the verifier is forbidden to import, so that item is INFO only.

**Second run (verifier agent, after the fixes): 39 PASS, 0 FAIL, 1 WARN, 5 INFO.**
- Check 24 was rewritten for the corrected columns and passes (max difference 3.5e-7, CSV rounding). Check 34 passes.
- A new check 24b compares every cell of the `CRITIC_GAP_DIAGNOSIS.md` table with seed means from the records. It found two more last-digit double-rounding errors: PN 1 v1 sensitivity, and LN 10 v2 fresh_def CE.
- The verifier also noted that "agree within about 0.01" holds for the seed means but not for single cells.
- *Fixed:* the table is now computed directly from the records, and the sentence gives both maxima (0.011 for seed means, 0.023 for single cells; the sign agrees in 45/45).

**Final run (2026-10-04 00:53Z): the verifier's script was run unchanged by the runner. Result: 40 PASS, 0 FAIL, 0 WARN, 5 INFO (45 checks).**
- 60/60 table values match.
- Claims A and B are NOT_ESTABLISHED.
- The 5 INFO items are the JP-replication receipt, the replayed selection outcome, the outer label map, the per-seed claim-A values, and the FARE seed dependence.
- Check 24's detail says the CSV is an uncommitted change; that was true at run time, and the CSV was committed immediately after.
- Runtime 137 s; peak memory 0.86 GB.

**Documentation corrections found at closeout (runner side).**
- The seed-2 income-view difference is +0.032, not +0.033.
- The "+0.067 vs JP" comparison uses the frozen per-seed JP reference (β = 0.1 on seed 0, β = 1 on seeds 1 and 2), not JP at β = 0.1, where the difference is +0.023. `RESEARCH_DECISION.md` now says so.
