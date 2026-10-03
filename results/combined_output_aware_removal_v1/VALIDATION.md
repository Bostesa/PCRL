# Validation

## Before any new fit

| Check | Result |
|---|---|
| Input admission | Every admitted input and checkpoint matches its sha256. All 9,717 reused benchmark outputs match the drive SHA256SUMS. |
| Registered predictions | PREDICTIONS.md, committed at f70bad3 before any fit, and scored in RESEARCH_DECISION.md |
| Exposure rule | Committed at 7a4a086 before any sensitivity result |
| Execution lock v1 | Pushed at 3130c8d before the first new fit. It pins the code, the dependencies, the official FARE tree (sha256 56a44700…), the inputs, the LEACE maps, the roles and support, the grid, the 12-row family (asserted) and the runtime. |
| FARE admission | Official commit 89cb1b66. The reproduction gate is bit-exact. 12/12 wrapper tests pass on synthetic fixtures. |
| Study tests | 17/17 pass: synthetic units, plus-aliasing, heads, the cell-conditional attacker, controls, the end-to-end FARE stage, and the regression test for L1 |

## During and after the run

| Check | Result |
|---|---|
| Technical failure | HMDA attempt 1 stopped before any fit with a map-id bug: the policy-set LEACE map name used sorted instead of declared order. Repaired with a regression test, re-locked as amendment L1 (7244537), and resumed. Adult was unaffected. |
| Units | 372 of 378 planned completed and hash-verified. 6 were not fitted because of 3 FARE aliases. 0 failed scientific units, 0 budget-unrun. |
| Real-data controls (seed 0, attacker_fit / attacker_val only) | 14/14 shuffled-label nulls at 0.47–0.53, none flagged above 0.55. 14/14 planted leaks detected at 0.90–1.00. |
| Selection hygiene | FARE nominees, σ\* and the view-3 heads use validation or defense_fit holdouts only. The verifier recomputed the nominees from validation data alone. |
| Custody of the original study | 24/24 primary endpoints reproduced to 1e-12 with identical decisions |

## Independent replay

`INDEPENDENT_VERIFICATION.json`; `verification/replay_oar.py`. It uses numpy only for scoring and does not import
`oar.*` or `stored_model_eval.*`.

**Totals.** 462 items: **458 PASS, 4 FAIL, 0 MC_BORDERLINE.**

**Agreement:**
- **Primary: 12/12.**
  - Points agree within 2.2e-16 and bounds within 3.1e-16; the mirrored bootstrap draw reproduces the bounds
    bit-for-bit.
  - Every decision is identical: 10 PASS and 2 NOT_ESTABLISHED (Adult P3, P4).
  - The competitive conjunction matches: HMDA yes, Adult no.
  - The nominees, recomputed from validation data only, match: Adult 4/2/4, HMDA 4/4/5.
  - The aliases match an independent comparison of `cells.npy`.
- **Exposure sensitivity: 24/24** endpoints agree within 5.6e-16, all labelled STABLE. As its own code check, the
  replay also reproduces the 24 original benchmark endpoints.
- **Corrections:**
  - C1: 50/50 rows agree. Re-pairing the untreated arm over seeds {0, 1, 2} reproduces the original wrong values
    exactly, which confirms the diagnosis.
  - C2: 435 unit points agree (max difference 1.1e-15), and so do the 9 corrected rows with their intervals.
- **View-3 heads:** all 24 reproduce their saved outputs exactly from the protected features alone. Their input width
  equals the feature width, and their fit and selection roles are the defense_fit holdout.
- **Roles, leakage, membership, noise and controls:**
  - no assessment row in any fit or validation set;
  - cert rows disjoint from all other roles;
  - all three encoder seeds present;
  - identical rows on both sides of every comparison;
  - σ\* = 2.0 / 4.0 with release seeds 0, 1, 2.

### The four failures

All four are reporting defects. None changes a number or a decision.

| Item | Cause | Correct values |
|---|---|---|
| Adult `excluded_exposure_rows` / `exposure_groups_removed` in ROLES_AND_SUPPORT.json show 0 | `oar/study.py` stores roles as 16-character strings, which truncates the 17-character label "excluded_exposure". The rows are still excluded from every scored role. | 17 rows / 17 groups: attacker_fit 6, attacker_val 4, assessment 7 |
| HMDA, same field | Same cause | 42 rows: attacker_fit 21, attacker_val 7, assessment 14 |
| Adult s0 `cert_rows_with_feature_vector_equal_to_a_fit_row` in amendment A1 reads 4 | The wrapper's `row_hashes` deduplicates, so the field counts distinct vectors, not rows | 8 rows (4 distinct vectors) |
| HMDA s1, same field, reads 1 | Same cause | **1,138 of 1,385** cert rows. HMDA s1's representation is heavily collapsed: 248 distinct cert vectors. That seed's amended bound is 3.18, which is vacuous. |

**Handling.** These were not "fixed" in the locked code after the results were in. The correct values are reported
here and in NATIVE_TEST_VS_RECOVERY.md and EXPOSURE_SENSITIVITY.md.

**An interpretation correction prompted by these counts.** The rows whose representation matches a training row are
**seed-dependent representation collisions** (largely collapse), not repeated input records. An earlier draft had said
otherwise; EXPOSURE_SENSITIVITY.md was corrected.

**Bugs in the replay itself.** The verifier found and fixed two bugs in its own replay during the run:
- an exposure-label definition that did not follow the rule's "same decision" definition of STABLE;
- an MC-borderline test that compared against the wrong tail.

Each was investigated first, and the run was re-run on the final code. The synthetic self-test passes on that code:
clean baseline, and 20/20 planted defects caught.

**Replay cost.** The final run took 1,084 s wall (about 1,080 CPU-s) with 1.7 GB peak memory. The 4 verifier runs
used about 73 CPU-min in total.

## Backup and restore

- **Drive copy:** `<drive>/private_oar_v1_20261003/oar_v1`, 30,624 files.
- **Read-back:** every file was re-read from the drive with F_NOCACHE: **30,624/30,624 sha256 match.**
- **Restoration from the drive copy alone:**
  - **Defense:** the FARE nominee tree for HMDA s0 (config 4) encodes to cells identical to the saved ones.
  - **Head:** the release head on FARE features reproduces its saved outputs (max |diff| 0.0).
  - **Attacker:** the nonlinear attacker on FARE + head reproduces its saved predictions (max |diff| 0.0).

  See `ARCHIVE_INDEX.json`.
- Nothing was deleted, and the originals are retained.
