# Validation

## Before nonzero fits

| Check | Result |
|---|---|
| Data and roles | `provenance/role_check.py` recomputed the partition independently and matches `smf.data.load()` on all 8 roles and subroles, by row-id sets and hashes. The numeric refit was verified against raw rows regenerated with the pinned loader. Assessment labels are −1 at load. The closed pools (old assessment, refreshed assessment, cert, exclusions) are absent. |
| Design review (`MATH_REVIEW.md`) | Four REQUIRED defects in the first `smf/train.py`, each with a failing fixture, were fixed before Phase A: **R1** zero-direction steps were under-counted; **R2** the controller's linear reader was blind to a rotated 1e-6 clue; **R3** gradient archive fields were missing; **R4** critic optimizer state and w_hyp were not saved. 63 reviewer fixtures pass, and 14 injected defects were each caught. Bounded prior-art check: Madras et al., Agarwal et al., Cotter et al.; no novelty claimed. |
| Tests | 108 pass: pipeline 14, math review 63, audit 29, late 2. |
| Parity | 36/36 bitwise. ρ = 0 / β = 0 equals the task-only continuation for RAW-J, RAW-L, NJ/NL × 3 schedules and J-F/L-F/J-N/L-N with controller probes running. Rerun under the Phase A lock; the pre-review receipts are kept. |
| Audit pre-lock (`AUDIT_PRELOCK_CHECKS.json`) | On U (inner roles) a rotated 1e-6 clue survives serialisation and is detected (held-out AUC 0.885). The selection-aware null on a separate split is 0.494. Planted leaks are detected. |
| Controller preflight (`PREFLIGHT.json`) | Intended +0.01 initial violation; response within one update; selectivity; exact local common-weight symmetry; joint direction change; asymmetric allocation in 5/6 real 2-epoch runs. One check failed because of a real zero-direction event at a reference, not the controller. No repair was authorised; the reviewer's multiplicative step was not adopted (PROTOCOL §11). |

## During the run

| Check | Result |
|---|---|
| Ordering | Every named lock was pushed before the stages it governs (see `RESEARCH_DECISION.md`). EVALUATION_LOCK was pushed at 01:33:58Z; the first outer unit started at 01:33:59Z. |
| Training health | 0 nonfinite steps, 0 rescues. Clip hits: 2 (normalized arms) and 18 (raw arms). Zero-direction events concentrate in REFRESHED at ρ ≥ 0.75 (`GRADIENT_MATCHING.csv`). |
| Audit controls (inner roles, assessment stage) | J-F, L-F, U and F: `all_ok = true` (`AUDIT_CONTROLS.json`). |
| Deployment | `smf.deploy` reproduces the descriptive J-F and the C\* = RAW-J releases bitwise from the 83 permitted columns, and refuses an 84-column input. |
| Privacy | The SEX prior is published only as a hash (provenance review). Every push was gated by a staged-file scan for home, drive and personal names. |

## Backup and restore (`BACKUP_VERIFICATION.json`)

**Backup.** About 01:52Z: 2,412 private files copied to `<DRIVE_ROOT>/private_smf_v1_20261005`, and all 2,412 re-read uncached and matching. This was an uncached read, not a physical cold-disk read.

**Restore from the drive copy alone** (runner's closeout):
- U (`tl__s1__e40`), J-F (`B__s1__J-F__r0.25__e20`) and L-F (`B__s1__L-F__r0.75__e20`) rebuild bitwise.
- The recorded AUC-selected attacker of `outer__s1__J-F` (DA_MLP, pair), refitted from the drive-copy release, reproduces its saved predictions (maximum difference 0).

**Custody gap.** The external drive was **disconnected after the backup**; `/Volumes` no longer lists it. Two consequences:
1. The independent verifier's own restore test from the drive is **PENDING**. It is implemented and runs with `replay_smf.py --drive-root <drive copy root>` once the drive is reconnected.
2. One private log written after the backup (`run/closeout_backup.log`), and the watchdog stop line added at closeout (`run/watchdog.log`), exist only on the internal disk.

No result file is affected.

## Independent replay (`INDEPENDENT_VERIFICATION.json`, `verification/replay_smf.py`)

**Result: 0 FAIL.** Top level: 11 PASS, 1 WARN, 1 PENDING, 1 INFO. All nodes: 81 PASS, 3 WARN, 1 PENDING, 1 INFO.

The final JSON comes from the verifier's script, run unchanged by the lead at 02:06:39Z (worktree HEAD 5a1a9e4). The verifier's own run gave the same counts.

**Independence.** The verifier is a separate agent. An import guard refused all of `smf`, `rgj`, `jcv`, `pnx`, `oar` and `stored_model_eval`, including through unpickling. Assessment labels stayed sealed until the lock check.

**Reproduced (PASS):**
- **Roles.** Matched with an exact-rational hash rule.
- **Releases.** 165 units rebuilt with its own forward pass are bit-exact, including official LEACE maps; 180 head refits are exact.
- **Inner records.** All 165 match.
- **Gradient matching.** 90 runs, all per-epoch identities against ρ·s. ONLINE_MATCHED extra updates equal the REFRESHED receipts per bank on all 18 units.
- **Controller replay.** 36 runs, 756 rows, every applied and hypothetical weight. The INACTIVE_OR_ALIAS local run is parameter-identical to its twin.
- **Selection.** Phase A and Phase B, every status, unit, guard and shortfall.
- **Endpoints.** All 18 primary and 34 secondary slots, and 954 levels and SEs to 1e-10, so the bootstrap draws are reproduced. z, aliases and the prior hash also match.
- **Conjunctions.** Both claims NOT_ESTABLISHED.
- **Integrity.** 583 COMPLETE.json files.
- **Critic tracking.** 324 rows.

**Warnings (process; no affected evidence found):**
1. **`smf/train.py` was re-locked with a changed hash** by PHASE_A_PROTOCOL_LOCK, without a dated amendment.
   - The change applied review fixes R1–R4: zero-direction counting, the controller reader, logging, and snapshot contents.
   - The warm starts, U and the raw controls (45 units) were produced under the DATA_AND_ENGINEERING_LOCK hash. The warm starts use `jcv.train` and the raw arms use `rgj.train`, both unchanged.
   - Before the Phase A lock, the lead recomputed U at epochs 20 and 40 on all three seeds with the fixed code: **bitwise identical**. The 36 parity receipts were rerun under the new lock.
   - The fixes appear in the Phase A lock's commit message, but a dated amendment would have recorded them more cleanly.
2. **AMENDMENT_A3 timing.** A3 (the preflight crash fix) was written at 01:18:09Z, but pushed at 01:20:28Z, after the three final preflight receipts had run (01:18:46–01:19:43Z). Phase B started only after the push, at 01:20:52Z.

**Information items.**
- 22 runs realize less than 0.9·ρ RMS strength. Every case is explained by zero-direction or cap events, so equal strength holds only on steps with a nonzero protection direction.
- The FARE inner results are identical across seeds, because the official trees do not depend on the seed here, so the F/F0 seeds are not independent replicates.
- The verifier found and fixed a sign error in its own LLR level. The runner's values were correct.
