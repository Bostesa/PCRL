# Validation

## Independent verification (`verification/replay_dpc.py` → `INDEPENDENT_VERIFICATION.json`)

**Independence.** The verifier is the sixth role, with its own code. An import guard refuses every study package (`dpc`, `osf`, `smf`, `rgj`, `jcv`, `pnx`, `oar`, `stored_model_eval`), including imports triggered while unpickling heads. The run asserts none of them was loaded: **PASS**.

The verifier read the lead's source only to learn file formats and fixed rules, and executed none of it.

**Phase 1 (fitting side): 547 PASS, 0 FAIL/WARN.**

| Area | Check |
|---|---|
| Roles | Exact-rational group-hash recomputation |
| Teachers | Own forward pass through the admitted `model.pt` and the deployed heads, bitwise against every teacher unit |
| References | LEACE and FARE: own application and tree traversal |
| Fine partitions | Own KL k-means |
| Policies | Every one of the 258 policy units: assignment, prototypes, smoothing, canonical tokens, fingerprints, every person's token/probability/decision; objectives D, I and F match the receipts to ≤ 1.5e-14 (search traces ≤ 2.2e-14) |
| Class preservation | Every row of every role, plus adversarial synthetic rows |
| Search traces | Every greedy merge, exchange move and sweep replayed from scratch |
| Joint dominance | Checked |
| Sequential stage-one rule | Checked |
| Aliases | 10 groups |
| Chronology | Locks before stages; predictions before fitting |
| Mutation power | Checked |

**Phase 2 (selection and assessment): final full run of 463 s on one process, generated 2026-10-06T02:39:30Z, with 0 lead workers at its start and end.**

| Area | Result |
|---|---|
| Inner audits | All 273 recomputed (17,658 stored candidates); selected attackers and inner utilities match (≤ 3e-16) |
| Selection | Own implementation gives identical statuses: J\*, P\* NO_FEASIBLE_NOMINEE; T\* = C_global = SRC\|U; C_match NO_FEASIBLE_CONTROL at all 6 cells. Also identical: deployable compact (none), composed winners, 23 scored labels |
| Evaluation lock | Statuses, roles, 69 unit file maps, 104 code hashes, SEX prior hash, assessment role and composed-policy rule all match. Assessment labels were read only after the lock was confirmed unmodified and on origin |
| Outer units | All 69 completed after the lock push. Labels and groups equal the verifier's; every probability and decision is bitwise equal to the verifier's own releases; row losses, AUCs and fallback counts match |
| Endpoints | All 33 primary slots and all 1,070 levels equal `inference.json` with zero difference (B = 1,999, seed 20261007, z = 3.1717657833516224). Clause counts are 8 / 9 / 9 of 11; claims A, B and C are NOT_ESTABLISHED. A wrong bootstrap seed or row-level resampling would fail the SE check |
| Attacker refits | All 54 candidates refit for `outer__s1__U_FINE-TASK_m8`; selections, fallback rules and assessment predictions are bitwise equal at seeds 0–2. The J\* fallback and SRC\|U selected attackers (including a composed candidate) are also bitwise equal |
| Controls | Null split, permutation hash and threshold (0.5654) match; coverage complete (28 releases, 5 plants per policy); null and CONF_r1 replay match; 0 of 15 null exceedances |
| Amendments and chronology | A1 and A2 change only their declared files and were pushed before the affected rerun. The lock was written after A2; the assessment opened 13 s after the lock push. No post-lock amendment; locked code unchanged. The 01:45Z resume tests pass the chronology check |
| Restore | INFO: same-device copy. All 2,364 checksums match. The teachers, the two policies and the pair attacker rebuilt from the copy are bitwise equal. **Not a drive restore** |
| Reports | All 138 utility rows agree |

**Totals (final run): 560 PASS, 2 INFO (custody, restore replay), 0 WARN, 0 FAIL.** Phase 1 PASS, Phase 2 PASS, independence PASS.

**The earlier WARN** (first Phase 2 run, 02:08Z) concerned the decision documents: 186 of 197 quoted numbers held, and 11 were rounding or transcription slips of at most 0.0025 each. The lead corrected all 11. None changed a conclusion, a decision or the label. A full re-run confirmed all 11 corrections and the claim-A disclosure, and found one further slip in the plain-language summary ("0.052" for P12 = 0.0515, now 0.051). The final re-check at 02:39Z confirmed all corrections, including the 0.052 → 0.051 fix and the income wording. The decision-documents check now passes 203 of 203.

The check records the sha256 of the files it read: RESEARCH_DECISION.md, PAPER_ADDENDUM.md, ADVISOR_BRIEF.md and RUN_STATUS.json. RUN_STATUS.json was edited once after that check, only in its `independent_verification` field, to record these final totals. Its `validity_note` is unchanged, so its current hash differs from the recorded one.

**Verifier corrections.** Nine corrections were made to the verifier's own comparison logic, all recorded in the JSON.

## Validity caveat (verifier note, adopted)

C_match is NO_FEASIBLE_CONTROL at every teacher/rate cell. Two readings apply:

| Reading | Claim A | Label |
|---|---|---|
| Locked per-claim rule (review R3, applied before the selection lock) | Valid NOT_ESTABLISHED | EXPERIMENTAL_NO_ADVANTAGE |
| Strict reading of PROTOCOL §8, "a missing comparator invalidates the dependent claim" | Invalid | INCOMPLETE_OR_INVALID |

The locked label is kept. No claim passes under either reading, because no JOINT policy is task-eligible. See `RESEARCH_DECISION.md`.

## Lead-side checks

| Check | Result |
|---|---|
| `dpc/tests` | 158 passed (50 s), including the AMENDMENT_A1 regression and the reviewer's 65 tests |
| Math review | 0 REQUIRED; R1–R3 applied before the selection lock; 52 of 52 non-equivalent injected defects caught |
| Real-data controls | All ok: 0/15 null exceedances; CONF, COLL and XOR plants detected on 3 policies; a decisions-only audit misses the confidence plant |
| Class preservation (`CLASS_PRESERVATION.json`) | 258 units, 20,211,720 recipient-row checks, all pass |
| Deployment (`QUICKSTART.md`) | Bound policy reproduces the stored release bitwise; 84 columns, reordered schema, fine-ID/raw-score export flags and a mismatched teacher are all refused (exit 2, nothing written) |
| Backup | Same-device copy, 2,364 files re-read uncached; restore checks 7/7 PASS |

## Not validated

- **Off-device custody.** The drive was absent.
- **Predecessor (osf/smf) drive restore.** PENDING.
- **Population validity.** Rows are reused, and no confirmation population was opened.
