# Validation

## Independent verification (`verification/replay_qpc.py` → `INDEPENDENT_VERIFICATION.json`)

**Independence.**
- The verifier is the sixth role and uses its own code.
- An import guard refuses qpc, dpc, osf, smf, rgj, jcv, pnx, oar, stored_model_eval and pcrl, including imports triggered while unpickling heads. The run asserts that none was loaded: **PASS**.
- It implements forward application, KL k-means and search replay, metrics, attacker refits, selection and inference separately. It read the study code only to learn formats and fixed rules.
- Every real-data replay ran under the shared semaphore.

**Phase 1** (roles, teachers, references, A1, A2, gate; 0 FAIL).
- Roles match both pinned manifests.
- All 6 teachers are bitwise equal to the qpc units, the dpc units and the admitted releases. All 9 references are bitwise.
- A1: the own 20-round rule reproduces dpc DIRECT-TASK m8 bitwise, including per-class receipts.
- A2: all 216 start receipts match (partitions, k-means++ draws, stop reasons, passes, objective trajectories within 2.5e-13).
- All 24 releases are bitwise. Class preservation holds on every row of every role and on adversarial rows.
- The gate decision is identical: CAPACITY_GATE_MET at i8o64 only. CONVERGENCE_DIAGNOSTIC.csv is exact, and CAPACITY_CURVE.csv is within print precision.
- Mutation power: 9 of 9 planted defects detected.

**Phase 2A** (Stage B; 0 FAIL).
- Fine partitions, start fingerprints and every row's assignment are exact.
- The search replay covers all 42 units: 15,948 merges plus 12,263 moves and sweep merges, every increment exactly equal.
- Sequential stage one uses F_joint with the class-only counterpart. The old D + 1.5λI rule would have chosen a different stage-one map in all 18 sequential units.
- All 9 JOINT winners match, with dominance over the unchanged witnesses.
- All 42 releases are bitwise. D, I1, I2 and I12 match within 1.4e-14, and the permutation-null MI within 1.7e-15.

**Phase 3** (inner audits, selection, controls, assessment, inference, tables; 0 FAIL).
- **Label gate:** EVALUATION_LOCK (4e92ce4) was verified on origin before any assessment label was read.
- **Inner audits:** all 81 (7,290 candidates). AUC and CE, banks and dual selection are exact. The composed SRC|U bank covers all 22 codes on every seed.
- **Selection:** the verifier's own §11 implementation gives identical statuses.
- **Attacker refits:** 108 of 108 bitwise equal (P*, J* fallback, C_rate, T* and SRC|U including composed winners) on INNER_SELECTION and assessment rows.
- **Controls:** own split, permutation hash and null threshold (0.56541); all 49 pass rules and the 15 null-calibration rows replay.
- **Outer units:** all 57 bitwise.
- **Endpoints and tables:** all 37 endpoints and 1,214 levels equal to 0.0. The label is CONFIDENCE_FEASIBILITY_ESTABLISHED under both rules. Published tables match (1,685 cells).

**Final phase** (PHASE_FINAL; every earlier check re-run in the same process): 155 PASS, 1 INFO, 3 WARN, 0 FAIL, 0 PENDING.
- **Deployment parity:** the input equals the verifier's own 83-column X. Its own forward pass equals the teachers. Q* and P* applied from `policy.json` equal the stored releases on all rows and seeds, and the lead's `qpc.deploy` outputs are bitwise equal with exactly six arrays.
- **Restore parity** from the first same-device copy (925 entries): the teacher U s1 forward pass, both codes and the selected pair attacker (HGB) refit all have max difference 0.
- **Decision documents:** 151/153 fragments match. The two slips (a CPU figure 201 → 202 and the AMENDMENT_A1 push time) are corrected in COST_AND_CLOSEOUT.md. A wording clarification ("seed means" for the guard numbers) was added to RESEARCH_DECISION.md §7 after the check.
- **The third WARN** is the document slips themselves, now corrected.
- **Verifier compute:** 18 semaphore holds, 3,426 CPU-s, peak 2.03 GB.

**WARNs (chronology; no result affected).**
1. **Two semaphore holds without a release record.**
   - The math-review "C:mutation" wrapper was killed at about 05:12:26Z. The reviewer confirmed no child outlived it.
   - The lead's "A:sematest" was a `sleep` test of the first AMENDMENT_A1 draft, which deadlocked; its child was gone before the wrapper was killed.
   - Both slots were re-acquired later. A per-slot reconstruction shows at most 2 concurrent heavy holds for the whole run.
2. **Activity after the assessment opened.**
   - A 2.7 s no-op `fit` resume at 06:08:31Z, the QUICKSTART test; no unit was created or changed.
   - The presentation module `qpc/report_outer.py` is not in EVALUATION_LOCK. The verifier checked every number it publishes.

## Math and protocol review (`MATH_REVIEW.md`)

- **0 REQUIRED open.** Four REQUIRED findings were fixed and re-tested before their governing locks. Two were runner bindings and two were protocol and code text (SA-R1, SA-R2, SEL-R1, SEL-R2).
- **Independent re-implementations** check:
  - k-means++ and convergence;
  - search deltas against brute force;
  - the sequential correction;
  - JOINT witness dominance;
  - selection on 60 random banks;
  - the composed-bank rule on 40 banks;
  - the label truth table, exhaustively.
- **Optimiser gaps** (tiny exhaustive fixtures): FINE-TASK and LOCAL are globally optimal in 12 of 12. JOINT is in 10 of 12, with gaps 0.0025 and 0.0058; the XOR case needs a coordinated two-cell move.
- **Mutation testing:** 74/74 non-equivalent injected defects caught by the post-lock test file (76 injected, 2 equivalent). The locked test file alone is weaker.

## Lead-side checks

| Check | Result |
|---|---|
| `qpc/tests` | 246 passed (run ending 07:00Z): admit 22, audit 31, closeout 26, late 2, method 40, math review 49 (locked) + 64 (post-lock), pipeline 6, truth table 6 |
| Class preservation (`CLASS_PRESERVATION.json`) | 66 policy units, 5,170,440 recipient-row checks, all pass |
| Real-data controls (`AUDIT_PRELOCK_CHECKS.json`) | All ok: 0 null exceedances; CONF, COLL, XOR and ROT plants detected; a decisions-only audit misses the confidence plant |
| Deployment (`QUICKSTART.md`) | Q* and P* reproduce the stored releases bitwise. 84 columns, a reordered schema, fine-ID and raw-score exports, a mismatched teacher and unknown flags are all refused (exit 2) |
| Backup | Same-device copy; restores of both teachers, Q*, P* and the selected attacker from the copy alone all PASS. **Not off-device** |
| Locks | Every stage started after its governing lock was pushed. A stage refused once (05:16:07Z) because a locked test file had changed, which is the intended behaviour |

## Not validated

- **Off-device custody:** the drive is absent. The dpc off-device backup and the osf/smf custody repair are pending.
- **Population validity:** the rows are reused, and no confirmation population was opened.
- **Global optimality of the search:** JOINT is a local search.
