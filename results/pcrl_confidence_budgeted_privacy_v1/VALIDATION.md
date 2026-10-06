# Validation (cbp)

## 1. Independent verification (`verification/replay_cbp.py` → `INDEPENDENT_VERIFICATION.json`, PHASE_3)

**Result: overall WARN; 26 PASS, 2 WARN, 0 FAIL (182 nodes: 180 PASS, 2 WARN). Independence PASS. No disagreement with the lead's
inference, selection or label.**

**Independence (role F).**
- The verifier is adapted from the qpc verifier and uses its own code throughout.
- A `sys.meta_path` guard refuses cbp, qpc, dpc, osf, smf, rgj, jcv, pnx, oar, stored_model_eval and pcrl. This covers
  imports triggered during joblib unpickling.
- Forbidden modules loaded: none.
- The deployment CLI was exercised only as an external process, never imported.
- Every real-data replay ran under the shared semaphore.

| Phase | Time and verdict | What was reproduced independently |
|---|---|---|
| 1 | 17:18Z; 14/14 PASS | <ul><li>Pins: 52 source modules equal the d0c8a45 blobs; 18 source result hashes.</li><li>Role counts: 15,434 / 1,500 / 6,065 / 2,235 / 13,936 rows, 13,929 groups.</li><li>Custody of all 84 admitted units (321 files).</li><li>6/6 teachers and 9/9 references bitwise.</li><li>24 DIRECT-TASK codes bitwise.</li><li>3/3 fine partitions.</li><li>Code-search replay of 42 reused or composition units: 15,948 merges and 12,263 moves, maximum difference 0.</li><li>Class preservation on every row and role, including adversarial rows.</li><li>Predictions pushed before any fit.</li></ul> |
| 2 | 18:19Z; 19/19 PASS; before the evaluation lock | <ul><li>All 48 new fits replayed in full: bitwise tokens, probabilities and decisions; final terms within 1.4e-14 (registered tolerance 1e-12); 12/12 JOINT winners, each dominating its four witnesses.</li><li>33/33 endpoint-parity records agree.</li><li>All 129 inner units (9,882 candidates re-scored, maximum difference 3.3e-16).</li><li>U composition over 38 maps per seed, with matching freeze lists.</li><li>All 49 control rules, at null threshold 0.56541.</li><li>Ordinary and headroom eligibility on all 32 configurations for every seed and task.</li><li>672 role fields with 0 differences: roles, guards, shortfalls, aliases, fallbacks and diagnostics.</li><li>Planted §14 defects (reversed best/worst, headroom at the ordinary limit, seed-averaged eligibility, flipped orientation) each change the result on the real rows, so the comparison would catch them.</li></ul> |
| 3 | run 20:05:57Z–20:59:04Z; 26 PASS, 2 WARN | <ul><li>EVALUATION_LOCK: one version, pushed 18:23:51Z, byte-identical on origin, verified before any label was read. The first assessment hold came 10 s after the push.</li><li>All 48 outer units bitwise.</li><li>All 37 slots: point, SE and bound differences 0.0; exact outcomes and decisions.</li><li>All 1,058 levels; the label and the claim and Q statuses.</li><li>All 7 public tables within print precision, which covers the figure source values.</li><li>Deployment parity, with 11/11 refusals.</li><li>Restore from the same-device copy alone: U, Q, P\*, the J\* fallback and the selected pair attacker (HGB) all bitwise, re-run after the refresh.</li><li>Budget: peak 2 concurrent holds.</li></ul> |

**WARNs (no result affected).**
1. **Assessment chronology.** Two load-balancing workers were added during the assessment. The seed-1 worker overlapped
   shard 1/2 on 5 units, so those units were computed twice.
   - Both copies are bitwise identical.
   - The atomic writer quarantined the first-written copies. They were moved, not deleted.
   - All work used the same lock, sha256 1ff1307c….
   - The overlap is recorded in ASSESSMENT_ATTEMPTS.json, and peak concurrency was 2. It was a scheduling error by the
     lead (a wrong timing estimate), not a rerun with changed inputs.
2. **Attacker refits.** 132 of 135 refits are bitwise. The other 3 (one LR C=0.01 candidate, JOINT λ 0.01 seed 0
   recipient 2) differ by 5.4e-15, inside the registered 1e-9.

**Registered tolerances.**
- Exact: tokens, decisions, partitions, bindings, hashes, decoded probabilities and teacher outputs.
- 1e-12 for objective terms, inner AUC/CE/utility, endpoint points and SEs; 1e-11 for bounds.
- 5e-7 for published 6-decimal prints.
- Exact agreement for verdicts, labels and roles.

## 2. Statistics and selection review (role C; `SELECTION_REVIEW.md`, `cbp/review_tests/test_selection_review.py`)

- **Method.** An independent transcription of prompt §§9, 11 and 14, with no cbp or qpc imports, compared against the
  lead's code on synthetic banks and synthetic saved predictions.
- **REQUIRED findings: 7 raised, all fixed and confirmed by failing-then-passing fixtures before FIT_LOCK.**
  - SEL-R1: the tie-tolerance text.
  - SEL-R2: the JOINT alias tie-break now names the simplest family.
  - SEL-R3: the Q-status classes.
  - SEL-R4: the inclusive guard at the bound.
  - SEL-R5: the missing-guard text versus the code.
  - SEL-R6: a missing guard can no longer be ranked as a zero shortfall.
  - SEL-R7: a NaN attacker score is now INVALID, never ranked.
- **RECOMMENDED: 11 raised.** 10 were adopted before the lock. SEL-C11 (the HEADROOM_VS_STANDARD_SELECTION.csv columns)
  was adopted at 0905d0a, also before the lock, and its fixture now XPASSes.
- **NOTEs: 5.** SEL-N4 is carried into RESEARCH_DECISION §2: the inner SE is about 0.003, the seeds share the inner
  rows, and per-seed guards are noisy. Its further point, that the 0.006 headroom leaves about one assessment z·SE
  before the 0.01 limit, is in SELECTION_REVIEW.md.
- **Final suite at the freeze:** 46 tests; 44 pass, 2 xfail (one NOTE and the then-open SEL-C11).
- **Verdict:** no objection to the freeze.
- **Erratum.** SELECTION_REVIEW.md's "17:24Z" is a clock error. That file was committed in 0905d0a at 17:20:02Z, and
  the last role-C run (C:selection-review-final) ended at 17:19:39Z with exit 0 (SEMA_LOG). Both are before FIT_LOCK
  (17:20:21Z).

## 3. Math and claims review (role E; `MATH_REVIEW.md`, `PRIOR_ART_AND_CLAIM_SCOPE.md`)

- **REQUIRED: 0.**
- **RECOMMENDED: 4, all addressed.**
  - E-R1: endpoint parity now uses witness_records and meta.
  - E-R2: Theorem 1 with assumptions A1–A7, and row checks presented as receipts. This is used in RESEARCH_DECISION §6
    and PAPER_ADDENDUM.
  - E-R3: name the MI null. RESEARCH_DECISION §5 names the qpc fit-receipt null (100 permutations, seed 20261006) and
    quotes no fitted MI value. No null was recorded for the DIRECT-TASK (Q) units.
  - E-R4: the "not the official Taylor solver" sentence was added to PROTOCOL §5.
- **Proofs.**
  - Theorem 1, exact decision preservation and containment, which is family- and λ-independent.
  - The objective decomposition.
  - The sequential correction λ·I(S; d_b | C_a).
  - Theorem 2, JOINT witness dominance on the fitting objective only. It is shown not to extend to components, held-out
    rows or DIRECT-TASK.
- **Prior art (re-checked 16:53–17:05Z).** The PURIFIER repository is empty, and Taylor et al. have released no code.
- **Review tests:** 28/28 pass.

## 4. Engineering tests

`pytest cbp/tests cbp/review_tests` at c9b017a (20:04:30Z–20:05:40Z) gave **187 passed, 1 xfailed (SEL-N3 NOTE), 1 xpassed (SEL-C11,
fixed)** across 189 tests:

| File | Tests | Role |
|---|---|---|
| test_admit | 3 | lead |
| test_truth_table | 7 | lead |
| test_select | 8 | lead |
| test_late | 5 | lead |
| test_fit | 28 | B |
| test_audit | 40 | D |
| test_closeout | 24 at that run; 26 after F's refresh job | F |
| review_tests/test_selection_review | 46 | C |
| review_tests/test_math_review | 28 | E |

The tests use synthetic data only. Deliberate-defect tests (prompt §14) are spread across test_audit (D) and the replay
self-tests (F).

## 5. Real-data controls and integrity receipts

- **Controls (AUDIT_PRELOCK_CHECKS.json): all_ok.**
  - The realised null threshold is 0.5654143765984265, equal to the source's.
  - Null calibration: 0 exceedances.
  - Every positive control passes: CONF, COLL and XOR on three codes; ROT on the U, RAW-J and E interfaces.
- **Endpoint parity:** 33/33 reused units pass every gate, including the 1e-12 relative term gate (run/endpoint_parity.json).
- **Inner validation:** cbp.audit.validate_all over the saved bank checked 129 units, with 0 defects and 0 missing.
- **Class preservation:** 114 policy units, 8,930,760 row checks, 0 failed (CLASS_PRESERVATION.json).
- **Assessment arrays:** the finiteness receipt is all finite (inference.json; LABEL_RESULT.json).

## 6. Corrections and post-hoc items

- **Lock order.** No scientific definition was changed after the locks. FIT_LOCK and AUDIT_AND_SELECTION_LOCK were
  pushed and remote-verified before the first new fit (17:20:47Z). The EVALUATION_LOCK was pushed before the
  assessment.
- **Assessment duplication.** It is recorded above. No selection, threshold or attacker choice was made after the
  assessment.
- **Decision documents.** They were checked by an adversarial read-only review: numbers against aggregate sources, claim
  scope against PRIOR_ART §5.3 and prompt §17, and completeness. Confirmed corrections are listed in
  COST_AND_CLOSEOUT.md §5.
- **Post-lock reporting completion.** After the assessment, `cbp/report_post.py` only added files:
  - ENDPOINT_RECEIPTS.json;
  - INNER_STATES_VS_RECOVERY_v2.csv;
  - figures `fig1b_inner_tradeoff`, `fig2_annotated`, `fig3_annotated` and `fig4_annotated`.

  It reads only aggregate inputs: the run files endpoint_parity.json, selection.json and inference.json, and the
  locked public tables INNER_STATES_VS_RECOVERY.csv and ASSESSMENT_COMPARISON.csv, read-only.
  It changes no endpoint, selection or label. The locked `cbp/report.py` and its outputs are unchanged. The added files
  are presentation and §13 completeness: they were not part of the locked scoring chain, and the independent replay
  verified the source tables they draw from.
