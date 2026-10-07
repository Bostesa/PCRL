# Validation — hcal

**Label:** NO_COMPETITIVE_RELEASE_CRITERION_ESTABLISHED
[OriginalCriterion NOT_ESTABLISHED; CalibrationMatchedCriterion NOT_ESTABLISHED; FittingRole MIXED;
ParameterSharing MIXED].

## 1. Admission (SOURCE_ADMISSION_LOCK f259db6)

- **Verified copies.** 540 lra units and 6 teacher directories.
  - Each file was hash-checked against its unit's COMPLETE.json, the lra same-device SHA256SUMS, and (for scored units)
    the lra EVALUATION_LOCK at the evidence commit.
  - Not admitted: lra assessment predictions (`outer__*`).
- **Bitwise parity.**
  - Teacher forward passes: 6 of 6.
  - Legacy D0 re-encodings: 81.
  - D1 re-encodings with every token re-solved: 171.
  - 171 frozen-bank tables, whose row checks hold. TEACHER-MEAN equals the admitted D0 on every legacy row, and lra's
    D0SAME for C-TASK.
- **Role split.** Built from group IDs only, with no label read.
  - CALIBRATION_HELDOUT: 2,000 groups / 2,001 rows.
  - ATTACK_FIT_NEW: 4,061 groups / 4,064 rows.
  - TRAIN_MATCHED: 2,000 groups / 2,002 rows.
  - All 13 disjointness pairs are zero.
- **Teacher-held-out provenance.** VERIFIED (PROVENANCE_REPORT.md). No AUDIT_FIT row entered teacher, head, head
  selection or preprocessing fitting. Disclosures D1–D4 are listed there.

## 2. Engineering (ENGINEERING_LOCK f1b2d81)

- **Registered checks.** 130 synthetic checks (ENGINEERING_CHECKS.json) cover every required correctness category:
  - identity paths;
  - a known non-identity optimum;
  - zero and small counts, class fallbacks, boundary α and absent classes;
  - NLL derivatives and convexity;
  - per-token certificates against independent K = 2/3/6 solves;
  - group partitioning;
  - common attack predictions across decoder variants;
  - orientation, unseen tokens and pairs, and ignore-recipient banks;
  - denominator and seed alignment;
  - nominee and comparator status cases;
  - deployment refusals;
  - control plants.
- **Verdict.** ENGINEERING_READY (130/130). It rests on correctness only.
- **Reviews.**
  - B (MATH_REVIEW.md): 0 blocking, 3 should-fix, all integrated.
  - D: the XOR risk led to the common-bank controls; an SVD non-convergence led to the registered fresh-view hygiene.
  - E (E_ENGINEERING_REVIEW.md): 1 blocking (a fail-safe controls key) and 7 should-fix, all integrated before the
    lock.
  - Dispositions: REVIEW_FINDINGS_DISPOSITION.json.

## 3. Science (SCIENCE_LOCK 05cebd6; AMENDMENT_A1 7fba038)

- **Calibrate.** 171 partition/seed units plus 3 U units. Every per-token solve is certified, every temperature
  certificate holds, and every fallback equals q0 exactly.
- **Utility.**
  - 292 releases × 3 seeds, with decision preservation on every row.
  - Ucal\* = H-GLOBAL-TEMP.
  - The registered audit plan audits 25 partitions and records 32 as PREDECLARED_UTILITY_INELIGIBLE.
- **Audit and compose.** 75 fresh banks with common records. Every decoder variant shares one record
  (COMPLETE_INTERFACE_EQUIVALENCE.json). U's composed bank covers the lra composition plus 25 fresh banks.
- **Controls** (CONTROLS_RESULT.json; required = common bank).
  - The first run hit a technical defect: a NaN-blind serialisation receipt.
  - AMENDMENT_A1 was pushed before the rerun; attempt 1 is preserved.
  - Rerun: all_ok. NULL, CONF, COLL and XOR pass on 4 codes; null calibration has 0/15 exceedances; ROT passes on U.
  - The realised null threshold equals the source's (0.5654143765984265).
  - No per-pipeline diagnostic failure.
- **Selection.** T\* = FINE-TASK H-GLOBAL-TEMP; P\* = JOINT λ0.1 H-CLASS-TEMP. Every rejection is recorded:
  - 174 utility-ineligible;
  - 36 local-guard failures;
  - 9 below the 0.02 benefit.
- **Replay.** 90/90 frozen winners refit bitwise on INNER_SELECTION.
- **Independent inner replay (role E).** PASS on all six checks:
  - role split;
  - frozen bank;
  - calibration (43,290 KKT-certified tokens; 1,392 temperature solves);
  - utility (41,875 comparisons, difference 0);
  - common records and both U compositions (24,234 comparisons);
  - T\* and P\* with every rejection.

## 4. Assessment and inference (EVALUATION_LOCK b23df38, pushed 15:00:31Z)

- **Single opening.** 15:00:43–15:04:30Z, through `hcal.assess`:
  - 81 probability units and 15 attack units;
  - every frozen winner refit with INNER predictions bitwise first.
- **Inference.**
  - 23 slots on 1,999 paired exact-record-group draws (seed 20261011), z = 3.0653831516447343.
  - Every statistic is finite. 13,936 rows and 13,929 groups.
- **Independent phase 3 (role E).** PASS:
  - lock and unit custody;
  - assessment role;
  - 27 releases × 3 seeds equal to table[tok], bitwise;
  - 20 attack winners refit bitwise;
  - all 23 slots exact (max difference 0.0), both criteria, both diagnostic summaries, the label, and the
    supplementary contrasts.

## 5. Deployment and custody

- **Deployment** (DEPLOYMENT_RECEIPT.json). Six releases deploy bitwise from the 83-column input with argmax equal to
  the decision: P\*, T\*, an H-TOKEN32 release re-solved from its counts, D1, D0, and an lra MEAN control. Ten refusals
  exit with code 2 and write nothing.
- **Custody** (BACKUP_VERIFICATION.json, RESTORE_INDEX.json).
  - Same-device copy: `<PRIVATE_CACHE>/hcal_v1_local_copy_20261007`, 4,098 store files plus the pinned input,
    re-read uncached.
  - Restores from the copy alone:
    - the pinned input;
    - U teachers s0–s2, bitwise;
    - frozen-bank tables;
    - every calibrator family, refitted bitwise (12 tables each for H-TOKEN32, H-GLOBAL-TEMP and H-CLASS-TEMP; 9 for
      T-TOKEN32; 6 for U);
    - P\* and T\* deployed from the copy, bitwise;
    - the selected reader (P\* pair winner, seed 0), refit exact on all assessment rows.
  - Off-device backup is PENDING (drive absent).

## 6. Disclosed deviations

1. **AMENDMENT_A1** (controls only; cause, fix, affected artifacts and unchanged estimands are in AMENDMENT_A1.json).
2. **OPS-1.** Stopping the first control run left its workers alive for about 2 s under `/usr/bin/time`. They were then
   stopped, and no third heavy process ran.
3. **Fitting-row losses.** The use of fitting-row task labels for descriptive losses was registered before
   SCIENCE_LOCK (E-S2).
4. **Fresh-view hygiene and common-bank controls.** Both were registered before SCIENCE_LOCK.
5. **Disclosure.** P\*'s partition is lra's P\* partition; only the decoder differs.
