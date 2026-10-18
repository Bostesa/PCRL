# Adjudications (append-only; times UTC)

## A1. Verifier CHECK_CHANGES change 1 (adjudicated by the coordinator, 20:31Z)
- Original result: the verifier's first reader-refit check of the D_FRONTIER grid (C5.D_FRONTIER.*.READER_REFIT__V1) FAILED.
- Cause reported: the check looked up the reader cache with the verifier's own recomposition of a release array, which differs from the
  stored array by 4e-16; the cache key hashes exact bytes, so the lookup missed and the comparison ran against nothing.
- Changed rule: none. The criterion (LR refits equal the cached readers exactly; winner AUCs equal those recomputed from cached scores)
  is unchanged; only the input bytes are now the hash-checked stored release arrays that the grid itself read.
- Re-run (V2): PASS. Both versions stay in the verifier's append-only history; V1 is recorded as superseded by a check-input bug, not
  as a passing result.
- Adjudication: accepted. The change corrects how the check obtains its input and does not widen the criterion. The independent
  replay of the 2,808 paired cells per seed and their SEs (exact) and the release-array recomposition (within 4.4e-16, hashes matching)
  support it.

## A2. Question A bootstrap scheme (verifier ATTENTION, 20:29Z): routed to the statistical reviewer for independent adjudication
(see roles/statistical_reviewer/ADJUDICATION_QA_BOOTSTRAP.md). Until adjudicated, every question A label is reported under both
schemes.
- Resolved 20:40Z. The statistical reviewer's adjudication (ADJUDICATION_QA_BOOTSTRAP.md, about 20:35Z) holds that RULES.md section 6
  defines one household resample per set and replicate, shared by every statistic and strength; the implementation departed from it.
  The coordinator applied the reviewer's fix as its own recorded correction (PROTOCOL_DEVIATIONS item 10; cts.rules
  prediction_comparison_registered with the data-free test cts/test_rules_correction.py). Primary labels: one run of the corrected
  code with the registered tag and B = 2000 on all 34 question A units; the first labels are kept as superseded. Criterion unchanged.

## A3. Verifier CHECK_CHANGES changes 2 and 3 (adjudicated by the coordinator, 20:38Z; change 1 is A1)
- Change 2 (C2.1 / C2.2 V3 -> V4). Original: two access lines reported as 'unknown partition role (P70 / P70A, FIT)' and a KeyError
  in C2.1. Cause: the verifier rebuilt partitions from the registered PARTITIONS.json only, and P70 / P70A were added later by
  amendment 9. Changed rule: amended partitions are rebuilt from their documented definition (FIT = the 14 non-panel pool blocks, no
  GUARD or CHECK; P70A = P70's rows with the admitted dictionary); their reads must equal those rows and come after the amendment
  commit 5002fed58. Accepted: the change adds a reconstruction that was missing and adds a timing condition; no criterion for the
  registered partitions is relaxed.
- Change 3 (C2.1 V4 -> V5). Original: FAIL for P70A plus a diagnostic item saying the registered 'same folds' was not implemented.
  Cause: the item compared the fold tags written in the two WITNESS.json records instead of the saved fold arrays, which are
  identical. Changed rule: sameness is judged on the saved arrays; a tag mismatch goes into a separate record note; the C2.1 FAIL
  criterion (the recorded tag must regenerate the saved folds) is unchanged and still FAILS for P70A. Accepted: it corrects a
  misleading diagnostic of the verifier's own and keeps the failing criterion.
- The P70A record defect itself is ERRATA.md E1. The C2.1 FAIL stays in the verifier's history. Any re-check against the erratum
  should be a separate item (the erratum's tags regenerate the saved folds), not a change to C2.1.

## A4. Verifier CHECK_CHANGES change 4 (adjudicated by the coordinator, 21:06Z)
- Original result: C6.3.SCREENS.READER_STABILITY_R_COMP0__V1 FAILED. Every mismatch had the form 'verifier label vs None'.
- Cause: the verifier computed the old gate, AGG1 and the stability label on the one-seed R_COMP0 bank (4 cells). RULES.md
  section 10 and rung 3 define those screens only over the 8 registered cells (both release seeds), so the table's empty labels
  are the registered output.
- Changed rule: records with fewer than 8 cells, or from a one-seed bank, are checked for the recorded mean and must carry no
  screen label. The 8-cell criteria are unchanged (exact gate, AGG1 U and stability numbers at 1e-12).
- Accepted: the change removes a computation the rules do not define, and no failing criterion is widened.

## A5. Primary R0 / R1 question A labels under PROTOCOL section 8 (verifier F3 V2): referred to the statistical reviewer at
21:04Z (PROTOCOL_DEVIATIONS item 16). Until the ruling, both R labels are reported side by side.
- Ruling (statistical reviewer, ADJUDICATION_SECTION8.md, written 21:04:46Z): option (b), with conditions.
  - The registered-text labels are primary on R0 and R1. The frozen-code labels stay beside them as 'frozen implementation
    output, superseded: departed from RULES.md section 6'.
  - Section 8 restricts revisions that use a repair outcome. This correction was triggered by a D-stage code audit, is fully
    fixed by text registered before any fit, applies to all units, and was specified without any R output.
  - Conditions: (1) the verifier confirms that its trigger used no R output, otherwise (a) applies; (2) the exact timeline is
    recorded (deviation 18); (3) both label sets are shown, with R1 REFIT_LOC full and R1 REFIT_PAIR_TR full flagged
    SCHEME_DEPENDENT; (4) the verifier records the variance detail of those two units descriptively.
  - The D-stage designation of A2 stands.
- Coordinator: conditions 2 and 3 are done (deviation 18; diagnosis/QUESTION_A_LABELS.csv has flag and superseded_note columns).
  Conditions 1 and 4 were requested from the verifier at 21:12Z. The documents report both labels until condition 1 is confirmed.
- **Conditions resolved (coordinator, 21:12Z).**
  - **Condition 1 holds.** The verifier's P4_QA_CORRECTION_TRIGGER_INPUTS__V2 is PASS: the finding used only D0 / D1 inputs (60
    files by SHA-256 in V_TRIGGER__V2.json), and none of its units read a repair panel. The message went out between 20:27:28Z and
    20:28:04Z; it reached the coordinator at 20:27:58Z.
  - **The disclosed window (P4b).** The verifier's R1 replays wrote the two scheme-dependent labels into its CHECKS.json at
    20:33:36Z, 29 s before the correction commit. The coordinator's tool log shows no read of that file between 20:33:00Z and
    20:34:10Z. A2, which fixed the correction's content, was written at 20:30:14Z, before any R1 scheme comparison existed.
  - **Condition 4 is done** (C3.QA.R1.VARIANCE_DETAIL__V3, descriptive). The R1 shifts follow from the superseded scheme's 20-25%
    smaller SE, not from heavy tails. The top 10 households carry about 5% of the variance, and dropping 1, 5 or 10 of them leaves
    the registered label NOT_IMPROVED.
  - **Decision (b) therefore stands.** On R0 and R1 the registered-text labels are primary, and the frozen-code labels are kept as
    "frozen implementation output, superseded: departed from RULES.md section 6", flagged SCHEME_DEPENDENT where they differ.

## A6. Verifier CHECK_CHANGES change 5 (adjudicated by the coordinator, 21:12Z)
- Original: the evidence list of P4_QA_CORRECTION_TRIGGER_INPUTS__V1 included the later registered-scheme replay records.
- Changed: the V2 evidence list holds only the inputs that existed before the finding was sent. The criterion is unchanged, and the
  status did not depend on the extra entries.
- Accepted: this corrects the evidence record and does not widen any criterion.

## A7. Claims review (roles/claims_review/REPORT.md, reviewer 21:00-21:22Z; disposition by the coordinator, 21:35Z)
- All 14 required edits were accepted and applied in RESEARCH_DECISION.md, START_HERE.md, ADVISOR_BRIEF.md and
  MANUSCRIPT_CLAIMS_HANDOFF.md:
  - nesting described as a one-partition, one-panel check; non-nested cross-fitting called the likely, not established, source;
    the CONT attribution removed;
  - A5's final wording for the SCHEME_DEPENDENT units, with the P4b window disclosed;
  - the eight wrong numbers corrected (max |z| 2.9; about 53% of MODEL_FIT for Track R; 1 failed unit and 1 never launched;
    70 times at full strength; GUARD or CHECK panel; slack, not margin, and about half the gap; OOF errors, not cost changes;
    SE ranges);
  - training-size language made non-causal;
  - the old-gate attribution completed (occupation view at s <= 0.5);
  - the strength-rule statements qualified (ensemble choices, e_0 crossing, rules not reader-evaluated, 'close to forced', the
    CONT exceptions);
  - the Track F and Track R scope; screen rates as properties of law L1; reader-relative wording;
  - 'replicated and independently replayed' and 'overlapping partitions' in place of 'validated' and 'genuinely different';
  - the disclosure lists completed; plain one-line answers added to sections 3, 6 and 7; the START_HERE records list completed;
    and every minor item.
- ERRATA E2 corrects F11's '75% of MODEL_FIT'. The verifier's numbers the reviewer relied on were spot-checked by the coordinator
  against TRANSFER_FACTORS.json, ORIGINAL_BENCHMARK.json and the witness records (max GUARD |z| 2.88; nested errors; occupation-view
  gate failures; FIT sizes 10,523 / 10,506), and all match.
