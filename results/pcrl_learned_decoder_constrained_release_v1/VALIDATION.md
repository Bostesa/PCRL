# Validation — lcr

**Label: MECHANISM_GATE_NOT_MET.** This file records what was checked, by whom and how, and what remains open.

## 1. Tests (synthetic data and registered fixture laws only; no real labels)

| Suite | Tests | Owner |
|---|---|---|
| lcr/tests/test_admit.py | 3 | A |
| lcr/tests/test_decoder.py | 17 | B |
| lcr/tests/test_fixtures.py | 15 | B, plus A's two pre-lock / A1 tests |
| lcr/tests/test_deploy.py | 6 | B |
| lcr/tests/test_mapper.py | 17 | C |
| lcr/tests/test_audit.py | 61 | D |
| lcr/tests/test_select.py | 8 | A |
| lcr/tests/test_truth_table.py | 8 | A |
| lcr/tests/test_late.py | 5 | A |
| lcr/tests/test_closeout.py | 33 | F |
| **Total** | **173, all pass** | `lcr.sema --label A:test-all … pytest -q lcr/tests` (about 96 s, final run at closeout) |

**Injected defects.** Role E's verifier injects each of the 15 defects named in the prompt and catches all 15
(INDEPENDENT_VERIFICATION.json; self-tests 26 PASS plus 1 timing INFO):
- omitted token identities;
- clean-output bypass;
- reversed AUC;
- pair misalignment;
- wrong label counts in the decoder;
- stale cache key;
- omitted prior;
- class constraint omitted;
- proxy KL substituted for true LL;
- missing Brier constraint;
- temporary sequential partner wrongly required to be feasible;
- infeasible paired update;
- mean-seed eligibility;
- missing comparator;
- missing source composition winner.

The owners' unit tests also cover the defects in their own scope. Role D's audit tests show these caught: omitted token
identities, clean-output bypass, reversed AUC, reversed best/worst reader, pair misalignment and a missing composition
winner.

## 2. Independent verification (role E; verification/replay_lcr.py; imports no study module)

- **Phase 0 / 1A.**
  - Self-tests pass.
  - Admission replay: source hashes, groups and the verified copy.
  - E's own exhaustive fixture oracle: law integrity, static properties, and 159 of E's own D1 solves, each certified
    by E's FW/KKT certificate, with KKT-enumeration and SLSQP cross-checks on samples.
- **Phase 1B (fixture gate): PASS.** 10 PASS / 0 WARN / 0 FAIL / 1 INFO top-level, 53 PASS nodes. E checked:
  - **Binding:** the rule and laws are bound to the FIXTURE_LOCK documents.
  - **Chronology:** both locks were on origin before each attempt started (01:38:50Z → 01:39:00Z; 01:42:28Z →
    01:42:45Z).
  - **Locked code:** equal at both evidence commits.
  - **Oracle tables:** match E's enumeration within ≤ 4.4e-16.
  - **Arms:** all 84 arms per fixture agree on terms, feasibility and local budget.
  - **Checks:** C1–C4 and C6 recomputed; the per-fixture triggers and the verdict GATE_NOT_MET / NO_FIXTURE_TRIGGERED
    reproduced; the descriptive flags reproduced.
  - **Structure:** I(S; d1, d2) = 0 by integer-table check in all four laws. No enumerated pair falls below CLASS.
    CLASS|D1 is budget-feasible on F2–F4 and not on F1.
  - **Amendment A1:** both its cause and its correction are verified. The attempts differ only in the C5 nodes, and the
    oracle CSVs are byte-identical.
- **Phase 3 (reports, deployment, custody, document numbers):** results in §6 below.

## 3. Reviews

- **Gate-rule text vs code** (reviews/FIXTURE_GATE_TEXT_VS_CODE_REVIEW.json; before FIXTURE_LOCK; 4 finders and 1
  skeptic per finding).
  - 5 confirmed findings, all resolved before the lock with the laws unchanged (PROTOCOL.md §9):
    - C3 scope;
    - C5 incremental terms;
    - C7 lock binding;
    - two descriptive flags;
    - the unregistered C1/C5 conditions and the C6 NOT_A_SEARCH text.
  - 4 refuted. One of them, the CLASS|D1 structural finding, was refuted as a text/code defect but confirmed as a fact,
    and was registered as the expected outcome (PREDICTIONS.json LP1-U1).
- **Selection stack** (reviews/SELECTION_STACK_REVIEW.json: select, family, infer, eval_lock).
  - 14 confirmed findings: no REQUIRED, 8 RECOMMENDED, 6 NOTE.
  - **NOT applied.** This code never executed: no SCIENCE_LOCK, no Adult fit. These are open items that any future use
    must apply and re-verify first. The main ones:
    - fit_feasible must treat an unreadable constrained record as a technical failure, not as CONSTRAINED_FIT_INFEASIBLE;
    - the winning family and construction must name a pair that some exact alias actually deploys;
    - restore cbp's `identical_to_untrained` alias disclosure;
    - give missing-guard roles a descriptive fallback;
    - wire rule 1 (gate failure → MECHANISM_GATE_NOT_MET) into infer;
    - make role-level aliases use release hashes, not config ids;
    - remove cbp role names (C_rate, C_global) from LABEL_TRUTH_TABLE.json.
- **Baseline fairness** (BASELINE_FAIRNESS_REVIEW.md, role F). R-1 to R-5 were resolved before any fit: R-4 (never
  nominate an infeasible constrained fit) and R-5 (rank fit-feasible fallbacks first) are in select.py and verified by F
  and E.
- **Claims review** (role F): of RESEARCH_DECISION, ADVISOR_BRIEF, PAPER_ADDENDUM, METHOD_CARD and COST_AND_CLOSEOUT
  against the claim scope (§6).
- **Closeout completeness critic** (reviews/CLOSEOUT_COMPLETENESS_REVIEW.json; 3 lenses, with a skeptic per gap). It
  checked the package against every prompt requirement and the evidence files. Each confirmed gap was fixed before the
  final commit, or was a closeout step completed afterwards: the handoffs, the custody refresh, the final CPU total and
  close time, and the push. Its refuted items and notes are recorded there.

## 4. Disclosed deviations and corrections

1. **Pre-lock oracle run.** Role E's independent oracle ran on the registered laws before FIXTURE_LOCK was pushed, which
   deviates from "push FIXTURE_LOCK before algorithms run on them". Its counts informed the pre-stage update LP1-U1.
   The laws were not changed in response.
2. **Pre-lock rule clarifications.** The rule changes listed in §3 (47beb1f) were made before the lock.
   - No study algorithm had run on the laws.
   - Role E's exhaustive oracle HAD enumerated them (01:11–01:12Z).
   - The clarifications changed correctness checks and descriptive definitions only: not the laws, the trigger, T* or
     the candidate lists.
   - FIXTURE_LAWS.json's bank note ("written before any algorithm runs") was accurate when written at dd1cf23. The
     laws themselves are unchanged since then.
3. **Amendment A1.** A defect in the lead's own pre-lock C5 check produced a spurious CORRECTNESS_FAILURE in attempt 1.
   - A1 is code-only and was pushed before attempt 2. Both attempts are kept.
   - The A1 reason text says both F1 K-SEQ arms hit INFEASIBLE_START at stage 1. In fact K-SEQ-12 fails at stage 2. This
     wording error, found by E, is disclosed in FIXTURE_ATTEMPTS.json; the pushed amendment file is unchanged.
4. **Post-hoc analysis.** POST_HOC_EQUAL_LEAKAGE_UTILITY.json was defined after the gate ran. It is labelled
   UNREGISTERED and cannot affect the gate or the label.
5. **FIT_MANIFEST.json.** It was frozen only by FIXTURE_LOCK (9ac4cb7, as a hashed document), never by a SCIENCE_LOCK,
   because none was written.
6. **Replay timing fields.** `lcr.report all` re-runs the mapper. Its measured CPU-time fields (`cpu_s_starts` in
   OPTIMIZATION_RECEIPTS.json) therefore differ between replays; every other field is identical.
7. **Learned-decoder restore.** The custody restore covers the teachers and the Q release and deployment. No learned
   decoder file exists to restore: there is no Adult D1, and fixture decoders are not persisted as units. They are
   regenerated deterministically by `lcr.report all`, which checks them against the registered gate.
8. **Process note.** A tracked empty file, lcr/review_tests/__init__.py (committed with the admission lock), was briefly
   removed when the review scratch tests were moved out of the tree. It was restored from git before any commit, so no
   tracked file is missing.

## 5. Not validated, because it never ran

- Adult D1 decodes and C-TASK; weighted and constrained Adult fits.
- Inner audits and controls (the cbp_parity receipt).
- Selection, the evaluation lock, the assessment and inference.

Their code is unit-tested on synthetic data only. The decoder-only ablation on Adult does not exist; the fixture
analogue is DECODER_ONLY_ABLATION.csv.

## 6. Phase-3 verification and claims review

**Role E, PHASE_3 (final): WARN, 0 FAIL.** 15 PASS / 2 WARN / 1 INFO top-level; 14 NOT_APPLICABLE (Adult); 64 PASS
nodes. Independence passes: the study CLI ran only as external subprocesses (one lcr.deploy refusal, two lcr.lock
verify calls).
- **Report tables, reproduced from E's own oracle.**
  - DECODER_ONLY_ABLATION.csv: 216 rows, max |diff| 4.4e-16.
  - BUDGET_FEASIBILITY.csv: 336 rows, max |diff| 6.7e-16. Feasible counts F1 0, F2 57, F3 84, F4 84 of 84.
  - CLASS_PRESERVATION, RUN_STATUS, POST_HOC_EQUAL_LEAKAGE_UTILITY and OPTIMIZATION_RECEIPTS: all match.
- **Decoder certificates.**
  - All 456 stats hashes are reproduced exactly.
  - All 1,921 supervised tokens are re-solved and certified (FW gap ≤ 2.8e-16, stationarity ≤ 4.3e-16).
  - 45 binding-tie tokens, with minimum margin 9.99978e-13 (the ε-level separation).
  - A trusted KKT-enumeration spot check gives the same point.
- **Figures.** Every plotted series equals E's recomputation.
- **Deployment.**
  - The CLI output is bitwise equal to the admitted Q release and to E's own deployment.
  - The input is byte-identical to cbp's.
  - The 84-column refusal exits with code 2 and writes nothing.
- **Custody.** `shasum -c` gives 731/731 OK. E restored from the copy alone, and the teacher and Q re-encode are
  bitwise. STATUS.json is PENDING with osf_assessment_opened false.
- **Budget.** At most 2 concurrent holds; 1.61 CPU-h measured across roles at E's run.
- **WARN 1 (custody), resolved afterwards.** The review scratch tests, moved into the private store after the 01:49Z
  copy, were not in it. Role F's refresh at 02:28:12Z added them, now at `<PRIVATE_CACHE>/lcr_v1/run/review_scratch/`,
  together with the deploy test output and the grown ledger. The copy holds 754 files, 754/754 OK. The final
  verification run re-checks this (below).
- **WARN 2 (documents).** E checked 58 claims and found four minor mismatches, all corrected:
  - the lock push time (01:38:50Z, not 01:38:38Z);
  - "about 0.0006" (0.000616);
  - the CPU total (about 1.6, not 1.5).
- **Note.** The C5 incremental-terms maximum is lcr.fixtures' own figure. The mapper start records are not persisted,
  so E could not recompute it independently.

Verifier commands (from the worktree root; each phase re-runs the earlier ones; `--out <file>` writes nothing to PKG):
```
OMP_NUM_THREADS=1 <python> -P lcr/sema.py --label E:phase3 -- env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 <python> results/pcrl_learned_decoder_constrained_release_v1/verification/replay_lcr.py \
    --phase 3 --label PHASE_3
```
Phases 0, 1A and 1B used `--phase 0 --label PHASE_0`, `--phase 1 --label PHASE_1A` and `--phase 1 --label PHASE_1B`.

**Role F, claims review.** It found no novelty claim, no "fixed fixtures" wording, and the post-hoc block labelled.
These REQUIRED items were applied:
- the short answer reworded so it cannot read as an Adult test;
- "no clause evaluated" in the claims table;
- the F2 generalisations scoped to that fixture;
- one unlabelled post-hoc sentence labelled;
- "never opened" scoped to this study;
- the custody wording tied to the 01:49Z copy and its closeout refresh.

The recommended scope edits (convexity per fixed token, exhaustive-optimal only on the small fixtures, the
predecessor custody, prior art in the brief) were also applied.
