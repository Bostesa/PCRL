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
| lcr/tests/test_closeout.py | 32 | F |
| **Total** | **172, all pass** | `lcr.sema --label A:test-all … pytest -q lcr/tests` (about 98 s) |

**Injected defects.** Role E's verifier injects each of the 15 defects named in the prompt and catches all 15
(INDEPENDENT_VERIFICATION.json, phase 0 self-tests, 27/27 pass):
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
  - E's own exhaustive fixture oracle: law integrity, static properties, and 159 certified D1 solves via an independent
    trusted solver.
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

## 4. Disclosed deviations and corrections

1. **Pre-lock oracle run.** Role E's independent oracle ran on the registered laws before FIXTURE_LOCK was pushed, which
   deviates from "push FIXTURE_LOCK before algorithms run on them". Its counts informed the pre-stage update LP1-U1.
   The laws were not changed in response.
2. **Pre-lock rule clarifications.** The rule changes listed in §3 were made before the lock. No study algorithm had run
   on the laws.
3. **Amendment A1.** A defect in the lead's own pre-lock C5 check produced a spurious CORRECTNESS_FAILURE in attempt 1.
   - A1 is code-only and was pushed before attempt 2. Both attempts are kept.
   - The A1 reason text says both F1 K-SEQ arms hit INFEASIBLE_START at stage 1. In fact K-SEQ-12 fails at stage 2. This
     wording error, found by E, is disclosed in FIXTURE_ATTEMPTS.json; the pushed amendment file is unchanged.
4. **Post-hoc analysis.** POST_HOC_EQUAL_LEAKAGE_UTILITY.json was defined after the gate ran. It is labelled
   UNREGISTERED and cannot affect the gate or the label.
5. **Process note.** A tracked empty file, lcr/review_tests/__init__.py (committed with the admission lock), was briefly
   removed when the review scratch tests were moved out of the tree. It was restored from git before any commit, so no
   tracked file is missing.

## 5. Not validated, because it never ran

- Adult D1 decodes and C-TASK; weighted and constrained Adult fits.
- Inner audits and controls (the cbp_parity receipt).
- Selection, the evaluation lock, the assessment and inference.

Their code is unit-tested on synthetic data only. The decoder-only ablation on Adult does not exist; the fixture
analogue is DECODER_ONLY_ABLATION.csv.

## 6. Phase-3 verification and claims review

(filled from role E's PHASE_3 record and role F's claims review; see below)
