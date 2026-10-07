# Research decision — learned decoders and confidence-constrained releases (lcr)

**Label: MECHANISM_GATE_NOT_MET.** No Adult fit was run, so there is no Adult result in either direction.

**Short answer.**
- Learning the probabilities did not improve confidence on Adult, because it was never tried there: the registered
  mechanism gate failed first.
- On the known-law fixtures, it improved confidence on identical tokens in the miscalibrated fixture and changed
  nothing in the calibrated null.
- None of changing the assignments, the calibrated existing method or a weighted baseline produced usable protection
  here, because no Adult release was fitted.
- The full development criterion did not pass.

| Item | Value |
|---|---|
| Branch | `research/pcrl-learned-decoder-constrained-release-v1` |
| Started from | cbp final tip `7f3ec67` |
| SOURCE_ADMISSION_LOCK | `4a91947` (pushed) |
| Predictions | `f0b78f6` (pushed before any run); pre-stage update LP1-U1 in `47beb1f` |
| Fixture laws and gate rule first registered | `dd1cf23` |
| FIXTURE_LOCK | `9ac4cb7` (pushed 2026-10-07T01:38:38Z) |
| AMENDMENT_A1_FIXTURE_C5_SCOPE | `c9a7150` (pushed) |
| Gate result (attempt 2) | `18e41ab`; attempt 1 kept at `cc0083a` |
| SCIENCE_LOCK / EVALUATION_LOCK | never written; no Adult fit was launched |

## 1. What ran

1. **Source admission.** 189 cbp units were admitted by verified copy: U and RAW-J teachers, deployed heads, fine
   partitions, the 81 D0 codes, references and cbp inner audits as custody. Teacher and release parity were bitwise
   (ADMITTED, 6/6 teachers, 81/81 codes).
2. **Pre-registration.** The fixture laws, the gate rule and predictions LP1–LP9 were pushed before any study
   algorithm ran on the laws.
3. **Pre-lock review.** A text-versus-code review of the gate rule found disagreements in C1, C2, C3, C5, C6, C7 and the
   descriptive flags. All were resolved before the lock with the laws unchanged (PROTOCOL.md §9).
4. **Structural problem found before the stage.** The same review found that the registered laws could not trigger the
   gate (§3). This expectation was registered before the stage (PREDICTIONS.json LP1-U1; PROTOCOL.md §9). The laws
   were deliberately not amended.
5. **Fixture stage.** It ran under the pushed FIXTURE_LOCK in one process (~27 CPU-s).
   - Attempt 1 flagged a spurious C5 failure. The check I added before the lock required an incremental search state
     for two F1 constrained arms in which no start was ever refined.
   - Amendment A1 (code only) scoped that check. Attempt 2 is identical to attempt 1 apart from that flag and the
     timings, and its oracle tables are byte-identical. Both attempts are kept (FIXTURE_ATTEMPTS.json).
6. **Stop rule.** The gate failed, so the registered stop rule applied (prompt §9): no D1 decode of Adult maps, no
   C-TASK, no weighted or constrained Adult fit, no inner audit, no selection, no assessment.
7. **What ran after the gate.**
   - fixture-scope reports;
   - the deployment CLI on the admitted Q map;
   - the same-device custody copy with a restore from it;
   - the independent fixture replay (role E).

## 2. Gate result

The trigger, verbatim: "At least one fixed nontrivial fixture shows that privacy-trained partitioning with D1 satisfies
all utility/local budgets and reduces pair MI by at least 0.01 nats beyond the strongest feasible task-only D1
compression, while both tasks have accuracy gain at least 0.03 over their constants."

| Fixture | Mandatory checks C1–C7 | Strongest feasible task-only D1 (T*) | I12(T*) | Triggered | Reason |
|---|---|---|---|---|---|
| F1 calibrated null | all pass | none: no task-only map meets the budgets | — | no | NO_FEASIBLE_TASK_ONLY |
| F2 miscalibrated | all pass | CLASS\|D1 | 0 | no | not nontrivial (I12(T*) < 0.01) |
| F3 complementary XOR | all pass | CLASS\|D1 | 0 | no | not nontrivial |
| F4 redundant | all pass | CLASS\|D1 | 0 | no | not nontrivial |

Verdict: **GATE_NOT_MET**, reasons `["NO_FIXTURE_TRIGGERED"]`; the route is not classified.

## 3. Why it failed: structural, registered before the stage, not fixed

- **The floor.** Every class-preserving release determines both predicted classes (c1, c2). By data processing,
  I12(R) ≥ I12(CLASS|D1) for every release R. So whenever the decision-only release meets the fitting budgets, it is
  the strongest feasible task-only compression and nothing can go below it.
- **The registered bank.**
  - SEX is balanced within every predicted-class pair in all four laws, so I12(CLASS|D1) = 0 exactly.
  - On F2–F4, CLASS|D1 is within budget. Its smallest slack is 0.0015 nats of log loss for F2 recipient 2, and
    0.0023–0.0027 nats on F3/F4.
  - On F1, no task-only map is within budget for recipient 2.
  - The trigger was therefore unattainable before anything ran.
- **Why the laws were not changed.** Role E's independent oracle had already enumerated the laws before FIXTURE_LOCK,
  which is a disclosed deviation from "push FIXTURE_LOCK before algorithms run on them". Changing the laws would have
  been the outcome-informed hunt for a favourable example that the prompt forbids. The original forecast LP1 (gate
  met, 0.85) is kept and scored as a miss. LP1-U1 (0.01) is a pre-stage update informed by those counts, not an
  independent forecast.
- **What this does not mean.**
  - It is not evidence that the mechanism fails on Adult.
  - It is a property of this fixture bank: the decisions carry no SEX information, and the decision-only release is
    affordable.
  - Neither property is known for Adult under these budgets. cbp measured a decision-only (CLASS-ONLY) pair AUC of
    0.739 with its slate, so Adult decisions do carry SEX signal. Whether CLASS|D1 meets the lcr confidence budgets on
    Adult was never computed.

## 4. What the fixtures do show (descriptive)

| Finding | Evidence |
|---|---|
| Implementation correctness | C1–C7 pass on every fixture. 228 D1 decoders were all certified and reloaded with a bitwise re-solve: stationarity ≤ 3.1e-16, simplex residual ≤ 2.2e-16, no fallback token (DECODER_CERTIFICATES.json). Decisions are preserved in every release (CLASS_PRESERVATION.json). The incremental search terms of every refined mapper winner match the from-scratch rebuild within 1.6e-15. |
| Calibrated null | D1 = D0 within 8.1e-13 per token. No loss change on identical tokens (\|ΔL\| ≤ 4e-16). |
| Decoder on identical tokens (F2, miscalibrated) | D1 lowers log loss by 0.076–0.083 nats (recipient 1) and 0.033–0.038 (recipient 2), and Brier by 0.024–0.049. Token arrays and I1, I2, I12 are identical (bitwise). All 27 fixed maps become budget-feasible, against 0 of 27 under the mean decoder (DECODER_ONLY_ABLATION.csv). |
| Decoder on identical tokens (F3, F4) | Changes ≤ 7e-5 nats (nearly calibrated teachers). |
| Search quality | Every constrained arm and C-TASK on F2–F4 is EXHAUSTIVE_OPTIMAL against the exhaustive reference for its own registered problem. On F1 the constrained arms correctly report INFEASIBLE. They are labelled HEURISTIC there because no feasible reference exists. Some unconstrained searches are HEURISTIC on F3/F4: the old mean-decoder SEQ/JOINT/LOCAL searches and some weighted D1 arms, 17 and 21 arms respectively. Their gaps in their own objectives are ≤ 7.0e-4, reported and not failures (OPTIMIZATION_RECEIPTS.json; FIXTURE_GATE.json C6). |
| Constrained vs weighted (descriptive flag) | F2: best constrained I12 = 0 against best weighted 0.0216, so the registered λ grid does not push the weighted arms to zero. F3, F4: both reach 0. |
| Joint vs sequential (descriptive flag) | 0 difference in every family on every fixture. |

**Post-hoc, UNREGISTERED** (POST_HOC_EQUAL_LEAKAGE_UTILITY.json; it cannot change the verdict or the label). Among
budget- and local-feasible privacy-trained D1 releases, the lowest task loss with pair MI no larger than T*'s:
- **F2:** 0.042 nats below CLASS|D1, from K-JOINT-PAIR (constrained). The decoded old JOINT λ0.025 map is within
  0.0006 nats of it.
- **F3:** 0.0038 nats below CLASS|D1. Decoded old maps, weighted and constrained arms all tie.
- **F4:** no difference.

So on these laws, privacy-trained D1 releases match the decision-only leakage while keeping more task utility. Almost
all of that is available from old maps once they are decoded with D1. The registered trigger measures only pair-MI
reduction, so this is not a gate result.

## 5. Claims and clauses

| Claim | Status | Exact failed clause |
|---|---|---|
| Gate (mechanism) | NOT MET | no fixture triggered: F1 has no feasible task-only comparator; F2–F4 are not nontrivial because I12(T*) = 0 |
| A: P* vs T* (privacy release) | NOT RUN | gate not met; no Adult fit |
| B: N* vs C* (constrained increment) | NOT RUN | gate not met |
| C: J* vs C_pair* (paired joint increment) | NOT RUN | gate not met |
| Q (confidence feasibility) | NOT RUN | gate not met |
| Decoder-only ablation on Adult | NOT RUN | gate not met; the fixture analogue is in §4 |

No Adult AUC, accuracy, log-loss or Brier figure is produced or implied. The ones quoted here come from cbp and are
labelled as cbp's.

## 6. What is runnable

- **Fixture stage:** `python -m lcr.run --lock …/FIXTURE_LOCK.json --stage fixture`, run under the semaphore.
- **Reports:** `python -m lcr.report all`, which replays the locked fixture engine and checks it against the registered
  FIXTURE_GATE.json before writing.
- **Deployment:** `python -m lcr.deploy` (QUICKSTART.md).
  - Tested on the admitted Q map (seed 1, 39,170 rows): BOUND, and bitwise equal to the admitted release.
  - Six refusal cases exit with code 2 and write nothing.
  - The D1 path is tested on synthetic data only.
- **Adult code that never executed.** The Adult stages are written and unit-tested but never ran on real data: the D1
  Adult decode, C-TASK, the mapper on Adult, audits, selection, inference and the evaluation lock. VALIDATION.md lists
  the unapplied review findings on that code.

## 7. Tests, replay, work, cost, custody

- **Tests:**
  - test_admit 3, test_decoder 17, test_fixtures 15, test_deploy 6, test_mapper 17;
  - test_audit 61, test_select 8, test_truth_table 8, test_late 5, test_closeout 32;
  - all 172 pass (VALIDATION.md).
- **Independent verification:** role E's replay imports no study module (INDEPENDENT_VERIFICATION.json,
  FIXTURE_ORACLE_REPORT.md).
- **Work and cost:**
  - Fixture stage: about 27 CPU-s per attempt.
  - Whole study: about 1.5 CPU-h through the shared semaphore, against the 20 CPU-h cap. At most 2 heavy processes, with
    the largest child at 1.3 GiB.
  - $0 cloud.
- **Custody:**
  - Same-device copy (731 files): verified uncached and restored from (teachers, Q release and deployment, all
    bitwise).
  - Status: LOCAL_SAME_DEVICE_COPY_VERIFIED_OFF_DEVICE_BACKUP_PENDING; the drive is absent.
  - cbp custody is PENDING, because it would open osf's assessment rows, the same rows as this study's never-opened
    assessment.

## 8. Decision

- **Close this exact gate.**
  - Prompt §17 says that when the fixture gate fails, there is no Adult claim and the engineering and oracle evidence is
    preserved.
  - No Adult result exists, so no "competitive negative" claim is made.
  - No weight or budget adjustment is recommended.
- **What a successor would need.**
  - A new, separately registered study with a new fixture bank written before any algorithm runs.
  - At least one fixture must have a budget-infeasible decision-only release, or decisions that carry SEX information.
  - Otherwise the "beyond the strongest task-only compression" trigger is unattainable by the data-processing floor.
  - That is a design consequence of this failure. It must not be a re-run under this registration.
- **Carry forward.**
  - The decoder (convex, certified, class-preserving, with a calibrated null).
  - The constrained mapper (exhaustive-optimal on the fixtures).
  - The decoder-only ablation finding: on a miscalibrated teacher, D1 makes old maps budget-usable at identical
    information.
  - All three are engineering facts on known laws, not privacy evidence.
- **Confirmation.** No confirmation population is spent.
