# Predecessor gate diagnosis (lcr → lra)

**Owner and date.** Role F (claims and custody reviewer), 2026-10-07, written between 04:40Z and 05:00Z. That is after
SOURCE_ADMISSION_LOCK and before CORRECTNESS_LOCK and SCIENCE_LOCK.

**Sources.**
- Read: the published lcr results only (source review, prompt §4):
  - SOURCE_FIXTURE_GATE.json, the historical lcr FIXTURE_GATE.json, sha256
    `772f17e35a471d7069a44c707890bcdfe76a3996f1ebab071331655ae5cb325e`, byte-identical to
    `results/pcrl_learned_decoder_constrained_release_v1/FIXTURE_GATE.json`;
  - the lcr RESEARCH_DECISION.md and PROTOCOL §9 at 091afc2.
- Not done: no fixture algorithm, oracle or Adult computation was run for this file.

**Scope.** This file explains:
- why lcr's registered mechanism gate could not trigger;
- why lcr has no Adult result;
- why the prompt's correction of the successor suggestion is right;
- why lra uses a correctness-only launch gate.

It changes no historical verdict.

## 1. What the predecessor registered and found

**The trigger, verbatim (lcr FIXTURE_GATE_RULE.json).** "At least one fixed nontrivial fixture shows that
privacy-trained partitioning with D1 satisfies all utility/local budgets and reduces pair MI by at least 0.01 nats
beyond the strongest feasible task-only D1 compression, while both tasks have accuracy gain at least 0.03 over their
constants."

**The result, from SOURCE_FIXTURE_GATE.json (unchanged).** Verdict GATE_NOT_MET, reasons `["NO_FIXTURE_TRIGGERED"]`,
route not classified.

| Fixture (N = 4096 exact expected counts) | Strongest feasible task-only D1 compression (T\*) | I12(T\*) | Trigger reason |
|---|---|---|---|
| F1 calibrated null | none | — | NO_FEASIBLE_TASK_ONLY: no task-only map meets recipient 2's budgets |
| F2 miscalibrated teacher | U\|CLASS\|i1o1\|D1 (feasible task-only maps: CLASS, DIRECT-TASK, FINE-TASK and C-TASK, all with D1) | 0.0 | not nontrivial |
| F3 complementary clues (XOR) | U\|CLASS\|i1o1\|D1 (same four feasible) | 0.0 | not nontrivial |
| F4 redundant / no coordination | U\|CLASS\|i1o1\|D1 (same four feasible) | 0.0 | not nontrivial |

lcr RESEARCH_DECISION §3 gives CLASS|D1's smallest fitting-budget slack:
- 0.0015 nats of log loss for F2 recipient 2;
- 0.0023–0.0027 nats on F3 and F4.

**The historical label is MECHANISM_GATE_NOT_MET.**
- It stays the lcr label.
- It is not this study's launch verdict, and it is never this study's label (PROTOCOL §12; `lra.closeout.study_label`
  never returns it).

## 2. The affordable CLASS comparator and the disclosure floor

**Setting.** Recipient i receives a class-preserving complete-token release R_i = (t_i, q_i, d_i):
- t_i is the full categorical token;
- q_i = h_i(t_i) is its public decoded vector;
- d_i = class(t_i) is the unchanged teacher decision. Every token lies inside one teacher-predicted class, so d_i is a
  function of t_i.

The decision-only release CLASS gives each recipient d_i alone, under either decoder.

**The floor.** For every class-preserving complete-token release R = (R_1, R_2):
- (d_1, d_2) is a deterministic function of (t_1, t_2), which is part of R.
- By the data-processing inequality, I(S; R) = I(S; t_1, t_2) ≥ I(S; d_1, d_2) = I(S; CLASS), and likewise
  I(S; R_i) ≥ I(S; CLASS_i).

The inequality holds in two senses:
- **For the population law**, by data processing.
- **Exactly for the fitted plug-in quantities.** Plug-in MI is the MI of the empirical joint distribution on the
  fitting rows. The empirical law obeys the same data-processing inequality, so I12(R) ≥ I12(CLASS) holds on
  OSF_DEFENSE_FIT for every map in the bank.

The floor binds the decoders too:
- A decoder cannot lower it. q_i is a public function of t_i, so I(S; R) does not depend on h_1 and h_2 (PRIOR_ART_AND_CLAIM_SCOPE
  §5.1).
- The Bayes-optimal recovery ROC on R dominates the one on CLASS: any test on (d_1, d_2) is also a test on R.

**The consequence for a "beyond the strongest task-only compression" trigger.**
- CLASS is privacy-untrained: neither the decision map nor D1's per-token objective reads SEX.
- So whenever CLASS meets the utility budgets, it is a feasible task-only compression with the smallest pair
  information any class-preserving release can have.
- Then no class-preserving privacy-trained release R can satisfy I12(CLASS) − I12(R) ≥ 0.01. That would need
  I12(R) < I12(CLASS), which the floor forbids.
- The trigger is therefore **unattainable by construction whenever CLASS is utility-feasible, whatever any algorithm
  returns.** This is check 10 of the new gate ("an affordable CLASS comparator may make privacy superiority impossible
  without making the implementation wrong").

**Where the trigger could fire.**
- Only in a narrow window: CLASS fails the budgets, some richer task-only compression meets them, and a privacy-trained
  release sits 0.01 nats or more below that richer comparator.
- F1 was outside the window from the other side: no task-only compression at all met the budgets.
- F2–F4 were outside it because CLASS|D1 was affordable.

## 3. lcr produced no Adult result

- **Under the registered stop rule, lcr fitted nothing on Adult:**
  - no Adult D1 decode;
  - no C-TASK;
  - no weighted or constrained Adult fit;
  - no inner audit;
  - no selection, EVALUATION_LOCK or assessment.
- **Claims A, B, C and Q are NOT RUN.**
  - There is no Adult result in either direction.
  - The gate failure says nothing about the Adult method.
- **The cause** is a property of the registered fixture bank and trigger:
  - CLASS|D1 was affordable on F2–F4, with zero pair information;
  - nothing was feasible on F1.
- **What lcr did show is engineering on known laws, not privacy evidence:**
  - certified D1 decoders;
  - the calibrated null on F1;
  - on F2, D1 made all 27 of the fixture's fixed maps budget-feasible at identical full-token information, against 0 of
    27 under D0;
  - the search reports.
- **The post-hoc equal-leakage observation** (POST_HOC_EQUAL_LEAKAGE_UTILITY.json) was unregistered. It passed no gate
  and establishes no method result.
- **The only Adult figure lcr quoted** is cbp's decision-only (CLASS-ONLY) pair AUC of about 0.739 on cbp's slate. It is
  cbp's measurement, not an lcr or lra result.

## 4. The prompt's correction of the successor suggestion

**What lcr suggested** (RESEARCH_DECISION §8). A successor fixture bank needs "a budget-infeasible decision-only release,
**or** decisions that carry SEX information."

**Why the second alternative is wrong.** Suppose the decisions carry SEX information, so I12(CLASS) > 0, and CLASS is
utility-feasible.
- CLASS is still the strongest feasible task-only comparator.
- Every class-preserving complete-token release still has I12(R) ≥ I12(CLASS).
- The floor has moved from zero to a positive value, but it still binds.
- So the old "less information than CLASS" gate stays unattainable **whenever CLASS is utility-feasible, whether the
  floor is zero or positive** (prompt §1).
- Only an unaffordable CLASS opens room for that trigger.

**Why this study does not chase that window.** Designing new fixture laws so that CLASS fails its budgets would be an
outcome-informed search for a favourable example, after the old bank's failure was seen. Prompt §9 forbids it ("Do not
… search for new favorable fixture laws").

## 5. Why lra uses a correctness-only launch gate

1. **Fixture superiority is not Adult evidence.** A synthetic win on a known law would say nothing about Adult. A
   synthetic loss is only a property of the law, as §2 shows. Neither is needed to justify a cheap development test on
   already-used rows (prompt, "Why this handoff is different").
2. **The real question is about engineering.** Before a real fit, the question that needs an answer is "is the
   implementation correct enough to run the registered development test?" (prompt §9).
   - The twelve mandatory checks answer it: law integrity, the D0/D1 identity, the calibrated null, decoder
     certificates, loss reconstruction, budgets and caps on every accepted state, the sequential-partner rule, replay from
     persisted traces, oracle enumeration, the disclosure floor, wiring, and the inherited findings.
   - The verdicts are ENGINEERING_READY or ENGINEERING_BLOCKED, nothing else.
3. **The floor becomes a correctness check, not a trigger.** Check 10 asks that the floor HOLD. An affordable CLASS that
   makes superiority impossible is the expected, correct behaviour, not a failure.
4. **A performance trigger would repeat the trap.** Any trigger of the form "beats the strongest task-only compression"
   on these laws is pinned by CLASS|D1 at zero. Any re-tuned law would be an outcome-informed hunt.
5. **What READY does not need** (prompt §9). ENGINEERING_READY permits the locked Adult study even if:
   - CLASS is zero-leakage on a fixture;
   - no fixture shows a superior release;
   - constrained search ties a weighted control;
   - joint ties sequential;
   - the source gate still reads GATE_NOT_MET.

   If the correctness gate itself fails, the study closes ENGINEERING_BLOCKED_NOT_RUN, with every Adult claim NOT RUN.
   That is not an Adult negative.

## 6. What the floor means for the Adult claims (registered before SCIENCE_LOCK)

- **CLASS is in the T\* pool with both decoders.**
  - U|CLASS|i1o1 (D0) and U|CLASS|i1o1|D1 are both in the pool (fairness review R-1, adopted at ab81a97; PROTOCOL §6:
    84 codes and 89 configurations per seed, 267 inner units).
  - Whether either CLASS variant meets the ORIGINAL inner utility rules on Adult is measured, not assumed. It is
    reported in DECISION_FLOOR_AND_FEASIBILITY.csv with CLASS's measured attacks.
  - No MI guarantee is ever derived from an AUC.
- **Claim A when a CLASS variant is eligible and is T\*.**
  - Clause 1 asks for a pair-AUC lower bound above 0.02 of P\* below CLASS.
  - Every class-preserving P\* has at least CLASS's information, and its Bayes-optimal recovery ROC dominates CLASS's.
  - Such a pass can therefore come only from finite-reader behaviour on the registered slate. It must be reported in
    exactly those words, never as information removal (PRIOR_ART_AND_CLAIM_SCOPE §6.2).
- **Claim A when CLASS is ineligible on Adult.** The floor still bounds every release from below. Report I12(CLASS) on
  the fitting rows and CLASS's inner/assessment recovery beside every privacy result.
- **Claim B.** N\* and C\* are both class-preserving privacy-trained codes, so the floor bounds both sides. It makes
  the claim neither impossible by construction nor easier.
- **Claim C.** The registered C_pair\* pool is every eligible release other than K-JOINT-PAIR, including the
  privacy-untrained ones.
  - If a CLASS variant is eligible, it can be C_pair\*. Claim C clause 1 is then attainable only through finite-reader
    behaviour, as for A.
  - If C_pair\* is a continuous reference (U, RAW-J, FARE, F0 or LEACE), the two sides are not nested, and the floor
    gives no ordering between them.
  - This conservative pool is a registered design choice (BASELINE_FAIRNESS_REVIEW.md S5).
- **The decoder.** A D1 change on identical tokens moves no release below or above the floor (§2). It can only make a
  release utility-eligible.

## 7. What this file does not claim

- No Adult result, in either direction.
- No statement that the mechanism works or fails on Adult.
- No new fixture law, and no re-scoring of the old gate.
- No population or all-attacker guarantee. The floor is an information inequality for the full-token interface. It
  says nothing about how well any finite attacker performs.
