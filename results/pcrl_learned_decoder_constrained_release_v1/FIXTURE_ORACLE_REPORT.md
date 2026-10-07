# Fixture oracle report (role E, independent verifier)

Verifier: `results/pcrl_learned_decoder_constrained_release_v1/verification/replay_lcr.py` (sha256 `f6086e808690304ccc202b66ae0d69a03b5c8500c00679c5fd53dc57db35c128`); generated 2026-10-07T02:39:17Z. Own code only: no lcr / cbp / qpc module was imported.

Laws: FIXTURE_LAWS.json, laws_sha256 `5c5e3bda08c971d50513093179207546059fe2259e8707ca866fd0e162942434`. Hash recomputed by the registered rule: verified. Gate rule binds the same laws: True. Registered tolerances equal: True.

## Method

- Atoms rebuilt from the explicit tables (pair counts, SEX numerators, per-cell label laws, north-west-corner joint labels), compared with the stored atoms; static properties recomputed from own atoms.
- Every canonical same-class partition of each recipient's fine cells under the per-class cap enumerated as restricted-growth strings (counts checked against Stirling sums); all mapping pairs evaluated.
- Exact law quantities from integer tables at N = 4096: plug-in MI of SEX with full token identities; D0 = smoothed token-mean teacher; D1 = own dual water-filling solve (kappa 32, eps 1e-12, class-dominant simplex) from exact expected counts, certified by the own Frank-Wolfe gap / KKT certificate; log loss (clip 1e-12), multiclass Brier, teacher KL of the D0 decoder.
- Exhaustive optimum of every registered own-problem objective; arms' values recomputed from their partitions; feasibility, local budget, T_star, qualifying set, trigger, route and verdict recomputed from FIXTURE_GATE_RULE.json by own code.
- Continuous optimisation is certified numerically (registered tolerances); exhaustive enumeration makes the DISCRETE references exact, not the convex solves.

## Per fixture

### F1_CALIBRATED_NULL

- Law integrity (PASS): atoms rebuilt exactly True (as a set True); static properties equal True; rows 4096; mapping pairs 4096 enumerated 4096.
- D1 class preservation on every partition: True; feasible maps under the fitting budgets (D1): {'1': 1, '2': 0}.
- Calibrated null: max |q_D1 - q_D0| = 8.13e-13 over 480 token evaluations; min (D1 - D0) law loss per row = -2.22e-16 (ok True).
- Exhaustive C-TASK: value 1.434305 at own-enumeration partition indices [27, 63].
- Exhaustive D0 FINE-TASK: value 0.013148 at own-enumeration partition indices [27, 63].
- Replay vs FIXTURE_GATE.json (PASS): oracle tables max |diff| 0.000; arms checked 84; partition identity (own CLASS / DIRECT-TASK) {'U|CLASS|i1o1': True, 'U|DIRECT-TASK|i8o64': True}.
- Own trigger: T_star None (I12 n/a); nontrivial False; triggered False; qualifying 0; decoder_enabled None; assignment_search_helps None; best fixed-map I12 n/a; best new I12 n/a; exhaustive most-private feasible local pair I12 n/a.
- Own heuristic labelling: {'EXHAUSTIVE_OPTIMAL': 50, 'HEURISTIC': 5}; max heuristic gap 0.000.

### F2_MISCALIBRATED

- Law integrity (PASS): atoms rebuilt exactly True (as a set True); static properties equal True; rows 4096; mapping pairs 32768 enumerated 32768.
- D1 class preservation on every partition: True; feasible maps under the fitting budgets (D1): {'1': 64, '2': 512}.
- Exhaustive C-TASK: value 2.120589 at own-enumeration partition indices [27, 165].
- Exhaustive D0 FINE-TASK: value 0.018193 at own-enumeration partition indices [27, 227].
- Replay vs FIXTURE_GATE.json (PASS): oracle tables max |diff| 0.000; arms checked 84; partition identity (own CLASS / DIRECT-TASK) {'U|CLASS|i1o1': True, 'U|DIRECT-TASK|i8o64': True}.
- Own trigger: T_star U|CLASS|i1o1|D1 (I12 0.000000); nontrivial False; triggered False; qualifying 0; decoder_enabled False; assignment_search_helps False; best fixed-map I12 n/a; best new I12 n/a; exhaustive most-private feasible local pair I12 0.000000.
- Own heuristic labelling: {'EXHAUSTIVE_OPTIMAL': 55, 'HEURISTIC': 0}; max heuristic gap 0.000.

### F3_COMPLEMENTARY_XOR

- Law integrity (PASS): atoms rebuilt exactly True (as a set True); static properties equal True; rows 4096; mapping pairs 4096 enumerated 4096.
- D1 class preservation on every partition: True; feasible maps under the fitting budgets (D1): {'1': 64, '2': 64}.
- Exhaustive C-TASK: value 1.485261 at own-enumeration partition indices [45, 45].
- Exhaustive D0 FINE-TASK: value 0.000334 at own-enumeration partition indices [45, 45].
- Replay vs FIXTURE_GATE.json (PASS): oracle tables max |diff| 0.000; arms checked 84; partition identity (own CLASS / DIRECT-TASK) {'U|CLASS|i1o1': True, 'U|DIRECT-TASK|i8o64': True}.
- Own trigger: T_star U|CLASS|i1o1|D1 (I12 0.000000); nontrivial False; triggered False; qualifying 0; decoder_enabled False; assignment_search_helps False; best fixed-map I12 n/a; best new I12 n/a; exhaustive most-private feasible local pair I12 0.000000.
- Own heuristic labelling: {'EXHAUSTIVE_OPTIMAL': 38, 'HEURISTIC': 17}; max heuristic gap 0.001.

### F4_REDUNDANT

- Law integrity (PASS): atoms rebuilt exactly True (as a set True); static properties equal True; rows 4096; mapping pairs 4096 enumerated 4096.
- D1 class preservation on every partition: True; feasible maps under the fitting budgets (D1): {'1': 64, '2': 64}.
- Exhaustive C-TASK: value 1.485261 at own-enumeration partition indices [45, 45].
- Exhaustive D0 FINE-TASK: value 0.000334 at own-enumeration partition indices [45, 45].
- Replay vs FIXTURE_GATE.json (PASS): oracle tables max |diff| 0.000; arms checked 84; partition identity (own CLASS / DIRECT-TASK) {'U|CLASS|i1o1': True, 'U|DIRECT-TASK|i8o64': True}.
- Own trigger: T_star U|CLASS|i1o1|D1 (I12 0.000000); nontrivial False; triggered False; qualifying 0; decoder_enabled False; assignment_search_helps False; best fixed-map I12 n/a; best new I12 n/a; exhaustive most-private feasible local pair I12 0.000000.
- Own heuristic labelling: {'EXHAUSTIVE_OPTIMAL': 34, 'HEURISTIC': 21}; max heuristic gap 0.001.

## Binding and chronology

- Rule FIXTURE_GATE_RULE.json sha256 `6f3bb44a3f4fd9db6aa986e697929689e1e49c43d9e5f88ae2136a482b0ed844` (re-bound; equals the FIXTURE_LOCK documents_sha256: True); FIXTURE_LAWS.json equals the lock documents: True; laws unchanged since dd1cf23: True.
- FIXTURE_LOCK 9ac4cb7 first on origin 2026-10-07T01:38:50Z; attempt 1 ran ['2026-10-07T01:38:58Z', '01:39:26Z']; AMENDMENT_A1 c9a7150 first on origin 2026-10-07T01:42:28Z; attempt 2 ran ['2026-10-07T01:42:42Z', '01:43:11Z']. Lock before attempt 1: True; amendment before attempt 2: True.
- FIXTURE_GATE.json binds the rule and laws: True; its oracle-table hashes re-hash: True; fix__ units of both attempts complete and bound to the laws hash: True.

## Registered expectation vs result

| Fixture | Registered expectation (PROTOCOL sec. 9, before the stage) | Lead result | Own replay |
|---|---|---|---|
| F1_CALIBRATED_NULL | NO_FEASIBLE_TASK_ONLY (no budget-feasible task-only map) | as own (replay PASS) | NO_FEASIBLE_TASK_ONLY; triggered False |
| F2_MISCALIBRATED | T* = CLASS|D1 with I12 = 0: not nontrivial | as own (replay PASS) | T* = U|CLASS|i1o1|D1, I12(T*) = 0.000000, nontrivial False; triggered False |
| F3_COMPLEMENTARY_XOR | T* = CLASS|D1 with I12 = 0: not nontrivial | as own (replay PASS) | T* = U|CLASS|i1o1|D1, I12(T*) = 0.000000, nontrivial False; triggered False |
| F4_REDUNDANT | T* = CLASS|D1 with I12 = 0: not nontrivial | as own (replay PASS) | T* = U|CLASS|i1o1|D1, I12(T*) = 0.000000, nontrivial False; triggered False |

## Structural bound I12(R) >= I12(CLASS|D1)

Every class-preserving token refines the teacher-predicted class, so the token tuple determines (d1, d2) and I(S; t1, t2) >= I(S; d1, d2) = I12(CLASS) (data processing). Checked exactly and exhaustively:

| Fixture | I(S; d1, d2) = 0 exactly (integer table) | I12(CLASS) | min I12 over every enumerated pair | pairs below CLASS | min reported arm I12 | CLASS|D1 budget-feasible |
|---|---|---|---|---|---|---|
| F1_CALIBRATED_NULL | True | 0.000000 | 0.000000 | 0 | 0.000000 | False |
| F2_MISCALIBRATED | True | 0.000000 | 0.000000 | 0 | 0.000000 | True |
| F3_COMPLEMENTARY_XOR | True | 0.000000 | 0.000000 | 0 | 0.000000 | True |
| F4_REDUNDANT | True | 0.000000 | 0.000000 | 0 | 0.000000 | True |

Consequence: on F2-F4 the feasible task-only reference already has I12 = 0, so no release (not even the exhaustive most-private feasible local pair, I12 = 0) can reduce pair MI by 0.01; on F1 no task-only candidate is feasible. The registered fixtures could not trigger; this is not evidence about the mechanism on Adult.

## Descriptive flags (own recomputation; never used by the verdict)

| Fixture | best I12 constrained (all / eligible) | best I12 weighted (all / eligible) | constrained - weighted (eligible) | joint - sequential (d1 / weighted / constrained, eligible) |
|---|---|---|---|---|
| F1_CALIBRATED_NULL | 0.042822 / n/a | 0.042822 / n/a | n/a | n/a / n/a / n/a |
| F2_MISCALIBRATED | 0.000000 / 0.000000 | 0.021584 / 0.021584 | -0.021584 | 0.000000 / 0.000000 / 0.000000 |
| F3_COMPLEMENTARY_XOR | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 0.000000 | 0.000000 / 0.000000 / 0.000000 |
| F4_REDUNDANT | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 0.000000 | 0.000000 / 0.000000 / 0.000000 |

## Amendment A1 (C5 scope): cause and correction

- Status PASS. Attempt 1 C5 failures on F1: ['U|K-SEQ-12|i8o64|D1:incremental_terms_missing', 'U|K-SEQ-21|i8o64|D1:incremental_terms_missing']; attempt 2 NO_SEARCH_STATE lists: {'F1_CALIBRATED_NULL': ['U|K-SEQ-12|i8o64|D1', 'U|K-SEQ-21|i8o64|D1'], 'F2_MISCALIBRATED': [], 'F3_COMPLEMENTARY_XOR': [], 'F4_REDUNDANT': []} (exactly K-SEQ-12 and K-SEQ-21 on F1: True).
- F1 constrained mapper statuses (persisted fix__ unit): {'U|K-LOCAL|i8o64|D1': {'status': 'INFEASIBLE', 'winner_kind': 'refined_descriptive', 'winner_start': 'U|C-TASK|i8o64|D1'}, 'U|K-SEQ-12|i8o64|D1': {'status': 'INFEASIBLE', 'winner_kind': 'unchanged_descriptive', 'winner_start': 'U|C-TASK|i8o64|D1'}, 'U|K-SEQ-21|i8o64|D1': {'status': 'INFEASIBLE', 'winner_kind': 'unchanged_descriptive', 'winner_start': 'U|C-TASK|i8o64|D1'}, 'U|K-JOINT-SINGLE|i8o64|D1': {'status': 'INFEASIBLE', 'winner_kind': 'unchanged_descriptive', 'winner_start': 'U|C-TASK|i8o64|D1'}, 'U|K-JOINT-PAIR|i8o64|D1': {'status': 'INFEASIBLE', 'winner_kind': 'unchanged_descriptive', 'winner_start': 'U|C-TASK|i8o64|D1'}}.
- Own cause derivation (own oracle + the registered sequential driver): recipient-2 maps that pass the search feasibility (budgets - 1e-10, local cap) on F1: 0; recipient-1 maps: 1. K-SEQ-21 (recipient 2 first) is infeasible at stage 1 from every start; K-SEQ-12 is feasible at stage 1 from every start and infeasible at stage 2. No start can reach a refined final: True. K-LOCAL refines recipient 1 (refined_descriptive) and the joint drivers record an 'unchanged' witness block, so a state exists for them; NO_SEARCH_STATE can apply only to the sequential arms.
- Wording disagreement (does not affect the correction): the amendment says both arms hit INFEASIBLE_START at stage 1; for K-SEQ-12 the failure is at stage 2 (7 starts).
- Identity: FIXTURE_GATE.json attempt 1 vs 2 differ only at ['.fixtures[0].checks.C5_TERM_RECONSTRUCTION.failures:len', '.fixtures[0].checks.C5_TERM_RECONSTRUCTION.incremental_not_applicable_no_search_state:missing', '.fixtures[0].checks.C5_TERM_RECONSTRUCTION.pass', '.fixtures[1].checks.C5_TERM_RECONSTRUCTION.incremental_not_applicable_no_search_state:missing', '.fixtures[2].checks.C5_TERM_RECONSTRUCTION.incremental_not_applicable_no_search_state:missing', '.fixtures[3].checks.C5_TERM_RECONSTRUCTION.incremental_not_applicable_no_search_state:missing', '.reasons:len', '.reasons[0]'] (outside C5 / reasons: []); oracle CSVs byte-identical: True; fix__ results identical outside C5 and timing: True; rule and laws unchanged by A1: True; code files changed: ['lcr/fixtures.py', 'lcr/tests/test_fixtures.py'].

## Gate

- Lead verdict GATE_NOT_MET, route None, reasons ['NO_FIXTURE_TRIGGERED'].
- Own verdict GATE_NOT_MET, route None, triggered [].
- Verdict agrees: True; route agrees: True; replay status PASS.

## Scope

Fixture MI is the exact law MI of these finite laws, not a population guarantee for Adult. Optimisers that differ from the exhaustive references are labelled HEURISTIC with their gap; that is a report, not a failure.
