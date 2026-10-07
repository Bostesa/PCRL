# Correctness oracle report (lra, independent verifier E)

Generated 2026-10-07T07:14:42Z by `results/pcrl_adult_learned_decoder_release_v1/verification/replay_lra.py` (sha256 `ef7148a1da5162a20f2fbfa21d35162680a2d1e15684b69f66abb9e22118048d`). The verifier imports no lra / lcr / study module. Study tests and the challenge harness ran as separate external processes.

## Verdict

- Own verdict, recomputed from the own checks: **ENGINEERING_READY**. Published verdict (ENGINEERING_GATE_RESULT.json): **ENGINEERING_READY**. Agreement: True.
- This is a correctness verdict on four KNOWN, ALREADY-OPENED fixture laws. It is not mechanism evidence and not an Adult result. The historical lcr GATE_NOT_MET is untouched and plays no part in this verdict.

## Exposure and chronology

- CORRECTNESS_LOCK `de4495f3a3da1ff2b3c6b6a271aea34fac3ed795`: one version, byte-identical on origin; first push 2026-10-07T05:07:48Z.
- Correctness stage slot acquired: [('A:correctness', '2026-10-07T05:07:54Z')]; cor__ units written ['2026-10-07T05:08:04Z'] onward; the gate result was committed after the lock (True).
- This verifier first read the laws for computation at 2026-10-07T07:15:11Z, after its own guard verified the pushed lock. In PHASE_0 every computation path on the laws refused, and that was tested.

## The twelve mandatory checks (own reproduction)

| Check | Own result | Evidence |
|---|---|---|
| E01 counts, routing, hashes | PASS | pins: file sha = rule pin = lcr bytes at 091afc2 (True), laws hash recomputed (True); atoms rebuilt, integer counts sum to 4096, static properties equal; 84 releases per law routed (token class = teacher decision = released decision, strict argmax, sum q within 1e-12, token function); 114 D1 tables per law with n_t and y_t exact and s_t within 1e-9 |
| E02 fixed-token information | PASS | every release: I(S; token, q) = I(S; token) within 1e-15 per recipient and for the pair; each D0 / D1 pair shares tokens, decisions and the oracle partition |
| E03 calibrated null (F1) | PASS | every token of every canonical partition: max abs(q_D1 - q_D0) = 8.13e-13 (limit 1e-9), min law-loss gain -2.22e-16 (limit -1e-12); every F1 D1 release: max abs(dq) = 8.13e-13, min gain -3.33e-16 |
| E04 D1 final-vector certificates | PASS | 1921 released token vectors: own solver-free certificate (affine identity, simplex, exact dominance, strict argmax, Frank-Wolfe gap max 2.15e-16 relative, stationarity max 4.31e-16); own solve agrees within 2.22e-16; projection magnitude max 0 |
| E05 loss reconstruction | PASS | own row-level LL / Brier / MI of every stored release vs the published arm terms: max abs diff 2.78e-17 |
| E06 accepted-state budgets | PASS | own engine on 120 persisted traces: 1384 starts, 1735 stages, 2043 accepted states rebuilt from fine-cell statistics; budgets (limit - 1e-10), local caps and state caps hold on every accepted state; paired moves accepted: 0 (the paired path is covered only by synthetic tests) |
| E07 temporary sequential partner | PASS | the CLASS-ONLY partner is never enforced in seq1; it fails its own budget in 98 of 392 seq1 stages (all on F1), so the rule is exercised; final releases are checked for both recipients |
| E08 incremental replay | PASS | terms max abs diff 2.44e-15, deltas 2.44e-15 (limit 1e-12); stats hashes exact; 1365 fresh-cache rebuilds identical; termination receipts, winners and final labels equal; stored release partitions equal the trace winners; q hashes agree for 1152 of 4074 (informational: a different solver legitimately differs in the last bits) |
| E09 exhaustive oracle | PASS | own canonical enumeration equal to the capped Stirling counts; tables vs correctness_oracle/*.csv max abs diff 4.44e-16; the 16 CSVs are byte-identical to lcr's at 091afc2 (True); arm terms and feasibility equal; heuristic labels equal the own exhaustive optima |
| E10 decision floor | PASS | every canonical partition: I_i >= I_i(CLASS) and I12 >= I12(CLASS), minimum difference 0; CLASS D1 utility-feasible on: F2_MISCALIBRATED, F3_COMPLEMENTARY_XOR, F4_REDUNDANT |
| E11 launch wiring | PASS | verdict strings exactly ENGINEERING_READY / ENGINEERING_BLOCKED; no hard-coded readiness; run / eval_lock / assess call engineering_ready(), infer compares with ENGINEERING_READY and refuses; the historical MECHANISM_GATE_NOT_MET appears only as historical text; external wiring tests: PASS (........................................................................ [ 88%] .........                                                                [100%] 81 passed in 6.51s) |
| E12 review findings | PASS | 14 findings, ordinals [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], duplicates 9->2, 11->7, 12->8, finding 10 superseded; every regression node exists in a locked test file, all_pass, pre-science commit an ancestor of the lock |

## Challenge of the study's critical branches (external harness)

Status: **PASS**. The harness `verification/challenge_lra.py` (sha256 `418240faca95631963267c98c0cf3540130bd668f0f8e5fd7b3dc7a91fcf7953`) runs the study's own `lra.select.select_all` (disk reads and writes redirected), `fit_feasible`, `lra.family.overall_label` and `lra.assess.verify_validity` in a separate process on the verifier's synthetic cases. The verifier then compares each answer with its own independent implementation.

- selection: 12 of 12 cases agree / refuse as required
- fit_feasible: 11 of 11 cases agree / refuse as required
- labels: 10 of 10 cases agree / refuse as required
- assessment_refusal: 5 of 5 cases agree / refuse as required

## Numerical notes and heuristic gaps (reported, not failures)

- F3_COMPLEMENTARY_XOR: the local cap is compared on exact float bits, so starts / witnesses whose TRUE MI equals the cap (0) but whose float MI exceeds it by about 1e-16 were excluded as infeasible: {'U|K-SEQ-12|i8o64|D1': 2, 'U|K-SEQ-21|i8o64|D1': 2, 'U|K-JOINT-SINGLE|i8o64|D1': 3, 'U|K-JOINT-PAIR|i8o64|D1': 3}. This is conservative (no violating state is accepted). On Adult it matters only if a different partition ties the C-TASK MI exactly.
- F1_CALIBRATED_NULL: arms labelled HEURISTIC (accurately labelled search gaps vs the exhaustive optimum, not correctness failures): 5
- F3_COMPLEMENTARY_XOR: arms labelled HEURISTIC (accurately labelled search gaps vs the exhaustive optimum, not correctness failures): 17
- F4_REDUNDANT: arms labelled HEURISTIC (accurately labelled search gaps vs the exhaustive optimum, not correctness failures): 21

## Scope

Fixture MI is the exact MI of finite laws, not a population guarantee. Exhaustive enumeration of partitions does not make a numerical decoder solution exact, so the D1 vectors carry their own solver-free certificates. A heuristic gap that is accurately labelled is not a correctness failure. This report says nothing about Adult performance.
