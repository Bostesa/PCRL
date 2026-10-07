# Fixture oracle report (role E, independent verifier)

Verifier: `results/pcrl_learned_decoder_constrained_release_v1/verification/replay_lcr.py` (sha256 `ffce2881de7fa600d28a2215b26313d98cf1d70dd1c92d848df23606927acc8b`); generated 2026-10-07T01:12:22Z. Own code only: no lcr / cbp / qpc module was imported.

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
- Exhaustive C-TASK: value 1.434305 at partitions [27, 63].
- Exhaustive D0 FINE-TASK: value 0.013148 at partitions [27, 63].

### F2_MISCALIBRATED

- Law integrity (PASS): atoms rebuilt exactly True (as a set True); static properties equal True; rows 4096; mapping pairs 32768 enumerated 32768.
- D1 class preservation on every partition: True; feasible maps under the fitting budgets (D1): {'1': 64, '2': 512}.
- Exhaustive C-TASK: value 2.120589 at partitions [27, 165].
- Exhaustive D0 FINE-TASK: value 0.018193 at partitions [27, 227].

### F3_COMPLEMENTARY_XOR

- Law integrity (PASS): atoms rebuilt exactly True (as a set True); static properties equal True; rows 4096; mapping pairs 4096 enumerated 4096.
- D1 class preservation on every partition: True; feasible maps under the fitting budgets (D1): {'1': 64, '2': 64}.
- Exhaustive C-TASK: value 1.485261 at partitions [45, 45].
- Exhaustive D0 FINE-TASK: value 0.000334 at partitions [45, 45].

### F4_REDUNDANT

- Law integrity (PASS): atoms rebuilt exactly True (as a set True); static properties equal True; rows 4096; mapping pairs 4096 enumerated 4096.
- D1 class preservation on every partition: True; feasible maps under the fitting budgets (D1): {'1': 64, '2': 64}.
- Exhaustive C-TASK: value 1.485261 at partitions [45, 45].
- Exhaustive D0 FINE-TASK: value 0.000334 at partitions [45, 45].

## Gate

- PENDING: FIXTURE_GATE.json not yet written (own references precomputed).

## Scope

Fixture MI is the exact law MI of these finite laws, not a population guarantee for Adult. Optimisers that differ from the exhaustive references are labelled HEURISTIC with their gap; that is a report, not a failure.
