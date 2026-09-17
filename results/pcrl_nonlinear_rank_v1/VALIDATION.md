# VALIDATION

What was checked, with the measured number. Nothing here certifies privacy.

## 1. Falsification fixtures (ran before any new ACS score)

`tests/pcrl_nonlinear_rank_v1/`: 23 passed in 3.23s

Covered: feature-family shape and frozen scaling; explicit bandwidth failure;
chunk-size invariance; the magnitude fixture (old first moment exactly 0 while the
quadratic block detects recovery from |Z|); the XOR fixture (marginal moment 0,
H-interaction moment 1, S exactly recoverable); the conditional-null fixture
(oracle-nuisance moments concentrate towards 0 and are NOT required to equal it);
the nuisance-error fixture (a misspecified m(H) manufactures a penalty more than
10x the oracle value under exact conditional independence); the dimension fixture
(fixed-rank versus variable-rank optimum, zero rank, repeated eigenvalues);
analytic gradient against central finite differences; retraction and tangency;
monotone descent and feasibility; the closed form beating descent on the original
family; covariance-normalised utility; and the three rotation-invariance tests.

## 2. Parity with the frozen historical objects

| Check | Max abs error |
|---|---|
| Recomputed three-fold household OOF moment Gram trace, all 5 roles x 3 seeds | 0.000e+00 |
| Recomputed per-class moment squared norms, all roles x 3 seeds | 0.000e+00 |
| Utility matrix raw trace | 0.000e+00 |
| Whitening covariance identity `V'V/n - I` | 1.389e-12 |
| Reconstruction-trace identity vs direct least squares | 9.770e-15 |
| Closed-form objective vs top-eigenvalue sum (original family) | 6.661e-16 |

The recomputed nuisance moments are **bit-identical** to the stored historical ones,
so the new penalty is built on exactly the moments the original baseline used.

## 3. The reused column is the historical column

The nine `original`-family rank-16 conditions have an objective identical to the
historical arms and are **never refitted or re-audited**:

* `spectral_lin16_C1` reuses the frozen `spectral_C1` unit.
* `spectral_lin16_L1` reuses the frozen `spectral_L1` unit.
* `spectral_lin16_L2` reuses the frozen `spectral_L2` unit.

Their maps reproduce the historical maps to ~1e-14 and their objectives to ~1e-10
(recorded per seed in `seed_N/fit_diagnostics.json`). The ledger counts them as
reused, not new.

## 4. Optimisation feasibility and orthogonality

| Check | Worst value over 36 fitted conditions |
|---|---|
| Stiefel feasibility `max|W'W - I|` | 2.665e-15 |
| Orthogonality of the saved map | 2.665e-15 |
| Released covariance `max|Z'Z/n - I|` | 3.619e-14 |

The released covariance identity is what makes feature collapse unable to look like
privacy: the utility term is covariance-normalised for every orthonormal `W`.

## 5. Score replay

Every selected endpoint's reported loss was recomputed from the stored per-person
predictions: 2112 checks, max absolute difference
4.441e-16
(clean).

## 6. Independent prediction replay (corruption detection)

27540 stored prediction arrays were recomputed in a
separate process that did not write them, and compared bitwise:
**0 mismatches**.

This check exists because the completed transport study found rare,
load-dependent corrupted CPU prediction blocks under extreme memory pressure.
That machine condition recurred during this run (swap near capacity throughout),
so the check is not hypothetical. Memory pressure remains a **suspected**
contributing condition, not a demonstrated root cause.

## 7. Anchor parity and role preservation

Every released view of every pool asserts `wire/A[:, :4] == hA` and `wire/B == hB`
bitwise, and `wire/AB == [wire/A | wire/B]`, in `float64`. A receives `(H_A, Z)`;
B receives `H_B` only; A's inference transform never touches `H_B`. Recorded per
arm and pool in `seed_N/anchor_parity.json`.

Consequence, stated so it is not misread: because B receives only `H_B`, every
B-view quantity is identical across all interfaces **by construction**. There is no
commute-capability or coverage-capability endpoint that any interface can win or
lose; those numbers are structural constants, not results.

## 8. Subset-index compatibility

The utility (2048) and attacker (4096) subset index arrays were regenerated from the
historical seed formulas and their hashes asserted equal to the historical
`indices.json` for all three seeds, so the new units are fitted on exactly the rows
the historical units used.

## 9. What none of this establishes

* No marginal privacy, conditional privacy, or bound on `I(S;Z|H)`.
* No calibrated conditional-independence test: a penalty at its optimum has no
  p-value and no Type-I rate. None of KCI's or RCoT's calibration is inherited.
* Equal penalty scale at the reference projection is not equal privacy strength
  away from it.
* The global-optimality statement belongs to the original fixed linear-moment
  matrix only. The refined objective is nonconvex and its solver is iterative.
* Development uncertainty is descriptive: these pools have been used repeatedly and
  a bootstrap cannot undo that.
