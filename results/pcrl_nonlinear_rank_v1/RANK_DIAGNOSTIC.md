# RANK_DIAGNOSTIC — is eigenvalue-sign rank selection a live issue?

Outcome-free. Computed **before any new fit** from the already-saved 2018 arm spectra of the
completed residual spectral study (`seed_N/matrix_diagnostics.json`). No ACS outcome, reserved
label, attack or fit is involved. Machine-readable: `RANK_SPECTRUM.json`, `RANK_SPECTRUM.csv`
(1536 rows: every eigenvalue of every objective in every seed).

## Rule under test

```
r_plus = min(16, #{ eigenvalues of U - (P_L + P_AB) above tol })
tol    = max(1e-12, 1e-10 * max|eigenvalues|)      # the existing relative convention, reused
```
Frozen from training matrices only. The same `r_plus` is used for L1, L2 and C1 within a seed, so
their comparison is not confounded by different dimensions.

## Result

| seed | rows | whitened rank q | U: pos/neg/zero | L1 pos | L2 pos | C1 pos | **r_plus** | nonpositive among top 16 |
|---|---|---|---|---|---|---|---|---|
| 0 | 10513 | 128 | 32/0/96 | 32 | 32 | 32 | **16** | 0 |
| 1 | 10428 | 128 | 32/0/96 | 32 | 32 | 32 | **16** | 0 |
| 2 | 10551 | 128 | 32/0/96 | 32 | 32 | 32 | **16** | 0 |

| seed | objective | tol | ev[1] | ev[16] | ev[17] | ev[128] |
|---|---|---|---|---|---|---|
| 0 | L1 | 3.319e-11 | 0.196483 | 0.00839733 | 0.00751748 | -0.331852 |
| 0 | L2 | 6.855e-11 | 0.189091 | 0.00777449 | 0.00731225 | -0.68555 |
| 0 | C1 | 5.894e-11 | 0.189988 | 0.00677338 | 0.00652261 | -0.589353 |
| 1 | L1 | 3.215e-11 | 0.204944 | 0.00870679 | 0.00739151 | -0.321498 |
| 1 | L2 | 6.689e-11 | 0.192993 | 0.00827458 | 0.00720375 | -0.668864 |
| 1 | C1 | 5.774e-11 | 0.195756 | 0.00782752 | 0.00636075 | -0.57744 |
| 2 | L1 | 3.462e-11 | 0.200761 | 0.00798297 | 0.00735351 | -0.34623 |
| 2 | L2 | 7.173e-11 | 0.189105 | 0.00766721 | 0.00704577 | -0.717298 |
| 2 | C1 | 5.962e-11 | 0.190176 | 0.00714122 | 0.00667083 | -0.596189 |

## Reading

**`r_plus = 16` in every seed and for every original objective. Not one of the top sixteen
directions is nonpositive.** The eigenvalue-sign rank proposal therefore makes no change to this
interface, and the reduced-rank recipe is an exact alias of the rank-16 recipe. This is not a
near miss: the margin at the boundary is a factor of two in count (32 positive against a target of 16).

### Why, structurally

`U_raw = (V'R/n)(R'V/n)` and the teacher residual `R` has **32** columns, so `rank(U) <= 32`.
Measured: `U` has exactly 32 eigenvalues above tolerance and **zero** below it, in all three seeds —
96 of the 128 whitened directions carry no teacher-reconstruction energy at all. The penalties are
positive semidefinite, so `#positive(U - lambda P) <= #positive(U) = 32`; measured, it is exactly 32.
`r_plus` could only fall below 16 if a penalty drove more than sixteen of `U`'s thirty-two positive
directions to or below zero. The penalties are far too small for that: at the boundary `ev[16]` is
~8e-3 while the largest is ~2e-1.

Note the qualitative difference between the objectives that the counts expose. `L1` and `L2` leave
38-42 directions numerically **at** zero, because `P_L` is built from the rank-limited `Q_A` basis and
does not reach every direction of `U`'s null space. `C1` has **no** zero eigenvalues and 96 strictly
negative ones: adding `P_AB` makes the penalty reach the whole complement. That is a real structural
difference between the local and coalition objectives, and it is visible without any outcome.

### Consequence for the matrix (predeclared branch)

Every seed has `r_plus = 16`, so the predeclared branch fires: a **rank-8 sensitivity** is run for both
penalty families and all three policies, and it is labelled **compression, not eigenvalue-sign
selection**. The `r_plus` column of the main matrix is recorded as *void by alias* and is neither
fitted nor audited a second time. `r_plus = 0` did not occur; had it occurred the additional channel
would have been empty and the interface would have equalled `H`.

### Scope limit that this diagnostic cannot escape

`r_plus` is the retained rank of the **old fixed linear-moment objective**. It is **not** the optimal
rank of the new nonlinear objective, which has no fixed matrix to take a spectrum of. Exact objective
optimality at FIXED rank (Ky Fan, applicable to the original family) and optimal rank selection under
a different feasible set are separate statements, and neither transfers to the refined objective.
