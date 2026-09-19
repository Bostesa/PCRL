# MATHEMATICAL_REVIEW — v5 addendum

The v4 review (`results/pcrl_manuscript_review_v4/MATHEMATICAL_REVIEW.md`, fixtures in its
`MATH_FIXTURES.json`) is carried unchanged. This addendum records (A) how the issues v4 raised about the
Study 5 projection were actually resolved in code and text, and (B) the review of the pending
utility-first residual extension, before any outcome.

## A. Study 5 projection — resolution

| v4 issue | Resolution in Study 5 | Evidence |
|---|---|---|
| Rank-deficient Σ: whitened projection deletes off-support components of new rows; LEACE's subtractive form keeps them; form must be declared | Declared the **whitened** form `z_out = mu + v P_k S`, relative support tolerance 1e-10. Measured supported rank **16/16 in all 204 fits**, 0 dropped directions in all 6 whitenings (2 channels × 3 anchors), stable across tolerances 1e-14…1e-6. So `R S = I` and the two forms coincide, including on transported rows. The rank-deficient case did not arise; were it to, the declared form would delete off-support directions. | `projection.py` docstring; `TRACK_E_FITS.json`; RUN_STATUS "B5" |
| Range of P must lie in the support | `P_k` is r×r in whitened coordinates, so its range lies in the support by construction. | METHOD §2.1, §2.5 |
| Globally vs per-role centred moments | Globally centred `v`, masked moments with valid denominators; tested. | METHOD §2.4; fixtures |
| Mass-scaled local projection is an alias | `L2 = 2 K_local` registered as an exact alias and not fitted. | METHOD §2.4 |
| Convex-correction statement needs fixed offset, no intercept | METHOD §2.7 states the result for a fixed-offset logit with the **same cross-fitted offset** that defines `e_j`, the same masks, correction linear in `b ⊗ v_out`, **no free intercept** (intercept gradient `−mean(e_j)` tested non-zero). Scope excludes retrained baselines, nonlinear auditors, population quantities; quadratic counterexample tested. | METHOD §2.7; `test_projection_math.py` |
| Partial projection guarantees | Partial projections leave non-zero training cross-moments (tested), so the stationarity statement **does not apply to the partial nominees**; it applies to full-span removal, which here is the constant map (made exactly zero by amendment 2). | METHOD §2.6 |
| Novelty of the eigensystem | Track E = utility-free, infinite-penalty limit (λ→∞ at retained rank r−k) of the project's own residual spectral eigenproblem, applied to a frozen neural channel and unwhitened into its metric. Limiting form of an existing eigensystem on a different input; not new eigenvalue mathematics. | METHOD §2.5 |

Wording enforced in the manuscript: removing linear dimensions is not certified conditional privacy;
nonsignificant effects are never "ruled out"; point agreement within .0011 is not equivalence.

## B. Pending utility-first extension `[H_A, Z_J, R]` — review before outcomes

Design as communicated to Terminal 2 (no protocol file committed at the time of writing; Terminal 1 at
tier 0, AWS session expired). R predicts residual teacher information not captured by a frozen decoder
from `(H_A, Z_J)` to `A0`.

1. **Monotonicity.** For any sensitive S, `inf_f E[−log f(S | H, Z_J, R)] ≤ inf_f E[−log f(S | H, Z_J)]`:
   the recipient can ignore R. Population sensitive recovery cannot decrease. The method can only aim at
   *bounded added* disclosure. It must never be described as removing information available in J.
2. **Utility preservation is population-level.** Population utility with R is at least that without it,
   but a finite probe fitted on `[H_A, Z_J, R]` can score worse than one fitted on `[H_A, Z_J]`
   (larger input, regularisation, selection). The utility audit must include the old-view predictor as an
   explicit candidate in the new view's slate, or report both. Exact bitwise preservation of `Z_J` ≠ equal
   fitted-probe scores.
3. **Proxy.** Residual reconstruction of A0 is a training proxy. Residence/commute labels do not enter
   fitting or selection; validation proxy improvement is not a utility claim.
4. **Deployment path.** Cross-fitted residual targets exist only at training time. Check the final deployed
   map `D0`: R must be a fixed function of permitted A-side inputs (the PCA_32 path and frozen maps). It
   must not index an out-of-fold target, a row identifier, `H_B`, or a sensitive label at inference.
   Request: the serialised inference function plus a replay test on held-out rows with targets removed.
5. **Two increments.** Report `Δ_H = loss(S|H) − loss(S|H,Z_J,R)` and `Δ_J = loss(S|H,Z_J) − loss(S|H,Z_J,R)`
   per endpoint. A small `Δ_J` can coexist with a large `Δ_H`. Show both in every table.
6. **Roles.** Baseline attackers must condition on the J-containing view; coalition attackers also see
   `H_B`; local and coalition roles and equal-opportunity-conditioned attacks remain. Guard against an
   apparent protection gain from worse auditor fitting in higher dimension: the old-view attack must be
   routed onto the new view as a candidate (as `H`'s attacks already are), so the new-view attacker is never
   weaker than the old-view one by construction of the slate.
7. **Objective.** A finite adversarial objective with baseline subtraction, clipping and frozen nuisances is
   not a mutual-information upper bound or independence guarantee. Clipping at zero in training is fine; in
   reporting, increments stay unclipped.
8. **Gate and counts.** The pilot gate screens reconstruction, source eligibility and validation sensitive
   increments; it must not read residence outcomes. Expansion is conditional and preregistered; counts are
   cumulative (pilot slots reused in the 135-slot plan must not be double counted); an untriggered tier is
   reported as "not triggered", not as missing evidence.
9. **Utility-first criterion.** Increased recovery must be *bounded* on every declared endpoint by one-sided
   adjusted upper bounds, not merely non-significant; the .001-nat allowance is named an allowance, not zero
   harm; comparisons against every strong external comparator and the local controls; and a non-coalition
   ("ordinary") residual extension that performs equally well is a counterexample to any coalition-specific
   claim.

Items needing Terminal 1 artefacts to close: the machine-readable pilot gate file, the D0 serialisation and
inference replay, the attacker slate listing (old-view candidates routed), and the family/margin declaration.
