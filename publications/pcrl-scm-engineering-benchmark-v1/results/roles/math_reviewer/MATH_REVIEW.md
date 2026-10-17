# Mathematical and baseline review of PROTOCOL.md (SCM engineering benchmark)

Math reviewer, 2026-10-16. Reviewed against the committed `PROTOCOL.md` at `be5cc210a` (sha256 `4817bac1…67c6`). I first read the 22:43 draft; between the two, only the guard text changed.

**Code checked.** `scm_bench/engine.py`, `fit_scm.py`, `select_scm.py`, `report.py` and `make_lock.py`, as committed in `be5cc210a`.

**Other inputs.** Designer 1's `PROPOSAL.md` (§2, §11, §13); `FIELD_MATRIX.md` (P1–P3); `PROPOSAL_COMPARISON.md`; `VERIFICATION.md` (C2); the frozen-hybrids `ANALYSIS_LOCK.json`.

**Primary sources, re-read today on the rendered pages.**
- Chechik and Tishby, NIPS 2002: Eq. 1, §3, Eq. 6, §5, §6.2.
- Slonim and Tishby, NIPS 1999: §1.2, §2.

**Access.** No real-person array was opened and nothing was fitted. Every number below comes from invented data and is reproduced by `math_checks.py` → `MATH_CHECKS.json`.

**Verdict.** The §4 identities, the KL k-means rules, the SEX gain and the PUB_IBSI signs are correct. The guard defect I found in the draft is already fixed in the committed version. Five MUST-FIX items remain:
- M1–M2 are text errors in frozen rules; the code is already right.
- M3–M5 are claim-rule gaps that exist in both the protocol and `report.py` / `select_scm.py`.

## Resolved before filing: the guard's shrinkage target

The draft re-shrank each merged cell toward its new mean μ_m. With κ > 0, n_{c,s}ỹ_{c,s} is not additive, so cumulative spending then depends on the order of merges.

Invented example, three merge orders reaching the same final cell:

| SEX | Spending under the three orders | Fixed-law value |
|---|---|---|
| 0 | 0.0756, 0.0657, 0.0648 | 0.0626 |
| 1 | −0.0401, −0.0350, −0.0524 | −0.0340 |

The committed §4 fixes ỹ at the fine cells and carries Ỹ_{c,s} = Σ_{f⊆c} n_{f,s}ỹ_{f,s}. Its Δ_s is algebraically identical to the fixed-law cost, and `engine.py` implements it. The single-merge identity holds to 3e-17 against brute force. Keep it. One nit: the §4 statistics list should name Ỹ_{c,s}, not "the sum of one-hot occupation labels".

## MUST-FIX

**M1. F1 sign reversed (§12.5).** `select_scm.py` already computes TWIN − SCM. Replace "as SCM − TWIN ≥ 0.020 on both seeds" with:
> as TWIN − SCM ≥ 0.020 on both seeds (SCM's proxy at least 0.020 below its matched twin's); otherwise F1 is reported as refuted.

**M2. Nonexistent clause count (§7.4).** There are 24 MODEL_SELECT clauses, 12 per seed; `select_scm.worst` already uses 24. Replace "max over the 48 clauses of value/cap" with:
> max over the 24 MODEL_SELECT clauses (12 per seed) of value/cap

**M3. Four claims at α/3 (§9; `make_lock.py` `n_claims: 3`; `report.py` claim ATTRIBUTION).** A, Attribution, B and C each have clause bounds, so the family-wise error can reach 4α/3. Replace the claim-clause bullet with:
> Claim clauses are one-sided at α/3, α = 0.025, one share each for A, B and C. Attribution is tested in fixed sequence inside A's share. Its four clauses (L1 > 0 against IBSI and against SHUF, on both seeds, combined by IUT) are evaluated at one-sided α/3 only if claim A is established; otherwise they are reported descriptively. The family-wise error over {A, Attribution, B, C} is at most α.

Make `report.py` set the Attribution decision to NOT_TESTED_A_NOT_ESTABLISHED when A is not established. The alternative, α/4 for all four claims, raises the critical t from 2.395 to 2.499 at df 3,954.

**M4. Unmatched twin capacity does not affect claim A (§5 TWIN, §9).** A twin left with more cells than K_β is expected to leak more, which biases the benefit toward SCM. `report.py` reads `twin_matched_capacity` from SELECTION_LOCK, but `select_scm.py` never writes that field and the claim A decision ignores it. Add to §9 claim A:
> If TWIN_b stops above K_β, its RECIPE keys are computed and shown, but claim A is UNMATCHED_CAPACITY (not established).

Write that flag in selection and test it in report.

**M5. Attribution can count an ineligible PUB_IBSI as beaten.** §7.6 shows an ineligible IBSI "descriptively", and §8 says ineligible releases are "never counted as beaten". But §9 and `report.py` establish Attribution without checking `sel['ibsi']['model_select_point_feasible_24']`. Add to §9 Attribution:
> If the PUB_IBSI nominee is utility-ineligible, Attribution is COMPARATOR_INELIGIBLE (not established) and its IBSI keys are descriptive. An ineligible SCM fallback makes Attribution NOMINEE_INELIGIBLE, as it does A and B. TWIN is a matched control and counts whatever its eligibility.

## SHOULD-FIX

**S1. Priority conventions (§5, §12).** Clipping at max(Δ_k, 0) and adding 1e-12 are departures from designer §2. They are needed because guard changes can be negative, and so can **exact grouped income spending relative to C0**: regrouping can improve the release. In 4,000 invented single merges, Brier spending was negative 318 times, NLL 315 times and accuracy 228 times. Add to §12:
> The designer's G/Σ_k Δ_k/R_k is used with negative Δ_k clipped to 0, a 1e-12 floor on the denominator, and R_k = max(B_k − spent_k, 1e-15) on the arm's own path. This is engine.py's floor, so no 0/0 arises. Income spending is the exact grouped value and, like the guard, can be negative, so R_k can exceed B_k.

**S2. Mixed income costs (§12.2).** Mixing is reasonable but under-declared. Replace §12.2 with:
> Admissibility uses the exact grouped income costs, recomputed by regrouping each candidate partition. For tractability, the SCM, TWIN and SHUF priorities keep designer §2's cell-level income Δ_k, with R_k the remaining grouped budget. The two can differ in either direction.

The appendix example shows both directions: a same-group merge has exact cost 0 against 0.015 at cell level; a cross-group merge has exact cost 0.0296 against 0.015. All three arms share this proxy, so the recipe contrast is not biased.

**S3. PUB_IBSI is faithful as the hard limit; state it.**
- **Objective.** IBSI §3: "when β goes to infinity the Lagrangian reduces to I(T;Y⁺) − γI(T;Y⁻)". AIB merges the pair with the smallest (p_i + p_j)·JS_Π. IBSI §5 lists "greedy agglomerative hard clustering [13]".
- **Signs.** ΔI is information lost, so ΔI ≥ 0. The JS forms equal the plug-in mutual-information drops to about 1e-16. Minimising ΔI⁺ − γΔI⁻ is therefore the greedy step for maximising I⁺ − γI⁻, and `engine.py` minimises nll1 + nll2 − γ·sexinfo.
- **Append to the IBSI row:**
  > Hard limit: IBSI's β → ∞, unrelated to the SCM budget β. X := the occupied fine cells, with p(x) = n_c/N. Plug-in conditionals are the cell means of OOF t1, t2 and π̂. There are two relevance variables of unit weight (IBSI Eq. 6), so the relevance term is I(T;Y1) + I(T;Y2), not I(T;(Y1,Y2)). The paper's own runs used sequential IBSI with a finite-β target. The path is independent of β, which only sets the stopping count.
- **γ.** The paper reports only γ = 0 (plain IB) and γ = 1 (§6.2: "IBSI (γ = 1) is compared with the IB method (i.e. γ = 0)"). Replace §12.6's rationale with:
  > γ = 1 is the published value; 0.5 and 2 are data-free factor-2 brackets.

  The §3 reading of γ = 1 as a likelihood ratio needs Y⁺ and Y⁻ to share an alphabet, which they do not here.
- **Diagnostic.** Report the share of IBSI merges with negative cost.

**S4. SHUF (§5, §7.7).**
- `fit_scm.py` correctly permutes rates with masses fixed (spi′ = n·π_{σ}). Say so in the protocol: "SEX rate" means the π̂ mean over a fine cell, and rates are permuted, not sums.
- Add, in both the protocol and `select_scm.py`: "a SHUF run that stops above K_β cannot be the shuffled nominee". The current code picks among eligible runs without checking the cell count.

**S5. Fixed K (§9).** `report.py` already keeps MISSING keys in K. Add the sentence:
> K_primary = 42 is fixed and is never reduced. MISSING or descriptive-only keys keep their slots and carry no status. The critical value stays t(1 − 0.05/84; G − 1), which is 3.244 at G = 3,955.

**S6. Make the frozen text match the code.**
- §3: seeding samples ∝ max(0, min_j KL-sum), the divergence itself and not its square; a cell that empties during Lloyd keeps its centroid.
- §7.2: reader seeds are MODEL_SEED_BASE + s, and LOOKUP's π is the fitting half's female rate.
- §5 TWIN: "Σ_k runs over the same eight costs as SCM, both guards included, so SCM and TWIN differ only in G. The guard shapes TWIN's merge order as well as its admissibility."

## NOTES

- **N1. Occupation identities.** Ward (Brier), KL (NLL) and accuracy are exact: maximum error 1.6e-17 over 300 invented merges, including one-hot rows and argmax ties. The accuracy form equals designer 1's.
- **N1. SEX gain.** G is the exact decrease of L(C) = Σ_c (n_c/N)(π_c − π̄)². L(C) is a Bayes SEX-Brier advantage only insofar as π̂ is the true propensity.
- **N2. Centroid.** Σ_i KL(t_i‖μ) = const − Σ_k S_k log μ_k is minimised on the simplex at μ = S/n, so the member mean is correct for this KL direction. The reverse direction, KL(μ‖t), would give the normalised geometric mean.
- **N2. Assignment.** argmin_c KL equals argmax_c Σ t log μ_c (0 mismatches in 200 checks).
- **N2. Zeros.** Clipping keeps 0·log 0 = 0. A term with t_k > 0 and μ_k = 0 becomes a finite 690.8·t_k instead of +∞. That cannot happen for occupied cells, because teacher probabilities are strictly positive.
- **N3. Selection proxy.** No leakage: MODEL_SELECT is disjoint from every fit, the halves are split by household, orientation is fixed, and keeping a mean below 0.5 is correct. Biases:
  - The maximum over three readers biases the proxy upward by a roughly common amount.
  - LOOKUP has about 4,000 people per half, i.e. 10–40 per cell at K = 100–400. It under-reads finer partitions, which favours smaller β.
  - The winner's curse in the β, γ and SHUF choices does not reach the DEV endpoints, which use fresh readers. Choosing the most protective IBSI and SHUF is conservative.
- **N4. Endpoint signs** match the frozen-hybrids lock. K = 2 + 4 + 2 + 2 + 2 + 2 + 24 + 4 = 42. Two things to state: the sign of the secondary "SCM vs C0" keys (C0 − SCM), and whether nested 0.020 also applies to H_LJ.
- **N5. Budgets versus caps.** By design, the β = 1.5 budgets (NLL 0.015, guard 0.0075) exceed the held-out NLL cap and the subgroup margin, and C0's own excess over U0 is not budgeted. Selection enforces held-out utility, not the budgets.
- **N6. Optional extra path.** The γ = 0 path (pure AIB, no caps) costs nothing. It would separate, descriptively, the effect of the caps from that of the SEX term.

## Grouped income costs: formulas and check

The income groups g have masses n_g and sums S_g = Σ_{i∈g} t1_i, and the released vector is ν_g = S_g/n_g. Under the teacher law:

| Metric | Plug-in value | Excess over the OOF teacher (engine.py's form) |
|---|---|---|
| Brier (summed) | 1 − Σ_g ‖S_g‖²/(n_g N) | [Σ_i ‖t1_i‖² − Σ_g ‖S_g‖²/n_g]/N |
| NLL | −Σ_g Σ_k S_gk log(S_gk/n_g)/N | [Σ_i Σ_k t_ik log t_ik − Σ_g Σ_k S_gk log(S_gk/n_g)]/N |
| Accuracy | Σ_g max_k S_gk/N | loss: [Σ_i max_k t_ik − Σ_g max_k S_gk]/N |

Spending is value(C) − value(C0), or value(C0) − value(C) for accuracy. The teacher terms cancel in that difference, so group sums and masses are enough.

**Invented check.** 10 people with P(income = 1) = (.1, .2 | .3, .4, .5 | .6, .7 | .8, .9, .7) in 4 cells. The protocol's rule, with 2 groups in place of 32, gives groups {c0, c1} and {c2, c3} with ν = (0.30, 0.74). For C0: Brier 0.4024, NLL 0.5920, accuracy 0.72, all equal to brute force.

| Merge | Groups after | Exact grouped change (Brier / NLL / accuracy loss) | Cell-level change |
|---|---|---|---|
| c0 + c1 (same group) | unchanged | 0 / 0 / 0 | 0.015 / 0.0190 / 0 |
| c1 + c2 (across groups) | {c0, c12}, {c3} | +0.0296 / +0.0293 / 0.060 | 0.015 / 0.0152 / 0.060 |

A random check with 32 groups gives a maximum error of 2.2e-16.
