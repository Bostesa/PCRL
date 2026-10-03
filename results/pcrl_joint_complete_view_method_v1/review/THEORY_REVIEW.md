# Theory review: joint complete-view (JCV) specification

**Role:** read-only mathematics reviewer, 2026-10-03. Numbers come from two small numpy/scipy/sklearn scripts kept
outside the repository (not committed); LEACE in the toys is implemented from Belrose et al. Thm 4.2-4.3 (the
official package was not installed), so the toys only illustrate algebra. **Labels:** [identity] elementary algebra;
[standard] established result; [conditional] true only under stated conditions. **Nothing below is new
mathematics;** the only spec-specific observation is block-disjointness (§7).

## Spec statements found false or misleading, and design gaps

1. **"Normalising by `H_fit(S)` makes β comparable across views" is false as stated.** All three views share the
   same `S` and the same fitting rows, so `H_fit(S)` is one common constant. It rescales β globally and cannot change
   the relative weight of `R_1`, `R_2` and `R_pair`. It makes β comparable across *attributes or datasets* only. (§8)
2. **"Exact for at most two constraints" is misleading.** `g1, h1` and `g2, h2` share no parameters, so
   `a_1 · a_2 = 0`: the exact projection is independent one-constraint (A-GEM) projections for any number of such
   guards, and collinear/anti-collinear cases cannot occur. If any parameter *is* shared, the unprojected task term
   can itself raise an active guard to first order. (§7)
3. **First-order non-increase holds for the projected privacy part, not automatically for the full step.** The full
   `d` also needs `a_j·(a_1+a_2) ≥ 0` (true when block-disjoint) and the same gradient estimate in the task term and
   in `a_j`; activity is judged on a *guard subset*, so a minibatch task gradient is not `a_j`. (§7, §10e)
4. **Fixed-epoch refits do not by themselves guard the release.** A map frozen while the encoder moves does not
   guard it; the last refit must follow the last encoder update. (§10a)
5. **Design gap: J vs L is confounded by penalty mass.** L decouples into two independent recipients; J adds a third
   term, and when the pair term's best critic ignores one view, J is L with one local term double-weighted. (§5)
6. **`R_v ≥ 0` only on the rows where the constant prior was fitted;** it can be negative on held-out rows. (§8)

## 1. Concatenation of per-view guarded features [identity; conditional on a common law]

`Cov([r_1; r_2], S) = [Cov(r_1, S); Cov(r_2, S)]` (stacked blocks). No `r_1`-`r_2` cross term appears. So zero blocks
under one law give zero for the concatenation, and by LEACE Thm 3.4 the concatenation is linearly guarded under
that law.

**What "the same law" means.** The official fitter uses empirical moments, so the guarantee is on the
defense-fit rows at refit time. Toy check (2 000 fit rows, 2 000 other rows):

| Quantity | max\|Cov\| |
|---|---|
| Fit rows: `r_1`, `r_2`, concatenation | 7.9e-16, 3.2e-16, 7.9e-16 |
| Held-out rows: concatenation | 2.0e-2 (the O(n^-1/2) sampling gap; the population value is not 0 either) |
| `r_2` fitted on a different subset F', scored on F | 1.4e-2 |

**Consequence.** With both maps fitted on one row set, the coalition is linearly guarded in-sample *by construction*.
J's pair term can then act only on nonlinear dependence, and PCRL's old linear cross-purpose R² constraint would be
vacuous here.

## 2. Affine vs nonlinear post-processing [identity + counterexample]

**Affine.** `Cov(A r + b, S) = A Cov(r, S) = 0`. Centred logits (row-centred, or minus a fixed offset) are affine in
`r_i`, so `v_i` is guarded whenever `r_i` is.

**Nonlinear: counterexample.** Take `r | S=1 = 0`, and `r | S=0 = -1` w.p. ¾ or `+3` w.p. ¼. Then
`E[r|S=0] = E[r|S=1] = 0`.

| Post-processing | `S=0` | `S=1` |
|---|---|---|
| Binary softmax `σ(r)`, conditional mean | 0.4398 | 0.5000 |
| Hard argmax `1[r>0]`, conditional mean | 0.25 | 0 |

Three-class head `l = W r` with `W = [[1,0],[0,1],[-1,-1]]` and guarded 2-d `r`:
- `r | S=0 ∈ {(±2, 0)}` gives mean softmax `(0.441, 0.117, 0.441)`;
- `r | S=1 ∈ {(0, ±2)}` gives `(0.117, 0.441, 0.441)`.

Both conditional means of `r` are 0.

## 3. Adding a view [standard] and finite fitted AUC

**Bayes risk.** For any loss, `inf_f E ℓ(f(V1,V2), S) ≤ inf_f E ℓ(f(V1), S)`, because `f(V1)` is a candidate.

**AUC.** Every test on `V1` is a test on `(V1, V2)`. By Neyman-Pearson, the likelihood-ratio test on `(V1, V2)` has
the highest power at every size, so the optimal ROC, and hence the optimal AUC, cannot decrease when a view is added.

**Finite attackers.** A fitted attacker is not Bayes. Toy: one informative 2-d view, plus 40 independent dimensions,
300 training rows, 32-unit MLP. AUC(pair) − AUC(view alone) = −0.093 (range −0.114 to −0.080 over 10 seeds).

**So the coalition bank must contain ignore-other-view candidates** (attackers on `v1` alone and `v2` alone, scored
as coalition candidates). Without them, an audited coalition AUC below the best local AUC is an estimation artefact,
and "synergy" (coalition − max local) can be negative noise. The same holds for the *training* bank: without them
`R_pair < max(R_1, R_2)` is possible and `P_J` is not monotone in coalition leakage; with them (reusing the local
critics) `R_pair ≥ max(R_1, R_2)` on the fitting rows.

## 4. A fixed known head adds no information [standard]

`h_i` is a fixed measurable function of `r_i`, so `σ(r_i, h_i(r_i)) = σ(r_i)` and `I(S; r_i, h_i(r_i)) = I(S; r_i)`;
Bayes risk and optimal AUC are unchanged. The head can still help a *finite* attacker by supplying a feature its
family might not build; centred logits are affine, so they cannot help a *linear* attacker. Caveats: this needs a
per-row map (batch-statistic centring at release would make the release depend on other rows); releasing
probabilities or decisions adds no Bayes information but changes the finite-attacker surface (§2).

## 5. Joint vs sequential [conditional]

Let `F_J` be J's objective and `C` the set satisfying the guards (same budgets, data roles, refit rule and critic
banks). If the S12 solution `θ^S ∈ C`, then `min_C F_J ≤ F_J(θ^S)`. This is weak dominance of J's *global optimum*,
in J's *own surrogate objective*. It does **not** imply that:
- **(i) the iterate is good.** The projected-SGD iterate need not reach `F_J(θ^S)`: the problem is nonconvex,
  critics move, and LEACE refits re-base it. Even starting J at `θ^S` gives no monotonicity, because steps are accepted on guards,
  not on `P`.
- **(ii) the audit improves.** Audited coalition AUC need not be lower: the surrogate is not the audit, and fitting
  rows are not test rows.
- **(iii) the advantage is algorithmic.** If J uses purpose-2 information when shaping `g1` and S12 does not, the
  feasible *design* set is larger by information. That is Taylor et al.'s online/offline gap (§II there).

**Same-information contrast.** J vs L holds information fixed.
- L decouples exactly: `P_L = R_1(θ_1) + R_2(θ_2)`, the task losses are block-separate and the guards are
  block-disjoint. So L is two independent single-recipient fits, coupled only by the schedule and the backtracking
  rule.
- J adds `R_pair(θ_1, θ_2)` *and* more penalty mass. If `R_pair`'s argmax critic is an ignore-other-view one, then
  `∇R_pair = ∇R_i` and J is L with `β` doubled on one term.
- **Control needed:** a β-matched L arm (or a β sweep), or report J's gain at matched local leakage. An alternative
  pair term is the excess `max(0, R_pair − max(R_1, R_2))`.

## 6. Local caps bind [identity]

`v_1` is a function of `(θ_1, X)` only, so changing `θ_2` cannot change the law of `(v_1, S)`. Hence
Bayes-coalition ≥ max(local_1, local_2), and

`coalition_J − coalition_L ≥ max(local_J1, local_J2) − coalition_L`.

The primary claim (J beats L by ≥ 0.02 on coalition AUC, with local guards) therefore needs either:
- an L synergy gap `coalition_L − max local ≥ 0.02` that J removes; or
- reductions in J's *local* leakage, which is a local effect confounded with penalty mass (§5).

**Recommendations.** Before the main run, measure L's synergy on a non-test role; if it is below 0.02, the claim is
infeasible as a coalition-specific effect. Report `Δcoalition = Δ(max local) + Δ(synergy)`.

## 7. Projection math [standard: QP projection onto a polyhedral cone]

**Problem (Q).** `min_q ½||q − p||²` subject to `a_j · q ≤ 0` for `j ∈ A`, with `|A| ≤ 2`.
- `K = {q : a_j·q ≤ 0}` is a closed convex cone containing 0, and the objective is strictly convex. So
  `q* = Π_K(p)` exists and is unique.
- With linear constraints, KKT is necessary and sufficient: `q = p − Σ μ_j a_j`, `μ_j ≥ 0`, `a_j·q ≤ 0`,
  `μ_j (a_j·q) = 0`.
- Moreau decomposition: `p = Π_K(p) + Π_{K°}(p)`, where `K° = cone{a_j}` and the two parts are orthogonal. The removed
  part is the projection of `p` onto the cone generated by the active guard gradients.

**Active-set solution.** Write `c_j = a_j·p`, `G_ij = a_i·a_j`, `det = ‖a_1‖²‖a_2‖² − (a_1·a_2)²`.

| Case | Condition | `q*` |
|---|---|---|
| 0. Zero gradient | `a_j = 0` | Drop constraint `j` (it reads `0 ≤ 0`); set `μ_j = 0` |
| 1. One nonzero `a` | — | `q* = p − max(0, c)/‖a‖² · a` (the A-GEM / PCGrad formula) |
| 2a. None active | `c_1 ≤ 0`, `c_2 ≤ 0` | `q* = p` |
| 2b. Only 1 active | `c_1 > 0` and `c_2 − c_1 (a_1·a_2)/‖a_1‖² ≤ 0` | `q* = p − (c_1/‖a_1‖²) a_1` (2b' is symmetric) |
| 2c. Both active | `det > 0`, `μ = G⁻¹c ≥ 0` | `μ_1 = (‖a_2‖² c_1 − (a_1·a_2) c_2)/det`, `μ_2 = (‖a_1‖² c_2 − (a_1·a_2) c_1)/det`; `q*` = projection of `p` onto `span{a_1, a_2}^⊥` |
| Collinear | `a_2 = κ a_1`, `κ > 0` | Same half-space, so case 1. `μ` is not unique (`μ_1 + κμ_2 = max(0, c_1)/‖a_1‖²`); `q*` is unique |
| Anti-collinear | `κ < 0` | `K` is the hyperplane `a_1·q = 0`, so `q* = p − (c_1/‖a_1‖²) a_1` for either sign of `c_1` |
| Ties | `c_j = 0`, a boundary equality in 2b, or `μ_j = 0` in 2c | Adjacent cases give the same `q*` (the projection is continuous). Only the reported multipliers differ, so pick by a fixed tolerance rule |
| Near-collinear | `det ≈ 0` | `q*` stays well defined, but `G⁻¹c` is ill-conditioned (cond ∝ 1/sin²∠): at sin∠ ≈ 1e-7 the naive solve erred by 4e-8. Use a collinearity threshold or least squares |
| Block-disjoint (this spec) | `a_1·a_2 = 0` | `G` is diagonal, so `μ_j = max(0, c_j)/‖a_j‖²` independently, which equals sequential A-GEM in either order |

**First order.** Each active guard: `a_j·q* ≤ 0` by feasibility. The surrogate: `∇P·q* = −(1/β) p·q* =
−(1/β)‖q*‖² ≤ 0` (Moreau), so the projected part never raises `P` to first order. **Full step**
`d = −(a_1+a_2) + q*`: `a_j·d = −‖a_j‖² − a_i·a_j + a_j·q*`.
- Block-disjoint guards: `a_j·d ≤ −‖a_j‖² ≤ 0`.
- Shared parameters, counterexample: `a_1 = (1, 0)`, `a_2 = (−2, 0.1)`, `p = 0` gives `q* = 0` and `a_1·d = +1 > 0`.
- Nothing bounds `∇P·(−a_1−a_2)`, so `P` may rise along `d`.

**Why first order says nothing more.**
- *Finite steps:* `L_j(θ+ηd) = L_j + η a_j·d + ½η² dᵀ∇²L_j d + o(η²)`; a tight guard (`a_j·q* = 0`) rises under
  positive curvature. *Inactive guards* are unconstrained and can cross the ε-band in one step; backtracking catches
  this on the guard subset only, not on held-out data.
- *Critics change* after the step, so `p` was the gradient of a different `P`. *LEACE refits* change `r_i`
  discontinuously and are not steps, so guard losses can jump with no acceptance test (§10f).

**Independent check (closed form vs `scipy.optimize.minimize(method='SLSQP')`, R^12, `ftol` 1e-14).** 5 900
instances: 3 000 random draws with `|A| ∈ {0,1,2}`; 300 each of zero `a_1`, both zero, collinear, anti-collinear,
`a·p = 0` tie, `p = 0`, `p` in the polar cone, near-collinear; and 500 block-disjoint.

| Case | max \|q_cf − q_SLSQP\| | min (obj_SLSQP − obj_cf) | max feasibility violation | max KKT residual |
|---|---|---|---|---|
| All non-near-collinear cases | ≤ 2.5e-13 | ≥ −2.0e-12 | ≤ 1.7e-14 | ≤ 1.7e-15 |
| Near-collinear | 3.7e-8 | ≥ −3.6e-14 | 4.9e-8 | 9.3e-16 |

The closed form is never beaten by SLSQP beyond round-off; SLSQP flagged non-success on 2/300 polar-cone cases but
returned the same point (`q* = 0`). Block-disjoint: exact = sequential A-GEM (max difference 6.7e-16). General
`a_1, a_2`: a one-pass sequential (PCGrad-style) projection differs from the exact one in 553/2000 draws (max 1.13).
Full direction: an active guard rises to first order in 3/2000 general draws and 0/2000 block-disjoint draws.

## 8. The surrogate `R_v = 1 − min_a CE_a / H_fit(S)` [identity + standard]

- **Best critic.** `R_v = max_a (1 − CE_a/H_fit)`, attained by the lowest-CE critic. Its encoder gradient, with
  critics held fixed, is `−∇CE_{a*}/H_fit` at a unique `a*` (finite-max Danskin), and a subgradient at ties.
  - If the constant prior is best, the gradient is **zero**: no pressure until a critic detects leakage.
- **Sign.**
  - The constant prior is the empirical marginal of the rows `F` it is fitted on, so `CE_const(F) = H_F(S)` and
    `R_v(F) ≥ 0`.
  - Elsewhere, `CE_const = H_eval + KL(p_eval || p̂_F)`, so `R_v` can be negative. Toy: P(S=1) of 0.30 on the fit rows
    vs 0.49 on the evaluation rows gives `R = −0.26`; small negatives also arise from ordinary sampling noise.
- **Not MI.** For a fixed critic in population, `H(S) − CE_q ≤ I(S; V)` (Barber-Agakov; the same as Song et al.'s
  `C2`).
  - So population `R_v` lower-bounds the uncertainty coefficient `I(S; V)/H(S)`: it can show leakage, never certify
    its absence.
  - In-sample with fitted critics it has no fixed direction: a memorising critic gives `R_v > 0` even when `S ⊥ V`.
- **Not an AUC bound.** On the XOR fixture with linear critics, `R_v = 0` while an MLP reaches AUC 1.000 (§9).
- **Normalisation.** `H_fit(S)` is identical for `v1`, `v2` and the pair (same `S`, same rows), so it is **not** a
  per-view calibration (finding 1).

## 9. XOR fixture [identity]

**Setup.** Let `A, B` be iid Bern(½) and `S = A ⊕ B`.
- `P(S=1 | A=a) = P(B ≠ a) = ½`, so `I(A; S) = I(B; S) = 0`.
- `S` is a function of `(A, B)`, so `I(A,B; S) = 1` bit.
- `Cov(A, S) = P(A=1, B=0) − ¼ = 0`. So even the *coalition* is linearly guarded: `S = A + B − 2AB` lives in the
  product term.

**Task-bit extension.** `Y1, Y2` are iid Bern(½), independent of `(A, B)`; `v1 = (Y1, A)`, `v2 = (Y2, B)`.
- **Locally safe.** `(Y1, A) ⊥ S`, because `S | (Y1, A)` is `A ⊕ B` with `B` fresh.
- **Jointly identifying.** The pair determines `S`.
- **Dropping `A` from `v1`:** task 1 is untouched (`Y1` is retained), and `(Y1, Y2, B) ⊥ S`. The coalition clue is
  gone at zero task cost (dropping `B` is symmetric).

| Check (10 000/10 000 split) | MLP AUC | Linear AUC |
|---|---|---|
| `v1` alone | 0.499 | — |
| `v2` alone | 0.495 | — |
| Pair | 1.000 | 0.497 |
| Pair after dropping `A` | 0.493 | — |

**What each arm sees.** Under L, `R_1 = R_2 = 0` (the constant prior wins), so there is no gradient to drop `A`, and
a randomly initialised encoder keeps `A` incidentally. Under J, the pair critic sees it. Per-purpose LEACE does
nothing here. **Forced trade-off variant:** `Y1 = A`; dropping `A` now costs task 1, and the guard should block it.

## 10. Counterexamples to the specification

- **a. Frozen LEACE map, moving encoder.** Fit on `z_0` (leakage along `e_1`): max|Cov| = 1.3e-16. Update the
  encoder so leakage also enters along `e_2`, keeping the map: max|Cov| = 0.250. LEACE annihilates only the fit-time
  `colsp Σ_XZ` (Thm 4.1). **Fix:** refit after the final encoder update, then refit or re-check the heads, freeze
  everything, and report held-out covariance.
- **b. Logits vs softmax.** Centred logits are affine (guarded); recipient-side softmax or argmax are not (§2). This
  adds no Bayes information (§4), but "linearly guarded release" does not mean "guarded decisions".
- **c. Weaker training critics than audit attackers.** J minimises only `max` over its own bank. With linear critics
  on XOR, `R_pair = 0` while the audit reaches 1.0. MLP critics with a finite budget or early stopping can miss
  structure that gradient boosting or longer-trained attackers find. This is the circumvention pattern of LEACE §7 and
  of the PCRL erase layer.
  **Report** training-bank AUC beside audit AUC on the same split (a "critic deficit"), and require the audit bank ⊇
  the training-bank architectures.
- **d. Shared parameters.** The task term can raise an active guard (§7 counterexample).
- **e. Guard subset vs minibatch.** If `a_j` is computed on the guard subset and the task term on a minibatch,
  `a_j(G)·(−∇L_j(B))` can be positive even with disjoint parameters. State which estimate each term uses.
- **f. Refit jumps.** A LEACE refit can push a guard over budget with no step to reject. Re-evaluate the guards after
  each refit, and log the jumps.
- **g. Pair bank without ignore-other-view critics.** `R_pair` can fall below `max(R_1, R_2)` (§3).

## 11. FARE extends to the coalition [identity + `jovanovic2023fare`]

For any score, `AUC − ½ = ∫₀¹ (TPR(u) − u) du ≤ sup_u (TPR(u) − u) ≤ TV(P_{Z|S=1}, P_{Z|S=0})`, where the ROC uses
randomised tests. The last term equals `sup_g Δ_DP(g)`, which FARE certifies.

`(f_1(x), f_2(x))` is a restricted encoder with at most `k_1 k_2` cells. If everything released is a function of the
cells, FARE's procedure run on rows unused by either tree gives `AUC_coalition ≤ ½ + T` with probability `1 − ε`
against *any* attacker. The neural arms have no analogue.
