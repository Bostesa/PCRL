# METHOD — nonlinear moment refinement (`nonlinear_moment_refinement`)

Written before any new interface was fitted. Every scale, seed and constant below is prospective.
Implementation: `experiments/pcrl_nonlinear_rank_v1/{nonlinear_moment,objective,maps}.py`.

## 1. What is reused unchanged

From the frozen 2018 `SpectralModel` of each seed (`results/redesign_20260910_acs_residual_spectral_v1/seed_N/maps.joblib`):

| Symbol | Meaning | Shape (seed 0) |
|---|---|---|
| `V` | whitened residual features on the representation-fit rows; training mean 0, training covariance `I_q` | `(10513, 128)` |
| `q` | whitened rank | `128` |
| `R` | teacher residual `T - Q_A Q_A^+ T`, recentred, from **original unstandardised** `T` | `(10513, 32)` |
| `U` | trace-normalised utility matrix, `U_raw = (V'R/n)(R'V/n)` | `(128, 128)` |
| `Q_A` | training-only degree-2 polynomial anchor basis in `hA[:,[1,3]]`, intercept retained | `(n, 6)` |
| `Q_AB` | same in `[hA[:,[1,3]], hB[:,1]]` | `(n, 10)` |
| `m_j(H_c)` | the original three-fold **household** OOF nuisance predictions, reconstructed exactly by applying saved fold model `f` to the rows of fold `f` using the saved `fold_assignments` | — |
| `P_j` | trace-normalised historical moment Gram of attribute `j` | `(128, 128)` |

The protected residual for role `j` is, exactly as before, with **all `K` one-hot columns and no reference-class deletion**:

```
e_j = onehot(S_j) - m_j(H_c)
```

Valid rows are that attribute's own mask `y_j >= 0`; the divisor is that attribute's valid row count `n_j`. Nuisances are frozen and shared by the old and new arms.

## 2. Reference projection (policy-independent)

For each seed and each rank `r`, the **common reference projection** is the top-`r` eigenvectors of `U` alone:

```
W_ref(r) = top-r eigenvectors of U,   sign-canonicalised
Z_ref    = V W_ref(r)
```

`W_ref` does not depend on the policy or the penalty family, so all policies of one rank and seed share one feature family and one set of normalisers. Residence and commute labels are never used anywhere in its construction.

## 3. The nonlinear feature family `chi_r(Z)`

Fitted once at `Z_ref`, on at most `min(1024, n)` rows chosen without replacement by `default_rng(20260918 + 100*seed + r)`:

* **Bandwidth.** `sigma_ref` = median strictly positive pairwise Euclidean distance among those rows' reference-projection coordinates. If no positive distance exists, fitting aborts with a bandwidth diagnostic (it never silently falls back).
* **Fourier directions and phases.** `default_rng(20260919 + 100*seed + r)` draws `omega ~ N(0,1)` of shape `(r, 96)` and `phase ~ U[0, 2pi)` of length 96. The 96 columns split into three blocks of 32; block `m` is divided by `band_m * sigma_ref` with `band = (0.5, 1.0, 2.0)`.
* **Feature scaling.** The raw feature mean and standard deviation are measured on the same reference rows and frozen.

The family is

```
chi_r(Z)[i,:] = ( [ z_ia z_ib for a<=b ] ++ [ sqrt(2/32) cos(z_i . omega_m/(band_m sigma_ref) + phase_m) ] - mu_ref ) / s_ref
```

giving `F = r(r+1)/2 + 96` features: **232** at `r=16`, **132** at `r=8`.

Conventions, declared in advance:
* A whitened reference projection cannot produce an exactly constant monomial, but a near-constant one would; any column with reference standard deviation `<= 1e-12` keeps scale `1` and is **flagged**, never amplified. Degenerate columns are listed in the diagnostics.
* Zero pairwise distances are counted and reported; only strictly positive ones enter the median.
* `mu_ref`, `s_ref`, `omega`, `phase` and `sigma_ref` are **frozen for every arm**. No bandwidth or feature-scale quantity is ever recomputed from the current optimised `W` — doing so would make the objective self-referential.

## 4. Conditional nonlinear moment blocks

For protected role `j` with recipient basis `b(H_c)` (`Q_A` for local roles, `Q_AB` for coalition roles, intercept retained so conditional cancellation such as XOR is representable):

```
M_j[f, k, c](W) = (1/n_j) * sum_{i valid} chi_r(V W)[i, f] * b(H_c)[i, k] * e_j[i, c]
```

Contractions are streamed in row chunks; **no n-by-n kernel matrix is ever formed.** Splitting the feature index `f` at `r(r+1)/2` gives the two blocks:

```
D_j^quad(W) = sum over quadratic f, all k, c  of  M_j[f,k,c]^2
D_j^rff(W)  = sum over Fourier   f, all k, c  of  M_j[f,k,c]^2
```

Both are reported separately as well as together.

**Fixed training-reference normalisers**, measured once at `W_ref` and frozen, with declared positive floor `FLOOR = 1e-12`:

```
N_j^quad = max(D_j^quad(W_ref), FLOOR)
N_j^rff  = max(D_j^rff(W_ref),  FLOOR)
```

The two blocks then receive **equal block weight** `1/2`:

```
D_j(W) = 0.5 * D_j^quad(W)/N_j^quad + 0.5 * D_j^rff(W)/N_j^rff
```

so that `D_j(W_ref) = 1` exactly. The raw scales `D_j^quad(W_ref)`, `D_j^rff(W_ref)`, the supported classes and any floored normaliser are recorded per role in `fit_diagnostics.json`. A role whose reference block is at the floor is flagged.

## 5. Combined role penalty and the three policies

The original projected linear-moment term is unchanged:

```
lin_j(W) = tr(W' P_j W)
```

The nonlinear term is **calibrated to the original role term at the common reference projection**:

```
c_j = max( lin_j(W_ref), FLOOR )        # equals lin_j(W_ref) unless degenerate
```

and the role penalty is the **equal-weight average** of the two:

```
penalty_j(W) = 0.5 * lin_j(W) + 0.5 * c_j * D_j(W)
```

At `W = W_ref` this equals `lin_j(W_ref)` exactly, which is the sense in which the scales are matched. If `lin_j(W_ref) <= FLOOR` the calibration is degenerate: the floor is used, the role is flagged, and that flag is carried into every table. **Matching a reference penalty scale is not equal privacy strength away from that projection**, and nothing below claims it is.

Aggregation is identical to the historical arms, with the same fixed denominators 3 and 2 — an unsupported or zero attribute does **not** shrink the denominator and never counts as a privacy success:

```
local_agg(W) = (1/3) * [ penalty_{A/SEX} + penalty_{A/RAC1P} + penalty_{A/public_coverage} ]
ab_agg(W)    = (1/2) * [ penalty_{AB/SEX} + penalty_{AB/RAC1P} ]
```

The three policies, matching the historical `spectral_L1`, `spectral_L2`, `spectral_C1`:

| Policy | Penalty |
|---|---|
| `L1` | `local_agg(W)` |
| `L2` | `2 * local_agg(W)` (equal total mass, local only) |
| `C1` | `local_agg(W) + ab_agg(W)` |

Class-support flags are preserved; no hard target is removed.

## 6. The optimised objective and its solver

Reported as a **training objective to be MINIMISED**, so that "lowest training objective" is unambiguous:

```
L(W) = penalty(W) - tr(W' U W) ,     subject to  W'W = I_r
```

The maximised matrix objective of the historical baseline is `-L(W)`.

**Penalty family `original`.** Setting `penalty_j = lin_j` makes `L(W) = -tr(W'(U - lambda P)W)` exactly, so the closed-form top-`r` eigenvectors are the global optimum of that fixed matrix. At `r = 16` this reproduces the historical arms bitwise (verified, `VALIDATION.md`). This is the only place a global-optimality statement is made.

**Penalty family `nonlinear`.** `D_j` depends on `W` through `chi_r(VW)`, so the objective is **not** a fixed quadratic form in `W`. There is no matrix to diagonalise, Ky Fan does not apply, and the closed-form guarantees of the spectral ARL line are **not** inherited. The objective is generally nonconvex and the solver is iterative:

* Riemannian gradient `G_R = G - W sym(W'G)` where `G = dL/dW` is analytic (§6.1).
* Retraction `W <- qf(W - t G_R)`, the QR `Q` factor with positive `R` diagonal.
* Backtracking line search from step `1.0`, shrink `0.5`, at most `20` backtracks; a step is accepted only if it improves `L` by more than `1e-14`.
* Stop on budget, on Riemannian gradient norm `< 1e-12`, or on line-search failure.
* Budget **200 full-objective updates per start**, frozen after the synthetic checks and one training-only runtime calibration.

**Two deterministic starts per unique condition:** (a) the corresponding original-moment closed-form solution; (b) a fixed perturbed/retracted version, `qf(W0 + 0.05 * G/||G|| * ||W0||)` with `G` drawn once by `default_rng(20260920 + 100*seed + r)`. The **unmoved initial point of each start is itself an eligible checkpoint.** The selected map is the **lowest training objective** across both starts and both initial points, ties broken by start name. Selection never consults attackers, residence, commute, development outcomes or the transport table. If selection returns the original point, that is recorded as a valid no-improvement outcome.

### 6.1 Analytic gradient

With `Z = VW`, `arg = Z omega_scaled + phase`, `g_chi` the derivative with respect to the standardised features and `g = g_chi / s_ref`:

```
dL/dM_j[f,k,c] = 2 * w_f * M_j[f,k,c]                      w_f = block weight / block normaliser
g_chi[i,f]     = (2/n_j) * sum_{k,c} w_f M_j[f,k,c] b[i,k] e_j[i,c]
```

Quadratic block: with `Gsym[i]` the symmetric matrix holding `g[i, idx(a,b)]` off-diagonal and `2 g[i, idx(a,a)]` on the diagonal,

```
dZ_quad[i,:] = Gsym[i] @ z_i
```

Fourier block:

```
dZ_rff[i,:] = ( -(g[i, F_q:] * sqrt(2/32)) * sin(arg[i,:]) ) @ omega_scaled'
```

and finally `dD_j/dW = V' (dZ_quad + dZ_rff)`. The linear term contributes `2 P_j W`, the utility term `2 U W`. Verified against central finite differences.

Feature collapse cannot masquerade as privacy: the whitening fixes `Z'Z/n = I_r` for any orthonormal `W`, so the utility term is covariance-normalised and no arbitrary rescaling of `Z` is available to the optimiser. Orthogonality residuals are recorded for every accepted map.

## 7. Endpoints and rules reused from the transport study

Reused verbatim for comparability and labelled as reused. Sources: `results/redesign_20260917_acs_spectral_transport_v1/{PROTOCOL.md,COMPARISONS.json}` and `scripts/report_acs_spectral_transport.py`.

* Five utility tasks: `income_binary`, `civilian_at_work`, `same_residence` (A-view); `public_coverage`, `commute_over20` (B-view).
* Eleven forbidden roles: `A/{public_coverage, commute_over20, SEX, RAC1P}`, `B/{income_binary, civilian_at_work, same_residence, SEX, RAC1P}`, `AB/{SEX, RAC1P}`. `RAC1P` is scored on all nine classes.
* Four family sensitive endpoints: `recovery/A/SEX`, `recovery/AB/SEX`, `recovery/A/RAC1P`, `recovery/AB/RAC1P`, plus `utility/same_residence`.
* Attack families per role: `logistic` (C=1, lbfgs, max_iter 500), two **restarted** MLPs `mlp_0`/`mlp_1` (hidden [64,32], seed offsets 0 and 10000, one 360-epoch trajectory with nested 120/360 checkpoints), two boosted trees `hist_gb_20`/`hist_gb_5` (HistGB, 150 iters, 15 leaves, min_samples_leaf 20 and 5), and the kernel family (256 random Fourier features, ridge `alpha in {1e-4, 1e-2, 1}`, median-distance bandwidth on training rows only). Role seed `1260000 + 100*seed + 10*view_index + target_index`; kernel seed `20263910 + 100*seed + 10*view_index + target_index`.
* Legal `H`-only ancestors routed to the anchor coordinates (`anchor__*`) and singleton-to-coalition projections (`inherited_A__*`, `inherited_B__*`) are included, exactly as for the historical arms.
* Selection: minimum **unweighted** attacker-validation log loss, then candidate ID. The diagnostic `saved_adversary` is excluded from every scope.
* Budgets 120 and 360; primary budget 360.
* Withholding controls at `p in {0, .25, .5, .75, 1}` for the frozen sources `E`, `A0`, `L025`, `L20`, `J` and `spectral_S0`, and for the new candidates under the **same** predetermined grid, using the existing explicit branch-routed mechanism with a fixed per-person independent branch (`sha256('...|year|seed|condition|SERIALNO|SPORDER')`). Expected loss interpolates as `(1-p) L_H + p L_aug` for this routed mechanism **only** — never for interpolated features or probability vectors, and the branch must persist for a person's release.

New candidate families: **none introduced.** The new interfaces use exactly the slate the historical spectral arms used, so attack opportunities are matched without adding anything to the controls.

Attack scopes are kept distinct: a common fresh scope (2017-fitted wire candidates), and, for the historical interfaces only, their saved-observer / catch-up scopes. Historical catch-up exposure is never merged into the common fresh scope, and no equal-total-history claim is made.

## 8. Attribution and boundaries

`nonlinear_moment_refinement` is an engineering adaptation, not a reproduction of any prior method and not a new theorem. Honest ingredient attribution:

* The **squared-Frobenius norm of a residualised cross-covariance between random features and a partialled-out variable** is the RCoT statistic of Strobl, Zhang & Visweswaran (2019), used here as a **penalty**; none of its null distribution or Type-I control is inherited.
* Multiplying a **residualised** sensitive side by functions of `(Z, H)` is a finite, fitted subset of the Daudin (1980) partial-association characterisation used by Zhang et al. (KCI, 2011), Lemma 2(v).
* **Random Fourier features**: Rahimi & Recht (2007); their use inside a spectral ARL solver: Sadeghi, Dehdashtian & Boddeti (K-TOpt, 2022) §5.1.
* **Cross-fitted / out-of-fold nuisance models**: standard cross-fitting practice; no DML orthogonality or rate result is inherited.
* **Utility minus lambda times dependence, solved by a matrix trace step**: Sadeghi, Yu & Boddeti (SARL, 2019) and K-TOpt; the trace step itself is Ky Fan (1949). Not a contribution here.
* A **conditional** dependence penalty inside this objective family, evaluated on ACS/Folktables: Dehdashtian, Sadeghi & Boddeti (U-FaTE, CVPR 2024) is the **direct antecedent**. U-FaTE conditions on the **discrete ground-truth label `Y` by stratification** and is explicitly scoped to non-continuous `Y`; it names *sufficiency* — the fairness family that conditions on the released score — as one of three classes and does not address it. Replacing that stratification with basis interactions in a continuous **released** variable costs U-FaTE's closed-form global optimum, which is exactly the trade this study makes.
* A **nonlinear player acting on the released representation** is prior art: OptNet-ARL (Sadeghi, Wang & Boddeti, 2021) uses kernel-ridge best responses of `z`, unconditionally and with iterative training.

Explicit non-claims: this is not a closed-form, spectral or globally optimal solver; not a new conditional-independence criterion; not an enforcement of `Z ⊥ S | H`; not a bound on `I(S;Z|H)`; not a calibrated test; not the first conditional fair-representation method; not the first to penalise nonlinear functions of a representation; and not evidence of robustness to nonlinear adversaries. A kernel on `S` alone would not address nonlinear functions of `Z`, and nonlinear features of the **input** followed by first-moment protection of `Z` would not either — which is why the feature maps here are applied to the actual released `Z`, conditional on the actual released `H` available to that recipient.

Any bounded-search novelty statement is phrased as absence of evidence after a bounded search, never as a first.

---

## POSTSCRIPT — a specification error found after fitting, recorded not repaired

**Timing, stated explicitly.** Everything above was written and hashed into
`PROTOCOL_FREEZE.json` before the fit phase began. The diagnosis below was made
**after** all 27 maps were fitted and **before** any 2018 score was read. It
changes nothing in §1-§7: the penalty that was registered is the penalty that was
fitted, and the fitted maps are the ones that are audited and reported. This
postscript is a finding *about* the registered specification, not an edit to it.

### The defect

Disclosure from the released channel is invariant under `Z -> Z Q` for orthogonal
`Q`, because `Z` and `Z Q` determine each other and any attacker able to use one
is able to use the other (verified in `tests/.../test_released_channel_information_is_rotation_invariant`).
The utility term `tr(W' U W)` and the original linear penalty `tr(W' P W)` are
invariant too. **The nonlinear penalty of §3-§4 is not.** So the optimiser can
lower it by re-basing the released channel at zero utility cost and with zero
change in what is recoverable — and `diagnostics.py` measures how much of the
training-objective gain is reachable exactly that way.

A penalty intended to measure disclosure from `Z` should be a function of the
information `Z` carries, not of the particular basis chosen for it. By that
standard the registered feature family is misspecified.

### Where the slack comes from, exactly

1. **The quadratic block (a genuine specification error, exactly fixable).** With
   the symmetric moment matrix `M_ab = E[z_a z_b b_k(H) e_c]`, rotation acts as
   `M -> Q' M Q`, which preserves `||M||_F`. But
   `||M||_F^2 = sum_a M_aa^2 + 2 sum_{a<b} M_ab^2`, whereas §3 sums the monomials
   `a <= b` with **equal** weight and so computes
   `sum_a M_aa^2 + sum_{a<b} M_ab^2`. The missing factor of two on the
   off-diagonals is the whole defect. Weighting each off-diagonal monomial by
   `sqrt(2)` makes the block exactly `||M||_F^2` and therefore **exactly**
   rotation invariant. Measured on a synthetic fixture: the shipped convention
   moves by ~10% under a random rotation; the `sqrt(2)`-weighted version moves by
   `< 1e-12`.
2. **Per-feature standardisation (a smaller, structural breakage).** Freezing a
   separate mean and scale per feature gives the features unequal weights, which
   breaks the Frobenius structure a second time. Measured slack ~1%.
3. **The Fourier block (Monte-Carlo error, not a specification error).** Because
   `omega ~ N(0, I)` is rotationally symmetric in distribution, the feature set is
   distributionally invariant and the block converges to a rotation-invariant
   limit; at 32 features per band it is only approximately invariant. Measured:
   the slack shrinks as features are added.

Items 1 and 2 are properties of the construction and would appear on any dataset.
They are not findings about ACS.

### Consequence for the claims

The measured rotation share bounds how much of the surrogate improvement could
possibly correspond to real disclosure reduction: the remainder provably does not.
This is reported as the study's central mechanism result and is the reason the
research decision does not treat the training-objective improvement as progress.
The fix is specified in `NEXT_CONFIRMATION_SPEC.md`; it was **not** applied here,
because applying it after seeing the diagnostic and then re-running would replace
a registered experiment with an unregistered one.
