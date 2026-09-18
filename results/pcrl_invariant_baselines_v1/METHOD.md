# METHOD — `rotation_invariant_moment_refinement`

Written **before** any new interface was fitted and before any new ACS performance
outcome was opened. Every scale, seed, subset and constant below is prospective and
is hashed into `PROTOCOL_FREEZE.json`.

Implementation: `experiments/pcrl_invariant_baselines_v1/{invariant_moment,objective,maps}.py`.

Predecessor: `results/pcrl_nonlinear_rank_v1/METHOD.md` at commit
`c37807e4f568ef38e5528fc09c1506083278bf4d`. This document changes **only** the
nonlinear penalty block. §1, §5-§7 of the predecessor are reused verbatim and are
restated here only where needed to make the new block unambiguous.

---

## 0. What problem this repairs, and what it does not

The predecessor's nonlinear penalty was **coordinate-dependent**: equal monomial
weights, per-feature standardisation, and a finite Fourier feature bank together
allowed the objective to change under rotations `W -> W Q` that leave the released
information unchanged. The defect was diagnosed after fitting and was **not**
repaired in the evaluated maps (predecessor `METHOD.md` postscript).

This document replaces that block with two components that are **exactly** invariant
under every orthogonal `Q`, by construction and not by Monte-Carlo convergence.

Four things this does **not** do, stated up front:

* It does not make the objective convex, closed-form, or globally solvable. It is
  nonlinear in `W` and generally nonconvex. Ky Fan does not apply.
* Rotation invariance does **not** imply the penalty captures all sensitive
  information, that the optimiser reaches a useful point, or that a low value
  certifies anything. It removes one provably inert direction of slack; that is all.
* The kernel component is **exact for a declared bounded subset**, not for the ACS
  population. It is a finite empirical V-statistic, not a calibrated
  conditional-independence test and not a privacy certificate.
* This is a repaired construction with **several** specified changes at once
  (Frobenius weighting, single scalar block scale, no centering, exact kernel in
  place of finite Fourier features, per-role frozen subsets). Any measured ACS
  effect is attributable to the **package**, never to `sqrt(2)` weighting alone.

---

## 1. Reused unchanged from the predecessor

From the frozen 2018 `SpectralModel` of each seed
(`results/redesign_20260910_acs_residual_spectral_v1/seed_N/maps.joblib`):

| Symbol | Meaning | Shape (seed 0) |
|---|---|---|
| `V` | whitened residual features on representation-fit rows; training mean 0, covariance `I_q` | `(10513, 128)` |
| `q` | whitened rank | `128` |
| `R` | teacher residual `T - Q_A Q_A^+ T`, recentred, from original unstandardised `T` | `(10513, 32)` |
| `U` | trace-normalised utility matrix, `U_raw = (V'R/n)(R'V/n)` | `(128, 128)` |
| `Q_A` | training-only degree-2 polynomial anchor basis in `hA[:,[1,3]]`, intercept retained | `(n, 6)` |
| `Q_AB` | same in `[hA[:,[1,3]], hB[:,1]]` | `(n, 10)` |
| `m_j(H_c)` | the original three-fold **household** OOF nuisance predictions, reconstructed exactly | — |
| `P_j` | trace-normalised historical linear moment Gram of attribute `j` | `(128, 128)` |

Protected residual, unchanged, with **all `K` one-hot columns and no reference-class
deletion**:

```
e_j = onehot(S_j) - m_j(H_c)
```

Valid rows are that attribute's own mask `y_j >= 0`; the divisor is `n_j`. Nuisances
are frozen and shared by the old, new and baseline arms. `A` inference uses `X` and
`H_A` only; coalition fitting may use `H_B`. `H`'s authoritative wire values and
dtypes are preserved bitwise everywhere.

**Reference projection (policy-independent).** For each seed and rank `r`:

```
W_ref(r) = top-r eigenvectors of U, sign-canonicalised
Z_ref    = V W_ref(r)
```

Residence and commute labels never enter its construction.

---

## 2. Component A — the quadratic moment block

For protected role `j`, anchor-basis coordinate `k` and class `c`, on that role's
valid representation-fitting rows, define the `(r, r)` **symmetric** moment matrix

```
M_jkc(W) = (1/n_j) * sum_{i valid}  z_i z_i^T * b_k(H_c i) * e_j[i, c]        (1)
```

with `z_i` the `i`-th row of `Z = V W`. The block is

```
D_quad_j(W) = sum_{k, c}  || M_jkc(W) ||_F^2                                  (2)
```

### 2.1 Exact rotation invariance — the algebra

Let `Q` be any `r x r` orthogonal matrix and `W' = W Q`. Then `Z' = V W Q = Z Q`, so
each row transforms as `z_i' = Q^T z_i`, hence

```
z_i' z_i'^T = Q^T z_i z_i^T Q
```

Substituting into (1), and noting `b_k` and `e_j[·,c]` are functions of `H` and the
labels only — **not** of `W` — every term carries the same conjugation, so

```
M_jkc(W Q) = Q^T M_jkc(W) Q                                                   (3)
```

The Frobenius norm is unitarily invariant:
`||Q^T M Q||_F^2 = tr(Q^T M^T Q Q^T M Q) = tr(Q^T M^T M Q) = tr(M^T M) = ||M||_F^2`.
Therefore

```
D_quad_j(W Q) = D_quad_j(W)   for every orthogonal Q                          (4)
```

exactly, with no distributional or large-sample argument. The orthogonal group
includes sign flips (`Q = diag(±1)`) and coordinate permutations (`Q` a permutation
matrix), so both are covered by (4) as special cases. This is checked numerically
against **many** random rotations, sign patterns and permutations across ranks and
seeds — not against one chosen `Q` (`INVARIANCE_VALIDATION.md`).

### 2.2 Why the predecessor failed, restated exactly

Writing `||M||_F^2 = sum_a M_aa^2 + 2 * sum_{a<b} M_ab^2`, the predecessor summed the
monomials `a <= b` with **equal** weight and so computed
`sum_a M_aa^2 + sum_{a<b} M_ab^2`. The missing factor of two on the off-diagonals is
the specification error. Per-feature standardisation broke the Frobenius structure a
second time.

### 2.3 Implementation and the packed convention

Two implementations are maintained and checked against each other (§4 of
`PROTOCOL.md`):

* **Reference:** accumulate the full symmetric `(r, r)` moment per `(k, c)` and take
  `||·||_F^2` directly.
* **Packed:** the `r(r+1)/2` monomials `z_a z_b` (`a <= b`) with weight `sqrt(2)` for
  `a < b` and `1` for `a = b`. The packed squared sum then equals
  `sum_a M_aa^2 + 2 sum_{a<b} M_ab^2 = ||M||_F^2` identically.

The packed form is used in the hot path; the reference form is the fixture. They must
agree to the declared tolerance.

### 2.4 Centering and scale

**No feature centering.** This is the simplest convention that is provably
equivariant, and it is the one adopted. A frozen, coordinate-specific centering
vector is **not** assumed harmless: subtracting a fixed non-isotropic reference
matrix `C` from `z_i z_i^T` does not transform as `Q^T(·)Q` unless `C` itself is
conjugated, which a frozen `C` is not. Since no proof of equivariant centering is
offered, no centering is applied.

**One scalar scale for the whole block**, fixed from the common reference projection:

```
N_j^quad = D_quad_j(W_ref)                                                    (5)
```

There are **no** coordinate-specific standard deviations anywhere in this block. A
scalar multiplies (4) through and so preserves invariance.

---

## 3. Component B — the exact radial-kernel block

An **exact** radial kernel on a fixed bounded subset replaces the finite Fourier
feature bank. More Fourier features are *not* treated as exact invariance: they give
a distributionally symmetric, Monte-Carlo-convergent approximation, which is a
different and weaker property.

### 3.1 The frozen subset

For each role `j`, a deterministic subset of at most **512** valid
representation-training rows:

```
idx_j = sort( default_rng(20260930 + 100*seed + 10*r + role_index).choice(
              valid_rows_j, size=min(512, n_j), replace=False) )
```

`role_index` is the position of the role in the fixed tuple
`('A/SEX', 'A/RAC1P', 'A/public_coverage', 'AB/SEX', 'AB/RAC1P')`.

* The seed and the realised indices are **frozen and recorded** (hashed) and are used
  for **every** policy, every penalty family and both the old-versus-new diagnostic
  comparisons.
* Selection uses only the role's own validity mask and the RNG. **No residence or
  commute label, and no outcome, enters subset selection.**
* Realised per-class support on the subset is **recorded**. It is **not** repaired,
  rebalanced, stratified or padded with placeholders. `RAC1P` has classes with
  population support as low as 0-1 rows (class index 3: 1, 1, 0 rows in seeds 0, 1,
  2), so classes with **zero** positive support on a 512-row subset are expected and
  are reported per role. Such a class still contributes through its residual
  `e_c = -m_c(H)`, which is generally nonzero; the limitation is recorded, not hidden.

### 3.2 The kernel

The average of three Gaussian kernels with the frozen bandwidth multipliers
`band = (0.5, 1.0, 2.0)`:

```
K_Z(i, l) = (1/3) * sum_{m=1..3} exp( -||z_i - z_l||^2 / (2 * h_m^2) ),
h_m = band_m * sigma_ref                                                      (6)
```

`sigma_ref` is set **once per (seed, rank)** from the common reference projection on
representation-fitting rows — the median strictly positive pairwise Euclidean
distance of `Z_ref` over the frozen bandwidth subset
(`min(1024, n)` rows, `default_rng(20260918 + 100*seed + r)`, the predecessor's
recipe reused unchanged) — and is then frozen. If no strictly positive distance
exists, fitting **aborts** with a bandwidth diagnostic; it never silently falls back.
Zero-distance pairs are counted and reported. No bandwidth is ever recomputed from an
optimised `W`, which would make the objective self-referential, and no bandwidth is
searched on a development score.

### 3.3 The block

Let `B_j` be the fixed `(m_j, B)` anchor-basis matrix and `E_j` the fixed `(m_j, K)`
nuisance-residual matrix on the subset rows, `m_j = |idx_j|`. Define the fixed
`(m_j, m_j)` matrix

```
L_j = (B_j B_j^T)  elementwise*  (E_j E_j^T)                                  (7)
```

and the block

```
D_kernel_j(W) = ( 1 / m_j^2 ) * sum_{i, l}  K_Z(i, l) * L_j(i, l)             (8)
```

### 3.4 What (8) is, exactly

Let `phi` be the RKHS feature map of the averaged kernel (6), so
`K_Z(i,l) = <phi(z_i), phi(z_l)>`. Define the empirical kernel-feature residual
moment `N_kc = (1/m) sum_i phi(z_i) b_k(i) e_c(i)`. Then

```
sum_{k,c} ||N_kc||^2
  = (1/m^2) sum_{i,l} <phi(z_i), phi(z_l)> * (sum_k b_k(i) b_k(l)) * (sum_c e_c(i) e_c(l))
  = (1/m^2) sum_{i,l} K_Z(i,l) * (B B^T)_{il} * (E E^T)_{il}
  = D_kernel_j(W)
```

So (8) **is** the squared norm of a finite empirical kernel-feature residual moment —
the same object as the quadratic block, with a different feature map. It is
**not** a calibrated conditional-independence test, not a privacy certificate, and
inherits no Type-I control from KCI or RCoT.

**Estimator convention, declared now.** The finite-sample **diagonal terms `i = l`
are kept**. (8) is therefore a **V-statistic**. The U-statistic (diagonal-deleted)
variant is biased differently and is *not* used. This convention is fixed before any
outcome and is **not** swapped later based on how a number comes out.

### 3.5 Exact rotation invariance

`W -> W Q` sends `z_i -> Q^T z_i`, and

```
||Q^T z_i - Q^T z_l||^2 = (z_i - z_l)^T Q Q^T (z_i - z_l) = ||z_i - z_l||^2
```

so every entry of `K_Z` is **unchanged**. `L_j` does not depend on `W` at all.
Therefore `D_kernel_j(W Q) = D_kernel_j(W)` exactly, for every orthogonal `Q`
including sign flips and permutations. A radial kernel depends on the data only
through pairwise distances, which orthogonal maps preserve exactly.

### 3.6 Scale and cost

One scalar per role, frozen at the reference projection:

```
N_j^kernel = D_kernel_j(W_ref)                                                (9)
```

Cost: `L_j` is built once per role (`512 x 512` doubles = 2.1 MB). Per objective
evaluation the kernel block needs `Z` on `m_j <= 512` rows and one `m_j x m_j`
distance matrix, streamed in row chunks. **No `n x n` matrix over the full
representation pool is ever formed.** The objective is exact **for the declared
subset only**; it is not an exact full-data kernel objective and is never described
as one. The bounded sample and the frozen nuisance models remain limitations.

---

## 4. Combination, calibration and the declared degenerate conventions

Positive floor `FLOOR = 1e-12`, declared before fitting.

**Block combination.** Where both references are nondegenerate the two blocks take
**equal weight**:

```
D_j(W) = 0.5 * D_quad_j(W) / N_j^quad  +  0.5 * D_kernel_j(W) / N_j^kernel    (10)
```

so `D_j(W_ref) = 1` exactly.

**Nondividing degenerate conventions, specified before fitting.** No quantity is ever
divided by the floor:

| Condition | Convention | Flag |
|---|---|---|
| `N_j^quad > FLOOR` and `N_j^kernel > FLOOR` | (10) as written | — |
| exactly one reference `<= FLOOR` | that block is **dropped**; the other takes weight `1.0`, so `D_j(W_ref) = 1` still holds | `<block>_reference_degenerate_dropped` |
| both references `<= FLOOR` | `D_j(W) := 0`; the role contributes its linear term only | `nonlinear_block_fully_degenerate` |

**Calibration to the original role term** at the common reference projection:

```
c_j = lin_j(W_ref)                if lin_j(W_ref) > FLOOR
c_j = 0  (nonlinear term dropped) otherwise                                   (11)
```

with flag `linear_reference_degenerate_nonlinear_term_dropped` in the second case.
Every flag is carried into every table. The predecessor recorded **no** degeneracy
flag at any role, seed or rank, so these branches are expected to be unexercised;
they are specified in advance so that an exercised branch is a recorded fact rather
than an improvised decision.

**Role penalty**, retaining the predecessor's equal-weight combination:

```
lin_j(W)     = tr(W^T P_j W)
penalty_j(W) = 0.5 * lin_j(W)  +  0.5 * c_j * D_j(W)                          (12)
```

At `W = W_ref` this equals `lin_j(W_ref)` exactly. **Matching a reference penalty
scale is not equal privacy strength away from that projection**, and nothing here
claims it is.

**Aggregation and policies**, identical to the historical arms with the same fixed
denominators 3 and 2 — an unsupported or zero attribute does **not** shrink the
denominator and never counts as a privacy success:

```
local_agg(W) = (1/3) * [ penalty_{A/SEX} + penalty_{A/RAC1P} + penalty_{A/public_coverage} ]
ab_agg(W)    = (1/2) * [ penalty_{AB/SEX} + penalty_{AB/RAC1P} ]
```

| Policy | Penalty |
|---|---|
| `L1` | `local_agg(W)` |
| `L2` | `2 * local_agg(W)` |
| `C1` | `local_agg(W) + ab_agg(W)` |

Original denominators, masks, one-hot schema and support flags are preserved.

---

## 5. The optimised objective

Reported as a **training objective to be MINIMISED**:

```
L(W) = penalty(W) - tr(W^T U W),     subject to  W^T W = I_r                  (13)
```

### 5.1 A consequence worth stating: `L` is a function on the Grassmannian

`tr(W^T U W)` is rotation invariant. `tr(W^T P_j W)` is rotation invariant. By (4)
and §3.5 both new blocks are rotation invariant. Therefore **the whole repaired
objective satisfies `L(W Q) = L(W)` for every orthogonal `Q`** — it depends only on
the subspace `span(W)`, exactly as the information content of the release `Z = V W`
does.

Two consequences, both checked:

1. The predecessor's `rotation_share` diagnostic must now return ~0. That is the
   predeclared acceptance gate (`PROTOCOL.md` §5).
2. The Riemannian gradient has **zero** component along rotation directions `W A` for
   skew-symmetric `A`, to numerical tolerance. The search is effectively on the
   Grassmannian while the solver remains the same Stiefel solver.

This is a property of the construction, not a finding about ACS.

### 5.2 Solver — reused unchanged

* Riemannian gradient `G_R = G - W sym(W^T G)`, `G = dL/dW` analytic (§5.3).
* Retraction `W <- qf(W - t G_R)`, QR `Q` factor with positive `R` diagonal.
* Backtracking line search from step `1.0`, shrink `0.5`, at most `20` backtracks;
  a step is accepted only if it improves `L` by more than `1e-14`.
* Stop on budget, on Riemannian gradient norm `< 1e-12`, or on line-search failure.
* Budget **200 full-objective updates per start**.
* **Two deterministic starts** per unique condition: (a) the corresponding
  original-moment closed-form solution; (b) `qf(W0 + 0.05 * G/||G|| * ||W0||)` with
  `G` drawn once by `default_rng(20260920 + 100*seed + r)`.
* The **unmoved initial point of each start is itself an eligible checkpoint.**
* Selection is the **lowest training objective** across both starts and both initial
  points, ties broken by start name. Selection never consults attackers, residence,
  commute, development outcomes or the transport table. Returning the original point
  is a valid recorded no-improvement outcome.
* Bandwidths, subsets and all normalisers are **fixed during optimisation**.
* Line-search failures, gradient norms, feasibility, start variability and runtime
  are recorded for every condition.

The penalty family `original` is unchanged and still has the closed-form top-`r`
eigenvector global optimum of its fixed matrix `U - lambda P`. That is the **only**
place a global-optimality statement is made. The repaired family is nonconvex and its
solution is at best locally optimal.

### 5.3 Analytic gradient

With `Z = V W`:

**Quadratic block.** From (2), `dD_quad_j / dM_jkc = 2 M_jkc`, and from (1) a
perturbation `dz_i` contributes `(1/n_j) w_ikc (dz_i z_i^T + z_i dz_i^T)` with
`w_ikc = b_k(i) e_j[i,c]`. Using symmetry of `M`,

```
dD_quad_j / dz_i = (4 / n_j) * sum_{k,c} w_ikc * M_jkc * z_i                  (14)
```

**Kernel block.** With `A(i,l) = -(1/3) sum_m exp(-d_il^2 / (2 h_m^2)) / h_m^2` and
`G = L_j elementwise* A` (symmetric),

```
dD_kernel_j / dZ = (2 / m_j^2) * ( diag(G 1) - G ) Z                          (15)
```

a graph-Laplacian contraction on the `m_j <= 512` subset rows.

**Assembly.** `dD_j/dW = V^T (dD/dZ)` with the kernel part restricted to the subset
rows; the linear term contributes `2 P_j W` and the utility term `2 U W`, both scaled
by their weights in (12) and (13). Verified against central finite differences and
against autodiff on a reduced fixture.

Feature collapse cannot masquerade as privacy: whitening fixes `Z^T Z / n = I_r` for
any orthonormal `W`, so the utility term is covariance-normalised and no rescaling of
`Z` is available to the optimiser. Orthogonality residuals are recorded for every
accepted map.

---

## 6. Endpoints, attacks and reporting — reused from the transport study

Reused verbatim and labelled as reused. Sources:
`results/redesign_20260917_acs_spectral_transport_v1/{PROTOCOL.md,COMPARISONS.json}`
and `scripts/report_acs_spectral_transport.py`.

* Five utility tasks: `income_binary`, `civilian_at_work`, `same_residence` (A-view);
  `public_coverage`, `commute_over20` (B-view).
* Eleven forbidden roles: `A/{public_coverage, commute_over20, SEX, RAC1P}`,
  `B/{income_binary, civilian_at_work, same_residence, SEX, RAC1P}`, `AB/{SEX, RAC1P}`.
  `RAC1P` scored on all nine classes.
* Four family sensitive endpoints: `recovery/A/SEX`, `recovery/AB/SEX`,
  `recovery/A/RAC1P`, `recovery/AB/RAC1P`, plus `utility/same_residence`.
* Attack families per role: `logistic` (C=1, lbfgs, max_iter 500), two **restarted**
  MLPs `mlp_0`/`mlp_1` (hidden [64,32], seed offsets 0 and 10000, one 360-epoch
  trajectory with nested 120/360 checkpoints), two boosted trees
  `hist_gb_20`/`hist_gb_5` (HistGB, 150 iters, 15 leaves, min_samples_leaf 20 and 5),
  and the kernel family (256 random Fourier features, ridge `alpha in {1e-4,1e-2,1}`,
  median-distance bandwidth on training rows only). Role seed
  `1260000 + 100*seed + 10*view_index + target_index`; kernel seed
  `20263910 + 100*seed + 10*view_index + target_index`.
* Legal `H`-only ancestors routed to the anchor coordinates (`anchor__*`) and
  singleton-to-coalition projections (`inherited_A__*`, `inherited_B__*`) included
  exactly as for the historical arms.
* Selection: minimum **unweighted** attacker-validation log loss, then candidate ID.
  The diagnostic `saved_adversary` is excluded from every scope.
* Budgets 120 and 360; primary budget 360.
* Withholding controls at `p in {0, .25, .5, .75, 1}` for the frozen sources `E`,
  `A0`, `L025`, `L20`, `J` and `spectral_S0`, and for the new candidates under the
  **same** predetermined grid, using the existing explicit branch-routed mechanism
  with a fixed per-person independent branch
  (`sha256('...|year|seed|condition|SERIALNO|SPORDER')`). Expected loss interpolates
  as `(1-p) L_H + p L_aug` for that routed mechanism **only** — never for interpolated
  features or probability vectors, and the branch must persist for a person's release.

**New candidate families: none introduced** by the repaired arm. It uses exactly the
slate the historical spectral arms used, so attack opportunities are matched without
adding anything to the controls. The external baselines of `BASELINE_ADAPTATIONS.md`
receive the same slate.

Attack scopes stay distinct: a common fresh scope, and, for the historical interfaces
only, their saved-observer / catch-up scopes. Historical catch-up exposure is never
merged into the common fresh scope, and no equal-total-history claim is made.

---

## 7. Attribution and boundaries

Honest ingredient attribution, unchanged from the predecessor except where the
repair touches it:

* The **squared Frobenius norm of a residualised cross-covariance between features
  and a partialled-out variable** is the RCoT statistic of Strobl, Zhang &
  Visweswaran (2019), used here as a **penalty**; no null distribution or Type-I
  control is inherited.
* Multiplying a **residualised** sensitive side by functions of `(Z, H)` is a finite,
  fitted subset of the Daudin (1980) partial-association characterisation used by
  Zhang et al. (KCI, 2011), Lemma 2(v).
* **Exact Gaussian kernels in place of random features**: the standard
  kernel-vs-random-feature trade (Rahimi & Recht 2007 is the approximation this
  replaces, not the thing implemented).
* **Cross-fitted / out-of-fold nuisance models**: standard practice; no DML
  orthogonality or rate result is inherited.
* **Utility minus lambda times dependence, solved by a matrix trace step**: SARL
  (Sadeghi, Yu & Boddeti 2019) and K-TOpt; the trace step is Ky Fan (1949). Not a
  contribution here, and the repaired objective is **not** in that family — it is not
  a trace form.
* A **conditional** dependence penalty inside this objective family on ACS/Folktables:
  U-FaTE (Dehdashtian, Sadeghi & Boddeti, CVPR 2024) is the direct antecedent.
* A **nonlinear player acting on the released representation**: OptNet-ARL (Sadeghi,
  Wang & Boddeti, 2021).

Explicit non-claims: not closed-form, spectral or globally optimal; not a new
conditional-independence criterion; not an enforcement of `Z ⊥ S | H`; not a bound on
`I(S;Z|H)`; not a calibrated test; not evidence of robustness to nonlinear
adversaries; and **not** a demonstration that rotation invariance improves anything —
that is the empirical question this study asks, and it may answer it negatively.

Any bounded-search novelty statement is phrased as absence of evidence after a
bounded search, never as a first.
