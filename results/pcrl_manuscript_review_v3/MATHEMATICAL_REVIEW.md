# MATHEMATICAL_REVIEW — what is proved, over what, and where each proof stops

Independent review of the release graph and the definitions. Every statement is given at
**four levels**, because conflating them is the failure mode this document exists to
prevent:

| Level | Question |
|---|---|
| **P** population definition | what is true of the distribution |
| **E** empirical estimator | what the finite-sample quantity estimates |
| **A** algorithmic approximation | what the implemented optimiser/kernel/subset actually computes |
| **N** numerical test | what was measured, at what tolerance |

Only statements M1–M4 are proved. Everything else in the manuscript is a measurement.
No synthetic benchmark matrix was launched for this review; the two fixtures mentioned
below are the ones already committed by the studies.

---

## Setup

Per person: covariates `X`, sensitive attributes `S`, task labels. Two functions are
already published and fixed: `H_A(X) ∈ [0,1]^4`, `H_B(X) ∈ [0,1]^2`. A mechanism produces
`Z = f(X) ∈ R^r`. The releases are

```
release to A  :  (H_A(X), Z)
release to B  :  (H_B(X))
coalition AB  :  (H_A(X), Z, H_B(X))
baseline H    :  (H_A(X))  or  (H_A(X), H_B(X))  according to the audit role
```

`Z` is **concatenated**, never composed with `H`. Write `L*(V)` for the population
infimum of expected log loss for predicting `S` from a view `V` over **all** measurable
predictors, and `L̂(V)` for the loss of the validation-selected member of the finite
attack collection actually fitted.

---

## M1. Appending `Z` without modifying `H` preserves `H` exactly

**P.** For every `x`, the first four coordinates of `(H_A(x), Z(x))` are `H_A(x)`. The
projection `π_{1:4}(H_A, Z) = H_A` is the identity on the published block.

**Proof.** Concatenation. `π_{1:4}` is a coordinate selection and does not depend on `Z`.
∎

**This is not a theorem about privacy, accuracy or stability, and it is not deep.** It is
recorded because it is the *structural* property that distinguishes this release setting
from representation learning, where preservation would be a bounded perturbation with a
norm to control.

**E/A.** No estimator is involved: `H_A` is a stored array copied into the release.

**N.** Verified as `np.array_equal` on every released view: 210 of 210 in Study 1, and
bitwise `H_A`/`H_B` parity on all seven pools for every arm in Study 3, **twice** — at
release time and again at evaluation time against the frozen historical anchors.

**Boundary.** Exactness of the released bits does **not** imply the service stays
accurate: measured, employment log loss rises by 0.0138 nats and income falls by 0.0145
between the two survey years on *identical* released vectors. An identical delivered
prediction is not a stable one.

---

## M2. An unrestricted recipient cannot be hurt by the appended channel

**P.** `L*(H, Z) ≤ L*(H)`.

**Proof.** Any predictor `g` measurable with respect to `σ(H)` is also measurable with
respect to `σ(H, Z)`, via `g̃(h,z) = g(h)`. So the feasible set of the second infimum
contains that of the first, and an infimum over a larger set is no larger. ∎

**E.** The empirical analogue is **false in general.** `L̂(H,Z) ≤ L̂(H)` does not follow,
because `L̂` is not an infimum: it is the test loss of a model chosen by minimum
*validation* loss from a finite candidate set, and a larger candidate set can yield a
selection that generalises worse.

**N.** This is not hypothetical here. Measured on the study's own data:

* 22 of 429 unweighted role-cells have **negative** additional recovery.
* `J`'s own local-sex increment is negative under person weighting with an adjusted
  interval excluding zero, and its absolute local-sex recovery (0.0092) is **below** `H`'s
  own (0.0101).
* In the reporting scope, `J`'s absolute local-sex recovery on 2018 *falls* from 0.00271
  to 0.00236 when it is given **more** attack candidates.

**Consequence adopted in the manuscript.** Negative increments are reported unclipped. A
negative increment is a property of selection, not protection: `H`'s own attack, routed
onto the augmented wire, remains executable, so nothing an `H`-only attacker had is
removed.

---

## M3. The population risk difference is a conditional mutual information — and the
## experiments do not estimate it

**P.** Under ordinary log loss and unrestricted Bayes prediction,

```
L*(H) − L*(H, Z)  =  I(S ; Z | H).
```

**Proof.** `L*(V) = H(S | V)` when the predictor ranges over all conditional
distributions and the loss is `−log q(S)`, since the infimum is attained at
`q = P(S | V)`. Hence the difference is `H(S|H) − H(S|H,Z) = I(S;Z|H)`. ∎

**E.** The reported quantity is `L̂(H) − L̂(H,Z)` with `L̂` a validation-selected finite
attack. Since `L̂ ≥ L*` with an unknown and view-dependent gap, the difference bounds
`I(S;Z|H)` in **neither** direction. It is not an upper bound (a stronger attacker may
exist), and by M2/E it is not even reliably a lower bound in finite samples.

**A.** The attack collection is logistic regression, two restarted MLPs at two budgets,
two boosted-tree configurations, and a random-feature kernel-ridge family, plus legal
projections of `H`'s own attacks and singleton candidates into the coalition view.

**N.** No information quantity is reported anywhere in the manuscript. Mutual-information
estimation carries its own formal limits independent of this design
(McAllester & Stratos, AISTATS 2020).

**The one place an exact interpolation *is* available.** For the declared randomised
withholding mechanism — `H` always released, an independent **visible** branch
`B ~ Bernoulli(p)` fixed per person deciding whether `Z` is also released — the released
object is `(B, Z·1{B=1})`, so `I(S; B, Z_B | H) = p·I(S; Z | H)` and expected routed loss
is exactly `(1−p)L_H + p·L_{HZ}`. This holds **only** for that routed mechanism, never
for interpolated features or interpolated probability vectors, and the branch must
persist for a person's release: independent re-draws would eventually reveal `Z`.

---

## M4. Exact rotation invariance of the repaired objective

Let `Q` be `r × r` orthogonal and `W' = W Q`, so `Z' = V W Q = Z Q` and each row
transforms as `z_i' = Q^⊤ z_i`.

### M4a. The quadratic moment block

**P/E.** With `M_jkc(W) = (1/n_j) Σ_i z_i z_i^⊤ b_k(H_i) e_j[i,c]`, where `b_k` and `e_j`
depend on `H` and the labels only and **not** on `W`:

```
M_jkc(W Q) = (1/n_j) Σ_i (Q^⊤ z_i)(Q^⊤ z_i)^⊤ b_k e_jc = Q^⊤ M_jkc(W) Q.
```

The Frobenius norm is unitarily invariant:
`‖Q^⊤MQ‖_F² = tr(Q^⊤M^⊤QQ^⊤MQ) = tr(M^⊤M) = ‖M‖_F²`. Hence
`D_quad(WQ) = D_quad(W)` for **every** orthogonal `Q`, with no distributional or
large-sample argument. Sign flips (`Q = diag(±1)`) and coordinate permutations are
special cases. ∎

**The predecessor's defect, located exactly.** `‖M‖_F² = Σ_a M_aa² + 2 Σ_{a<b} M_ab²`.
Summing the monomials `a ≤ b` with **equal** weight computes `Σ_a M_aa² + Σ_{a<b} M_ab²`.
The missing factor of two on the off-diagonals is the specification error; per-feature
standardisation broke the Frobenius structure a second time. This is an arithmetic
identity, and the diagnosis is correct.

**A — and this is the condition that matters.** Invariance holds for the *packed*
implementation **only if** the off-diagonal monomials carry weight `√2`, and **only if**
no coordinate-specific rescaling is applied afterwards. A frozen centering vector would
break it: subtracting a fixed non-isotropic `C` from `z_i z_i^⊤` does not transform as
`Q^⊤(·)Q` unless `C` is itself conjugated, which a frozen `C` is not. The implementation
applies **no** centering and **one scalar** scale per role, which is why the property
survives the implementation. A single scalar multiplies through and preserves invariance.

### M4b. The radial kernel block

**P/E.** `‖Q^⊤z_i − Q^⊤z_l‖² = (z_i−z_l)^⊤QQ^⊤(z_i−z_l) = ‖z_i−z_l‖²`, so every entry of
a kernel that depends on the data only through pairwise Euclidean distances is unchanged,
and `L_j` does not depend on `W` at all. Hence `D_kernel(WQ) = D_kernel(W)` exactly. ∎

**A — three conditions, all material.**

1. The bandwidths `h_m` must be **frozen**, set once from a `W`-independent reference
   projection. A bandwidth recomputed from the optimised `W` would make the objective
   self-referential and the invariance argument would not close. The implementation
   freezes them per `(seed, rank)` and aborts rather than falling back if no strictly
   positive pairwise distance exists.
2. The subset `idx_j` must be frozen and `W`-independent. It is, and it is selected from
   the role's validity mask and an RNG only — no residence label, commute label or
   outcome enters it.
3. The kernel must be **exact**, not a random-feature approximation. A finite Fourier
   bank gives a *distributionally symmetric, Monte-Carlo-convergent* property, which is
   strictly weaker and is what the predecessor had.

**N.** Checked against many random rotations, sign patterns and permutations across ranks
{3,5,8} and seeds {0,1}, with a **control** fixture confirming the predecessor is still
rotation-sensitive; 40 fixtures at tolerances declared before any fit. On real ACS data
the gate reports rotation-only share `≤ 1.4×10⁻¹⁵` over 18 cells and
`|L(WQ) − L(W)| ≤ 5.6×10⁻¹⁷`. Orthogonality residual `max|W^⊤W − I| ≤ 6.7×10⁻¹⁶` against a
declared `10⁻¹³`.

**Finite-precision condition.** These are floating-point identities to round-off, not exact
in IEEE arithmetic: the QR retraction, the accumulation order of the moment sums and the
chunked distance computation all introduce `O(ε‖·‖)` error. The measured `10⁻¹⁵`–`10⁻¹⁷`
residuals are consistent with double-precision round-off on these magnitudes and are
reported as such, not as exact zeros.

### M4c. What M4 does and does not buy

**Corollary (correct).** The utility term `tr(W^⊤UW)` and the linear penalty
`tr(W^⊤P_jW)` are rotation invariant, so with M4a and M4b the **whole** repaired objective
satisfies `L(WQ) = L(W)`: it is a function on the Grassmannian, and the Riemannian gradient
has zero component along rotation directions to numerical tolerance.

**Four things this does not give, stated because each is a natural over-reading.**

1. It does not make the objective convex, closed-form or globally solvable. It is
   nonlinear in `W` and generally nonconvex; Ky Fan does not apply. **A suboptimal solution
   remains a live explanation for Study 3's empirical result**, and the study does not
   exclude it.
2. It does not imply the penalty captures the sensitive information that matters. It
   removes one provably inert direction of slack. Study 3 is precisely a case where an
   exactly correct objective produced *worse* measured outcomes.
3. A low fitted value certifies nothing. The kernel block is an empirical **V-statistic**
   (diagonal retained) on at most 512 rows with a measured `1/m` floor —
   `7.13×10⁻⁴` at `m = 512` under an oracle-nuisance conditional null. **A small nonzero
   value is not evidence of residual conditional dependence.** The floor applies equally to
   every arm, so it does not bias comparisons; it bounds what an absolute value can mean.
4. It is not exact for the ACS population — only for the declared bounded subset.

---

## M5. What the moment condition is, and is not (not a theorem: a scoping statement)

The penalty is a finite fitted subset of the `L²` characterisation of conditional
independence used by kernel CI tests — Lemma 2(v) of Zhang et al. (KCI, 2011), attributed
there to Daudin (1980). Our centring on the `S` side is algebraically equivalent to
residualising the `(Z,H)` side: with `f̃ = f − E[f|H]`,
`E[f(Z,H)(1{S=c} − m_c(H))] = E[f̃ · 1{S=c}]`.

The characterisation quantifies over **every** square-integrable `f`. The implementation
tests `f ∈ {Z_ℓ b_k(H)}` — linear in `Z`, degree-2 in `H` — plus, in the repaired version,
degree-2 monomials in `Z` and an exact radial kernel on 512 rows, with **estimated**
nuisances and finite samples. Therefore:

* Vanishing fitted moments do **not** establish `Z ⊥ S | H`.
* A non-zero moment may reflect nuisance error rather than leakage. One committed fixture
  shows a misspecified `m(H)` manufacturing a penalty **more than ten times** the oracle
  value under **exact** conditional independence.
* No Type-I control, null distribution or calibration is inherited from KCI or RCoT. The
  squared Frobenius norm of a residualised cross-covariance *is* the RCoT statistic, used
  here as a **penalty**; borrowing a statistic does not borrow its test.

Two committed fixtures bound the function class concretely, and they fail differently:

* **XOR.** A channel with exactly zero marginal moment while `S = ZH` is recovered
  perfectly. Detected by an interaction term **in `H`** — enriching the conditioning side
  suffices.
* **Magnitude.** `(Z,S)` uniform on `{(−1,0),(1,0),(−2,1),(2,1)}` with `H` constant has
  exactly zero first moment, yet `1{|Z| > 1.5}` recovers `S` exactly. Because `H` is
  constant, **no** enrichment of the conditioning side can ever see it; only a nonlinear
  function of `Z` can (`E[Z²(S−½)] = 0.75`). The attacks that defeat these arms on ACS are
  MLPs and boosted trees, i.e.\ the second kind.

---

## M6. Erasure guarantees do not survive the augmented view (not a theorem: a scope audit)

LEACE's guarantee, as stated by its authors: no **affine** predictor `b + Wx` under any
non-negative convex loss beats the best constant predictor. Three scope facts follow, and
all three bind here.

1. **It is about optimal loss, not thresholded accuracy**, and it is a population
   statement that holds exactly for the moments it was fitted on.
2. **It covers the transformed channel, not the release.** The release is
   `(H_A, r(Z))`, which still contains `H_A`. Nothing in the theorem constrains what a
   predictor can do with `H_A` and `r(Z)` jointly. Study 3's registered forecast Q6 is
   exactly this and it was confirmed: linear cross-covariance is driven to `4×10⁻¹⁶` while
   LEACE's additional recovery from the augmented `A` wire remains strictly positive on
   every endpoint.
3. **It says nothing about nonlinear adversaries**, and the authors conjecture that
   nondestructive editing against a general nonlinear adversary is intractable. A committed
   fixture in this study makes the boundary concrete: after erasure a linear probe reads
   `R² < 10⁻⁸` while a **quadratic** feature of the same erased channel still recovers.

**A design consequence that must be stated before any number is read.** SPLINCE shares
LEACE's kernel `colsp(Σ_xz)` and differs only in range, so under SPLINCE's own Theorem 2
the two give **identical predictions** after refitting a strictly convex *unregularised*
model with a unique minimiser. This study's attacker slate is regularised logistic
regression, restarted MLPs, boosted trees and ridge kernel features, so a difference *can*
appear — but any observed LEACE-versus-SPLINCE difference is attributable to **attacker
regularisation and nonlinearity**, never to a stronger erasure guarantee.

**One clean algebraic fact recorded by the study and not by its sources.** SPLINCE's joint
infeasibility — a direction `v ∈ colsp(WΣ_xz) ∩ colsp(WΣ_xy)` would need `Pv = 0` and
`Pv = v` simultaneously — is immediate, but it is **not** in the paper, and the study
labels the derivation as its own wherever it uses it. The paper offers no infeasibility
theorem, no fallback and no soft-constraint variant. Measured, `cond(U'V)` sits at 2.2–3.5
against a declared `10⁶` gate, so the near-infeasible regime the paper does not treat was
never entered.

---

## M7. Rank arithmetic (elementary, and it is the one place a support fact becomes structural)

`rank(P*) = rank(Σ_XX) − rank(Σ_XZ)`: erasure deletes exactly `rank(Σ_XZ)` directions. A
centred `k`-ary one-hot has rank `k−1`, so a concatenated joint one-hot over sex (2), race
(9) and coverage (2) generically costs `1 + 8 + 1 = 10`, leaving `16 − 10 = 6`.

Measured: 6, 6, 7 across seeds 0, 1, 2. Seed 2 differs because its race class 3 has **zero
support in that seed's own representation-fitting pool** (counts 1, 1, 0), so the centred
joint one-hot has rank 9 rather than 10.

**Scope.** This is a statement about the *fitting pool*, not the population. The category
is not absent from California. A class without support is unmeasured, not protected. The
structural finding worth keeping is the conditional one: *given* the fitted moments,
protected-class support determines the realised width of the released channel.

---

## M8. Where the utility term's exactness stops

`U = C_{VR}C_{RV}` with `V` whitened to `V^⊤V/n = I`. For `W^⊤W = I`,
`tr(W^⊤UW)` is exactly the **training** reduction in least-squares reconstruction error of
the residualised teacher `R`. Exact, on the training rows, for that teacher.

It is not the utility the paper measures. The reported utility endpoint is held-out
residence log loss under an independently fitted probe, which is a different quantity on
different rows. The design deliberately keeps them apart, and no claim transfers from one
to the other.

Also worth recording: whitening fixes `Z^⊤Z/n = I_r` for any orthonormal `W`, so feature
collapse cannot masquerade as privacy — the utility term is covariance-normalised and no
rescaling of `Z` is available to the optimiser.

---

## Defects found, and their disposition

1. **`VALIDATION.md` of Study 3 contradicts its own `RUN_STATUS.md`** on whether numerical
   faults occurred and units were quarantined. `VALIDATION.md` (commit `4e9dc127`) predates
   the exploratory-2017 stage (`34ff87d8`) in which four faults occurred and four
   `QUARANTINE.json` records were written. **Reported to Terminal 1 through the handoff.**
   No number changes; the manuscript follows the run ledger. → `CORRECTIONS.md` B10.
2. **Registered forecast Q2 was ill-posed**, comparing values across two differently
   normalised objectives — a comparison the study's own protocol calls invalid. The study
   discloses this itself, and the manuscript reports it as a registration failure rather
   than quietly dropping it. No mathematical consequence.
3. **No defect found** in M1, M2, M3, M4a, M4b, M7 or M8 as implemented. The invariance
   algebra is correct, the packed `√2` convention is the right one, the no-centering
   decision is justified rather than merely convenient, and the frozen-bandwidth rule is
   what makes the invariance argument close rather than circular.
