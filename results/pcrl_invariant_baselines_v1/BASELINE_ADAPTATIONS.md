# BASELINE_ADAPTATIONS — fair external method adaptations

Written **after** primary-source review and **before** the corresponding fits. Every
departure from a published algorithm is named as a departure. **A homegrown
substitute is never labelled as the published algorithm.**

The goal is a small **measured comparison** with published methods. It is **not** an
exhaustive state-of-the-art claim and **not** a comprehensive benchmark of every cited
method.

## 0. Sources actually read

| Method | arXiv | Verified title / venue | Code |
|---|---|---|---|
| SARL (Spectral-ARL) | 1910.07423 | *On the Global Optima of Kernelized Adversarial Representation Learning*, Sadeghi, Yu, Boddeti, ICCV 2019 | `human-analysis/kernel-adversarial-representation-learning` |
| OptNet-ARL | 2109.05535 | *Adversarial Representation Learning With Closed-Form Solvers*, Sadeghi, Wang, Boddeti. **Venue not stated on the arXiv page**; repo README says ECML 2021 | `human-analysis/closed-form-adversarial-representation-learning` |
| LEACE | 2306.03819 | *LEACE: Perfect linear concept erasure in closed form*, Belrose et al., NeurIPS 2023 | `EleutherAI/concept-erasure` (installed v0.2.4) |
| SPLINCE | 2506.10703 | *Preserving Task-Relevant Information Under Linear Concept Removal*, Holstege, Ravfogel, Wouters, NeurIPS 2025 | **No official code found from any primary source** |

Both papers' bodies were read from extracted PDF text, not from abstract pages. One
recorded hazard: a first automated pass returned a **hallucinated** LEACE formula; it
was discarded and every equation below is from the PDF body.

## 1. Fairness conditions binding on all four arms

Declared in `PROTOCOL.md` §6.2 and repeated here because they are what makes the
comparison fair:

* Every compared release **preserves `H` exactly**. Erasure is applied to the
  *auxiliary channel* and the transformed channel is appended to **unchanged** `H_A`;
  `H_B` is unchanged.
* **No residence or commute label enters any representation or baseline fit.**
* A baseline's guarantee concerns **only the channel it transformed**. No guarantee is
  claimed for the augmented release, which still contains `H`.
* Same declared width, same attacker families, same validation pool, same selection
  rule; absolute and incremental recovery reported separately.
* Shared policy loss coefficients do **not** imply identical effective regularisation
  across methods. Disclosed, not resolved.

### 1.1 Two errors in the predecessor specification, not reproduced

`results/pcrl_evidence_review_v1/CONTRIBUTION_ASSESSMENT.md` §6 specifies, for
LEACE/SPLINCE, "Apply to the concatenated `[H_A, Z]` wire" and, for SPLINCE,
"additionally preserves covariance with the residence label". Both are **rejected
here**, with reasons:

1. **Editing the combined `[H_A, Z]` wire can alter `H`.** LEACE's `P*` is a single
   `d x d` oblique projection over all `d` coordinates; nothing constrains it to act
   as the identity on the first four. Applying it to the combined wire would generally
   change the published service output, breaking the structural guarantee that is the
   study's first deliverable. The channel is therefore transformed **alone** and
   appended.
2. **`same_residence` is the held-out task.** Preserving covariance with it during
   representation fitting would leak the held-out task into the representation and
   defeat the test. SPLINCE's preservation target here is the **authorised training
   tasks** `income_binary` and `civilian_at_work`. This is named an **adaptation**.

---

## 2. §6a — Audit of the existing spectral arms as SARL adaptations

**Aim: avoid fits, not add them.** If the existing marginal spectral arms already are
SARL-style residual-teacher adaptations under their exact parametrisation, they are
**credited and reused**, not duplicated to attach a published name.

### 2.1 The derivation

SARL's empirical construction (Appendix B, eq. 24) minimises `Tr[G' B G]` over
`G'G = I`, with

```
B = L_x' ( lambda * S~' S~  -  (1 - lambda) * Y~' Y~ ) L_x
```

where `L_x` is an orthonormal basis for `range(X~')`. The verified facts that make
this comparable:

* **`L_x` IS the whitening.** With `X~ = V_x Sigma U_x'`, the Appendix-B `L_x` equals
  `U_x`, and `C_x^{-1/2} X~ = sqrt(n) U_x'`. There is no separate whitening step, and
  adding one would double-whiten. This repository's `V` is already the whitened
  residual feature matrix, so `V` plays the role of `sqrt(n) U_x'`.
* Writing `U := C_{y,x_w}' C_{y,x_w}` and `P := C_{s,x_w}' C_{s,x_w}` for the whitened
  cross-covariance Grams, `B = lambda*P - (1-lambda)*U`, so minimising `Tr[G'BG]` is
  **maximising** `Tr[G'((1-lambda)U - lambda P)G]`. Dividing by `(1-lambda) > 0`:

```
SARL solution = top eigenvectors of  U - lambda_bar * P,   lambda_bar = lambda/(1-lambda)
```

which is exactly this repository's spectral step.

* **Trace normalisation re-parametrises `lambda_bar` but changes nothing else.** A
  positive global scalar changes no eigenvector and flips no eigenvalue sign, so the
  solution path and the rank rule are identical; only the `lambda -> solution` map
  differs. A fixed `lambda` is therefore **not** comparable between the two
  parametrisations, and that is disclosed rather than glossed.

### 2.2 The three precise divergence points to be checked numerically

| # | Divergence | What is verified on the saved training matrices |
|---|---|---|
| 1 | **Ordering.** SARL sorts `B` **ascending** and takes the most negative; the repository sorts `U - lambda_bar P` **descending** and takes the most positive. Same eigenvectors, reversed index order -- truncating by index would select **opposite** subspaces. | that the selected subspaces coincide, not merely the eigenvalue multisets |
| 2 | **Rank rule.** SARL Theorem 3 takes `min{r, #{beta < 0}}` -- a strict **sign test**, justified as a minimal-rank tie-break (including zero eigenvalues "does not change the minimum value"). The repository uses a fixed `r`. These differ whenever `#{positive} < r`. | the realised count of positive eigenvalues at each policy, against `r = 16` and `r = 8` |
| 3 | **Repeated eigenspaces.** The objective depends only on `span(G)`; a tie across the selection boundary makes the selected subspace ill-posed. | the eigenvalue gap at the truncation boundary, with the subspace compared by projector Frobenius distance rather than by basis |

**Conditional vs marginal, stated plainly.** SARL's `P` is a **marginal** cross-covariance
Gram of the sensitive attribute. This repository's local/coalition arms use the
**residualised** moment `e_j = onehot(S_j) - m_j(H_c)`, which is a *conditional*
construction and is U-FaTE-adjacent, **not** SARL. Only the **marginal** arms are
candidate SARL aliases. The audit is therefore scoped to those arms and says so.

**Known repository-vs-paper hazard, recorded:** the official SARL code takes the `r`
algebraically smallest eigenvectors **with no sign test**, which Theorem 3 excludes.
Any claim of equivalence is to the **paper**, not to that code path.

**K-TOpt and U-FaTE are related formulations, not automatically distinct implemented
baselines**, and are not fitted here.

---

## 3. §6b — LEACE auxiliary-channel baseline

### 3.1 The published method, verified

With `X` the channel, `Z` the one-hot protected labels, `W = (Sigma_XX)^{+1/2}` the
psd square root of the Moore-Penrose pseudo-inverse, and
`P_{W Sigma_XZ} = (W Sigma_XZ)(W Sigma_XZ)^+` the orthogonal projector onto
`colsp(W Sigma_XZ)`:

```
r_LEACE(x) = x  -  W^+ P_{W Sigma_XZ} W (x - E[X])          (Thm 4.2 / 4.3, Eq. 1)
```

De-mean, whiten, orthogonally project, unwhiten, subtract. **The mean is preserved
exactly.** `P*` is **oblique**, not orthogonal, and is idempotent. It minimises
`E||PX - X||_M^2` for **every** psd inner product `M` simultaneously.

* **Rank:** `rank(P*) = rank(Sigma_XX) - rank(Sigma_XZ)`; erasure deletes exactly
  `rank(Sigma_XZ)` directions and nothing more. A centered `k`-ary one-hot has rank
  `k-1`, so a concatenated joint one-hot over SEX (2), RAC1P (9) and public_coverage
  (2) generically costs `1 + 8 + 1 = 10` directions. Applied to the 16-coordinate
  channel this leaves rank **6**, which is recorded as the realised width.
* **Guarantee scope, exactly:** no affine predictor `b + Wx` under **any** nonnegative
  convex loss beats the best constant predictor. It is a statement about **optimal
  loss**, not about thresholded accuracy, and it is a **population** statement that
  holds exactly for the moments it was fitted on.
* **The paper's own statements on nonlinear adversaries** are quoted rather than
  paraphrased in the report: it *conjectures* nondestructive editing against a general
  nonlinear adversary is intractable, notes erasers fitted with one kernel do not
  generalise to others, and warns that multiclass softmax outputs can leak the removed
  information when another classifier is stacked on top.

### 3.2 Implementation — reuse, not reimplementation

This repository already contains a **verified** wrapper with bit-level regression
tests: `experiments/acs_protection_maps.py:fit_joint_leace`, tested in
`tests/test_acs_protection_maps.py`. It is reused unchanged. It matters because it
overrides three library defaults that would otherwise silently change the estimator
away from the Thm 4.2 optimum:

| Default | Library value | Why it is overridden |
|---|---|---|
| `constrain_cov_trace` | `True` | applies the Appendix-H convex mix with the orthogonal SAL projector, not the paper optimum |
| `shrinkage` | `True` | Ledoit-Wolf-style shrinkage of `Sigma_XX`, not the sample covariance |
| `svd_tol` | `0.01`, **absolute** | an absolute singular-value threshold on `W Sigma_XZ` can silently drop real concept directions at small feature scale |

The wrapper additionally eigendecomposes `Sigma_XX`, keeps eigenvalues
`> 1e-10 * lambda_max`, fits in the retained coordinates and lifts back -- rank
stabilisation the library does not provide.

**Protected target:** concatenated joint one-hot over SEX, RAC1P and public_coverage
from the representation-fitting pool, which is the wrapper's documented convention
("concatenated one-hot columns, not intersection labels").

### 3.3 What is recorded

Rank loss, realised rank, support, the transformation itself, the cross-covariance
residual `max |Cov(r(X), Z)|`, and mean preservation. **No claim of full-wire erasure
and no threshold-accuracy guarantee.** Combined access to `H_A` and the transformed
channel may recover more than either alone; that is measured, not assumed away.

**If this condition exactly matches the existing frozen `E` control, the alias is
proved and reused** rather than refitted.

---

## 4. §6c — SPLINCE auxiliary-channel baseline

### 4.1 The published method, verified

Same starting channel. SPLINCE adds a **range** constraint to LEACE's kernel
constraint (Eq. 3):

```
P Sigma_x,z = 0            (linear guardedness for z -- identical to LEACE)
P Sigma_x,y = Sigma_x,y    (EXACT preservation of Cov(x, y))
```

With `W = (Sigma_xx)^{+1/2}`, `U = colsp(W Sigma_xz)^perp`,
`U^- = U ∩ (colsp(W Sigma_xz) + colsp(W Sigma_xy))^perp`,
`V = colsp(W Sigma_xy) + U^-`, and orthonormal bases `U`, `V`:

```
P*_SPLINCE = W^+ V (U' V)^{-1} U' W                          (Thm 1, Eq. 4)
b*          = E[x] - P* E[x]
```

* **Rank is identical to LEACE.** Both are oblique projections with the **same kernel**
  `colsp(Sigma_x,z)`; they differ only in the **range**. SPLINCE costs no extra rank.
* **Feasibility condition:** `colsp(W Sigma_xz) ∩ colsp(W Sigma_xy) = {0}` -- trivial
  intersection after whitening. The paper's prose ("do not perfectly overlap") is
  **looser than its own formal condition** and is accurate only in the binary/binary
  case; the **formal** version is implemented.
* **The paper states no infeasibility theorem.** The joint infeasibility is immediate
  (a shared direction `v` would need `Pv = 0` and `Pv = v` at once) but that derivation
  is **not** in the paper and is labelled as ours wherever it is used. The paper offers
  no fallback, no relaxation and no soft-constraint variant.

### 4.2 The design consequence that must be stated before any number is read

**SPLINCE Thm 2:** for a model `f(x'theta)` with a strictly convex **unregularized**
loss with a unique minimiser, two projections with the **same kernel** but different
ranges give **identical predictions after re-fitting**. So SPLINCE ≡ LEACE ≡ SAL in
that setting, and "the choice of the range does not determine how much linear
information about the target is lost... that is solely determined by the kernel."

The range matters only when the last layer is re-trained **with** regularisation, or
is **frozen**. This study's attacker slate is regularised logistic regression (`C=1`),
restarted MLPs, boosted trees and ridge-regularised kernel features -- none of which
is the unregularized-unique-minimiser case -- so a difference *can* appear. But any
observed LEACE-vs-SPLINCE difference here is attributable to **attacker
regularisation and nonlinearity**, not to a stronger erasure guarantee. Stated now, so
it cannot be read as a method advantage later.

### 4.3 Implementation and the infeasibility rule

`pcrl/baselines/splince.py:fit_splince` implements Thm 1 faithfully (verified line by
line against Eq. 4, including the `W = Sigma^{+1/2}` convention, the `U^-` computation
and the mean convention). It is **untested** in this repository, so this study adds
fixtures for `P Sigma_xz ≈ 0`, `P Sigma_xy ≈ Sigma_xy`, `P^2 = P`,
`rank(P) = d - k_z`, and the infeasible case.

**Its built-in silent fallback to LEACE is disabled for this study.** The protocol
requires that infeasibility be **reported as a scoped infeasible result**, not
silently relaxed. If the dimension test fails or `cond(U'V)` exceeds the declared
gate, the arm is recorded as `SCOPED INFEASIBLE` with its measured principal angles
and condition number, and it is **not** replaced by a LEACE fit wearing the SPLINCE
name.

**Preservation target:** `y = [income_binary, civilian_at_work]` on the
representation-fitting pool. **Not** residence. **Not** commute. Named an adaptation.

### 4.4 What is recorded

Actual rank, `k_z`, `k_y`, `dim U^-`, `cond(U'V)`, feasibility, the **target
covariance property measured separately from empirical utility**, and the principal
angles between the whitened concept and target column spaces.

---

## 5. §6d — OptNet-ARL adaptation

### 5.1 The published method, verified

Three entities, **exactly one optimised player**: a deep encoder trained by
SGD/Adam, with target and adversary as **kernel ridge regressors solved in closed form
at every step**. There is no descent-ascent. Lemma 1:

```
J(Z) = (1/n)||Y~||_F^2  -  (1/n)|| P_M [Y~' ; 0_n] ||_F^2
M    = [ K~ ; sqrt(n*gamma) I_n ]  in R^{2n x n},  full column rank
K~   = D' K D
```

The stacked `sqrt(n gamma) I` block is the standard augmented-least-squares encoding
of Tikhonov regularisation and is what makes `M'M` invertible. Expanded, this is
`Lambda_hat = (K~^2 + n gamma I)^{-1} K~ * label` and
`||P_M[u;0]||^2 = ||K~ Lambda_hat||^2 + n gamma ||Lambda_hat||^2`, which is exactly
what the official code computes.

Encoder objective (the **corrected** form -- the arXiv Eq. 11 as printed swaps the
multipliers and the index bounds relative to its own Eq. 3, and the official code
implements the corrected form):

```
min_{Theta_E} { lambda * sum_k ||P_{M_s} u_s^k||^2  -  (1-lambda) * sum_m ||P_{M_y} u_y^m||^2 }
```

**Gradient:** the Golub-Pereyra (1973) variable-projection derivative in the paper;
the official code obtains the same gradient by **plain autodiff through the inverse**.
Amos & Kolter's OptNet is **not cited anywhere** in the paper -- "OptNet-ARL" is a
name, not a QP layer. This is recorded because it is a natural misreading.

**Theorem 4.1 rank rule:** the optimal embedding dimensionality is the number of
negative eigenvalues of `B = lambda S~'S~ - (1-lambda) Y~'Y~`, under two assumptions
that do **not** hold during training (`z` free and disconnected from the encoder;
**linear** predictors). The paper therefore frames it as an **upper bound**.
Structurally `#{negative} <= rank(Y~)`, so the rank is capped by the **target** label
rank, not the sensitive one.

**Scalability:** the full projector needs an `O(n^3)` pseudo-inverse, so the paper
approximates the projector and its gradient on a **single mini-batch**, `O(b^3)` per
step -- a deliberately biased estimator justified by analogy with Nystrom.

### 5.2 Every departure, named

| # | Departure | Why | Status |
|---|---|---|---|
| 1 | **Utility target is the residualised teacher `R`**, not a class label | the whole study measures utility as reconstruction beyond what `H` explains | **Structurally free.** `Y` enters only as `Y~` inside `Y~'Y~`; Lemma 1 is stated for an arbitrary random vector with no discreteness assumption. `R` has 32 columns so the `rank(Y~)` cap is not binding at `r = 16`. |
| 2 | **Multiple `lambda` over multiple attributes** (the `L1`/`L2`/`C1` policies) | the study's policies weight several protected roles | **Extrapolation, declared.** The paper asserts multi-lambda generalisation in one sentence with **no equation, no normalisation rule and no code**. The form used is one independent `J` term per role with its policy weight, matching the study's existing aggregation -- *that specific form is not in the paper.* |
| 3 | **Adversary players receive `[H_c, Z]`**, not `Z` alone | the recipient actually holds the service probabilities; an adversary denied them would be a strawman | Adaptation, declared. Original is single-view. |
| 4 | **Mini-batch exact Gaussian kernel, batch 512** | `K~` over 10,513 rows is 884 MB and the machine is at swap capacity | **This is the paper's own practice**, not an invention: it approximates the projector on one mini-batch. Declared as approximate; it is **not** claimed exact for the infinite-dimensional kernel. |
| 5 | Encoder is a fixed small MLP on the frozen `V` | "deep encoder with closed-form ridge players" is the method | Architecture frozen before fitting; no expansion authorised. |
| 6 | Rank fixed at 16, not chosen by Theorem 4.1 | comparisons must be at equal realised rank | Declared. Theorem 4.1's count is **reported** alongside, as the paper's upper bound, but is not used to select. |

### 5.3 Frozen before fitting

Kernel: Gaussian, bandwidth by the median heuristic on the reference projection,
frozen. Ridge `gamma = 1e-4` (the paper's value). Adam, lr `3e-4`, weight decay
`2e-4` (the paper's values). Batch 512. Encoder `128 -> 64 -> 16`, ReLU. Budget: the
same **200 full-objective updates, two deterministic starts** as every other arm, or a
justified prospectively documented comparable budget after a **training-only** runtime
calibration. Selection uses training/allowed-validation objectives only -- **never**
residence or commute performance.

Closed-form player fits and the differentiation through them are verified on a **tiny
reference problem** against an explicit ridge solve before any ACS fit.

### 5.4 Two assertions carried from the source review

* `Cov(z) ≈ I_r` is the cheapest correctness test on any SARL-family solution and is
  asserted.
* `#{negative eigenvalues} > 0` is asserted before an encoder is constructed:
  residualisation can annihilate the utility signal entirely, and the eigendecomposition
  would not error. `||R~ L_x||^2 / ||R~||^2` is reported before any run is trusted.

---

## 6. Fit ledger for this document

| Arm | Nominal | Reduced if |
|---|---:|---|
| §6a spectral/SARL alias audit | **0 fits** | this is the point: credit and reuse, never duplicate |
| §6b LEACE x 3 seeds | 3 | alias with the frozen `E` control is proved |
| §6c SPLINCE x 3 seeds | 3 | infeasibility is documented |
| §6d OptNet-ARL x 3 policies x 3 seeds | 9 | faithful implementation proves impossible with accessible materials |

**No extra baseline tuning sweep is authorised.**

---

# RESULTS APPENDIX — what the adaptations measured

Written after the fits, before any 2018 or 2017 score was read for these arms.

## A. §6a — the SARL alias audit: **equivalent up to one measured mismatch, so reused**

Verdict: the marginal spectral arms `spectral_M025` and `spectral_M1` **are** SARL-style
residual-teacher adaptations. They are **credited and reused as explicit adaptations**.
**Zero duplicate models were fitted.** (`ALIAS_AUDIT.json`)

The derivation is confirmed constructively: rebuilding the arm from
`top-r eigenvectors of (U − coefficient · P_marginal)`, with `U` the trace-normalised
Gram of the residualised teacher and `P_marginal` built from an intercept-only basis and
the **class prior** as the nuisance, reproduces the stored historical map **bitwise**
(max abs difference `0.0`) for both arms in all three seeds.

The three divergence points, measured rather than asserted:

| # | Divergence | Measured |
|---|---|---|
| 1 | **Ordering** | Ascending truncation (SARL's sort order) selects a subspace at projector distance **5.657** from the correct one — the **maximum possible** value `sqrt(2·16)`, i.e. a completely orthogonal subspace. This is the single most damaging bug a reimplementation can ship, and it is what the official SARL code's `sort` + `indices[0:r]` would produce without the sign convention reversed. |
| 2 | **Rank rule** | `U − λ̄P` has **32 positive eigenvalues** in every seed and both arms, so SARL's sign rule would select `min(16, 32) = 16`. **The sign rule and the fixed `r = 16` agree**; the rule is not binding here. |
| 3 | **Repeated eigenspaces** | The gap at the truncation boundary is `4.8e-4` to `1.7e-3` — small but strictly positive, so the selected subspace is well posed. No tie straddles the cut. |

**The one real mismatch.** The repository trace-normalises each attribute's moment Gram
**separately** and then averages; SARL would take one raw sum. Because the per-attribute
raw traces differ by a factor of ~1.9 (`SEX ≈ 0.047`, `RAC1P ≈ 0.045`,
`public_coverage ≈ 0.024`), this is a **per-block reweighting, not a global scalar**, and
so is **not** absorbable into `λ̄`. Measured effect: the selected subspace moves by a
projector Frobenius distance of **0.10–0.20** (out of a maximum 5.657).

So: equivalent in construction, family and solution method; **not** identical in the
sensitive-attribute weighting. Credited, reused, and the mismatch stated precisely.

**Scope.** Only the **marginal** arms are SARL aliases. The local and coalition arms use
the residualised moment `e_j = onehot(S_j) − m_j(H_c)`, which is a **conditional**
construction (U-FaTE-adjacent), not SARL.

## B. §6b — LEACE: exact linear erasure, at the rank the theory predicts

All three seeds **FITTED**. Applied to the 16-coordinate `A0` auxiliary channel **alone**
and appended to **unchanged** `H_A`.

| Seed | Realised rank | `k_z` | Linear cross-covariance before → after | Idempotency error | Mean-preservation error |
|---|---|---|---|---|---|
| 0 | 6 | 10 | `2.48e-1` → `4.56e-16` | `4.9e-15` | `7.7e-15` |
| 1 | 6 | 10 | `2.72e-1` → `3.86e-16` | `6.7e-15` | `4.9e-15` |
| 2 | **7** | **9** | `2.76e-1` → `7.21e-16` | `3.0e-15` | `6.0e-15` |

The rank arithmetic is exactly the theory: a concatenated joint one-hot over SEX (2),
RAC1P (9) and public_coverage (2) has centred rank `1 + 8 + 1 = 10`, leaving `16 − 10 = 6`.
**Seed 2 differs for a real and recorded reason**: its `RAC1P` class index 3 has **zero**
population support, so the centred joint one-hot has rank 9, not 10, and the channel keeps
7 dimensions. Support is recorded, never repaired — and here the support directly
determines the realised width of a released channel.

**Guarantee scope, unchanged from the paper:** no **affine** predictor under any
nonnegative convex loss beats the best constant predictor, **on the transformed channel**
and for the moments it was fitted on. A fixture in this study confirms the limit
concretely: after erasure a linear probe recovers `R² < 1e-8` while a **quadratic** feature
of the same erased channel still recovers. Nothing is claimed for the augmented release,
which still contains `H_A`.

## C. §6c — SPLINCE: feasible, same rank as LEACE, covariance preserved exactly

All three seeds **FITTED and FEASIBLE** — the `SCOPED INFEASIBLE` branch was not exercised.
The silent LEACE fallback was disabled and did not fire (`fallback_to_leace = False`).

| Seed | Rank | `k_z` | `k_y` | `dim U⁻` | `cond(U'V)` | Target-covariance preservation error | `‖Σ_xy‖_F` |
|---|---|---|---|---|---|---|---|
| 0 | 6 | 10 | 2 | 4 | 3.45 | `1.24e-14` | 2.18 |
| 1 | 6 | 10 | 2 | 4 | 2.19 | `6.01e-14` | 2.41 |
| 2 | 7 | 9 | 2 | 5 | 2.96 | `5.55e-15` | 2.02 |

`cond(U'V)` sits between 2.2 and 3.5, five to six orders of magnitude below the declared
`1e6` gate, so this is nowhere near the near-infeasible regime the paper does not treat.
SPLINCE costs **no extra rank** over LEACE, exactly as the paper claims (same kernel
`colsp(Σ_xz)`, different range), and it preserves the authorised training-task
cross-covariance to `~1e-14` against a signal of norm ~2.

**Preservation target:** `income_binary` and `civilian_at_work`. **Not** residence.
**Not** commute. Named an adaptation.

**Two limitations restated before any recovery number is read.**

1. **Thm 2 caveat.** SPLINCE and LEACE share a kernel, so after re-fitting a strictly
   convex *unregularized* model with a unique minimiser they give **identical**
   predictions. This study's attacker slate is regularised logistic regression, restarted
   MLPs, boosted trees and ridge kernel features, so a difference *can* appear — but any
   difference is attributable to **attacker regularisation and nonlinearity**, not to a
   stronger erasure guarantee.
2. **Label-access asymmetry.** No other arm uses task labels during representation
   fitting; the spectral and repaired arms optimise against the residualised teacher `R`.
   SPLINCE sees two authorised task labels its comparators do not. This is an asymmetry
   **in SPLINCE's favour on utility**, and it is disclosed rather than resolved.

## D. §6d — OptNet-ARL: budget set from a training-only probe

Nine conditions, rank 16, three policies, three seeds. Every departure from the published
single-view supervised formulation is enumerated in §5.2 and carried in the machine-readable
`DEPARTURES` list inside `OPTNET_SUMMARY.json`. The two that most affect interpretation:
the **multi-lambda multi-attribute form is an extrapolation** (the paper asserts it in one
sentence with no equation, no normalisation rule and no code), and the **mini-batch
projector is the paper's own practice** but is approximate and is never called exact.

Budget: **1200 Adam steps per start, two starts**, set from a training-only convergence
probe rather than copied from the other arms — 200 mini-batch steps is not comparable to
200 full-objective Riemannian updates. See `RUN_STATUS.md` amendment 2 and
`OPTNET_CALIBRATION.json`.

## E. Fit ledger actually realised

| Arm | Nominal | Realised | Why |
|---|---:|---:|---|
| §6a spectral/SARL alias audit | 0 | **0** | equivalence established up to a measured mismatch; credited and reused, never duplicated |
| §6b LEACE x 3 seeds | 3 | **3** | fitted |
| §6c SPLINCE x 3 seeds | 3 | **3** | fitted; feasible, so no scoped-infeasible reduction |
| §6d OptNet-ARL x 9 | 9 | **9** | fitted |
| **External total** | **15** | **15** | |

Registered forecast **Q7** ("at least one external adaptation proves to be an alias or
infeasible") is **met by the alias route, not the infeasibility route**: the SARL audit
avoided six fits, while SPLINCE turned out comfortably feasible.
