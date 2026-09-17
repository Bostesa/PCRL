# Independent mathematical review of the proposed nonlinear-penalty / free-rank follow-up

Terminal B, branch `research/pcrl-evidence-paper-v1`, baseline `349efa4`.
Fixtures: `tests/pcrl_evidence_review_v1/test_penalty_fixtures.py` (8 tests, all passing,
exact enumeration rather than simulation).

This note reviews the *design* proposed in `RESEARCH_DECISION.md` — "replace the finite
first-moment penalty with a penalty a nonlinear attacker cannot walk around … and select
rank by the sign of the eigenvalues rather than fixing r = 16" — before any of Terminal A's
outcomes are available. It reviews the two factors separately, because they are not
independent and one of them silently breaks a control in the existing comparison.

## 1. Objects and the legal inference inputs

* `X` covariates; `S = (S_SEX ∈ {0,1}, S_RAC1P ∈ {0,…,8})`; `T ∈ R^32` the frozen PCA teacher.
* `H_A(X) ∈ [0,1]^4`: income and employment probability *vectors* (two 2-simplices).
  `H_B(X) ∈ [0,1]^2`: coverage. Both are released bitwise unchanged.
* `Z = W' V ∈ R^r`, where `V` is the whitened residual random-feature map. **`V` is a function
  of `(T, H_A)` only.** `H_B` enters the coalition penalty at fitting time and the audits; it is
  never an input to the released map. Any new penalty must preserve this: a penalty that makes
  the *map* depend on `H_B` changes the interface, not just the objective.
* Recipient-specific conditioning variables: local penalties condition on `q_A(H_A)`
  (intercept + degree-≤2 monomials in the two positive-class probabilities); coalition
  penalties condition on `q_AB(H_A, H_B)`.
* Legal inference inputs at release time: `T` and `H_A`. Nothing else.

## 2. The current objective and its fixed-rank optimum

With `A_arm = U_norm − λ P` symmetric and fixed, the released map solves

    max over W'W = I_r of tr(W' A_arm W)   =   a_1 + … + a_r  (Ky Fan),

attained by the top-`r` eigenvectors. This is a **global optimum of that fixed matrix** and
nothing more: it is silent about `φ`, the basis, the nuisances, `λ`, `r`, and every attacker.
`test_free_rank_optimum_keeps_exactly_the_positive_eigenvalues` pins the accompanying fact:
if the width `r` is *free*, the optimiser of the same trace functional keeps exactly the
eigenvectors with positive eigenvalues, and the fixed-`r = d` choice is strictly worse whenever
`A_arm` has a negative eigenvalue.

## 3. Free-rank sign selection: state the feasible set, and note what it costs

The sign-selection argument is correct **only** under the feasible set
`{W : W'W = I_r, r ∈ {0,…,d} free}` with no other constraint. Two consequences that the
proposal does not currently acknowledge:

1. **It breaks the equal-width control.** The study's headline negative result — "every
   spectral arm leaks more than J *at the same width*" — is a width-matched comparison at
   `r = 16`. A free-rank arm has `r = #{i : a_i > 0}`, which will generally not be 16. If
   `r < 16`, a lower leakage number is then partly a *capacity* effect and is no longer
   comparable to J on the term that made the original comparison fair. **Recommended repair:**
   report the realised `r` for every arm, and either (a) additionally evaluate the free-rank
   arm truncated/padded to `r = 16`, or (b) evaluate J-equivalents at the realised width.
   Otherwise the new arm cannot be placed in the existing F3 comparison at all.
2. **It is only an optimum for the matrix it is computed from.** `test_sign_selection_on_the_old_matrix_does_not_optimise_a_new_matrix`
   exhibits a symmetric `A` and a new penalty `P` for which the positive-eigenvalue subspace of
   `A` is strictly suboptimal for `A − λP`: the transplanted value is strictly below the native
   optimum. So sign selection must be recomputed on whatever matrix the *new* objective
   induces; carrying the old arm's retained set over is not a valid shortcut.

## 4. Why "nonlinear penalty" and "free rank" cannot both be closed-form

This is the central finding of this review.

`tr(W' P W)` is proportional to `Σ_{c,k} || E_n[ Z b_k(H) (1{S=c} − m_c(H)) ] ||²` — i.e. the
output-side feature is `g(Z,H) = Z_l b_k(H)`, **linear in Z**. That linearity is exactly why
the objective is a trace form and why an eigensolver applies.

Replace `Z_l` by any genuinely nonlinear `ψ(Z)` — an RBF-kernel HSIC on `Z`, a polynomial in
`Z`, anything a kernel CI test would use — and the penalty is no longer `tr(W' A W)` for any
`W`-independent `A`. `test_kernelised_output_features_break_the_trace_form` pins this with the
cleanest available invariant: the linear penalty is invariant under `W → W Q` for orthogonal
`Q` (as any trace form must be, since `Q` does not change the column space), whereas the
squared-output penalty is not. A functional that is not invariant under a change of basis
within the retained subspace **cannot** be written as `tr(W'AW)`, so:

* **No closed-form spectral optimiser remains.** The follow-up needs iterative optimisation
  (projected gradient / Stiefel-manifold or Riemannian methods), with all the usual local-optimum
  caveats. A "linearise at `W_0`, eigendecompose `A(W_0)`, iterate" scheme is a fixed-point
  iteration, not a global optimum, and must be reported as such.
* **Free-rank sign selection loses its justification simultaneously.** "Keep the positive
  eigenvalues" is a statement about a trace form. Once the objective is not a trace form there
  are no eigenvalues to take the sign of. If Terminal A wants both factors, they need a
  *separate* rank rule (e.g. a validated width sweep) and must say so.

**A cheaper alternative that preserves the closed form** — worth pricing before committing to
iterative optimisation: keep `g` linear in `Z` but enrich the *`H` side* (more basis functions
`b_k`, e.g. higher-degree monomials or random features of `H`). That stays a trace form and an
eigensolver still applies. §5 explains exactly which failure modes this does and does not fix.

## 5. What the counterexamples prove, and which output-side features detect them

| Fixture | Construction | Linear-in-Z penalty | What detects it |
|---|---|---|---|
| **XOR** | `S, H` independent fair signs, `Z = S·H` | `E[Z(S−½)] = 0` exactly | `E[Z·H·(S−½)] = 0.5 ≠ 0` — **linear in Z**, nonlinear in `H`. Fixed by enriching the `H` basis. |
| **Magnitude** | `(Z,S)` uniform on `{(−1,0),(1,0),(−2,1),(2,1)}`, `H` constant | `E[Z(S−½)] = 0` exactly | `E[Z²(S−½)] = 0.75` — **requires a nonlinear function of Z**. `H` is constant, so *no* enrichment of the `H` basis can ever help (`test_polynomial_basis_in_H_alone_cannot_fix_magnitude`), yet `1{|Z| > 1.5}` recovers `S` perfectly. |

So the two counterexamples are not interchangeable, and the distinction is directly actionable:

* The **XOR** failure is a *conditioning* failure and is cheap to repair inside the existing
  closed form.
* The **magnitude** failure is an *output-feature* failure and is exactly the one an MLP or a
  boosted tree exploits. It is provably unreachable by enriching the conditioning side alone.
  Since the ACS attacks that beat the spectral arms are MLPs and boosted trees, this is the
  failure mode the follow-up must target — and, by §4, targeting it costs the closed form.

## 6. Conditional nulls, nuisance error and finite function classes

`test_conditional_penalty_is_zero_when_all_dependence_flows_through_H` constructs `Z = H`,
`S | H ~ Bernoulli(σ(H))`: *all* apparent dependence between `Z` and `S` flows through `H`.
With the correct nuisance, every conditional moment vanishes for `Z`, `Z²` and `tanh(Z)`
(max |moment| < 1e-12), while the *marginal* moment is > 0.05. A marginal penalty would spend
capacity destroying a direction that leaks nothing beyond the already-published `H`. This is
the affirmative case for conditioning, and it is why the marginal arms (M025/M1) are the right
control to keep.

`test_misspecified_nuisance_manufactures_a_penalty_under_conditional_independence` is the
mirror image: on the *same* conditionally-independent distribution, a constant nuisance
`m(H) ≡ ½` produces a moment > 0.05. The penalty is then a measurement of nuisance error, and
paying utility to shrink it buys nothing. Two consequences for the follow-up:

* Cross-fitting controls overfitting; it does **not** certify the conditional model. Any
  new penalty should carry a nuisance-quality diagnostic on held-out rows, reported next to
  the penalty value, so that a large penalty can be attributed.
* A richer output-side class makes this *worse*, not better: more functions `g` means more
  opportunities for nuisance error to show up as apparent dependence.

**Finite conditional-function classes.** The population statement behind the penalty is the L2
(Daudin) characterisation: `E[g(Z,H)(1{S=c} − P(S=c|H))] = 0` for **every** square-integrable
`g` iff `S ⟂ Z | H`. "Every" is doing all the work. A finite class `G` gives an implication in
one direction only (conditional independence ⟹ zero moments), never the converse. Therefore:

* a zero empirical penalty at any finite `λ` and any finite `G` is **not** a certificate;
* what a population conditional-independence implication would additionally need is: (i) `G`
  dense in `L²(Z,H)` — a universal kernel gets you this, a degree-2 polynomial does not;
  (ii) a correctly specified (or nonparametrically consistent) nuisance; (iii) population,
  not empirical, moments, i.e. a uniform-over-`G` concentration argument with an explicit rate.
  None of these is supplied by adding a kernel to a finite-sample penalty. A follow-up that
  claims "a penalty a nonlinear attacker cannot walk around" should be worded as
  *a penalty that is not blind to the specific nonlinearities in `G`*.

## 7. Kernels on X, kernels on Z, and features on S — three different objects

These are routinely conflated and the distinction decides whether the follow-up changes anything:

* **A kernel on `X` (or on `T`).** This is what the random Fourier features `φ(T)` already are.
  Enriching it changes the *input* representation `V`, i.e. what utility is available. It does
  **not** make the penalty nonlinear in `Z`, because the penalty acts on `Z = W'V` after the
  projection. Swapping in a richer `φ` and keeping `g` linear in `Z` leaves the magnitude
  counterexample untouched.
* **A kernel on the released `Z`.** This is the only change that addresses §5's magnitude
  failure — and the one that, by §4, destroys the trace form.
* **One-hot / kernel features on categorical `S`.** With `S` categorical and the one-hot
  encoding already used, a linear kernel on `S` is *already* characteristic for a finite
  discrete variable: every function of a 9-class `S` is a linear combination of its indicators.
  So **enriching the `S` side buys nothing here.** Note the corollary: an HSIC with a linear
  kernel on `Z` and a linear kernel on one-hot `S` is, up to normalisation and the nuisance
  residual, *the penalty already implemented*. Calling the follow-up "HSIC" is not by itself a
  change of method; the change has to be the kernel on `Z`.

## 8. Specific findings for Terminal A

1. **Blocking, if the equal-width claim is retained.** Free-rank selection breaks the `r = 16`
   width match against J. Either report realised `r` and drop "same width", or add a
   width-matched arm. (§3)
2. **Blocking, if a closed form is claimed.** A penalty nonlinear in `Z` admits no eigensolver;
   report the optimiser, its initialisation, and that its optimum is local. Do not describe a
   linearise-and-eigendecompose iteration as a global optimum. (§4)
3. **Confounded design.** The proposal changes the penalty *and* the rank rule at once. With
   three seeds and one evaluation year, a single joint arm cannot attribute an effect to either
   factor. Recommend at minimum a 2×2: {old first-moment penalty, new penalty} × {r = 16, free r}.
4. **Cheap control worth including.** An `H`-basis-enriched arm that keeps `g` linear in `Z`
   isolates the XOR-type failure from the magnitude-type failure at closed-form cost. It is the
   single most informative extra cell.
5. **Diagnostic to add.** Held-out nuisance quality alongside the penalty value, so a nonzero
   penalty can be attributed to leakage rather than nuisance error. (§6)
6. **Wording.** No finite empirical penalty is a certificate. "Cannot walk around" should become
   "not blind to `G`". (§6)
7. **Not a blocker, and explicitly not a request for more experiments.** Nothing here asks
   Terminal A to add a cell it has already locked. If A's design is frozen, items 1, 2 and 6 are
   *reporting* repairs, and items 3 and 4 belong in future work.

## 9. What the ACS result does and does not isolate

The study's own surrogate-mismatch diagnosis is sound but weaker than its phrasing suggests.
Verified by Proposition 1 and the arm's own least-squares check, the spectral step *is* a global
optimum of its fixed matrix, so the failure is **not** a failure to solve the stated problem.
But that rules out exactly one of six candidate causes. The remaining five —
finite input features `φ`; nuisance misspecification; the whitening/normalisation of `U` and `P`;
fixed rank 16; the reconstruction surrogate standing in for downstream utility; and
finite-sample generalisation from fit to final rows — are **not** separated by any measurement in
this study. The counterexamples of §5 prove that a linear-in-`Z` penalty *can* be defeated by a
nonlinear attacker; the ACS numbers are consistent with that being what happened, but they do not
establish it against the other five. The corrected report should say "consistent with", not
"the gap is the surrogate".

**Family-wide conclusion must be dropped.** `RESEARCH_DECISION.md` proposes that if the new
penalty also leaks more than J, "the conclusion generalizes from this design to the closed-form
spectral family for this interface". It does not. Two arms of a family failing is evidence about
two arms. Moreover §4 shows the proposed successor *is not in the closed-form family* — it
requires iterative optimisation — so its failure would say nothing at all about closed-form
spectral methods. This inference is removed from the corrected paper.
