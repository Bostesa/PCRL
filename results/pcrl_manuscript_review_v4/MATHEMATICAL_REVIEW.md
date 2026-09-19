# MATHEMATICAL_REVIEW — v4

Independent review of (i) the mathematical statements in Study 4 and (ii) the pending coalition-
conditioned projection and neural factorial **as specified** — no protocol for either has been
committed (see PENDING_INTEGRATION.md). Every claim here is backed by an executable fixture in
`checks/math_fixtures.py` → `MATH_FIXTURES.json`. No ACS data is read by the fixtures.

Each item is labelled **BLOCKING** (the statement or construction is invalid as written and should
be fixed before fitting) or **DISCLOSE** (a limitation to state; not a reason to stop measuring).
No item is a veto of an empirical comparison, and none requires an arbitrary-nonlinear privacy theorem.

## M1 Conventions (F1, F1b) — DISCLOSE + one BLOCKING condition
With row vectors, v=(z−μ)Σ^{-1/2}, z_out=μ+vPΣ^{1/2} give z_out=μ+(z−μ)M, M=Σ^{-1/2}PΣ^{1/2}.
M²=M and rank M=rank P (error 1.7e-15), but M is oblique (asymmetry 1.02). Code must not assume a
symmetric projector or use Mᵀ for M.
**BLOCKING condition:** if Σ is rank deficient, Σ^{1/2}Σ^{+1/2}=Π_supp≠I; M is idempotent only if
range(P)⊆supp (fixture violation 0.085). Declare a RELATIVE rank tolerance for the support.
H is appended unchanged; M acts on z only, so A's inference needs no H_B.

## M2 Masks and centring (F4 centring numbers) — BLOCKING if violated
Compute the moment on the declared valid-role masks with a GLOBALLY centred channel. Raw cross-moment
and centred covariance differ (0.0033 max abs in the fixture). Per-role re-centring makes the
between-role mean shift exactly zero (6.5e-17 vs 0.046 globally) — erasing such a moment erases nothing.
An implementation must not switch silently between covariance and raw cross-moment.

## M3 Why H_A vs H_AB changes the target, and duplication does not (F5) — supported
S=a⊕b (a in H_A, b in Z): marginal channel moment 0.0007; H_A-interacting basis 0.125; duplicated H_A
columns bit-identical 0.125. S₂=b_{H_B}⊕b: A-view basis 0.0016, AB-view 0.124. The distinction is the
span of the conditioning set. Cross-fitting controls own-row overfit; it is not evidence of correct
nuisance estimation nor of conditional independence (DISCLOSE).

## M4 Fixed-rank alias (F2) — proved; one exception BLOCKING for registry accounting
cM has the same eigenvectors, eigenvalues scaled by c>0, order and ties preserved ⇒ identical top-k
projector (difference 0.0, including a tied boundary). A mass-doubled local HARD projection is NOT a
distinct method; register it as an alias, not a comparator. Exception: an absolute-tolerance rank rule
(rank 2 at c=1, 3 at c=2) or ridge shrinkage breaks the alias — say which rule is used.
Local-EXPANDED control changes the feature family: useful, imperfect capacity control.

## M5 Rank reduction and information (F3) — DISCLOSE
Invertible shrinkage cannot reduce information available to unrestricted attackers; a hard projection
can. Neither guarantees improvement in a finite attacker benchmark. Full-span projection annihilates
the specified empirical moments (8.7e-16); top-k generally does not (1.73). The trace criterion supports
optimality of the retained subspace for that trace on the fitted moments; not minimal leakage, not
attacker optimality, not a population statement.

## M6 Convex-correction claim (F4) — TRUE ONLY NARROWLY; counterexample given
For fixed offset ℓ₀ and correction Φθ linear in a fixed basis: mean CE is convex in θ; ∇θ at 0 =
Φᵀ(p₀−Y)/n = the empirical cross-moment the projection annihilates. Zero gradient + convexity ⇒ θ=0 is
the empirical global minimiser over that family (direct search: improvement 0.0).
**Smallest counterexample (BLOCKING if the implemented family has it):** include an intercept with ℓ₀
fitted on a disjoint fold (the p0_fit/mapper_fit/monitor design). Intercept gradient = in-sample
calibration error 0.026; shifting by 0.125 lowers CE by 0.0017.
**Corrected statement:** fixed offset; family exactly {Φθ}; Φ's empirical cross-moment with (p₀−Y) zero
on the masked rows scored; no intercept or column outside the annihilated span ⇒ θ=0 empirically optimal
for that family. Does not extend to a freely retrained H baseline, nonlinear corrections, population
leakage, or the augmented release.

## M7 Completeness counterexample (F6) — standing caveat
S~Bern(½); Z|S=1 uniform ±1; Z|S=0 ~ N(0,1). First and second cross-moments vanish exactly in population;
1{|Z|=1} recovers S with accuracy 1.0. Moment annihilation (LEACE or the proposal) is not sufficiency;
low measured recovery is not independence.

## M8 Relation to LEACE (F7) — BLOCKING for the specification
LEACE r(x)=x−W⁺P_{WΣ_XZ}W(x−E X) (Thm 4.2/4.3, Eq. 1; transcribed from the PDF in
BASELINE_ADAPTATIONS.md §3.1 at 73903b7f). Full-rank Σ: proposed map = LEACE with a different
cross-moment (5.1e-15). Rank-deficient Σ: differ by 2.28; LEACE keeps off-support components of new rows,
the proposal deletes them. Declare the form. Novelty is at most the choice of coalition-conditioned
cross-moment and rank control on strong existing channels — an adaptation.

## M9 Study 4 statements
- "11 > 8 ⇒ rank 0": nominal centred rank is 10; realised width-16 loss 10,10,9. Result measured, reason withdrawn.
- "A0 is the minimiser of U": only the distortion term. Penalty is maximal at step 0.
- g = CE(p0) − CE(q) adds no encoder gradient beyond CE(q) (p0 independent of θ) — Study 4 states this correctly.
- Sign convention: verified by Study 4 fixture and by the monotone dose-response in T1's diagnosis.

## M10 Neural factorial (Track N), as specified — checks to perform on commit
1. Actual J weights loaded and hash-verified, not re-trained or A0.
2. Teacher = frozen A0 in every arm, so init and teacher are not confounded.
3. γ=0 removes teacher preservation but keeps authorised source losses; source heads read Z only.
4. Checkpoint 0 gets the same fresh-probe budget as every checkpoint; repeated step-0 selection is an outcome.
5. Saved-trajectory rescoring labelled counterfactual selection (T1 already does this correctly).
6. Attacker machinery held fixed; no ensemble-size or architecture sweep.
7. Effects via matched contrasts; any J-init gain compared with untouched J.
