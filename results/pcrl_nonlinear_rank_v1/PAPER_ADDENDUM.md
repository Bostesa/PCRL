# Experimental addendum — a nonlinear conditional moment penalty, and why it did not help

Terminal A, study `pcrl_nonlinear_rank_v1`, branch `research/pcrl-nonlinear-rank-v1`.
This is a **development** addendum. It spends no confirmation set, and 2016 is untouched.
The main revised manuscript is Terminal B's and lives in a separate directory; this document
is self-contained and does not restate Terminal B's evidence or conclusions.

## 1. What this addendum adds

The completed residual spectral study closed with a specific next step: replace its finite
first-moment penalty with one "a nonlinear attacker cannot walk around", and select the
channel rank by the sign of the eigenvalues rather than fixing `r = 16`. This addendum runs
that step as a two-factor development experiment — penalty family × channel dimension — and
reports two structural findings and one mechanism finding. All three are negative for the
proposed design, and all three are sharper than "it did not work".

We adopt neither the phrase nor the hope behind "cannot walk around": no finite empirical
penalty over a finite function class is a certificate of anything, and this addendum's own
fixtures show a misspecified nuisance manufacturing a penalty under exact conditional
independence. The accurate statement is that a penalty can be *not blind* to a given family
of functions.

## 2. Finding 1 — eigenvalue-sign rank selection is vacuous for this interface

`r_plus = min(16, #{eigenvalues of U − (P_L + P_AB) above tolerance})` equals **16 in every
seed**, and **not one** of the top sixteen directions is nonpositive. The reduced-rank recipe
is therefore an exact alias of the rank-16 recipe, not a different arm.

The reason is structural, not empirical. `U_raw = (V'R/n)(R'V/n)` and the teacher residual `R`
has 32 columns, so `rank(U) ≤ 32`; measured, `U` has exactly 32 eigenvalues above tolerance and
**zero** below it, so 96 of the 128 whitened directions carry no teacher-reconstruction energy
at all. Subtracting a positive semidefinite penalty cannot raise the positive count, and the
penalties are far too small to remove sixteen of the thirty-two: at the boundary `ev[16] ≈ 8e-3`
against a largest eigenvalue `≈ 2e-1`. `r_plus` can only bind if a penalty drives more than
sixteen of `U`'s thirty-two positive directions to zero, which would require a penalty two
orders of magnitude larger than the ones this interface uses.

A side observation the counts expose without any outcome: the local objectives `U − P_L` and
`U − 2P_L` leave 38–42 directions numerically **at** zero, because `P_L` is built from the
rank-limited `Q_A` basis and does not reach all of `U`'s null space, while `U − (P_L + P_AB)`
has **no** zero eigenvalues and 96 strictly negative ones. Adding the coalition term is what
makes the penalty reach the whole complement. That is a real structural difference between the
local and coalition objectives.

Because every seed gave `r_plus = 16`, the predeclared branch fired and a **rank-8** arm was
run instead, labelled **compression, not eigenvalue-sign selection**, and compared at equal
realised rank with no zero padding.

## 3. Finding 2 — the conditioning variable is nearly uninformative here

The penalty conditions on the released service probabilities through the original polynomial
anchor bases and the frozen three-fold household out-of-fold nuisances `m_j(H)`. Those
nuisances barely beat a constant prior on the protected attributes (seed 0,
representation-fit rows):

| role | prior log loss | OOF log loss | gain | gain as % of prior |
|---|---|---|---|---|
| `A/SEX` | 0.6928 | 0.6914 | +0.0014 | 0.21% |
| `A/RAC1P` | 1.2956 | 1.2831 | +0.0125 | 0.96% |
| `A/public_coverage` | 0.5526 | 0.5050 | +0.0476 | 8.60% |
| `AB/SEX` | 0.6928 | 0.6882 | +0.0046 | 0.66% |
| `AB/RAC1P` | 1.2956 | 1.2652 | +0.0305 | 2.35% |

The released income/employment/coverage probabilities predict sex by well under one percent of
the prior loss and race by one to two percent. The protected residual `e_j = onehot(S_j) −
m_j(H)` is therefore close to the *marginal* residual, so any penalty conditioned on `H` has
little room to differ from the marginal penalty on this interface — **whatever function class
it uses in `Z`**. This bounds the leverage of the whole conditional-penalty idea here, and it
is worth knowing before anyone ports a U-FaTE-style conditional criterion to this data. It is
a property of the interface, not of our construction.

## 4. Finding 3 — the penalty is not invariant under a transformation the disclosure is

This is the mechanism result, and it needs no fitted attacker.

The released channel is `Z = V W`. For orthogonal `Q`, `Z` and `Z Q` determine each other, so
any attacker able to use one is able to use the other: **disclosure is invariant under
`W → W Q`**. So is the utility term `tr(W' U W)`, exactly. So is the original linear penalty
`tr(W' P W)`, exactly. Measured, applying a random `Q` to the original solution changes utility
by `≤ 3e-16` and the original penalty by `≤ 3e-17`.

The nonlinear penalty changes by `4e-4` to `7.5e-3`. It is therefore **not** a trace form for
any `W`-independent matrix, which also means no closed-form eigensolution exists and the
optimality statements of the closed-form spectral line are not inherited. That part was
expected and is stated in the method.

The part that matters is the consequence: the optimiser can lower the penalty by re-basing the
released channel at **zero** utility cost and with **zero** change in what is recoverable. We
measure how much of its progress is exactly that, by minimising the same objective over
rotations of the original subspace alone:

| condition | rotation-only share of the training gain, by seed |
|---|---|
| `nlr16_L1` | 0.907, 0.807, … |
| `nlr16_L2` | 0.645, 0.656, … |
| `nlr16_C1` | 0.625, 0.717, … |
| `nlr8_L1` | 0.590, 0.768, … |
| `nlr8_L2` | 0.367, 0.478, … |
| `nlr8_C1` | 0.514, … |

(complete table in `DEVELOPMENT_2018.md`; per-seed values in `DIAGNOSTICS_SUMMARY.json`)

Between roughly a third and nine tenths of the surrogate improvement is reachable without
changing the released information at all. For `nlr16_C1` seed 0: the objective falls from
−0.5902 to −0.6719 under free optimisation, but reaches −0.6413 by rotation alone, and the
utility at the best rotation is *bit-identical* to the utility at the original solution
(0.80071059 in both). The remainder does move the subspace — projector distance 1.88, largest
principal angle 43.9° — so the refinement is not purely cosmetic; but most of its headline
progress is.

### 4.1 Where the slack comes from, exactly

Localised to the construction, not to the data, and reproducible on a synthetic fixture:

1. **The quadratic block — a specification error, exactly fixable.** With
   `M_ab = E[z_a z_b b_k(H) e_c]`, rotation acts as `M → Q'MQ`, which preserves `‖M‖_F`. But
   `‖M‖_F² = Σ_a M_aa² + 2 Σ_{a<b} M_ab²`, while the registered family sums the monomials
   `a ≤ b` with **equal** weight and so computes `Σ_a M_aa² + Σ_{a<b} M_ab²`. The missing
   factor of two on the off-diagonals is the entire defect. Weighting each off-diagonal
   monomial by `√2` makes the block exactly `‖M‖_F²` and exactly invariant. Measured: the
   shipped convention moves by ~10% under a random rotation, the `√2` version by `< 1e-12`.
   The ~10% figure is a good predictor of the largest measured rotation shares.
2. **Per-feature standardisation — a second, smaller breakage.** Freezing a separate mean and
   scale per feature gives features unequal weights and breaks the Frobenius structure again.
   Measured slack ~1%.
3. **The Fourier block — Monte-Carlo error, not a specification error.** `ω ~ N(0, I)` is
   rotationally symmetric in distribution, so the feature set is distributionally invariant and
   the block converges to a rotation-invariant limit; at 32 features per band it is only
   approximately invariant, and the measured slack shrinks as features are added.

The general lesson is stateable independently of this interface: **a penalty intended to
measure disclosure from a released representation should be a function of the information that
representation carries, not of the basis chosen for it.** A fixed nonlinear feature map of `Z`
does not satisfy that unless its aggregation is built to be reparametrisation invariant. The
fix is closed-form and is specified in `NEXT_CONFIRMATION_SPEC.md`. It was deliberately **not**
applied here: applying it after seeing the diagnostic and re-running would have replaced a
registered experiment with an unregistered one.

## 5. What the penalty does measure correctly

The construction is not broken as a *measure*. Its fixtures show it detecting exactly the
disclosure the original first-moment penalty is blind to:

* **Magnitude fixture.** `(Z,S)` equally likely `(−1,0),(1,0),(−2,1),(2,1)`: the old first
  moment is exactly zero, thresholding `|Z|` recovers `S` perfectly, and the quadratic block
  detects it. This is also why enriching the `H` basis alone cannot help — with `H` constant,
  every `H`-basis moment vanishes while `|Z|` still reveals `S`.
* **XOR fixture.** Independent fair signs `S, H` with `Z = S·H`: the marginal moment is 0, the
  `H`-interaction moment is 1, and `S = Z·H` exactly.
* **Conditional-null fixture.** With oracle conditionals the population moments vanish and the
  finite-sample moments concentrate — they are not required to equal zero.
* **Nuisance-error fixture.** A deliberately misspecified `m(H)` produces a penalty more than
  ten times the oracle value under *exact* conditional independence. A nonzero penalty is
  therefore not evidence of incremental disclosure.

So the negative result is not "the measure is blind". It is "the measure is informative but the
objective built from it is partly gameable by an information-preserving transformation, and on
this interface the conditioning variable has little to offer anyway".

## 6. Scope and attribution

`nonlinear_moment_refinement` is an engineering adaptation. The squared-Frobenius norm of a
residualised cross-covariance between random features and a partialled-out variable is the RCoT
statistic (Strobl, Zhang & Visweswaran, 2019), used here as a penalty, inheriting none of its
null distribution or Type-I control. Multiplying a residualised sensitive side by functions of
`(Z, H)` is a finite fitted subset of the Daudin (1980) partial-association characterisation
used by KCI (Zhang et al., 2011, Lemma 2(v)). Random Fourier features are Rahimi & Recht (2007);
their use inside a spectral adversarial-representation solver is K-TOpt (Sadeghi, Dehdashtian &
Boddeti, 2022). Utility-minus-dependence solved by a matrix trace step is SARL (Sadeghi, Yu &
Boddeti, 2019) and K-TOpt, and the trace step itself is Ky Fan (1949). The direct antecedent
for a *conditional* dependence penalty in this objective family, evaluated on ACS/Folktables, is
U-FaTE (Dehdashtian, Sadeghi & Boddeti, 2024), which conditions on the **discrete ground-truth
label by stratification**, is explicitly scoped to non-continuous labels, and names *sufficiency*
— the fairness family that conditions on the released score — as one it does not address.
Conditioning on a continuous *released* variable instead costs U-FaTE's closed-form global
optimum, which is exactly the trade made here. A nonlinear player acting on the released
representation is itself prior art: OptNet-ARL (Sadeghi, Wang & Boddeti, 2021) uses kernel-ridge
best responses of `z`, unconditionally and with iterative training.

Explicit non-claims: not closed-form, spectral or globally optimal; not a new
conditional-independence criterion; not enforcement of `Z ⊥ S | H`; not a bound on `I(S;Z|H)`;
not a calibrated test; not the first conditional fair-representation method; not the first to
penalise nonlinear functions of a representation; and not evidence of robustness to nonlinear
adversaries. Any novelty statement is phrased as absence of evidence after a bounded search.

Two inferences are explicitly **not** drawn. First, two penalty families failing is evidence
about two penalty families, not about the closed-form spectral family as a whole — and by §4 the
refined objective is not in that family at all. Second, because B receives only `H_B`, every
B-view quantity including commute and coverage capability is identical across all interfaces by
construction; those numbers are structural constants and no interface can win or lose them.
