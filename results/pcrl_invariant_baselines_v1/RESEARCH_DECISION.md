# Research decision

## The verdict in one line

**The repair worked mathematically and failed scientifically.** The objective is now
exactly rotation invariant — verified to machine precision, gate passed 18/18 — and the
measured disclosure it produces is **worse**, not better, than the defective objective
it repairs. The predecessor's central forward-looking inference is refuted by direct
experiment, and three external baselines beat the repaired mechanism.

## Disclosure: what was registered, and how it turned out

Seven directional forecasts were registered in `PROTOCOL.md` §2 and hashed into
`PROTOCOL_FREEZE.json` **before** the fit phase started. **Three of the seven were
wrong.** That is reported first, because the registration is what makes the rest of
this document worth anything.

| # | Registered forecast | Outcome |
|---|---|---|
| **Q1** | Rotation-only share `< 0.01` in every seed, rank and policy | **Confirmed**, and structurally. Max abs share `1.4e-15` over 18 cells; the penalty moves by `5.6e-17` under random rotations; on real ACS data `abs(L(WQ) − L(W)) = 1.1e-16`. Predecessor, same diagnostic, same cells: mean **0.628** |
| **Q2** | The repaired family attains a **higher** (worse) training objective than the defective family in ≥15 of 18 matched conditions | **Refuted — and the forecast was ill-posed.** Under the repaired objective, the only common yardstick, the repaired map is **better in 18 of 18** cells, and both beat the linear-only solution. My registration compared values across two differently normalised objectives, which is exactly the comparison this study's own protocol says is invalid. Disclosed as my error |
| **Q3** | The repaired family's measured `A/SEX` recovery is **not worse** than the defective family's in ≥2 of 3 seeds | **Mixed on its own terms, and refuted on the wider endpoint set.** Evaluated exactly as registered, per seed: the criterion is **met in 5 of 12** policy x rank x weighting cells and **not met in 7**. It is strongly policy-dependent — at rank 16 the `L1` and `L2` arms are *better* on `A/SEX` (up to −0.0077), while `C1`, the headline coalition policy, is worse in **0 of 3** seeds under both weightings. Beyond `A/SEX`, the repaired arm is **significantly worse** on `AB/RAC1P` at rank 16 and on `A/RAC1P`, `AB/SEX` and `AB/RAC1P` at rank 8, with **no endpoint significantly better** |
| **Q4** | The repaired family still fails the coordination rule against both local controls | **Confirmed**, all 4 registered cells NOT SUPPORTED |
| **Q5** | The repaired best candidate remains significantly worse than `J` on `A/SEX` under at least one weighting | **Confirmed, under both.** `riv16_C1 − J` on `A/SEX`: **+0.0135** [+0.0059, +0.0212] unweighted, **+0.0153** [+0.0071, +0.0234] person-weighted. Also significantly worse on `A/RAC1P` and `AB/RAC1P` |
| **Q6** | LEACE achieves near-zero *linear* recovery from the transformed channel while **selected** recovery from the augmented wire stays above zero | **Confirmed.** Linear cross-covariance driven to `4e-16`, yet LEACE's additional recovery from the augmented `A` wire remains strictly positive on every endpoint. `H_A` still discloses, exactly as predicted |
| **Q7** | At least one external adaptation proves an alias or infeasible | **Confirmed by the alias route.** The SARL audit avoided six fits; SPLINCE turned out comfortably feasible (`cond(U'V)` 2.2–3.5 against a `1e6` gate) |

I predicted the repair would cost descent power and preserve measured protection. The
opposite happened on both counts: it gained descent power and, on the coalition policy
and on race recovery, lost protection.

Per-seed accounting for Q3 is in `Q3_PER_SEED.csv`; the wider endpoint set is in
`PAIRED_INTERVALS.csv`.

## Conclusion

**The repaired method is not a candidate for confirmation, and the repair makes the
case against this mechanism family stronger rather than weaker.**

### 1. The repair is real, exact, and verified

Not a convergence argument. The quadratic block is the true `‖M‖_F^2` and rotation acts
as `M → Q'MQ`; the kernel block sees the data only through pairwise distances. Together
with the already-invariant utility and linear terms, **the whole objective is a function
on the Grassmannian**. Measured: rotation-only share `1.4e-15`, direct identity
`5.6e-17`, 40 validation fixtures passing at tolerances declared before fitting.

The predecessor's gate — "the recipe is only eligible to be scored if the measured
rotation share is below 0.10" — passes by fifteen orders of magnitude.

### 2. The predecessor's central inference is refuted

`NEXT_CONFIRMATION_SPEC.md` §1 argued: *"roughly 37% of the surrogate movement bought
the real reductions. Removing the provably inert fraction should let the same compute
buy more."*

It does not. Removing it lets the optimiser move **further** in the repaired objective
(18/18 cells) and **leak more**:

| Contrast (repaired − defective) | Endpoint | Unweighted | Person-weighted |
|---|---|---|---|
| `riv16_C1 − nlr16_C1` | `AB/RAC1P` | **+0.0078** [+0.0026, +0.0131] | **+0.0066** [+0.0003, +0.0129] |
| `riv8_C1 − nlr8_C1` | `A/RAC1P` | **+0.0079** [+0.0027, +0.0132] | **+0.0085** [+0.0018, +0.0153] |
| `riv8_C1 − nlr8_C1` | `AB/SEX` | **+0.0033** [+0.0003, +0.0062] | +0.0028 (n.s.) |
| `riv8_C1 − nlr8_C1` | `AB/RAC1P` | **+0.0066** [+0.0006, +0.0126] | +0.0073 (n.s.) |

Bold cells have adjusted simultaneous intervals entirely above zero. **No endpoint is
significantly better.** The measured cost of removing the provably inert slack is real
and one-directional.

**The mechanism reading, stated carefully.** This does *not* show that rotation
invariance is harmful in principle. It shows that in this objective the coordinate
dependence was acting as an **implicit regulariser** on an under-determined surrogate:
the defective penalty's extra freedom kept the optimiser near the linear-moment
solution, and removing it let the optimiser travel further into a region where the
finite empirical moment family and the actual attacker slate disagree. The repaired
subspaces are far from both the defective and the original ones (projector distance
1.2–2.0 of a maximum 5.66 at rank 16). **Surrogate–attacker mismatch, not rotation, is
the binding problem**, and the repair made that visible by removing the confound.

### 3. The external baselines beat it

This is the first time this line of work has had method-level controls, and the answer
is unambiguous. Against `spectral_riv16_C1`:

| Baseline | `A/SEX` | `AB/SEX` | `A/RAC1P` | `AB/RAC1P` | residence cost |
|---|---|---|---|---|---|
| `leace_A0` | **−0.0081** | **−0.0086** | **−0.0131** | **−0.0158** | +0.0036, **not significant** |
| `splince_A0` | **−0.0088** | **−0.0077** | **−0.0183** | **−0.0165** | **+0.0089**, significant |
| `optnet16_C1` | **−0.0074** | **−0.0110** | **−0.0161** | **−0.0145** | **+0.0086**, significant |

(unweighted; person-weighted agrees. Bold = adjusted interval excludes zero. For
recovery, negative means the baseline leaks **less**; for residence, positive means it
costs **more** utility.)

**LEACE dominates the repaired mechanism**: significantly less recovery on **all four**
sensitive endpoints with **no statistically detectable residence cost**. It also clears
the legacy source allowance in **3 of 3** seeds where the repaired arm clears it in 1–2.
SPLINCE and OptNet-C1 also beat it on every sensitive endpoint but pay a significant
residence cost, so they are trades rather than dominations.

Against `J`, the frozen neural channel: `leace_A0`, `splince_A0` and `optnet16_C1` are
**statistically indistinguishable on all five family endpoints**, while the repaired
spectral arm is **significantly worse on three of four**. `optnet16_L1` is significantly
worse than `J` on `A/SEX`.

**A closed-form 2023 linear-erasure method, applied to a channel this project already
had, matches `J` and beats the mechanism this line of work has been developing for four
studies.** That is the most consequential finding here.

Two caveats that keep this honest, both declared before the numbers were read:
* SPLINCE and LEACE share a kernel, so under SPLINCE's own Thm 2 they would give
  identical predictions after refitting an *unregularized* unique-minimiser model. The
  differences seen here are attributable to **attacker regularisation and
  nonlinearity**, not to a stronger erasure guarantee.
* SPLINCE sees two authorised task labels (`income_binary`, `civilian_at_work`) that no
  other arm uses in representation fitting. That asymmetry favours SPLINCE on utility.

### 4. Rank behaviour

`r_plus = 16` in all three seeds, as already known; the sign-selection rule changes
nothing and **no alias was refitted**. Rank 8 remains the predeclared compression
alternative, never presented as the rule's output.

The genuinely new rank finding is on the **erasure** side and it is support-driven:
LEACE and SPLINCE delete exactly `rank(Σ_XZ)` directions, so the 16-coordinate channel
drops to rank **6, 6, 7** across seeds 0, 1, 2. Seed 2 differs because its `RAC1P` class
index 3 has **zero** population support, making the centred joint one-hot rank 9 instead
of 10. **Class support directly determines the realised width of a released channel** —
recorded, not repaired. SPLINCE costs no extra rank over LEACE, exactly as its paper
claims.

### 5. The SARL alias

The marginal spectral arms **are** SARL-style residual-teacher adaptations: rebuilding
from the derivation reproduces the stored maps **bitwise**. The sign rank rule and the
fixed `r = 16` agree (32 positive eigenvalues). The one real mismatch is per-attribute
trace reweighting of the sensitive Gram, which is **not** absorbable into `λ̄` and moves
the selected subspace by projector distance 0.10–0.20. Credited and reused; **zero
duplicate fits**.

Recorded because it is the easiest bug to ship: copying SARL's ascending sort order
without reversing the sign convention selects a subspace at projector distance **5.657
of a maximum 5.657** — completely orthogonal to the correct one.

## One next decision

**Do not run a confirmation on this design, and do not fix it again.** The predecessor
said "fix the rotation defect first, then decide." The defect is fixed, exactly, and the
answer is that it was not what was limiting the method. Another penalty repair on this
interface is not indicated by anything measured here.

**2016 is not authorised by this decision and was not touched.** It remains **UNSCORED**.
A 2016 confirmation would need its own prospective protocol, its own registered
predictions and an outcome-free admission pass, and must not be started on the strength
of a development result — including this one.

### What, if anything, merits future confirmation

Nothing in **this** study's mechanism line. One thing from outside it does, and it is
not our method:

> **`leace_A0` is the only arm that matches `J` on every sensitive endpoint while
> costing no statistically detectable residence capability, and it clears the source
> allowance in 3 of 3 seeds.**

That is a *candidate observation*, not a nomination. Before it could deserve the one
unused year it would need: a prospective protocol registering it as the primary arm;
registered directional predictions; an explanation of why a linear-erasure guarantee
should hold up against this nonlinear attacker slate when its own authors conjecture it
should not; and a resolution of the fact that its advantage here is measured on
development pools this project has used repeatedly. **A development result is not a
reason to spend 2016.**

## Uncertainty this study cannot remove

* These are **development** numbers on pools used repeatedly. The paired household
  bootstrap (2000 replicates, 20147 cohort-household clusters) quantifies sampling
  variability for **fixed fitted systems**. It cannot undo repeated use, and it is
  neither Census replicate-weight variance nor retraining variability.
* Three seeds cannot resolve seed-to-seed variability on race recovery, which remains
  large relative to the effects.
* The kernel block has an **`m`-dependent floor** (`METHOD.md` §3.7): its conditional-null
  value is governed by the frozen 512-row subset, not the pool, decaying as `~1/m`. It
  applies equally to every arm so it does not bias comparisons, but it bounds what an
  absolute block value can mean.
* Nuisances were held fixed by design. The misspecification fixture shows a wrong
  `m(H)` manufacturing a penalty more than ten times the oracle value under **exact**
  conditional independence. A nonzero penalty is **not** evidence of incremental
  disclosure.
* Nothing here is a certificate. Vanishing fitted finite moments imply neither
  `Z ⊥ S | H` nor any bound on `I(S;Z|H)`.
* The `.001` coordination rule is a **point-estimate** rule reused for comparability.
  A pass is not equivalence; equivalence and noninferiority are reported separately.
  Absence of significance is neither.
* Shared policy coefficients do **not** imply equal effective regularisation across
  methods. The OptNet multi-lambda form is an **extrapolation** not present in its paper.
* This bounded comparison is **not** a comprehensive benchmark of every cited method.

## Corrections carried into this report

From Terminal 2's independent review, applied here and **not** retro-fitted into the
historical evidence:

* The `nlr8_C1` vs `J` deficit on `recovery/A/SEX` person-weighted is **+0.01210**, not
  the +0.0098 printed in the predecessor's `RESEARCH_DECISION.md` and `PAPER_ADDENDUM.md`.
  The generated tables were always right; the prose was not.
* The predecessor's attribution row "no endpoint significantly worse" is **false** and
  is not repeated: `spectral_nlr8_L1 − spectral_lin8_L1` on `recovery/A/RAC1P` unweighted
  is +0.00862, adjusted [+0.00168, +0.01556].
* The 63% rotation share is an **attained** value, not a supremum and not a causal
  decomposition: every rotation search stopped on budget exhaustion or line-search
  failure. It is described that way throughout.
