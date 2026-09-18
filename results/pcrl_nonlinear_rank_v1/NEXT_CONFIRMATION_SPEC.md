# NEXT_CONFIRMATION_SPEC — one fixed recipe, and the alternative if it is declined

This document specifies **one** candidate recipe and its controls. Whether it is
worth running at all depends on the research decision in `RESEARCH_DECISION.md`;
if that decision is no-go, §5 is the alternative.

**Nothing here authorises scoring 2016.** The 2016 admission has never been run,
and admission preparation is Terminal B's scope. Its provenance report is
admission preparation only and is never performance evidence.

## 1. Why a next study is specifiable at all

This study's negative result is not "a nonlinear penalty did not help." The
penalty demonstrably did help: at matched rank and policy it produced
statistically significant reductions in measured sensitive recovery (`A/SEX`
−0.0064 at r16, −0.0065 at r8; `AB/SEX` −0.0055 at r8; `A/RAC1P` −0.0073 and
`AB/RAC1P` −0.0116 at r16) with no endpoint significantly worse and no residence
cost. It simply did not go far enough to beat J on every role.

What makes a next experiment well-posed is that a specific, closed-form defect
was found and measured: the registered penalty **is not invariant under a
transformation that the disclosure it measures is invariant under**, and a mean
of **63%** (range 37–91%) of its optimiser's progress was reachable through
exactly that transformation. So roughly 37% of the surrogate movement bought the
real reductions above. Removing the provably inert fraction should let the same
compute buy more — which is a stronger reason to run the fix than a pure
negative would have been.

## 2. The fixed recipe: `rotation_invariant_moment_refinement`

Identical to `METHOD.md` §1-§7 in every respect — same frozen features `V`, same
utility matrix, same frozen three-fold household OOF nuisances, same anchor
bases, same policies and denominators, same Stiefel solver, same two deterministic
starts, same 200-update budget, same selection on the training objective alone —
with exactly **three** changes, all inside the feature family:

1. **Frobenius weighting of the quadratic block.** Weight each off-diagonal
   monomial `z_a z_b` (`a < b`) by `sqrt(2)`, leaving diagonal monomials at weight
   1. The block then equals `||M||_F^2` for the symmetric moment matrix
   `M_ab = E[z_a z_b b_k(H) e_c]`, and since rotation acts as `M -> Q' M Q` it
   becomes **exactly** rotation invariant. Verified on a fixture to `< 1e-12`.
2. **No per-feature standardisation within a block.** Replace the per-feature
   `(raw - mu)/s` by a single scalar scale per block, fitted once at the reference
   projection. Per-feature scales give the features unequal weights and break the
   Frobenius structure a second time (measured slack ~1%). Centring stays, since a
   constant offset does not break invariance.
3. **Fourier features raised to 1024 per bandwidth** (from 32), keeping the three
   bandwidths `(0.5, 1, 2) * sigma_ref`. The Fourier block's residual slack is
   Monte-Carlo error, not a specification error: `omega ~ N(0, I)` is rotationally
   symmetric in distribution, so the block converges to a rotation-invariant limit.
   1024 is chosen because the slack shrinks with feature count and this is the
   largest value that keeps one optimiser start under ~5 CPU-minutes at `r = 16`.

**Predeclared acceptance gate, to be checked before any outcome is read.** Run
`diagnostics.rotation_share` on the fitted maps. The recipe is only eligible to be
scored if the measured rotation-only share is **below 0.10** in every seed, rank
and policy. If it is not, the construction is still misspecified, the study stops
there, and that is itself the reportable result. This gate is an outcome-free
check on the mechanism and costs nothing.

## 3. Controls, matched exactly

| Arm | Role |
|---|---|
| `original` family at the same rank and policy | the registered linear-moment baseline; at rank 16 it **is** the historical `spectral_L1`/`L2`/`C1` and is reused, not refitted |
| `rotation_invariant` family, policies `L1`, `L2`, `C1` | the candidate and its equal-strength and equal-total-mass local controls |
| frozen neural `J` | the competing mechanism, reused unchanged |
| `H`, `E`, `A0` | simple interfaces |
| this study's `nonlinear` family | the misspecified predecessor, so the fix is attributable to the fix |

Same three seeds, same rank 16 **and** the rank-8 compression arm, so penalty
family and dimension stay separable as two factors. Attack slate, selection rule,
budgets, roles, masks, subset indices and seed formulas are the historical ones;
**no new candidate family is introduced**, so nothing has to be added to the
controls. Comparisons are at equal realised rank, with no zero padding.

## 4. Evaluation, and what it can and cannot be

* **Development:** the 2018 pools, exactly as here.
* **Cross-year development:** the 2017 partitions, exploratory only. The 2017
  seal is spent and cannot be restored.
* **Confirmation:** the only unused candidate year is **2016**, whose admission
  has never been run. A 2016 confirmation requires its own prospective protocol,
  its own registered directional predictions, and an admission pass that is
  outcome-free. **It is not authorised by this document and must not be started
  on the strength of a development result.**
* Endpoints, both weightings, the `.01` residence reference, the original
  half-headroom and source-allowance criteria, the advantage rule, the `.001`
  point-difference coordination rule, the paired household-cluster bootstrap with
  studentized max-|t| adjustment, and the withholding grid all carry over
  unchanged and stay labelled as reused.
* Register directional predictions **before** fitting. This study's experience is
  the argument: its six registered predictions made the outcome interpretable in a
  way the predecessor's unregistered pass was not.

## 5. If the recipe is declined: the alternative research decision

The honest alternative is **not** another penalty. It is to stop trying to
control a released channel through a finite empirical moment family on this
interface, and instead attack the question the interface actually poses.

**Alternative: measure the disclosure headroom before designing another
mechanism.** Every mechanism tested on this interface — eight spectral arms, the
frozen neural `J`, and now two more penalty families across two ranks — is
compared against attackers whose strength is unknown in absolute terms. What is
missing is an estimate of how much sensitive information the *appended channel
capacity itself* makes available, independent of any mechanism: i.e. an upper
bound on achievable residence capability at a given level of additional sensitive
recovery, for **any** `r`-dimensional function of `(T, H_A)`. Obtain it by
deliberately maximising the attacker's advantage — fit the channel to *maximise*
sensitive recovery at matched residence capability — and report the resulting
frontier as an envelope. That is a small, bounded, mechanism-free study on
already-used development data. It would tell the next designer whether the
interface has any room at all, which is currently unknown and is the reason
mechanism studies here keep returning qualified negatives.

This alternative is explicitly labelled an optimistically selected descriptive
envelope, not an achievable guarantee, and it would be kept out of any primary
selected-attack comparison.

## 6. What would make either study worth the compute

A candidate merits confirmation only if a complete development result identifies
a **specific better practical option** against `J` **and** both local controls,
with a stated utility/disclosure trade-off and no silently ignored harmful role.
A strict Pareto improvement would be unambiguous; it is not assumed. Mixed
evidence is to be frozen and reported, not re-metricised until it reads as a win.
