# CONTRIBUTION_ASSESSMENT — an independent review memo

Written as a reviewer would write it, against the v3 manuscript and the three completed
studies. It does not predict acceptance, and a clean LaTeX build is not submission
readiness.

---

## The one-line verdict

**This is an empirical and problem-formulation contribution. It does not support a
competitive method claim, and after three studies the evidence increasingly argues against
the mechanism family the project has been developing.**

The paper is unusually honest about that, and the honesty is not decorative: the
measurement contract is what made three successive negative results *legible* rather than
ambiguous, and that is the part worth keeping.

---

## The three strongest contributions

### 1. A release constraint that changes what "preservation" means, implemented and audited

Most fair- and invariant-representation work learns a representation from scratch, so
preserving anything is a bounded perturbation with a norm to control. Here the published
output is immutable and the mechanism may only **append**. Preservation becomes an
architectural identity checked bitwise on 210 of 210 released views, and the paper
immediately separates it from the thing it does not imply — service *accuracy*, which moves
by up to 0.015 nats across one year on identical released bits.

That separation is the contribution, not the concatenation. It is the kind of distinction
that is obvious once stated and is routinely elided.

### 2. A measurement contract that survives its own negative results

Recipient-specific and coalition-specific audit roles; fresh and frozen attackers never
pooled; absolute and incremental recovery always side by side; negative increments retained
unclipped; denominators printed beside every count; per-endpoint results never averaged into
one privacy score; multiplicity fixed before the seal is opened.

The proof that this contract works is that it caught things. It caught that
`J`'s own local-sex increment is negative and below `H`'s own recovery — a selection
artifact, not protection. It caught that 22 of 429 role-cells have negative increments. It
caught that adding attack candidates can *lower* a measured maximum. A contract that only
ever confirms is not doing work; this one repeatedly embarrassed its authors.

### 3. One confirmed effect on a sealed year, correctly bounded

On a locked, previously unused survey year, the coalition-conditioned penalty beats both a
strength-matched and a mass-matched local control: 14 of 16 sensitive cells strictly better,
0 worse, 2 unresolved, under simultaneous intervals and both weightings. The controls are
the right ones — they exist precisely so that "a coalition effect" cannot be "more penalty".

And the paper does not inflate it. It states that the paired residence rule was a *point*
criterion, that interval non-inferiority at the same margin holds in **0 of 4** comparisons,
and that the two unresolved cells are undecided rather than unaffected.

---

## The three strongest objections

### O1. The confirmed effect is one cell of a large grid, and the mechanism it validates is
### the one the paper then spends three studies failing to improve

The sealed result concerns *coalition conditioning versus local controls*. It says nothing
about whether this mechanism family is a good way to build a channel — and by the paper's
own evidence it is not. A reviewer is entitled to ask what the confirmed effect is *for*,
if the only construction that exhibits it is beaten by a closed-form 2023 eraser applied to
a channel the project already had.

**The honest answer, which the paper gives:** the effect is evidence that coalition-aware
penalties can do something local penalties cannot, in this interface. Whether any
competitive mechanism should use one is open. That is a narrower contribution than the
framing of earlier drafts, and v3 is the first version to say so plainly.

### O2. Three seeds, one state, one age band, two survey years — and one of them is spent

Most of the grid in every attribution family is *unresolved* at three seeds. Seed-to-seed
variability on race recovery is large relative to the effects being reported. The 2018 pools
have been used repeatedly; the 2017 seal is spent; 2016 is the only unused year and is
correctly not spent on a development result. So the paper's forward evidential capacity is
essentially one year, and it has no design registered for it.

This is a real ceiling, and the paper acknowledges it without solving it. A reviewer may
reasonably regard the whole programme as under-powered for the questions it asks.

### O3. The external comparison is three adapted methods, not a benchmark — and the
### adaptations are load-bearing

LEACE and SPLINCE act on the auxiliary channel alone (necessarily: applying them to the
joint wire could alter the published output, which would break the paper's first
deliverable). SPLINCE preserves covariance with the *authorised training* tasks, an
asymmetry in its favour on utility that no other arm enjoys. OptNet-ARL's multi-`λ`
multi-attribute form is an extrapolation asserted in one sentence of the source with no
equation and no code. Shared policy coefficients do not imply equal effective
regularisation.

Every one of these is disclosed. But a reader should not take "LEACE beats the developed
mechanism" as a benchmark result. It is a measured comparison, on one interface, with
declared adaptations, on development pools.

---

## Claim corrections this review required

Fourteen, in `CORRECTIONS.md` B1–B14, with `CLAIM_LEDGER.csv` giving per-claim provenance.
The four that changed what the paper asserts:

| # | Was | Is |
|---|---|---|
| B1 | "LEACE **dominates** the repaired mechanism with **no** detectable residence cost" | Recovery advantage holds on all four endpoints. The residence difference is **unresolved** with an interval admitting 0.010 nats, it **contains SPLINCE's significant cost**, and against the channel LEACE actually transformed the cost **is significant** (+0.0098 unweighted). *Dominates* withdrawn. |
| B2 | Three external arms are "statistically **indistinguishable** from `J`" | **Unresolved** differences. No equivalence test was run; all three point estimates lean towards `J` on local sex, two missing significance by under 0.0007; `OptNet-L1` **is** significantly worse. |
| B4 | Coordinate dependence "was acting as an **implicit regulariser**" | A hypothesis. Five ingredients changed together and no ablation separates them. Surrogate–attacker mismatch remains the live explanation. |
| — | "4 of 24 sensitive cells worse" (repair vs defective) | **8 of 48**. The prose denominator counted contrast × endpoint rows; the cells are contrast × endpoint × weighting. Found by recomputation, not by reading. |

Plus one new finding this review contributed rather than corrected: **every headline paired
contrast is invariant to the attack scope**, because the shared `H` baseline cancels, while
`H`-relative levels move by a factor of five between scopes (`CORRECTIONS.md` B11). That
matters because it is what makes the comparisons robust to a reporting choice no incoming
document had documented.

---

## The smallest remaining scientific gap

**One prospective protocol, registered before any fit, testing whether the coalition
structure buys anything on top of a strong erasure baseline.**

Everything else is either done or not worth doing. Specifically:

The paper now has (a) a confirmed coalition effect against *local spectral* controls, and
(b) a demonstration that a closed-form linear eraser beats the spectral mechanism outright.
It does **not** have the one comparison that would make the formulation matter: *a
coalition-aware eraser against a local-only eraser on the same channel.* If coalition
conditioning is a real design principle rather than an artifact of the spectral family, it
should show up when applied to the family that actually works here.

That is a small, cheap, well-posed question. It needs no new mathematics, it reuses the
existing interface and attacker slate, and it is the only thing in this programme that would
deserve the one unused year — and then only after a registered protocol with directional
predictions, an outcome-free admission pass, and an account of why a linear-erasure
guarantee should survive a nonlinear attacker slate when its own authors conjecture it will
not.

**What does not deserve it:** another penalty repair on this interface. Study 3 settled
that. The defect Study 2 diagnosed was fixed exactly, and it was not what was limiting the
method.

---

## Two notes on how this paper should be read

**On the negative results.** Three failed refinements are reported at the same length and
with the same rigour as the one success. That is the right call and it should not be read
as weakness: the rotation repair in particular is a clean, fully registered experiment whose
prediction was refuted by its own authors' design. Papers that only report what worked
cannot produce that.

**On what would change the verdict.** Not a fourth refinement of the same family. A
demonstration that the recipient/coalition structure changes the answer for a mechanism that
is competitive on its own terms. Until then the defensible claim is the one the paper makes:
an evaluated release problem, one confirmed effect, and three honest negatives.
