# Corrected research decision (companion report)

This is a **companion** to `results/redesign_20260917_acs_spectral_transport_v1/RESEARCH_DECISION.md`,
not a replacement. The original decision, its registered rules and its outcome are preserved
unchanged and remain the record of what was decided on the evidence available at the time. This
document states the same conclusions with the corrections in [`ERRATA.md`](ERRATA.md) applied, and
adds what an independent reanalysis of the stored evidence establishes.

**No numerical result changed.** All 90 family endpoints reproduce under an independently written
scorer to 1.1e-15; candidate selection re-derived from stored validation losses gives 0 mismatches
in 924 cells; the lock verifies over all 17,639 inputs with every amendment applied.

## 1. What is established

**Exact preservation of the published service outputs.** Structural, and verified bitwise on all 210
released 2017 views. This is the strongest claim in the study and it does not depend on any
statistical assumption. It does **not** mean the service stays accurate: on identical released
vectors, employment log loss rises 0.0138 nats and income falls 0.0145 nats from 2018 to 2017.

**Reusable residence capability beyond the fixed service, and it transports.** Every augmented
interface clears the 0.01-nat reference over H in every seed, weighting and mode; the spectral arms
carry 0.025–0.030 nats against 0.017–0.019 for the frozen neural channel J. The qualification the
original report omits belongs here: the stronger half-headroom reference — is the channel's residence
probe within half the headroom between the released service and an unrestricted PCA32 / rich-bank
representation — **passes in only 0–2 of 3 seeds for every interface in both modes**. The channel adds
real capability relative to what is already published; it does not approach what an unrestricted
representation of the same covariates would give.

**Coalition conditioning lowers measured recovery against both local controls.** On the sealed 2017
partition, `spectral_C1` beats `spectral_L1` and the equal-total-mass `spectral_L2` on the sensitive
endpoints under simultaneous 95% intervals and both weightings: 8 of 8 endpoints better, 0 worse,
critical value 2.9301. Frozen 2018 attacks (Mode A) reproduce the ordering, so it is not an artifact
of refitting attackers on the new year. **The registered F1 rule passed and this audit reproduces it
exactly.**

**The residence side of that rule is a point criterion, and only a point criterion.** As implemented,
it is `seed-mean(C1 − comparator) ≤ .001`, one-sided. The four values are 0.00029, 0.00010, 0.00059
and 0.00017. Retrospectively, **0 of 4 comparisons support noninferiority at 0.001** — the pointwise
one-sided 95% bootstrap upper bounds are 0.00112, 0.00103, 0.00135 and 0.00102, all above the margin,
and the simultaneous bounds are higher still — and 0 of 4 support two-sided equivalence. The honest
statement is: *the point difference meets the registered threshold; the data are compatible with a
residence cost somewhat larger than it.* "Within the prespecified 0.001 band" must not be read as an
interval claim.

**The tested design is not competitive with the frozen adversarial channel.** F3 fails, and it fails
more comprehensively than the original text says: `spectral_C1` is significantly worse than J on
**all four** sensitive endpoints under **both** weightings, plus both source probes, and better only
on residence (−0.0093 [−0.0150, −0.0036]). On A/race the paired difference is 0.0377 [0.0294, 0.0459].

**That comparison is not a traversable frontier.** Under the declared withholding mechanism the
expected residence gain is exactly `p` times the full-release gain, so the probability matching a
target's mean residence gain is identified in closed form. **J would need p\* = 1.49 to be as useful as
C1** — it cannot get there at any withholding schedule, and neither can E (1.07), L025 (1.62) or L20
(1.56). Where matching *is* identifiable the ordering survives in both directions: C1 beats withheld
A0 (p\*=0.976) and withheld S0 (p\*=0.929) on all four sensitive endpoints, and J beats withheld C1
(p\*=0.670) on all four. This is post-hoc average-utility matching, and the fixed-p intervals do not
transfer to it; it is reported as exploratory context that *strengthens* rather than softens the
negative verdict.

## 2. What is measured but not certified

* **Every recovery number is a floor under the tested attack families.** The penalty optimises a
  finite set of first moments, linear in Z over a degree-2 basis in the service probabilities.
  Zero fitted moments certify nothing: the fixtures show a distribution with exactly zero first
  moments from which thresholding recovers S perfectly.
* **Negative increments are measurements, not protection.** 22 of 429 unweighted role-cells are
  negative in `transport_all`. J's own A/SEX increment is negative under PWGTP with an interval
  excluding zero, and its absolute A/SEX recovery (0.0092) is below H's own (0.0101). Validation
  selection generalises imperfectly; H's attack routed onto the augmented wire remains executable,
  so a negative increment removes nothing an H-only attacker already had.
* **"No significantly worse endpoint" is not "no harm".** Two F1 A/SEX endpoints have adjusted
  intervals that cross zero on the harmful side ([−0.00585, +0.00020] and [−0.00568, +0.00052]).
  The rule is satisfied; the endpoints are not shown to be unaffected.
* **A class with no support is unmeasured, not protected.** RAC1P class 3 has one person in the final
  partition and none in 2017 attacker validation. Full nine-class log loss is always scored with the
  1e-12 floor; class-level statements are withheld; the comparable-category diagnostic reproduces the
  F1 race differences on supported classes only.

## 3. Where the design fails — corrected

* **Consistent with a surrogate mismatch; not isolated to it.** The spectral step *is* a global
  optimum of its fixed matrix (Ky Fan, checked against direct least squares for every arm), so a
  failure to solve the stated problem is excluded. That excludes one of six candidate causes. The
  finite feature map, nuisance misspecification, the whitening and trace normalisation, the fixed
  rank 16, the reconstruction surrogate standing in for downstream utility, and finite-sample
  generalisation are **not** separated by any measurement in this study. The counterexamples prove a
  linear-in-Z penalty *can* be defeated by a nonlinear attacker; the ACS numbers are consistent with
  that, and do not establish it against the alternatives.
* **Capacity spent on leaky directions.** With r fixed at 16, the retained set can include directions
  whose penalty exceeds their utility. SARL's Theorem 3, OptNet-ARL's Theorem 4.1 and K-TOpt's
  Corollary 4.1 all already select rank by eigenvalue sign; fixing r = 16 was a width-matching choice,
  not an optimum, and the study says so.
* **Not a withholding artifact.** Neither family robustly dominates the other: withholding dominates a
  given spectral arm in 1–3 of 180 fixed comparisons, and a spectral arm dominates withholding in 0–3
  of 180. Non-dominance by a finite control list is not evidence that C1 is efficient.

## 4. What is *not* concluded

The original decision proposed that if a nonlinear conditional penalty also leaks more than J, "the
conclusion generalizes … to the closed-form spectral family for this interface." **This inference is
withdrawn.** Two arms failing is evidence about two arms. And a penalty acting on nonlinear functions
of Z is provably not a trace form `tr(W'AW)` for any `W`-independent `A` — it is not invariant under
`W → WQ` for orthogonal `Q`, while every trace form is — so such a successor is **not in the
closed-form spectral family at all**, and its failure would say nothing about that family. See
[`METHOD_REVIEW.md`](METHOD_REVIEW.md) §4 and §9.

## 5. The audit trail

Three code-only lock amendments exist, not two. Amendment 1 (17:02:29Z) fixed a release-file write
race; amendment 2 (17:29:09Z) fixed rare load-dependent prediction corruption observed under extreme
machine memory pressure; amendment 3 (18:30:13Z) changed three published tables from CSV to gzip CSV.
All ten final-partition reads occurred between 16:46:33Z and 17:54:45Z under one lock digest, so
amendment 3 post-dates every final read and cannot have touched a score. An independent recheck of
all 17,639 hashed inputs (1.94 GB) with every amendment applied passes with 0 changed, 0 missing and
0 invalid entries.

The gap is in the published record, not the science: the **committed**
`INDEPENDENT_VERIFICATION.json` lists only amendments 1 and 2. The verifier was re-run locally after
amendment 3 and passes, but that re-run was never committed. It should be regenerated and committed,
and the four "two amendments" sentences corrected.

Memory pressure remains the **observed condition** under which the corruption appeared, not a
demonstrated cause. The study says this already and it should stay that way. The repair is sound
independently of the diagnosis: every prediction is now computed twice and must agree, all final
units were rescored, originals are preserved, and the post-fix replay found 0 mismatches in 59,874
predictions with 0 selection changes in 37,122 validation predictions.

## 6. The decision, restated

**Stop developing the fixed-rank residual spectral channel as a candidate mechanism.** That conclusion
is unchanged and this audit strengthens it: the negative verdict survives average-utility matching in
both directions.

The defensible contribution of this study is **not** a method. It is (i) an interface and evaluation
design — fixed published outputs, recipient-specific and coalition-specific audit roles, fresh and
frozen attackers kept separate, absolute and incremental recovery reported separately, negative
increments unclipped — and (ii) a locked temporal transport that shows a coalition-conditioning knob
behaving as designed on a sealed new year while the mechanism it is attached to remains uncompetitive.

The next study that would change a decision must separate the two factors it would change (the
penalty's output-side function class, and the rank rule), must report realised width if it selects
rank by eigenvalue sign, and must not claim a closed-form optimum for an objective that no longer has
one. The 2017 final partition is spent. **2016 has now been admitted on provenance and schema
grounds, with a prospective label-blind partition and a lock-enforcing loader, and without a single
model fit, transformed row, prediction or final-partition label read** — see
[`DATA_2016_ADMISSION.md`](DATA_2016_ADMISSION.md).
