# CORRECTED_RESEARCH_DECISION — the conclusion both studies jointly support

Supersedes, as a *reading*, the conclusions of
`results/redesign_20260917_acs_spectral_transport_v1/RESEARCH_DECISION.md` (Study 1) and
`results/pcrl_nonlinear_rank_v1/RESEARCH_DECISION.md` (Study 2). **Both originals are preserved
unedited** and remain the record of what each study concluded at the time. Nothing numerical
changed; the corrections are listed in [`CLAIM_CHANGES.md`](CLAIM_CHANGES.md) and the prior
study-level errata remain at `results/pcrl_evidence_review_v1/ERRATA.md`.

## The decision

**Do not run a confirmation on any current design, and do not spend ACS 2016.** Neither the
original fixed-matrix mechanism nor its nonlinear/rank refinement reaches a state worth a
confirmation year, and the refinement carries a known, unrepaired defect in its objective.

**Publish the work as an empirical and problem-formulation paper**, carrying both the positive
coalition result and both negative method results. The formulation and the evaluation design are
the contribution; there is no competitive algorithm here and the paper should not be titled or
abstracted as though there were.

## What is established

1. **Exact preservation of an already-published service is structural** and holds bitwise on 210 of
   210 released views. It is a property of concatenation, not a theorem, and it does not preserve
   service *accuracy* under a year shift (employment +0.0138 nats, income −0.0145, coverage
   −0.0067).
2. **Coalition conditioning does what it claims against its own local controls, on a sealed year.**
   14 of 16 F1 sensitive cells are strictly better under simultaneous 95 % intervals, 0 worse, 2
   unresolved (local sex, unweighted). The registered `advantage` rule fired in 4 of 4 contrast ×
   weighting cells and the registered primary coordination decision was recorded as supported.
   This is a prespecified-rule confirmation on evidence that was sealed until the lock — **the one
   confirmatory result in this line of work.** It stands, and the later negative results do not
   retroactively erase it.
3. **The registered residence criterion is a one-sided point rule.** It passed. The interval
   evidence does **not** establish non-inferiority at the same 0.001 margin: 0 of 4 F1 comparisons,
   under any procedure computed. "No statistically detected cost" is not "a cost bounded below
   0.001".
4. **Eigenvalue-sign rank selection changes nothing on this interface.** `r_plus = 16` in every seed
   and for every original objective, with no nonpositive direction in the top sixteen. The rank-8
   arm is **compression**, not sign selection. The rank rule itself is prior work, three times over.
5. **The refined objective is coordinate-dependent.** Utility and the original penalty are
   invariant under `W → WQ`; the refined penalty changes by up to 7.5e-3 under the same rotation.
   That is a defect in the construction, localised exactly and reproduced on a synthetic fixture.

## What is not established, and must not be claimed

* **Any competitiveness against the literature.** No external method was run. `J` is an internal
  adversarially trained channel, not an external benchmark; where `J` wins, that is evidence about
  `J`. Equally, `J`'s position is not an upper bound on attainable utility at a privacy constraint.
* **Any Pareto verdict.** `J` carries materially less residence capability than every spectral arm
  and materially less measured recovery. Neither dominates. The correct statement is which
  registered decision rules failed: Study 1's F3, and all eight of Study 2's coordination cells.
* **Anything about a method family.** Two original arms and twelve refined ones failing is evidence
  about fourteen arms. Spectral, kernel, nonlinear and fixed-service approaches are not ruled out —
  and the refined objective is not a trace form, so it was never in the closed-form family anyway.
* **Any privacy guarantee.** No certificate, no conditional independence, no mutual-information
  bound. Vanishing fitted finite moments imply none of these, and every recovery number is a floor
  set by the tested attack families.
* **That the optimiser wasted a measurable fraction of its work.** The rotation diagnostic reports
  a share *also attainable* by a separately optimised invariant transformation, under a bounded
  search that stopped on budget or a failed line search. It does not decompose the actual
  trajectory, and it predicts nothing about a repaired objective.
* **That `H` carries little sensitive information.** The frozen nuisances barely beat the prior on
  sex and race (0.03–0.66 %, 0.96–2.35 %), while the same family reaches 8.1–8.9 % on an authorised
  attribute. That is a diagnostic of those fitted models, not a property of the interface.
* **That residence is a pristine unseen task.** Residence labels were excluded from every
  representation fit — that is structural. But residence outcomes have been used repeatedly across
  this line of work as the utility endpoint, so its selection is itself a development decision.

## The two negative results, stated once each

* **Study 1, registered secondary comparison F3: not supported.** C1 is significantly worse than J
  on all four sensitive endpoints under both weightings, and on both source probes. The ordering
  survives post-hoc average-utility matching in both directions, and four of six comparators cannot
  reach C1's utility at any withholding schedule.
* **Study 2, all eight registered coordination decisions: not supported.** The refinement moved
  measured recovery substantially in the intended direction — composed, the two factors take
  additional A/race recovery from 0.0274 to 0.0055 against J's 0.0071 while holding residence gain
  at 0.0259 against J's 0.0215 — but the best candidate is significantly worse than J on local sex
  under both weightings (+0.0106 unweighted, +0.0121 person-weighted), and the exploratory
  cross-year look makes the gap wider, not narrower.

## Data-pool decisions

* **ACS 2018** is development and has been used repeatedly. It cannot confirm anything.
* **ACS 2017 under its original seal** is spent. Its historical status is preserved; Study 2's
  later reuse of the same rows is exploratory development evidence with no intervals, and never
  restates or overwrites the frozen result.
* **ACS 2016** is admitted on documented provenance and schema checks, with a prospective
  label-blind household split prepared, and is **unscored**. It was not touched by either study or
  by this integration. Spending it would require its own prospective protocol and its own
  registered directional predictions, and must not be started on the strength of a development
  result. Its income estimand differs again, which any future evaluation must handle.

## If work continues

The next step specified by Study 2 is to repair the coordinate-dependence defect — the closed-form
`√2` off-diagonal weighting restores exact invariance to < 1e-12 — and to verify it through an
**outcome-free acceptance gate** requiring the measured rotation share below 0.10 *before* any
outcome may be read. If the gate fails, the construction is still misspecified and that is the
reportable result. A gate pass is not a reason to expect the mechanism to become competitive, and
it is not by itself a reason to spend 2016.

The larger gap is external baselines. `RELATED_WORK_SCOPE.md` §3 specifies them, with two
corrections that matter: linear-erasure baselines may act only on the auxiliary channel if `H` is
immutable, and no baseline may consume residence labels if it is to be compared on transfer to
residence. Until those are run, this work has no method-level control and should not claim one.
