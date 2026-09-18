# CLAIM_CHANGES — what the integrated manuscript changes, and on what evidence

Branch `research/pcrl-manuscript-integrated-v2`. Sources integrated:

| Role | Branch | SHA |
|---|---|---|
| Common baseline | — | `349efa454afd907389760fd1f59fd8806a215efd` |
| Evidence audit + previous manuscript | `research/pcrl-evidence-paper-v1` | `0d8f4b67b6d4961dfa133289d0167c874d2f4794` |
| Nonlinear/rank development study | `research/pcrl-nonlinear-rank-v1` | `c37807e4f568ef38e5528fc09c1506083278bf4d` |

Nothing in `results/redesign_20260917_acs_spectral_transport_v1`, `results/pcrl_evidence_review_v1`
or `results/pcrl_nonlinear_rank_v1` was edited. Those trees are historical artifacts and are
preserved exactly as committed. Every change below is made in the **new** manuscript
(`papers/pcrl_manuscript_v2/`) and in this review directory.

**No stored result, table or interval changed anywhere in this revision.** All but one correction
is to wording, scope, a denominator or a causal reading; the exception is A11, a figure
mis-transcribed into a prose summary, where the evidence file was always right. Each row names the
artifact that settles it.

---

## A. New in this integration

### A1. The F1 sensitive denominator: "8 of 8" → 14 of 16

**Was** (`papers/pcrl_evidence_v1/main.tex` §7.2, `results/pcrl_evidence_review_v1/CLAIM_AUDIT.csv`
row C04, `CORRECTED_RESEARCH_DECISION.md` line 31): *"C1 has a simultaneous-interval advantage over
both spectral_L1 and spectral_L2, under both weightings: 8 of 8 sensitive endpoints better, 0
worse."*

**Evidence.** `results/redesign_20260917_acs_spectral_transport_v1/TRANSPORT_DECISION.json`,
`families.F1_primary.decisions`. The four contrast × weighting cells record
`significantly_better` lists of length 3, 4, 3, 4 over the four sensitive roles:

| cell | strictly better | strictly worse | unresolved |
|---|---|---|---|
| C1 vs L1 / unweighted | 3 of 4 | 0 | `recovery/A/SEX`, `[-0.00585, +0.00020]` |
| C1 vs L1 / person-weighted | 4 of 4 | 0 | — |
| C1 vs L2 / unweighted | 3 of 4 | 0 | `recovery/A/SEX`, `[-0.00568, +0.00052]` |
| C1 vs L2 / person-weighted | 4 of 4 | 0 | — |
| **total** | **14 of 16** | **0 of 16** | **2** |

**Now.** The manuscript reports 14 of 16 strictly better, 0 worse, 2 unresolved, and separately
reports that the **registered** `advantage` rule — "≥1 sensitive endpoint adjusted upper < 0 and no
sensitive endpoint adjusted lower > 0, per weighting" (`COMPARISONS.json`, `rules.advantage`) —
fired in 4 of 4 cells. These are different objects and the paper now names which one it means every
time. Table `denominators.tex` prints both for all four families.

"8 of 8" is recoverable as the endpoint × comparator grid scored as better in *at least one*
weighting. As written — "under both weightings" — it was not supported.

**Why it matters.** It is exactly the substitution the reader should not have to make: a count over
coalition-sex and race cells presented as though every sensitive outcome improved. The unresolved
cells are on **local sex**, and there up to 5 × 10⁻⁴ nats of *extra* recovery is not excluded.

### A2. The nonlinear penalty is not harmless: one cell is significantly worse

**Was** (`results/pcrl_nonlinear_rank_v1/RESEARCH_DECISION.md`, attribution table, "Nonlinear
penalty" row): *"… No endpoint significantly worse; no residence cost."*

**Evidence.** `results/pcrl_nonlinear_rank_v1/PAIRED_INTERVALS.csv`, family
`C6_nonlinear_vs_original_rank8`:

```
spectral_nlr8_L1 - spectral_lin8_L1, recovery/A/RAC1P, unweighted
estimate +0.00862,  adjusted 95% [+0.00168, +0.01556],  significantly_worse = True
```

**Now.** §8.3 and Table `attribution.tex` report, per factor: nonlinear penalty 7 better / 0 worse
of 24 at *r*=16, and **5 better / 1 worse of 24** at *r*=8; rank-8 compression 15 better / 0 worse
of 48; coalition-vs-local 29 better / **2 worse** of 64. The "no residence cost" half survives: 0 of
6, 0 of 6 and 0 of 12 residence cells move significantly in either direction. Every significant
cell is listed in Appendix A.

### A3. The rotation share is not a causal budget partition

**Was** (`RESEARCH_DECISION.md`): *"roughly 37% of the surrogate movement bought statistically
significant reductions in measured recovery, so removing the provably inert 63% should let the same
compute buy more"*, under the heading *"This makes the fix better motivated than a pure negative."*

**Evidence.** `results/pcrl_nonlinear_rank_v1/DIAGNOSTICS_SUMMARY.json`, `seeds[*].rotation`. The
measured quantity is `rotation_only_training_gain / total_training_gain`, where the numerator comes
from a **separate** bounded optimisation over orthogonal transformations of the original solution's
subspace. Its `rotation_stop_reason` is `budget_exhausted` or
`line_search_failed_no_improvement` in every applicable condition — so each value is an *attained*
share under a fixed budget, not a supremum.

**Now.** §8.5 reports the diagnostic as *the share of training gain also attainable by a separately
optimised information-preserving transformation* (mean 0.628, range 0.367–0.907, n = 18), states
that the two searches optimise over different feasible sets so the share does not decompose the
actual trajectory into effective and wasted parts, and removes the prediction that a repaired
objective will do better. What survives is the finding the diagnostic actually supports: the
objective is coordinate-dependent, which is a defect in the construction
(`nonlinear_penalty_change_abs` up to 7.5 × 10⁻³ under a rotation, against
`utility_change_abs` ≤ 3.3 × 10⁻¹⁶).

### A4. The frozen-nuisance observation is about the frozen nuisances

**Was** (`RESEARCH_DECISION.md`): *"The protected residual is therefore close to the marginal
residual, so any H-conditioned penalty has limited room to differ from the marginal one **on this
interface, whatever function class it uses in Z**. That is a property of the interface."*

**Evidence.** `DIAGNOSTICS_SUMMARY.json`, `seeds[*].nuisance_calibration`. Out-of-fold improvement
over the prior loss, across local and coalition roles and all three seeds: sex 0.03–0.66 %, race
0.96–2.35 %. The same cross-fitted multinomial-logistic family reaches 8.1–8.9 % on
`A/public_coverage`.

**Now.** §8.6 keeps the measurement and drops the inference to "a property of the interface". A
restricted nuisance class can miss information and can miss conditional interactions entirely;
weak fitted prediction does not establish small population information in *H*, accurate conditional
means, or general closeness of conditional and marginal protection. The study's own
nuisance-error fixture — a misspecified *m(H)* manufacturing a penalty more than ten times the
oracle value under *exact* conditional independence — is cited in the same paragraph.

### A5. Optimisation checks exclude three failures, not suboptimality

**Was** (`RESEARCH_DECISION.md`, "Controls that came out clean"): *"**Not an optimisation
artifact.** The refinement never returned its initial point in any of the 18 conditions; restart
spread tracks the gain; feasibility max|W'W − I| ≤ 1.4e-15."*

**Now.** §8.7 keeps all three checks and states what they rule out: returning the initialisation,
an incomplete restart schedule, and a feasibility violation. They do not establish that a nonconvex
objective was optimised well. Ky Fan gives a restricted global optimum for the **original fixed
matrix only**; the refinement has no fixed matrix and inherits no optimality statement, so a
suboptimal solution remains a live explanation for its results (§4.3).

### A6. The rank statement rests on the measurement, not on a rank argument

**Was** (`RANK_DIAGNOSTIC.md`, "Why, structurally"): the passage reads as though
`rank(U) ≤ 32` plus positive-semidefinite penalties settles that all sixteen selected eigenvalues
are positive.

**Now.** §8.1 reports the measured counts as the finding (32 positive / 0 negative / 96 within
tolerance for *U*; 32 positive for each of L1, L2 and C1; `nonpositive_among_top_16 = 0` in every
seed and objective) and states explicitly that `rank(U) = 32` alone does **not** logically imply
the sign pattern without an inertia argument whose assumptions were not established. The rank-8 arm
is labelled **compression**, not eigenvalue-sign selection, throughout; sign-based rank selection
is prior work (SARL Thm 3, OptNet-ARL Thm 4.1, K-TOpt Cor 4.1) and made no change here
(`reduced_rank_recipe_is_alias_of_rank_16 = true`).

### A7. Two stale cross-terminal status assertions removed

* `results/pcrl_evidence_review_v1/REVIEW_INDEX.md` records the nonlinear study as *"not
  available … fit phase still running"*. It is complete at `c37807e`, and this manuscript
  integrates it.
* `results/pcrl_nonlinear_rank_v1/RESEARCH_DECISION.md` records that 2016's *"admission has never
  been run"*. It has: `results/pcrl_evidence_review_v1/DATA_2016_ADMISSION.md` and
  `ACS_2016_ADMISSION_MANIFEST.json`. 2016 is **admitted on provenance and schema with a
  prospective label-blind household split prepared, and unscored**. Both files are preserved
  unedited; `REVIEW_INDEX.md` in this directory carries the current status.

### A8. Minor numeric correction

`RESEARCH_DECISION.md` gives the frozen-nuisance improvement as "sex by 0.2–0.7 %". The minimum
across seeds and roles is 0.027 % (seed 2, `A/SEX`). The manuscript states 0.03–0.66 %. No
conclusion depends on the difference.

### A11. A mis-transcribed figure in Study 2's prose summary

**Was** (`results/pcrl_nonlinear_rank_v1/RESEARCH_DECISION.md` line 32, repeated in
`PAPER_ADDENDUM.md` line 154): *"significantly worse than J on `A/SEX` under both weightings
(+0.0106 unweighted, **+0.0098** person-weighted; adjusted intervals exclude zero)"*.

**Evidence.** `results/pcrl_nonlinear_rank_v1/PAIRED_INTERVALS.csv`, family
`C5_candidate_vs_reference`:

```
spectral_nlr8_C1 - J, recovery/A/SEX, unweighted       +0.01065  [+0.00331, +0.01799]  worse
spectral_nlr8_C1 - J, recovery/A/SEX, person_weighted  +0.01210  [+0.00441, +0.01978]  worse
```

No cell anywhere in that file supports `+0.0098` for this contrast. The value `0.0098` does appear
in the same study's development table — it is `spectral_lin16_C1`'s additional `AB/SEX` recovery
(`DEVELOPMENT_2018.md` line 11) — which is the likely source of the slip.

**Now.** §8.4 reports `+0.0106` unweighted and **`+0.0121`** person-weighted, and Table 11, which is
generated directly from the interval file, carried the correct value throughout. This is the one
**numerical** correction in this revision, and it makes the candidate's deficit against J *larger*,
not smaller. Recorded as withdrawn-claims item 17.

The original documents are preserved unedited; this is the correction of record.

### A9. Evidence status made consistent in four places

The abstract, the design section, every figure caption and the limitations now use the same four
labels, and Table 1 prints them: ACS 2018 **development**; ACS 2017 under its original seal
**confirmatory**; ACS 2017 after the seal **exploratory reuse, no intervals by design**; ACS 2016
**admitted, unscored**. Figure 7 shows the two 2017 roles side by side with the distinction in the
panel titles rather than only in the text.

### A10. The residence task described precisely

Residence labels were excluded from every representation fit — that much is structural and is now
asserted in the condition table. The manuscript no longer lets this read as a pristine unseen-task
selection: residence outcomes have been used repeatedly across this line of work as *the* utility
endpoint, so choosing residence as the capability measure is itself a development decision (§6).
Exact output preservation does not preserve future service accuracy under shift (§3, §7.7).

---

## B. Carried forward from the previous revision

These were corrected at `0d8f4b6` and are restated in §9 of the new manuscript so that it is
self-contained. `results/pcrl_evidence_review_v1/ERRATA.md` remains the primary record and is not
superseded.

| # | Correction | Primary evidence |
|---|---|---|
| B1 | "±0.001 band" → one-sided point rule; 0 of 4 comparisons non-inferior at the margin | `EQUIVALENCE_F1.csv` |
| B2 | "resolution, not direction" withdrawn; magnitudes moved 0.054×–14.8× | `DEV_VS_TRANSPORT.csv` |
| B3 | "20/20 development signs" is a seed-mean count; 13 of 20 in every seed | `PER_SEED_DIRECTION.csv` |
| B4 | "C1 vs J fails on local race" → fails on all four sensitive endpoints, both weightings | `INDEPENDENT_FAMILIES.csv` |
| B5 | "several times more" is scale-dependent; quote the paired difference | `MODEB_TRANSPORT_TABLE.csv` |
| B6 | A pass/fail inversion in the frozen source-probe sentence | `SERVICE_VS_PROBE.csv` |
| B7 | "the gap is the surrogate, not the solver" → "consistent with" | `METHOD_REVIEW.md` |
| B8 | Family-level negative inference withdrawn (trace-form argument) | `METHOD_REVIEW.md` §4, §9 |
| B9 | "two lock amendments" → three, the third post-dating every final read | `LOCK_RECHECK.json` |
| B10 | Half-headroom failure reported beside the H-relative residence gain | `SERVICE_VS_PROBE.csv` |
| B11 | Rank-rule misattribution corrected to SARL / OptNet-ARL / K-TOpt | `NOVELTY_MATRIX.md` N3 |
| B12 | OptNet-ARL "Theorem 4" → Theorem 4.1; KCI "Lemma 2(iii)/(v)" → Lemma 2(v) | `NOVELTY_MATRIX.md` N1, N2 |
| B13 | Withholding non-dominance reported in **both** directions | `WITHHOLDING_DOMINANCE.csv` |

---

## C. Claims deliberately **not** changed

* **The F1 pass itself.** The registered rule is reproduced exactly and stands. A1 sharpens the
  denominator; it does not retract the decision.
* **"No directional prediction was registered" (Study 1).** Correct as stated, and worth keeping:
  prespecified contrasts, endpoint families, multiplicity and stopping point still fix the estimand
  before the outcome. A run without a called shot is not thereby exploratory.
* **Study 2's registration disclosure.** Reporting first that three of six registered predictions
  were wrong is the right ordering and is preserved.
* **The numerical-recovery disclosure.** Amendments 1 and 2 are reported honestly with originals
  preserved. Memory pressure remains the *observed condition*, not a demonstrated cause, in both
  studies.
* **The exploratory 2017 table.** Study 2 reports a cross-year result that cuts against its own
  candidate and computes no intervals there. That is kept verbatim in substance.

---

## D. What no correction here can fix

* No external published method was executed, so the paper supports no competitiveness claim against
  the literature, in either direction. See `RELATED_WORK_SCOPE.md`.
* The 2017 seal is spent and 2018 is development. Reanalysis creates no fresh evaluation.
* Attack strength is a floor. No finite attack collection bounds what is recoverable.
* Study 2's evaluated maps carry a known, **unrepaired** coordinate-dependence defect. Its numbers
  are numbers for the defective objective and the manuscript says so where they appear.
