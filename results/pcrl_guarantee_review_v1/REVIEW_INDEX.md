# Guarantee review — index

**Role:** claims foundation and guarantee verification (assignment 2 under the new numbering; formerly
"Terminal 3").

**Branch:** `research/pcrl-guarantee-review-v1`, based on `33124f965861ccfcaa710aff4a56880c5ab3957e`.

**Handoffs:** `<common git dir>/pcrl_week_finish_v2/claims/`.

## Answers

- **Strongest valid guarantee** (FULL_VIEW_GUARANTEE_REVIEW, Theorem 1). For a fixed public kernel with
  radius κ (equal to its capacity), I(S;Z|W) ≤ κ and I(Y;Z|W) ≤ κ for every joint law and every view W
  that determines the kernel's public state. That includes both H_A and H_AB.
  - An attribute-specific alternative is also valid: I(S;Z|W) ≤ η_TV·log|S|.
  - Both are established mathematics (Gallager–Ryabko, Csiszár, van Erven–Harremoës Thm 34;
    Polyanskiy–Wu Thm 1 and eq. 17), not PCRL theorems.
- **Actual assumptions.**
  - The code and kernel are fixed, and the release uses fresh randomness.
  - Support conditions hold.
  - The kernel may depend on b(H_A) but not on H_B.
  - The bound is about information and Bayes-optimal loss. It is not about the registered +0.001 clause
    or any finite attacker's increment, and not about training data.
- **Useful at the proposed utility level?**
  - A release that adds Q's measured 2016 task gain (~0.026 nats over H) needs κ ≥ 0.026. That gives a
    0.026-nat guarantee for every attribute and view, which is informative against log 2 and log 9.
  - Whether such a kernel can realise that gain is untested.
  - The evaluated D17 has a vacuous κ (log 17). Q's κ is unknown because its kernel is archived
    privately; entry point `kernel_guarantees.py`.
  - Attribute-specific certificates at the level of full H need an extra assumption: within-bin
    homogeneity T ⊥ H | (S,B), or H-smoothness. The actual code T0 violates homogeneity by construction.
- **Remaining novelty gap** (PRIOR_ART_DELTA). The only non-covered increment is an attribute-specific
  certificate conditional on a continuous third-party view, under a stated assumption, with an exact
  oracle and a properly calibrated envelope. That is an adaptation of Diaz et al.'s uniform mechanisms and
  of RLDP. It is valuable only if it beats a matched-κ radius kernel on utility at equal certificate.
- **Prospective 2016** (PROSPECTIVE_INTERPRETATION). Both registered decisions reproduce, with 8/10 for
  each release. Three wording repairs for the manuscript are in `tex/prospective_fragments.tex`. The main
  one, P-1, is a real error: the sensitive clauses cap recovery in excess of J's, not added recovery.

## Files

| File | Content |
|---|---|
| PROSPECTIVE_INTERPRETATION.md, PROSPECTIVE_RECOMPUTATION.json | the five distinctions, reproduction, manuscript repairs P-1 to P-3 |
| MATH_REVIEW.md | early handoff to the method owner (M1–M6) |
| FULL_VIEW_GUARANTEE_REVIEW.md | Theorem 1, Corollary 1, Propositions 2–4, coverage table |
| PRIOR_ART_DELTA.md | Taylor; Diaz; RLDP; van Erven–Harremoës; Polyanskiy–Wu; the smallest distinction |
| COMPOSITION_LEMMA.tex | clean lemma (i)–(iv), proof, tightness, mapping to the implemented statistic; compiles |
| AMENDMENT_2026-09-23_ridge_premise.md | corrects the claims audit's over-cautious T3-12 |
| MAIN_REPAIR_REVIEW.md, 0001-followup-on-fix-retire-accuracy-guarantee.patch | review of 5d4eda04 and follow-up patch (R1, R2, R4) |
| REVIEWED_CLAIMS_MATRIX.csv | 26 rows: 6 proved, 10 refuted, 1 conditional, 7 empirically assessed, 2 unverified |
| FULLVIEW_FIXTURE_OUTPUTS.json, COMPOSITION_FIXTURE_OUTPUTS.json | fixture outputs |
| tex/prospective_fragments.tex | manuscript fragments |

Code: `analysis/pcrl_guarantee_review_v1/{verify_prospective,fullview_fixtures,composition_fixtures,kernel_guarantees}.py`.
Tests: `tests/pcrl_guarantee_review_v1/` (10 tests; with the claims tests, 20 pass).

## Incremental review of the method (entry point)

No method branch existed at this writing. When one lands:

1. Pin its SHA. Read its probability model, and list for each variable {S, Y, H_A, H_B, B, T, Z} who
   observes it and whether it is fixed or optimised.
2. **Kernel guarantee.** Export each fitted kernel (per public state) as
   `{"Q": ...}` or `{"states": {...}}`, then run
   `python analysis/pcrl_guarantee_review_v1/kernel_guarantees.py kernel.json`. Report κ and η_TV beside
   every certified number.
3. **Oracle.** If it samples scenarios, it is a lower bound (MATH_REVIEW M5.2). Require vertex
   enumeration or an upper bound. Test it against `fullview_fixtures.g5_envelope` (exact 0.0502 at TV
   radius 0.05).
4. **Radius.** Identify which level it covers (bin or H), and the assumption that transfers it to H.
   Check `g5c` and `g8` against the method's code construction.
5. **Calibration.** It must account for households, weights, support gaps, the learned code and
   selection.
6. **Compare** the certified value with log|S| at the utility level claimed, and with a matched-κ kernel.
7. **Local encoder.** Check that no H_B enters it.
