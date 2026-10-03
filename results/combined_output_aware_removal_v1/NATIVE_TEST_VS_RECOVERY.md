# Native tests vs measured recovery

Each method's own test is kept separate from the study's recovery measurement. Three different situations are
distinguished:
- **(i) the implementation fails its own intended test**;
- **(ii) the intended test holds, but recovery happens outside its stated scope**;
- **(iii) a separately available output bypasses the protected features.**

A mismatch between a fairness metric and a privacy metric is not a broken theorem. An invalid or non-estimable
certificate is not proof that a representation is unsafe.

| Method | Native test (what it bounds; on which rows) | Result here | Measured recovery (supported macro AUC, common assessment) | Category |
|---|---|---|---|---|
| PCRL (untreated) | Historical in-sample one-hot ridge R² ≤ 0.05 on the test split (benchmark N0) | HMDA underwriting/race fails on s1 and s2; Adult income/sex passes on all 3 seeds; stable under exposure removal | Adult 0.817, HMDA 0.870 | (i) on HMDA s1/s2; (ii) elsewhere |
| Official target LEACE (B) | Implementation bound: whitened residual cross-covariance ≤ svd_tol on defense_fit (benchmark maps, reused unchanged) | Holds on every map | Adult 0.812, HMDA 0.864 | (ii): linear erasure holds, nonlinear recovery remains |
| Policy LEACE (C) | The same bound for the disallowed set | Holds | Adult 0.803, HMDA 0.863 | (ii) |
| Noise σ\* | None (no native certificate) | — | Adult 0.524, HMDA 0.508 on the features | — |
| FARE\* (official) | Upper bound on the demographic-parity distance of any classifier that sees **only the FARE cell**, with probability ≥ 1 − δ (paper §5, App. D.1), computed on the reserved certification rows | Mostly UNAVAILABLE or vacuous (details below) | Adult 0.550, HMDA 0.501 | Recovery is low **without** a usable certificate |
| FARE\* + clean outputs | Not covered by the certificate: the classifier sees other channels | — | Adult 0.788, HMDA 0.768 | (iii) bypass |

## FARE certificate details

Sources: `run/certificates/AMENDMENT_A1.json`, summarised in `certificates/`. δ = 0.05; certification rows are
Adult 1,500 and HMDA 1,385, split 50/50 into D_val and D_test.

**Original run.** Every certificate is UNAVAILABLE. The wrapper refused because some certification rows' representations are byte-identical to
those of fit rows, although they are distinct records in disjoint roles. The cert rows affected are Adult 8 / 1 / 2 and
HMDA 6 / 1,138 / 6 for seeds 0 / 1 / 2; the HMDA s1 representation is collapsed. The amendment field
`cert_rows_with_feature_vector_equal_to_a_fit_row` counts *distinct vectors*, not rows. Use these row counts instead.

**Amendment A1** (dated, post-run, implementation guard only). The duplicate check uses row identity instead of
feature hashes; the official computation is otherwise unchanged.

| | s0 | s1 | s2 |
|---|---|---|---|
| Adult nominee (DP-distance bound, 1 pair) | 0.497 | UNAVAILABLE (premise: a cell absent from a certificate split) | 0.504 |
| Adult zero-fairness twin | 0.552 | UNAVAILABLE | 0.569 |
| HMDA nominee, all 10 race pairs | UNAVAILABLE (premise) | **3.18 (vacuous: > 1)** | UNAVAILABLE |
| HMDA nominee, race {0, 1, 2} (secondary, declared before the run) | 1.05 (vacuous) | **0.175** | 0.947 |
| HMDA zero-fairness twin, {0, 1, 2} | UNAVAILABLE | UNAVAILABLE | 1.14 (vacuous) |

**Interpretation:**
- With about 1,400–1,500 certification rows, the official certificate is either unavailable (premise failure) or too
  loose to be informative in all but one case: HMDA s1 on race {0, 1, 2}, bound 0.175. That case is the degenerate
  single-cell nominee, for which the true DP distance is 0.
- This is a **sample-size and role limitation of this study**. It does not show that FARE is unsafe, and it does not
  refute FARE's theorem. Where it is available, the certificate is about demographic parity of cell-only classifiers.
  It is not an attack-AUC bound, and it does not cover the view-2 release (cell plus clean outputs). In that release,
  recovery is 0.77–0.79 because of the outputs, not the cells.
- The measured recovery on FARE features alone (0.550 / 0.501) is an empirical, finite-sample result for this
  attacker slate on already-used rows. It is not a certificate.
