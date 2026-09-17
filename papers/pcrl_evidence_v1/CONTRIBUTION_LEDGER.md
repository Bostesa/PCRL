# Contribution ledger

Every claim in the abstract and Section 1 of `main.tex`, linked to the evidence that supports it.
"Regenerate" commands assume repository root with `PYTHONPATH=.`; `<local>` is the read-only study
directory holding the per-person arrays.

| # | Claim (abstract / §1) | Type | Evidence | Regenerate |
|---|---|---|---|---|
| 1 | The mechanism adds a channel for one recipient "without changing a single published number" | structural, verified | 210/210 released 2017 views bitwise equal; `source_output_identity` check | `INDEPENDENT_VERIFICATION.json` (study) |
| 2 | "Exact service preservation … does not imply stable service accuracy, which moves by ±0.014 nats across one year" | measured | employment +0.0138, income −0.0145, coverage −0.0067 | `tradeoff_tables` → `SERVICE_QUALITY_SUMMARY.csv` |
| 3 | "it lowers additional sensitive recovery against both an equal-strength and an equal-total-mass local control, under simultaneous 95% intervals and both weightings" | registered-rule outcome, reproduced | F1: 8/8 sensitive endpoints adjusted-high < 0, 0 with adjusted-low > 0, c = 2.930106 | `reanalyse_transport` → `INDEPENDENT_FAMILIES.csv`, `FAMILY_AGREEMENT.json` |
| 4 | "satisfying a rule the development year had failed" | historical comparison | development C1−L2 AB/SEX −0.0003, AB/race −0.0006, not resolved on a 2,982-person pool | study `DEVELOPMENT_RESULTS.md`; `DEV_VS_TRANSPORT.csv` |
| 5 | "That rule is a one-sided constraint on a *point* estimate" | code reading | `residence_difference <= .001 + 1e-12` on the seed-mean estimate, `scripts/report_acs_spectral_transport.py:456` | read the source |
| 6 | "the interval evidence does not establish non-inferiority at the same 0.001 margin — zero of four comparisons, under any procedure we computed" | new analysis | pointwise one-sided 95% upper bounds 0.00112 / 0.00103 / 0.00135 / 0.00102; simultaneous 0.00166 / 0.00164 / 0.00182 / 0.00162 | `reanalyse_transport` → `EQUIVALENCE_F1.csv` |
| 7 | "significantly worse than a frozen adversarially trained channel of the same width on all four sensitive endpoints under both weightings" | registered-rule outcome, reproduced | F3: 8/8 sensitive rows adjusted-low > 0, c = 2.728486 | `INDEPENDENT_FAMILIES.csv`, family `F3_secondary` |
| 8 | "this ordering survives matching the two channels on average utility, in both directions" | new analysis, exploratory | J needs p\* = 1.493 to match C1 (not identifiable); C1 at p\* = 0.670 still leaks 0.0303 vs J's 0.0076 on A/race | `tradeoff_tables` → `WITHHOLDING_MATCHING.csv` |
| 9 | "the negative verdict is robust to matching the two on average utility" (§1) | same as 8 | same | same |
| 10 | "No number changed: all 90 pre-declared endpoints reproduce under an independently written scorer to 1.1e-15" | verification | max abs diff: estimate 1.11e-15, se 4.42e-17, unadjusted 2.40e-15, adjusted 1.73e-15 | `FAMILY_AGREEMENT.json` |
| 11 | "The useful output is the problem formulation and the evaluation design" (§1) | assessment, not measurement | argued in `CONTRIBUTION_ASSESSMENT.md` against the verified `NOVELTY_MATRIX.md` | — |
| 12 | "Selecting rank by eigenvalue sign … is SARL's Theorem 3, OptNet-ARL's Theorem 4.1 and K-TOpt's Corollary 4.1" (§4.1) | primary-source verified | full texts read; theorem statements quoted in `NOVELTY_MATRIX.md` | — |
| 13 | "it is Lemma 2(v) of Zhang et al." (§4.1) | primary-source verified + derivation | Lemma 2 located verbatim with the Daudin attribution; equivalence derived in `NOVELTY_MATRIX.md` N2 | — |
| 14 | "a penalty acting on nonlinear functions of Z is provably not a trace form" (§8) | proved by fixture | a trace form is invariant under `W → WQ`; the squared-output penalty is not | `pytest tests/pcrl_evidence_review_v1 -q` |
| 15 | "the half-headroom reference … passes in only 0–2 of 3 seeds" (§7.1) | measured | `half_headroom_pass_seeds` per interface and mode | study `evidence/CRITERIA_SUMMARY.csv`; `SERVICE_VS_PROBE.csv` |
| 16 | "13 of 20 agree in every seed" (§7.8) | measured | per-seed sign agreement | `PER_SEED_DIRECTION.csv` |
| 17 | "point estimates changed by factors between 0.054× and 14.8×" (§7.8) | measured | magnitude ratios over F1 | `DEV_VS_TRANSPORT.csv` |
| 18 | "Three code-only lock amendments … the third post-dates every final read" (§9) | verified | amendments at 17:02:29Z / 17:29:09Z / 18:30:13Z; final reads 16:46:33–17:54:45Z; 17,639 inputs re-verified, 0 changed | `recheck_lock` → `LOCK_RECHECK.json` |

## Claims deliberately **not** made

* No certified bound on `I(S;Z|H)` or on any information quantity.
* No claim that any attack family is strong, only that it is what was tested.
* No priority claim; §5 states that a bounded search cannot establish novelty.
* No claim that 2017 respondents are distinct people from 2018 respondents.
* No claim about retraining variability or Census design variance.
* No family-level inference about closed-form spectral methods (withdrawn; §8 item 8).
* No acceptance probability or readiness score anywhere in this package.
