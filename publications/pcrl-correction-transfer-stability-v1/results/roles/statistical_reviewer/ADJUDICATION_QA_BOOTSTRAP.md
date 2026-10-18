# Adjudication: the question-A bootstrap scheme (verifier finding of 20:29Z, ATTENTION)

Statistical reviewer, 2026-10-18 about 20:35Z.
- This adjudication is independent of the verifier and the coordinator.
- I read no real data. My own check below uses invented data only.
- **The error is in code I wrote** (`cts.rules.prediction_comparison`). My invented-data test checked the sign of D and of its 5%
  quantile, but not how the bootstrap resamples households, so it did not catch the error.

## 1. Which scheme does the registered text define?

**One household resample per set per replicate, shared by every statistic and every strength.** RULES.md §6 and RULES.json
(`rule_evaluation.prediction_comparison.uncertainty`) both read:
- "paired household bootstrap";
- "households resampled independently within FIT (OOF people), GUARD and CHECK";
- "D recomputed on each [replicate]".

**Why this reading.**
- A replicate is one resampled data set per set. "Independently" refers to the three sets (FIT, GUARD and CHECK are resampled
  separately), not to the statistics.
- D is one function of the households in the three sets. Its bootstrap distribution must come from resampling those households
  jointly.
- Drawing a fresh resample for each statistic is not a bootstrap of the data. Within one replicate it even makes the overall Brier
  mean disagree with the men's and women's means it is built from.

**How the implementation departs.** It calls `household_bootstrap_means` once per (set, statistic), and each call draws a new
multinomial resample. Strengths share a resample (they are columns of one call), and so does the CHECK resample used for the GUARD
and OOF errors of the same statistic. The five statistics do not.

They are positively correlated on the same households: Brier, NLL and accuracy loss move together, and the SEX-group Brier is part of
the overall Brier. Independent resamples therefore average away correlated noise and **understate Var(D)**, which makes the 5%
quantile anti-conservative.

**Independent invented-data check** (scratchpad; correlated statistics with household effects; B = 1000; 4 invented data sets):
- D is identical under both schemes.
- The bootstrap SE under the registered scheme is 1.04-1.19 times the per-statistic SE.
- The 5% quantile moves down by 0.001-0.012.

This agrees in direction and size with the verifier's 7-25%.

## 2. Correction record (the implementation brought into line with the registered text; no new criterion)

**Original result.** The labels computed with the per-statistic scheme, as recorded. On the D partitions only one label depends on the
scheme: D0 REFIT_CONT ens OOF is recorded as `PREDICTION_IMPROVED` with q05 = +0.0096.

**Changed implementation.**
- The procedure is otherwise unchanged: the same D, the same MAE_rel, B = 2000, seed tag `CTS1_transfer_boot`, and the same decision
  rule (`PREDICTION_IMPROVED` iff D > 0 and q05 > 0).
- The random generator is seeded from the tag. For each set, in the order OOF, GUARD, CHECK, draw **one** household multinomial
  matrix W_set (B × G_set).
- For every statistic k (with its SEX mask) and all 10 strengths: replicate means = (W_set @ S_k) / (W_set @ c_k), where S_k holds the
  per-household sums of the masked rows and c_k their counts.
- The draw order is fixed, so the primary run is deterministic.

**Reason.** The code departed from the registered text, and the departure understates Var(D) (verifier: SEs 7-25% larger under the
registered scheme; my invented-data check: 4-19%).

**Primary labels.** The primary labels are those from the corrected implementation, run once with the registered tag and B = 2000,
on D0, D1 and R0, R1.
- The verifier's 5-seed results are supporting evidence of robustness. In 5 of 5 seeds, D0 REFIT_CONT ens OOF has q05 between -0.065
  and -0.007, which gives `NOT_IMPROVED`.
- If the registered-tag run were to disagree with those 5 seeds, it would still be the primary label and the disagreement would be
  reported. No seed choice is allowed.
- The original per-statistic labels stay in the record, marked **superseded (implementation departed from RULES.md §6)**, beside the
  primary labels.

**Code and test.**
- The fix is about 10 lines in `prediction_comparison`, plus a test that two identical statistic columns give identical replicates,
  i.e. one shared resample.
- I have **not** edited `cts/rules.py`. Its hash is recorded by the screen study and the pre-outcome record, and other units may be
  importing it. The coordinator should apply the fix as its own recorded correction commit; I can write it on request.

## 3. Effect on the claim ladder

**No change.** Rung 1 (`IMPROVED_COST_PREDICTION`) is **NOT_MET under both schemes**, for two independent reasons:
1. The outcome is `NO_RECIPE`, so no recipe was selected and the rung has nothing to apply to.
2. No direction arm (PAIR or LOC) is `PREDICTION_IMPROVED` on D0 under either scheme. Rung 1 needs D0, D1, R0 and R1.

CONT is a control and never a recipe.

**The descriptive question-A statement changes.**
- Under the registered scheme, `PREDICTION_IMPROVED` appears **only on D1** (FROZEN_PAIR ens; REFIT_CONT full and ens), never on D0.
- **No direction improved cost prediction on both diagnostic partitions.**
- Any sentence implying a D0 improvement (from REFIT_CONT ens) must be withdrawn.
- The D1 FROZEN_PAIR label also carries `A1_NOT_SUPPORTED`. FROZEN routing saw the GUARD and CHECK labels, so a GUARD-to-CHECK
  improvement there can reflect behaviour shared in-sample.

**R labels.** The corrected implementation gives the primary R labels, and any label that depends on the scheme is reported beside
them. R labels cannot create a claim, because rung 1 needs a selected recipe.
