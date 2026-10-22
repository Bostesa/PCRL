# Failure map of earlier PCRL release-mechanism families (campaign `pcrl_input_and_pair_release_v1`; coordinator)

Written 2026-10-22 from about 01:38Z from public records only (no label, nothing under `.private/` except the label-free admission
hashes). Rows 1-13 are the dual-design release campaign's map (`results/pcrl_dual_design_release_v1/FAILURE_MAP.md`, published in
e30cf1c6b and pinned by `SOURCE_ADMISSION.json`), which stays the reference for those rows; rows 1-10 are detailed in
`results/pcrl_new_release_mechanisms_v1/FAILURE_MAP.md`. Row 14 is the dual-design release campaign itself. Paired values a / b are
versions 0 / 1.

## Rows 1-13 (inherited, one line each)

| # | Family | Dominant failure |
|---|---|---|
| 1 | Adversarial penalties with refreshed critics (Adult) | Utility cost; "strength is the lever"; leakage moved to income |
| 2 | Leaf-score corrections and safeguards (ACS FL) | Recovery not reduced at a feasible strength; CONT / LOC absorb the effect; line closed |
| 3 | Tree leaf-score privacy training and frozen hybrids (ACS FL; H_LJ, NEURAL_RAW come from here) | Benefit below 0.020 against eligible references; JOINT income costs income utility |
| 4 | Frozen-bank random, sampled and replacement channels | Infeasible bank or recovery not reduced; randomness never evaluated on an attacked real release |
| 5 | Post-hoc shared-cell merging (SCM, IBSI) | Benefit not separated from task spending; frontier at the candidate's spend unmeasured |
| 6 | Calibration and learned decoders | Utility cost; relabelings remove no information |
| 7 | Latent bottlenecks and finite codes | Utility cost; joint codes about equal to sequential ones |
| 8 | Grouped feature selection (Adult) | Infeasible; one-recipient drops vanish in the linked pair |
| 9 | ACS income codes and transplants (B3CTL, H3) | Benefit below 0.020, bought with the whole income Brier budget; reader-sensitive |
| 10 | Task-conditioned representations and small adapters | Benefit below 0.020; local / unconditional / sequential twins reproduce it |
| 11 | Propensity-routed finite cells (FA) and the SEX-targeted coupled Gaussian logit channel (FB2) | SEX targeting not separated from SEX-blind spending of the same form |
| 12 | Propensity-routed finite cells, fitted (PRC) | No reduction beyond any control; a task-score-only grouping (HQ) reproduced it |
| 13 | Solver adequacy of the PRC objective (SV1) | A better optimum of the surrogate is not a better release |

## Row 14 (since that map)

| # | Family (campaign; publication) | Operation | Best measured outcome (MODEL_SELECT registered grid, 16-recipe bank) | Dominant failure | Lesson for controls |
|---|---|---|---|---|---|
| 14 | SEX-standardized posteriors (STD, IPW-STD) and propensity-tilt cancellation (PTC); one idea family (dual-design release; e30cf1c6b) | STD: fixed-weight average over SEX of SEX-aware task posteriors; IPW-STD: inverse-propensity-weighted task models. PTC: log-probabilities tilted against a cross-fitted SEX propensity with covariance-cancelling weights; R1 split strengths 0.35 / 0.65 | Twin DA_TWIN 0.6506 (income 0.5606, occupation 0.6450). DA_STD 0.6331 (+0.0175, se 0.0090 vs the twin; not separated from the SEX-blind spend path DQ). DB_PTC_R1 (adaptive) 0.6267: +0.0239 (0.0101) vs the twin, +0.0245 (0.0122) vs DQ, +0.0265 (0.0139) vs its SEX-independent-attribute control, +0.0128 (0.0081) vs the release-only tilt DB_ZTILT_R1 (which overspent). D2 met in 0 of 4 looks; best D4 values +0.011 over NEURAL_RAW and -0.006 over H_LJ | **Not separated from release-only information**: what can be estimated from the twin's own release reproduced most of the gain; realized se_agg 0.009-0.018 | Every candidate needs a release-only control (the operation driven only by the twin's released vectors) bracketing its spend, beside the SEX-blind spend path and the SEX-independent-attribute path; plan for se_agg about 0.01 |

## Cross-family notes carried into this campaign

1. **Spend explains the large drops** (rows 1, 2, 5, 7, 9, 11, 12). With spend held, benefits were about 0.002-0.018.
2. **The linked pair is strongly super-additive.** TWJ (a strong task-only income output, HC_STRONG, + the JOINT-tree occupation): income view 0.5606,
   occupation view 0.5600, linked pair 0.6476 (dual-design registered grid). H_LJ: 0.5464, 0.5600, 0.6209. Protecting one view
   (or each view separately) left most of the pair leakage in place.
3. **Leakage moves between recipients** (rows 1, 3, 10): per-recipient views and the linked pair are both required.
4. **Readers decide near 0.01** and are data-limited (pair AUC rises with reader FIT data, row 12). Finite releases favour LOOKUP.
5. **Twins must be equally trained** on the same people with the same effort (rows 2, 9).
6. **Release-only information** (row 14): if the twin's released vectors carry what the readers use, a SEX-targeted function of the
   twin's release fitted with SEX labels is the simplest alternative explanation of a targeted gain.
7. **All evidence is development evidence** on adaptively used MODEL_SELECT records; a 0.020 benefit is not confirmable here.

## Untested operations at this campaign's start

Input-side generalization, smoothing or erasure of the six raw fields on ACS FL; training-target or training-row changes other than
SEX standardization and inverse-propensity reweighting; deterministic joint (pair-level) release maps other than linear tilts,
propensity routing and codes; randomized response on decisions; decision- and top-confidence-preserving flattening (partly tested as
CTL_REM, row 11). This campaign's designers chose from these or proposed others (`roles/designer_a/`, `roles/designer_b/`).
