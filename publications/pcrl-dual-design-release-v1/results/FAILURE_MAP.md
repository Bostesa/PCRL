# Failure map of earlier PCRL release-mechanism families (campaign `pcrl_dual_design_release_v1`; coordinator)

Written 2026-10-21 16:57Z from public records only (no label, nothing under `.private/` except the label-free admission hashes).
This extends the challenger's map of the new-release-mechanisms campaign
(`results/pcrl_new_release_mechanisms_v1/FAILURE_MAP.md`, rows 1-10, published in 682068d52 and pinned by `SOURCE_ADMISSION.json`),
which stays the reference for those rows. Rows 11-13 are the three campaigns since then. Paired values a / b are versions 0 / 1.

## Rows 1-10 (inherited, one line each; numbers and sources in the inherited map)

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

## Rows 11-13 (since that map)

| # | Family (campaign; publication) | Operation | Best measured outcome (MODEL_SELECT registered grid, 16-recipe bank) | Dominant failure | Lesson for controls |
|---|---|---|---|---|---|
| 11 | Propensity-routed finite cells (FA) and post-hoc SEX-targeted Gaussian logit noise (FB2) (new release mechanisms; 682068d52) | FA: routing to (income cell, occupation cell) pairs trading task cross-entropy against the cells' plug-in SEX information. FB2: one N(0,1) scalar per person along a person-specific Fisher-regularized SEX direction in logit space, on the strongest task-only base (FB2_BASE, total NLL -0.0245 / -0.0212 vs U0) | FA closed at its capacity gate (twin spend over the continuous teachers +0.0107 / +0.0136 nats). FB2_SEX_S5 pair 0.6496 vs its base 0.6605 (+0.0120, se 0.0073); vs the same-strength SEX-blind channel +0.0019 (se 0.0089). Full collapse of the occupation remainder (CTL_REM_1p00): occupation view 0.643 -> 0.560 but occupation NLL 1.8-2.2x the cap and the men's guard 2.7-3.0x | **SEX targeting not separated from SEX-blind spending** of the same form (F1). Randomness bought nothing beyond its spend | Every candidate needs a same-form SEX-blind control at the same strength and a spend path that brackets it |
| 12 | Propensity-routed finite cells, fitted (finite-routing comparison; 04ac9161f) | PRC objective with the repaired B1 teacher-mean tables on 16 x 64 cells; strength 8 (one protected partition per version) | Twin 0.6555 (income 0.5681, occupation 0.6379); PRC_S2 0.6485 (+0.0070, se 0.0040); task-score-only grouping HQ_S2 0.6489; untargeted moves 0.6516; SEX-blind coarsening ORD_S2 0.6538, ORD_S3 0.6474. 0.018-0.023 above NEURAL_RAW (0.6309) and H_LJ (0.6252) | **No reduction beyond any control**: the propensity added nothing over task-score-only grouping at equal spend; finite releases leak more than continuous ones to table readers (twin 0.6555 vs U0 0.6497) | A control that sees only task scores (HQ) can explain a routing effect; ER1-type grid validity needs effective reader seeds |
| 13 | Solver adequacy of the PRC objective (finite-solver adequacy; ddc231a97) | Exact tilt line search (SV1) for the same objective | SV1 lowers the fitted objective in 12 / 12 cells but pooled nested held-out pair LOOKUP -0.0028 (2-SE -0.0065..+0.0009): not resolved | **A better optimum of the surrogate is not a better release** | Optimizing a plug-in dependence objective does not transfer to held-out readers |

## Cross-family notes carried into this campaign

1. **Spend explains the large drops** (rows 1, 2, 5, 7, 9, 11, 12). With spend held, benefits were about 0.002-0.018.
2. **The linked pair is super-additive.** H_LJ: income view 0.5504, occupation view 0.5727, linked pair 0.6252; the finite twin:
   0.5681, 0.6379, 0.6555. Every family whose per-view recovery fell left most of the pair leakage in place.
3. **Leakage moves between recipients** (rows 1, 3, 10): per-recipient views and the linked pair are both required.
4. **Readers decide near 0.01** and are data-limited (pair AUC rises with reader FIT data, row 12). Finite releases favour LOOKUP.
5. **Twins must be equally trained** on the same people with the same effort (rows 2, 9).
6. **All evidence is development evidence** on adaptively used MODEL_SELECT records; a 0.020 benefit is not confirmable here.

## Untested operations named by the inherited map (status at this campaign's start)

Randomized response on decisions; coalition-aware correlated randomness across recipients; input-side generalization or erasure of
the six raw fields on ACS FL; training-time SEX-balanced reweighting or resampling; the MMP mixture over SEX values (data-free only);
a continuous mixture of two task-only models (no coarsening); decision- and top-confidence-preserving flattening (partly tested as
CTL_REM, row 11). This campaign's designers chose from these or proposed others (`roles/designer_a/`, `roles/designer_b/`).
