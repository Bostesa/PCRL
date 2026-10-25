# DR_DRF_R2: revision 2 of the direct-recovery finite release (family DR, coordinator)

Campaign `pcrl_direct_recovery_frontier_v1` (DRF1). Written 2026-10-25 from 03:58Z (`date -u`), AFTER the MODEL_SELECT outcomes of the
R1 reading (`claims/DRF_R1/DECISIONS.json`): this revision is ADAPTIVE as a whole (RULES.md 7) and is never a registered reading; the
headline stays the registered R1 reading. It does not re-read MODEL_SELECT to choose any parameter. Governs units `DRF_FIT_R2_*`
(`python3 -m drf.dr fit --revision R2 ...`). Everything not changed below is exactly card DR_DRF_R1.

## Prospective note (observed failure, changed operation, trigger)

- **Trigger: MODEL_SELECT** (the R1 reading) together with MODEL_FIT search records.
- **Observed failure:** the objective comparison is not separated (DR_AUC_R1 vs DR_MI_R1 pair benefit +0.0076, se 0.0117; version
  means -0.0004 / +0.0156), and the R1 search is far from converged on MODEL_FIT: the best objective of the four seeds spreads by
  0.022 / 0.016 in J_auc (DR_AUC_R1, versions 0 / 1) and 0.0025 / 0.0044 in J_mi (DR_MI_R1), so the tie may be partly an optimizer
  artefact rather than an objective result (the brief's optimizer-defect / failed-objective distinction, on real data).
- **Changed operation (search budget only, identical for both objectives):** 12 seeds x (1 + 3 perturbation restarts), at most 30
  sweeps, 16 moves per perturbation, seeds 1000 v + 10 ... + 21 (full fit) and 1000 v + 100 (fold + 1) + 10 ... + 21 (nested folds):
  a strict superset of R1's four seeds, so each R2 solution's MODEL_FIT objective is at most R1's under the same estimator and data.
- **Not changed:** the expected-cost constraint estimator, the frozen budgets, the exact-AUC objective under the estimated (pi) law,
  the MI control objective, the fine cells, capacity, B1 emission, deployment, the nested estimator, the CTRL path and bracket rule.

## Configurations and accounting

DR_AUC_R2 (candidate), DR_MI_R2 (control), DR_CTRL_LO_R2, DR_CTRL_HI_R2 (controls re-saved for R2's nested spend; a point equal to an
R1 point is still saved under its R2 id and counted): at most 8 specifications, 20 + 8 = 28 of 32.

## Predictions (written after the R1 MODEL_SELECT outcomes, so they are adaptive)

- MODEL_FIT: R2 lowers J_auc below R1 in both versions (certain by construction for the full fit) by 0.005-0.02; DR_MI_R2 likewise
  for J_mi; the in-sample estimated-law pair AUC* changes by at most 0.005.
- MODEL_SELECT: both releases keep G1 (probability 0.75); the DR_AUC vs DR_MI pair difference stays unresolved (probability 0.75);
  no change of D3 or D4.
