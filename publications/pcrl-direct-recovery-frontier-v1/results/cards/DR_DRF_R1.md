# DR_DRF_R1: revision 1 of the direct-recovery finite release (family DR, coordinator)

Campaign `pcrl_direct_recovery_frontier_v1` (DRF1). Written 2026-10-25 from 03:24Z (`date -u`), after the MODEL_FIT fit units of card
DR_DRF_v1 (DRF_FIT_V0_V1, DRF_FIT_V1_V1; records `dr/FIT_V0.json`, `dr/FIT_V1.json`) and BEFORE any MODEL_SELECT read of this campaign
(no utility, paired cost or reader output of any release has been computed). Governs units `DRF_FIT_R1_*` (`python3 -m drf.dr fit
--revision R1 ...`). Everything not changed below is exactly card DR_DRF_v1 (representation, emission, objectives, law, budgets,
capacity, controls, nested estimator, specifications rule).

## Prospective note (RULES.md 7: observed failure, changed operation, trigger)

- **Trigger: MODEL_FIT** (the nested held-out estimator of card DR_DRF_v1; no MODEL_SELECT outcome).
- **Observed failure 1: the in-sample constraints do not transfer.** DR_AUC satisfies every budget in-sample on MODEL_FIT (occupation
  NLL spend 0.0097 / 0.0099 of 0.0097 / 0.0100; men's occupation Brier 0.0035 / 0.0027 of 0.0035 / 0.0029), but its nested held-out
  spends are about three times larger: occupation NLL +0.028 (SE 0.0016-0.0017), men's Brier +0.012 (0.001), occupation accuracy
  +0.010, total NLL +0.039 / +0.040 (versions 0 / 1). DR_MI shows the same (+0.030 / +0.028 occupation NLL). The SEX-blind
  coarsening path does not: its in-sample and nested spends agree within about 20% (step 66: 0.032 in-sample, 0.037 nested). The
  search evaluates about 1.4-2.0 million candidate moves against constraints computed from the realized MODEL_FIT labels of small
  fine cells, so it selects moves whose realized cost is low by chance (selection on label noise). On MODEL_SELECT this would break
  the original occupation NLL and men's-guard caps by a wide margin (twin -0.0071 + about 0.028 against the 0.010 cap).
- **Observed failure 2: an optimizer shortfall in version 1.** DR_AUC's own MODEL_FIT objective (J_auc 0.2319) is above the J_auc of
  the DR_MI solution (0.2167) in version 1 (version 0: 0.2086 vs 0.2226), so the AUC search did not reach a point the MI search found.
- **Changed operation 1 (the constraint estimator):** every task cost inside the search, for DR_AUC, DR_MI and the SEX-blind
  coarsening path alike, is the EXPECTED cost under the frozen out-of-fold teachers instead of the realized labels: income costs
  with the NRM1 income teacher's out-of-fold posterior, occupation costs with the occupation teacher's, the men's / women's
  occupation Brier with the FRC1 SEX-conditional occupation teacher's out-of-fold posterior for the person's own SEX group (`occ_s`
  in the pinned SEXSIDE cache). The cost formulas are linear in the label counts, so the soft counts give exact teacher-expected
  costs (`drf.solver.Problem(soft=...)`). No realized task label enters the search; the realized labels and the true SEX enter only
  the nested held-out evaluation and the true SEX selects the within-SEX group. The budgets are the frozen `dr/BUDGETS.json` values.
- **Changed operation 2 (the search budget, identical for both objectives):** 4 seeds x (1 + 3 perturbation restarts), at most 30
  sweeps, 16 moves per perturbation (was 2 x (1 + 2)); seeds 1000 v + 10 and up (full fit), 1000 v + 100 (fold + 1) + 10 and up
  (nested folds).
- **Not changed:** the exact-AUC objective under the estimated (pi) law, the fine cells, capacity, B1 emission, the deployment
  function, the nested estimator (realized held-out labels), the CTRL bracket rule on nested realized spend, the budgets.

## Configurations and accounting

DR_AUC_R1 (candidate), DR_MI_R1 (control), DR_CTRL_LO_R1 and DR_CTRL_HI_R1 (controls; a bracket point at the origin is the twin and
costs none): at most 8 new version-specific specifications, 12 + 8 = 20 of 32. The card DR_DRF_v1 releases stay saved and are
evaluated as registered (their own reading); OPT_L and OPT_S run as registered by DR_DRF_v1 (realized labels in-sample: the optimistic
construction).

## Registered predictions (before any MODEL_SELECT read)

- The expected-cost search is more constrained: DR_AUC_R1's MODEL_FIT exact pair AUC* (estimated law) lies above DR_AUC's (0.6089 /
  0.6209), and its nested held-out spends lie within about 1.5 x its budgets (probability 0.6).
- P(DR_AUC_R1 meets all 28 conditions on MODEL_SELECT in both versions) 0.5; P(D2 met) 0.08; P(D3) 0.02; P(D4) 0.01.
- If the nested spends still exceed the budgets by more than 1.5 x, the teachers' calibration inside merged atoms (not label noise)
  limits constraint transfer; that would be the reading's stated limitation, not a new revision on the same people.
