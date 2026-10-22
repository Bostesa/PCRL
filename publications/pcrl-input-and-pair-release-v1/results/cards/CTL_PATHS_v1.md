# Card CTL_PATHS_v1: the shared SPEND and RO path points of the shared twin (family CTL)

Coordinator, campaign `pcrl_input_and_pair_release_v1`, written 2026-10-22 after 02:36Z (PROTOCOL_AMENDMENT_1 item 10). Governs every
configuration CTL_DQ_<step> and CTL_RO_<strength>. Code: `ipr/paths.py` (`spend_and_ro`, `bracket_points`, `save_shared`,
`predict_path`) and `ipr/ctlkit.py` (DQ, RO), deployed by `ipr/ctl/twin.py`; committed with this card before the first fit.

1. **Operations** (on the shared twin CTL_TWIN's vectors; no runtime SEX):
   - SPEND, DQ (`ipr.ctlkit.dq_release`): subtractive-dithered lattice quantization of each recipient's twin log-ratio vector in its
     standardized principal-component coordinates, basis fitted on the twin's MODEL_FIT out-of-fold vectors, dither keyed on the six
     compact fields with the family-independent tag IPR_DQ_S<v>; step grid `ctlkit.DQ_GRID_FINE` (0.1 x 1.1^j, j = 0..39). SEX is
     never read.
   - RO, GENERIC_ZTILT (`ipr.ctlkit.ro_path` / `ro_release`): log r = log p - alpha (q_z - qbar) w + const for both recipients, q_z =
     P(SEX | the twin's released log-probability features) by HGB (SEX-side settings, 300 iterations) cross-fitted on the household
     folds NRM_FA_xf_S<v> (deployed: the full fit), covariance-cancelling weights (income one vector; occupation one vector per twin
     decision stratum, global fallback under 200 people or Cov <= 1e-4) on the out-of-fold statistics with MODEL_FIT SEX; out-of-fold
     path values with nested weights (outer household folds IPR_RO_ncv_S<v>); strength grid `ctlkit.RO_GRID` (0.05 x 1.1^j,
     j = 0..41).
2. **Saved points.** For each family candidate, inside that family's fit unit, the two bracket points of each path by the registered
   rule (PROTOCOL 4: LO = the largest grid point whose MODEL_FIT out-of-fold total-NLL spend over the twin is at most CAND's; HI = the
   smallest grid point whose spend is at least CAND's + 2 household-clustered SE of the paired difference; the largest grid point
   flagged CONTROL_PATH_MAY_NOT_BRACKET if none). A point equal to one already saved (same path, step or strength, version) is reused,
   not saved again, and counted once. The fineness inputs (lower, upper, upper SE, CAND's spend, the path's largest spend) go to the
   family's fit record for `ipr.rules.grid_fineness`.
3. **Role.** SPEND is a gating path of every family; RO (GENERIC_ZTILT) is family IA's declared RO path (fallback) and is not built
   for the SEX-blind family IB.
4. **Configuration count and CPU.** At most 2 points x 2 paths x 2 versions per family, fewer where they coincide; under 5 CPU-minutes
   per family and version (the RO teacher dominates).
