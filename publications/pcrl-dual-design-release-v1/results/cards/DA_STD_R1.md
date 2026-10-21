# Card DA_STD_R1: IPW-STD (family DA, revision R1)

Committed before its fit. It follows the registered DA reading (claims/DA_REG, a MODEL_SELECT outcome), so it is ADAPTIVE_R1 under
RULES.md 7 and is reported beside the registered reading, never as the family's registered result. Code: `ddr/da/ipw.py` (entry
`python3 -m ddr.da.ipw fit --version V --card <this file>`); deployment through `ddr.da.predict` (operation `ipw`).

## 1. Observed failure and the declared trigger (DA_STD_v1 section 6, R1 (b))

- Registered comparison not separated: DA_REG has D2 `PARTIAL_SBC_FRONTIER_ONLY`; against the SPEND path the frontier is
  `NOT_SEPARATED_FROM_TASK_SPENDING` (benefit +0.0137, se_agg 0.0174 at DA_SPD_HI) and against TWIN the pair benefit is
  `AGG_POSITIVE_POINT_ONLY` (+0.0175, se_agg 0.0090).
- Wording: DA_STD_v1's trigger (b) names `NOT_SEPARATED`. In the registered rules (RULES.md 3.9) that label means that no D2
  comparison passes, and a reading in which some pass is `PARTIAL_<passed>_ONLY`. DA_REG passed the SBC frontier, so **under a
  strict reading the first condition of (b) does not hold.** The revision is run instead under the campaign's standing authority for
  justified revisions that address an observed failure. The observed failure is that D2 is not met: the SPEND frontier is not
  separated, the TWIN benefit is point-only, and (second condition, MODEL_FIT) the release kept part of the propensity dependence.
  This is disclosed as deviation 7 in PROTOCOL_DEVIATIONS.md. The reading is ADAPTIVE_R1 either way, and it changes no registered
  label.
- Second condition (MODEL_FIT out-of-fold, `da/R1_TRIGGER_B2.json`, queue unit DDR_DA_R1_TRIGGER_V1, 19:13Z): for at least one
  recipient, the linear R^2 of logit pi on the lam = 1 release's log-probabilities exceeds half the twin's. **Held in both versions
  for the income recipient**: R^2 0.0665 vs the twin's 0.0999 (ratio 0.666) in version 0, 0.0648 vs 0.0980 (0.661) in version 1.
  The occupation recipient's ratios were 0.387 and 0.391. The SEX-aware trees attributed part of the SEX effect to proxy fields, so
  the plug-in standardization kept part of the income release's propensity dependence. Trigger (a) did not hold: the occupation
  out-of-fold NLL spend at lam* was 0.0057 / 0.0054 against the limit 0.0166.

## 2. Operation

One SEX-blind HGB per task on Z (A1_PO_16 settings) trained on MODEL_FIT with inverse-propensity weights omega_i = w / pi_i (women) or
(1 - w) / (1 - pi_i) (men); pi = the cached cross-fitted SEX-side propensity clipped to [0.05, 0.95]; w = the registered fit's
guard-balanced weight (0.45 / 0.44); omega normalized to mean 1. T* by the weighted inner household-CV NLL, full refit, out-of-fold on
`NRM_FA_xf_S<v>`. In the reweighted population SEX is independent of x, so the model targets m_w(x) = sum_s w_s P(y | x, s) without
evaluating a SEX input counterfactually. Release (1 - lam) twin + lam m_IPW with the DA_STD_v1 grid (0.5, 0.75, 1.0) and S3 rule, per
version. None qualifying: NO_FEASIBLE_STRENGTH.

## 3. Controls (PROTOCOL 8 reuse rule)

TWIN unchanged (DA_TWIN). SPEND path reused (DA_SPD_LO / DA_SPD_HI). SBC path refitted (the operation changed): the same IPW operation
with SEX replaced by the DA_STD_v1 SBC target rule (fixed-seed permuted SEX first; PSEUDO2 with gamma the smallest of 1, 2, 4 reaching
the candidate's spend), its propensity a cross-fitted HGB of the target (ddr.ctlkit.attribute_profile) and its mixture weight the
target's share; two bracket points DA_IPW_SBC_LO / _HI. Ablations are not refitted.

## 4. Stop rule and predictions

DA's first revision; at most one more. Predictions (coordinator, before the fit): lam* = 1.0 in both versions (p 0.6); income view
lower than DA_STD's by 0.000-0.010; pair benefit against TWIN larger than DA_STD's by -0.003 to +0.008; D2 met p 0.10.
Configurations: 3 x 2 = 6 (DA_IPW, DA_IPW_SBC_LO, DA_IPW_SBC_HI). CPU about 0.1 h.
