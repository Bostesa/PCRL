# Card DB_ZTILT_R1: spend-matched release-only control for DB's strongest release (family DB, revision R1)

Committed before its fit. Its purpose is the "simplest alternative explanation" control for the strongest feasible comparison.

DB_PTC_R1 is DB's strongest feasible nonzero release (`claims/STRONGEST.json`). Against the registered DB_ZTILT its pair benefit is
+0.0245 (se_agg 0.0075, both version means positive; `claims/BESIDE_DDR1_R1_BESIDE.json`). That comparison is not spend-matched.
DB_ZTILT was matched to DB_PTC's spend at the single registered strength, and the card DB_PTC_R1 did not refit it. On MODEL_SELECT
the total-NLL cost against the twin is:
- DB_ZTILT: 0.0085 / 0.0086 (versions 0 / 1);
- DB_PTC_R1: 0.0099 / 0.0099.

This card applies the registered ZTILT rule (DB_PTC_v1 table; PROTOCOL_AMENDMENT_1 finding 2) to DB_PTC_R1. It is part of the
ADAPTIVE_R1 reading and reported beside D2, as the registered ZTILT is. It changes no D label.

Code: `ddr/db/ptc_r1.py`, entry `python3 -m ddr.db.ptc_r1 ztilt --version V --card <this file>`. Deployment goes through
`ddr.db.predict` (operation `ztilt_r1`).

## Operation

- **Amplitude:** q_z = P(SEX | the twin's released log-probabilities). This is the registered ZTILT teacher `ddr.db.ptc.ztilt_teacher`:
  cross-fitted HGB on the twin's log-probability features, with a full fit for deployment. It is a function of the twin's release
  only.
- **Tilt weights:** fitted by the same rule as DB_PTC, with q_z as the amplitude.
- **Strengths:** the split law of DB_PTC_R1 along rho = alpha_O* / alpha_I* (0.65 / 0.35). The pairs are k x (alpha_I* m_z0,
  alpha_O* m_z0) for k in (0.5, 0.75, 1, 1.5, 2, 3), where m_z0 is the RMS-normalizing factor, as for DB_ZTILT.
- **Choice:** the first pair whose nested MODEL_FIT mean total-NLL spend over the twin reaches DB_PTC_R1's nested spend. The rule and
  the grid are those of DB_ZTILT. If none reaches it, the largest pair is saved and flagged (`reached_cand_spend` false).

## Evaluation and reading

- MODEL_SELECT utility against the twin and DB_PTC_R1.
- Grid bank `DDR1_R1_ZTILT`: DA_TWIN, DB_PTC_R1 and DB_ZTILT_R1, on R1-R4 x k0-k3. DB_PTC_R1's cells are cache hits of the R1 and
  refit grids.
- Declared reading, as for DB_ZTILT:
  - If DB_PTC_R1 is not resolved below DB_ZTILT_R1 (aggregate pair benefit - 2 se_agg <= 0 on the registered cells), the gain is
    reproduced by release-only information, which repeats PRC's failure.
  - If it is resolved below, the gain needs propensity information beyond the release. That carries no removal guarantee.
- The refit cells are reported beside with the inherited robustness screen.

## Budget and predictions

Configurations: 1 x 2 = 2 (DB_ZTILT_R1), bringing the total to 48 of 48. No configuration is left after it. CPU: under 0.05 h for
the fit; about 0.1 h for the grid.

Coordinator predictions, made before the fit:
- DB_ZTILT_R1 reaches DB_PTC_R1's spend at k <= 1.5 in both versions (p 0.7).
- Its pair AUC is within +-0.008 of the twin's (p 0.7).
- DB_PTC_R1 vs DB_ZTILT_R1: +0.015 to +0.030 (resolved on the registered cells, p 0.55).
