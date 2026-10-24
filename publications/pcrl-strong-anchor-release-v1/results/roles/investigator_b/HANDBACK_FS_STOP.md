# Investigator B hand-back: FS stop memo (saved verbatim by the coordinator, 06:37Z on 2026-10-24)

# Stop memo for FS: no FS_R1 card (investigator B, 2026-10-24, 06:35Z by `date -u`)

**Decision: close FS. I am not proposing a revision.** My honest prediction for a decision-preserving FS revision is a pair benefit over SUCC of about 0.005 on MODEL_SELECT at its budget-feasible strength (80% interval 0.000–0.011). That is below the 0.010 threshold. On the fitting data, the part of it that SUCC's own operation (ALT) does not reproduce is about 0.001–0.004, so D2 cannot be met.

All evidence below comes from committed fitting records: `fs/FIT_CPI_V0.json` and `fs/FIT_CPI_V1.json`. They hold CALIBRATION spends over START_CF and descriptive MODEL_FIT readers. No MODEL_SELECT value enters this memo. Scripts: `roles/investigator_b/fs_r1_extract.py` and `fs_r1_interp.py`.

## 1. What FS_CPI_v1 showed

- **CALIBRATION accuracy is noisy, and accuracy is the binding component.** In v0 the first strength (λ 0.025) fails on income accuracy, +0.00268 (se 0.00125) against 0.0040. NLL (+0.00062) and Brier (+0.00042) are far inside their budgets. The accuracy path is non-monotone: 0.00235 (0.0018) at 0.05, 0.00335 (0.0026) at 0.1. Version 1 selected λ* 0.05, where CAND spends +0.00104 (se 0.0004) nats over START_CF.
- **Without the accuracy component, Brier would bind at about λ 0.05 / 0.1.**
  - v0: Brier at λ 0.1 is +0.00236 (0.00062), bound 0.0036, just above the 0.0035 budget.
  - v1: Brier passes at λ 0.1 (+0.00158 (0.00053), bound 0.0026) and fails at 0.2 (bound 0.0053).
  - So a decision-preserving residual, whose accuracy spend is zero, could go at most one grid step further in v1, and none in v0.
- **The CAND moves the pair about 0.010 on MODEL_FIT, but SUCC's own transplant (ALT) moves it almost as much at the same spend.** Descriptive MODEL_FIT readers, version 1, pair AUC. SUCC and the residual are in-sample there; the readers are cross-fitted. The occupation view is identical in every row (0.5658).

  | Release | CALIBRATION spend over START | LOG_LR pair | HGB15 pair |
  |---|---|---|---|
  | SUCC | 0 | 0.6101 | 0.6239 |
  | CAND, λ 0.05 | +0.00104 | 0.5997 (−0.0105) | 0.6154 (−0.0085) |
  | ALT LO, β 0.30 | +0.00084 | 0.6046 (−0.0055) | 0.6179 (−0.0060) |
  | ALT HI, β 0.40 | +0.00189 | 0.5989 (−0.0112) | 0.6123 (−0.0116) |
  | ALT interpolated at CAND's spend | +0.00104 | −0.0066 | −0.0071 |
  | CTRL DQ HI | +0.00564 | 0.6080 (−0.0021) | 0.6183 (−0.0056) |

  - Beyond SUCC's own operation at matched spend, CAND gains +0.0039 (LOG_LR) / +0.0014 (HGB15). This is in-sample for the residual.
  - The in-batch training proxy agrees. The pair term falls from 0.1326 / 0.1374 at START to 0.1158 / 0.1325 at λ 0.05 and 0.1074 / 0.1278 at λ 0.1 (v0 / v1).

## 2. Why a decision-preserving revision would not reach 0.010 against SUCC

The revision you suggest is z' = z·exp(g), with every decision kept equal to H3's. It removes the binding accuracy noise by construction, but it does not create new protection:

1. **The strength gain is at most one grid step.** Brier then binds at λ about 0.05 (v0) and 0.1 (v1). At those strengths the unconstrained residual's MODEL_FIT pair drop is about 0.010. This is measured on the very people whose SEX trained the residual, so it is an upper estimate.
2. **The constraint removes a channel.** With decisions frozen, H3's income decisions are released unchanged, and so is their SEX information given the occupation output. Only the confidence part of the increment can move. At the same Brier spend, the reduction is at most the unconstrained one. I take it as about 0.8 of it.
3. **MODEL_FIT overstates the gain.** In-sample optimism of the protection, and FB_R1's lesson that MODEL_FIT understated MODEL_SELECT spend by 1.95x, both point to a smaller MODEL_SELECT gain. I take about 0.6 of the MODEL_FIT drop.

**Prediction.** About 0.010–0.014 × 0.8 × 0.6, i.e. about 0.005–0.007; I state 0.005 with an 80% interval of 0.000–0.011. P(≥ 0.010) is about 0.2.

**D2 / D3 / D4.** The advantage over ALT at matched spend is 0.001–0.004 on the fitting data, before any transfer loss. The reviewer's power is below 0.15 even for a 0.010 total benefit.
- D2 (START, CTRL and ALT): about 0.03.
- D3: about 0.02.
- D4: about 0.01.

## 3. Other points you asked about

- **DQ does not preserve decisions.** Subtractive-dithered quantization moves the standardized income log-odds by up to δ/2, so it crosses 0 for people near H3's threshold. It is therefore not the matched untargeted control for a decision-preserving candidate. That would need a sign-preserving quantization of |z|, which no saved control implements.
- **Saved controls.** FS_CPI_DQ_HI and FS_CPI_BETA_LO / _HI (v1 only) were bracketed for CAND's +0.00104 spend. A revision with a different spend would need re-bracketed points, i.e. new specifications.
- **I recommend closing FS.** The evidence says the income-side increment is reachable only at about the rate of SUCC's own β transplant. The remaining budget is better spent on follow-through.

**Label-free CPU for this memo:** about 1 CPU-s (two record-reading scripts; no fit, no selftest). No code was written or changed, and no git command was run.

The memo is also saved as `<repository>/.claude/worktrees/pcrl-strong-anchor-release-v1/results/pcrl_strong_anchor_release_v1/roles/investigator_b/STOP_MEMO_FS_R1.md`. The scripts are `fs_r1_extract.py` and `fs_r1_interp.py` in the same directory.
