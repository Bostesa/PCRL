# Utility-reserve rule (registered summary)

The full rule, its justification and the reference calibration were written by the baseline and statistical reviewer before
any new fit and registered at `74376912c` (receipt `77dd6710f`): `roles/statistical_reviewer/UTILITY_RESERVE_RULE.md`, the
machine-readable `roles/statistical_reviewer/RESERVE_RULE.json` (SHA-256 `aca4d8e45fbe523629ba74f3f36eb51cf30ee00941cc13b18927a9a1d0accd70`)
and its implementation `ihs/reserve.py` (13 invented-data checks, `ihs/test_reserve.py`). This page restates it and records how it
was applied. It is a development selection heuristic on adaptively used data: not a confidence guarantee and not a changed cap.

## The rule

- **Income screen (12 conditions per configuration).** For each income clause (U0 / Ucal x accuracy loss, NLL excess, summed
  Brier excess) on each seed: margin = cap - value >= k x se_used on MODEL_SELECT, with k = 2.0 and se_used = max(the
  configuration's own household-clustered paired SE, H_LJ's MODEL_SELECT SE for the same clause and seed). No projection: the
  MODEL_SELECT scale (5,306 households) is used. This equals 1.7267 SEs at DEV_REPORT's 3,955 households and 3.633 SEs at the
  17,510 households that exist for a same-population study.
- **Occupation side.** The inherited H_LJ / JOINT_TREE occupation output is not screened; it is reported as an inherited
  constraint (it binds: men's guard 0.32 / 0.62 SE). A changed occupation output must, for each of its 6 clauses and 2 guards,
  meet the same reserve or keep at least the inherited margin.
- **Labels.** Per seed RESERVE_MET / RESERVE_NOT_MET; per configuration RESERVE_MET only if both seeds meet it.
- **Why k = 2 at MODEL_SELECT scale.** At 17,510 households a one-sided 5% / 80%-power bound needs about 1.37 MODEL_SELECT SEs of
  plug-in margin; the remaining 0.63 SE is a buffer for adaptive selection on MODEL_SELECT. The reviewer rejected the more lenient
  "2 SEs at 17,510 households" (1.10 MODEL_SELECT SEs).

## How it was applied (MODEL_SELECT; seed 0 / seed 1)

| Release | Reserve | Minimum income ratio (margin / se_used) |
|---|---|---|
| RETIRED_B3CTL (retired candidate) | NOT_MET (income Brier 0.40 / 0.35) | 0.35 |
| B2TWIN (6-level LOCAL code) | MET | 2.80 |
| H_LJ (eligible reference) | MET | 4.16 |
| NEURAL_RAW (eligible reference) | seed 0 only | (1.56-1.65 floored SEs on seed 1) |
| **H3_STRONG_LJDIR_PB (nominated successor)** | **MET** | **2.862** (binding: seed-1 income accuracy) |
| P3_JR_B60 | MET | 2.677 |
| PC_JSCALE_B60 (control) | MET | 2.212 |
| H2_STRONG_LJDIR_DP | NOT_MET (income Brier +0.00497) | 0.02 |

Every configuration's status is in `NOMINATION.json` / `NOMINATION_TABLE.csv` and the ledger.

## On DEV_REPORT (claim R2 of the frozen claims draft: margin >= 1.7267 x max(SE_DEV(successor), SE_DEV(H_LJ)))

`RESERVE_ERODED`: all 12 income margins are positive; 10 of 12 meet the requirement. The two that do not are the seed-1 income
accuracy-loss clauses (identical against U0 and Ucal): accuracy loss 0.007067, margin 0.002933 = 0.83 SE against 1.73 required.
The income summed-Brier margins are 3.8-5.0 SE and the NLL margins 5.8-7.0 SE (the retired candidate's binding Brier clause had
0.03-0.04 SE).
