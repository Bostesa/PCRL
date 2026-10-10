# A FIT feasibility closeout — aggregate receipt review

Status: R11 review draft for independent R04 check. This records the frozen producer FIT results under amendment SHA `526f509f8077f2ee1c412f6d290df939966a6b81f1ae5a87e9639d4bd317271f`. It uses only aggregate private FIT receipts and the public R03 closeout (`r03/A_PRODUCER_FIT_CLOSEOUT.json`, SHA `beb24a699c696adf0c0f7b483c7a1c9e074510393907e05a710d1fac892a9d39`). No role arrays, saved checkpoint arrays, ATT/INNER outcomes, new fit, or cap change entered this review.

## Frozen identity action on FIT

R04 independent source replay (`r04/A_PREFIT_INDEPENDENT_REPLAY.json`, SHA `2f5fed77088d665aeac2be0147b73cbe9bba4f09e4b15807e5c4e78c85e19b8d`) gives exact expected identity-action utility. It preserves both decisions, and its income loss meets both U0 and Ucal caps in both seeds. Occupation meets Ucal caps, but fails the required U0 NLL cap in both seeds and the U0 Brier cap in seed 1. The fitted cap is reference loss plus 0.010 NLL or 0.005 Brier on identical FIT rows; both reference comparisons must pass.

| Seed | Task | Reference | Identity NLL | NLL cap | NLL gap | Identity Brier | Brier cap | Brier gap |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | income | U0 | 0.297393442 | 0.301554854 | −0.004161412 | 0.189996240 | 0.192308697 | −0.002312456 |
| 0 | income | Ucal | 0.297393442 | 0.306080187 | −0.008686745 | 0.189996240 | 0.194175314 | −0.004179073 |
| 0 | occupation | U0 | 1.223729798 | 1.219417787 | +0.004312011 | 0.632198531 | 0.632221459 | −0.000022929 |
| 0 | occupation | Ucal | 1.223729798 | 1.225193527 | −0.001463730 | 0.632198531 | 0.634417011 | −0.002218480 |
| 1 | income | U0 | 0.294109299 | 0.297318931 | −0.003209632 | 0.187169669 | 0.189199775 | −0.002030105 |
| 1 | income | Ucal | 0.294109299 | 0.303949948 | −0.009840649 | 0.187169669 | 0.191795158 | −0.004625489 |
| 1 | occupation | U0 | 1.217416573 | 1.211996017 | +0.005420556 | 0.628522722 | 0.628174646 | +0.000348077 |
| 1 | occupation | Ucal | 1.217416573 | 1.218659153 | −0.001242581 | 0.628522722 | 0.630668265 | −0.002145543 |

Thus the frozen identity action is not a FIT-feasible witness. The source-bank label-aware lower bounds in the same R04 receipt are optimistic necessary bounds, not a deployable feasible law.

## All 16 paired arm/seed FIT units

`r2` is occupation. Gaps below are metric minus the frozen U0-based occupation cap, rounded to nine decimals. A positive gap is a violation. For stochastic, product, LOCAL and SEQ12, the NLL/Brier entries are **separate minima over recorded checkpoints** and need not come from one checkpoint; positive minimum NLL alone rules out a feasible recorded checkpoint. For deterministic arms, only the identity-biased terminal replay is reported in this table; the complete two-start selection receipt supplies the no-feasible status. For SEQ21 the minima are over stage 1 only. Full unit IDs, receipt SHAs, queue return codes and charges are in the public R03 closeout.

| Arm | Seed | Queue rc | FIT status | Recorded occupation NLL gap | Recorded Brier gap | Diagnostic scope |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| A_JOINT_STOCH | 0 | 0 | COMPLETE_NO_FEASIBLE_POLICY | +0.004342792 | −0.000010686 | 26 complete two-start checkpoints; retry FIT ID `A-FIT-A_JOINT_STOCH-S0-R1` |
| A_JOINT_STOCH | 1 | 0 | COMPLETE_NO_FEASIBLE_POLICY | +0.005433225 | +0.000352808 | 26 complete two-start checkpoints |
| A_JOINT_DET | 0 | 0 | COMPLETE_NO_FEASIBLE_POLICY | +0.004312011 | −0.000022929 | Identity-biased terminal; both 300-update starts and fixed polish complete |
| A_JOINT_DET | 1 | 0 | COMPLETE_NO_FEASIBLE_POLICY | +0.005420556 | +0.000348077 | Identity-biased terminal; both 300-update starts and fixed polish complete |
| A_TASK_STOCH | 0 | 0 | COMPLETE_NO_FEASIBLE_POLICY | +0.012343085 | +0.003207095 | 26 complete two-start checkpoints |
| A_TASK_STOCH | 1 | 0 | COMPLETE_NO_FEASIBLE_POLICY | +0.005921044 | +0.000528665 | 26 complete two-start checkpoints |
| A_TASK_DET | 0 | 0 | COMPLETE_NO_FEASIBLE_POLICY | +0.004312011 | −0.000022929 | Identity-biased terminal; both 300-update starts and fixed polish complete |
| A_TASK_DET | 1 | 0 | COMPLETE_NO_FEASIBLE_POLICY | +0.005420556 | +0.000348077 | Identity-biased terminal; both 300-update starts and fixed polish complete |
| A_CENTRAL_PRODUCT | 0 | 0 | COMPLETE_NO_FEASIBLE_POLICY | +0.004333544 | −0.000014415 | 26 complete two-start checkpoints |
| A_CENTRAL_PRODUCT | 1 | 0 | COMPLETE_NO_FEASIBLE_POLICY | +0.005433275 | +0.000352825 | 26 complete two-start checkpoints |
| A_LOCAL | 0 | 0 | COMPLETE_NO_FEASIBLE_POLICY | +0.004414507 | +0.000017622 | Recipient 1 found a feasible local checkpoint; recipient 2 did not |
| A_LOCAL | 1 | 0 | COMPLETE_NO_FEASIBLE_POLICY | +0.005470639 | +0.000366893 | Recipient 1 found a feasible local checkpoint; recipient 2 did not |
| A_SEQ12 | 0 | 0 | COMPLETE_NO_FEASIBLE_POLICY | +0.004711827 | +0.000137611 | Stage 1 completed; stage 2 full-law replay has no feasible checkpoint |
| A_SEQ12 | 1 | 0 | COMPLETE_NO_FEASIBLE_POLICY | +0.005544582 | +0.000394977 | Stage 1 completed; stage 2 full-law replay has no feasible checkpoint |
| A_SEQ21 | 0 | 1 | INCOMPLETE_RESOURCE *private diagnostic* | +0.004407116 | +0.000014672 | Stage 1 only; full arm incomplete and queue failed |
| A_SEQ21 | 1 | 1 | INCOMPLETE_RESOURCE *private diagnostic* | +0.005478903 | +0.000369972 | Stage 1 only; full arm incomplete and queue failed |

Both SEQ21 units ran two 150-update stage-1 starts with seven checkpoints each. Every stage-1 checkpoint violated its occupation constraints, so no first checkpoint was selected and stage 2 was not run. `controls/seq_train.py` returns `complete=false` for this condition; `controls/run_admitted_controls.py` writes a private diagnostic receipt, then raises `TimeoutError`. R03 recorded rc 1, `timed_out=false`, no resource limit, and about 9 seconds of wall time against 300 seconds reserved. The `INCOMPLETE_RESOURCE` label is therefore not evidence of a CPU cutoff. It is also not an admitted completed SEQ21 scientific null: the full sequential arm is incomplete. R04 independently checked this source/receipt distinction before this summary; its public review receipt is pending.

## Interpretation and dependency rule

Fourteen FIT units completed their frozen starts and selected no feasible policy; two SEQ21 units failed after a complete but infeasible first stage. No fitted A policy package was selected in either seed. The predeclared selected-policy dependency therefore does not trigger fitted A release, fitted Q, or fitted attack descendants. Fixed historical references and identity/class diagnostics have separate sources and may proceed under their own admission. These outcomes describe the bounded frozen algorithms and recorded checkpoints; they do not establish global infeasibility of the same-bank channel class or a privacy result. Keep all utility thresholds, algorithms, and incomplete statuses fixed.
