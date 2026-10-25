# Failure map (campaign `pcrl_direct_recovery_frontier_v1`; coordinator; registered before any label read)

Written 2026-10-25 from 02:57Z from public records only. Rows 1-15 are the strong-anchor campaign's map
(`results/pcrl_strong_anchor_release_v1/FAILURE_MAP.md`, published in c8aaa2df4), which stays the reference for them. The rows below
are the closed recipes this campaign must not resume and the finite-routing facts its design answers.

| # | Family or recipe | Dominant failure | Status here |
|---|---|---|---|
| A | SAR1 FA: SUCC-anchored residuals (FA_TFR, FA_TFRO_R1, FA_TFRB_R2: distillation residual, online SEX critics, cross-fit bag, guard estimator) | Point-feasible but protection profile not kept across versions (version 0 occupation view +0.025); D2 not met; D4 point only | closed: residuals, guard estimators and strength sweeps are not resumed |
| B | SAR1 FB: pair Fisher-discriminant penalties on NEURAL_RAW (FB_PFD, FB_PFO) | Gain over NEURAL_RAW not separated from its own recipe continued (ALT); sign change on the robustness split | closed: Fisher penalties are not resumed |
| C | SAR1 FS: conditional pair-increment residual on SUCC's income | No feasible strength | closed |
| D | NRM1 FA_PRC_v1 / FRC1 PRC: propensity-routed finite cells, Lagrangian of task cross-entropy + lambda x pi-weighted plug-in MI (pair + local) + lambda kappa x per-SEX guard; damped Lloyd with seven tilts; lambda chosen afterwards by a nested utility screen (budgets S1-S3) | Protected path saturated (spent about 0.002 nats of a 0.008-0.012 budget; strength probe saturated to 128); PRC_S2 +0.0070 (0.0040) vs the twin, not separated from HQ / UNT / ORD controls; 0.018-0.023 above NEURAL_RAW / H_LJ | the previous objective is re-run only as the matched control DR_MI inside this campaign's family, with explicit constraints instead of the lambda path |
| E | FSA1 SV1: exact tilt-line solver for the same surrogate | Lowers the surrogate only; no resolved held-out gain; the shrunk linearized routing is anti-ranked against the exact objective | not resumed; the DR optimizer evaluates the exact objective of every candidate move |
| F | SCM / IBSI shared-cell merging (structural-mechanism pilot, SCM benchmark, SCM utility-matched) | Benefit not separated from task spending; utility match did not carry to held-out people; men's guard missed | overlap acknowledged (privacy funnel / IBSI engineering); matched CTRL coarsening path and the guard as an explicit constraint |
| G | FRC1 / ORD SEX-blind coarsening | Nested LOOKUP AUC rises at moderate coarsening (fine supports defeat finite readers); large task cost for small reductions | used as the untargeted control path (DR_CTRL) |

Facts the design uses (published): the twin TWB_C64 meets all 28 conditions with slack (MODEL_SELECT men's guard -0.0008 / +0.0002,
occupation NLL vs U0 -0.0071 / -0.0066); grid pair AUC 0.6555 on FRC1's grid, occupation view 0.6379, income view 0.5681; SUCC 0.6140
on that grid (SAR1 grid 0.6178 / 0.6155).
