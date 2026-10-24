# Failure map of earlier PCRL release-mechanism families (campaign `pcrl_strong_anchor_release_v1`; coordinator)

Written 2026-10-24 from about 04:33Z from public records only (no label; nothing under `.private/` except the label-free anchor table
and admission hashes). Rows 1-14 are the input-and-pair release campaign's map (`results/pcrl_input_and_pair_release_v1/FAILURE_MAP.md`,
published in 9c2aa4d99 and pinned by `SOURCE_ADMISSION.json`), which stays the reference for those rows. Row 15 is the input-and-pair
campaign itself. Paired values a / b are versions 0 / 1.

## Rows 1-14 (inherited, one line each)

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
| 11 | Propensity-routed finite cells (FA) and the SEX-targeted coupled Gaussian logit channel (FB2) | SEX targeting not separated from SEX-blind spending of the same form |
| 12 | Propensity-routed finite cells, fitted (PRC) | No reduction beyond any control; a task-score-only grouping (HQ) reproduced it |
| 13 | Solver adequacy of the PRC objective (SV1) | A better optimum of the surrogate is not a better release |
| 14 | SEX standardization, IPW and propensity-tilt cancellation (dual-design) | Not separated from release-only information (the twin's own release reproduced most of the gain) |

## Row 15 (since that map)

| # | Family (campaign; publication) | Operation | Best measured outcome (IPR1 registered grid, 16-recipe bank) | Dominant failure | Lesson |
|---|---|---|---|---|---|
| 15 | Input-side SEX-ordered field generalization (IA_SFG), occupation-only generalization (IA_OSG, adaptive) and SEX-blind coordinated keyed collapse to decision-cell means (IB_CSF) (input-and-pair; 9c2aa4d99) | IA: a ladder of coarser raw-field levels inside budget S3, both task models refitted; IA_OSG: only the occupation recipient's input; IB: for a keyed share of people both vectors collapsed to their decision-cell means | Twin CTL_TWIN 0.6480 (income 0.5498, occupation 0.6443). IB_CSF 0.6411 (+0.0069, se 0.0057 vs the twin; +0.0007 vs its SEX-blind spend bracket). IA_SFG 0.6464 (+0.0016). IA_OSG occupation view 0.6351 but pair 0.6463. D2 met in 0 of 3 looks; best -0.0158 vs NEURAL_RAW, -0.0225 vs H_LJ | **A weak start**: every candidate stayed 0.016-0.028 more recoverable than NEURAL_RAW and H_LJ; S3 stopped the generalization ladder at its first informative step; an occupation-only reduction did not reach the linked pair | Start from strong releases; price operations on the pair; choose budgets from the actual margins |

## Cross-family notes carried into this campaign

1. **Spend explains the large drops** (rows 1, 2, 5, 7, 9, 11, 12, 15). With spend held, benefits were about 0.002-0.018.
2. **The linked pair is strongly super-additive** for the JOINT-occupation releases (IPR1_MAIN): SUCC occupation view 0.5731, income
   view 0.5312, pair 0.6074 (+0.034 above its best single view); TWJ +0.063; H_LJ +0.046; RETIRED_B3CTL +0.023 with a 6-level income
   code. For NEURAL_RAW and the task-only releases the pair is about its occupation view.
3. **The strong occupation protection binds a within-SEX guard.** The JOINT_TREE correction costs both SEX groups about +0.005 summed
   occupation Brier against its own task-only routing (TASK_REFIT), but the tree base leaves men only about 0.001 of slack against U0
   (women about 0.010), so the men's guard sits at 0.88 / 0.79 of its cap in every JOINT-occupation release.
4. **Leakage moves between recipients** (rows 1, 3, 10): per-recipient views and the linked pair are both required.
5. **Readers decide near 0.01** and are data-limited; finite releases favour LOOKUP; few-valued views make reader seeds ineffective
   (count n_eff).
6. **Controls must match the mechanism and its cost** (rows 2, 9, 14): equally exposed continuations, the simplest untargeted
   operation at comparable task cost, and release-only explanations where the operation could be read off the start's release.
7. **All evidence is development evidence** on adaptively used MODEL_SELECT records; a 0.020 benefit is not confirmable here.

## Closed as mechanisms (not renamed in this campaign)

Scalar leaf-score corrections on any routing; decoder rescue, learned decoders and calibrators; another controller; selective
flattening and coordinated keyed collapse; another ordinary cell merge, propensity-routed cells and their solvers; input-side field
generalization inside S3; Gaussian and coupled logit channels or a slightly different noise strength; SEX standardization,
inverse-propensity reweighting, linear propensity tilts and release-only tilts as mechanisms; the 6-level income codes; temperature
and per-class bias rescue. The cross-recipient swap FS_SWAP was declined on unfavourable planning evidence and needs a new rationale.
