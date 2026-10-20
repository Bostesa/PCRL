# Failure map of earlier PCRL release-mechanism families (challenger)

Campaign `pcrl_new_release_mechanisms_v1`, challenger role, 2026-10-20 (written 14:53-15:00Z).

Sources: public records under `results/pcrl_*`, read only. No data, no labels, nothing under `.private/` was read.

- Numbers are quoted from the cited decision documents. Paired values a/b are seed (version) 0/1. Brackets are registered intervals.
- (*) marks arithmetic done by the extraction pass, not a registered endpoint.
- Eight key numbers were re-checked against the sources before writing (rows 1, 2, 3, 5, 7, 9); all eight matched.

**Comparability.** Only rows 2, 5 and 9, and the ACS halves of row 3, ran on the current ACS 2019 FL income/occupation
setting, where pair AUCs are about 0.60-0.67. The other rows used different settings:

- Rows 1, 6, 7, 8 and 10, and the Adult parts of rows 3-4, used reused UCI Adult people. Occupation has 6 classes, several
  studies also released 16-d features or logits, and pair AUCs are about 0.74-0.90.
- The two residence channel studies in row 4 used ACS 2018 residence.

Absolute AUCs do not compare across settings. Deltas and failure modes do.

| # | Family (campaigns; data) | Operation that changed the transmitted output | Best measured outcome (role; readers) | Dominant failure | Lesson for controls |
|---|---|---|---|---|---|
| 1 | **Adversarial penalties, refreshed critics.** joint_complete_view, penalty_no_erasure, refreshed_guarded_joint, strength_matched_feedback, online_strength_frontier (Adult, 10-03..10-05) | Two MLP encoders (64-64-16) and heads trained on task loss + β·(per-view + coalition SEX-critic penalties). Variants: LEACE in the loop or as the final map; no eraser; critic refits every 4 epochs with per-recipient feedback; normalized strength ρ with an AUC controller; raw β {0.1, 0.3, 0.6} vs ρ {1.5, 3, 5} | Incumbent RAW-J β0.3: pair 0.802 vs U 0.883 (outer dev assessment of reused rows; LR/MLP/HGB/defense-aware readers), at occupation NLL excess +0.016/+0.012/+0.012 (*). NORM-J ρ3: 0.786 (−0.016 [−0.021, −0.011]) at +0.021..+0.024 (*). J-G vs J-O: −0.039. J-F vs RAW-J: −0.054. All LEACE arms infeasible (accuracy −0.068/−0.035) | **Utility cost.** Every route to lower recovery spent occupation accuracy or NLL (β 0.3→0.6: −0.038 pair for −1.1 accuracy points). Refresh, feedback and controller never beat raw strength. Leakage moved to the income recipient (P02 +0.010 [0.001, 0.019]) | "Strength is the lever": the raw-strength path of the same penalty explains each "smarter" variant (NORM-J lies on the raw-β line, 0.784 interpolated vs 0.786 measured). No study had a SEX-blind control matched to the candidate's spend. Always report per-recipient views |
| 2 | **Leaf-score corrections, safeguards.** correction_transfer_stability (10-18), controlled_training_size (10-19), occupation_headroom_overnight (10-18) (ACS FL) | A critic-trained correction added to the occupation HGB leaf scores on frozen T_TREE / A1_PO_16 routing, scaled by s. Arms: PAIR (coalition critic), LOC (occupation critic), CONT (zero-critic continuation). Income frozen at H3. Budget rules B50/B80/M90; nested 10-fold strength rule | At the selected s=0.3 (MODEL_SELECT 8-cell grid, 16-recipe bank): all 24 clauses + 4 guards pass, reserve 3.83/3.64. LOC vs W +0.0046 (POSITIVE_NOT_STABLE); LOC pair 0.644 vs W 0.649; −0.003 to −0.045 against all six references. At s=1.0, PAIR vs W is +0.041 STABLE but fails utility (men's cost 14-70× the slack) and is +0.010, sign-changing, vs LOC. P2A_PAIR_B80: 0.5813/0.6118, +0.040 vs the CONT twin, +0.004 vs SUCC (better in 18/18 s0 cells but only 3/18 s1), and spends the s0 men's reserve (0.85) | **Recovery not reduced at a feasible strength.** The benefit appears only far outside the budget and is unstable across versions. The training-size gain is not recipe-specific (IR3 CONTROL_EXPLAINS_GAIN) | CONT and LOC absorb most of the effect. The equally trained TASK_REFIT shares the training-size gain, so the twin must be trained on the same people at the same size. The line is CLOSED: no retuning of strength, controller or temperature |
| 3 | **Tree leaf-score privacy training, frozen hybrids.** task_headroom_protected_outputs, frozen_tree_control_completion (Adult, 10-15); acs_protected_tree_replication, acs_frozen_hybrids (ACS FL, 10-16) | CatBoost depth-4 leaf scores trained against online SEX critics: L_TREE (local) and J_TREE (coalition), ρ ∈ {1, 3, 10}, up to E20. ACS: LOCAL_TREE at ρ1 E20. H_LJ composes LOCAL income with JOINT occupation, with no training | ACS DEV_REPORT, 7 readers × 3 views. LOCAL_TREE: pair 0.657/0.649 vs TASK_REFIT 0.664/0.666 (+0.0070, an established shortfall / +0.0175 unresolved); −0.017/−0.011 vs NEURAL_RAW. H_LJ: 0.641/0.643, +0.022/+0.024 vs TASK_REFIT (clause bounds 0.012/0.014); −0.001/−0.005 vs NEURAL_RAW. H_JJ: 0.603/0.606 but utility-ineligible (JOINT income accuracy loss 0.0140). Adult L_TREE: +0.0505 [0.0257, 0.0754] vs RAW_PROJ, but only +0.014..0.018 (unresolved) vs the historical codes | **Benefit below 0.020** against eligible references. Confirmation is infeasible (about 317k groups needed) | The equally exposed NEURAL_RAW and TASK_REFIT explain the ACS effect. Larger reductions came only from JOINT income, which costs income utility. On Adult, RAW_PROJ was a weak incumbent |
| 4 | **Frozen-bank random, sampled and replacement channels.** latent_and_sampled (10-10), incumbent_view_risk (10-10), conditional_budgeted_release (10-11), bank_anchored_release (10-10) (Adult); stochastic_channel (09-20), stochastic_replacement (09-21) (ACS 2018 residence) | One draw per person of a code from a fixed bank: identity, the 2 nearest same-class codes, or the class medoid, chosen by a 480-parameter gate. Other variants: LP-designed finite-law kernels over a frozen code bank; identity plus at most 3 nearest same-class states from exact P0; re-choosing existing ten-bit codes; an appended or replacement label-free quantized code T (\|T\| 64/256) | latent_and_sampled: 16/16 COMPLETE_NO_FEASIBLE_POLICY. The identity action alone fails the occupation U0 NLL cap (+0.0043/+0.0054), and no sampled release was attacked. incumbent_view_risk: 24/24 utility; best V-S21 −0.0024 [−0.176, 0.181] vs P0 0.8031. conditional_budgeted: W_STRICT +0.0020 [−0.0037, 0.0076], and sequential 1→2 beat W by 0.0021. bank_anchored: −0.006 vs B0 0.838, far above P* 0.803. Residence: T is subsumed by J (−0.0027/−0.0114 nats); the replacement code costs +0.021/+0.035 nats in quantization | **Infeasible bank** (the identity action is not a utility witness) or **recovery not reduced**. Randomness was never evaluated on an attacked real release. The coder's quantization was the limit | The identity action must be a utility witness before any channel. The strongest controls were the deterministic same-bank and freeze-first sequential ones. Randomness earns no credit without its deterministic counterpart at matched spend |
| 5 | **Post-hoc shared-cell merging, SCM, IBSI.** structural_mechanism_pilot, scm_engineering_benchmark, scm_utility_matched (10-17); mechanism_design (10-16, data-free) (ACS FL) | 512 KL k-means cells of out-of-fold CatBoost posteriors. Greedy merges by SEX-variance gain per unit task cost under β·(0.010 NLL, 0.005 Brier, within-SEX occupation Brier guard), β ∈ {0.5, 1, 1.5}. The occupation recipient receives the merged cell's pooled posterior; income receives one of 32 SEX-blind groups | DEV_REPORT pair (7 inherited readers + LOOKUP): SCM 0.6175/0.6219 vs TWIN 0.6584/0.6595, NEURAL_RAW 0.6400/0.6377, H_LJ 0.6414/0.6428. Occupation NLL vs Ucal 0.0106 (s0) exceeds 0.010; men's Brier 0.0062/0.0078. Utility-matched follow-up: SCM 0.6234/0.6326 vs ORD_NLLM 0.6570/0.6579, but SCM spent more on DEV (occupation +0.0070/+0.0017 and income +0.0064/+0.0035 nats over ORD_NLLM). Men's guard 0.0054/0.0064. Benefits vs references: +0.017/+0.005 and +0.018/+0.010 | **Benefit not separated from task spending.** No ordinary release reached SCM's DEV spend, so the frontier there is unmeasured. The 0.001-nat MODEL_SELECT match did not carry over (paired DEV SE 0.0016-0.0020). The S3 ruling: an engineering combination (privacy funnel / IBSI) | Extend the SEX-blind path beyond the candidate's spend (bracketing) and compare on the frontier, not at one matched point. "TWIN" used SEX through the within-SEX guard, so a truly SEX-blind control is needed. Coarse releases change the reader budget per vector (16-17 vs 53-64 people per vector), so LOOKUP-type readers are needed |
| 6 | **Calibration, learned decoders.** learned_decoder_constrained (fixtures), adult_learned_decoder, heldout_calibration (Adult, 10-07) | Same tokens; only the token→vector decoder changed. Learned per-token decoder D1 (κ=32) vs the mean-teacher decoder D0. Held-out per-token, global-temperature and per-class-temperature calibrators fitted on 2,000 held-out rows | P* (JOINT i8o64 λ0.1 + class temperature): pair 0.8137 vs calibrated T* 0.8456 (+0.0319 [0.0272, 0.0366]), but occupation LL vs U0 0.0075 [0.0031, 0.0120], precision not established. Temperature ties D0 (0.0013 [−0.0002, 0.0027]). D1 is worse held out (+0.014 nats occupation LL; 0/57 eligible) | **Utility cost.** The benefit belongs to the same frozen partition; changing the decoder removes no information. The fixture gate was unattainable because decisions carried zero SEX | Calibrate the twin the same way (calibrated U gains 0.0017). Decisions-only (CLASS) is the floor (Adult pair 0.739). Near-invertible relabelings are controls, not mechanisms |
| 7 | **Latent bottlenecks, finite codes.** decision_preserving_compression, confidence_capacity, confidence_budgeted_privacy (10-06); method_portfolio B (10-08); occupation_bottleneck (10-13) (Adult) | Each recipient receives a token (at most m states per predicted class) decoded to a mean vector, plus the unchanged decision. Codes: task-only, LOCAL, SEQ, JOINT with privacy weight λ; capacity i{4,8} × o{8..64}. Also VIB bottlenecks and a post-hoc head-row-space projection Pz | JOINT i8o64 λ0.1: pair 0.813 vs task-only T* 0.846 (+0.0336 [0.0286, 0.0387]) at occupation LL +0.0081 (UB 0.0121; 10/11 clauses). With confidence held (λ0.01): +0.0022 [0.0002, 0.0042]. Capacity frontier at i8o8: 0.820 at +0.021. Projection, RAW vs RAW-projected: +0.0107 [0.0004, 0.0210]. B-JVIB: 0.800 but fails occupation NLL (0.0108) | **Utility cost.** The benefit came with about +0.005 nats over T*; with spend held it falls below 0.020. Joint ≈ sequential (within 0.003) | The task-only code at the same capacity explains the confidence loss. LOCAL/SEQ codes match JOINT. Outputs-only readers carry most of the recovery (decisions alone 0.723) |
| 8 | **Grouped feature selection.** capacity_matched_feature_selection (10-08/09), utility_reference_repair (10-09), method_portfolio C (10-08) (Adult) | Per-recipient raw-group masks (delete one categorical group; keep 42 or 76 columns), with retrained heads + temperature | capacity_matched: 0/5 masks feasible and no recovery measured; the full-input F head itself fails U0 NLL by 0.045-0.060. reference_repair: J pair 0.8973 vs S21 0.8960 (−0.0013), and J and the task-only T sent the same pair. method_portfolio C: 0.801 for one recipient, but pair 0.897 | **Infeasible**, on a weak full-input baseline. Drops for one recipient vanish in the linked pair because the union covers all columns | Use a union-matched (linked-pair) control. Establish full-input feasibility first |
| 9 | **ACS income codes and transplants.** autonomous_method_development, frozen_finalist_validation (10-17), income_headroom_successor (10-18) (ACS FL) | Occupation is the stored JOINT_TREE output. Income is a 6-level code of 0.5·LOCAL + 0.5·JOINT tree log-odds, with level probabilities fitted on CALIBRATION (B3CTL). Successor: stacked HGB income + a budgeted transplant of the JOINT direction (β 0.25/0.20) + a Platt map | B3CTL, DEV linked pair 0.613/0.618: +0.027/+0.020 vs NEURAL_RAW (bounds 0.0135/0.0059) and +0.028/+0.025 vs H_LJ, at the utility edge (income Brier vs Ucal 0.0049/0.0049; s1 accuracy loss 0.0096). With the enhanced 16-recipe readers it is READER_SENSITIVE (H_LJ's income reader was picked on a near tie), and bootstrap-trained readers shrank the benefit in 3/4 cells. Successor H3: 0.6272/0.6263, +0.014/+0.017 vs H_LJ; component control +0.0020/−0.0026; 3/4 bootstrap refits reverse s1 | **Benefit below 0.020,** bought with about the whole income Brier budget. **Reader-sensitive.** Confirmation is infeasible: about 4.6M-60M households would be needed for 80% power, against about 17,500 | Missing controls: a continuous-mixture control (coarsening vs mixing never separated) and a matched temperature control. The enhanced reader bank is now mandatory. Recipient clauses can fail on reader near-ties |
| 10 | **Task-conditioned representations, small adapters.** task_conditioned_representations (10-12), minimal / two_recipient task-wire adapters (10-14/15), separate_pool_utility_guard (10-14), compressed_income_safeguard (10-13) (Adult) | Two encoders + logistic heads with an MMD conditional penalty (SEX matched within task labels and within aligned pair labels), strengths .3/1/3. Low-rank residual adapters in the head row space (325 + 65 parameters) under KL/NLL/Brier budgets. Continuation with a safeguard QP | Task-conditioned JOINT: pair 0.8619 vs U0 0.8701 (+0.0083 [0.0001, 0.0164]); +0.0048 vs the local twin; −0.0006 vs unconditional; −0.0051 vs RAW_JOINT, with the income view worse by 0.063 (VIOLATION). Adapter J_TRUST: +0.0049 [−0.0017, 0.0115] vs RAW_PROJ, and the local twin −0.0002. Two-recipient adapter: +0.0036. Separate-pool guard: held-out losses were 1.5-2× those on the guard pool | **Benefit below 0.020.** Local, unconditional and sequential twins reproduce it. A guard checked on a pool inside the objective did not certify held-out utility | Use local-twin, unconditional-twin and sequential controls. Check budgets on held-out people outside the objective |

**Cross-family notes**

1. **Spend explains the large drops.** Wherever pair recovery fell by at least 0.020 against eligible references, one of two things
   held (rows 1, 2, 5, 7, 9):
   - the release failed a point clause or guard; or
   - it spent more task utility than any SEX-blind or task-only control was evaluated at.

   With spend held, or at feasible points, the benefits were about 0.002-0.018 (rows 2, 3, 4, 7, 9, 10).
2. **Coalition terms rarely beat local or sequential versions.** The gaps were within 0.000-0.005 in rows 3, 4, 7 and 10. Row 1 is
   the exception (+0.012 and +0.029), but there the leakage moved to income.
3. **Leakage moves between recipients.** Examples: JOINT income in H_JJ; the task-conditioned income view (+0.063); P02 in row 1.
   Linked recovery (pair plus the ignore-one-recipient readers) and per-recipient recovery are both required.
4. **Readers decide near 0.01.**
   - Enhanced banks and reader near-ties reversed claims (row 9).
   - Coarse releases need LOOKUP-type readers (row 5).
   - Invertible relabelings (temperature, decoders) remove no information (row 6).
5. **Undertrained or weak baselines flattered candidates.** Examples: the Adult RAW_PROJ; the training-size effect; the Adult
   full-input F head. The twin must use the same people and the same effort.
6. **All evidence is development evidence.** MODEL_SELECT was used adaptively by every campaign, and a 0.020 benefit is not
   confirmable on about 17,500 households.

**Operations not found in the records read**

These are candidates either for new families or for simpler explanations:

1. SEX-blind coarsening of a full-size task-only twin's outputs, run as a strength path that brackets a candidate's spend. SCM's ORD
   path used SEX through the guard and never reached SCM's DEV spend.
2. Decision- and top-confidence-preserving flattening that replaces only the non-argmax coordinates with SEX-blind fitted class
   shares. This removes secondary-class information only.
3. A continuous mixture of two task-only models' outputs, with no coarsening (the control B3CTL lacked), and a temperature-matched
   control for transplants.
4. Any randomized release evaluated by readers on ACS FL. Randomized response on the decisions (SEX-blind flip probabilities) would be
   the simplest randomized baseline.
5. Coalition-aware correlated randomness across the two recipients. This was positive only on synthetic data in the replacement
   study (+0.061/+0.175/+0.302 at budgets 0/.002/.01).
6. Input-side operations on ACS FL: SEX-blind or SEX-aware generalization of the six raw fields, or concept erasure on the
   145-column encoding before an ordinary task model. Grouped masks were tried only on Adult.
7. Training-time SEX-balanced reweighting or resampling of MODEL_FIT people. The successor's family O was a post-hoc recalibration
   only.
8. The MMP mixture over SEX values (fit with SEX as an input, release a fixed-weight average over SEX). It exists data-free only;
   masking was never tested.

**File references** (under `results/`)

1. `pcrl_online_strength_frontier_v1/RESEARCH_DECISION.md` (l.17-25, 61, 142-148), `ALL_LEVELS.csv`;
   `pcrl_strength_matched_feedback_v1/RESEARCH_DECISION.md` (l.29-52); `pcrl_refreshed_guarded_joint_v1/RESEARCH_DECISION.md`
   (l.33-79); `pcrl_penalty_no_erasure_v1/RESEARCH_DECISION.md` (l.15-36), `CRITIC_GAP_DIAGNOSIS.md`;
   `pcrl_joint_complete_view_method_v1/RESEARCH_DECISION.md` (l.23-48).
2. `pcrl_controlled_training_size_v1/RESEARCH_DECISION.md` §4-6, §8 (table l.186-193);
   `pcrl_correction_transfer_stability_v1/RESEARCH_DECISION.md` §3-7 (l.166-171), `HANDOFF.json`;
   `pcrl_occupation_headroom_overnight_v1/RESEARCH_DECISION.md` (l.103-110, 195-230, 305-318).
3. `pcrl_acs_protected_tree_replication_v1/RESEARCH_DECISION.md` §1, §3-4, §8; `pcrl_acs_frozen_hybrids_v1/ADVISOR_BRIEF.md` §2-4
   (l.35-36); `pcrl_task_headroom_protected_outputs_v1/RESEARCH_DECISION.md` §2-4;
   `pcrl_frozen_tree_control_completion_v1/ADVISOR_BRIEF.md`.
4. `pcrl_latent_and_sampled_methods_v1/RESEARCH_DECISION.md` ("Sampled score channel"),
   `sampled_channel_v1/design/A_FIT_FEASIBILITY_SUMMARY.md`; `pcrl_incumbent_view_risk_v1/RESEARCH_DECISION.md`;
   `pcrl_conditional_budgeted_release_v1/RESEARCH_DECISION.md`, `COMPARISON_TABLE.md`;
   `pcrl_bank_anchored_release_v1/RESEARCH_DECISION.md`.
   - The residence studies are not in this worktree: local branches `research/pcrl-stochastic-channel-v1` (deb6992ea) and
     `research/pcrl-stochastic-replacement-overnight-v1` (e3415b94d), `RESEARCH_DECISION.md`.
5. `pcrl_scm_utility_matched_v1/RESEARCH_DECISION.md` §2, §6, `SELECTION_AND_MATCHING.md` (l.25-53), `RECOVERY_TABLE.csv`;
   `pcrl_scm_engineering_benchmark_v1/DECISION.md` (l.25, 39, 54), `RESULTS_TABLES.md` (l.22-57);
   `pcrl_structural_mechanism_pilot_v1/SELECTION_DECISION.md` (l.24-44, 72-92); `pcrl_mechanism_design_v1/CORRECTIONS.md`.
6. `pcrl_heldout_calibration_v1/RESEARCH_DECISION.md` (l.60-86); `pcrl_adult_learned_decoder_release_v1/RESEARCH_DECISION.md`
   §1, §3-4; `pcrl_learned_decoder_constrained_release_v1/RESEARCH_DECISION.md` §2-3.
7. `pcrl_confidence_capacity_v1/RESEARCH_DECISION.md` (l.39-70, 144), `CAPACITY_CURVE.csv`;
   `pcrl_confidence_budgeted_privacy_v1/RESEARCH_DECISION.md` (l.102-111);
   `pcrl_decision_preserving_compression_v1/RESEARCH_DECISION.md` (l.32-57); `pcrl_occupation_bottleneck_v1/BOTTLENECK_DECISION.md`;
   `pcrl_method_portfolio_v1/RESEARCH_DECISION.md`.
8. `pcrl_capacity_matched_feature_selection_v1/RESEARCH_DECISION.md`, `m2/FIT_SCREEN_SUMMARY.md`;
   `pcrl_utility_reference_repair_v1/RESEARCH_DECISION.md`.
9. `pcrl_autonomous_method_development_v1/START_HERE.md` (l.35), `RESEARCH_DECISION.md` (l.114; §8 l.239-247);
   `pcrl_frozen_finalist_validation_v1/RESEARCH_DECISION.md` (l.28-58, 104-131), `CONFIRMATION_FEASIBILITY.md` (l.54-74);
   `pcrl_income_headroom_successor_v1/RESEARCH_DECISION.md` (l.81-105, 145).
10. `pcrl_task_conditioned_representations_v1/RESEARCH_DECISION.md` ("Separate verdicts"), `COMPLETE_VIEW_RECOVERY.md`
    (l.15-27, 55-65); `pcrl_minimal_task_wire_adapter_v1/RESEARCH_DECISION.md`;
    `pcrl_two_recipient_task_wire_adapter_v1/RESEARCH_DECISION.md`; `pcrl_separate_pool_utility_guard_v1/FEASIBILITY_DECISION.md`;
    `pcrl_compressed_income_safeguard_v1/RESEARCH_DECISION.md`.
