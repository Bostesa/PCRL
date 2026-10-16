# Closed-lane helper report (verbatim content, paths made repository-relative)

A read-only helper agent compiled this from summary documents only (START_HERE, ADVISOR_BRIEF, RESEARCH_DECISION, FEASIBILITY_DECISION, HANDOFF). It did not open private data or arrays. The coordinator spot-checked six of its claims against the files (see `CLOSED_LANES.md`).

**Path conventions.**
- A worktree path `X/results/...` means the file in worktree `X`.
- `git show <branch>:<path>` marks files on branches that have no worktree.
- Tip dates are those of `git log -1`.

**Excluded from the ledger:**
- the publication and `-public` worktrees;
- the custody-only directories `pcrl-method-portfolio-custody-20261008*`;
- the git-permission probe worktree.

**Not extracted:**
- `combined-matched-removal-benchmark-v1` and `combined-stored-model-pilot-v1`, which are LEACE/noise benchmarks;
- the remaining `combined-*` worktrees, which are paper preparation;
- three earlier branches, of which only the headlines were read. All three closed negative:
  - full-view-protection, which ran 0 of its 27 planned fits;
  - task-directed-release;
  - objective-diagnosis.

**Column key:**
- **Pair: T+E** — the linked two-recipient coalition term was used in training and the pair was attacked. **E** — attacked only. **none** — no pair attack.
- **NN** — the neural adversarial control.

## Table 1. Closed lanes on Adult

| Lane | Campaign @ tip | Mechanism tried | Label | Measured bottleneck | Pair; NN | Evidence |
|---|---|---|---|---|---|---|
| 1 | refreshed-guarded-joint @ ccdcc5a37 | Joint penalty; critics refit every 4 epochs; AUC-budget local feedback (J-G) | EXPERIMENTAL_NO_ADVANTAGE | "The bottleneck is not critic staleness". Refit gap ≈0 vs 0.06–0.07 nats online, yet recovery did not fall. Penalty/task ratio 1.55 (J-O) vs 0.52. Coalition vs L-G +0.012 [0.006, 0.018], below 0.02 | T+E; J-O was C\*: 0.816 vs J-G 0.855 | pcrl-refreshed-guarded-joint-v1/results/pcrl_refreshed_guarded_joint_v1/RESEARCH_DECISION.md:15,35,52-53,62-80,146-150 |
| 2 | strength-matched-feedback @ a9951ed2f | Normalised strength; online/refreshed/matched schedules; AUC controller | EXPERIMENTAL_NO_ADVANTAGE | Schedules tie at matched strength (REFRESHED−ONLINE −0.003 [−0.008, 0.003]). Controller switch-like, no benefit. Grid ≤1.5 never reached the raw ratio 2.6–3.2. Vs L-F 0.006 | T+E; RAW-J β0.3 = C\*, beat J-F by 0.054 | pcrl-strength-matched-feedback-v1/results/pcrl_strength_matched_feedback_v1/RESEARCH_DECISION.md:20,25,47,56-74,94-104,165-172 |
| 2 | online-strength-frontier @ 925e0fddf | Raw β 0.1/0.3/0.6; normalised ρ 1.5/3/5; allocations a 0.5/2; joint vs local | EXPERIMENTAL_NO_ADVANTAGE (complete) | "measured limit is the occupation-accuracy cost of removing SEX recovery, not a shortage of update strength". NORM-J ρ3 vs RAW-J −0.016 [−0.021, −0.011] at −0.5 occupation points, inner-infeasible. Allocation only moved leakage | T+E; incumbent RAW-J (pair 0.802) best | pcrl-online-strength-frontier-v1/results/pcrl_online_strength_frontier_v1/RESEARCH_DECISION.md:5,20-22,44-53,64-73,142-148 |
| 2 | penalty-no-erasure @ b628ff5b9 | Joint coalition+local penalty, mandatory erasure removed (PN vs LN) | EXPERIMENTAL_NO_ADVANTAGE | Income-view guard P02 +0.010 [0.001, 0.019] vs <0.01. Seed 0 not a nominee. Coalition +0.029 passed. Online critics weaker than fresh by +0.047 nats (45/45) | T+E; C\* = LN | pcrl-penalty-no-erasure-v1/results/pcrl_penalty_no_erasure_v1/RESEARCH_DECISION.md:3,8-9,17-18,54-57,99-103 |
| 2 | joint-complete-view-method @ 568f97068 | Encoders → official erasure → heads; coalition critic; projected step | Claims 1, 2 NOT_ESTABLISHED | Erasure alone costs 4.9 income / 3.7 occupation points, so no arm was feasible. J−L 0.017 [0.009, 0.024] | T+E; plain penalty 0.040 lower than J | pcrl-joint-complete-view-method-v1/results/pcrl_joint_complete_view_method_v1/RESEARCH_DECISION.md:13-17,43,82-83,145 |
| 3 | task-conditioned-representations @ b21c4cc5e | Conditional MMD-B-Fair (SEX matched within task labels), local + pair | COMPLETE_EXPLORATORY_NEGATIVE_COMPARISON | Vs task-only 0.008252 [0.000111, 0.016393]. Conditioning −0.000635 [−0.006684, 0.005414]. Income +0.063 vs RAW_JOINT. 9.57% of pair mass unsupported | T+E; RAW_JOINT lower | pcrl-task-conditioned-representations-v1/results/pcrl_task_conditioned_representations_v1/RESEARCH_DECISION.md:5,13-17,24,30 |
| 4 | separate-pool-utility-guard @ 968c5e8a7 | Occupation-only RAW-J continuation; KKT-projected steps under NLL/Brier budgets on a 2,043-row guard pool | COMPLETE_NEGATIVE_FEASIBILITY (privacy NOT_TESTED) | Guard Brier held at +0.005, but validation occupation NLL +0.012 to +0.016 and Brier +0.006 to +0.008. Held-out loss grew 1.5–2× faster. Epoch 0 selected | no AUC run | pcrl-separate-pool-utility-guard-v1/results/pcrl_separate_pool_utility_guard_v1/FEASIBILITY_DECISION.md:3,13-26,30-56 |
| 4 | compressed-income-safeguard @ b58b47224 | Occupation-only minibatch NLL/Brier safeguard over frozen income; pair term | NO_COMPETITIVE_ADVANTAGE_ESTABLISHED | Every protection checkpoint failed the FIT2 proxy, so epoch 0 was selected. Vs RAW_PROJ −0.0019 [−0.0111, 0.0074]. Epoch-20 ATT 0.862→0.797 at FIT2 NLL +0.054/+0.057 | T+E; RAW_PROJ strongest | pcrl-compressed-income-safeguard-v1/results/pcrl_compressed_income_safeguard_v1/RESEARCH_DECISION.md:3,22,28,36,40 |
| 5 | minimal-task-wire-adapter @ 059c5a60e | 325-parameter trust-region occupation correction; income frozen | POINT_FEASIBLE_NO_COMPETITIVE_CRITERION | Pair 0.0049 [−0.0017, +0.0115], established shortfall vs 0.020. Occupation-only −0.010, but L_TRUST equal. "frozen income recipient plus the occupation predictions still reveal SEX" | T+E (one recipient); RAW_PROJ | pcrl-minimal-task-wire-adapter-v1/results/pcrl_minimal_task_wire_adapter_v1/RESEARCH_DECISION.md:4,12,14,33,49,104 |
| 5 | two-recipient-task-wire-adapter @ ee67c29ee | Adapters on both recipients; joint/local/S12/S21 | POINT_FEASIBLE_NO_COMPETITIVE_CRITERION | Pair 0.0036 [−0.0025, +0.0097]. Simultaneous utility 5/24 vs RAW_PROJ 10/24. Income KL budget never active | T+E; RAW_PROJ | pcrl-two-recipient-task-wire-adapter-v1/results/pcrl_two_recipient_task_wire_adapter_v1/RESEARCH_DECISION.md:4,10,31-32,38,55,106-108 |
| 6 | capacity-matched-feature-selection @ 9e6bfea62 | Five grouped-feature masks, retrained heads | INCOMPLETE_MATCHED_COMPARISON | 0 feasible masks. Full-input F head NLL excess 0.045–0.060 vs 0.010 | none | pcrl-capacity-matched-feature-selection-v1/results/pcrl_capacity_matched_feature_selection_v1/RESEARCH_DECISION.md:3,5 |
| 6 | utility-reference-repair @ ce1308a52 | Reference refit, then a 5-mask T/L/J/S12/S21 screen | UTILITY_FEASIBILITY_ESTABLISHED; UTILITY_PRESERVATION_UNRESOLVED | J picked the same pair as task-only. Control−J −0.0013 vs +0.020 | T+E | pcrl-utility-reference-repair-v1/results/pcrl_utility_reference_repair_v1/RESEARCH_DECISION.md:3,9,19 |
| 7 | decision-preserving-compression @ 0a7b05a52 | Decision-preserving codes, joint/local/sequential | EXPERIMENTAL_NO_ADVANTAGE | Finest code +0.020 nats occupation log loss, +0.007 Brier. Joint vs task-only 0.018 [0.014, 0.021]. Decisions alone 0.74 | T+E | pcrl-decision-preserving-compression-v1/results/pcrl_decision_preserving_compression_v1/RESEARCH_DECISION.md:12-18,26,70,91,105 |
| 7 | confidence-capacity @ d0c8a45c8 | Capacity ladder to i8o64; privacy-trained codes | CONFIDENCE_FEASIBILITY_ESTABLISHED | Q\* passes confidence but pair 0.849 vs 0.858. P\* −0.034 pair, but occupation log loss +0.0081, upper bound 0.0121. Joint−sequential 0.003 | T+E; RAW-J 0.788, occupation log loss +0.0134 | pcrl-confidence-capacity-v1/results/pcrl_confidence_capacity_v1/RESEARCH_DECISION.md:3,10-25,32,107,120 |
| 7 | confidence-budgeted-privacy @ 7f3ec67b2 | λ 0.01–0.1 at i8o64 with headroom selection | CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION | "Headroom passed, but the privacy margin disappeared": 0.00217 [0.00019, 0.00415] vs >0.02 | T+E | pcrl-confidence-budgeted-privacy-v1/results/pcrl_confidence_budgeted_privacy_v1/RESEARCH_DECISION.md:3,85,134,188-199,247 |
| 7 | occupation-bottleneck @ b0243f6a5 | Leakage diagnosis; repair pilot never triggered | NOT_TRIGGERED_DIAGNOSIS_UNRESOLVED | RAW projection pair 0.0107 [0.0004, 0.0210]. Logits-only 0.841, decisions 0.723 | E | pcrl-occupation-bottleneck-v1/results/pcrl_occupation_bottleneck_v1/RESEARCH_DECISION.md:16,25,31,39,45 |
| 7 | latent-and-sampled-methods @ 3e742098d | 10-bit codes + unconditional MMD repair; sampled confidence channel | NONE_ESTABLISHED | −0.034 recovery, but occupation NLL +0.032, Brier +0.015. All 10-bit arms fail utility | T+E | pcrl-latent-and-sampled-methods-v1/results/pcrl_latent_and_sampled_methods_v1/RESEARCH_DECISION.md:12-18,24 |
| 8 | bank-anchored-release @ dda8e450c | Re-route codes of a frozen bank | EXPLORATORY_NEGATIVE_COMPETITIVE_COMPARISON | Best P vs B0 0.006. J 0.004 | T+E | pcrl-bank-anchored-release-v1/results/pcrl_bank_anchored_release_v1/RESEARCH_DECISION.md:11,32-34 |
| 8 | code-utility-witness @ 4723190ff | Hard-bit latent learner gating a conditional-kernel repair | LEARNER_RECIPE_NOT_UTILITY_FEASIBLE | B16 passed 21/24. Kernel stage never ran | none | pcrl-code-utility-witness-v1/results/pcrl_code_utility_witness_v1/RESEARCH_DECISION.md:9,13 |
| 8 | conditional-budgeted-release @ ecf01fe17 | LP frozen-bank change kernels | NO_COMPETITIVE_METHOD_OR_JOINT_ADVANTAGE_ESTABLISHED | W vs P0 0.0020 [−0.0037, 0.0076]. S12 lower than W | E | pcrl-conditional-budgeted-release-v1/results/pcrl_conditional_budgeted_release_v1/RESEARCH_DECISION.md:7,13,17 |
| 8 | incumbent-view-risk @ 320c07c81 | Complete-view joint/product/local/sequential policies from exact P0 | Claims A/B/C NOT_ESTABLISHED | Best reduction 0.0024 (S21) vs 0.020 | T+E | pcrl-incumbent-view-risk-v1/results/pcrl_incumbent_view_risk_v1/RESEARCH_DECISION.md:19-31,37 |
| 9 | learned-decoder-constrained-release @ 091afc200 | Certified learned decoder (synthetic fixtures only) | MECHANISM_GATE_NOT_MET | Trigger unattainable: decision-only pair MI 0 on every fixture | fixtures | pcrl-learned-decoder-constrained-release-v1/results/pcrl_learned_decoder_constrained_release_v1/RESEARCH_DECISION.md:3,58-81 |
| 9 | adult-learned-decoder-release @ 976202546 | D1 on frozen codes | CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION | D1 raised held-out occupation log loss +0.0102 [0.0081, 0.0123]. P\* upper bound 0.0121 | T+E | pcrl-adult-learned-decoder-release-v1/results/pcrl_adult_learned_decoder_release_v1/RESEARCH_DECISION.md:3,21-24,39-42,49-53 |
| 9 | heldout-calibration @ 1baf5bbda | Held-out per-token / temperature decoders | NO_COMPETITIVE_RELEASE_CRITERION_ESTABLISHED | Pair 0.0319 PASS, but occupation log loss 0.0075 [0.0031, 0.0120] vs U0 and 0.0092 [0.0060, 0.0124] vs Ucal | E | pcrl-heldout-calibration-v1/results/pcrl_heldout_calibration_v1/RESEARCH_DECISION.md:3,65-73,104-107 |
| 9 | confidence-constrained-mechanism @ d85f0a6f4 | Pointwise-guarded score channel | PREMISE_NOT_SUPPORTED | Occupation ≤36.6% coverage, 85% fallback (identity release) | none | pcrl-confidence-constrained-mechanism-v1/results/pcrl_confidence_constrained_mechanism_v1/RESEARCH_DECISION.md:3,13-16,32-44 |
| 11 | task-headroom-protected-outputs @ 2c2be7830 | CatBoost teachers + local leaf-score correction | POINT_FEASIBLE_NO_COMPETITIVE_CRITERION; JOINT_COMPONENT_NOT_ESTABLISHED | Vs historical code 0.0177 [−0.0064, 0.0419]. J_TREE−L_TREE −0.0155 | T+E; beats RAW_PROJ 0.0505 [0.0257, 0.0754] | pcrl-task-headroom-protected-outputs-v1/results/pcrl_task_headroom_protected_outputs_v1/RESEARCH_DECISION.md:3-6,44-48,62-77,92-99 |
| 11 | frozen-tree-control-completion @ 5ca95e47b | Missing controls and power for frozen L_TREE | ORIGINAL_CONFIRMATION_CRITERION_NOT_PRACTICALLY_FEASIBLE | Claim B needs ≈317,000 groups, set by occupation Brier slack 0.000775 | E; fresh RAW-J beaten by 0.066 [0.045, 0.088] | pcrl-frozen-tree-control-completion-v1/results/pcrl_frozen_tree_control_completion_v1/ADVISOR_BRIEF.md:3,14-35 |
| 11 | method-portfolio (v1 and a–d) @ 38d641189 | Sampled channel (not run), stochastic VIB, masks, latent repair | DEVELOPMENT_SHORTFALL_INDEPENDENTLY_REPLAYED | B-JVIB fails occupation caps. C-INDEP1 coalition 0.897 (union = all 83 columns) | T+E; B-RAWJ unbeaten | pcrl-method-portfolio-v1/results/pcrl_method_portfolio_v1/RESEARCH_DECISION.md:3,7-13,18-22 |
| other | combined-output-aware-removal @ f7425b15c; combined-output-diagnosis @ dc6a2e961 (Adult and HMDA) | Benchmarks of erasure/FARE/noise/output contracts | "No new algorithm is suggested" | Clean outputs bypass feature protection. Hard decision 0.513 vs logits 0.776. Pair beats both singles (0.917 vs 0.845) | mostly single-recipient | combined-output-aware-removal-v1/results/combined_output_aware_removal_v1/RESEARCH_DECISION.md:64-66,116-117,203; combined-output-diagnosis-v1/results/combined_output_diagnosis_v1/RESEARCH_DECISION.md:76-83 |

## Table 2. Closed lanes on ACS data

| Lane | Campaign @ tip | Data | Mechanism tried | Label | Measured bottleneck | Pair; NN | Evidence |
|---|---|---|---|---|---|---|---|
| 1 | direct-adversarial (branch only) @ 106de9afa | ACS 2018 dev | Refreshed multi-attacker refinement of J; C1 vs local L1/L2 | "nominee conjunction is UNSUPPORTED" | Refreshed attacker kept 84/1350 (6%). C1 beats locals, but residence +0.00717 [+0.0031, +0.0113] | T+E; frozen neural J not beaten | git show research/pcrl-direct-adversarial-v1:results/pcrl_direct_adversarial_v1/RESEARCH_DECISION.md L5-9,39-48,62-71,94-103 |
| 8 | stochastic channel RCSR (branch only) @ cd895e429 | ACS 2018 dev | Role-constrained stochastic release appended to J | CLOSED_NEGATIVE_at_G2_capacity | Advantage over J −0.00268 (64 states), −0.01136 (256). Mechanism never constrained. Its "subsumption" claim was later withdrawn | none (utility only) | git show research/pcrl-stochastic-channel-v1:results/pcrl_stochastic_channel_v1/RESEARCH_DECISION.md L3-9,41-42 |
| 8 | stochastic replacement (branch only) @ e3415b94d | ACS 2018 dev | Replacement release [H_A, Z]; 64/256-state codes | "No competitive operating point was established" | Best deficit vs J +0.00576 vs .001. Quantisation cost +0.02109 ("The failure is the coder, not the probe") | none | git show research/pcrl-stochastic-replacement-overnight-v1:results/pcrl_stochastic_replacement_overnight_v1/RESEARCH_DECISION.md L3-5,35-42,82-90,103-110 |
| 10 | acs-frozen-hybrids @ 54f089d3e | ACS FL 2019 | LOCAL income + JOINT occupation | H_LJ_FIXED_COMPOSITION_NOT_SUPPORTED | Vs TASK_REFIT 0.022/0.024, clause bounds 0.012/0.014. Larger drops need JOINT income (fails utility) | E; NEURAL_RAW lower by 0.001/0.005 | pcrl-acs-frozen-hybrids-v1/results/pcrl_acs_frozen_hybrids_v1/START_HERE.md:3-7,37-40 |
| 11 | acs-protected-tree-replication @ 7eb9f2beb | ACS FL 2019 | Transported L_TREE | RECIPE_EFFECT_NOT_ESTABLISHED; NOT_SUPERIOR_TO_ALL_ELIGIBLE; SUBGROUP_UNRESOLVED | Each recipient fell, but coalition only +0.007/+0.0175. Pair keeps 91–96% of the teacher's above-chance AUC | E; NEURAL_RAW lower by 0.0167/0.0112 | pcrl-acs-protected-tree-replication-v1/results/pcrl_acs_protected_tree_replication_v1/RESEARCH_DECISION.md:11-19,87-100,114 |
| 11 | adaptive-release @ 2ce5d171f | ACS 2018 dev | 17-token release; utility and privacy routes | EXPERIMENTAL_NO_ADVANTAGE | Privacy route residence +0.009539 violates the +0.001 guard | E | pcrl-adaptive-release-v1/results/pcrl_adaptive_release_v1/RESEARCH_DECISION.md:3-7,11 |
| 11 | shared-context-release @ 537e74a44 | ACS CA 2018 | Richer-input context policies mixed by LP | EXPERIMENTAL_NO_ADVANTAGE | Task +.00781 (adverse); "no usable additional task information" | E | pcrl-shared-context-release-v1/results/pcrl_shared_context_release_v1/RESEARCH_DECISION.md:3-7,46-49,63,73,84 |
| 11 | privacy-first-selector @ 8fdc61e39 | ACS CA 2018 | Privacy-first selector; randomised variant | NOT_ESTABLISHED | D4 −.00363 is exactly the H-only floor; task +.00264 vs +.001 cap | E | pcrl-privacy-first-selector-v1/results/pcrl_privacy_first_selector_v1/RESEARCH_DECISION.md:3-10,23-37,105 |
| other | six earlier branches (competitive-method, nonlinear-rank, invariant-baselines, utility-extension, task-aligned-cuts, final-prospective) | ACS 2018 / 2016 | Projections and refinements of J; invariant objective; appended channel; constraint exchange; prospective test | "No competitive tradeoff"; no-go; "failed scientifically"; "Tier 2 (pilot), FAIL"; EXPERIMENTAL_NO_ADVANTAGE | Residence guard binding; task clauses 0/2 | E; frozen J never beaten | each branch's RESEARCH_DECISION.md |

## Patterns that the files themselves state

1. **Occupation confidence binds on Adult.** The capacity, calibration and budget lines all show it. In the separate-pool guard, Brier bound. In the frozen-tree power analysis, occupation Brier slack sets the sample size.
2. **Update strength, not critic freshness, moved recovery,** and more strength cost occupation accuracy. The exception is penalty-no-erasure, whose report names weak online critics.
3. **Joint design never beat local or sequential** in any lane that tested it.
4. **Recipient gains did not carry to the linked pair.**
5. **Ordinary neural adversarial joint training was strongest or unbeaten,** except the Adult L_TREE.
6. **Privacy-trained score codes** buy about 0.03 pair AUC at an occupation log-loss upper bound near 0.012.
7. **Decisions alone set a floor** for decision-preserving releases (0.72–0.74 on Adult).
8. **Fitting-pool checks did not hold out:** guards, safeguards, learned decoders and per-token calibration.
9. **Frozen income leaves 10–12 income clauses unresolved** in the adapter studies.
10. **Conditional kernels:** only task-conditioned-representations ran one, and conditioning contributed −0.000635. No campaign used an HSIC penalty.
11. **In the ACS 2018 lineage the residence guard or margin was the binding clause.**
12. **The pair was attacked** in every Adult and ACS Florida campaign that reached the attack stage. There was no pair attack in:
    - separate-pool;
    - feature selection;
    - code-utility-witness;
    - the confidence-constrained mechanism;
    - learned-decoder fixtures;
    - RCSR and replacement;
    - most of the combined-\* campaigns.
