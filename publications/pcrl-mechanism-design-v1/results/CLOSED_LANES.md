# Closed lanes: measured bottlenecks (decision aid)

This is a short decision aid, not a new audit. Each row names a lane, what was tried and the bottleneck the original report measured.

**Evidence.** The evidence column gives `branch @ tip : file : lines`. A read-only helper compiled the rows from each campaign's own summary documents. The coordinator spot-checked six claims against the files before writing this page: the online-strength frontier, the strength-matched schedules, the minimal adapter's frozen income, task-headroom's 0.0505, the separate-pool guard table, and confidence-capacity's 0.034 / 0.0121. Every other number is quoted from the file named.

**Conventions.**
- Unless stated otherwise, the Adult rows use the same utility caps as ACS: accuracy loss ≤ 0.010, NLL ≤ 0.010, Brier ≤ 0.005 against U0 and Ucal, and a 0.020 coalition margin.
- "Pair" means the linked two-recipient coalition view.
- The neural adversarial control (NN) is RAW-J / RAW_PROJ / J-O on Adult, J on ACS 2018 and NEURAL_RAW on ACS Florida 2019.

| Lane (prompt name) | What was tried (campaign; data) | Measured bottleneck as reported | Pair? NN result | Evidence |
|---|---|---|---|---|
| **Critic refresh** | Critics refit every 4 epochs, plus AUC-budget feedback (refreshed guarded joint; Adult) | "The bottleneck is not critic staleness". The refit gap was ≈0 against 0.06–0.07 nats online, yet recovery did not fall. Coalition vs the local twin was +0.012 [0.006, 0.018], below 0.020 | Pair trained and attacked. Online-critic J-O stayed the best (0.816 vs 0.855) | `research/pcrl-refreshed-guarded-joint-v1 @ ccdcc5a37 : results/pcrl_refreshed_guarded_joint_v1/RESEARCH_DECISION.md : 15, 35, 52–53, 62–80` |
| | Refreshed multi-attacker ensemble refinement of J (direct adversarial; ACS 2018) | The refreshed attacker was kept in 84/1350 decisions (6%). The best arm was significantly worse than J on two audits | The frozen neural J was not beaten | `research/pcrl-direct-adversarial-v1 @ 106de9afa : results/pcrl_direct_adversarial_v1/RESEARCH_DECISION.md : 5–9, 39–48` |
| **Penalty strength / allocation / schedule** | Normalised strength, online/refreshed/matched schedules, AUC controller (strength-matched feedback; Adult) | Schedules tie at matched strength: REFRESHED − ONLINE −0.003 [−0.008, 0.003]. The controller had no benefit | RAW-J β0.3 beat the new arm by 0.054 | `research/pcrl-strength-matched-feedback-v1 @ a9951ed2f : results/pcrl_strength_matched_feedback_v1/RESEARCH_DECISION.md : 56–74` |
| | Raw β grid, normalised ρ grid, fixed allocations a ∈ {0.5, 2}, joint vs local (online strength frontier; Adult) | "The measured limit is the occupation-accuracy cost of removing SEX recovery, not a shortage of update strength". "Allocation moved leakage between recipients, or traded it for occupation accuracy" | The incumbent RAW-J (pair 0.802) stayed the best | `research/pcrl-online-strength-frontier-v1 @ 925e0fddf : results/pcrl_online_strength_frontier_v1/RESEARCH_DECISION.md : 64–73, 144` |
| | Joint coalition-plus-local penalty with the mandatory erasure removed (penalty, no erasure; Adult) | The income-view guard was +0.010 [0.001, 0.019] against a limit below 0.01. Online critics were weaker than fresh ones by +0.047 nats (45/45) | Both C\* candidates had lower recovery but were utility-infeasible | `research/pcrl-penalty-no-erasure-v1 @ b628ff5b9 : results/pcrl_penalty_no_erasure_v1/RESEARCH_DECISION.md : 8–9, 17–18, 54–57` |
| | Erasure, then heads, with a coalition critic and projected steps (joint complete view; Adult) | Erasure alone cost 4.9 income and 3.7 occupation points, so no arm was feasible | The plain penalty was 0.040 below J | `research/pcrl-joint-complete-view-method-v1 @ 568f97068 : results/pcrl_joint_complete_view_method_v1/RESEARCH_DECISION.md : 13–17, 43` |
| **Conditional kernels** | MMD-B-Fair conditioned on task labels and aligned label pairs, local plus pair (task-conditioned representations; Adult). The only conditional-kernel run | The conditioning contribution was −0.000635 [−0.006684, 0.005414]. Vs the task-only twin 0.008. Income +0.063 vs RAW_JOINT against a cap of 0.010 | RAW_JOINT was lower | `research/pcrl-task-conditioned-representations-v1 @ b21c4cc5e : results/pcrl_task_conditioned_representations_v1/RESEARCH_DECISION.md : 5, 13–17` |
| **Utility guards** | Occupation-only continuation, with KKT-projected steps under NLL and Brier budgets on a separate 2,043-row guard pool (separate-pool guard; Adult) | The guard held on its own pool, but held-out occupation NLL rose +0.012 to +0.016 and Brier +0.006 to +0.008 by epochs 1–5. Epoch 0 was selected | Privacy was not tested | `research/pcrl-separate-pool-utility-guard-v1 @ 968c5e8a7 : results/pcrl_separate_pool_utility_guard_v1/FEASIBILITY_DECISION.md : 13–26` |
| | Occupation-only minibatch NLL/Brier safeguard over a frozen income output (compressed income safeguard; Adult) | Every protection checkpoint failed the held-out proxy, so epoch 0 was selected | RAW_PROJ was the strongest eligible | `research/pcrl-compressed-income-safeguard-v1 @ b58b47224 : results/pcrl_compressed_income_safeguard_v1/RESEARCH_DECISION.md : 3, 22, 28` |
| **Small adapters** | A 325-parameter occupation-head correction with income frozen (minimal task-wire adapter; Adult) | Pair benefit 0.0049 [−0.0017, 0.0115], an established shortfall against 0.020. The occupation-only drop did not survive into the pair, "because the frozen income recipient plus the occupation predictions still reveal SEX" | RAW_PROJ was the reference | `research/pcrl-minimal-task-wire-adapter-v1 @ 059c5a60e : results/pcrl_minimal_task_wire_adapter_v1/RESEARCH_DECISION.md : 4, 12, 14, 33` |
| | Adapters on both recipients, joint/local/S12/S21 (two-recipient adapter; Adult) | Pair 0.0036 [−0.0025, 0.0097]. The income KL budget was never active | RAW_PROJ was the reference | `research/pcrl-two-recipient-task-wire-adapter-v1 @ ee67c29ee : results/pcrl_two_recipient_task_wire_adapter_v1/RESEARCH_DECISION.md : 4, 10, 31–32` |
| **Feature masks** | Five grouped-feature masks with retrained heads (capacity-matched feature selection; Adult) | 0 feasible masks. The full-input head's NLL excess was 0.045–0.060 against 0.010 | No pair test | `research/pcrl-capacity-matched-feature-selection-v1 @ 9e6bfea62 : results/pcrl_capacity_matched_feature_selection_v1/RESEARCH_DECISION.md : 3, 5` |
| | Masks with stochastic VIB and latent repair (method portfolio teams; Adult) | B-JVIB failed the occupation caps. The independent masks' union covered all 83 columns, giving coalition 0.897 | B-RAWJ was unbeaten | `research/pcrl-method-portfolio-v1 @ 38d641189 : results/pcrl_method_portfolio_v1/RESEARCH_DECISION.md : 3, 7–13` |
| **Latent bottlenecks / codes** | Decision-preserving codes (Adult) | The finest code cost +0.020 nats of occupation log loss and +0.007 Brier. Decisions alone gave pair 0.74 | — | `research/pcrl-decision-preserving-compression-v1 @ 0a7b05a52 : results/pcrl_decision_preserving_compression_v1/RESEARCH_DECISION.md : 12–18, 26` |
| | Capacity ladder to the i8o64 code, plus privacy-trained codes (confidence capacity; Adult) | P\* lowered pair recovery by 0.034 [0.029, 0.039], but its occupation log loss was +0.0081 with upper bound 0.0121 against 0.010. Joint − sequential was 0.003 | Pair trained and attacked | `research/pcrl-confidence-capacity-v1 @ d0c8a45c8 : results/pcrl_confidence_capacity_v1/RESEARCH_DECISION.md : 20–21, 40–50` |
| | λ grid at i8o64 with headroom selection (confidence-budgeted privacy; Adult) | "Headroom passed, but the privacy margin disappeared": 0.00217 [0.00019, 0.00415] against 0.02 | Pair trained and attacked | `research/pcrl-confidence-budgeted-privacy-v1 @ 7f3ec67b2 : results/pcrl_confidence_budgeted_privacy_v1/RESEARCH_DECISION.md : 3, 85, 188–199` |
| | 10-bit codes plus MMD repair (latent and sampled methods; Adult) | −0.034 recovery, but occupation NLL +0.032 and Brier +0.015. Every 10-bit arm failed utility | Pair trained and attacked | `research/pcrl-latent-and-sampled-methods-v1 @ 3e742098d : results/pcrl_latent_and_sampled_methods_v1/RESEARCH_DECISION.md : 12–18` |
| | Occupation leakage diagnosis; the repair pilot never triggered (occupation bottleneck; Adult) | RAW projection pair 0.0107 [0.0004, 0.0210]. Decisions alone 0.723 | RAW 0.3 was the strongest | `research/pcrl-occupation-bottleneck-v1 @ b0243f6a5 : results/pcrl_occupation_bottleneck_v1/RESEARCH_DECISION.md : 16–45` |
| **Frozen-bank channels** | Re-routing the codes of a frozen bank (bank-anchored release; Adult) | Best vs B0 0.006 | — | `research/pcrl-bank-anchored-release-v1 @ dda8e450c : results/pcrl_bank_anchored_release_v1/RESEARCH_DECISION.md : 11, 32–34` |
| | LP change kernels over a frozen bank (conditional budgeted release; Adult) | W vs P0 0.0020 [−0.0037, 0.0076]. S12 was lower than W | — | `research/pcrl-conditional-budgeted-release-v1 @ ecf01fe17 : results/pcrl_conditional_budgeted_release_v1/RESEARCH_DECISION.md : 7, 13, 17` |
| | Complete-view joint/product/local/sequential policies from exact P0 (incumbent view risk; Adult) | Best reduction 0.0024 (S21) against 0.020 | — | `research/pcrl-incumbent-view-risk-v1 @ 320c07c81 : results/pcrl_incumbent_view_risk_v1/RESEARCH_DECISION.md : 19–31` |
| | Role-constrained stochastic channel; replacement channel (ACS 2018) | Advantage over J −0.003 / −0.011, and the mechanism never constrained anything. The replacement coder's quantisation cost +0.021 ("the coder, not the probe") | J was not beaten | `research/pcrl-stochastic-channel-v1 @ cd895e429` and `research/pcrl-stochastic-replacement-overnight-v1 @ e3415b94d : RESEARCH_DECISION.md` |
| **Decoder / calibration rescue** | A certified learned decoder on synthetic fixtures (learned-decoder constrained release) | The trigger could not be met: decision-only pair MI was 0 on every fixture | Fixtures only | `research/pcrl-learned-decoder-constrained-release-v1 @ 091afc200 : results/…/RESEARCH_DECISION.md : 58–81` |
| | Learned decoder on frozen codes (Adult) | Held-out occupation log loss rose +0.0102 [0.0081, 0.0123] | — | `research/pcrl-adult-learned-decoder-release-v1 @ 976202546 : results/…/RESEARCH_DECISION.md : 21–24` |
| | Held-out per-token and temperature decoders (Adult) | Pair 0.0319 PASS, but occupation log-loss upper bounds were 0.0120 and 0.0124 | — | `research/pcrl-heldout-calibration-v1 @ 1baf5bbda : results/…/RESEARCH_DECISION.md : 65–73` |
| | Pointwise-guarded score channel (Adult) | Occupation coverage ≤ 36.6%, i.e. essentially an identity release | — | `research/pcrl-confidence-constrained-mechanism-v1 @ d85f0a6f4 : results/…/RESEARCH_DECISION.md : 13–16` |
| | Reference refit plus a T/L/J/S12/S21 mask screen (utility reference repair; Adult) | J picked the same pair as task-only. Control − J −0.0013 | — | `research/pcrl-utility-reference-repair-v1 @ ce1308a52 : results/…/RESEARCH_DECISION.md : 3, 9, 19` |
| **Protected trees** | CatBoost teacher plus local leaf-score correction (task headroom; Adult) | Beat RAW_PROJ by 0.0505 [0.0257, 0.0754] (established). Vs historical score codes 0.0177 [−0.0064, 0.0419]. J_TREE vs L_TREE unresolved | **Beat the Adult NN control** | `research/pcrl-task-headroom-protected-outputs-v1 @ 2c2be7830 : results/…/RESEARCH_DECISION.md : 3–6, 62` |
| | Controls and power for the frozen L_TREE (Adult) | Confirmation would need ≈317,000 groups, set by an occupation Brier slack of 0.000775 | Fresh RAW-J beaten by 0.066 | `research/pcrl-frozen-tree-control-completion-v1 @ 5ca95e47b : results/…/ADVISOR_BRIEF.md : 3, 14–35` |
| | Transported L_TREE (ACS FL 2019) | Each recipient fell, but the coalition fell only +0.007 / +0.0175, keeping 91–96% of the teacher's above-chance pair AUC | **NEURAL_RAW lower by 0.017 / 0.011** | `research/pcrl-acs-protected-tree-replication-v1 @ 7eb9f2beb : results/…/RESEARCH_DECISION.md : 11–19, 87–100` |
| **Frozen hybrids** | LOCAL income plus JOINT occupation (ACS FL 2019) | 0.022 / 0.024 vs TASK_REFIT, with clause bounds 0.012 / 0.014. Larger drops need JOINT income, which fails utility | NEURAL_RAW lower by 0.001 / 0.005 | `research/pcrl-acs-frozen-hybrids-v1 @ 54f089d3e : results/pcrl_acs_frozen_hybrids_v1/START_HERE.md : 3–7, 37–40` |
| **Other ACS-2018 lanes** | Adaptive release, shared-context release, privacy-first selector, competitive method, nonlinear rank, invariant baselines, utility extension, task-aligned cuts, final prospective | The residence task guard bound in each, and audits saturated at the route floor. No competitive trade-off was found | Neural J was never beaten | Each branch's `RESEARCH_DECISION.md` (see the helper's ledger in `roles/coordinator/CLOSED_LANES_HELPER_REPORT.md`) |

## What the closed lanes jointly say, as their own reports state it

1. **Utility binds before protection runs out.**
   - On Adult, occupation confidence is the binding clause: privacy-trained codes buy about 0.03 pair AUC but land at an occupation log-loss upper bound near 0.012.
   - On ACS Florida 2019, income Brier and accuracy bind: JOINT_TREE, and every NEURAL_RAW configuration stronger than ρ1 E2.
   - "Update strength, not critic freshness, moved recovery", and more strength cost task utility.
2. **Recipient-level gains did not carry to the linked pair** (minimal adapter, ACS replication, portfolio masks). A frozen recipient's own disclosure plus the other output still revealed SEX.
3. **Joint design never beat local or sequential designs** in any lane that tested it. This is consistent with the joint-versus-sequential fact in `PROBLEM_SPECIFICATION.md` §9: joint and sequential designs differ only through the algorithm, not through what they can express.
4. **Ordinary neural adversarial joint training was the strongest or unbeaten comparator in every lane but one.**
   - The exception is the Adult protected-tree recipe (L_TREE), which beat RAW_PROJ by 0.0505 and a fresh RAW-J by 0.066. Its transport to ACS Florida did not repeat that.
5. **Checks that held on the fitting pool did not hold on held-out data** (guards, safeguards, learned decoders, per-token calibration).
6. **Decisions alone set a recovery floor** for decision-preserving releases (Adult pair 0.72–0.74 from decisions only). Methods allowed to change decisions are not bound by it.
7. **The conditional-kernel lane ran once**, and conditioning contributed nothing measurable (−0.0006). No campaign used an HSIC penalty.

## What a new proposal must do differently

A proposal is not distinct merely by changing any of these:
- the strength, schedule or allocation of an adversarial penalty;
- the critic refresh rate;
- a guard pool;
- an adapter's size;
- a code's capacity;
- a post-hoc decoder or calibrator on frozen outputs;
- a composition of frozen outputs.

Each was measured above, and the bottleneck was task utility or held-out generalisation, not the knob.

A credible proposal must therefore do one of two things:
- name information the neural control demonstrably retains and explain why removing it costs less utility than the measured trade-offs; or
- change the contract explicitly and evaluate every comparator under the changed contract.
