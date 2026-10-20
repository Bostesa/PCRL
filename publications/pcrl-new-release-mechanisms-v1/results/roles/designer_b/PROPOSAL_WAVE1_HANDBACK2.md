# Designer B, wave 1, second hand-back (saved verbatim with attribution)

Author: Designer B (family B) of campaign pcrl_new_release_mechanisms_v1. This second final message arrived at about 15:05Z on
2026-10-20, after the coordinator's no-sub-agent instruction, and carries the sub-agent disclosure. It also states the training
hyperparameters and the per-strength invented-data diagnostics. Saved verbatim by the coordinator; nothing below was edited. The
first hand-back is PROPOSAL_WAVE1.md.

---

DESIGNER B: wave-1 hand-back (family B, pcrl_new_release_mechanisms_v1). Written 15:00Z.

SUB-AGENT DISCLOSURE (coordinator rule)
- Count: 2. Both were read-only Explore agents launched at about 14:37Z, before the six-agent rule reached me. Both finished, so none is live, and I spawned nothing after the rule arrived.
  1. "Summarize joint-training campaigns": about 14:37Z to about 14:44Z (399 s). It summarized refreshed_guarded_joint, online_strength_frontier, strength_matched_feedback, penalty_no_erasure, joint_complete_view_method, confidence_capacity, confidence_budgeted_privacy, decision_preserving_compression, the minimal and two-recipient adapters, and controlled_training_size/conditional.
  2. "Summarize sampled-channel campaigns": about 14:37Z to about 14:45Z (457 s). It summarized latent_and_sampled_methods, conditional_budgeted_release, bank_anchored_release, incumbent_view_risk, method_portfolio, and the ACS-2018 stochastic branches.
- Neither ran Python or loaded data, and both reported reading nothing under nrm/fa/ or roles/designer_a/.
- The out-of-queue CPU for my selftests and timing script (40 s, upper estimate) is charged under label designer_b. The agents' token work involved no numerical computation.

ACCESS STATEMENT
- I read no real label and no real data. The only data-like arrays I touched were invented arrays inside the selftests and one timing script.
- I did not read Designer A's files. No real fit was run.

=====================================================================
PROPOSAL / MECHANISM-CARD DRAFT: family B, "FSC-CA"
Channel-aware joint training with a SEX-targeted, Fisher-shaped, cross-recipient-coupled Gaussian logit channel
=====================================================================

1. THE OPERATION THAT CHANGES THE TRANSMITTED OUTPUT

Learner
- Two task networks, one per task (income K=2, occupation K=5).
- Each network is Linear(145,128)-ReLU-Linear(128,128)-ReLU-Linear(128,32)-Linear(32,K). This is the U0 Recipe A shape (acs_replication/neural.py) at twice the width.
- Both networks are trained jointly from scratch under the configuration's own channel.
- The loss is the summed income and occupation cross-entropy of the EMITTED vectors. Each training example gets one draw per step, reparameterized: the gradient flows through the deterministic logits, and the per-person noise operator is recomputed from the detached logits.

Coordinates
- z = [H2·l_inc ; H5·l_occ] in R^5, where H_k are orthonormal Helmert rows orthogonal to the all-ones vector.
- These are linked centred logits (1 income coordinate + 4 occupation coordinates). Mapping back leaves the softmax unchanged.

Channel law (person i, release instance r)
- F_i: the per-person Fisher of the summed task log loss in z. It is block-diagonal and computed from the deterministic probabilities, so it needs no label.
- Geometry, frozen per configuration from MODEL_FIT, computed on the final learner's in-sample logits:
  - δ = mean z(women) − mean z(men);
  - Σ_w = pooled within-SEX covariance.
- Direction:
  - d_i = (F_i + s²·Σ_w⁻¹)⁻¹ Σ_w⁻¹ δ;
  - v_i = d_i / sqrt(d_iᵀ F_i d_i), so v_i has unit Fisher norm.
- Per-person scale: c_i = min(s, 1.5/|v_i|). The predeclared cap is 1.5 centred-logit units per unit draw.
- Perturbation: z̃_i = z_i/T + c_i·ε_{i,r}·v_i.
  - T holds the per-task CALIBRATION temperatures; the geometry is rescaled exactly to match.
  - ε_{i,r} ~ N(0,1) is ONE scalar per person per instance, shared by both recipients. This is maximal coupling along the person's own direction.
- Emission:
  - p̃_inc = softmax(H2ᵀ z̃_inc) and p̃_occ = softmax(H5ᵀ z̃_occ);
  - each decision is the argmax of the emitted vector;
  - each recipient receives only its own vector and decision. The runtime service supplies exactly one vector and one decision per recipient per person.
- Cost: to second order, the expected NLL cost of the noise is s²/2 nats per person (linked total, split between recipients by the direction). For calibrated vectors, the summed-Brier excess is at most the NLL cost to second order.
- d_i is the rank-1 minimizer of the linked linear-Gaussian SEX discriminability δᵀ(Σ_w+Σ_n)⁻¹δ at second-order Fisher cost s²/2. With s→0 it tends to F⁻¹Σ_w⁻¹δ; with large s it tends to δ.

Seed handling (frozen before fitting)
- ε_{i,r} is the Box-Muller transform of keyed BLAKE2b(key_{config,v}, "NRM-FB|<role>|r<r>|<row_id>").
- key_{config,v} = BLAKE2b(root_v, "NRM-FB|<config>|V<v>").
- root_v is 32 bytes from os.urandom at fit time. It is stored privately (mode 0600, write-once, .private/.../fb/V<v>/RELEASE_ROOT.bin) and is never published or committed. Replays use the stored root.
- A draw depends only on (key, role, r, row_id). It does not depend on row order, batch or n (selftest-checked).
- Instances r = 0..7 are independent.
- Other draws use separate keys:
  - CALIBRATION temperature fitting uses its own key, with common random numbers across configurations;
  - early stopping uses fixed fit-time-only validation draws;
  - training noise and permutations come from numpy PCG64 streams seeded by (version, phase, epoch), identical for every learner of a version, so learners are paired.
- Re-querying a person within one instance returns the same draw. Repeated-query protection across instances is NOT claimed.

Monte Carlo repetitions
- R = 8 independent one-draw-per-person release instances per randomized configuration, stored as (8, n, k) stacks for all six roles.
- Utility is computed on each emitted vector and its emitted argmax, then averaged over the 8 instances. The MC SE (sd/√8) is reported separately from the household-sampling SE.

2. CLOSEST PRIOR PCRL ATTEMPTS, WHAT IS DIFFERENT, WHICH BOTTLENECK IT ADDRESSES

Closest attempt: results/pcrl_latent_and_sampled_methods_v1/sampled_channel_v1/, score channel "A" (Adult).
- It mapped frozen U0 scores to a frozen T* bank canonicalized to 16/321 code vectors.
- A central gate chose among at most 16 joint same-class actions, with one joint categorical draw per person (HMAC), R=3.
- 14 of 16 FIT searches had no feasible policy, and no sampled release was ever attacked.
- The measured bottleneck was not randomization: the identity action itself broke the FIT occupation U0 NLL cap (+0.0043 / +0.0054; design/A_FIT_FEASIBILITY_SUMMARY.md).

Related attempts:
- pcrl_conditional_budgeted_release_v1: LP mixtures toward frozen donors. Pair reduction 0.0020 while using 0.0098 of the 0.010 NLL cap; "no randomness advantage".
- pcrl_bank_anchored_release_v1 and pcrl_incumbent_view_risk_v1: re-routing among existing codes. Reductions of 0.002 to 0.006; "no distinct measured advantage".
- pcrl_method_portfolio_v1, B-JVIB: a jointly trained sampled Gaussian latent. Pair 0.800 against a deterministic twin's 0.869 (Adult), but it released the latent z (not allowed here) and failed seed-0 occupation NLL (0.0108 vs U0).
- decision_preserving_compression: every finite code cost at least about 0.02 nats of occupation NLL.

What is substantively different here:
- (a) No codebook or bank. The identity is the learner's own continuous release, so there is no quantization cost. This removes the frozen-bank bottleneck: identity already over the cap, and finite codes costing at least 0.02 nats.
- (b) The task learner is trained through the actual channel from scratch. Its logits are mean-calibrated under the noise, and the channel geometry follows the learner's evolving SEX geometry. In the earlier attempts the scores were frozen.
- (c) Noise is continuous and placed in the Fisher-optimal SEX-targeted direction. The per-person cost is fixed at s²/2, so the only design question is where to place it.
- (d) A single shared scalar couples the two recipients' noise. Per-recipient utility is blind to coupling, while the linked view is where the references leak most. The ACS-2018 synthetic certified model (branch records) found zero randomization advantage under local-only constraints, but +0.061 / +0.175 / +0.302 under coalition constraints at budgets 0 / 0.002 / 0.01.

3. WHY THE TRADE CAN BE FAVOURABLE DESPITE THE JENSEN GAP, AND WHEN IT IS NOT

The cost of randomization:
- For zero-mean logit noise n, E[NLL(z+n)] − NLL(z) ≈ ½ nᵀFn > 0. For Brier, E‖p̃−e_y‖² = ‖E p̃ − e_y‖² + Σ_k Var(p̃_k).
- So the randomized release always pays the Jensen gap relative to a deterministic vector with the same mean. This cost is second order in amplitude and direction-dependent in the Fisher metric.

Why it can still pay:
- (i) A deterministic release lowers a re-trained flexible reader's recovery only by merging people. An injective change, such as shrinkage or interpolation, is inverted by a reader trained on it. Merging is lumpy and pays refinement loss, which is the measured finite-code bottleneck. Randomization buys partial ambiguity continuously, at second-order cost.
- (ii) With unit-Fisher-norm directions every person pays s²/2, so only placement matters.
  - In the linear-Gaussian approximation, the reduction in linked discriminability per nat at small s is 2·aᵀF⁻¹a, with a = Σ_w⁻¹δ.
  - This is large exactly when SEX information lies in low-Fisher directions: log-odds among outcomes the predictor already considers unlikely. An example is log P(construction/maintenance) for profiles where it is 1-4%.
  - Readers of probability vectors see those coordinates at full rank resolution, while the task losses barely depend on them.
  - In those directions any deterministic change is either invertible or a merge, but noise is nearly free.
- (iii) Coupling costs nothing per recipient and aligns the noise with the pair discriminant.
- (iv) Training through the channel, plus temperatures fitted on emitted draws, removes the avoidable miscalibration part of post-hoc noise. Zero-mean logit noise makes the mean emitted vector under-confident; the learner re-centres. Only the irreducible variance term is left. The frozen-bank channels paid both terms plus quantization.

When it is NOT favourable:
- If SEX information sits in task-dominant, high-Fisher directions (the occupation-group composition itself), masking costs about ½·F_dir·σ_w² per unit of discriminability.
- At s²/2 ≤ 0.007 nats, the linear-Gaussian reduction is then a few thousandths of AUC.
- The fit-time diagnostic shows which regime holds before any reader runs. It is recorded per configuration as the in-sample one-draw linked Δ² of the emitted release against the identity release, on MODEL_FIT.

Honest expectation:
- The twin is a U0-class neural release, with little NLL slack relative to U0. The A1_PO_16 occupation slack is not available to it.
- The 0.020 target against NEURAL_RAW (0.6245) and H_LJ (0.6183) needs pair AUC ≤ 0.6045 / 0.5983. That is about 0.035-0.04 below a twin plausibly near 0.64.
- I do not expect FSC-CA to be competitive. Its value is a clean frontier test of whether targeted randomization beats SEX-blind spending, with the identity, deterministic and post-hoc controls.

4. CONTROLS (the simplest explanations of a positive result) AND FALSIFICATION

Controls:
- FB_TWIN: the equally trained task-only twin. Same architecture, initialisation, split, permutations, training-noise stream (unused), early-stopping rule and CALIBRATION temperature procedure, under the identity channel.
- FB_SEX_S2_ID: the exact identity channel of the S2 candidate learner, i.e. its deterministic softmax. It tests whether channel-aware training changed the predictor rather than the noise doing the work.
- FB_SEX_S2_MEAN: the relevant deterministic construction. It releases the exact mean vector E_ε[p̃] of the S2 channel (48-node Gauss-Hermite). It tests whether the mean shift, rather than the randomness, protects.
- FB_PCA_S1/S2/S3: the SEX-blind channel of matched utility cost.
  - Same rank-1 coupled construction, same s, same cap, same R, same keys scheme and same channel-aware training.
  - The only change is the direction: δ is replaced by the top principal axis of the total covariance of z, and Σ_w by that covariance.
  - Cost is matched to second order by construction (s²/2). Realized costs are measured, and the frontier is read on realized MODEL_SELECT costs.
- FB_POSTHOC_S2: the S2 SEX-targeted channel on the twin learner, with geometry from the twin and its own temperatures. This is the frozen-learner analogue, and it tests the joint-training part of the hypothesis.
- Not requested: deterministic erasure along v. Partial shrinkage is invertible, and full LEACE-type output erasure already broke the utility gates (joint_complete_view_method). The challenger may add a deterministic SEX-blind coarsening frontier (nrm/ctl).

Falsification:
- F1 (targeting): at matched realized utility cost, the SEX path's pair recovery is not below the PCA path's beyond reader-cell noise at S2 and S3. The targeting rationale is then falsified.
- F2 (randomness): FB_SEX_S2_ID or FB_SEX_S2_MEAN recovers no more than FB_SEX_S2 (within reader noise). Randomization is then not the active ingredient.
- F3 (joint adaptation): FB_POSTHOC_S2 matches FB_SEX_S2 on both recovery and utility cost. The family-specific claim that joint training makes randomization affordable then fails, even if the channel works.
- F4 (reader mismatch): the fit-time Δ² falls by at least 50% at S2, but the screen pair AUC falls by less than 0.003. The readers then exploit nonlinear structure that a rank-1 global-geometry channel does not cover.

5. TASK-ONLY TWIN AND CAPACITY

- FB_TWIN is the same learner trained under the identity channel. It is the U0 shape at twice the width, with Adam lr 1e-3, batch 512, at most 40 epochs, early stopping (patience 5, minimum delta 1e-4) on a household-grouped 15% internal split, then a refit on all of MODEL_FIT for the selected epoch count. Per-task CALIBRATION temperatures follow.
- It must pass the original limits: 24 clauses + 4 guards in both versions. If it fails, revision R1 applies.

6. FITTING PLAN AND EXPOSURE

- MODEL_FIT (20,000 people / 13,337 households):
  - Use: all network parameters and the channel geometry.
  - Labels read:
    - income and occupation are the task losses;
    - SEX enters only the closed-form geometry (δ, Σ_w). It is re-estimated each epoch from the current deterministic logits (fit rows during search, all rows during refit), and through the channel-aware loss it reaches the sex:* learners.
  - The internal split is household-grouped (BLAKE2b of household id, keyed by version) and used only for early stopping. Validation uses summed emitted NLL with fixed draws.
  - The scheme is non-nested: the final refit uses the same 20,000 people, so the actual training size of the deployed parameters is 20,000 / 13,337. The release geometry comes from in-sample logits (5-dimensional statistics over 20,000 people), which is disclosed as non-nested.
- CALIBRATION (2,982 people): only two temperatures per configuration, from golden-section search on log T in [log 0.5, log 3] with 2 coordinate sweeps. Randomized configurations use the mean emitted NLL over 8 keyed calibration draws; the MEAN configuration uses its exact NLL. No SEX.
- MODEL_SELECT: family B never reads its labels. Feedback comes only through nrm.evaluate and nrm.screen outputs.
- Versions v ∈ {0, 1}: the initialisation seeds, split, permutations and training noise depend on v. Both versions are fitted.
- Queue: one fit unit per version, for example:
  python3 -m nrm.jobqueue run --unit FB_FIT_V0 --stage fit --rss-gib 2 -- <repository>/.venv/bin/python -m nrm.fb.learner fit --version 0 --card <committed card>
  A retry needs new configuration ids, because the root-key file and the releases are write-once.

7. STRENGTH PATH (predeclared)

- s ∈ {S1 0.06, S2 0.09, S3 0.12}, giving second-order linked NLL cost s²/2 ∈ {0.0018, 0.0041, 0.0072} nats per person.
- The SEX-blind PCA path uses the same s values.
- The result is read as pair recovery against realized MODEL_SELECT utility cost, against the PCA path, with the twin, identity and MEAN points marked.
- No other strengths except R2 below. No penalty weights, coupling variants or temperature tuning.

8. READER NEEDS FOR RANDOMIZED OUTPUTS

- Readers train and score under the same channel law.
- In a reader cell, every person takes exactly one draw from one instance r:
  - auxiliary people (reader FIT and VALIDATION) contribute one draw each;
  - SCORE people contribute one draw each;
  - all from the SAME instance.
- I request no multi-draw variant, and a target person never supplies more than one draw.
- Linked readers pool the two recipients' draws of the same instance, which is the actual joint release.
- Proposed cell-to-instance map: cell (v, repetition j, reader seed k) uses instance r = j for both seeds. This separates seed variation from Monte Carlo variation.
- Monte Carlo variation: refit one reader recipe per cell on instances j+2 and j+4, with the same split and seed. To save CPU, do this only for FB_SEX_S2 and FB_PCA_S2. Report it separately from household-split and reader-seed variation.
- Deterministic configurations need nothing special (R = 1).

9. CONFIGURATION REQUEST AND CPU

- 10 configurations per version × 2 versions = 20 of the global 60:
  - FB_TWIN (twin);
  - FB_SEX_S1, FB_SEX_S2, FB_SEX_S3 (candidate);
  - FB_PCA_S1, FB_PCA_S2, FB_PCA_S3 (control);
  - FB_SEX_S2_ID, FB_SEX_S2_MEAN, FB_POSTHOC_S2 (ablation).
- If the budget must be cut, drop in this order: FB_POSTHOC_S2, then FB_SEX_S2_MEAN (giving 16). Never drop the twin, the SEX path or the PCA path.
- Per version this involves 7 learners × 2 model fits each (search + refit), 10 calibrator fits and 0 critic fits. The geometry is closed-form, not a critic.
- CPU (torch, one thread):
  - Timing on invented arrays of the real size (20,000 × 145) gave about 0.3 CPU-s for 3 search epochs + 1 refit epoch per learner. Emitting 3,000 people × 8 instances took 0.016 s.
  - Estimate: ≤ 3 CPU-min per version.
  - Request: 0.25 CPU-h for fits, including one retry. Evaluation is small; readers come from the coordinator's grid plus the MC replicates above.
- Memory: < 1 GiB (request --rss-gib 2).

10. REGISTERED PREDICTIONS (committed before any fit)

- P1: FB_TWIN passes all 24 clauses + 4 guards in both versions (p ≈ 0.7).
- P2: FB_SEX_S2 lowers pair AUC against FB_TWIN by 0.002-0.012 (central 0.006). FB_PCA_S2 lowers it by ≤ 0.003.
- P3: the SEX path lies below the PCA path at matched realized cost at S2 and S3 (p ≈ 0.6).
- P4: FB_SEX_S2_ID and FB_SEX_S2_MEAN are within 0.003 of FB_TWIN's recovery (p ≈ 0.75).
- P5: FB_POSTHOC_S2 is about equal to FB_SEX_S2 in recovery. Channel-aware training saves less than 0.001 nats (p ≈ 0.65).
- P6: the 0.020 target against NEURAL_RAW and H_LJ is NOT met (p > 0.9).

11. REVISION PLAN

- R1 (utility): if FB_TWIN fails any original clause or guard in either version, make one capacity revision: FB2_*, d-256-256-64-K, Adam weight decay 1e-4, same channel, new configuration ids. If it still fails, the outcome is UTILITY_INFEASIBLE and the family closes.
- R2 (strength): if FB_SEX_S3 passes everything with at least 0.004 nats of occupation-NLL slack in both versions, add S4 = 0.15 for both SEX and PCA (4 configurations).
- R3 (F4): replace the global geometry with decision-cell-conditional geometry: δ_c and Σ_w,c per linked decision cell (10 cells). This targets readers that condition on decisions. It adds FB3_SEX_S2/S3 and FB3_PCA_S2/S3 (8 configurations).
- Closure:
  - If F1 holds after R3, or F2 holds at any point, the family is CLOSED, with no further strength, penalty or coupling tuning.
  - If only F3 holds, I report that joint adaptation adds nothing. The channel is then carried only as a frozen-learner component.
- Alternative successor if FB_TWIN passes but lacks budget: a W-anchored residual. Frozen A1_PO_16 + H3 logits would serve as an offset, with a channel-aware residual network; the identity channel then equals W exactly. It is not requested now because it needs a different exposure map: the cross-fitted A1 base from the tsz correction, and H3 income logits that are in-sample on MODEL_FIT.

=====================================================================
FILES WRITTEN / COMMITTED (commit ef0bc1ddf on research/pcrl-new-release-mechanisms-v1; plain message, no trailers; my paths only)
=====================================================================
- <repository>/.claude/worktrees/pcrl-correction-transfer-stability-v1/nrm/fb/__init__.py
- <repository>/.claude/worktrees/pcrl-correction-transfer-stability-v1/nrm/fb/channel.py
  - numpy only;
  - Helmert coordinates, per-person Fisher, geometry ('sex', 'pca'; 'iso' is implemented but not requested);
  - per-person operators, keyed one-draw-per-person emission;
  - identity channel, Gauss-Hermite mean-vector construction, metrics of emitted vectors with MC SE.
- <repository>/.claude/worktrees/pcrl-correction-transfer-stability-v1/nrm/fb/learner.py
  - torch: paired joint learners, emitted-NLL early stopping, refit, CALIBRATION temperatures;
  - fit_version (all 10 configurations of a version through nrm.release.save, with records including draws, seed_handling, noise_coupling, the frozen geometry, temperatures and fit-time diagnostics);
  - release_from_features (deployment from admitted features alone);
  - CLI: selftest | fit --version V --card <card>.
- Not committed: a timing script in my scratchpad (invented data only).
- The mechanism card is not committed yet; this text is the draft. I will commit it as results/pcrl_new_release_mechanisms_v1/cards/<FB card> when you approve.

Other records:
- CPU charge: python3 -m nrm.jobqueue charge --label designer_b --cpu-s 40 (selftests and timing, upper estimate). Confirmed in the compute ledger.

=====================================================================
SELFTEST OUTPUT (invented arrays only)
=====================================================================
python3 -m nrm.fb.channel selftest: exit 0. The same module under the torch venv also exits 0.
CHANNEL {"ok":true,"checks":{"helmert_orthonormal":true,"softmax_preserved":true,"z_from_probs_roundtrip":true,"fisher_equals_hessian":true,"identity_channel_exact":true,"shapes_one_draw_per_person":true,"valid_probability_rows":true,"draws_independent_of_row_order":true,"draws_reproducible":true,"repetitions_differ":true,"keys_differ":true,"rank1_coupled_draw":true,"coupled_scalar_is_the_keyed_draw":true,"keyed_normals_moments":true,"second_order_cost_matches":true,"mean_vector_matches_mc":true,"mean_vector_identity_at_s0":true,"targeted_beats_blind_on_invented_data":true,"scaled_geometry_equals_refit":true,"deterministic_mc_se_zero":true,"refusals":true,"no_data_access_module":true},"cpu_s":2.32}
CHANNEL_INFO {"normal_moments":{"mean":0.0011,"var":0.9998,"max_abs_corr":0.0062},"second_order_cost_s0.1":{"sex":0.00476,"pca":0.00448,"iso":0.00523},"mean_vector_vs_mc_max_abs":0.0045,"lda_delta2_one_draw_s0.3":{"identity":11.19,"sex":0.399,"pca":11.069,"iso":1.737}}
- The predicted second-order cost at s = 0.1 is 0.005. The Δ² numbers come from invented data built with the SEX shift in a low-variance direction, so they say nothing about the real data.

<repository>/.venv/bin/python -m nrm.fb.learner selftest: exit 0. It used fake access hooks, a temporary folder, no ledger and no real access; the real access path was refused before anything was written.
LEARNER {"ok":true,"checks":{"real_access_refused":true,"all_configurations_saved":true,"randomized_flags":true,"randomized_stack_shape":true,"label_reads_policy":true,"paired_streams_across_learners":true,"identity_config_deterministic":true,"mean_vector_config_deterministic":true,"deployment_from_features_bitwise":true,"twin_and_s2_learners_differ":true,"geometry_frozen_in_record":true,"temperatures_in_bounds":true,"second_fit_refused":true},"cpu_s":1.6}
LEARNER_INFO (invented data, in-sample, one draw; the values vary run to run because the release root comes from os.urandom):
FB_TWIN nll inc .5838 occ 1.4486, Δ² .593
FB_SEX_S1 .5848 / 1.4495, .407
FB_SEX_S2 .5844 / 1.4507, .263
FB_SEX_S3 .5870 / 1.4520, .253
FB_PCA_S1 .5842 / 1.4510, .506
FB_PCA_S2 .5848 / 1.4508, .326
FB_PCA_S3 .5879 / 1.4524, .301
FB_SEX_S2_ID .5836 / 1.4483, .594
FB_SEX_S2_MEAN .5836 / 1.4485, .594
FB_POSTHOC_S2 .5858 / 1.4508, .235

Invented-data timing (real size 20,000 × 145, label-free): 3 search epochs + 1 refit epoch per learner took 0.3-0.9 CPU-s; emitting 3,000 people × 8 instances took 0.016 CPU-s.

Waiting on the coordinator for:
- the receipt verification;
- approval of this card text, so I can commit it under cards/;
- the configuration allocation, before running the two fit units through the queue.
