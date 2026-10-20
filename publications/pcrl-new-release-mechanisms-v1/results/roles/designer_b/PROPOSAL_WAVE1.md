# Designer B, wave 1 proposal (saved verbatim with attribution)

Author: Designer B, the independent mechanism designer and implementer for family B, of campaign pcrl_new_release_mechanisms_v1.
Returned to the coordinator at 15:00Z on 2026-10-20 as its final message. Saved verbatim by the coordinator; nothing below was
edited. This is a DRAFT card. The governing card is the committed file under cards/, with the coordinator's and the statistical
reviewer's decisions recorded there.

---

DESIGNER B: wave-1 report for pcrl_new_release_mechanisms_v1, returned 15:00Z

No real label was read and no real fit was run. The code is committed at ef0bc1ddf (nrm/fb/), and both invented-data selftests pass. The mechanism card below is a draft. It is not committed under cards/ yet; I will commit it when you approve it.

# Proposal: family B, channel-aware joint training with a SEX-targeted, Fisher-shaped, coupled Gaussian logit channel ("FSC")

## 1. What changes the transmitted output

**Learner.**
- Two task networks:
  - income, K = 2;
  - occupation, K = 5.
- Each network is Linear(145,128)-ReLU-Linear(128,128)-ReLU-Linear(128,32)-Linear(32,K). This is the U0 Recipe A shape at twice the width.
- Both networks are trained jointly from scratch, under the configuration's own channel.
- The loss is the summed income and occupation cross-entropy of the EMITTED vectors:
  - one training draw per example per step, reparameterized;
  - the gradient flows through the deterministic logits;
  - the per-person noise operator is recomputed from the detached logits.

**Coordinates.**
- z = [H2 l_inc ; H5 l_occ] in R^5.
- H_k are orthonormal Helmert rows, orthogonal to the all-ones vector, so softmax is unchanged.

**Channel law.** This is frozen per configuration. For person i and release instance r:
- **Fisher.** F_i is the per-person Fisher of the summed task log loss in z. It is block-diagonal and computed from the deterministic probabilities, so it is label-free.
- **Geometry.** It is estimated on MODEL_FIT from the final learner's in-sample logits and then frozen:
  - delta = mean z(women) - mean z(men);
  - Sigma_w = the pooled within-SEX covariance.
- **Direction.**
  - d_i = (F_i + s^2 Sigma_w^-1)^-1 Sigma_w^-1 delta.
  - v_i = d_i / sqrt(d_i' F_i d_i), so v_i has unit Fisher norm.
  - c_i = min(s, 1.5 / |v_i|). The cap is 1.5 centred-logit units per unit draw.
- **Release.**
  - z~_i = z_i/T + c_i * eps_{i,r} * v_i, with eps_{i,r} ~ N(0,1).
  - T is the pair of per-task CALIBRATION temperatures; delta and Sigma_w are rescaled to match.
  - eps is ONE scalar per person per instance, shared by the income and occupation recipients. That is maximal coupling along the person's own direction.
  - Emitted vectors are p~_inc = softmax(H2' z~_inc) and p~_occ = softmax(H5' z~_occ).
  - Each decision is the argmax of the emitted vector. Each recipient receives only its own vector and decision: one probability vector and one decision per recipient per person.
- **Why this direction.** d_i is the rank-1 minimiser of the linked linear-Gaussian SEX discriminability delta'(Sigma_w + Sigma_n)^-1 delta at a second-order Fisher cost of s^2/2.
- **Cost.**
  - The expected NLL cost of the noise is s^2/2 nats per person to second order. That is the linked total; the direction splits it between the two recipients.
  - The expected summed-Brier excess is at most the NLL cost to second order.
- **SEX-blind modes.**
  - The PCA mode uses the same formula, with delta replaced by the top principal axis of the total covariance of z, and Sigma_w replaced by that total covariance.
  - A full-rank, uncoupled, Fisher-isotropic mode ("iso") is implemented but not requested.

**Seed handling.** This is frozen.
- eps_{i,r} is the Box-Muller transform of keyed BLAKE2b(key, "NRM-FB|<role>|r<r>|<row_id>").
- key = BLAKE2b(root_v, "NRM-FB|<config_id>|V<v>").
- root_v is 32 bytes from os.urandom at fit time. It is stored only in the private tree (mode 0600) and never published. A replay uses the stored root.
- A draw depends only on (key, role, r, row_id). It does not depend on row order, batch or n, and the selftest checks this.
- Instances r = 0..7 are independent.
- Other draws are kept separate:
  - CALIBRATION temperatures use a separate calibration key, with common random numbers across configurations;
  - early stopping uses fixed validation draws that exist only at fit time;
  - training noise and permutations come from numpy PCG64 streams seeded by (version, phase, epoch). They are identical across all seven learners of a version (paired).

**Monte Carlo and repeated queries.**
- R = 8 independent one-draw-per-person release instances per randomized configuration, stored as (8, n, k) stacks for all six roles.
- Repeated-query protection is not claimed. Within one instance a re-query returns the same draw; across instances the draws are independent.

## 2. Closest prior PCRL attempts and what is substantively different

**Prior attempts.**
- **results/pcrl_latent_and_sampled_methods_v1/sampled_channel_v1/ (score channel A, Adult)** is the closest.
  - Design:
    - frozen U0 scores mapped to a frozen T* bank (16/321 canonical vectors);
    - a gate choosing among at most 4 same-class actions per recipient;
    - one HMAC-keyed joint categorical draw per person, R = 3.
  - Outcome: 14 of 16 FIT searches were infeasible, and the other 2 were incomplete; no protected release was ever attacked.
  - Measured bottleneck: the identity action itself broke the occupation U0 NLL cap, by +0.0043 and +0.0054 (design/A_FIT_FEASIBILITY_SUMMARY.md). The bank's quantization, not the randomization, used up the budget.
- **pcrl_conditional_budgeted_release_v1** (LP mixtures toward frozen donors):
  - nominee W reduced pair AUC by 0.0020 [-0.0037, 0.0076];
  - it used 0.00977 of the 0.010 NLL cap;
  - recorded verdict: "no randomness advantage".
- **pcrl_bank_anchored_release_v1 and pcrl_incumbent_view_risk_v1** (re-routing existing codes): reductions of 0.002-0.006, and "randomness and joint coordination have no distinct measured advantage".
- **pcrl_method_portfolio_v1 B-JVIB** (jointly trained, sampled Gaussian latent):
  - pair AUC 0.7999 against 0.8686 for its deterministic twin;
  - but it released the latent z, which this contract does not allow;
  - it failed seed-0 occupation NLL (0.0108 against U0).
- **ACS-2018 research branches** (synthetic certified model): randomization gave zero advantage under local-only constraints, but +0.061/+0.175/+0.302 under coalition constraints (budgets 0/0.002/0.01).

**Differences from those attempts.**
1. **No bank and no codebook.** The identity action is the learner's own continuous release, so there is no quantization cost. This removes both measured bottlenecks:
   - the frozen-bank identity action that already exceeded the cap;
   - "every finite code costs at least about 0.02 nats of occupation NLL" (decision_preserving_compression).
2. **The task learner is trained through the actual channel, from scratch.** It is not a channel fitted over fixed scores. Its logits become mean-calibrated under the noise, and the channel geometry follows the learner's evolving SEX geometry, re-estimated every epoch.
3. **Continuous, Fisher-optimal, SEX-targeted rank-1 noise.** The cost is second order, and it is placed where SEX discriminability per nat of task cost is highest.
4. **Free coupling.** One shared scalar per person aligns both recipients' noise with the linked discriminant. This costs nothing in per-recipient utility, and it follows the synthetic finding that randomization helps only under coalition constraints.

## 3. Why the trade can still be favourable despite the Jensen gap

**The cost being paid.**
- Randomization costs E[-log p~_y] >= -log E[p~_y] and E||p~ - e_y||^2 = ||E p~ - e_y||^2 + sum_k Var(p~_k).
- With logit noise of unit Fisher norm, the expected NLL cost is exactly s^2/2 nats per person to second order, whatever the direction. So the only design question is where to spend a fixed cost.

**Four arguments that the trade can be favourable.**
- **(i) Deterministic releases cannot buy partial ambiguity.**
  - A deterministic release lowers a retrained flexible reader's recovery only by merging people. An injective map, such as shrinkage or interpolation, is inverted by a reader trained on it.
  - Merging is lumpy and pays refinement loss; that is the measured cost of at least about 0.02 nats for finite codes.
  - Randomization buys ambiguity continuously, at a cost that is second order in the amplitude.
- **(ii) Noise is cheapest where the predictor is already confident.**
  - In the linear-Gaussian approximation, the reduction of linked discriminability per nat at small s is 2 a' F^-1 a, with a = Sigma_w^-1 delta.
  - That is large exactly when the SEX signal sits in low-Fisher directions: log-odds among outcomes the predictor already considers unlikely. An example is log P(construction/maintenance) at 1-4% for profiles where men and women differ.
  - Probability-vector readers see those coordinates at full rank resolution, while the task losses barely depend on them.
  - There, any deterministic change is either invertible (no protection) or a merge (expensive), and noise is nearly free.
- **(iii) Coupling is free for per-recipient utility** and targets the pair view, where the references leak most.
- **(iv) Channel-aware training removes the avoidable part of a post-hoc channel's cost.**
  - Zero-mean logit noise leaves the mean emitted vector under-confident.
  - Training on emitted losses, plus CALIBRATION temperatures fitted on emitted draws, re-centres the logits.
  - Only the irreducible variance term is left. The frozen-bank channels paid both terms, and quantization on top.

**When the trade is not favourable.**
- If the SEX signal is in task-dominant, high-Fisher directions (the occupation-group composition itself), masking costs about F_dir * sigma_w^2 / 2 per unit of reduction.
- At s^2/2 <= 0.0072 nats the reduction is then only a few thousandths of AUC.
- The fit-time diagnostic shows which regime holds before any reader runs. It is the in-sample, one-draw Delta^2 of the emitted vectors against the identity, recorded per configuration.

**Competitiveness.**
- The 0.020 target requires pair AUC <= 0.6045 against NEURAL_RAW (0.6245) and <= 0.5983 against H_LJ (0.6183).
- A U0-class twin probably sits near 0.64, so the target needs a reduction of about 0.035-0.04. That is not plausible at this budget.
- The family's realistic value is a clean test of whether targeted randomization beats SEX-blind spending on the utility-recovery frontier.

## 4. Controls, and what would falsify the rationale

Every control uses the same architecture, initialisation, split, permutations, training-noise stream, early-stopping rule, temperature procedure and R.

| Configuration | What it is | What it tests |
|---|---|---|
| FB_TWIN | equally trained task-only twin under the identity channel | baseline |
| FB_SEX_S2_ID | exact identity channel of the S2 candidate learner | protection coming only from the retrained deterministic predictor |
| FB_SEX_S2_MEAN | deterministic construction: E[emitted] by 48-node Gauss-Hermite quadrature, released as an actual deterministic release | the mean shift, rather than the randomness, doing the work |
| FB_PCA_S1/S2/S3 | SEX-blind channel at matched second-order cost (same s, same rank-1 coupled form, channel-aware trained); realized costs measured | "any noise of this cost" (the spending explanation) |
| FB_POSTHOC_S2 | the S2 SEX channel on the twin learner, no adaptation | the frozen-learner analogue of the earlier frozen-bank approach |

The other deterministic option, erasure or projection along v, is not requested:
- partial shrinkage is invertible;
- full LEACE in output space already broke the utility gates (joint_complete_view_method).

**Falsification.**
- **F1.** At matched realized utility cost (MODEL_SELECT NLL/Brier excess from nrm.evaluate), the SEX path's linked recovery is not below the PCA path's at S2 and S3. That falsifies targeting.
- **F2.** FB_SEX_S2_ID or FB_SEX_S2_MEAN reaches FB_SEX_S2's recovery. Then randomization is not the active ingredient.
- **F3.** FB_POSTHOC_S2 matches FB_SEX_S2 on both recovery and cost. Then joint adaptation adds nothing.
- **F4.** Fit-time Delta^2 falls by at least 50% at S2, but screen pair AUC falls by less than 0.003. Then the readers use nonlinear structure that a global rank-1 channel does not cover.

## 5. Task-only twin

- It is FB_TWIN as described above.
- Its capacity is twice U0's width, with early stopping and a CALIBRATION temperature.
- It must pass the original 24 clauses and 4 guards in both versions; otherwise revision R1 applies.
- I am not using A1_PO_16's occupation slack in wave 1. A W-anchored successor would be a channel-aware residual over frozen A1+H3 logits, but it needs:
  - the cross-fitted A1 base;
  - in-sample H3 income logits (H3 was fitted on all of MODEL_FIT).

  That is a different exposure map, so I only note it as a possible later step.

## 6. Fitting plan and exposure (non-nested)

**MODEL_FIT** (all 20,000 people / 13,337 households; labels income, occupation, SEX):
- all network parameters and the channel geometry;
- SEX enters only delta and Sigma_w, which are closed-form and re-estimated per epoch. Through the emitted loss it reaches the sex:* learners;
- a household-grouped 15% internal split keyed by version (BLAKE2b of the household id) is used ONLY for the early-stopping epoch count;
- the deployed networks are refit from the same initialisation on all 20,000 people for that epoch count;
- the release geometry is estimated in-sample on all of MODEL_FIT. It is 5-dimensional, but in-sample and non-nested, which I disclose.

**CALIBRATION** (2,982 people; income, occupation; no SEX):
- two temperatures per configuration only;
- golden section on log T in [log 0.5, log 3], two coordinate sweeps, minimising the mean emitted NLL over 8 calibration draws (exact NLL for deterministic configurations).

**MODEL_SELECT:** labels are never read by family B. Feedback comes only through nrm.evaluate and nrm.screen outputs.

**Versions:** v in {0, 1}. Initialisation, split, permutations and training noise depend on v. Both versions are fitted.

**Fit counts per version:**
- 7 learners x 2 model fits (early-stopping search + refit) = 14 model fits;
- 10 calibrator fits;
- 0 critic fits;
- the per-epoch geometry estimates are recorded in each learner's training log.

## 7. Strength path (predeclared)

| Setting | s | Second-order linked cost (s^2/2) |
|---|---|---|
| S1 | 0.06 | 0.0018 nats |
| S2 | 0.09 | 0.0041 nats |
| S3 | 0.12 | 0.0072 nats |

- The SEX-blind PCA path uses the same s at each setting.
- The frontier is read on realized costs (nrm.evaluate) against linked and single-view recovery (nrm.screen and the grid).

## 8. Reader needs for randomized outputs

- **Same law for training and scoring.** Readers train and score under the same channel law.
  - In a reader cell, every auxiliary person (reader FIT and VALIDATION) contributes ONE draw.
  - That draw comes from the same release instance as the SCORE people, and each SCORE person contributes exactly one draw.
  - No multi-draw variant is requested, and target people never contribute more than one draw.
- **Linked readers** pool the two recipients' draws of the same instance, which is the actual joint release.
- **Proposed cell-to-instance mapping.**
  - Cell (v, repetition j, reader seed k) uses instance r = j for both seeds, which separates seed variation from Monte Carlo variation.
  - Monte Carlo variation: refit one reader recipe per cell on instances j+2 and j+4, with the same split and seed. Do this only for FB_SEX_S2 and FB_PCA_S2.
  - Report Monte Carlo variation separately from household-split and reader-seed variation.
- **Utility (nrm.evaluate).**
  - Computed per instance on each emitted vector and its emitted argmax, then averaged over the 8 instances.
  - The Monte Carlo SE (sd/sqrt(8)) is reported separately from the household-sampling SE.
  - No averaged vector is used, except in FB_SEX_S2_MEAN, where it is the actual release.
- **Deterministic configurations** use R = 1.

## 9. Configuration request and CPU

**Request: 20 of the 60** (10 configurations x 2 versions).

| Configuration | Kind |
|---|---|
| FB_TWIN | twin |
| FB_SEX_S1, FB_SEX_S2, FB_SEX_S3 | candidate (randomized) |
| FB_PCA_S1, FB_PCA_S2, FB_PCA_S3 | matched SEX-blind control (randomized) |
| FB_SEX_S2_ID | ablation (deterministic) |
| FB_SEX_S2_MEAN | ablation (deterministic) |
| FB_POSTHOC_S2 | ablation (randomized) |

- If you need to cut, drop FB_POSTHOC_S2 first, then FB_SEX_S2_MEAN (16 left). The twin and both strength paths are the core.

**Queue.** Two fit units, one per version:

`python3 -m nrm.jobqueue run --unit FB_FIT_V<v> --stage fit --rss-gib 2 -- <repository>/.venv/bin/python -m nrm.fb.learner fit --version <v> --card <committed card>`

**CPU.**
- Measured on invented arrays of the real size (20,000 x 145): a learner costs about 0.1 CPU-s per epoch, and emitting 3,000 people x 8 draws takes 0.016 s.
- Per version: 7 learners x at most 80 epochs, plus 10 temperature fits, comes to about 2-4 CPU-min.
- I request at most 0.25 CPU-h for family-B fits, including one retry.
- Reader cost belongs to your grid, plus the Monte Carlo replicate fits in section 8.
- Out-of-queue work so far is charged as 40 CPU-s (label designer_b): selftests and the timing script.

**Note on retries.** A version's fit writes RELEASE_ROOT.bin once and refuses a second fit, so a retry needs new configuration ids.

## 10. Predictions, registered before any fit

| | Prediction | Probability |
|---|---|---|
| P1 | FB_TWIN passes all 24 clauses + 4 guards in both versions | 0.70 |
| P2 | FB_SEX_S2 pair-AUC reduction against FB_TWIN is 0.002-0.012 (central 0.006); FB_PCA_S2's is at most 0.003 | (central estimate) |
| P3 | The SEX path is below the PCA path at matched realized cost at S2 and S3 | 0.60 |
| P4 | FB_SEX_S2_ID and FB_SEX_S2_MEAN are within 0.003 of FB_TWIN in recovery | 0.75 |
| P5 | FB_POSTHOC_S2 is about equal to FB_SEX_S2 in recovery, and channel-aware training saves less than 0.001 nats of cost | 0.65 |
| P6 | The 0.020 target against NEURAL_RAW and H_LJ is not met | > 0.90 |

## 11. Revision plan

- **R1 (utility).** If FB_TWIN fails any original clause or guard in either version:
  - one capacity revision, FB2_*: d-256-256-64-K with Adam weight decay 1e-4, same channel;
  - if it still fails, the family is UTILITY_INFEASIBLE.
- **R2 (strength).** If FB_SEX_S3 passes with at least 0.004 nats of occupation NLL slack in both versions, add S4 (s = 0.15) for both the SEX and PCA paths (4 configurations).
- **R3 (nonlinear readers).** If F4 holds:
  - replace the global geometry with decision-cell-conditional geometry: delta_c and Sigma_w,c per linked decision cell, 10 cells;
  - new configurations FB3_SEX_S2/S3 and FB3_PCA_S2/S3 (8 configurations).
- **Close the family:**
  - if F1 still holds after R3;
  - or immediately if F2 holds.

  No strength, penalty or coupling tuning beyond the predeclared path.
- **F3 alone:** report that joint adaptation adds nothing. The channel, if useful, is then only a frozen-learner component.

# Files written and committed

Commit ef0bc1ddf on research/pcrl-new-release-mechanisms-v1. Only my files were added, and the message has no trailer.
- <repository>/.claude/worktrees/pcrl-correction-transfer-stability-v1/nrm/fb/__init__.py
- <repository>/.claude/worktrees/pcrl-correction-transfer-stability-v1/nrm/fb/channel.py
  - numpy only;
  - coordinates, Fisher, geometry, per-person operators, keyed one-draw-per-person emission, identity channel, mean-vector construction, emitted-vector metrics with Monte Carlo SE.
- <repository>/.claude/worktrees/pcrl-correction-transfer-stability-v1/nrm/fb/learner.py
  - torch, one thread;
  - joint from-scratch learners, temperatures, the `fit --version V --card C` entry point (all 10 configurations of a version), deployment from features alone.

Not committed: a scratch timing script in my scratchpad (invented data only). No card has been committed yet.

# Selftest output (invented arrays only)

`python3 -m nrm.fb.channel selftest` (exit 0; also exit 0 under the torch venv):
CHANNEL {"ok":true,"checks":{"helmert_orthonormal":true,"softmax_preserved":true,"z_from_probs_roundtrip":true,"fisher_equals_hessian":true,"identity_channel_exact":true,"shapes_one_draw_per_person":true,"valid_probability_rows":true,"draws_independent_of_row_order":true,"draws_reproducible":true,"repetitions_differ":true,"keys_differ":true,"rank1_coupled_draw":true,"coupled_scalar_is_the_keyed_draw":true,"keyed_normals_moments":true,"second_order_cost_matches":true,"mean_vector_matches_mc":true,"mean_vector_identity_at_s0":true,"targeted_beats_blind_on_invented_data":true,"scaled_geometry_equals_refit":true,"deterministic_mc_se_zero":true,"refusals":true,"no_data_access_module":true},"cpu_s":2.32}
CHANNEL_INFO {"normal_moments":{"mean":0.0011,"var":0.9998,"max_abs_corr":0.0062},"second_order_cost_s0.1":{"sex":0.00476,"pca":0.00448,"iso":0.00523},"mean_vector_vs_mc_max_abs":0.0045,"lda_delta2_one_draw_s0.3":{"identity":11.19,"sex":0.399,"pca":11.069,"iso":1.737}}

Notes on these numbers:
- The predicted second-order cost at s = 0.1 is 0.005.
- The Delta^2 figures come from invented data with the SEX shift deliberately placed in a low-variance direction. They are a construction check, not evidence.

`<repository>/.venv/bin/python -m nrm.fb.learner selftest` (exit 0):
- It ran the full fit_version pipeline through fake access hooks with a reduced recipe.
- It also checked that the real access layer is refused before anything is written.

LEARNER {"ok":true,"checks":{"real_access_refused":true,"all_configurations_saved":true,"randomized_flags":true,"randomized_stack_shape":true,"label_reads_policy":true,"paired_streams_across_learners":true,"identity_config_deterministic":true,"mean_vector_config_deterministic":true,"deployment_from_features_bitwise":true,"twin_and_s2_learners_differ":true,"geometry_frozen_in_record":true,"temperatures_in_bounds":true,"second_fit_refused":true},"cpu_s":1.6}

Invented-data fit-time diagnostics. The emitted NLL and one-draw Delta^2 vary run to run because the release root comes from os.urandom.
- FB_TWIN: NLL income 0.5838, occupation 1.4486; Delta^2 0.593
- FB_SEX_S2: NLL 0.5844 / 1.4507; Delta^2 0.263
- FB_PCA_S2: NLL 0.5848 / 1.4508; Delta^2 0.326
- FB_SEX_S2_ID: Delta^2 0.594
- FB_SEX_S2_MEAN: Delta^2 0.594
- FB_POSTHOC_S2: Delta^2 0.235

# Process notes

- I read neither nrm/fa/ nor roles/designer_a/.
- Two read-only search agents summarised the earlier campaigns for me. They ran no Python and read no data.
- Five of the ten Adult-era campaigns I was pointed to release 16-d features plus logits, not probability vectors. The ACS figures cited above come from the controlled_training_size records and the brief.
